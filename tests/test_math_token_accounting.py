"""Zero-provider reservation, crash recovery and guarded transport regressions."""
import json
from pathlib import Path
import asyncio

import httpx
from openai import OpenAI, APITimeoutError, InternalServerError
import pytest

from multi_dataset_diverse_rl.governance.token_accounting import (
    TokenLedger, OperationalAbort, serialized_request, reservation,
)
from multi_dataset_diverse_rl.governance.autonomous_math import create_transport
from multi_dataset_diverse_rl.benchmarks.math_accounting_prep import (
    ValidationReserve, escaped_prompt_bytes, solver_request,
)
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker

ROOT=Path(__file__).resolve().parents[1]
TASK="a"*64


def request(**extra):
    return dict(model="qwen3-8b",max_tokens=1800,temperature=0.,extra_body={"enable_thinking":False},
        messages=[{"role":"system","content":"Formatting interface"},{"role":"user","content":"Synthetic public 数学 fixture"}],**extra)


def contract():
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_execution_v1_2.json').read_bytes())
    return {**c,"execution_attempt_id":"synthetic_attempt","execution_phase":"canary"}


def broker(tmp_path,transport,**kwargs):
    ledger=TokenLedger(tmp_path,task_sha256=TASK)
    b=RequestBroker(contract=contract(),transport=transport,arm="A1",seed=81,token_ledger=ledger,**kwargs)
    return b,ledger


def complete(b):
    return b.complete(role="solver",split="optimize",stage="initial",messages=request()["messages"])


def test_exact_utf8_reservation_includes_roles_and_thinking():
    r=request()
    raw=serialized_request(r)
    assert json.loads(raw)["enable_thinking"] is False
    assert "extra_body" not in json.loads(raw)
    assert reservation(r)["amount"]==len(raw)+4096+1800
    assert "数学".encode() in raw
    r["messages"][0]["content"]="Longer immutable formatting interface"
    assert reservation(r)["amount"]>len(raw)+4096+1800


@pytest.mark.parametrize("usage",[dict(input_tokens=23,output_tokens=11),
    dict(input_tokens=None,output_tokens=None),{},dict(input_tokens=-1,output_tokens=3),
    dict(input_tokens=True,output_tokens=3),dict(input_tokens=1000000,output_tokens=3)])
def test_success_usage_or_full_fallback(tmp_path,usage):
    b,l=broker(tmp_path,lambda r:dict(text="FINAL_ANSWER: 1",**usage))
    try:
        result=complete(b)
        if usage==dict(input_tokens=23,output_tokens=11):
            assert l.view()["provider_reported_actual"]==34
            assert l.view()["fallback_charged"]==0
        else:
            assert l.view()["fallback_charged"]==result["input_tokens"]+result["output_tokens"]
            assert l.view()["fallback_charged"]>1800
        assert l.reserved==0
        before=l.view()["charged_total"]
        assert complete(b)["provider_called"] is False
        assert l.view()["charged_total"]==before
    finally:
        l.close()


@pytest.mark.parametrize("kind",["timeout","5xx"])
def test_failed_retry_is_fully_charged(tmp_path,monkeypatch,kind):
    calls=[]
    def transport(r):
        calls.append(r)
        if len(calls)==1:
            wire=httpx.Request("POST","https://example.invalid")
            if kind=="timeout":
                raise APITimeoutError(request=wire)
            raise InternalServerError("synthetic",response=httpx.Response(500,request=wire),body={})
        return dict(text="FINAL_ANSWER: 1",input_tokens=10,output_tokens=5)
    monkeypatch.setattr("multi_dataset_diverse_rl.search.provider_runtime.time.sleep",lambda _:None)
    b,l=broker(tmp_path,transport)
    try:
        complete(b)
        assert len(calls)==2 and b.usage["failures"]==1
        assert l.view()["fallback_charged"]==reservation(calls[0])["amount"]
        assert l.view()["provider_reported_actual"]==15
        assert l.view()["by_attempt"]["synthetic_attempt"]["physical_attempts"]==2
        assert l.view()["charged_total"]==l.view()["fallback_charged"]+15
    finally:
        l.close()


def test_failed_call_with_reliable_partial_usage(tmp_path):
    def transport(r):
        exc=ValueError("synthetic integration failure")
        exc.token_usage=dict(input_tokens=12,output_tokens=4)
        raise exc
    b,l=broker(tmp_path,transport)
    try:
        with pytest.raises(OperationalAbort,match="PROVIDER_TERMINAL"):
            complete(b)
        assert l.view()["charged_total"]==16 and l.reserved==0
    finally:
        l.close()


def test_validation_reserve_denies_before_transport(tmp_path):
    calls=[]
    b,l=broker(tmp_path,lambda r:calls.append(r),reserve_reader=lambda:30_000_000)
    try:
        with pytest.raises(OperationalAbort,match="INSUFFICIENT_FOR_VALIDATION"):
            complete(b)
        assert not calls and l.view()["charged_total"]==0 and l.reserved==0
    finally:
        l.close()


def test_charge_monotonic_and_budget_exhaustion(tmp_path):
    with TokenLedger(tmp_path,task_sha256=TASK) as l:
        large={**request(),"max_tokens":29_995_000}
        k=l.reserve(large,attempt_id="a",stage="canary",role="solver",model="qwen3-8b")
        l.reconcile(k,None,outcome="failure")
        before=l.view()["charged_total"]
        with pytest.raises(OperationalAbort,match="BUDGET_EXHAUSTED"):
            l.reserve(request(),attempt_id="b",stage="pilot",role="solver",model="qwen3-8b")
        assert l.view()["charged_total"]==before
        assert before+l.reserved<=30_000_000


def test_crash_unresolved_reservation_fully_charged_on_recovery(tmp_path):
    l=TokenLedger(tmp_path,task_sha256=TASK)
    l.reserve(request(),attempt_id="a",stage="canary",role="solver",model="qwen3-8b")
    l.close()  # No reconciliation; models loss of the owning process.
    with TokenLedger(tmp_path,task_sha256=TASK) as recovered:
        assert recovered.reserved==0
        assert recovered.view()["charged_total"]==reservation(request())["amount"]
        assert recovered.view()["fallback_charged"]==recovered.view()["charged_total"]
        old=recovered.view()["charged_total"]
    with TokenLedger(tmp_path,task_sha256=TASK) as again:
        assert again.view()["charged_total"]==old


def test_concurrent_owner_and_corrupt_journal_fail_closed(tmp_path):
    with TokenLedger(tmp_path,task_sha256=TASK):
        with pytest.raises(OperationalAbort,match="ALREADY_OWNED"):
            TokenLedger(tmp_path,task_sha256=TASK)
    with (tmp_path/'events.jsonl').open('ab') as f:
        f.write(b'{broken}\n')
    with pytest.raises(OperationalAbort,match="LEDGER_CORRUPTION"):
        TokenLedger(tmp_path,task_sha256=TASK)


def test_truncation_aborts_after_charge_without_regeneration(tmp_path):
    calls=[]
    def transport(r):
        calls.append(r)
        return dict(text="incomplete",input_tokens=10,output_tokens=1800,finish_reason="length")
    b,l=broker(tmp_path,transport)
    try:
        with pytest.raises(OperationalAbort,match="OUTPUT_TRUNCATION"):
            complete(b)
        assert len(calls)==1 and l.view()["charged_total"]==1810
    finally:
        l.close()


def test_opaque_transport_sends_exact_reserved_body(monkeypatch):
    captured=[]
    def handler(r):
        assert r.headers['authorization']=='Bearer synthetic'
        assert r.headers['content-type']=='application/json'
        assert r.url.path=='/v1/chat/completions'
        captured.append(r.content)
        return httpx.Response(200,json={"choices":[{"message":{"content":"FINAL_ANSWER: 1"},"finish_reason":"stop"}],
                                      "usage":{"prompt_tokens":8,"completion_tokens":4}})
    client=OpenAI(api_key="synthetic",base_url="https://example.invalid/v1",max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr("multi_dataset_diverse_rl.provider_factory.ProviderClientFactory.from_environment",lambda **_:client)
    transport,client=create_transport(contract())
    try:
        result=transport(request())
        assert captured==[serialized_request(request())]
        assert result["input_tokens"]==8 and result["output_tokens"]==4
    finally:
        client.close()


@pytest.mark.parametrize("status,payload", [(403,{"error":{"message":"synthetic denial"}}),
    (500,{"error":{"message":"synthetic server failure"},"usage":{"prompt_tokens":7,"completion_tokens":2}}),
    (503,"synthetic non-JSON failure")])
def test_installed_sdk_http_error_mapping_preserves_private_evidence(monkeypatch,status,payload):
    def handler(r):
        return httpx.Response(status,**({"json":payload} if isinstance(payload,dict) else {"text":payload}))
    client=OpenAI(api_key="synthetic",base_url="https://example.invalid/v1",max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr("multi_dataset_diverse_rl.provider_factory.ProviderClientFactory.from_environment",lambda **_:client)
    transport,client=create_transport(contract())
    try:
        from openai import PermissionDeniedError
        with pytest.raises(PermissionDeniedError if status==403 else InternalServerError) as error:
            transport(request())
        assert error.value.provider_evidence["http_status"]==status
        if status==500:
            assert error.value.token_usage==dict(input_tokens=7,output_tokens=2)
    finally:
        client.close()


def test_broker_persists_failed_http_evidence_and_full_charge(tmp_path):
    raw=[]
    def transport(r):
        from openai import PermissionDeniedError
        wire=httpx.Request("POST","https://example.invalid")
        error=PermissionDeniedError("synthetic",response=httpx.Response(403,request=wire),body={})
        error.provider_evidence={"http_status":403,"response_body":{"error":{"message":"synthetic"}}}
        raise error
    b,l=broker(tmp_path,transport,raw_writer=raw.append)
    try:
        with pytest.raises(OperationalAbort,match="PROVIDER_TERMINAL_PermissionDeniedError"):
            complete(b)
        assert len(raw)==1 and raw[0]["provider_evidence"]["http_status"]==403
        assert l.view()["fallback_charged"]==reservation(raw[0]["request"])["amount"]
    finally:
        l.close()


@pytest.mark.parametrize("usage",[None,"invalid",[1,2],-1])
def test_http_success_with_invalid_usage_continues_and_charges_full(tmp_path,monkeypatch,usage):
    client=OpenAI(api_key="synthetic",base_url="https://example.invalid/v1",max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={
            "choices":[{"message":{"content":"FINAL_ANSWER: 1"},"finish_reason":"stop"}],"usage":usage}))))
    monkeypatch.setattr("multi_dataset_diverse_rl.provider_factory.ProviderClientFactory.from_environment",lambda **_:client)
    transport,client=create_transport(contract())
    b,l=broker(tmp_path,transport)
    try:
        result=complete(b)
        assert result['provider_called'] and not result['usage_reliable']
        assert l.view()['fallback_charged']==reservation(request())["amount"]
        assert b.usage['successes']==1 and b.usage['failures']==0
    finally:
        client.close()
        l.close()


def test_sdk_security_path_and_no_implicit_physical_redirect(monkeypatch):
    captured=[]
    def handler(r):
        captured.append(r)
        assert r.headers['authorization']=='Bearer synthetic'
        return httpx.Response(307,headers={'location':'https://example.invalid/redirect'},json={'error':{'message':'synthetic redirect'}})
    client=OpenAI(api_key='synthetic',base_url='https://example.invalid/v1',max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler),follow_redirects=True))
    # Negative control captures the integration defect without credentials or network.
    assert 'Authorization' not in client.default_headers
    monkeypatch.setattr('multi_dataset_diverse_rl.provider_factory.ProviderClientFactory.from_environment',lambda **_:client)
    transport,client=create_transport(contract())
    try:
        from openai import APIStatusError
        with pytest.raises(APIStatusError) as error:
            transport(request())
        assert len(captured)==1 and error.value.provider_evidence['http_status']==307
        assert captured[0].content==serialized_request(request())
    finally:
        client.close()


def test_provider_sdk_binding_drift_denies_before_transport():
    from multi_dataset_diverse_rl.benchmarks.math_autonomous import MATHAutonomousBinding
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_accounting_canary_v1_4.json').read_bytes())
    c['provider_sdk']={'name':'openai','version':'not-installed'}
    assert MATHAutonomousBinding(ROOT,c).blockers()==('PROVIDER_SDK_IDENTITY_MISMATCH',)


def test_validation_metadata_reserve_handles_json_escaping():
    c=contract()
    problems=("Synthetic public a", "Synthetic public \"b\" 数学")
    metadata=dict(decoding=c["decoding"],examples=[dict(blank_prompt_serialized_request_bytes=len(serialized_request(solver_request(c,"",p)))) for p in problems])
    prompts=("a","b\nc",'"d"',"e\\f","数学")
    reserve=ValidationReserve(metadata,prompts)
    exact=sum(reservation(solver_request(c,p,q))["amount"] for q in problems for p in prompts)
    assert reserve.cost(reserve.initial_sizes)==exact
    before=reserve.remaining()
    reserve.observe("A much longer reasoning procedure containing \"quotes\" and\nline breaks.")
    assert reserve.remaining()>before
    reserve.observe("short")
    assert reserve.remaining()>before
    assert escaped_prompt_bytes("a\nb")>len("a\nb".encode())


def test_validation_accounting_metadata_is_content_free_and_search_denied():
    from multi_dataset_diverse_rl.benchmarks.math_accounting_prep import validation_rows
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    m=json.loads((ROOT/'experiments/accounting/math_validation_accounting_metadata_v2.json').read_bytes())
    assert len(m['examples'])==300 and m['model_calls']==m['correctness_evaluations']==0
    assert not m['content_exposed_to_search']
    assert all(set(r)=={'example_id','input_sha256','blank_prompt_serialized_request_bytes'} for r in m['examples'])
    with pytest.raises(SearchContractError,match='CONTEXT_FORBIDDEN'):
        validation_rows(ROOT,contract(),context='SEARCH_CONTEXT')
    with pytest.raises(SearchContractError,match='SEARCH_COMPLETE_RECEIPT_REQUIRED'):
        validation_rows(ROOT,contract(),context='POST_SEARCH_VALIDATION_CONTEXT')


@pytest.mark.parametrize("phase",["canary","pilot"])
def test_actual_public_gepa_with_accounting_and_phase_persistence(tmp_path,monkeypatch,phase,*,binding_path=None,adapter=None):
    from multi_dataset_diverse_rl.governance import unified_execution as gov
    from multi_dataset_diverse_rl.governance import autonomous_math as execution
    from multi_dataset_diverse_rl.benchmarks.math_execution import MATHExecutionBinding
    from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.benchmark import BenchmarkInput
    c=json.loads((ROOT/(binding_path or f'experiments/execution_bindings/math_v2_accounting_{phase}_v1_3.json')).read_bytes())
    # Isolate every test ledger, namespace, authorization and run.
    c={**c,'token_ledger_directory':(tmp_path/'budget').relative_to(ROOT).as_posix()}
    binding=tmp_path/'binding.json'
    c['binding_path']=binding.relative_to(ROOT).as_posix()
    binding.write_bytes((json.dumps(c,sort_keys=True,indent=2)+'\n').encode())
    monkeypatch.setattr(gov,'verify_source_commit',lambda *_:None)
    monkeypatch.setattr(execution,'consumption_path',lambda *_:tmp_path/'global_consumption.json')
    manifest=gov.preexecution_manifest(ROOT,source_sha='a'*40,binding_path=c['binding_path'],experiment_id='synthetic_'+phase)
    prep=tmp_path/'prep'
    payload=gov.prepare_canary(ROOT,manifest,destination=prep)
    auth=json.loads((prep/'authorization.json').read_bytes())
    auth['explicit_user_authorized']=True
    (prep/'authorization.json').write_bytes(json.dumps(auth).encode())
    adapter=adapter or MATHBenchmarkAdapter()
    good="Evaluate the mathematical relationships independently. Check each step and verify the conclusion using reliable general principles."
    def examples(self,role):
        return tuple(CorrectnessExample(BenchmarkInput(f'{role}{i}',f'Synthetic {role} arithmetic {i}: one plus zero.',
            adapter.output_contract,benchmark_id='math'),'1') for i in range(6 if role=='optimize' else 300))
    monkeypatch.setattr(MATHExecutionBinding,'examples',examples)
    calls=[]
    def transport(r):
        calls.append(r)
        if r['model']=='qwen3-8b':
            answer='1' if good in r['messages'][1]['content'] else '2'
            text='Synthetic reasoning.\nFINAL_ANSWER: '+answer
        else:
            text='```'+good+'```'
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop')
    class Client:
        def close(self):pass
    monkeypatch.setattr(execution,'create_transport',lambda _: (transport,Client()))
    run=tmp_path/'run'
    summary=asyncio.run(gov.execute_canary(ROOT,prep,run))
    assert summary['ledger']['attempts']==len(calls)
    assert summary['accounting']['charged_total']==4*len(calls)
    assert summary['accounting']['reserved_inflight']==0
    assert not summary['validation_calls'] and not summary['test_calls'] and not summary['pattern_calls']
    assert (run/'raw_evidence_inventory.json').is_file()
    assert (run/'trajectory_private.jsonl').is_file()
    assert (prep/'authorization_consumed.json').is_file()
    if phase=='canary':
        assert summary['result']['stop_reason']=='CANARY_PARENT_EPOCH_COMPLETE'
        assert len(summary['result']['transitions'])==1
        assert not (run/'SEARCH_COMPLETE_RECEIPT.json').exists()
    else:
        assert summary['result']['stop_reason'] in {'SATURATION_REACHED','NO_FEASIBLE_OPPORTUNITY'}
        assert (run/'SEARCH_COMPLETE_RECEIPT.json').is_file()
    assert gov.inventory(run)['files']  # Frozen files have read-back evidence.
