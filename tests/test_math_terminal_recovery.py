"""Fresh semantic retries, resolved-only caching, and immutable budget accounting."""
import json
from pathlib import Path
from dataclasses import asdict
import pytest
import httpx
from openai import APITimeoutError
from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import (
    prediction_validity_contract,invalid_recovery_contract,prediction_from_persisted,
)
from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATHV21BenchmarkAdapter
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.persistence.exact_output_cache import DurableExactOutputCache,FIELDS
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,OperationalAbort

ROOT=Path(__file__).resolve().parents[1]


def contract():
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_canary_v6.json').read_bytes())
    c.update(identity=versions.MATH_LOW_COST_EXECUTION_BINDING_VERSION,
        prediction_validity_policy=dict(prediction_validity_contract(),
            identity=versions.MATH_RECOVERY_PREDICTION_VERSION,invalid_response_retries=3),
        invalid_recovery_policy=invalid_recovery_contract(),
        low_cost_protocol=dict(identity=versions.MATH_LOW_COST_PROTOCOL_VERSION),
        cache_policy=versions.DURABLE_EXACT_OUTPUT_CACHE_VERSION)
    return c


def ports(responses,**kwargs):
    c=contract();b=MATHV21BenchmarkAdapter(c);calls=[];events=[];iterator=iter(responses)
    def transport(request):
        calls.append(request);value=next(iterator)
        if isinstance(value,Exception):raise value
        return dict(text=value[0],finish_reason=value[1],input_tokens=2,output_tokens=3)
    broker=RequestBroker(contract=c,transport=transport,arm='A1',seed=81,ledger_writer=events.append,**kwargs)
    item=protocol_input('math','synthetic',{'problem':'Synthetic arithmetic.'},b.output_contract,protocol=b.protocol)
    return b,broker,BenchmarkSolver(b,broker),item,calls,events


def solve(solver,item,stage='initial'):
    return solver.solve('Synthetic procedure.',item,stage=stage,split='optimize')


@pytest.mark.parametrize('responses,attempts,invalid,recovered,terminal',[
    ([('FINAL_ANSWER: 1','stop')],1,0,False,False),
    ([('FINAL_ANSWER: 2','stop')],1,0,False,False),
    ([('missing','stop'),('FINAL_ANSWER: 1','stop')],2,1,True,False),
    ([('FINAL_ANSWER: 1','length'),('FINAL_ANSWER:','stop'),('FINAL_ANSWER: 1','stop')],3,2,True,False),
    ([('missing','stop')]*4,4,4,False,True)])
def test_resolved_contract_and_cache(responses,attempts,invalid,recovered,terminal):
    b,broker,solver,item,calls,events=ports(responses)
    result=solve(solver,item)
    assert (result.semantic_attempt_count,result.raw_invalid_count,result.recovered_invalid,result.terminal_invalid)==(
        attempts,invalid,recovered,terminal)
    assert len(calls)==attempts and all(r==calls[0] for r in calls)
    assert prediction_from_persisted(json.loads(json.dumps(asdict(result))))==result
    assert b.score_member_output(result.parsed(),'1')==int(result.prediction_valid and result.answer=='1')
    assert solve(solver,item,'full')==result and len(calls)==attempts
    assert len(broker.cache)==1 and events[-1]['provider_called'] is False
    assert broker.usage['input_tokens']==2*attempts and broker.usage['output_tokens']==3*attempts


def test_transport_retry_does_not_consume_semantic_attempt(monkeypatch):
    monkeypatch.setattr('multi_dataset_diverse_rl.search.provider_runtime.time.sleep',lambda _:None)
    timeout=APITimeoutError(request=httpx.Request('POST','https://example.invalid'))
    _,broker,solver,item,calls,_=ports([timeout,('missing','stop'),timeout,('FINAL_ANSWER: 1','stop')])
    result=solve(solver,item)
    assert len(calls)==4 and result.semantic_attempt_count==2 and result.transport_retry_attempts==2
    assert broker.usage['failures']==2 and broker.usage['successes']==2


def test_budget_blocked_retry_cannot_be_written_as_terminal(tmp_path):
    with TokenLedger(tmp_path,task_sha256='a'*64) as ledger:
        _,broker,solver,item,calls,_=ports([('missing','stop')]*4,token_ledger=ledger,
            reserve_reader=lambda:0 if not calls else ledger.remaining)
        with pytest.raises(OperationalAbort,match='INSUFFICIENT_FOR_INVALID_RECOVERY'):
            solve(solver,item)
        assert len(calls)==1 and not broker.cache and ledger.view()['charged_total']==5


def test_same_attempt_durable_reuse_rejects_changed_scope_and_corruption(tmp_path):
    context={k:('a'*64) for k in FIELDS}
    from multi_dataset_diverse_rl.persistence.exact_output_cache import digest
    c=contract()
    context.update(execution_attempt_id=c['execution_attempt_id'],cache_namespace=c['cache_namespace'],
        binding_sha256=digest(c),generation_policy_sha256=digest(c['solver_decoding_policy']),
        recovery_policy_sha256=digest(c['invalid_recovery_policy']))
    cache=DurableExactOutputCache(tmp_path,context)
    _,broker,solver,item,calls,_=ports([('missing','stop'),('FINAL_ANSWER: 1','stop')],durable_cache=cache)
    result=solve(solver,item);assert len(calls)==2
    b=MATHV21BenchmarkAdapter(contract())
    restarted=RequestBroker(contract=contract(),transport=lambda _:pytest.fail('durable hit must not dispatch'),
        arm='A1',seed=81,durable_cache=DurableExactOutputCache(tmp_path,context))
    assert solve(BenchmarkSolver(b,restarted),item,'full')==result and restarted.usage['attempts']==0
    for field in FIELDS:
        with pytest.raises(SearchContractError,match='ATTEMPT_MISMATCH'):
            DurableExactOutputCache(tmp_path,{**context,field:'b'*64})
    entry=next(p for p in tmp_path.glob('*.json') if p.name!='scope.json')
    data=json.loads(entry.read_bytes());data['result']['resolved_prediction']['terminal_invalid']=True
    entry.write_text(json.dumps(data))
    with pytest.raises(SearchContractError,match='CORRUPTION'):cache.get(entry.stem)


def test_unresolved_and_forged_terminal_metadata_fail_closed():
    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import classify_prediction,resolve_predictions
    with pytest.raises(SearchContractError,match='UNRESOLVED'):
        resolve_predictions([classify_prediction('missing')])
    result=asdict(resolve_predictions([classify_prediction('missing')]*4))
    result['semantic_attempt_count']=3
    with pytest.raises(SearchContractError,match='CORRUPTION'):prediction_from_persisted(result)
