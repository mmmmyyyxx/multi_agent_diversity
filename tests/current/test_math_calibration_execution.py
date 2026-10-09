"""Fresh accounting, integrity, recovery and transport are tested with fakes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.diagnostics.math_calibration_accounting import (
    CAP,CalibrationLedger,CalibrationAbort,read_ledger,replay_events)
from multi_dataset_diverse_rl.diagnostics.math_calibration_execution import (
    run_logical,run_panel,validate_response,verify_bundle,IDENTITY,ATTEMPT)
from multi_dataset_diverse_rl.diagnostics.math_calibration_plan import request_for
from multi_dataset_diverse_rl.diagnostics.math_baseline_calibration import digest
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts


def response(text='Final answer: 2',finish='stop',model='qwen3-8b',output=10):
    body=dict(model=model,id='FAKE',choices=[dict(message=dict(content=text),finish_reason=finish)],
        usage=dict(prompt_tokens=5,completion_tokens=output))
    return dict(text=text,finish_reason=finish,input_tokens=5,output_tokens=output,
        provider_http_response_body=body,response_id='FAKE')


def scope():
    return dict(startup_identity_sha256='f'*64,attempt_id=ATTEMPT,source_sha='e'*40)


def row(i=0):
    return dict(stable_example_id='SYNTHETIC_'+str(i),content=dict(problem='SYNTHETIC',type='Algebra',level='Level 1'),reference_final_answer='2')


def resources(tmp_path):
    ledger=CalibrationLedger(tmp_path/'accounting','f'*64)
    receipts=ProviderResponseReceipts(tmp_path/'receipts',attempt_id=ATTEMPT,startup_identity_sha256='f'*64)
    return ledger,receipts


def test_fresh_scope_and_missing_usage_do_not_reuse_old_balance(tmp_path):
    ledger,_=resources(tmp_path)
    request=request_for('A','SYNTHETIC')
    key=ledger.reserve(request,dict(arm='A'));bound=ledger.state['reserved_total']
    charged,reliable=ledger.charge(key,None,'TIMEOUT')
    assert not reliable and charged==bound
    assert ledger.state['unknown_usage_charge']==bound
    ledger.terminal('EXECUTION_ABORTED');ledger.close()
    assert read_ledger(tmp_path/'accounting')['terminal']=='EXECUTION_ABORTED'
    with pytest.raises(CalibrationAbort,match='FRESH'):CalibrationLedger(tmp_path/'accounting','f'*64)


def test_budget_reservation_prevents_next_call_before_network(tmp_path):
    ledger,_=resources(tmp_path)
    for i in range(180):
        key=ledger.reserve(request_for('A','SYNTHETIC'),dict(i=i))
        ledger.charge(key,None,'TIMEOUT')
        if ledger.state['charged_total']>CAP-8000:break
    before=ledger.state['physical_attempts']
    with pytest.raises(CalibrationAbort,match='RESOURCE_CEILING'):
        ledger.reserve(request_for('A','SYNTHETIC'),dict(i=999))
    assert ledger.state['physical_attempts']==before
    ledger.terminal('EXECUTION_ABORTED');ledger.close()


def test_hash_chain_tamper_and_unresolved_terminal_full_charge(tmp_path):
    ledger,_=resources(tmp_path);ledger.reserve(request_for('A','SYNTHETIC'),{})
    bound=ledger.state['reserved_total'];ledger.terminal('EXECUTION_ABORTED')
    assert ledger.state['charged_total']==bound and ledger.state['reserved_total']==0
    events=deepcopy(ledger.events);ledger.close();events[1]['wire_sha256']='0'*64
    with pytest.raises(CalibrationAbort,match='CORRUPT'):replay_events(events)


def test_snapshot_failure_detaches_without_repeating_transport(tmp_path,monkeypatch):
    ledger,_=resources(tmp_path)
    def fail(*a,**k):raise OSError('SYNTHETIC_SHARING_ERROR')
    monkeypatch.setattr('multi_dataset_diverse_rl.diagnostics.math_calibration_accounting.atomic_write_json',fail)
    key=ledger.reserve(request_for('A','SYNTHETIC'),{});ledger.charge(key,response(),'RESPONSE')
    assert ledger.snapshot_detached and ledger.state['physical_attempts']==1
    ledger.terminal('EXECUTION_COMPLETE');ledger.close()


def test_native_valid_wrong_is_one_draw_and_never_diagnostic_retry(tmp_path):
    ledger,receipts=resources(tmp_path);calls=[]
    def transport(request):calls.append(request);return response('Final answer: 3')
    result=run_logical('A',row(),scope(),transport,ledger,receipts,sleep=lambda _:None)
    assert result['semantic_draws']==1 and result['first_native']['valid']
    assert result['reference']=='2' and len(calls)==1
    ledger.terminal('EXECUTION_COMPLETE');ledger.close()


def test_format_recovery_capacity_and_gold_independent_dispatch(tmp_path):
    ledger,receipts=resources(tmp_path);calls=[]
    values=[response('Final answer: 2','length'),response('Nothing concluded.'),response('Final answer: 3')]
    def transport(request):calls.append(request);return values.pop(0)
    result=run_logical('B',row(),scope(),transport,ledger,receipts,sleep=lambda _:None)
    assert [r['max_tokens'] for r in calls]==[3600,6144,3600]
    assert result['semantic_draws']==3
    assert all(r['extra_body']['enable_thinking'] is False for r in calls)
    assert len({json.dumps(r['messages']) for r in calls})==1
    ledger.terminal('EXECUTION_COMPLETE');ledger.close()


def test_timeout_receipt_and_same_request_do_not_consume_semantic_draw(tmp_path):
    ledger,receipts=resources(tmp_path);calls=[]
    APITimeoutError=type('APITimeoutError',(Exception,),{})
    def transport(request):
        calls.append(request)
        if len(calls)==1:raise APITimeoutError()
        return response()
    result=run_logical('A',row(),scope(),transport,ledger,receipts,sleep=lambda _:None)
    assert result['semantic_draws']==1 and result['transport_retries']==1 and calls[0]==calls[1]
    assert len(list((tmp_path/'receipts').glob('*.json')))==2
    assert ledger.state['unknown_usage_charge']>0
    ledger.terminal('EXECUTION_COMPLETE');ledger.close()


@pytest.mark.parametrize('failure',[response(model='wrong-alias'),response(output=4000)])
def test_provider_identity_or_output_violation_is_preserved_then_abort(tmp_path,failure):
    ledger,receipts=resources(tmp_path)
    with pytest.raises(CalibrationAbort):
        run_logical('A',row(),scope(),lambda _:failure,ledger,receipts,sleep=lambda _:None)
    assert len(list((tmp_path/'receipts').glob('*.json')))==1
    assert ledger.state['charged_total']>0
    ledger.terminal('EXECUTION_ABORTED');ledger.close()


def test_reasoning_and_content_loss_fail_closed():
    request=request_for('A','SYNTHETIC');value=response()
    value['provider_http_response_body']['choices'][0]['message']['reasoning_content']='SYNTHETIC'
    with pytest.raises(CalibrationAbort,match='NONTHINKING'):validate_response(value,request)
    value=response();value['text']='CHANGED'
    with pytest.raises(CalibrationAbort,match='CONTENT'):validate_response(value,request)


def test_receipt_failure_charges_once_no_retry(tmp_path,monkeypatch):
    ledger,receipts=resources(tmp_path);calls=[]
    def fail(*a,**k):raise OSError('SYNTHETIC')
    monkeypatch.setattr(receipts,'persist',fail)
    def transport(r):calls.append(r);return response()
    with pytest.raises(OSError):run_logical('A',row(),scope(),transport,ledger,receipts,sleep=lambda _:None)
    assert len(calls)==1 and ledger.state['reserved_total']==0 and ledger.state['unknown_usage_charge']>0
    ledger.terminal('EXECUTION_ABORTED');ledger.close()


def test_rotated_pairing_same_order_and_complete_sealed_checkpoint(tmp_path):
    calls=[]
    def transport(r):
        calls.append(r)
        return response(r'\boxed{2}' if r'\boxed' in r['messages'][0]['content'] else 'Final answer: 2')
    summary=run_panel([row(0),row(1),row(2)],scope(),tmp_path/'run',transport,sleep=lambda _:None,progress=lambda _:None)
    assert summary['status']=='EXECUTION_COMPLETE' and summary['completed_logicals']==9
    checkpoint=json.loads((tmp_path/'run/checkpoint_private.json').read_bytes())
    seal=checkpoint.pop('integrity_sha256');assert digest(checkpoint)==seal
    assert [l['arm'] for l in checkpoint['completed']]==list('ABCBCACAB')
    assert summary['validation_calls']==summary['test_calls']==summary['optimizer_calls']==0
    assert summary['accounting']['terminal']=='EXECUTION_COMPLETE'


def test_invalid_four_draws_are_complete_but_provider_failure_is_partial(tmp_path):
    summary=run_panel([row()],scope(),tmp_path/'run',lambda _:response('Nothing concluded.'),
        sleep=lambda _:None,progress=lambda _:None)
    assert summary['status']=='EXECUTION_COMPLETE' and summary['accounting']['physical_attempts']==12
    def fail(r):raise ValueError('SYNTHETIC_NO_PRIVATE_MESSAGE_IN_PUBLIC_REASON')
    failed=run_panel([row()],scope(),tmp_path/'failed',fail,sleep=lambda _:None,progress=lambda _:None)
    assert failed['status']=='EXECUTION_ABORTED' and failed['completed_logicals']==0
    assert failed['termination_reason']=='PROVIDER_TERMINAL_ValueError'


def test_exact_http_wire_flattens_false_and_preserves_content_without_sdk_retries(monkeypatch):
    import httpx
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.diagnostics.math_calibration_execution import create_exact_transport
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    captured=[];settings=[]
    def handler(request):
        captured.append(request.content)
        return httpx.Response(200,json=response()['provider_http_response_body'])
    http=httpx.Client(transport=httpx.MockTransport(handler))
    fake=SimpleNamespace(_client=http,_prepare_url=lambda _: 'https://synthetic.invalid/chat/completions',
        _build_headers=lambda _: {},close=http.close)
    def factory(**kwargs):settings.append(kwargs);return fake
    monkeypatch.setattr(ProviderClientFactory,'from_environment',factory)
    transport,client=create_exact_transport(dict(timeout_seconds=120))
    try:
        for capacity in (3600,6144):
            request=request_for('B','SYNTHETIC',capacity)
            result=transport(request)
            assert result['text']=='Final answer: 2' and captured[-1]==serialized_request(request)
            body=json.loads(captured[-1]);assert body['enable_thinking'] is False and body['top_k']==20
            assert 'extra_body' not in body and body['max_tokens']==capacity
        assert settings[0]['max_retries']==0 and settings[0]['timeout']==120
    finally:client.close()


def test_post_execution_pairing_and_bootstrap_do_not_compare_partial_panels():
    from multi_dataset_diverse_rl.diagnostics.math_calibration_results import summarize_scored,bootstrap_paired
    values=[]
    for arm in ('A','B','C'):
        values.append(dict(arm=arm,example_id_sha256='f'*64,first_valid=arm!='A',first_correct=arm!='A',
            recovered_valid=True,recovered_correct=True,input_tokens=5,output_tokens=10,charged_tokens=15,
            semantic_draws=1,transport_retries=0,first_diagnostic_status='EXPLICIT_PARSEABLE_RESULT',
            final_diagnostic_status='EXPLICIT_PARSEABLE_RESULT',first_diagnostic_correct=True,final_diagnostic_correct=True,
            truncated_draws=0,capacity_expanded_draws=0,terminal_native_reason=None,attempt_native_reasons=['VALID'],
            subject='Algebra',level='Level 1'))
    complete=summarize_scored(values,dict(charged_total=45),True)
    assert complete['paired_comparisons']['A_B']['metrics']['first_correct']['improved']==1
    assert complete['paired_comparisons']['A_C']['contrast']=='OUTPUT_INTERFACE_WITH_DIFFERENT_PARSER'
    partial=summarize_scored(values[:2],dict(charged_total=30),False)
    assert partial['paired_comparisons']=={} and not partial['efficacy_comparisons_permitted']
    assert bootstrap_paired([1,0,-1])==bootstrap_paired([1,0,-1])


def test_journal_failure_never_appends_another_unknown_charge(tmp_path,monkeypatch):
    ledger,_=resources(tmp_path)
    def fail(*a,**k):raise OSError('SYNTHETIC_FSYNC_FAILURE')
    monkeypatch.setattr('multi_dataset_diverse_rl.diagnostics.math_calibration_accounting.append_jsonl',fail)
    with pytest.raises(OSError):ledger.reserve(request_for('A','SYNTHETIC'),{})
    with pytest.raises(CalibrationAbort,match='JOURNAL'):ledger.terminal('EXECUTION_ABORTED')
    ledger.close()


@pytest.mark.parametrize('body',[{'model':'qwen3-8b','usage':{'prompt_tokens':5,'completion_tokens':10}},'NON_JSON_SYNTHETIC'])
def test_malformed_http_paid_body_is_preserved_before_abort(monkeypatch,body):
    import httpx
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.diagnostics.math_calibration_execution import create_exact_transport
    http=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json=body)
        if isinstance(body,dict) else httpx.Response(200,text=body)))
    fake=SimpleNamespace(_client=http,_prepare_url=lambda _:'https://synthetic.invalid/chat/completions',
        _build_headers=lambda _:{},close=http.close)
    monkeypatch.setattr(ProviderClientFactory,'from_environment',lambda **kwargs:fake)
    transport,client=create_exact_transport(dict(timeout_seconds=120))
    try:
        with pytest.raises(Exception) as failure:transport(request_for('A','SYNTHETIC'))
        evidence=failure.value.provider_evidence
        assert evidence.get('response_body',evidence.get('response_text'))==body
    finally:client.close()
