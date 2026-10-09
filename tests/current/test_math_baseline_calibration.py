"""Synthetic contract tests; no benchmark private content or provider calls."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.benchmarks.math_structured_answer import classify_prediction
from multi_dataset_diverse_rl.diagnostics.math_baseline_calibration import (
    diagnostic_extract,grade_diagnostic,primary_correctness,primary_reason,native_extract,
    balanced_box,summarize_rounds,paired_comparison,digest,NATIVE_ID,BOXED_ID)
from multi_dataset_diverse_rl.diagnostics.math_baseline_audit import (
    cache_read,validate_initial_panel,verify_ledger_events,optimize_rows,deployment_audit)
from multi_dataset_diverse_rl.diagnostics.math_calibration_plan import (
    choose_membership,plan_artifact,validate_plan,budget_artifact,request_for,FakeProvider,fake_recovery)


@pytest.mark.parametrize('text,finish,reason',[
    ('Nothing concluded.','stop','MISSING_FINAL_LABEL'),
    ('## Final answer: 2','stop','MARKDOWN_WRAPPED_LABEL'),
    ('**Final answer: 2**','stop','MARKDOWN_WRAPPED_LABEL'),
    ('Final answer: 2\nMore text.','stop','FINAL_LABEL_NOT_LAST'),
    (r'\boxed{2}','stop','BOXED_ONLY'),
    ('Final answer: unknown result','stop','MALFORMED_FINAL_PAYLOAD'),
    ('Final answer: 2\nFinal answer: 3','stop','CONFLICTING_FINAL_DECLARATIONS'),
    ('Final answer: 2','length','OUTPUT_TRUNCATED'),
    ('Final answer: 2','tool_calls','OTHER_OUTPUT_CONTRACT_FAILURE'),
    (None,'stop','INSUFFICIENT_EVIDENCE')])
def test_mutually_exclusive_reasons_respect_native_and_finish(text,finish,reason):
    assert primary_reason(text,finish,classify_prediction(text,finish))==reason


def test_math_parse_failure_is_distinct_from_label_failure():
    native=SimpleNamespace(prediction_valid=False,invalid_reason='PAYLOAD_PARSE_FAILURE')
    assert primary_reason('Final answer: @','stop',native)=='MATH_PAYLOAD_PARSE_FAILURE'


@pytest.mark.parametrize('text',[
    '**Final answer:** $2$',r'\boxed{2}', 'The final answer is 2.',
    'Final answer:\n\\[\n\\boxed{2}\n\\]', 'Final answer:\n$2$',
    '\\[\n\\boxed{2}\n\\]', '2'])
def test_explicit_gold_blind_diagnostic_forms(text):
    extraction=diagnostic_extract(text)
    assert extraction.status=='EXPLICIT_PARSEABLE_RESULT'
    assert grade_diagnostic(extraction,'2') is True
    assert grade_diagnostic(extraction,'3') is False
    assert asdict(extraction)==asdict(diagnostic_extract(text))


def test_ambiguity_and_no_gold_matching_intermediate_selection():
    ambiguous=diagnostic_extract('Final answer: 2\nFinal answer: 3')
    assert ambiguous.status=='AMBIGUOUS_MULTIPLE_RESULTS' and ambiguous.payload is None
    assert grade_diagnostic(ambiguous,'2') is None
    assert diagnostic_extract(r'Intermediate \boxed{2}'+ '\nNo result established.').payload is None
    assert diagnostic_extract('Unfinished work.').status=='UNKNOWN_NO_EXPLICIT_RESULT'
    with pytest.raises(TypeError):diagnostic_extract('Final answer: 3',reference='2')


def test_truncated_payload_cannot_become_official_correct():
    extraction=diagnostic_extract('Final answer: 2','length')
    assert extraction.truncated and extraction.status=='EXPLICIT_RESULT_TRUNCATED'
    assert native_extract('A','Final answer: 2','length')['valid'] is False


def test_secondary_cannot_supply_primary_score():
    with pytest.raises(ValueError,match='DIAGNOSTIC_CANNOT'):
        primary_correctness(asdict(diagnostic_extract('Final answer: 2')),'2')


def test_terminal_boxed_parser_identity_and_intermediate_rejection():
    assert native_extract('A','Final answer: 2')==native_extract('B','Final answer: 2')
    assert native_extract('C',r'\boxed{2}')['policy_identity']==BOXED_ID
    assert native_extract('A',r'\boxed{2}')['policy_identity']==NATIVE_ID
    assert not native_extract('A',r'\boxed{2}')['valid']
    for text in (r'\boxed{2}'+ '\nMore text.',r'\boxed{2}.',r'\boxed{2} \boxed{3}'):
        assert not native_extract('C',text)['valid']
    assert balanced_box(r'\boxed{\frac{1}{2}}')==r'\frac{1}{2}'


def panel():
    return [[dict(solver_trajectory=dict(source=dict(member_id=m,example_id=str(i))))
        for i in range(12)] for m in range(5)]


def test_sixty_logical_panel_uniqueness_and_same_example_order():
    original=panel();validate_initial_panel(original)
    duplicate=deepcopy(original);duplicate[1][1]=duplicate[1][0]
    with pytest.raises(ValueError):validate_initial_panel(duplicate)
    different=deepcopy(original);different[2].reverse()
    with pytest.raises(ValueError):validate_initial_panel(different)
    with pytest.raises(ValueError):validate_initial_panel(original[:4])


def test_retry_round_denominators_cost_and_transport_errors_separate():
    attempts=[dict(semantic_attempt_index=i,native_valid=i==2,official_correctness_under_frozen_rule=i==2,
        input_tokens=10,output_tokens=20,finish_reason='stop') for i in (1,1,2)]
    result=summarize_rounds([{},{}],attempts,[dict(semantic_attempt_no=2,conservative_charge=50)])
    assert result[0]['eligible_logical_examples']==2 and result[1]['eligible_logical_examples']==1
    assert result[1]['original_logical_denominator']==2
    assert result[1]['became_correct']==1 and result[1]['reported_tokens']==30
    assert result[1]['unknown_usage_reservation']==50


def test_cache_reader_is_read_only_and_rejects_corruption(tmp_path):
    from dataclasses import asdict
    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import resolve_predictions
    prediction=resolve_predictions((classify_prediction('Final answer: 2'),),0)
    result=dict(original_realizations=[{}],resolved_prediction=asdict(prediction))
    scope=dict(identity='SYNTHETIC');key='f'*64
    row=dict(context=scope,request_sha256=key,result=result,response_sha256=digest(result))
    row['integrity_seal']=digest(row)
    for filename,value in (('scope.json',scope),(key+'.json',row)):
        (tmp_path/filename).write_text(json.dumps(value),encoding='utf-8')
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert cache_read(tmp_path,key)==json.loads(json.dumps(result))
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    row['response_sha256']='0'*64;(tmp_path/(key+'.json')).write_text(json.dumps(row))
    with pytest.raises(ValueError):cache_read(tmp_path,key)


def test_ledger_hash_chain_checked_without_ledger_owner():
    row=dict(kind='SYNTHETIC',sequence=0,previous_sha256=None);row['event_sha256']=digest(row)
    assert verify_ledger_events([row])['hash_chain_verified']
    row['sequence']=1
    with pytest.raises(ValueError):verify_ledger_events([row])
    from multi_dataset_diverse_rl.diagnostics.math_baseline_audit import verify_receipt_wire_join,ATTEMPT
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    request=request_for('A','SYNTHETIC')
    record=dict(role='solver',stage='initial',request=request)
    event=dict(request_sha256=hashlib.sha256(serialized_request(request)).hexdigest(),
        role='solver',stage='canary',model='qwen3-8b',attempt_id=ATTEMPT)
    verify_receipt_wire_join({'1':event},[('1',record,'seal')],'canary')
    event['request_sha256']='0'*64
    with pytest.raises(ValueError):verify_receipt_wire_join({'1':event},[('1',record,'seal')],'canary')


def test_optimize_reader_never_requests_validation_test_or_shadow(monkeypatch,tmp_path):
    calls=[]
    class Reader:
        def __init__(self,*a,**k):pass
        def rows(self,split,purpose):calls.append(split);return []
    monkeypatch.setattr('multi_dataset_diverse_rl.diagnostics.math_baseline_audit.ExperimentSplitReader',Reader)
    optimize_rows(tmp_path,dict(canonical_root='synthetic',split_directory='splits',split_manifest_sha256='f'*64,split_version='synthetic'))
    assert calls==['optimize']


def rows():
    return [dict(stable_example_id=str(i),content=dict(problem='SYNTHETIC',type='Algebra',level='Level '+str(i%5+1)),
        content_sha256='a'*64,input_sha256='b'*64,reference_final_answer='2') for i in range(150)]


def plan():
    selected,selection=choose_membership(rows(),[str(i) for i in range(12)])
    return plan_artifact(selected,selection,dict(source_sha256='c'*64)),selected


def test_deterministic_legal_membership_all_arms_and_no_outcome_selection():
    p,selected=plan();validate_plan(p)
    assert len(selected)==60 and p['data']['selection']['available_after_exclusion']==138
    assert not {r['stable_example_id'] for r in selected}&{str(i) for i in range(12)}
    assert choose_membership(rows(),[str(i) for i in range(12)])[0]==selected
    fewer,meta=choose_membership(rows()[:20],[str(i) for i in range(12)])
    assert len(fewer)==meta['actual_n']==8


def test_no_thinking_or_sampling_differences_across_arms():
    requests=[request_for(arm,'SYNTHETIC') for arm in ('A','B','C')]
    assert all(r['extra_body']['enable_thinking'] is False for r in requests)
    assert all(r['messages'][1]==requests[0]['messages'][1] for r in requests)
    assert len({json.dumps({k:v for k,v in r.items() if k!='messages'},sort_keys=True) for r in requests})==1
    p,_=plan();p['sampling']['enable_thinking']=True
    with pytest.raises(ValueError):validate_plan(p)


def example():return dict(example_id='SYNTHETIC',problem='RAW_SYNTHETIC_PROBLEM',reference='2')


def test_native_valid_wrong_stops_without_diagnostic_retry():
    provider=FakeProvider({('A','SYNTHETIC',0):[dict(text='Final answer: 3')]})
    result=fake_recovery('A',example(),0,provider)
    assert result['first_valid'] and not result['first_correct'] and result['semantic_draws']==1
    assert provider.calls[0]['request']['messages'][1]['content']==example()['problem']


def test_transport_retry_does_not_consume_semantic_draw_and_capacity_only_after_length():
    provider=FakeProvider({('A','SYNTHETIC',0):[
        [dict(error='TIMEOUT'),dict(text='Final answer: 2',finish_reason='length')],
        dict(text='No conclusion.',finish_reason='stop'),dict(text='Final answer: 2')]})
    result=fake_recovery('A',example(),0,provider)
    assert result['semantic_draws']==3 and result['transport_retries']==1
    assert [r['request']['max_tokens'] for r in provider.calls]==[3600,3600,6144,3600]
    assert not result['first_valid'] and result['recovered_correct']


def test_four_invalid_draws_are_complete_recovery_and_transport_exhaustion_is_not():
    provider=FakeProvider({('A','SYNTHETIC',0):[dict(text='No conclusion.')]*4})
    result=fake_recovery('A',example(),0,provider)
    assert result['semantic_draws']==4 and not result['recovered_valid']
    failures=FakeProvider({('A','SYNTHETIC',0):[[dict(error='TIMEOUT')]*21]})
    with pytest.raises(ValueError,match='INCOMPLETE'):fake_recovery('A',example(),0,failures)


def test_five_realization_lanes_have_separate_identities():
    provider=FakeProvider({('A','SYNTHETIC',m):[dict(text='Final answer: '+str(m))] for m in range(5)})
    for m in range(5):fake_recovery('A',example(),m,provider,stage='stage2')
    assert len(provider.calls)==5
    assert len({r['realization_identity'] for r in provider.calls})==5
    with pytest.raises(ValueError,match='NO_REAL_EXECUTION_PORT'):fake_recovery('A',example(),0,object())


def test_budget_includes_all_recovery_transport_and_capacity_and_separate_stages():
    p,selected=plan()
    historical=dict(reported_success_tokens=209860,logical_examples=60,successful_responses=192,transport_timeouts=2,conservative_initial_timeout_charge=21678,
        rounds=[dict(reported_tokens=53092,input_tokens=6790,output_tokens=46302)])
    budget=budget_artifact(p,selected,historical)
    a,b=budget['stages']['stage1'],budget['stages']['stage2']
    assert (a['logical_requests'],a['successful_semantic_responses_upper'],a['transport_attempts_upper'])==(180,720,15120)
    assert a['successful_recovered_output_tokens_upper']==180*(3600+3*6144)
    from multi_dataset_diverse_rl.governance.token_accounting import reservation
    expected=reservation(request_for('A',selected[0]['content']['problem'],6144))['input_upper_bound']
    assert budget['input_reservation_by_arm']['A']['min']==expected
    assert a['all_transport_attempts_conservatively_charged_envelope']==21*a['no_transport_failure_reserved_envelope']
    assert b['logical_requests']==900 and b['transport_attempts_upper']==75600
    assert not budget['old_2m_scope_reusable'] and not a['saturation_or_complete_panel_guaranteed']


def test_paired_improved_regressed_and_incomplete_membership_rejected():
    left={str(i):dict(first_valid=False,first_correct=False,recovered_valid=False,recovered_correct=False) for i in range(2)}
    right=deepcopy(left);right['0']['first_valid']=True
    result=paired_comparison(left,right)
    assert result['metrics']['first_valid']==dict(improved=1,regressed=0,paired_delta=.5)
    with pytest.raises(ValueError):paired_comparison(left,{'0':right['0']})


def test_command_fails_closed_without_preimport_guard(monkeypatch):
    from scripts.audit_math_baseline_calibration import require_zero_api_guard
    monkeypatch.delenv('FORMAL_ZERO_API_GUARD_REQUIRED',raising=False)
    with pytest.raises(RuntimeError,match='PREIMPORT'):require_zero_api_guard()
