"""Synthetic fixtures for offline compatibility, sampling and five-member votes."""
from copy import deepcopy
import pytest

from multi_dataset_diverse_rl.diagnostics.math_baseline_followup import (
    display_box_diagnostic, grade_compatibility, choose_fresh_membership,
    initial_team, vote_selection, panel_metrics, CAP,
)
from multi_dataset_diverse_rl.diagnostics.math_baseline_calibration import native_extract,primary_correctness
from multi_dataset_diverse_rl.diagnostics.math_calibration_plan import ROLE,STRATEGY,ANSWERS


@pytest.mark.parametrize('text', [
    '\\[\n\\boxed{\\frac{1}{2}}\n\\]',
    '$$\n\\boxed{\\frac{1}{2}}\n$$',
    '\\[\n\\boxed{\n\\frac{1}{2}\n}\n\\]',
    '\\boxed{\\frac{1}{2}}',
])
def test_gold_blind_display_compatibility_preserves_official_score(text):
    old=native_extract('C',text)
    probe=display_box_diagnostic(text)
    assert probe['valid']
    assert grade_compatibility(probe,'1/2') is True
    assert grade_compatibility(probe,'2') is False
    assert native_extract('C',text)==old
    with pytest.raises(ValueError,match='DIAGNOSTIC_CANNOT'):
        primary_correctness(probe,'1/2')
    with pytest.raises(TypeError):
        display_box_diagnostic(text,reference='1/2')


@pytest.mark.parametrize('text,finish', [
    ('\\[\n\\boxed{2}\n\\]','length'),
    ('\\[\n\\boxed{2}\n\\]','tool_calls'),
    ('\\[\n\\boxed{2}\n\\]\nExplanation.','stop'),
    ('\\[\n\\boxed{2}\n','stop'),
    ('\\[\n\\boxed{2}.\n\\]','stop'),
    ('\\[\n\\boxed{2} \\boxed{3}\n\\]','stop'),
    ('\\[\nThus \\boxed{2}\n\\]','stop'),
    ('Final answer: 3\n\\[\n\\boxed{2}\n\\]','stop'),
    ('```\n\\[\n\\boxed{2}\n\\]\n```','stop'),
    ('```\n\\[\n\\boxed{2}\n\\]','stop'),
])
def test_rejects_nonterminal_malformed_conflicting_and_unfinished(text,finish):
    assert not display_box_diagnostic(text,finish)['valid']


def synthetic_rows():
    return [dict(stable_example_id=str(i),content=dict(type='Algebra',level='Level 1')) for i in range(150)]


def test_fresh_membership_is_disjoint_fixed_order_and_fails_on_shortage():
    rows=synthetic_rows();excluded={str(i) for i in range(72)}
    chosen,meta=choose_fresh_membership(rows,excluded)
    assert len(chosen)==60 and meta['seed']==82
    assert all(r['stable_example_id'] not in excluded for r in chosen)
    assert chosen==choose_fresh_membership(rows,excluded)[0]
    assert [int(r['stable_example_id']) for r in chosen]==sorted(int(r['stable_example_id']) for r in chosen)
    with pytest.raises(ValueError,match='INSUFFICIENT'):
        choose_fresh_membership(rows,{str(i) for i in range(91)})


def test_b_team_has_five_identical_unmodified_role_and_strategy():
    team=initial_team()
    assert team['ordered_member_ids']==list(range(5))
    assert all(m['prompt']==dict(role=ROLE,strategy=STRATEGY,answer=ANSWERS['B']) for m in team['members'])
    assert len({m['prompt_sha256'] for m in team['members']})==1
    assert CAP==1000000


def native(text):
    return native_extract('B',text)


def test_vote_ties_abstain_oracle_and_unique_coverage_remain_separate():
    outputs=[native('Final answer: '+s) for s in ('1','1','2','2','3')]
    metrics=panel_metrics(outputs,'3')
    assert metrics['selection']['winner_member'] is None
    assert not metrics['vote_correct'] and metrics['oracle_correct']
    assert metrics['unique_cover_member']==4
    assert vote_selection(outputs)==metrics['selection']
    outputs[-1]=native('No result.')
    assert panel_metrics(outputs,'1')['selection']['reason']=='TIE'


def test_equivalence_plurality_uses_lowest_member_and_no_gold_to_select():
    outputs=[native('Final answer: '+s) for s in ('0.5',r'\frac{1}{2}','2','2','1/2')]
    assert vote_selection(outputs)['winner_member']==0
    assert panel_metrics(outputs,'1/2')['vote_correct']
    assert not panel_metrics(outputs,'2')['vote_correct']
    with pytest.raises(TypeError):vote_selection(outputs,reference='2')
    with pytest.raises(ValueError,match='FIVE_MEMBERS'):
        vote_selection(outputs[:4])
    changed=deepcopy(outputs);changed[0]['usage_scope']='DIAGNOSTIC_ONLY'
    with pytest.raises(ValueError,match='B_NATIVE_ONLY'):vote_selection(changed)


def response(text,finish='stop'):
    usage=dict(prompt_tokens=10,completion_tokens=20)
    body=dict(model='qwen3-8b',usage=usage,
        choices=[dict(message=dict(content=text),finish_reason=finish)])
    return dict(text=text,finish_reason=finish,input_tokens=10,output_tokens=20,provider_http_response_body=body)


def test_five_member_runner_rotates_independent_lanes_and_recovers_only_invalid(tmp_path):
    import json
    from multi_dataset_diverse_rl.diagnostics.math_b5_execution import run_panel
    from multi_dataset_diverse_rl.diagnostics.math_b5_accounting import read_ledger
    selected=[dict(stable_example_id=str(i),reference_final_answer='2',
        content=dict(problem='Synthetic arithmetic.',type='Algebra',level='Level 1')) for i in range(2)]
    scope=dict(startup_identity_sha256='s',attempt_id='fake-only',source_sha='fake-source')
    calls=[]
    def provider(request):
        calls.append(deepcopy(request))
        return response('No result.','length') if len(calls)==1 else response('Final answer: 3')
    result=run_panel(selected,scope,tmp_path/'run',provider,sleep=lambda _:None,progress=lambda _:None)
    assert result['status']=='EXECUTION_COMPLETE' and len(calls)==11
    assert [r['max_tokens'] for r in calls[:3]]==[3600,6144,3600]
    cp=json.loads((tmp_path/'run/checkpoint_private.json').read_bytes())['completed']
    assert [r['member'] for r in cp]==[0,1,2,3,4,1,2,3,4,0]
    assert cp[0]['semantic_draws']==2 and all(r['semantic_draws']==1 for r in cp[1:])
    ledger=read_ledger(tmp_path/'run/accounting')
    assert ledger['physical_attempts']==11 and ledger['charged_total']==330 and ledger['reserved_total']==0
    assert ledger['policy_identity']=='FRESH_MATH_B_FIVE_MEMBER_ACCOUNTING_V1'
    from multi_dataset_diverse_rl.diagnostics.math_b5_results import summarize_complete
    events=[json.loads(line) for line in (tmp_path/'run/accounting/events.jsonl').read_bytes().splitlines()]
    lanes={e['reservation_id']:e['lane'] for e in events if e['kind']=='RESERVE'}
    charges=[e|dict(lane=lanes[e['reservation_id']]) for e in events if e['kind']=='CHARGE']
    scored=summarize_complete(cp,ledger,charges)
    assert scored['phases']['first']['total_valid_count']==9
    assert scored['phases']['recovered']['total_valid_count']==10
    assert scored['phases']['recovered']['correct_coverage_histogram']['0']==2
    assert scored['cost']['format_recovery_charged_tokens']==30
    assert scored['format_stability_gate']=='PASSED'


def test_b5_ledger_never_reuses_directory_or_old_scope_and_charges_unknown(tmp_path):
    from multi_dataset_diverse_rl.diagnostics.math_b5_accounting import CalibrationLedger,CalibrationAbort,replay_events
    from multi_dataset_diverse_rl.diagnostics.math_calibration_plan import request_for
    ledger=CalibrationLedger(tmp_path/'accounting','new-b5-scope')
    request=request_for('B','Synthetic.')
    key=ledger.reserve(request,dict(member=0))
    bound=ledger.state['reserved_total']
    amount,reliable=ledger.charge(key,None,'TIMEOUT')
    assert amount==bound and reliable is False
    ledger.terminal('EXECUTION_ABORTED');ledger.close()
    with pytest.raises(CalibrationAbort,match='FRESH_CALIBRATION'):
        CalibrationLedger(tmp_path/'accounting','new-b5-scope')
    events=deepcopy(ledger.events);events[0]['policy_identity']='FRESH_MATH_BASELINE_STAGE1_ACCOUNTING_V1'
    from multi_dataset_diverse_rl.diagnostics.math_baseline_calibration import digest
    events[0]['event_sha256']=digest({k:v for k,v in events[0].items() if k!='event_sha256'})
    with pytest.raises(CalibrationAbort,match='SCOPE_MISMATCH'):
        replay_events(events)


def test_b5_paid_entry_fails_before_any_transport_without_frozen_review(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.diagnostics import math_b5_execution as execution
    called=[]
    monkeypatch.setattr(execution,'create_exact_transport',lambda _:called.append(True))
    with pytest.raises(FileNotFoundError):execution.execute_paid(tmp_path,tmp_path/'missing')
    assert not called
    from multi_dataset_diverse_rl.diagnostics.math_b5_accounting import CalibrationAbort
    monkeypatch.setattr(execution,'git',lambda root,*args:'frozen' if args[0]=='rev-parse' else ' M tracked.py')
    with pytest.raises(CalibrationAbort,match='SOURCE_CHANGED'):
        execution.check_dispatch_source(tmp_path,dict(source_sha='frozen',source_files=[]))


def test_authorization_cannot_reuse_stage1_or_change_cap(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.diagnostics import math_b5_execution as execution
    from multi_dataset_diverse_rl.diagnostics.math_b5_accounting import CalibrationAbort
    scope=dict(review_scope_sha256='new-review',source_sha='new-source',protocol_sha256='new-protocol',
        membership_sha256='new-membership',phase='b_five_member_development')
    monkeypatch.setattr(execution,'verify_review',lambda *a:(scope,{},[]))
    with pytest.raises(CalibrationAbort,match='FRESH_EXACT'):
        execution.authorize_review(tmp_path,tmp_path/'prep',dict(stage='stage1',cap=1450000,exact_user_message='old approval'))


def test_b5_resource_cap_stops_before_dispatch_and_does_not_claim_complete(tmp_path):
    from multi_dataset_diverse_rl.diagnostics.math_b5_accounting import CalibrationLedger,CalibrationAbort
    from multi_dataset_diverse_rl.diagnostics.math_calibration_plan import request_for
    ledger=CalibrationLedger(tmp_path/'accounting','new-scope')
    too_large=request_for('B','x'*1000000)
    with pytest.raises(CalibrationAbort,match='RESOURCE_CEILING'):ledger.reserve(too_large,dict(member=0))
    assert ledger.state['physical_attempts']==0
    ledger.terminal('EXECUTION_ABORTED');ledger.close()


def test_transport_retry_preserves_body_and_native_invalid_four_draw_stop(tmp_path):
    from multi_dataset_diverse_rl.diagnostics.math_b5_execution import run_panel
    selected=[dict(stable_example_id='synthetic',reference_final_answer='2',
        content=dict(problem='Synthetic.',type='Algebra',level='Level 1'))]
    scope=dict(startup_identity_sha256='fake',attempt_id='fake',source_sha='fake')
    calls=[];sleeps=[]
    class APITimeoutError(Exception):pass
    def provider(request):
        calls.append(deepcopy(request))
        if len(calls)==1:raise APITimeoutError()
        return response('No final declaration.')
    result=run_panel(selected,scope,tmp_path/'run',provider,sleep=sleeps.append,progress=lambda _:None)
    assert result['status']=='EXECUTION_COMPLETE' and len(calls)==21
    assert calls[0]==calls[1] and sleeps==[1.5]
    assert result['accounting']['unknown_usage_charge']>0
    assert result['accounting']['reserved_total']==0
