"""Diagnostic separation and retry attribution; no production controls."""
import json
import pytest
from multi_dataset_diverse_rl.diagnostics.pilot_observations import (
    opportunity_streaks,gradient_shape_annotation,gradient_cost_summary)


def valid_gradient(**changes):
    value=dict(disposition='ACTIONABLE',observed_failure='A wrong conclusion.',
        diagnosis='A hypothesis.',reusable_correction='Check every branch before simplifying.',
        suggested_block='strategy',expected_effect='Preserve all legal branches.')
    value.update(changes)
    return json.dumps(value)


def test_target_only_commit_and_rejected_candidate_have_different_streaks():
    result=opportunity_streaks([
        dict(committed=True,parent_vote=10,child_vote=10),
        dict(committed=False,parent_vote=10,child_vote=10),
        dict(committed=True,parent_vote=10,child_vote=11)])
    assert [(r['consecutive_no_commit_opportunities'],r['consecutive_no_vote_improvement_opportunities'])
        for r in result['rows']]==[(0,1),(1,2),(0,0)]
    assert result['max_no_commit']==1 and result['max_no_vote_improvement']==2
    assert result['scientific_stop_inference'].startswith('NONE_')


@pytest.mark.parametrize('rows',[
    [dict(committed=False,parent_vote=2,child_vote=3)],
    [dict(committed=True,parent_vote=2,child_vote=3),dict(committed=False,parent_vote=2,child_vote=2)],
    [dict(committed=1,parent_vote=2,child_vote=2)],
    [dict(committed=True,parent_vote=2,child_vote=float('nan'))]])
def test_streaks_fail_on_inconsistent_or_nonfinite_observation(rows):
    with pytest.raises(ValueError):opportunity_streaks(rows)


def test_shape_diagnostics_keep_primary_and_overlapping_length_issues_separate():
    annotation=gradient_shape_annotation(valid_gradient(observed_failure='x'*241,
        diagnosis='y'*250,reusable_correction='z'*410),'stop')
    assert annotation['primary']=='OBSERVED_FAILURE_OVER_240'
    assert annotation['issues']==['OBSERVED_FAILURE_OVER_240','DIAGNOSIS_OVER_240','REUSABLE_CORRECTION_OVER_400']
    assert gradient_shape_annotation(valid_gradient(observed_failure='x'*240,
        diagnosis='y'*240,reusable_correction='z'*400),'stop')['primary']=='REQUIRES_FROZEN_CONTENT_VALIDATION'


@pytest.mark.parametrize('text,finish,category',[
    ('{"diagnosis":"\\alpha"}','stop','JSON_INVALID_BACKSLASH_ESCAPE'),
    ('```json\n{}\n```','stop','JSON_FENCED_ENVELOPE'),
    (valid_gradient(),'length','NONSTOP_PROVIDER_FINISH'),
    ('null','stop','SCHEMA_KEYS_OR_OBJECT'),
    (valid_gradient(suggested_block=['strategy']),'stop','SCHEMA_BLOCK'),
    (valid_gradient(suggested_block='Strategy'),'stop','SCHEMA_BLOCK_CASE'),
    (valid_gradient(disposition='UNCERTAIN'),'stop','SCHEMA_UNCERTAIN_NULLS')])
def test_diagnostic_never_repairs_or_accepts_schema_content(text,finish,category):
    assert gradient_shape_annotation(text,finish)['primary']==category


def draw(accepted,category,input_tokens=100,output_tokens=10):
    return dict(accepted=accepted,category=category,issues=[],finish_reason='stop',
        input_tokens=input_tokens,output_tokens=output_tokens)


def test_retry_cost_belongs_to_previous_rejection_and_first_draw_is_excluded():
    result=gradient_cost_summary([
        [draw(False,'LENGTH'),draw(False,'JSON'),draw(True,'ACCEPTED',200,20)],
        [draw(True,'ACCEPTED')],
        [draw(False,'CONTENT')]*3])
    assert result['logical_gradients']==3 and result['physical_draws']==7
    assert result['first_pass_accepted']==1 and result['recovered']==1 and result['exhausted']==1
    assert result['extra_draws']==4 and result['extra_usage_tokens']==550
    assert result['retry_usage_by_preceding_rejection']=={'LENGTH':110,'JSON':220,'CONTENT':220}
    assert result['exhausted_total_usage_tokens']==330 and result['exhausted_extra_usage_tokens']==220
    assert result['successful_recovery_extra_usage_tokens']==330


@pytest.mark.parametrize('groups',[
    [[draw(True,'ACCEPTED'),draw(False,'JSON')]],
    [[draw(False,'JSON')]],
    [[draw(False,'JSON',-1)]],
    [[draw(True,'ACCEPTED',True)]]])
def test_cost_diagnostics_reject_incomplete_or_inconsistent_draws(groups):
    with pytest.raises(ValueError):gradient_cost_summary(groups)
