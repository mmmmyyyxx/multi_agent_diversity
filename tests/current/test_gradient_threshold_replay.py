"""Counterfactual isolation and local retry accounting, without private inputs."""
import ast
from pathlib import Path
import subprocess
import pytest

from multi_dataset_diverse_rl.diagnostics.gradient_threshold_replay import (
    threshold_tree, counterfactual_summary)


SOURCE = """def validate(value):
    if any(not isinstance(value[k], str) or not value[k].strip() or len(value[k]) > 240
           for k in ('observed_failure', 'diagnosis')):
        return False
    return len(value['expected_effect']) <= 240 and len(value['reusable_correction']) <= 400
"""


def validator(a, b):
    scope = {}
    exec(compile(threshold_tree(SOURCE, a, b), '<test>', 'exec'), scope)
    return scope['validate']


def payload(a=240, b=240, correction=400, effect=240):
    return dict(observed_failure='观'*a, diagnosis='d'*b,
        reusable_correction='c'*correction, expected_effect='e'*effect)


@pytest.mark.parametrize('limit', [240, 320, 400, 480, 640])
def test_unicode_limits_are_inclusive_without_changing_other_fields(limit):
    validate = validator(limit, limit)
    assert validate(payload(limit, limit))
    assert not validate(payload(limit+1, limit))
    assert not validate(payload(limit, limit+1))
    assert not validate(payload(limit, limit, correction=401))
    assert not validate(payload(limit, limit, effect=241))


def test_fields_can_be_replayed_independently_and_empty_fields_stay_invalid():
    validate = validator(320, 480)
    assert validate(payload(320, 480))
    assert not validate(payload(321, 480))
    assert not validate(payload(320, 481))
    assert not validate(payload(0, 480))
    assert not validate(payload(320, 0))


@pytest.mark.parametrize('source', [SOURCE.replace('> 240', '> 400'),
    SOURCE.replace("('observed_failure', 'diagnosis')", "('expected_effect', 'diagnosis')"),
    SOURCE+'\n'+SOURCE, 'def validate(value): return True'])
def test_unrecognized_or_multiple_source_comparisons_fail_closed(source):
    with pytest.raises(ValueError, match='FROZEN_TWO_FIELD_COMPARISON_MISMATCH'):
        threshold_tree(source, 400, 400)


@pytest.mark.parametrize('limit', [True, 241, 1000, 400.0])
def test_only_registered_caps_can_be_evaluated(limit):
    with pytest.raises(ValueError, match='UNREGISTERED_DIAGNOSTIC_THRESHOLD'):
        threshold_tree(SOURCE, limit, 400)


def test_frozen_source_changes_only_the_registered_comparison():
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(['git', 'show',
        '01306fbf9cde002291ab21e2f93a2c02399f7e9e:multi_dataset_diverse_rl/search/textual_gradients.py'], cwd=root).decode()
    revised = threshold_tree(source, 320, 480)
    # Removing the one substituted lookup must recover the entire original AST.
    substituted = [n for n in ast.walk(revised) if isinstance(n, ast.Compare)
        and isinstance(n.comparators[0], ast.Subscript)
        and isinstance(n.comparators[0].value, ast.Dict)
        and ast.unparse(n.left) == 'len(value[k])']
    assert len(substituted) == 1
    substituted[0].comparators[0] = ast.Constant(value=240)
    assert ast.dump(revised, include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)


def test_current_formal_policy_cannot_be_mistaken_for_the_frozen_240_source():
    source = (Path(__file__).resolve().parents[2]/'multi_dataset_diverse_rl/search/textual_gradients.py').read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='FROZEN_TWO_FIELD_COMPARISON_MISMATCH'):
        threshold_tree(source, 400, 400)


def draw(original=False, accepted=False, cost=100, disposition='ACTIONABLE'):
    return dict(original_accepted=original, accepted=accepted, input_tokens=cost-10,
        output_tokens=10, disposition=disposition)


def test_savings_only_include_suffix_not_the_newly_accepted_draw():
    result = counterfactual_summary([[draw(accepted=True, cost=100), draw(cost=200),
        draw(original=True, accepted=True, cost=300)]])
    assert result['all_response_full_pass'] == 2
    assert result['accepted_diagnoses'] == 1
    assert result['avoidable_historical_retry_responses'] == 2
    assert result['avoidable_usage_tokens'] == 500
    assert result['avoidable_input_tokens'] == 480
    assert result['rescued_exhausted_diagnoses'] == 0


def test_last_draw_rescue_adds_coverage_without_saving_a_recorded_response():
    result = counterfactual_summary([[draw(), draw(), draw(accepted=True)]])
    assert result['rescued_exhausted_diagnoses'] == 1
    assert result['avoidable_usage_tokens'] == 0
    assert result['avoidable_historical_retry_responses'] == 0


def test_uncertain_is_full_valid_but_not_actionable_and_baseline_saves_nothing():
    result = counterfactual_summary([[draw(), draw(original=True, accepted=True, disposition='UNCERTAIN')],
        [draw(), draw(), draw()]])
    assert result['uncertain_diagnoses'] == 1
    assert result['actionable_diagnoses'] == 0
    assert result['remaining_exhausted_diagnoses'] == 1
    assert result['avoidable_usage_tokens'] == 0


@pytest.mark.parametrize('groups', [[[]], [[draw()]],
    [[draw(original=True, accepted=False)]],
    [[draw(original=True, accepted=True), draw()]],
    [[draw(), draw(), draw(), draw()]],
    [[dict(draw(original=True, accepted=True), input_tokens=-1)]],
    [[dict(draw(original=True, accepted=True), accepted=1)]]])
def test_invalid_observation_sequences_fail_closed(groups):
    with pytest.raises(ValueError):
        counterfactual_summary(groups)
