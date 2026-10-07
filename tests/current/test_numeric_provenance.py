"""Admissibility positives, leakage negatives and frozen amendment controls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from multi_dataset_diverse_rl import versions as historical, current_contract as current
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.search.numeric_provenance import numeric_content_leaked,numeric_content_leaked_v5
from multi_dataset_diverse_rl.search.textual_gradients import validate_gradient,GRADIENT_PROMPT,POLICY
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT=Path(__file__).resolve().parents[2]
BP='experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v2.json'
def row(problem='A synthetic quantity is given.',gold='1',prediction='0'):
    return NS(example_id='synthetic',signals=dict(input_payload=problem,gold=gold,target_output=prediction))


@pytest.mark.parametrize('gradient',[
    'Check that probabilities sum to 1.',
    'Check whether a denominator is 0 before dividing.',
    'Check that primitive integer coefficients have greatest common divisor 1.',
    'Check that probabilities sum to one.',
    'Use 2 independent derivations to verify consistency.',
    'Check whether a denominator is 1.000 before simplifying.',
])
def test_generic_structural_values_pass_despite_source_overlap(gradient):
    r=row('The given quantities are 0, 1, 2 and 1.000.',gold='1.000',prediction='0')
    assert not numeric_content_leaked(gradient,(r,))
    assert validate_gradient(gradient,(r,))==gradient
    assert validate_gradient(gradient,(r,),generalized=True)==gradient


@pytest.mark.parametrize('problem,gold,prediction,gradient',[
    ('There are 123 items.','x','y','Use constant 123 when transforming the expression.'),
    ('There are 123 items.','x','y','Use constant one hundred and twenty-three when transforming the expression.'),
    ('A measured value is 0.125.','x','y','Use 0.125 in the transformation.'),
    ('A measured value is 0.125.','x','y','Use 1.25e-1 in the transformation.'),
    ('There are 1,234 items.','x','y','Use １２３４ in the transformation.'),
    ('A fraction is \\frac{3}{7}.','x','y','Use the ratio 3/7.'),
    ('A fraction is 3/7.','x','y','Use the ratio \\frac{3}{7}.'),
    ('A symbolic example.','42','x','The answer is forty-two.'),
    ('A symbolic example.','x','17','The answer is 17.'),
    ('A symbolic example.','1','x','The answer is one.'),
    ('There are 7 students.','x','y','Count seven students in each selection.'),
    ('The given length is 7.','x','y','Use the given length 7.'),
    ('The interval lasts 1 hour.','x','y','Use one hour as the duration.'),
    ('The denominator is 0.','x','y','Use the given denominator 0.'),
    ('Solve 3x+5=14.','x','y','Transform 3x + 5 = 14 before simplifying.'),
])
def test_clear_numeric_provenance_rejected_in_both_boundaries(problem,gold,prediction,gradient):
    r=row(problem,gold,prediction)
    assert numeric_content_leaked_v5(gradient,(r,))
    strong=not gradient.startswith(('Use constant ','Use 0.125','Use 1.25e-1','Use １２３４','Use the ratio '))
    assert numeric_content_leaked(gradient,(r,)) is strong
    if not strong:
        assert validate_gradient(gradient,(r,))==gradient
        return
    for generalized in (False,True):
        with pytest.raises(SearchContractError,match='GRADIENT_.*_INVALID'):
            validate_gradient(gradient,(r,),generalized=generalized)


def test_small_unanchored_overlap_is_not_a_provenance_claim():
    assert not numeric_content_leaked('Check 7 candidate transformations for consistency.',
        (row('There are 7 apples.','x','y'),))
    assert not numeric_content_leaked('Use constant 123.',(row('A symbolic example.','x','y'),))


@pytest.mark.parametrize('gold',['one','zero','two','\\frac{1}{2}'])
def test_generic_numeric_rule_with_written_numeric_reference(gold):
    gradient='Check that probabilities sum to one and use 2 independent derivations.'
    assert validate_gradient(gradient,(row('A symbolic example.',gold,'x'),))==gradient


def test_generalized_gradient_checks_all_original_wrong_sources():
    rows=(row('A symbolic example.','x','y'),row('The given value is 123.','z','w'))
    with pytest.raises(SearchContractError,match='CLUSTER_INVALID'):
        validate_gradient('Use the given value 123.',rows,generalized=True)


def test_guard_prompt_identities_and_single_generation_are_versioned():
    assert historical.PATTERN_CONSTRAINT_ENTITY_GUARD_VERSION=='PATTERN_ABSTRACTION_SPECIFIC_CONTENT_GUARD_V4'
    assert historical.GRADIENT_PROMPT_VERSION=='PER_EXAMPLE_TEXTUAL_GRADIENT_PROMPT_V2'
    assert current.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION=='PATTERN_ABSTRACTION_SPECIFIC_CONTENT_GUARD_V6'
    assert historical.PATTERN_NUMERIC_PROVENANCE_GUARD_VERSION=='PATTERN_ABSTRACTION_SPECIFIC_CONTENT_GUARD_V5'
    assert current.GRADIENT_PROMPT_VERSION=='PER_EXAMPLE_TEXTUAL_GRADIENT_PROMPT_V3'
    assert 'problem-specific numeric constants' in GRADIENT_PROMPT
    assert 'Generic mathematical constants or structural quantities are allowed only when' in GRADIENT_PROMPT
    g=POLICY['gradient_policy']
    assert g['max_characters']==400 and g['successful_generations_per_wrong']==1
    assert g['semantic_regeneration'] is False and g['logical_calls_per_wrong']==1


def test_fresh_pilot_preserves_all_other_fields_and_closes_old_treatment():
    c=json.loads((ROOT/BP).read_bytes());parent=json.loads((ROOT/c['numeric_parent_binding_path']).read_bytes())
    from multi_dataset_diverse_rl.benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
    assert not MATHGradientPatternBinding(ROOT,c).blockers()
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):execution_binding(ROOT,c)
    for key in ('models','provider_bounds','decoding','memory_limits','shared_risk_policy',
            'layer1_search_policy','low_cost_protocol','membership_hashes','initial_team_sha256',
            'initial_team_artifact_sha256','post_search_validation_policy','pattern_prompt_sha256',
            'stop_policy','pilot_observation_policy','search_only_scope'):
        assert c[key]==parent[key]
    old_policy=deepcopy(c['pattern_policy'])
    old_policy['gradient_policy'].update(prompt_identity=historical.GRADIENT_PROMPT_VERSION,
        abstraction_guard=historical.PATTERN_CONSTRAINT_ENTITY_GUARD_VERSION)
    assert old_policy==parent['pattern_policy']
    assert c['execution_attempt_id']!=parent['execution_attempt_id'] and c['cache_namespace']!=parent['cache_namespace']
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        execution_binding(ROOT,parent)


@pytest.mark.parametrize('key,value',[
    ('numeric_admissibility_amendment_sha256','0'*64),('numeric_parent_binding_sha256','0'*64),
    ('numeric_user_scope_sha256','0'*64),('gradient_prompt_sha256','0'*64),
    ('initial_team_sha256','0'*64),('models',{}),('provider_bounds',{}),
    ('cache_namespace','old_attempt'),('execution_phase','canary'),('pilot_observation_policy','unknown')])
def test_treatment_mutations_fail_before_dispatch(key,value):
    c=json.loads((ROOT/BP).read_bytes());c[key]=value
    from multi_dataset_diverse_rl.benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
    assert MATHGradientPatternBinding(ROOT,c).blockers()[0].startswith('CURRENT_NUMERIC_ADMISSIBILITY_')
