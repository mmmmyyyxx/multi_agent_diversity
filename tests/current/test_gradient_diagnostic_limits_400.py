"""Current limits, unchanged safety, and fresh identity through fake providers."""
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
import subprocess
import sys
from pathlib import Path

import pytest

from multi_dataset_diverse_rl import versions, current_contract as current
from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
from multi_dataset_diverse_rl.search.schemas import SearchContractError, SearchMethodConfig
from multi_dataset_diverse_rl.search.textual_gradients import (
    GradientExtractor, PerExampleGradientProvider, REFERENCE_GRADIENT_PROMPT, GRADIENT_OUTPUT_LIMITS)
from tests.current.population_support import graph, contract, response, BASE, HYPOTHESIS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def observed_row(tmp_path_factory):
    run, broker, _, _ = graph(tmp_path_factory.mktemp('diagnostic_400'))
    state = run.state.snapshot()
    diagnosis = run.analyzer.analyze(state, run.history)
    row = run.opportunities.source.for_member(state, diagnosis, 0)[0]
    return broker, row


def output(**changes):
    value = dict(disposition='ACTIONABLE', observed_failure='A wrong observed result.',
        diagnosis='A repair hypothesis.', reusable_correction=HYPOTHESIS,
        suggested_block='strategy', expected_effect='Retain applicable constraints.')
    value.update(changes)
    return value


def extract(observed_row, value):
    broker, row = observed_row
    calls = []
    text = value if isinstance(value, str) else json.dumps(value)
    broker.transport = lambda req: (calls.append(deepcopy(req)) or response(text))
    extractor = GradientExtractor(PerExampleGradientProvider(broker))
    result = extractor.extract(BASE, (row,))
    return result, extractor.evidence_diagnostics[0], calls


@pytest.mark.parametrize('field', ['observed_failure', 'diagnosis'])
@pytest.mark.parametrize('length', [240, 241, 399, 400, 401])
def test_diagnostic_boundaries_use_original_unicode_characters(observed_row, field, length):
    result, diagnostic, calls = extract(observed_row, output(**{field: '观'*length}))
    if length <= 400:
        assert len(result) == len(calls) == 1
        assert diagnostic[field] == '观'*length
        assert result[0]['gradient'] == HYPOTHESIS
    else:
        assert not result and len(calls) == 3 and diagnostic['exhausted']


@pytest.mark.parametrize('changes', [dict(observed_failure=''), dict(diagnosis=' '),
    dict(observed_failure=None), dict(diagnosis=['synthetic']),
    dict(reusable_correction='a'*401), dict(expected_effect='a'*241),
    dict(suggested_block='both'), dict(extra='synthetic'),
    dict(reusable_correction='Copy the reference solution.'),
    dict(disposition='UNCERTAIN')])
def test_other_schema_length_and_content_guards_still_exhaust(observed_row, changes):
    value = dict(observed_failure='观'*400, diagnosis='d'*400)
    value.update(changes)
    result, diagnostic, calls = extract(observed_row, output(**value))
    assert not result and len(calls) == 3 and diagnostic['exhausted']


def test_malformed_json_remains_rejected_without_repair(observed_row):
    result, diagnostic, calls = extract(observed_row, '{"diagnosis":"bad\\escape"}')
    assert not result and len(calls) == 3 and diagnostic['exhausted']


def test_valid_uncertain_400_stops_once_without_a_pattern_correction(observed_row):
    result, diagnostic, calls = extract(observed_row, output(disposition='UNCERTAIN',
        observed_failure='观'*400, diagnosis='d'*400, reusable_correction=None, expected_effect=None))
    assert not result and len(calls) == 1 and diagnostic['disposition'] == 'UNCERTAIN'
    assert not diagnostic['exhausted']


def test_provenance_failure_is_fatal_before_transport(observed_row):
    broker, row = observed_row
    calls = []
    broker.transport = lambda req: (calls.append(req) or response(json.dumps(output())))
    corrupt = replace(row, signals={**row.signals, 'solver_trajectory': {
        **row.signals['solver_trajectory'], 'source': {
            **row.signals['solver_trajectory']['source'], 'example_id': 'synthetic_mismatch'}}})
    with pytest.raises(SearchContractError, match='PROVENANCE'):
        GradientExtractor(PerExampleGradientProvider(broker)).extract(BASE, (corrupt,))
    assert not calls


def test_formal_policy_carries_limits_and_rejects_unfrozen_tampering():
    c = contract()
    policy = c['pattern_policy']['gradient_policy']
    assert policy['output_limits'] == GRADIENT_OUTPUT_LIMITS
    assert policy['prompt_identity'] == current.GRADIENT_PROMPT_VERSION
    assert c['identity'] != versions.MATH_ROLE_ROUTED_GENERATION_RECOVERY_BINDING_VERSION
    CURRENT_POLICY_BUNDLE.validate_contract(c)
    c['pattern_policy']['gradient_policy']['output_limits']['diagnosis'] = 480
    with pytest.raises(SearchContractError, match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        CURRENT_POLICY_BUNDLE.validate_contract(c)


def test_method_identity_includes_limits_and_rejects_pre_amendment_method():
    c = contract()
    method = CURRENT_POLICY_BUNDLE.method(aggregation=versions.EQUIVALENCE_PLURALITY_VERSION,
        provider_binding='0'*64, successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
        solver_trajectory_policy=c['solver_trajectory_policy'], optimization_evidence_policy=c['optimization_evidence_policy'])
    old = method.identity()
    method.mechanism_config['pattern_policy']['gradient_policy']['output_limits']['diagnosis'] = 480
    assert method.identity() != old
    with pytest.raises(SearchContractError):
        CURRENT_POLICY_BUNDLE.validate_method(method)
    payload = asdict(method)
    payload['method'] = versions.UNIFIED_GENERATION_RECOVERY_VERSION
    with pytest.raises(SearchContractError, match='unsupported unified method identity'):
        SearchMethodConfig.from_mapping(payload)


def test_old_binding_and_cache_never_become_current_execution():
    old = json.loads((ROOT/'experiments/execution_bindings/a4_v25_arm_b_openlux_seed84_pilot_attempt1.json').read_bytes())
    with pytest.raises(SearchContractError):
        CURRENT_POLICY_BUNDLE.validate_contract(old)
    c = contract()
    c['cache_policy'] = 'ROLE_ROUTED_RECOVERY_MEMBER_LANE_CACHE_V6'
    with pytest.raises(SearchContractError, match='CURRENT_STRUCTURED_SYSTEM_POLICY_REQUIRED'):
        CURRENT_POLICY_BUNDLE.validate_contract(c)


def test_current_prompt_names_each_field_limit_and_rejects_stale_prompt(observed_row):
    assert 'observed_failure is at most\n400 characters; diagnosis at most 400' in REFERENCE_GRADIENT_PROMPT
    assert 'reusable_correction\nat most 400; expected_effect at most 240' in REFERENCE_GRADIENT_PROMPT
    broker, _ = observed_row
    with pytest.raises(SearchContractError, match='PROMPT_MISMATCH'):
        PerExampleGradientProvider(broker, prompt=REFERENCE_GRADIENT_PROMPT.replace('diagnosis at most 400', 'diagnosis at most 240'))


def test_fake_full_graph_accepts_400_diagnostics_on_first_draw_and_isolates_solver(tmp_path):
    run, broker, requests, _ = graph(tmp_path)
    original = broker.transport
    def transport(req):
        value = original(req)
        if req['model'] != 'gpt-4o-mini' and len(req['messages']) == 2:
            packet = json.loads(req['messages'][1]['content'])
            if 'example' in packet:
                sentinel = 'SYNTHETIC_DIAGNOSTIC_SENTINEL_'
                value['text'] = json.dumps(output(observed_failure=sentinel+'x'*(400-len(sentinel)),
                    diagnosis='d'*400))
        return value
    broker.transport = transport
    asyncio.run(run.run(max_opportunities=1))
    diagnostics = run.opportunities.patterns.extractor.evidence_diagnostics
    assert diagnostics and all(d['structural_draws'] == 1 and not d['exhausted'] for d in diagnostics)
    for req in requests:
        if req['model'] == 'gpt-4o-mini':
            assert 'SYNTHETIC_DIAGNOSTIC_SENTINEL' not in json.dumps(req)
        elif len(req['messages']) == 2 and 'gradients' in json.loads(req['messages'][1]['content']):
            assert 'SYNTHETIC_DIAGNOSTIC_SENTINEL' not in json.dumps(req)


@pytest.mark.parametrize('module', ['multi_dataset_diverse_rl.search.schemas', 'multi_dataset_diverse_rl.current_contract'])
def test_closed_identity_imports_work_in_either_entry_order(module):
    subprocess.run([sys.executable, '-c', 'import '+module], cwd=ROOT, check=True, capture_output=True)


def preparation_manifest():
    from multi_dataset_diverse_rl.governance.registries import load_yaml
    frontier = load_yaml(ROOT/'experiments/current_frontier.yaml')
    return load_yaml(ROOT/frontier['current_preparation_manifest'])


def test_current_preparation_is_strict_and_cannot_import_historical_measurement():
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    m = preparation_manifest()
    assert not validate_manifest_v2(ROOT, m)
    assert m['method_identity'] == current.CURRENT_METHOD_VERSION
    assert m['mechanism_config']['pattern_policy'] == contract()['pattern_policy']
    assert m['lifecycle']['status'] == 'DRAFT' and m['source_sha'] is None
    assert not m['authorization']['real_api_authorized'] and m['authorization']['attempt_id'] is None
    assert m['execution_binding']['path'] is None and m['execution_binding']['sha256'] is None
    assert all(v is None for v in m['hash_closure'].values())
    assert m['split_identity'] is None and m.get('initial_competence_binding') is None


@pytest.mark.parametrize('field', ['observed_failure', 'diagnosis', 'reusable_correction', 'expected_effect'])
def test_manifest_rejects_unregistered_field_limits(field):
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    m = preparation_manifest()
    m['mechanism_config']['pattern_policy']['gradient_policy']['output_limits'][field] += 1
    assert validate_manifest_v2(ROOT, m)


def test_historical_manifest_schema_branches_keep_their_original_contracts():
    old = json.loads(subprocess.check_output(['git', 'show',
        'cb4e19bca664c9ea5d8aad0ba9396ae3fe266592:experiments/schema/experiment_manifest_v2.schema.json'], cwd=ROOT))
    now = json.loads((ROOT/'experiments/schema/experiment_manifest_v2.schema.json').read_bytes())
    assert now['oneOf'][1:len(old['oneOf'])] == old['oneOf'][1:]
    fallback = deepcopy(now['oneOf'][0])
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(current.CURRENT_METHOD_VERSION)
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(versions.UNIFIED_GROUP_DIAGNOSED_VERSION)
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(versions.UNIFIED_WEIGHTED_RESPONSIBILITY_VERSION)
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(versions.UNIFIED_VOTE_POPULATION_VERSION)
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(versions.UNIFIED_SCOPED_EXPERIENCE_VERSION)
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(
        versions.UNIFIED_REQUEST_RECOVERY_VERSION)
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(
        versions.UNIFIED_OUTPUT_CONTRACT_FEEDBACK_VERSION)
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(
        'unified_team_prompt_search_v2_5_diagnostic_limits_400_v2')
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(
        'unified_team_prompt_search_v2_6_parallel_compact_v1')
    fallback['allOf'][0]['not']['properties']['method_family']['enum'].remove(
        'unified_team_prompt_search_v2_6_readable_evidence_v2')
    assert fallback == old['oneOf'][0]


def test_new_normative_limits_are_in_both_source_hash_closures():
    from multi_dataset_diverse_rl.governance.source_identity import current_scientific_files
    for execution_closure in (False, True):
        files = current_scientific_files(ROOT, execution_closure=execution_closure)
        assert ROOT/'docs/design/GRADIENT_DIAGNOSTIC_LIMITS_400_V2.md' in files
