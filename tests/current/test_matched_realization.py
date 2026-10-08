"""Offline common-realization isolation and authorization regression tests."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from tests.current.test_visible_solver_trajectories import visible_contract
from tests.current.test_optimization_evidence_v23 import contract as evidence_contract
from multi_dataset_diverse_rl.governance.matched_realization import (
    paired_policy, group_scope, validate_policy, validate_authorization, cache_context)
from multi_dataset_diverse_rl.persistence.exact_output_cache import DurableExactOutputCache
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.schemas import SearchContractError


def contracts():
    attempts = dict(A='matched_synthetic_a', B='matched_synthetic_b')
    paths = dict(A='experiments/execution_bindings/matched_synthetic_a.json',
                 B='experiments/execution_bindings/matched_synthetic_b.json')
    cells = []
    for cell, c in [('A', visible_contract()), ('B', evidence_contract())]:
        c['execution_attempt_id'] = c['cache_namespace'] = attempts[cell]
        c['binding_path'] = paths[cell]
        c['operational_pilot'].update(max_opportunities=5, token_ceiling=2_000_000)
        c['provider_bounds']['max_opportunities'] = 5
        c['paired_realization_policy'] = paired_policy(c, group_id='synthetic_pair', cell=cell,
            attempts=attempts, binding_paths=paths, protocol_sha256='c' * 64)
        cells.append(c)
    return cells


def kwargs(**changes):
    return dict(role='solver', split='optimize', stage='initial', member_slot=0,
        messages=[dict(role='system', content='Synthetic immutable interface.'),
                  dict(role='user', content='Synthetic public input.')], **changes)


def test_pair_shares_only_identical_solver_realization_keys():
    a, b = contracts()
    left = RequestBroker(contract=a, transport=lambda _: pytest.fail('provider'), arm='A4', seed=81)
    right = RequestBroker(contract=b, transport=lambda _: pytest.fail('provider'), arm='A4', seed=81)
    values = kwargs(); values.pop('stage')
    assert left._request_identity(**values) == right._request_identity(**values)
    for changed in [dict(member_slot=1), dict(split='shadow'),
            dict(messages=[dict(role='user', content='Different synthetic public input.')])]:
        assert left._request_identity(**values)[1] != right._request_identity(**{**values, **changed})[1]
    r = dict(role='reflection', split='optimize', messages=[])
    assert left._request_identity(**r)[1] != right._request_identity(**r)[1]
    historical = deepcopy(a); historical.pop('paired_realization_policy')
    original = RequestBroker(contract=historical, transport=lambda _: None, arm='A4', seed=81)
    assert original._request_identity(**values)[1] != left._request_identity(**values)[1]


@pytest.mark.parametrize('change', ['seed', 'model', 'thinking', 'horizon', 'budget', 'cell', 'path', 'identity'])
def test_unfrozen_pair_policy_fails_before_provider(change):
    a, _ = contracts()
    if change == 'seed': a['seeds'] = [82]
    elif change == 'model': a['models']['solver'] = 'other'
    elif change == 'thinking': a['models']['solver_thinking'] = True
    elif change == 'horizon': a['operational_pilot']['max_opportunities'] = 6
    elif change == 'budget': a['operational_pilot']['token_ceiling'] += 1
    elif change == 'cell': a['paired_realization_policy']['cell'] = 'B'
    elif change == 'path': a['paired_realization_policy']['group_id'] = '../escape'
    else: a['paired_realization_policy']['identity'] = 'unknown'
    with pytest.raises(SearchContractError, match='MATCHED_REALIZATION_POLICY'):
        RequestBroker(contract=a, transport=lambda _: pytest.fail('provider'), arm='A4', seed=81)


def authorize(root, a, b):
    p = a['paired_realization_policy']
    destination = root / p['authorization_file']; destination.parent.mkdir(parents=True)
    auth = dict(explicit_user_authorized=True, single_use_per_cell=True, scope=group_scope(p),
        approved_startups={a['execution_attempt_id']: 'a' * 64, b['execution_attempt_id']: 'b' * 64},
        source_sha='1' * 40)
    destination.write_text(json.dumps(auth), encoding='utf-8')
    return auth


def test_authorization_is_fresh_exact_and_required(tmp_path):
    a, b = contracts(); payload = dict(manifest=dict(source_sha='1' * 40), startup_identity_sha256='a' * 64)
    with pytest.raises(SearchContractError, match='EXACT_PAIRED_AUTHORIZATION_REQUIRED'):
        validate_authorization(tmp_path, a, payload)
    auth = authorize(tmp_path, a, b)
    assert validate_authorization(tmp_path, a, payload) == auth
    payload['startup_identity_sha256'] = 'c' * 64
    with pytest.raises(SearchContractError, match='EXACT_PAIRED_AUTHORIZATION_REQUIRED'):
        validate_authorization(tmp_path, a, payload)


def test_common_cache_reuses_original_response_with_zero_second_charge(tmp_path):
    a, b = contracts(); authorize(tmp_path, a, b)
    calls = []
    def transport(request):
        calls.append(request)
        return dict(text='A synthetic written operation.\nFINAL_ANSWER: 2', input_tokens=2,
                    output_tokens=3, finish_reason='stop')
    brokers = []
    for c, checksum in [(a, 'a' * 64), (b, 'b' * 64)]:
        payload = dict(manifest=dict(source_sha='1' * 40), startup_identity_sha256=checksum)
        cache = DurableExactOutputCache(tmp_path / a['paired_realization_policy']['cache_directory'],
            cache_context(tmp_path, c, payload))
        brokers.append(RequestBroker(contract=c, transport=transport, arm='A4', seed=81, durable_cache=cache))
    first = brokers[0].complete(**kwargs()); second = brokers[1].complete(**kwargs())
    assert len(calls) == 1 and first['text'] == second['text']
    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import prediction_from_persisted
    assert prediction_from_persisted(first['resolved_prediction']) == prediction_from_persisted(second['resolved_prediction'])
    assert first['request_sha256'] == second['request_sha256']
    assert second['input_tokens'] == second['output_tokens'] == 0
    assert second['provider_called'] is False and brokers[1].usage['attempts'] == 0
    path = tmp_path / a['paired_realization_policy']['cache_directory'] / (first['request_sha256'] + '.json')
    record = json.loads(path.read_bytes()); record['result']['text'] = 'tampered'
    path.write_text(json.dumps(record), encoding='utf-8')
    cache = DurableExactOutputCache(path.parent, cache_context(tmp_path, b,
        dict(manifest=dict(source_sha='1' * 40), startup_identity_sha256='b' * 64)))
    with pytest.raises(SearchContractError, match='DURABLE_CACHE_SCIENTIFIC_ATTEMPT_MISMATCH|seal|IMMUTABLE|MISMATCH|CORRUPTION'):
        cache.get(first['request_sha256'])


def test_both_initial_floors_are_measured_before_search_and_review_is_bound(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.benchmarks import math_domain_binding
    from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATHV21BenchmarkAdapter
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.governance import matched_realization as paired
    from multi_dataset_diverse_rl.governance.autonomous_math import plain
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.current_composition import build_current_team_prompt_search
    from multi_dataset_diverse_rl.search.current_layer1 import CurrentOptimizer
    from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
    from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver, ReflectionProvider
    from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
    from multi_dataset_diverse_rl.search.textual_gradients import GradientClusterProvider, PerExampleGradientProvider
    a, b = contracts(); auth = authorize(tmp_path, a, b)
    for c in (a, b):
        path = tmp_path / c['binding_path']; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(c), encoding='utf-8')
    path = tmp_path / b['pattern_prompt_path']; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(prompt='synthetic cluster instruction')), encoding='utf-8')
    from multi_dataset_diverse_rl.benchmarks.math_evidence_binding import evidence_gradient_prompt_artifact
    path = tmp_path / b['gradient_prompt_path']; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence_gradient_prompt_artifact()), encoding='utf-8')
    payload = dict(manifest=dict(source_sha='1' * 40), startup_identity_sha256='a' * 64)
    cache = DurableExactOutputCache(tmp_path / a['paired_realization_policy']['cache_directory'],
        cache_context(tmp_path, a, payload))
    calls = []
    def transport(request):
        calls.append(request)
        return dict(text='Synthetic calculation with a visible check.\nFINAL_ANSWER: 2',
            input_tokens=2, output_tokens=3, finish_reason='stop')
    broker = RequestBroker(contract=a, transport=transport, arm='A4', seed=81, durable_cache=cache)
    def compose(c, current_broker, destination):
        adapter = MATHV21BenchmarkAdapter(c)
        examples = tuple(CorrectnessExample(protocol_input('math', 'synthetic' + str(i),
            dict(problem='Synthetic public problem ' + str(i)), adapter.output_contract, protocol=adapter.protocol),
            '2', 'Synthetic reference operation.' if c.get('optimization_evidence_policy') else None)
            for i in range(18))
        solver = BenchmarkSolver(adapter, current_broker)
        reflection = ReflectionProvider(current_broker)
        pattern = GradientClusterProvider(current_broker, gradient_provider=PerExampleGradientProvider(current_broker))
        method = CURRENT_POLICY_BUNDLE.method(aggregation='equal_equivalence_plurality_consistency_v1',
            provider_binding='0' * 64, successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
            solver_trajectory_policy=c['solver_trajectory_policy'],
            optimization_evidence_policy=c.get('optimization_evidence_policy'))
        return build_current_team_prompt_search(benchmark=adapter, aggregation=EquivalencePluralityAggregation(),
            examples=examples, prompts=('Solve the problem.',) * 5, solver=solver,
            optimizer=CurrentOptimizer(evaluator=solver, reflection_lm=reflection,
                accounting_reader=reflection.accounting, run_root=destination), method=method, seed=81,
            shadow_loader=lambda: examples[:3], shadow_count=3, runtime_readiness=lambda: (),
            pattern_provider=pattern, provider_call_reader=lambda: current_broker.successes)
    own = compose(a, broker, tmp_path / 'runs/own'); own.state.initialize()
    state = own.state.snapshot(); run_root = tmp_path / 'runs/own'; run_root.mkdir(parents=True)
    atomic_write_json(run_root / 'initial_state_private.json', plain(state))
    peers = []
    def peer_compose(**ports):
        peer = compose(b, ports['solver'].broker, ports['run_root']); peers.append(peer)
        return peer
    monkeypatch.setattr(math_domain_binding, 'execution_binding', lambda root, c:
        SimpleNamespace(benchmark=lambda: MATHV21BenchmarkAdapter(c), compose=peer_compose))
    def reviewed_write(path, value):
        atomic_write_json(path, value)
        if path.name == 'INITIAL_CANARY_REVIEW_REQUIRED.json':
            review = {k: v for k, v in value.items() if k.endswith('sha256')}
            review.update(approved_for_search=True, visible_solution_useful=True,
                final_payload_parser_confirmed=True, solver_wire_confirmed=True,
                thinking_control_confirmed=True, reviewed_example_ids=['synthetic0'])
            atomic_write_json(path.parent / 'initial_canary_review.json', review)
    monkeypatch.setattr(paired, 'atomic_write_json', reviewed_write)
    paired.freeze_initial_and_review(tmp_path, a, payload, run_root, state)
    assert len(calls) == 90 and peers[0].state.snapshot().member_scores == state.member_scores
    assert peers[0].state.initial_state_id is not None
    assert (run_root / 'initial_peer_state_private.json').is_file()
    assert broker.usage['reflection'] == broker.usage['pattern'] == 0
    # A false or mismatched owner review cannot permit B's optimization.
    peer_root = tmp_path / 'runs/peer'; peer_root.mkdir()
    atomic_write_json(peer_root / 'initial_state_private.json', plain(peers[0].state.snapshot()))
    monkeypatch.setattr(paired, 'atomic_write_json', atomic_write_json)
    atomic_write_json(peer_root / 'initial_canary_review.json', dict(approved_for_search=False))
    with pytest.raises(SearchContractError, match='INITIAL_CANARY_REVIEW_MISMATCH'):
        paired.freeze_initial_and_review(tmp_path, b,
            dict(manifest=dict(source_sha='1' * 40), startup_identity_sha256='b' * 64),
            peer_root, peers[0].state.snapshot())
