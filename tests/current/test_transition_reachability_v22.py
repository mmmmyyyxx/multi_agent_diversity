"""V2.2 zero-API deployment reachability, replay separation and admission safety."""
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from multi_dataset_diverse_rl import current_contract, versions
from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
from multi_dataset_diverse_rl.search.initial_competence_transition import InitialCompetenceTransitionV2
from multi_dataset_diverse_rl.search.target_or_team_transition import InitialCompetenceTargetOrTeamProgressV3, progress_path
from multi_dataset_diverse_rl.search.schemas import (
    EvaluatedCandidate, SearchCandidate, SearchContractError, SearchMethodConfig, TeamEvaluation,
)

ROOT = Path(__file__).resolve().parents[2]


def measurement(target=22, vote=22, invalid_delta=0, **metrics):
    return TeamEvaluation(vote, None, (target, 22, 22, 22, 22),
        aggregation_diagnostics={'terminal_invalid_delta': invalid_delta, **metrics})


def policy():
    result = InitialCompetenceTargetOrTeamProgressV3()
    result.bind_initial((22,) * 5, 'initial')
    return result


@pytest.mark.parametrize('target,vote,invalid,expected', [
    (25, 22, 0, True), (22, 23, 0, True), (25, 23, 0, True),
    (21, 23, 0, False), (25, 21, 0, False), (22, 22, 0, False),
    (25, 23, 1, False),
])
def test_transition_reachability_and_safety(target, vote, invalid, expected):
    assert policy().allows(measurement(), measurement(target, vote, invalid), 0) is expected


def test_team_progress_can_lower_incumbent_above_immutable_floor():
    p = policy()
    assert p.allows(measurement(25), measurement(23, 23), 0)
    assert not p.allows(measurement(25), measurement(21, 23), 0)
    with pytest.raises(SearchContractError, match='CANNOT_REBASE'):
        p.bind_initial((25,) * 5, 'child')
    assert p.initial_scores == (22,) * 5 and p.initial_state_id == 'initial'


def test_v21_does_not_inherit_v22_gate_or_identity():
    old = InitialCompetenceTransitionV2()
    old.bind_initial((22,) * 5, 'initial')
    assert not old.allows(measurement(), measurement(25, 22), 0)
    assert policy().allows(measurement(), measurement(25, 22), 0)
    assert old.identity == versions.UNIFIED_COMPETENCE_TRANSITION_VERSION == 'initial_competence_team_gain_v2'
    assert policy().identity == current_contract.UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION
    assert SearchMethodConfig.v2_1().identity() != SearchMethodConfig.v2_2().identity()
    for factory in (SearchMethodConfig.v2_1, SearchMethodConfig.v2_2):
        assert SearchMethodConfig.from_mapping(asdict(factory())) == factory()
    assert not hasattr(current_contract, 'UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION')


def test_invalid_prediction_policy_keeps_existing_frozen_behavior():
    p = InitialCompetenceTargetOrTeamProgressV3(invalid_predictions_are_incorrect=True)
    p.bind_initial((22,) * 5, 'initial')
    # The frozen MATH validity policy scores terminal-invalid as incorrect;
    # it already makes invalidity telemetry observation-only in V2.1.
    assert p.allows(measurement(), measurement(25, 22, 1), 0)
    with pytest.raises(SearchContractError, match='GUARD_NOT_MEASURED'):
        p.allows(measurement(), replace(measurement(25), aggregation_diagnostics={}), 0)
    with pytest.raises(SearchContractError, match='NONFINITE'):
        p.allows(measurement(float('nan')), measurement(25), 0)


def row(cid, target, vote=22, soft=0, local=0):
    return EvaluatedCandidate(SearchCandidate(cid, 'Check signs. ' + cid, search_score=local),
        None, measurement(target, vote, mean_soft_vote_utility=soft), True, False, {'target_member': 0})


def test_winner_team_first_target_second_not_local_or_soft_or_input_order():
    a, b = row('A', 23, soft=999, local=999), row('B', 25, soft=0, local=-999)
    for candidates in ((a, b), (b, a)):
        assert policy().select(measurement(), candidates).candidate == b
    c = row('C', 22, vote=23, soft=-1000)
    assert policy().select(measurement(), (b, c)).candidate == c


@pytest.mark.parametrize('team,target,path', [(1, -2, 'TEAM'), (0, 3, 'TARGET'),
    (1, 3, 'TARGET_AND_TEAM'), (0, 0, 'NONE')])
def test_progress_telemetry_paths(team, target, path):
    assert progress_path(team, target) == path


def test_target_commit_resets_scientific_patience_without_using_team_delta():
    from multi_dataset_diverse_rl.search.policies import GlobalStopPolicy
    stop = GlobalStopPolicy(2)
    assert stop.observe_opportunity(parent_state_id='parent', eligible_members=(0,),
        selected_member=0, local_update=False, committed=False) is None
    assert stop.no_commit_epochs == 1
    assert stop.observe_opportunity(parent_state_id='parent', eligible_members=(0,),
        selected_member=0, local_update=False, committed=True) is None
    assert stop.no_commit_epochs == 0
    assert stop.observe_opportunity(parent_state_id='child', eligible_members=(0,),
        selected_member=0, local_update=False, committed=False) is None
    assert stop.observe_opportunity(parent_state_id='child', eligible_members=(0,),
        selected_member=0, local_update=False, committed=False) == 'SATURATION_REACHED'


def test_target_only_teamprobe_signal_reaches_full_with_ceiling_two():
    from multi_dataset_diverse_rl.search.evaluation import CandidateEvaluationPipeline, FixedPeerPromotion
    from multi_dataset_diverse_rl.search.schemas import SearchResult
    from multi_dataset_diverse_rl.team_search.schemas import TeamMiniBatchMetrics
    calls = []
    class Provider:
        async def team_probe(self, opportunity, candidate):
            return replace(measurement(), aggregation_diagnostics={'team_probe_metrics':
                TeamMiniBatchMetrics(target_delta=1, vote_delta=0, team_net_vote_delta=0,
                    responsibility_delta=1, broad_delta=1, invalid_delta=0)})
        async def full(self, opportunity, candidate):
            calls.append(candidate.candidate_id)
            return measurement(25)
    candidates = tuple(SearchCandidate(str(i), 'Check constraints. ' + str(i)) for i in range(4))
    result = asyncio.run(CandidateEvaluationPipeline(Provider(), FixedPeerPromotion()).evaluate(
        NS(target_member=0, evaluation_plan={}), SearchResult(candidates, 'frozen_fake')))
    assert len(calls) == sum(r.promoted for r in result) == 2
    assert policy().select(measurement(), result).candidate is not None


def test_pilot_bound_and_builder_fail_before_any_provider(tmp_path):
    from multi_dataset_diverse_rl.benchmarks.gradient_pilot_contract import pilot_provider_bounds, derive_current_pilot_contract
    from multi_dataset_diverse_rl.search.current_composition import build_current_team_prompt_search
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('provider')
        raise AssertionError('No provider may be called')
    with pytest.raises(SearchContractError, match='TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN'):
        pilot_provider_bounds({})
    with pytest.raises(SearchContractError, match='TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN'):
        derive_current_pilot_contract({}, attempt='synthetic', binding_path='unbound', parent_path='unbound',
            parent_sha256='0'*64, authorization_path='unbound', authorization_sha256='0'*64)
    with pytest.raises(SearchContractError, match='TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN'):
        build_current_team_prompt_search(benchmark=None, aggregation=None, examples=(), prompts=(),
            solver=forbidden, optimizer=forbidden, method=None, seed=81, shadow_loader=forbidden,
            shadow_count=40, runtime_readiness=forbidden, pattern_provider=forbidden, execution_phase='pilot')
    assert calls == []


def test_current_entrypoint_rejects_old_and_unfrozen_new_binding(tmp_path, monkeypatch):
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.governance import unified_execution as gov
    from scripts.run_experiment import preflight, execute_frozen
    old = json.loads((ROOT/'experiments/execution_bindings/math_v2_1_gradient_pattern_pilot_offline_profile_v5.json').read_bytes())
    calls = []
    for contract in (old, {'identity': 'MATH_V2_2_EXECUTION_BINDING_V1'}):
        with pytest.raises(SearchContractError, match='CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN'):
            binding = execution_binding(ROOT, contract)
            binding.compose(solver=lambda: calls.append('solver'))
    old_manifest = json.loads(json.dumps(__import__('yaml').safe_load(
        (ROOT/'experiments/manifests/math_v2_1_gradient_pattern_seed81_pilot_v6.yaml').read_text(encoding='utf-8'))))
    # A current preflight must reject before reading/instantiating the old binding.
    monkeypatch.setattr(gov, 'read_json', lambda path: (_ for _ in ()).throw(AssertionError('old binding read')))
    status = preflight(old_manifest)
    assert status['gate'] == 'HOLD' and status['provider_attempts'] == 0
    assert 'CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN' in status['blockers']
    with pytest.raises(SearchContractError, match='CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN'):
        asyncio.run(execute_frozen(tmp_path/'prep', tmp_path/'run'))
    assert calls == [] and not (tmp_path/'run').exists()


def test_current_default_profile_and_manifest_identity_are_v22(monkeypatch):
    import yaml
    from multi_dataset_diverse_rl.governance import unified_execution as gov
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    paths = []
    original = gov.read_json
    def read(path):
        paths.append(path.relative_to(ROOT).as_posix())
        return original(path)
    monkeypatch.setattr(gov, 'read_json', read)
    with pytest.raises(SearchContractError, match='CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN'):
        gov.preexecution_manifest(ROOT, source_sha='0'*40)
    assert paths == ['experiments/execution_bindings/math_v2_2_offline_profile_v1.json']
    manifest = yaml.safe_load((ROOT/'experiments/templates/unified_experiment_v2_2.yaml').read_text(encoding='utf-8'))
    assert validate_manifest_v2(ROOT, manifest) == []
    for key, value in [('transition_identity', versions.UNIFIED_COMPETENCE_TRANSITION_VERSION),
            ('method_identity', versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION),
            ('pattern_identity', 'null_pattern_v1')]:
        assert validate_manifest_v2(ROOT, {**manifest, key: value})
    method = CURRENT_POLICY_BUNDLE.method(aggregation=versions.EQUIVALENCE_PLURALITY_VERSION,
        provider_binding='0'*64, successful_provider_calls=10000)
    assert method.method == 'unified_team_prompt_search_v2_2'
    with pytest.raises(SearchContractError, match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        CURRENT_POLICY_BUNDLE.validate_method(replace(method, method=versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION))


def test_current_graph_excludes_v21_replay_and_archive_has_provenance():
    import hashlib
    from multi_dataset_diverse_rl.governance.current_dependencies import current_dependency_graph
    graph = current_dependency_graph(ROOT)
    assert graph['legacy_namespace_dependencies'] == graph['legacy_class_definitions'] == []
    assert not any(p.endswith('legacy_current_contract_v21.py') for p in graph['modules'])
    index = json.loads((ROOT/'docs/archive/specs/unified_v2_1_replay_index.json').read_bytes())
    for item in index['archives']:
        content = (ROOT/item['archive']).read_bytes().replace(b'\r\n', b'\n')
        assert b'archive_status: HISTORICAL_REPLAY_ONLY' in content
        assert hashlib.sha256(content.split(b'---\n\n', 1)[1]).hexdigest() == item['body_sha256']
    for item in index['replay_modules']:
        content = (ROOT/item['replay']).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(content).hexdigest() == item['replay_normalized_sha256']


def test_source_identity_binds_v22_authority_and_excludes_replay_receipt_constructors():
    from multi_dataset_diverse_rl.governance.source_identity import (
        build_unified_source_identity, HISTORICAL_RECEIPT_PATHS,
    )
    identity = build_unified_source_identity(ROOT)
    scientific = {r['path'] for r in identity['scopes']['scientific']['files']}
    benchmark = {r['path'] for r in identity['scopes']['benchmark']['files']}
    assert 'docs/design/TRANSITION_TARGET_OR_TEAM_PROGRESS_V3.md' in scientific
    assert 'docs/design/UNIFIED_METHOD_SEMANTIC_CONTRACT.md' not in scientific
    assert not benchmark.intersection(HISTORICAL_RECEIPT_PATHS)
    assert all('/legacy/' not in row['path'] and not row['path'].endswith('legacy_current_contract_v21.py')
        for scope in identity['scopes'].values() for row in scope['files'])


def make_current_fake_graph(tmp_path, *, shadow_progress=False, shadow_regression=False):
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATHV21BenchmarkAdapter
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.current_composition import build_current_team_prompt_search
    from multi_dataset_diverse_rl.search.current_layer1 import CurrentOptimizer
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker, BenchmarkSolver, ReflectionProvider
    from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
    from multi_dataset_diverse_rl.search.textual_gradients import GradientClusterProvider, PerExampleGradientProvider
    # Reuse unchanged decoding/interface settings as test data, with fresh synthetic identities.
    c = json.loads((ROOT/'experiments/execution_bindings/math_v2_1_gradient_pattern_offline_profile_v6.json').read_bytes())
    c.update(identity='MATH_V2_2_EXECUTION_BINDING_V1', execution_attempt_id='synthetic_v22',
        cache_namespace='synthetic_v22', transition_policy=policy().identity,
        method_identity='unified_team_prompt_search_v2_2')
    # Explicit finite test fixture ceilings; no scientific Pilot bound is derived.
    c['provider_bounds'].update({k:10000 for k in ('solver_calls', 'reflection_calls',
        'pattern_gradient_calls', 'pattern_cluster_calls', 'pattern_calls',
        'successful_provider_calls', 'transport_attempts')})
    adapter = MATHV21BenchmarkAdapter(c)
    def examples(role, count):
        return tuple(CorrectnessExample(protocol_input('math', f'{role}{i}',
            {'problem':f'Synthetic {role} arithmetic {i}.'}, adapter.output_contract, protocol=adapter.protocol), '1')
            for i in range(count))
    gradients, requests = [], []
    text = 'Check constraints before transforming intermediate expressions.'
    def transport(req):
        requests.append(deepcopy(req))
        if req['model'] == 'qwen3-8b':
            prompt, problem = req['messages'][1]['content'].split('\n\n', 1)
            i = int(problem.split('arithmetic ')[1].split('.')[0])
            initial = prompt == 'Solve the problem.'
            correct = i < 22 or (not initial and i in gradients[:3]
                and ('Synthetic optimize' in problem or shadow_progress))
            if shadow_regression and not initial and 'Synthetic shadow' in problem and 19 <= i < 22:
                correct = False
            output = 'FINAL_ANSWER: ' + ('1' if correct else '2')
        elif len(req['messages']) == 2:
            payload = json.loads(req['messages'][1]['content'])
            if 'example' in payload:
                problem = payload['example']['problem']
                if isinstance(problem, dict): problem = problem['problem']
                gradients.append(int(problem.split('arithmetic ')[1].split('.')[0]))
                output = json.dumps({'gradient':text})
            else:
                ids = [g['example_id'] for g in payload['gradients']]
                output = json.dumps({'patterns':[{'generalized_gradient':text, 'support_ids':ids[:3]}],
                    'unassigned_ids':ids[3:]})
        else:
            output = json.dumps({'decision_procedure':'Solve the problem. Check constraints and verify signs.',
                'change_summary':'Add constraint and sign checks.'})
        return dict(text=output, input_tokens=2, output_tokens=2, finish_reason='stop',
            provider_response_accepted=True, provider_metadata_loss_audited=True,
            provider_reasoning_content_present=False, provider_reasoning_character_count=None,
            provider_usage_details={}, provider_thinking_indicators=[])
    broker = RequestBroker(contract=c, transport=transport, arm='A4', seed=81)
    solver, reflection = BenchmarkSolver(adapter, broker), ReflectionProvider(broker)
    pattern = GradientClusterProvider(broker, gradient_provider=PerExampleGradientProvider(broker))
    method = CURRENT_POLICY_BUNDLE.method(aggregation=versions.EQUIVALENCE_PLURALITY_VERSION,
        provider_binding='0'*64, successful_provider_calls=10000)
    run = build_current_team_prompt_search(benchmark=adapter, aggregation=EquivalencePluralityAggregation(),
        examples=examples('optimize', 60), prompts=('Solve the problem.',)*5, solver=solver,
        optimizer=CurrentOptimizer(evaluator=solver, reflection_lm=reflection, accounting_reader=reflection.accounting,
            run_root=tmp_path), method=method, seed=81, shadow_loader=lambda:examples('shadow', 40),
        shadow_count=40, runtime_readiness=lambda:(), pattern_provider=pattern,
        provider_call_reader=lambda:broker.successes)
    run.state.initialize()
    return run, broker, requests


def test_complete_current_fake_graph_commits_first_target_only_update(tmp_path):
    run, broker, requests = make_current_fake_graph(tmp_path)
    result = asyncio.run(run.run(max_opportunities=1))
    assert len(result.transitions) == 1 and result.trace[0].committed_candidate_id
    trace = result.trace[0].allocation_audit
    assert trace['parent_team_score'] == trace['child_team_score'] == 22
    assert trace['realized_team_gain'] == 0 and trace['realized_target_gain'] == 3
    assert trace['realized_progress_path'] == 'TARGET' and trace['committed']
    assert run.transition.initial_scores == run.state.initial_member_scores == (22,)*5
    assert run.gate.winner_count == 1 and run.gate.feedback.passed
    # Shadow target is unchanged (0 gain), proving safety rather than gain replication.
    assert run.memory.audit()['success_writes'] == 1 and len(run.memory.private) == 1
    success = run.memory.private[0]
    assert 'realized_team_gain=0' in success.outcome
    assert 'realized_target_gain=3' in success.outcome and 'progress_path=TARGET' in success.outcome
    visible = run.memory.read_for_opportunity(NS(target_member=success.owner_member,
        evidence=NS(mutation_evidence=(NS(source_split='optimize'),),search_validation_evidence=()),
        diagnosis=NS(responsibility={success.owner_member:NS(primary_lane=success.lane)})))
    assert any('progress_path=TARGET' in row['outcome'] for row in visible['private_success'])
    assert not run.memory.shared
    assert broker.usage['validation'] == broker.usage['test'] == 0
    assert result.stop_reason == 'CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'


def test_sequential_commits_use_current_peers_and_floors_and_reject_stale_parent(tmp_path):
    from multi_dataset_diverse_rl.search.transition import TeamStateCommitter
    from multi_dataset_diverse_rl.search.history import NullMemoryProvider
    from multi_dataset_diverse_rl.search.schemas import TransitionDecision
    from multi_dataset_diverse_rl.search.binary_runtime import BinaryEvidenceSource
    run, broker, requests = make_current_fake_graph(tmp_path, shadow_progress=True)
    first = asyncio.run(run.run(max_opportunities=1))
    first_target = first.trace[0].target_member
    state1 = run.state.snapshot()
    evidence = BinaryEvidenceSource(run.state, run.history)
    diagnosis = run.analyzer.analyze(state1, run.history)
    residual = evidence.for_member(state1, diagnosis, first_target)
    assert sum(r.signals['target_member_correct'] is False for r in residual) == 35
    floor = run.state.initial_member_scores
    for index, target in enumerate(m for m in range(5) if m != first_target):
        if index == 2:
            break
        parent = run.state.snapshot()
        opportunity = NS(parent_state_id=parent.team_state_id, target_member=target,
            parent_prompt=parent.member_prompts[target], opportunity_id=f'sequential_{index}',
            evaluation_plan={'current_parent_binding':True})
        candidate = SearchCandidate(f'candidate_{index}', 'Check constraints and verify signs. ' + str(index),
            backend_details={'parent_state_id':parent.team_state_id})
        full = asyncio.run(run.evaluation.provider.full(opportunity, candidate))
        active = asyncio.run(run.evaluation.provider.active(opportunity))
        assert full.member_scores[first_target] == 25  # A', never the initial A.
        assert run.transition.allows(active, full, target)
        assert full.aggregate_score == (22 if index == 0 else 25)
        row = EvaluatedCandidate(candidate, None, full, True, False, {'target_member':target})
        assert asyncio.run(run.gate.check(opportunity, row))
        # The second Shadow incumbent includes both previously committed lanes.
        assert run.gate.feedback.passed
        assert run.gate.feedback.aggregate_score == (22 if index == 0 else 25)
        history_before = run.history.snapshot()
        with pytest.raises(SearchContractError, match='COMMIT_CURRENT_PARENT_FULL_MISMATCH'):
            TeamStateCommitter(run.state).commit(opportunity,
                TransitionDecision(replace(row, full=replace(full, aggregate_score=full.aggregate_score+1)), 'bad_full'),
                run.history, NullMemoryProvider())
        assert run.state.snapshot() == parent and run.history == history_before
        assert run.state.initial_member_scores == floor
        TeamStateCommitter(run.state).commit(opportunity, TransitionDecision(row, 'synthetic'),
            run.history, NullMemoryProvider())
        child = run.state.snapshot()
        assert child.member_outputs[first_target] == parent.member_outputs[first_target]
        assert child.member_scores == full.member_scores
        assert run.state.initial_member_scores == run.transition.initial_scores == floor == (22,)*5
        before_requests = len(requests)
        with pytest.raises(SearchContractError, match='EVALUATION_PARENT_MISMATCH'):
            asyncio.run(run.evaluation.provider.full(opportunity, candidate))
        assert len(requests) == before_requests
    assert run.state.snapshot().team_scores['vote_correct_count'] == 25
    assert broker.usage['validation'] == broker.usage['test'] == 0


def test_shadow_rejection_has_no_atomic_update_or_success_memory(tmp_path):
    run, broker, _ = make_current_fake_graph(tmp_path, shadow_regression=True)
    initial = run.state.snapshot()
    result = asyncio.run(run.run(max_opportunities=1))
    assert run.gate.winner_count == 1 and run.gate.feedback.passed is False
    assert result.trace[0].selected_candidate_id and not result.transitions
    assert run.state.snapshot() == initial
    assert run.memory.audit()['success_writes'] == 0
    assert run.state.initial_member_scores == (22,)*5


def test_stale_candidate_fails_before_any_teamprobe_provider_call():
    from multi_dataset_diverse_rl.search.evaluation import CandidateEvaluationPipeline, FixedPeerPromotion
    from multi_dataset_diverse_rl.search.schemas import SearchResult
    calls = []
    class Provider:
        async def team_probe(self, *args):
            calls.append('provider')
            raise AssertionError('Stale candidate must be rejected first')
    opportunity = NS(parent_state_id='active', target_member=0,
        evaluation_plan={'current_parent_binding':True})
    candidate = SearchCandidate('stale', 'Check constraints.', backend_details={'parent_state_id':'old'})
    with pytest.raises(SearchContractError, match='CANDIDATE_PARENT_MISMATCH'):
        asyncio.run(CandidateEvaluationPipeline(Provider(), FixedPeerPromotion()).evaluate(
            opportunity, SearchResult((candidate,), 'synthetic')))
    assert not calls


@pytest.mark.parametrize('mutation', ['fractional','outside','missing_support','nan_tie','peer_change'])
def test_complete_binary_measurements_are_required_for_potential_proof(mutation):
    p = InitialCompetenceTargetOrTeamProgressV3(evaluation_count=60)
    p.bind_initial((22,)*5, 'initial')
    parent = measurement(evaluation_count=60)
    full = measurement(25, evaluation_count=60)
    if mutation == 'fractional': full = replace(full, aggregate_score=22.5)
    if mutation == 'outside': full = replace(full, aggregate_score=61)
    if mutation == 'missing_support': full = measurement(25)
    if mutation == 'nan_tie': full = measurement(25, evaluation_count=60, mean_soft_vote_utility=float('nan'))
    if mutation == 'peer_change': full = replace(full, member_scores=(25,23,22,22,22))
    with pytest.raises(SearchContractError): p.allows(parent, full, 0)


def test_v22_potential_exhaustively_covers_target_team_and_competence_decline():
    from multi_dataset_diverse_rl.benchmarks.gradient_pilot_contract import progress_potential
    n = 4
    for floor in range(n+1):
        p = InitialCompetenceTargetOrTeamProgressV3(evaluation_count=n)
        p.bind_initial((floor,)*5, 'initial')
        for old_target in range(floor,n+1):
            for new_target in range(n+1):
                for old_vote in range(n+1):
                    for new_vote in range(n+1):
                        def value(target, vote):
                            return TeamEvaluation(vote, None, (target,floor,floor,floor,floor),
                                aggregation_diagnostics={'terminal_invalid_delta':0,'evaluation_count':n})
                        before, after = value(old_target,old_vote), value(new_target,new_vote)
                        expected = new_target >= floor and new_vote >= old_vote and (
                            new_target > old_target or new_vote > old_vote)
                        assert p.allows(before,after,0) == expected
                        if expected:
                            assert progress_potential(new_vote,after.member_scores,optimize_count=n) > \
                                progress_potential(old_vote,before.member_scores,optimize_count=n)
    # Simple unweighted sum is false for a legal competence drop.
    assert 1+0 < 0+4
    assert progress_potential(1,(0,0,0,0,0),optimize_count=4) > \
        progress_potential(0,(4,0,0,0,0),optimize_count=4)


def test_finite_mathematical_bound_does_not_unlock_impractical_real_execution():
    from multi_dataset_diverse_rl.benchmarks.gradient_pilot_contract import finite_bound_assessment, pilot_provider_bounds
    bound = finite_bound_assessment()
    assert bound['potential_maximum'] == bound['max_commits'] == 3960
    assert bound['max_epoch_segments'] == 7922
    assert bound['conservative_opportunity_ceiling_min_decimal_digits'] > 18000
    assert not bound['resource_realistic_completion_bound_established']
    assert bound['execution_gate'] == 'HOLD'
    measured = finite_bound_assessment(initial_vote=22, initial_scores=(22,)*5)
    assert measured['max_commits'] == 2508
    with pytest.raises(SearchContractError, match='PILOT_BOUND_NOT_FROZEN'):
        pilot_provider_bounds({'bound_proof':bound})


@pytest.mark.parametrize('code,failures', [(5,2),(32,2),(33,2),(5,8),(13,1)])
def test_windows_atomic_replace_retries_only_bounded_sharing_failures(monkeypatch, code, failures):
    from multi_dataset_diverse_rl.persistence import durable_io
    calls, sleeps = [], []
    def replacement(*args):
        calls.append(args)
        if len(calls) <= failures:
            error = PermissionError('Synthetic Windows file sharing failure')
            error.winerror = code
            raise error
    monkeypatch.setattr(durable_io.os, 'replace', replacement)
    monkeypatch.setattr(durable_io.time, 'sleep', sleeps.append)
    if code in {5,32,33} and failures < 8:
        durable_io.atomic_replace('synthetic.tmp', 'synthetic.json')
        assert len(calls) == failures+1 and len(sleeps) == failures
    else:
        with pytest.raises(PermissionError): durable_io.atomic_replace('synthetic.tmp','synthetic.json')
        assert len(calls) == (8 if code == 5 else 1)
    assert len(sleeps) <= 7 and sum(sleeps) <= 1.11
