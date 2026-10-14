"""Module integration, repeated-seed isolation and strict gate failure boundaries."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import json

import pytest

from multi_dataset_diverse_rl.research.ablations.config import (
    default_config, build_plan, validate_plan, request_namespace,
)
from multi_dataset_diverse_rl.research.ablations.engine import run_cell
from multi_dataset_diverse_rl.research.ablations.interfaces import (
    OptimizeContext, OptimizeOutcome, PreparedFeedback, ModuleProvider,
)
from multi_dataset_diverse_rl.research.ablations.modules import NoModule, ModuleRegistry, default_registry
from multi_dataset_diverse_rl.research.ablations.pattern import project
from multi_dataset_diverse_rl.research.ablations.formal_adapters import (
    GradientTransportAdapter, ClusterTransportAdapter, new_production_memory, production_memory_binding,
)
from multi_dataset_diverse_rl.research.ablations.analysis import summarize
from multi_dataset_diverse_rl.research.ablations.synthetic import smoke, synthetic_rows
from multi_dataset_diverse_rl.search.textual_gradients import PerExampleGradientProvider, GradientClusterProvider
from multi_dataset_diverse_rl.search.population_memory import PopulationEditMemory
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest

ROOT = Path(__file__).resolve().parents[2]


def small_plan(rounds=3, registry=None, module='none'):
    c = default_config()
    c['rounds'] = c['runtime']['rounds'] = rounds
    c['arms'] = [dict(arm_id='B', error_source='random_wrong_batch', module=module)]
    return build_plan(c, registry)


class ScriptedPort:
    phase = 'train'
    execution_policy = dict(gradient_max_concurrency=1)

    def __init__(self, outputs=None, failure=None):
        self.usage = dict(provider_calls=0, input_tokens=0, output_tokens=0)
        self.evaluations = []
        self.packets = []
        self.outputs = iter(outputs) if outputs is not None else None
        self.failure = failure

    def bind_cell(self, cell, plan):
        self.cell, self.plan = cell, plan

    def usage_view(self):
        return dict(self.usage)

    def complete(self, **kwargs):
        self.usage['provider_calls'] += 1
        self.usage['input_tokens'] += 10
        self.usage['output_tokens'] += 5
        packet = json.loads(kwargs['messages'][-1]['content'])
        self.packets.append(packet)
        if self.failure:
            raise self.failure
        text = next(self.outputs) if self.outputs else json.dumps(dict(block='strategy',
            content='Check transformations carefully. ' + 'Verify assumptions. ' * self.round_index))
        return dict(text=text, finish_reason='stop', request_sha256=digest(kwargs))

    def evaluate(self, prompt, rows, split, seed, ordinal):
        self.evaluations.append((prompt.to_dict(), split, ordinal))
        # One accepted candidate then a tie and then a regression: no incumbent rescore.
        correct = 1 if ordinal == 0 else {1: 2, 2: 2, 3: 0}.get(ordinal, 0)
        return {r['stable_example_id']: dict(correct=i < correct, valid=True, answer='fixture',
            prediction=dict(text='Synthetic response', invalid_reason=None, semantic_attempt_count=1))
            for i, r in enumerate(rows)}


def run_small(plan, broker=None, registry=None, cell=None, resume=None):
    artifacts = {}
    result = run_cell(plan, cell or plan['cells'][0], SEED.to_dict(), synthetic_rows('optimize', 4),
        synthetic_rows('validation', 4), broker or ScriptedPort(),
        lambda n, v: artifacts.__setitem__(n, deepcopy(v)), registry=registry, resume=resume)
    return result, artifacts


def test_default_three_repeats_same_seed_and_no_real_grant():
    plan = build_plan(default_config())
    assert len(plan['cells']) == 9 and {c['seed'] for c in plan['cells']} == {42}
    assert {c['repeat_id'] for c in plan['cells']} == {1, 2, 3}
    assert len({c['cell_id'] for c in plan['cells']}) == 9
    assert not plan['ready_to_run'] and not plan['config']['real_api_authorized']
    assert plan['training_semantic_call_ceiling'] == 190440
    assert plan['final_test_semantic_call_ceiling'] == 10800
    validate_plan(plan)
    assert json.loads((ROOT / 'experiments/ablations/pattern/default.json').read_text(encoding='utf-8')) == default_config()


def test_repeat_namespace_pairs_arms_but_never_repeats():
    plan = build_plan(default_config())
    values = [request_namespace(plan['plan_identity'], 42, c['repeat_id']) for c in plan['cells']]
    assert len(set(values)) == 3 and values[0] == values[3] == values[6]
    assert values[0] != values[1]


@pytest.mark.parametrize('change', ['authorization', 'runtime', 'duplicate_arm', 'unknown_module', 'wrong_pattern_source', 'float_seed', 'tampered_plan'])
def test_draft_or_binding_mutation_fails_closed(change):
    c = default_config()
    if change == 'authorization': c['real_api_authorized'] = True
    elif change == 'runtime': c['runtime']['optimizer_structural_draws'] = 2
    elif change == 'duplicate_arm': c['arms'].append(deepcopy(c['arms'][0]))
    elif change == 'unknown_module': c['arms'][0]['module'] = 'memory'
    elif change == 'wrong_pattern_source': c['arms'][2]['error_source'] = 'all_wrong'
    elif change == 'float_seed': c['seed'] = 42.0
    else:
        plan = build_plan(c); plan['cells'][0]['repeat_id'] = 2
        with pytest.raises(SearchContractError): validate_plan(plan)
        return
    with pytest.raises(SearchContractError): build_plan(c)


def test_formal_algorithms_are_actual_inherited_methods_and_memory_instance():
    assert GradientTransportAdapter.extract is PerExampleGradientProvider.extract
    assert ClusterTransportAdapter.cluster is GradientClusterProvider.cluster
    assert ClusterTransportAdapter._cluster_draw is GradientClusterProvider._cluster_draw
    assert type(new_production_memory()) is PopulationEditMemory
    assert production_memory_binding()['status'] == 'ADAPTATION_POLICY_REQUIRED'
    with pytest.raises(SearchContractError, match='NOT_REGISTERED'): default_registry().create('memory')


def feedback():
    return dict(errors=[dict(example_id=str(i)) for i in range(3)], structured_feedback=dict(
        gradients=[dict(example_id=str(i)) for i in range(3)],
        diagnostics=[dict(example_id=str(i)) for i in range(3)],
        patterns=[dict(pattern_id='large', support_ids=['0', '1'], generalized_gradient='large correction'),
                  dict(pattern_id='small', support_ids=['2'] * 5, generalized_gradient='excluded correction')],
        partition_audit=[dict(discarded_patterns=['excluded draft'])], unassigned_ids=[], cluster_structural_errors=[]))


def test_projection_matches_frozen_semantics_but_accepts_seed42():
    from multi_dataset_diverse_rl.research.gated_prompt_largest import project_feedback
    packet = feedback()
    expected, old_audit = project_feedback(packet, 88, 1)
    actual, audit = project(packet, 88, 1)
    assert actual == expected and all(audit[k] == old_audit[k] for k in audit)
    actual, audit = project(packet, 42, 1)
    assert audit['selected_pattern_id'] == 'large' and len(actual['errors']) == 2
    assert 'excluded' not in json.dumps(actual) and packet == feedback()
    packet['structured_feedback']['patterns'] = []
    fallback, audit = project(packet, 42, 1)
    assert fallback == dict(errors=packet['errors'], structured_feedback=None) and audit['fallback_to_B']


def test_strict_gate_records_one_update_then_tie_and_regression_rejections():
    class OptimizeRegression(ScriptedPort):
        def evaluate(self, prompt, rows, split, seed, ordinal):
            values = super().evaluate(prompt, rows, split, seed, ordinal)
            if split == 'optimize' and ordinal:
                for v in values.values(): v['correct'] = False
            return values
    broker = OptimizeRegression()
    result, artifacts = run_small(small_plan(), broker)
    assert [m['accepted'] for m in result['metrics']] == [True, False, False]
    assert result['final_validation_correct'] == 2
    assert result['final_optimize_correct'] == 0  # Optimize loss is not an extra gate.
    assert len(broker.evaluations) == 8  # initial Optimize/Validation + three candidates
    assert result['prompt_history'][1:] == [result['final_prompt']] * 3
    assert artifacts['checkpoint_3']['validation'] == artifacts['checkpoint_1']['validation']


def test_invalid_overlong_and_unchanged_retry_before_one_opportunity():
    outputs = ['not JSON', json.dumps(dict(block='strategy', content='x' * 3001)),
               json.dumps(dict(block='strategy', content=SEED.strategy)),
               json.dumps(dict(block='strategy', content='Check transformations carefully.'))]
    broker = ScriptedPort(outputs)
    result, artifacts = run_small(small_plan(rounds=1), broker)
    assert result['metrics'][0]['generation_draws'] == 4 and result['metrics'][0]['accepted']
    assert len(broker.evaluations) == 4
    assert len([k for k in artifacts if 'generation_' in k]) == 4
    assert 'x' * 50 not in json.dumps(broker.packets)


def test_four_failed_draws_exhaust_one_opportunity_without_candidate_evaluation():
    broker = ScriptedPort(['not JSON'] * 4)
    result, _ = run_small(small_plan(rounds=1), broker)
    assert result['metrics'][0]['decision'] == 'ABANDONED_GENERATION_RETRIES_EXHAUSTED'
    assert len(broker.evaluations) == 2 and result['final_prompt'] == SEED.to_dict()


def test_generated_prompt_cannot_depend_on_new_module_context_field():
    broker = ScriptedPort([
        json.dumps(dict(block='strategy', content='Consult module_context while solving.')),
        json.dumps(dict(block='strategy', content='Check each transformation carefully.'))])
    result, artifacts = run_small(small_plan(rounds=1), broker)
    assert result['metrics'][0]['generation_draws'] == 2
    assert artifacts['round_1_generation_1']['errors'] == ['optimizer_only_dependency']
    assert 'module_context' not in json.dumps(result['final_prompt'])


def test_non_generation_fault_propagates_without_completed_checkpoint():
    artifacts = {}
    plan = small_plan(rounds=1)
    with pytest.raises(RuntimeError, match='ledger failed'):
        run_cell(plan, plan['cells'][0], SEED.to_dict(), synthetic_rows('optimize', 4),
            synthetic_rows('validation', 4), ScriptedPort(failure=RuntimeError('ledger failed')),
            artifacts.__setitem__)
    assert 'checkpoint_0' in artifacts and 'checkpoint_1' not in artifacts and 'result' not in artifacts


def test_module_capabilities_exclude_validation_test_and_solver():
    assert not any('validation' in f.name or 'test' in f.name for t in (OptimizeContext, OptimizeOutcome) for f in fields(t))
    port = ModuleProvider(ScriptedPort())
    for role, split in [('solver', 'optimize'), ('pattern_gradient', 'validation'), ('pattern_cluster', 'test')]:
        with pytest.raises(SearchContractError): port.complete(role=role, split=split, messages=[])
    assert not hasattr(port, 'evaluate')


def test_module_cannot_exceed_its_declared_call_budget():
    port = ModuleProvider(ScriptedPort(), maximum_calls=0)
    with pytest.raises(SearchContractError, match='SEMANTIC_CALL_BUDGET'):
        port.complete(role='pattern_gradient', split='optimize', messages=[])


class FixtureStateModule(NoModule):
    def binding(self):
        return dict(identity='SYNTHETIC_STATE_MODULE', semantic_calls_per_opportunity=0,
                    maximum_state_bytes=128, implementation='test_fixture_only')

    def initial_state(self): return dict(seen=0)

    def prepare(self, context, state, provider):
        base = super().prepare(context, state, provider)
        return PreparedFeedback(base.errors, None, dict(context_only_sentinel=state['seen']), dict(), state)

    def observe(self, outcome, state): return dict(seen=state['seen'] + 1)


def fixture_registry():
    r = ModuleRegistry(); r.register('fixture_state', FixtureStateModule)
    return r


def test_stateful_extension_is_per_repeat_and_optimizer_only():
    registry = fixture_registry()
    plan = small_plan(registry=registry, module='fixture_state')
    for cell in plan['cells'][:2]:
        broker = ScriptedPort()
        result, _ = run_small(plan, broker, registry, cell)
        assert result['module_state'] == dict(seen=3)
        assert broker.packets[0]['module_context'] == dict(context_only_sentinel=0)
        assert 'context_only_sentinel' not in json.dumps(broker.evaluations)


def test_completed_boundary_resume_preserves_module_state_and_paid_prefix():
    registry = fixture_registry(); plan = small_plan(registry=registry, module='fixture_state')
    complete, artifacts = run_small(plan, registry=registry)
    # Resume just the first completed opportunity under exactly the same plan.
    resume = dict(checkpoints=[artifacts['checkpoint_0'], artifacts['checkpoint_1']],
        historical_usage=dict(provider_calls=1, input_tokens=10, output_tokens=5))
    broker = ScriptedPort()
    continued, _ = run_small(plan, broker, registry, resume=resume)
    assert continued['final_prompt'] == complete['final_prompt'] and continued['module_state'] == dict(seen=3)
    assert broker.packets[0]['module_context'] == dict(context_only_sentinel=1)
    assert len(broker.evaluations) == 4 and continued['usage'] == complete['usage']
    corrupted = deepcopy(resume); corrupted['checkpoints'][-1]['module_state']['seen'] = 900
    with pytest.raises(SearchContractError, match='INTEGRITY'): run_small(plan, registry=registry, resume=corrupted)
    other_repeat = plan['cells'][1]
    with pytest.raises(SearchContractError, match='IDENTITY'): run_small(plan, registry=registry, cell=other_repeat, resume=resume)


def test_optimizer_error_copying_cannot_be_smuggled_by_module():
    class Bad(NoModule):
        def prepare(self, context, state, provider):
            base = super().prepare(context, state, provider)
            base.errors[0]['reference'] = 'foreign reference'
            return base
    registry = ModuleRegistry(); registry.register('bad', Bad)
    plan = small_plan(registry=registry, module='bad')
    with pytest.raises(SearchContractError, match='PROVENANCE'): run_small(plan, registry=registry)


def test_default_full_nine_cell_twenty_opportunity_smoke(tmp_path):
    plan = build_plan(default_config())
    summary = smoke(plan, tmp_path / 'fresh_smoke')
    assert summary['status'] == 'SYNTHETIC_SMOKE_COMPLETE_NOT_EFFICACY'
    assert summary['real_api_calls'] == summary['network_attempts'] == summary['unresolved_reservations'] == 0
    assert len(summary['rows']) == 9 and len(summary['within_arm_repeat_pairs']) == 9
    assert all(g['repeats'] == 3 and g['rejected'] == 60 for g in summary['groups'])
    assert all(g['updates'] == g['exhausted'] == 0 for g in summary['groups'])
    # Request realization identities differ by repeat, but pair A/B identical
    # initial Prompts within one repeat. These are actual checkpoint observations.
    cp = lambda arm, repeat: json.loads((tmp_path / 'fresh_smoke' / f'{arm}_seed42_repeat{repeat:02d}' / 'checkpoint_0.json').read_text())
    assert cp('A', 1)['optimize'] == cp('B', 1)['optimize']
    assert cp('A', 1)['optimize'] != cp('A', 2)['optimize']
    for n in range(1, 21):
        audit = json.loads((tmp_path / 'fresh_smoke/C_seed42_repeat01' / f'round_{n}_module.json').read_text())
        assert audit['audit']['focus']['selected_pattern_id'] is not None


def test_draft_preflight_is_not_current_team_execution():
    from scripts.run_experiment import preflight
    result = preflight(default_config())
    assert not result['ready_to_run'] and result['real_api_calls'] == 0 and result['cells'] == 9


@pytest.mark.parametrize('failure_type', ['timeout', 'connection'])
def test_accounted_network_retries_do_not_consume_semantic_or_generation_budget(tmp_path, monkeypatch, failure_type):
    import httpx
    from openai import APITimeoutError, APIConnectionError
    from multi_dataset_diverse_rl.research.ablations.offline_runtime import OfflineAblationBroker
    from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_GATED_32_UNCAPPED
    from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
    import multi_dataset_diverse_rl.research.single_prompt_runtime as runtime
    delays = []
    monkeypatch.setattr(runtime.time, 'sleep', delays.append)
    class Flaky(SyntheticTransport):
        def __init__(self):
            super().__init__(); self.remaining = 8
        def for_role(self, role, request):
            if self.remaining:
                self.remaining -= 1
                req = httpx.Request('POST', 'https://synthetic.invalid')
                if failure_type == 'timeout': raise APITimeoutError(request=req)
                raise APIConnectionError(request=req)
            return super().for_role(role, request)
    plan = build_plan(default_config()); scope_id = digest(['synthetic_retry', failure_type])
    ledger = TokenLedger(tmp_path / 'ledger', task_sha256=scope_id, policy=POLICY_GATED_32_UNCAPPED)
    broker = OfflineAblationBroker(plan=plan, phase='train', scope=dict(scope_identity=scope_id,
        token_ceiling=None, call_ceiling=1, network_recovery=True), transport=Flaky(), ledger=ledger,
        receipts=ProviderResponseReceipts(tmp_path / 'receipts', attempt_id='fixture', startup_identity_sha256=scope_id),
        dispatch_guard=lambda _: None)
    try:
        broker.bind_cell(plan['cells'][0], plan)
        response = broker.complete(role='reflection', split='optimize', stage='synthetic_generation',
            messages=[dict(role='system', content='Synthetic instruction'), dict(role='user', content='Synthetic input')])
        assert response['finish_reason'] == 'stop'
        assert broker.usage['provider_calls'] == 9 and broker.usage['network_failed_calls'] == 8
        assert broker.budgeted_provider_calls() == 1 and not ledger.inflight
        assert delays == [2, 4, 8, 16, 30, 60, 60, 60]
    finally:
        ledger.close()
