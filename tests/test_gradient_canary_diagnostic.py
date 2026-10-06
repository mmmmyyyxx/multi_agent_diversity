import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.governance import gradient_canary_diagnostic as diagnostic
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ALIASES = {'e1': 'id-one', 'e2': 'id-two', 'e3': 'id-three'}


def partition():
    return dict(patterns=[dict(failure_mechanism='Incomplete constraint tracking',
        update_direction='Track admissibility constraints during each transformation', support_ids=['e1', 'e2'])], unassigned_ids=['e3'])


def test_support_and_unassigned_accounting():
    result = diagnostic.parse_raw_partition(partition(), ALIASES)
    metrics = result['metrics']
    assert metrics['shared_support_count'] == 2
    assert metrics['shared_support_fraction'] == pytest.approx(2 / 3)
    assert metrics['assigned_fraction'] + metrics['unassigned_fraction'] == 1
    assert result['patterns'][0]['support_ids'] == ['id-one', 'id-two']


@pytest.mark.parametrize('invalid', [
    dict(patterns=[dict(failure_mechanism='a', update_direction='b', support_ids=['e1', 'unknown'])], unassigned_ids=['e2', 'e3']),
    dict(patterns=[dict(failure_mechanism='a', update_direction='b', support_ids=['e1', 'e1'])], unassigned_ids=['e2', 'e3']),
    dict(patterns=[dict(failure_mechanism='a', update_direction='b', support_ids=['e1'])], unassigned_ids=['e1', 'e2', 'e3']),
    dict(patterns=[dict(failure_mechanism='a', update_direction='b', support_ids=['e1'])], unassigned_ids=['e2']),
])
def test_membership_corruption_is_rejected(invalid):
    with pytest.raises(SearchContractError, match='MEMBERSHIP'):
        diagnostic.parse_raw_partition(invalid, ALIASES)


def test_structural_merge_retains_frozen_normalization():
    value = partition()
    value['patterns'][0]['support_ids'] = ['e1']
    value['patterns'].append(dict(failure_mechanism='  INCOMPLETE constraint tracking ',
        update_direction='Track admissibility constraints during each transformation', support_ids=['e2']))
    assert diagnostic.parse_raw_partition(value, ALIASES)['metrics']['pattern_count'] == 1


def test_singleton_metrics_are_descriptive():
    value = dict(patterns=[dict(failure_mechanism='Distinct correction ' + letter,
        update_direction='Track ' + letter, support_ids=[alias]) for alias, letter in zip(ALIASES, 'abc')], unassigned_ids=[])
    metrics = diagnostic.parse_raw_partition(value, ALIASES)['metrics']
    assert metrics['singleton_fraction'] == 1
    assert metrics['shared_support_fraction'] == 0
    assert metrics['normalized_entropy'] == pytest.approx(1)


def test_early_diagnostic_is_forbidden(tmp_path):
    atomic_write_json(tmp_path / 'lifecycle.json', dict(status='RUNNING'))
    atomic_write_json(tmp_path / 'execution_summary.json', {})
    with pytest.raises(SearchContractError, match='CLOSED_ACTIVE'):
        diagnostic.matched_examples(tmp_path)


def test_failed_transport_is_one_call_and_does_not_mutate_active_run(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    binding = json.loads((root / 'experiments/execution_bindings/math_v2_1_gradient_pattern_offline_profile_v1.json').read_text(encoding='utf-8'))
    binding['token_ledger_directory'] = 'runs/fake_ledger'
    binding['initial_team_path'] = 'team.json'
    binding['validation_accounting_metadata_path'] = 'reserve.json'
    atomic_write_json(tmp_path / 'binding.json', binding)
    atomic_write_json(tmp_path / 'reserve.json', {})
    atomic_write_json(tmp_path / 'raw_prompt.json', dict(identity='set_level_wrong_pattern_discovery_v3', prompt='Frozen historical prompt'))
    active = tmp_path / 'runs/active'
    active.mkdir(parents=True)
    atomic_write_json(active / 'lifecycle.json', dict(status='EXECUTION_COMPLETE'))
    atomic_write_json(active / 'execution_summary.json', dict(pattern_gradient_calls=1))
    atomic_write_json(active / 'consumed_authorization.json', dict(scope=dict(attempt_id='attempt', source_sha='source'), startup_identity_sha256='startup'))
    row = dict(role='pattern_gradient', response={}, request=dict(messages=[{}, dict(content=json.dumps(dict(
        current_member_procedure='Synthetic member procedure', example=dict(example_id='id-one', problem='synthetic variable task', reference='symbolic answer', prediction='wrong'))))]))
    (active / 'provider_trace_private.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    scope = dict(role=diagnostic.ROLE, mode=diagnostic.MODE, physical_call_ceiling=1, cache_hits=0,
        active_attempt_id='attempt', active_startup_identity_sha256='startup', source_sha='source',
        binding_path='binding.json', binding_sha256=sha256((tmp_path / 'binding.json').read_bytes()).hexdigest(),
        diagnostic_module_sha256=sha256(Path(diagnostic.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        prompt_path='raw_prompt.json', prompt_sha256=sha256((tmp_path / 'raw_prompt.json').read_bytes()).hexdigest(),
        raw_partition_schema='synthetic-schema', model=binding['models']['pattern'], generation_policy=binding['optimizer_generation_policy'])
    atomic_write_json(tmp_path / 'auth.json', dict(explicit_user_authorized=True, single_use=True, consumed=False, scope=scope))
    from multi_dataset_diverse_rl.governance import unified_execution
    monkeypatch.setattr(unified_execution, 'execution_identity', lambda *a: {})
    monkeypatch.setattr(unified_execution, 'verify_source_commit', lambda *a: None)
    monkeypatch.setattr(diagnostic, 'initial_prompts', lambda *a: ())
    monkeypatch.setattr(diagnostic, 'ValidationReserve', lambda *a: SimpleNamespace(remaining=lambda: 0))
    calls = []
    charges = []
    class FakeLedger:
        def __init__(self, *a, **k): pass
        def view(self): return {}
        def reserve(self, request, **kwargs):
            assert kwargs['role'] == 'diagnostic_raw_pattern'
            return 'reservation'
        def reconcile(self, key, result, **kwargs):
            charges.append(kwargs['outcome'])
            return dict(input_tokens=1, output_tokens=1, usage_reliable=False)
        def close(self): pass
    monkeypatch.setattr(diagnostic, 'TokenLedger', FakeLedger)
    def failed_transport(request):
        calls.append(request)
        raise ConnectionError('Synthetic transport failure')
    monkeypatch.setattr(diagnostic, 'create_transport', lambda *a: (failed_transport, SimpleNamespace(close=lambda: None)))
    outcome = diagnostic.execute_diagnostic(tmp_path, active, tmp_path / 'runs/diagnostic', tmp_path / 'auth.json')
    assert len(calls) == outcome['physical_calls'] == 1
    assert charges == ['DIAGNOSTIC_FAILURE_NO_RETRY']
    assert outcome['status'] == 'NOT_COMPARABLE' and outcome['active_run_unchanged']
    with pytest.raises(SearchContractError, match='EXACT_FRESH'):
        diagnostic.execute_diagnostic(tmp_path, active, tmp_path / 'runs/diagnostic', tmp_path / 'auth.json')
