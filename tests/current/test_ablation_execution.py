"""Fresh phase authorization, terminal audits and Solver-only cache boundaries."""
from copy import deepcopy
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from multi_dataset_diverse_rl.research.ablations import governance as gov
from multi_dataset_diverse_rl.research.ablations.config import build_plan, batch_config as default_config
from multi_dataset_diverse_rl.research.ablations.offline_runtime import OfflineAblationBroker
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport, synthetic_rows
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_GATED_32_UNCAPPED
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json, read_json
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.system_prompt import SEED

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def broker(tmp_path):
    plan = build_plan(default_config())
    scope = dict(scope_identity=digest('test'), call_ceiling=plan['training_semantic_call_ceiling'],
        token_ceiling=None, network_recovery=True)
    ledger = TokenLedger(tmp_path / 'ledger', task_sha256=scope['scope_identity'], policy=POLICY_GATED_32_UNCAPPED)
    transport = SyntheticTransport()
    value = OfflineAblationBroker(plan=plan, scope=scope, phase='train', transport=transport, ledger=ledger,
        receipts=ProviderResponseReceipts(tmp_path / 'receipts', attempt_id='fake', startup_identity_sha256=scope['scope_identity']),
        dispatch_guard=lambda _: None)
    value.bind_cell(plan['cells'][0], plan)
    yield value, plan, transport
    assert not ledger.inflight
    ledger.close()


def test_same_repeat_solver_reused_across_arms_rounds_only(broker):
    b, plan, transport = broker
    row = synthetic_rows('optimize', 1)[0]
    first = b.prediction(SEED, row, 'optimize', 42, 0)
    b.bind_cell(next(c for c in plan['cells'] if c['arm_id'] == 'B' and c['repeat_id'] == 1), plan)
    assert b.prediction(SEED, row, 'optimize', 42, 17) == first
    assert b.usage_view()['solver'] == 1 and b.usage_view()['cache_hits'] == 1
    b.bind_cell(plan['cells'][1], plan)
    second = b.prediction(SEED, row, 'optimize', 42, 17)
    assert second['request_sha256'] != first['request_sha256'] and b.usage_view()['solver'] == 2
    log = [json.loads(line) for line in (b.receipts.directory / 'solver_cache_access.jsonl').read_text().splitlines()]
    assert log[0]['observation_sha256'] == log[1]['observation_sha256']
    assert log[0]['request_sha256'] != log[2]['request_sha256']


@pytest.mark.parametrize('changed', ['prompt', 'problem', 'reference', 'example_id', 'split', 'capacity'])
def test_complete_solver_cache_identity(broker, changed):
    b, _, _ = broker
    row = synthetic_rows('optimize', 1)[0]
    first = b.prediction(SEED, row, 'optimize', 42, 0)
    prompt, split = SEED, 'optimize'
    if changed == 'prompt': prompt = SEED.edit('strategy', 'Check arithmetic carefully.')
    elif changed == 'problem': row['content']['problem'] = 'Another synthetic fixture value: 1'
    elif changed == 'reference': row['reference_final_answer'] = '1'
    elif changed == 'example_id': row['stable_example_id'] += '_other'
    elif changed == 'split': split = 'validation'
    else:
        b.plan['config']['runtime']['first_solver_capacity'] = 3601
    second = b.prediction(prompt, row, split, 42, 9)
    assert second['request_sha256'] != first['request_sha256']
    assert b.usage_view()['solver'] == 2


@pytest.mark.parametrize('role', ['reflection', 'pattern_gradient', 'pattern_cluster'])
def test_identical_non_solver_messages_always_call_transport(broker, role):
    b, _, _ = broker
    message = json.dumps(dict(gradients=[dict(example_id='g1')])) if role == 'pattern_cluster' else '{}'
    args = dict(role=role, split='optimize', stage='identical', messages=[dict(role='system', content='fixture'),
        dict(role='user', content=message)])
    b.complete(**args)
    b.complete(**args)
    assert b.usage_view()[role] == 2 and b.usage_view()['cache_hits'] == 0


def test_paid_teacher_replay_explicitly_forbidden(broker):
    b, _, _ = broker
    b.replay = object()
    with pytest.raises(SearchContractError, match='REPLAY_FORBIDDEN'):
        b.complete(role='reflection', split='optimize', stage='x', messages=[])
    assert b.usage_view()['provider_calls'] == 0


def test_shared_terminal_semantic_recovery_not_invalid_first_response(broker):
    b, _, transport = broker
    original = transport.for_role
    calls = []
    def invalid_then_valid(role, request):
        result = original(role, request)
        calls.append(request)
        if len(calls) == 1:
            result.update(text='Incomplete derivation', finish_reason='length')
        return result
    transport.for_role = invalid_then_valid
    row = synthetic_rows('optimize', 1)[0]
    first = b.prediction(SEED, row, 'optimize', 42, 0)
    assert first['valid'] and len(calls) == 2 and calls[1]['max_tokens'] == 6144
    assert b.prediction(SEED, row, 'optimize', 42, 1) == first and len(calls) == 2


def test_inflight_identical_solver_requests_share_one_realization(broker):
    b, _, _ = broker
    row = synthetic_rows('optimize', 1)[0]
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: b.prediction(SEED, row, 'optimize', 42, 0), range(8)))
    assert all(v == values[0] for v in values)
    assert b.usage_view()['solver'] == 1 and b.usage_view()['cache_hits'] == 7


def test_frozen_manifest_metadata_and_formal_test_alignment():
    manifest = gov.manifest_template(ROOT)
    assert read_json(ROOT / gov.MANIFEST) == manifest
    reader = gov.validate_manifest(ROOT, manifest)
    final = gov.test_plan(ROOT, reader, manifest['plan'])
    assert final['count'] == 300 and final['solver_concurrency'] == 16
    assert final['successful_semantic_call_ceiling'] == 10800 and final['physical_call_ceiling'] is None
    assert final['solver_decoding'] == manifest['config']['runtime']['solver_decoding_policy']


def test_five_repeat_parent_is_preserved_but_not_executable():
    from scripts.run_experiment import preflight
    historical = read_json(ROOT / gov.HISTORICAL_MANIFEST)
    gov.validate_historical_manifest(ROOT, historical)
    assert historical['config']['repeats_per_arm'] == 5 and len(historical['plan']['cells']) == 15
    assert preflight(historical)['status'] == 'ABLATION_HISTORICAL_PREREGISTRATION_ONLY_NOT_EXECUTABLE'
    with pytest.raises(SearchContractError): gov.validate_manifest(ROOT, historical)
    with pytest.raises(SearchContractError):
        gov.validate_prep(ROOT, ROOT / 'runs/ablations/pattern_seed42_repeat5_20261015/prep')


def test_three_repeat_handoff_counts_are_derived_from_frozen_plan(tmp_path):
    scope = gov._scope(ROOT, ROOT / 'runs/ablations/synthetic_metadata/prep')
    atomic_write_json(tmp_path / 'prep.json', dict(scope=scope))
    # Use an ignored repository-local destination so all relative handoff paths bind.
    destination = ROOT / 'runs/ablations/synthetic_metadata/prep'
    import unittest.mock
    with unittest.mock.patch.object(gov, 'file_hash', wraps=gov.file_hash) as hash_file:
        hash_file.side_effect = lambda p: digest('synthetic prep') if p == destination / 'prep.json' else hash_file._mock_wraps(p)
        handoff = gov.handoff_template(ROOT, destination, scope)
    assert handoff['execution']['repeat_ids'] == [1, 2, 3]
    assert handoff['execution']['opportunity_budget'] == 180
    assert len(handoff['execution']['expected_checkpoint_paths']) == 189
    assert scope['call_ceiling'] == 190440
    assert scope['test_plan']['successful_semantic_call_ceiling'] == 10800


@pytest.mark.parametrize('mutation', ['unapproved', 'wrong_scope', 'wrong_attempt', 'validation', 'test', 'evidence'])
def test_exact_phase_authorization_fail_closed(mutation):
    scope = dict(scope_identity=digest('scope'), plan=build_plan(default_config()))
    auth = gov.authorization_template(scope)
    auth.update(approved=True, adaptive_validation_access=True, test_access=True,
        user_authorization_evidence_sha256=digest('human'))
    if mutation == 'unapproved': auth['approved'] = False
    elif mutation == 'wrong_scope': auth['scope_identity'] = digest('foreign')
    elif mutation == 'wrong_attempt': auth['authorized_attempt_ids'][0] += '_foreign'
    elif mutation == 'validation': auth['adaptive_validation_access'] = False
    elif mutation == 'test': auth['test_access'] = False
    else: auth['user_authorization_evidence_sha256'] = None
    with pytest.raises(SearchContractError): gov.validate_authorization(scope, auth, required=True, test=True)


def test_unapproved_execution_cannot_construct_provider_or_create_run(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(gov, '_transport', lambda: called.append(True))
    monkeypatch.setattr(gov, 'validate_prep', lambda *a, **k: (_ for _ in ()).throw(SearchContractError('UNAUTHORIZED')))
    with pytest.raises(SearchContractError, match='UNAUTHORIZED'):
        gov.execute(ROOT, ROOT / 'runs/ablations/nonexistent/prep', ROOT / 'runs/ablations/nonexistent/train_run')
    assert not called and not (ROOT / 'runs/ablations/nonexistent/train_run').exists()


def test_gate_capability_must_authorize_before_content_read():
    class Reader:
        members = [dict(stable_example_id=f'v{i}', project_split='validation') for i in range(300)]
        def _read_rows(self, *a, **k): raise AssertionError('content read')
    reader = Reader()
    access = gov.GateAccess(digest('scope'), gov.gate_membership(reader),
        lambda: (_ for _ in ()).throw(SearchContractError('DENIED')))
    with pytest.raises(SearchContractError, match='DENIED'): gov.read_gate_rows(reader, access)


def test_test_requires_exact_complete_training_terminal_before_access(tmp_path):
    scope = dict(scope_identity=digest('scope'), source_sha='0' * 40)
    atomic_write_json(tmp_path / 'terminal.json', dict(status='EXECUTION_ABORTED', phase='train'))
    atomic_write_json(tmp_path / 'access.json', dict(test_access=False))
    with pytest.raises(SearchContractError, match='SEALED_TRAINING'):
        gov.verify_training(tmp_path, scope)


@pytest.fixture
def phase_smoke(tmp_path, monkeypatch, request):
    if getattr(request,'param',False):
        from multi_dataset_diverse_rl.research.ablations.config import default_config as full_config
        config=full_config()
    else:
        config=default_config()
    config['rounds'] = config['runtime']['rounds'] = 1 if getattr(request,'param',False) else 2
    config['repeats_per_arm'] = config['runtime']['repeats_per_arm'] = 1
    plan = build_plan(config)
    optimize = synthetic_rows('optimize', 150)
    pool = synthetic_rows('validation', 300)
    test = synthetic_rows('test', 300)
    class Reader:
        manifest_sha256 = digest('split')
        members = [dict(stable_example_id=r['stable_example_id'], project_split='validation') for r in pool]
        def rows(self, split, purpose):
            assert split == 'optimize'
            return optimize
        def _read_rows(self, split, frozen_subset_ids):
            assert split == 'validation'
            return tuple(r for r in pool if r['stable_example_id'] in frozen_subset_ids)
        def heldout_rows(self, access):
            assert (tmp_path / 'runs/attempt/train_run/terminal.json').exists()
            access.require_authorization()
            return test
    reader = Reader()
    prep = tmp_path / 'runs/attempt/prep'
    train = prep.parent / 'train_run'
    manifest = dict(initial_prompt_path='initial.json')
    atomic_write_json(tmp_path / gov.MANIFEST, manifest)
    atomic_write_json(tmp_path / 'initial.json', dict(members=[dict(prompt=SEED.to_dict())]))
    atomic_write_json(prep / 'prep.json', dict(fake=True))
    atomic_write_json(prep / 'handoff.json', dict(fake=True))
    atomic_write_json(prep / 'execution_plan.json', dict(fake=True))
    scope = dict(phase='train', source_sha='0' * 40, source_hashes={}, plan=plan,
        gate_membership=gov.gate_membership(reader), initial_prompt_sha256=SEED.prompt_hash,
        optimize_sample_ids=[r['stable_example_id'] for r in optimize],
        test_plan=dict(sample_ids=[r['stable_example_id'] for r in test], successful_semantic_call_ceiling=3600),
        call_ceiling=plan['training_semantic_call_ceiling'], token_ceiling=None,
        network_recovery=plan['config']['runtime']['timeout_recovery'])
    scope['scope_identity'] = digest(scope)
    auth = gov.authorization_template(scope)
    auth.update(approved=True, adaptive_validation_access=True, test_access=True,
        user_authorization_evidence_sha256=digest('fake authorization'))
    atomic_write_json(prep / 'authorization.json', auth)
    monkeypatch.setattr(gov, 'validate_prep', lambda *a, **k: scope)
    monkeypatch.setattr(gov, 'validate_manifest', lambda *a, **k: reader)
    class Clients:
        def close(self): pass
    transport = SyntheticTransport()
    monkeypatch.setattr(gov, '_transport', lambda: (transport, Clients()))
    gov._execute_phase(tmp_path, prep, train, scope, train_scope=scope)
    return tmp_path, prep, train, scope, transport


def test_governed_fake_training_seals_gates_receipts_and_then_test(phase_smoke):
    root, prep, train, scope, transport = phase_smoke
    results, terminal_sha = gov.verify_training(train, scope)
    assert len(results) == 3 and all(r['rounds_completed'] == 2 for r in results)
    assert read_json(train / 'access.json')['test_access'] is False
    summary = read_json(train / 'summary.json')
    assert all(g['rejected'] == 2 and g['exhausted'] == 0 and g['updates'] == 0 for g in summary['groups'])
    body = dict(phase='test', source_sha=scope['source_sha'], parent_scope_identity=scope['scope_identity'],
        parent_terminal_sha256=terminal_sha, plan=scope['plan'], test_plan=scope['test_plan'],
        call_ceiling=3600, token_ceiling=None, network_recovery=scope['network_recovery'])
    test_scope = dict(body, scope_identity=digest(body))
    result = gov._execute_phase(root, prep, train.parent / 'test_run', test_scope,
        train_scope=scope, parent_results=results, terminal_sha=terminal_sha)
    assert result['status'] == 'EXECUTION_COMPLETE'
    test_summary = read_json(train.parent / 'test_run/execution_summary.json')
    assert test_summary['usage']['solver'] == 300 and test_summary['usage']['cache_hits'] == 600
    assert gov.verify_training(train, scope)[1] == terminal_sha


def test_training_mutation_during_test_cannot_be_sealed_complete(phase_smoke):
    root, prep, train, scope, transport = phase_smoke
    results, terminal_sha = gov.verify_training(train, scope)
    original = transport.for_role
    changed = []
    def mutate_parent(role, request):
        value = original(role, request)
        if not changed:
            changed.append(True)
            atomic_write_json(train / 'unexpected.json', dict(corrupted=True))
        return value
    transport.for_role = mutate_parent
    test_root = train.parent / 'test_run'
    with pytest.raises(SearchContractError, match='SEALED_TRAINING'):
        gov._execute_phase(root, prep, test_root, gov.final_test_scope(scope, terminal_sha),
            train_scope=scope, parent_results=results, terminal_sha=terminal_sha)
    assert read_json(test_root / 'lifecycle.json')['status'] == 'EXECUTION_ABORTED'
    assert not (test_root / 'terminal.json').exists()


@pytest.mark.parametrize('tamper', ['checkpoint', 'result', 'receipt', 'test_access', 'new_file'])
def test_tampered_training_cannot_enable_test(phase_smoke, tamper):
    _, _, train, scope, _ = phase_smoke
    gov.verify_training(train, scope)
    cell = scope['plan']['cells'][0]['cell_id']
    if tamper == 'checkpoint':
        path = train / cell / 'checkpoint_1.json'
        value = read_json(path); value['metric']['accepted'] = True; atomic_write_json(path, value)
    elif tamper == 'result':
        path = train / 'results.json'; value = read_json(path); value[0]['final_validation_correct'] += 1; atomic_write_json(path, value)
    elif tamper == 'receipt':
        path = next((train / 'provider_receipts').glob('[0-9]*.json')); value = read_json(path); value['record']['response']['text'] += 'x'; atomic_write_json(path, value)
    elif tamper == 'test_access': atomic_write_json(train / 'access.json', dict(test_access=True))
    else: atomic_write_json(train / 'extra.json', dict(unregistered=True))
    with pytest.raises((SearchContractError, ValueError)):
        gov.verify_training(train, scope)
