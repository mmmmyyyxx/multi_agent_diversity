"""Zero-API tests of independent scientific controls and governed boundaries."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.research.single_prompt import run_cell, batch_ids, generate_candidate
from multi_dataset_diverse_rl.research.single_prompt_policy import policy, call_ceiling, ARMS
from multi_dataset_diverse_rl.research.single_prompt_analysis import pair_metrics, analyze_final
from multi_dataset_diverse_rl.research.single_prompt_runtime import StudyBroker
from multi_dataset_diverse_rl.research.single_prompt_governance import (
    validate_manifest, private_path, StudySplitReader, verify_search, execute, validate_prep,
)
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_V27_FORMAL_UNCAPPED, OperationalAbort
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.durable_io import read_json, atomic_write_json
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.math_flexible_answer import classify_prediction
from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import resolve_predictions

ROOT = Path(__file__).resolve().parents[2]


class ToyTransport:
    """Synthetic arithmetic/control fixture; never a real efficacy observation."""
    def __init__(self):
        self.calls = []
        self.reflections = 0

    def for_role(self, role, request):
        self.calls.append((role, deepcopy(request)))
        if role == 'solver':
            number = int(request['messages'][1]['content'].split(':')[-1])
            text = f'Final answer: {number + 1}'
        elif role == 'reflection':
            self.reflections += 1
            text = json.dumps(dict(block='strategy', content='Check calculations carefully. ' + 'Verify the result. ' * self.reflections))
        elif role == 'pattern_gradient':
            text = json.dumps(dict(disposition='ACTIONABLE', observed_failure='The final value disagrees with the reference.',
                diagnosis='A calculation step may require checking.', reusable_correction='Check arithmetic consistency before finalizing.',
                suggested_block='strategy', expected_effect='Reduce calculation errors.'))
        elif role == 'pattern_cluster':
            packet = json.loads(request['messages'][1]['content'])
            text = json.dumps(dict(patterns=[dict(generalized_gradient='Check arithmetic consistency before finalizing.',
                support_ids=[g['example_id'] for g in packet['gradients']])], unassigned_ids=[]))
        else:
            raise AssertionError(role)
        return dict(text=text, finish_reason='stop', input_tokens=80, output_tokens=40,
                    provider_thinking_indicators=[], provider_reasoning_character_count=0,
                    provider_reasoning_content_present=False,
                    provider_usage_details=dict(completion_tokens_details=dict(reasoning_tokens=0)))


def toy_rows(n=5):
    return tuple(dict(stable_example_id=f'toy_{i}', reference_final_answer=str(i),
        content=dict(problem=f'Synthetic fixture value: {i}', solution=f'The reference value is {i}.')) for i in range(n))


def cell(arm='full_errors', seed=88):
    return dict(cell_id=f'{arm}_seed{seed}', arm=arm, seed=seed, initial_prompt=SEED.to_dict())


def broker_for(tmp_path, *, phase='train', transport=None, cap=10**9):
    ledger = TokenLedger(tmp_path / 'ledger', task_sha256='a' * 64, policy=POLICY_V27_FORMAL_UNCAPPED)
    receipts = ProviderResponseReceipts(tmp_path / 'receipts', attempt_id='toy', startup_identity_sha256='a' * 64)
    broker = StudyBroker(phase=phase, scope=dict(scope_identity='a' * 64, token_ceiling=cap, call_ceiling=100000),
        transport=transport or ToyTransport(), ledger=ledger, receipts=receipts, dispatch_guard=lambda _: None)
    broker.attempt_id, broker.cell_id, broker.seed = 'toy', 'toy', 88
    return broker


def observation(correct=False):
    pred = resolve_predictions([classify_prediction('Final answer: 1')])
    return dict(correct=correct, valid=True, answer='1', prediction=asdict(pred))


def test_confirmed_manifest_and_resource_arithmetic():
    p = policy()
    assert p['rounds'] == 20 and p['error_batch_size'] == 3 and p['seeds'] == [88, 89, 90]
    assert [call_ceiling(x) for x in ('train', 'validation', 'test')] == [114480, 226800, 10800]
    manifest = read_json(ROOT / 'experiments/manifests/single_prompt_gradient_pattern_v1.json')
    reader = validate_manifest(ROOT, manifest)
    assert reader.metadata('test')['count'] == 300
    with pytest.raises(SearchContractError, match='POLICY_MISMATCH'):
        validate_manifest(ROOT, dict(manifest, policy=dict(p, rounds=19)))
    with pytest.raises(SearchContractError, match='HELDOUT_SEARCH_ACCESS'):
        from multi_dataset_diverse_rl.benchmarks.access import DataPurpose
        reader.rows('test', DataPurpose.SEARCH_EVALUATION)
    with pytest.raises(SearchContractError, match='HELDOUT_CAPABILITY'):
        reader.heldout_rows(None)


def test_paired_error_batch_has_no_sampling_in_full_arm():
    ids = tuple(f'error_{i}' for i in range(50))
    assert batch_ids(ids, 88, 1, 3) == batch_ids(ids, 88, 1, 3)
    assert len(batch_ids(ids, 88, 1, 3)) == 3
    assert batch_ids(ids[:2], 88, 1, 3) == ids[:2]
    assert len({batch_ids(ids, 88, round_index, 3) for round_index in range(1, 21)}) > 1


@pytest.mark.parametrize('arm', ('full_errors', 'batch_errors'))
def test_accuracy_decline_is_installed_and_last_round_is_final(tmp_path, arm):
    broker = broker_for(tmp_path)
    artifacts = {}
    def evaluate(prompt, rows, split, seed, ordinal):
        return {r['stable_example_id']: observation(correct=ordinal != 20 and (ordinal != 1 or i == 0)) for i, r in enumerate(rows)}
    # Include initial errors so the first declining update actually generates.
    def evaluator(prompt, rows, split, seed, ordinal):
        result = evaluate(prompt, rows, split, seed, ordinal)
        if ordinal == 0:
            result[rows[-1]['stable_example_id']]['correct'] = False
        return result
    try:
        result = run_cell(cell(arm), toy_rows(5), evaluator, broker, lambda n, v: artifacts.setdefault(n, v))
        assert result['rounds_completed'] == 20 and len(result['prompt_history']) == 21
        first = result['metrics'][0]
        assert first['accuracy_after'] < first['accuracy_before'] and first['update_status'] == 'REPLACED'
        assert result['final_accuracy'] == 0 and result['final_prompt'] == result['prompt_history'][20]
        assert max(m['accuracy_after'] for m in result['metrics']) == 1
    finally:
        broker.ledger.close()


def test_full_errors_complete_records_and_batch_optimizer_instruction_match(tmp_path):
    packets = []
    for arm in ('full_errors', 'batch_errors'):
        broker = broker_for(tmp_path / arm)
        rows = toy_rows(5)
        def evaluator(*args):
            return {r['stable_example_id']: observation() for r in rows}
        try:
            result = run_cell(cell(arm), rows, evaluator, broker, lambda *_: None)
            calls = [(r, p) for r, p in broker.transport.calls if r == 'reflection']
            packet = json.loads(calls[0][1]['messages'][1]['content'])
            packets.append((calls[0][1]['messages'][0]['content'], packet))
            assert all(m['optimizer_example_count'] == (5 if arm == 'full_errors' else 3) for m in result['metrics'])
            assert all(e['reference_solution'] == next(r['content']['solution'] for r in rows if r['stable_example_id'] == e['example_id']) for e in packet['errors'])
        finally:
            broker.ledger.close()
    assert packets[0][0] == packets[1][0]
    assert packets[0][1]['structured_feedback'] is None and packets[1][1]['structured_feedback'] is None


def test_no_errors_carries_for_all_twenty_rounds_without_optimizer(tmp_path):
    broker = broker_for(tmp_path)
    rows = toy_rows(2)
    try:
        result = run_cell(cell(), rows, lambda *args: {r['stable_example_id']: observation(True) for r in rows}, broker, lambda *_: None)
        assert result['rounds_completed'] == 20
        assert all(m['update_status'] == 'NO_ERRORS_CARRY' for m in result['metrics'])
        assert not broker.transport.calls
    finally:
        broker.ledger.close()


def test_invalid_candidate_two_draws_then_carry_and_provider_failure_not_swallowed(tmp_path):
    class Invalid(ToyTransport):
        def for_role(self, role, request):
            self.calls.append((role, request))
            return dict(text='bad json', finish_reason='stop', input_tokens=1, output_tokens=1,
                        provider_reasoning_content_present=False,
                        provider_usage_details=dict(completion_tokens_details=dict(reasoning_tokens=0)))
    broker = broker_for(tmp_path / 'invalid', transport=Invalid())
    try:
        child, draws = generate_candidate(SEED, dict(errors=[], structured_feedback=None), broker, 1)
        assert child == SEED and len(draws) == 2
    finally:
        broker.ledger.close()
    class Failure(ToyTransport):
        def for_role(self, role, request):
            raise RuntimeError('Synthetic provider failure')
    broker = broker_for(tmp_path / 'failed', transport=Failure())
    try:
        with pytest.raises(OperationalAbort, match='PROVIDER_FAILURE'):
            generate_candidate(SEED, dict(errors=[], structured_feedback=None), broker, 1)
        assert broker.usage['provider_calls'] == 1 and not broker.ledger.inflight
        assert len(list((tmp_path / 'failed/receipts').glob('*.json'))) == 1
    finally:
        broker.ledger.close()


def test_real_gradient_clustering_pipeline_and_no_team_state(tmp_path):
    broker = broker_for(tmp_path)
    artifacts = {}
    try:
        result = run_cell(cell('gradient_pattern'), toy_rows(3), broker.evaluate, broker, lambda n, v: artifacts.setdefault(n, v))
        assert result['rounds_completed'] == 20
        assert broker.usage['pattern_gradient'] == 60 and broker.usage['pattern_cluster'] == 20
        for metric in result['metrics']:
            feedback = metric['structured_feedback']
            assert len(feedback['gradients']) == 3 and len(feedback['patterns']) == 1
            assert len(feedback['patterns'][0]['support_ids']) == 3
        for role, request in broker.transport.calls:
            if role == 'solver':
                assert request['messages'][1]['content'].startswith('Synthetic fixture value:')
                assert len(request['messages']) == 2
                assert 'reference_solution' not in request['messages'][0]['content']
            elif role == 'reflection':
                feedback = json.loads(request['messages'][1]['content'])
                assert len(feedback['errors']) == 3
                assert 'patterns' in feedback['structured_feedback']
                assert not any(k in feedback for k in ('memory', 'team', 'history', 'validation'))
    finally:
        broker.ledger.close()


def test_paired_solver_cache_shares_same_seed_realization(tmp_path):
    broker = broker_for(tmp_path)
    row = toy_rows(1)[0]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(lambda _: broker.prediction(SEED, row, 'optimize', 88, 0), range(4)))
        assert all(o == outputs[0] for o in outputs)
        assert broker.usage['solver'] == 1
        broker.seed = 89
        broker.prediction(SEED, row, 'optimize', 89, 0)
        assert broker.usage['solver'] == 2
    finally:
        broker.ledger.close()


def test_context_and_budget_reject_before_dispatch_and_heldout_optimizer_denied(tmp_path):
    broker = broker_for(tmp_path / 'budget', cap=1)
    try:
        with pytest.raises(OperationalAbort, match='TOKEN_CEILING'):
            broker.evaluate(SEED, toy_rows(1), 'optimize', 88, 0)
        assert not broker.transport.calls
    finally:
        broker.ledger.close()
    broker = broker_for(tmp_path / 'context')
    try:
        with pytest.raises(SearchContractError, match='CONTEXT_LIMIT'):
            broker.complete(role='reflection', split='optimize', stage='test', messages=[dict(role='user', content='x' * 1000000)])
        assert not broker.transport.calls
    finally:
        broker.ledger.close()
    broker = broker_for(tmp_path / 'heldout', phase='test')
    try:
        with pytest.raises(SearchContractError, match='ROLE_SPLIT'):
            broker.complete(role='reflection', split='test', stage='test', messages=[])
        assert not broker.transport.calls
    finally:
        broker.ledger.close()


def test_equivalence_diversity_and_complementarity_do_not_count_formatting():
    left = dict(a=dict(answer='1/2', valid=True, correct=True), b=dict(answer='4', valid=True, correct=False))
    right = dict(a=dict(answer='0.5', valid=True, correct=True), b=dict(answer='5', valid=True, correct=True))
    from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
    result = pair_metrics(left, right, MATHBenchmarkAdapter().equivalent)
    assert result['answer_disagreement'] == .5
    assert result['oracle_gain'] == .25 and result['double_fault'] == 0
    invalid = {k: dict(answer='', valid=False, correct=False) for k in left}
    result = pair_metrics(invalid, invalid, lambda *_: pytest.fail('Invalid output must not enter equivalence'))
    assert result['answer_disagreement'] is None and result['double_fault'] == 1


def test_all_nine_analysis_has_complete_36_pairs():
    cells = [dict(cell(a, s), final_prompt=SEED.to_dict()) for a in ARMS for s in policy()['seeds']]
    predictions = {c['cell_id']: dict(x=dict(answer='1', valid=True, correct=True)) for c in cells}
    result = analyze_final(cells, predictions, lambda a, b: a == b)
    assert len(result['all_36_pairs']) == 36
    assert all(len(x) == 3 for x in result['within_arm_pairs'].values())
    assert all(x['all_cross_seed_pair_count'] == 9 and x['paired_seed_pair_count'] == 3 for x in result['between_arms'].values())
    assert result['all_nine_coverage']['oracle_gain'] == 0


def test_duplicate_destination_preserves_terminal_before_any_auth(tmp_path):
    directory = ROOT / 'runs' / 'single_prompt_duplicate_guard_test'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'terminal.json'
    target.write_bytes(b'{"synthetic":"preserve"}\n')
    before = target.read_bytes()
    with pytest.raises(SearchContractError, match='EXISTING_RUN_ROOT'):
        execute(ROOT, ROOT / 'runs/nonexistent_prep', directory)
    assert target.read_bytes() == before
    with pytest.raises(SearchContractError, match='PRIVATE_RUNS'):
        private_path(ROOT, ROOT / 'reports')


def test_search_predecessor_requires_complete_twenty_rounds(tmp_path):
    atomic_write_json(tmp_path / 'terminal.json', dict(status='EXECUTION_COMPLETE', phase='train', source_sha='x', artifacts={}))
    atomic_write_json(tmp_path / 'results.json', [])
    with pytest.raises(SearchContractError, match='RESULT_INCOMPLETE'):
        verify_search(tmp_path)


def test_receipts_bind_each_cell_attempt_and_unknown_candidate_failure_is_fatal(tmp_path, monkeypatch):
    broker = broker_for(tmp_path)
    try:
        broker.attempt_id = 'different_cell_attempt1'
        broker.evaluate(SEED, toy_rows(1), 'optimize', 88, 0)
        row = read_json(next((tmp_path / 'receipts').glob('*.json')))
        assert row['attempt_id'] == 'different_cell_attempt1'
        assert not broker.ledger.inflight
        import multi_dataset_diverse_rl.research.single_prompt as engine
        def integrity_failure(*args, **kwargs):
            raise SearchContractError('SYNTHETIC_INTEGRITY_FAILURE')
        monkeypatch.setattr(engine, 'candidate_failed_checks', integrity_failure)
        with pytest.raises(SearchContractError, match='INTEGRITY_FAILURE'):
            generate_candidate(SEED, dict(errors=[], structured_feedback=None), broker, 1)
        assert broker.usage['reflection'] == 1
    finally:
        broker.ledger.close()


def test_false_authorization_blocks_execution_before_client_construction(tmp_path, monkeypatch):
    import multi_dataset_diverse_rl.research.single_prompt_governance as gov
    from multi_dataset_diverse_rl.benchmarks.data_freeze import digest, file_hash
    root = tmp_path / 'root'
    manifest_path = root / gov.MANIFEST
    manifest_path.parent.mkdir(parents=True)
    manifest = read_json(ROOT / gov.MANIFEST)
    atomic_write_json(manifest_path, manifest)
    monkeypatch.setattr(gov, 'validate_manifest', lambda *_: None)
    monkeypatch.setattr(gov, 'require_selected_runtime', lambda _: None)
    monkeypatch.setattr(gov, 'source_snapshot', lambda _: {})
    monkeypatch.setattr(gov.subprocess, 'check_output', lambda args, **kwargs: 'a' * 40 if args[1] == 'rev-parse' else '')
    scope = dict(study_id=gov.STUDY, phase='train', source_sha='a' * 40, source_hashes={},
        runtime_identity=gov.runtime_identity(),
        policy=policy(), manifest_sha256=file_hash(manifest_path), token_ceiling=None,
        call_ceiling=call_ceiling('train'), search_predecessor=None,
        cells=[dict(c, attempt_id=c['cell_id'] + '_train_attempt1') for c in manifest['cells']])
    scope['execution'] = dict(prep_path='runs/prep', run_root='runs/train_run',
        command=[policy()['runtime_interpreter'], 'scripts/run_experiment.py', '--prep', 'runs/prep', '--run-root', 'runs/train_run', '--execute'])
    scope['scope_identity'] = digest(scope)
    prep = root / 'runs' / 'prep'
    atomic_write_json(prep / 'prep.json', dict(schema_version=gov.PREP_SCHEMA, scope=scope))
    atomic_write_json(prep / 'authorization.json', dict(scope_identity=scope['scope_identity'], approved=False,
        user_authorization_evidence_sha256=None, authorized_attempt_ids=[c['attempt_id'] for c in scope['cells']]))
    with pytest.raises(SearchContractError, match='EXACT_USER_AUTHORIZATION'):
        execute(root, prep, root / 'runs' / 'unapproved')
    assert not (root / 'runs' / 'unapproved').exists()
    assert not (root / 'runs' / 'single_prompt_consumed').exists()


def test_legal_unchanged_candidate_stops_after_one_draw(tmp_path):
    class Unchanged(ToyTransport):
        def for_role(self, role, request):
            self.calls.append((role, request))
            return dict(text=json.dumps(dict(block='strategy', content=SEED.strategy)), finish_reason='stop', input_tokens=1, output_tokens=1,
                        provider_reasoning_content_present=False,
                        provider_usage_details=dict(completion_tokens_details=dict(reasoning_tokens=0)))
    broker = broker_for(tmp_path, transport=Unchanged())
    try:
        child, draws = generate_candidate(SEED, dict(errors=[], structured_feedback=None), broker, 1)
        assert child == SEED and len(draws) == 1 and draws[0]['errors'] == []
    finally:
        broker.ledger.close()


def test_governed_phase_execution_and_ledger_closure_with_frozen_fake_scope(tmp_path, monkeypatch):
    """Exact execution function, synthetic authority/data/transport in temp repo."""
    import shutil
    import multi_dataset_diverse_rl.research.single_prompt_governance as gov
    import multi_dataset_diverse_rl.governance.autonomous_math as transport_module
    from multi_dataset_diverse_rl.benchmarks.data_freeze import digest, file_hash
    root = tmp_path / 'synthetic_repo'
    manifest = read_json(ROOT / gov.MANIFEST)
    for relative in [gov.MANIFEST, manifest['initial_prompt_path'], *[c['manifest_path'] for c in manifest['cells']]]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    class Reader:
        def rows(self, *args):
            return toy_rows(2)
    class Clients:
        closed = False
        def close(self):
            self.closed = True
    clients = Clients()
    monkeypatch.setattr(gov, 'validate_manifest', lambda *_: Reader())
    monkeypatch.setattr(gov, 'require_selected_runtime', lambda _: None)
    monkeypatch.setattr(gov, 'source_snapshot', lambda _: {})
    monkeypatch.setattr(gov.subprocess, 'check_output', lambda args, **kwargs: 'a' * 40 if args[1] == 'rev-parse' else '')
    monkeypatch.setattr(transport_module, 'create_transport', lambda _: (ToyTransport(), clients))
    scope = dict(study_id=gov.STUDY, phase='train', source_sha='a' * 40, source_hashes={},
        runtime_identity=gov.runtime_identity(), policy=policy(), manifest_sha256=file_hash(root / gov.MANIFEST),
        token_ceiling=None, call_ceiling=call_ceiling('train'), search_predecessor=None,
        cells=[dict(c, attempt_id=c['cell_id'] + '_train_attempt1') for c in manifest['cells']],
        execution=dict(prep_path='runs/train_prep', run_root='runs/train_run',
            command=[policy()['runtime_interpreter'], 'scripts/run_experiment.py', '--prep', 'runs/train_prep', '--run-root', 'runs/train_run', '--execute']))
    scope['scope_identity'] = digest(scope)
    prep = root / 'runs/train_prep'
    atomic_write_json(prep / 'prep.json', dict(schema_version=gov.PREP_SCHEMA, scope=scope))
    atomic_write_json(prep / 'authorization.json', dict(scope_identity=scope['scope_identity'], approved=True,
        user_authorization_evidence_sha256='b' * 64, authorized_attempt_ids=[c['attempt_id'] for c in scope['cells']]))
    result = execute(root, prep, root / 'runs/train_run')
    assert result['cells_completed'] == 9 and clients.closed
    results, _ = verify_search(root / 'runs/train_run', implementation_sha='a' * 40)
    assert all(r['rounds_completed'] == 20 and len(r['prompt_history']) == 21 for r in results)
    before = read_json(root / 'runs/train_run/terminal.json')
    with pytest.raises(SearchContractError, match='EXISTING_RUN_ROOT'):
        execute(root, prep, root / 'runs/train_run')
    assert read_json(root / 'runs/train_run/terminal.json') == before


def test_solver_invalid_recovery_capacity_and_first_valid_wrong_stops(tmp_path):
    class TruncatedThenWrong(ToyTransport):
        def for_role(self, role, request):
            self.calls.append((role, request))
            return dict(text='Final answer: 10', finish_reason='length' if len(self.calls) == 1 else 'stop', input_tokens=1, output_tokens=1)
    broker = broker_for(tmp_path / 'length', transport=TruncatedThenWrong())
    try:
        result = broker.evaluate(SEED, toy_rows(1), 'optimize', 88, 0)['toy_0']
        assert result['valid'] and not result['correct']
        assert [r['max_tokens'] for _, r in broker.transport.calls] == [3600, 6144]
        assert result['prediction']['semantic_attempt_count'] == 2
    finally:
        broker.ledger.close()
    class Invalid(ToyTransport):
        def for_role(self, role, request):
            self.calls.append((role, request))
            return dict(text='Missing final declaration', finish_reason='stop', input_tokens=1, output_tokens=1)
    broker = broker_for(tmp_path / 'invalid', transport=Invalid())
    try:
        result = broker.evaluate(SEED, toy_rows(1), 'optimize', 88, 0)['toy_0']
        assert not result['valid'] and not result['correct']
        assert len(broker.transport.calls) == 4
        assert all(r['max_tokens'] == 3600 for _, r in broker.transport.calls)
    finally:
        broker.ledger.close()


def test_selected_sdk_shared_transport_exact_wire_and_response_with_mock_http(monkeypatch):
    """Real installed SDK/helper, synthetic HTTP only; no credentials/network."""
    import hashlib
    import httpx
    import openai
    from multi_dataset_diverse_rl.governance.autonomous_math import create_transport
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.provider_routing import routing_contract, IDENTITY
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    calls = []
    def handler(request):
        calls.append(dict(body=request.content, parsed=json.loads(request.content)))
        assert request.headers['authorization'] == 'Bearer offline-fixture'
        return httpx.Response(200, json=dict(id='SYNTHETIC_HTTP',
            choices=[dict(message=dict(content='SYNTHETIC_RESPONSE'), finish_reason='stop')],
            usage=dict(prompt_tokens=12, completion_tokens=8,
                       completion_tokens_details=dict(reasoning_tokens=0))))
    def fake_factory(**kwargs):
        assert kwargs['max_retries'] == 0
        return openai.OpenAI(api_key='offline-fixture', base_url='https://offline.invalid',
            max_retries=0, timeout=kwargs['timeout'], http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(ProviderClientFactory, 'from_environment', fake_factory)
    contract = dict(provider=IDENTITY, provider_routing_policy=routing_contract(),
        models=dict(solver='gpt-4o-mini', optimizer_reflection='qwen3.7-flash', pattern='qwen3.7-flash', solver_thinking=False),
        decoding=dict(timeout_seconds=120), runtime_persistence_policy=True,
        optimizer_generation_policy=policy()['optimizer_generation_policy'])
    transport, clients = create_transport(contract)
    try:
        for role in ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'):
            request = dict(model='gpt-4o-mini' if role == 'solver' else 'qwen3.7-flash',
                           messages=[dict(role='user', content='SYNTHETIC_INPUT')])
            if role == 'solver':
                request.update(max_tokens=3600, temperature=.2, top_p=.8)
            else:
                request.update(max_completion_tokens=8192 if role == 'pattern_cluster' else 1800,
                    temperature=.7, top_p=.8, presence_penalty=1.5, frequency_penalty=0,
                    extra_body=dict(enable_thinking=False, top_k=20))
            result = transport.for_role(role, request)
            assert calls[-1]['body'] == serialized_request(request)
            assert result['text'] == 'SYNTHETIC_RESPONSE' and result['finish_reason'] == 'stop'
            assert result['input_tokens'] == 12 and result['output_tokens'] == 8
            assert result['provider_metadata_loss_audited'] and result['provider_response_accepted']
            assert not result['provider_thinking_indicators']
        assert len(calls) == 4
    finally:
        clients.close()


def test_selected_runtime_requires_exact_environment_interpreter_and_sdk(monkeypatch):
    import multi_dataset_diverse_rl.research.single_prompt_governance as gov
    monkeypatch.setattr(gov.sys, 'prefix', str(ROOT / policy()['runtime_environment']))
    monkeypatch.setattr(gov.sys, 'executable', str(ROOT / policy()['runtime_interpreter']))
    monkeypatch.setattr(gov.importlib.metadata, 'version', lambda _: '2.37.0')
    gov.require_selected_runtime(ROOT)
    for field in ('prefix', 'executable'):
        with monkeypatch.context() as changed:
            changed.setattr(gov.sys, field, str(ROOT / 'other_environment'))
            with pytest.raises(SearchContractError, match='SELECTED_REPOSITORY_RUNTIME'):
                gov.require_selected_runtime(ROOT)
    monkeypatch.setattr(gov.importlib.metadata, 'version', lambda _: '2.3.0')
    with pytest.raises(SearchContractError, match='SELECTED_REPOSITORY_RUNTIME'):
        gov.require_selected_runtime(ROOT)
