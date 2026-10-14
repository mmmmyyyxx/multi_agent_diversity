"""Exact fresh freezes for the isolated repeat-scoped single-Prompt study.

Planning opens split metadata only. Execution consumes one authorization and
opens adaptive Validation explicitly; Test opens only after audited training.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import subprocess

from ...benchmarks.access import DataPurpose
from ...benchmarks.data_freeze import digest, file_hash
from ...benchmarks.math_flexible_answer import validity_contract
from ...benchmarks.math_prediction_validity import structured_recovery_contract
from ...governance.token_accounting import (
    TokenLedger, POLICY_GATED_32_UNCAPPED, POLICY_V27_FORMAL_UNCAPPED, audit_journal,
)
from ...persistence.durable_io import atomic_write_json, read_json
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt, block_lineage, candidate_failed_checks
from ..single_prompt_governance import (
    private_path, inventory, runtime_identity, require_selected_runtime,
    StudySplitReader, StudyHeldoutAccess,
)
from ..gated_prompt_policy import gate_membership
from ..gated_prompt_test import formal_test_plan
from .config import build_plan, default_config, validate_plan, request_namespace
from .engine import run_cell
from .analysis import summarize
from .runtime import AblationBroker
from ...local_optimizers.schemas import LocalEvidenceExample

MANIFEST = 'experiments/manifests/single_prompt_pattern_repeat_solver_cache_v2.json'
PROTOCOL = 'docs/design/SINGLE_PROMPT_PATTERN_REPEAT_SOLVER_CACHE_V2.md'
MANIFEST_SCHEMA = 'single_prompt_ablation_execution_manifest_v2'
PREP_SCHEMA = 'single_prompt_ablation_execution_prep_v2'
DATA_PARENT = 'experiments/manifests/single_prompt_gradient_pattern_v1.json'
INTERPRETER = 'runs/math_v2_1/runtime/Scripts/python.exe'


def manifest_template(root):
    parent = read_json(root / DATA_PARENT)
    return dict(schema_version=MANIFEST_SCHEMA,
        experiment_id=default_config()['experiment_id'],
        parents=['experiments/manifests/single_prompt_largest_pattern_v1.json',
                 'experiments/manifests/modular_mechanisms_v212_preparation_v1.yaml'],
        protocol_path=PROTOCOL, config=default_config(), plan=build_plan(default_config()),
        data=parent['data'], initial_prompt_path=parent['initial_prompt_path'],
        initial_prompt_sha256=parent['initial_prompt_sha256'],
        authorization=dict(real_api_authorized=False, adaptive_validation_access=False, test_access=False),
        status='PREPARED_NOT_EXECUTED')


def validate_manifest(root, manifest):
    if manifest != manifest_template(root):
        raise SearchContractError('ABLATION_EXECUTION_MANIFEST_MISMATCH')
    data = manifest['data']
    artifact = root / manifest['initial_prompt_path']
    if file_hash(artifact) != manifest['initial_prompt_sha256']:
        raise SearchContractError('ABLATION_INITIAL_PROMPT_HASH')
    members = read_json(artifact)['members']
    if not members or len({digest(m['prompt']) for m in members}) != 1:
        raise SearchContractError('ABLATION_IDENTICAL_INITIAL_PROMPT_REQUIRED')
    SystemPrompt.from_dict(members[0]['prompt'])
    reader = StudySplitReader(root / data['canonical_root'], root / data['split_directory'], 'math',
        expected_manifest_sha256=data['split_manifest_sha256'], expected_protocol='benchmark_experiment_split_v2')
    if (reader.manifest['counts'] != data['counts']
            or file_hash(root / data['canonical_root'] / 'manifests/math.json') != data['canonical_manifest_sha256']
            or file_hash(root / data['split_directory'] / 'math.ids.jsonl') != data['membership_sha256']):
        raise SearchContractError('ABLATION_DATA_IDENTITY')
    return reader


def test_plan(root, reader, plan):
    formal = formal_test_plan(root, reader)
    return dict(formal, identity='ABLATION_REPEAT_SCOPED_FINAL_TEST300_V2',
        selection='all_fifteen_last_validation_retained_prompts_after_audited_training',
        physical_call_ceiling=None,
        successful_semantic_call_ceiling=plan['final_test_semantic_call_ceiling'],
        cache=plan['cache_policy'], non_solver_cache='forbidden')


def source_snapshot(root):
    paths = set((root / 'multi_dataset_diverse_rl').rglob('*.py'))
    paths.update((root / 'docs/design').glob('*.md'))
    paths.update((root / 'tests/current').glob('*.py'))
    paths.update((root / 'tests/formal_zero_network').rglob('*.py'))
    paths.update(root / p for p in (
        'AGENTS.md', MANIFEST, DATA_PARENT, PROTOCOL,
        'docs/workflows/EXPERIMENT_HANDOFF.md', 'docs/failures/registry.yaml',
        'docs/REPOSITORY_MAP.md', 'experiments/current_frontier.yaml',
        'experiments/registry.yaml', 'experiments/lineage.yaml',
        'experiments/ablations/pattern/default.json', 'scripts/run_experiment.py',
        'scripts/ablations/study.py', 'scripts/ablations/prepare.py',
        'scripts/tooling_classification.json', 'tests/suite_classification.json',
        'tests/conftest.py', 'tests/formal_zero_api_runner.py',
        'requirements-benchmark-evaluators.txt',
        'experiments/manifests/math_v27_direct_seed86_formal_v1.yaml'))
    manifest = read_json(root / MANIFEST)
    paths.update(root / p for p in manifest['parents'])
    paths.add(root / manifest['initial_prompt_path'])
    paths.update(p for p in (root / manifest['data']['split_directory']).iterdir() if p.is_file())
    paths.add(root / manifest['data']['canonical_root'] / 'manifests/math.json')
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}


def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def _scope(root, destination):
    manifest = read_json(root / MANIFEST)
    reader = validate_manifest(root, manifest)
    plan = manifest['plan']
    prep = destination.relative_to(root).as_posix()
    run = (destination.parent / 'train_run').relative_to(root).as_posix()
    value = dict(study_id=manifest['experiment_id'], phase='train',
        source_sha=_git(root, 'rev-parse', 'HEAD'), source_hashes=source_snapshot(root),
        runtime_identity=runtime_identity(), manifest_sha256=file_hash(root / MANIFEST),
        plan=plan, gate_membership=gate_membership(reader), test_plan=test_plan(root, reader, plan),
        optimize_sample_ids=[m['stable_example_id'] for m in reader.members if m['project_split'] == 'optimize'],
        initial_prompt_sha256=SystemPrompt.from_dict(read_json(root / manifest['initial_prompt_path'])['members'][0]['prompt']).prompt_hash,
        call_ceiling=plan['training_semantic_call_ceiling'], token_ceiling=None,
        network_recovery=deepcopy(plan['config']['runtime']['timeout_recovery']),
        execution=dict(prep_path=prep, run_root=run,
            command=[INTERPRETER, 'scripts/run_experiment.py', '--prep', prep, '--run-root', run, '--execute']))
    return dict(value, scope_identity=digest(value))


def authorization_template(scope):
    return dict(scope_identity=scope['scope_identity'], approved=False,
        adaptive_validation_access=False, test_access=False,
        user_authorization_evidence_sha256=None,
        authorized_attempt_ids=[scope['scope_identity'] + '_train_' + c['cell_id'] for c in scope['plan']['cells']])


def prepare(root, destination):
    destination = private_path(root, destination)
    if destination.exists():
        raise SearchContractError('ABLATION_FRESH_PREPARATION_REQUIRED')
    require_selected_runtime(root)
    if _git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise SearchContractError('ABLATION_CLEAN_COMMITTED_SOURCE_REQUIRED')
    scope = _scope(root, destination)
    if set(scope['source_hashes']) - set(_git(root, 'ls-files').splitlines()):
        raise SearchContractError('ABLATION_UNCOMMITTED_FREEZE_FILE')
    destination.mkdir(parents=True)
    atomic_write_json(destination / 'prep.json', dict(schema_version=PREP_SCHEMA, scope=scope))
    atomic_write_json(destination / 'authorization.json', authorization_template(scope))
    atomic_write_json(destination / 'execution_plan.json', execution_plan_template(scope))
    atomic_write_json(destination / 'handoff.json', handoff_template(root, destination, scope))
    return dict(status='FROZEN_NOT_AUTHORIZED', source_sha=scope['source_sha'],
        scope_identity=scope['scope_identity'], cells=len(scope['plan']['cells']), real_api_calls=0)


def execution_plan_template(scope):
    return dict(
        command=scope['execution']['command'], cells=scope['plan']['cells'],
        train_call_ceiling=scope['call_ceiling'], test_plan=scope['test_plan'], token_ceiling=None,
        source_sha=scope['source_sha'], scope_identity=scope['scope_identity'])


def handoff_template(root, destination, scope):
    return dict(
        schema_version='sol_luna_experiment_handoff_v1', experiment_id=scope['study_id'],
        source=dict(repository_commit=scope['source_sha'], tracked_worktree_status='clean'),
        protocol=dict(name='Single Prompt Pattern repeat-scoped Solver cache', version='V2',
            protocol_sha256=file_hash(root / PROTOCOL), manifest_path=MANIFEST,
            manifest_sha256=scope['manifest_sha256'], preregistration_path=PROTOCOL,
            preregistration_sha256=file_hash(root / PROTOCOL)),
        data=dict(dataset='math', split_identity=read_json(root / MANIFEST)['data']['split_manifest_sha256'],
            split_sha256=read_json(root / MANIFEST)['data']['membership_sha256'],
            adaptive_validation_membership_sha256=scope['gate_membership']['membership_sha256']),
        models=dict(solver='gpt-4o-mini', optimizer_or_reflection='qwen3.7-flash', thinking=False,
            local_optimizer_backend='single_prompt_one_block', local_optimizer_version_or_sha256=scope['plan']['plan_identity']),
        execution=dict(seeds=[42], repeat_ids=[1, 2, 3, 4, 5], opportunity_budget=300,
            early_stop_rule='none_efficacy_independent_stop_after_twenty_per_cell',
            exact_runner_command=scope['execution']['command'], expected_output_directory=scope['execution']['run_root'],
            expected_checkpoint_path=scope['execution']['run_root'] + '/' + scope['plan']['cells'][0]['cell_id'] + '/checkpoint_0.json',
            expected_checkpoint_paths=[scope['execution']['run_root'] + '/' + c['cell_id'] + f'/checkpoint_{n}.json'
                for c in scope['plan']['cells'] for n in range(scope['plan']['config']['rounds'] + 1)],
            expected_ledger_path=scope['execution']['run_root'] + '/ledger',
            allowed_retries=scope['network_recovery'], train_solver_concurrency=32, gradient_concurrency=8,
            test_solver_concurrency=16, train_successful_semantic_call_ceiling=scope['call_ceiling'],
            test_successful_semantic_call_ceiling=scope['test_plan']['successful_semantic_call_ceiling'],
            cumulative_token_ceiling=None),
        authorization=dict(api_scope=scope['scope_identity'], validation_access_policy='exact_adaptive_validation100',
            test_access_policy='conditional_test300_after_all_fifteen_audited_cells'),
        fail_closed_conditions=['identity_or_hash_mismatch', 'tracked_source_mutation',
            'command_seed_model_split_budget_mismatch', 'non_network_provider_failure',
            'checkpoint_ledger_receipt_corruption', 'budget_overrun', 'unexpected_data_access', 'protocol_change'],
        READY_TO_RUN=False, status='FROZEN_AWAITING_EXACT_USER_AUTHORIZATION',
        source_sha=scope['source_sha'], scope_identity=scope['scope_identity'],
        prep_sha256=file_hash(destination / 'prep.json'),
        command=scope['execution']['command'], runtime_identity=scope['runtime_identity'],
        source_hashes_sha256=digest(scope['source_hashes']), real_api_calls=0,
        resume='no_legacy_state_no_process_restart_network_recovery_inside_frozen_process',
        authorization_path=(destination / 'authorization.json').relative_to(root).as_posix())


def validate_authorization(scope, authorization, *, required=False, test=False):
    expected = authorization_template(scope)
    if (set(authorization) != set(expected)
            or any(authorization[k] != expected[k] for k in ('scope_identity', 'authorized_attempt_ids'))
            or any(type(authorization[k]) is not bool for k in ('approved', 'adaptive_validation_access', 'test_access'))):
        raise SearchContractError('ABLATION_EXACT_AUTHORIZATION_SCOPE_REQUIRED')
    if required and (not authorization['approved'] or not authorization['adaptive_validation_access']
            or re.fullmatch('[0-9a-f]{64}', str(authorization['user_authorization_evidence_sha256'])) is None
            or (test and not authorization['test_access'])):
        raise SearchContractError('ABLATION_FRESH_USER_PHASE_AUTHORIZATION_REQUIRED')


def validate_prep(root, directory, *, authorized=False):
    directory = private_path(root, directory)
    prep = read_json(directory / 'prep.json')
    if set(prep) != {'schema_version', 'scope'} or prep['schema_version'] != PREP_SCHEMA:
        raise SearchContractError('ABLATION_PREP_SCHEMA')
    require_selected_runtime(root)
    if (_git(root, 'status', '--porcelain', '--untracked-files=no') or prep['scope'] != _scope(root, directory)):
        raise SearchContractError('ABLATION_FROZEN_SOURCE_RUNTIME_INPUT_CHANGED')
    validate_authorization(prep['scope'], read_json(directory / 'authorization.json'), required=authorized)
    if read_json(directory / 'execution_plan.json') != execution_plan_template(prep['scope']):
        raise SearchContractError('ABLATION_EXECUTION_PLAN_MISMATCH')
    handoff = read_json(directory / 'handoff.json')
    expected_handoff = handoff_template(root, directory, prep['scope'])
    if any(handoff.get(k) != v for k, v in expected_handoff.items() if k not in ('READY_TO_RUN', 'status')) or set(handoff) != set(expected_handoff):
        raise SearchContractError('ABLATION_HANDOFF_MISMATCH')
    if authorized and handoff.get('READY_TO_RUN') is not True:
        raise SearchContractError('ABLATION_AUTHORIZED_HANDOFF_REQUIRED')
    return prep['scope']


@dataclass(frozen=True)
class GateAccess:
    scope_identity: str
    membership: dict
    require_authorization: object


def read_gate_rows(reader, access):
    if (type(access) is not GateAccess or re.fullmatch('[0-9a-f]{64}', access.scope_identity) is None
            or access.membership != gate_membership(reader)):
        raise SearchContractError('ABLATION_ADAPTIVE_VALIDATION_CAPABILITY_REQUIRED')
    access.require_authorization()
    return reader._read_rows('validation', frozen_subset_ids=access.membership['sample_ids'])


def audit_cells(directory, scope):
    """Recheck each persisted gate transition before authorizing Test access."""
    plan = scope['plan']
    validate_plan(plan)
    results = read_json(directory / 'results.json')
    if [r['cell'] for r in results] != plan['cells']:
        raise SearchContractError('ABLATION_COMPLETE_ORDERED_CELLS_REQUIRED')
    for cell, result in zip(plan['cells'], results):
        folder = directory / cell['cell_id']
        cps = [read_json(folder / f'checkpoint_{n}.json') for n in range(plan['config']['rounds'] + 1)]
        first = cps[0]
        if (result != read_json(folder / 'result.json')
                or result['rounds_completed'] != plan['config']['rounds']
                or result['prompt_history'] != [cp['prompt'] for cp in cps]
                or result['metrics'] != [cp['metric'] for cp in cps[1:]]
                or result['final_prompt'] != cps[-1]['prompt']
                or first['scope']['cell'] != cell
                or result['module_binding'] != plan['module_bindings'][cell['arm_id']]
                or first['scope']['module_binding'] != result['module_binding']
                or first['scope']['plan_identity'] != plan['plan_identity']
                or first['scope']['initial_prompt_sha256'] != scope['initial_prompt_sha256']
                or first['scope']['optimize_membership_sha256'] != digest(scope['optimize_sample_ids'])
                or first['scope']['validation_membership_sha256'] != scope['gate_membership']['membership_sha256']
                or first['scope']['repeat_namespace'] != request_namespace(plan['plan_identity'], cell['seed'], cell['repeat_id'])):
            raise SearchContractError('ABLATION_CHECKPOINT_RESULT_SCOPE_MISMATCH')
        for n, cp in enumerate(cps):
            if (cp['integrity_sha256'] != digest({k: v for k, v in cp.items() if k != 'integrity_sha256'})
                    or cp['scope_identity'] != digest(first['scope']) or cp['scope_identity'] != result['scope_identity']
                    or len(cp['optimize']) != 150 or len(cp['validation']) != 100
                    or set(cp['optimize']) != set(scope['optimize_sample_ids'])
                    or set(cp['validation']) != set(scope['gate_membership']['sample_ids'])
                    or any(type(v.get('correct')) is not bool or type(v.get('valid')) is not bool
                           or (v['correct'] and not v['valid']) for values in (cp['optimize'], cp['validation']) for v in values.values())):
                raise SearchContractError('ABLATION_CHECKPOINT_INTEGRITY')
            if not n:
                continue
            previous, metric = cps[n - 1], cp['metric']
            opportunity = read_json(folder / f'round_{n}_opportunity.json')
            module = read_json(folder / f'round_{n}_module.json')
            if (opportunity['parent_prompt'] != previous['prompt']
                    or opportunity['retained_validation_correct'] != sum(v['correct'] for v in previous['validation'].values())
                    or module['module_binding'] != plan['module_bindings'][cell['arm_id']]):
                raise SearchContractError('ABLATION_OPPORTUNITY_PROVENANCE')
            if cell['module'] == 'largest_pattern':
                from .pattern import project
                projected, focus = project(module['audit']['complete_diagnosis'], cell['seed'], n)
                if (projected['errors'] != opportunity['errors']
                        or projected['structured_feedback'] != opportunity['structured_feedback']
                        or focus != module['audit']['focus'] or opportunity['module_context'] is not None):
                    raise SearchContractError('ABLATION_PATTERN_PROJECTION_MISMATCH')
            before = sum(v['correct'] for v in previous['validation'].values())
            candidate = cp['candidate_prompt']
            decision = metric['decision']
            if metric['round'] != n or metric['validation_before'] != before or cp['parent_prompt'] != previous['prompt']:
                raise SearchContractError('ABLATION_GATE_PARENT_MISMATCH')
            if candidate is None:
                if (decision != 'ABANDONED_GENERATION_RETRIES_EXHAUSTED' or metric['accepted']
                        or metric['generation_draws'] != 4 or metric['candidate_evaluated']
                        or cp['candidate_validation'] is not None or cp['candidate_optimize'] is not None):
                    raise SearchContractError('ABLATION_EXHAUSTION_MISMATCH')
            else:
                prompt = SystemPrompt.from_dict(candidate)
                lineage = block_lineage(SystemPrompt.from_dict(previous['prompt']), prompt)
                if (len(prompt.render()) > 3000 or lineage != metric['lineage']
                        or len(lineage['edited_blocks']) != 1
                        or candidate_failed_checks(prompt, parent_prompt=SystemPrompt.from_dict(previous['prompt']),
                            examples=tuple(LocalEvidenceExample(e['example_id'], e['problem'], e['reference'])
                                           for e in opportunity['errors']))
                        or re.search(r'\bmodule_context\b', prompt.render(), re.I)
                        or not 1 <= metric['generation_draws'] <= 4 or not metric['candidate_evaluated']
                        or len(cp['candidate_validation']) != 100 or len(cp['candidate_optimize']) != 150
                        or set(cp['candidate_validation']) != set(previous['validation'])
                        or set(cp['candidate_optimize']) != set(previous['optimize'])):
                    raise SearchContractError('ABLATION_CANDIDATE_CONFORMANCE')
                score = sum(v['correct'] for v in cp['candidate_validation'].values())
                accepted = score > before
                if (metric['validation_candidate'] != score or metric['accepted'] is not accepted
                        or decision != ('ACCEPTED_STRICT_VALIDATION_GAIN' if accepted else 'REJECTED_NO_STRICT_VALIDATION_GAIN')):
                    raise SearchContractError('ABLATION_STRICT_GATE_MISMATCH')
                if accepted and any(cp[k] != cp['candidate_' + k] for k in ('prompt', 'optimize', 'validation')):
                    raise SearchContractError('ABLATION_ACCEPTED_PROFILE_MISMATCH')
            if not metric['accepted'] and any(cp[k] != previous[k] for k in ('prompt', 'optimize', 'validation')):
                raise SearchContractError('ABLATION_REJECTED_PARENT_CHANGED')
        if (result['final_validation_correct'] != sum(v['correct'] for v in cps[-1]['validation'].values())
                or result['final_optimize_correct'] != sum(v['correct'] for v in cps[-1]['optimize'].values())):
            raise SearchContractError('ABLATION_FINAL_SCORE_MISMATCH')
    return results


def audit_accounting(directory, scope, phase):
    policy = POLICY_GATED_32_UNCAPPED if phase == 'train' else POLICY_V27_FORMAL_UNCAPPED
    view, events = audit_journal(directory / 'ledger', task_sha256=scope['scope_identity'], policy=policy)
    summary = read_json(directory / 'execution_summary.json')
    usage = summary['usage']
    reserves = {e['reservation_id']: e for e in events if e['kind'] == 'RESERVE'}
    charges = [e for e in events if e['kind'] == 'CHARGE']
    if (view != summary['token_ledger'] or len(reserves) != usage['provider_calls']
            or len(charges) != len(reserves) or view['charged_total'] != usage['input_tokens'] + usage['output_tokens']
            or usage['provider_calls'] - usage['network_failed_calls'] > scope['call_ceiling']):
        raise SearchContractError('ABLATION_ACCOUNTING_MISMATCH')
    network_failed = 0
    expected_attempts = {scope['scope_identity'] + '_' + phase + '_' + c['cell_id'] for c in scope['plan']['cells']}
    if any(r['attempt_id'] not in expected_attempts
           or r['role'] not in (('solver',) if phase == 'test' else ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'))
           for r in reserves.values()):
        raise SearchContractError('ABLATION_LEDGER_ATTEMPT_ROLE_SCOPE')
    for e in charges:
        receipt = e.get('response_receipt')
        if (receipt is None or receipt['reservation_id'] != e['reservation_id']
                or receipt['attempt_id'] != reserves[e['reservation_id']]['attempt_id']):
            raise SearchContractError('ABLATION_RECEIPT_REQUIRED')
        store = ProviderResponseReceipts(directory / 'provider_receipts', attempt_id=receipt['attempt_id'],
            startup_identity_sha256=scope['scope_identity'])
        record = store.read(receipt['reservation_id'])
        if read_json(store.directory / (receipt['reservation_id'] + '.json'))['integrity_sha256'] != receipt['integrity_sha256']:
            raise SearchContractError('ABLATION_RECEIPT_SEAL_CHANGED')
        network_failed += record.get('error_category') in ('APITimeoutError', 'APIConnectionError')
    if network_failed != usage['network_failed_calls'] or usage.get('replayed_calls', 0) != 0:
        raise SearchContractError('ABLATION_NETWORK_OR_REPLAY_ACCOUNTING')
    for role in ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'):
        values = view['by_role'].get(role, {})
        if (usage[role] != values.get('physical_attempts', 0)
                or any(usage[role + '_' + k] != values.get(k, 0) for k in ('input_tokens', 'output_tokens'))):
            raise SearchContractError('ABLATION_ROLE_COST_MISMATCH')
    for result in read_json(directory / 'results.json'):
        ident = scope['scope_identity'] + '_' + phase + '_' + result['cell']['cell_id']
        values = view['by_attempt'].get(ident, {})
        if (result['usage']['provider_calls'] != values.get('physical_attempts', 0)
                or any(result['usage'][k] != values.get(k, 0) for k in ('input_tokens', 'output_tokens'))):
            raise SearchContractError('ABLATION_CELL_COST_MISMATCH')
    return dict(physical_calls=len(reserves), charged_tokens=view['charged_total'], network_failed_calls=network_failed)


def verify_training(directory, scope):
    terminal = read_json(directory / 'terminal.json')
    actual = inventory(directory)
    actual.pop('terminal.json', None)
    if (terminal != dict(status='EXECUTION_COMPLETE', phase='train', source_sha=scope['source_sha'],
            scope_identity=scope['scope_identity'], artifacts=actual)
            or read_json(directory / 'access.json')['test_access'] is not False):
        raise SearchContractError('ABLATION_SEALED_TRAINING_REQUIRED')
    results = audit_cells(directory, scope)
    audit_accounting(directory, scope, 'train')
    return results, file_hash(directory / 'terminal.json')


def _consume(root, scope, directory):
    marker = root / 'runs/ablations/consumed' / (scope['scope_identity'] + '.json')
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(dict(scope_identity=scope['scope_identity'], run_root=directory.relative_to(root).as_posix()), stream)
        stream.flush()
        os.fsync(stream.fileno())


def _transport():
    # This function is reached only after authorization, handoff and single-use checks.
    from ...governance.autonomous_math import create_transport
    from ...provider_routing import routing_contract, IDENTITY
    from ...benchmarks.math_optimizer_generation import optimizer_generation_contract
    return create_transport(dict(provider=IDENTITY, provider_routing_policy=routing_contract(),
        models=dict(solver='gpt-4o-mini', optimizer_reflection='qwen3.7-flash', pattern='qwen3.7-flash', solver_thinking=False),
        decoding=dict(timeout_seconds=120), runtime_persistence_policy=True,
        optimizer_generation_policy=optimizer_generation_contract()))


def _execute_phase(root, prep, directory, scope, *, train_scope, parent_results=None, terminal_sha=None):
    phase = scope['phase']
    plan = train_scope['plan']
    if directory.exists():
        raise SearchContractError('ABLATION_EXISTING_RUN_FORBIDDEN')
    validate_prep(root, prep, authorized=True)
    validate_authorization(train_scope, read_json(prep / 'authorization.json'), required=True, test=phase == 'test')
    if phase == 'test':
        checked_results, checked_terminal = verify_training(directory.parent / 'train_run', train_scope)
        if checked_results != parent_results or checked_terminal != terminal_sha or scope != final_test_scope(train_scope, terminal_sha):
            raise SearchContractError('ABLATION_EXACT_FINAL_TEST_SCOPE_REQUIRED')
    _consume(root, scope, directory)
    directory.mkdir(parents=True)
    atomic_write_json(directory / 'scope.json', scope)
    atomic_write_json(directory / 'authorization.json', read_json(prep / 'authorization.json'))
    atomic_write_json(directory / 'lifecycle.json', dict(status='RUNNING', phase=phase, cells_completed=0))
    signatures = {p: (root / p).stat() for p in train_scope['source_hashes']}
    pins = {name: file_hash(prep / name) for name in ('authorization.json', 'prep.json', 'handoff.json', 'execution_plan.json')}
    active = None
    ledger = clients = broker = None
    completed = []
    predictions = {}
    def guard(cell):
        if (cell != active or any(file_hash(prep / name) != sha for name, sha in pins.items())
                or (terminal_sha is not None and file_hash(directory.parent / 'train_run/terminal.json') != terminal_sha)):
            raise SearchContractError('ABLATION_ACTIVE_SCOPE_MUTATED')
        for p, old in signatures.items():
            new = (root / p).stat()
            if (new.st_size, new.st_mtime_ns) != (old.st_size, old.st_mtime_ns):
                raise SearchContractError('ABLATION_SOURCE_MUTATED')
    try:
        transport, clients = _transport()
        ledger = TokenLedger(directory / 'ledger', task_sha256=scope['scope_identity'],
            policy=POLICY_GATED_32_UNCAPPED if phase == 'train' else POLICY_V27_FORMAL_UNCAPPED)
        broker = AblationBroker(plan=plan, phase=phase, scope=scope, transport=transport, ledger=ledger,
            receipts=ProviderResponseReceipts(directory / 'provider_receipts', attempt_id=scope['scope_identity'],
                startup_identity_sha256=scope['scope_identity']), dispatch_guard=guard)
        manifest = read_json(root / MANIFEST)
        reader = validate_manifest(root, manifest)
        if phase == 'train':
            optimize = reader.rows('optimize', DataPurpose.SEARCH_EVALUATION)
            validation = read_gate_rows(reader, GateAccess(scope['scope_identity'], scope['gate_membership'],
                lambda: validate_prep(root, prep, authorized=True)))
            initial = read_json(root / manifest['initial_prompt_path'])['members'][0]['prompt']
            atomic_write_json(directory / 'access.json', dict(optimize_ids=[r['stable_example_id'] for r in optimize],
                validation_gate_ids=scope['gate_membership']['sample_ids'], test_access=False))
        else:
            # All train files/ledger/receipts are checked again before opening Test rows.
            verify_training(directory.parent / 'train_run', train_scope)
            rows = reader.heldout_rows(StudyHeldoutAccess('test', reader.manifest_sha256, terminal_sha,
                lambda: validate_authorization(train_scope, read_json(prep / 'authorization.json'), required=True, test=True)))
            if [r['stable_example_id'] for r in rows] != train_scope['test_plan']['sample_ids']:
                raise SearchContractError('ABLATION_TEST_MEMBERSHIP_CHANGED')
            atomic_write_json(directory / 'access.json', dict(test_ids=train_scope['test_plan']['sample_ids'],
                optimize_access=False, validation_access=False))
        for index, cell in enumerate(plan['cells']):
            validate_prep(root, prep, authorized=True)
            active = cell['cell_id']
            atomic_write_json(directory / 'lifecycle.json', dict(status='RUNNING', phase=phase,
                current_cell=active, cells_completed=len(completed)))
            folder = directory / active
            folder.mkdir()
            def persist(name, data):
                path = folder / (name + '.json')
                if path.exists():
                    raise SearchContractError('ABLATION_ARTIFACT_OVERWRITE_FORBIDDEN')
                atomic_write_json(path, data)
            if phase == 'train':
                result = run_cell(plan, cell, initial, optimize, validation, broker, persist)
            else:
                broker.bind_cell(cell, plan)
                begin = broker.usage_view()
                values = broker.evaluate(SystemPrompt.from_dict(parent_results[index]['final_prompt']), rows, 'test', cell['seed'], 0)
                result = dict(cell=cell, predictions=values,
                    usage={k: broker.usage_view()[k] - v for k, v in begin.items()})
                persist('result', result)
                predictions[active] = values
            completed.append(result)
            print(json.dumps(dict(cell=active, phase=phase, status='CELL_COMPLETE',
                provider_calls=broker.usage_view()['provider_calls'])), flush=True)
        validate_prep(root, prep, authorized=True)
        if ledger.inflight:
            raise SearchContractError('ABLATION_UNRESOLVED_RESERVATION')
        atomic_write_json(directory / 'results.json', completed)
        atomic_write_json(directory / 'execution_summary.json', dict(usage=broker.usage_view(),
            token_ledger=ledger.view(), cells_completed=len(completed)))
        if phase == 'train':
            audit_cells(directory, scope)
            summary = summarize(plan, completed)
        else:
            verify_training(directory.parent / 'train_run', train_scope)
            if any(len(v) != 300 or set(v) != set(train_scope['test_plan']['sample_ids']) for v in predictions.values()):
                raise SearchContractError('ABLATION_FINAL_TEST_INCOMPLETE')
            summary = summarize(plan, parent_results, predictions)
        audit_accounting(directory, scope, phase)
        atomic_write_json(directory / 'summary.json', summary)
        ledger.close()
        ledger = None
        clients.close()
        clients = None
        atomic_write_json(directory / 'lifecycle.json', dict(status='EXECUTION_COMPLETE', phase=phase, cells_completed=len(completed)))
        atomic_write_json(directory / 'terminal.json', dict(status='EXECUTION_COMPLETE', phase=phase,
            source_sha=scope['source_sha'], scope_identity=scope['scope_identity'], artifacts=inventory(directory)))
        return dict(status='EXECUTION_COMPLETE', phase=phase, cells_completed=len(completed))
    except BaseException as exc:
        if ledger is not None:
            for reservation_id in tuple(ledger.inflight):
                ledger.reconcile(reservation_id, {}, outcome='ABORT_FULL_RESERVATION_FALLBACK')
            atomic_write_json(directory / 'abort.json', dict(error_type=type(exc).__name__,
                usage=None if broker is None else broker.usage_view(), token_ledger=ledger.view()))
        atomic_write_json(directory / 'lifecycle.json', dict(status='EXECUTION_ABORTED', phase=phase,
            current_cell=active, cells_completed=len(completed), error_type=type(exc).__name__))
        raise
    finally:
        if ledger is not None:
            ledger.close()
        if clients is not None:
            clients.close()


def execute(root, prep, run):
    prep, run = private_path(root, prep), private_path(root, run)
    scope = validate_prep(root, prep, authorized=True)
    if run != (root / scope['execution']['run_root']).resolve():
        raise SearchContractError('ABLATION_EXACT_RUN_ROOT_REQUIRED')
    result = _execute_phase(root, prep, run, scope, train_scope=scope)
    if read_json(prep / 'authorization.json')['test_access']:
        results, terminal_sha = verify_training(run, scope)
        test_scope = final_test_scope(scope, terminal_sha)
        return _execute_phase(root, prep, run.parent / 'test_run', test_scope,
            train_scope=scope, parent_results=results, terminal_sha=terminal_sha)
    return result


def final_test_scope(scope, terminal_sha):
    body = dict(phase='test', source_sha=scope['source_sha'], parent_scope_identity=scope['scope_identity'],
        parent_terminal_sha256=terminal_sha, plan=scope['plan'], test_plan=scope['test_plan'],
        call_ceiling=scope['test_plan']['successful_semantic_call_ceiling'], token_ceiling=None,
        network_recovery=scope['network_recovery'])
    return dict(body, scope_identity=digest(body))
