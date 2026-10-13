"""Separate phase freezes and single-use capabilities for the isolated study."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import importlib.metadata
import subprocess

from ..benchmarks.data_freeze import file_hash, digest
from ..benchmarks.experiment_splits import ExperimentSplitReader
from ..benchmarks.access import DataPurpose
from ..persistence.durable_io import atomic_write_json, read_json
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..governance.token_accounting import TokenLedger, POLICY_V27_FORMAL_UNCAPPED, audit_journal
from ..search.schemas import SearchContractError
from ..search.system_prompt import SystemPrompt
from .single_prompt_policy import policy, canary_contract, call_ceiling, STUDY, ARMS
from .single_prompt_runtime import StudyBroker
from .single_prompt import run_cell

MANIFEST = 'experiments/manifests/single_prompt_gradient_pattern_v1.json'
PREP_SCHEMA = 'single_prompt_phase_prep_v1'
CELL_SCHEMA = 'single_prompt_cell_manifest_v1'
SOURCE_BASE = 'd7de59818da609cff6b03c6ebb334fa79011aedd'
CANARY_MANIFEST = 'experiments/manifests/single_prompt_gradient_pattern_canary_v1.json'
CANARY_STUDY = 'single_prompt_gradient_pattern_canary_v1'


def validate_canary_manifest(root, value):
    expected = dict(schema_version='single_prompt_canary_manifest_v1', experiment_id=CANARY_STUDY,
        parent_manifest=MANIFEST, contract=canary_contract(),
        protocol_path='docs/design/SINGLE_PROMPT_CANARY_V1.md', real_api_authorized=False)
    if value != expected:
        raise SearchContractError('STUDY_CANARY_MANIFEST_INVALID')


def phase_cells(manifest, phase):
    cells = manifest['cells']
    if phase == 'canary':
        cells = [c for c in cells if c['seed'] == canary_contract()['seed']]
    return [dict(c, attempt_id=c['cell_id'] + '_' + phase + '_attempt1') for c in cells]


def frozen_canary_contract(reader):
    c = canary_contract()
    ids = [m['stable_example_id'] for m in reader.members if m['project_split'] == 'optimize'][:c['optimize_count']]
    if len(ids) != c['optimize_count']:
        raise SearchContractError('STUDY_CANARY_MEMBERSHIP_INCOMPLETE')
    return dict(c, sample_ids=ids, membership_sha256=digest(ids))


def private_path(root, path):
    path = Path(path)
    value = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not value.is_relative_to((root / 'runs').resolve()):
        raise SearchContractError('STUDY_ARTIFACT_DESTINATION_MUST_BE_PRIVATE_RUNS')
    return value


def validate_manifest(root, manifest):
    expected_keys = {'schema_version', 'experiment_id', 'identity', 'source_base_sha',
        'policy', 'parents', 'data', 'initial_prompt_path', 'initial_prompt_sha256',
        'cells', 'authorization', 'status', 'protocol_path'}
    if set(manifest) != expected_keys or manifest['schema_version'] != 'single_prompt_study_manifest_v1':
        raise SearchContractError('STUDY_MANIFEST_SCHEMA_INVALID')
    if (manifest['experiment_id'] != STUDY or manifest['identity'] != policy()['identity']
            or manifest['policy'] != policy() or manifest['source_base_sha'] != SOURCE_BASE
            or manifest['authorization'] != dict(real_api_authorized=False, validation_access=False, test_access=False)
            or manifest['status'] != 'OFFLINE_PREPARATION_ONLY'):
        raise SearchContractError('STUDY_MANIFEST_POLICY_MISMATCH')
    if (manifest['parents'] != ['experiments/manifests/weighted_error_responsibility_v210_preparation_v1.yaml',
                               'experiments/manifests/math_v27_seed86_precluster_continuation_v1.yaml']
            or manifest['protocol_path'] != 'docs/design/SINGLE_PROMPT_GRADIENT_PATTERN_V1.md'):
        raise SearchContractError('STUDY_LINEAGE_OR_PROTOCOL_MISMATCH')
    artifact = read_json(root / manifest['initial_prompt_path'])
    if file_hash(root / manifest['initial_prompt_path']) != manifest['initial_prompt_sha256']:
        raise SearchContractError('STUDY_INITIAL_PROMPT_HASH_MISMATCH')
    if len({digest(m['prompt']) for m in artifact['members']}) != 1:
        raise SearchContractError('STUDY_IDENTICAL_INITIAL_PROMPT_REQUIRED')
    prompt = SystemPrompt.from_dict(artifact['members'][0]['prompt'])
    expected = {(a, s) for a in ARMS for s in policy()['seeds']}
    if (len(manifest['cells']) != 9 or {(c['arm'], c['seed']) for c in manifest['cells']} != expected
            or len({c['cell_id'] for c in manifest['cells']}) != 9):
        raise SearchContractError('STUDY_MANIFEST_NINE_CELLS_REQUIRED')
    for c in manifest['cells']:
        if set(c) != {'cell_id', 'arm', 'seed', 'manifest_path'}:
            raise SearchContractError('STUDY_CELL_SCHEMA_INVALID')
        item = read_json(root / c['manifest_path'])
        wanted = dict(schema_version=CELL_SCHEMA, study_id=STUDY, cell_id=c['cell_id'],
            arm=c['arm'], seed=c['seed'], initial_prompt_path=manifest['initial_prompt_path'],
            initial_prompt_sha256=prompt.prompt_hash,
            policy_sha256=digest(policy()), source_base_sha=SOURCE_BASE,
            status='OFFLINE_PREPARATION_ONLY', real_api_authorized=False)
        if item != wanted:
            raise SearchContractError('STUDY_CELL_MANIFEST_MISMATCH')
    data = manifest['data']
    if data != dict(canonical_root='data/benchmark_suite_v1',
        split_directory='experiments/data_splits/math_v2_1_scorable_v1',
        split_manifest_sha256='196f359dad5b90e5dcf5b3b4bf2d11f86ca85825becce5ccb34687ce59d14507',
        canonical_manifest_sha256='7688acc214417677431dd34f63572740986c4dff32c70832274c659c24796a6f',
        membership_sha256='bddbaef2fd356fa32b06d5076a2708aadca4ee72905b0cc918790bfc13c978e3',
        counts=dict(optimize=150, shadow=300, validation=300, test=300)):
        raise SearchContractError('STUDY_DATA_IDENTITY_MISMATCH')
    reader = StudySplitReader(root / data['canonical_root'], root / data['split_directory'], 'math',
        expected_manifest_sha256=data['split_manifest_sha256'], expected_protocol='benchmark_experiment_split_v2')
    if reader.manifest['counts'] != data['counts']:
        raise SearchContractError('STUDY_DATA_COUNT_MISMATCH')
    # Metadata only: constructor never reads held-out row content.
    return reader


def source_snapshot(root):
    paths = set(root.joinpath('multi_dataset_diverse_rl').rglob('*.py'))
    paths.update(root / p for p in ('AGENTS.md', 'scripts/run_experiment.py',
        'scripts/prepare_single_prompt_study.py', MANIFEST, CANARY_MANIFEST,
        'scripts/dry_run_single_prompt_study.py', 'tests/current/test_single_prompt_study.py',
        'scripts/analyze_single_prompt_study.py',
        'tests/formal_zero_api_runner.py', 'tests/conftest.py',
        'docs/design/SINGLE_PROMPT_GRADIENT_PATTERN_V1.md',
        'docs/design/CURRENT_SPEC.md', 'method.md', 'docs/REPOSITORY_MAP.md',
        'docs/workflows/EXPERIMENT_HANDOFF.md', 'docs/failures/registry.yaml',
        'experiments/registry.yaml', 'experiments/lineage.yaml', 'tests/suite_classification.json'))
    paths.update((root / 'docs/design').glob('*.md'))
    paths.add(root / 'requirements-benchmark-evaluators.txt')
    paths.update((root / 'tests/formal_zero_network').rglob('*.py'))
    manifest = read_json(root / MANIFEST)
    paths.update(root / c['manifest_path'] for c in manifest['cells'])
    paths.update(root / p for p in manifest['parents'])
    paths.add(root / manifest['initial_prompt_path'])
    for p in (root / manifest['data']['split_directory']).iterdir():
        if p.is_file():
            paths.add(p)
    paths.add(root / manifest['data']['canonical_root'] / 'manifests/math.json')
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}


def inventory(directory):
    return {p.relative_to(directory).as_posix(): file_hash(p)
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'owner.lock'}


def runtime_identity():
    names = ('math-verify', 'latex2sympy2_extended', 'antlr4-python3-runtime',
             'sympy', 'mpmath', 'openai', 'httpx', 'jsonschema', 'PyYAML')
    return dict(environment_name=Path(sys.prefix).name, python_version=sys.version,
                packages={n: importlib.metadata.version(n) for n in names})


def require_selected_runtime(root):
    p = policy()
    if (Path(sys.prefix).resolve() != (root / p['runtime_environment']).resolve()
            or Path(sys.executable).resolve() != (root / p['runtime_interpreter']).resolve()
            or importlib.metadata.version('openai') != p['provider_sdk_version']):
        raise SearchContractError('STUDY_USER_SELECTED_REPOSITORY_RUNTIME_REQUIRED')


def verify_search(directory, *, implementation_sha=None):
    terminal = read_json(directory / 'terminal.json')
    if terminal['status'] != 'EXECUTION_COMPLETE' or terminal['phase'] != 'train':
        raise SearchContractError('STUDY_COMPLETE_SEARCH_REQUIRED')
    if implementation_sha is not None and terminal['source_sha'] != implementation_sha:
        raise SearchContractError('STUDY_SEARCH_SOURCE_MISMATCH')
    for relative, sha in terminal['artifacts'].items():
        if file_hash(directory / relative) != sha:
            raise SearchContractError('STUDY_SEARCH_ARTIFACT_CORRUPTION')
    results = read_json(directory / 'results.json')
    if (len(results) != 9 or {(r['arm'], r['seed']) for r in results} != {(a, s) for a in ARMS for s in policy()['seeds']}
            or any(r['rounds_completed'] != policy()['rounds'] or len(r['prompt_history']) != policy()['rounds'] + 1
                or r['final_prompt'] != r['prompt_history'][-1] for r in results)):
        raise SearchContractError('STUDY_SEARCH_RESULT_INCOMPLETE')
    if len({r['cell_id'] for r in results}) != 9:
        raise SearchContractError('STUDY_SEARCH_RESULT_INCOMPLETE')
    audit_complete_phase(directory, 'train')
    return results, file_hash(directory / 'terminal.json')


def audit_complete_phase(directory, phase):
    terminal = read_json(directory / 'terminal.json')
    if terminal.get('status') != 'EXECUTION_COMPLETE' or terminal.get('phase') != phase:
        raise SearchContractError('STUDY_PHASE_NOT_COMPLETE')
    expected = terminal['artifacts']
    actual = {k: v for k, v in inventory(directory).items() if k != 'terminal.json'}
    if actual != expected:
        raise SearchContractError('STUDY_PHASE_ARTIFACT_INVENTORY_MISMATCH')
    scope = read_json(directory / 'scope.json')
    if scope['scope_identity'] != terminal['scope_identity'] or scope['source_sha'] != terminal['source_sha']:
        raise SearchContractError('STUDY_PHASE_SCOPE_MISMATCH')
    view, events = audit_journal(directory / 'ledger', task_sha256=scope['scope_identity'], policy=POLICY_V27_FORMAL_UNCAPPED)
    authorized_ids = {c['attempt_id'] for c in scope['cells']}
    calls = [e for e in events if e['kind'] == 'RESERVE']
    if len(calls) > scope['call_ceiling'] or any(e['attempt_id'] not in authorized_ids for e in calls):
        raise SearchContractError('STUDY_PHASE_LEDGER_SCOPE_MISMATCH')
    by_reservation = {e['reservation_id']: e for e in calls}
    for event in events:
        if event['kind'] != 'CHARGE':
            continue
        reserve = by_reservation[event['reservation_id']]
        receipt = event.get('response_receipt')
        if receipt is None:
            raise SearchContractError('STUDY_COMPLETE_PHASE_RECEIPT_REQUIRED')
        store = ProviderResponseReceipts(directory / 'provider_receipts',
            attempt_id=reserve['attempt_id'], startup_identity_sha256=scope['scope_identity'])
        store.read(event['reservation_id'])
        raw_receipt = read_json(directory / 'provider_receipts' / (event['reservation_id'] + '.json'))
        if raw_receipt['integrity_sha256'] != receipt['integrity_sha256']:
            raise SearchContractError('STUDY_LEDGER_RECEIPT_SEAL_MISMATCH')
    return dict(phase=phase, source_sha=scope['source_sha'], scope_identity=scope['scope_identity'],
        physical_calls=len(calls), ledger=view, terminal_sha256=file_hash(directory / 'terminal.json'))


def audit_canary(directory):
    audit = audit_complete_phase(directory, 'canary')
    scope = read_json(directory / 'scope.json')
    results = read_json(directory / 'results.json')
    c = canary_contract()
    if (scope.get('canary_contract', {}).get('identity') != c['identity']
            or len(results) != 3 or {(r['arm'], r['seed']) for r in results} != {(a, c['seed']) for a in ARMS}
            or {r['cell_id'] for r in results} != {r['cell_id'] for r in scope['cells']}):
        raise SearchContractError('STUDY_CANARY_RESULTS_INVALID')
    problems = []
    usage = read_json(directory / 'execution_summary.json')['usage']
    if any(usage[role] == 0 for role in ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster')):
        problems.append('REQUIRED_ROLE_NOT_COVERED')
    for result in results:
        if (result['rounds_completed'] != 1 or len(result['prompt_history']) != 2
                or result['final_prompt'] != result['prompt_history'][-1] or len(result['metrics']) != 1):
            raise SearchContractError('STUDY_CANARY_HORIZON_INVALID')
        for ordinal in (0, 1):
            checkpoint = read_json(directory / result['cell_id'] / f'checkpoint_{ordinal}.json')
            if (set(checkpoint['observations']) != set(scope['canary_contract']['sample_ids'])
                    or checkpoint['prompt'] != result['prompt_history'][ordinal]):
                raise SearchContractError('STUDY_CANARY_CHECKPOINT_INVALID')
        metric = result['metrics'][0]
        expected = metric['error_count'] if result['arm'] == 'full_errors' else min(policy()['error_batch_size'], metric['error_count'])
        if metric['optimizer_example_count'] != expected:
            raise SearchContractError('STUDY_CANARY_ERROR_FEEDBACK_INVALID')
        if metric['update_status'] not in ('REPLACED', 'UNCHANGED_LEGAL_CANDIDATE'):
            problems.append(result['arm'] + '_NO_LEGAL_CANDIDATE')
        if result['arm'] == 'gradient_pattern':
            feedback = metric['structured_feedback']
            if not feedback['gradients'] or not feedback['patterns']:
                problems.append('LEGAL_GRADIENT_PATTERN_NOT_COVERED')
    return dict(status='CANARY_PASS' if not problems else 'CANARY_NOT_PASSED',
        reasons=problems, phase_integrity=audit, role_calls={r: usage[r] for r in
            ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster')},
        accuracy_gate_used=False, heldout_access=False, scientific_results=False)


def prepare(root, destination, *, phase, token_ceiling=None, search_root=None):
    destination = private_path(root, destination)
    if destination.exists():
        raise SearchContractError('STUDY_PREP_ALREADY_EXISTS')
    if phase not in ('canary', 'train', 'validation', 'test') or token_ceiling is not None:
        raise SearchContractError('STUDY_PHASE_OR_BUDGET_INVALID')
    manifest = read_json(root / MANIFEST)
    reader = validate_manifest(root, manifest)
    if phase == 'canary':
        validate_canary_manifest(root, read_json(root / CANARY_MANIFEST))
    require_selected_runtime(root)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip():
        raise SearchContractError('STUDY_SOURCE_NOT_CLEAN')
    sources = source_snapshot(root)
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines())
    if set(sources) - tracked:
        raise SearchContractError('STUDY_SOURCE_NOT_COMMITTED')
    predecessor = None
    if phase in ('validation', 'test'):
        if search_root is None:
            raise SearchContractError('STUDY_HELDOUT_SEARCH_PREDECESSOR_REQUIRED')
        search_root = private_path(root, search_root)
        _, terminal_hash = verify_search(search_root, implementation_sha=sha)
        predecessor = dict(path=search_root.relative_to(root).as_posix(), terminal_sha256=terminal_hash)
    scope = dict(study_id=STUDY, phase=phase, source_sha=sha,
        runtime_identity=runtime_identity(),
        source_hashes=sources, policy=policy(), manifest_sha256=file_hash(root / MANIFEST),
        token_ceiling=token_ceiling, call_ceiling=call_ceiling(phase),
        search_predecessor=predecessor,
        cells=phase_cells(manifest, phase))
    if phase == 'canary':
        scope['canary_contract'] = frozen_canary_contract(reader)
    prep_relative = destination.relative_to(root).as_posix()
    run_relative = (destination.parent / (phase + '_run')).relative_to(root).as_posix()
    scope['execution'] = dict(prep_path=prep_relative, run_root=run_relative,
        command=[policy()['runtime_interpreter'], 'scripts/run_experiment.py', '--prep', prep_relative, '--run-root', run_relative, '--execute'])
    scope['scope_identity'] = digest(scope)
    destination.mkdir(parents=True)
    atomic_write_json(destination / 'prep.json', dict(schema_version=PREP_SCHEMA, scope=scope))
    # Template is never approval. It is checked on every dispatch.
    atomic_write_json(destination / 'authorization.json', dict(scope_identity=scope['scope_identity'],
        approved=False, user_authorization_evidence_sha256=None,
        authorized_attempt_ids=[c['attempt_id'] for c in scope['cells']]))
    return dict(status='FROZEN_NOT_AUTHORIZED',
        scope_identity=scope['scope_identity'], source_sha=sha, phase=phase,
        token_ceiling=token_ceiling, call_ceiling=scope['call_ceiling'], real_api_calls=0)


def validate_prep(root, prep, *, authorized=False):
    prep = private_path(root, prep)
    value = read_json(prep / 'prep.json')
    if value.get('schema_version') != PREP_SCHEMA or set(value) != {'schema_version', 'scope'}:
        raise SearchContractError('STUDY_PREP_SCHEMA_INVALID')
    scope = value['scope']
    extra_keys = {'canary_contract'} if scope.get('phase') == 'canary' else set()
    if (set(scope) != {'study_id', 'phase', 'source_sha', 'source_hashes', 'policy', 'manifest_sha256', 'runtime_identity',
                      'token_ceiling', 'call_ceiling', 'search_predecessor', 'cells', 'scope_identity', 'execution'} | extra_keys
            or scope.get('phase') not in ('canary', 'train', 'validation', 'test') or scope.get('study_id') != STUDY
            or re.fullmatch('[0-9a-f]{40}', str(scope.get('source_sha'))) is None
            or scope.get('policy') != policy() or scope.get('scope_identity') != digest({k: v for k, v in scope.items() if k != 'scope_identity'})
            or scope.get('call_ceiling') != call_ceiling(scope['phase']) or scope.get('token_ceiling') is not None):
        raise SearchContractError('STUDY_SCOPE_IDENTITY_MISMATCH')
    require_selected_runtime(root)
    if scope['runtime_identity'] != runtime_identity():
        raise SearchContractError('STUDY_FROZEN_RUNTIME_CHANGED')
    prep_relative = prep.relative_to(root).as_posix()
    run_relative = (prep.parent / (scope['phase'] + '_run')).relative_to(root).as_posix()
    if scope['execution'] != dict(prep_path=prep_relative, run_root=run_relative,
        command=[policy()['runtime_interpreter'], 'scripts/run_experiment.py', '--prep', prep_relative, '--run-root', run_relative, '--execute']):
        raise SearchContractError('STUDY_FROZEN_COMMAND_MISMATCH')
    manifest = read_json(root / MANIFEST)
    reader = validate_manifest(root, manifest)
    if scope['phase'] == 'canary':
        validate_canary_manifest(root, read_json(root / CANARY_MANIFEST))
        if scope['canary_contract'] != frozen_canary_contract(reader):
            raise SearchContractError('STUDY_CANARY_CONTRACT_MISMATCH')
    if scope['manifest_sha256'] != file_hash(root / MANIFEST) or scope['source_hashes'] != source_snapshot(root):
        raise SearchContractError('STUDY_FROZEN_SOURCE_CHANGED')
    current_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if current_sha != scope['source_sha'] or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip():
        raise SearchContractError('STUDY_FROZEN_COMMIT_MISMATCH')
    wanted = phase_cells(manifest, scope['phase'])
    if scope['cells'] != wanted:
        raise SearchContractError('STUDY_SCOPE_CELL_MISMATCH')
    predecessor = scope['search_predecessor']
    if scope['phase'] in ('train', 'canary'):
        if predecessor is not None:
            raise SearchContractError('STUDY_TRAIN_PREDECESSOR_FORBIDDEN')
    else:
        directory = private_path(root, predecessor['path'])
        _, actual = verify_search(directory, implementation_sha=scope['source_sha'])
        if actual != predecessor['terminal_sha256']:
            raise SearchContractError('STUDY_SEARCH_TERMINAL_CHANGED')
    auth = read_json(prep / 'authorization.json')
    if (set(auth) != {'scope_identity', 'approved', 'user_authorization_evidence_sha256', 'authorized_attempt_ids'}
            or auth['scope_identity'] != scope['scope_identity']
            or auth['authorized_attempt_ids'] != [c['attempt_id'] for c in wanted]):
        raise SearchContractError('STUDY_AUTHORIZATION_SCOPE_MISMATCH')
    if authorized:
        evidence = auth['user_authorization_evidence_sha256']
        if auth['approved'] is not True or not isinstance(evidence, str) or re.fullmatch('[0-9a-f]{64}', evidence) is None:
            raise SearchContractError('STUDY_EXACT_USER_AUTHORIZATION_REQUIRED')
    return scope


@dataclass(frozen=True)
class StudyHeldoutAccess:
    phase: str
    split_manifest_sha256: str
    search_terminal_sha256: str
    require_authorization: object


class StudySplitReader(ExperimentSplitReader):
    def heldout_rows(self, access):
        if (type(access) is not StudyHeldoutAccess or access.phase not in ('validation', 'test')
                or access.split_manifest_sha256 != self.manifest_sha256
                or not isinstance(access.search_terminal_sha256, str) or len(access.search_terminal_sha256) != 64):
            raise SearchContractError('STUDY_HELDOUT_CAPABILITY_REQUIRED')
        access.require_authorization()
        # Exact canonical source/member/reference checks remain the shared reader's.
        return self._read_rows(access.phase)


def execute(root, prep, run_root):
    prep, run_root = private_path(root, prep), private_path(root, run_root)
    # Reject foreign or completed destinations before any lifecycle write.
    if run_root.exists():
        raise SearchContractError('STUDY_EXISTING_RUN_ROOT_FORBIDDEN')
    scope = validate_prep(root, prep, authorized=True)
    if run_root != private_path(root, scope['execution']['run_root']):
        raise SearchContractError('STUDY_FROZEN_RUN_ROOT_MISMATCH')
    consumed = root / 'runs' / 'single_prompt_consumed' / (scope['scope_identity'] + '.json')
    consumed.parent.mkdir(parents=True, exist_ok=True)
    with consumed.open('x', encoding='utf-8') as stream:
        json.dump(dict(scope_identity=scope['scope_identity'], run_root=run_root.relative_to(root).as_posix()), stream)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    run_root.mkdir(parents=True)
    atomic_write_json(run_root / 'scope.json', scope)
    atomic_write_json(run_root / 'authorization.json', read_json(prep / 'authorization.json'))
    atomic_write_json(run_root / 'lifecycle.json', dict(status='RUNNING', phase=scope['phase']))
    clients = None
    ledger = None
    completed = []
    current_cell = None
    ledger_snapshot = None
    # Normal source changes are rejected per dispatch; complete hashes also at
    # each cell boundary and terminal closure. No in-attempt source repair.
    signatures = {p: (root / p).stat() for p in scope['source_hashes']}
    auth_sha = file_hash(prep / 'authorization.json')

    def dispatch_guard(cell_id):
        if file_hash(prep / 'authorization.json') != auth_sha or file_hash(prep / 'prep.json') != prep_sha:
            raise SearchContractError('STUDY_RUNTIME_SCOPE_CHANGED')
        for relative, before in signatures.items():
            actual = (root / relative).stat()
            if (actual.st_size, actual.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise SearchContractError('STUDY_SOURCE_MUTATED_DURING_EXECUTION')
        if cell_id != current_cell:
            raise SearchContractError('STUDY_CELL_CAPABILITY_MISMATCH')

    prep_sha = file_hash(prep / 'prep.json')
    try:
        from ..governance.autonomous_math import create_transport
        from ..provider_routing import routing_contract, IDENTITY
        from ..benchmarks.math_optimizer_generation import optimizer_generation_contract
        transport_contract = dict(provider=IDENTITY, provider_routing_policy=routing_contract(),
            models=dict(solver='gpt-4o-mini', optimizer_reflection='qwen3.7-flash', pattern='qwen3.7-flash', solver_thinking=False),
            decoding=dict(timeout_seconds=policy()['provider_timeout_seconds']), runtime_persistence_policy=True,
            optimizer_generation_policy=optimizer_generation_contract())
        transport, clients = create_transport(transport_contract)
        ledger = TokenLedger(run_root / 'ledger', task_sha256=scope['scope_identity'], policy=POLICY_V27_FORMAL_UNCAPPED)
        receipts = ProviderResponseReceipts(run_root / 'provider_receipts',
            attempt_id=scope['scope_identity'], startup_identity_sha256=scope['scope_identity'])
        broker = StudyBroker(phase=scope['phase'], scope=scope, transport=transport,
            ledger=ledger, receipts=receipts, dispatch_guard=dispatch_guard)
        reader = validate_manifest(root, read_json(root / MANIFEST))
        results = None
        if scope['phase'] in ('train', 'canary'):
            rows = reader.rows('optimize', DataPurpose.SEARCH_EVALUATION)
            if scope['phase'] == 'canary':
                rows = rows[:canary_contract()['optimize_count']]
                if [r['stable_example_id'] for r in rows] != scope['canary_contract']['sample_ids']:
                    raise SearchContractError('STUDY_CANARY_DATA_ORDER_MISMATCH')
        else:
            parent_root = private_path(root, scope['search_predecessor']['path'])
            results, terminal_hash = verify_search(parent_root, implementation_sha=scope['source_sha'])
            rows = reader.heldout_rows(StudyHeldoutAccess(scope['phase'], reader.manifest_sha256, terminal_hash,
                lambda: validate_prep(root, prep, authorized=True)))
        for cell in scope['cells']:
            validate_prep(root, prep, authorized=True)
            current_cell = cell['cell_id']
            cell_root = run_root / current_cell
            cell_root.mkdir()
            atomic_write_json(cell_root / 'consumed_scope.json', dict(attempt_id=cell['attempt_id'], scope_identity=scope['scope_identity']))
            broker.cell_id, broker.seed, broker.attempt_id = current_cell, cell['seed'], cell['attempt_id']
            broker.round_index = 0
            cell_manifest = read_json(root / cell['manifest_path'])
            atomic_write_json(cell_root / 'manifest.json', dict(cell_manifest,
                implementation_sha=scope['source_sha'], phase=scope['phase'], scope_identity=scope['scope_identity']))

            def persist(name, payload):
                target = cell_root / (name + '.json')
                if target.exists():
                    raise SearchContractError('STUDY_ARTIFACT_ALREADY_EXISTS')
                atomic_write_json(target, payload)

            if scope['phase'] in ('train', 'canary'):
                initial = read_json(root / cell_manifest['initial_prompt_path'])['members'][0]['prompt']
                item = run_cell(dict(cell_manifest, initial_prompt=initial), rows, broker.evaluate, broker, persist)
            else:
                selected = next(r for r in results if r['cell_id'] == current_cell)
                prompts = selected['prompt_history'] if scope['phase'] == 'validation' else [selected['final_prompt']]
                checkpoints = []
                for ordinal, prompt in enumerate(prompts):
                    step = ordinal if scope['phase'] == 'validation' else policy()['rounds']
                    observations = broker.evaluate(SystemPrompt.from_dict(prompt), rows, scope['phase'], cell['seed'], step)
                    checkpoints.append(dict(checkpoint=step, accuracy=sum(r['correct'] for r in observations.values()) / len(rows)))
                    persist(f'checkpoint_{step}', dict(prompt=prompt, observations=observations))
                item = dict(cell_id=current_cell, arm=cell['arm'], seed=cell['seed'], checkpoints=checkpoints)
                persist('result', item)
            completed.append(item)
            print(json.dumps(dict(phase=scope['phase'], cell=current_cell, status='CELL_COMPLETE',
                                  provider_calls=broker.usage_view()['provider_calls'])), flush=True)
        atomic_write_json(run_root / 'results.json', completed)
        validate_prep(root, prep, authorized=True)
        if ledger.inflight:
            raise SearchContractError('STUDY_UNRESOLVED_RESERVATION')
        ledger_snapshot = ledger.view()
        atomic_write_json(run_root / 'execution_summary.json', dict(phase=scope['phase'], cells_completed=len(completed),
            usage=broker.usage_view(), token_ledger=ledger_snapshot))
        # Cleanup errors cannot follow a published completion receipt.
        ledger.close()
        ledger = None
        clients.close()
        clients = None
        atomic_write_json(run_root / 'lifecycle.json', dict(status='EXECUTION_COMPLETE', phase=scope['phase']))
        atomic_write_json(run_root / 'terminal.json', dict(status='EXECUTION_COMPLETE', phase=scope['phase'],
            source_sha=scope['source_sha'], scope_identity=scope['scope_identity'], artifacts=inventory(run_root)))
        return dict(status='EXECUTION_COMPLETE', phase=scope['phase'], cells_completed=len(completed))
    except BaseException as exc:
        # The evaluator drains its issued worker requests before this boundary.
        # Failed persistence cannot leave a free or reusable reservation.
        if ledger is not None:
            for key in tuple(ledger.inflight):
                ledger.reconcile(key, {}, outcome='ABORT_FULL_RESERVATION_FALLBACK')
        atomic_write_json(run_root / 'lifecycle.json', dict(status='EXECUTION_ABORTED', phase=scope['phase'],
            error_category=type(exc).__name__, cell_id=current_cell, cells_completed=len(completed)))
        atomic_write_json(run_root / 'abort.json', dict(error_category=type(exc).__name__,
            message=str(exc) if isinstance(exc, SearchContractError) else 'OPERATIONAL_FAILURE_PRIVATE_EVIDENCE_RETAINED',
            cells_completed=len(completed), token_ledger=ledger.view() if ledger is not None else ledger_snapshot))
        raise
    finally:
        if ledger is not None:
            ledger.close()
        if clients is not None:
            clients.close()
