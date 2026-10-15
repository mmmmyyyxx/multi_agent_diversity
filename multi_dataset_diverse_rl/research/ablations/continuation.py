"""Frozen, paid-prefix continuation; only Solver terminal observations are reused."""
from copy import deepcopy
from pathlib import Path
import json
import re
import shutil

from ...benchmarks.data_freeze import digest, file_hash
from ...persistence.durable_io import read_json, atomic_write_json
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt, solver_messages
from ..single_prompt_governance import private_path, inventory, require_selected_runtime

MANIFEST = 'experiments/manifests/single_prompt_pattern_truncation_continuation_v4.json'
PROTOCOL = 'docs/design/SINGLE_PROMPT_PATTERN_TRUNCATION_CONTINUATION_V4.md'
PREP_SCHEMA = 'single_prompt_ablation_continuation_prep_v4'
MANIFEST_SCHEMA = 'single_prompt_ablation_continuation_manifest_v4'


def recovery_policy():
    return dict(identity='ABLATION_DISCARDED_LENGTH_BOUNDED_RECOVERY_V1',
        discarded_finish_reasons=['length', 'max_tokens', 'max_output_tokens'],
        nonthinking='discard_length_before_missing_witness_guard_never_accept_positive_thinking',
        gradient_draws_per_example=3, cluster_draws_per_opportunity=3,
        optimizer_draws_per_opportunity=4, capacity_change=False,
        diagnostic_truncation_exhaustion='skip_failed_gradient_item_cluster_successful_items_continue_optimizer',
        ordinary_no_pattern='unchanged_raw_batch_B_fallback',
        genuine_control_integrity_or_unknown_failure='abort_preserve_evidence',
        incomplete_teacher='no_response_replay_carry_semantic_draw_counts')


def manifest_template(root):
    from . import governance as gov
    return dict(schema_version=MANIFEST_SCHEMA,
        experiment_id='single_prompt_pattern_truncation_continuation_v4',
        parents=[gov.MANIFEST], protocol_path=PROTOCOL,
        generation_recovery=recovery_policy(),
        parent_expectation=dict(source_sha='241f725797080b1b5fc714834b222b675a09d0e8',
            scope_identity='624d623e7ef83118f58d953b7b34cdd01110b7d1e07a8bbaba4b001267d6c1d6',
            completed_opportunities=133, provider_calls=38053, charged_tokens=45105780),
        scientific_plan='unchanged_parent_three_repeats_seed42_twenty_opportunities',
        cache='exact_parent_solver_terminal_observations_same_repeat_only_no_teacher_replay',
        budgets='parent_plus_new_training_190440_semantic_test_separate_10800_tokens_uncapped',
        real_api_authorized=False)


def validate_manifest(root, value):
    from . import governance as gov
    if value != manifest_template(root):
        raise SearchContractError('ABLATION_CONTINUATION_MANIFEST_MISMATCH')
    return gov.validate_manifest(root, read_json(root / gov.MANIFEST))


def _partial_result(cell, folder, usage):
    cps = []
    while (folder / f'checkpoint_{len(cps)}.json').exists():
        cps.append(read_json(folder / f'checkpoint_{len(cps)}.json'))
    if not cps:
        return None
    last = cps[-1]
    return dict(cell=cell, scope_identity=cps[0]['scope_identity'], rounds_completed=len(cps)-1,
        module_binding=cps[0]['scope']['module_binding'], final_prompt=last['prompt'],
        prompt_history=[c['prompt'] for c in cps], metrics=[c['metric'] for c in cps[1:]],
        final_optimize_correct=sum(v['correct'] for v in last['optimize'].values()), optimize_count=150,
        final_validation_correct=sum(v['correct'] for v in last['validation'].values()), validation_count=100,
        module_state=last['module_state'], usage=usage)


def audit_parent(root, parent):
    """Offline audit of the closed parent, including every receipt and transition."""
    from . import governance as gov
    parent = private_path(root, parent)
    scope, abort, life = (read_json(parent / n) for n in ('scope.json', 'abort.json', 'lifecycle.json'))
    expected = manifest_template(root)['parent_expectation']
    if (scope['scope_identity'] != digest({k:v for k,v in scope.items() if k != 'scope_identity'})
            or scope['scope_identity'] != expected['scope_identity'] or scope['source_sha'] != expected['source_sha']
            or scope['plan'] != read_json(root / gov.MANIFEST)['plan'] or scope['phase'] != 'train'
            or life['status'] != 'EXECUTION_ABORTED' or life['error_type'] != 'OperationalAbort'
            or (parent / 'terminal.json').exists() or (parent.parent / 'test_run').exists()
            or read_json(parent / 'access.json')['test_access'] is not False):
        raise SearchContractError('ABLATION_EXACT_CLOSED_PARENT_REQUIRED')
    gov.validate_authorization(scope, read_json(parent / 'authorization.json'), required=True, test=True)
    if read_json(root / 'runs/ablations/consumed' / (scope['scope_identity']+'.json')) != dict(
            scope_identity=scope['scope_identity'], run_root=parent.relative_to(root).as_posix()):
        raise SearchContractError('ABLATION_PARENT_CONSUMPTION_MISMATCH')
    usage = abort['usage']
    if usage['provider_calls'] != expected['provider_calls'] or usage['input_tokens']+usage['output_tokens'] != expected['charged_tokens']:
        raise SearchContractError('ABLATION_PARENT_COST_MISMATCH')
    results, costs, completed = [], {}, {}
    remaining = deepcopy(usage)
    for cell in scope['plan']['cells']:
        folder = parent / cell['cell_id']
        if not (folder / 'checkpoint_0.json').exists():
            break
        if (folder / 'result.json').exists():
            result = read_json(folder / 'result.json')
            for k in remaining:
                remaining[k] -= result['usage'][k]
        else:
            result = _partial_result(cell, folder, remaining)
        results.append(result)
        costs[cell['cell_id']] = result['usage']
        completed[cell['cell_id']] = result['rounds_completed']
    if sum(completed.values()) != expected['completed_opportunities']:
        raise SearchContractError('ABLATION_PARENT_PREFIX_COUNT')
    gov.audit_cells(parent, scope, results=results, allow_partial=True)
    gov.audit_accounting(parent, scope, 'train', summary=abort, results=results)
    # Closed journal was verified above. Inspect only unfinished role/round metadata
    # and request example identities; never feed successful Teacher responses back.
    draws, unfinished_receipts = {}, []
    events = [json.loads(l) for l in (parent/'ledger/events.jsonl').read_text(encoding='utf-8').splitlines()]
    reserves = {e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    truncation = None
    for charge in (e for e in events if e['kind']=='CHARGE'):
        reservation = reserves[charge['reservation_id']]
        cell_id = reservation['attempt_id'].removeprefix(scope['scope_identity']+'_train_')
        if completed.get(cell_id) == scope['plan']['config']['rounds']:
            continue
        path = parent/'provider_receipts'/f"{charge['reservation_id']}.json"
        record = read_json(path)['record']  # Seal already audited above.
        logical = record.get('logical_identity', {})
        if logical.get('ordinal', 0) <= completed.get(cell_id, 0) or record.get('error_category'):
            continue
        role = logical['role']
        if role not in ('pattern_gradient', 'pattern_cluster'):
            raise SearchContractError('ABLATION_UNFINISHED_STAGE_NOT_DIAGNOSIS')
        packet = json.loads(record['request']['messages'][-1]['content'])
        ident = packet['example']['example_id'] if role == 'pattern_gradient' else 'whole_partition'
        counts = draws.setdefault(cell_id, {}).setdefault(str(logical['ordinal']), {})
        key = role+':'+ident
        counts[key] = counts.get(key, 0)+1
        unfinished_receipts.append(dict(reservation_id=charge['reservation_id'], sha256=file_hash(path)))
        response = record.get('response', {})
        if response.get('finish_reason') in recovery_policy()['discarded_finish_reasons']:
            from ...benchmarks.math_optimizer_diagnostics import optimizer_response_telemetry, nonthinking_evidence_contract
            p = scope['plan']['config']['runtime']
            telemetry = optimizer_response_telemetry(record['request'], response,
                p['cluster_generation_policy'] if role=='pattern_cluster' else p['optimizer_generation_policy'],
                nonthinking_evidence_contract())
            if telemetry['enable_thinking'] is not False or telemetry['nonthinking_evidence_level']=='CONTRADICTORY':
                raise SearchContractError('ABLATION_PARENT_NOT_ORDINARY_TRUNCATION')
            truncation = dict(cell_id=cell_id, ordinal=logical['ordinal'], key=key,
                receipt_sha256=file_hash(path), telemetry=telemetry)
    if (truncation is None or draws[truncation['cell_id']][str(truncation['ordinal'])][truncation['key']] != 3
            or any(n > 3 for cells in draws.values() for counts in cells.values() for n in counts.values())):
        raise SearchContractError('ABLATION_PARENT_TRUNCATION_DRAW_BOUND')
    payload = dict(parent_run=parent.relative_to(root).as_posix(), parent_scope_identity=scope['scope_identity'],
        parent_inventory=inventory(parent), inherited_usage=usage, cell_usage=costs,
        completed_rounds=completed, unfinished_module_draws=draws,
        unfinished_receipts=unfinished_receipts, truncation=truncation)
    return dict(payload, integrity_sha256=digest(payload))


def _scope(root, destination):
    from . import governance as gov
    value = gov._scope(root, destination)
    value.pop('scope_identity')
    audit = read_json(destination / 'parent_audit.json')
    if audit.get('integrity_sha256') != digest({k:v for k,v in audit.items() if k != 'integrity_sha256'}):
        raise SearchContractError('ABLATION_PARENT_AUDIT_INTEGRITY')
    expectation = manifest_template(root)['parent_expectation']
    if (audit['parent_scope_identity'] != expectation['scope_identity']
            or sum(audit['completed_rounds'].values()) != expectation['completed_opportunities']
            or audit['inherited_usage']['provider_calls'] != expectation['provider_calls']):
        raise SearchContractError('ABLATION_PARENT_AUDIT_IDENTITY')
    value.update(study_id=manifest_template(root)['experiment_id'], manifest_sha256=file_hash(root/MANIFEST),
        generation_recovery=recovery_policy(), continuation=dict(
            **{k:audit[k] for k in ('parent_run','parent_inventory','inherited_usage','cell_usage',
                                   'completed_rounds','unfinished_module_draws')},
            parent_audit_sha256=file_hash(destination/'parent_audit.json'),
            request_identity_scope=audit['parent_scope_identity'], teacher_response_replay=False))
    return dict(value, scope_identity=digest(value))


def handoff_template(root, destination, scope):
    from . import governance as gov
    value = gov.handoff_template(root, destination, scope)
    value['protocol'].update(name='Single Prompt bounded truncation continuation', version='V4',
        protocol_sha256=file_hash(root/PROTOCOL), manifest_path=MANIFEST,
        preregistration_path=PROTOCOL, preregistration_sha256=file_hash(root/PROTOCOL))
    value.update(resume='audited_133_completed_opportunities_exact_solver_only_cache_no_teacher_replay')
    value['execution']['allowed_retries'] = dict(network=scope['network_recovery'], generation=scope['generation_recovery'])
    return value


def prepare(root, destination, parent):
    from . import governance as gov
    destination = private_path(root, destination)
    require_selected_runtime(root)
    if destination.exists() or gov._git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise SearchContractError('ABLATION_FRESH_PREPARATION_CLEAN_SOURCE_REQUIRED')
    validate_manifest(root, read_json(root/MANIFEST))
    audit = audit_parent(root, parent)
    destination.mkdir(parents=True)
    atomic_write_json(destination/'parent_audit.json', audit)
    scope = _scope(root, destination)
    if set(scope['source_hashes'])-set(gov._git(root, 'ls-files').splitlines()):
        raise SearchContractError('ABLATION_UNCOMMITTED_FREEZE_FILE')
    atomic_write_json(destination/'prep.json', dict(schema_version=PREP_SCHEMA, scope=scope))
    atomic_write_json(destination/'authorization.json', gov.authorization_template(scope))
    atomic_write_json(destination/'execution_plan.json', gov.execution_plan_template(scope))
    atomic_write_json(destination/'handoff.json', handoff_template(root,destination,scope))
    return dict(status='FROZEN_NOT_AUTHORIZED', source_sha=scope['source_sha'],
        scope_identity=scope['scope_identity'], inherited_opportunities=sum(audit['completed_rounds'].values()), real_api_calls=0)


def validate_prep(root, directory, *, authorized=False):
    from . import governance as gov
    require_selected_runtime(root)
    value = read_json(directory/'prep.json')
    validate_manifest(root, read_json(root/MANIFEST))
    if (value.get('schema_version') != PREP_SCHEMA or set(value) != {'schema_version','scope'}
            or gov._git(root,'status','--porcelain','--untracked-files=no') or value['scope'] != _scope(root,directory)):
        raise SearchContractError('ABLATION_CONTINUATION_FROZEN_IDENTITY_CHANGED')
    scope = value['scope']
    auth = read_json(directory/'authorization.json')
    gov.validate_authorization(scope, auth, required=authorized)
    expected = handoff_template(root,directory,scope)
    handoff = read_json(directory/'handoff.json')
    if (read_json(directory/'execution_plan.json') != gov.execution_plan_template(scope)
            or set(handoff) != set(expected) or any(handoff[k] != v for k,v in expected.items() if k not in ('READY_TO_RUN','status'))
            or (authorized and handoff['READY_TO_RUN'] is not True)):
        raise SearchContractError('ABLATION_CONTINUATION_HANDOFF_CHANGED')
    if authorized and file_hash(directory/'user_authorization_evidence.json') != auth['user_authorization_evidence_sha256']:
        raise SearchContractError('ABLATION_CONTINUATION_HUMAN_EVIDENCE_CHANGED')
    return scope


def verify_parent(directory, scope):
    root = Path(directory).resolve()
    for _ in Path(scope['execution']['run_root']).parts:
        root = root.parent
    parent = private_path(root, root/scope['continuation']['parent_run'])
    if inventory(parent) != scope['continuation']['parent_inventory']:
        raise SearchContractError('ABLATION_SEALED_PARENT_CHANGED')


def restore_prefix(root, scope, cell, folder):
    old = scope['continuation']
    count = old['completed_rounds'].get(cell['cell_id'])
    if count is None:
        return None
    source = root/old['parent_run']/cell['cell_id']
    for path in source.glob('*.json'):
        match = re.fullmatch(r'(?:checkpoint_(\d+)|round_(\d+)_.+)\.json', path.name)
        if match and int(next(g for g in match.groups() if g is not None)) <= count:
            shutil.copyfile(path, folder/path.name)
    return dict(checkpoints=[read_json(folder/f'checkpoint_{n}.json') for n in range(count+1)],
        historical_usage=deepcopy(old['cell_usage'][cell['cell_id']]))


def hydrate_solver(root, scope, broker, optimize, validation, directory):
    rows = {split:{r['stable_example_id']:r for r in values} for split,values in
            (('optimize',optimize),('validation',validation))}
    imported = {}
    for cell in scope['plan']['cells']:
        count = scope['continuation']['completed_rounds'].get(cell['cell_id'])
        if count is None:
            continue
        broker.bind_cell(cell, scope['plan'])
        folder = root/scope['continuation']['parent_run']/cell['cell_id']
        for n in range(count+1):
            path = folder/f'checkpoint_{n}.json'
            cp = read_json(path)
            cp_sha = file_hash(path)
            for prefix in ('','candidate_'):
                if cp.get(prefix+'prompt') is None:
                    continue
                prompt = SystemPrompt.from_dict(cp[prefix+'prompt'])
                for split in ('optimize','validation'):
                    for ident, observation in cp[prefix+split].items():
                        row = rows[split][ident]
                        broker.example_context.identity = dict(example_id=ident, prompt_sha256=prompt.prompt_hash,
                            problem_sha256=digest(row['content']['problem']),
                            scoring_reference_sha256=digest(row['reference_final_answer']),
                            scoring_contract='PINNED_MATH_FIRST_VALID_FOUR_DRAWS_V1')
                        _, key = broker._request_identity(role='solver',split=split,messages=solver_messages(prompt,row['content']['problem']))
                        del broker.example_context.identity
                        if key != observation['request_sha256'] or (key in broker.cache and broker.cache[key] != observation):
                            raise SearchContractError('ABLATION_SOLVER_CACHE_RESTORE_IDENTITY')
                        if key not in broker.cache:
                            broker.cache[key] = deepcopy(observation)
                            imported[key] = dict(repeat_id=cell['repeat_id'], split=split,
                                observation_sha256=digest(observation), checkpoint_sha256=cp_sha)
    atomic_write_json(directory/'solver_cache_import.json', dict(
        identity='EXACT_PARENT_SOLVER_ONLY_RESTORE_V1', observations=imported,
        imported_count=len(imported), teacher_responses_replayed=0))
