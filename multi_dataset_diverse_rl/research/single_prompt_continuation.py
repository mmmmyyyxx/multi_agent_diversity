"""Explicit timeout-only continuation of the independent study's paid prefix.

No historical team execution or generic receipt resume capability is enabled.
Successful responses are replayed by exact logical request and draw identity;
unsuccessful transport attempts retain their charges and produce no model draw.
"""
from collections import defaultdict, deque
from copy import deepcopy
import re
import json

from ..benchmarks.data_freeze import digest, file_hash
from ..governance.token_accounting import audit_journal, POLICY_V27_FORMAL_UNCAPPED, serialized_request
import hashlib
from ..persistence.durable_io import read_json, append_jsonl
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..search.schemas import SearchContractError

MANIFEST = 'experiments/manifests/single_prompt_timeout_continuation_v1.json'
PREP_SCHEMA = 'single_prompt_timeout_continuation_prep_v1'


def operation_contract():
    return dict(identity='SINGLE_PROMPT_TIMEOUT_DURABLE_CONTINUATION_V1',
        phase='train', parent='sealed_timeout_aborted_same_study_only',
        retry_errors=['APITimeoutError'], timeout_seconds=120,
        backoff_seconds=[2, 4, 8, 16, 30, 60], backoff_tail_seconds=60,
        retries='until_success_or_combined_physical_call_ceiling',
        timeout_consumes_semantic_draw=False, successful_draw_reuse='exact_request_seed_round_role_stage_fifo',
        preserve_optimizer_response=True, replay_does_not_charge_twice=True,
        request_namespace='preserve_root_parent_scope_for_byte_identical_optimizer_evidence',
        combined_physical_call_ceiling=114480, cumulative_token_ceiling=None,
        automatic_timeout_recovery_authorized=True, non_timeout_failure='abort',
        validation_access=False, test_access=False)


def validate_operation_manifest(value):
    if value != dict(schema_version='single_prompt_timeout_continuation_manifest_v1',
        experiment_id='single_prompt_timeout_continuation_v1',
        parent_manifest='experiments/manifests/single_prompt_gradient_pattern_v1.json',
        operation_policy=operation_contract(), protocol_path='docs/design/SINGLE_PROMPT_TIMEOUT_CONTINUATION_V1.md',
        real_api_authorized=False):
        raise SearchContractError('STUDY_CONTINUATION_MANIFEST_MISMATCH')


def replay_key(request, seed, ordinal, role, stage):
    return digest(dict(request_sha256=hashlib.sha256(serialized_request(request)).hexdigest(),
        seed=seed, ordinal=ordinal, role=role, stage=stage))


def replay_guard(request, seed, ordinal, role, stage):
    subject = request['messages'][-1]['content'] if role == 'solver' else (
        json.loads(request['messages'][-1]['content'])['example']['example_id'] if role == 'pattern_gradient' else '')
    return digest(dict(seed=seed, ordinal=ordinal, role=role, stage=stage, subject=subject))


def continuation_cells(manifest, prep_relative):
    suffix = digest(prep_relative)[:16]
    return [dict(c, attempt_id=c['cell_id'] + '_train_continuation_' + suffix) for c in manifest['cells']]


def audit_parent(root, directory, *, expected_inventory=None, seen=None):
    # Import here avoids a module cycle with governed preparation.
    from .single_prompt_governance import inventory, private_path, PREP_SCHEMA as ORIGINAL_SCHEMA, STUDY
    from .single_prompt_policy import policy
    directory = private_path(root, directory)
    seen = set() if seen is None else set(seen)
    if directory in seen:
        raise SearchContractError('STUDY_CONTINUATION_PARENT_CYCLE')
    seen.add(directory)
    actual_inventory = inventory(directory)
    if expected_inventory is not None and actual_inventory != expected_inventory:
        raise SearchContractError('STUDY_CONTINUATION_PARENT_MUTATED')
    scope = read_json(directory / 'scope.json')
    if (scope.get('phase') != 'train' or scope.get('study_id') != STUDY or scope.get('policy') != policy()
            or digest({k:v for k,v in scope.items() if k != 'scope_identity'}) != scope.get('scope_identity')
            or scope.get('search_predecessor') is not None
            or scope.get('call_ceiling') != operation_contract()['combined_physical_call_ceiling']):
        raise SearchContractError('STUDY_CONTINUATION_PARENT_SCOPE_MISMATCH')
    lifecycle = read_json(directory / 'lifecycle.json')
    aborted = read_json(directory / 'abort.json')
    if (lifecycle.get('status') != 'EXECUTION_ABORTED' or lifecycle.get('error_category') != 'OperationalAbort'
            or aborted.get('error_category') != 'OperationalAbort' or (directory / 'terminal.json').exists()):
        raise SearchContractError('STUDY_CONTINUATION_PARENT_NOT_TIMEOUT_ABORT')
    prep = private_path(root, scope['execution']['prep_path'])
    if read_json(prep / 'prep.json') != dict(schema_version=PREP_SCHEMA if 'continuation' in scope else ORIGINAL_SCHEMA, scope=scope):
        raise SearchContractError('STUDY_CONTINUATION_PARENT_PREP_MISMATCH')
    auth = read_json(directory / 'authorization.json')
    if (auth != read_json(prep / 'authorization.json') or auth.get('approved') is not True
            or auth.get('scope_identity') != scope['scope_identity']
            or auth.get('authorized_attempt_ids') != [c['attempt_id'] for c in scope['cells']]):
        raise SearchContractError('STUDY_CONTINUATION_PARENT_AUTHORIZATION_MISMATCH')
    consumed = read_json(root / 'runs/single_prompt_consumed' / (scope['scope_identity'] + '.json'))
    if consumed != dict(scope_identity=scope['scope_identity'], run_root=directory.relative_to(root).as_posix()):
        raise SearchContractError('STUDY_CONTINUATION_PARENT_NOT_CONSUMED')
    view, events = audit_journal(directory / 'ledger', task_sha256=scope['scope_identity'], policy=POLICY_V27_FORMAL_UNCAPPED)
    if view != aborted['token_ledger']:
        raise SearchContractError('STUDY_CONTINUATION_PARENT_LEDGER_MISMATCH')
    cells = {c['attempt_id']:c for c in scope['cells']}
    if len(cells) != 9 or {(c['arm'],c['seed']) for c in cells.values()} != {(a,s) for a in policy()['arms'] for s in policy()['seeds']}:
        raise SearchContractError('STUDY_CONTINUATION_PARENT_CELLS_MISMATCH')
    historical_cost_by_cell = {}
    replay, inherited = [], dict(provider_calls=0, input_tokens=0, output_tokens=0,
        supplier_reported_tokens=0, fallback_usage_calls=0)
    for role in ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'):
        inherited.update({role:0, role+'_input_tokens':0, role+'_output_tokens':0})
    if 'continuation' in scope:
        old = scope['continuation']
        if old['operation_policy'] != operation_contract():
            raise SearchContractError('STUDY_CONTINUATION_PARENT_POLICY_MISMATCH')
        ancestor = audit_parent(root, root/old['parent_run_root'], expected_inventory=old['parent_inventory'], seen=seen)
        replay.extend(ancestor['replay'])
        inherited.update(ancestor['inherited_usage'])
        historical_cost_by_cell = deepcopy(ancestor['historical_cost_by_cell'])
    else:
        # The one explicitly authorized migration predates durable logical IDs.
        if scope['source_sha'] != '93c7f7f6804dfda30c0ab2452f78c806d3c3814c':
            raise SearchContractError('STUDY_CONTINUATION_UNAPPROVED_HISTORICAL_SOURCE')
    reserves = {e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    failures = 0
    for e in events:
        if e['kind'] != 'CHARGE':
            continue
        reserve = reserves[e['reservation_id']]
        if reserve['attempt_id'] not in cells or not e.get('response_receipt'):
            raise SearchContractError('STUDY_CONTINUATION_PARENT_RECEIPT_REQUIRED')
        role = reserve['role']
        if role not in ('solver','reflection','pattern_gradient','pattern_cluster'):
            raise SearchContractError('STUDY_CONTINUATION_PARENT_ROLE_INVALID')
        inherited['provider_calls'] += 1
        inherited[role] += 1
        for k in ('input_tokens','output_tokens'):
            inherited[k] += e[k]
            inherited[role+'_'+k] += e[k]
        inherited['supplier_reported_tokens'] += e['charged'] if e['reliable_usage'] else 0
        inherited['fallback_usage_calls'] += not e['reliable_usage']
        store = ProviderResponseReceipts.__new__(ProviderResponseReceipts)
        store.directory, store.attempt_id, store.startup = directory/'provider_receipts', reserve['attempt_id'], scope['scope_identity']
        record = store.read(e['reservation_id'])
        if read_json(store.directory/(e['reservation_id']+'.json'))['integrity_sha256'] != e['response_receipt']['integrity_sha256']:
            raise SearchContractError('STUDY_CONTINUATION_RECEIPT_LEDGER_MISMATCH')
        seed, stage = cells[reserve['attempt_id']]['seed'], reserve['stage']
        if 'logical_identity' in record:
            logical = record['logical_identity']
            ordinal = logical['ordinal']
            if logical != dict(seed=seed, ordinal=ordinal, role=role, stage=stage):
                raise SearchContractError('STUDY_CONTINUATION_LOGICAL_IDENTITY_MISMATCH')
        else:
            match = re.fullmatch(r'optimize_checkpoint_(\d+)_draw_[1-4]', stage) if role=='solver' else re.fullmatch(r'round_(\d+)_single_candidate_draw_[1-2]', stage) if role=='reflection' else None
            if match is None:
                raise SearchContractError('STUDY_CONTINUATION_LEGACY_STAGE_UNSUPPORTED')
            ordinal = int(match[1])
        if not 0 <= ordinal <= 20:
            raise SearchContractError('STUDY_CONTINUATION_ROUND_INVALID')
        cost = historical_cost_by_cell.setdefault(cells[reserve['attempt_id']]['cell_id'], {}).setdefault(str(ordinal), {})
        delta = dict(provider_calls=1, input_tokens=e['input_tokens'], output_tokens=e['output_tokens'],
            supplier_reported_tokens=e['charged'] if e['reliable_usage'] else 0,
            fallback_usage_calls=int(not e['reliable_usage']), **{role:1,role+'_input_tokens':e['input_tokens'],role+'_output_tokens':e['output_tokens']})
        for k,v in delta.items():cost[k]=cost.get(k,0)+v
        if 'error_category' in record:
            if record['error_category'] != 'APITimeoutError' or e['outcome'] not in ('PROVIDER_FAILURE','PROVIDER_TIMEOUT'):
                raise SearchContractError('STUDY_CONTINUATION_NON_TIMEOUT_PARENT_FAILURE')
            failures += 1
            continue
        if e['outcome'] != 'PROVIDER_RESPONSE' or set(record) - {'request','response','logical_identity'}:
            raise SearchContractError('STUDY_CONTINUATION_PARENT_RESPONSE_INVALID')
        replay.append(dict(key=replay_key(record['request'],seed,ordinal,role,stage),
            guard_key=replay_guard(record['request'],seed,ordinal,role,stage),
            response=record['response'], source_scope_identity=scope['scope_identity'],
            reservation_id=e['reservation_id'], source_receipt_sha256=e['response_receipt']['integrity_sha256']))
    if not failures or inherited['provider_calls'] >= operation_contract()['combined_physical_call_ceiling']:
        raise SearchContractError('STUDY_CONTINUATION_PARENT_TIMEOUT_OR_BUDGET_REQUIRED')
    return dict(parent_run_root=directory.relative_to(root).as_posix(), parent_inventory=actual_inventory,
        parent_scope_identity=scope['scope_identity'], parent_source_sha=scope['source_sha'],
        request_identity_scope=scope.get('continuation',{}).get('request_identity_scope',scope['scope_identity']),
        inherited_usage=inherited, historical_cost_by_cell=historical_cost_by_cell,
        replay=replay, operation_policy=operation_contract())


def frozen_parent(root, directory):
    return {k:v for k,v in audit_parent(root,directory).items() if k != 'replay'}


class ResponseReplay:
    def __init__(self, entries, log_path):
        self.entries = defaultdict(deque)
        self.log_path = log_path
        for row in entries:
            self.entries[row['key']].append(row)

    def take(self, key, guard_key):
        if not self.entries[key]:
            if any(row['guard_key'] == guard_key for rows in self.entries.values() for row in rows):
                raise SearchContractError('STUDY_CONTINUATION_REPLAY_REQUEST_MISMATCH')
            return None
        row = self.entries[key].popleft()
        append_jsonl(self.log_path, {k:v for k,v in row.items() if k != 'response'})
        return deepcopy(row['response'])

    def require_consumed(self):
        if any(self.entries.values()):
            raise SearchContractError('STUDY_CONTINUATION_UNCONSUMED_PARENT_RESPONSES')
