"""Closed, user-selected two-seed continuation; historical V2 stays frozen."""
from ..persistence.durable_io import read_json
from ..search.schemas import SearchContractError
from ..governance.token_accounting import POLICY_GATED_32_UNCAPPED, POLICY_V27_FORMAL_UNCAPPED
from .gated_prompt_policy import policy

MANIFEST='experiments/manifests/single_prompt_gated_two_seed32_v1.json'
PROTOCOL='docs/design/SINGLE_PROMPT_GATED_TWO_SEED32_V1.md'
PREP_SCHEMA='single_prompt_gated_two_seed32_prep_v1'


def contract():
    return dict(identity='GATED_TWO_SEED_SOLVER32_CONTINUATION_V1',seeds=[88,89],
        cells='three_arms_two_seeds_sequential',rounds_per_cell=20,training_solver_concurrency=32,
        gradient_concurrency=8,test_solver_concurrency=16,test_final_prompts=6,
        test_semantic_call_ceiling=7200,training_semantic_call_ceiling=190440,
        parent_source_sha='7a04de51636065cc5a15a4704c499fc55425f46f',
        parent_stop='USER_REQUESTED_EXECUTION_AMENDMENT',
        checkpoint='preserve_completed_and_exact_unfinished_paid_prefix',
        excluded_seed='preserve_evidence_and_costs_no_new_seed90_dispatch',
        scientific_gate='unchanged_validation100_strict_gain_retry3',
        network_recovery='unchanged_unlimited_typed_delayed_retry',
        test_alignment='unchanged_formal_test300_solver_scoring_decoding_and_concurrency')


def manifest_template():
    return dict(schema_version='single_prompt_gated_two_seed32_manifest_v1',
        experiment_id='single_prompt_gated_two_seed32_v1',
        parent_manifest='experiments/manifests/single_prompt_gated_network_continuation_v1.json',
        execution_amendment=contract(),protocol_path=PROTOCOL,real_api_authorized=False)


def validate_manifest(value):
    if value!=manifest_template():raise SearchContractError('GATED_EXECUTION_AMENDMENT_MANIFEST_MISMATCH')


def amendment(scope):
    value=scope.get('execution_amendment')
    if 'execution_amendment' in scope and value!=contract():raise SearchContractError('GATED_EXECUTION_AMENDMENT_MISMATCH')
    return value


def selected_cells(manifest,enabled=False):
    return [c for c in manifest['cells'] if not enabled or c['seed'] in contract()['seeds']]


def runtime_policy(scope,phase):
    p=policy();a=amendment(scope)
    if a:p.update(seeds=a['seeds'],solver_concurrency=a['training_solver_concurrency'] if phase=='train' else a['test_solver_concurrency'])
    return p


def bounded_training_map(function,items,scope):
    """Explicit study-only32; production's default map remains bounded at16."""
    from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
    if amendment(scope) is None:raise SearchContractError('GATED_SOLVER32_SCOPE_REQUIRED')
    items=tuple(items);results=[None]*len(items);cursor=0;failure=None
    with ThreadPoolExecutor(max_workers=32,thread_name_prefix='gated_solver32') as pool:
        pending={}
        while cursor<len(items) or pending:
            while failure is None and cursor<len(items) and len(pending)<32:
                pending[pool.submit(function,items[cursor])]=cursor;cursor+=1
            if not pending:break
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in sorted(completed,key=pending.get):
                index=pending.pop(future)
                try:results[index]=future.result()
                except BaseException as exc:
                    if failure is None:failure=exc
        if failure is not None:raise failure
    return tuple(results)


def ledger_policy(scope,phase='train'):
    return POLICY_GATED_32_UNCAPPED if amendment(scope) and phase=='train' else POLICY_V27_FORMAL_UNCAPPED


def test_plan(plan,enabled=False):
    if not enabled:return plan
    return dict(plan,identity='GATED_AUTOMATIC_TWO_SEED_FINAL_TEST300_V1',
        selection='last_validation_retained_prompt_after_all_six_twenty_opportunity_cells',
        physical_call_ceiling=7200)


def validate_stop(root,directory,scope,abort):
    from ..benchmarks.data_freeze import file_hash
    receipt=directory.parent/'user_stop_20261014/closure.json'
    row=read_json(receipt)
    if (scope['source_sha']!=contract()['parent_source_sha']
            or abort['owner_stop_receipt_sha256']!=file_hash(receipt)
            or row['identity']!='GATED_USER_REQUESTED_EXECUTION_AMENDMENT_STOP_V1'
            or row['source_sha']!=scope['source_sha'] or row['scope_identity']!=scope['scope_identity']
            or row['processes_verified_dead'] is not True or row['model_calls_from_closure']!=0):
        raise SearchContractError('GATED_OWNER_STOP_EVIDENCE_MISMATCH')
    return set(row['orphan_reservations'])
