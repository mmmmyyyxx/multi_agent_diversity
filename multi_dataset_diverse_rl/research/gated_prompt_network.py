"""Accounted network-only continuation of compatible, sealed V2 evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib
import shutil
import threading
from ..benchmarks.data_freeze import digest,file_hash
from ..governance.token_accounting import audit_journal,POLICY_V27_FORMAL_UNCAPPED,serialized_request
from ..persistence.durable_io import read_json,atomic_write_json
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..search.schemas import SearchContractError
from .gated_prompt import GatedPromptBroker
from .gated_prompt_policy import policy,call_ceiling
from .single_prompt_continuation import ResponseReplay,replay_key,replay_guard

MANIFEST='experiments/manifests/single_prompt_gated_network_continuation_v1.json'
PROTOCOL='docs/design/SINGLE_PROMPT_GATED_NETWORK_CONTINUATION_V1.md'
PREP_SCHEMA='single_prompt_gated_network_continuation_prep_v1'


def operation_contract():
    return dict(identity='GATED_ACCOUNTED_TIMEOUT_CONNECTION_CONTINUATION_V1',
        retry_errors=['APITimeoutError','APIConnectionError'],backoff_seconds=[2,4,8,16,30,60],
        combined_physical_call_ceiling=None,test_physical_call_ceiling=None,cumulative_token_ceiling=None,
        successful_semantic_call_ceiling=call_ceiling(),test_successful_semantic_call_ceiling=10800,
        network_failed_attempts='unlimited_until_success_or_human_stop_never_consume_opportunity',
        parent='sealed_compatible_network_aborted_gated_v2_only',checkpoint='restore_completed_opportunities',
        response_reuse='exact_request_seed_round_role_stage_fifo_unfinished_opportunity',
        cache='hydrate_completed_paired_observations_preserve_root_request_namespace',
        costs='all_ancestor_attempts_plus_new_calls_no_double_charge',non_network_failure='abort',
        scientific_policy='unchanged_validation100_retry3_twenty_opportunity_gate',final_test='preserve_conditional_formal_test300')


def validate_operation_manifest(value):
    expected=dict(schema_version='single_prompt_gated_network_continuation_manifest_v1',
        experiment_id='single_prompt_gated_network_continuation_v1',
        parent_manifest='experiments/manifests/single_prompt_validation_gated_v2.json',
        operation_policy=operation_contract(),protocol_path=PROTOCOL,real_api_authorized=False)
    if value!=expected:raise SearchContractError('GATED_NETWORK_MANIFEST_MISMATCH')


class NetworkRecoveryMixin:
    def network_enabled(self):return 'continuation' in self.scope or 'network_recovery' in self.scope
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.network_abort=threading.Event()
        if self.network_enabled():self.usage.setdefault('network_failed_calls',0)
    def recoverable_transport_error(self,exc):
        if not self.network_enabled():return super().recoverable_transport_error(exc)
        from openai import APITimeoutError,APIConnectionError
        return isinstance(exc,(APITimeoutError,APIConnectionError))
    def record_recovery_failure(self):
        if self.network_enabled():self.usage['network_failed_calls']+=1
        else:super().record_recovery_failure()
    def budgeted_provider_calls(self):
        if not self.network_enabled():return super().budgeted_provider_calls()
        if self.network_abort.is_set():
            from ..governance.token_accounting import OperationalAbort
            raise OperationalAbort('GATED_NETWORK_FATAL_DRAIN')
        return self.usage['provider_calls']-self.usage['network_failed_calls']
    def complete(self,**kwargs):
        try:return super().complete(**kwargs)
        except BaseException as exc:
            from ..search.generation_failures import GeneratedOutputFailure
            if self.network_enabled() and not isinstance(exc,GeneratedOutputFailure):self.network_abort.set()
            raise
    def prediction(self,*args,**kwargs):
        try:return super().prediction(*args,**kwargs)
        except BaseException:
            if self.network_enabled():self.network_abort.set()
            raise


class NetworkBroker(NetworkRecoveryMixin,GatedPromptBroker):
    def timeout_policy(self):return self.scope['continuation']['operation_policy']


def audit_parent(root,directory,*,expected_inventory=None,seen=None):
    from . import gated_prompt_governance as gov
    directory=gov.shared.private_path(root,directory)
    seen=set() if seen is None else set(seen)
    if str(directory) in seen:raise SearchContractError('GATED_NETWORK_PARENT_CYCLE')
    seen.add(str(directory));actual=gov.shared.inventory(directory)
    if expected_inventory is not None and actual!=expected_inventory:raise SearchContractError('GATED_NETWORK_PARENT_CHANGED')
    scope=read_json(directory/'scope.json');abort=read_json(directory/'abort.json');life=read_json(directory/'lifecycle.json')
    if (scope['scope_identity']!=digest({k:v for k,v in scope.items() if k!='scope_identity'})
            or scope['study_id']!=gov.STUDY or scope['phase']!='train' or scope['policy']!=policy()
            or scope['manifest_sha256']!=file_hash(root/gov.MANIFEST)
            or scope['call_ceiling']!=call_ceiling() or scope['token_ceiling'] is not None
            or life['status']!='EXECUTION_ABORTED' or abort['reason']!='STUDY_PROVIDER_FAILURE'
            or (directory/'terminal.json').exists()):
        raise SearchContractError('GATED_NETWORK_COMPATIBLE_ABORT_REQUIRED')
    auth=read_json(directory/'authorization.json')
    if (auth['approved'] is not True or auth['adaptive_validation_access'] is not True
            or auth['test_access'] is not True or auth['scope_identity']!=scope['scope_identity']
            or auth['authorized_attempt_ids']!=[c['attempt_id'] for c in scope['cells']]):
        raise SearchContractError('GATED_NETWORK_PARENT_AUTHORIZATION_MISMATCH')
    marker=read_json(root/'runs/gated_prompt_consumed'/(scope['scope_identity']+'.json'))
    if marker!=dict(scope_identity=scope['scope_identity'],run_root=directory.relative_to(root).as_posix()):
        raise SearchContractError('GATED_NETWORK_PARENT_NOT_CONSUMED')
    view,events=audit_journal(directory/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
    if view!=abort['token_ledger']:raise SearchContractError('GATED_NETWORK_PARENT_LEDGER_MISMATCH')
    parent_cells={c['attempt_id']:c for c in scope['cells']}
    if [(c['cell_id'],c['arm'],c['seed']) for c in scope['cells']]!=[(c['cell_id'],c['arm'],c['seed']) for c in read_json(root/gov.MANIFEST)['cells']]:
        raise SearchContractError('GATED_NETWORK_PARENT_CELLS_MISMATCH')
    inherited={k:0 for k in ('provider_calls','input_tokens','output_tokens','supplier_reported_tokens','fallback_usage_calls','timeout_retries','network_failed_calls')}
    costs={};entries=[]
    if 'continuation' in scope:
        old=scope['continuation']
        if old['operation_policy']!=operation_contract():raise SearchContractError('GATED_NETWORK_ANCESTOR_POLICY_MISMATCH')
        ancestor=audit_parent(root,root/old['parent_run_root'],expected_inventory=old['parent_inventory'],seen=seen)
        inherited.update(ancestor['inherited_usage']);costs=deepcopy(ancestor['historical_cost_by_cell']);entries.extend(ancestor['replay'])
    elif scope['source_sha']!='4d1b7125926ca6353b269a71110fee880d722261':
        raise SearchContractError('GATED_NETWORK_UNAPPROVED_ROOT_SOURCE')
    for role in ('solver','reflection','pattern_gradient','pattern_cluster'):
        for key in (role,role+'_input_tokens',role+'_output_tokens'):inherited.setdefault(key,0)
    reserves={e['reservation_id']:e for e in events if e['kind']=='RESERVE'};failures=0
    for e in events:
        if e['kind']!='CHARGE':continue
        reserve=reserves[e['reservation_id']];receipt=e.get('response_receipt')
        if receipt is None or reserve['attempt_id'] not in parent_cells:raise SearchContractError('GATED_NETWORK_RECEIPT_REQUIRED')
        cell=parent_cells[reserve['attempt_id']];role=reserve['role']
        store=ProviderResponseReceipts.__new__(ProviderResponseReceipts)
        store.directory=directory/'provider_receipts';store.attempt_id=reserve['attempt_id'];store.startup=scope['scope_identity']
        record=store.read(e['reservation_id'])
        if read_json(store.directory/(e['reservation_id']+'.json'))['integrity_sha256']!=receipt['integrity_sha256']:
            raise SearchContractError('GATED_NETWORK_RECEIPT_LEDGER_SEAL_MISMATCH')
        logical=record['logical_identity'];ordinal=logical['ordinal'];stage=reserve['stage']
        if (logical!=dict(seed=cell['seed'],ordinal=ordinal,role=role,stage=stage) or not 0<=ordinal<=20
                or hashlib.sha256(serialized_request(record['request'])).hexdigest()!=reserve['request_sha256']):
            raise SearchContractError('GATED_NETWORK_LOGICAL_REQUEST_MISMATCH')
        delta=dict(provider_calls=1,input_tokens=e['input_tokens'],output_tokens=e['output_tokens'],
            supplier_reported_tokens=e['charged'] if e['reliable_usage'] else 0,fallback_usage_calls=int(not e['reliable_usage']),
            **{role:1,role+'_input_tokens':e['input_tokens'],role+'_output_tokens':e['output_tokens']})
        per_round=costs.setdefault(cell['cell_id'],{}).setdefault(str(ordinal),{})
        for k,v in delta.items():inherited[k]=inherited.get(k,0)+v;per_round[k]=per_round.get(k,0)+v
        if 'error_category' in record:
            if record['error_category'] not in operation_contract()['retry_errors'] or e['outcome'] not in ('PROVIDER_TIMEOUT','PROVIDER_FAILURE'):
                raise SearchContractError('GATED_NETWORK_NON_NETWORK_PARENT_FAILURE')
            failures+=1;inherited['timeout_retries']+=record['error_category']=='APITimeoutError';inherited['network_failed_calls']+=1;continue
        if e['outcome']!='PROVIDER_RESPONSE':raise SearchContractError('GATED_NETWORK_RESPONSE_OUTCOME_MISMATCH')
        entries.append(dict(key=replay_key(record['request'],cell['seed'],ordinal,role,stage),
            guard_key=replay_guard(record['request'],cell['seed'],ordinal,role,stage),response=record['response'],
            source_scope_identity=scope['scope_identity'],reservation_id=e['reservation_id'],
            source_receipt_sha256=receipt['integrity_sha256'],cell_id=cell['cell_id'],ordinal=ordinal,
            request=record['request'],seed=cell['seed'],role=role,stage=stage))
    if not failures or inherited['provider_calls']-inherited['network_failed_calls']>=call_ceiling():raise SearchContractError('GATED_NETWORK_FAILURE_AND_REMAINING_BUDGET_REQUIRED')
    progress={};checkpoint_counts={}
    for cell in scope['cells']:
        folder=directory/cell['cell_id'];paths=sorted(folder.glob('checkpoint_*.json'),key=lambda p:int(p.stem.split('_')[-1]))
        checkpoints=[read_json(p) for p in paths]
        if [int(p.stem.split('_')[-1]) for p in paths]!=list(range(len(paths))):raise SearchContractError('GATED_NETWORK_CHECKPOINT_GAP')
        if len(paths)>21:raise SearchContractError('GATED_NETWORK_TOO_MANY_CHECKPOINTS')
        for n,cp in enumerate(checkpoints):
            if len(cp['optimize'])!=150 or len(cp['validation'])!=100:raise SearchContractError('GATED_NETWORK_CHECKPOINT_MEMBERSHIP_COUNT')
            if n:
                before=checkpoints[n-1];metric=cp['metric']
                if metric['round']!=n or metric['validation_before']!=sum(v['correct'] for v in before['validation'].values()):
                    raise SearchContractError('GATED_NETWORK_CHECKPOINT_GATE_MISMATCH')
                if metric['accepted']:
                    if not metric['validation_candidate']>metric['validation_before'] or cp['prompt']!=cp['candidate_prompt']:
                        raise SearchContractError('GATED_NETWORK_ACCEPTANCE_MISMATCH')
                elif any(cp[k]!=before[k] for k in ('prompt','optimize','validation')):
                    raise SearchContractError('GATED_NETWORK_RETAINED_PROFILE_CHANGED')
                if metric['decision']=='ABANDONED_GENERATION_RETRIES_EXHAUSTED':
                    attempts=[read_json(folder/f'round_{n}_generation_{draw}.json') for draw in range(1,5)]
                    if metric['generation_draws']!=4 or any(not a['audit']['errors'] or a['candidate'] is not None for a in attempts):
                        raise SearchContractError('GATED_NETWORK_FALSE_COMPLETED_ABANDONMENT')
        if (folder/'result.json').exists() and (len(paths)!=21 or read_json(folder/'result.json')['final_prompt']!=checkpoints[-1]['prompt']):
            raise SearchContractError('GATED_NETWORK_COMPLETE_CELL_MISMATCH')
        checkpoint_counts[cell['cell_id']]=len(paths)
        round_costs=costs.get(cell['cell_id'],{});usage={}
        for delta in round_costs.values():
            for k,v in delta.items():usage[k]=usage.get(k,0)+v
        progress[cell['cell_id']]=dict(checkpoints=checkpoints,round_costs=round_costs,historical_usage=usage)
    active=[e for e in entries if e['ordinal']>=checkpoint_counts[e['cell_id']]]
    frozen=dict(parent_run_root=directory.relative_to(root).as_posix(),parent_inventory=actual,
        parent_scope_identity=scope['scope_identity'],parent_source_sha=scope['source_sha'],
        request_identity_scope=scope.get('continuation',{}).get('request_identity_scope',scope['scope_identity']),
        inherited_usage=inherited,historical_cost_by_cell=costs,checkpoint_counts=checkpoint_counts,
        replay_sha256=digest(active),operation_policy=operation_contract())
    return dict(**frozen,replay=active,progress=progress)


def frozen_parent(root,directory,**kwargs):
    return {k:v for k,v in audit_parent(root,directory,**kwargs).items() if k not in ('replay','progress')}


def initialize_resume(root,run,scope,broker):
    parent=scope['continuation'];audited=audit_parent(root,root/parent['parent_run_root'],expected_inventory=parent['parent_inventory'])
    if {k:v for k,v in audited.items() if k not in ('replay','progress')}!=parent:
        raise SearchContractError('GATED_NETWORK_FROZEN_PARENT_MISMATCH')
    broker.replay=ResponseReplay(audited['replay'],run/'response_replay.jsonl')
    for cell in scope['cells']:
        for cp in audited['progress'][cell['cell_id']]['checkpoints']:
            for field in ('optimize','validation','candidate_optimize','candidate_validation'):
                for value in (cp.get(field) or {}).values():
                    key=value['request_sha256']
                    if key in broker.cache and broker.cache[key]!=value:raise SearchContractError('GATED_NETWORK_PAIRED_CACHE_CONFLICT')
                    broker.cache[key]=deepcopy(value)
    atomic_write_json(run/'parent_accounting.json',parent)
    return audited['progress']


def import_cell(root,run,scope,cell):
    parent=root/scope['continuation']['parent_run_root']/cell['cell_id'];target=run/cell['cell_id']
    target.mkdir()
    if parent.exists():
        for path in parent.glob('*.json'):
            if path.is_file():shutil.copyfile(path,target/path.name)
    return target
