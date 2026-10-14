"""Closed largest-support focus and exact paid-response reuse for corrected C."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random
import shutil
from ..benchmarks.data_freeze import digest,file_hash
from ..persistence.durable_io import read_json,atomic_write_json,append_jsonl
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..governance.token_accounting import audit_journal,serialized_request
from ..search.schemas import SearchContractError
from .single_prompt_continuation import replay_key

MANIFEST='experiments/manifests/single_prompt_largest_pattern_v1.json'
PROTOCOL='docs/design/SINGLE_PROMPT_LARGEST_PATTERN_V1.md'
PREP_SCHEMA='single_prompt_largest_pattern_prep_v1'
STUDY='single_prompt_largest_pattern_v1'
PARENT_SOURCE='fd27547f3044903a496d28289aefd25f165a7b5c'


def contract():
    return dict(identity='GATED_LARGEST_SUPPORT_PATTERN_FOCUS_V1',seeds=[88,89],
        focus='one_pattern_max_unique_support_count',tie='sorted_pattern_ids_seed_round_sha256_random_choice',
        tie_namespace='largest_pattern_tie_v1',evidence='selected_support_errors_gradients_diagnostics_only',
        no_pattern='raw_original_batch_structured_feedback_none_exact_B',
        auxiliary_audits='retain_only_when_entire_batch_is_selected_else_strip_from_optimizer',
        optimizer_wire_template='unchanged_gated_generator',rounds=20,
        parent_source_sha=PARENT_SOURCE,controls='reference_all_A_B_results_no_new_control_calls',
        continuation='only_contiguous_byte_identical_projected_feedback_prefix',
        response_cache='exact_request_seed_round_role_stage_parent_C_then_B_then_A',
        solver_cache='exact_frozen_request_identity_and_membership_only',
        test_cache='separate_final_phase_only_exact_parent_test_request_identity',
        additional_train_semantic_ceiling=42640,additional_test_semantic_ceiling=2400,
        cumulative_token_ceiling=None,network='unchanged_unbounded_accounted_typed_recovery')


def manifest_template():
    return dict(schema_version='single_prompt_largest_pattern_manifest_v1',experiment_id=STUDY,
        parent_manifest='experiments/manifests/single_prompt_gated_two_seed32_v1.json',
        method_amendment=contract(),protocol_path=PROTOCOL,real_api_authorized=False)


def validate_manifest(value):
    if value!=manifest_template():raise SearchContractError('LARGEST_PATTERN_MANIFEST_MISMATCH')


def enabled(scope):
    if 'method_amendment' not in scope:return False
    if scope['method_amendment']!=contract():raise SearchContractError('LARGEST_PATTERN_METHOD_MISMATCH')
    return True


def project_feedback(feedback,seed,ordinal):
    if seed not in contract()['seeds'] or not 1<=ordinal<=20:raise SearchContractError('LARGEST_PATTERN_SELECTION_SCOPE')
    full=deepcopy(feedback);sf=full['structured_feedback'];errors=full['errors']
    if not isinstance(sf,dict):raise SearchContractError('LARGEST_PATTERN_STRUCTURE_REQUIRED')
    if set(sf)-{'gradients','diagnostics','patterns','unassigned_ids','partition_audit','cluster_structural_errors'}:raise SearchContractError('LARGEST_PATTERN_UNKNOWN_FEEDBACK_CHANNEL')
    batch={e['example_id'] for e in errors};patterns=sf['patterns'];grad_ids={g['example_id'] for g in sf['gradients']}
    if len(batch)!=len(errors) or len(grad_ids)!=len(sf['gradients']):raise SearchContractError('LARGEST_PATTERN_DUPLICATE_EVIDENCE')
    ids=[p['pattern_id'] for p in patterns]
    if len(set(ids))!=len(ids):raise SearchContractError('LARGEST_PATTERN_DUPLICATE_ID')
    supports={p['pattern_id']:set(p['support_ids']) for p in patterns}
    if any(not support or not support<=batch&grad_ids for support in supports.values()):raise SearchContractError('LARGEST_PATTERN_SUPPORT_MISMATCH')
    audit=dict(identity=contract()['identity'],seed=seed,round=ordinal,
        pattern_sizes={p:len(v) for p,v in sorted(supports.items())},selected_pattern_id=None,
        selected_support_ids=[],tie_ids=[],fallback_to_B=not bool(patterns))
    if not patterns:return dict(errors=errors,structured_feedback=None),audit
    largest=max(map(len,supports.values()));ties=sorted(p for p,v in supports.items() if len(v)==largest)
    key=digest([seed,ordinal,contract()['tie_namespace']]);chosen=random.Random(int(key,16)).choice(ties)
    support=supports[chosen];selected=next(p for p in patterns if p['pattern_id']==chosen)
    audit.update(selected_pattern_id=chosen,selected_support_ids=sorted(support),tie_ids=ties)
    # Keep byte-identical full-support single-pattern inputs when no excluded
    # evidence exists. Other inputs cannot carry whole-Batch audit channels.
    complete=(len(patterns)==1 and support==batch==grad_ids
        and not sf.get('unassigned_ids') and not sf.get('cluster_structural_errors')
        and all(d.get('example_id') in support for d in sf.get('diagnostics',[]))
        and all(not row.get('discarded_patterns') for row in sf.get('partition_audit',[])))
    if not complete:
        sf=dict(sf,patterns=[selected],gradients=[g for g in sf['gradients'] if g['example_id'] in support],
            diagnostics=[d for d in sf.get('diagnostics',[]) if d.get('example_id') in support],
            unassigned_ids=[],partition_audit=[],cluster_structural_errors=[])
    return dict(errors=[e for e in errors if e['example_id'] in support],structured_feedback=sf),audit


class ExactResponseLibrary:
    """A counterfactual cache miss dispatches new work; never coerces old input."""
    def __init__(self,entries,path):
        self.entries={};self.path=Path(path);self.hits=0
        priority={'gradient_pattern':0,'batch_errors':1,'full_errors':2}
        for e in sorted(entries,key=lambda e:(priority[e['arm']],e['source_scope_identity'],int(e['reservation_id']))):
            self.entries.setdefault(e['key'],e)
    def take(self,key,guard):
        entry=self.entries.get(key)
        if entry is None:return None
        append_jsonl(self.path,dict(key=key,source_scope_identity=entry['source_scope_identity'],
            source_reservation_id=entry['reservation_id'],source_receipt_sha256=entry['source_receipt_sha256']))
        self.hits+=1;return deepcopy(entry['response'])
    def require_consumed(self):pass  # Unmatched counterfactual requests are not required.


def audit_parent(root,directory,*,expected_inventory=None):
    from . import gated_prompt_governance as gov,gated_prompt_test as final,gated_prompt_execution as ex
    directory=gov.shared.private_path(root,directory);scope=read_json(directory/'scope.json')
    if (scope['source_sha']!=PARENT_SOURCE or scope['scope_identity']!=digest({k:v for k,v in scope.items() if k!='scope_identity'})
            or scope['policy']!=gov.policy() or scope['execution_amendment']!=ex.contract()):
        raise SearchContractError('LARGEST_PATTERN_PARENT_IDENTITY')
    actual=gov.shared.inventory(directory)
    if expected_inventory is not None and actual!=expected_inventory:raise SearchContractError('LARGEST_PATTERN_PARENT_CHANGED')
    results,terminal_sha=final.verify_training(directory,scope)
    auth=read_json(directory/'authorization.json')
    if (auth!=read_json(root/scope['execution']['prep_path']/'authorization.json')
            or auth['approved'] is not True or auth['adaptive_validation_access'] is not True or auth['test_access'] is not True
            or auth['scope_identity']!=scope['scope_identity']):raise SearchContractError('LARGEST_PATTERN_PARENT_AUTHORIZATION')
    if read_json(root/'runs/gated_prompt_consumed'/(scope['scope_identity']+'.json'))!=dict(scope_identity=scope['scope_identity'],run_root=directory.relative_to(root).as_posix()):
        raise SearchContractError('LARGEST_PATTERN_PARENT_CONSUMPTION')
    progress={};prefixes={};observations={}
    for c,result in zip(scope['cells'],results):
        folder=directory/c['cell_id'];checkpoints=[read_json(folder/f'checkpoint_{n}.json') for n in range(21)]
        for cp in checkpoints:
            for field in ('optimize','validation','candidate_optimize','candidate_validation'):
                for value in (cp.get(field) or {}).values():
                    key=value['request_sha256']
                    if key in observations and observations[key]!=value:raise SearchContractError('LARGEST_PATTERN_OBSERVATION_CONFLICT')
                    observations[key]=value
        if c['arm']!='gradient_pattern':continue
        count=0
        for n in range(1,21):
            old=read_json(folder/f'round_{n}_opportunity.json')['feedback'];new,_=project_feedback(old,c['seed'],n)
            if json.dumps(new,sort_keys=True,ensure_ascii=True)!=json.dumps(old,sort_keys=True,ensure_ascii=True):break
            count=n
        prefixes[c['cell_id']]=count
        usage={}
        for m in result['metrics'][:count]:
            for k,v in m['cost'].items():usage[k]=usage.get(k,0)+v
        progress[c['cell_id']]=dict(checkpoints=checkpoints[:count+1],round_costs={},historical_usage=usage)
    view,events=audit_journal(directory/'ledger',task_sha256=scope['scope_identity'],policy=ex.ledger_policy(scope))
    reserves={e['reservation_id']:e for e in events if e['kind']=='RESERVE'};cells={c['attempt_id']:c for c in scope['cells']};entries=[]
    for event in events:
        if event['kind']!='CHARGE':continue
        rid=event['reservation_id'];reserve=reserves[rid]
        if reserve['role']=='solver':continue  # Complete scored observations already cover Solver work.
        store=ProviderResponseReceipts.__new__(ProviderResponseReceipts);store.directory=directory/'provider_receipts';store.attempt_id=reserve['attempt_id'];store.startup=scope['scope_identity'];record=store.read(rid)
        if 'error_category' in record:continue
        logical=record['logical_identity'];cell=cells[reserve['attempt_id']]
        if (logical!=dict(seed=cell['seed'],ordinal=logical['ordinal'],role=reserve['role'],stage=reserve['stage'])
                or hashlib.sha256(serialized_request(record['request'])).hexdigest()!=reserve['request_sha256']
                or read_json(store.directory/(rid+'.json'))['integrity_sha256']!=event['response_receipt']['integrity_sha256']):
            raise SearchContractError('LARGEST_PATTERN_PARENT_WIRE_MISMATCH')
        entries.append(dict(key=replay_key(record['request'],cell['seed'],logical['ordinal'],reserve['role'],reserve['stage']),
            response=record['response'],source_scope_identity=scope['scope_identity'],reservation_id=rid,
            source_receipt_sha256=event['response_receipt']['integrity_sha256'],arm=cell['arm'],request=record['request'],
            seed=cell['seed'],ordinal=logical['ordinal'],role=reserve['role'],stage=reserve['stage']))
    # Parent Test is only sealed metadata here. Prediction rows open in the new final phase.
    test=directory.parent/'test_run';test_scope=read_json(test/'scope.json');test_terminal=read_json(test/'terminal.json')
    test_inventory=gov.shared.inventory(test);test_inventory.pop('terminal.json',None)
    if (test_terminal['status']!='EXECUTION_COMPLETE' or test_terminal['artifacts']!=test_inventory
            or test_scope['parent_terminal_sha256']!=terminal_sha or test_scope['parent_scope_identity']!=scope['scope_identity']
            or test_scope['policy']!=scope['test_plan'] or test_scope['scope_identity']!=digest({k:v for k,v in test_scope.items() if k!='scope_identity'})):
        raise SearchContractError('LARGEST_PATTERN_PARENT_TEST_SEAL')
    inherited=read_json(directory/'execution_summary.json')['usage'];test_usage=read_json(test/'execution_summary.json')['usage']
    frozen=dict(parent_run_root=directory.relative_to(root).as_posix(),parent_inventory=actual,
        parent_scope_identity=scope['scope_identity'],parent_source_sha=scope['source_sha'],parent_terminal_sha256=terminal_sha,
        parent_test_terminal_sha256=file_hash(test/'terminal.json'),parent_test_inventory=gov.shared.inventory(test),
        request_identity_scope=scope['continuation']['request_identity_scope'],inherited_usage=inherited,
        inherited_test_usage=test_usage,parent_test_scope_identity=test_scope['scope_identity'],
        compatible_prefixes=prefixes,observation_cache_sha256=digest(observations),wire_library_sha256=digest(entries),
        operation_policy=scope['continuation']['operation_policy'])
    return dict(**frozen,progress=progress,observations=observations,entries=entries)


def frozen_parent(root,directory,**kwargs):
    return {k:v for k,v in audit_parent(root,directory,**kwargs).items() if k not in ('progress','observations','entries')}


def initialize_resume(root,run,scope,broker):
    parent=scope['continuation'];a=audit_parent(root,root/parent['parent_run_root'],expected_inventory=parent['parent_inventory'])
    if {k:v for k,v in a.items() if k not in ('progress','observations','entries')}!=parent:raise SearchContractError('LARGEST_PATTERN_FROZEN_PARENT_CHANGED')
    broker.cache=deepcopy(a['observations']);broker.replay=ExactResponseLibrary(a['entries'],run/'response_reuse.jsonl')
    atomic_write_json(run/'parent_accounting.json',parent)
    return a['progress']


def import_cell(root,run,scope,cell):
    parent=root/scope['continuation']['parent_run_root']/cell['cell_id'];target=run/cell['cell_id'];target.mkdir()
    prefix=scope['continuation']['compatible_prefixes'][cell['cell_id']]
    for n in range(prefix+1):shutil.copyfile(parent/f'checkpoint_{n}.json',target/f'checkpoint_{n}.json')
    for n in range(1,prefix+1):
        for p in parent.glob(f'round_{n}_*.json'):shutil.copyfile(p,target/p.name)
        _,audit=project_feedback(read_json(parent/f'round_{n}_opportunity.json')['feedback'],cell['seed'],n)
        atomic_write_json(target/f'round_{n}_focus.json',audit)
    atomic_write_json(target/'prefix_certificate.json',dict(parent_scope_identity=scope['continuation']['parent_scope_identity'],completed_prefix=prefix,feedback_wire_identical=True))
    return target
