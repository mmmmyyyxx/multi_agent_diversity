"""Explicitly authorized, immutable pre-cluster prefix continuation.

This is not general checkpoint resume. No completed opportunity, adaptive
Memory or selected candidate can be imported. Model outputs are replayed
through current strict validators; actual parent charges remain attributable.
"""
from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import json
import threading

from ..persistence.durable_io import read_json, atomic_write_json
from ..persistence.exact_output_cache import DurableExactOutputCache
from ..search.schemas import SearchContractError
from .startup_identity import canonical_sha256
from .token_accounting import audit_journal, serialized_request

IDENTITY='AUTHORIZED_V27_PRECLUSTER_PREFIX_CONTINUATION_V1'
POLICY_FIELDS={'identity','parent_prep_path','parent_run_path','parent_inventory_sha256',
    'parent_charged_tokens','parent_attempt_id','parent_source_sha','user_instruction_sha256'}


def require_policy(value):
    if value is None:return
    if (not isinstance(value,dict) or set(value)!=POLICY_FIELDS or value['identity']!=IDENTITY
            or type(value['parent_charged_tokens']) is not int or not 0<value['parent_charged_tokens']<30_000_000
            or any(not isinstance(value[k],str) or len(value[k])!=64 or any(c not in '0123456789abcdef' for c in value[k])
                for k in ('parent_inventory_sha256','user_instruction_sha256'))
            or not isinstance(value['parent_source_sha'],str) or len(value['parent_source_sha'])!=40
            or not isinstance(value['parent_attempt_id'],str) or not value['parent_attempt_id']
            or any(not isinstance(value[k],str) or not value[k].startswith('runs/') for k in ('parent_prep_path','parent_run_path'))):
        raise SearchContractError('PRECLUSTER_CONTINUATION_POLICY_INVALID')


def _path(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to((root/'runs').resolve()):raise SearchContractError('CONTINUATION_PATH_ESCAPE')
    return p


def _lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]
def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def _key(role,split,stage,request):return canonical_sha256([role,split,stage,request])


def compatibility(old,new):
    """Everything that determines initial scores/search evidence remains fixed."""
    from ..versions import UNIFIED_OUTPUT_CONTRACT_FEEDBACK_VERSION,UNIFIED_REQUEST_RECOVERY_VERSION
    if old.get('method_identity')!=UNIFIED_OUTPUT_CONTRACT_FEEDBACK_VERSION or new.get('method_identity')!=UNIFIED_REQUEST_RECOVERY_VERSION:
        raise SearchContractError('CONTINUATION_METHOD_PAIR_NOT_FROZEN')
    fixed=('execution_phase','execution_seed','models','provider','provider_routing_policy',
        'decoding','solver_decoding_policy','solver_execution_policy','initial_team_version','initial_team_sha256',
        'initial_team_artifact_sha256','system_prompt_policy','answer_extraction_policy','solver_output_interface',
        'solver_trajectory_policy','prediction_validity_policy','invalid_recovery_policy','cache_policy',
        'repair_probe_policy','optimizer_generation_policy',
        'optimizer_nonthinking_evidence_policy','pattern_cluster_generation_policy',
        'gradient_prompt_sha256','pattern_prompt_sha256','pattern_support_id_transport','pattern_abstraction_guard',
        'partition_completion_policy','transition_policy','layer1_search_policy','memory_policy_identity','memory_limits',
        'shared_risk_policy','panel_evidence_policy','optimizer_input_schema','initial_competence_binding',
        'low_cost_protocol','low_cost_subsets_sha256','canonical_manifest_sha256','split_manifest_sha256',
        'benchmark_protocol_sha256','token_reporting_policy','heldout_accounting_reserve')
    if (old.get('execution_phase')!='formal' or old.get('execution_attempt_id')==new.get('execution_attempt_id')
            or any(old.get(k)!=new.get(k) for k in fixed)):
        raise SearchContractError('CONTINUATION_SCIENTIFIC_COMPATIBILITY_MISMATCH')
    # Only generated membership rejection classification is amended. Every
    # other Pattern field, including source-context safety, stays identical.
    def pattern(c):
        value=deepcopy(c['pattern_policy']);value['gradient_policy']['output_recovery'].pop('identity',None)
        value['optimization_evidence_policy'].pop('identity',None)
        value['optimization_evidence_policy'].pop('generated_output_recovery',None)
        return value
    if pattern(old)!=pattern(new):raise SearchContractError('CONTINUATION_PATTERN_COMPATIBILITY_MISMATCH')
    a=deepcopy(old['operational_pilot']);b=deepcopy(new['operational_pilot'])
    for value in (a,b):value.pop('identity');value.pop('token_ceiling')
    if a!=b or new['operational_pilot']['token_ceiling'] is not None or new['scientific_no_commit_patience']!=4:
        raise SearchContractError('CONTINUATION_USER_SCOPE_AMENDMENT_MISMATCH')
    a=deepcopy(old['provider_bounds']);b=deepcopy(new['provider_bounds'])
    for value in (a,b):
        for key in ('attempt_charged_token_ceiling','cumulative_charged_token_ceiling','reservation_peak_upper_bound'):value.pop(key)
        value['bound_proof'].pop('token_bound')
    if a!=b:raise SearchContractError('CONTINUATION_CALL_BOUND_MISMATCH')


class PreclusterContinuation:
    def __init__(self,root,contract,run_root):
        self.policy=contract['precluster_continuation_policy'];require_policy(self.policy)
        self.contract=contract;self.run_root=run_root;self.lock=threading.RLock()
        self.parent=_path(root,self.policy['parent_run_path']);self.prep=_path(root,self.policy['parent_prep_path'])
        payload=read_json(self.prep/'prep.json')
        if (canonical_sha256({k:v for k,v in payload.items() if k!='startup_identity_sha256'})!=payload['startup_identity_sha256']
                or payload['scope']['source_sha']!=self.policy['parent_source_sha']
                or payload['scope']['attempt_id']!=self.policy['parent_attempt_id']
                or _sha(self.parent/'raw_evidence_inventory.json')!=self.policy['parent_inventory_sha256']):
            raise SearchContractError('CONTINUATION_PARENT_IDENTITY_MISMATCH')
        from .unified_execution import verify_source_commit
        verify_source_commit(root,self.policy['parent_source_sha'],payload['source_identity'])
        reference=payload['manifest']['execution_binding'];old=read_json(root/reference['path'])
        if _sha(root/reference['path'])!=reference['sha256']:raise SearchContractError('CONTINUATION_PARENT_BINDING_MISMATCH')
        compatibility(old,contract)
        life=read_json(self.parent/'lifecycle.json')
        if (life.get('status')!='EXECUTION_ABORTED' or life.get('stop_category')!='PATTERN_GRADIENT_CLUSTER_INVALID_MEMBERSHIP'
                or (self.parent/'trajectory_private.jsonl').read_bytes()!=b''):
            raise SearchContractError('CONTINUATION_NOT_PRECLUSTER_PREFIX')
        inventory=read_json(self.parent/'raw_evidence_inventory.json')['files']
        actual={p.relative_to(self.parent).as_posix() for p in self.parent.rglob('*') if p.is_file() and p.name!='raw_evidence_inventory.json'}
        if {r['path'] for r in inventory}!=actual:raise SearchContractError('CONTINUATION_PARENT_INVENTORY_MISMATCH')
        for row in inventory:
            path=(self.parent/row['path']).resolve()
            if not path.is_relative_to(self.parent) or path.stat().st_size!=row['bytes'] or _sha(path)!=row['sha256']:
                raise SearchContractError('CONTINUATION_PARENT_EVIDENCE_MISMATCH')
        from .autonomous_math import ledger_policy
        view,events=audit_journal(root/old['token_ledger_directory'],task_sha256=old['task_authorization_sha256'],policy=ledger_policy(old),cost_policy=old.get('cost_policy'))
        if (view!=read_json(self.parent/'accounting_end.json') or view['reserved_inflight']!=0
                or view['charged_total']!=self.policy['parent_charged_tokens']):
            raise SearchContractError('CONTINUATION_PARENT_LEDGER_MISMATCH')
        authorization=read_json(self.parent/'consumed_authorization.json')
        if authorization.get('consumed') is not True or authorization.get('scope')!=payload['scope']:
            raise SearchContractError('CONTINUATION_PARENT_AUTHORIZATION_MISMATCH')
        self.parent_usage=life['provider_usage'];self.initial=read_json(self.parent/'initial_state_private.json')
        self.trace=_lines(self.parent/'provider_trace_private.jsonl')
        if (len(self.trace)!=self.parent_usage['attempts'] or sum(r['role']=='pattern_cluster' for r in self.trace)!=1
                or any(r['split']!='optimize' or r['role'] not in {'solver','pattern_gradient','pattern_cluster'} for r in self.trace)):
            raise SearchContractError('CONTINUATION_PREFIX_ACCESS_MISMATCH')
        self.pending=defaultdict(deque);self.imported=0;self.imported_by_role=defaultdict(int)
        for row in self.trace:
            if row['role']=='solver':continue
            if 'response' not in row:raise SearchContractError('CONTINUATION_OPTIMIZER_TRANSPORT_PREFIX_UNSUPPORTED')
            self.pending[_key(row['role'],row['split'],row['stage'],row['request'])].append(row)
        self.remaining=sum(map(len,self.pending.values()))
        self.old_cache=DurableExactOutputCache.__new__(DurableExactOutputCache)
        self.old_cache.directory=self.parent/'resolved_output_cache';self.old_cache.context=read_json(self.old_cache.directory/'scope.json')
        if self.old_cache.context['execution_attempt_id']!=self.policy['parent_attempt_id']:
            raise SearchContractError('CONTINUATION_PARENT_CACHE_MISMATCH')

    def bind(self,broker):
        """Seed only exact initial Solver lanes; preserve old provenance keys."""
        from ..search.system_prompt import SystemPrompt,solver_messages
        for member,profiles in enumerate(self.initial['diagnostics']['raw_profiles']):
            prompt=SystemPrompt.from_dict(self.initial['member_prompts'][member])
            for profile,state in zip(profiles,self.initial['diagnostics']['team_states'],strict=True):
                source=profile['solver_trajectory']['source'];old_key=source['request_sha256']
                result=self.old_cache.get(old_key)
                if result is None or result['resolved_prediction']!=profile['prediction'] or source['member_id']!=member:
                    raise SearchContractError('CONTINUATION_INITIAL_CACHE_PROFILE_MISMATCH')
                original=next(r for r in self.trace if r['role']=='solver' and r['request_sha256']==old_key)
                problem=original['request']['messages'][1]['content']
                if hashlib.sha256(problem.encode()).hexdigest()!=source['benchmark_input_sha256']:
                    raise SearchContractError('CONTINUATION_INITIAL_PROVENANCE_MISMATCH')
                request,new_key=broker._request_identity(role='solver',split='optimize',messages=solver_messages(prompt,problem),member_slot=member)
                if request['messages']!=original['request']['messages']:
                    raise SearchContractError('CONTINUATION_INITIAL_WIRE_MISMATCH')
                broker.durable_cache.put(old_key,result)
                rebound=deepcopy(result);rebound['request_sha256']=new_key
                broker.durable_cache.put(new_key,rebound)
        atomic_write_json(self.run_root/'precluster_continuation_receipt.json',dict(
            identity=IDENTITY,policy=self.policy,parent_attempts=self.parent_usage['attempts'],
            parent_usage=self.parent_usage,initial_profiles=750,pending_optimizer_responses=self.remaining,
            historical_scores_changed=False,new_api_calls_for_import=0))
        (self.run_root/'imported_provider_trace_private.jsonl').write_bytes((self.parent/'provider_trace_private.jsonl').read_bytes())

    def restore_initial(self,store):
        from .autonomous_math import plain
        store.profiles={m:tuple(profiles) for m,profiles in enumerate(self.initial['diagnostics']['raw_profiles'])}
        snapshot=store.snapshot();store.initial_member_scores=snapshot.member_scores;store.initial_state_id=snapshot.team_state_id
        if plain(store.snapshot())!=self.initial:raise SearchContractError('CONTINUATION_INITIAL_STATE_MISMATCH')

    def take(self,broker,*,role,split,stage,messages,member_slot=None):
        if role not in {'pattern_gradient','pattern_cluster'}:return None
        request,new_key=broker._request_identity(role=role,split=split,messages=messages,member_slot=member_slot)
        with self.lock:
            if not self.remaining:return None
            key=_key(role,split,stage,request)
            if not self.pending[key]:raise SearchContractError('CONTINUATION_OPTIMIZER_WIRE_MISMATCH')
            record=self.pending[key].popleft();self.remaining-=1;self.imported+=1
            self.imported_by_role[role]+=1
            broker._write(dict(kind='CONTINUATION_IMPORTED_RESPONSE',role=role,split=split,stage=stage,
                parent_attempt_id=self.policy['parent_attempt_id'],parent_physical_attempt_no=record['physical_attempt_no'],
                parent_request_sha256=record['request_sha256'],request_sha256=new_key,
                parent_charged_tokens_included=True,new_api_call=False))
            if not self.remaining:
                atomic_write_json(self.run_root/'precluster_continuation_drained.json',dict(
                    identity=IDENTITY,parent_attempt_id=self.policy['parent_attempt_id'],
                    imported_optimizer_responses=self.imported,pending_optimizer_responses=0,
                    strict_current_validators_required=True,new_api_calls_for_import=0))
            return {**deepcopy(record['response']),'provider_called':False,'request_sha256':new_key}
