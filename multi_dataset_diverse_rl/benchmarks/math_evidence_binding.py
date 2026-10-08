"""Fresh V2.3 preparation; historical bindings never grant eligibility or API access."""
from copy import deepcopy
from dataclasses import asdict
import json
from ..search.current_layer1 import CurrentOptimizer
from ..search.current_composition import build_current_team_prompt_search
from ..search.scientific_aggregation import EquivalencePluralityAggregation
from .experiment_splits import ExperimentSplitReader
from .access import DataPurpose
from .protocols import protocol_input,PROTOCOLS
from .math_v21_interface import MATHV21BenchmarkAdapter
import hashlib

from ..search.current_layer1 import EvidenceLayer1Config
from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.optimization_evidence import POLICY, BINDING, METHOD, MEMORY, EVIDENCE, INPUT
from ..search.schemas import SearchContractError
from ..search.textual_gradients import REFERENCE_GRADIENT_PROMPT, pattern_policy_for_trajectory
from .data_freeze import file_hash, digest


def evidence_gradient_prompt_artifact():
    return dict(identity='REFERENCE_SOLUTION_GRADIENT_PROMPT_V5',prompt=REFERENCE_GRADIENT_PROMPT)


def prepare_v23_only_inputs(root,*,user_task_sha256):
    """Create one fresh preparation scope, without credentials or providers."""
    from ..persistence.durable_io import atomic_write_json
    from ..governance.token_accounting import POLICY_V23_2M
    from ..governance.canary_review import POLICY as canary_policy
    if not isinstance(user_task_sha256,str) or len(user_task_sha256)!=64:
        raise SearchContractError('V23_ONLY_USER_TASK_HASH_REQUIRED')
    attempt='a4_v23_only_seed81_20261008_attempt1'
    parent_path='experiments/execution_bindings/a4_v23_matched_seed81_20261008_attempt1_b.json'
    parent=json.loads((root/parent_path).read_bytes())
    if parent.get('optimization_evidence_policy')!=POLICY or parent.get('method_identity')!=METHOD:
        raise SearchContractError('V23_ONLY_PARENT_SCIENTIFIC_SETTINGS_MISMATCH')
    protocol_dir='experiments/protocols/a4_v23_only_pilot_v1'
    binding_path='experiments/execution_bindings/'+attempt+'.json'
    scope_path=protocol_dir+'/preparation_scope.json'
    gradient_path=protocol_dir+'/gradient.json'
    metadata_path=protocol_dir+'/v6_accounting_metadata.json'
    accounting_path=protocol_dir+'/accounting_policy.json'
    if (root/binding_path).exists():
        raise SearchContractError('V23_ONLY_FRESH_PREPARATION_PATH_REQUIRED')
    def write_input(path,value):
        if (root/path).exists():
            if json.loads((root/path).read_bytes())!=value:
                raise SearchContractError('V23_ONLY_PREPARATION_INPUT_CHANGED')
        else:atomic_write_json(root/path,value)
    parent_hash=file_hash(root/parent_path)
    write_input(scope_path,dict(schema_version='math_optimization_evidence_user_scope_v1',
        attempt_id=attempt,one_attempt_only=True,user_authorized=True,real_api_authorized=False,
        optimization_evidence_policy=POLICY,parent_binding_sha256=parent_hash,
        user_task_sha256=user_task_sha256,validation_authorized=False,test_authorized=False,
        preparation_only=True,exact_api_approval_required=True,push_authorized=True,
        cancelled_paired_scope_reusable=False,historical_40m_authorization_reusable=False))
    write_input(gradient_path,evidence_gradient_prompt_artifact())
    # This frozen V6 byte-length metadata is copied, never recomputed from
    # held-out questions. It grants no held-out execution or future reserve.
    metadata=(root/parent['validation_accounting_metadata_path']).read_bytes()
    if (root/metadata_path).exists():
        if (root/metadata_path).read_bytes()!=metadata:
            raise SearchContractError('V23_ONLY_PREPARATION_INPUT_CHANGED')
    else:(root/metadata_path).write_bytes(metadata)
    write_input(accounting_path,POLICY_V23_2M)
    c=derive_evidence_contract(parent,attempt=attempt,binding_path=binding_path,
        parent_path=parent_path,parent_sha256=parent_hash,
        authorization_path=scope_path,authorization_sha256=file_hash(root/scope_path),
        gradient_prompt_path=gradient_path,gradient_prompt_sha256=file_hash(root/gradient_path),
        validation_metadata_path=metadata_path,validation_metadata_sha256=file_hash(root/metadata_path),
        max_opportunities=5,token_ceiling=2_000_000,
        accounting_scope='runs/'+attempt+'/accounting',
        accounting_policy_path=accounting_path,accounting_policy_sha256=file_hash(root/accounting_path))
    blockers=MATHEvidenceBinding(root,c).blockers()
    if blockers:raise SearchContractError('HOLD_PRE_PROVIDER:'+','.join(blockers))
    atomic_write_json(root/binding_path,c)
    atomic_write_json(root/(protocol_dir+'/protocol.json'),dict(
        identity='V23_ONLY_CONTINUOUS_CANARY_PILOT_SCOPE_V1',attempt_id=attempt,
        method=METHOD,seed=81,members=5,optimize=60,shadow=40,
        canary_review_policy=canary_policy,max_opportunities_total=5,
        max_generations_per_opportunity=6,max_exported_candidates=4,max_full_candidates=2,
        charged_token_ceiling=2_000_000,inflight_reservations_count_against_ceiling=True,
        validation_calls=0,test_calls=0,experimental_retry=False,
        scientific_stopper=c['stop_policy'],efficacy_based_stopping=False,
        freeze_api_approval_required=True,source_reference='7342d85a877880717b00cd19ddc4ab6e598e81e4',
        parent_binding_sha256=parent_hash,binding_sha256=file_hash(root/binding_path),
        accounting_scope_policy=c['accounting_scope_policy'],heldout_accounting_reserve=0))
    return c


def derive_evidence_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authorization_path, authorization_sha256, gradient_prompt_path,
        gradient_prompt_sha256, validation_metadata_path, validation_metadata_sha256,
        paired_realization_policy=None, max_opportunities=None, token_ceiling=None,
        accounting_scope=None, accounting_policy_path=None, accounting_policy_sha256=None):
    """Frozen parent is data provenance; it never selects an executable method."""
    from .math_v21_interface import v6_interface_contract
    from .math_visible_trajectory import trajectory_policy
    if (paired_realization_policy is not None or not isinstance(attempt,str) or not attempt
            or attempt in {parent.get('execution_attempt_id'),parent.get('cache_namespace')}
            or any(new==parent.get(key) for new,key in (
                (binding_path,'binding_path'),(authorization_path,'current_user_scope_path'),
                (gradient_prompt_path,'gradient_prompt_path'),
                (validation_metadata_path,'validation_accounting_metadata_path')))
            or any(not isinstance(h,str) or len(h)!=64 for h in
                (parent_sha256,authorization_sha256,gradient_prompt_sha256,validation_metadata_sha256))):
        raise SearchContractError('MATH_EVIDENCE_FRESH_FREEZE_REQUIRED')
    c=deepcopy(parent)
    c.pop('paired_realization_policy',None)
    c.update(execution_attempt_id=attempt,cache_namespace=attempt,binding_path=binding_path,
        solver_output_interface=v6_interface_contract(),solver_trajectory_policy=trajectory_policy(),
        gradient_prompt_path=gradient_prompt_path,gradient_prompt_sha256=gradient_prompt_sha256,
        trajectory_parent_binding_path=parent_path,trajectory_parent_binding_sha256=parent_sha256,
        current_user_scope_path=authorization_path,current_user_scope_sha256=authorization_sha256,
        continuation_authorization_sha256=authorization_sha256,
        validation_accounting_metadata_path=validation_metadata_path,
        validation_accounting_metadata_sha256=validation_metadata_sha256)
    if max_opportunities is not None or token_ceiling is not None:
        if (type(max_opportunities) is not int or not 0<max_opportunities<=5
                or type(token_ceiling) is not int or not 0<token_ceiling<=2_000_000):
            raise SearchContractError('V23_ONLY_OPERATIONAL_SCOPE_INVALID')
        c['execution_phase']='pilot'
        c['operational_pilot'].update(identity='TARGET_OR_TEAM_PROGRESS_OPERATIONAL_PILOT_BOUND_V1',
            max_opportunities=max_opportunities,token_ceiling=token_ceiling)
        c['provider_bounds']['max_opportunities']=max_opportunities
        if (accounting_scope is None or not accounting_policy_path
                or not isinstance(accounting_policy_sha256,str) or len(accounting_policy_sha256)!=64):
            raise SearchContractError('FRESH_V23_ACCOUNTING_SCOPE_REQUIRED')
        from ..governance.canary_review import POLICY as canary_policy
        c.update(accounting_policy_path=accounting_policy_path,
            accounting_policy_sha256=accounting_policy_sha256,
            accounting_scope_policy='FRESH_V23_SINGLE_ARM_2M_V1',heldout_accounting_reserve=0,
            task_authorization_sha256=authorization_sha256,
            canary_review_policy=deepcopy(canary_policy),
            canary_phase='continuous_initial_profile_and_first_opportunity_review_v1',
            arms={'A4':[True,True]},execution_arm='A4')
        c['budget']['metric_calls']=POLICY['metric_limit']
        # The complete parent receipt remains hash-pinned above. Its obsolete
        # authorizations and operational ancestry cannot grant execution scope.
        keep={'current_user_scope_path','current_user_scope_sha256',
            'task_authorization_sha256','continuation_authorization_sha256',
            'trajectory_parent_binding_path','trajectory_parent_binding_sha256'}
        for key in tuple(c):
            if key not in keep and ('user_scope' in key or 'amendment' in key
                    or 'parent_binding' in key or key.startswith('baseline_scientific_settings')
                    or key.startswith('gradient_recovery_')):
                c.pop(key)
    if accounting_scope is not None:
        if not isinstance(accounting_scope,str) or not accounting_scope.startswith('runs/'):
            raise SearchContractError('V23_ONLY_ACCOUNTING_SCOPE_INVALID')
        c['token_ledger_directory']=accounting_scope
    c.update(identity=BINDING,method_identity=METHOD,optimization_evidence_policy=deepcopy(POLICY),
        optimizer_input_schema=INPUT,layer1_search_policy=asdict(EvidenceLayer1Config()),
        memory_policy_identity=MEMORY,panel_evidence_policy=EVIDENCE,
        pattern_policy=pattern_policy_for_trajectory(c['solver_trajectory_policy'],POLICY))
    # One accepted independent diagnostic per wrong example. UNCERTAIN is valid,
    # not a contract retry or semantic regeneration.
    c.pop('gradient_recovery_policy',None)
    for key in tuple(c):
        if key.startswith('gradient_recovery_amendment_'):c.pop(key)
    bound=deepcopy(c['provider_bounds']);k=c['operational_pilot']['max_opportunities']
    proof=bound['bound_proof'];bootstrap=proof['bootstrap_logical_solver_calls']
    multiplier=proof['solver_semantic_multiplier'];transport=proof['transport_multiplier']
    per_op=dict(local=42,probe=12,full=120,shadow=40)
    logical=sum(per_op.values());solver=multiplier*(bootstrap+k*logical)
    gradient=60*k;cluster=k;reflection=6*k;successful=solver+gradient+cluster+reflection
    bound.update(solver_per_opportunity=logical,physical_solver_per_opportunity=multiplier*logical,
        solver_calls=solver,pattern_gradient_calls=gradient,pattern_cluster_calls=cluster,
        pattern_calls=gradient+cluster,reflection_calls=reflection,
        successful_provider_calls=successful,transport_attempts=successful*transport)
    proof.update(identity='V2_3_OPTIMIZE_EVIDENCE_RESOURCE_DERIVATION_V1',
        per_op_logical_solver=per_op,gradient_semantic_multiplier=1)
    c['provider_bounds']=bound
    return c


class MATHEvidenceBinding:
    def __init__(self,root,contract):
        self.root=root.resolve();self.contract=contract

    def blockers(self):
        c=self.contract
        try:
            parent_path=c['trajectory_parent_binding_path']
            if file_hash(self.path(parent_path))!=c['trajectory_parent_binding_sha256']:
                raise SearchContractError('MATH_EVIDENCE_PARENT_HASH_MISMATCH')
            parent=json.loads(self.path(parent_path).read_bytes())
            expected=derive_evidence_contract(parent,attempt=c['execution_attempt_id'],
                binding_path=c['binding_path'],parent_path=parent_path,parent_sha256=c['trajectory_parent_binding_sha256'],
                authorization_path=c['current_user_scope_path'],authorization_sha256=c['current_user_scope_sha256'],
                gradient_prompt_path=c['gradient_prompt_path'],gradient_prompt_sha256=c['gradient_prompt_sha256'],
                validation_metadata_path=c['validation_accounting_metadata_path'],
                validation_metadata_sha256=c['validation_accounting_metadata_sha256'],
                max_opportunities=(c['operational_pilot']['max_opportunities'] if c['operational_pilot']['identity']=='TARGET_OR_TEAM_PROGRESS_OPERATIONAL_PILOT_BOUND_V1' else None),
                token_ceiling=(c['operational_pilot']['token_ceiling'] if c['operational_pilot']['identity']=='TARGET_OR_TEAM_PROGRESS_OPERATIONAL_PILOT_BOUND_V1' else None),
                accounting_scope=(c['token_ledger_directory'] if c['token_ledger_directory']!=parent['token_ledger_directory'] else None),
                accounting_policy_path=(c['accounting_policy_path'] if c.get('accounting_scope_policy') else None),
                accounting_policy_sha256=(c['accounting_policy_sha256'] if c.get('accounting_scope_policy') else None))
            if c!=expected:raise SearchContractError('MATH_EVIDENCE_FROZEN_SETTINGS_CHANGED')
            for p in sorted(k for k in c if k.endswith('_path')):
                h=p[:-5]+'_sha256'
                if p=='initial_team_path':h='initial_team_artifact_sha256'
                if h not in c:continue
                actual=(digest(
                    json.loads(self.path(c[p]).read_bytes())) if p=='verify_settings_path' else file_hash(self.path(c[p])))
                if actual!=c[h]:raise SearchContractError('MATH_EVIDENCE_DEPENDENCY_HASH_MISMATCH:'+p)
            if c.get('accounting_scope_policy'):
                from ..governance.token_accounting import POLICY_V23_2M
                if json.loads(self.path(c['accounting_policy_path']).read_bytes())!=POLICY_V23_2M:
                    raise SearchContractError('FRESH_V23_ACCOUNTING_POLICY_MISMATCH')
            if json.loads(self.path(c['gradient_prompt_path']).read_bytes())!=evidence_gradient_prompt_artifact():
                raise SearchContractError('MATH_EVIDENCE_GRADIENT_PROMPT_MISMATCH')
            scope=json.loads(self.path(c['current_user_scope_path']).read_bytes())
            required=dict(schema_version='math_optimization_evidence_user_scope_v1',
                attempt_id=c['execution_attempt_id'],one_attempt_only=True,user_authorized=True,
                real_api_authorized=False,optimization_evidence_policy=POLICY,
                parent_binding_sha256=c['trajectory_parent_binding_sha256'],
                validation_authorized=False,test_authorized=False)
            if (any(scope.get(k)!=v for k,v in required.items())
                    or not isinstance(scope.get('user_task_sha256'),str)
                    or len(scope['user_task_sha256'])!=64):
                raise SearchContractError('MATH_EVIDENCE_FRESH_USER_SCOPE_REQUIRED')
            CURRENT_POLICY_BUNDLE.validate_contract(c)
            from .current_math_dependencies import validate_effective_math_dependencies
            validate_effective_math_dependencies(self)
            return ()
        except (SearchContractError,KeyError,TypeError,ValueError,OSError) as exc:
            return (str(exc) if isinstance(exc,SearchContractError) else 'MATH_EVIDENCE_FRESH_FREEZE_REQUIRED',)

    def compose(self,**kwargs):
        blockers=self.blockers()
        if blockers:raise SearchContractError(blockers[0])
        return self._compose(**kwargs)

    def path(self,relative):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root):raise SearchContractError('EXECUTION_BINDING_PATH_ESCAPE')
        return path

    def reader(self):
        c=self.contract
        return ExperimentSplitReader(self.path(c['canonical_root']),self.path(c['split_directory']),'math',
            expected_manifest_sha256=c['split_manifest_sha256'],expected_protocol='benchmark_experiment_split_v2')

    def benchmark(self):return MATHV21BenchmarkAdapter(self.contract)

    def method(self,arm):
        if arm!='A4':raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')
        if self.blockers():raise SearchContractError(self.blockers()[0])
        c=self.contract;CURRENT_POLICY_BUNDLE.validate_contract(c)
        identity=dict(provider=c['provider'],model=c['models']['pattern'],
            gradient_prompt=c['gradient_prompt_sha256'],cluster_prompt=c['pattern_prompt_sha256'],
            generation_policy=c['optimizer_generation_policy'],discovery=c['pattern_policy'],
            support_id_transport=c['pattern_support_id_transport'],abstraction_guard=c['pattern_abstraction_guard'])
        if 'partition_completion_policy' in c:
            identity['partition_completion_policy']=c['partition_completion_policy']
        if 'pattern_cluster_generation_policy' in c:
            identity['pattern_cluster_generation_policy']=c['pattern_cluster_generation_policy']
        binding=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return CURRENT_POLICY_BUNDLE.method(aggregation=c['aggregation'],provider_binding=binding,
            successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
            partition_completion_policy=c.get('partition_completion_policy'),
            pattern_cluster_generation_policy=c.get('pattern_cluster_generation_policy'),
            solver_trajectory_policy=c.get('solver_trajectory_policy'),
            optimization_evidence_policy=c.get('optimization_evidence_policy'))

    def _compose(self,*,arm,seed,solver,reflection,pattern_provider,run_root,optimize_fn=None):
        blockers=self.blockers()
        if blockers or seed not in self.contract['seeds']:
            raise SearchContractError('HOLD_PRE_PROVIDER: '+','.join(blockers or ('SEED_NOT_FROZEN',)))
        if optimize_fn is not None:raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')
        if pattern_provider is None or getattr(pattern_provider,'gradient_provider',None) is None:
            raise SearchContractError('HOLD_PRE_PROVIDER: CURRENT_GRADIENT_PROVIDER_NOT_BOUND')
        if (solver.output_contract_id!=PROTOCOLS['math'].output_contract_id
                or solver.solver_contract_id!='COMMON_SOLVER_CONTRACT_V1'
                or solver.broker.contract!=self.contract or reflection.broker is not solver.broker
                or pattern_provider.broker is not solver.broker
                or pattern_provider.gradient_provider.broker is not solver.broker
                or solver.broker.arm!=arm or solver.broker.seed!=seed):
            raise SearchContractError('MATH_PROVIDER_PORT_BINDING_MISMATCH')
        team=json.loads(self.path(self.contract['initial_team_path']).read_bytes())
        optimizer=CurrentOptimizer(evaluator=solver,reflection_lm=reflection,
            accounting_reader=reflection.accounting,run_root=run_root)
        return build_current_team_prompt_search(benchmark=self.benchmark(),aggregation=EquivalencePluralityAggregation(),
            examples=self.examples('optimize'),prompts=tuple(m['prompt'] for m in team['members']),solver=solver,
            optimizer=optimizer,method=self.method(arm),seed=seed,shadow_loader=lambda:self.examples('shadow'),
            shadow_count=self.contract['shadow_count'],runtime_readiness=self.blockers,pattern_provider=pattern_provider,
            provider_call_reader=lambda:solver.broker.successes,execution_phase=self.contract['execution_phase'],operational_bound=self.contract['operational_pilot'])

    def examples(self,role):
        if role not in {'optimize','shadow'}:raise SearchContractError('HELDOUT_SEARCH_ACCESS_FORBIDDEN')
        from .math_low_cost import read_subsets
        reader=self.reader();c=self.contract
        name='pilot_shadow' if role=='shadow' else 'canary_optimize' if c['execution_phase']=='canary' else 'pilot_optimize'
        members=read_subsets(self.root,c)['memberships'][name]
        wanted={r['stable_example_id'] for r in members}
        reader.members=[r for r in reader.members if r['project_split']!=role or r['stable_example_id'] in wanted]
        reader.manifest=dict(reader.manifest,counts={**reader.manifest['counts'],role:len(members)})
        rows=reader.rows(role,DataPurpose.ADAPTIVE_GATE if role=='shadow' else DataPurpose.EVIDENCE)
        b=self.benchmark()
        from ..search.binary_runtime import CorrectnessExample
        return tuple(CorrectnessExample(protocol_input('math',r['stable_example_id'],r['content'],b.output_contract,
            protocol=b.protocol),r['reference_final_answer'],
            r['content']['solution'] if role=='optimize' and c.get('optimization_evidence_policy') else None,
            tuple((k,str(r['content'][k])) for k in ('type','level'))
                if role=='optimize' and c.get('optimization_evidence_policy') else ()) for r in rows)
