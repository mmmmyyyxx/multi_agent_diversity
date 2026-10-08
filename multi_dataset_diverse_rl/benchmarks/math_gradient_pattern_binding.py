"""Current V2.2 MATH binding; old settings are immutable data, never authority."""
from dataclasses import asdict
import hashlib,json
from .. import current_contract as versions
from ..search.current_layer1 import CurrentOptimizer
from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.current_composition import build_current_team_prompt_search
from ..search.schemas import SearchContractError
from ..search.scientific_aggregation import EquivalencePluralityAggregation
from .experiment_splits import ExperimentSplitReader
from .access import DataPurpose
from .protocols import protocol_input,PROTOCOLS
from .math_v21_interface import MATHV21BenchmarkAdapter
from .current_math_dependencies import validate_effective_math_dependencies
from .data_freeze import file_hash,digest
from .gradient_pilot_contract import derive_current_pilot_contract
BASELINE_PATH='experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v6.json'
BASELINE_SHA='02efd18e8431937f2cf1b196289a36cd0ff8381546fe0f4603f46011b330da22'

class MATHGradientPatternBinding:
    def __init__(self,root,contract):
        if contract.get('identity')!=versions.MATH_V2_2_EXECUTION_BINDING_VERSION:
            raise SearchContractError('CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN')
        self.root=root.resolve();self.contract=contract

    def compose(self,**kwargs):
        blockers=self.blockers()
        if blockers:raise SearchContractError(blockers[0])
        return self._compose(**kwargs)

    def blockers(self):
        c=self.contract
        if 'operational_pilot' not in c:return ('CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN',)
        try:
            if c['baseline_scientific_settings_path']!=BASELINE_PATH or c['baseline_scientific_settings_sha256']!=BASELINE_SHA or file_hash(self.path(BASELINE_PATH))!=BASELINE_SHA:
                raise SearchContractError('CURRENT_BASELINE_SETTINGS_MISMATCH')
            parent=json.loads(self.path(BASELINE_PATH).read_bytes())
            scope_path=self.path(c['current_user_scope_path'])
            if file_hash(scope_path)!=c['current_user_scope_sha256']:raise SearchContractError('CURRENT_USER_SCOPE_HASH_MISMATCH')
            scope=json.loads(scope_path.read_bytes());op=c['operational_pilot']
            cluster_amendment = None
            if 'pattern_cluster_generation_policy' in c:
                from .math_optimizer_generation import frozen_cluster_policy
                policy = frozen_cluster_policy(c)
                for p,h in [('cluster_output_amendment_path','cluster_output_amendment_sha256'),
                        ('cluster_output_parent_binding_path','cluster_output_parent_binding_sha256')]:
                    if file_hash(self.path(c[p])) != c[h]:
                        raise SearchContractError('PATTERN_CLUSTER_GENERATION_AMENDMENT_HASH_MISMATCH')
                amendment=json.loads(self.path(c['cluster_output_amendment_path']).read_bytes())
                prior=json.loads(self.path(c['cluster_output_parent_binding_path']).read_bytes())
                if (amendment.get('schema_version') != 'pattern_cluster_output_amendment_v1'
                        or amendment.get('policy') != policy
                        or amendment.get('scientific_method_changed') is not False
                        or amendment.get('user_task_sha256') != scope.get('user_task_sha256')
                        or amendment.get('parent_binding_sha256') != c['cluster_output_parent_binding_sha256']
                        or prior.get('identity') != versions.MATH_V2_2_EXECUTION_BINDING_VERSION
                        or prior.get('execution_attempt_id') == c['execution_attempt_id']):
                    raise SearchContractError('PATTERN_CLUSTER_GENERATION_AMENDMENT_INVALID')
                cluster_amendment=dict(policy=policy,path=c['cluster_output_amendment_path'],
                    sha256=c['cluster_output_amendment_sha256'],parent_path=c['cluster_output_parent_binding_path'],
                    parent_sha256=c['cluster_output_parent_binding_sha256'])
            expected=derive_current_pilot_contract(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
                parent_path=BASELINE_PATH,parent_sha256=BASELINE_SHA,authorization_path=c['current_user_scope_path'],
                authorization_sha256=c['current_user_scope_sha256'],max_opportunities=op['max_opportunities'],
                token_ceiling=op['token_ceiling'],cluster_generation_amendment=cluster_amendment)
            if c!=expected:raise SearchContractError('CURRENT_FROZEN_PILOT_SETTING_CHANGED')
            required=dict(schema_version='math_v2_2_operational_pilot_user_scope_v1',attempt_id=c['execution_attempt_id'],
                arm='A4',seed=81,user_authorized=True,one_attempt_only=True,max_opportunities=op['max_opportunities'],
                token_ceiling=op['token_ceiling'],scientific_method_changed=False,validation_authorized=False,
                test_authorized=False,push_authorized=True,initial_memory_entries=0,
                raw_diagnostic_authorized=False,llm_judge_authorized=False,canary_authorized=False,
                operational_retry_limit=0,exact_frozen_api_authorization_required=True)
            if (any(scope.get(k)!=v for k,v in required.items()) or len(scope.get('user_task_sha256',''))!=64):
                raise SearchContractError('CURRENT_USER_SCOPE_MISMATCH')
            CURRENT_POLICY_BUNDLE.validate_contract(c)
            for p,h in [('gradient_prompt_path','gradient_prompt_sha256'),('pattern_prompt_path','pattern_prompt_sha256'),
                ('accounting_policy_path','accounting_policy_sha256'),
                ('validation_accounting_metadata_path','validation_accounting_metadata_sha256')]:
                if file_hash(self.path(c[p]))!=c[h]:raise SearchContractError('CURRENT_DEPENDENCY_HASH_MISMATCH')
            if digest(json.loads(self.path(c['verify_settings_path']).read_bytes()))!=c['verify_settings_sha256']:
                raise SearchContractError('CURRENT_EVALUATOR_SETTINGS_IDENTITY_MISMATCH')
            validate_effective_math_dependencies(self)
            return ()
        except (SearchContractError,KeyError,TypeError,ValueError,OSError) as e:
            return (str(e) if isinstance(e,SearchContractError) else 'CURRENT_BINDING_INVALID',)

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
        if 'gradient_recovery_policy' in c:
            identity['gradient_recovery_policy']=c['gradient_recovery_policy']
        if 'pattern_cluster_generation_policy' in c:
            identity['pattern_cluster_generation_policy']=c['pattern_cluster_generation_policy']
        binding=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return CURRENT_POLICY_BUNDLE.method(aggregation=c['aggregation'],provider_binding=binding,
            successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
            partition_completion_policy=c.get('partition_completion_policy'),
            gradient_recovery_policy=c.get('gradient_recovery_policy'),
            pattern_cluster_generation_policy=c.get('pattern_cluster_generation_policy'))

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
            protocol=b.protocol),r['reference_final_answer']) for r in rows)
