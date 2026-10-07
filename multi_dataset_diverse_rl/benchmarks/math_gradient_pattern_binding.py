"""Flat current MATH contract; historical parents are integrity receipts only."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from .. import current_contract as versions
from ..search.current_layer1 import GradientPatternLayer1Config, CurrentOptimizer
from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.current_composition import build_current_team_prompt_search
from ..search.textual_gradients import POLICY, GRADIENT_PROMPT, CLUSTER_PROMPT
from ..search.schemas import SearchContractError
from ..search.scientific_aggregation import EquivalencePluralityAggregation
from ..governance.provenance_receipts import verify_receipt_dependencies
from .data_freeze import file_hash
from .gradient_contract_receipt import derive_gradient_contract
from .gradient_pilot_contract import derive_current_pilot_contract, OBSERVATION_POLICY
from .experiment_splits import ExperimentSplitReader
from .access import DataPurpose
from .protocols import protocol_input, PROTOCOLS
from .math_v21_interface import MATHV21BenchmarkAdapter
from .current_math_dependencies import validate_effective_math_dependencies
from .initial_condition_contract import initial_condition_provenance


class MATHGradientPatternBinding:
    def __init__(self,root,contract):
        self.root=root.resolve();self.contract=contract

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
        c=self.contract;CURRENT_POLICY_BUNDLE.validate_contract(c)
        identity=dict(provider=c['provider'],model=c['models']['pattern'],
            gradient_prompt=c['gradient_prompt_sha256'],cluster_prompt=c['pattern_prompt_sha256'],
            generation_policy=c['optimizer_generation_policy'],discovery=c['pattern_policy'],
            support_id_transport=c['pattern_support_id_transport'],abstraction_guard=c['pattern_abstraction_guard'])
        if 'partition_completion_policy' in c:
            identity['partition_completion_policy']=c['partition_completion_policy']
        binding=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return CURRENT_POLICY_BUNDLE.method(aggregation=c['aggregation'],provider_binding=binding,
            successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
            partition_completion_policy=c.get('partition_completion_policy'))

    def compose(self,*,arm,seed,solver,reflection,pattern_provider,run_root,optimize_fn=None):
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
            provider_call_reader=lambda:solver.broker.successes,execution_phase=self.contract['execution_phase'])

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

    def blockers(self):
        try:
            if 'partition_completion_amendment_path' in self.contract:
                from .partition_completion_contract import validate_partition_completion
                validate_partition_completion(self)
                validate_effective_math_dependencies(self)
                return ()
            if 'operational_user_scope_path' in self.contract:
                from .operational_pilot_contract import validate_operational_pilot
                validate_operational_pilot(self)
                validate_effective_math_dependencies(self)
                return ()
            if 'numeric_admissibility_amendment_path' in self.contract:
                from .numeric_admissibility_contract import validate_numeric_admissibility
                validate_numeric_admissibility(self)
                validate_effective_math_dependencies(self)
                return ()
            c,historical=initial_condition_provenance(self)
            CURRENT_POLICY_BUNDLE.validate_contract(c)
            for path,sha in [('gradient_parent_binding_path','gradient_parent_binding_sha256'),
                ('pattern_amendment_authorization_path','pattern_amendment_authorization_sha256'),
                ('gradient_prompt_path','gradient_prompt_sha256'),('pattern_prompt_path','pattern_prompt_sha256')]:
                if file_hash(self.path(c[path]))!=c[sha]:return ('GRADIENT_PATTERN_DEPENDENCY_HASH_MISMATCH',)
            parent=json.loads(self.path(c['gradient_parent_binding_path']).read_bytes())
            verify_receipt_dependencies(self.root,parent,historical_initial_team=historical)
            approval=json.loads(self.path(c['pattern_amendment_authorization_path']).read_bytes())
            verify_receipt_dependencies(self.root,approval,historical_initial_team=historical)
            if (approval.get('schema_version')!='gradient_pattern_code_amendment_v1'
                    or approval.get('code_change_authorized') is not True
                    or any(approval.get(k) is not False for k in ('real_api_authorized','canary_authorized',
                        'pilot_authorized','validation_authorized','test_authorized','push_authorized'))
                    or approval.get('provider_call_budget')!=0 or approval.get('api_token_budget')!=0
                    or approval.get('parent_binding_sha256')!=c['gradient_parent_binding_sha256']
                    or approval.get('pattern_policy')!=POLICY
                    or approval.get('layer1_search_policy')!=asdict(GradientPatternLayer1Config())):
                return ('GRADIENT_PATTERN_CODE_AUTHORITY_MISMATCH',)
            request_hash=approval.get('user_request_sha256')
            if not isinstance(request_hash,str) or len(request_hash)!=64 or any(x not in '0123456789abcdef' for x in request_hash):
                return ('GRADIENT_PATTERN_CODE_AUTHORITY_MISMATCH',)
            if (json.loads(self.path(c['gradient_prompt_path']).read_bytes())!={'identity':versions.GRADIENT_PROMPT_VERSION,'prompt':GRADIENT_PROMPT}
                    or json.loads(self.path(c['pattern_prompt_path']).read_bytes())!={'identity':versions.GRADIENT_PATTERN_DISCOVERY_VERSION,'prompt':CLUSTER_PROMPT}):
                return ('GRADIENT_PATTERN_PROMPT_CONTRACT_MISMATCH',)
            expected=derive_gradient_contract(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
                parent_path=c['gradient_parent_binding_path'],parent_sha256=c['gradient_parent_binding_sha256'],
                approval_path=c['pattern_amendment_authorization_path'],approval_sha256=c['pattern_amendment_authorization_sha256'],
                gradient_prompt_path=c['gradient_prompt_path'],gradient_prompt_sha256=c['gradient_prompt_sha256'],
                cluster_prompt_path=c['pattern_prompt_path'],cluster_prompt_sha256=c['pattern_prompt_sha256'])
            if c.get('execution_phase') == 'pilot':
                pilot_parent=json.loads(self.path(c['pilot_parent_binding_path']).read_bytes())
                authority=json.loads(self.path(c['pilot_execution_authorization_path']).read_bytes())
                if (file_hash(self.path(c['pilot_parent_binding_path']))!=c['pilot_parent_binding_sha256']
                        or file_hash(self.path(c['pilot_execution_authorization_path']))!=c['pilot_execution_authorization_sha256']
                        or pilot_parent!=derive_gradient_contract(parent,
                            attempt=pilot_parent['execution_attempt_id'],binding_path=pilot_parent['binding_path'],
                            parent_path=c['gradient_parent_binding_path'],parent_sha256=c['gradient_parent_binding_sha256'],
                            approval_path=c['pattern_amendment_authorization_path'],approval_sha256=c['pattern_amendment_authorization_sha256'],
                            gradient_prompt_path=c['gradient_prompt_path'],gradient_prompt_sha256=c['gradient_prompt_sha256'],
                            cluster_prompt_path=c['pattern_prompt_path'],cluster_prompt_sha256=c['pattern_prompt_sha256'])):
                    return ('CURRENT_PILOT_PARENT_RECEIPT_MISMATCH',)
                required=dict(schema_version='current_gradient_pilot_user_scope_v1',arm='A4',seed=81,
                    phase='pilot_search_only',attempt_id=c['execution_attempt_id'],user_authorized=True,
                    parent_binding_sha256=c['pilot_parent_binding_sha256'],scientific_change_authorized=False,
                    validation_authorized=False,test_authorized=False,raw_diagnostic_authorized=False,
                    llm_judge_authorized=False,push_authorized=False,operational_fresh_retry_limit=1,
                    observation_policy=OBSERVATION_POLICY)
                if (any(authority.get(k)!=v for k,v in required.items())
                        or not isinstance(authority.get('user_task_sha256'),str)
                        or len(authority['user_task_sha256'])!=64):
                    return ('CURRENT_PILOT_USER_SCOPE_MISMATCH',)
                expected=derive_current_pilot_contract(pilot_parent,attempt=c['execution_attempt_id'],
                    binding_path=c['binding_path'],parent_path=c['pilot_parent_binding_path'],
                    parent_sha256=c['pilot_parent_binding_sha256'],authorization_path=c['pilot_execution_authorization_path'],
                    authorization_sha256=c['pilot_execution_authorization_sha256'])
            if (c!=expected or not c['execution_attempt_id'].startswith('math_v2_1_gradient_pattern_A4_seed81_')
                    or c['execution_attempt_id']==parent['execution_attempt_id'] or c['cache_namespace']==parent['cache_namespace']):
                return ('GRADIENT_PATTERN_FROZEN_CONTRACT_MISMATCH',)
            validate_effective_math_dependencies(self)
            return ()
        except SearchContractError as error:
            if str(error)=='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN' or str(error).startswith(('CURRENT_DATA_', 'CURRENT_INITIAL_CONDITION_', 'CURRENT_NUMERIC_ADMISSIBILITY_', 'OPERATIONAL_PILOT_')):return (str(error),)
            return ('GRADIENT_PATTERN_BINDING_INVALID',)
        except (KeyError,TypeError,ValueError,OSError):return ('GRADIENT_PATTERN_BINDING_INVALID',)
