"""Fresh V2.4 freeze and composition; parents supply data provenance only."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

from ..current_contract import MATH_INITIAL_TEAM_VERSION,CURRENT_CACHE_POLICY
from ..search.system_prompt import SEED, SystemPrompt, CONTRACT
from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.current_layer1 import CurrentOptimizer, EvidenceLayer1Config
from ..search.current_composition import build_current_team_prompt_search
from ..search.scientific_aggregation import EquivalencePluralityAggregation
from ..search.optimization_evidence import POLICY, BINDING, METHOD, MEMORY, EVIDENCE, INPUT, GRADIENT_PROMPT_ID, PROBE_POLICY
from ..search.textual_gradients import REFERENCE_GRADIENT_PROMPT, CLUSTER_PROMPT, pattern_policy_for_trajectory
from ..search.schemas import SearchContractError
from .math_structured_interface import MATHStructuredSystemBenchmark, SYSTEM_PROMPT_POLICY, system_interface_contract, system_prompt_policy
from .math_flexible_answer import POLICY as EXTRACTION, validity_contract, IDENTITY as PARSER
from .math_response_evidence import trajectory_policy
from .math_prediction_validity import structured_recovery_contract
from .data_freeze import file_hash, digest
from .protocols import protocol_input, PROTOCOLS
from .experiment_splits import ExperimentSplitReader
from .access import DataPurpose


def initial_team_artifact(team_version=MATH_INITIAL_TEAM_VERSION):
    from .math_canary_inputs import initial_prompt
    prompt=initial_prompt(team_version)
    hashes=[prompt.prompt_hash]*5
    return dict(team_version=team_version,initial_team_data_dependency='NONE',
        semantic_contract=['complete editable system prompt'],ordered_member_ids=list(range(5)),
        members=[dict(member_id=i,prompt=prompt.to_dict(),prompt_sha256=prompt.prompt_hash,
            block_hashes={k:prompt.block_hash(k) for k in ('role','strategy','answer')},
            rendered_system_sha256=prompt.system_hash) for i in range(5)],ordered_team_sha256=digest(hashes))


def gradient_prompt_artifact():
    return dict(identity=GRADIENT_PROMPT_ID,prompt=REFERENCE_GRADIENT_PROMPT)


def prepare_structured_inputs(root,*,experiment_id,attempt,user_task_sha256,execution_phase='pilot',
        seed=81,team_version=MATH_INITIAL_TEAM_VERSION,initial_team_path=None,development_subsets=None):
    """Prepare a fresh unapproved scope without loading examples or credentials."""
    from ..persistence.durable_io import atomic_write_json
    from ..governance.token_accounting import POLICY_V25_2M
    parent_path='experiments/execution_bindings/a4_v24_structured_seed81_canary_attempt1.json'
    parent=json.loads((root/parent_path).read_bytes());parent_hash=file_hash(root/parent_path)
    protocol='experiments/protocols/'+experiment_id
    binding_path='experiments/execution_bindings/'+attempt+'.json'
    paths=dict(scope=protocol+'/preparation_scope.json',gradient=protocol+'/gradient.json',
        pattern=protocol+'/pattern.json',metadata=protocol+'/heldout_seal.json',
        accounting=protocol+'/accounting_policy.json',team=initial_team_path or 'experiments/initial_teams/math_explicit_final_seed_v2.json')
    if (root/binding_path).exists():raise SearchContractError('V25_FRESH_PATH_REQUIRED')
    scope=dict(schema_version='structured_system_preparation_scope_v1',attempt_id=attempt,
        one_attempt_only=True,preparation_only=True,real_api_authorized=False,
        validation_authorized=False,test_authorized=False,optimization_evidence_policy=POLICY,
        parent_binding_sha256=parent_hash,user_task_sha256=user_task_sha256,
        exact_post_freeze_api_authorization_required=True,old_authorization_reusable=False,
        old_responses_cache_and_competence_reusable=False,push_authorized=False)
    for path,value in ((paths['scope'],scope),(paths['gradient'],gradient_prompt_artifact()),
            (paths['pattern'],dict(identity='STRUCTURED_GRADIENT_CLUSTER_PROMPT_V2',prompt=CLUSTER_PROMPT)),
            (paths['accounting'],POLICY_V25_2M),(paths['team'],initial_team_artifact(team_version)),
            (paths['metadata'],dict(identity='V25_HELDOUT_SEAL_NO_ACCESS_METADATA_V1',
                validation_model_calls=0,test_model_calls=0,heldout_accounting_reserve=0,
                validation_access='not_authorized',test_access='sealed'))):
        if (root/path).exists():
            if json.loads((root/path).read_bytes())!=value:raise SearchContractError('V25_PREPARATION_INPUT_CHANGED')
        else:atomic_write_json(root/path,value)
    c=derive_structured_contract(parent,attempt=attempt,binding_path=binding_path,
        parent_path=parent_path,parent_sha256=parent_hash,authorization_path=paths['scope'],
        authorization_sha256=file_hash(root/paths['scope']),gradient_prompt_path=paths['gradient'],
        gradient_prompt_sha256=file_hash(root/paths['gradient']),pattern_prompt_path=paths['pattern'],
        pattern_prompt_sha256=file_hash(root/paths['pattern']),validation_metadata_path=paths['metadata'],
        validation_metadata_sha256=file_hash(root/paths['metadata']),initial_team_path=paths['team'],
        initial_team_artifact_sha256=file_hash(root/paths['team']),accounting_policy_path=paths['accounting'],
        accounting_policy_sha256=file_hash(root/paths['accounting']),execution_phase=execution_phase,
        max_opportunities=1 if execution_phase=='canary' else 5,seed=seed,
        team_version=team_version,development_subsets=development_subsets)
    blockers=MATHStructuredBinding(root,c).blockers()
    if blockers:raise SearchContractError('V25_PREPARATION_HOLD:'+','.join(blockers))
    atomic_write_json(root/binding_path,c)
    atomic_write_json(root/(protocol+'/protocol.json'),dict(identity='STRUCTURED_GENERATED_OUTPUT_RECOVERY_PROTOCOL_V3',
        attempt_id=attempt,method=METHOD,seed=seed,members=5,phase=execution_phase,
        optimize=c['initial_competence_binding']['count'],shadow=40,models=c['models'],
        system_prompt_policy=c['system_prompt_policy'],answer_extraction_policy=EXTRACTION,
        optimization_evidence_policy=POLICY,provider_bounds=c['provider_bounds'],
        generated_output_recovery_policy=c['generated_output_recovery_policy'],
        solver_execution_policy=c['solver_execution_policy'],invalid_recovery_policy=c['invalid_recovery_policy'],
        canary_review_policy=c['canary_review_policy'],initial_accuracy='UNMEASURED_NEW_REAL_REQUESTS_REQUIRED',
        real_api_authorized=False,validation_access='not_authorized',test_access='sealed'))
    return c


def derive_structured_contract(parent, *, attempt, binding_path, parent_path, parent_sha256,
        authorization_path, authorization_sha256, gradient_prompt_path, gradient_prompt_sha256,
        validation_metadata_path, validation_metadata_sha256, initial_team_path,
        initial_team_artifact_sha256, pattern_prompt_path, pattern_prompt_sha256,
        accounting_policy_path, accounting_policy_sha256, max_opportunities=5,
        token_ceiling=2_000_000, execution_phase='pilot',seed=81,
        team_version=MATH_INITIAL_TEAM_VERSION,development_subsets=None):
    if (not isinstance(attempt,str) or not attempt or execution_phase not in {'canary','pilot'}
            or attempt in {parent.get('execution_attempt_id'),parent.get('cache_namespace')}
            or type(max_opportunities) is not int or not 0<max_opportunities<=5
            or token_ceiling!=2_000_000 or type(seed) is not int or seed<0
            or any(new==parent.get(key) for new,key in (
                (binding_path,'binding_path'),(authorization_path,'current_user_scope_path'),
                (gradient_prompt_path,'gradient_prompt_path'),(initial_team_path,'initial_team_path'),
                (pattern_prompt_path,'pattern_prompt_path'),(accounting_policy_path,'accounting_policy_path'),
                (validation_metadata_path,'validation_accounting_metadata_path')))):
        raise SearchContractError('V25_FRESH_FREEZE_REQUIRED')
    initial=initial_team_artifact(team_version)
    c=deepcopy(parent)
    # No historical authorization, response import, operational ancestry or
    # held-out evaluation can become a current execution policy.
    for key in tuple(c):
        if (key.startswith('initial_evidence_reuse') or 'amendment' in key
                or 'user_scope' in key or 'parent_binding' in key
                or key.startswith('baseline_scientific') or key.startswith('gradient_recovery')
                or key in {'post_search_validation_policy','canary_review_policy',
                    'validation_accounting_metadata_path','validation_accounting_metadata_sha256',
                    'rolling_memory_freeze_path','rolling_memory_freeze_sha256'}):
            c.pop(key)
    from ..search.solver_execution import POLICY as execution
    from ..governance.canary_review import STRUCTURED_POLICY
    from ..search.generation_failures import POLICY as output_recovery
    from ..provider_routing import IDENTITY as provider, SOLVER_MODEL, OPTIMIZER_MODEL, routing_contract
    from .math_solver_decoding import solver_decoding_contract
    c.pop('verify_settings_path', None)
    c.pop('verify_settings_sha256', None)
    c.update(identity=BINDING,method_identity=METHOD,execution_attempt_id=attempt,
        provider=provider,provider_routing_policy=routing_contract(),
        models=dict(solver=SOLVER_MODEL,optimizer_reflection=OPTIMIZER_MODEL,
            pattern=OPTIMIZER_MODEL,solver_thinking=False),solver_decoding_policy=solver_decoding_contract(),
        canary_attempt_id=attempt,cache_namespace=attempt,binding_path=binding_path,
        execution_phase=execution_phase,execution_seed=seed,seeds=[seed],cache_policy=CURRENT_CACHE_POLICY,
        generated_output_recovery_policy=deepcopy(output_recovery),
        system_prompt_policy=system_prompt_policy(team_version),answer_extraction_policy=deepcopy(EXTRACTION),
        solver_output_interface=system_interface_contract(team_version),solver_trajectory_policy=trajectory_policy(),
        prediction_validity_policy=validity_contract(),invalid_recovery_policy=structured_recovery_contract(),
        payload_parser_identity=PARSER,optimization_evidence_policy=deepcopy(POLICY),repair_probe_policy=deepcopy(PROBE_POLICY),
        optimizer_input_schema=INPUT,layer1_search_policy=asdict(EvidenceLayer1Config()),
        candidate_contract_identity=CONTRACT,memory_policy_identity=MEMORY,panel_evidence_policy=EVIDENCE,
        gradient_prompt_path=gradient_prompt_path,gradient_prompt_sha256=gradient_prompt_sha256,
        pattern_prompt_path=pattern_prompt_path,pattern_prompt_sha256=pattern_prompt_sha256,
        trajectory_parent_binding_path=parent_path,trajectory_parent_binding_sha256=parent_sha256,
        current_user_scope_path=authorization_path,current_user_scope_sha256=authorization_sha256,
        task_authorization_sha256=authorization_sha256,continuation_authorization_sha256=authorization_sha256,
        validation_accounting_metadata_path=validation_metadata_path,
        validation_accounting_metadata_sha256=validation_metadata_sha256,
        initial_team_path=initial_team_path,initial_team_version=team_version,
        initial_team_artifact_sha256=initial_team_artifact_sha256,
        initial_team_sha256=initial['ordered_team_sha256'],
        solver_execution_policy=deepcopy(execution),
        concurrency=dict(solver=8,optimizer_reflection=1,pattern=1),
        accounting_policy_path=accounting_policy_path,accounting_policy_sha256=accounting_policy_sha256,
        accounting_scope_policy='FRESH_V25_REPAIR_SINGLE_ARM_2M_V2',heldout_accounting_reserve=0,
        token_ledger_directory='runs/'+attempt+'/accounting',execution_arm='A4',arms={'A4':[True,True]},
        canary_review_policy=deepcopy(STRUCTURED_POLICY),canary_phase='initial_profile_first_opportunity',
        operational_pilot=dict(identity='TARGET_OR_TEAM_PROGRESS_OPERATIONAL_PILOT_BOUND_V1',
            max_opportunities=max_opportunities,token_ceiling=token_ceiling,
            scientific_stopper='team_epoch_no_commit_v1',guarantees_saturation=False))
    if development_subsets is not None:
        if (set(development_subsets)!={'path','sha256','protocol'} or
                (execution_phase!='canary' and development_subsets['protocol'].get('identity')!='MATH_FRESH_RECOVERY_DEVELOPMENT_MEMBERSHIP_V3')):
            raise SearchContractError('FRESH_CANARY_SUBSET_BINDING_INVALID')
        c.update(low_cost_subsets_path=development_subsets['path'],
            low_cost_subsets_sha256=development_subsets['sha256'],
            low_cost_protocol=deepcopy(development_subsets['protocol']))
    c['pattern_policy']=pattern_policy_for_trajectory(c['solver_trajectory_policy'],POLICY)
    from ..current_contract import PATTERN_SPECIFIC_CONTENT_GUARD_VERSION, GRADIENT_PARTITION_COMPLETION_VERSION
    c['pattern_abstraction_guard']=PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
    c['partition_completion_policy']=GRADIENT_PARTITION_COMPLETION_VERSION
    from .protocols import MATH_PROTOCOL_FLEXIBLE_V7
    c['benchmark_protocol_sha256']=MATH_PROTOCOL_FLEXIBLE_V7.identity()
    c['budget']['metric_calls']=42
    count=12 if execution_phase=='canary' else 60
    name='canary_optimize' if execution_phase=='canary' else 'pilot_optimize'
    c['initial_competence_binding'].update(count=count,support_identity=c['low_cost_protocol']['membership_hashes'][name])
    bootstrap=5*(count+c['shadow_count']);per_op=dict(local=42,probe=36,full=2*count,shadow=c['shadow_count'])
    solver=4*(bootstrap+max_opportunities*sum(per_op.values()))
    gradients=3*count*max_opportunities;clusters=3*max_opportunities;reflection=6*max_opportunities
    successful=solver+gradients+clusters+reflection;transport=c['decoding']['transport_retries']+1
    c['provider_bounds']=dict(max_opportunities=max_opportunities,max_proposals_per_opportunity=6,
        solver_per_opportunity=sum(per_op.values()),physical_solver_per_opportunity=4*sum(per_op.values()),
        solver_calls=solver,pattern_gradient_calls=gradients,pattern_cluster_calls=clusters,
        pattern_calls=gradients+clusters,reflection_calls=reflection,successful_provider_calls=successful,
        transport_attempts=successful*transport,attempt_charged_token_ceiling=token_ceiling,
        cumulative_charged_token_ceiling=token_ceiling,reservation_peak_upper_bound=token_ceiling,
        bound_proof=dict(identity='V25_STRUCTURED_RESOURCE_DERIVATION_V1',
            bootstrap_logical_solver_calls=bootstrap,per_op_logical_solver=per_op,
            solver_semantic_multiplier=4,gradient_semantic_multiplier=3,cluster_semantic_multiplier=3,transport_multiplier=transport,
            scientific_stopper_unchanged=True,guarantees_saturation=False,
            token_bound='PRE_TRANSPORT_EXACT_RESERVATION_AND_CUMULATIVE_ATTEMPT_ADMISSION'))
    return c


class MATHStructuredBinding:
    def __init__(self,root,contract):
        self.root=root.resolve();self.contract=contract

    def path(self,relative):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root):raise SearchContractError('EXECUTION_BINDING_PATH_ESCAPE')
        return path

    def blockers(self):
        c=self.contract
        try:
            CURRENT_POLICY_BUNDLE.validate_contract(c)
            parent_path=c['trajectory_parent_binding_path']
            if file_hash(self.path(parent_path))!=c['trajectory_parent_binding_sha256']:
                raise SearchContractError('V25_PARENT_HASH_MISMATCH')
            parent=json.loads(self.path(parent_path).read_bytes())
            expected=derive_structured_contract(parent,attempt=c['execution_attempt_id'],
                binding_path=c['binding_path'],parent_path=parent_path,parent_sha256=c['trajectory_parent_binding_sha256'],
                authorization_path=c['current_user_scope_path'],authorization_sha256=c['current_user_scope_sha256'],
                gradient_prompt_path=c['gradient_prompt_path'],gradient_prompt_sha256=c['gradient_prompt_sha256'],
                validation_metadata_path=c['validation_accounting_metadata_path'],
                validation_metadata_sha256=c['validation_accounting_metadata_sha256'],
                initial_team_path=c['initial_team_path'],initial_team_artifact_sha256=c['initial_team_artifact_sha256'],
                pattern_prompt_path=c['pattern_prompt_path'],pattern_prompt_sha256=c['pattern_prompt_sha256'],
                accounting_policy_path=c['accounting_policy_path'],accounting_policy_sha256=c['accounting_policy_sha256'],
                max_opportunities=c['operational_pilot']['max_opportunities'],
                token_ceiling=c['operational_pilot']['token_ceiling'],execution_phase=c['execution_phase'],
                seed=c['execution_seed'],team_version=c['initial_team_version'],
                development_subsets=(dict(path=c['low_cost_subsets_path'],sha256=c['low_cost_subsets_sha256'],
                    protocol=c['low_cost_protocol']) if c['low_cost_protocol']['identity'] in {
                        'MATH_FRESH_CANARY_MEMBERSHIP_V2','MATH_FRESH_RECOVERY_DEVELOPMENT_MEMBERSHIP_V3'} else None))
            if c!=expected:raise SearchContractError('V25_FROZEN_SETTINGS_CHANGED')
            for key in sorted(k for k in c if k.endswith('_path')):
                h='initial_team_artifact_sha256' if key=='initial_team_path' else key[:-5]+'_sha256'
                if h not in c:continue
                actual=digest(json.loads(self.path(c[key]).read_bytes())) if key=='verify_settings_path' else file_hash(self.path(c[key]))
                if actual!=c[h]:raise SearchContractError('V25_DEPENDENCY_HASH_MISMATCH:'+key)
            if json.loads(self.path(c['gradient_prompt_path']).read_bytes())!=gradient_prompt_artifact():
                raise SearchContractError('V25_GRADIENT_PROMPT_MISMATCH')
            if json.loads(self.path(c['pattern_prompt_path']).read_bytes())['prompt']!=CLUSTER_PROMPT:
                raise SearchContractError('V25_CLUSTER_PROMPT_MISMATCH')
            from ..governance.token_accounting import POLICY_V25_2M
            if json.loads(self.path(c['accounting_policy_path']).read_bytes())!=POLICY_V25_2M:
                raise SearchContractError('V25_ACCOUNTING_POLICY_MISMATCH')
            scope=json.loads(self.path(c['current_user_scope_path']).read_bytes())
            required=dict(schema_version='structured_system_preparation_scope_v1',
                attempt_id=c['execution_attempt_id'],one_attempt_only=True,preparation_only=True,
                real_api_authorized=False,validation_authorized=False,test_authorized=False,
                optimization_evidence_policy=POLICY,parent_binding_sha256=c['trajectory_parent_binding_sha256'])
            if any(scope.get(k)!=v for k,v in required.items()):raise SearchContractError('V25_PREPARATION_SCOPE_MISMATCH')
            from .current_math_dependencies import validate_effective_math_dependencies
            validate_effective_math_dependencies(self)
            return ()
        except (SearchContractError,KeyError,TypeError,ValueError,OSError) as exc:
            return (str(exc) if isinstance(exc,SearchContractError) else 'V25_FRESH_FREEZE_REQUIRED',)

    def reader(self):
        c=self.contract
        return ExperimentSplitReader(self.path(c['canonical_root']),self.path(c['split_directory']),'math',
            expected_manifest_sha256=c['split_manifest_sha256'],expected_protocol='benchmark_experiment_split_v2')

    def benchmark(self):return MATHStructuredSystemBenchmark(self.contract)

    def method(self,arm):
        if arm!='A4':raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')
        if self.blockers():raise SearchContractError(self.blockers()[0])
        c=self.contract
        provider_binding=digest({k:c[k] for k in ('provider','provider_routing_policy','models','gradient_prompt_sha256',
            'pattern_prompt_sha256','optimizer_generation_policy','pattern_policy',
            'partition_completion_policy','pattern_cluster_generation_policy')})
        return CURRENT_POLICY_BUNDLE.method(aggregation=c['aggregation'],provider_binding=provider_binding,
            successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
            partition_completion_policy=c['partition_completion_policy'],
            pattern_cluster_generation_policy=c['pattern_cluster_generation_policy'],
            solver_trajectory_policy=c['solver_trajectory_policy'],optimization_evidence_policy=POLICY)

    def compose(self,*,arm,seed,solver,reflection,pattern_provider,run_root,optimize_fn=None):
        if self.blockers():raise SearchContractError(self.blockers()[0])
        if seed!=self.contract['execution_seed'] or optimize_fn is not None:raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')
        if (pattern_provider is None or pattern_provider.gradient_provider is None
                or solver.broker.contract!=self.contract or reflection.broker is not solver.broker
                or pattern_provider.broker is not solver.broker
                or pattern_provider.gradient_provider.broker is not solver.broker
                or solver.broker.arm!=arm or solver.broker.seed!=seed):
            raise SearchContractError('MATH_PROVIDER_PORT_BINDING_MISMATCH')
        team=json.loads(self.path(self.contract['initial_team_path']).read_bytes())
        optimizer=CurrentOptimizer(evaluator=solver,reflection_lm=reflection,
            accounting_reader=reflection.accounting,run_root=run_root)
        return build_current_team_prompt_search(benchmark=self.benchmark(),aggregation=EquivalencePluralityAggregation(),
            examples=self.examples('optimize'),prompts=tuple(SystemPrompt.from_dict(m['prompt']) for m in team['members']),
            solver=solver,optimizer=optimizer,method=self.method(arm),seed=seed,
            shadow_loader=lambda:self.examples('shadow'),shadow_count=self.contract['shadow_count'],
            runtime_readiness=self.blockers,pattern_provider=pattern_provider,
            provider_call_reader=lambda:solver.broker.successes,execution_phase=self.contract['execution_phase'],
            operational_bound=self.contract['operational_pilot'])

    def examples(self,role):
        if role not in {'optimize','shadow'}:raise SearchContractError('HELDOUT_SEARCH_ACCESS_FORBIDDEN')
        from .math_low_cost import read_subsets
        from ..search.binary_runtime import CorrectnessExample
        reader=self.reader();c=self.contract
        name='pilot_shadow' if role=='shadow' else 'canary_optimize' if c['execution_phase']=='canary' else 'pilot_optimize'
        members=read_subsets(self.root,c)['memberships'][name];wanted={r['stable_example_id'] for r in members}
        reader.members=[r for r in reader.members if r['project_split']!=role or r['stable_example_id'] in wanted]
        reader.manifest=dict(reader.manifest,counts={**reader.manifest['counts'],role:len(members)})
        rows=reader.rows(role,DataPurpose.ADAPTIVE_GATE if role=='shadow' else DataPurpose.EVIDENCE);b=self.benchmark()
        return tuple(CorrectnessExample(protocol_input('math',r['stable_example_id'],r['content'],b.output_contract,
            protocol=b.protocol),r['reference_final_answer'],r['content']['solution'] if role=='optimize' else None,
            tuple((k,str(r['content'][k])) for k in ('type','level')) if role=='optimize' else ()) for r in rows)
