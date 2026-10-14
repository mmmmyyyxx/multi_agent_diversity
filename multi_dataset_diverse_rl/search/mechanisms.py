"""Small explicit factory bundle. Scientific defaults remain V2.12.

Interventions are synthetic-only until a separate method/manifest is frozen.
There is no plugin loader and no implicit experiment or provider authorization.
"""
from dataclasses import dataclass,fields
from .binary_responsibility import BinaryPluralityResponsibilityAnalyzer
from .policies import TargetPolicyV1
from .textual_gradients import GradientExtractor
from .pattern_selection import WeightedSupportScorer,WeightedResponsibilitySelector
from .population_evidence import PatternSupportEvidence
from .reflection_steps import (FrozenParentSelection,TeamGroupChannel,ReflectionEvidenceComposition,
    SingleStepReflection,DisabledGroupChannel,DisabledReflection)
from .population_policies import FitnessWeightedSampling,BestThreeRetention,PerMemberArgmaxAssembly
from .memory_components import (MeasuredEditExperienceRecorder,ParentScopedMemoryRetrieval,
    WholeMeasuredCardDelivery,DisabledMemoryDelivery)
from .measurement_policies import CompleteOptimizeFitness,MeasuredPoolUpdate,JointFinalAdmission
from .target_or_team_transition import InitialCompetenceTargetOrTeamProgressV3
from .parent_population import digest
from .schemas import SearchContractError

ARCHITECTURE='V212_MODULAR_MECHANISMS_ARCHITECTURE_V1'

@dataclass(frozen=True)
class TeamModules:
    responsibility: object=BinaryPluralityResponsibilityAnalyzer
    who: object=TargetPolicyV1
    gradient: object=GradientExtractor
    pattern_scoring: object=WeightedSupportScorer
    pattern_selection: object=WeightedResponsibilitySelector
    support_sampling: object=PatternSupportEvidence
    parent_selection: object=FrozenParentSelection
    group_channel: object=TeamGroupChannel
    reflection_evidence: object=ReflectionEvidenceComposition
    reflection_generation: object=SingleStepReflection
    population_sampling: object=FitnessWeightedSampling
    population_retention: object=BestThreeRetention
    final_selection: object=PerMemberArgmaxAssembly
    memory_recording: object=MeasuredEditExperienceRecorder
    memory_retrieval: object=ParentScopedMemoryRetrieval
    memory_delivery: object=WholeMeasuredCardDelivery
    fitness: object=CompleteOptimizeFitness
    pool_update: object=MeasuredPoolUpdate
    team_admission: object=InitialCompetenceTargetOrTeamProgressV3
    final_admission: object=JointFinalAdmission

    def binding(self):
        entries={}
        for field in fields(self):
            factory=getattr(self,field.name)
            spec=CATALOG.get((field.name,factory))
            if spec is None:raise SearchContractError('UNREGISTERED_TEAM_MECHANISM:'+field.name)
            entries[field.name]={**spec,'implementation':factory.__module__+'.'+factory.__qualname__}
        return dict(architecture=ARCHITECTURE,modules=entries)

    def changed(self):return tuple(f.name for f in fields(self) if getattr(self,f.name) is not f.default)

    def validate(self,*,synthetic_intervention=False):
        binding=self.binding()
        if self.changed():
            if not synthetic_intervention:raise SearchContractError('TEAM_INTERVENTION_REQUIRES_NEW_FROZEN_METHOD')
            try:
                import sitecustomize
                active=sitecustomize.network_attempt_count()==0
                import socket
                active=active and socket.socket.connect.__module__=='sitecustomize'
            except (ImportError,AttributeError):active=False
            if not active:raise SearchContractError('TEAM_INTERVENTION_REQUIRES_OFFLINE_NETWORK_GUARD')
        return binding

    def configure_optimizer(self,optimizer):
        optimizer.parent_selection=self.parent_selection()
        optimizer.group_channel=self.group_channel()
        optimizer.evidence_composition=self.reflection_evidence()
        optimizer.candidate_generation=self.reflection_generation()


# Each row describes a scientific capability, not a file split.
_ROWS={
 'responsibility':('five_boolean_member_results','exclusive_error_labels',(),(),'stateless','classify_member_errors',False),
 'who':('member_responsibility_and_failure_history','selected_member',('responsibility',),(),'stateless','select_member',False),
 'gradient':('root_member_error_response_and_reference','validated_repair_hypotheses',('who',),('pattern_gradient',),'per_opportunity_diagnostics','diagnose_only',True),
 'pattern_scoring':('validated_partition_and_responsibility_labels','support_weight_sums',('gradient','responsibility'),(),'stateless','score_only',False),
 'pattern_selection':('scored_partition','chosen_pattern',('pattern_scoring',),(),'stateless','select_repair_direction',False),
 'support_sampling':('chosen_pattern_wrong_supports','representative_supports',('pattern_selection',),(),'stateless','select_supports',False),
 'parent_selection':('population_and_opportunity_identity','three_frozen_parents',('population_sampling',),(),'persisted_selection_plan','select_parents_only',False),
 'group_channel':('verified_root_team_supports','auxiliary_experience',('support_sampling',),('group_diagnosis',),'private_canonical_draw_audit','diagnose_only',False),
 'reflection_evidence':('root_and_actual_parent_observations','readable_reflection_view',('support_sampling','parent_selection','memory_delivery','group_channel'),(),'canonical_private_audit','compose_only',True),
 'reflection_generation':('one_parent_readable_view','legal_single_block_candidate',('reflection_evidence',),('reflection',),'bounded_slot_audit','propose_edit_only',True),
 'population_sampling':('measured_parent_fitness','positive_probabilities_and_draws',('fitness',),(),'seeded_stateless','sample_only',False),
 'population_retention':('measured_parents','at_most_three_parents',('fitness',),(),'population_checkpoint','retain_search_state_only',False),
 'final_selection':('five_populations','one_argmax_prompt_per_member',('population_retention',),(),'frozen_final_selection','propose_final_team_only',False),
 'memory_recording':('actual_complete_edit_measurements_and_admission','scoped_experience_records',('fitness','team_admission'),(),'transaction_delta','record_observations_only',False),
 'memory_retrieval':('stored_experiences_and_parent_hash','ranked_experiences',('memory_recording',),(),'stateless','rank_only',True),
 'memory_delivery':('ranked_measured_cards_and_budget','optimizer_only_context',('memory_retrieval',),(),'delivery_counter','deliver_only',True),
 'fitness':('complete_optimize_member_and_team_scores','provenance_bearing_fitness',(),(),'measured_profile_cache','assign_measured_score_only',False),
 'pool_update':('measured_child_fitness','population_upsert',('fitness','population_retention'),(),'population_checkpoint','retain_search_state_only',False),
 'team_admission':('complete_optimize_candidate_scores_and_initial_floor','deployment_proposal',('fitness',),(),'initial_competence_binding','select_safe_proposal_no_commit',False),
 'final_admission':('joint_final_scores_and_initial_floor','joint_full_pass',('final_selection','fitness'),(),'stateless','admit_full_before_private_shadow',False),
}
_METHODS={'responsibility':'analyze','who':'select','gradient':'extract','pattern_scoring':'score',
 'pattern_selection':'select','support_sampling':'compose_opportunity','parent_selection':'select',
 'group_channel':'deliver','reflection_evidence':'compose','reflection_generation':'generate',
 'population_sampling':'probabilities','population_retention':'keep','final_selection':'select',
 'memory_recording':'records','memory_retrieval':'rank','memory_delivery':'deliver',
 'fitness':'assign','pool_update':'update','team_admission':'select','final_admission':'allowed'}
CATALOG={}
_ARGUMENTS={
 'responsibility':('state','history'),'who':('state','diagnosis','feasible_members','history'),
 'gradient':('procedure','rows'),'pattern_scoring':('patterns','universe'),
 'pattern_selection':('patterns','seed','state_id','ordinal'),
 'support_sampling':('state','diagnosis','member_id','rows','context','ordinal'),
 'parent_selection':('population','member','seed','opportunity_id'),
 'group_channel':('evidence_source','provider','opportunity','pattern','rows','run_root','observer'),
 'reflection_evidence':('opportunity','parent','rows','pattern','diagnostics','canonical_group','group_experience','profile_reader','memory','config'),
 'reflection_generation':('reflection_lm','config','opportunity','parent','rows','pattern','plan','slot','packet','view','private','canonical_group','seen'),
 'population_sampling':('parents',),'population_retention':('ordered',),'final_selection':('ordered_members',),
 'memory_recording':('outcome','revision'),'memory_retrieval':('entries','parent_hash'),
 'memory_delivery':('ranked','parent_hash','limits','identity'),
 'fitness':('state','member','source','observation_id','count'),
 'pool_update':('pool','member','prompt','vote','proxy'),
 'team_admission':('parent','candidates'),
 'final_admission':('child','baseline','changed','initial_scores','invalid','parent_invalid','invalidity')}
def register_mechanism(kind,factory,*,identity,disabled=False):
    if (kind not in _ROWS or not isinstance(identity,str) or not identity
            or not isinstance(factory,type) or not callable(getattr(factory,_METHODS[kind],None))
            or (kind,factory) in CATALOG):raise SearchContractError('TEAM_MECHANISM_INTERFACE_OR_DUPLICATE')
    import inspect
    try:inspect.signature(getattr(factory,_METHODS[kind])).bind(None,**{k:None for k in _ARGUMENTS[kind]})
    except TypeError as error:raise SearchContractError('TEAM_MECHANISM_SIGNATURE_MISMATCH') from error
    if kind=='population_sampling':
        try:inspect.signature(factory.draw).bind(None,None,None,scope=None,seed=None,opportunity_id=None,member=None)
        except (AttributeError,TypeError) as error:raise SearchContractError('TEAM_MECHANISM_SIGNATURE_MISMATCH') from error
    input_schema,output_schema,deps,roles,state,authority,single=_ROWS[kind]
    CATALOG[kind,factory]=dict(identity=identity,input_schema=input_schema,output_schema=output_schema,
        dependencies=list(deps),model_roles=[] if disabled else list(roles),data_scopes=['optimize'],
        state_contract=state,authority=authority,disabled=disabled,
        disable_behavior='MODULE_DISABLED_NO_MODEL_CALL_OR_FABRICATED_OBSERVATION' if disabled else 'explicit_control_required',
        single_prompt_adaptation='team_context_adapter_required' if not single else 'compatible_context_required')

for field in fields(TeamModules):
    register_mechanism(field.name,field.default,identity=getattr(field.default,'identity',field.default.__name__))
register_mechanism('group_channel',DisabledGroupChannel,identity=DisabledGroupChannel.identity,disabled=True)
register_mechanism('reflection_generation',DisabledReflection,identity=DisabledReflection.identity,disabled=True)
register_mechanism('memory_delivery',DisabledMemoryDelivery,identity=DisabledMemoryDelivery.identity,disabled=True)

DEFAULT_TEAM_MODULES=TeamModules()

# Existing typed ports and safety authorities are inventoried separately from
# selectable factories. A description never grants authority to replace a guard.
BOUNDARY_CAPABILITIES={
 'gradient_provider':('textual_gradients.PerExampleGradientProvider','root_wrong_evidence','validated_diagnostic_draws',('who',),('pattern_gradient',),'private_draw_audit','optimize','provider_port','direct_adapter'),
 'pattern_clustering':('textual_gradients.GradientClusterProvider','validated_gradients','semantic_partition',('gradient',),('pattern_cluster',),'private_draw_audit','optimize','provider_port','direct_adapter'),
 'partition_validation':('textual_gradients.normalize_gradient_partition','partition_and_known_wrong_support','validated_partition',('pattern_clustering',),(),'stateless','optimize','shared_guard','direct_adapter'),
 'root_team_evidence':('group_diagnosis.RootTeamEvidence','verified_root_snapshot_and_support','canonical_root_team',('support_sampling',),(),'private_snapshot','optimize','evidence_source_port','team_only'),
 'group_generation':('group_diagnosis.GroupDiagnoser','canonical_root_team','validated_auxiliary_experience',('root_team_evidence',),('group_diagnosis',),'private_draw_audit','optimize','group_channel_generate_port','team_only'),
 'group_validation':('group_diagnosis.validate_experience','raw_diagnosis_and_canonical_sources','source_checked_experience',('root_team_evidence',),(),'stateless','optimize','shared_guard','team_only'),
 'reflection_projection':('reflector_view.project_reflector','canonical_packet','lossless_readable_view',('reflection_evidence',),(),'canonical_evidence_retained','optimize','evidence_composition_port','context_adapter_required'),
 'prompt_edit_guard':('system_prompt.candidate_failed_checks','parent_child_and_support','legality_findings',(),(),'stateless','optimize','shared_guard','direct_adapter'),
 'solver_measurement':('population_evaluation.PopulationEvaluationProvider','prompt_and_fixed_peers','complete_actual_profiles',(),('solver',),'identity_bound_profile_cache','optimize','evaluation_provider_port','member_measurement_adapter_required'),
 'private_shadow':('private_gate.private_binary_gate','full_admitted_proposal','private_pass_or_reject',('team_admission',),('solver',),'private_gate_profiles','adaptive_gate','gate_port','team_only'),
 'team_commit':('transition.TeamStateCommitter','admitted_proposal_and_gate_receipt','deployed_team',('private_shadow',),(),'team_state','optimize_plus_gate_receipt','committer_port','team_only'),
 'scientific_stop':('policies.GlobalStopPolicy','opportunity_commit_outcomes','stop_or_continue',('team_commit',),(),'boundary_checkpoint','optimize','stop_port','team_only'),
 'population_storage':('parent_population.ParentPopulation','measured_parents_and_plans','identity_bound_pool',('pool_update',),(),'private_population_checkpoint','optimize','storage_authority','team_only'),
 'memory_storage':('population_memory.PopulationEditMemory','actual_edit_records','scoped_store',('memory_recording',),(),'private_memory_transaction','optimize','storage_authority','measurement_adapter_unresolved'),
 'boundary_recovery':('population_checkpoint.PopulationCheckpoint','complete_identity_bound_boundary','verified_atomic_restore',('population_storage','memory_storage'),(),'private_boundary_checkpoint','optimize','recovery_authority','separate_framework_checkpoint'),
}

def capability_inventory(modules=DEFAULT_TEAM_MODULES):
    """Serializable capabilities with implementation and explicit adaptation limits."""
    boundaries={}
    for name,(source,inputs,outputs,deps,roles,state,scope,port,adapter) in BOUNDARY_CAPABILITIES.items():
        boundaries[name]=dict(implementation='multi_dataset_diverse_rl.search.'+source,
            input_schema=inputs,output_schema=outputs,dependencies=list(deps),model_roles=list(roles),
            data_scopes=[scope],state_contract=state,port=port,single_prompt_adaptation=adapter,
            disable_behavior='requires_explicit_legal_control_and_new_frozen_protocol')
    return {**modules.binding(),'boundary_capabilities':boundaries}
