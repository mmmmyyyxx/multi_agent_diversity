"""Auditable flow contracts; imported classes alone do not establish equivalence."""
from copy import deepcopy

IDENTITY='FORMAL_MODULE_FLOW_REVIEW_V1'


def module_review(name,binding):
    known={
        'none':('ABLATION_NO_MODULE_V1','APPROVED_CONTROL',
            'explicit_all_errors_or_random_batch_control_no_diagnostic_module',[]),
        'full_gradient_pattern':('ABLATION_FULL_WRONG_GRADIENT_PATTERN_REPRESENTATIVES_V1',
            'FORMAL_WITH_APPROVED_ADAPTATIONS',
            'all_wrong_then_formal_gradient_then_cluster_then_max_support_then_formal_representatives',
            ['single_prompt_max_support_instead_of_team_responsibility',
             'no_pattern_B_fallback', 'single_prompt_root_sampling_scope']),
        'largest_pattern':('ABLATION_FORMAL_DISCOVERY_LARGEST_SUPPORT_V1','HISTORICAL_BATCH_ONLY',
            'legacy_preselected_three_error_diagnosis',['not_current_full_diagnosis']),
    }
    if name not in known or binding.get('identity')!=known[name][0]:
        return dict(identity=IDENTITY,status='UNREVIEWED',module=name,
            reason='explicit_module_flow_and_single_prompt_adapter_review_required')
    expected,status,flow,adaptations=known[name]
    return dict(identity=IDENTITY,status=status,module=name,binding_identity=expected,
        flow=flow,approved_adaptations=adaptations,
        implementation=binding['implementation'],required_error_source=binding.get('required_error_source'))


def inventory():
    from ...search.mechanisms import capability_inventory
    team=capability_inventory()
    ports={}
    for name,entry in team['modules'].items():
        ports[name]=dict(implementation=entry['implementation'],
            team_status='ACTUAL_FORMAL_COMPONENT',
            single_prompt_status=('BOUND_FULL_PATTERN_ADAPTER' if name in
                ('gradient','support_sampling') else 'TEAM_CONTEXT_OR_NEW_PROTOCOL_REQUIRED'),
            authority=entry['authority'],dependencies=entry['dependencies'])
    boundaries={name:dict(implementation=row['implementation'],port=row['port'],
        single_prompt_adaptation=row['single_prompt_adaptation'],data_scopes=row['data_scopes'])
        for name,row in team['boundary_capabilities'].items()}
    return dict(identity=IDENTITY,team_modules=ports,boundary_capabilities=boundaries,
        independent=dict(
            none=dict(status='APPROVED_CONTROL',activation='explicit_A_all_errors_B_random_three'),
            full_gradient_pattern=dict(status='FORMAL_WITH_APPROVED_ADAPTATIONS',
                stages=['all_current_wrong_optimize','formal_per_example_gradient',
                    'successful_gradients_only','formal_cluster_and_partition',
                    'approved_maximum_support_selection','formal_up_to_three_representatives',
                    'representative_raw_errors_gradients_diagnostics_only']),
            largest_pattern=dict(status='HISTORICAL_BATCH_ONLY',activation='historical_schema_only'),
            formal_gradient=dict(status='DISABLED_BY_USER',activation=False,
                issue='old_adapter_preselected_three_errors; no_new_gradient_only_feedback_policy_authorized'),
            memory=dict(status='DISABLED_BY_USER_PENDING_ADAPTATION',activation=False,
                official_core='search.population_memory.PopulationEditMemory',
                preserved=['storage','parent_scoped_retrieval','whole_card_delivery','native_limits'],
                unresolved=['single_prompt_bootstrap','truthful_absent_team_vote_shadow',
                    'which_measured_candidate_edits_to_record','optimizer_delivery_schema'])),
        synthetic_team_controls=dict(
            group_channel='only_LLM_Group_generation_disabled_root_evidence_preserved',
            reflection_generation='pipeline_negative_control_zero_candidates_not_optimizer_efficacy',
            memory_delivery='delivery_only_disabled_true_recording_storage_preserved'),
        common_kernel=dict(status='USER_APPROVED_INDEPENDENT_KERNEL_NOT_TEAM_LAYER1',
            preserved=['one_candidate_per_opportunity','twenty_opportunities','strict_Validation100_gain',
                'formal_Solver_scoring_decoding','repeat_scoped_Solver_only_cache',
                'bounded_generated_output_recovery','accounted_unbounded_network_retry'],
            formal_shared_guards=['nonempty_three_block_structure','single_block_legality',
                'formal_4096_worked_reference_with_explicit_provenance',
                'actual_solver_input_isolation_without_keyword_filters','no_op_skip_without_retry'],
            excluded_team_only=['WHO','five_member_Group_inputs','three_Parent_population_slots',
                'Team_Vote_fitness','Full_Shadow_team_admission','joint_team_assembly']),
        real_api_authorized=False)
