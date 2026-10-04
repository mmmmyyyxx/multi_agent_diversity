# Experiment Lineage

DO NOT EDIT DIRECTLY. Generated from experiments/lineage.yaml.
Registry provides node metadata; current_frontier.yaml provides readiness only.

## Current frontier

```yaml
schema_version: current_frontier_v1
current_architecture: Unified Team Prompt Search
method_identity: unified_team_prompt_search_v2_1
current_method: unified_team_prompt_search_v2_1
current_implementation: Versioned method semantic contract; benchmark-neutral production composition; zero-API audit
current_experiment: math_v2_1_a3_seed81_low_cost_canary_v1
current_benchmark_suite:
- math
- ifbench
- hotpotqa
historical_benchmark_only: true
historical_benchmark: bbh
current_dataset_migration: multibench_dataset_migration_v1
future_experiment_plan: v2_pattern_memory_multibench_v1
last_governance_milestone: repository_hygiene_alignment_v1
last_scientific_contract_milestone: benchmark_scientific_contract_freeze_v1
real_execution_ready: false
real_api_authorized: false
validation_access: not_authorized
test_access: sealed
open_questions: docs/research/OPEN_QUESTIONS.md
last_method_milestone: math_v2_1_layer1_memory_amendment_v1
canary_manifest: experiments/manifests/math_v2_1_a3_seed81_low_cost_canary_v1.yaml
last_preexecution_milestone: math_v2_1_memory_preexecution_v1
last_canary_milestone: math_v2_1_a3_seed81_low_cost_canary_v1
current_canary_status: PASS
formal_a1_ready: false
formal_a1_authorized: false
current_execution_blocker: null
default_optimizer_generation_policy: OPTIMIZER_REFLECTION_GENERATION_POLICY_V3
default_optimizer_enable_thinking: false
optimizer_policy_amendment_status: LAYER1_GENERATION_V3_FROZEN_AND_DISPATCHED
optimizer_policy_amendment_authority: experiments/protocols/math_v2_1_layer1_redesign_v1/canary_only_authorization_decision.json
next_canary_milestone: math_v2_1_a3_seed81_low_cost_canary_v1
next_canary_attempt_id: math_v2_1_memory_A3_seed81_canary_attempt1
last_autonomous_milestone: math_v2_1_layer1_preexecution_repair_v1
autonomous_user_authorization_received: true
autonomous_token_authorization: 40000000
autonomous_tokens_consumed: 1167499
autonomous_tokens_remaining: 38832501
token_accounting_policy: RESERVATION_V2
prior_accounting_stop_resolution: RESOLVED_BY_USER_ACCOUNTING_POLICY_V2
autonomous_authorization_status: CANARY_CONSUMED_CLOSED_USER_SCOPE_COMPLETE
pilot_search_complete: false
pilot_validation_complete: false
pilot_final_status: NOT_RUN_USER_DEFERRED
task_stop_reason: USER_STOP_AFTER_CANARY
optimizer_nonthinking_wire_confirmed: YES_EQUIVALENT_EVIDENCE
last_candidate_contract_audit: math_v2_1_proposal_semantic_postmortem_v1
pilot_scientific_interpretation: NOT_ESTIMATED
candidate_contract_audit_status: COMPLETED_ZERO_API
next_search_contract_alignment: BOUNDED_MEMORY_CANARY_PASS
current_layer1_backend: LAYER1_BOUNDED_MEMORY_SEARCH_V2
next_pilot_authorized: false
next_validation_authorized: false
last_completed_pilot_milestone: math_v2_1_a1_seed81_low_cost_pilot_v4
historical_pilot_status:
  search_complete: true
  validation_complete: true
  final_status: COMPLETE
  interpretation: ZERO_INTERVENTION_NO_ADMISSIBLE_CANDIDATE
new_pilot_search_complete: false
new_pilot_validation_complete: false
new_pilot_final_status: NOT_RUN_USER_DEFERRED
current_memory_policy: structured_action_failure_memory_v3
current_configuration_arm: A3
current_pattern_enabled: false
current_memory_enabled: true
memory_policy_amendment: math_v2_1_shared_risk_memory_amendment_v1
memory_policy_amendment_status: FROZEN_ZERO_API
available_opt_in_memory_policy: structured_cross_member_rolling_risk_memory_v4
memory_policy_real_execution_authorized: false
```

## Experiment and engineering DAG

Engineering edges are metadata progression, never efficacy evidence.

```mermaid
flowchart TD
  subgraph era0["BENCHMARK_GENERALIZATION"]
    n159["unified_benchmark_migration_v1<br/>BENCHMARK_MIGRATION<br/>COMPLETED"]
    n160["benchmark_data_freeze_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n161["benchmark_scientific_contract_freeze_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n162["repository_hygiene_alignment_v1<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n182["multibench_dataset_migration_v1<br/>DATASET_FREEZE_AND_PROTOCOL_MIGRATION<br/>PREPARED_NOT_EXECUTED"]
    n183["v2_pattern_memory_multibench_v1<br/>FORMAL_EXPERIMENT<br/>PREPARED_NOT_EXECUTED"]
    n184["math_v2_preexecution_freeze_v1<br/>PREEXECUTION_FREEZE<br/>HOLD"]
    n185["math_v2_preexecution_freeze_v1_1<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n186["math_v2_pattern_memory_v1<br/>FORMAL_EXPERIMENT<br/>PREEXECUTION_FROZEN"]
    n187["math_v2_a1_seed81_real_canary_v1<br/>REAL_CANARY<br/>HOLD"]
    n188["math_v2_preexecution_freeze_v1_2<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n189["math_v2_solver_interface_repair_v1_2<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n190["math_v2_autonomous_accounting_preflight_v1<br/>ZERO_API_AUDIT<br/>HOLD"]
    n191["math_v2_token_accounting_repair_v2<br/>PREEXECUTION_FREEZE<br/>COMPLETED"]
    n192["math_v2_a1_seed81_real_canary_v2<br/>REAL_CANARY<br/>INVALID"]
    n193["math_v2_a1_seed81_pilot_v1<br/>PILOT<br/>HOLD"]
    n194["math_v2_provider_error_mapping_repair_v1<br/>PREEXECUTION_FREEZE<br/>COMPLETED"]
    n195["math_v2_a1_seed81_real_canary_v3<br/>REAL_CANARY<br/>INVALID"]
    n196["math_v2_reference_parser_domain_audit_v1<br/>FORENSIC_AUDIT<br/>COMPLETED"]
    n197["math_v2_answer_domain_amendment_v2<br/>PROTOCOL_FREEZE<br/>PREPARED_NOT_EXECUTED"]
    n198["math_v2_preexecution_freeze_v1_4<br/>PREEXECUTION_FREEZE<br/>PREPARED_NOT_EXECUTED"]
    n199["math_v2_a1_seed81_real_canary_v4<br/>REAL_CANARY<br/>PREPARED_NOT_EXECUTED"]
    n200["math_v2_a1_seed81_pilot_v2<br/>PILOT<br/>PREPARED_NOT_EXECUTED"]
    n202["math_v2_1_answer_domain_amendment_v1<br/>BENCHMARK_MIGRATION<br/>COMPLETED"]
    n203["math_v2_1_preexecution_freeze_v1<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n204["math_v2_1_a1_seed81_real_canary_v1<br/>REAL_CANARY<br/>INVALID"]
    n205["math_v2_1_a1_seed81_pilot_v1<br/>PILOT<br/>SUPERSEDED"]
    n206["math_v2_1_output_interface_repair_v1<br/>ARCHITECTURE_REFACTOR<br/>COMPLETED"]
    n207["math_v2_1_preexecution_freeze_v2<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n208["math_v2_1_a1_seed81_real_canary_v2<br/>REAL_CANARY<br/>INVALID"]
    n209["math_v2_1_a1_seed81_pilot_v2<br/>PILOT<br/>SUPERSEDED"]
    n210["math_v2_1_output_interface_repair_v2<br/>ARCHITECTURE_REFACTOR<br/>COMPLETED"]
    n211["math_v2_1_preexecution_freeze_v3<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n212["math_v2_1_a1_seed81_real_canary_v3<br/>REAL_CANARY<br/>INVALID"]
    n213["math_v2_1_a1_seed81_pilot_v3<br/>PILOT<br/>SUPERSEDED"]
    n214["math_v2_1_solver_output_envelope_repair_v1<br/>ARCHITECTURE_REFACTOR<br/>COMPLETED"]
    n215["math_v2_1_preexecution_freeze_v4<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n216["math_v2_1_a1_seed81_real_canary_v4<br/>REAL_CANARY<br/>INVALID"]
    n217["math_v2_1_a1_seed81_pilot_v4<br/>PILOT<br/>HOLD"]
    n218["math_v2_1_solver_decoding_policy_v1<br/>ARCHITECTURE_REFACTOR<br/>COMPLETED"]
    n219["math_v2_1_preexecution_freeze_v5<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n220["math_v2_1_a1_seed81_real_canary_v5<br/>REAL_CANARY<br/>INVALID"]
    n221["math_v2_1_a1_seed81_pilot_v5<br/>PILOT<br/>HOLD"]
    n222["math_v2_1_prediction_validity_policy_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n223["math_v2_1_preexecution_freeze_v6<br/>PREEXECUTION_FREEZE<br/>PREEXECUTION_FROZEN"]
    n224["math_v2_1_a1_seed81_real_canary_v6<br/>REAL_CANARY<br/>HOLD"]
    n225["math_v2_1_a1_seed81_pilot_v6<br/>PILOT<br/>HOLD"]
    n226["math_v2_1_output_reuse_audit_v1<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n227["math_v2_1_terminal_invalid_recovery_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n228["math_v2_1_low_cost_protocol_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n229["math_v2_1_preexecution_low_cost_v1<br/>PREEXECUTION_FREEZE<br/>COMPLETED"]
    n230["math_v2_1_a1_seed81_low_cost_canary_v1<br/>REAL_CANARY<br/>INVALID"]
    n231["math_v2_1_a1_seed81_low_cost_pilot_v1<br/>PILOT<br/>HOLD"]
    n232["math_v2_1_optimizer_wire_amendment_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n233["math_v2_1_preexecution_low_cost_v2<br/>PREEXECUTION_FREEZE<br/>COMPLETED"]
    n234["math_v2_1_a1_seed81_low_cost_canary_v2<br/>REAL_CANARY<br/>INVALID"]
    n235["math_v2_1_a1_seed81_low_cost_pilot_v2<br/>PILOT<br/>HOLD"]
    n236["math_v2_1_optimizer_truncation_root_cause_v1<br/>DIAGNOSTIC<br/>HOLD"]
    n237["math_v2_1_preexecution_low_cost_v3<br/>PREEXECUTION_FREEZE<br/>HOLD"]
    n238["math_v2_1_a1_seed81_low_cost_canary_v3<br/>REAL_CANARY<br/>HOLD"]
    n239["math_v2_1_a1_seed81_low_cost_pilot_v3<br/>PILOT<br/>HOLD"]
    n240["math_v2_1_autonomous_constitution_v1<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n241["math_v2_1_preexecution_low_cost_v4<br/>PREEXECUTION_FREEZE<br/>COMPLETED"]
    n242["math_v2_1_a1_seed81_low_cost_canary_v4<br/>REAL_CANARY<br/>COMPLETED"]
    n243["math_v2_1_a1_seed81_low_cost_pilot_v4<br/>PILOT<br/>COMPLETED"]
    n244["math_v2_1_proposal_semantic_postmortem_v1<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n245["math_v2_1_layer1_gepa_flow_audit_v1<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n246["math_v2_1_layer1_redesign_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n247["math_v2_1_preexecution_low_cost_v5<br/>PREEXECUTION_FREEZE<br/>INVALID"]
    n248["math_v2_1_a1_seed81_low_cost_canary_v5<br/>REAL_CANARY<br/>HOLD"]
    n249["math_v2_1_a1_seed81_low_cost_pilot_v5<br/>PILOT<br/>HOLD"]
    n250["math_v2_1_layer1_preexecution_repair_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n251["math_v2_1_preexecution_low_cost_v6<br/>PREEXECUTION_FREEZE<br/>COMPLETED"]
    n252["math_v2_1_a1_seed81_low_cost_canary_v6<br/>REAL_CANARY<br/>COMPLETED"]
    n253["math_v2_1_layer1_panel_feedback_audit_v1<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n254["math_v2_1_layer1_memory_amendment_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n255["math_v2_1_memory_preexecution_v1<br/>PREEXECUTION_FREEZE<br/>COMPLETED"]
    n256["math_v2_1_a3_seed81_low_cost_canary_v1<br/>REAL_CANARY<br/>COMPLETED"]
    n257["math_v2_1_shared_risk_memory_amendment_v1<br/>PROTOCOL_FREEZE<br/>COMPLETED"]
    n258["math_v2_1_pattern_aware_refactor_v1<br/>PROTOCOL_FREEZE<br/>DRAFT"]
    n259["math_v2_1_pattern_preexecution_v1<br/>PREEXECUTION_FREEZE<br/>DRAFT"]
    n260["math_v2_1_a4_seed81_pattern_canary_v1<br/>REAL_CANARY<br/>DRAFT"]
  end
  subgraph era1["FORMAL_V3_V4"]
    n50["gepa_layer2_local_to_team_transfer_diagnostic_v4<br/>DIAGNOSTIC<br/>STATUS_UNRESOLVED"]
    n51["gepa_saturation_comparison_v3<br/>FORMAL_EXPERIMENT<br/>STATUS_UNRESOLVED"]
    n87["formal_saturation_v3_attempt2_preexecution_20260928<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n88["formal_saturation_v3_attempt2_validation50_refreeze_20260928<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n89["formal_saturation_v3_preexecution_closure_20260928<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n90["formal_saturation_v3_zero_api_incident_20260927<br/>FORENSIC_AUDIT<br/>STATUS_UNRESOLVED"]
    n91["formal_v3_attempt3_json_repair_preexecution_20260929<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n92["formal_v3_attempt3_last_mile_hardening_20260929<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n93["formal_v3_attempt3_seed80_preliminary_20260930<br/>FORMAL_EXPERIMENT<br/>STATUS_UNRESOLVED"]
    n94["formal_v3_attempt4_concurrent_execution_preexecution_20260930<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n120["stage0_formal_v3_telemetry_preflight_20260928<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n142["v4_current_source_attempt2_preexecution_20260928<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n147["v4_preexecution_closure_20260927<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n148["v4_seed81_attempt2_execution_20260928<br/>PILOT<br/>INVALID"]
    n165["formal_v3_attempt2_seed80_native_abort_20260929<br/>FORENSIC_AUDIT<br/>INVALID"]
    n170["v4_seed81_attempt2_forensics_20260928<br/>FORENSIC_AUDIT<br/>STATUS_UNRESOLVED"]
    n171["v4_seed81_attempt3_execution_20260928<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n174["gepa_saturation_comparison_v3_seed80_native_attempt2<br/>INVALID_ATTEMPT<br/>INVALID"]
    n175["gepa_saturation_comparison_v3_seed80_layer2_attempt3<br/>FORMAL_EXPERIMENT<br/>COMPLETED"]
    n176["gepa_saturation_comparison_v3_seed80_native_attempt3<br/>FORMAL_EXPERIMENT<br/>COMPLETED"]
    n177["gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2<br/>INVALID_ATTEMPT<br/>INVALID"]
    n178["gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt3<br/>DIAGNOSTIC<br/>COMPLETED"]
  end
  subgraph era2["GEPA_TWO_LAYER"]
    n7["gepa_candidate_selection_breadth<br/>PILOT<br/>COMPLETED"]
    n8["gepa_critic_gate_audit<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n25["primary_responsibility_persistent_realizability_v1<br/>PILOT<br/>IMPLEMENTED_NOT_EXECUTED"]
    n26["sequential_symmetry_breaking_online_pilot_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n27["accepted_local_mutation_team_transfer_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n28["accepted_local_mutation_team_transfer_v2<br/>PILOT<br/>COMPLETED"]
    n29["accepted_mutation_compositional_state_graph_v1<br/>PILOT<br/>COMPLETED"]
    n30["accepted_mutation_exact_state_graph_v2<br/>PILOT<br/>COMPLETED"]
    n31["accepted_mutation_profile_materialization_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n33["gepa_layer2_real_canary_v2<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n34["gepa_saturation_comparison_v2<br/>FORMAL_EXPERIMENT<br/>STATUS_UNRESOLVED"]
    n35["level_b_gepa_local_acceptance_rate_pilot_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n36["level_b_gepa_real_canary_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n37["level_b_gepa_real_canary_v2<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n38["local_gepa_acceptance_rate_pilot_phase_b_v2<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n39["local_gepa_acceptance_rate_pilot_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n40["local_gepa_parent_acquisition_v1<br/>PILOT<br/>COMPLETED"]
    n41["seed78_primary_responsibility_ab_v1<br/>PILOT<br/>COMPLETED"]
    n42["sequential_symmetry_breaking_online_pilot_v2<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n46["gepa_candidate_breadth_pilot_20260831<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n47["gepa_layer2_local_to_team_transfer_diagnostic_v1<br/>DIAGNOSTIC<br/>STATUS_UNRESOLVED"]
    n48["gepa_layer2_local_to_team_transfer_diagnostic_v2<br/>DIAGNOSTIC<br/>STATUS_UNRESOLVED"]
    n49["gepa_layer2_local_to_team_transfer_diagnostic_v3<br/>DIAGNOSTIC<br/>STATUS_UNRESOLVED"]
    n76["contract_adapted_rg_gepa_fixed_parent_pilot_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n77["contract_adapted_rg_gepa_fixed_parent_pilot_v2<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n82["final_layer2_semantic_closure<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n96["gepa_canary_runtime_compatibility_fix_20260922<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n97["gepa_layer2_real_canary_post_refactor_v2_20260924<br/>ARCHITECTURE_REFACTOR<br/>STATUS_UNRESOLVED"]
    n98["gepa_layer2_transfer_v2_engineering_abort_20260924<br/>FORENSIC_AUDIT<br/>STATUS_UNRESOLVED"]
    n99["gepa_post_refactor_proposal_contract_audit_20260923<br/>ARCHITECTURE_REFACTOR<br/>STATUS_UNRESOLVED"]
    n100["layer2_pre_pilot_repository_audit_20260921<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n101["layer2_raw_legal_responsibility_correction_20260925<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n102["level_b_gepa_adapter_closure_20260914<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n103["level_b_gepa_reflection_input_attribution_20260914<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n104["level_b_gepa_v2_precanary_closure_20260914<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n110["responsibility_guided_gepa_fixed_parent_pilot_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n112["rg_gepa_durable_ledger_repair_20260909<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n113["rg_gepa_execution_smoke_20260909<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n114["rg_gepa_hypothesis_interface_v2_qualification<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n116["seed78_gepa_differential_audit_20260914<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n117["seed78_gepa_teamminibatch_postmortem_20260913<br/>FORENSIC_AUDIT<br/>STATUS_UNRESOLVED"]
    n124["two_layer_gepa_grounded_refactor_20260910<br/>ARCHITECTURE_REFACTOR<br/>STATUS_UNRESOLVED"]
    n164["final_layer2_semantic_refactor<br/>ARCHITECTURE_REFACTOR<br/>STATUS_UNRESOLVED"]
    n166["layer2_semantic_alignment_audit<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n167["production_architecture_refactor<br/>ARCHITECTURE_REFACTOR<br/>STATUS_UNRESOLVED"]
    n168["saturation_mode_prep<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n169["unified_execution_governance_binding_20260923<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
  end
  subgraph era3["HISTORICAL_V15_V16_AND_EARLIER"]
    n32["common_contract_seed77_p1_e2e_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n44["backend_unified_four_mode_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n52["model_headroom_screening_20260901<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n53["solver_headroom_multimodel_seed65_20260901<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n54["solver_headroom_screening_20260901<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n55["v16_m2e_scoped_patch_probe_20260812<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n56["v16_m2f_feedback_necessity_fixed_candidate_20260813<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n57["v16_m2f_online_mechanism_pilot_seed52_20260813<br/>DIAGNOSTIC<br/>STATUS_UNRESOLVED"]
    n58["v16_module2_candidate_design_fixed_trajectory_isolation_20260811<br/>ABLATION<br/>STATUS_UNRESOLVED"]
    n59["v16_module2_compute_matched_efficacy_20260813<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n61["v15_bottleneck_isolation_offline_audit_seed48_50_20260811<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n62["v15_coverage_fragmentation_consensus_audit_seed48_50_20260811<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n63["v16_generic_vs_m20_fixed_parent_probe_20260812<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n64["v16_m20_collateral_structure_audit_20260812<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n65["v16_m2d_raw_responsibility_minimal_edit_fixed_parent_probe_20260812<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n66["v16_m2f_candidate_specific_compatibility_repair_probe_20260812<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n67["v16_m2f_critical_competence_exchange_audit_20260812<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n68["v16_residual_diagnosis_fixed_parent_probe_20260812<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n69["v16_responsibility_coherence_audit_20260812<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n70["backend_branch_consolidation<br/>ARCHITECTURE_REFACTOR<br/>STATUS_UNRESOLVED"]
    n71["common_contract_seed77_sensitivity_20260907<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n72["common_solver_contract_v1_prep_20260906<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n73["common_solver_contract_v1_replay_20260906<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n74["common_solver_contract_v1_seed77_prep_20260907<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n75["common_solver_contract_v1_seed77_replay_20260907<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n78["cross_method_cost_accounting_20260907<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n79["cross_repo_p0_parity_audit_20260906<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n80["diversity_writeback_cost_mechanism_audit_20260908<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n81["execution_harness_refreeze_20260922<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n83["final_method_agent_clustering_20260803<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n84["final_method_code_audit_20260731<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n85["final_method_complete_20260803<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n86["final_pre_experiment_freeze_20260922<br/>PROTOCOL_FREEZE<br/>STATUS_UNRESOLVED"]
    n95["gate_runs<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n105["matched_gpt4omini_seed42_20260725<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n106["post_refactor_readiness_20260923<br/>ARCHITECTURE_REFACTOR<br/>STATUS_UNRESOLVED"]
    n107["pre_authorization_startup_identity_audit_20260922<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n108["pre_run_full_stack_hardening_20260924<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n109["qwen37_flash_qualification_20260805<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n111["responsibility_mechanism_history_audit<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n115["saturation_mode_refactor_20260922<br/>ARCHITECTURE_REFACTOR<br/>STATUS_UNRESOLVED"]
    n118["seed78_precanary_interface_closure_20260914<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n119["solver_stage_attribution_closure_20260915<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n121["static_checkpoint_restore_repair_20260901<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n122["strict_v2_disambiguation_s345_20260803<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n123["strict_v2_s345_20260803<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n125["v11_full_seed43_32updates_20260727<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n126["v12_pilot_seed46_20260806<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n127["v14_formal_seed44_20260809<br/>FORMAL_EXPERIMENT<br/>STATUS_UNRESOLVED"]
    n128["v14_pilot_seed46_20260808<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n129["v14_qwen3_14b_seed46_20260809<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n130["v15_three_seed_development_formal_20260811<br/>FORMAL_EXPERIMENT<br/>STATUS_UNRESOLVED"]
    n139["v3_preexecution_fault_campaign_20260927<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n140["v3_scientific_blocker_decision_closure_20260927<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n141["v4_baseline_full_seed42_20260726<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n143["v4_full_highfreq_seed42_20260726<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n144["v4_gpt4omini_seed44_32updates_20260727<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n145["v4_gpt4omini_seed45_32updates_20260728<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n146["v4_independent_seed42_20260726<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n149["v4_v5_gpt4omini_seed45_matched_pair_20260728<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n150["v5_gpt4omini_seed44_32updates_20260727<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n151["v5_gpt4omini_seed44_final7_rejection_audit_20260727<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n152["v5_gpt4omini_seed45_32updates_20260728<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n153["v6_seed44_control_32updates_20260729<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n154["v6_seed44_memory_32updates_20260729<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n155["v6_seed45_control_32updates_20260729<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n156["v6_seed45_memory_32updates_20260729<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n157["v7_frontier_seed46_stage1_20260730<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n163["common_contract_seed77_p1_e2e_prep_20260907<br/>PREEXECUTION_FREEZE<br/>STATUS_UNRESOLVED"]
    n172["v6_seed44_control_memory_pair_20260729<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n173["v6_seed45_control_memory_pair_20260729<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
  end
  subgraph era4["UNIFIED_TEAM_PROMPT_SEARCH"]
    n158["unified_team_prompt_search_refactor_v1<br/>ARCHITECTURE_REFACTOR<br/>COMPLETED"]
    n179["unified_team_prompt_search_v2_method_refactor<br/>ARCHITECTURE_REFACTOR<br/>IMPLEMENTED_NOT_EXECUTED"]
    n180["v2_pattern_memory_factorial_v1_preexecution_audit<br/>ZERO_API_AUDIT<br/>HOLD"]
    n181["v2_pattern_memory_factorial_v1_1<br/>FORMAL_EXPERIMENT<br/>SUPERSEDED"]
    n201["unified_semantic_contract_v2_1<br/>ARCHITECTURE_REFACTOR<br/>IMPLEMENTED_NOT_EXECUTED"]
  end
  subgraph era5["V17_V18_MEMBER_AWARE"]
    n0["v17_formal_5arm_3seed<br/>FORMAL_EXPERIMENT<br/>COMPLETED"]
    n1["v17_failure_decomposition<br/>FORENSIC_AUDIT<br/>COMPLETED"]
    n2["v17_module1_2x2_isolation<br/>ABLATION<br/>COMPLETED"]
    n3["v17_hybrid_target_allocation<br/>PILOT<br/>COMPLETED"]
    n4["v18_online_accumulation<br/>PILOT<br/>COMPLETED"]
    n5["v18_writeback_quality_audit<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n6["v18_m2f_trigger_extension<br/>PILOT<br/>COMPLETED"]
    n9["v18_safety_only_counterfactual<br/>PILOT<br/>COMPLETED"]
    n10["v18_safety_only_prospective<br/>PILOT<br/>COMPLETED"]
    n11["v18_historical_teacher_safety<br/>PILOT<br/>COMPLETED"]
    n12["v18_shadow_raw_critic<br/>PILOT<br/>COMPLETED"]
    n13["v18_teacher_critic_four_arm<br/>PILOT<br/>COMPLETED"]
    n14["v18_no_semantic_critic_online<br/>PILOT<br/>COMPLETED"]
    n15["v18_no_semantic_critic_online_seed69_70_extension<br/>PILOT<br/>COMPLETED"]
    n16["v18_no_semantic_critic_transfer_decomposition<br/>DIAGNOSTIC<br/>COMPLETED"]
    n17["v18_no_semantic_critic_candidate_ranking_audit<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n18["v18_qwen3_8b_no_semantic_critic_light_replication<br/>PILOT<br/>COMPLETED"]
    n19["diversity_matrix_d0_d5<br/>ABLATION<br/>COMPLETED"]
    n20["diversity_matrix_split_balance_audit<br/>ZERO_API_AUDIT<br/>COMPLETED"]
    n21["anti_overfitting_shadow_gate_v1<br/>PILOT<br/>PREPARED_NOT_EXECUTED"]
    n22["vote_aligned_generic_shadow_pilot_v1<br/>PILOT<br/>COMPLETED"]
    n23["vote_aligned_seed75_static_control<br/>PILOT<br/>COMPLETED"]
    n24["vote_aligned_confirmatory_seed76_77_v1<br/>PILOT<br/>COMPLETED"]
    n43["anti_overfitting_split_v1<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n45["diversity_matrix_d0_d5_20260903<br/>ABLATION<br/>STATUS_UNRESOLVED"]
    n60["v18_revision_parity_semantics_audit_20260824<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n131["v17_conversion_aware_hybrid_5parent_pilot_20260822<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n132["v17_conversion_aware_hybrid_pilot_20260822<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n133["v17_conversion_priority_hybrid_three_arm_pilot_20260822<br/>PILOT<br/>STATUS_UNRESOLVED"]
    n134["v17_conversion_target2_differentiation_audit_20260822<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n135["v17_hybrid_recovered_update_conversion_audit_20260822<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n136["v17_w1_target_allocation_mechanism_audit_20260820<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n137["v18_harmful_commit_m2f_repair_pilot_20260824<br/>ZERO_API_AUDIT<br/>STATUS_UNRESOLVED"]
    n138["v18_trajectory_gain_loss_decomposition_20260824<br/>PILOT<br/>STATUS_UNRESOLVED"]
  end
  n0 -->|audit_of| n1
  n1 -->|followup_of| n2
  n2 -->|derived_from| n3
  n3 -->|derived_from| n4
  n4 -->|audit_of| n5
  n5 -->|followup_of| n6
  n5 -->|followup_of| n7
  n7 -->|audit_of| n8
  n8 -->|followup_of| n9
  n9 -->|derived_from| n10
  n10 -->|followup_of| n11
  n4 -->|audit_of| n11
  n8 -->|derived_from| n12
  n11 -->|followup_of| n12
  n12 -->|derived_from| n13
  n11 -->|followup_of| n13
  n13 -->|derived_from| n14
  n14 -->|followup_of| n15
  n15 -->|audit_of| n16
  n16 -->|audit_of| n17
  n17 -->|followup_of| n18
  n15 -->|derived_from| n19
  n19 -->|audit_of| n20
  n20 -->|derived_from| n21
  n21 -->|derived_from| n22
  n22 -->|followup_of| n23
  n22 -->|followup_of| n24
  n23 -->|followup_of| n24
  n24 -->|derived_from| n25
  n25 -->|derived_from| n26
  n38 -->|derived_from| n27
  n27 -->|derived_from| n28
  n28 -->|derived_from| n29
  n29 -->|derived_from| n30
  n30 -->|derived_from| n31
  n24 -->|derived_from| n32
  n37 -->|derived_from| n35
  n102 -->|derived_from| n36
  n36 -->|derived_from| n37
  n103 -->|derived_from| n37
  n40 -->|derived_from| n38
  n37 -->|derived_from| n39
  n39 -->|derived_from| n40
  n22 -->|derived_from| n41
  n25 -->|derived_from| n41
  n51 -->|refactor_of| n158
  n158 -->|migration_of| n159
  n159 -->|freeze_of| n160
  n160 -->|freeze_of| n161
  n161 -->|audit_of| n162
  n51 -->|replication_of| n174
  n51 -->|replication_of| n175
  n51 -->|replication_of| n176
  n50 -->|diagnostic_of| n177
  n50 -->|diagnostic_of| n178
  n0 -->|refactor_of| n124
  n124 -->|freeze_of| n34
  n34 -->|derived_from| n51
  n162 -->|refactor_of| n179
  n179 -->|audit_of| n180
  n179 -->|derived_from| n181
  n180 -->|followup_of| n181
  n160 -->|migration_of| n182
  n161 -->|migration_of| n182
  n179 -->|migration_of| n182
  n179 -->|derived_from| n183
  n182 -->|derived_from| n183
  n182 -->|freeze_of| n184
  n179 -->|freeze_of| n184
  n184 -->|resolves_blockers_of| n185
  n185 -->|derived_from| n186
  n185 -->|derived_from| n187
  n185 -->|repairs_operational_blocker_of| n188
  n187 -->|repairs_operational_blocker_of| n188
  n188 -->|freeze_of| n189
  n188 -->|audit_of| n190
  n190 -->|resolves_blockers_of| n191
  n191 -->|derived_from| n192
  n192 -->|derived_from| n193
  n192 -->|repairs_operational_blocker_of| n194
  n194 -->|derived_from| n195
  n195 -->|audit_of| n196
  n196 -->|derived_from| n197
  n197 -->|derived_from| n198
  n198 -->|derived_from| n199
  n199 -->|derived_from| n200
  n198 -->|derived_from| n201
  n201 -->|derived_from| n202
  n197 -->|derived_from| n202
  n202 -->|derived_from| n203
  n203 -->|derived_from| n204
  n204 -->|derived_from| n205
  n204 -->|derived_from| n206
  n206 -->|derived_from| n207
  n207 -->|derived_from| n208
  n208 -->|derived_from| n209
  n208 -->|derived_from| n210
  n210 -->|derived_from| n211
  n211 -->|derived_from| n212
  n212 -->|derived_from| n213
  n212 -->|derived_from| n214
  n214 -->|derived_from| n215
  n215 -->|derived_from| n216
  n216 -->|derived_from| n217
  n216 -->|derived_from| n218
  n218 -->|derived_from| n219
  n219 -->|derived_from| n220
  n220 -->|derived_from| n221
  n220 -->|derived_from| n222
  n222 -->|derived_from| n223
  n223 -->|derived_from| n224
  n224 -->|derived_from| n225
  n224 -->|derived_from| n226
  n226 -->|derived_from| n227
  n227 -->|derived_from| n228
  n228 -->|derived_from| n229
  n229 -->|derived_from| n230
  n230 -->|derived_from| n231
  n230 -->|derived_from| n232
  n232 -->|derived_from| n233
  n233 -->|derived_from| n234
  n234 -->|derived_from| n235
  n234 -->|derived_from| n236
  n236 -->|derived_from| n237
  n237 -->|derived_from| n238
  n238 -->|derived_from| n239
  n236 -->|derived_from| n240
  n240 -->|derived_from| n241
  n241 -->|derived_from| n242
  n242 -->|derived_from| n243
  n243 -->|audit_of| n244
  n244 -->|audit_of| n245
  n245 -->|derived_from| n246
  n246 -->|derived_from| n247
  n247 -->|derived_from| n248
  n248 -->|derived_from| n249
  n247 -->|derived_from| n250
  n250 -->|derived_from| n251
  n251 -->|derived_from| n252
  n252 -->|audit_of| n253
  n253 -->|derived_from| n254
  n254 -->|derived_from| n255
  n255 -->|derived_from| n256
  n254 -->|derived_from| n257
  n257 -->|derived_from| n258
  n258 -->|derived_from| n259
  n259 -->|derived_from| n260
```

## Archived branches and unresolved evidence

| Node | Era | Lifecycle | Scientific classifier |
|---|---|---|---|
| v17_formal_5arm_3seed | V17_V18_MEMBER_AWARE | COMPLETED | MULTI_CONTRAST_MIXED |
| v17_failure_decomposition | V17_V18_MEMBER_AWARE | COMPLETED | TARGET_CONCENTRATION_ASSOCIATED_WITH_MEMBER_TRANSFER_REGRESSION |
| v17_module1_2x2_isolation | V17_V18_MEMBER_AWARE | COMPLETED | TARGET_ALLOCATION_DOMINANT |
| v17_hybrid_target_allocation | V17_V18_MEMBER_AWARE | COMPLETED | HYBRID_THROUGHPUT_ONLY |
| v18_online_accumulation | V17_V18_MEMBER_AWARE | COMPLETED | LONGITUDINAL_ACCUMULATION_WITH_VOTE_CONVERSION |
| v18_writeback_quality_audit | V17_V18_MEMBER_AWARE | COMPLETED | COMMON_SAFE_FEASIBLE_SET_QUALITY_GAP_WITH_EXISTING_TRAIN_VOTE_LOSS_RISK_SIGNAL |
| v18_m2f_trigger_extension | V17_V18_MEMBER_AWARE | COMPLETED | EXTENDED_M2F_TRIGGER_NOT_SUPPORTED |
| gepa_candidate_selection_breadth | GEPA_TWO_LAYER | COMPLETED | CANDIDATE_SELECTION_NOT_PRIMARY__BREADTH_NOT_EVALUATED |
| gepa_critic_gate_audit | GEPA_TWO_LAYER | COMPLETED | PRE_STUDENT_CRITIC_GATE_BOTTLENECK_CONFIRMED |
| v18_safety_only_counterfactual | V17_V18_MEMBER_AWARE | COMPLETED | SAFETY_ONLY_REACH_BOUNDS_ONLY |
| v18_safety_only_prospective | V17_V18_MEMBER_AWARE | COMPLETED | NO_CLEAR_SIGNAL |
| v18_historical_teacher_safety | V17_V18_MEMBER_AWARE | COMPLETED | STABLE_PATTERN_NOT_DISCRIMINATIVE_FOR_CANONICAL_BLOCKING |
| v18_shadow_raw_critic | V17_V18_MEMBER_AWARE | COMPLETED | CRITIC_OVER_FILTERING_CAUSALLY_SUPPORTED |
| v18_teacher_critic_four_arm | V17_V18_MEMBER_AWARE | COMPLETED | C_NO_SEMANTIC_CRITIC |
| v18_no_semantic_critic_online | V17_V18_MEMBER_AWARE | COMPLETED | ONLINE_THROUGHPUT_AND_VOTE_SUPPORTED |
| v18_no_semantic_critic_online_seed69_70_extension | V17_V18_MEMBER_AWARE | COMPLETED | NO_CLEAR_ONLINE_ADVANTAGE |
| v18_no_semantic_critic_transfer_decomposition | V17_V18_MEMBER_AWARE | COMPLETED | ACCEPTED_UPDATE_CROSS_SPLIT_AND_PLURALITY_CONVERSION_QUALITY |
| v18_no_semantic_critic_candidate_ranking_audit | V17_V18_MEMBER_AWARE | COMPLETED | COUNTERFACTUAL_ALTERNATIVE_VALIDATION_UNOBSERVED |
| v18_qwen3_8b_no_semantic_critic_light_replication | V17_V18_MEMBER_AWARE | COMPLETED | FRONTEND_REPLICATED__TRANSFER_UNSTABLE__TRAIN_RANKING_NOT_IMPLICATED |
| diversity_matrix_d0_d5 | V17_V18_MEMBER_AWARE | COMPLETED | GENERIC_POSITIVE__D2_TIED_BEST__W1_NEGATIVE__RCE_NOT_SUPPORTED |
| diversity_matrix_split_balance_audit | V17_V18_MEMBER_AWARE | COMPLETED | SPLIT_IMBALANCE_AND_METHOD_INTERACTION_SUPPORTED |
| anti_overfitting_shadow_gate_v1 | V17_V18_MEMBER_AWARE | PREPARED_NOT_EXECUTED | PHASE_A_EXECUTION_READY__PHASE_B_RUNNING |
| vote_aligned_generic_shadow_pilot_v1 | V17_V18_MEMBER_AWARE | COMPLETED | NO_CLEAR_SIGNAL |
| vote_aligned_seed75_static_control | V17_V18_MEMBER_AWARE | COMPLETED | P1_ABSOLUTE_MEMBER_GAIN__ENSEMBLE_STRUCTURE_POSITIVE |
| vote_aligned_confirmatory_seed76_77_v1 | V17_V18_MEMBER_AWARE | COMPLETED | CONFIRMATORY_REPLICATION_NOT_SUPPORTED |
| primary_responsibility_persistent_realizability_v1 | GEPA_TWO_LAYER | IMPLEMENTED_NOT_EXECUTED | ONLINE_BINDING_IMPLEMENTED__EXECUTION_NOT_FROZEN |
| sequential_symmetry_breaking_online_pilot_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | API_AUTHORIZATION_PENDING |
| accepted_local_mutation_team_transfer_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | PREREGISTERED_NOT_EXECUTED |
| accepted_local_mutation_team_transfer_v2 | GEPA_TWO_LAYER | COMPLETED | LOCAL_IMPROVEMENT_OFTEN_FAILS_TO_CHANGE_TEAM_OUTCOME |
| accepted_mutation_compositional_state_graph_v1 | GEPA_TWO_LAYER | COMPLETED | BOUNDED_AUDIT_COMPLETE_EXACT_GRAPH_NOT_RECONSTRUCTIBLE |
| accepted_mutation_exact_state_graph_v2 | GEPA_TWO_LAYER | COMPLETED | CACHE_ONLY_PROFILE_RECOVERY_INCOMPLETE |
| accepted_mutation_profile_materialization_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| common_contract_seed77_p1_e2e_v1 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | NOT_RUN |
| gepa_layer2_real_canary_v2 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| gepa_saturation_comparison_v2 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| level_b_gepa_local_acceptance_rate_pilot_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| level_b_gepa_real_canary_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| level_b_gepa_real_canary_v2 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| local_gepa_acceptance_rate_pilot_phase_b_v2 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | PREREGISTERED_NOT_EXECUTED |
| local_gepa_acceptance_rate_pilot_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | PARENT_CATALOG_INSUFFICIENT |
| local_gepa_parent_acquisition_v1 | GEPA_TWO_LAYER | COMPLETED | VALID |
| seed78_primary_responsibility_ab_v1 | GEPA_TWO_LAYER | COMPLETED | HOLD_NO_SCHEDULER_INTERVENTION_EVALUABLE |
| sequential_symmetry_breaking_online_pilot_v2 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| anti_overfitting_split_v1 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| backend_unified_four_mode_v1 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| diversity_matrix_d0_d5_20260903 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| gepa_candidate_breadth_pilot_20260831 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| gepa_layer2_local_to_team_transfer_diagnostic_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| gepa_layer2_local_to_team_transfer_diagnostic_v2 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| gepa_layer2_local_to_team_transfer_diagnostic_v3 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| gepa_layer2_local_to_team_transfer_diagnostic_v4 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| gepa_saturation_comparison_v3 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| model_headroom_screening_20260901 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| solver_headroom_multimodel_seed65_20260901 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| solver_headroom_screening_20260901 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_m2e_scoped_patch_probe_20260812 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_m2f_feedback_necessity_fixed_candidate_20260813 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_m2f_online_mechanism_pilot_seed52_20260813 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_module2_candidate_design_fixed_trajectory_isolation_20260811 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_module2_compute_matched_efficacy_20260813 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v18_revision_parity_semantics_audit_20260824 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v15_bottleneck_isolation_offline_audit_seed48_50_20260811 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v15_coverage_fragmentation_consensus_audit_seed48_50_20260811 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_generic_vs_m20_fixed_parent_probe_20260812 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_m20_collateral_structure_audit_20260812 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_m2d_raw_responsibility_minimal_edit_fixed_parent_probe_20260812 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_m2f_candidate_specific_compatibility_repair_probe_20260812 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_m2f_critical_competence_exchange_audit_20260812 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_residual_diagnosis_fixed_parent_probe_20260812 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v16_responsibility_coherence_audit_20260812 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| backend_branch_consolidation | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| common_contract_seed77_sensitivity_20260907 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| common_solver_contract_v1_prep_20260906 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| common_solver_contract_v1_replay_20260906 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| common_solver_contract_v1_seed77_prep_20260907 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| common_solver_contract_v1_seed77_replay_20260907 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| contract_adapted_rg_gepa_fixed_parent_pilot_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| contract_adapted_rg_gepa_fixed_parent_pilot_v2 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| cross_method_cost_accounting_20260907 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| cross_repo_p0_parity_audit_20260906 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| diversity_writeback_cost_mechanism_audit_20260908 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| execution_harness_refreeze_20260922 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| final_layer2_semantic_closure | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| final_method_agent_clustering_20260803 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| final_method_code_audit_20260731 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| final_method_complete_20260803 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| final_pre_experiment_freeze_20260922 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| formal_saturation_v3_attempt2_preexecution_20260928 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| formal_saturation_v3_attempt2_validation50_refreeze_20260928 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| formal_saturation_v3_preexecution_closure_20260928 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| formal_saturation_v3_zero_api_incident_20260927 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| formal_v3_attempt3_json_repair_preexecution_20260929 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| formal_v3_attempt3_last_mile_hardening_20260929 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| formal_v3_attempt3_seed80_preliminary_20260930 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| formal_v3_attempt4_concurrent_execution_preexecution_20260930 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| gate_runs | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| gepa_canary_runtime_compatibility_fix_20260922 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| gepa_layer2_real_canary_post_refactor_v2_20260924 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| gepa_layer2_transfer_v2_engineering_abort_20260924 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| gepa_post_refactor_proposal_contract_audit_20260923 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| layer2_pre_pilot_repository_audit_20260921 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| layer2_raw_legal_responsibility_correction_20260925 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| level_b_gepa_adapter_closure_20260914 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| level_b_gepa_reflection_input_attribution_20260914 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| level_b_gepa_v2_precanary_closure_20260914 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| matched_gpt4omini_seed42_20260725 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| post_refactor_readiness_20260923 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| pre_authorization_startup_identity_audit_20260922 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| pre_run_full_stack_hardening_20260924 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| qwen37_flash_qualification_20260805 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| responsibility_guided_gepa_fixed_parent_pilot_v1 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| responsibility_mechanism_history_audit | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| rg_gepa_durable_ledger_repair_20260909 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| rg_gepa_execution_smoke_20260909 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| rg_gepa_hypothesis_interface_v2_qualification | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| saturation_mode_refactor_20260922 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| seed78_gepa_differential_audit_20260914 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| seed78_gepa_teamminibatch_postmortem_20260913 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| seed78_precanary_interface_closure_20260914 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| solver_stage_attribution_closure_20260915 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| stage0_formal_v3_telemetry_preflight_20260928 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| static_checkpoint_restore_repair_20260901 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| strict_v2_disambiguation_s345_20260803 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| strict_v2_s345_20260803 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| two_layer_gepa_grounded_refactor_20260910 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| v11_full_seed43_32updates_20260727 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v12_pilot_seed46_20260806 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v14_formal_seed44_20260809 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v14_pilot_seed46_20260808 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v14_qwen3_14b_seed46_20260809 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v15_three_seed_development_formal_20260811 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v17_conversion_aware_hybrid_5parent_pilot_20260822 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v17_conversion_aware_hybrid_pilot_20260822 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v17_conversion_priority_hybrid_three_arm_pilot_20260822 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v17_conversion_target2_differentiation_audit_20260822 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v17_hybrid_recovered_update_conversion_audit_20260822 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v17_w1_target_allocation_mechanism_audit_20260820 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v18_harmful_commit_m2f_repair_pilot_20260824 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v18_trajectory_gain_loss_decomposition_20260824 | V17_V18_MEMBER_AWARE | STATUS_UNRESOLVED | Not established |
| v3_preexecution_fault_campaign_20260927 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v3_scientific_blocker_decision_closure_20260927 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v4_baseline_full_seed42_20260726 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v4_current_source_attempt2_preexecution_20260928 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| v4_full_highfreq_seed42_20260726 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v4_gpt4omini_seed44_32updates_20260727 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v4_gpt4omini_seed45_32updates_20260728 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v4_independent_seed42_20260726 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v4_preexecution_closure_20260927 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| v4_seed81_attempt2_execution_20260928 | FORMAL_V3_V4 | INVALID | Not established |
| v4_v5_gpt4omini_seed45_matched_pair_20260728 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v5_gpt4omini_seed44_32updates_20260727 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v5_gpt4omini_seed44_final7_rejection_audit_20260727 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v5_gpt4omini_seed45_32updates_20260728 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v6_seed44_control_32updates_20260729 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v6_seed44_memory_32updates_20260729 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v6_seed45_control_32updates_20260729 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v6_seed45_memory_32updates_20260729 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v7_frontier_seed46_stage1_20260730 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| common_contract_seed77_p1_e2e_prep_20260907 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| final_layer2_semantic_refactor | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| formal_v3_attempt2_seed80_native_abort_20260929 | FORMAL_V3_V4 | INVALID | Not established |
| layer2_semantic_alignment_audit | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| production_architecture_refactor | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| saturation_mode_prep | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| unified_execution_governance_binding_20260923 | GEPA_TWO_LAYER | STATUS_UNRESOLVED | Not established |
| v4_seed81_attempt2_forensics_20260928 | FORMAL_V3_V4 | STATUS_UNRESOLVED | Not established |
| v4_seed81_attempt3_execution_20260928 | FORMAL_V3_V4 | COMPLETED | Not established |
| v6_seed44_control_memory_pair_20260729 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| v6_seed45_control_memory_pair_20260729 | HISTORICAL_V15_V16_AND_EARLIER | STATUS_UNRESOLVED | Not established |
| gepa_saturation_comparison_v3_seed80_native_attempt2 | FORMAL_V3_V4 | INVALID | Not established |
| gepa_saturation_comparison_v3_seed80_layer2_attempt3 | FORMAL_V3_V4 | COMPLETED | Not established |
| gepa_saturation_comparison_v3_seed80_native_attempt3 | FORMAL_V3_V4 | COMPLETED | Not established |
| gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2 | FORMAL_V3_V4 | INVALID | Not established |
| gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt3 | FORMAL_V3_V4 | COMPLETED | Not established |
| v2_pattern_memory_factorial_v1_1 | UNIFIED_TEAM_PROMPT_SEARCH | SUPERSEDED | SUPERSEDED_UNEXECUTED_BBH_DRAFT |
| math_v2_a1_seed81_real_canary_v2 | BENCHMARK_GENERALIZATION | INVALID | SDK_STATUS_ERROR_CONSTRUCTOR_SIGNATURE_MISMATCH |
| math_v2_a1_seed81_real_canary_v3 | BENCHMARK_GENERALIZATION | INVALID | MATH_REFERENCE_PARSER_DOMAIN_MISMATCH |
| math_v2_1_a1_seed81_real_canary_v1 | BENCHMARK_GENERALIZATION | INVALID | V2_1_MATH_FRESH_PREEXECUTION |
| math_v2_1_a1_seed81_pilot_v1 | BENCHMARK_GENERALIZATION | SUPERSEDED | V2_1_MATH_FRESH_PREEXECUTION |
| math_v2_1_a1_seed81_real_canary_v2 | BENCHMARK_GENERALIZATION | INVALID | V2_1_MATH_OUTPUT_INTERFACE_REPAIR |
| math_v2_1_a1_seed81_pilot_v2 | BENCHMARK_GENERALIZATION | SUPERSEDED | V2_1_MATH_OUTPUT_INTERFACE_REPAIR |
| math_v2_1_a1_seed81_real_canary_v3 | BENCHMARK_GENERALIZATION | INVALID | V2_1_MATH_OUTPUT_INTERFACE_REPAIR |
| math_v2_1_a1_seed81_pilot_v3 | BENCHMARK_GENERALIZATION | SUPERSEDED | V2_1_MATH_OUTPUT_INTERFACE_REPAIR |
| math_v2_1_a1_seed81_real_canary_v4 | BENCHMARK_GENERALIZATION | INVALID | INVALID_OPERATIONAL_FAILURE |
| math_v2_1_a1_seed81_real_canary_v5 | BENCHMARK_GENERALIZATION | INVALID | STOP_SOLVER_DECODING_POLICY_INSUFFICIENT |
| math_v2_1_a1_seed81_low_cost_canary_v1 | BENCHMARK_GENERALIZATION | INVALID | CANARY_OPERATIONAL_FAILURE |
| math_v2_1_a1_seed81_low_cost_canary_v2 | BENCHMARK_GENERALIZATION | INVALID | CANARY_OPTIMIZER_OUTPUT_TRUNCATION |
| math_v2_1_preexecution_low_cost_v5 | BENCHMARK_GENERALIZATION | INVALID | REGISTRY_MUTATED_AFTER_FREEZE_NO_REAL_CALLS |
