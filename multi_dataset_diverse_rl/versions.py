"""Frozen runtime identities. Values are unchanged; historical constants support replay.

Active identity is selected explicitly by the method manifest, never by section
order. Shared compatibility constants retain their original names and values.
"""

# ACTIVE_UNIFIED_IDENTITIES
BENCHMARK_EXPERIMENT_SPLIT_VERSION = "benchmark_experiment_split_v1"
MATH_EXPERIMENT_SPLIT_VERSION = "benchmark_experiment_split_v1_1"
MATH_REFERENCE_EXTRACTOR_VERSION = "MATH_REFERENCE_EXTRACTOR_V1"
MATH_REFERENCE_VALIDITY_VERSION = "MATH_REFERENCE_VALIDITY_V1"
MATH_EXECUTION_BINDING_VERSION = "MATH_EXECUTION_BINDING_V1_2"
MATH_AUTONOMOUS_EXECUTION_BINDING_VERSION = "MATH_EXECUTION_BINDING_V1_3"
MATH_ANSWER_DOMAIN_VERSION = "MATH_ANSWER_DOMAIN_V2"
MATH_SCORABLE_REFERENCE_VERSION = "MATH_SCORABLE_REFERENCE_V2"
MATH_DOMAIN_EXECUTION_BINDING_VERSION = "MATH_EXECUTION_BINDING_V1_4"
MATH_V2_1_EXECUTION_BINDING_VERSION = "MATH_V2_1_EXECUTION_BINDING_V1"
MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION = "MATH_V2_1_EXECUTION_BINDING_V2"
MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION = "MATH_V2_1_EXECUTION_BINDING_V3"
MATH_PREDICTION_VALIDITY_VERSION = "MATH_PREDICTION_VALIDITY_V1"
MATH_SOLVER_DECODING_POLICY_VERSION = "SOLVER_DECODING_POLICY_V1"
MATH_SCORABLE_SPLIT_VERSION = "benchmark_experiment_split_v2"
MATH_TOKEN_ACCOUNTING_VERSION = "MATH_TOKEN_ACCOUNTING_V2"
MATH_EXECUTION_BINDING_PATH = "experiments/execution_bindings/math_v2_execution_v1_2.json"
MATH_SOLVER_INTERFACE_VERSION = "MATH_SOLVER_INTERFACE_V2"
MATH_SOLVER_INTERFACE_V3_VERSION = "MATH_SOLVER_INTERFACE_V3"
MATH_SOLVER_INTERFACE_V4_VERSION = "MATH_SOLVER_INTERFACE_V4"
MATH_SOLVER_INTERFACE_V5_VERSION = "MATH_SOLVER_INTERFACE_V5"
MATH_RUNTIME_READINESS_VERSION = "MATH_RUNTIME_READINESS_V1"
MULTIBENCH_MODEL_BINDING_VERSION = "multibench_qwen3_8b_qwen3_7_flash_v1"
CURRENT_RESEARCH_BENCHMARK_SUITE = ("math", "ifbench", "hotpotqa")
CURRENT_EXPERIMENT_SOLVER_MODEL = "qwen3-8b"
CURRENT_EXPERIMENT_OPTIMIZER_MODEL = "qwen3.7-flash"
CURRENT_EXPERIMENT_PATTERN_MODEL = "qwen3.7-flash"
UNIFIED_TEAM_PROMPT_SEARCH_VERSION = "unified_team_prompt_search_v1"
UNIFIED_GEPA_DERIVED_ENGINE_VERSION = "gepa_derived_v1"
# V1 identities above and below remain frozen replay contracts.
UNIFIED_TEAM_PROMPT_SEARCH_V2_VERSION = "unified_team_prompt_search_v2"
UNIFIED_GEPA_EXPOSURE_V2_VERSION = "gepa_derived_team_candidate_exposure_v2"
UNIFIED_DECOUPLED_ACCEPTANCE_VERSION = "local_survival_team_admission_decoupled_v1"
UNIFIED_VARIABLE_EVIDENCE_VERSION = "variable_pattern_capable_evidence_v1"
UNIFIED_VARIABLE_FEASIBILITY_VERSION = "variable_evidence_feasibility_v1"
UNIFIED_PATTERN_DIAGNOSTIC_VERSION = "pattern_diagnostic_v1"
UNIFIED_STRUCTURED_MEMORY_VERSION = "structured_agent_memory_v1"
# Semantic-contract amendment; V1/V2 identities remain replay contracts.
UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION = "unified_team_prompt_search_v2_1"
UNIFIED_FOCUSED_PATTERN_VERSION = "focused_pattern_diagnostic_v2"
UNIFIED_FOCUSED_EVIDENCE_VERSION = "single_mechanism_evidence_v2"
UNIFIED_EXPERIENCE_MEMORY_VERSION = "strategy_experience_memory_v2"
UNIFIED_COMPETENCE_TRANSITION_VERSION = "initial_competence_team_gain_v2"
UNIFIED_PLURALITY_RESPONSIBILITY_VERSION = "plurality_raw_responsibility_v1"
UNIFIED_TARGET_POLICY_VERSION = "responsibility_failure_discount_v1"
UNIFIED_FEASIBILITY_POLICY_VERSION = "v4_exact_evidence_feasibility_v1"
UNIFIED_EVIDENCE_POLICY_VERSION = "role_view_4_4_4_v1"
UNIFIED_EVALUATION_POLICY_VERSION = "team_probe_full_v1"
UNIFIED_TRANSITION_POLICY_VERSION = "common_safe_v1"
UNIFIED_ADAPTIVE_GATE_VERSION = "winner_only_shadow_v1"
UNIFIED_PLURALITY_AGGREGATION_VERSION = "equal_plurality_abstain_v1"
UNIFIED_LLM_AGGREGATION_VERSION = "equal_status_llm_aggregation_v1"
UNIFIED_NULL_MEMORY_VERSION = "null_memory_v1"
UNIFIED_NULL_PATTERN_VERSION = "null_pattern_v1"
UNIFIED_GEPA_ACCEPTANCE_VERSION = "gepa_strict_local_improvement_v1"
UNIFIED_SEARCH_STOP_VERSION = "gepa_strict_local_saturation_v1"
UNIFIED_GLOBAL_STOP_VERSION = "team_epoch_no_commit_v1"

# ACTIVE_BENCHMARK_CONTRACT_IDENTITIES
BENCHMARK_SCIENTIFIC_CONTRACT_VERSION = "benchmark_scientific_contract_freeze_v1"
HOTPOT_TASK_CONTRACT = "HOTPotQA_GEPA_ANSWER_V1"
HOVER_TASK_CONTRACT = "HOVER_GEPA_RETRIEVAL_V1"
IFBENCH_TASK_CONTRACT = "IFBENCH_GEPA_CONSTRAINT_V1"
MATH_TASK_CONTRACT = "MATH_ANSWER_EQUIVALENCE_V1"
PUPA_TASK_CONTRACT = "PUPA_PAPILLON_PRIVACY_V1"
BINARY_PLURALITY_RESPONSIBILITY_VERSION = "binary_plurality_raw_responsibility_v1"
EQUIVALENCE_PLURALITY_VERSION = "equal_equivalence_plurality_consistency_v1"
HOTPOT_RETRIEVAL_HOP_COUNTS = (7, 7)
HOVER_RETRIEVAL_HOP_COUNTS = (7, 7, 10)
HOVER_AGGREGATE_MAX_TITLES = 24
MATH_EVALUATOR_PROCESS_DEADLINE_SECONDS = 8
MATH_EVALUATOR_INNER_TIMEOUT_SECONDS = 3
MATH_EVALUATOR_FLOAT_ROUNDING = 6
MATH_EVALUATOR_NUMERIC_PRECISION = 15
MATH_EVALUATOR_DEPENDENCY_PINS = {
    "math-verify": "0.6.0", "latex2sympy2_extended": "1.0.9", "sympy": "1.14.0",
    "antlr4-python3-runtime": "4.13.2", "mpmath": "1.3.0",
}

# HISTORICAL_TWO_LAYER_IDENTITIES
LOCAL_GEPA_RESULT_SEMANTICS_VERSION = "changed_candidates_only_v1"
LOCAL_GEPA_PROPOSER_CONTRACT_VERSION = "decision_procedure_proposer_v1"
LOCAL_GEPA_REFLECTIVE_DATASET_VERSION = "component_specific_reasoning_evidence_v1"
LOCAL_OPTIMIZER_FIDELITY_LEVEL = "LEVEL_B_API_COMPATIBLE_ADAPTATION"
LOCAL_GEPA_CANDIDATE_COMPONENT = "decision_procedure"
LOCAL_GEPA_ENGINE_ACCEPTANCE_SEMANTICS = "pinned_gepa_v0.1.1_strict_improvement"
NATIVE_FEED_OUTER_CONTRACT_VERSION = "backend_native_feed_request_v1"
LAYER2_RESPONSIBILITY_CONTEXT_VERSION = "layer2_responsibility_context_v1"
GEPA_NATIVE_FEED_VERSION = "GEPA_NATIVE_FEED_V1"
GEPA_LAYER2_RESPONSIBILITY_OVERLAY_VERSION = (
    "GEPA_LAYER2_RESPONSIBILITY_OVERLAY_V1"
)
MARS_NATIVE_FEED_VERSION = "MARS_OFFICIAL_CODE_FEED_V1"
MARS_LAYER2_RESPONSIBILITY_OVERLAY_VERSION = (
    "MARS_LAYER2_RESPONSIBILITY_OVERLAY_V1"
)
LAYER2_EVIDENCE_PACKET_VERSION = "responsibility_evidence_packet_v3_frozen_local_eval"
UNIFIED_EXPERIMENT_ENGINE_VERSION = "backend_neutral_layer1_layer2_engine_v1"
LAYER2_EVIDENCE_SELECTION_POLICY_VERSION = (
    "responsibility_plus_latest_transition_frozen_eval_v2"
)
GEPA_LAYER2_EVIDENCE_BACKEND_VERSION = (
    "GEPA_SEARCH_CORE_WITH_LAYER2_TRANSITION_EVIDENCE_V2"
)
MARS_LAYER2_EVIDENCE_BACKEND_VERSION = (
    "MARS_SEARCH_CORE_WITH_LAYER2_TRANSITION_EVIDENCE_V2"
)
LOCAL_GEPA_TOKEN_EDIT_SIMILARITY_VERSION = "token_edit_similarity_v1"
TEAM_MINIBATCH_CONTRACT_VERSION = "repair_preservation_team_hard_4_4_4_v2"
LAYER2_TEAM_SEARCH_PROTOCOL_VERSION = "two_layer_responsibility_v2"
TEAM_MINIBATCH_DIAGNOSTIC_GATE_VERSION = "TEAM_MINIBATCH_DIAGNOSTIC_GATE_V1"
PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION = (
    "primary_responsibility_persistent_realizability_v2"
)
PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION = (
    "primary_responsibility_feasibility_constrained_v1"
)
LAYER2_RESPONSIBILITY_SOURCE_VERSION = "raw_legal_pre_routing_v1"
PERSISTENT_REALIZABILITY_SEMANTICS_VERSION = "eventual_write_back_realizability_v1"
LAYER2_EVIDENCE_EPOCH_POLICY_VERSION = "replay_same_frozen_packet_v1"
PRIMARY_RESPONSIBILITY_DIRECT_WEIGHT = 4
PRIMARY_RESPONSIBILITY_NEAR_MARGIN_WEIGHT = 2
PRIMARY_RESPONSIBILITY_COVERAGE_WEIGHT = 1

# HISTORICAL_V15_IDENTITIES
METHOD_VERSION = "member_aware_peer_state_v15"
RESPONSIBILITY_VERSION = "counterfactual_vote_margin_responsibility_v1"
SERVICE_ROUTING_VERSION = "single_service_anchor_routing_no_freeze_v2"
TARGET_SELECTION_VERSION = (
    "repairability_adjusted_expected_update_value_wait_coupled_v2"
)
REPAIRABILITY_VERSION = "state_local_branch_failure_discount_v1"
DUAL_TARGET_SEARCH_VERSION = "dual_target_single_commit_search_v1"
TCS_CONTEXT_VERSION = "compact_single_lane_responsibility_context_v1"
CHECKPOINT_VERSION = 25
EXPERIMENTAL_MODULE2_VERSION = (
    "v16_residual_diagnosis_minimal_edit_experimental_v2"
)
CANDIDATE_ACCEPTANCE_VERSION = "fixed_peer_monotone_target_or_vote_v2"
CANDIDATE_SELECTION_VERSION = "common_safe_final_update_v1"
CANDIDATE_ACCEPTANCE_POLICY = "fixed_peer_monotone_target_or_vote"
PROPOSAL_MEMORY_VERSION = "agent_isolated_state_local_proposal_memory_v1"
PRESERVATION_POLICY_VERSION = "diagnostic_only_sample_preservation_v1"
EVALUATION_PROTOCOL_VERSION = "final_active_state_no_validation_v1"
CHECKPOINT_SELECTION_VERSION = "none_final_state_v1"
TEST_ISOLATION_VERSION = "post_training_final_state_test_once_v1"
STUDENT_INVALID_RECOVERY_VERSION = "feedback_retry_then_upstream_regenerate_v1"
MUTABLE_PROMPT_CONTRACT_VERSION = "reasoning_only_no_response_format_v2"
STUDENT_PROMPT_CONTRACT_VERSION = "mutable_reasoning_only_v2"
CANDIDATE_PROTOCOL_FILTER_VERSION = "output_contract_contamination_v2"
MODEL_THINKING_MODE_VERSION = "explicitly_disabled_v1"
LEGACY_DIVERSITY_SOLVER_CONTRACT_ID = "DIVERSITY_SOLVER_CONTRACT_LEGACY_V1"
COMMON_SOLVER_CONTRACT_V1_ID = "COMMON_SOLVER_CONTRACT_V1"
RCRU_VERSION = "responsibility_conditioned_robust_contribution_update_v1"
RESPONSIBILITY_UTILITY_VERSION = "three_lane_responsibility_utility_v1"
COALITION_CONTRIBUTION_VERSION = "leave_one_out_pivotal_contribution_v1"
ROBUST_SUPPORT_VERSION = "paired_positive_no_negative_bootstrap_v2"
MINIMAL_EDIT_VERSION = "token_diff_minimal_edit_v1"
EXPERIMENT_MATRIX_VERSION = "reduced_two_module_ablation_v1"
PROTOCOL_RESOLUTION_VERSION = "reduced_two_module_protocol_resolution_v1"
COMMON_UPDATE_POLICY_VERSION = "common_fixed_peer_monotone_safe_v1"
TARGET_SCORE_DIRECT_WEIGHT = 0.5
TARGET_SCORE_SUPPORT_WEIGHT = 0.3
TARGET_SCORE_UPLIFT_WEIGHT = 0.2
TARGET_SCORE_WAIT_WEIGHT = 0.05

# HISTORICAL_FORMAL_IDENTITIES
LAYER2_EVIDENCE_PACKET_V4_VERSION = "responsibility_evidence_packet_v4_bounded_search_view"
LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION = (
    "raw_universe_exact_transition_bounded_nominal_schedule_v1"
)
LOCAL_GEPA_PROPOSAL_TELEMETRY_VERSION = "proposal_behavior_telemetry_v1"
LOCAL_GEPA_PHASE_B_TELEMETRY_VERSION = "single_state_member_task_telemetry_v2"
LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION = "two_layer_feasibility_bounded_view_v4"
LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION = (
    "raw_positive_score_preselection_exact_minibatch_packet_v1"
)
ACCEPTED_LOCAL_MUTATION_TEAM_TRANSFER_VERSION = "accepted_local_mutation_team_transfer_v1"
TEAM_TRANSFER_DECOMPOSITION_VERSION = "fixed_baseline_read_only_case_mapping_v1"
ACCEPTED_LOCAL_MUTATION_TEAM_TRANSFER_V2_VERSION = "accepted_local_mutation_team_transfer_v2"
TEAM_TRANSFER_DECOMPOSITION_V2_VERSION = "mandatory_full_fixed_baseline_case_mapping_v2"
ACCEPTED_MUTATION_COMPOSITIONAL_STATE_GRAPH_VERSION = "accepted_mutation_compositional_state_graph_v1"
ACCEPTED_MUTATION_EXACT_STATE_GRAPH_VERSION = "accepted_mutation_exact_state_graph_v2"
SEQUENTIAL_SYMMETRY_BREAKING_ONLINE_PILOT_VERSION = (
    "sequential_symmetry_breaking_online_pilot_v1"
)
SATURATION_STOPPING_CONTRACT_VERSION = "backend_neutral_saturation_stopping_v1"
TEAM_EPOCH_SEMANTICS_VERSION = "scheduler_eligible_coverage_v1"
TEAM_EPOCH_SEMANTICS_V4_VERSION = "feasible_parent_scoped_coverage_commit_reset_v1"
