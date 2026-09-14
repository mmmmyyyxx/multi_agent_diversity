"""Frozen protocol identifiers shared by runtime, metadata, and persistence."""

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
LOCAL_GEPA_RESULT_SEMANTICS_VERSION = "changed_candidates_only_v1"
LOCAL_GEPA_PROPOSER_CONTRACT_VERSION = "decision_procedure_proposer_v1"
LOCAL_OPTIMIZER_FIDELITY_LEVEL = "LEVEL_B_API_COMPATIBLE_ADAPTATION"
LOCAL_GEPA_CANDIDATE_COMPONENT = "decision_procedure"
LOCAL_GEPA_ENGINE_ACCEPTANCE_SEMANTICS = "pinned_gepa_v0.1.1_strict_improvement"
TEAM_MINIBATCH_CONTRACT_VERSION = "primary_lane_strict_4_4_4_v1"
# Legacy v14 RCRU identifiers remain frozen for historical replay only.
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

# Experimental Layer-2 target scheduler.  This identity is opt-in and does not
# change the canonical v15 target-selection or repairability versions above.
PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION = (
    "primary_responsibility_persistent_realizability_v1"
)
PERSISTENT_REALIZABILITY_SEMANTICS_VERSION = "eventual_write_back_realizability_v1"
PRIMARY_RESPONSIBILITY_DIRECT_WEIGHT = 4
PRIMARY_RESPONSIBILITY_NEAR_MARGIN_WEIGHT = 2
PRIMARY_RESPONSIBILITY_COVERAGE_WEIGHT = 1
