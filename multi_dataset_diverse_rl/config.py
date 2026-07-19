from dataclasses import asdict, dataclass, field, fields
import argparse

from .config_sections import canonical_config_dict, section_for_field, split_flat_config


DEFAULT_TEMPERATURE = 0.0
DEFAULT_OPTIMIZER_TEMPERATURE = 0.5
DEFAULT_EVALUATOR_TEMPERATURE = 0.0


@dataclass
class _FlatConfigSchema:
    task_type: str = "auto"
    dataset_format: str = "legacy"
    comparison_task_id: str = ""
    experiment_setting: str = ""
    benchmark: str = ""
    answer_format: str = ""

    agent_model: str = "deepseek-chat"
    optimizer_model: str = "deepseek-chat"
    evaluator_model: str = "deepseek-chat"

    train_path: str = "train.jsonl"
    val_path: str = ""
    test_path: str = "test.jsonl"
    train_size: int = 200
    val_size: int = 100
    val_split_ratio: float = 0.2
    test_size: int = 200
    eval_test_each_epoch: bool = False

    agents: int = 5
    init_mode: str = "shared"
    shared_prompt: str = "You are a careful reasoning solver. Produce a compact, explicit reasoning trace, make your decision procedure visible, verify key logic, and give exactly one final answer."
    epochs: int = 2
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    update_every: int = 10
    candidate_eval_batch_size: int = 20
    baseline_only: bool = False

    search_mode: str = "evolutionary_beam"
    reward_mode: str = "vote_useful_diversity"
    candidate_selection_mode: str = "scalar_reward"
    best_state_selection_mode: str = "vote_first"
    beam_size: int = 3
    num_candidates_per_parent: int = 2
    optimizer_parent_concurrency: int = 2
    # Compatibility alias for the historical per-epoch beam rescore.
    beam_refresh_each_epoch: bool = True
    legacy_beam_rescore_each_epoch: bool = True
    homogeneity_overlap_threshold: float = 0.55
    homogeneity_pressure_tie_eps: float = 0.03
    max_homogeneous_cases_per_agent: int = 4
    random_window_cases_per_agent: int = 2
    hard_validity_cases_per_agent: int = 2
    invalid_repair_rate_threshold: float = 0.25

    accuracy_guard_epsilon: float = 0.02
    reward_weight_div_delta: float = 0.3
    reward_weight_invalid_delta: float = 0.5
    reward_weight_vote_delta: float = 0.3
    reward_weight_vote_margin: float = 0.2
    reward_weight_boundary_diversity: float = 0.2
    reward_weight_coverage: float = 0.3
    reward_weight_useful_diversity: float = 0.2
    invalid_guard_epsilon: float = 0.05
    use_baseline_relative_reward: bool = True
    reward_schedule_mode: str = "phase_adaptive"
    reward_diversity_warmup_updates: int = 10
    reward_weight_div_delta_early: float = 0.8
    reward_weight_div_delta_late: float = 0.2
    reward_weight_vote_delta_early: float = 0.4
    reward_weight_vote_delta_late: float = 0.3
    reward_weight_vote_margin_early: float = 0.5
    reward_weight_vote_margin_late: float = 0.25
    reward_weight_boundary_diversity_early: float = 0.3
    reward_weight_boundary_diversity_late: float = 0.2
    reward_weight_coverage_early: float = 0.4
    reward_weight_coverage_late: float = 0.3
    reward_weight_useful_diversity_early: float = 0.5
    reward_weight_useful_diversity_late: float = 0.25
    reward_weight_target_accuracy_early: float = 0.9
    reward_weight_target_accuracy_late: float = 1.0
    accuracy_guard_epsilon_early: float = 0.03
    accuracy_guard_epsilon_late: float = 0.01
    optimizer_architecture: str = "teacher_critic_student"
    teacher_critic_max_rounds: int = 3
    teacher_rewrite_max_count: int = 1
    teacher_question_pass_threshold: float = 0.75
    teacher_critic_direct_pass_threshold: float = 0.75
    teacher_critic_rewrite_threshold: float = 0.50
    teacher_critic_forced_best_threshold: float = 0.60
    tcs_repair_candidates_per_parent: int = 1
    open_exploration_candidates_per_parent: int = 1
    teacher_temperature: float = 0.4
    critic_temperature: float = 0.0
    student_temperature: float = 0.5
    teacher_max_tokens: int = 1200
    critic_max_tokens: int = 1000
    student_max_tokens: int = 1800
    student_json_retry_on_parse_fail: bool = True
    student_json_max_retries: int = 5
    student_json_repair_enabled: bool = True
    student_json_repair_max_tokens: int = 1200
    student_json_repair_temperature: float = 0.0
    student_candidate_schema_mode: str = "compact"
    student_candidate_max_chars_per_field: int = 320
    student_candidate_prompt_max_chars: int = 900
    student_candidate_prompt_soft_max_chars: int = 1100
    student_candidate_prompt_hard_max_chars: int = 1400
    student_force_minified_json: bool = True
    teacher_critic_use_voting_failure: bool = True
    optimizer_fallback_mode: str = "none"
    no_effective_evolution_patience: int = 10
    no_effective_evolution_min_optimizer_candidates: int = 1
    no_effective_evolution_stop_enabled: bool = True

    boundary_selector_enabled: bool = False
    shared_error_metrics_enabled: bool = False
    residual_specialization_enabled: bool = False
    error_dependence_guard_enabled: bool = False
    residual_cycle_guard_enabled: bool = False
    mechanism_trust_region_enabled: bool = False
    specialization_ema: float = 0.20
    specialization_support_shrinkage: float = 3.0
    capability_loss_weight: float = 1.5
    specialization_update_period: int = 2
    capability_affinity_weight: float = 0.25
    capability_coverage_gap_weight: float = 0.25
    pivotal_loss_guard_epsilon: float = 0.0
    shared_error_creation_epsilon: float = 0.02
    behavior_cycle_guard_enabled: bool = True
    behavior_archive_size: int = 16
    behavior_cycle_similarity_threshold: float = 0.95
    behavior_cycle_min_overlap: int = 16
    behavior_cycle_improvement_epsilon: float = 0.01
    behavior_cycle_margin_epsilon: float = 0.05
    prompt_trust_region_enabled: bool = True
    prompt_max_change_ratio: float = 0.45
    prompt_large_shift_warmup_accepts: int = 2
    prompt_large_shift_min_vote_delta: float = 0.02
    baseline_allowed_vote_loss: float = 0.0

    competence_depth_enabled: bool = False
    competence_depth2_aux_enabled: bool = False
    competence_progressive_residual_enabled: bool = False
    competence_floor_low: float = 0.55
    competence_floor_high: float = 0.65
    competence_selector_weight: float = 1.0
    competence_extra_support_shrinkage: float = 3.0
    competence_weight_accuracy_gain: float = 1.0
    competence_weight_accuracy_loss: float = 1.5
    competence_weight_depth2_gain: float = 0.8
    competence_weight_depth2_loss: float = 1.0
    competence_weight_vote_gain_early: float = 0.4
    competence_weight_vote_loss_early: float = 1.0
    competence_schedule_mode: str = "absolute_legacy"
    competence_schedule_version: str = "competence_depth_v1"
    competence_probe_size: int = 0
    competence_probe_seed_offset: int = 7000
    competence_relative_low_delta: float = 0.01
    competence_relative_high_delta: float = 0.06
    competence_schedule_ema: float = 0.50
    competence_schedule_max_step: float = 0.35
    competence_schedule_monotonic: bool = True
    competence_mean_guard_epsilon: float = 0.01
    competence_c1_guard_epsilon: float = 0.01
    competence_c2_guard_epsilon: float = 0.01
    competence_depth1_candidate_guard_enabled: bool = False
    competence_depth1_candidate_guard_epsilon: float = 0.0
    competence_min_effective_specialization_epochs: int = 1
    method_version: str = "legacy"
    target_selector_mode: str = "legacy"
    target_selector_version: str = "legacy"
    beam_policy_version: str = "legacy"
    tcs_candidate_policy_version: str = "legacy"
    mechanism_signature_version: str = "legacy"
    competence_weight_depth1_gain: float = 0.80
    competence_weight_depth1_loss: float = 1.20
    competence_residual_floor: float = 0.30
    catastrophic_target_accuracy_loss_epsilon: float = 0.05
    soft_guard_error_dependence_weight: float = 0.50
    soft_guard_cycle_weight: float = 0.20
    soft_guard_mechanism_shift_weight: float = 0.20
    soft_guard_accuracy_regression_weight: float = 0.50
    mechanism_novelty_bonus_weight: float = 0.20
    active_team_selector_version: str = "legacy"
    candidate_generation_policy_version: str = "legacy"
    joint_refresh_policy_version: str = "legacy"
    representative_probe_policy_version: str = "legacy"
    lineage_policy_version: str = "legacy"
    mechanism_distance_version: str = "legacy"
    mechanism_sequence_distance_weight: float = 0.50
    mechanism_embedding_distance_weight: float = 0.50
    mechanism_near_duplicate_similarity_threshold: float = 0.97
    semantic_niche_merge_threshold: float = 0.88
    semantic_mechanism_novelty_distance: float = 0.12
    behavior_correct_set_weight: float = 0.40
    behavior_rescue_weight: float = 0.30
    behavior_shared_wrong_weight: float = 0.15
    behavior_support_shrinkage: float = 5.0
    team_diversity_mean_behavior_weight: float = 0.45
    team_diversity_min_behavior_weight: float = 0.25
    team_diversity_mechanism_weight: float = 0.20
    team_diversity_rescue_balance_weight: float = 0.10
    joint_team_vote_epsilon_questions: int = 1
    joint_team_mean_epsilon_questions: int = 1
    joint_team_bottom2_epsilon_questions: int = 1
    joint_team_c1_epsilon_questions: int = 1
    joint_team_c2_epsilon_questions: int = 1
    joint_team_per_agent_accuracy_epsilon: float = 0.03
    lineage_provisional_epochs: int = 2
    lineage_commit_epochs: int = 3
    lineage_switch_confirmation_epochs: int = 2
    lineage_mechanism_drift_weight: float = 0.50
    lineage_behavior_drift_weight: float = 0.50
    lineage_soft_drift_threshold: float = 0.35
    lineage_hard_drift_threshold: float = 0.75
    lineage_switch_min_accuracy_gain: float = 0.03
    lineage_switch_min_vote_gain: float = 0.02
    peer_collapse_soft_similarity: float = 0.85
    peer_collapse_hard_similarity: float = 0.97
    validation_stable_specialization_tie_break_enabled: bool = True
    candidate_refill_version: str = "legacy"
    archive_policy_version: str = "legacy"
    joint_quality_filter_version: str = "legacy"
    probe_stability_version: str = "legacy"
    parent_selection_version: str = "legacy"
    candidate_refill_enabled: bool = True
    candidate_refill_max_rounds: int = 2
    candidate_refill_candidates_per_round: int = 2
    candidate_refill_max_unique_candidates_per_parent: int = 6
    candidate_refill_min_safe_non_incumbent: int = 2
    candidate_refill_require_task_repair: bool = True
    candidate_refill_require_distinct_mechanism: bool = True
    candidate_refill_feed_rejection_reasons: bool = True
    candidate_refill_stop_when_requirements_met: bool = True
    candidate_refill_max_solver_calls_per_agent_update: int = 0
    probation_archive_enabled: bool = True
    probation_archive_size_per_agent: int = 1
    probation_archive_ttl_updates: int = 2
    probation_max_accuracy_loss: float = 0.03
    probation_max_c1_loss_questions: int = 1
    probation_max_c2_loss_questions: int = 1
    probation_require_mechanism_novelty: bool = True
    candidate_c1_catastrophic_loss_questions: int = 2
    candidate_c2_catastrophic_loss_questions: int = 2
    qd_archive_size_per_agent: int = 6
    quality_anchor_archive_size: int = 5
    joint_representative_beam_size: int = 3
    joint_refresh_mode: str = "event_driven"
    joint_refresh_on_safe_archive_change: bool = True
    joint_refresh_on_probation_promotion: bool = True
    joint_refresh_on_representative_change: bool = True
    joint_refresh_interval_epochs: int = 2
    joint_refresh_force_final_epoch: bool = True
    joint_refresh_min_new_safe_candidates: int = 1
    joint_refresh_max_dirty_candidates_per_agent: int = 2
    joint_refresh_skip_when_no_dirty_prompt: bool = True
    qd_parent_selection_mode: str = "active_plus_round_robin_niche"
    qd_niche_min_parent_opportunities_per_epoch: int = 1
    probation_parent_enabled: bool = True
    probe_stability_fold_count: int = 2
    probe_stability_seed_offset: int = 9100
    joint_vote_band_questions: int = 1
    joint_mean_band_correct_count: int = 2
    joint_bottom2_band_correct_count: int = 1
    joint_c1_band_questions: int = 1
    joint_c2_band_questions: int = 1
    joint_allowed_vote_loss_questions: int = 1
    joint_allowed_c1_loss_questions: int = 1
    joint_allowed_c2_loss_questions: int = 1
    joint_allowed_total_agent_correct_loss: int = 2
    joint_allowed_bottom2_correct_loss: int = 1
    joint_allowed_per_agent_correct_loss: int = 2
    joint_team_max_active_changes_early: int = 3
    joint_team_max_active_changes_late: int = 2
    joint_team_change_limit_switch_strength: float = 0.30
    joint_team_no_diversification_patience: int = 2
    joint_team_change_limit_relaxation: int = 1
    lineage_commit_required_snapshots: int = 2
    lineage_switch_confirmation_snapshots: int = 2
    qd_readiness_min_distinct_niches: int = 2
    qd_readiness_min_diversity: float = 0.10
    qd_readiness_max_fold_gap: float = 0.15
    residual_specialization_qd_floor: float = 0.15
    behavior_error_overlap_weight: float = 0.15
    behavior_wrong_answer_dispersion_weight: float = 0.15
    behavior_wrong_support_shrinkage: float = 5.0
    min_optimizer_updates_per_agent_per_epoch: int = 1
    target_selector_fairness_enabled: bool = True

    diversity_metric: str = "trace_embedding"
    use_joint_trace_diversity_evaluator: bool = False
    invalid_binary: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    trace_embedding_chunk_words: int = 320
    trace_embedding_chunk_overlap: int = 40

    max_tokens: int = 1000
    optimizer_max_tokens: int = 1400
    evaluator_max_tokens: int = 1200
    temperature: float = DEFAULT_TEMPERATURE
    optimizer_temperature: float = DEFAULT_OPTIMIZER_TEMPERATURE
    evaluator_temperature: float = DEFAULT_EVALUATOR_TEMPERATURE

    out_dir: str = "runs_trace_beam"
    seed: int = 42
    resume_from_checkpoint: bool = False
    max_retries: int = 3
    retry_sleep: float = 1.5
    transient_retry_forever: bool = True
    max_transient_retries: int = 0
    max_retry_backoff: float = 30.0
    llm_call_logging: bool = True
    llm_call_timeout: float = 120.0
    candidate_eval_concurrency: int = 0
    candidate_eval_strategy: str = "random"
    candidate_eval_pool_size: int = 100
    candidate_eval_pool_actual_size: int = 0
    candidate_eval_data_source: str = "optimization_train"
    candidate_eval_total_count: int = 0
    candidate_eval_unique_question_count: int = 0
    candidate_eval_repeats: int = 1
    candidate_eval_seed_offset: int = 1000
    candidate_reuse_recorded_rollouts: bool = True
    candidate_eval_execution_mode: str = "legacy"
    solver_rollout_singleflight: bool = True
    candidate_eval_prompt_dedup: bool = True
    candidate_eval_cache_logging: bool = True
    train_rollout_concurrency: int = 0
    eval_solver_call_concurrency: int = 225
    solver_api_key_env: str = ""
    solver_base_url_env: str = ""
    evaluator_api_key_env: str = ""
    evaluator_base_url_env: str = ""
    vote_tie_break: str = "random"
    aggregation_mode: str = "majority"
    split_integrity_json: str = ""

    def __post_init__(self):
        if not str(self.agent_model or "").strip():
            self.agent_model = "deepseek-chat"
        if not str(self.optimizer_model or "").strip():
            self.optimizer_model = "deepseek-chat"
        if not str(self.evaluator_model or "").strip():
            self.evaluator_model = "deepseek-chat"
        if str(self.method_version) == "v8_stable_qd_lineage":
            self.legacy_beam_rescore_each_epoch = False
            self.teacher_critic_max_rounds = 2
        probability_fields = (
            "specialization_ema",
            "behavior_cycle_similarity_threshold",
            "behavior_cycle_improvement_epsilon",
            "behavior_cycle_margin_epsilon",
            "prompt_max_change_ratio",
            "prompt_large_shift_min_vote_delta",
            "baseline_allowed_vote_loss",
            "pivotal_loss_guard_epsilon",
            "shared_error_creation_epsilon",
            "competence_floor_low",
            "competence_floor_high",
            "competence_relative_low_delta",
            "competence_relative_high_delta",
            "competence_schedule_ema",
            "competence_schedule_max_step",
            "competence_mean_guard_epsilon",
            "competence_c1_guard_epsilon",
            "competence_c2_guard_epsilon",
            "competence_depth1_candidate_guard_epsilon",
            "competence_residual_floor",
            "catastrophic_target_accuracy_loss_epsilon",
            "mechanism_sequence_distance_weight",
            "mechanism_embedding_distance_weight",
            "mechanism_near_duplicate_similarity_threshold",
            "semantic_niche_merge_threshold",
            "semantic_mechanism_novelty_distance",
            "behavior_correct_set_weight",
            "behavior_rescue_weight",
            "behavior_shared_wrong_weight",
            "behavior_error_overlap_weight",
            "behavior_wrong_answer_dispersion_weight",
            "team_diversity_mean_behavior_weight",
            "team_diversity_min_behavior_weight",
            "team_diversity_mechanism_weight",
            "team_diversity_rescue_balance_weight",
            "joint_team_per_agent_accuracy_epsilon",
            "lineage_mechanism_drift_weight",
            "lineage_behavior_drift_weight",
            "lineage_soft_drift_threshold",
            "lineage_hard_drift_threshold",
            "lineage_switch_min_accuracy_gain",
            "lineage_switch_min_vote_gain",
            "peer_collapse_soft_similarity",
            "peer_collapse_hard_similarity",
        )
        for field in probability_fields:
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {value}")
        for field in (
            "specialization_support_shrinkage",
            "capability_loss_weight",
            "capability_affinity_weight",
            "capability_coverage_gap_weight",
            "competence_selector_weight",
            "competence_extra_support_shrinkage",
            "competence_weight_accuracy_gain",
            "competence_weight_accuracy_loss",
            "competence_weight_depth2_gain",
            "competence_weight_depth2_loss",
            "competence_weight_vote_gain_early",
            "competence_weight_vote_loss_early",
            "competence_weight_depth1_gain",
            "competence_weight_depth1_loss",
            "soft_guard_error_dependence_weight",
            "soft_guard_cycle_weight",
            "soft_guard_mechanism_shift_weight",
            "soft_guard_accuracy_regression_weight",
            "mechanism_novelty_bonus_weight",
            "behavior_support_shrinkage",
        ):
            if float(getattr(self, field)) < 0.0:
                raise ValueError(f"{field} must be non-negative")
        for field in (
            "prompt_large_shift_warmup_accepts", "joint_team_vote_epsilon_questions",
            "joint_team_mean_epsilon_questions", "joint_team_bottom2_epsilon_questions",
            "joint_team_c1_epsilon_questions", "joint_team_c2_epsilon_questions",
            "teacher_rewrite_max_count", "tcs_repair_candidates_per_parent",
            "open_exploration_candidates_per_parent", "joint_refresh_interval_epochs",
            "joint_refresh_min_new_safe_candidates", "joint_refresh_max_dirty_candidates_per_agent",
        ):
            if int(getattr(self, field)) < 0:
                raise ValueError(f"{field} must be non-negative")
        for field in ("specialization_update_period", "behavior_archive_size", "behavior_cycle_min_overlap"):
            if int(getattr(self, field)) < 1:
                raise ValueError(f"{field} must be at least 1")
        if float(self.competence_floor_high) <= float(self.competence_floor_low):
            raise ValueError("competence_floor_high must be greater than competence_floor_low")
        if float(self.competence_relative_high_delta) <= float(self.competence_relative_low_delta):
            raise ValueError("competence_relative_high_delta must be greater than competence_relative_low_delta")
        if int(self.competence_probe_size) < 0:
            raise ValueError("competence_probe_size must be non-negative")
        if int(self.competence_min_effective_specialization_epochs) < 1:
            raise ValueError("competence_min_effective_specialization_epochs must be at least 1")
        for field in ("lineage_provisional_epochs", "lineage_commit_epochs", "lineage_switch_confirmation_epochs"):
            if int(getattr(self, field)) < 1:
                raise ValueError(f"{field} must be at least 1")
        if int(self.lineage_commit_epochs) < int(self.lineage_provisional_epochs):
            raise ValueError("lineage_commit_epochs must be >= lineage_provisional_epochs")
        if float(self.lineage_hard_drift_threshold) <= float(self.lineage_soft_drift_threshold):
            raise ValueError("lineage_hard_drift_threshold must be greater than lineage_soft_drift_threshold")
        if float(self.peer_collapse_hard_similarity) <= float(self.peer_collapse_soft_similarity):
            raise ValueError("peer_collapse_hard_similarity must be greater than peer_collapse_soft_similarity")
        if int(self.student_candidate_prompt_hard_max_chars) < int(self.student_candidate_prompt_soft_max_chars):
            raise ValueError("student_candidate_prompt_hard_max_chars must be >= soft max")
        behavior_weight_sum = (
            float(self.behavior_correct_set_weight)
            + float(self.behavior_rescue_weight)
            + float(self.behavior_error_overlap_weight)
            + float(self.behavior_wrong_answer_dispersion_weight)
        )
        if abs(behavior_weight_sum - 1.0) > 1e-9:
            raise ValueError(f"behavior distance weights must sum to 1, got {behavior_weight_sum}")
        nonnegative_integer_fields = (
            "candidate_refill_max_rounds", "candidate_refill_min_safe_non_incumbent",
            "candidate_refill_max_solver_calls_per_agent_update", "probation_archive_size_per_agent",
            "probation_archive_ttl_updates", "probation_max_c1_loss_questions",
            "probation_max_c2_loss_questions", "qd_archive_size_per_agent",
            "joint_allowed_vote_loss_questions", "joint_allowed_c1_loss_questions",
            "joint_allowed_c2_loss_questions", "joint_allowed_total_agent_correct_loss",
            "joint_allowed_bottom2_correct_loss", "joint_allowed_per_agent_correct_loss",
            "joint_vote_band_questions", "joint_mean_band_correct_count",
            "joint_bottom2_band_correct_count", "joint_c1_band_questions", "joint_c2_band_questions",
            "joint_team_max_active_changes_early", "joint_team_max_active_changes_late",
            "joint_team_no_diversification_patience", "joint_team_change_limit_relaxation",
        )
        for field in nonnegative_integer_fields:
            if int(getattr(self, field)) < 0:
                raise ValueError(f"{field} must be non-negative")
        positive_integer_fields = (
            "candidate_refill_candidates_per_round", "candidate_refill_max_unique_candidates_per_parent",
            "candidate_c1_catastrophic_loss_questions", "candidate_c2_catastrophic_loss_questions",
            "joint_representative_beam_size", "probe_stability_fold_count",
            "lineage_commit_required_snapshots", "lineage_switch_confirmation_snapshots",
            "qd_readiness_min_distinct_niches", "min_optimizer_updates_per_agent_per_epoch",
        )
        for field in positive_integer_fields:
            if int(getattr(self, field)) < 1:
                raise ValueError(f"{field} must be at least 1")
        if int(self.candidate_refill_max_unique_candidates_per_parent) < int(self.num_candidates_per_parent):
            raise ValueError("candidate_refill_max_unique_candidates_per_parent must cover the initial candidates per parent")
        if int(self.candidate_refill_min_safe_non_incumbent) > int(self.candidate_refill_max_unique_candidates_per_parent):
            raise ValueError("candidate_refill_min_safe_non_incumbent cannot exceed the per-parent unique candidate limit")
        if int(self.joint_representative_beam_size) > int(self.qd_archive_size_per_agent):
            raise ValueError("joint_representative_beam_size must not exceed qd_archive_size_per_agent")
        if float(self.probation_max_accuracy_loss) > float(self.catastrophic_target_accuracy_loss_epsilon):
            raise ValueError("probation_max_accuracy_loss must not exceed catastrophic_target_accuracy_loss_epsilon")
        if int(self.probation_max_c1_loss_questions) >= int(self.candidate_c1_catastrophic_loss_questions):
            raise ValueError("probation C1 loss must remain below the catastrophic C1 threshold")
        if int(self.probation_max_c2_loss_questions) >= int(self.candidate_c2_catastrophic_loss_questions):
            raise ValueError("probation C2 loss must remain below the catastrophic C2 threshold")
        if int(self.probe_stability_fold_count) != 2:
            raise ValueError("Stable QD currently requires exactly two deterministic probe folds")



@dataclass(init=False)
class Config:
    """Section-owned configuration with a flat compatibility API."""

    data: object = field(init=False)
    models: object = field(init=False)
    runtime: object = field(init=False)
    generation: object = field(init=False)
    evaluation: object = field(init=False)
    quality: object = field(init=False)
    archive: object = field(init=False)
    joint: object = field(init=False)
    lineage: object = field(init=False)
    output: object = field(init=False)
    identity: object = field(init=False)

    _SECTION_NAMES = (
        "data", "models", "runtime", "generation", "evaluation", "quality",
        "archive", "joint", "lineage", "output", "identity",
    )
    _FLAT_FIELDS = {item.name for item in fields(_FlatConfigSchema)}

    def __init__(self, **overrides):
        unknown = sorted(set(overrides) - self._FLAT_FIELDS)
        if unknown:
            raise TypeError(f"Unknown Config fields: {unknown}")
        flat = _FlatConfigSchema(**overrides)
        sections = split_flat_config(asdict(flat))
        for name, section in sections.items():
            object.__setattr__(self, name, section)

    def __getattr__(self, name):
        if name in self._FLAT_FIELDS:
            section = object.__getattribute__(self, section_for_field(name))
            return section.values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._SECTION_NAMES:
            object.__setattr__(self, name, value)
            return
        if name in self._FLAT_FIELDS:
            section = object.__getattribute__(self, section_for_field(name))
            section.values[name] = value
            return
        raise AttributeError(f"Unknown Config field: {name}")

    @classmethod
    def flat_field_registry(cls):
        return {name: f"{section_for_field(name)}.{name}" for name in sorted(cls._FLAT_FIELDS)}

    def to_flat_dict(self):
        result = {}
        for name in self._SECTION_NAMES:
            result.update(dict(object.__getattribute__(self, name).values))
        return {item.name: result[item.name] for item in fields(_FlatConfigSchema)}

    def sections(self):
        return {name: object.__getattribute__(self, name) for name in self._SECTION_NAMES}

    def to_canonical_dict(self):
        return canonical_config_dict(self)


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default=defaults.task_type, choices=["auto", "gsm8k", "mmlu", "bbh"])
    parser.add_argument("--dataset_format", type=str, default=defaults.dataset_format, choices=["legacy", "mars"])
    parser.add_argument("--comparison_task_id", type=str, default=defaults.comparison_task_id)
    parser.add_argument("--experiment_setting", type=str, default=defaults.experiment_setting)
    parser.add_argument("--benchmark", type=str, default=defaults.benchmark)
    parser.add_argument("--answer_format", type=str, default=defaults.answer_format, choices=["", "option_letter", "boolean", "yes_no", "valid_invalid", "numeric", "free_text"])

    parser.add_argument("--agent_model", type=str, default=defaults.agent_model)
    parser.add_argument("--optimizer_model", type=str, default=defaults.optimizer_model)
    parser.add_argument("--evaluator_model", type=str, default=defaults.evaluator_model)

    parser.add_argument("--train_path", type=str, default=defaults.train_path)
    parser.add_argument("--val_path", type=str, default=defaults.val_path)
    parser.add_argument("--test_path", type=str, default=defaults.test_path)
    parser.add_argument("--train_size", type=int, default=defaults.train_size)
    parser.add_argument("--val_size", type=int, default=defaults.val_size)
    parser.add_argument("--val_split_ratio", type=float, default=defaults.val_split_ratio)
    parser.add_argument("--test_size", type=int, default=defaults.test_size)
    parser.add_argument("--eval_test_each_epoch", type=int, default=int(defaults.eval_test_each_epoch), choices=[0, 1])

    parser.add_argument("--agents", type=int, default=defaults.agents)
    parser.add_argument("--init_mode", type=str, default=defaults.init_mode, choices=["shared", "bank"])
    parser.add_argument(
        "--shared_prompt",
        type=str,
        default=defaults.shared_prompt,
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--early_stopping_patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--early_stopping_min_delta", type=float, default=defaults.early_stopping_min_delta)
    parser.add_argument("--update_every", type=int, default=defaults.update_every)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=defaults.candidate_eval_batch_size)
    parser.add_argument("--baseline_only", type=int, default=int(defaults.baseline_only), choices=[0, 1])

    parser.add_argument("--search_mode", type=str, default=defaults.search_mode, choices=["evolutionary_beam"])
    parser.add_argument("--reward_mode", type=str, default=defaults.reward_mode, choices=["accuracy_only", "guarded_diversity", "coverage_useful_diversity", "vote_useful_diversity", "competence_depth_schedule"])
    parser.add_argument("--candidate_selection_mode", type=str, default=defaults.candidate_selection_mode, choices=["scalar_reward", "vote_pareto", "vote_error_pareto", "competence_depth_pareto"])
    parser.add_argument("--best_state_selection_mode", type=str, default=defaults.best_state_selection_mode, choices=["existing", "vote_first", "vote_competence_first", "vote_generalization_first"])
    parser.add_argument("--beam_size", type=int, default=defaults.beam_size)
    parser.add_argument("--num_candidates_per_parent", type=int, default=defaults.num_candidates_per_parent)
    parser.add_argument("--optimizer_parent_concurrency", type=int, default=defaults.optimizer_parent_concurrency)
    parser.add_argument("--beam_refresh_each_epoch", type=int, default=int(defaults.beam_refresh_each_epoch), choices=[0, 1])
    parser.add_argument("--legacy_beam_rescore_each_epoch", type=int, default=int(defaults.legacy_beam_rescore_each_epoch), choices=[0, 1])
    parser.add_argument("--homogeneity_overlap_threshold", type=float, default=defaults.homogeneity_overlap_threshold)
    parser.add_argument("--homogeneity_pressure_tie_eps", type=float, default=defaults.homogeneity_pressure_tie_eps)
    parser.add_argument("--max_homogeneous_cases_per_agent", type=int, default=defaults.max_homogeneous_cases_per_agent)
    parser.add_argument("--random_window_cases_per_agent", type=int, default=defaults.random_window_cases_per_agent)
    parser.add_argument("--hard_validity_cases_per_agent", type=int, default=defaults.hard_validity_cases_per_agent)
    parser.add_argument("--invalid_repair_rate_threshold", type=float, default=defaults.invalid_repair_rate_threshold)

    parser.add_argument("--accuracy_guard_epsilon", type=float, default=defaults.accuracy_guard_epsilon)
    parser.add_argument("--reward_weight_div_delta", type=float, default=defaults.reward_weight_div_delta)
    parser.add_argument("--reward_weight_invalid_delta", type=float, default=defaults.reward_weight_invalid_delta)
    parser.add_argument("--reward_weight_vote_delta", type=float, default=defaults.reward_weight_vote_delta)
    parser.add_argument("--reward_weight_vote_margin", type=float, default=defaults.reward_weight_vote_margin)
    parser.add_argument("--reward_weight_boundary_diversity", type=float, default=defaults.reward_weight_boundary_diversity)
    parser.add_argument("--reward_weight_coverage", type=float, default=defaults.reward_weight_coverage)
    parser.add_argument("--reward_weight_useful_diversity", type=float, default=defaults.reward_weight_useful_diversity)
    parser.add_argument("--invalid_guard_epsilon", type=float, default=defaults.invalid_guard_epsilon)
    parser.add_argument("--use_baseline_relative_reward", type=int, default=int(defaults.use_baseline_relative_reward), choices=[0, 1])
    parser.add_argument("--reward_schedule_mode", type=str, default=defaults.reward_schedule_mode, choices=["static", "phase_adaptive"])
    parser.add_argument("--reward_diversity_warmup_updates", type=int, default=defaults.reward_diversity_warmup_updates)
    parser.add_argument("--reward_weight_div_delta_early", type=float, default=defaults.reward_weight_div_delta_early)
    parser.add_argument("--reward_weight_div_delta_late", type=float, default=defaults.reward_weight_div_delta_late)
    parser.add_argument("--reward_weight_vote_delta_early", type=float, default=defaults.reward_weight_vote_delta_early)
    parser.add_argument("--reward_weight_vote_delta_late", type=float, default=defaults.reward_weight_vote_delta_late)
    parser.add_argument("--reward_weight_vote_margin_early", type=float, default=defaults.reward_weight_vote_margin_early)
    parser.add_argument("--reward_weight_vote_margin_late", type=float, default=defaults.reward_weight_vote_margin_late)
    parser.add_argument("--reward_weight_boundary_diversity_early", type=float, default=defaults.reward_weight_boundary_diversity_early)
    parser.add_argument("--reward_weight_boundary_diversity_late", type=float, default=defaults.reward_weight_boundary_diversity_late)
    parser.add_argument("--reward_weight_coverage_early", type=float, default=defaults.reward_weight_coverage_early)
    parser.add_argument("--reward_weight_coverage_late", type=float, default=defaults.reward_weight_coverage_late)
    parser.add_argument("--reward_weight_useful_diversity_early", type=float, default=defaults.reward_weight_useful_diversity_early)
    parser.add_argument("--reward_weight_useful_diversity_late", type=float, default=defaults.reward_weight_useful_diversity_late)
    parser.add_argument("--reward_weight_target_accuracy_early", type=float, default=defaults.reward_weight_target_accuracy_early)
    parser.add_argument("--reward_weight_target_accuracy_late", type=float, default=defaults.reward_weight_target_accuracy_late)
    parser.add_argument("--accuracy_guard_epsilon_early", type=float, default=defaults.accuracy_guard_epsilon_early)
    parser.add_argument("--accuracy_guard_epsilon_late", type=float, default=defaults.accuracy_guard_epsilon_late)
    parser.add_argument("--optimizer_architecture", type=str, default=defaults.optimizer_architecture, choices=["one_shot", "teacher_critic_student"])
    parser.add_argument("--teacher_critic_max_rounds", type=int, default=defaults.teacher_critic_max_rounds)
    parser.add_argument("--teacher_rewrite_max_count", type=int, default=defaults.teacher_rewrite_max_count)
    parser.add_argument("--teacher_question_pass_threshold", type=float, default=defaults.teacher_question_pass_threshold)
    parser.add_argument("--teacher_critic_direct_pass_threshold", type=float, default=defaults.teacher_critic_direct_pass_threshold)
    parser.add_argument("--teacher_critic_rewrite_threshold", type=float, default=defaults.teacher_critic_rewrite_threshold)
    parser.add_argument("--teacher_critic_forced_best_threshold", type=float, default=defaults.teacher_critic_forced_best_threshold)
    parser.add_argument("--tcs_repair_candidates_per_parent", type=int, default=defaults.tcs_repair_candidates_per_parent)
    parser.add_argument("--open_exploration_candidates_per_parent", type=int, default=defaults.open_exploration_candidates_per_parent)
    parser.add_argument("--teacher_temperature", type=float, default=defaults.teacher_temperature)
    parser.add_argument("--critic_temperature", type=float, default=defaults.critic_temperature)
    parser.add_argument("--student_temperature", type=float, default=defaults.student_temperature)
    parser.add_argument("--teacher_max_tokens", type=int, default=defaults.teacher_max_tokens)
    parser.add_argument("--critic_max_tokens", type=int, default=defaults.critic_max_tokens)
    parser.add_argument("--student_max_tokens", type=int, default=defaults.student_max_tokens)
    parser.add_argument("--student_json_retry_on_parse_fail", type=int, default=int(defaults.student_json_retry_on_parse_fail), choices=[0, 1])
    parser.add_argument("--student_json_max_retries", type=int, default=defaults.student_json_max_retries)
    parser.add_argument("--student_json_repair_enabled", type=int, default=int(defaults.student_json_repair_enabled), choices=[0, 1])
    parser.add_argument("--student_json_repair_max_tokens", type=int, default=defaults.student_json_repair_max_tokens)
    parser.add_argument("--student_json_repair_temperature", type=float, default=defaults.student_json_repair_temperature)
    parser.add_argument("--student_candidate_schema_mode", type=str, default=defaults.student_candidate_schema_mode, choices=["compact", "verbose"])
    parser.add_argument("--student_candidate_max_chars_per_field", type=int, default=defaults.student_candidate_max_chars_per_field)
    parser.add_argument("--student_candidate_prompt_max_chars", type=int, default=defaults.student_candidate_prompt_max_chars)
    parser.add_argument("--student_candidate_prompt_soft_max_chars", type=int, default=defaults.student_candidate_prompt_soft_max_chars)
    parser.add_argument("--student_candidate_prompt_hard_max_chars", type=int, default=defaults.student_candidate_prompt_hard_max_chars)
    parser.add_argument("--student_force_minified_json", type=int, default=int(defaults.student_force_minified_json), choices=[0, 1])
    parser.add_argument("--teacher_critic_use_voting_failure", type=int, default=int(defaults.teacher_critic_use_voting_failure), choices=[0, 1])
    parser.add_argument("--optimizer_fallback_mode", type=str, default=defaults.optimizer_fallback_mode, choices=["none", "template"])
    parser.add_argument("--no_effective_evolution_patience", type=int, default=defaults.no_effective_evolution_patience)
    parser.add_argument("--no_effective_evolution_min_optimizer_candidates", type=int, default=defaults.no_effective_evolution_min_optimizer_candidates)
    parser.add_argument("--no_effective_evolution_stop_enabled", type=int, default=int(defaults.no_effective_evolution_stop_enabled), choices=[0, 1])
    parser.add_argument("--boundary_selector_enabled", type=int, default=int(defaults.boundary_selector_enabled), choices=[0, 1])
    parser.add_argument("--shared_error_metrics_enabled", type=int, default=int(defaults.shared_error_metrics_enabled), choices=[0, 1])
    parser.add_argument("--residual_specialization_enabled", type=int, default=int(defaults.residual_specialization_enabled), choices=[0, 1])
    parser.add_argument("--error_dependence_guard_enabled", type=int, default=int(defaults.error_dependence_guard_enabled), choices=[0, 1])
    parser.add_argument("--residual_cycle_guard_enabled", type=int, default=int(defaults.residual_cycle_guard_enabled), choices=[0, 1])
    parser.add_argument("--mechanism_trust_region_enabled", type=int, default=int(defaults.mechanism_trust_region_enabled), choices=[0, 1])
    parser.add_argument("--specialization_ema", type=float, default=defaults.specialization_ema)
    parser.add_argument("--specialization_support_shrinkage", type=float, default=defaults.specialization_support_shrinkage)
    parser.add_argument("--capability_loss_weight", type=float, default=defaults.capability_loss_weight)
    parser.add_argument("--specialization_update_period", type=int, default=defaults.specialization_update_period)
    parser.add_argument("--capability_affinity_weight", type=float, default=defaults.capability_affinity_weight)
    parser.add_argument("--capability_coverage_gap_weight", type=float, default=defaults.capability_coverage_gap_weight)
    parser.add_argument("--pivotal_loss_guard_epsilon", type=float, default=defaults.pivotal_loss_guard_epsilon)
    parser.add_argument("--shared_error_creation_epsilon", type=float, default=defaults.shared_error_creation_epsilon)
    parser.add_argument("--behavior_cycle_guard_enabled", type=int, default=int(defaults.behavior_cycle_guard_enabled), choices=[0, 1])
    parser.add_argument("--behavior_archive_size", type=int, default=defaults.behavior_archive_size)
    parser.add_argument("--behavior_cycle_similarity_threshold", type=float, default=defaults.behavior_cycle_similarity_threshold)
    parser.add_argument("--behavior_cycle_min_overlap", type=int, default=defaults.behavior_cycle_min_overlap)
    parser.add_argument("--behavior_cycle_improvement_epsilon", type=float, default=defaults.behavior_cycle_improvement_epsilon)
    parser.add_argument("--behavior_cycle_margin_epsilon", type=float, default=defaults.behavior_cycle_margin_epsilon)
    parser.add_argument("--prompt_trust_region_enabled", type=int, default=int(defaults.prompt_trust_region_enabled), choices=[0, 1])
    parser.add_argument("--prompt_max_change_ratio", type=float, default=defaults.prompt_max_change_ratio)
    parser.add_argument("--prompt_large_shift_warmup_accepts", type=int, default=defaults.prompt_large_shift_warmup_accepts)
    parser.add_argument("--prompt_large_shift_min_vote_delta", type=float, default=defaults.prompt_large_shift_min_vote_delta)
    parser.add_argument("--baseline_allowed_vote_loss", type=float, default=defaults.baseline_allowed_vote_loss)
    parser.add_argument("--competence_depth_enabled", type=int, default=int(defaults.competence_depth_enabled), choices=[0, 1])
    parser.add_argument("--competence_depth2_aux_enabled", type=int, default=int(defaults.competence_depth2_aux_enabled), choices=[0, 1])
    parser.add_argument("--competence_progressive_residual_enabled", type=int, default=int(defaults.competence_progressive_residual_enabled), choices=[0, 1])
    parser.add_argument("--competence_floor_low", type=float, default=defaults.competence_floor_low)
    parser.add_argument("--competence_floor_high", type=float, default=defaults.competence_floor_high)
    parser.add_argument("--competence_selector_weight", type=float, default=defaults.competence_selector_weight)
    parser.add_argument("--competence_extra_support_shrinkage", type=float, default=defaults.competence_extra_support_shrinkage)
    parser.add_argument("--competence_weight_accuracy_gain", type=float, default=defaults.competence_weight_accuracy_gain)
    parser.add_argument("--competence_weight_accuracy_loss", type=float, default=defaults.competence_weight_accuracy_loss)
    parser.add_argument("--competence_weight_depth2_gain", type=float, default=defaults.competence_weight_depth2_gain)
    parser.add_argument("--competence_weight_depth2_loss", type=float, default=defaults.competence_weight_depth2_loss)
    parser.add_argument("--competence_weight_vote_gain_early", type=float, default=defaults.competence_weight_vote_gain_early)
    parser.add_argument("--competence_weight_vote_loss_early", type=float, default=defaults.competence_weight_vote_loss_early)
    parser.add_argument("--competence_schedule_mode", type=str, default=defaults.competence_schedule_mode, choices=["absolute_legacy", "baseline_relative_opt_snapshot"])
    parser.add_argument("--competence_schedule_version", type=str, default=defaults.competence_schedule_version)
    parser.add_argument("--competence_probe_size", type=int, default=defaults.competence_probe_size)
    parser.add_argument("--competence_probe_seed_offset", type=int, default=defaults.competence_probe_seed_offset)
    parser.add_argument("--competence_relative_low_delta", type=float, default=defaults.competence_relative_low_delta)
    parser.add_argument("--competence_relative_high_delta", type=float, default=defaults.competence_relative_high_delta)
    parser.add_argument("--competence_schedule_ema", type=float, default=defaults.competence_schedule_ema)
    parser.add_argument("--competence_schedule_max_step", type=float, default=defaults.competence_schedule_max_step)
    parser.add_argument("--competence_schedule_monotonic", type=int, default=int(defaults.competence_schedule_monotonic), choices=[0, 1])
    parser.add_argument("--competence_mean_guard_epsilon", type=float, default=defaults.competence_mean_guard_epsilon)
    parser.add_argument("--competence_c1_guard_epsilon", type=float, default=defaults.competence_c1_guard_epsilon)
    parser.add_argument("--competence_c2_guard_epsilon", type=float, default=defaults.competence_c2_guard_epsilon)
    parser.add_argument("--competence_depth1_candidate_guard_enabled", type=int, default=int(defaults.competence_depth1_candidate_guard_enabled), choices=[0, 1])
    parser.add_argument("--competence_depth1_candidate_guard_epsilon", type=float, default=defaults.competence_depth1_candidate_guard_epsilon)
    parser.add_argument("--competence_min_effective_specialization_epochs", type=int, default=defaults.competence_min_effective_specialization_epochs)
    parser.add_argument("--method_version", default=defaults.method_version)
    parser.add_argument("--target_selector_mode", default=defaults.target_selector_mode, choices=["legacy", "hybrid_competence_boundary"])
    parser.add_argument("--target_selector_version", default=defaults.target_selector_version)
    parser.add_argument("--beam_policy_version", default=defaults.beam_policy_version)
    parser.add_argument("--tcs_candidate_policy_version", default=defaults.tcs_candidate_policy_version)
    parser.add_argument("--mechanism_signature_version", default=defaults.mechanism_signature_version)
    parser.add_argument("--competence_weight_depth1_gain", type=float, default=defaults.competence_weight_depth1_gain)
    parser.add_argument("--competence_weight_depth1_loss", type=float, default=defaults.competence_weight_depth1_loss)
    parser.add_argument("--competence_residual_floor", type=float, default=defaults.competence_residual_floor)
    parser.add_argument("--catastrophic_target_accuracy_loss_epsilon", type=float, default=defaults.catastrophic_target_accuracy_loss_epsilon)
    parser.add_argument("--soft_guard_error_dependence_weight", type=float, default=defaults.soft_guard_error_dependence_weight)
    parser.add_argument("--soft_guard_cycle_weight", type=float, default=defaults.soft_guard_cycle_weight)
    parser.add_argument("--soft_guard_mechanism_shift_weight", type=float, default=defaults.soft_guard_mechanism_shift_weight)
    parser.add_argument("--soft_guard_accuracy_regression_weight", type=float, default=defaults.soft_guard_accuracy_regression_weight)
    parser.add_argument("--mechanism_novelty_bonus_weight", type=float, default=defaults.mechanism_novelty_bonus_weight)
    parser.add_argument("--active_team_selector_version", default=defaults.active_team_selector_version)
    parser.add_argument("--candidate_generation_policy_version", default=defaults.candidate_generation_policy_version)
    parser.add_argument("--joint_refresh_policy_version", default=defaults.joint_refresh_policy_version)
    parser.add_argument("--representative_probe_policy_version", default=defaults.representative_probe_policy_version)
    parser.add_argument("--lineage_policy_version", default=defaults.lineage_policy_version)
    parser.add_argument("--mechanism_distance_version", default=defaults.mechanism_distance_version)
    for name in (
        "candidate_refill_version", "archive_policy_version", "joint_quality_filter_version",
        "probe_stability_version", "parent_selection_version", "qd_parent_selection_mode",
        "joint_refresh_mode",
    ):
        parser.add_argument(f"--{name}", default=getattr(defaults, name))
    for name in (
        "mechanism_sequence_distance_weight", "mechanism_embedding_distance_weight",
        "mechanism_near_duplicate_similarity_threshold", "behavior_correct_set_weight",
        "behavior_rescue_weight", "behavior_shared_wrong_weight", "behavior_support_shrinkage",
        "team_diversity_mean_behavior_weight", "team_diversity_min_behavior_weight",
        "team_diversity_mechanism_weight", "team_diversity_rescue_balance_weight",
        "joint_team_per_agent_accuracy_epsilon", "lineage_mechanism_drift_weight",
        "lineage_behavior_drift_weight", "lineage_soft_drift_threshold", "lineage_hard_drift_threshold",
        "lineage_switch_min_accuracy_gain", "lineage_switch_min_vote_gain",
        "peer_collapse_soft_similarity", "peer_collapse_hard_similarity",
        "probation_max_accuracy_loss", "qd_readiness_min_diversity",
        "qd_readiness_max_fold_gap", "residual_specialization_qd_floor",
        "behavior_error_overlap_weight", "behavior_wrong_answer_dispersion_weight",
        "behavior_wrong_support_shrinkage", "joint_team_change_limit_switch_strength",
    ):
        parser.add_argument(f"--{name}", type=float, default=getattr(defaults, name))
    for name in (
        "joint_team_vote_epsilon_questions", "joint_team_mean_epsilon_questions",
        "joint_team_bottom2_epsilon_questions", "joint_team_c1_epsilon_questions",
        "joint_team_c2_epsilon_questions", "lineage_provisional_epochs", "lineage_commit_epochs",
        "lineage_switch_confirmation_epochs",
        "candidate_refill_max_rounds", "candidate_refill_candidates_per_round",
        "candidate_refill_max_unique_candidates_per_parent", "candidate_refill_min_safe_non_incumbent",
        "candidate_refill_max_solver_calls_per_agent_update", "probation_archive_size_per_agent",
        "probation_archive_ttl_updates", "probation_max_c1_loss_questions",
        "probation_max_c2_loss_questions", "candidate_c1_catastrophic_loss_questions",
        "candidate_c2_catastrophic_loss_questions", "qd_archive_size_per_agent",
        "joint_representative_beam_size", "qd_niche_min_parent_opportunities_per_epoch",
        "probe_stability_fold_count", "probe_stability_seed_offset", "joint_vote_band_questions",
        "joint_mean_band_correct_count", "joint_bottom2_band_correct_count",
        "joint_c1_band_questions", "joint_c2_band_questions", "joint_allowed_vote_loss_questions",
        "joint_allowed_c1_loss_questions", "joint_allowed_c2_loss_questions",
        "joint_allowed_total_agent_correct_loss", "joint_allowed_bottom2_correct_loss",
        "joint_allowed_per_agent_correct_loss", "joint_team_max_active_changes_early",
        "joint_team_max_active_changes_late", "joint_team_no_diversification_patience",
        "joint_team_change_limit_relaxation", "lineage_commit_required_snapshots",
        "lineage_switch_confirmation_snapshots", "qd_readiness_min_distinct_niches",
        "min_optimizer_updates_per_agent_per_epoch", "joint_refresh_interval_epochs",
        "joint_refresh_min_new_safe_candidates", "joint_refresh_max_dirty_candidates_per_agent",
    ):
        parser.add_argument(f"--{name}", type=int, default=getattr(defaults, name))
    parser.add_argument(
        "--validation_stable_specialization_tie_break_enabled", type=int,
        default=int(defaults.validation_stable_specialization_tie_break_enabled), choices=[0, 1],
    )
    for name in (
        "candidate_refill_enabled", "candidate_refill_require_task_repair",
        "candidate_refill_require_distinct_mechanism", "candidate_refill_feed_rejection_reasons",
        "candidate_refill_stop_when_requirements_met", "probation_archive_enabled",
        "probation_require_mechanism_novelty", "probation_parent_enabled",
        "target_selector_fairness_enabled", "joint_refresh_on_safe_archive_change",
        "joint_refresh_on_probation_promotion", "joint_refresh_on_representative_change",
        "joint_refresh_force_final_epoch", "joint_refresh_skip_when_no_dirty_prompt",
    ):
        parser.add_argument(f"--{name}", type=int, default=int(getattr(defaults, name)), choices=[0, 1])
    parser.add_argument("--diversity_metric", type=str, default=defaults.diversity_metric, choices=["trace_embedding"])
    parser.add_argument("--use_joint_trace_diversity_evaluator", type=int, default=int(defaults.use_joint_trace_diversity_evaluator), choices=[0, 1])
    parser.add_argument("--invalid_binary", type=int, default=int(defaults.invalid_binary), choices=[0, 1])
    parser.add_argument("--embedding_model", type=str, default=defaults.embedding_model)
    parser.add_argument("--trace_embedding_chunk_words", type=int, default=defaults.trace_embedding_chunk_words)
    parser.add_argument("--trace_embedding_chunk_overlap", type=int, default=defaults.trace_embedding_chunk_overlap)

    parser.add_argument("--max_tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--optimizer_max_tokens", type=int, default=defaults.optimizer_max_tokens)
    parser.add_argument("--evaluator_max_tokens", type=int, default=defaults.evaluator_max_tokens)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--optimizer_temperature", type=float, default=defaults.optimizer_temperature)
    parser.add_argument("--evaluator_temperature", type=float, default=defaults.evaluator_temperature)

    parser.add_argument("--out_dir", type=str, default=defaults.out_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--resume_from_checkpoint", type=int, default=int(defaults.resume_from_checkpoint), choices=[0, 1])
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries)
    parser.add_argument("--retry_sleep", type=float, default=defaults.retry_sleep)
    parser.add_argument("--transient_retry_forever", type=int, default=int(defaults.transient_retry_forever), choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=defaults.max_transient_retries)
    parser.add_argument("--max_retry_backoff", type=float, default=defaults.max_retry_backoff)
    parser.add_argument("--llm_call_logging", type=int, default=int(defaults.llm_call_logging), choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=defaults.llm_call_timeout)
    parser.add_argument("--candidate_eval_concurrency", type=int, default=defaults.candidate_eval_concurrency)
    parser.add_argument("--candidate_eval_strategy", type=str, default=defaults.candidate_eval_strategy, choices=["random", "fixed_pool", "stratified"])
    parser.add_argument("--candidate_eval_pool_size", type=int, default=defaults.candidate_eval_pool_size)
    parser.add_argument("--candidate_eval_repeats", type=int, default=defaults.candidate_eval_repeats)
    parser.add_argument("--candidate_eval_seed_offset", type=int, default=defaults.candidate_eval_seed_offset)
    parser.add_argument("--candidate_reuse_recorded_rollouts", type=int, default=int(defaults.candidate_reuse_recorded_rollouts), choices=[0, 1])
    parser.add_argument("--candidate_eval_execution_mode", type=str, default=defaults.candidate_eval_execution_mode, choices=["legacy", "factorized_cached"])
    parser.add_argument("--solver_rollout_singleflight", type=int, default=int(defaults.solver_rollout_singleflight), choices=[0, 1])
    parser.add_argument("--candidate_eval_prompt_dedup", type=int, default=int(defaults.candidate_eval_prompt_dedup), choices=[0, 1])
    parser.add_argument("--candidate_eval_cache_logging", type=int, default=int(defaults.candidate_eval_cache_logging), choices=[0, 1])
    parser.add_argument("--split_integrity_json", type=str, default=defaults.split_integrity_json)
    parser.add_argument("--train_rollout_concurrency", type=int, default=defaults.train_rollout_concurrency)
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=defaults.eval_solver_call_concurrency)
    parser.add_argument("--solver_api_key_env", type=str, default=defaults.solver_api_key_env)
    parser.add_argument("--solver_base_url_env", type=str, default=defaults.solver_base_url_env)
    parser.add_argument("--evaluator_api_key_env", type=str, default=defaults.evaluator_api_key_env)
    parser.add_argument("--evaluator_base_url_env", type=str, default=defaults.evaluator_base_url_env)
    parser.add_argument("--vote_tie_break", type=str, default=defaults.vote_tie_break, choices=["first", "random", "abstain"])
    parser.add_argument("--aggregation_mode", type=str, default=defaults.aggregation_mode, choices=["majority", "plurality", "weighted_vote", "verifier_select"])
    return parser
