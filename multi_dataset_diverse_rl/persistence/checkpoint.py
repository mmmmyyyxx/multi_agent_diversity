"""Canonical checkpoint serialization and compatibility validation."""

import hashlib
import json
import os
import random
import time
import uuid

import numpy as np

from ..config import Config
from ..diagnostics.candidate_funnel import (
    empty_candidate_channel_funnel,
    restore_funnel_seen,
    serialize_funnel_seen,
)
from ..utils import canonical_aggregation_mode
from ..qd.quality_anchors import build_quality_anchor, update_quality_anchor_archive


def checkpoint_path(cfg):
    return os.path.join(cfg.out_dir, "training_checkpoint.json")


def read_json_file(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json_atomic(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".{uuid.uuid4().hex[:12]}.tmp")
    for attempt in range(3):
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            return
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt == 2:
                raise
            time.sleep(0.1 * (attempt + 1))


def restore_system_state(system, state_payload):
    agents = state_payload.get("agents", []) if isinstance(state_payload, dict) else []
    if not isinstance(agents, list) or len(agents) != len(system.agents):
        raise ValueError(f"Cannot restore state: got {len(agents) if isinstance(agents, list) else 0} agents, expected {len(system.agents)}")
    for agent, saved in zip(system.agents, agents):
        if not isinstance(saved, dict):
            continue
        agent.initial_prompt = str(saved.get("initial_prompt", agent.initial_prompt))
        agent.current_prompt = str(saved.get("current_prompt", agent.current_prompt))
        prompt_beam = saved.get("prompt_beam", [])
        if isinstance(prompt_beam, list) and prompt_beam:
            agent.prompt_beam = [dict(item) for item in prompt_beam if isinstance(item, dict)]
        else:
            agent.prompt_beam = [system._make_beam_item(agent.current_prompt, None, {}, None, 0)]
        history = saved.get("history", [])
        agent.history = [str(x) for x in history] if isinstance(history, list) and history else [agent.current_prompt]
        agent.accept_count = int(saved.get("accept_count", 0) or 0)
        agent.reject_count = int(saved.get("reject_count", 0) or 0)
        if hasattr(agent, "restore_trajectory_state"):
            agent.restore_trajectory_state(saved)
    recent_window_records = state_payload.get("recent_window_records", [])
    system.recent_window_records = (
        [dict(record) for record in recent_window_records if isinstance(record, dict)]
        if isinstance(recent_window_records, list)
        else []
    )
    system.specialization_strength = float(state_payload.get("specialization_strength", 0.0) or 0.0)
    system.effective_residual_strength = float(state_payload.get("effective_residual_strength", system.specialization_strength) or system.specialization_strength)
    system.previous_epoch_per_agent_acc = [float(x) for x in state_payload.get("previous_epoch_per_agent_acc", [])]
    system.previous_epoch_bottom2_mean_acc = float(state_payload.get("previous_epoch_bottom2_mean_acc", 0.0) or 0.0)
    system.competence_phase_epoch = int(state_payload.get("competence_phase_epoch", 1) or 1)
    system.competence_schedule_version = str(state_payload.get("competence_schedule_version", "competence_depth_v1"))
    system.specialization_strength_history = [float(x) for x in state_payload.get("specialization_strength_history", [])]
    system.competence_probe_indices = [int(x) for x in state_payload.get("competence_probe_indices", [])]
    system.competence_probe_question_hashes = [str(x) for x in state_payload.get("competence_probe_question_hashes", [])]
    system.initial_competence_probe_metrics = dict(state_payload.get("initial_competence_probe_metrics", {}) or {})
    system.latest_competence_probe_metrics = dict(state_payload.get("latest_competence_probe_metrics", {}) or {})
    system.competence_probe_history = [dict(x) for x in state_payload.get("competence_probe_history", []) if isinstance(x, dict)]
    system.initial_active_prompt_hashes = [str(x) for x in state_payload.get("initial_active_prompt_hashes", [])]
    first_nonzero = state_payload.get("first_nonzero_specialization_epoch")
    system.first_nonzero_specialization_epoch = int(first_nonzero) if first_nonzero is not None else None
    system.effective_specialization_epoch_count = int(state_payload.get("effective_specialization_epoch_count", 0) or 0)
    system.depth1_guard_rejection_count = int(state_payload.get("depth1_guard_rejection_count", 0) or 0)
    for field in (
        "catastrophic_accuracy_guard_rejection_count",
        "soft_error_dependence_penalty_count",
        "soft_cycle_penalty_count",
        "soft_mechanism_shift_penalty_count",
        "exploration_candidate_count",
        "exploration_slot_occupancy_count",
        "exploration_to_active_conversion_count",
    ):
        setattr(system, field, int(state_payload.get(field, 0) or 0))
    system.hybrid_selector_history = [dict(x) for x in state_payload.get("hybrid_selector_history", []) if isinstance(x, dict)]
    system.mechanism_signature_history = [dict(x) for x in state_payload.get("mechanism_signature_history", []) if isinstance(x, dict)]
    system.mechanism_signature_by_prompt_hash = {
        str(key): [str(value) for value in values]
        for key, values in dict(state_payload.get("mechanism_signature_by_prompt_hash", {}) or {}).items()
        if isinstance(values, list)
    }
    system.beam_slot_state = dict(state_payload.get("beam_slot_state", {}) or {})
    system.exploration_slot_candidates = [dict(x) for x in state_payload.get("exploration_slot_candidates", []) if isinstance(x, dict)]
    system.prompt_overlength_rejection_count = int(state_payload.get("prompt_overlength_rejection_count", 0) or 0)
    system.truncated_prompt_count = int(state_payload.get("truncated_prompt_count", 0) or 0)
    system.mechanism_embedding_cache = {str(key): list(value) for key, value in dict(state_payload.get("mechanism_embedding_cache", {}) or {}).items()}
    system.semantic_mechanism_families = {
        str(key): dict(value) for key, value in dict(state_payload.get("semantic_mechanism_families", {}) or {}).items()
    }
    system.prompt_probe_cache = {str(key): dict(value) for key, value in dict(state_payload.get("prompt_probe_cache", {}) or {}).items()}
    system.mechanism_embedding_cache_hit_count = int(state_payload.get("mechanism_embedding_cache_hit_count", 0) or 0)
    system.mechanism_embedding_cache_miss_count = int(state_payload.get("mechanism_embedding_cache_miss_count", 0) or 0)
    system.full_probe_cache_hit_count = int(state_payload.get("full_probe_cache_hit_count", 0) or 0)
    system.full_probe_missing_pair_evaluation_count = int(state_payload.get("full_probe_missing_pair_evaluation_count", 0) or 0)
    system.behavior_profile_by_prompt_hash = {str(key): dict(value) for key, value in dict(state_payload.get("behavior_profile_by_prompt_hash", {}) or {}).items()}
    restored_funnel = state_payload.get("candidate_channel_funnel", {})
    system.candidate_channel_funnel = (
        {str(channel): {str(stage): int(count or 0) for stage, count in counts.items()}
         for channel, counts in restored_funnel.items() if isinstance(counts, dict)}
        if isinstance(restored_funnel, dict) and restored_funnel
        else empty_candidate_channel_funnel()
    )
    system.candidate_channel_funnel_seen = restore_funnel_seen(
        state_payload.get("candidate_channel_funnel_seen", {})
        if isinstance(state_payload.get("candidate_channel_funnel_seen", {}), dict) else {}
    )
    system.joint_team_selection_history = [dict(value) for value in state_payload.get("joint_team_selection_history", []) if isinstance(value, dict)]
    system.lineage_history = [dict(value) for value in state_payload.get("lineage_history", []) if isinstance(value, dict)]
    system.quality_diversity_archive_history = [dict(value) for value in state_payload.get("quality_diversity_archive_history", []) if isinstance(value, dict)]
    system.behavior_profile_history = [dict(value) for value in state_payload.get("behavior_profile_history", []) if isinstance(value, dict)]
    system.latest_joint_team_metrics = dict(state_payload.get("latest_joint_team_metrics", {}) or {})
    system.joint_quality_anchor_metrics = {}
    system.quality_anchor_archive = [
        dict(value) for value in state_payload.get("quality_anchor_archive", []) if isinstance(value, dict)
    ]
    system.quality_anchor_created_count = int(state_payload.get("quality_anchor_created_count", len(system.quality_anchor_archive)) or 0)
    for field in (
        "last_joint_refresh_epoch", "epochs_since_last_joint_refresh",
        "archive_material_change_version", "last_probation_promotion_count",
        "joint_refresh_count", "joint_refresh_skipped_count",
        "legacy_beam_refresh_call_count", "new_full_probe_prompt_count",
        "offline_team_combination_count", "joint_team_solver_call_count",
        "tcs_repair_generation_count", "open_exploration_generation_count",
        "tcs_repair_candidate_count", "open_exploration_candidate_count",
    ):
        setattr(system, field, int(state_payload.get(field, getattr(system, field, 0)) or 0))
    system.representative_version_per_agent = {
        str(key): int(value) for key, value in dict(
            state_payload.get("representative_version_per_agent", {}) or {}
        ).items()
    }
    system.dirty_prompt_hashes = {
        str(key): [str(value) for value in values]
        for key, values in dict(state_payload.get("dirty_prompt_hashes", {}) or {}).items()
        if isinstance(values, list)
    }
    system.prompt_probe_version = str(state_payload.get("prompt_probe_version", getattr(system, "prompt_probe_version", "legacy")))
    system.current_fixed_probe_hash = str(state_payload.get("current_fixed_probe_hash", ""))
    system.last_archive_material_snapshot = dict(state_payload.get("last_archive_material_snapshot", {}) or {})
    system.last_representative_snapshot = dict(state_payload.get("last_representative_snapshot", {}) or {})
    system.last_active_prompt_hashes = [str(value) for value in state_payload.get("last_active_prompt_hashes", [])]
    for field in (
        "qd_no_diversification_epochs", "qd_change_limit_relaxed_epoch",
        "qd_previous_active_niche_count",
        "probation_to_safe_conversion_count", "probation_expired_count",
        "candidate_starvation_count", "mechanism_starvation_count",
        "search_branch_starvation_count", "refill_requirements_unmet_count",
    ):
        setattr(system, field, int(state_payload.get(field, 0) or 0))
    system.per_agent_optimizer_update_count = {
        str(key): int(value) for key, value in dict(state_payload.get("per_agent_optimizer_update_count", {}) or {}).items()
    }
    for field in (
        "total_agent_update_count", "task_repair_niche_occupancy_count",
        "mechanism_niche_occupancy_count", "peer_collapse_soft_count",
        "peer_collapse_hard_rejection_count",
    ):
        setattr(system, field, int(state_payload.get(field, 0) or 0))
    python_state = state_payload.get("python_random_state")
    if isinstance(python_state, list):
        def as_tuple(value):
            return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value
        random.setstate(as_tuple(python_state))
    numpy_state = state_payload.get("numpy_random_state")
    if isinstance(numpy_state, list) and len(numpy_state) == 5:
        np.random.set_state((str(numpy_state[0]), np.array(numpy_state[1], dtype=np.uint32), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))


def restore_prompt_history(system):
    path = os.path.join(system.cfg.out_dir, "prompt_history.json")
    payload = read_json_file(path)
    if isinstance(payload, dict):
        system.prompt_history = payload
    if hasattr(system, "sync_prompt_history_current_state"):
        system.sync_prompt_history_current_state(event="checkpoint_resume", epoch="resume", step=0)


def restore_cost_summary(system):
    payload = read_json_file(os.path.join(system.cfg.out_dir, "cost_summary.json"))
    if isinstance(payload, dict):
        base = system._empty_cost_summary() if hasattr(system, "_empty_cost_summary") else {}
        base.update(payload)
        system.cost_summary = base
CHECKPOINT_VERSION = 6

# Fields that can change the objective, candidate distribution, optimizer
# behavior, validation decision, or final aggregation of an interrupted run.
BEHAVIOR_CONFIG_FIELDS = (
        "task_type",
        "dataset_format",
        "comparison_task_id",
        "benchmark",
        "answer_format",
        "train_path",
        "val_path",
        "test_path",
        "train_size",
        "val_size",
        "val_split_ratio",
        "test_size",
        "agents",
        "init_mode",
        "shared_prompt",
        "reward_mode",
        "accuracy_guard_epsilon",
        "invalid_guard_epsilon",
        "reward_weight_div_delta",
        "reward_weight_invalid_delta",
        "reward_weight_vote_delta",
        "reward_weight_vote_margin",
        "reward_weight_boundary_diversity",
        "reward_weight_coverage",
        "reward_weight_useful_diversity",
        "use_baseline_relative_reward",
        "reward_schedule_mode",
        "reward_diversity_warmup_updates",
        "reward_weight_div_delta_early",
        "reward_weight_div_delta_late",
        "reward_weight_vote_delta_early",
        "reward_weight_vote_delta_late",
        "reward_weight_vote_margin_early",
        "reward_weight_vote_margin_late",
        "reward_weight_boundary_diversity_early",
        "reward_weight_boundary_diversity_late",
        "reward_weight_coverage_early",
        "reward_weight_coverage_late",
        "reward_weight_useful_diversity_early",
        "reward_weight_useful_diversity_late",
        "reward_weight_target_accuracy_early",
        "reward_weight_target_accuracy_late",
        "accuracy_guard_epsilon_early",
        "accuracy_guard_epsilon_late",
        "candidate_selection_mode",
        "best_state_selection_mode",
        "vote_tie_break",
        "aggregation_mode",
        "optimizer_architecture",
        "optimizer_fallback_mode",
        "teacher_critic_max_rounds",
        "teacher_question_pass_threshold",
        "teacher_critic_use_voting_failure",
        "teacher_temperature",
        "critic_temperature",
        "student_temperature",
        "teacher_max_tokens",
        "critic_max_tokens",
        "student_max_tokens",
        "student_json_retry_on_parse_fail",
        "student_json_max_retries",
        "student_json_repair_enabled",
        "student_json_repair_max_tokens",
        "student_json_repair_temperature",
        "student_candidate_schema_mode",
        "student_candidate_max_chars_per_field",
        "student_candidate_prompt_max_chars",
        "student_candidate_prompt_soft_max_chars",
        "student_candidate_prompt_hard_max_chars",
        "student_force_minified_json",
        "beam_size",
        "num_candidates_per_parent",
        "optimizer_parent_concurrency",
        "beam_refresh_each_epoch",
        "update_every",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "candidate_eval_batch_size",
        "candidate_eval_strategy",
        "candidate_eval_pool_size",
        "candidate_eval_repeats",
        "candidate_eval_seed_offset",
        "candidate_eval_data_source",
        "candidate_eval_execution_mode",
        "candidate_reuse_recorded_rollouts",
        "candidate_eval_concurrency",
        "solver_rollout_singleflight",
        "candidate_eval_prompt_dedup",
        "candidate_eval_cache_logging",
        "agent_model",
        "optimizer_model",
        "evaluator_model",
        "max_tokens",
        "temperature",
        "optimizer_max_tokens",
        "optimizer_temperature",
        "evaluator_max_tokens",
        "evaluator_temperature",
        "solver_base_url_env",
        "evaluator_base_url_env",
        "diversity_metric",
        "use_joint_trace_diversity_evaluator",
        "invalid_binary",
        "embedding_model",
        "trace_embedding_chunk_words",
        "trace_embedding_chunk_overlap",
        "eval_test_each_epoch",
        "no_effective_evolution_patience",
        "no_effective_evolution_min_optimizer_candidates",
        "no_effective_evolution_stop_enabled",
        "boundary_selector_enabled",
        "shared_error_metrics_enabled",
        "residual_specialization_enabled",
        "error_dependence_guard_enabled",
        "residual_cycle_guard_enabled",
        "mechanism_trust_region_enabled",
        "specialization_ema",
        "specialization_support_shrinkage",
        "capability_loss_weight",
        "specialization_update_period",
        "capability_affinity_weight",
        "capability_coverage_gap_weight",
        "pivotal_loss_guard_epsilon",
        "shared_error_creation_epsilon",
        "behavior_cycle_guard_enabled",
        "behavior_archive_size",
        "behavior_cycle_similarity_threshold",
        "behavior_cycle_min_overlap",
        "behavior_cycle_improvement_epsilon",
        "behavior_cycle_margin_epsilon",
        "prompt_trust_region_enabled",
        "prompt_max_change_ratio",
        "prompt_large_shift_warmup_accepts",
        "prompt_large_shift_min_vote_delta",
        "baseline_allowed_vote_loss",
        "competence_depth_enabled",
        "competence_depth2_aux_enabled",
        "competence_progressive_residual_enabled",
        "competence_floor_low",
        "competence_floor_high",
        "competence_selector_weight",
        "competence_extra_support_shrinkage",
        "competence_weight_accuracy_gain",
        "competence_weight_accuracy_loss",
        "competence_weight_depth2_gain",
        "competence_weight_depth2_loss",
        "competence_weight_vote_gain_early",
        "competence_weight_vote_loss_early",
        "competence_schedule_version",
        "competence_schedule_mode",
        "competence_probe_size",
        "competence_probe_seed_offset",
        "competence_relative_low_delta",
        "competence_relative_high_delta",
        "competence_schedule_ema",
        "competence_schedule_max_step",
        "competence_schedule_monotonic",
        "competence_mean_guard_epsilon",
        "competence_c1_guard_epsilon",
        "competence_c2_guard_epsilon",
        "competence_depth1_candidate_guard_enabled",
        "competence_depth1_candidate_guard_epsilon",
        "competence_min_effective_specialization_epochs",
        "method_version", "target_selector_mode", "target_selector_version", "beam_policy_version",
        "tcs_candidate_policy_version", "mechanism_signature_version",
        "competence_weight_depth1_gain", "competence_weight_depth1_loss", "competence_residual_floor",
        "catastrophic_target_accuracy_loss_epsilon", "soft_guard_error_dependence_weight",
        "soft_guard_cycle_weight", "soft_guard_mechanism_shift_weight",
        "soft_guard_accuracy_regression_weight", "mechanism_novelty_bonus_weight",
        "active_team_selector_version", "lineage_policy_version", "mechanism_distance_version",
        "mechanism_sequence_distance_weight", "mechanism_embedding_distance_weight",
        "mechanism_near_duplicate_similarity_threshold", "behavior_correct_set_weight",
        "behavior_rescue_weight", "behavior_shared_wrong_weight", "behavior_support_shrinkage",
        "team_diversity_mean_behavior_weight", "team_diversity_min_behavior_weight",
        "team_diversity_mechanism_weight", "team_diversity_rescue_balance_weight",
        "joint_team_vote_epsilon_questions", "joint_team_mean_epsilon_questions",
        "joint_team_bottom2_epsilon_questions", "joint_team_c1_epsilon_questions",
        "joint_team_c2_epsilon_questions", "joint_team_per_agent_accuracy_epsilon",
        "lineage_provisional_epochs", "lineage_commit_epochs", "lineage_switch_confirmation_epochs",
        "lineage_mechanism_drift_weight", "lineage_behavior_drift_weight",
        "lineage_soft_drift_threshold", "lineage_hard_drift_threshold",
        "lineage_switch_min_accuracy_gain", "lineage_switch_min_vote_gain",
        "peer_collapse_soft_similarity", "peer_collapse_hard_similarity",
        "validation_stable_specialization_tie_break_enabled",
        "candidate_refill_version", "archive_policy_version", "joint_quality_filter_version",
        "probe_stability_version", "parent_selection_version", "candidate_refill_enabled",
        "candidate_refill_max_rounds", "candidate_refill_candidates_per_round",
        "candidate_refill_max_unique_candidates_per_parent", "candidate_refill_min_safe_non_incumbent",
        "candidate_refill_require_task_repair", "candidate_refill_require_distinct_mechanism",
        "candidate_refill_feed_rejection_reasons", "candidate_refill_stop_when_requirements_met",
        "candidate_refill_max_solver_calls_per_agent_update", "probation_archive_enabled",
        "probation_archive_size_per_agent", "probation_archive_ttl_updates", "probation_max_accuracy_loss",
        "probation_max_c1_loss_questions", "probation_max_c2_loss_questions",
        "probation_require_mechanism_novelty", "candidate_c1_catastrophic_loss_questions",
        "candidate_c2_catastrophic_loss_questions", "qd_archive_size_per_agent",
        "joint_representative_beam_size", "qd_parent_selection_mode",
        "qd_niche_min_parent_opportunities_per_epoch", "probation_parent_enabled",
        "probe_stability_fold_count", "probe_stability_seed_offset", "joint_vote_band_questions",
        "joint_mean_band_correct_count", "joint_bottom2_band_correct_count", "joint_c1_band_questions",
        "joint_c2_band_questions", "joint_allowed_vote_loss_questions", "joint_allowed_c1_loss_questions",
        "joint_allowed_c2_loss_questions", "joint_allowed_total_agent_correct_loss",
        "joint_allowed_bottom2_correct_loss", "joint_allowed_per_agent_correct_loss",
        "joint_team_max_active_changes_early", "joint_team_max_active_changes_late",
        "joint_team_change_limit_switch_strength", "joint_team_no_diversification_patience",
        "joint_team_change_limit_relaxation", "lineage_commit_required_snapshots",
        "lineage_switch_confirmation_snapshots", "qd_readiness_min_distinct_niches",
        "qd_readiness_min_diversity", "qd_readiness_max_fold_gap", "residual_specialization_qd_floor",
        "behavior_error_overlap_weight", "behavior_wrong_answer_dispersion_weight",
        "behavior_wrong_support_shrinkage", "min_optimizer_updates_per_agent_per_epoch",
        "target_selector_fairness_enabled",
        "teacher_rewrite_max_count", "teacher_critic_direct_pass_threshold",
        "teacher_critic_rewrite_threshold", "teacher_critic_forced_best_threshold",
        "legacy_beam_rescore_each_epoch", "candidate_generation_policy_version",
        "joint_refresh_policy_version", "representative_probe_policy_version",
        "joint_refresh_mode", "joint_refresh_on_safe_archive_change",
        "joint_refresh_on_probation_promotion", "joint_refresh_on_representative_change",
        "joint_refresh_interval_epochs", "joint_refresh_force_final_epoch",
        "joint_refresh_min_new_safe_candidates", "joint_refresh_max_dirty_candidates_per_agent",
        "joint_refresh_skip_when_no_dirty_prompt", "tcs_repair_candidates_per_parent",
        "open_exploration_candidates_per_parent",
        "split_integrity_json",
)


def _normalize_behavior_config_types(payload):
    defaults = Config()
    normalized = dict(payload)
    for field, value in list(normalized.items()):
        if isinstance(getattr(defaults, field, None), bool):
            value = bool(value)
        normalized[field] = value
    return normalized


def checkpoint_behavior_config(cfg):
    payload = _normalize_behavior_config_types(
        {field: getattr(cfg, field, None) for field in BEHAVIOR_CONFIG_FIELDS}
    )
    if bool(getattr(cfg, "competence_depth_enabled", False)):
        payload["effective_aggregation_mode"] = canonical_aggregation_mode(
            str(getattr(cfg, "aggregation_mode", "majority") or "majority")
        )
        payload["plurality_boundary_version"] = "plurality_boundary_v1"
    if not bool(getattr(cfg, "competence_depth_enabled", False)):
        for field in (
            "student_candidate_prompt_soft_max_chars", "student_candidate_prompt_hard_max_chars",
            "competence_depth_enabled", "competence_depth2_aux_enabled",
            "competence_progressive_residual_enabled", "competence_floor_low", "competence_floor_high",
            "competence_selector_weight", "competence_extra_support_shrinkage",
            "competence_weight_accuracy_gain", "competence_weight_accuracy_loss",
            "competence_weight_depth2_gain", "competence_weight_depth2_loss",
            "competence_weight_vote_gain_early", "competence_weight_vote_loss_early",
            "competence_schedule_version",
            "competence_schedule_mode", "competence_probe_size", "competence_probe_seed_offset",
            "competence_relative_low_delta", "competence_relative_high_delta",
            "competence_schedule_ema", "competence_schedule_max_step", "competence_schedule_monotonic",
            "competence_mean_guard_epsilon", "competence_c1_guard_epsilon", "competence_c2_guard_epsilon",
            "competence_depth1_candidate_guard_enabled", "competence_depth1_candidate_guard_epsilon",
            "competence_min_effective_specialization_epochs",
        ):
            payload.pop(field, None)
    if str(getattr(cfg, "competence_schedule_mode", "absolute_legacy")) != "baseline_relative_opt_snapshot":
        for field in (
            "competence_schedule_mode", "competence_probe_size", "competence_probe_seed_offset",
            "competence_relative_low_delta", "competence_relative_high_delta", "competence_schedule_ema",
            "competence_schedule_max_step", "competence_schedule_monotonic", "competence_mean_guard_epsilon",
            "competence_c1_guard_epsilon", "competence_c2_guard_epsilon",
            "competence_depth1_candidate_guard_enabled", "competence_depth1_candidate_guard_epsilon",
            "competence_min_effective_specialization_epochs",
        ):
            payload.pop(field, None)
    if str(getattr(cfg, "method_version", "legacy")) not in {"v8_2_hybrid_progressive", "v8_stable_qd_lineage"}:
        for field in (
            "method_version", "target_selector_mode", "target_selector_version", "beam_policy_version",
            "tcs_candidate_policy_version", "mechanism_signature_version", "competence_weight_depth1_gain",
            "competence_weight_depth1_loss", "competence_residual_floor",
            "catastrophic_target_accuracy_loss_epsilon", "soft_guard_error_dependence_weight",
            "soft_guard_cycle_weight", "soft_guard_mechanism_shift_weight",
            "soft_guard_accuracy_regression_weight", "mechanism_novelty_bonus_weight",
        ):
            payload.pop(field, None)
    qd_fields = (
        "active_team_selector_version", "lineage_policy_version", "mechanism_distance_version",
        "mechanism_sequence_distance_weight", "mechanism_embedding_distance_weight",
        "mechanism_near_duplicate_similarity_threshold", "behavior_correct_set_weight",
        "behavior_rescue_weight", "behavior_shared_wrong_weight", "behavior_support_shrinkage",
        "team_diversity_mean_behavior_weight", "team_diversity_min_behavior_weight",
        "team_diversity_mechanism_weight", "team_diversity_rescue_balance_weight",
        "joint_team_vote_epsilon_questions", "joint_team_mean_epsilon_questions",
        "joint_team_bottom2_epsilon_questions", "joint_team_c1_epsilon_questions",
        "joint_team_c2_epsilon_questions", "joint_team_per_agent_accuracy_epsilon",
        "lineage_provisional_epochs", "lineage_commit_epochs", "lineage_switch_confirmation_epochs",
        "lineage_mechanism_drift_weight", "lineage_behavior_drift_weight",
        "lineage_soft_drift_threshold", "lineage_hard_drift_threshold",
        "lineage_switch_min_accuracy_gain", "lineage_switch_min_vote_gain",
        "peer_collapse_soft_similarity", "peer_collapse_hard_similarity",
        "validation_stable_specialization_tie_break_enabled",
        "candidate_refill_version", "archive_policy_version", "joint_quality_filter_version",
        "probe_stability_version", "parent_selection_version", "candidate_refill_enabled",
        "candidate_refill_max_rounds", "candidate_refill_candidates_per_round",
        "candidate_refill_max_unique_candidates_per_parent", "candidate_refill_min_safe_non_incumbent",
        "candidate_refill_require_task_repair", "candidate_refill_require_distinct_mechanism",
        "candidate_refill_feed_rejection_reasons", "candidate_refill_stop_when_requirements_met",
        "candidate_refill_max_solver_calls_per_agent_update", "probation_archive_enabled",
        "probation_archive_size_per_agent", "probation_archive_ttl_updates", "probation_max_accuracy_loss",
        "probation_max_c1_loss_questions", "probation_max_c2_loss_questions",
        "probation_require_mechanism_novelty", "candidate_c1_catastrophic_loss_questions",
        "candidate_c2_catastrophic_loss_questions", "qd_archive_size_per_agent",
        "joint_representative_beam_size", "qd_parent_selection_mode",
        "qd_niche_min_parent_opportunities_per_epoch", "probation_parent_enabled",
        "probe_stability_fold_count", "probe_stability_seed_offset", "joint_vote_band_questions",
        "joint_mean_band_correct_count", "joint_bottom2_band_correct_count", "joint_c1_band_questions",
        "joint_c2_band_questions", "joint_allowed_vote_loss_questions", "joint_allowed_c1_loss_questions",
        "joint_allowed_c2_loss_questions", "joint_allowed_total_agent_correct_loss",
        "joint_allowed_bottom2_correct_loss", "joint_allowed_per_agent_correct_loss",
        "joint_team_max_active_changes_early", "joint_team_max_active_changes_late",
        "joint_team_change_limit_switch_strength", "joint_team_no_diversification_patience",
        "joint_team_change_limit_relaxation", "lineage_commit_required_snapshots",
        "lineage_switch_confirmation_snapshots", "qd_readiness_min_distinct_niches",
        "qd_readiness_min_diversity", "qd_readiness_max_fold_gap", "residual_specialization_qd_floor",
        "behavior_error_overlap_weight", "behavior_wrong_answer_dispersion_weight",
        "behavior_wrong_support_shrinkage", "min_optimizer_updates_per_agent_per_epoch",
        "target_selector_fairness_enabled",
    )
    if str(getattr(cfg, "method_version", "legacy")) != "v8_stable_qd_lineage":
        for field in qd_fields:
            payload.pop(field, None)
    else:
        for field in (
            "teacher_rewrite_max_count", "teacher_critic_direct_pass_threshold",
            "teacher_critic_rewrite_threshold", "teacher_critic_forced_best_threshold",
            "legacy_beam_rescore_each_epoch", "candidate_generation_policy_version",
            "joint_refresh_policy_version", "representative_probe_policy_version",
            "joint_refresh_mode", "joint_refresh_on_safe_archive_change",
            "joint_refresh_on_probation_promotion", "joint_refresh_on_representative_change",
            "joint_refresh_interval_epochs", "joint_refresh_force_final_epoch",
            "joint_refresh_min_new_safe_candidates", "joint_refresh_max_dirty_candidates_per_agent",
            "joint_refresh_skip_when_no_dirty_prompt", "tcs_repair_candidates_per_parent",
            "open_exploration_candidates_per_parent",
        ):
            payload[field] = getattr(cfg, field, None)
    if str(getattr(cfg, "method_version", "legacy")) != "v8_stable_qd_lineage":
        for field in (
            "teacher_rewrite_max_count", "teacher_critic_direct_pass_threshold",
            "teacher_critic_rewrite_threshold", "teacher_critic_forced_best_threshold",
            "legacy_beam_rescore_each_epoch", "candidate_generation_policy_version",
            "joint_refresh_policy_version", "representative_probe_policy_version",
            "joint_refresh_mode", "joint_refresh_on_safe_archive_change",
            "joint_refresh_on_probation_promotion", "joint_refresh_on_representative_change",
            "joint_refresh_interval_epochs", "joint_refresh_force_final_epoch",
            "joint_refresh_min_new_safe_candidates", "joint_refresh_max_dirty_candidates_per_agent",
            "joint_refresh_skip_when_no_dirty_prompt", "tcs_repair_candidates_per_parent",
            "open_exploration_candidates_per_parent",
        ):
            payload.pop(field, None)
    if str(getattr(cfg, "reward_mode", "")) != "coverage_useful_diversity":
        for field in (
            "reward_weight_coverage", "reward_weight_useful_diversity", "reward_weight_coverage_early",
            "reward_weight_coverage_late", "reward_weight_useful_diversity_early",
            "reward_weight_useful_diversity_late",
        ):
            payload.pop(field, None)
    return payload


def checkpoint_behavior_config_fingerprint(cfg):
    payload = checkpoint_behavior_config(cfg)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_config_signature(cfg):
    """Compatibility alias retained for older callers and tests."""
    return checkpoint_behavior_config(cfg)


def build_training_checkpoint(
    cfg,
    system,
    *,
    epoch_index,
    cursor,
    order,
    train_accumulators,
    best_score,
    best_epoch,
    epochs_without_improvement,
    stopped_early,
    no_effective_evolution_counter,
    no_effective_evolution_stopped,
    no_effective_evolution_reason,
    stage="training",
    epoch_record=None,
):
    payload = {
        "version": CHECKPOINT_VERSION,
        "stage": str(stage),
        "updated_at": time.time(),
        "seed": int(cfg.seed),
        "execution_session_id": str(getattr(system, "execution_session_id", "") or ""),
        "epochs": int(cfg.epochs),
        "train_size": int(cfg.train_size),
        "behavior_config": checkpoint_behavior_config(cfg),
        "behavior_config_fingerprint": checkpoint_behavior_config_fingerprint(cfg),
        "config_signature": checkpoint_config_signature(cfg),
        "epoch_index": int(epoch_index),
        "cursor": int(cursor),
        "order": [int(x) for x in order],
        "train_accumulators": train_accumulators,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "stopped_early": bool(stopped_early),
        "no_effective_evolution_counter": int(no_effective_evolution_counter),
        "no_effective_evolution_stopped": bool(no_effective_evolution_stopped),
        "no_effective_evolution_reason": str(no_effective_evolution_reason),
        "state": {
            "recent_window_records": list(getattr(system, "recent_window_records", [])),
            "specialization_strength": float(getattr(system, "specialization_strength", 0.0)),
            "effective_residual_strength": float(getattr(system, "effective_residual_strength", 0.0)),
            "previous_epoch_per_agent_acc": list(getattr(system, "previous_epoch_per_agent_acc", [])),
            "previous_epoch_bottom2_mean_acc": float(getattr(system, "previous_epoch_bottom2_mean_acc", 0.0)),
            "competence_phase_epoch": int(getattr(system, "competence_phase_epoch", 1)),
            "competence_schedule_version": str(getattr(system, "competence_schedule_version", "competence_depth_v1")),
            "specialization_strength_history": list(getattr(system, "specialization_strength_history", [0.0])),
            "competence_probe_indices": list(getattr(system, "competence_probe_indices", [])),
            "competence_probe_question_hashes": list(getattr(system, "competence_probe_question_hashes", [])),
            "initial_competence_probe_metrics": dict(getattr(system, "initial_competence_probe_metrics", {})),
            "latest_competence_probe_metrics": dict(getattr(system, "latest_competence_probe_metrics", {})),
            "competence_probe_history": list(getattr(system, "competence_probe_history", [])),
            "initial_active_prompt_hashes": list(getattr(system, "initial_active_prompt_hashes", [])),
            "first_nonzero_specialization_epoch": getattr(system, "first_nonzero_specialization_epoch", None),
            "effective_specialization_epoch_count": int(getattr(system, "effective_specialization_epoch_count", 0)),
            "depth1_guard_rejection_count": int(getattr(system, "depth1_guard_rejection_count", 0)),
            "catastrophic_accuracy_guard_rejection_count": int(getattr(system, "catastrophic_accuracy_guard_rejection_count", 0)),
            "soft_error_dependence_penalty_count": int(getattr(system, "soft_error_dependence_penalty_count", 0)),
            "soft_cycle_penalty_count": int(getattr(system, "soft_cycle_penalty_count", 0)),
            "soft_mechanism_shift_penalty_count": int(getattr(system, "soft_mechanism_shift_penalty_count", 0)),
            "exploration_candidate_count": int(getattr(system, "exploration_candidate_count", 0)),
            "exploration_slot_occupancy_count": int(getattr(system, "exploration_slot_occupancy_count", 0)),
            "exploration_to_active_conversion_count": int(getattr(system, "exploration_to_active_conversion_count", 0)),
            "hybrid_selector_history": list(getattr(system, "hybrid_selector_history", [])),
            "mechanism_signature_history": list(getattr(system, "mechanism_signature_history", [])),
            "mechanism_signature_by_prompt_hash": dict(getattr(system, "mechanism_signature_by_prompt_hash", {})),
            "beam_slot_state": dict(getattr(system, "beam_slot_state", {})),
            "exploration_slot_candidates": list(getattr(system, "exploration_slot_candidates", [])),
            "prompt_overlength_rejection_count": int(getattr(system, "prompt_overlength_rejection_count", 0)),
            "truncated_prompt_count": int(getattr(system, "truncated_prompt_count", 0)),
            "mechanism_embedding_cache": dict(getattr(system, "mechanism_embedding_cache", {})),
            "semantic_mechanism_families": dict(getattr(system, "semantic_mechanism_families", {})),
            "prompt_probe_cache": dict(getattr(system, "prompt_probe_cache", {})),
            "mechanism_embedding_cache_hit_count": int(getattr(system, "mechanism_embedding_cache_hit_count", 0)),
            "mechanism_embedding_cache_miss_count": int(getattr(system, "mechanism_embedding_cache_miss_count", 0)),
            "full_probe_cache_hit_count": int(getattr(system, "full_probe_cache_hit_count", 0)),
            "full_probe_missing_pair_evaluation_count": int(getattr(system, "full_probe_missing_pair_evaluation_count", 0)),
            "behavior_profile_by_prompt_hash": dict(getattr(system, "behavior_profile_by_prompt_hash", {})),
            "candidate_channel_funnel": dict(getattr(system, "candidate_channel_funnel", {})),
            "candidate_channel_funnel_seen": serialize_funnel_seen(
                getattr(system, "candidate_channel_funnel_seen", {})
            ),
            "joint_team_selection_history": list(getattr(system, "joint_team_selection_history", [])),
            "lineage_history": list(getattr(system, "lineage_history", [])),
            "quality_diversity_archive_history": list(getattr(system, "quality_diversity_archive_history", [])),
            "behavior_profile_history": list(getattr(system, "behavior_profile_history", [])),
            "latest_joint_team_metrics": dict(getattr(system, "latest_joint_team_metrics", {})),
            "quality_anchor_archive": list(getattr(system, "quality_anchor_archive", [])),
            "quality_anchor_created_count": int(getattr(system, "quality_anchor_created_count", 0)),
            "last_joint_refresh_epoch": int(getattr(system, "last_joint_refresh_epoch", 0)),
            "epochs_since_last_joint_refresh": int(getattr(system, "epochs_since_last_joint_refresh", 0)),
            "archive_material_change_version": int(getattr(system, "archive_material_change_version", 0)),
            "representative_version_per_agent": dict(getattr(system, "representative_version_per_agent", {})),
            "dirty_prompt_hashes": dict(getattr(system, "dirty_prompt_hashes", {})),
            "prompt_probe_version": str(getattr(system, "prompt_probe_version", "legacy")),
            "current_fixed_probe_hash": str(getattr(system, "current_fixed_probe_hash", "")),
            "last_archive_material_snapshot": dict(getattr(system, "last_archive_material_snapshot", {})),
            "last_representative_snapshot": dict(getattr(system, "last_representative_snapshot", {})),
            "last_active_prompt_hashes": list(getattr(system, "last_active_prompt_hashes", [])),
            "last_probation_promotion_count": int(getattr(system, "last_probation_promotion_count", 0)),
            "joint_refresh_count": int(getattr(system, "joint_refresh_count", 0)),
            "joint_refresh_skipped_count": int(getattr(system, "joint_refresh_skipped_count", 0)),
            "legacy_beam_refresh_call_count": int(getattr(system, "legacy_beam_refresh_call_count", 0)),
            "new_full_probe_prompt_count": int(getattr(system, "new_full_probe_prompt_count", 0)),
            "offline_team_combination_count": int(getattr(system, "offline_team_combination_count", 0)),
            "joint_team_solver_call_count": int(getattr(system, "joint_team_solver_call_count", 0)),
            "tcs_repair_generation_count": int(getattr(system, "tcs_repair_generation_count", 0)),
            "open_exploration_generation_count": int(getattr(system, "open_exploration_generation_count", 0)),
            "tcs_repair_candidate_count": int(getattr(system, "tcs_repair_candidate_count", 0)),
            "open_exploration_candidate_count": int(getattr(system, "open_exploration_candidate_count", 0)),
            "total_agent_update_count": int(getattr(system, "total_agent_update_count", 0)),
            "task_repair_niche_occupancy_count": int(getattr(system, "task_repair_niche_occupancy_count", 0)),
            "mechanism_niche_occupancy_count": int(getattr(system, "mechanism_niche_occupancy_count", 0)),
            "peer_collapse_soft_count": int(getattr(system, "peer_collapse_soft_count", 0)),
            "peer_collapse_hard_rejection_count": int(getattr(system, "peer_collapse_hard_rejection_count", 0)),
            "qd_no_diversification_epochs": int(getattr(system, "qd_no_diversification_epochs", 0)),
            "qd_change_limit_relaxed_epoch": int(getattr(system, "qd_change_limit_relaxed_epoch", -1)),
            "qd_previous_active_niche_count": int(getattr(system, "qd_previous_active_niche_count", 0)),
            "probation_to_safe_conversion_count": int(getattr(system, "probation_to_safe_conversion_count", 0)),
            "probation_expired_count": int(getattr(system, "probation_expired_count", 0)),
            "candidate_starvation_count": int(getattr(system, "candidate_starvation_count", 0)),
            "mechanism_starvation_count": int(getattr(system, "mechanism_starvation_count", 0)),
            "search_branch_starvation_count": int(getattr(system, "search_branch_starvation_count", 0)),
            "refill_requirements_unmet_count": int(getattr(system, "refill_requirements_unmet_count", 0)),
            "per_agent_optimizer_update_count": dict(getattr(system, "per_agent_optimizer_update_count", {})),
            "python_random_state": random.getstate(),
            "numpy_random_state": (
                lambda state: [state[0], state[1].tolist(), state[2], state[3], state[4]]
            )(np.random.get_state()),
            "agents": [
                {
                    "agent_id": i,
                    "initial_prompt": a.initial_prompt,
                    "current_prompt": a.current_prompt,
                    "prompt_beam": a.prompt_beam,
                    "history": a.history,
                    "accept_count": a.accept_count,
                    "reject_count": a.reject_count,
                    **a.trajectory_state_dict(),
                }
                for i, a in enumerate(system.agents)
            ],
        },
    }
    if epoch_record is not None:
        payload["epoch_record"] = epoch_record
    return payload


def write_training_checkpoint(cfg, system, **kwargs):
    write_json_atomic(checkpoint_path(cfg), build_training_checkpoint(cfg, system, **kwargs))


def clear_training_checkpoint(cfg):
    path = checkpoint_path(cfg)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def checkpoint_incompatibility_reasons(payload, cfg, train_data):
    reasons = []
    if not isinstance(payload, dict):
        return ["checkpoint payload is missing or is not a JSON object"]
    version = int(payload.get("version", 0) or 0)
    if version not in {5, CHECKPOINT_VERSION}:
        reasons.append(f"version: checkpoint={payload.get('version')!r} current={CHECKPOINT_VERSION}")
    if (
        version == CHECKPOINT_VERSION
        and str(getattr(cfg, "method_version", "legacy")) == "v8_stable_qd_lineage"
        and not isinstance(payload.get("behavior_config"), dict)
    ):
        reasons.append("Stable-QD checkpoint lacks the event-driven refresh and dual-channel policy fingerprint")
    if version == 5 and str(getattr(cfg, "method_version", "legacy")) == "v8_stable_qd_lineage":
        state = payload.get("state", {}) if isinstance(payload.get("state", {}), dict) else {}
        history = state.get("joint_team_selection_history", [])
        migratable = any(
            isinstance(row, dict) and row.get("selected_prompt_hashes") and row.get("selected_metrics")
            for row in history if isinstance(history, list)
        )
        if not migratable:
            reasons.append("version 5 Stable-QD checkpoint lacks real selected-team anchor evidence required for v6 migration")
    if int(payload.get("seed", -1)) != int(cfg.seed):
        reasons.append(f"seed: checkpoint={payload.get('seed')!r} current={cfg.seed!r}")
    if int(payload.get("epochs", -1)) != int(cfg.epochs):
        reasons.append(f"epochs: checkpoint={payload.get('epochs')!r} current={cfg.epochs!r}")
    if int(payload.get("train_size", -1)) != int(cfg.train_size):
        reasons.append(f"train_size: checkpoint={payload.get('train_size')!r} current={cfg.train_size!r}")
    saved_config = payload.get("behavior_config")
    saved_fingerprint = str(payload.get("behavior_config_fingerprint", "") or "")
    current_config = checkpoint_behavior_config(cfg)
    current_fingerprint = checkpoint_behavior_config_fingerprint(cfg)
    if not isinstance(saved_config, dict) or not saved_fingerprint:
        reasons.append("behavior_config: checkpoint behavior configuration or fingerprint is missing")
    elif saved_fingerprint != current_fingerprint and _normalize_behavior_config_types(saved_config) != current_config:
        reasons.append(
            "behavior_config_fingerprint: checkpoint and current optimization behavior differ"
        )
        if (
            str(saved_config.get("method_version", "")) == "v8_2_hybrid_progressive"
            and str(current_config.get("method_version", "")) == "v8_stable_qd_lineage"
        ):
            reasons.append("V8 behavior fingerprint mismatch: joint quality-diversity lineage policy changed")
        if (
            str(saved_config.get("method_version", "")) == "v8_stable_qd_lineage"
            and not str(saved_config.get("candidate_refill_version", "") or "")
        ):
            reasons.append("Stable-QD checkpoint predates the refill/probation search-loop policy")
        for key in sorted(set(BEHAVIOR_CONFIG_FIELDS) | set(saved_config) | set(current_config)):
            saved_value = saved_config.get(key)
            current_value = current_config.get(key)
            if json.dumps(saved_value, sort_keys=True, default=str) != json.dumps(
                current_value, sort_keys=True, default=str
            ):
                label = f"{key} mismatch" if key in {"competence_schedule_version", "competence_schedule_mode"} else key
                reasons.append(f"{label}: checkpoint={saved_value!r} current={current_value!r}")
    epoch_index = int(payload.get("epoch_index", -1))
    if epoch_index < 0 or epoch_index > int(cfg.epochs):
        reasons.append(f"epoch_index: checkpoint={payload.get('epoch_index')!r} current_epochs={cfg.epochs!r}")
    stage = str(payload.get("stage", "training") or "training")
    order = payload.get("order", [])
    if stage in {"between_epochs", "epoch_evaluated"}:
        if not isinstance(order, list):
            reasons.append("order: checkpoint value is not a list")
        return reasons
    if stage != "training":
        reasons.append(f"stage: unsupported checkpoint stage {stage!r}")
        return reasons
    cursor = int(payload.get("cursor", -1))
    if not isinstance(order, list):
        reasons.append("order: checkpoint value is not a list")
    elif len(order) != len(train_data):
        reasons.append(f"order length: checkpoint={len(order)} current_train={len(train_data)}")
    if not (0 <= cursor <= len(order) if isinstance(order, list) else False):
        reasons.append(f"cursor: checkpoint={payload.get('cursor')!r} order_length={len(order) if isinstance(order, list) else 'invalid'}")
    state = payload.get("state", {})
    saved_window = state.get("recent_window_records") if isinstance(state, dict) else None
    expected_window_size = cursor % max(1, int(cfg.update_every))
    if saved_window is None and expected_window_size:
        reasons.append(
            "recent_window_records: checkpoint stopped inside an update window but does not contain window state"
        )
    elif saved_window is not None and not isinstance(saved_window, list):
        reasons.append("recent_window_records: checkpoint value is not a list")
    elif isinstance(saved_window, list) and len(saved_window) != expected_window_size:
        reasons.append(
            f"recent_window_records: checkpoint={len(saved_window)} expected={expected_window_size} for cursor={cursor}"
        )

    # Old resume code could advance a boundary after silently losing its window.
    train_step_path = os.path.join(cfg.out_dir, "train_step_logs.jsonl")
    if cursor > 0 and cursor % max(1, int(cfg.update_every)) == 0 and os.path.exists(train_step_path):
        try:
            with open(train_step_path, "r", encoding="utf-8") as f:
                last_row = next((json.loads(line) for line in reversed(f.readlines()) if line.strip()), {})
            update_summary = last_row.get("update_summary", {}) if isinstance(last_row, dict) else {}
            if (
                int(last_row.get("epoch", 0) or 0) == epoch_index + 1
                and int(last_row.get("step", 0) or 0) == cursor
                and isinstance(update_summary, dict)
                and str(update_summary.get("skipped_reason", "")) == "window_not_ready"
            ):
                reasons.append(
                    "recent_window_records: update boundary was skipped as window_not_ready; this run cannot be resumed faithfully"
                )
        except (OSError, json.JSONDecodeError):
            pass
    return reasons


def checkpoint_compatible(payload, cfg, train_data):
    return not checkpoint_incompatibility_reasons(payload, cfg, train_data)


def migrate_checkpoint(payload, cfg):
    """Migrate only the immediately preceding checkpoint version."""
    version = int(payload.get("version", 0) or 0) if isinstance(payload, dict) else 0
    if version == CHECKPOINT_VERSION:
        return payload
    if version != 5:
        raise ValueError(f"checkpoint version {version} cannot migrate to {CHECKPOINT_VERSION}")
    migrated = json.loads(json.dumps(payload))
    state = migrated.setdefault("state", {})
    state.setdefault("semantic_mechanism_families", {})
    anchors = []
    if str(getattr(cfg, "method_version", "legacy")) == "v8_stable_qd_lineage":
        for order, row in enumerate(state.get("joint_team_selection_history", [])):
            if not isinstance(row, dict) or not row.get("selected_prompt_hashes") or not row.get("selected_metrics"):
                continue
            team = {**dict(row["selected_metrics"]), "prompt_hashes": list(row["selected_prompt_hashes"])}
            anchors.append(build_quality_anchor(team, epoch=int(row.get("epoch", 0) or 0), created_order=order))
        if not anchors:
            raise ValueError("version 5 Stable-QD checkpoint has no real selected-team anchor evidence")
    frontier = update_quality_anchor_archive([], anchors, capacity=int(getattr(cfg, "quality_anchor_archive_size", 5)))
    state["quality_anchor_archive"] = [anchor.to_dict() for anchor in frontier]
    state["quality_anchor_created_count"] = len(anchors)
    state.pop("joint_quality_anchor_metrics", None)
    migrated["version"] = CHECKPOINT_VERSION
    return migrated


def abort_incompatible_checkpoint(cfg, reasons):
    print(
        "[RESUME] ERROR: Incompatible training_checkpoint.json; refusing to start from scratch in the same run directory.",
        flush=True,
    )
    print(f"[RESUME] Checkpoint path: {checkpoint_path(cfg)}", flush=True)
    for reason in reasons[:20]:
        print(f"[RESUME] Incompatibility: {reason}", flush=True)
    if len(reasons) > 20:
        print(f"[RESUME] Incompatibility: ... {len(reasons) - 20} more", flush=True)
    raise SystemExit(2)
