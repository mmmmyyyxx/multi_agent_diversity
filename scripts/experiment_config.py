from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.config_sections import canonical_field_registry


@dataclass(frozen=True, init=False)
class ExperimentPreset:
    name: str
    base: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        init_mode: str | None = None,
        baseline_only: bool | None = None,
        reward_mode: str | None = None,
        *,
        base: str | None = None,
        overrides: Mapping[str, Any] | None = None,
        **legacy_overrides: Any,
    ) -> None:
        values = dict(overrides or {})
        if init_mode is not None:
            legacy_overrides["init_mode"] = init_mode
        if baseline_only is not None:
            legacy_overrides["baseline_only"] = baseline_only
        if reward_mode is not None:
            legacy_overrides["reward_mode"] = reward_mode
        registry = canonical_field_registry(Config)
        for key, value in legacy_overrides.items():
            if key not in registry:
                raise ValueError(f"Unknown preset override for {name}: {key}")
            values[registry[key]] = value
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "overrides", values)

    def resolved_overrides(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        if self.base:
            values.update(PRESET_BASES[self.base])
        for path, value in self.overrides.items():
            flat_name = path.rsplit(".", 1)[-1]
            values[flat_name] = value
        return values

    def __getattr__(self, name: str) -> Any:
        values = self.resolved_overrides()
        if name in values:
            return values[name]
        if name in canonical_field_registry(Config):
            return None
        raise AttributeError(name)

    def compatibility_dict(self) -> Dict[str, Any]:
        """Expose the old sparse setting view without restoring 71 fields."""
        values = {name: None for name in canonical_field_registry(Config)}
        values.update(self.resolved_overrides())
        values["name"] = self.name
        return values


ExperimentSetting = ExperimentPreset


@dataclass(frozen=True)
class DatasetPaths:
    task_type: str
    train: str
    val: str
    test: str


SHARED_TCS_SEARCH_BASE = {
    "init_mode": "shared",
    "baseline_only": False,
    "best_state_selection_mode": "vote_first",
    "optimizer_architecture": "teacher_critic_student",
    "optimizer_fallback_mode": "none",
    "teacher_critic_use_voting_failure": True,
    "candidate_eval_strategy": "fixed_pool",
    "candidate_eval_pool_size": 50,
    "candidate_eval_batch_size": 24,
    "candidate_eval_execution_mode": "factorized_cached",
    "solver_rollout_singleflight": True,
    "candidate_eval_prompt_dedup": True,
    "candidate_eval_cache_logging": True,
}

SHARED_VOTE_SEARCH_BASE = {
    **SHARED_TCS_SEARCH_BASE,
    "reward_mode": "vote_useful_diversity",
}

PRESET_BASES = {
    "shared_tcs_base": SHARED_TCS_SEARCH_BASE,
    "shared_vote_base": SHARED_VOTE_SEARCH_BASE,
}


def _preset(name: str, *, base: str | None = None, **overrides: Any) -> ExperimentPreset:
    registry = canonical_field_registry(Config)
    inherited = PRESET_BASES.get(base or "", {})
    unknown = sorted((set(inherited) | set(overrides)) - set(registry))
    if unknown:
        raise ValueError(f"Unknown preset overrides for {name}: {unknown}")
    dotted = {registry[key]: value for key, value in overrides.items()}
    return ExperimentPreset(name=name, base=base, overrides=dotted)


ALL_EXPERIMENT_SETTINGS = [
    ExperimentSetting("shared_baseline", "shared", True, "guarded_diversity"),
    ExperimentSetting("bank_baseline", "bank", True, "guarded_diversity"),
    ExperimentSetting("shared_guarded_beam", "shared", False, "guarded_diversity"),
    ExperimentSetting("bank_guarded_beam", "bank", False, "guarded_diversity"),
    ExperimentSetting(
        name="shared_scalar_tcs_vote_first",
        candidate_selection_mode="scalar_reward",
        **SHARED_VOTE_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_pareto_tcs",
        candidate_selection_mode="vote_pareto",
        **SHARED_VOTE_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_pareto_tcs_static",
        candidate_selection_mode="vote_pareto",
        reward_schedule_mode="static",
        **SHARED_VOTE_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_pareto_tcs_boundary_selector",
        candidate_selection_mode="vote_pareto",
        reward_schedule_mode="static",
        boundary_selector_enabled=True,
        **SHARED_VOTE_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_error_pareto_tcs",
        candidate_selection_mode="vote_error_pareto",
        reward_schedule_mode="static",
        boundary_selector_enabled=True,
        shared_error_metrics_enabled=True,
        error_dependence_guard_enabled=True,
        **SHARED_VOTE_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_error_pareto_tcs_residual_specialization",
        candidate_selection_mode="vote_error_pareto",
        reward_schedule_mode="static",
        boundary_selector_enabled=True,
        shared_error_metrics_enabled=True,
        residual_specialization_enabled=True,
        error_dependence_guard_enabled=True,
        capability_affinity_weight=0.25,
        capability_coverage_gap_weight=0.25,
        specialization_support_shrinkage=3.0,
        capability_loss_weight=1.5,
        specialization_update_period=2,
        **SHARED_VOTE_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_error_pareto_tcs_residual_cycle_guard",
        candidate_selection_mode="vote_error_pareto",
        reward_schedule_mode="static",
        boundary_selector_enabled=True,
        shared_error_metrics_enabled=True,
        residual_specialization_enabled=True,
        error_dependence_guard_enabled=True,
        residual_cycle_guard_enabled=True,
        mechanism_trust_region_enabled=True,
        capability_affinity_weight=0.25,
        capability_coverage_gap_weight=0.25,
        specialization_support_shrinkage=3.0,
        capability_loss_weight=1.5,
        specialization_update_period=2,
        pivotal_loss_guard_epsilon=0.0,
        shared_error_creation_epsilon=0.02,
        **SHARED_VOTE_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_legacy_coverage_useful_tcs_strict",
        candidate_selection_mode="scalar_reward",
        reward_mode="coverage_useful_diversity",
        reward_schedule_mode="phase_adaptive",
        residual_specialization_enabled=False,
        residual_cycle_guard_enabled=False,
        mechanism_trust_region_enabled=False,
        **SHARED_TCS_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_tcs_competence_schedule",
        candidate_selection_mode="competence_depth_pareto",
        reward_mode="competence_depth_schedule",
        best_state_selection_mode="vote_competence_first",
        competence_depth_enabled=True,
        competence_depth2_aux_enabled=False,
        competence_progressive_residual_enabled=False,
        **{key: value for key, value in SHARED_TCS_SEARCH_BASE.items() if key != "best_state_selection_mode"},
    ),
    ExperimentSetting(
        name="shared_vote_tcs_competence_depth2",
        candidate_selection_mode="competence_depth_pareto",
        reward_mode="competence_depth_schedule",
        best_state_selection_mode="vote_competence_first",
        competence_depth_enabled=True,
        competence_depth2_aux_enabled=True,
        competence_progressive_residual_enabled=False,
        **{key: value for key, value in SHARED_TCS_SEARCH_BASE.items() if key != "best_state_selection_mode"},
    ),
    ExperimentSetting(
        name="shared_vote_tcs_competence_depth2_progressive_residual",
        candidate_selection_mode="competence_depth_pareto",
        reward_mode="competence_depth_schedule",
        best_state_selection_mode="vote_competence_first",
        reward_schedule_mode="static",
        boundary_selector_enabled=True,
        shared_error_metrics_enabled=True,
        residual_specialization_enabled=True,
        error_dependence_guard_enabled=True,
        residual_cycle_guard_enabled=True,
        mechanism_trust_region_enabled=True,
        competence_depth_enabled=True,
        competence_depth2_aux_enabled=True,
        competence_progressive_residual_enabled=True,
        **{key: value for key, value in SHARED_TCS_SEARCH_BASE.items() if key != "best_state_selection_mode"},
    ),
    ExperimentSetting(
        name="shared_vote_tcs_competence_depth2_progressive_residual_hybrid",
        candidate_selection_mode="competence_depth_pareto",
        reward_mode="competence_depth_schedule",
        best_state_selection_mode="vote_generalization_first",
        reward_schedule_mode="static",
        boundary_selector_enabled=True,
        shared_error_metrics_enabled=True,
        residual_specialization_enabled=True,
        error_dependence_guard_enabled=True,
        residual_cycle_guard_enabled=True,
        mechanism_trust_region_enabled=True,
        competence_depth_enabled=True,
        competence_depth2_aux_enabled=True,
        competence_progressive_residual_enabled=True,
        competence_schedule_mode="baseline_relative_opt_snapshot",
        competence_schedule_version="competence_depth_v2_opt_snapshot_c1_guard",
        competence_probe_size=0,
        competence_probe_seed_offset=7000,
        competence_relative_low_delta=0.01,
        competence_relative_high_delta=0.06,
        competence_schedule_ema=0.50,
        competence_schedule_max_step=0.35,
        competence_schedule_monotonic=True,
        competence_mean_guard_epsilon=0.01,
        competence_c1_guard_epsilon=0.01,
        competence_c2_guard_epsilon=0.01,
        competence_depth1_candidate_guard_enabled=True,
        competence_depth1_candidate_guard_epsilon=0.0,
        competence_min_effective_specialization_epochs=1,
        method_version="v8_stable_qd_lineage",
        target_selector_mode="hybrid_competence_boundary",
        target_selector_version="hybrid_competence_boundary_v2",
        beam_policy_version="quality_diversity_archive_v1",
        active_team_selector_version="joint_quality_diversity_v1",
        lineage_policy_version="stable_lineage_anchor_v1",
        mechanism_distance_version="mechanism_sequence_embedding_v1",
        candidate_refill_version="quality_feedback_refill_v1",
        archive_policy_version="safe_probation_qd_archive_v1",
        joint_quality_filter_version="hierarchical_epsilon_band_v1",
        probe_stability_version="deterministic_two_fold_v1",
        parent_selection_version="active_plus_round_robin_niche_v1",
        tcs_candidate_policy_version="repair_mechanism_alternative_v1",
        candidate_generation_policy_version="tcs_repair_open_exploration_v1",
        joint_refresh_policy_version="event_driven_incremental_v1",
        representative_probe_policy_version="dirty_shortlist_probe_v1",
        mechanism_signature_version="mechanism_signature_v1",
        legacy_beam_rescore_each_epoch=False,
        beam_refresh_each_epoch=False,
        teacher_critic_max_rounds=2,
        teacher_rewrite_max_count=1,
        teacher_critic_direct_pass_threshold=0.75,
        teacher_critic_rewrite_threshold=0.50,
        teacher_critic_forced_best_threshold=0.60,
        tcs_repair_candidates_per_parent=1,
        open_exploration_candidates_per_parent=1,
        joint_refresh_mode="event_driven",
        joint_refresh_on_safe_archive_change=True,
        joint_refresh_on_probation_promotion=True,
        joint_refresh_on_representative_change=True,
        joint_refresh_interval_epochs=2,
        joint_refresh_force_final_epoch=True,
        joint_refresh_min_new_safe_candidates=1,
        joint_refresh_max_dirty_candidates_per_agent=2,
        joint_refresh_skip_when_no_dirty_prompt=True,
        competence_weight_depth1_gain=0.80,
        competence_weight_depth1_loss=1.20,
        competence_residual_floor=0.30,
        catastrophic_target_accuracy_loss_epsilon=0.05,
        soft_guard_error_dependence_weight=0.50,
        soft_guard_cycle_weight=0.20,
        soft_guard_mechanism_shift_weight=0.20,
        soft_guard_accuracy_regression_weight=0.50,
        mechanism_novelty_bonus_weight=0.20,
        **{key: value for key, value in SHARED_TCS_SEARCH_BASE.items() if key != "best_state_selection_mode"},
    ),
    ExperimentSetting(
        name="shared_accuracy_only_tcs_vote_first",
        reward_mode="accuracy_only",
        candidate_selection_mode="scalar_reward",
        **SHARED_TCS_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_accuracy_rollout_embedding_tcs",
        method_version="v8_accuracy_rollout_embedding",
        reward_mode="rollout_accuracy_diversity",
        candidate_selection_mode="scalar_reward",
        best_state_selection_mode="rollout_vote_first",
        beam_policy_version="rollout_archive_v1",
        active_team_selector_version="accuracy_rollout_joint_v1",
        candidate_generation_policy_version="tcs_repair_open_rollout_v1",
        tcs_candidate_policy_version="rollout_minimal_schema_v1",
        archive_policy_version="rollout_signature_archive_v1",
        joint_quality_filter_version="accuracy_first_rollout_v1",
        probe_stability_version="rollout_fixed_probe_v1",
        parent_selection_version="active_plus_rollout_representatives_v1",
        legacy_beam_rescore_each_epoch=False,
        beam_refresh_each_epoch=False,
        competence_depth_enabled=False,
        residual_specialization_enabled=False,
        **{key: value for key, value in SHARED_TCS_SEARCH_BASE.items() if key != "best_state_selection_mode"},
    ),
    ExperimentSetting(
        name="shared_vote_ready_rollout_diversity_tcs",
        method_version="v8_rollout_qd_vote_ready",
        reward_mode="rollout_vote_ready",
        candidate_selection_mode="rollout_vote_ready",
        best_state_selection_mode="rollout_vote_first",
        beam_policy_version="rollout_archive_v1",
        active_team_selector_version="vote_ready_rollout_joint_v1",
        candidate_generation_policy_version="tcs_repair_open_rollout_v1",
        tcs_candidate_policy_version="rollout_minimal_schema_v1",
        archive_policy_version="rollout_signature_archive_v1",
        joint_quality_filter_version="vote_c3_lexicographic_v1",
        probe_stability_version="rollout_fixed_probe_v1",
        parent_selection_version="active_plus_rollout_representatives_v1",
        legacy_beam_rescore_each_epoch=False,
        beam_refresh_each_epoch=False,
        competence_depth_enabled=False,
        residual_specialization_enabled=False,
        **{key: value for key, value in SHARED_TCS_SEARCH_BASE.items() if key != "best_state_selection_mode"},
    ),
    ExperimentSetting(
        name="shared_guarded_diversity_tcs_vote_first",
        reward_mode="guarded_diversity",
        candidate_selection_mode="scalar_reward",
        **SHARED_TCS_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_no_margin_tcs_vote_first",
        candidate_selection_mode="scalar_reward",
        **SHARED_VOTE_SEARCH_BASE,
    ),
    ExperimentSetting(
        name="shared_vote_no_boundary_tcs_vote_first",
        candidate_selection_mode="scalar_reward",
        **SHARED_VOTE_SEARCH_BASE,
    ),
]

# Historical batch runs intentionally remain the four baseline/guarded settings.
DEFAULT_EXPERIMENT_SETTING_NAMES = [
    "shared_baseline",
    "bank_baseline",
    "shared_guarded_beam",
    "bank_guarded_beam",
]
DEFAULT_EXPERIMENT_SETTINGS = ALL_EXPERIMENT_SETTINGS


DEFAULT_DATASET_PATHS: Dict[str, DatasetPaths] = {
    "mmlu": DatasetPaths("mmlu", "mmlu_train.jsonl", "mmlu_val.jsonl", "mmlu_test.jsonl"),
    "bbh": DatasetPaths("bbh", "bbh_train.jsonl", "bbh_val.jsonl", "bbh_test.jsonl"),
}


DEFAULT_SEED_BASELINES = 1


def setting_names(settings: Iterable[ExperimentSetting] = ALL_EXPERIMENT_SETTINGS) -> List[str]:
    return [setting.name for setting in settings]


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def select_settings(raw: str, settings: Iterable[ExperimentSetting] = ALL_EXPERIMENT_SETTINGS) -> List[ExperimentSetting]:
    available = list(settings)
    if not raw or str(raw).strip().lower() == "all":
        return available
    wanted = set(parse_csv_list(raw))
    selected = [setting for setting in available if setting.name in wanted]
    missing = wanted - {setting.name for setting in selected}
    if missing:
        raise ValueError(f"Unknown run_settings: {sorted(missing)}")
    return selected


def setting_from_run_name(name: str, settings: Iterable[ExperimentSetting] = ALL_EXPERIMENT_SETTINGS) -> str:
    text = str(name or "")
    for setting_name in setting_names(settings):
        if text == setting_name or text.startswith(f"{setting_name}_seed"):
            return setting_name
    return ""


def dataset_paths_from_args(args, dataset: str) -> Dict[str, str]:
    key = str(dataset or "").strip().lower()
    defaults = DEFAULT_DATASET_PATHS.get(key)
    if defaults is not None:
        return {
            "task_type": defaults.task_type,
            "train": getattr(args, f"{key}_train_path", defaults.train),
            "val": getattr(args, f"{key}_val_path", defaults.val),
            "test": getattr(args, f"{key}_test_path", defaults.test),
        }
    return {
        "task_type": getattr(args, "task_type", "auto"),
        "train": getattr(args, "train_path", "train.jsonl"),
        "val": getattr(args, "val_path", ""),
        "test": getattr(args, "test_path", "test.jsonl"),
    }
