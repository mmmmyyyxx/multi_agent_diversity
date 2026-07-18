from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ExperimentSetting:
    name: str
    init_mode: str
    baseline_only: bool
    reward_mode: str = "guarded_diversity"
    candidate_selection_mode: str = "scalar_reward"
    best_state_selection_mode: str = "existing"
    optimizer_architecture: str = ""
    optimizer_fallback_mode: str = ""
    teacher_critic_use_voting_failure: Optional[bool] = None
    candidate_eval_strategy: str = ""
    candidate_eval_pool_size: int = 0
    candidate_eval_batch_size: int = 0
    candidate_eval_execution_mode: str = ""
    solver_rollout_singleflight: Optional[bool] = None
    candidate_eval_prompt_dedup: Optional[bool] = None
    candidate_eval_cache_logging: Optional[bool] = None
    reward_schedule_mode: str = ""
    boundary_selector_enabled: Optional[bool] = None
    shared_error_metrics_enabled: Optional[bool] = None
    residual_specialization_enabled: Optional[bool] = None
    error_dependence_guard_enabled: Optional[bool] = None
    residual_cycle_guard_enabled: Optional[bool] = None
    mechanism_trust_region_enabled: Optional[bool] = None
    capability_affinity_weight: Optional[float] = None
    capability_coverage_gap_weight: Optional[float] = None
    specialization_support_shrinkage: Optional[float] = None
    capability_loss_weight: Optional[float] = None
    specialization_update_period: Optional[int] = None
    pivotal_loss_guard_epsilon: Optional[float] = None
    shared_error_creation_epsilon: Optional[float] = None
    competence_depth_enabled: Optional[bool] = None
    competence_depth2_aux_enabled: Optional[bool] = None
    competence_progressive_residual_enabled: Optional[bool] = None


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
        name="shared_accuracy_only_tcs_vote_first",
        reward_mode="accuracy_only",
        candidate_selection_mode="scalar_reward",
        **SHARED_TCS_SEARCH_BASE,
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
