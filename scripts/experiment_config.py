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


@dataclass(frozen=True)
class DatasetPaths:
    task_type: str
    train: str
    val: str
    test: str


DEFAULT_EXPERIMENT_SETTINGS = [
    ExperimentSetting("shared_baseline", "shared", True, "guarded_diversity"),
    ExperimentSetting("bank_baseline", "bank", True, "guarded_diversity"),
    ExperimentSetting("shared_guarded_beam", "shared", False, "guarded_diversity"),
    ExperimentSetting("bank_guarded_beam", "bank", False, "guarded_diversity"),
    ExperimentSetting(
        "shared_oracle_pareto_tcs",
        "shared",
        False,
        "coverage_useful_diversity",
        candidate_selection_mode="oracle_pareto",
        best_state_selection_mode="oracle_first",
        optimizer_architecture="teacher_critic_student",
        optimizer_fallback_mode="none",
        teacher_critic_use_voting_failure=False,
        candidate_eval_strategy="fixed_pool",
        candidate_eval_pool_size=100,
        candidate_eval_batch_size=24,
    ),
]


DEFAULT_DATASET_PATHS: Dict[str, DatasetPaths] = {
    "mmlu": DatasetPaths("mmlu", "mmlu_train.jsonl", "mmlu_val.jsonl", "mmlu_test.jsonl"),
    "bbh": DatasetPaths("bbh", "bbh_train.jsonl", "bbh_val.jsonl", "bbh_test.jsonl"),
}


DEFAULT_SEED_BASELINES = 1


def setting_names(settings: Iterable[ExperimentSetting] = DEFAULT_EXPERIMENT_SETTINGS) -> List[str]:
    return [setting.name for setting in settings]


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def select_settings(raw: str, settings: Iterable[ExperimentSetting] = DEFAULT_EXPERIMENT_SETTINGS) -> List[ExperimentSetting]:
    available = list(settings)
    if not raw or str(raw).strip().lower() == "all":
        return available
    wanted = set(parse_csv_list(raw))
    selected = [setting for setting in available if setting.name in wanted]
    missing = wanted - {setting.name for setting in selected}
    if missing:
        raise ValueError(f"Unknown run_settings: {sorted(missing)}")
    return selected


def setting_from_run_name(name: str, settings: Iterable[ExperimentSetting] = DEFAULT_EXPERIMENT_SETTINGS) -> str:
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
