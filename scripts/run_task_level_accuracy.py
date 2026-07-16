import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from multi_dataset_diverse_rl.cli import build_dataset
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.utils import load_jsonl
    from scripts.experiment_config import ALL_EXPERIMENT_SETTINGS, ExperimentSetting, parse_csv_list, select_settings
    from scripts.experiment_io import append_jsonl, read_json, write_csv, write_jsonl
    from scripts.task_level_accuracy_utils import ACCURACY_RESULT_COLUMNS, build_accuracy_result_row
except ModuleNotFoundError:
    from multi_dataset_diverse_rl.cli import build_dataset
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.utils import load_jsonl
    from experiment_config import ALL_EXPERIMENT_SETTINGS, ExperimentSetting, parse_csv_list, select_settings
    from experiment_io import append_jsonl, read_json, write_csv, write_jsonl
    from task_level_accuracy_utils import ACCURACY_RESULT_COLUMNS, build_accuracy_result_row

from multi_dataset_diverse_rl.task_manifest import ComparisonTask, load_task_manifest, resolve_task_ids


SETTINGS = ALL_EXPERIMENT_SETTINGS


def _selected_settings(raw: str) -> List[ExperimentSetting]:
    return select_settings(raw, SETTINGS)


def _setting_reward_mode(args: argparse.Namespace, setting: ExperimentSetting) -> str:
    override = str(getattr(args, "reward_mode", "") or "").strip()
    if override and not setting.baseline_only:
        return override
    return setting.reward_mode


def _setting_value(setting: ExperimentSetting, name: str, fallback: Any) -> Any:
    value = getattr(setting, name, None)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value or fallback
    if isinstance(value, int) and value == 0 and isinstance(fallback, int):
        return fallback
    return fallback if value is None else value


def _explicit_cli_or_setting(args: argparse.Namespace, setting: ExperimentSetting, name: str, default: Any) -> Any:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    return _setting_value(setting, name, default)


def _append_common_cli_args(
    cmd: List[str],
    args: argparse.Namespace,
    task: ComparisonTask,
    setting: ExperimentSetting,
    seed: int,
    split_integrity: Dict[str, Any] | None = None,
):
    reward_mode = _setting_reward_mode(args, setting)
    candidate_selection_mode = (
        setting.candidate_selection_mode
        if str(getattr(setting, "candidate_selection_mode", "") or "") in {"vote_pareto", "vote_error_pareto"}
        else getattr(args, "candidate_selection_mode", Config().candidate_selection_mode)
    )
    best_state_selection_mode = (
        setting.best_state_selection_mode
        if str(getattr(setting, "best_state_selection_mode", "") or "") == "vote_first"
        else getattr(args, "best_state_selection_mode", Config().best_state_selection_mode)
    )
    optimizer_architecture = _setting_value(setting, "optimizer_architecture", args.optimizer_architecture)
    optimizer_fallback_mode = _setting_value(setting, "optimizer_fallback_mode", args.optimizer_fallback_mode)
    teacher_voting_failure = _setting_value(setting, "teacher_critic_use_voting_failure", args.teacher_critic_use_voting_failure)
    candidate_eval_strategy = _explicit_cli_or_setting(args, setting, "candidate_eval_strategy", Config().candidate_eval_strategy)
    candidate_eval_pool_size = _explicit_cli_or_setting(args, setting, "candidate_eval_pool_size", Config().candidate_eval_pool_size)
    candidate_eval_execution_mode = _explicit_cli_or_setting(args, setting, "candidate_eval_execution_mode", Config().candidate_eval_execution_mode)
    defaults = Config()
    cmd.extend(
        [
            "--task_type", task.task_type,
            "--dataset_format", args.dataset_format,
            "--comparison_task_id", task.task_id,
            "--benchmark", task.benchmark,
            "--answer_format", task.answer_format,
            "--split_integrity_json", json.dumps(split_integrity or {}, sort_keys=True),
            "--agent_model", args.agent_model,
            "--optimizer_model", args.optimizer_model,
            "--evaluator_model", args.evaluator_model,
            "--search_mode", "evolutionary_beam",
            "--reward_mode", reward_mode,
            "--candidate_selection_mode", str(candidate_selection_mode),
            "--best_state_selection_mode", str(best_state_selection_mode),
            "--agents", str(args.agents),
            "--init_mode", setting.init_mode,
            "--shared_prompt", args.shared_prompt,
            "--beam_size", str(args.beam_size),
            "--num_candidates_per_parent", str(args.num_candidates_per_parent),
            "--optimizer_parent_concurrency", str(args.optimizer_parent_concurrency),
            "--beam_refresh_each_epoch", str(args.beam_refresh_each_epoch),
            "--accuracy_guard_epsilon", str(args.accuracy_guard_epsilon),
            "--reward_weight_div_delta", str(args.reward_weight_div_delta),
            "--reward_weight_invalid_delta", str(args.reward_weight_invalid_delta),
            "--reward_weight_vote_delta", str(args.reward_weight_vote_delta),
            "--reward_weight_vote_margin", str(args.reward_weight_vote_margin),
            "--reward_weight_boundary_diversity", str(args.reward_weight_boundary_diversity),
            "--invalid_guard_epsilon", str(args.invalid_guard_epsilon),
            "--use_baseline_relative_reward", str(args.use_baseline_relative_reward),
            "--reward_schedule_mode", str(getattr(setting, "reward_schedule_mode", "") or args.reward_schedule_mode),
            "--reward_diversity_warmup_updates", str(args.reward_diversity_warmup_updates),
            "--reward_weight_div_delta_early", str(args.reward_weight_div_delta_early),
            "--reward_weight_div_delta_late", str(args.reward_weight_div_delta_late),
            "--reward_weight_vote_delta_early", str(args.reward_weight_vote_delta_early),
            "--reward_weight_vote_delta_late", str(args.reward_weight_vote_delta_late),
            "--reward_weight_vote_margin_early", str(args.reward_weight_vote_margin_early),
            "--reward_weight_vote_margin_late", str(args.reward_weight_vote_margin_late),
            "--reward_weight_boundary_diversity_early", str(args.reward_weight_boundary_diversity_early),
            "--reward_weight_boundary_diversity_late", str(args.reward_weight_boundary_diversity_late),
            "--reward_weight_target_accuracy_early", str(args.reward_weight_target_accuracy_early),
            "--reward_weight_target_accuracy_late", str(args.reward_weight_target_accuracy_late),
            "--accuracy_guard_epsilon_early", str(args.accuracy_guard_epsilon_early),
            "--accuracy_guard_epsilon_late", str(args.accuracy_guard_epsilon_late),
            "--optimizer_architecture", str(optimizer_architecture),
            "--teacher_critic_max_rounds", str(args.teacher_critic_max_rounds),
            "--teacher_question_pass_threshold", str(args.teacher_question_pass_threshold),
            "--teacher_temperature", str(args.teacher_temperature),
            "--critic_temperature", str(args.critic_temperature),
            "--student_temperature", str(args.student_temperature),
            "--teacher_max_tokens", str(args.teacher_max_tokens),
            "--critic_max_tokens", str(args.critic_max_tokens),
            "--student_max_tokens", str(args.student_max_tokens),
            "--student_json_retry_on_parse_fail", str(int(args.student_json_retry_on_parse_fail)),
            "--student_json_max_retries", str(args.student_json_max_retries),
            "--student_json_repair_enabled", str(int(args.student_json_repair_enabled)),
            "--student_json_repair_max_tokens", str(args.student_json_repair_max_tokens),
            "--student_json_repair_temperature", str(args.student_json_repair_temperature),
            "--student_candidate_schema_mode", args.student_candidate_schema_mode,
            "--student_candidate_max_chars_per_field", str(args.student_candidate_max_chars_per_field),
            "--student_candidate_prompt_max_chars", str(args.student_candidate_prompt_max_chars),
            "--student_force_minified_json", str(int(args.student_force_minified_json)),
            "--teacher_critic_use_voting_failure", str(int(teacher_voting_failure)),
            "--optimizer_fallback_mode", str(optimizer_fallback_mode),
            "--no_effective_evolution_patience", str(args.no_effective_evolution_patience),
            "--no_effective_evolution_min_optimizer_candidates", str(args.no_effective_evolution_min_optimizer_candidates),
            "--no_effective_evolution_stop_enabled", str(args.no_effective_evolution_stop_enabled),
            "--specialization_ema", str(getattr(args, "specialization_ema", defaults.specialization_ema)),
            "--behavior_cycle_guard_enabled", str(int(getattr(args, "behavior_cycle_guard_enabled", defaults.behavior_cycle_guard_enabled))),
            "--behavior_archive_size", str(getattr(args, "behavior_archive_size", defaults.behavior_archive_size)),
            "--behavior_cycle_similarity_threshold", str(getattr(args, "behavior_cycle_similarity_threshold", defaults.behavior_cycle_similarity_threshold)),
            "--behavior_cycle_min_overlap", str(getattr(args, "behavior_cycle_min_overlap", defaults.behavior_cycle_min_overlap)),
            "--behavior_cycle_improvement_epsilon", str(getattr(args, "behavior_cycle_improvement_epsilon", defaults.behavior_cycle_improvement_epsilon)),
            "--behavior_cycle_margin_epsilon", str(getattr(args, "behavior_cycle_margin_epsilon", defaults.behavior_cycle_margin_epsilon)),
            "--prompt_trust_region_enabled", str(int(getattr(args, "prompt_trust_region_enabled", defaults.prompt_trust_region_enabled))),
            "--prompt_max_change_ratio", str(getattr(args, "prompt_max_change_ratio", defaults.prompt_max_change_ratio)),
            "--prompt_large_shift_warmup_accepts", str(getattr(args, "prompt_large_shift_warmup_accepts", defaults.prompt_large_shift_warmup_accepts)),
            "--prompt_large_shift_min_vote_delta", str(getattr(args, "prompt_large_shift_min_vote_delta", defaults.prompt_large_shift_min_vote_delta)),
            "--baseline_allowed_vote_loss", str(getattr(args, "baseline_allowed_vote_loss", defaults.baseline_allowed_vote_loss)),
            "--candidate_eval_strategy", str(candidate_eval_strategy),
            "--candidate_eval_concurrency", str(args.candidate_eval_concurrency),
            "--candidate_eval_pool_size", str(candidate_eval_pool_size),
            "--candidate_eval_repeats", str(args.candidate_eval_repeats),
            "--candidate_eval_seed_offset", str(args.candidate_eval_seed_offset),
            "--candidate_reuse_recorded_rollouts", str(args.candidate_reuse_recorded_rollouts),
            "--candidate_eval_execution_mode", str(candidate_eval_execution_mode),
            "--solver_rollout_singleflight", str(int(getattr(args, "solver_rollout_singleflight", Config().solver_rollout_singleflight) if getattr(setting, "solver_rollout_singleflight", None) is None else setting.solver_rollout_singleflight)),
            "--candidate_eval_prompt_dedup", str(int(getattr(args, "candidate_eval_prompt_dedup", Config().candidate_eval_prompt_dedup) if getattr(setting, "candidate_eval_prompt_dedup", None) is None else setting.candidate_eval_prompt_dedup)),
            "--candidate_eval_cache_logging", str(int(getattr(args, "candidate_eval_cache_logging", Config().candidate_eval_cache_logging) if getattr(setting, "candidate_eval_cache_logging", None) is None else setting.candidate_eval_cache_logging)),
            "--resume_from_checkpoint", str(int(args.resume_from_checkpoint)),
            "--train_rollout_concurrency", str(args.train_rollout_concurrency),
            "--eval_solver_call_concurrency", str(args.eval_solver_call_concurrency),
            "--max_tokens", str(args.max_tokens),
            "--optimizer_max_tokens", str(args.optimizer_max_tokens),
            "--evaluator_max_tokens", str(args.evaluator_max_tokens),
            "--temperature", str(args.temperature),
            "--optimizer_temperature", str(args.optimizer_temperature),
            "--evaluator_temperature", str(args.evaluator_temperature),
            "--max_retries", str(args.max_retries),
            "--retry_sleep", str(args.retry_sleep),
            "--transient_retry_forever", str(args.transient_retry_forever),
            "--max_transient_retries", str(args.max_transient_retries),
            "--max_retry_backoff", str(args.max_retry_backoff),
            "--llm_call_logging", str(args.llm_call_logging),
            "--llm_call_timeout", str(args.llm_call_timeout),
            "--vote_tie_break", args.vote_tie_break,
            "--aggregation_mode", args.aggregation_mode,
            "--test_size", str(args.test_size),
            "--eval_test_each_epoch", str(args.eval_test_each_epoch),
            "--early_stopping_patience", str(args.early_stopping_patience),
            "--early_stopping_min_delta", str(args.early_stopping_min_delta),
            "--seed", str(seed),
        ]
    )
    for name in (
        "boundary_selector_enabled", "shared_error_metrics_enabled", "residual_specialization_enabled",
        "error_dependence_guard_enabled", "residual_cycle_guard_enabled", "mechanism_trust_region_enabled",
        "capability_affinity_weight", "capability_coverage_gap_weight", "specialization_support_shrinkage",
        "capability_loss_weight", "specialization_update_period", "pivotal_loss_guard_epsilon",
        "shared_error_creation_epsilon",
    ):
        value = _setting_value(setting, name, getattr(args, name, getattr(defaults, name)))
        if isinstance(getattr(defaults, name), bool):
            value = int(bool(value))
        cmd.extend([f"--{name}", str(value)])


def run_one(
    task: ComparisonTask,
    setting: ExperimentSetting,
    seed: int,
    args: argparse.Namespace,
    split_integrity: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    run_dir = Path(args.out_root) / task.task_id / f"{setting.name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, "-m", "multi_dataset_diverse_rl.cli"]
    _append_common_cli_args(cmd, args, task, setting, seed, split_integrity)
    cmd.extend(
        [
            "--test_path", task.test_path,
            "--out_dir", str(run_dir),
            "--baseline_only", "1" if setting.baseline_only else "0",
        ]
    )
    if not setting.baseline_only:
        cmd.extend(
            [
                "--train_path", task.train_path,
                "--val_path", task.val_path,
                "--train_size", str(args.train_size),
                "--val_size", str(args.val_size),
                "--val_split_ratio", str(args.val_split_ratio),
                "--epochs", str(args.epochs),
                "--update_every", str(args.update_every),
                "--candidate_eval_batch_size", str(_explicit_cli_or_setting(args, setting, "candidate_eval_batch_size", Config().candidate_eval_batch_size)),
            ]
        )
    start = time.time()
    print(f"\n[RUN] task={task.task_id} setting={setting.name} seed={seed}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=args.workspace)
    elapsed = time.time() - start
    row = {
        "task_id": task.task_id,
        "benchmark": task.benchmark,
        "setting": setting.name,
        "seed": seed,
        "reward_mode": _setting_reward_mode(args, setting),
        "init_mode": setting.init_mode,
        "baseline_only": int(setting.baseline_only),
        "answer_format": task.answer_format,
        "task_type": task.task_type,
        "dataset_format": args.dataset_format,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "run_dir": str(run_dir),
    }
    return row


def _latest_test_vote_acc(run_dir: Path) -> float:
    history = read_json(run_dir / "history.json") or []
    if not isinstance(history, list):
        return 0.0
    for record in reversed(history):
        if isinstance(record, dict) and isinstance(record.get("test"), dict):
            return float(record["test"].get("vote_acc", 0.0) or 0.0)
    return 0.0


def _latest_test_metrics(run_dir: Path) -> Dict[str, Any]:
    history = read_json(run_dir / "history.json") or []
    if not isinstance(history, list):
        return {}
    for record in reversed(history):
        if isinstance(record, dict) and isinstance(record.get("test"), dict):
            return record["test"]
    return {}


def is_completed_run_dir(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    if not (run_dir / "history.json").exists():
        return False
    if not (run_dir / "cost_summary.json").exists():
        return False
    if not (run_dir / "run_meta.json").exists():
        return False
    test = _latest_test_metrics(run_dir)
    if not isinstance(test, dict) or not test:
        return False
    return "vote_acc" in test or "num_test_samples" in test or "size" in test


def _completed_run_row(task: ComparisonTask, setting: ExperimentSetting, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.out_root) / task.task_id / f"{setting.name}_seed{seed}"
    return {
        "task_id": task.task_id,
        "benchmark": task.benchmark,
        "setting": setting.name,
        "seed": seed,
        "reward_mode": _setting_reward_mode(args, setting),
        "init_mode": setting.init_mode,
        "baseline_only": int(setting.baseline_only),
        "answer_format": task.answer_format,
        "task_type": task.task_type,
        "dataset_format": args.dataset_format,
        "status": "reused_completed",
        "returncode": 0,
        "elapsed_sec": 0.0,
        "run_dir": str(run_dir),
        "resume_completed": 1,
    }


def run_precheck(
    task: ComparisonTask,
    seed: int,
    args: argparse.Namespace,
    split_integrity: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    precheck_setting = ExperimentSetting("precheck_baseline", "shared", True, "guarded_diversity")
    run_dir = Path(args.out_root) / task.task_id / f"precheck_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, "-m", "multi_dataset_diverse_rl.cli"]
    _append_common_cli_args(cmd, args, task, precheck_setting, seed, split_integrity)
    cmd.extend(
        [
            "--test_path", task.test_path,
            "--out_dir", str(run_dir),
            "--baseline_only", "1",
            "--test_size", str(max(1, int(args.precheck_steps))),
        ]
    )
    start = time.time()
    print(f"\n[PRECHECK] task={task.task_id} seed={seed}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=args.workspace)
    elapsed = time.time() - start
    vote_acc = _latest_test_vote_acc(run_dir) if proc.returncode == 0 else 0.0
    threshold = float(args.precheck_acc_threshold)
    return {
        "task_id": task.task_id,
        "benchmark": task.benchmark,
        "setting": precheck_setting.name,
        "seed": seed,
        "reward_mode": precheck_setting.reward_mode,
        "init_mode": precheck_setting.init_mode,
        "baseline_only": 1,
        "answer_format": task.answer_format,
        "task_type": task.task_type,
        "dataset_format": args.dataset_format,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "run_dir": str(run_dir),
        "precheck": 1,
        "precheck_steps": int(args.precheck_steps),
        "precheck_vote_acc": float(vote_acc),
        "precheck_acc_threshold": threshold,
        "skip_task": bool(vote_acc > threshold),
    }


def _skip_row(task: ComparisonTask, setting: ExperimentSetting, seed: int, args: argparse.Namespace, precheck_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "benchmark": task.benchmark,
        "setting": setting.name,
        "seed": seed,
        "reward_mode": _setting_reward_mode(args, setting),
        "init_mode": setting.init_mode,
        "baseline_only": int(setting.baseline_only),
        "answer_format": task.answer_format,
        "task_type": task.task_type,
        "dataset_format": args.dataset_format,
        "status": "skipped_high_baseline_acc",
        "returncode": 0,
        "elapsed_sec": 0.0,
        "run_dir": "",
        "precheck": 0,
        "precheck_steps": int(args.precheck_steps),
        "precheck_vote_acc": float(precheck_row.get("precheck_vote_acc", 0.0) or 0.0),
        "precheck_acc_threshold": float(args.precheck_acc_threshold),
        "skip_reason": f"precheck_vote_acc>{float(args.precheck_acc_threshold):.4f}",
    }


def _task_split_protocol(task: ComparisonTask) -> Dict[str, Any]:
    paths = {str(task.train_path), str(task.val_path), str(task.test_path)}
    if len(paths) < 3:
        return {"split_protocol": "paper_compatible_reused_file", "leakage_warning": True}
    return {"split_protocol": "task_manifest_split", "leakage_warning": False}


def _normalized_question_hash(question: Any) -> str:
    normalized = " ".join(str(question or "").split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _resolved_path(path: str, workspace: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(workspace) / value


def _split_rows(path: Path, dataset_format: str) -> List[Dict[str, Any]]:
    return build_dataset(load_jsonl(str(path), -1), dataset_format)


def _task_split_integrity(task: ComparisonTask, dataset_format: str, workspace: str) -> Dict[str, Any]:
    split_paths = {
        "opt": _resolved_path(task.train_path, workspace),
        "val": _resolved_path(task.val_path, workspace),
        "test": _resolved_path(task.test_path, workspace),
    }
    split_rows = {name: _split_rows(path, dataset_format) for name, path in split_paths.items()}
    split_hashes = {
        name: {_normalized_question_hash(row.get("question", "")) for row in rows}
        for name, rows in split_rows.items()
    }
    overlaps = {
        "opt_val_question_overlap": len(split_hashes["opt"] & split_hashes["val"]),
        "opt_test_question_overlap": len(split_hashes["opt"] & split_hashes["test"]),
        "val_test_question_overlap": len(split_hashes["val"] & split_hashes["test"]),
    }
    protocol = _task_split_protocol(task)
    integrity = {
        **protocol,
        "opt_count": len(split_rows["opt"]),
        "val_count": len(split_rows["val"]),
        "test_count": len(split_rows["test"]),
        **overlaps,
        "opt_file_sha256": hashlib.sha256(split_paths["opt"].read_bytes()).hexdigest(),
        "val_file_sha256": hashlib.sha256(split_paths["val"].read_bytes()).hexdigest(),
        "test_file_sha256": hashlib.sha256(split_paths["test"].read_bytes()).hexdigest(),
    }
    if protocol["split_protocol"] == "task_manifest_split" and any(overlaps.values()):
        raise ValueError(
            f"Strict split overlap for task={task.task_id}: "
            f"opt_val={overlaps['opt_val_question_overlap']} "
            f"opt_test={overlaps['opt_test_question_overlap']} "
            f"val_test={overlaps['val_test_question_overlap']}"
        )
    return integrity


def _mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def write_accuracy_summary(rows: List[Dict[str, Any]], out_root: Path):
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row.get("task_id", ""), row.get("benchmark", ""), row.get("setting", "")), []).append(row)
    summary_rows = []
    for (task_id, benchmark, setting), group in sorted(groups.items()):
        out: Dict[str, Any] = {"task_id": task_id, "benchmark": benchmark, "setting": setting, "n": len(group)}
        for metric in [
            "vote_acc",
            "majority_vote_acc",
            "weighted_vote_acc",
            "mean_individual_acc",
            "best_individual_acc",
            "oracle_acc",
            "aggregation_gap",
            "correct_disagreement_rate",
            "mean_useful_diversity",
            "mean_vote_margin",
            "mean_boundary_useful_diversity",
            "mean_pairwise_double_fault",
            "mean_pairwise_error_covariance",
            "same_wrong_pair_rate",
            "triple_joint_error_rate",
            "majority_failure_tail_rate",
            "mean_boundary_conditional_error",
            "mean_pivotal_fix_rate",
            "mean_pivotal_hold_rate",
            "shared_error_rescue_rate",
            "shared_error_creation_rate",
            "boundary_shared_error_net_gain",
            "dominant_wrong_cluster_size",
            "gold_vs_largest_wrong_margin",
            "total_llm_calls",
            "total_tokens",
            "estimated_cost",
        ]:
            values = [float(row.get(metric, 0.0) or 0.0) for row in group]
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_std"] = _std(values)
        summary_rows.append(out)
    write_csv(out_root / "accuracy_summary.csv", summary_rows, empty_fieldnames=["task_id", "benchmark", "setting", "n"])
    lines = ["# Task-Level Accuracy Summary", ""]
    if not summary_rows:
        lines.append("No completed runs.")
    else:
        columns = [
            "task_id",
            "benchmark",
            "setting",
            "n",
            "vote_acc_mean",
            "mean_individual_acc_mean",
            "best_individual_acc_mean",
            "oracle_acc_mean",
            "aggregation_gap_mean",
            "mean_useful_diversity_mean",
            "mean_vote_margin_mean",
            "mean_boundary_useful_diversity_mean",
        ]
        lines.extend(["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"])
        for row in summary_rows:
            lines.append("| " + " | ".join(str(round(row.get(c, 0.0), 6)) if isinstance(row.get(c), float) else str(row.get(c, "")) for c in columns) + " |")
    (out_root / "accuracy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run MAD at task_id granularity and export standardized accuracy results.")
    cli_defaults = Config()
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--manifest", type=str, default="configs/task_level_comparison.yaml")
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--benchmarks", type=str, default="")
    parser.add_argument("--settings", type=str, default="shared_baseline,bank_baseline,shared_guarded_beam,bank_guarded_beam")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--dataset_format", type=str, default="mars", choices=["legacy", "mars"])
    parser.add_argument("--out_root", type=str, default="runs_task_level_accuracy")
    parser.add_argument("--run_concurrency", type=int, default=1)
    parser.add_argument("--warmup_serial_runs", type=int, default=1)
    parser.add_argument("--run_start_stagger_seconds", type=float, default=5.0)
    parser.add_argument("--resume_completed", type=int, default=0, choices=[0, 1])
    parser.add_argument("--resume_from_checkpoint", type=int, default=int(cli_defaults.resume_from_checkpoint), choices=[0, 1])
    parser.add_argument("--skip_high_baseline_acc", type=int, default=0, choices=[0, 1])
    parser.add_argument("--precheck_steps", type=int, default=20)
    parser.add_argument("--precheck_acc_threshold", type=float, default=0.95)

    parser.add_argument("--agent_model", type=str, default=cli_defaults.agent_model)
    parser.add_argument("--optimizer_model", type=str, default=cli_defaults.optimizer_model)
    parser.add_argument("--evaluator_model", type=str, default=cli_defaults.evaluator_model)
    parser.add_argument("--reward_mode", type=str, default="", choices=["", "accuracy_only", "guarded_diversity", "vote_useful_diversity"])
    parser.add_argument("--candidate_selection_mode", type=str, default=cli_defaults.candidate_selection_mode, choices=["scalar_reward", "vote_pareto", "vote_error_pareto"])
    parser.add_argument("--best_state_selection_mode", type=str, default=cli_defaults.best_state_selection_mode, choices=["existing", "vote_first"])
    parser.add_argument("--agents", type=int, default=cli_defaults.agents)
    parser.add_argument("--train_size", type=int, default=cli_defaults.train_size)
    parser.add_argument("--val_size", type=int, default=cli_defaults.val_size)
    parser.add_argument("--val_split_ratio", type=float, default=cli_defaults.val_split_ratio)
    parser.add_argument("--test_size", type=int, default=cli_defaults.test_size)
    parser.add_argument("--epochs", type=int, default=cli_defaults.epochs)
    parser.add_argument("--update_every", type=int, default=cli_defaults.update_every)
    parser.add_argument("--eval_test_each_epoch", type=int, default=int(cli_defaults.eval_test_each_epoch), choices=[0, 1])
    parser.add_argument("--early_stopping_patience", type=int, default=cli_defaults.early_stopping_patience)
    parser.add_argument("--early_stopping_min_delta", type=float, default=cli_defaults.early_stopping_min_delta)
    parser.add_argument("--shared_prompt", type=str, default=cli_defaults.shared_prompt)
    parser.add_argument("--beam_size", type=int, default=cli_defaults.beam_size)
    parser.add_argument("--num_candidates_per_parent", type=int, default=cli_defaults.num_candidates_per_parent)
    parser.add_argument("--optimizer_parent_concurrency", type=int, default=cli_defaults.optimizer_parent_concurrency)
    parser.add_argument("--beam_refresh_each_epoch", type=int, default=int(cli_defaults.beam_refresh_each_epoch), choices=[0, 1])
    parser.add_argument("--accuracy_guard_epsilon", type=float, default=cli_defaults.accuracy_guard_epsilon)
    parser.add_argument("--reward_weight_div_delta", type=float, default=cli_defaults.reward_weight_div_delta)
    parser.add_argument("--reward_weight_invalid_delta", type=float, default=cli_defaults.reward_weight_invalid_delta)
    parser.add_argument("--reward_weight_vote_delta", type=float, default=cli_defaults.reward_weight_vote_delta)
    parser.add_argument("--reward_weight_vote_margin", type=float, default=cli_defaults.reward_weight_vote_margin)
    parser.add_argument("--reward_weight_boundary_diversity", type=float, default=cli_defaults.reward_weight_boundary_diversity)
    parser.add_argument("--invalid_guard_epsilon", type=float, default=cli_defaults.invalid_guard_epsilon)
    parser.add_argument("--use_baseline_relative_reward", type=int, default=int(cli_defaults.use_baseline_relative_reward), choices=[0, 1])
    parser.add_argument("--reward_schedule_mode", type=str, default=cli_defaults.reward_schedule_mode, choices=["static", "phase_adaptive"])
    parser.add_argument("--reward_diversity_warmup_updates", type=int, default=cli_defaults.reward_diversity_warmup_updates)
    parser.add_argument("--reward_weight_div_delta_early", type=float, default=cli_defaults.reward_weight_div_delta_early)
    parser.add_argument("--reward_weight_div_delta_late", type=float, default=cli_defaults.reward_weight_div_delta_late)
    parser.add_argument("--reward_weight_vote_delta_early", type=float, default=cli_defaults.reward_weight_vote_delta_early)
    parser.add_argument("--reward_weight_vote_delta_late", type=float, default=cli_defaults.reward_weight_vote_delta_late)
    parser.add_argument("--reward_weight_vote_margin_early", type=float, default=cli_defaults.reward_weight_vote_margin_early)
    parser.add_argument("--reward_weight_vote_margin_late", type=float, default=cli_defaults.reward_weight_vote_margin_late)
    parser.add_argument("--reward_weight_boundary_diversity_early", type=float, default=cli_defaults.reward_weight_boundary_diversity_early)
    parser.add_argument("--reward_weight_boundary_diversity_late", type=float, default=cli_defaults.reward_weight_boundary_diversity_late)
    parser.add_argument("--reward_weight_target_accuracy_early", type=float, default=cli_defaults.reward_weight_target_accuracy_early)
    parser.add_argument("--reward_weight_target_accuracy_late", type=float, default=cli_defaults.reward_weight_target_accuracy_late)
    parser.add_argument("--accuracy_guard_epsilon_early", type=float, default=cli_defaults.accuracy_guard_epsilon_early)
    parser.add_argument("--accuracy_guard_epsilon_late", type=float, default=cli_defaults.accuracy_guard_epsilon_late)
    parser.add_argument("--optimizer_architecture", type=str, default=cli_defaults.optimizer_architecture, choices=["one_shot", "teacher_critic_student"])
    parser.add_argument("--teacher_critic_max_rounds", type=int, default=cli_defaults.teacher_critic_max_rounds)
    parser.add_argument("--teacher_question_pass_threshold", type=float, default=cli_defaults.teacher_question_pass_threshold)
    parser.add_argument("--teacher_temperature", type=float, default=cli_defaults.teacher_temperature)
    parser.add_argument("--critic_temperature", type=float, default=cli_defaults.critic_temperature)
    parser.add_argument("--student_temperature", type=float, default=cli_defaults.student_temperature)
    parser.add_argument("--teacher_max_tokens", type=int, default=cli_defaults.teacher_max_tokens)
    parser.add_argument("--critic_max_tokens", type=int, default=cli_defaults.critic_max_tokens)
    parser.add_argument("--student_max_tokens", type=int, default=cli_defaults.student_max_tokens)
    parser.add_argument("--student_json_retry_on_parse_fail", type=int, default=int(cli_defaults.student_json_retry_on_parse_fail), choices=[0, 1])
    parser.add_argument("--student_json_max_retries", type=int, default=cli_defaults.student_json_max_retries)
    parser.add_argument("--student_json_repair_enabled", type=int, default=int(cli_defaults.student_json_repair_enabled), choices=[0, 1])
    parser.add_argument("--student_json_repair_max_tokens", type=int, default=cli_defaults.student_json_repair_max_tokens)
    parser.add_argument("--student_json_repair_temperature", type=float, default=cli_defaults.student_json_repair_temperature)
    parser.add_argument("--student_candidate_schema_mode", type=str, default=cli_defaults.student_candidate_schema_mode, choices=["compact", "verbose"])
    parser.add_argument("--student_candidate_max_chars_per_field", type=int, default=cli_defaults.student_candidate_max_chars_per_field)
    parser.add_argument("--student_candidate_prompt_max_chars", type=int, default=cli_defaults.student_candidate_prompt_max_chars)
    parser.add_argument("--student_force_minified_json", type=int, default=int(cli_defaults.student_force_minified_json), choices=[0, 1])
    parser.add_argument("--teacher_critic_use_voting_failure", type=int, default=int(cli_defaults.teacher_critic_use_voting_failure), choices=[0, 1])
    parser.add_argument("--optimizer_fallback_mode", type=str, default=cli_defaults.optimizer_fallback_mode, choices=["none", "template"])
    parser.add_argument("--no_effective_evolution_patience", type=int, default=cli_defaults.no_effective_evolution_patience)
    parser.add_argument("--no_effective_evolution_min_optimizer_candidates", type=int, default=cli_defaults.no_effective_evolution_min_optimizer_candidates)
    parser.add_argument("--no_effective_evolution_stop_enabled", type=int, default=int(cli_defaults.no_effective_evolution_stop_enabled), choices=[0, 1])
    parser.add_argument("--boundary_selector_enabled", type=int, default=int(cli_defaults.boundary_selector_enabled), choices=[0, 1])
    parser.add_argument("--shared_error_metrics_enabled", type=int, default=int(cli_defaults.shared_error_metrics_enabled), choices=[0, 1])
    parser.add_argument("--residual_specialization_enabled", type=int, default=int(cli_defaults.residual_specialization_enabled), choices=[0, 1])
    parser.add_argument("--error_dependence_guard_enabled", type=int, default=int(cli_defaults.error_dependence_guard_enabled), choices=[0, 1])
    parser.add_argument("--residual_cycle_guard_enabled", type=int, default=int(cli_defaults.residual_cycle_guard_enabled), choices=[0, 1])
    parser.add_argument("--mechanism_trust_region_enabled", type=int, default=int(cli_defaults.mechanism_trust_region_enabled), choices=[0, 1])
    parser.add_argument("--specialization_support_shrinkage", type=float, default=cli_defaults.specialization_support_shrinkage)
    parser.add_argument("--capability_loss_weight", type=float, default=cli_defaults.capability_loss_weight)
    parser.add_argument("--specialization_update_period", type=int, default=cli_defaults.specialization_update_period)
    parser.add_argument("--capability_affinity_weight", type=float, default=cli_defaults.capability_affinity_weight)
    parser.add_argument("--capability_coverage_gap_weight", type=float, default=cli_defaults.capability_coverage_gap_weight)
    parser.add_argument("--pivotal_loss_guard_epsilon", type=float, default=cli_defaults.pivotal_loss_guard_epsilon)
    parser.add_argument("--shared_error_creation_epsilon", type=float, default=cli_defaults.shared_error_creation_epsilon)
    parser.add_argument("--specialization_ema", type=float, default=cli_defaults.specialization_ema)
    parser.add_argument("--behavior_cycle_guard_enabled", type=int, default=int(cli_defaults.behavior_cycle_guard_enabled), choices=[0, 1])
    parser.add_argument("--behavior_archive_size", type=int, default=cli_defaults.behavior_archive_size)
    parser.add_argument("--behavior_cycle_similarity_threshold", type=float, default=cli_defaults.behavior_cycle_similarity_threshold)
    parser.add_argument("--behavior_cycle_min_overlap", type=int, default=cli_defaults.behavior_cycle_min_overlap)
    parser.add_argument("--behavior_cycle_improvement_epsilon", type=float, default=cli_defaults.behavior_cycle_improvement_epsilon)
    parser.add_argument("--behavior_cycle_margin_epsilon", type=float, default=cli_defaults.behavior_cycle_margin_epsilon)
    parser.add_argument("--prompt_trust_region_enabled", type=int, default=int(cli_defaults.prompt_trust_region_enabled), choices=[0, 1])
    parser.add_argument("--prompt_max_change_ratio", type=float, default=cli_defaults.prompt_max_change_ratio)
    parser.add_argument("--prompt_large_shift_warmup_accepts", type=int, default=cli_defaults.prompt_large_shift_warmup_accepts)
    parser.add_argument("--prompt_large_shift_min_vote_delta", type=float, default=cli_defaults.prompt_large_shift_min_vote_delta)
    parser.add_argument("--baseline_allowed_vote_loss", type=float, default=cli_defaults.baseline_allowed_vote_loss)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=None)
    parser.add_argument("--candidate_eval_concurrency", type=int, default=cli_defaults.candidate_eval_concurrency)
    parser.add_argument("--candidate_eval_strategy", type=str, default=None, choices=["random", "fixed_pool", "stratified"])
    parser.add_argument("--candidate_eval_pool_size", type=int, default=None)
    parser.add_argument("--candidate_eval_repeats", type=int, default=cli_defaults.candidate_eval_repeats)
    parser.add_argument("--candidate_eval_seed_offset", type=int, default=cli_defaults.candidate_eval_seed_offset)
    parser.add_argument("--candidate_reuse_recorded_rollouts", type=int, default=int(cli_defaults.candidate_reuse_recorded_rollouts), choices=[0, 1])
    parser.add_argument("--candidate_eval_execution_mode", type=str, default=None, choices=["legacy", "factorized_cached"])
    parser.add_argument("--solver_rollout_singleflight", type=int, default=int(cli_defaults.solver_rollout_singleflight), choices=[0, 1])
    parser.add_argument("--candidate_eval_prompt_dedup", type=int, default=int(cli_defaults.candidate_eval_prompt_dedup), choices=[0, 1])
    parser.add_argument("--candidate_eval_cache_logging", type=int, default=int(cli_defaults.candidate_eval_cache_logging), choices=[0, 1])
    parser.add_argument("--train_rollout_concurrency", type=int, default=cli_defaults.train_rollout_concurrency)
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=cli_defaults.eval_solver_call_concurrency)
    parser.add_argument("--max_tokens", type=int, default=cli_defaults.max_tokens)
    parser.add_argument("--optimizer_max_tokens", type=int, default=cli_defaults.optimizer_max_tokens)
    parser.add_argument("--evaluator_max_tokens", type=int, default=cli_defaults.evaluator_max_tokens)
    parser.add_argument("--temperature", type=float, default=cli_defaults.temperature)
    parser.add_argument("--optimizer_temperature", type=float, default=cli_defaults.optimizer_temperature)
    parser.add_argument("--evaluator_temperature", type=float, default=cli_defaults.evaluator_temperature)
    parser.add_argument("--max_retries", type=int, default=cli_defaults.max_retries)
    parser.add_argument("--retry_sleep", type=float, default=cli_defaults.retry_sleep)
    parser.add_argument("--transient_retry_forever", type=int, default=int(cli_defaults.transient_retry_forever), choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=cli_defaults.max_transient_retries)
    parser.add_argument("--max_retry_backoff", type=float, default=cli_defaults.max_retry_backoff)
    parser.add_argument("--llm_call_logging", type=int, default=int(cli_defaults.llm_call_logging), choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=cli_defaults.llm_call_timeout)
    parser.add_argument("--vote_tie_break", type=str, default=cli_defaults.vote_tie_break, choices=["first", "random", "abstain"])
    parser.add_argument("--aggregation_mode", type=str, default=cli_defaults.aggregation_mode, choices=["majority", "weighted_vote", "verifier_select"])
    args = parser.parse_args()

    args.workspace = str(Path(args.workspace).resolve())
    args.out_root = str((Path(args.workspace) / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root).resolve())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = load_task_manifest(args.manifest)
    task_ids = resolve_task_ids(args.tasks, tasks, benchmarks=args.benchmarks)
    split_integrities = {
        task_id: _task_split_integrity(tasks[task_id], args.dataset_format, args.workspace)
        for task_id in task_ids
    }
    settings = _selected_settings(args.settings)
    seeds = [int(x) for x in parse_csv_list(args.seeds)] or [42]
    resume_completed = bool(int(getattr(args, "resume_completed", 0) or 0))

    runs_jsonl = out_root / "experiment_runs.jsonl"
    runs_csv = out_root / "experiment_runs.csv"
    accuracy_jsonl = out_root / "accuracy_results.jsonl"
    accuracy_csv = out_root / "accuracy_results.csv"
    if resume_completed:
        print(f"[RESUME] Rebuilding summaries and reusing completed run dirs under {out_root}", flush=True)
    runs_jsonl.write_text("", encoding="utf-8")
    accuracy_jsonl.write_text("", encoding="utf-8")

    run_rows: List[Dict[str, Any]] = []
    accuracy_rows: List[Dict[str, Any]] = []
    pending_runs: List[Tuple[ComparisonTask, ExperimentSetting, int]] = []

    def record_run_row(row: Dict[str, Any]):
        run_rows.append(row)
        append_jsonl(runs_jsonl, row)
        write_csv(runs_csv, run_rows, empty_fieldnames=["task_id", "setting", "status"])

    def record_accuracy_row(task: ComparisonTask, setting: ExperimentSetting, seed: int, run_row: Dict[str, Any]):
        accuracy_row = build_accuracy_result_row(
            run_dir=Path(run_row["run_dir"]),
            task_id=task.task_id,
            benchmark=task.benchmark,
            setting=setting.name,
            seed=seed,
            dataset_format=args.dataset_format,
            **_task_split_protocol(task),
        )
        accuracy_rows.append(accuracy_row)
        append_jsonl(accuracy_jsonl, accuracy_row)
        write_csv(accuracy_csv, accuracy_rows, fieldnames=ACCURACY_RESULT_COLUMNS)
        write_accuracy_summary(accuracy_rows, out_root)

    for task_id in task_ids:
        task = tasks[task_id]
        skip_task_for_seed: Dict[int, Dict[str, Any]] = {}
        if int(args.skip_high_baseline_acc):
            for seed in seeds:
                precheck_row = run_precheck(task, seed, args, split_integrities[task.task_id])
                record_run_row(precheck_row)
                if precheck_row["status"] != "ok":
                    raise SystemExit(precheck_row["returncode"])
                if precheck_row.get("skip_task"):
                    skip_task_for_seed[seed] = precheck_row
                    print(
                        f"[SKIP] task={task.task_id} seed={seed} "
                        f"precheck_vote_acc={float(precheck_row.get('precheck_vote_acc', 0.0)):.4f} "
                        f"> threshold={float(args.precheck_acc_threshold):.4f}",
                        flush=True,
                    )
        for setting in settings:
            for seed in seeds:
                if seed in skip_task_for_seed:
                    run_row = _skip_row(task, setting, seed, args, skip_task_for_seed[seed])
                    record_run_row(run_row)
                    continue
                run_dir = Path(args.out_root) / task.task_id / f"{setting.name}_seed{seed}"
                if resume_completed and is_completed_run_dir(run_dir):
                    run_row = _completed_run_row(task, setting, seed, args)
                    print(
                        f"[RESUME] Reusing completed run task={task.task_id} setting={setting.name} seed={seed}",
                        flush=True,
                    )
                    record_run_row(run_row)
                    record_accuracy_row(task, setting, seed, run_row)
                    continue
                pending_runs.append((task, setting, seed))

    failed_rows: List[Dict[str, Any]] = []

    def run_and_record(task: ComparisonTask, setting: ExperimentSetting, seed: int):
        run_row = run_one(task, setting, seed, args, split_integrities[task.task_id])
        record_run_row(run_row)
        if run_row["status"] != "ok":
            raise SystemExit(run_row["returncode"])
        record_accuracy_row(task, setting, seed, run_row)

    run_concurrency = max(1, int(getattr(args, "run_concurrency", 1) or 1))
    warmup_count = 0
    if run_concurrency > 1:
        warmup_count = min(max(0, int(getattr(args, "warmup_serial_runs", 0) or 0)), len(pending_runs))
    warmup_runs = pending_runs[:warmup_count]
    remaining_runs = pending_runs[warmup_count:]

    if warmup_runs:
        print(f"\n[WARMUP] Running {len(warmup_runs)} job(s) serially before enabling concurrency.", flush=True)
        for task, setting, seed in warmup_runs:
            run_and_record(task, setting, seed)

    if run_concurrency <= 1:
        for task, setting, seed in remaining_runs:
            run_and_record(task, setting, seed)
    elif remaining_runs:
        print(f"\n[CONCURRENCY] Running up to {run_concurrency} task/setting/seed jobs in parallel.", flush=True)

        def run_one_staggered(index: int, task: ComparisonTask, setting: ExperimentSetting, seed: int) -> Dict[str, Any]:
            stagger = max(0.0, float(getattr(args, "run_start_stagger_seconds", 0.0) or 0.0))
            delay = (index % run_concurrency) * stagger
            if delay > 0:
                print(
                    f"[STAGGER] task={task.task_id} setting={setting.name} seed={seed} sleep={delay:.1f}s before launch",
                    flush=True,
                )
                time.sleep(delay)
            return run_one(task, setting, seed, args, split_integrities[task.task_id])

        with ThreadPoolExecutor(max_workers=run_concurrency) as executor:
            futures = {
                executor.submit(run_one_staggered, idx, task, setting, seed): (task, setting, seed)
                for idx, (task, setting, seed) in enumerate(remaining_runs)
            }
            for future in as_completed(futures):
                task, setting, seed = futures[future]
                try:
                    run_row = future.result()
                except Exception as exc:
                    run_row = {
                        "task_id": task.task_id,
                        "benchmark": task.benchmark,
                        "setting": setting.name,
                        "seed": seed,
                        "reward_mode": _setting_reward_mode(args, setting),
                        "init_mode": setting.init_mode,
                        "baseline_only": int(setting.baseline_only),
                        "answer_format": task.answer_format,
                        "task_type": task.task_type,
                        "dataset_format": args.dataset_format,
                        "status": "failed",
                        "returncode": 1,
                        "elapsed_sec": 0.0,
                        "run_dir": str(Path(args.out_root) / task.task_id / f"{setting.name}_seed{seed}"),
                        "error": str(exc),
                    }
                record_run_row(run_row)
                if run_row["status"] != "ok":
                    failed_rows.append(run_row)
                    continue
                record_accuracy_row(task, setting, seed, run_row)

    if failed_rows:
        first = failed_rows[0]
        raise SystemExit(int(first.get("returncode", 1) or 1))

    write_jsonl(out_root / "accuracy_results.jsonl", accuracy_rows)
    write_csv(accuracy_csv, accuracy_rows, fieldnames=ACCURACY_RESULT_COLUMNS)
    write_accuracy_summary(accuracy_rows, out_root)
    print(f"\n[DONE] Wrote {accuracy_jsonl}")
    print(f"[DONE] Wrote {out_root / 'accuracy_summary.csv'}")


if __name__ == "__main__":
    main()
