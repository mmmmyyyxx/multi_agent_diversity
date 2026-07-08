import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from multi_dataset_diverse_rl.config import Config
    from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTINGS, ExperimentSetting, parse_csv_list, select_settings
    from scripts.experiment_io import append_jsonl, read_json, write_csv, write_jsonl
    from scripts.task_level_accuracy_utils import ACCURACY_RESULT_COLUMNS, build_accuracy_result_row
except ModuleNotFoundError:
    from multi_dataset_diverse_rl.config import Config
    from experiment_config import DEFAULT_EXPERIMENT_SETTINGS, ExperimentSetting, parse_csv_list, select_settings
    from experiment_io import append_jsonl, read_json, write_csv, write_jsonl
    from task_level_accuracy_utils import ACCURACY_RESULT_COLUMNS, build_accuracy_result_row

from multi_dataset_diverse_rl.task_manifest import ComparisonTask, load_task_manifest, resolve_task_ids


SETTINGS = DEFAULT_EXPERIMENT_SETTINGS


def _selected_settings(raw: str) -> List[ExperimentSetting]:
    return select_settings(raw, SETTINGS)


def _setting_reward_mode(args: argparse.Namespace, setting: ExperimentSetting) -> str:
    override = str(getattr(args, "reward_mode", "") or "").strip()
    if override and not setting.baseline_only:
        return override
    return setting.reward_mode


def _append_common_cli_args(cmd: List[str], args: argparse.Namespace, task: ComparisonTask, setting: ExperimentSetting, seed: int):
    reward_mode = _setting_reward_mode(args, setting)
    cmd.extend(
        [
            "--task_type", task.task_type,
            "--dataset_format", args.dataset_format,
            "--comparison_task_id", task.task_id,
            "--benchmark", task.benchmark,
            "--answer_format", task.answer_format,
            "--agent_model", args.agent_model,
            "--optimizer_model", args.optimizer_model,
            "--evaluator_model", args.evaluator_model,
            "--search_mode", "evolutionary_beam",
            "--reward_mode", reward_mode,
            "--agents", str(args.agents),
            "--init_mode", setting.init_mode,
            "--shared_prompt", args.shared_prompt,
            "--beam_size", str(args.beam_size),
            "--num_candidates_per_parent", str(args.num_candidates_per_parent),
            "--beam_refresh_each_epoch", str(args.beam_refresh_each_epoch),
            "--accuracy_guard_epsilon", str(args.accuracy_guard_epsilon),
            "--reward_weight_div_delta", str(args.reward_weight_div_delta),
            "--reward_weight_invalid_delta", str(args.reward_weight_invalid_delta),
            "--reward_weight_coverage", str(args.reward_weight_coverage),
            "--reward_weight_useful_diversity", str(args.reward_weight_useful_diversity),
            "--invalid_guard_epsilon", str(args.invalid_guard_epsilon),
            "--use_baseline_relative_reward", str(args.use_baseline_relative_reward),
            "--reward_schedule_mode", args.reward_schedule_mode,
            "--reward_diversity_warmup_updates", str(args.reward_diversity_warmup_updates),
            "--reward_weight_div_delta_early", str(args.reward_weight_div_delta_early),
            "--reward_weight_div_delta_late", str(args.reward_weight_div_delta_late),
            "--reward_weight_coverage_early", str(args.reward_weight_coverage_early),
            "--reward_weight_coverage_late", str(args.reward_weight_coverage_late),
            "--reward_weight_useful_diversity_early", str(args.reward_weight_useful_diversity_early),
            "--reward_weight_useful_diversity_late", str(args.reward_weight_useful_diversity_late),
            "--reward_weight_target_accuracy_early", str(args.reward_weight_target_accuracy_early),
            "--reward_weight_target_accuracy_late", str(args.reward_weight_target_accuracy_late),
            "--accuracy_guard_epsilon_early", str(args.accuracy_guard_epsilon_early),
            "--accuracy_guard_epsilon_late", str(args.accuracy_guard_epsilon_late),
            "--optimizer_architecture", args.optimizer_architecture,
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
            "--teacher_critic_use_voting_failure", str(args.teacher_critic_use_voting_failure),
            "--optimizer_fallback_mode", args.optimizer_fallback_mode,
            "--no_effective_evolution_patience", str(args.no_effective_evolution_patience),
            "--no_effective_evolution_min_optimizer_candidates", str(args.no_effective_evolution_min_optimizer_candidates),
            "--no_effective_evolution_stop_enabled", str(args.no_effective_evolution_stop_enabled),
            "--candidate_eval_strategy", args.candidate_eval_strategy,
            "--candidate_eval_pool_size", str(args.candidate_eval_pool_size),
            "--candidate_eval_repeats", str(args.candidate_eval_repeats),
            "--candidate_eval_seed_offset", str(args.candidate_eval_seed_offset),
            "--candidate_reuse_recorded_rollouts", str(args.candidate_reuse_recorded_rollouts),
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


def run_one(task: ComparisonTask, setting: ExperimentSetting, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.out_root) / task.task_id / f"{setting.name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, "-m", "multi_dataset_diverse_rl.cli"]
    _append_common_cli_args(cmd, args, task, setting, seed)
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
                "--candidate_eval_batch_size", str(args.candidate_eval_batch_size),
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


def run_precheck(task: ComparisonTask, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    precheck_setting = ExperimentSetting("precheck_baseline", "shared", True, "guarded_diversity")
    run_dir = Path(args.out_root) / task.task_id / f"precheck_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, "-m", "multi_dataset_diverse_rl.cli"]
    _append_common_cli_args(cmd, args, task, precheck_setting, seed)
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
    if len(paths) == 1:
        return {"split_protocol": "paper_compatible_reused_file", "leakage_warning": True}
    return {"split_protocol": "task_manifest_split", "leakage_warning": False}


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
            "rescue_available_rate",
            "correct_disagreement_rate",
            "mean_useful_diversity",
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
        columns = ["task_id", "benchmark", "setting", "n", "vote_acc_mean", "oracle_acc_mean", "aggregation_gap_mean", "rescue_available_rate_mean", "mean_individual_acc_mean", "best_individual_acc_mean"]
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
    parser.add_argument("--settings", type=str, default="shared_baseline,shared_guarded_beam,bank_guarded_beam")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--dataset_format", type=str, default="mars", choices=["legacy", "mars"])
    parser.add_argument("--out_root", type=str, default="runs_task_level_accuracy")
    parser.add_argument("--skip_high_baseline_acc", type=int, default=0, choices=[0, 1])
    parser.add_argument("--precheck_steps", type=int, default=20)
    parser.add_argument("--precheck_acc_threshold", type=float, default=0.95)

    parser.add_argument("--agent_model", type=str, default="deepseek-chat")
    parser.add_argument("--optimizer_model", type=str, default="deepseek-chat")
    parser.add_argument("--evaluator_model", type=str, default="deepseek-chat")
    parser.add_argument("--reward_mode", type=str, default="", choices=["", "accuracy_only", "guarded_diversity", "coverage_useful_diversity"])
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
    parser.add_argument("--beam_refresh_each_epoch", type=int, default=int(cli_defaults.beam_refresh_each_epoch), choices=[0, 1])
    parser.add_argument("--accuracy_guard_epsilon", type=float, default=cli_defaults.accuracy_guard_epsilon)
    parser.add_argument("--reward_weight_div_delta", type=float, default=cli_defaults.reward_weight_div_delta)
    parser.add_argument("--reward_weight_invalid_delta", type=float, default=cli_defaults.reward_weight_invalid_delta)
    parser.add_argument("--reward_weight_coverage", type=float, default=cli_defaults.reward_weight_coverage)
    parser.add_argument("--reward_weight_useful_diversity", type=float, default=cli_defaults.reward_weight_useful_diversity)
    parser.add_argument("--invalid_guard_epsilon", type=float, default=cli_defaults.invalid_guard_epsilon)
    parser.add_argument("--use_baseline_relative_reward", type=int, default=int(cli_defaults.use_baseline_relative_reward), choices=[0, 1])
    parser.add_argument("--reward_schedule_mode", type=str, default=cli_defaults.reward_schedule_mode, choices=["static", "phase_adaptive"])
    parser.add_argument("--reward_diversity_warmup_updates", type=int, default=cli_defaults.reward_diversity_warmup_updates)
    parser.add_argument("--reward_weight_div_delta_early", type=float, default=cli_defaults.reward_weight_div_delta_early)
    parser.add_argument("--reward_weight_div_delta_late", type=float, default=cli_defaults.reward_weight_div_delta_late)
    parser.add_argument("--reward_weight_coverage_early", type=float, default=cli_defaults.reward_weight_coverage_early)
    parser.add_argument("--reward_weight_coverage_late", type=float, default=cli_defaults.reward_weight_coverage_late)
    parser.add_argument("--reward_weight_useful_diversity_early", type=float, default=cli_defaults.reward_weight_useful_diversity_early)
    parser.add_argument("--reward_weight_useful_diversity_late", type=float, default=cli_defaults.reward_weight_useful_diversity_late)
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
    parser.add_argument("--candidate_eval_batch_size", type=int, default=cli_defaults.candidate_eval_batch_size)
    parser.add_argument("--candidate_eval_strategy", type=str, default=cli_defaults.candidate_eval_strategy, choices=["random", "fixed_pool", "stratified"])
    parser.add_argument("--candidate_eval_pool_size", type=int, default=cli_defaults.candidate_eval_pool_size)
    parser.add_argument("--candidate_eval_repeats", type=int, default=cli_defaults.candidate_eval_repeats)
    parser.add_argument("--candidate_eval_seed_offset", type=int, default=cli_defaults.candidate_eval_seed_offset)
    parser.add_argument("--candidate_reuse_recorded_rollouts", type=int, default=int(cli_defaults.candidate_reuse_recorded_rollouts), choices=[0, 1])
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
    settings = _selected_settings(args.settings)
    seeds = [int(x) for x in parse_csv_list(args.seeds)] or [42]

    runs_jsonl = out_root / "experiment_runs.jsonl"
    runs_csv = out_root / "experiment_runs.csv"
    accuracy_jsonl = out_root / "accuracy_results.jsonl"
    accuracy_csv = out_root / "accuracy_results.csv"
    runs_jsonl.write_text("", encoding="utf-8")
    accuracy_jsonl.write_text("", encoding="utf-8")

    run_rows: List[Dict[str, Any]] = []
    accuracy_rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        task = tasks[task_id]
        skip_task_for_seed: Dict[int, Dict[str, Any]] = {}
        if int(args.skip_high_baseline_acc):
            for seed in seeds:
                precheck_row = run_precheck(task, seed, args)
                run_rows.append(precheck_row)
                append_jsonl(runs_jsonl, precheck_row)
                write_csv(runs_csv, run_rows, empty_fieldnames=["task_id", "setting", "status"])
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
                    run_rows.append(run_row)
                    append_jsonl(runs_jsonl, run_row)
                    write_csv(runs_csv, run_rows, empty_fieldnames=["task_id", "setting", "status"])
                    continue
                run_row = run_one(task, setting, seed, args)
                run_rows.append(run_row)
                append_jsonl(runs_jsonl, run_row)
                write_csv(runs_csv, run_rows, empty_fieldnames=["task_id", "setting", "status"])
                if run_row["status"] != "ok":
                    raise SystemExit(run_row["returncode"])
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

    write_jsonl(out_root / "accuracy_results.jsonl", accuracy_rows)
    write_csv(accuracy_csv, accuracy_rows, fieldnames=ACCURACY_RESULT_COLUMNS)
    write_accuracy_summary(accuracy_rows, out_root)
    print(f"\n[DONE] Wrote {accuracy_jsonl}")
    print(f"[DONE] Wrote {out_root / 'accuracy_summary.csv'}")


if __name__ == "__main__":
    main()
