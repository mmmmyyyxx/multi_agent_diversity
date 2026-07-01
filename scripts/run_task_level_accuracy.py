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
    from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTINGS, ExperimentSetting, parse_csv_list, select_settings
    from scripts.experiment_io import append_jsonl, write_csv, write_jsonl
    from scripts.task_level_accuracy_utils import ACCURACY_RESULT_COLUMNS, build_accuracy_result_row
except ModuleNotFoundError:
    from experiment_config import DEFAULT_EXPERIMENT_SETTINGS, ExperimentSetting, parse_csv_list, select_settings
    from experiment_io import append_jsonl, write_csv, write_jsonl
    from task_level_accuracy_utils import ACCURACY_RESULT_COLUMNS, build_accuracy_result_row

from multi_dataset_diverse_rl.task_manifest import ComparisonTask, load_task_manifest, resolve_task_ids


SETTINGS = DEFAULT_EXPERIMENT_SETTINGS


def _selected_settings(raw: str) -> List[ExperimentSetting]:
    return select_settings(raw, SETTINGS)


def _append_common_cli_args(cmd: List[str], args: argparse.Namespace, task: ComparisonTask, setting: ExperimentSetting, seed: int):
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
            "--reward_mode", setting.reward_mode,
            "--agents", str(args.agents),
            "--init_mode", setting.init_mode,
            "--shared_prompt", args.shared_prompt,
            "--beam_size", str(args.beam_size),
            "--num_candidates_per_parent", str(args.num_candidates_per_parent),
            "--beam_refresh_each_epoch", str(args.beam_refresh_each_epoch),
            "--reward_weight_diversity", str(args.reward_weight_diversity),
            "--reward_weight_local_validity", str(args.reward_weight_local_validity),
            "--reward_weight_team_accuracy", str(args.reward_weight_team_accuracy),
            "--reward_weight_invalid_score", str(args.reward_weight_invalid_score),
            "--accuracy_guard_epsilon", str(args.accuracy_guard_epsilon),
            "--reward_weight_div_delta", str(args.reward_weight_div_delta),
            "--reward_weight_invalid_delta", str(args.reward_weight_invalid_delta),
            "--use_baseline_relative_reward", str(args.use_baseline_relative_reward),
            "--candidate_eval_strategy", args.candidate_eval_strategy,
            "--candidate_eval_pool_size", str(args.candidate_eval_pool_size),
            "--candidate_eval_repeats", str(args.candidate_eval_repeats),
            "--candidate_eval_seed_offset", str(args.candidate_eval_seed_offset),
            "--candidate_reuse_recorded_rollouts", str(args.candidate_reuse_recorded_rollouts),
            "--train_rollout_concurrency", str(args.train_rollout_concurrency),
            "--eval_solver_call_concurrency", str(args.eval_solver_call_concurrency),
            "--local_evaluator_batch_size", str(args.local_evaluator_batch_size),
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
        "reward_mode": setting.reward_mode,
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
        for metric in ["vote_acc", "mean_individual_acc", "best_individual_acc", "total_llm_calls", "total_tokens", "estimated_cost"]:
            values = [float(row.get(metric, 0.0) or 0.0) for row in group]
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_std"] = _std(values)
        summary_rows.append(out)
    write_csv(out_root / "accuracy_summary.csv", summary_rows, empty_fieldnames=["task_id", "benchmark", "setting", "n"])
    lines = ["# Task-Level Accuracy Summary", ""]
    if not summary_rows:
        lines.append("No completed runs.")
    else:
        columns = ["task_id", "benchmark", "setting", "n", "vote_acc_mean", "vote_acc_std", "mean_individual_acc_mean", "best_individual_acc_mean"]
        lines.extend(["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"])
        for row in summary_rows:
            lines.append("| " + " | ".join(str(round(row.get(c, 0.0), 6)) if isinstance(row.get(c), float) else str(row.get(c, "")) for c in columns) + " |")
    (out_root / "accuracy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run MAD at task_id granularity and export standardized accuracy results.")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--manifest", type=str, default="configs/task_level_comparison.yaml")
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--benchmarks", type=str, default="")
    parser.add_argument("--settings", type=str, default="shared_baseline,shared_guarded_beam,bank_guarded_beam")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--dataset_format", type=str, default="mars", choices=["legacy", "mars"])
    parser.add_argument("--out_root", type=str, default="runs_task_level_accuracy")

    parser.add_argument("--agent_model", type=str, default="deepseek-chat")
    parser.add_argument("--optimizer_model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--evaluator_model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--val_split_ratio", type=float, default=0.2)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--update_every", type=int, default=10)
    parser.add_argument("--eval_test_each_epoch", type=int, default=0, choices=[0, 1])
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--shared_prompt", type=str, default="You are a careful reasoning solver. Produce a compact, explicit reasoning trace, make your decision procedure visible, verify key logic, and give exactly one final answer.")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--num_candidates_per_parent", type=int, default=2)
    parser.add_argument("--beam_refresh_each_epoch", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reward_weight_diversity", type=float, default=0.5)
    parser.add_argument("--reward_weight_local_validity", type=float, default=0.2)
    parser.add_argument("--reward_weight_team_accuracy", type=float, default=0.1)
    parser.add_argument("--reward_weight_invalid_score", type=float, default=0.2)
    parser.add_argument("--accuracy_guard_epsilon", type=float, default=0.02)
    parser.add_argument("--reward_weight_div_delta", type=float, default=0.3)
    parser.add_argument("--reward_weight_invalid_delta", type=float, default=0.5)
    parser.add_argument("--use_baseline_relative_reward", type=int, default=1, choices=[0, 1])
    parser.add_argument("--candidate_eval_batch_size", type=int, default=20)
    parser.add_argument("--candidate_eval_strategy", type=str, default="fixed_pool", choices=["random", "fixed_pool", "stratified"])
    parser.add_argument("--candidate_eval_pool_size", type=int, default=100)
    parser.add_argument("--candidate_eval_repeats", type=int, default=1)
    parser.add_argument("--candidate_eval_seed_offset", type=int, default=1000)
    parser.add_argument("--candidate_reuse_recorded_rollouts", type=int, default=1, choices=[0, 1])
    parser.add_argument("--train_rollout_concurrency", type=int, default=0)
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=225)
    parser.add_argument("--local_evaluator_batch_size", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--optimizer_max_tokens", type=int, default=1400)
    parser.add_argument("--evaluator_max_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--optimizer_temperature", type=float, default=0.5)
    parser.add_argument("--evaluator_temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_logging", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)
    parser.add_argument("--vote_tie_break", type=str, default="random", choices=["first", "random", "abstain"])
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
        for setting in settings:
            for seed in seeds:
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
