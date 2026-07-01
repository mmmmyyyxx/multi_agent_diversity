import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExperimentSetting:
    name: str
    init_mode: str
    baseline_only: bool


SETTINGS = [
    #ExperimentSetting("shared_beam", "shared", False),
    #ExperimentSetting("bank_beam", "bank", False),
    ExperimentSetting("shared_baseline", "shared", True),
    ExperimentSetting("bank_baseline", "bank", True),
]


def _load_history_metrics(history_path: Path) -> Dict[str, Any]:
    empty = {
        "epochs_completed": 0,
        "latest_train_embedding_diversity": None,
        "latest_train_embedding_overlap": None,
        "latest_train_invalid_rate": None,
        "latest_train_vote_acc": None,
        "latest_test_embedding_diversity": None,
        "latest_test_embedding_overlap": None,
        "latest_test_invalid_rate": None,
        "latest_test_vote_acc": None,
    }
    if not history_path.exists():
        return empty
    hist = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(hist, list) or not hist:
        return empty
    last = hist[-1] if isinstance(hist[-1], dict) else {}
    train = last.get("train", {}) if isinstance(last.get("train", {}), dict) else {}
    test = last.get("test", {}) if isinstance(last.get("test", {}), dict) else {}
    return {
        "epochs_completed": len(hist),
        "latest_train_embedding_diversity": train.get("mean_embedding_diversity"),
        "latest_train_embedding_overlap": train.get("mean_embedding_overlap"),
        "latest_train_invalid_rate": train.get("mean_invalid_rate"),
        "latest_train_vote_acc": train.get("vote_acc"),
        "latest_test_embedding_diversity": test.get("mean_embedding_diversity"),
        "latest_test_embedding_overlap": test.get("mean_embedding_overlap"),
        "latest_test_invalid_rate": test.get("mean_invalid_rate"),
        "latest_test_vote_acc": test.get("vote_acc"),
    }


def _write_jsonl_append(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else ["setting", "status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append_common_cli_args(cmd: List[str], args: argparse.Namespace):
    cmd.extend(
        [
            "--task_type", args.task_type,
            "--agent_model", args.agent_model,
            "--optimizer_model", args.optimizer_model,
            "--evaluator_model", args.evaluator_model,
            "--search_mode", args.search_mode,
            "--reward_mode", args.reward_mode,
            "--beam_size", str(args.beam_size),
            "--num_candidates_per_parent", str(args.num_candidates_per_parent),
            "--beam_refresh_each_epoch", str(int(args.beam_refresh_each_epoch)),
            "--homogeneity_overlap_threshold", str(args.homogeneity_overlap_threshold),
            "--homogeneity_pressure_tie_eps", str(args.homogeneity_pressure_tie_eps),
            "--max_homogeneous_cases_per_agent", str(args.max_homogeneous_cases_per_agent),
            "--random_window_cases_per_agent", str(args.random_window_cases_per_agent),
            "--hard_validity_cases_per_agent", str(args.hard_validity_cases_per_agent),
            "--invalid_repair_rate_threshold", str(args.invalid_repair_rate_threshold),
            "--reward_weight_diversity", str(args.reward_weight_diversity),
            "--reward_weight_local_validity", str(args.reward_weight_local_validity),
            "--reward_weight_team_accuracy", str(args.reward_weight_team_accuracy),
            "--reward_weight_invalid_score", str(args.reward_weight_invalid_score),
            "--diversity_metric", args.diversity_metric,
            "--use_joint_trace_diversity_evaluator", str(int(args.use_joint_trace_diversity_evaluator)),
            "--local_validity_binary", str(int(args.local_validity_binary)),
            "--invalid_binary", str(int(args.invalid_binary)),
            "--embedding_model", args.embedding_model,
            "--trace_embedding_chunk_words", str(args.trace_embedding_chunk_words),
            "--trace_embedding_chunk_overlap", str(args.trace_embedding_chunk_overlap),
            "--max_retries", str(args.max_retries),
            "--retry_sleep", str(args.retry_sleep),
            "--transient_retry_forever", str(int(args.transient_retry_forever)),
            "--max_transient_retries", str(args.max_transient_retries),
            "--max_retry_backoff", str(args.max_retry_backoff),
            "--llm_call_logging", str(int(args.llm_call_logging)),
            "--llm_call_timeout", str(args.llm_call_timeout),
            "--candidate_eval_concurrency", str(args.candidate_eval_concurrency),
            "--train_rollout_concurrency", str(args.train_rollout_concurrency),
            "--eval_solver_call_concurrency", str(args.eval_solver_call_concurrency),
            "--local_evaluator_batch_size", str(args.local_evaluator_batch_size),
            "--agents", str(args.agents),
            "--test_size", str(args.test_size),
            "--eval_test_each_epoch", str(int(args.eval_test_each_epoch)),
            "--early_stopping_patience", str(args.early_stopping_patience),
            "--early_stopping_min_delta", str(args.early_stopping_min_delta),
            "--init_mode", args.current_init_mode,
            "--shared_prompt", args.shared_prompt,
            "--max_tokens", str(args.max_tokens),
            "--optimizer_max_tokens", str(args.optimizer_max_tokens),
            "--evaluator_max_tokens", str(args.evaluator_max_tokens),
            "--seed", str(args.current_seed),
        ]
    )


def run_one(setting: ExperimentSetting, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    run_name = f"{setting.name}_seed{seed}" if args.multi_seed_names else setting.name
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.current_init_mode = setting.init_mode
    args.current_seed = seed

    cmd = [args.python, "-m", "multi_dataset_diverse_rl.cli"]
    _append_common_cli_args(cmd, args)
    cmd.extend(["--test_path", args.test_path, "--out_dir", str(run_dir), "--baseline_only", "1" if setting.baseline_only else "0"])
    if not setting.baseline_only:
        cmd.extend(
            [
                "--train_path", args.train_path,
                "--val_path", args.val_path,
                "--train_size", str(args.train_size),
                "--val_size", str(args.val_size),
                "--val_split_ratio", str(args.val_split_ratio),
                "--epochs", str(args.epochs),
                "--update_every", str(args.update_every),
                "--candidate_eval_batch_size", str(args.candidate_eval_batch_size),
            ]
        )

    start = time.time()
    print(f"\n[RUN] {run_name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=args.workspace)
    elapsed = time.time() - start
    status = "ok" if proc.returncode == 0 else "failed"
    row = {
        "setting": setting.name,
        "run_name": run_name,
        "seed": seed,
        "init_mode": setting.init_mode,
        "baseline_only": int(setting.baseline_only),
        "status": status,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "run_dir": str(run_dir),
        "agent_model": args.agent_model,
        "optimizer_model": args.optimizer_model,
        "evaluator_model": args.evaluator_model,
        "search_mode": args.search_mode,
        "reward_mode": args.reward_mode,
        "beam_size": args.beam_size,
    }
    row.update(_load_history_metrics(run_dir / "history.json"))
    return row


def main():
    parser = argparse.ArgumentParser(description="Run the trace-embedding evolutionary beam experiments.")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out_root", type=str, default="runs_trace_beam")
    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--train_path", type=str, default="mmlu_train.jsonl")
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="mmlu_test.jsonl")

    parser.add_argument("--agent_model", type=str, default="deepseek-chat")
    parser.add_argument("--optimizer_model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--evaluator_model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--search_mode", type=str, default="evolutionary_beam", choices=["evolutionary_beam"])
    parser.add_argument("--reward_mode", type=str, default="embedding_local_acc_invalid", choices=["embedding_local_acc_invalid", "accuracy_only"])
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--num_candidates_per_parent", type=int, default=2)
    parser.add_argument("--beam_refresh_each_epoch", type=int, default=1, choices=[0, 1])
    parser.add_argument("--homogeneity_overlap_threshold", type=float, default=0.55)
    parser.add_argument("--homogeneity_pressure_tie_eps", type=float, default=0.03)
    parser.add_argument("--max_homogeneous_cases_per_agent", type=int, default=4)
    parser.add_argument("--random_window_cases_per_agent", type=int, default=2)
    parser.add_argument("--hard_validity_cases_per_agent", type=int, default=2)
    parser.add_argument("--invalid_repair_rate_threshold", type=float, default=0.25)
    parser.add_argument("--reward_weight_diversity", type=float, default=0.5)
    parser.add_argument("--reward_weight_local_validity", type=float, default=0.2)
    parser.add_argument("--reward_weight_team_accuracy", type=float, default=0.1)
    parser.add_argument("--reward_weight_invalid_score", type=float, default=0.2)
    parser.add_argument("--diversity_metric", type=str, default="trace_embedding", choices=["trace_embedding"])
    parser.add_argument("--use_joint_trace_diversity_evaluator", type=int, default=0, choices=[0, 1])
    parser.add_argument("--local_validity_binary", type=int, default=1, choices=[0, 1])
    parser.add_argument("--invalid_binary", type=int, default=1, choices=[0, 1])
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--trace_embedding_chunk_words", type=int, default=320)
    parser.add_argument("--trace_embedding_chunk_overlap", type=int, default=40)

    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--val_split_ratio", type=float, default=0.2)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval_test_each_epoch", type=int, default=0, choices=[0, 1])
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--update_every", type=int, default=10)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--optimizer_max_tokens", type=int, default=1400)
    parser.add_argument("--evaluator_max_tokens", type=int, default=1200)
    parser.add_argument("--shared_prompt", type=str, default="You are a careful reasoning solver. Produce a compact, explicit reasoning trace, make your decision procedure visible, verify key logic, and give exactly one final answer.")

    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_logging", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)
    parser.add_argument("--candidate_eval_concurrency", type=int, default=0)
    parser.add_argument("--train_rollout_concurrency", type=int, default=0)
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=225)
    parser.add_argument("--local_evaluator_batch_size", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--seed_baselines", type=int, default=0, choices=[0, 1])
    parser.add_argument("--multi_seed_names", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()
    for name in [
        "beam_refresh_each_epoch",
        "use_joint_trace_diversity_evaluator",
        "local_validity_binary",
        "invalid_binary",
        "eval_test_each_epoch",
        "transient_retry_forever",
        "llm_call_logging",
        "seed_baselines",
        "multi_seed_names",
    ]:
        setattr(args, name, bool(int(getattr(args, name))))

    args.workspace = str(Path(args.workspace).resolve())
    args.out_root = str((Path(args.workspace) / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root).resolve())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    runs_jsonl = out_root / "experiment_runs.jsonl"
    runs_csv = out_root / "experiment_runs.csv"
    runs_jsonl.write_text("", encoding="utf-8")

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if not seeds:
        seeds = [42]

    rows = []
    settings = list(SETTINGS)
    if str(args.reward_mode).lower() == "accuracy_only":
        settings = [
            ExperimentSetting("shared_accuracy_only", "shared", False),
        ]

    for setting in settings:
        setting_seeds = seeds if (not setting.baseline_only or args.seed_baselines) else [seeds[0]]
        for seed in setting_seeds:
            row = run_one(setting, seed, args)
            rows.append(row)
            _write_jsonl_append(runs_jsonl, row)
            _write_csv(runs_csv, rows)
            if row["status"] != "ok":
                raise SystemExit(row["returncode"])
    print(f"\n[DONE] Wrote {runs_jsonl} and {runs_csv}")


if __name__ == "__main__":
    main()
