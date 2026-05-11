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


SETTINGS: List[ExperimentSetting] = [
    ExperimentSetting("shared_div", "shared", False),
    ExperimentSetting("bank_div", "bank", False),
    ExperimentSetting("shared_baseline", "shared", True),
    ExperimentSetting("bank_baseline", "bank", True),
]


def _reward_params(baseline_only: bool) -> Dict[str, float]:
    if baseline_only:
        return {
            "lambda_diversity": 0.0,
            "lambda_homogeneity": 0.0,
            "lambda_invalid_trace": 0.0,
        }
    return {
        "lambda_diversity": 0.5,
        "lambda_homogeneity": 0.35,
        "lambda_invalid_trace": 0.30,
    }


def _load_history_metrics(history_path: Path) -> Dict[str, Any]:
    empty = {
        "epochs_completed": 0,
        "latest_train_mean_family_diversity": None,
        "latest_test_mean_family_diversity": None,
        "latest_train_mean_family_homogeneity_rate": None,
        "latest_test_mean_family_homogeneity_rate": None,
        "latest_train_mean_llm_direct_diversity_score": None,
        "latest_test_mean_llm_direct_diversity_score": None,
        "latest_train_vote_acc": None,
        "latest_test_vote_acc": None,
    }
    if not history_path.exists():
        return empty

    with history_path.open("r", encoding="utf-8") as f:
        hist = json.load(f)
    if not isinstance(hist, list) or not hist:
        return empty

    last = hist[-1]
    if not isinstance(last, dict):
        return empty
    train = last.get("train", {}) if isinstance(last.get("train", {}), dict) else {}
    test = last.get("test", {}) if isinstance(last.get("test", {}), dict) else {}
    return {
        "epochs_completed": len(hist),
        "latest_train_mean_family_diversity": train.get("mean_family_diversity"),
        "latest_test_mean_family_diversity": test.get("mean_family_diversity"),
        "latest_train_mean_family_homogeneity_rate": train.get("mean_family_homogeneity_rate"),
        "latest_test_mean_family_homogeneity_rate": test.get("mean_family_homogeneity_rate"),
        "latest_train_mean_llm_direct_diversity_score": train.get("mean_llm_direct_diversity_score"),
        "latest_test_mean_llm_direct_diversity_score": test.get("mean_llm_direct_diversity_score"),
        "latest_train_vote_acc": train.get("vote_acc"),
        "latest_test_vote_acc": test.get("vote_acc"),
    }


def _write_jsonl_append(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _reset_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("")


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("setting,status\n")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append_common_cli_args(cmd: List[str], args: argparse.Namespace):
    cmd.extend(
        [
            "--task_type",
            args.task_type,
            "--model",
            args.model,
            "--critic_model",
            args.critic_model,
            "--rewriter_model",
            args.rewriter_model,
            "--family_expansion_model",
            args.family_expansion_model,
            "--family_expansion_enabled",
            str(int(args.family_expansion_enabled)),
            "--family_taxonomy_path",
            args.family_taxonomy_path,
            "--use_dual_family_labels",
            str(int(args.use_dual_family_labels)),
            "--primary_family_weight",
            str(args.primary_family_weight),
            "--secondary_family_weight",
            str(args.secondary_family_weight),
            "--same_major_family_weight",
            str(args.same_major_family_weight),
            "--macro_diversity_weight",
            str(args.macro_diversity_weight),
            "--family_confidence_threshold",
            str(args.family_confidence_threshold),
            "--family_rejudge_on_low_confidence",
            str(int(args.family_rejudge_on_low_confidence)),
            "--min_summary_words",
            str(args.min_summary_words),
            "--max_summary_tokens",
            str(args.max_summary_tokens),
            "--min_evidence_spans",
            str(args.min_evidence_spans),
            "--reward_tie_eps",
            str(args.reward_tie_eps),
            "--invalid_tolerance",
            str(args.invalid_tolerance),
            "--max_retries",
            str(args.max_retries),
            "--retry_sleep",
            str(args.retry_sleep),
            "--transient_retry_forever",
            str(int(args.transient_retry_forever)),
            "--max_transient_retries",
            str(args.max_transient_retries),
            "--max_retry_backoff",
            str(args.max_retry_backoff),
            "--llm_call_logging",
            str(int(args.llm_call_logging)),
            "--llm_call_timeout",
            str(args.llm_call_timeout),
            "--agents",
            str(args.agents),
            "--test_size",
            str(args.test_size),
            "--eval_test_each_epoch",
            str(int(args.eval_test_each_epoch)),
            "--early_stopping_patience",
            str(args.early_stopping_patience),
            "--early_stopping_min_delta",
            str(args.early_stopping_min_delta),
            "--early_stopping_metric",
            args.early_stopping_metric,
            "--init_mode",
            args.current_init_mode,
            "--shared_prompt",
            args.shared_prompt,
            "--max_tokens",
            str(args.max_tokens),
            "--critic_max_tokens",
            str(args.critic_max_tokens),
            "--rewriter_max_tokens",
            str(args.rewriter_max_tokens),
            "--seed",
            str(args.current_seed),
        ]
    )


def _build_command(args: argparse.Namespace, setting: ExperimentSetting, out_dir: Path) -> List[str]:
    reward = _reward_params(setting.baseline_only)
    args.current_init_mode = setting.init_mode
    cmd = [args.python, "-m", "multi_dataset_diverse_rl.cli"]
    _append_common_cli_args(cmd, args)
    cmd.extend(
        [
            "--test_path",
            args.test_path,
            "--out_dir",
            str(out_dir),
            "--baseline_only",
            str(int(setting.baseline_only)),
            "--lambda_diversity",
            str(reward["lambda_diversity"]),
            "--lambda_homogeneity",
            str(reward["lambda_homogeneity"]),
            "--lambda_invalid_trace",
            str(reward["lambda_invalid_trace"]),
        ]
    )
    if not setting.baseline_only:
        cmd.extend(
            [
                "--train_path",
                args.train_path,
                "--val_path",
                args.val_path,
                "--train_size",
                str(args.train_size),
                "--val_size",
                str(args.val_size),
                "--val_split_ratio",
                str(args.val_split_ratio),
                "--epochs",
                str(args.epochs),
                "--update_every",
                str(args.update_every),
                "--candidate_eval_batch_size",
                str(args.candidate_eval_batch_size),
            ]
        )
    return cmd


def _run_one(args: argparse.Namespace, setting: ExperimentSetting, summary_jsonl: Path) -> Dict[str, Any]:
    if not Path(args.test_path).exists():
        return {
            "setting": setting.name,
            "status": "missing_data",
            "test_path": args.test_path,
            "error": "required test path not found",
        }
    if not setting.baseline_only and not Path(args.train_path).exists():
        return {
            "setting": setting.name,
            "status": "missing_data",
            "train_path": args.train_path,
            "error": "required train path not found",
        }

    reward = _reward_params(setting.baseline_only)
    out_name = setting.name if setting.baseline_only and not args.seed_baselines else f"{setting.name}_seed{args.current_seed}"
    out_dir = Path(args.out_root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(args, setting, out_dir)

    print("=" * 120)
    print(
        f"[RUN] setting={setting.name} init_mode={setting.init_mode} "
        f"baseline_only={int(setting.baseline_only)} seed={args.current_seed}"
    )
    print("Command:", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=args.workspace, check=False)
    elapsed = time.time() - t0

    metrics = _load_history_metrics(out_dir / "history.json")
    rec = {
        "setting": setting.name,
        "run_name": out_name,
        "seed": args.current_seed,
        "init_mode": setting.init_mode,
        "baseline_only": int(setting.baseline_only),
        "enable_diversification_reward": int(not setting.baseline_only),
        "lambda_diversity": reward["lambda_diversity"],
        "lambda_homogeneity": reward["lambda_homogeneity"],
        "lambda_invalid_trace": reward["lambda_invalid_trace"],
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "elapsed_sec": elapsed,
        "out_dir": str(out_dir),
        "task_type": args.task_type,
        "train_path": "" if setting.baseline_only else args.train_path,
        "val_path": "" if setting.baseline_only else (args.val_path or f"{args.train_path}:split"),
        "test_path": args.test_path,
        "agents": args.agents,
        "train_size": 0 if setting.baseline_only else args.train_size,
        "val_size": 0 if setting.baseline_only else args.val_size,
        "val_split_ratio": 0.0 if setting.baseline_only else args.val_split_ratio,
        "test_size": args.test_size,
        "epochs_target": 0 if setting.baseline_only else args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopping_metric": args.early_stopping_metric,
        "update_every": 0 if setting.baseline_only else args.update_every,
        "candidate_eval_batch_size": 0 if setting.baseline_only else args.candidate_eval_batch_size,
        "family_expansion_model": args.family_expansion_model,
        "family_expansion_enabled": int(args.family_expansion_enabled),
        "family_taxonomy_path": args.family_taxonomy_path,
        "use_dual_family_labels": int(args.use_dual_family_labels),
        "primary_family_weight": args.primary_family_weight,
        "secondary_family_weight": args.secondary_family_weight,
        "same_major_family_weight": args.same_major_family_weight,
        "macro_diversity_weight": args.macro_diversity_weight,
        "family_confidence_threshold": args.family_confidence_threshold,
        "family_rejudge_on_low_confidence": int(args.family_rejudge_on_low_confidence),
        "min_summary_words": args.min_summary_words,
        "max_summary_tokens": args.max_summary_tokens,
        "min_evidence_spans": args.min_evidence_spans,
        "reward_tie_eps": args.reward_tie_eps,
        "invalid_tolerance": args.invalid_tolerance,
        **metrics,
    }
    _write_jsonl_append(summary_jsonl, rec)
    return rec


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment settings with optional multi-seed training runs."
    )
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out_root", type=str, default="runs_experiments")

    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--train_path", type=str, default="mmlu_train.jsonl")
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="mmlu_test.jsonl")

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--critic_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--rewriter_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_enabled", type=int, default=1, choices=[0, 1])
    parser.add_argument("--family_taxonomy_path", type=str, default="auto")
    parser.add_argument("--use_dual_family_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--primary_family_weight", type=float, default=0.7)
    parser.add_argument("--secondary_family_weight", type=float, default=0.3)
    parser.add_argument("--same_major_family_weight", type=float, default=0.5)
    parser.add_argument("--macro_diversity_weight", type=float, default=0.5)
    parser.add_argument("--family_confidence_threshold", type=float, default=0.3)
    parser.add_argument("--family_rejudge_on_low_confidence", type=int, default=1, choices=[0, 1])
    parser.add_argument("--min_summary_words", type=int, default=60)
    parser.add_argument("--max_summary_tokens", type=int, default=512)
    parser.add_argument("--min_evidence_spans", type=int, default=1)
    parser.add_argument("--reward_tie_eps", type=float, default=0.03)
    parser.add_argument("--invalid_tolerance", type=float, default=0.1)

    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_logging", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)

    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--val_split_ratio", type=float, default=0.2)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval_test_each_epoch", type=int, default=0, choices=[0, 1])
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.005)
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="val_mean_family_diversity",
        choices=[
            "val_mean_family_diversity",
            "val_mean_family_homogeneity_rate",
        ],
    )
    parser.add_argument("--update_every", type=int, default=5)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=10)

    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--rewriter_max_tokens", type=int, default=1000)
    parser.add_argument(
        "--shared_prompt",
        type=str,
        default="You are a helpful reasoning assistant. Think step by step.",
    )

    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--seed_baselines", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    args.transient_retry_forever = bool(int(args.transient_retry_forever))
    args.family_expansion_enabled = bool(int(args.family_expansion_enabled))
    args.use_dual_family_labels = bool(int(args.use_dual_family_labels))
    args.family_rejudge_on_low_confidence = bool(int(args.family_rejudge_on_low_confidence))
    args.llm_call_logging = bool(int(args.llm_call_logging))
    args.eval_test_each_epoch = bool(int(args.eval_test_each_epoch))
    args.seed_baselines = bool(int(args.seed_baselines))
    seeds = []
    for raw_seed in str(args.seeds).split(","):
        raw_seed = raw_seed.strip()
        if raw_seed:
            seeds.append(int(raw_seed))
    if not seeds:
        seeds = [42]

    workspace = Path(args.workspace).resolve()
    args.workspace = str(workspace)
    out_root = (workspace / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    args.out_root = str(out_root)

    summary_jsonl = out_root / "experiment_runs.jsonl"
    summary_csv = out_root / "experiment_runs.csv"
    _reset_jsonl(summary_jsonl)

    rows: List[Dict[str, Any]] = []
    for setting in SETTINGS:
        setting_seeds = seeds if (not setting.baseline_only or args.seed_baselines) else [seeds[0]]
        for seed in setting_seeds:
            args.current_seed = int(seed)
            rec = _run_one(args, setting, summary_jsonl)
            rows.append(rec)
            _write_csv(summary_csv, rows)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    fail = sum(1 for r in rows if r.get("status") != "ok")
    print("=" * 120)
    print(f"Finished {len(rows)} experiment runs: success={ok}, failed={fail}")
    print(f"Summary JSONL: {summary_jsonl}")
    print(f"Summary CSV  : {summary_csv}")
    print(f"Out root     : {out_root}")
    print("Next: run scripts/analyze_experiments.py to analyze and plot existing experiment results.")


if __name__ == "__main__":
    main()
