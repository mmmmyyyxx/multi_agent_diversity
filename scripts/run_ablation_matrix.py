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
class AblationSetting:
    name: str
    init_mode: str
    enable_diversification_reward: bool


SETTINGS: List[AblationSetting] = [
    #AblationSetting("A_shared_no_div", "shared", False),
    AblationSetting("B_shared_div", "shared", True),
    # Optional full matrix:
    AblationSetting("C_bank_no_div", "bank", False),
    AblationSetting("D_bank_div", "bank", True),
]


def _reward_params(enabled: bool) -> Dict[str, float]:
    if enabled:
        return {
            "lambda_diversity": 0.5,
            "lambda_homogeneity": 0.35,
            "lambda_invalid_trace": 0.30,
        }
    return {
        "lambda_diversity": 0.0,
        "lambda_homogeneity": 0.0,
        "lambda_invalid_trace": 0.0,
    }


def _load_history_metrics(history_path: Path) -> Dict[str, Any]:
    empty = {
        "epochs_completed": 0,
        "final_train_mean_family_diversity": None,
        "final_test_mean_family_diversity": None,
        "final_train_mean_family_homogeneity_rate": None,
        "final_test_mean_family_homogeneity_rate": None,
        "final_train_vote_acc": None,
        "final_test_vote_acc": None,
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
        "final_train_mean_family_diversity": train.get("mean_family_diversity"),
        "final_test_mean_family_diversity": test.get("mean_family_diversity"),
        "final_train_mean_family_homogeneity_rate": train.get("mean_family_homogeneity_rate"),
        "final_test_mean_family_homogeneity_rate": test.get("mean_family_homogeneity_rate"),
        "final_train_vote_acc": train.get("vote_acc"),
        "final_test_vote_acc": test.get("vote_acc"),
    }


def _write_jsonl_append(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _build_command(
    args: argparse.Namespace,
    setting: AblationSetting,
    out_dir: Path,
    reward: Dict[str, float],
) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "multi_dataset_diverse_rl.cli",
        "--task_type",
        args.task_type,
        "--train_path",
        args.train_path,
        "--test_path",
        args.test_path,
        "--out_dir",
        str(out_dir),
        "--train_size",
        str(args.train_size),
        "--test_size",
        str(args.test_size),
        "--epochs",
        str(args.epochs),
        "--agents",
        str(args.agents),
        "--update_every",
        str(args.update_every),
        "--candidate_eval_batch_size",
        str(args.candidate_eval_batch_size),
        "--init_mode",
        setting.init_mode,
        "--lambda_diversity",
        str(reward["lambda_diversity"]),
        "--lambda_homogeneity",
        str(reward["lambda_homogeneity"]),
        "--lambda_invalid_trace",
        str(reward["lambda_invalid_trace"]),
        "--seed",
        str(args.seed),
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
        "--max_tokens",
        str(args.max_tokens),
        "--critic_max_tokens",
        str(args.critic_max_tokens),
        "--rewriter_max_tokens",
        str(args.rewriter_max_tokens),
        "--baseline_only",
        str(int(args.baseline_only)),
    ]
    return cmd


def _run_one(args: argparse.Namespace, setting: AblationSetting, summary_jsonl: Path) -> Dict[str, Any]:
    if (not Path(args.test_path).exists()) or ((not args.baseline_only) and (not Path(args.train_path).exists())):
        return {
            "setting": setting.name,
            "status": "missing_data",
            "train_path": args.train_path,
            "test_path": args.test_path,
            "error": "required train/test path not found",
        }

    reward = _reward_params(setting.enable_diversification_reward)
    out_dir = Path(args.out_root) / setting.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(args, setting, out_dir, reward)
    print("=" * 120)
    print(f"[RUN] setting={setting.name} init_mode={setting.init_mode} div_reward={setting.enable_diversification_reward}")
    print("Command:", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=args.workspace, check=False)
    elapsed = time.time() - t0

    metrics = _load_history_metrics(out_dir / "history.json")
    rec = {
        "setting": setting.name,
        "init_mode": setting.init_mode,
        "enable_diversification_reward": int(setting.enable_diversification_reward),
        "family_expansion_model": args.family_expansion_model,
        "family_expansion_enabled": int(args.family_expansion_enabled),
        "family_taxonomy_path": args.family_taxonomy_path,
        "use_dual_family_labels": int(args.use_dual_family_labels),
        "primary_family_weight": args.primary_family_weight,
        "secondary_family_weight": args.secondary_family_weight,
        "same_major_family_weight": args.same_major_family_weight,
        "macro_diversity_weight": args.macro_diversity_weight,
        "lambda_diversity": reward["lambda_diversity"],
        "lambda_homogeneity": reward["lambda_homogeneity"],
        "lambda_invalid_trace": reward["lambda_invalid_trace"],
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "elapsed_sec": elapsed,
        "out_dir": str(out_dir),
        "task_type": args.task_type,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "baseline_only": int(args.baseline_only),
        "agents": args.agents,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "epochs_target": args.epochs,
        "update_every": args.update_every,
        "candidate_eval_batch_size": args.candidate_eval_batch_size,
        **metrics,
    }
    _write_jsonl_append(summary_jsonl, rec)
    return rec


def _call_analyzer(args: argparse.Namespace, runs: List[Path]):
    analyzer = Path(args.workspace) / "scripts" / "analyze_ablation.py"
    if not analyzer.exists():
        print("[WARN] scripts/analyze_ablation.py not found, skip analyzer step")
        return

    cmd = [
        args.python,
        str(analyzer),
        "--runs",
        *[str(p) for p in runs],
        "--out_csv",
        str(Path(args.out_root) / "ablation_summary.csv"),
        "--out_md",
        str(Path(args.out_root) / "ablation_summary.md"),
    ]
    print("=" * 120)
    print("[ANALYZE]", " ".join(cmd))
    subprocess.run(cmd, cwd=args.workspace, check=False)


def main():
    parser = argparse.ArgumentParser(description="Run A/B/C/D ablation sequentially and record summaries.")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out_root", type=str, default="runs_abcd")

    parser.add_argument("--task_type", type=str, default="auto", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--train_path", type=str, default="mmlu_train.jsonl")
    parser.add_argument("--test_path", type=str, default="mmlu_test.jsonl")

    # 与 baseline 风格参数保持一致
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--critic_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--rewriter_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--family_expansion_enabled", type=int, default=1, choices=[0, 1])
    parser.add_argument("--family_taxonomy_path", type=str, default="family_taxonomy.json")
    parser.add_argument("--use_dual_family_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--primary_family_weight", type=float, default=0.7)
    parser.add_argument("--secondary_family_weight", type=float, default=0.3)
    parser.add_argument("--same_major_family_weight", type=float, default=0.5)
    parser.add_argument("--macro_diversity_weight", type=float, default=0.5)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)

    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--update_every", type=int, default=5)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=3)

    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--rewriter_max_tokens", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_logging", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)

    parser.add_argument("--baseline_only", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()
    args.transient_retry_forever = bool(int(args.transient_retry_forever))
    args.baseline_only = bool(int(args.baseline_only))
    args.family_expansion_enabled = bool(int(args.family_expansion_enabled))
    args.use_dual_family_labels = bool(int(args.use_dual_family_labels))
    args.llm_call_logging = bool(int(args.llm_call_logging))

    workspace = Path(args.workspace).resolve()
    args.workspace = str(workspace)
    out_root = (workspace / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    args.out_root = str(out_root)

    summary_jsonl = out_root / "abcd_runs.jsonl"
    summary_csv = out_root / "abcd_runs.csv"

    rows: List[Dict[str, Any]] = []
    run_dirs: List[Path] = []
    for setting in SETTINGS:
        rec = _run_one(args, setting, summary_jsonl)
        rows.append(rec)
        if "out_dir" in rec:
            run_dirs.append(Path(rec["out_dir"]))
        _write_csv(summary_csv, rows)

    _call_analyzer(args, run_dirs)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    fail = sum(1 for r in rows if r.get("status") != "ok")
    print("=" * 120)
    print(f"Finished {len(rows)} runs: success={ok}, failed={fail}")
    print(f"Summary JSONL: {summary_jsonl}")
    print(f"Summary CSV  : {summary_csv}")
    print(f"Out root     : {out_root}")


if __name__ == "__main__":
    main()
