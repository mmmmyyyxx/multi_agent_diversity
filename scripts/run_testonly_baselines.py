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
class BaselineSetting:
    name: str
    init_mode: str


BASELINE_SETTINGS: List[BaselineSetting] = [
    BaselineSetting("E_shared_testonly", "shared"),
    BaselineSetting("F_bank_testonly", "bank"),
]


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


def _load_history_metrics(history_path: Path) -> Dict[str, Any]:
    empty = {
        "epochs_completed": 0,
        "final_train_mean_family_diversity": None,
        "final_test_mean_family_diversity": None,
        "final_train_mean_family_homogeneity_rate": None,
        "final_test_mean_family_homogeneity_rate": None,
        "final_train_mean_llm_direct_diversity_score": None,
        "final_test_mean_llm_direct_diversity_score": None,
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
        "final_train_mean_llm_direct_diversity_score": train.get("mean_llm_direct_diversity_score"),
        "final_test_mean_llm_direct_diversity_score": test.get("mean_llm_direct_diversity_score"),
        "final_train_vote_acc": train.get("vote_acc"),
        "final_test_vote_acc": test.get("vote_acc"),
    }


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _build_command(args: argparse.Namespace, setting: BaselineSetting, out_dir: Path) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "multi_dataset_diverse_rl.cli",
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
        "--max_retries",
        str(args.max_retries),
        "--retry_sleep",
        str(args.retry_sleep),
        "--max_transient_retries",
        str(args.max_transient_retries),
        "--max_retry_backoff",
        str(args.max_retry_backoff),
        "--transient_retry_forever",
        str(int(args.transient_retry_forever)),
        "--test_path",
        args.test_path,
        "--agents",
        str(args.agents),
        "--baseline_only",
        "1",
        "--test_size",
        str(args.test_size),
        "--init_mode",
        setting.init_mode,
        "--shared_prompt",
        args.shared_prompt,
        "--max_tokens",
        str(args.max_tokens),
        "--critic_max_tokens",
        str(args.critic_max_tokens),
        "--rewriter_max_tokens",
        str(args.rewriter_max_tokens),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(out_dir),
    ]
    return cmd


def _run_one(args: argparse.Namespace, setting: BaselineSetting, summary_jsonl: Path) -> Dict[str, Any]:
    if not Path(args.test_path).exists():
        return {
            "setting": setting.name,
            "status": "missing_data",
            "test_path": args.test_path,
            "error": "required test path not found",
        }

    out_dir = Path(args.out_root) / setting.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(args, setting, out_dir)
    print("=" * 120)
    print(f"[RUN] setting={setting.name} init_mode={setting.init_mode} baseline_only=1")
    print("Command:", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=args.workspace, check=False)
    elapsed = time.time() - t0

    metrics = _load_history_metrics(out_dir / "history.json")
    rec = {
        "setting": setting.name,
        "init_mode": setting.init_mode,
        "enable_diversification_reward": 0,
        "family_expansion_model": args.family_expansion_model,
        "family_expansion_enabled": int(args.family_expansion_enabled),
        "family_taxonomy_path": args.family_taxonomy_path,
        "use_dual_family_labels": int(args.use_dual_family_labels),
        "primary_family_weight": args.primary_family_weight,
        "secondary_family_weight": args.secondary_family_weight,
        "same_major_family_weight": args.same_major_family_weight,
        "macro_diversity_weight": args.macro_diversity_weight,
        "lambda_diversity": 0.0,
        "lambda_homogeneity": 0.0,
        "lambda_invalid_trace": 0.0,
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "elapsed_sec": elapsed,
        "out_dir": str(out_dir),
        "task_type": args.task_type,
        "test_path": args.test_path,
        "baseline_only": 1,
        "agents": args.agents,
        "test_size": args.test_size,
        "epochs_target": 0,
        "update_every": 0,
        "candidate_eval_batch_size": 0,
        **metrics,
    }
    _write_jsonl_append(summary_jsonl, rec)
    return rec


def _collect_existing_run_dirs(out_root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    if not out_root.exists():
        return run_dirs
    for p in sorted(out_root.iterdir()):
        if p.is_dir() and (p / "run_meta.json").exists() and (p / "history.json").exists():
            run_dirs.append(p)
    return run_dirs


def _build_merged_rows(abcd_rows: List[Dict[str, Any]], baseline_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for r in abcd_rows:
        rec = dict(r)
        rec.setdefault("setting", Path(str(rec.get("out_dir", ""))).name or "unknown")
        rec["group"] = "train"
        rec["baseline_only"] = _safe_int(rec.get("baseline_only", 0), 0)
        merged.append(rec)
    for r in baseline_rows:
        rec = dict(r)
        rec["group"] = "test_only"
        rec["baseline_only"] = _safe_int(rec.get("baseline_only", 1), 1)
        merged.append(rec)

    order = {
        "A_shared_no_div": 0,
        "B_shared_div": 1,
        "C_bank_no_div": 2,
        "D_bank_div": 3,
        "E_shared_testonly": 4,
        "F_bank_testonly": 5,
    }
    merged.sort(key=lambda x: (order.get(str(x.get("setting", "")), 99), str(x.get("setting", ""))))
    return merged


def _call_analyzer(args: argparse.Namespace, run_dirs: List[Path]):
    analyzer = Path(args.workspace) / "scripts" / "analyze_ablation.py"
    if not analyzer.exists():
        print("[WARN] scripts/analyze_ablation.py not found, skip analyzer step")
        return

    cmd = [
        args.python,
        str(analyzer),
        "--runs",
        *[str(p) for p in run_dirs],
        "--out_csv",
        str(Path(args.out_root) / "ablation_summary_with_baselines.csv"),
        "--out_md",
        str(Path(args.out_root) / "ablation_summary_with_baselines.md"),
    ]
    print("=" * 120)
    print("[ANALYZE]", " ".join(cmd))
    subprocess.run(cmd, cwd=args.workspace, check=False)


def _call_plotter(args: argparse.Namespace, csv_path: Path):
    plotter = Path(args.workspace) / "scripts" / "plot_ablation_with_baselines.py"
    if not plotter.exists():
        print("[WARN] scripts/plot_ablation_with_baselines.py not found, skip plotting")
        return

    cmd = [
        args.python,
        str(plotter),
        "--csv",
        str(csv_path),
        "--out_dir",
        str(Path(args.out_root)),
    ]
    print("=" * 120)
    print("[PLOT]", " ".join(cmd))
    subprocess.run(cmd, cwd=args.workspace, check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run two no-training baselines (shared/bank) and aggregate visualization with existing ablation runs."
    )
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out_root", type=str, default="runs_abcd")

    parser.add_argument("--task_type", type=str, default="auto", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--test_path", type=str, default="test.jsonl")

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
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)

    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--rewriter_max_tokens", type=int, default=1000)

    parser.add_argument(
        "--shared_prompt",
        type=str,
        default="You are a helpful reasoning assistant. Think step by step.",
    )

    args = parser.parse_args()
    args.transient_retry_forever = bool(int(args.transient_retry_forever))
    args.family_expansion_enabled = bool(int(args.family_expansion_enabled))
    args.use_dual_family_labels = bool(int(args.use_dual_family_labels))

    workspace = Path(args.workspace).resolve()
    args.workspace = str(workspace)
    out_root = (workspace / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    args.out_root = str(out_root)

    baseline_jsonl = out_root / "baseline_runs.jsonl"
    baseline_csv = out_root / "baseline_runs.csv"
    merged_jsonl = out_root / "abcd_plus_baselines.jsonl"
    merged_csv = out_root / "abcd_plus_baselines.csv"

    # 让脚本可重复运行：每次重置本脚本产物，避免重复追加
    _reset_jsonl(baseline_jsonl)
    _reset_jsonl(merged_jsonl)

    baseline_rows: List[Dict[str, Any]] = []
    for setting in BASELINE_SETTINGS:
        rec = _run_one(args, setting, baseline_jsonl)
        baseline_rows.append(rec)
        _write_csv(baseline_csv, baseline_rows)

    # 合并已有 A/B/C/D 汇总（若存在）与新增 baseline 汇总
    abcd_rows = _read_csv_rows(out_root / "abcd_runs.csv")
    merged_rows = _build_merged_rows(abcd_rows, baseline_rows)
    _write_csv(merged_csv, merged_rows)
    for row in merged_rows:
        _write_jsonl_append(merged_jsonl, row)

    # 对 runs_abcd 下所有有效 run 统一再分析一次（包含 A/B/C/D/E/F）
    all_run_dirs = _collect_existing_run_dirs(out_root)
    _call_analyzer(args, all_run_dirs)
    _call_plotter(args, out_root / "ablation_summary_with_baselines.csv")

    ok = sum(1 for r in baseline_rows if r.get("status") == "ok")
    fail = sum(1 for r in baseline_rows if r.get("status") != "ok")
    print("=" * 120)
    print(f"Finished baseline runs: success={ok}, failed={fail}")
    print(f"Baseline JSONL: {baseline_jsonl}")
    print(f"Baseline CSV  : {baseline_csv}")
    print(f"Merged JSONL  : {merged_jsonl}")
    print(f"Merged CSV    : {merged_csv}")
    print(f"Out root      : {out_root}")


if __name__ == "__main__":
    main()
