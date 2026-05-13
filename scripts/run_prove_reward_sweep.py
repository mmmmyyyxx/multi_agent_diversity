import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


CONDITIONS = [
    {
        "name": "no_div",
        "lambda_diversity": 0.0,
        "lambda_homogeneity": 0.0,
        "same_major_family_weight": 0.5,
    },
    {
        "name": "weak",
        "lambda_diversity": 0.25,
        "lambda_homogeneity": 0.15,
        "same_major_family_weight": 0.5,
    },
    {
        "name": "default",
        "lambda_diversity": 0.5,
        "lambda_homogeneity": 0.35,
        "same_major_family_weight": 0.5,
    },
    {
        "name": "strong",
        "lambda_diversity": 0.8,
        "lambda_homogeneity": 0.55,
        "same_major_family_weight": 0.5,
    },
    {
        "name": "softened_tree",
        "lambda_diversity": 0.5,
        "lambda_homogeneity": 0.35,
        "same_major_family_weight": 0.7,
    },
    {
        "name": "strict_tree",
        "lambda_diversity": 0.5,
        "lambda_homogeneity": 0.35,
        "same_major_family_weight": 0.25,
    },
]


def _parse_conditions(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return list(CONDITIONS)
    wanted = {x.strip() for x in raw.split(",") if x.strip()}
    return [c for c in CONDITIONS if c["name"] in wanted]


def _read_history(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        hist = json.load(f)
    if not isinstance(hist, list) or not hist:
        return {}
    last = hist[-1] if isinstance(hist[-1], dict) else {}
    out: Dict[str, Any] = {
        "history_len": len(hist),
        "last_epoch": last.get("epoch", ""),
        "selected_epoch": last.get("selected_epoch", ""),
        "early_stopped": last.get("early_stopped", ""),
    }
    for split in ["train", "val", "test"]:
        block = last.get(split, {}) if isinstance(last.get(split, {}), dict) else {}
        for key, value in block.items():
            out[f"latest_{split}_{key}"] = value
    return out


def _write_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["condition"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_cmd(args: argparse.Namespace, cond: Dict[str, Any], out_dir: Path, seed: int) -> List[str]:
    return [
        args.python,
        "-m",
        "multi_dataset_diverse_rl.cli",
        "--task_type",
        args.task_type,
        "--train_path",
        args.train_path,
        "--val_path",
        args.val_path,
        "--test_path",
        args.test_path,
        "--train_size",
        str(args.train_size),
        "--val_size",
        str(args.val_size),
        "--test_size",
        str(args.test_size),
        "--agents",
        str(args.agents),
        "--init_mode",
        args.init_mode,
        "--epochs",
        str(args.epochs),
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--early_stopping_min_delta",
        str(args.early_stopping_min_delta),
        "--early_stopping_metric",
        args.early_stopping_metric,
        "--candidate_eval_batch_size",
        str(args.candidate_eval_batch_size),
        "--update_every",
        str(args.update_every),
        "--model",
        args.model,
        "--critic_model",
        args.critic_model,
        "--rewriter_model",
        args.rewriter_model,
        "--family_expansion_model",
        args.family_expansion_model,
        "--family_expansion_enabled",
        str(args.family_expansion_enabled),
        "--family_taxonomy_path",
        args.family_taxonomy_path,
        "--use_dual_family_labels",
        str(args.use_dual_family_labels),
        "--same_major_family_weight",
        str(cond["same_major_family_weight"]),
        "--lambda_diversity",
        str(cond["lambda_diversity"]),
        "--lambda_homogeneity",
        str(cond["lambda_homogeneity"]),
        "--lambda_invalid_trace",
        str(args.lambda_invalid_trace),
        "--max_tokens",
        str(args.max_tokens),
        "--critic_max_tokens",
        str(args.critic_max_tokens),
        "--rewriter_max_tokens",
        str(args.rewriter_max_tokens),
        "--max_retries",
        str(args.max_retries),
        "--retry_sleep",
        str(args.retry_sleep),
        "--transient_retry_forever",
        str(args.transient_retry_forever),
        "--max_retry_backoff",
        str(args.max_retry_backoff),
        "--llm_call_timeout",
        str(args.llm_call_timeout),
        "--out_dir",
        str(out_dir),
        "--seed",
        str(seed),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run P5 reward-weight sweep for prove_experiments.")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--conditions", type=str, default="", help="Comma-separated subset: no_div,weak,default,strong,softened_tree,strict_tree")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--train_path", type=str, default="mmlu_train_500.jsonl")
    parser.add_argument("--val_path", type=str, default="mmlu_val_150.jsonl")
    parser.add_argument("--test_path", type=str, default="mmlu_test_200.jsonl")
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--init_mode", type=str, default="shared", choices=["shared", "bank"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--early_stopping_patience", type=int, default=1)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.005)
    parser.add_argument("--early_stopping_metric", type=str, default="val_mean_family_diversity")
    parser.add_argument("--candidate_eval_batch_size", type=int, default=10)
    parser.add_argument("--update_every", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--critic_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--rewriter_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_enabled", type=int, default=0, choices=[0, 1])
    parser.add_argument("--family_taxonomy_path", type=str, default="auto")
    parser.add_argument("--use_dual_family_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lambda_invalid_trace", type=float, default=0.30)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--rewriter_max_tokens", type=int, default=1000)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    out_root = (workspace / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    conditions = _parse_conditions(args.conditions)
    rows: List[Dict[str, Any]] = []
    summary_path = out_root / "reward_sweep_runs.csv"

    for cond in conditions:
        for seed in seeds:
            run_name = f"P5_{cond['name']}_seed{seed}"
            out_dir = out_root / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            if int(args.skip_existing) and (out_dir / "history.json").exists():
                print(f"[SKIP] {run_name} exists")
                rec = {
                    "run_name": run_name,
                    "condition": cond["name"],
                    "seed": seed,
                    "status": "skipped_existing",
                    "out_dir": str(out_dir),
                    **cond,
                    **_read_history(out_dir / "history.json"),
                }
                rows.append(rec)
                _write_csv(rows, summary_path)
                continue
            cmd = _build_cmd(args, cond, out_dir, seed)
            print("=" * 120)
            print(f"[RUN] {run_name}")
            print("Command:", " ".join(cmd))
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(workspace), check=False)
            elapsed = time.time() - t0
            rec = {
                "run_name": run_name,
                "condition": cond["name"],
                "seed": seed,
                "status": "ok" if proc.returncode == 0 else "failed",
                "return_code": proc.returncode,
                "elapsed_sec": elapsed,
                "out_dir": str(out_dir),
                **cond,
                **_read_history(out_dir / "history.json"),
            }
            rows.append(rec)
            _write_csv(rows, summary_path)

    print("=" * 120)
    print(f"Reward sweep finished: {len(rows)} rows")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
