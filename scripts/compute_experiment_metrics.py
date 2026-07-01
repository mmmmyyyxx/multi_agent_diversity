import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


PUBLIC_METRIC_COLUMNS = [
    "run_dir",
    "run_name",
    "setting",
    "seed",
    "baseline_only",
    "init_mode",
    "agents",
    "epochs",
    "train_size",
    "test_size",
    "latest_train_embedding_diversity",
    "latest_train_embedding_overlap",
    "latest_train_invalid_rate",
    "latest_train_vote_acc",
    "latest_test_embedding_diversity",
    "latest_test_embedding_overlap",
    "latest_test_invalid_rate",
    "latest_test_vote_acc",
    "reward",
    "embedding_diversity",
    "mean_embedding_overlap",
    "target_overlap_pressure",
    "homogeneous_case_count",
    "resolved_case_count",
    "new_homogeneous_case_count",
    "local_validity_mean",
    "team_accuracy",
    "invalid_rate",
    "invalid_score",
    "solver_reuse_hits",
    "solver_reuse_misses",
    "solver_calls",
    "solver_reuse_total",
    "solver_reuse_hit_rate",
    "beam_rank",
    "beam_refresh_count",
    "active_prompt_changed_count",
]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def _safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _latest_metrics(history: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    for row in reversed(history):
        if isinstance(row, dict) and isinstance(row.get(key), dict):
            return row[key]
    return {}


def _collect_beam_update_metrics(update_records: List[Dict[str, Any]]) -> Dict[str, float]:
    candidate_rows = [r for r in update_records if isinstance(r, dict) and "reward" in r and r.get("event") != "beam_refresh"]
    refresh_rows = [r for r in update_records if isinstance(r, dict) and r.get("event") == "beam_refresh"]
    accepted_rows = [r for r in candidate_rows if bool(r.get("accepted", False))]

    def vals(key: str, rows: List[Dict[str, Any]] = candidate_rows) -> List[float]:
        return [_float(r.get(key)) for r in rows if r.get(key) not in (None, "")]

    return {
        "candidate_eval_count": float(len(candidate_rows)),
        "reward": _safe_mean(vals("reward")),
        "embedding_diversity": _safe_mean(vals("embedding_diversity")),
        "mean_embedding_overlap": _safe_mean(vals("mean_embedding_overlap")),
        "target_overlap_pressure": _safe_mean(vals("target_overlap_pressure")),
        "homogeneous_case_count": _safe_mean(vals("homogeneous_case_count")),
        "resolved_case_count": _safe_mean(vals("resolved_case_count")),
        "new_homogeneous_case_count": _safe_mean(vals("new_homogeneous_case_count")),
        "local_validity_mean": _safe_mean(vals("local_validity_mean")),
        "team_accuracy": _safe_mean(vals("team_accuracy")),
        "invalid_rate": _safe_mean(vals("invalid_rate")),
        "invalid_score": _safe_mean(vals("invalid_score")),
        "solver_reuse_hits": _safe_mean(vals("solver_reuse_hits")),
        "solver_reuse_misses": _safe_mean(vals("solver_reuse_misses")),
        "solver_calls": _safe_mean(vals("solver_calls")),
        "solver_reuse_total": _safe_mean(vals("solver_reuse_total")),
        "solver_reuse_hit_rate": _safe_mean(vals("solver_reuse_hit_rate")),
        "beam_rank": _safe_mean(vals("rank_in_beam", accepted_rows)),
        "beam_refresh_count": float(len(refresh_rows)),
        "active_prompt_changed_count": float(sum(1 for r in refresh_rows if bool(r.get("active_prompt_changed", False)))),
    }


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    run_meta = _read_json(run_dir / "run_meta.json") or {}
    cfg = run_meta.get("config", {}) if isinstance(run_meta.get("config", {}), dict) else {}
    history = _read_json(run_dir / "history.json") or []
    if not isinstance(history, list):
        history = []
    train = _latest_metrics(history, "train")
    test = _latest_metrics(history, "test")
    update_logs = _read_jsonl(run_dir / "update_logs.jsonl")
    name = run_dir.name
    setting = name.split("_seed")[0] if "_seed" in name else name
    seed = None
    if "_seed" in name:
        try:
            seed = int(name.rsplit("_seed", 1)[1])
        except Exception:
            seed = None
    out = {
        "run_dir": str(run_dir),
        "run_name": name,
        "setting": setting,
        "seed": seed,
        "baseline_only": bool(cfg.get("baseline_only", False)),
        "init_mode": cfg.get("init_mode", ""),
        "agents": cfg.get("agents", ""),
        "epochs": cfg.get("epochs", ""),
        "train_size": cfg.get("train_size", ""),
        "test_size": cfg.get("test_size", ""),
        "agent_model": cfg.get("agent_model", ""),
        "optimizer_model": cfg.get("optimizer_model", ""),
        "evaluator_model": cfg.get("evaluator_model", ""),
        "search_mode": cfg.get("search_mode", ""),
        "reward_mode": cfg.get("reward_mode", ""),
        "latest_train_embedding_diversity": train.get("mean_embedding_diversity"),
        "latest_train_embedding_overlap": train.get("mean_embedding_overlap"),
        "latest_train_invalid_rate": train.get("mean_invalid_rate"),
        "latest_train_vote_acc": train.get("vote_acc"),
        "latest_test_embedding_diversity": test.get("mean_embedding_diversity"),
        "latest_test_embedding_overlap": test.get("mean_embedding_overlap"),
        "latest_test_invalid_rate": test.get("mean_invalid_rate"),
        "latest_test_vote_acc": test.get("vote_acc"),
    }
    out.update(_collect_beam_update_metrics(update_logs))
    return out


def _to_float_str(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def write_markdown(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        path.write_text("# Experiment Summary\n\nNo valid runs found.\n", encoding="utf-8")
        return
    columns = PUBLIC_METRIC_COLUMNS
    lines = ["# Experiment Summary", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_to_float_str(row.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Aggregate trace-overlap beam experiment outputs.")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--runs_root", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_md", type=str, default="")
    args = parser.parse_args()

    run_dirs: List[Path] = []
    for raw in args.runs:
        p = Path(raw)
        if p.exists() and p.is_dir():
            run_dirs.append(p)
    if args.runs_root:
        root = Path(args.runs_root)
        if root.exists():
            run_dirs.extend([p for p in sorted(root.iterdir()) if p.is_dir() and (p / "run_meta.json").exists()])
    seen = set()
    dedup = []
    for p in run_dirs:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    rows = [analyze_run(p) for p in dedup]
    rows.sort(key=lambda r: (str(r.get("setting", "")), str(r.get("seed", ""))))

    out_csv = Path(args.out_csv) if args.out_csv else (Path(args.runs_root) / "experiment_metrics.csv" if args.runs_root else Path("experiment_metrics.csv"))
    out_md = Path(args.out_md) if args.out_md else (Path(args.runs_root) / "experiment_metrics.md" if args.runs_root else Path("experiment_metrics.md"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = PUBLIC_METRIC_COLUMNS + sorted({k for r in rows for k in r.keys() if k not in PUBLIC_METRIC_COLUMNS})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_markdown(rows, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
