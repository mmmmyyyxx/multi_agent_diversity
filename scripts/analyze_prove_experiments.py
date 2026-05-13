import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.utils import infer_strategy_family_major


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs]
    return float(statistics.mean(vals)) if vals else 0.0


def _find_pred_file(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("test*_predictions.jsonl"))
    if not files:
        return None
    final_files = [p for p in files if "final" in p.name]
    return final_files[-1] if final_files else files[-1]


def _load_probe_targets(run_dir: Path) -> Dict[int, List[str]]:
    probe = _read_json(run_dir / "probe_prompts.json")
    targets: Dict[int, List[str]] = {}
    if not isinstance(probe, dict):
        return targets
    for agent in probe.get("agents", []):
        if not isinstance(agent, dict):
            continue
        aid = int(agent.get("agent_id", len(targets)))
        target = agent.get("target_family", [])
        if isinstance(target, str):
            target = [target]
        targets[aid] = [str(x) for x in target] if isinstance(target, list) else []
    return targets


def _target_hit(primary: str, secondary: str, targets: List[str]) -> Tuple[int, int]:
    if not targets:
        return 0, 0
    target_set = {str(x) for x in targets}
    exact = int(primary in target_set or secondary in target_set)
    target_majors = {infer_strategy_family_major(x) for x in target_set}
    major = infer_strategy_family_major(primary)
    same_major = int(bool(target_majors) and major in target_majors)
    return exact, max(exact, same_major)


def _prediction_metrics(preds: List[Dict[str, Any]], targets: Dict[int, List[str]]) -> Dict[str, Any]:
    if not preds:
        return {
            "eval_size": 0,
            "mean_family_diversity": 0.0,
            "mean_family_homogeneity_rate": 0.0,
            "all_same_pair_rate": 0.0,
            "disagreement_rate": 0.0,
            "vote_acc": 0.0,
            "target_exact_hit_rate": None,
            "target_same_major_hit_rate": None,
        }
    exact_hits = []
    major_hits = []
    for rec in preds:
        primary = rec.get("primary_family_labels", [])
        secondary = rec.get("secondary_family_labels", primary)
        if not isinstance(primary, list):
            primary = []
        if not isinstance(secondary, list):
            secondary = primary
        for aid, target in targets.items():
            p = str(primary[aid]) if aid < len(primary) else ""
            s = str(secondary[aid]) if aid < len(secondary) else p
            exact, same_major = _target_hit(p, s, target)
            exact_hits.append(exact)
            major_hits.append(same_major)
    disagreements = []
    for rec in preds:
        answers = rec.get("answers", [])
        if isinstance(answers, list) and answers:
            disagreements.append(int(len(set(str(x) for x in answers)) > 1))
    return {
        "eval_size": len(preds),
        "mean_family_diversity": _safe_mean(float(r.get("team_family_diversity", 0.0) or 0.0) for r in preds),
        "mean_family_homogeneity_rate": _safe_mean(float(r.get("team_family_homogeneity_rate", 0.0) or 0.0) for r in preds),
        "all_same_pair_rate": _safe_mean(int(bool(r.get("all_same_pair", False))) for r in preds),
        "disagreement_rate": _safe_mean(disagreements),
        "vote_acc": _safe_mean(float(r.get("vote_correct", 0.0) or 0.0) for r in preds),
        "target_exact_hit_rate": _safe_mean(exact_hits) if exact_hits else None,
        "target_same_major_hit_rate": _safe_mean(major_hits) if major_hits else None,
    }


def _latest_history_metrics(run_dir: Path) -> Dict[str, Any]:
    hist = _read_json(run_dir / "history.json")
    if not isinstance(hist, list) or not hist:
        return {}
    latest = hist[-1] if isinstance(hist[-1], dict) else {}
    out = {}
    for split in ["train", "val", "test"]:
        block = latest.get(split, {}) if isinstance(latest.get(split, {}), dict) else {}
        for k, v in block.items():
            out[f"latest_{split}_{k}"] = v
    out["history_last_epoch"] = latest.get("epoch", "")
    out["selected_epoch"] = latest.get("selected_epoch", "")
    out["early_stopped"] = latest.get("early_stopped", "")
    return out


def _update_metrics(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "train_step_logs.jsonl")
    if not rows:
        return {
            "train_steps": 0,
            "update_applied_rate": None,
            "mean_updated_agents": None,
            "candidate_family_shift_rate": None,
            "candidate_rho_reduction": None,
            "candidate_invalid_delta": None,
        }
    applied = []
    updated_counts = []
    family_shift = []
    rho_reduction = []
    invalid_delta = []
    for row in rows:
        upd = row.get("update", {}) if isinstance(row.get("update", {}), dict) else {}
        updated = upd.get("updated_agent_ids", [])
        if not isinstance(updated, list):
            updated = []
        applied.append(int(len(updated) > 0))
        updated_counts.append(len(updated))
        for record in upd.get("candidate_behavior_records", []) if isinstance(upd.get("candidate_behavior_records", []), list) else []:
            if not isinstance(record, dict):
                continue
            if record.get("family_shift_rate") is not None:
                family_shift.append(float(record.get("family_shift_rate", 0.0) or 0.0))
            if record.get("rho_reduction") is not None:
                rho_reduction.append(float(record.get("rho_reduction", 0.0) or 0.0))
            if record.get("invalid_delta") is not None:
                invalid_delta.append(float(record.get("invalid_delta", 0.0) or 0.0))
        if upd.get("family_shift_rate") is not None:
            family_shift.append(float(upd.get("family_shift_rate", 0.0) or 0.0))
        if upd.get("rho_reduction") is not None:
            rho_reduction.append(float(upd.get("rho_reduction", 0.0) or 0.0))
        if upd.get("invalid_delta") is not None:
            invalid_delta.append(float(upd.get("invalid_delta", 0.0) or 0.0))
    return {
        "train_steps": len(rows),
        "update_applied_rate": _safe_mean(applied),
        "mean_updated_agents": _safe_mean(updated_counts),
        "candidate_family_shift_rate": _safe_mean(family_shift) if family_shift else None,
        "candidate_rho_reduction": _safe_mean(rho_reduction) if rho_reduction else None,
        "candidate_invalid_delta": _safe_mean(invalid_delta) if invalid_delta else None,
    }


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    meta = _read_json(run_dir / "run_meta.json") or {}
    cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    pred_file = _find_pred_file(run_dir)
    preds = _read_jsonl(pred_file) if pred_file else []
    targets = _load_probe_targets(run_dir)
    probe = meta.get("probe", {}) if isinstance(meta.get("probe", {}), dict) else {}
    out = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "probe_name": probe.get("probe_name", ""),
        "has_probe_targets": int(bool(targets)),
        "model": cfg.get("model", ""),
        "critic_model": cfg.get("critic_model", ""),
        "seed": cfg.get("seed", ""),
        "init_mode": meta.get("init_mode", cfg.get("init_mode", "")),
        "baseline_only": int(bool(cfg.get("baseline_only", False))),
        "lambda_diversity": cfg.get("lambda_diversity", ""),
        "lambda_homogeneity": cfg.get("lambda_homogeneity", ""),
        "same_major_family_weight": cfg.get("same_major_family_weight", ""),
        "prediction_file": str(pred_file) if pred_file else "",
    }
    out.update(_prediction_metrics(preds, targets))
    out.update(_latest_history_metrics(run_dir))
    out.update(_update_metrics(run_dir))
    return out


def _write_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["run_name"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if v is None:
        return ""
    return str(v)


def _write_md(rows: List[Dict[str, Any]], path: Path):
    columns = [
        "run_name",
        "probe_name",
        "model",
        "eval_size",
        "mean_family_diversity",
        "mean_family_homogeneity_rate",
        "all_same_pair_rate",
        "target_exact_hit_rate",
        "target_same_major_hit_rate",
        "vote_acc",
        "lambda_diversity",
        "lambda_homogeneity",
        "same_major_family_weight",
        "update_applied_rate",
        "candidate_family_shift_rate",
    ]
    lines = ["# Prove Experiment Summary", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in columns) + " |")

    by_probe = defaultdict(list)
    for row in rows:
        by_probe[str(row.get("probe_name", "") or row.get("run_name", ""))].append(row)
    lines.extend(["", "## 自动对照提示", ""])
    names = {row.get("probe_name") or row.get("run_name"): row for row in rows}
    same = [r for r in rows if "same" in str(r.get("probe_name", r.get("run_name", ""))).lower()]
    mixed = [r for r in rows if "mixed" in str(r.get("probe_name", r.get("run_name", ""))).lower() or "bank" in str(r.get("run_name", "")).lower()]
    if same and mixed:
        same_div = _safe_mean(r.get("mean_family_diversity", 0.0) for r in same)
        mixed_div = _safe_mean(r.get("mean_family_diversity", 0.0) for r in mixed)
        lines.append(f"- mixed/same diversity delta: {mixed_div - same_div:.4f}；应为正，才支持显式策略干预有效。")
    training = [r for r in rows if not int(r.get("baseline_only", 0) or 0)]
    if training:
        no_div = [r for r in training if float(r.get("lambda_diversity", 0.0) or 0.0) == 0.0]
        div = [r for r in training if float(r.get("lambda_diversity", 0.0) or 0.0) > 0.0]
        if no_div and div:
            no_div_val = _safe_mean(float(r.get("latest_val_mean_family_diversity", r.get("mean_family_diversity", 0.0)) or 0.0) for r in no_div)
            best_div = max(float(r.get("latest_val_mean_family_diversity", r.get("mean_family_diversity", 0.0)) or 0.0) for r in div)
            lines.append(f"- best nonzero reward val diversity minus no_div: {best_div - no_div_val:.4f}；应为正，才支持 reward 可优化。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze prove experiment runs.")
    parser.add_argument("--runs_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_md", type=str, default="")
    args = parser.parse_args()
    root = Path(args.runs_root)
    run_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "run_meta.json").exists()] if root.exists() else []
    rows = [analyze_run(p) for p in run_dirs]
    out_csv = Path(args.out_csv) if args.out_csv else root / "prove_summary.csv"
    out_md = Path(args.out_md) if args.out_md else root / "prove_summary.md"
    _write_csv(rows, out_csv)
    _write_md(rows, out_md)
    print(f"Analyzed runs: {len(rows)}")
    print(f"CSV: {out_csv}")
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()
