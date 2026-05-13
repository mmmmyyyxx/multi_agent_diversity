import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from multi_dataset_diverse_rl.utils import extract_question_answer  # noqa: E402
from prove_experiment_utils import (  # noqa: E402
    bootstrap_mean_ci,
    find_prediction_file,
    infer_probe_kind,
    prompt_hash,
    read_json,
    read_jsonl,
    safe_float,
    safe_mean,
    write_csv,
)


def _normalize_spaces(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", str(text or "").strip())


def _question_subject_map(path: str, limit: int = -1) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path:
        return out
    rows = read_jsonl(Path(path))
    if limit > 0:
        rows = rows[:limit]
    for row in rows:
        try:
            q, _ = extract_question_answer(row)
        except Exception:
            continue
        qh = prompt_hash(_normalize_spaces(q))
        subject = str(row.get("subject", row.get("category", row.get("task", "")))).strip() or "unknown"
        out[qh] = subject
    return out


def _load_run_rows(run_dir: Path, subject_by_hash: Dict[str, str], dataset_name: str) -> List[Dict[str, Any]]:
    meta = read_json(run_dir / "run_meta.json") or {}
    cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    probe = meta.get("probe", {}) if isinstance(meta.get("probe", {}), dict) else {}
    pred_file = find_prediction_file(run_dir)
    if not pred_file:
        return []
    rows = []
    for rec in read_jsonl(pred_file):
        qh = str(rec.get("question_hash", ""))
        if not qh:
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "probe_name": probe.get("probe_name", ""),
                "probe_kind": infer_probe_kind(probe.get("probe_name", ""), run_dir.name),
                "model": cfg.get("model", ""),
                "seed": cfg.get("seed", ""),
                "question_hash": qh,
                "subject": subject_by_hash.get(qh, "unknown"),
                "team_family_diversity": safe_float(rec.get("team_family_diversity")),
                "team_family_homogeneity_rate": safe_float(rec.get("team_family_homogeneity_rate")),
                "team_major_family_diversity": safe_float(rec.get("team_major_family_diversity")),
                "target_same_major_hit_rate_available": int(bool(probe)),
            }
        )
    return rows


def _summarize_subjects(rows: List[Dict[str, Any]], bootstrap_iterations: int, seed: int) -> List[Dict[str, Any]]:
    by_key: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["dataset"], row["subject"], row["model"])].append(row)

    out = []
    for (dataset, subject, model), vals in sorted(by_key.items()):
        same = [r for r in vals if r.get("probe_kind") == "same"]
        mixed = [r for r in vals if r.get("probe_kind") == "mixed"]
        same_by_q = {str(r["question_hash"]): r for r in same}
        mixed_by_q = {str(r["question_hash"]): r for r in mixed}
        qhs = sorted(set(same_by_q) & set(mixed_by_q))
        deltas = [safe_float(mixed_by_q[q]["team_family_diversity"]) - safe_float(same_by_q[q]["team_family_diversity"]) for q in qhs]
        homo_deltas = [safe_float(mixed_by_q[q]["team_family_homogeneity_rate"]) - safe_float(same_by_q[q]["team_family_homogeneity_rate"]) for q in qhs]
        ci = bootstrap_mean_ci(deltas, iterations=bootstrap_iterations, seed=seed)
        out.append(
            {
                "dataset": dataset,
                "subject": subject,
                "model": model,
                "same_count": len(same),
                "mixed_count": len(mixed),
                "paired_question_count": len(qhs),
                "same_mean_family_diversity": safe_mean([r["team_family_diversity"] for r in same]),
                "mixed_mean_family_diversity": safe_mean([r["team_family_diversity"] for r in mixed]),
                "intervention_effect_mean": ci["mean"],
                "intervention_effect_ci_low": ci["ci_low"],
                "intervention_effect_ci_high": ci["ci_high"],
                "homogeneity_delta_mean": safe_mean(homo_deltas),
            }
        )
    return out


def _summarize_datasets(subject_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    out = []
    for dataset, rows in sorted(by_dataset.items()):
        paired = [r for r in rows if int(r.get("paired_question_count", 0) or 0) > 0]
        effects = [safe_float(r.get("intervention_effect_mean")) for r in paired]
        out.append(
            {
                "dataset": dataset,
                "subject_count": len(rows),
                "paired_subject_count": len(paired),
                "mean_subject_intervention_effect": safe_mean(effects),
                "positive_subject_rate": safe_mean([int(x > 0.0) for x in effects]),
                "low_reachable_subjects": ";".join(str(r.get("subject", "")) for r in paired if safe_float(r.get("intervention_effect_mean")) <= 0.0),
            }
        )
    return out


def _write_md(subject_rows: List[Dict[str, Any]], dataset_rows: List[Dict[str, Any]], out_md: Path):
    lines = ["# P8 Task Dependence Check", "", "## Dataset Summary", ""]
    cols = ["dataset", "subject_count", "paired_subject_count", "mean_subject_intervention_effect", "positive_subject_rate", "low_reachable_subjects"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in dataset_rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) if not isinstance(row.get(c), float) else f"{row.get(c):.4f}" for c in cols) + " |")
    lines.extend(["", "## Subject Summary", ""])
    cols2 = ["dataset", "subject", "model", "paired_question_count", "intervention_effect_mean", "intervention_effect_ci_low", "intervention_effect_ci_high"]
    lines.append("| " + " | ".join(cols2) + " |")
    lines.append("|" + "|".join(["---"] * len(cols2)) + "|")
    for row in subject_rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) if not isinstance(row.get(c), float) else f"{row.get(c):.4f}" for c in cols2) + " |")
    lines.extend(
        [
            "",
            "判读：不同 subject 的 intervention_effect 可不同；受限 subject 接近 0 不必然反驳指标，多方法数据集或开放 subject 应显示更高效应。",
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="P8: subject/dataset dependence analysis for reachable strategy diversity.")
    parser.add_argument("--runs_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--dataset_name", type=str, default="mmlu")
    parser.add_argument("--test_path", type=str, default="mmlu_test_200.jsonl")
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--out_dir", type=str, default="prove_experiments/p8_task_dependence")
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subject_by_hash = _question_subject_map(args.test_path, args.test_size)
    root = Path(args.runs_root)
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []:
        rows.extend(_load_run_rows(run_dir, subject_by_hash, args.dataset_name))

    subject_rows = _summarize_subjects(rows, args.bootstrap_iterations, args.seed)
    dataset_rows = _summarize_datasets(subject_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "p8_question_rows.csv")
    write_csv(subject_rows, out_dir / "p8_subject_summary.csv")
    write_csv(dataset_rows, out_dir / "p8_dataset_summary.csv")
    (out_dir / "p8_subject_summary.json").write_text(json.dumps(subject_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(subject_rows, dataset_rows, out_dir / "p8_task_dependence_summary.md")
    print(f"P8 analyzed question rows: {len(rows)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()

