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

from multi_dataset_diverse_rl.utils import (  # noqa: E402
    compute_strategy_family_profile_metrics,
    infer_strategy_family_major,
    strategy_family_major_categories,
)
from prove_experiment_utils import (  # noqa: E402
    bootstrap_mean_ci,
    find_prediction_file,
    read_json,
    read_jsonl,
    safe_float,
    safe_mean,
    spearman_corr,
    write_csv,
)


def _load_allowed_labels(path: str) -> List[str]:
    if path and path != "auto":
        payload = read_json(Path(path))
    else:
        payload = read_json(ROOT / "taxonomies" / "mmlu_reasoning_family_taxonomy.json")
    if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
        return [str(x) for x in payload["labels"]]
    labels: List[str] = []
    for families in strategy_family_major_categories().values():
        labels.extend(families)
    return labels


def _labels(rec: Dict[str, Any]) -> tuple[List[str], List[str]]:
    primary = rec.get("primary_family_labels", [])
    secondary = rec.get("secondary_family_labels", primary)
    if not isinstance(primary, list):
        primary = []
    if not isinstance(secondary, list):
        secondary = primary
    return [str(x) for x in primary], [str(x) for x in secondary]


def _granularity_metrics(rec: Dict[str, Any], allowed_labels: List[str], same_major_weight: float, macro_weight: float) -> Dict[str, Any]:
    primary, secondary = _labels(rec)
    majors = [infer_strategy_family_major(x) for x in primary]
    major_allowed = sorted(strategy_family_major_categories().keys())

    major_only = compute_strategy_family_profile_metrics(
        majors,
        majors,
        allowed_labels=major_allowed,
        use_dual_family=False,
        allow_fallback=True,
    )
    weighted = compute_strategy_family_profile_metrics(
        primary,
        secondary,
        allowed_labels=allowed_labels,
        use_dual_family=True,
        primary_weight=0.7,
        secondary_weight=0.3,
        same_major_weight=same_major_weight,
        macro_diversity_weight=macro_weight,
        allow_fallback=True,
    )
    strict_leaf = compute_strategy_family_profile_metrics(
        primary,
        primary,
        allowed_labels=allowed_labels,
        use_dual_family=False,
        allow_fallback=True,
    )
    return {
        "major_only_diversity": major_only.get("team_family_diversity", 0.0),
        "major_only_homogeneity": major_only.get("team_family_homogeneity_rate", 0.0),
        "weighted_tree_diversity": weighted.get("team_family_diversity", 0.0),
        "weighted_tree_homogeneity": weighted.get("team_family_homogeneity_rate", 0.0),
        "strict_leaf_diversity": strict_leaf.get("team_family_diversity", 0.0),
        "strict_leaf_homogeneity": strict_leaf.get("team_family_homogeneity_rate", 0.0),
        "weighted_minus_major_diversity": safe_float(weighted.get("team_family_diversity")) - safe_float(major_only.get("team_family_diversity")),
        "strict_minus_weighted_diversity": safe_float(strict_leaf.get("team_family_diversity")) - safe_float(weighted.get("team_family_diversity")),
    }


def _human_scores(path: str) -> Dict[str, float]:
    if not path:
        return {}
    rows = read_jsonl(Path(path)) if str(path).lower().endswith(".jsonl") else []
    if not rows:
        import csv

        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8", newline="") as f:
                rows = [dict(r) for r in csv.DictReader(f)]
    out: Dict[str, float] = {}
    for row in rows:
        qh = str(row.get("question_hash", "") or row.get("group_id", "")).strip()
        score = row.get(
            "human_method_diversity_score",
            row.get(
                "gpt_method_diversity_score",
                row.get("method_diversity_score", row.get("score")),
            ),
        )
        if qh:
            out[qh] = safe_float(score)
    return out


def _summarize(rows: List[Dict[str, Any]], human: Dict[str, float], bootstrap_iterations: int, seed: int) -> Dict[str, Any]:
    metrics = [
        "major_only_diversity",
        "weighted_tree_diversity",
        "strict_leaf_diversity",
        "major_only_homogeneity",
        "weighted_tree_homogeneity",
        "strict_leaf_homogeneity",
        "weighted_minus_major_diversity",
        "strict_minus_weighted_diversity",
    ]
    summary: Dict[str, Any] = {"question_count": len(rows)}
    for metric in metrics:
        ci = bootstrap_mean_ci([safe_float(r.get(metric)) for r in rows], iterations=bootstrap_iterations, seed=seed)
        summary[f"{metric}_mean"] = ci["mean"]
        summary[f"{metric}_ci_low"] = ci["ci_low"]
        summary[f"{metric}_ci_high"] = ci["ci_high"]

    if human:
        matched = [r for r in rows if str(r.get("question_hash", "")) in human]
        summary["human_matched_count"] = len(matched)
        for metric in ["major_only_diversity", "weighted_tree_diversity", "strict_leaf_diversity"]:
            corr = spearman_corr([r.get(metric, 0.0) for r in matched], [human[str(r.get("question_hash", ""))] for r in matched])
            summary[f"{metric}_human_spearman_rho"] = corr["rho"]
            summary[f"{metric}_human_spearman_n"] = corr["n"]
    return summary


def _write_md(summary_by_run: List[Dict[str, Any]], out_md: Path):
    cols = [
        "run_name",
        "question_count",
        "major_only_diversity_mean",
        "weighted_tree_diversity_mean",
        "strict_leaf_diversity_mean",
        "weighted_minus_major_diversity_mean",
        "strict_minus_weighted_diversity_mean",
        "weighted_tree_diversity_human_spearman_rho",
    ]
    lines = ["# P6 Taxonomy Granularity Sensitivity", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in summary_by_run:
        lines.append("| " + " | ".join(f"{safe_float(row.get(c)):.4f}" if isinstance(row.get(c), float) else str(row.get(c, "")) for c in cols) + " |")
    lines.extend(
        [
            "",
            "判读：weighted_tree 应在 major-only 与 strict leaf-only 之间；如果 human_spearman 可用，优先看 weighted_tree 是否最好或接近最好。",
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="P6: offline taxonomy granularity sensitivity analysis.")
    parser.add_argument("--runs_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--taxonomy_path", type=str, default="auto")
    parser.add_argument("--human_annotations", type=str, default="", help="Backward-compatible name for blind score CSV/JSONL.")
    parser.add_argument("--blind_annotations", type=str, default="", help="CSV/JSONL with question_hash and GPT/human method-diversity score.")
    parser.add_argument("--gpt_annotations", type=str, default="", help="Alias for --blind_annotations.")
    parser.add_argument("--same_major_family_weight", type=float, default=0.5)
    parser.add_argument("--macro_diversity_weight", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="prove_experiments/p6_taxonomy")
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    allowed = _load_allowed_labels(args.taxonomy_path)
    annotation_path = args.blind_annotations or args.gpt_annotations or args.human_annotations
    human = _human_scores(annotation_path)
    root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []:
        pred_file = find_prediction_file(run_dir)
        if not pred_file:
            continue
        preds = read_jsonl(pred_file)
        run_rows = []
        for rec in preds:
            qh = str(rec.get("question_hash", ""))
            if not qh:
                continue
            metrics = _granularity_metrics(rec, allowed, args.same_major_family_weight, args.macro_diversity_weight)
            row = {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "question_hash": qh,
                **metrics,
            }
            if qh in human:
                row["human_method_diversity_score"] = human[qh]
            run_rows.append(row)
        all_rows.extend(run_rows)
        summary = _summarize(run_rows, human, args.bootstrap_iterations, args.seed)
        summary_rows.append({"run_name": run_dir.name, "run_dir": str(run_dir), **summary})

    write_csv(all_rows, out_dir / "p6_question_granularity.csv")
    write_csv(summary_rows, out_dir / "p6_granularity_summary.csv")
    (out_dir / "p6_granularity_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(summary_rows, out_dir / "p6_granularity_summary.md")
    print(f"P6 analyzed runs: {len(summary_rows)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
