#!/usr/bin/env python
"""Compare P7 trace text metrics against GPT blind method-diversity scores.

This script is deliberately offline: it reuses existing P7 GPT-5.5
analysis rows and does not call any model API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prove_experiment_utils import bootstrap_mean_ci, safe_float, safe_mean, spearman_corr, write_csv  # noqa: E402


METRICS = [
    ("trace_embedding_div", "trace_embedding_cosine_diversity"),
    ("trace_token_div", "trace_token_cosine_diversity"),
    ("family_div", "team_family_diversity"),
    ("major_div", "team_major_family_diversity"),
]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _is_valid(row: Dict[str, Any]) -> bool:
    return safe_float(row.get("gpt_method_diversity_score")) > 0


def _top_bottom(rows: List[Dict[str, Any]], metric_key: str, top_k: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda r: (
            safe_float(r.get(metric_key)),
            str(r.get("blinded_id", "")),
        ),
    )
    low = ordered[:top_k]
    high = list(reversed(ordered[-top_k:]))
    return high, low


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = [(safe_float(s), int(l)) for s, l in zip(scores, labels)]
    positives = [p for p in pairs if p[1] == 1]
    negatives = [p for p in pairs if p[1] == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def _score_distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        score = str(int(round(safe_float(row.get("gpt_method_diversity_score")))))
        counts[score] = counts.get(score, 0) + 1
    return counts


def _metric_summary(rows: List[Dict[str, Any]], label: str, key: str, top_k: int, bootstrap_iterations: int, seed: int) -> Dict[str, Any]:
    scores = [safe_float(r.get("gpt_method_diversity_score")) for r in rows]
    values = [safe_float(r.get(key)) for r in rows]
    high, low = _top_bottom(rows, key, top_k)
    high_scores = [safe_float(r.get("gpt_method_diversity_score")) for r in high]
    low_scores = [safe_float(r.get("gpt_method_diversity_score")) for r in low]
    deltas = [h - l for h, l in zip(high_scores, low_scores)]
    labels_ge2 = [1 if s >= 2.0 else 0 for s in scores]
    labels_ge3 = [1 if s >= 3.0 else 0 for s in scores]
    high_ids = {str(r.get("blinded_id", "")) for r in high}
    low_ids = {str(r.get("blinded_id", "")) for r in low}
    return {
        "metric": label,
        "metric_key": key,
        "n": len(rows),
        "top_k": top_k,
        "spearman_vs_gpt_score": spearman_corr(values, scores),
        "auc_predict_gpt_score_ge_2": _auc(values, labels_ge2),
        "auc_predict_gpt_score_ge_3": _auc(values, labels_ge3),
        "mean_metric": safe_mean(values),
        "mean_high_metric": safe_mean([r.get(key) for r in high]),
        "mean_low_metric": safe_mean([r.get(key) for r in low]),
        "mean_gpt_high_metric": safe_mean(high_scores),
        "mean_gpt_low_metric": safe_mean(low_scores),
        "high_minus_low_gpt_score_ci": bootstrap_mean_ci(deltas, iterations=bootstrap_iterations, seed=seed),
        "high_gpt_ge_2_rate": safe_mean([1 if safe_float(r.get("gpt_method_diversity_score")) >= 2.0 else 0 for r in high]),
        "low_gpt_ge_2_rate": safe_mean([1 if safe_float(r.get("gpt_method_diversity_score")) >= 2.0 else 0 for r in low]),
        "high_score_distribution": _score_distribution(high),
        "low_score_distribution": _score_distribution(low),
        "high_ids": sorted(high_ids),
        "low_ids": sorted(low_ids),
    }


def _bucket_rows(rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(r.get("blinded_id", "")): r for r in rows}
    out: List[Dict[str, Any]] = []
    for summary in summaries:
        metric = str(summary["metric"])
        for side in ["high", "low"]:
            ids = summary[f"{side}_ids"]
            for rank, blinded_id in enumerate(ids, start=1):
                row = by_id.get(blinded_id, {})
                key = str(summary["metric_key"])
                out.append(
                    {
                        "metric": metric,
                        "side": side,
                        "rank_by_id": rank,
                        "blinded_id": blinded_id,
                        "original_bucket": row.get("bucket", ""),
                        "run_name": row.get("run_name", ""),
                        "model": row.get("model", ""),
                        "question_hash": row.get("question_hash", ""),
                        "metric_value": row.get(key, ""),
                        "trace_embedding_div": row.get("trace_embedding_cosine_diversity", ""),
                        "trace_token_div": row.get("trace_token_cosine_diversity", ""),
                        "family_div": row.get("team_family_diversity", ""),
                        "major_div": row.get("team_major_family_diversity", ""),
                        "gpt_method_diversity_score": row.get("gpt_method_diversity_score", ""),
                        "gpt_distinct_methods_count": row.get("gpt_distinct_methods_count", ""),
                    }
                )
    return out


def _overlap_summary(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for left in summaries:
        for right in summaries:
            if left["metric"] >= right["metric"]:
                continue
            for side in ["high", "low"]:
                left_ids = set(left[f"{side}_ids"])
                right_ids = set(right[f"{side}_ids"])
                union = left_ids | right_ids
                out.append(
                    {
                        "left_metric": left["metric"],
                        "right_metric": right["metric"],
                        "side": side,
                        "left_n": len(left_ids),
                        "right_n": len(right_ids),
                        "overlap_n": len(left_ids & right_ids),
                        "jaccard": (len(left_ids & right_ids) / len(union)) if union else 0.0,
                    }
                )
    return out


def _write_md(out_path: Path, rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]], overlaps: List[Dict[str, Any]], top_k: int) -> None:
    ranked = sorted(
        summaries,
        key=lambda s: (
            -safe_float(s["spearman_vs_gpt_score"].get("rho") if isinstance(s.get("spearman_vs_gpt_score"), dict) else 0.0),
            -safe_float(s.get("auc_predict_gpt_score_ge_2")),
            -safe_float(s["high_minus_low_gpt_score_ci"].get("mean") if isinstance(s.get("high_minus_low_gpt_score_ci"), dict) else 0.0),
        ),
    )
    lines = [
        "# P7 Text Metric Alignment",
        "",
        f"- input_rows: {len(rows)}",
        f"- top_bottom_k: {top_k}",
        "- gpt_positive_threshold: score >= 2",
        "",
        "## Metric Summary",
        "",
        "| metric | Spearman rho | AUC score>=2 | AUC score>=3 | high mean metric | low mean metric | high GPT | low GPT | delta GPT | 95% CI | high score>=2 | low score>=2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ranked:
        corr = item.get("spearman_vs_gpt_score", {})
        ci = item.get("high_minus_low_gpt_score_ci", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["metric"]),
                    _fmt(corr.get("rho") if isinstance(corr, dict) else 0.0),
                    _fmt(item.get("auc_predict_gpt_score_ge_2")),
                    _fmt(item.get("auc_predict_gpt_score_ge_3")),
                    _fmt(item.get("mean_high_metric")),
                    _fmt(item.get("mean_low_metric")),
                    _fmt(item.get("mean_gpt_high_metric")),
                    _fmt(item.get("mean_gpt_low_metric")),
                    _fmt(ci.get("mean") if isinstance(ci, dict) else 0.0),
                    f"[{_fmt(ci.get('ci_low') if isinstance(ci, dict) else 0.0)}, {_fmt(ci.get('ci_high') if isinstance(ci, dict) else 0.0)}]",
                    _fmt(item.get("high_gpt_ge_2_rate")),
                    _fmt(item.get("low_gpt_ge_2_rate")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Embedding vs Token Bucket Overlap", ""])
    wanted = [
        r
        for r in overlaps
        if {r["left_metric"], r["right_metric"]} == {"trace_embedding_div", "trace_token_div"}
    ]
    lines.extend(["| side | overlap | Jaccard |", "|---|---:|---:|"])
    for row in wanted:
        lines.append(f"| {row['side']} | {row['overlap_n']}/{top_k} | {_fmt(row['jaccard'])} |")

    best = ranked[0] if ranked else {}
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"On this fixed P7 sample, the strongest single metric by Spearman is `{best.get('metric', 'NA')}`. "
            "Because these 120 rows were originally sampled with embedding-diversity extremes, the result should be read as an in-sample diagnostic, not a fresh sampling experiment.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline P7 comparison of trace embedding/token diversity alignment with GPT scores.")
    parser.add_argument("--analysis_rows", default="prove_experiments/p7_gpt55_blind_embedding_extreme/p7_gpt55_analysis_rows.csv")
    parser.add_argument("--out_dir", default="prove_experiments/p7_gpt55_blind_embedding_extreme")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [r for r in _read_csv(Path(args.analysis_rows)) if _is_valid(r)]
    if not rows:
        raise ValueError(f"No parsed GPT rows found in {args.analysis_rows}")
    top_k = max(1, min(int(args.top_k), len(rows) // 2))
    summaries = [
        _metric_summary(rows, label, key, top_k=top_k, bootstrap_iterations=args.bootstrap_iterations, seed=args.seed)
        for label, key in METRICS
    ]
    overlaps = _overlap_summary(summaries)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_rows": len(rows),
        "top_bottom_k": top_k,
        "metrics": summaries,
        "overlaps": overlaps,
    }
    (out_dir / "p7_text_metric_alignment.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        [
            {
                "metric": s["metric"],
                "n": s["n"],
                "top_k": s["top_k"],
                "spearman_rho": s["spearman_vs_gpt_score"].get("rho"),
                "spearman_n": s["spearman_vs_gpt_score"].get("n"),
                "auc_predict_gpt_score_ge_2": s["auc_predict_gpt_score_ge_2"],
                "auc_predict_gpt_score_ge_3": s["auc_predict_gpt_score_ge_3"],
                "mean_metric": s["mean_metric"],
                "mean_high_metric": s["mean_high_metric"],
                "mean_low_metric": s["mean_low_metric"],
                "mean_gpt_high_metric": s["mean_gpt_high_metric"],
                "mean_gpt_low_metric": s["mean_gpt_low_metric"],
                "delta_gpt_mean": s["high_minus_low_gpt_score_ci"].get("mean"),
                "delta_gpt_ci_low": s["high_minus_low_gpt_score_ci"].get("ci_low"),
                "delta_gpt_ci_high": s["high_minus_low_gpt_score_ci"].get("ci_high"),
                "high_gpt_ge_2_rate": s["high_gpt_ge_2_rate"],
                "low_gpt_ge_2_rate": s["low_gpt_ge_2_rate"],
                "high_score_distribution": json.dumps(s["high_score_distribution"], ensure_ascii=False, sort_keys=True),
                "low_score_distribution": json.dumps(s["low_score_distribution"], ensure_ascii=False, sort_keys=True),
            }
            for s in summaries
        ],
        out_dir / "p7_text_metric_alignment_summary.csv",
    )
    write_csv(_bucket_rows(rows, summaries), out_dir / "p7_text_metric_alignment_buckets.csv")
    write_csv(overlaps, out_dir / "p7_text_metric_alignment_overlaps.csv")
    _write_md(out_dir / "p7_text_metric_alignment_summary.md", rows, summaries, overlaps, top_k)
    print(f"Wrote {out_dir / 'p7_text_metric_alignment_summary.md'}")


if __name__ == "__main__":
    main()
