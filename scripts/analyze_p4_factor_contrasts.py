#!/usr/bin/env python
"""Factor contrast analysis for P4 model identity vs prompt effects."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prove_experiment_utils import find_prediction_file, read_json, read_jsonl, safe_float


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def prompt_family(run_name: str) -> str:
    text = run_name.lower()
    if "mixed_strategy" in text:
        return "mixed_strategy"
    if "same_definition" in text:
        return "same_definition"
    if "same_elimination" in text:
        return "same_elimination"
    return "other"


def model_name(run_dir: Path) -> str:
    meta = read_json(run_dir / "run_meta.json")
    cfg = meta.get("config", {}) if isinstance(meta, dict) and isinstance(meta.get("config", {}), dict) else {}
    return str(cfg.get("model", ""))


def distribution_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    return 0.5 * sum(abs(safe_float(a.get(k)) - safe_float(b.get(k))) for k in keys)


def run_records(runs_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("P4_")):
        pred_file = find_prediction_file(run_dir)
        preds = read_jsonl(pred_file) if pred_file else []
        for rec in preds:
            qh = str(rec.get("question_hash", ""))
            if not qh:
                continue
            dist = rec.get("major_family_distribution", {})
            if not isinstance(dist, dict):
                dist = {}
            out.append(
                {
                    "run_name": run_dir.name,
                    "model": model_name(run_dir),
                    "prompt_family": prompt_family(run_dir.name),
                    "question_hash": qh,
                    "major_family_distribution": dist,
                    "team_family_diversity": safe_float(rec.get("team_family_diversity")),
                    "team_major_family_diversity": safe_float(rec.get("team_major_family_diversity")),
                    "team_family_homogeneity_rate": safe_float(rec.get("team_family_homogeneity_rate")),
                }
            )
    return out


def contrast_kind(a: dict[str, Any], b: dict[str, Any]) -> str:
    same_model = a["model"] == b["model"]
    same_prompt = a["prompt_family"] == b["prompt_family"]
    if same_model and same_prompt:
        return "same_model_same_prompt"
    if same_model and not same_prompt:
        return "same_model_different_prompt"
    if not same_model and same_prompt:
        return "different_model_same_prompt"
    return "different_model_different_prompt"


def summarize(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_question[rec["question_hash"]].append(rec)

    pairs: list[dict[str, Any]] = []
    for qh, vals in by_question.items():
        for a, b in itertools.combinations(vals, 2):
            kind = contrast_kind(a, b)
            if kind == "same_model_same_prompt":
                continue
            pairs.append(
                {
                    "question_hash": qh,
                    "contrast": kind,
                    "left_run": a["run_name"],
                    "right_run": b["run_name"],
                    "left_model": a["model"],
                    "right_model": b["model"],
                    "left_prompt_family": a["prompt_family"],
                    "right_prompt_family": b["prompt_family"],
                    "major_distribution_distance": distribution_distance(
                        a["major_family_distribution"],
                        b["major_family_distribution"],
                    ),
                }
            )

    # same_model_same_prompt is represented by within-team diversity in each run.
    within_rows = [
        {
            "contrast": "same_model_same_prompt",
            "unit": "within_team",
            "n": len(records),
            "mean_major_distribution_distance": mean([r["team_major_family_diversity"] for r in records]),
            "mean_family_diversity": mean([r["team_family_diversity"] for r in records]),
            "mean_homogeneity": mean([r["team_family_homogeneity_rate"] for r in records]),
        }
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[row["contrast"]].append(row)
    summary = list(within_rows)
    for kind in [
        "same_model_different_prompt",
        "different_model_same_prompt",
        "different_model_different_prompt",
    ]:
        vals = grouped.get(kind, [])
        summary.append(
            {
                "contrast": kind,
                "unit": "between_team_same_question",
                "n": len(vals),
                "mean_major_distribution_distance": mean([safe_float(v["major_distribution_distance"]) for v in vals]),
                "mean_family_diversity": "",
                "mean_homogeneity": "",
            }
        )
    return summary, pairs


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()}) if rows else ["id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(v) for v in row) + " |")
    return lines


def write_md(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# P4 模型身份与 Prompt 因子对比",
        "",
        "本分析把 P4 分成四种对比。`same_model_same_prompt` 用同一个 run 内的 team 主类多样性表示；其余三类用同一道题下两个 team 的 `major_family_distribution` 距离表示。",
        "",
    ]
    lines.extend(
        md_table(
            ["contrast", "unit", "n", "mean major distribution distance", "mean family diversity", "mean homogeneity"],
            [
                [
                    r["contrast"],
                    r["unit"],
                    r["n"],
                    r["mean_major_distribution_distance"],
                    r.get("mean_family_diversity", ""),
                    r.get("mean_homogeneity", ""),
                ]
                for r in summary
            ],
        )
    )
    lines.extend(
        [
            "",
            "读法：如果 `different_model_same_prompt` 明显高于 `same_model_different_prompt`，说明模型身份/输出风格是更强的策略分布来源；如果反过来，说明策略 prompt 是更强来源。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="prove_experiments/cleaned_runs")
    parser.add_argument("--out_dir", default="prove_experiments/p4_factor_contrasts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    records = run_records(Path(args.runs_root))
    summary, pairs = summarize(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary, out_dir / "p4_factor_contrast_summary.csv")
    write_csv(pairs, out_dir / "p4_factor_contrast_pairs.csv")
    (out_dir / "p4_factor_contrast_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(summary, out_dir / "p4_factor_contrast_summary.md")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
