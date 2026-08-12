from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


G0 = "g0_fixed_target_generic"
M20 = "m20_current_v15"
GEOMETRIES = tuple("ABCDEF")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sum(rows: Iterable[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key, 0) or 0) for row in rows)


def _cell_metrics(cell: dict[str, Any]) -> dict[str, Any]:
    candidates = list(cell.get("candidates", []))
    funnel = cell.get("funnel", {})
    geometries = {
        letter: sum(row.get("candidate_geometry") == letter for row in candidates)
        for letter in GEOMETRIES
    }
    cost = cell.get("cost", {})
    return {
        "case_id": cell["case_id"],
        "seed": int(cell["seed"]),
        "variant": cell["variant"],
        "requested_candidates": int(cell["requested_candidate_count"]),
        "valid_candidates": int(funnel.get("valid_candidate_count", 0)),
        "evaluated_candidates": len(candidates),
        "feasible_candidates": sum(
            bool(row.get("constraint", {}).get("passed")) for row in candidates
        ),
        **{f"{letter}_count": count for letter, count in geometries.items()},
        "F_count": geometries["F"],
        "target_regression_count": sum(
            int(row.get("target_gain", 0)) < 0 for row in candidates
        ),
        "target_gain": _sum(candidates, "target_gain"),
        "vote_gain_count": _sum(candidates, "vote_gain_count"),
        "vote_loss_count": _sum(candidates, "vote_loss_count"),
        "vote_net_gain": _sum(candidates, "vote_net_gain"),
        "responsibility_repair_candidate_count": sum(
            int(row.get("responsibility_residual_gain_count", 0)) > 0
            for row in candidates
        ),
        "responsibility_repair_rate": sum(
            int(row.get("responsibility_residual_gain_count", 0)) > 0
            for row in candidates
        ) / max(1, len(candidates)),
        "responsibility_residual_gain_count": _sum(
            candidates, "responsibility_residual_gain_count"
        ),
        "responsibility_residual_loss_count": _sum(
            candidates, "responsibility_residual_loss_count"
        ),
        "coverage_responsibility_gain": _sum(
            candidates, "coverage_responsibility_gain"
        ),
        "conversion_responsibility_gain": _sum(
            candidates, "conversion_responsibility_gain"
        ),
        "critic_calls": int(funnel.get("critic_calls", 0)),
        "critic_semantic_rejections": int(
            funnel.get("critic_semantic_rejections", 0)
        ),
        "critic_exhausted": int(
            funnel.get("terminal_failure_role") == "critic"
        ),
        "student_reached": int(int(funnel.get("student_calls", 0)) > 0),
        "context_char_count": int(
            cell.get("generation_context", {}).get("context_char_count", 0)
        ),
        "context_item_count": int(
            cell.get("generation_context", {}).get("context_item_count", 0)
        ),
        "input_tokens": int(cost.get("prompt_tokens", 0)),
        "output_tokens": int(cost.get("completion_tokens", 0)),
        "total_tokens": int(cost.get("total_tokens", 0)),
    }


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    evaluated = _sum(selected, "evaluated_candidates")
    return {
        "variant": variant,
        "branches": len(selected),
        "requested_candidates": _sum(selected, "requested_candidates"),
        "valid_candidates": _sum(selected, "valid_candidates"),
        "evaluated_candidates": evaluated,
        "feasible_candidates": _sum(selected, "feasible_candidates"),
        "feasible_rate": _sum(selected, "feasible_candidates") / max(1, evaluated),
        **{
            f"{letter}_count": _sum(selected, f"{letter}_count")
            for letter in GEOMETRIES
        },
        "F_fraction": _sum(selected, "F_count") / max(1, evaluated),
        "target_regression_count": _sum(selected, "target_regression_count"),
        "target_regression_rate": (
            _sum(selected, "target_regression_count") / max(1, evaluated)
        ),
        "target_gain": _sum(selected, "target_gain"),
        "vote_gain_count": _sum(selected, "vote_gain_count"),
        "vote_loss_count": _sum(selected, "vote_loss_count"),
        "vote_net_gain": _sum(selected, "vote_net_gain"),
        "responsibility_repair_candidate_count": _sum(
            selected, "responsibility_repair_candidate_count"
        ),
        "responsibility_repair_rate": (
            _sum(selected, "responsibility_repair_candidate_count")
            / max(1, evaluated)
        ),
        "responsibility_residual_gain_count": _sum(
            selected, "responsibility_residual_gain_count"
        ),
        "responsibility_residual_loss_count": _sum(
            selected, "responsibility_residual_loss_count"
        ),
        "coverage_responsibility_gain": _sum(
            selected, "coverage_responsibility_gain"
        ),
        "conversion_responsibility_gain": _sum(
            selected, "conversion_responsibility_gain"
        ),
        "critic_calls": _sum(selected, "critic_calls"),
        "critic_semantic_rejections": _sum(
            selected, "critic_semantic_rejections"
        ),
        "critic_exhausted_branches": _sum(selected, "critic_exhausted"),
        "critic_exhausted_rate": (
            _sum(selected, "critic_exhausted") / max(1, len(selected))
        ),
        "student_reached_branches": _sum(selected, "student_reached"),
        "student_reached_rate": (
            _sum(selected, "student_reached") / max(1, len(selected))
        ),
        "mean_context_char_count": mean(
            row["context_char_count"] for row in selected
        ),
        "mean_context_item_count": mean(
            row["context_item_count"] for row in selected
        ),
        "input_tokens": _sum(selected, "input_tokens"),
        "output_tokens": _sum(selected, "output_tokens"),
        "total_tokens": _sum(selected, "total_tokens"),
    }


def _wtl(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    lower_is_better: bool = False,
) -> dict[str, int]:
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["variant"]] = int(row[metric])
    wins = ties = losses = 0
    for arms in by_case.values():
        difference = arms[M20] - arms[G0]
        if lower_is_better:
            difference *= -1
        wins += int(difference > 0)
        ties += int(difference == 0)
        losses += int(difference < 0)
    return {"wins": wins, "ties": ties, "losses": losses}


def _classify(
    rows: list[dict[str, Any]], aggregates: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    repair_wtl = _wtl(rows, "responsibility_residual_gain_count")
    seed_differences = {}
    for seed in (48, 49, 50, 51):
        seed_rows = [row for row in rows if row["seed"] == seed]
        values = {
            variant: _sum(
                [row for row in seed_rows if row["variant"] == variant],
                "responsibility_residual_gain_count",
            )
            for variant in (G0, M20)
        }
        seed_differences[str(seed)] = values[M20] - values[G0]
    g0, m20 = aggregates[G0], aggregates[M20]
    targeting_better = (
        m20["responsibility_residual_gain_count"]
        > g0["responsibility_residual_gain_count"]
        and repair_wtl["wins"] > repair_wtl["losses"]
        and sum(value > 0 for value in seed_differences.values()) >= 2
    )
    quality_noninferior = (
        m20["feasible_candidates"] >= g0["feasible_candidates"]
        and m20["F_count"] <= g0["F_count"]
        and m20["target_regression_count"] <= g0["target_regression_count"]
    )
    negative = (
        m20["responsibility_residual_gain_count"]
        < g0["responsibility_residual_gain_count"]
        and repair_wtl["losses"] > repair_wtl["wins"]
        and sum(value < 0 for value in seed_differences.values()) >= 2
        and (
            m20["feasible_candidates"] < g0["feasible_candidates"]
            or m20["F_count"] > g0["F_count"]
            or m20["target_regression_count"]
            > g0["target_regression_count"]
        )
    )
    value = (
        "SUPPORTED" if targeting_better and quality_noninferior
        else "USEFUL_SIGNAL_WITH_COLLATERAL" if targeting_better
        else "NEGATIVE" if negative
        else "NOT_ESTABLISHED"
    )
    return value, {
        "targeting_better": targeting_better,
        "quality_noninferior": quality_noninferior,
        "negative": negative,
        "repair_wtl": repair_wtl,
        "responsibility_repair_seed_differences_m20_minus_g0": seed_differences,
    }


def analyze(summary: dict[str, Any]) -> dict[str, Any]:
    rows = [_cell_metrics(cell) for cell in summary.get("cells", [])]
    aggregates = {
        variant: _aggregate(rows, variant) for variant in (G0, M20)
    }
    value, classifier = _classify(rows, aggregates)
    return {
        "cell_rows": rows,
        "variant_rows": [aggregates[G0], aggregates[M20]],
        "paired": {
            "feasible": _wtl(rows, "feasible_candidates"),
            "responsibility_repair": _wtl(
                rows, "responsibility_residual_gain_count"
            ),
            "F": _wtl(rows, "F_count", lower_is_better=True),
            "target_regression": _wtl(
                rows, "target_regression_count", lower_is_better=True
            ),
        },
        "classifier": classifier,
        "responsibility_conditioning_value": value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(
        (args.run_root / "probe_summary.json").read_text(encoding="utf-8")
    )
    result = analyze(summary)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "case_metrics.csv", result["cell_rows"])
    write_csv(out / "variant_metrics.csv", result["variant_rows"])
    seed_rows = []
    for seed in (48, 49, 50, 51):
        for variant in (G0, M20):
            aggregate = _aggregate(
                [row for row in result["cell_rows"] if row["seed"] == seed],
                variant,
            )
            seed_rows.append({"seed": seed, **aggregate})
    write_csv(out / "seed_metrics.csv", seed_rows)
    write_csv(out / "responsibility_repair_metrics.csv", [
        {
            key: row[key]
            for key in (
                "case_id", "seed", "variant",
                "responsibility_repair_candidate_count",
                "responsibility_repair_rate",
                "responsibility_residual_gain_count",
                "responsibility_residual_loss_count",
                "coverage_responsibility_gain",
                "conversion_responsibility_gain",
            )
        }
        for row in result["cell_rows"]
    ])
    write_csv(out / "pipeline_cost_metrics.csv", [
        {
            key: row[key]
            for key in (
                "case_id", "seed", "variant", "critic_calls",
                "critic_semantic_rejections", "critic_exhausted",
                "student_reached", "context_char_count", "context_item_count",
                "input_tokens", "output_tokens", "total_tokens",
            )
        }
        for row in result["cell_rows"]
    ])
    paired_rows = [
        {"comparison": f"M20_vs_G0_{name}", **values}
        for name, values in result["paired"].items()
    ]
    write_csv(out / "paired_comparisons.csv", paired_rows)
    summary_payload = {
        "analysis_version": "v16_generic_m20_analysis_v1",
        "responsibility_conditioning_value": result[
            "responsibility_conditioning_value"
        ],
        "paired": result["paired"],
        "classifier": result["classifier"],
        "variants": result["variant_rows"],
    }
    (out / "analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    repo_root = Path(__file__).resolve().parents[1]
    freeze_path = (
        repo_root
        / "runs"
        / "v16_responsibility_coherence_generic_m20_prep"
        / "source_freeze_manifest.json"
    )
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        (out / "source_freeze_sanitized.json").write_text(
            json.dumps({
                "execution_commit": freeze.get("execution_commit"),
                "method_version": freeze.get("canonical_method_version"),
                "checkpoint_version": freeze.get("checkpoint_version"),
                "source_file_count": freeze.get("source_file_count"),
                "source_tree_hash": freeze.get("working_tree_source_hash"),
                "source_freeze_status": freeze.get("source_freeze_status"),
                "method_source_changed_after_first_api_call": False,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
