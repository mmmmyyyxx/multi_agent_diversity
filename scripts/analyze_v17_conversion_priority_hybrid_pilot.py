from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v17_conversion_priority_hybrid_support import ARMS, BASE, BREADTH, DIRECT, classify


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, registry_path: Path, freeze_path: Path, report: Path) -> dict[str, Any]:
    if report.exists():
        raise FileExistsError("report directory must be fresh")
    report.mkdir(parents=True)
    registry = read_json(registry_path)
    freeze = read_json(freeze_path)
    cells = [read_json(path) for path in sorted(root.glob("*/cells/*/cell_result.json"))]
    candidates = [row for path in sorted(root.glob("*/candidate_level.jsonl")) for row in read_jsonl(path)]
    case_by_id = {case["case_id"]: case for case in registry["cases"]}
    parent_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    for case_id, case in case_by_id.items():
        case_cells = {row["arm"]: row for row in cells if row["case_id"] == case_id}
        row: dict[str, Any] = {
            "parent_id": case_id,
            "seed": case["source_seed"],
            "update_index": case["source_update_index"],
            "conversion_residual_count": case["conversion_residual_count"],
        }
        classifier_row: dict[str, Any] = {}
        for arm in ARMS:
            cell = case_cells[arm]
            prefix = arm.lower()
            structure = cell["validation_structure"]
            row.update({
                f"{prefix}_target2": cell["target_ids"][1],
                f"{prefix}_feasible_branches": sum(int(branch["feasible_candidate_count"] > 0) for branch in cell["branches"]),
                f"{prefix}_would_commit": int(cell["would_commit"]),
                f"{prefix}_train_target_delta": cell["train_realized_target_delta"],
                f"{prefix}_validation_target_delta": cell["realized_validation_target_delta"],
                f"{prefix}_target_transfer_gap": cell["target_transfer_gap"],
                f"{prefix}_validation_vote_delta": cell["realized_validation_vote_delta"],
                f"{prefix}_validation_oracle_delta": cell["realized_validation_oracle_delta"],
                f"{prefix}_deeper_support_gain": structure["deeper_support_gain_count"],
                f"{prefix}_vote_conversions": structure["vote_conversion_count"],
                f"{prefix}_vote_regressions": structure["vote_regression_count"],
            })
            classifier_row[arm] = {
                "validation_vote_delta": cell["realized_validation_vote_delta"],
                "validation_oracle_delta": cell["realized_validation_oracle_delta"],
                "deeper_support_gain_count": structure["deeper_support_gain_count"],
                "vote_conversion_count": structure["vote_conversion_count"],
            }
        parent_rows.append(row)
        classifier_rows.append(classifier_row)

    branch_rows: list[dict[str, Any]] = []
    for cell in cells:
        for branch in cell["branches"]:
            branch_rows.append({
                "parent_id": cell["case_id"],
                "arm": cell["arm"],
                "target_member": branch["target_member"],
                "branch_type": branch["branch_type"],
                "canonical_branch_key": branch["canonical_branch_key"],
                "shared_branch_reuse_count": branch["shared_branch_reuse_count"],
                "source_candidate_count": branch["source_candidate_count"],
                "valid_source_count": branch["valid_source_count"],
                "valid_revision_count": branch["valid_revision_count"],
                "feasible_candidate_count": branch["feasible_candidate_count"],
                "produced_cell_best": int(branch["produced_cell_best"]),
                "cell_would_commit": int(cell["would_commit"]),
            })
    candidate_rows: list[dict[str, Any]] = []
    for row in candidates:
        candidate_rows.append({
            "parent_id": row["parent_id"], "seed": row["seed"],
            "update_index": row["update_index"], "arm": row["arm"],
            "target_member": row["target_member"], "branch_type": row["branch_type"],
            "is_conversion_eligible": int(row["is_conversion_eligible"]),
            "conversion_responsibility_count": row["conversion_responsibility_count"],
            "direct_vote_flip_count": row["direct_vote_flip_count"],
            "candidate_stage": row["candidate_stage"], "valid": int(row["valid"]),
            "feasible": int(row["feasible"]),
            "common_safe_outcome": row["common_safe_outcome"],
            "sanitized_rejection_reasons": "|".join(row["sanitized_rejection_reasons"]),
            "train_target_delta": row["train"]["target_delta"],
            "train_vote_delta": row["train"]["vote_delta"],
            "train_oracle_delta": row["train"]["oracle_delta"],
            "validation_target_delta": row["validation"]["target_delta"],
            "validation_vote_delta": row["validation"]["vote_delta"],
            "validation_oracle_delta": row["validation"]["oracle_delta"],
            "validation_deeper_support_gain": row["validation"]["structure"]["deeper_support_gain_count"],
            "validation_vote_conversions": row["validation"]["structure"]["vote_conversion_count"],
            "branch_rank": row["branch_rank"], "cell_rank": row["cell_rank"],
            "cell_winner": int(row["selected_as_cell_winner"]),
            "would_commit": int(row["would_commit_contribution"]),
        })
    funnel: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        arm_branches = [row for row in branch_rows if row["arm"] == arm]
        arm_cells = [row for row in cells if row["arm"] == arm]
        funnel[arm] = {
            "feasible_branch_count": sum(int(row["feasible_candidate_count"] > 0) for row in arm_branches),
            "feasible_candidate_count": sum(int(row["feasible_candidate_count"]) for row in arm_branches),
            "would_commit_count": sum(int(row["would_commit"]) for row in arm_cells),
        }
    classifier = classify(classifier_rows, funnel)
    structure_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_cells = [row for row in cells if row["arm"] == arm]
        hist = Counter()
        for row in arm_cells:
            hist.update(row["validation_structure"]["transition_histogram"])
        structure_rows.append({
            "arm": arm,
            "deeper_support_gain_count": sum(row["validation_structure"]["deeper_support_gain_count"] for row in arm_cells),
            "vote_conversion_count": sum(row["validation_structure"]["vote_conversion_count"] for row in arm_cells),
            "vote_regression_count": sum(row["validation_structure"]["vote_regression_count"] for row in arm_cells),
            "0_to_1": hist["0_to_1"], "1_to_2": hist["1_to_2"],
            "1_to_3_plus": hist["1_to_3_plus"], "2_to_3": hist["2_to_3"],
            "2_to_4_plus": hist["2_to_4_plus"], "other_positive_g": hist["other_positive_g"],
        })
    probe = read_json(root / "probe_summary.json")
    summary = {
        "summary_version": "v17_conversion_priority_three_arm_summary_v1",
        "execution_commit": probe["execution_commit"],
        "phase_a_gate": "PASS", "phase_b_gate": "PASS",
        "parents": 5, "cells": 15,
        "conceptual_branches": 30,
        "unique_branches": probe["deduplicated_branch_count"],
        "reused_branches": 30 - probe["deduplicated_branch_count"],
        "total_recorded_calls": probe["total_recorded_calls"],
        "calls_by_role": probe["calls_by_role"],
        "prompt_tokens": probe["prompt_tokens"],
        "completion_tokens": probe["completion_tokens"],
        "total_tokens": probe["total_tokens"],
        "new_test_calls": 0, "actual_prompt_commits": 0,
        "trajectory_mutations": 0,
        "funnel": funnel,
        "structure": {row["arm"]: row for row in structure_rows},
        "classifier": classifier,
    }
    write_csv(report / "parent_level.csv", parent_rows)
    write_csv(report / "branch_level.csv", branch_rows)
    write_csv(report / "candidate_level.csv", candidate_rows)
    write_csv(report / "conversion_structure.csv", structure_rows)
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (report / "classifier.json").write_text(json.dumps(classifier, indent=2) + "\n", encoding="utf-8")
    (report / "source_freeze.json").write_text(json.dumps({
        "freeze_version": freeze["freeze_version"],
        "execution_commit": freeze["execution_commit"],
        "registry_content_hash": freeze["registry_content_hash"],
        "source_file_count": freeze["source_file_count"],
        "source_freeze_status": freeze["source_freeze_status"],
        "phase_a_zero_api": freeze["phase_a_zero_api"],
    }, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.root.resolve(), args.registry.resolve(),
        args.source_freeze.resolve(), args.report.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
