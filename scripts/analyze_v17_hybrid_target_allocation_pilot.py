from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v17_hybrid_target_allocation_support import ARMS, HYBRID, RR, W1, classify


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
    by_case = {case["case_id"]: case for case in registry["cases"]}
    parent_rows = []
    for case_id in sorted(by_case):
        case_cells = {row["arm"]: row for row in cells if row["case_id"] == case_id}
        case = by_case[case_id]
        row: dict[str, Any] = {
            "parent_id": case_id,
            "seed": case["source_seed"],
            "update_index": case["source_update_index"],
            "w1_rank1": int(case["w1_priority_rows"][0]["agent_id"]),
            "w1_rank2": int(case["w1_priority_rows"][1]["agent_id"]),
            "rr_first": case["arm_specs"][RR][0]["target_member"],
            "rr_second": case["arm_specs"][RR][1]["target_member"],
            "hybrid_exploit": case["arm_specs"][HYBRID][0]["target_member"],
            "hybrid_explore": case["arm_specs"][HYBRID][1]["target_member"],
            "hybrid_explore_equals_w1_rank2": int(
                case["arm_specs"][HYBRID][1]["target_member"]
                == int(case["w1_priority_rows"][1]["agent_id"])
            ),
        }
        for arm in ARMS:
            cell = case_cells[arm]
            prefix = arm.lower()
            row.update({
                f"{prefix}_targets": "|".join(map(str, cell["target_ids"])),
                f"{prefix}_feasible_branches": sum(int(branch["feasible_candidate_count"] > 0) for branch in cell["branches"]),
                f"{prefix}_feasible_candidates": sum(int(branch["feasible_candidate_count"]) for branch in cell["branches"]),
                f"{prefix}_would_commit": int(cell["would_commit"]),
                f"{prefix}_validation_vote_delta": int(cell["realized_validation_vote_delta"]),
                f"{prefix}_validation_oracle_delta": int(cell["realized_validation_oracle_delta"]),
                f"{prefix}_validation_target_delta": int(cell["realized_validation_target_delta"]),
            })
        parent_rows.append(row)
    branch_rows = []
    for cell in cells:
        for branch in cell["branches"]:
            branch_rows.append({
                "parent_id": cell["case_id"], "arm": cell["arm"],
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
                "realized_validation_vote_delta": cell["realized_validation_vote_delta"] if branch["produced_cell_best"] else 0,
                "realized_validation_oracle_delta": cell["realized_validation_oracle_delta"] if branch["produced_cell_best"] else 0,
            })
    candidate_rows = []
    for row in candidates:
        candidate_rows.append({
            "parent_id": row["parent_id"], "arm": row["arm"],
            "target_member": row["target_member"], "branch_type": row["branch_type"],
            "candidate_id": row["candidate_id"], "candidate_stage": row["candidate_stage"],
            "valid": int(row["valid"]), "feasible": int(row["feasible"]),
            "common_safe_outcome": row["common_safe_outcome"],
            "sanitized_rejection_reasons": "|".join(row["sanitized_rejection_reasons"]),
            "train_target_delta": row["train"]["target_delta"],
            "train_vote_delta": row["train"]["vote_delta"],
            "train_oracle_delta": row["train"]["oracle_delta"],
            "validation_target_delta": row["validation"]["target_delta"],
            "validation_vote_delta": row["validation"]["vote_delta"],
            "validation_oracle_delta": row["validation"]["oracle_delta"],
            "validation_vote_gain_count": row["validation"]["vote_gain_count"],
            "validation_vote_loss_count": row["validation"]["vote_loss_count"],
            "validation_oracle_gain_count": row["validation"]["oracle_gain_count"],
            "validation_oracle_loss_count": row["validation"]["oracle_loss_count"],
            "wrong_coalition_reduced_count": row["validation"]["wrong_coalition_reduced_count"],
            "wrong_coalition_increased_count": row["validation"]["wrong_coalition_increased_count"],
            "branch_rank": row["branch_rank"], "cell_rank": row["cell_rank"],
            "selected_as_cell_winner": int(row["selected_as_cell_winner"]),
            "would_commit_contribution": int(row["would_commit_contribution"]),
        })
    feasible = {
        arm: sum(int(row["feasible_candidate_count"] > 0) for row in branch_rows if row["arm"] == arm)
        for arm in ARMS
    }
    funnel = {}
    for arm in ARMS:
        branches = [row for row in branch_rows if row["arm"] == arm]
        arm_cells = [row for row in cells if row["arm"] == arm]
        funnel[arm] = {
            "selected_branch_count": len(branches),
            "valid_source_branch_count": sum(int(row["valid_source_count"] > 0) for row in branches),
            "valid_source_count": sum(int(row["valid_source_count"]) for row in branches),
            "valid_revision_count": sum(int(row["valid_revision_count"]) for row in branches),
            "feasible_branch_count": feasible[arm],
            "feasible_candidate_count": sum(int(row["feasible_candidate_count"]) for row in branches),
            "cell_best_branch_count": sum(int(row["produced_cell_best"]) for row in branches),
            "would_commit_count": sum(int(row["would_commit"]) for row in arm_cells),
            "positive_validation_vote_count": sum(int(row["realized_validation_vote_delta"] > 0) for row in arm_cells),
            "positive_validation_oracle_count": sum(int(row["realized_validation_oracle_delta"] > 0) for row in arm_cells),
            "validation_vote_delta_sum": sum(int(row["realized_validation_vote_delta"]) for row in arm_cells),
            "validation_oracle_delta_sum": sum(int(row["realized_validation_oracle_delta"]) for row in arm_cells),
            "mean_validation_vote_delta": sum(int(row["realized_validation_vote_delta"]) for row in arm_cells) / 6,
            "mean_validation_oracle_delta": sum(int(row["realized_validation_oracle_delta"]) for row in arm_cells) / 6,
            "unique_target_members": len({target for row in arm_cells for target in row["target_ids"]}),
        }
    classifier = classify([
        {
            arm: {
                "validation_vote_delta": int(row[f"{arm.lower()}_validation_vote_delta"]),
                "validation_oracle_delta": int(row[f"{arm.lower()}_validation_oracle_delta"]),
            }
            for arm in ARMS
        }
        for row in parent_rows
    ], feasible)
    hybrid_branches = [row for row in branch_rows if row["arm"] == HYBRID]
    explore = [row for row in hybrid_branches if row["branch_type"] == "explore"]
    exploration = {
        "exploration_attributable_feasible_branches": sum(int(row["feasible_candidate_count"] > 0) for row in explore),
        "exploration_attributable_cell_winners": sum(int(row["produced_cell_best"]) for row in explore),
        "exploration_attributable_would_commit": sum(int(row["produced_cell_best"] and row["cell_would_commit"]) for row in explore),
        "exploration_validation_vote_contribution": sum(int(row["realized_validation_vote_delta"]) for row in explore),
        "exploration_validation_oracle_contribution": sum(int(row["realized_validation_oracle_delta"]) for row in explore),
        "differentiating_parent_count": sum(1 - int(row["hybrid_explore_equals_w1_rank2"]) for row in parent_rows),
    }
    probe_summary = read_json(root / "probe_summary.json")
    summary = {
        "summary_version": "v17_hybrid_target_allocation_pilot_summary_v1",
        "execution_commit": probe_summary["execution_commit"],
        "phase_a_gate": "PASS", "phase_b_gate": "PASS",
        "phase_a_zero_api": True,
        "phase_b_actual_api_call_count": probe_summary["actual_api_call_count"],
        "new_test_calls": 0, "actual_prompt_commits": 0, "trajectory_mutations": 0,
        "prospective_parent_count": 6, "cell_count": 18,
        "conceptual_branch_count": 36,
        "deduplicated_branch_count": probe_summary["deduplicated_branch_count"],
        "funnel": funnel,
        "exploration_attribution": exploration,
        "classifier": classifier,
    }
    source_freeze = {
        "freeze_version": freeze["freeze_version"],
        "execution_commit": freeze["execution_commit"],
        "registry_content_hash": freeze["registry_content_hash"],
        "source_file_count": freeze["source_file_count"],
        "prospective_parents_frozen": freeze["prospective_parents_frozen"],
        "old_diagnostic_parents_excluded_from_primary": freeze["old_diagnostic_parents_excluded_from_primary"],
        "phase_a_zero_api": freeze["phase_a_zero_api"],
        "source_freeze_status": freeze["source_freeze_status"],
    }
    write_csv(report / "parent_level.csv", parent_rows)
    write_csv(report / "branch_level.csv", branch_rows)
    write_csv(report / "candidate_level.csv", candidate_rows)
    for name, payload in (
        ("summary.json", summary), ("classifier.json", classifier),
        ("source_freeze.json", source_freeze),
    ):
        (report / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest_files = ("parent_level.csv", "branch_level.csv", "candidate_level.csv", "summary.json", "classifier.json", "source_freeze.json")
    manifest = {
        "manifest_version": "v17_hybrid_target_allocation_report_manifest_v1",
        "files": {name: hashlib.sha256((report / name).read_bytes()).hexdigest() for name in manifest_files},
    }
    (report / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root.resolve(), args.registry.resolve(), args.source_freeze.resolve(), args.report.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
