from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v17_conversion_priority_hybrid_support import ARMS, arm_specs


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(root: Path, registry_path: Path) -> dict[str, Any]:
    registry = read_json(registry_path)
    cases = {row["case_id"]: row for row in registry["cases"]}
    cells = [read_json(path) for path in sorted(root.glob("*/cells/*/cell_result.json"))]
    branches = [read_json(path) for path in sorted(root.glob("*/branches/*/branch_result.json"))]
    candidates = [row for path in sorted(root.glob("*/candidate_level.jsonl")) for row in read_jsonl(path)]
    blockers: list[str] = []
    if len(cells) != 15:
        blockers.append("cell_count")
    if len(branches) != 15:
        blockers.append("unique_branch_count")
    by_case: dict[str, list[dict[str, Any]]] = {}
    conceptual_keys: list[str] = []
    for cell in cells:
        case = cases.get(cell.get("case_id"))
        by_case.setdefault(str(cell.get("case_id")), []).append(cell)
        if case is None or cell.get("arm") not in ARMS:
            blockers.append("cell_identity")
            continue
        expected = arm_specs(case)[cell["arm"]]
        expected_targets = [int(row["target_member"]) for row in expected]
        if list(map(int, cell.get("target_ids", []))) != expected_targets:
            blockers.append("selector_replay")
        if not cell.get("decision_frozen_before_validation"):
            blockers.append("validation_leakage")
        if int(cell.get("team_prompt_commit_count", -1)) or int(cell.get("trajectory_mutation_count", -1)):
            blockers.append("fixed_parent_mutation")
        if int(cell.get("test_calls", -1)):
            blockers.append("test_call")
        for rank, branch in enumerate(cell.get("branches", [])):
            conceptual_keys.append(str(branch.get("canonical_branch_key", "")))
            if int(branch.get("target_member", -1)) != expected_targets[rank]:
                blockers.append("branch_target")
            if int(branch.get("source_candidate_count", 3)) > 2:
                blockers.append("source_budget")
            if int(branch.get("valid_revision_count", -1)) != int(branch.get("valid_source_count", -2)):
                blockers.append("revision_parity")
        if not cell.get("would_commit"):
            for key in (
                "realized_validation_vote_delta", "realized_validation_oracle_delta",
                "realized_validation_target_delta", "train_realized_vote_delta",
                "train_realized_target_delta", "train_realized_oracle_delta",
                "target_transfer_gap",
            ):
                if int(cell.get(key, 1)):
                    blockers.append("no_commit_realized_metric")
            if any(int(value) for key, value in cell.get("validation_structure", {}).items() if key != "transition_histogram"):
                blockers.append("no_commit_structure")
    if len(by_case) != 5 or any(
        len(rows) != 3 or {row["arm"] for row in rows} != set(ARMS)
        for rows in by_case.values()
    ):
        blockers.append("matrix_inventory")
    for rows in by_case.values():
        if len({json.dumps(row["parent_validation"], sort_keys=True) for row in rows}) != 1:
            blockers.append("parent_validation_drift")
    branch_by_key = {row["canonical_branch_key"]: row for row in branches}
    if len(branch_by_key) != len(branches) or set(conceptual_keys) != set(branch_by_key):
        blockers.append("branch_sharing_identity")
    for branch in branches:
        if branch.get("state_hash_before") != branch.get("state_hash_after"):
            blockers.append("branch_state_mutation")
        if int(branch.get("funnel", {}).get("infrastructure_failed_updates", 0)):
            blockers.append("infrastructure_failure")
        if int(branch.get("validation_calls", -1)) or int(branch.get("test_calls", -1)):
            blockers.append("branch_evaluation_isolation")
    required = {
        "parent_id", "seed", "update_index", "arm", "target_member", "branch_type",
        "is_conversion_eligible", "conversion_responsibility_count",
        "direct_vote_flip_count", "candidate_stage", "valid", "feasible",
        "common_safe_outcome", "sanitized_rejection_reasons", "train", "validation",
        "branch_rank", "cell_rank", "selected_as_cell_winner", "would_commit_contribution",
    }
    if not candidates or any(not required.issubset(row) for row in candidates):
        blockers.append("candidate_logging_schema")
    summary = read_json(root / "probe_summary.json") if (root / "probe_summary.json").is_file() else {}
    if int(summary.get("team_prompt_commit_count", -1)) or int(summary.get("trajectory_mutation_count", -1)):
        blockers.append("summary_mutation")
    if int(summary.get("test_calls", -1)) or int(summary.get("validation_selection_count", -1)):
        blockers.append("summary_test_isolation")
    return {
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "parent_count": len(by_case),
        "cell_count": len(cells),
        "conceptual_branch_count": len(conceptual_keys),
        "unique_branch_count": len(branches),
        "reused_branch_count": len(conceptual_keys) - len(branches),
        "candidate_record_count": len(candidates),
        "total_recorded_calls": int(summary.get("total_recorded_calls", 0)),
        "calls_by_role": summary.get("calls_by_role", {}),
        "new_test_calls": int(summary.get("test_calls", -1)),
        "actual_prompt_commits": int(summary.get("team_prompt_commit_count", -1)),
        "trajectory_mutations": int(summary.get("trajectory_mutation_count", -1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.registry.resolve())
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
