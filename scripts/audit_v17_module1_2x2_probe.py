from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path) -> dict:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*/*/cell_result.json"))]
    blockers: list[str] = []
    if len(rows) != 24:
        blockers.append("complete_cell_count")
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
        if len(row.get("target_ids", [])) != 2 or len(set(row.get("target_ids", []))) != 2:
            blockers.append("target_budget")
        if len(row.get("branches", [])) != 2:
            blockers.append("branch_count")
        for branch in row.get("branches", []):
            if int(branch.get("source_candidate_count", 3)) > 2:
                blockers.append("source_candidate_budget")
            if int(branch.get("valid_source_candidate_count", -1)) > int(branch.get("source_candidate_count", -2)):
                blockers.append("valid_source_count")
            if int(branch.get("loss_blind_revision_count", -1)) != int(branch.get("valid_source_candidate_count", -2)):
                blockers.append("generic_revision_parity")
            if int(branch.get("validation_calls", 0)) or int(branch.get("test_calls", 0)):
                blockers.append("branch_evaluation_isolation")
        if int(row.get("team_prompt_commit_count", -1)) or int(row.get("trajectory_mutation_count", -1)):
            blockers.append("fixed_parent_mutation")
        if int(row.get("test_calls", -1)):
            blockers.append("test_call")
    if len(by_case) != 6 or any(len(group) != 4 for group in by_case.values()):
        blockers.append("matrix_inventory")
    for group in by_case.values():
        if len({json.dumps(row["parent_validation"], sort_keys=True) for row in group}) != 1:
            blockers.append("parent_validation_drift")
        if {row["cell"] for row in group} != {"A", "B", "C", "D"}:
            blockers.append("cell_inventory")
    return {
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "case_count": len(by_case), "cell_count": len(rows),
        "team_prompt_commit_count": sum(int(row.get("team_prompt_commit_count", 0)) for row in rows),
        "trajectory_mutation_count": sum(int(row.get("trajectory_mutation_count", 0)) for row in rows),
        "test_calls": sum(int(row.get("test_calls", 0)) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
