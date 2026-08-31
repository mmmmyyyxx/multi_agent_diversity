from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gepa_candidate_breadth_support import (
    BREADTHS, REQUESTED_SOURCE_COUNT, ROOT, canonical_hash, read_json,
    sha256_file, write_json,
)


def audit(raw: Path, registry_path: Path, freeze_path: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh gate root required")
    registry, freeze = read_json(registry_path), read_json(freeze_path)
    if sha256_file(registry_path) != freeze["registry_file_sha256"]:
        raise ValueError("registry file hash mismatch")
    if canonical_hash({k: v for k, v in registry.items() if k != "registry_content_hash"}) != registry["registry_content_hash"]:
        raise ValueError("registry content hash mismatch")
    blockers: list[str] = []
    results = []
    for case in registry["cases"]:
        path = raw / case["case_id"] / "case_result.json"
        train_path = raw / case["case_id"] / "train_decision.json"
        if not path.is_file() or not train_path.is_file():
            blockers.append(f"{case['case_id']}:missing_result")
            continue
        row, train = read_json(path), read_json(train_path)
        results.append(row)
        if row["parent_team_hash"] != case["parent_team_hash"]:
            blockers.append(f"{case['case_id']}:parent_mismatch")
        if int(row["target_agent_id"]) != int(case["target_agent_id"]):
            blockers.append(f"{case['case_id']}:target_mismatch")
        if row["state_hash_before"] != row["state_hash_after"]:
            blockers.append(f"{case['case_id']}:state_mutation")
        if not row["decision_frozen_before_validation"] or not train["decision_frozen_before_validation"]:
            blockers.append(f"{case['case_id']}:validation_leakage")
        if row["team_prompt_commit_count"] or row["trajectory_mutation_count"]:
            blockers.append(f"{case['case_id']}:forbidden_commit")
        if row["test_calls"] or row["infrastructure_failure_count"]:
            blockers.append(f"{case['case_id']}:test_or_infrastructure_failure")
        if int(row["requested_source_candidate_count"]) != REQUESTED_SOURCE_COUNT:
            blockers.append(f"{case['case_id']}:source_budget_mismatch")
        if int(row["actual_source_candidate_count"]) > REQUESTED_SOURCE_COUNT:
            blockers.append(f"{case['case_id']}:source_budget_exceeded")
        if int(row["revision_attempt_count"]) != int(row["actual_source_candidate_count"]):
            blockers.append(f"{case['case_id']}:revision_attempt_parity")
        if not bool(row["n4"]["n2_pool_is_subset"]):
            blockers.append(f"{case['case_id']}:nested_pool_violation")
        if set(train["n2"]["pool_candidate_hashes"]) - set(train["n4"]["pool_candidate_hashes"]):
            blockers.append(f"{case['case_id']}:train_nested_pool_violation")
        if any(key.startswith("validation_") for key in train["n2"]):
            blockers.append(f"{case['case_id']}:validation_present_before_freeze")
    if len(results) != 2:
        blockers.append("case_inventory_mismatch")
    summary = read_json(raw / "probe_summary.json")
    if summary["case_count"] != 2 or summary["test_calls"] or summary["infrastructure_failure_count"]:
        blockers.append("probe_summary_contract_failure")
    gate = {
        "gate_version": "gepa_candidate_breadth_gate_v1",
        "gate": "PASS" if not blockers else "HOLD",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "case_count": len(results),
        "breadths": list(BREADTHS),
        "requested_source_candidate_count": REQUESTED_SOURCE_COUNT,
        "decision_frozen_before_validation": all(
            bool(row["decision_frozen_before_validation"]) for row in results
        ),
        "team_prompt_commit_count": sum(int(row["team_prompt_commit_count"]) for row in results),
        "trajectory_mutation_count": sum(int(row["trajectory_mutation_count"]) for row in results),
        "new_test_calls": sum(int(row["test_calls"]) for row in results),
    }
    out.mkdir(parents=True)
    write_json(out / "audit_summary.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.raw, args.registry, args.freeze, args.out.parent):
        resolved = path.resolve()
        if resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
            raise SystemExit("all paths must be project-local")
    print(json.dumps(audit(
        args.raw.resolve(), args.registry.resolve(), args.freeze.resolve(), args.out.resolve()
    ), indent=2))


if __name__ == "__main__":
    main()
