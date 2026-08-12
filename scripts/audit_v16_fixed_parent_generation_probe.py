from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    summary = json.loads((args.run_root / "probe_summary.json").read_text(encoding="utf-8"))
    blockers = []
    expected = {(case["case_id"], variant) for case in registry["cases"] for variant in registry["variants"]}
    observed = {(row["case_id"], row["variant"]) for row in summary.get("cells", [])}
    if expected != observed:
        blockers.append("cell_inventory_mismatch")
    if summary.get("registry_hash") != registry.get("registry_content_hash"):
        blockers.append("registry_hash_mismatch")
    for row in summary.get("cells", []):
        if row.get("commit_performed") or row.get("parent_state_hash_before") != row.get("parent_state_hash_after"):
            blockers.append(f"parent_mutation:{row.get('case_id')}:{row.get('variant')}")
        if row.get("validation_calls") or row.get("test_calls"):
            blockers.append(f"evaluation_isolation:{row.get('case_id')}:{row.get('variant')}")
        if row.get("generated_candidate_count", 0) > 2:
            blockers.append(f"candidate_budget:{row.get('case_id')}:{row.get('variant')}")
        terminal = str(row.get("funnel", {}).get("terminal_failure_class", ""))
        if terminal in {"transport_failure", "persistence_failure"}:
            blockers.append(f"infrastructure_failure:{row.get('case_id')}:{row.get('variant')}:{terminal}")
        expected_case = next(case for case in registry["cases"] if case["case_id"] == row.get("case_id"))
        if (
            "assigned_question_hashes" in expected_case
            and sorted(row.get("assigned_question_hashes", []))
            != sorted(expected_case["assigned_question_hashes"])
        ):
            blockers.append(f"responsibility_membership:{row.get('case_id')}:{row.get('variant')}")
    report = {
        "audit_version": "v16_fixed_parent_generation_probe_audit_v1",
        "gate": "PASS" if not blockers else "FAIL", "blockers": blockers,
        "expected_cell_count": len(expected), "observed_cell_count": len(observed),
        "commit_count": sum(int(bool(row.get("commit_performed"))) for row in summary.get("cells", [])),
        "validation_calls": summary.get("validation_calls"), "test_calls": summary.get("test_calls"),
        "raw_artifacts_modified": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
