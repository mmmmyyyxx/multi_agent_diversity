"""Single zero-API, zero-dataset-evaluation governance preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.artifacts import (
    build_sha256_manifest,
    scan_sanitized_artifacts,
    validate_report_contract,
)
from multi_dataset_diverse_rl.governance.registries import (
    load_yaml,
    validate_experiment_registry,
    validate_failure_registry,
    validate_invariants,
    validate_lineage,
)
from scripts.render_experiment_lineage import render as render_lineage
from scripts.render_known_failures import render as render_failures


def run(workspace: Path) -> dict:
    errors: list[str] = []
    schema = load_yaml(workspace / "infrastructure" / "experiment_manifest.schema.json")
    registry = load_yaml(workspace / "experiments" / "registry.yaml")
    registry_errors, manifests = validate_experiment_registry(workspace, registry, schema)
    errors.extend(registry_errors)

    lineage = load_yaml(workspace / "experiments" / "lineage.yaml")
    lineage_errors, order = validate_lineage(lineage, manifests)
    errors.extend(lineage_errors)

    failure_registry = load_yaml(workspace / "docs" / "failures" / "registry.yaml")
    failure_errors = validate_failure_registry(workspace, failure_registry)
    errors.extend(failure_errors)
    failure_ids = [row["failure_id"] for row in failure_registry.get("failures", [])]

    invariants = load_yaml(workspace / "docs" / "design" / "invariants.yaml")
    invariant_errors = validate_invariants(workspace, invariants, failure_ids)
    errors.extend(invariant_errors)

    synthetic_schema_rows = {
        "api_ledger.schema.json": {
            "experiment_id": "synthetic", "run_id": "run", "phase": "pilot",
            "role": "Student", "call_index": 0, "provider_call": False,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "retry_index": 0, "success": True, "failure_category": None,
        },
        "optimization_funnel.schema.json": {
            "experiment_id": "synthetic", "run_id": "run", "branch_id": None,
            "target_member": None, "stage": "NOT_APPLICABLE",
            "reason_category": None, "timestamp": "synthetic",
        },
        "evaluation_access.schema.json": {
            "experiment_id": "synthetic", "run_id": "run", "split": "train",
            "purpose": "fixture", "phase": "PREFLIGHT", "row_count": 0,
            "selection_frozen_before_access": False, "timestamp": "synthetic",
        },
        "fact_assertions.schema.json": {"api_calls_zero": True},
    }
    for name, row in synthetic_schema_rows.items():
        try:
            jsonschema.validate(row, load_yaml(workspace / "infrastructure" / name))
        except jsonschema.ValidationError as exc:
            errors.append(f"{name} synthetic validation failed: {exc.message}")

    fixture = workspace / "infrastructure" / "fixtures" / "new_format_report"
    report_errors = validate_report_contract(fixture, optimization=True, validation=True)
    errors.extend(f"report fixture missing: {item}" for item in report_errors)
    sanitization_findings = scan_sanitized_artifacts(fixture)
    errors.extend(f"sanitization fixture: {item}" for item in sanitization_findings)
    first_hash = build_sha256_manifest(fixture)
    second_hash = build_sha256_manifest(fixture)
    if first_hash != second_hash:
        errors.append("deterministic SHA256 replay mismatch")
    stored_hash = json.loads((fixture / "sha256_manifest.json").read_text(encoding="utf-8"))
    if stored_hash != first_hash:
        errors.append("stored fixture SHA256 manifest is stale")

    lineage_target = workspace / "docs" / "experiments" / "LINEAGE.md"
    if not lineage_target.is_file() or lineage_target.read_text(encoding="utf-8") != render_lineage(workspace):
        errors.append("generated LINEAGE.md is stale")
    failure_target = workspace / "docs" / "failures" / "KNOWN_FAILURES.md"
    if not failure_target.is_file() or failure_target.read_text(encoding="utf-8") != render_failures(workspace):
        errors.append("generated KNOWN_FAILURES.md is stale")

    return {
        "ok": not errors,
        "mode": "zero_api_zero_dataset_evaluation",
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "experiment_count": len(manifests),
        "active_invariant_count": sum(
            row.get("status") == "ACTIVE" for row in invariants.get("invariants", [])
        ),
        "failure_count": len(failure_ids),
        "lineage_node_count": len(order),
        "deterministic_hash_replay": "PASS" if first_hash == second_hash else "FAIL",
        "sanitization_fixture": "PASS" if not sanitization_findings else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.workspace.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
