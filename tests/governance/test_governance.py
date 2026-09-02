from __future__ import annotations

import json
from pathlib import Path

import pytest
import jsonschema

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.governance.artifacts import (
    build_sha256_manifest,
    compare_tree_snapshot,
    scan_sanitized_artifacts,
    snapshot_tree_hashes,
    validate_fact_assertions,
    validate_report_contract,
)
from multi_dataset_diverse_rl.governance.authorization import (
    AuthorizationError,
    require_api_authorization,
)
from multi_dataset_diverse_rl.governance.manifest import (
    preregistration_hash,
    validate_manifest,
)
from multi_dataset_diverse_rl.governance.provenance import build_provenance
from multi_dataset_diverse_rl.governance.registries import (
    load_yaml,
    validate_experiment_registry,
    validate_invariants,
    validate_lineage,
)
from multi_dataset_diverse_rl.governance.telemetry import (
    summarize_api_ledger,
    summarize_funnel,
    validate_evaluation_access,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = load_yaml(ROOT / "infrastructure" / "experiment_manifest.schema.json")


def manifest(*, status: str = "DRAFT", api: bool = False) -> dict:
    history = [{"status": "DRAFT", "timestamp": "synthetic"}]
    if status == "RUNNING":
        history.extend(
            {"status": value, "timestamp": "synthetic"}
            for value in ("PREREGISTERED", "IMPLEMENTED", "PREFLIGHT_PASS", "RUNNING")
        )
    value = {
        "schema_version": "experiment_manifest_v1",
        "experiment_id": "synthetic_case",
        "title": "Synthetic case",
        "status": status,
        "legacy_index": False,
        "lifecycle_history": history,
        "lineage": {"parents": [], "derives_from": None},
        "scientific_question": "Synthetic question",
        "hypotheses": [],
        "design": {
            "changed": ["synthetic flag"],
            "unchanged": ["all runtime behavior"],
            "forbidden_changes": ["test access"],
        },
        "method_identity": versions.METHOD_VERSION,
        "runtime_version": "synthetic_v1",
        "data": {
            "task": "synthetic",
            "formal": False,
            "split_ids": {},
            "split_hashes": {},
            "validation_policy": "not_accessed",
            "test_policy": "prohibited",
        },
        "model": {"solver": None, "optimizer_roles": {}},
        "seeds": [],
        "budget": {"type": "synthetic", "frozen_before_run": status == "RUNNING", "limit": 1},
        "api_authorization": {
            "authorized": api,
            "allowed_roles": ["Student"] if api else [],
            "allowed_phases": ["pilot"] if api else [],
        },
        "selection": {
            "primary_metric": "synthetic",
            "frozen_rule": "fixed",
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
        },
        "artifacts": {"preregistration": None, "report": None, "provenance": None},
        "git": {"design_commit": None, "implementation_commit": None},
        "result": {"classifier": None, "conclusion": None, "evidence_type": "not_yet_available"},
    }
    if status == "RUNNING" and api:
        value["git"]["implementation_commit"] = "0" * 40
        value["artifacts"]["preregistration"] = {"sha256": preregistration_hash(value)}
    return value


def test_manifest_valid_and_invalid() -> None:
    """INV-MANIFEST-001: schema-valid manifests pass and missing policy fails."""
    valid = manifest()
    assert validate_manifest(valid, SCHEMA) == []
    invalid = manifest()
    del invalid["data"]["test_policy"]
    assert validate_manifest(invalid, SCHEMA)


def test_runtime_identity_mismatch_fails() -> None:
    """INV-ID-001: a new manifest cannot contradict versions.py runtime identity."""
    value = manifest()
    value["method_identity"] = "not_the_runtime_method"
    assert any("runtime method identity" in error for error in validate_manifest(value, SCHEMA))


def test_test_split_cannot_affect_selection() -> None:
    """INV-TEST-001: manifest validation rejects test-based selection."""
    value = manifest()
    value["selection"]["test_used_for_selection"] = True
    assert any("test data cannot" in error for error in validate_manifest(value, SCHEMA))


def test_manifest_lifecycle_cannot_skip_to_completed() -> None:
    """INV-MANIFEST-001: DRAFT cannot jump directly to COMPLETED."""
    value = manifest(status="DRAFT")
    value["status"] = "COMPLETED"
    value["lifecycle_history"].append({"status": "COMPLETED", "timestamp": "synthetic"})
    assert any("illegal lifecycle" in error for error in validate_manifest(value, SCHEMA))


def test_api_manifest_requires_frozen_preregistration() -> None:
    """INV-AUTH-001: running API manifests require source, budget, and preregistration freeze."""
    value = manifest(status="RUNNING", api=True)
    assert validate_manifest(value, SCHEMA) == []
    value["artifacts"]["preregistration"] = None
    assert any("preregistration" in error for error in validate_manifest(value, SCHEMA))


def test_manifest_prereg_hash_is_immutable() -> None:
    """INV-MANIFEST-001: a frozen preregistration hash detects design mutation."""
    value = manifest(status="RUNNING", api=True)
    recorded = value["artifacts"]["preregistration"]["sha256"]
    value["design"]["changed"].append("post-freeze mutation")
    assert preregistration_hash(value) != recorded
    assert any("hash does not match" in error for error in validate_manifest(value, SCHEMA))
    with pytest.raises(AuthorizationError, match="hash mismatch"):
        require_api_authorization(
            value, phase="pilot", role="Student", explicit_user_authorized=True
        )


def test_api_requires_explicit_authorization() -> None:
    """INV-AUTH-001: user authorization is independent and fail-closed."""
    value = manifest(status="RUNNING", api=True)
    with pytest.raises(AuthorizationError, match="explicit user"):
        require_api_authorization(value, phase="pilot", role="Student")


def test_api_rejects_unauthorized_lifecycle_state() -> None:
    """INV-AUTH-001: preregistered facts cannot call providers before RUNNING."""
    value = manifest()
    value["api_authorization"] = {
        "authorized": True, "allowed_roles": ["Student"], "allowed_phases": ["pilot"]
    }
    value["budget"]["frozen_before_run"] = True
    value["artifacts"]["preregistration"] = {"sha256": preregistration_hash(value)}
    with pytest.raises(AuthorizationError, match="RUNNING"):
        require_api_authorization(
            value, phase="pilot", role="Student", explicit_user_authorized=True
        )


def test_api_role_and_phase_are_bounded() -> None:
    """INV-AUTH-001: an authorized experiment cannot silently widen role or phase."""
    value = manifest(status="RUNNING", api=True)
    with pytest.raises(AuthorizationError, match="role"):
        require_api_authorization(
            value, phase="pilot", role="Solver", explicit_user_authorized=True
        )
    with pytest.raises(AuthorizationError, match="phase"):
        require_api_authorization(
            value, phase="formal", role="Student", explicit_user_authorized=True
        )


def test_duplicate_experiment_ids_fail(tmp_path: Path) -> None:
    """INV-MANIFEST-001: registry experiment identifiers are unique."""
    (tmp_path / "manifest.yaml").write_text("placeholder", encoding="utf-8")
    registry = {
        "experiments": [
            {"experiment_id": "same", "manifest": "manifest.yaml"},
            {"experiment_id": "same", "manifest": "manifest.yaml"},
        ]
    }
    errors, _ = validate_experiment_registry(tmp_path, registry, SCHEMA)
    assert any("duplicate experiment IDs" in error for error in errors)


def test_unknown_parent_and_lineage_cycle_fail() -> None:
    """INV-MANIFEST-001: DAG validation rejects missing nodes and cycles."""
    unknown, _ = validate_lineage(
        {"edges": [{"from": "missing", "to": "a", "relation": "followup_of"}]},
        ["a"],
    )
    assert any("unknown lineage parent" in error for error in unknown)
    cyclic, _ = validate_lineage(
        {"edges": [
            {"from": "a", "to": "b", "relation": "followup_of"},
            {"from": "b", "to": "a", "relation": "followup_of"},
        ]},
        ["a", "b"],
    )
    assert "lineage cycle detected" in cyclic


def test_missing_invariant_test_and_unknown_failure_fail(tmp_path: Path) -> None:
    """INV-MANIFEST-001: critical invariants bind real tests and known failures."""
    (tmp_path / "impl.py").write_text("pass\n", encoding="utf-8")
    registry = {"invariants": [{
        "id": "INV-X-001",
        "status": "ACTIVE",
        "critical": True,
        "implementation_refs": ["impl.py"],
        "tests": ["missing_test.py"],
        "failure_refs": ["FAIL-MISSING"],
    }]}
    errors = validate_invariants(tmp_path, registry, [])
    assert any("missing test" in error for error in errors)
    assert any("unknown failure" in error for error in errors)


def test_sanitization_accepts_aggregates(tmp_path: Path) -> None:
    """INV-ARTIFACT-001: hashes, counters, metrics, and categories are publishable."""
    (tmp_path / "safe.json").write_text(
        json.dumps({"prompt_hash": "a" * 64, "count": 2, "category": "blocked"}),
        encoding="utf-8",
    )
    assert scan_sanitized_artifacts(tmp_path) == []


def test_prompt_text_not_in_sanitized_report(tmp_path: Path) -> None:
    """INV-ARTIFACT-001: raw prompt-bearing fields are rejected."""
    (tmp_path / "bad.json").write_text(
        json.dumps({"raw_prompt": "reason about the hidden example"}), encoding="utf-8"
    )
    assert any("forbidden_field" in row["reason"] for row in scan_sanitized_artifacts(tmp_path))


def test_output_artifact_has_no_absolute_path(tmp_path: Path) -> None:
    """INV-ARTIFACT-001: absolute Windows and user-home paths are rejected."""
    (tmp_path / "windows.txt").write_text("source=C:\\private\\run", encoding="utf-8")
    (tmp_path / "posix.txt").write_text("source=/home/user/run", encoding="utf-8")
    findings = scan_sanitized_artifacts(tmp_path)
    assert {row["file"] for row in findings} == {"posix.txt", "windows.txt"}


def test_sanitization_rejects_cache_and_checkpoint(tmp_path: Path) -> None:
    """INV-ARTIFACT-001: cache and checkpoint payloads are never publishable."""
    (tmp_path / "cache.sqlite").write_bytes(b"synthetic")
    (tmp_path / "state.checkpoint").write_bytes(b"synthetic")
    assert len(scan_sanitized_artifacts(tmp_path)) == 2


def test_sha256_manifest_is_deterministic_and_relative(tmp_path: Path) -> None:
    """INV-ARTIFACT-001: manifest replay is deterministic and host-independent."""
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    first = build_sha256_manifest(tmp_path)
    second = build_sha256_manifest(tmp_path)
    assert first == second
    assert [row["file"] for row in first["files"]] == ["a.txt", "b.txt"]
    assert all(":" not in row["file"] for row in first["files"])


def test_historical_reports_are_read_only(tmp_path: Path) -> None:
    """INV-MANIFEST-001: evidence snapshots detect any attempted historical rewrite."""
    (tmp_path / "summary.json").write_text('{"count":1}\n', encoding="utf-8")
    frozen = snapshot_tree_hashes(tmp_path)
    assert compare_tree_snapshot(tmp_path, frozen) == []
    (tmp_path / "summary.json").write_text('{"count":2}\n', encoding="utf-8")
    assert compare_tree_snapshot(tmp_path, frozen) == ["summary.json"]


def test_funnel_aggregation() -> None:
    """FAIL-LOW-UPDATE-THROUGHPUT: funnel stages retain distinct denominators."""
    rows = [
        {"branch_id": "b1", "stage": "TARGET_SELECTED", "target_member": 1},
        {"branch_id": "b1", "stage": "STUDENT_REACHED", "target_member": 1},
        {"branch_id": "b1", "stage": "STUDENT_CANDIDATE_VALID", "target_member": 1},
        {"branch_id": "b1", "stage": "COMMON_SAFE_FEASIBLE", "target_member": 1},
        {"branch_id": "b1", "stage": "WOULD_COMMIT", "target_member": 1},
        {"branch_id": "b1", "stage": "COMMIT", "target_member": 1},
        {"branch_id": "b2", "stage": "TARGET_SELECTED", "target_member": 2},
    ]
    summary = summarize_funnel(rows)
    assert summary["branches_attempted"] == 2
    assert summary["student_reach"] == 1
    assert summary["rates"]["feasible_per_branch"] == 0.5
    assert summary["distinct_members_updated"] == 1


def test_not_applicable_funnel_is_not_zero_event() -> None:
    """FAIL-LOW-UPDATE-THROUGHPUT: NOT_APPLICABLE is distinct from observed zero."""
    summary = summarize_funnel(
        [{"branch_id": None, "stage": "NOT_APPLICABLE", "target_member": None}]
    )
    assert summary["branches_attempted"] == 0
    assert summary["not_applicable_event_count"] == 1
    assert summary["rates"]["reach_per_branch"] is None


def test_validation_before_freeze_is_rejected() -> None:
    """INV-DATA-001: validation access requires selection freeze."""
    result = validate_evaluation_access([{
        "split": "validation", "purpose": "report", "phase": "TRAIN_FROZEN",
        "row_count": 3, "selection_frozen_before_access": False,
    }])
    assert result["status"] == "FAIL"
    assert any("validation accessed" in error for error in result["errors"])


def test_test_split_cannot_be_accessed_before_final_freeze() -> None:
    """INV-TEST-001: test access is valid only after final freeze."""
    result = validate_evaluation_access([{
        "split": "test", "purpose": "final report", "phase": "TRAIN_FROZEN",
        "row_count": 3, "selection_frozen_before_access": True,
    }])
    assert result["status"] == "FAIL"
    passing = validate_evaluation_access([{
        "split": "test", "purpose": "final report", "phase": "FINAL_FROZEN",
        "row_count": 3, "selection_frozen_before_access": True,
    }])
    assert passing["status"] == "PASS"


def test_evaluation_access_zero_is_mechanical() -> None:
    """INV-TEST-001: an empty access ledger mechanically establishes zero test calls."""
    result = validate_evaluation_access([])
    assert result["test_calls_zero"] is True
    assert result["test_access_count"] == 0


def test_api_ledger_summary_keeps_roles_and_failures_separate() -> None:
    """FAIL-PRE-STUDENT-CRITIC-GATE: API role and failure counts are explicit."""
    summary = summarize_api_ledger([
        {"role": "Teacher", "input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "success": True},
        {"role": "Critic", "input_tokens": 3, "output_tokens": 1, "total_tokens": 4, "success": False, "failure_category": "semantic"},
    ], candidate_count=1, feasible_candidate_count=1, accepted_update_count=1)
    assert summary["call_count"] == 2
    assert summary["by_role"]["Critic"]["total_tokens"] == 4
    assert summary["failure_categories"] == {"semantic": 1}
    assert summary["tokens_per_candidate"] == 7


def test_fact_assertions_are_machine_checked() -> None:
    """INV-MANIFEST-001: README claims do not replace explicit boolean assertions."""
    assert validate_fact_assertions({"api_calls_zero": True, "test_calls_zero": True}) == []
    assert validate_fact_assertions({"api_calls_zero": False}) == ["api_calls_zero"]


def test_report_contract() -> None:
    """INV-ARTIFACT-001: the bundled synthetic future-report fixture is complete."""
    fixture = ROOT / "infrastructure" / "fixtures" / "new_format_report"
    missing = validate_report_contract(fixture, optimization=True, validation=True)
    assert missing in ([], ["sha256_manifest.json"])


def test_telemetry_schemas_reject_unstructured_payloads() -> None:
    """INV-MANIFEST-001: ledger and funnel events have explicit schemas."""
    ledger_schema = load_yaml(ROOT / "infrastructure" / "api_ledger.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"role": "Student", "raw_response": "unsafe"}, ledger_schema)
    funnel_schema = load_yaml(ROOT / "infrastructure" / "optimization_funnel.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"stage": "SILENT_ZERO"}, funnel_schema)


def test_runtime_identity_matches_versions() -> None:
    """INV-ID-001: normative prose mirrors the runtime identity authority."""
    spec = (ROOT / "docs" / "design" / "CURRENT_SPEC.md").read_text(encoding="utf-8")
    assert versions.METHOD_VERSION in spec
    assert f"checkpoint v{versions.CHECKPOINT_VERSION}" in spec


def test_standard_provenance_is_host_independent() -> None:
    """INV-ARTIFACT-001: standard provenance exposes identity but no host-local path."""
    value = build_provenance(
        ROOT,
        manifest(),
        run_id="synthetic-run",
        config_hashes={"config": "a" * 64},
        timestamp="synthetic",
    )
    rendered = json.dumps(value, sort_keys=True)
    assert str(ROOT) not in rendered
    assert value["method_version"] == versions.METHOD_VERSION
    assert value["checkpoint_version"] == versions.CHECKPOINT_VERSION
    assert len(value["source_tree_sha256"]) == 64
