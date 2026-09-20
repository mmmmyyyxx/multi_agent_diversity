from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from multi_dataset_diverse_rl.governance.artifacts import (
    build_sha256_manifest,
    scan_sanitized_artifacts,
)
from multi_dataset_diverse_rl.governance.manifest import (
    preregistration_hash,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/accepted_mutation_exact_state_graph_v2_20260920"
MANIFEST = ROOT / "experiments/manifests/accepted_mutation_exact_state_graph_v2.yaml"
MATERIALIZATION_MANIFEST = (
    ROOT / "experiments/manifests/accepted_mutation_profile_materialization_v1.yaml"
)


def read(name: str):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_cache_recovery_fails_closed_with_all_requests_identified():
    summary = read("summary.json")
    assert summary["status"] == "CACHE_ONLY_PROFILE_RECOVERY_INCOMPLETE"
    assert summary["required_unique_requests"] == 500
    assert summary["recovered_unique_requests"] == 0
    assert summary["missing_unique_requests"] == 500
    assert summary["exact_graph_constructed"] is False
    assert summary["provider_calls"] == 0
    assert summary["future_materialization_would_be_prospective"] is True

    misses = read("cache_miss_manifest.json")
    assert misses["missing_count"] == len(misses["missing_requests"]) == 500
    assert set(misses["missing_count_by_candidate"].values()) == {100}
    assert len({row["request_identity"] for row in misses["missing_requests"]}) == 500
    assert all(set(row) == {
        "candidate_identity", "example_id", "request_identity",
    } for row in misses["missing_requests"])


def test_historical_requests_are_verified_but_responses_are_not_persisted():
    recovery = read("cache_recovery_audit.json")
    assert set(recovery["historical_ledger_success_coverage"].values()) == {100}
    assert recovery["v2_runtime_cache_kind"] == "PROCESS_LOCAL_DICTIONARY_NOT_PERSISTED"
    assert recovery["v2_provider_ledger_contains_response_text"] is False
    assert recovery["exact_json_cache_hits"] == 0
    assert recovery["sqlite_scan"]["direct_request_identity_hits"] == 0
    assert recovery["sqlite_scan"]["candidate_prompt_hash_hits"] == 0
    assert recovery["stop_reason"] == "ANY_REQUIRED_RESPONSE_ABSENT"
    assert recovery["historical_exact_graph_recoverable_from_current_artifacts"] is False
    assert not (REPORT / "exact_state_catalog.json").exists()
    assert not (REPORT / "sanitized_profile_matrix.json").exists()


def test_integrity_and_zero_api_invariants_hold():
    integrity = read("integrity_audit.json")
    assert integrity["historical_execution_complete"] is True
    assert integrity["historical_files_byte_identical_before_after"] is True
    assert integrity["candidate_prompt_hashes_match"] is True
    assert integrity["required_request_count"] == 500
    assert integrity["required_requests_have_historical_provider_success"] is True
    assert integrity["model_matches_frozen_contract"] is True
    assert integrity["thinking_mode_matches_frozen_contract"] is True
    assert integrity["provider_access_used_by_audit"] is False
    assert integrity["raw_response_or_reasoning_published"] is False
    assert all(value == 0 for value in read("api_ledger_summary.json").values())
    source = (
        ROOT / "scripts/recover_accepted_mutation_exact_state_graph_v2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith("openai") for name in imported | imported_from)


def test_materialization_is_draft_only_and_exactly_scoped():
    draft = read("profile_materialization_draft.json")
    assert draft["status"] == "AUTHORIZATION_REQUIRED"
    assert draft["READY_TO_RUN"] is False
    assert draft["missing_unique_solver_requests"] == 500
    assert draft["successful_solver_call_ceiling"] == 500
    assert draft["transport_attempt_ceiling"] == 2000
    assert draft["automatic_execution_permitted"] is False
    assert draft["evidence_semantics"] == "PROSPECTIVE_FRESH_PROVIDER_REALIZATION"
    assert draft["historical_v2_profile_reconstruction"] is False
    materialization = yaml.safe_load(MATERIALIZATION_MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    assert validate_manifest(materialization, schema) == []
    assert materialization["status"] == "DRAFT"
    assert materialization["api_authorization"]["authorized"] is False
    assert materialization["artifacts"]["preregistration"]["sha256"] == preregistration_hash(materialization)
    assert materialization["budget"]["limit"]["successful_solver_provider_calls"] == 500
    assert materialization["design"]["historical_v2_profile_reconstruction"] is False


def test_manifest_and_report_are_valid_sanitized_and_replayable():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    assert validate_manifest(manifest, schema) == []
    assert manifest["artifacts"]["preregistration"]["sha256"] == preregistration_hash(manifest)
    assert manifest["api_authorization"]["authorized"] is False
    assert manifest["result"]["classifier"] == "CACHE_ONLY_PROFILE_RECOVERY_INCOMPLETE"
    assert scan_sanitized_artifacts(REPORT) == []
    assert read("sanitization_manifest.json")["status"] == "PASS"
    assert read("sha256_manifest.json") == build_sha256_manifest(REPORT)
