from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / "scripts/run_level_b_gepa_real_canary_v2.py"
    spec = importlib.util.spec_from_file_location("level_b_gepa_real_canary_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_governance_refreeze_is_narrow_and_manifest_authorized() -> None:
    module = load()
    protocol = module.protocol_document()
    assert protocol["experiment_id"] == "level_b_gepa_real_canary_v2"
    assert protocol["attempt_id"].endswith("stagefix2_transportfix1_authorized1")
    assert protocol["reflection_input_repair"] == (
        "component_specific_reasoning_evidence_v1"
    )
    assert protocol["minimum_technical_success"] == (
        "contract_valid_changed_proposal_reaches_solver_gt_zero"
    )
    assert protocol["official_gepa"]["reflective_dataset_version"] == (
        "component_specific_reasoning_evidence_v1"
    )
    assert protocol["validation50_calls"] == protocol["test50_calls"] == 0
    assert protocol["classifier_version"] == (
        "level_b_local_empirical_path_classifier_v1"
    )
    assert protocol["source_freeze_hash_semantics"] == "normalized_lf_text_v1"
    assert protocol["transport_retry_classifier_version"] == (
        "openai_sdk_connection_error_v1"
    )
    assert protocol["failed_attempt_durability_version"] == (
        "solver_provider_attempt_failure_v1"
    )
    manifest = module.base.yaml.safe_load(module.base.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["api_authorization"]["authorized"] is True
    assert protocol["logical_split_identities"] == {
        "optimize100": {
            "count": 100,
            "question_hashes_sha256": "1678f8339929dab5d728cc25940c8e452f74b3f0508fda8fc9a19532207d494f",
        },
        "shadow50": {
            "count": 50,
            "question_hashes_sha256": "254b6a4ae1f00f7a913795be4f0e3522790b537de4ca8e9bd73c04dfc23d1557",
        },
        "validation50": {
            "count": 50,
            "question_hashes_sha256": "95cd6cfd2ed3de66a16625b133a678c9422c41637356faa1073d6a908482d359",
        },
        "test50": {
            "count": 50,
            "question_hashes_sha256": "c1a90894d094b099a4bc2fc09fd7f529e8ddc4a667237e5f572f77358d7330b2",
        },
    }


def test_v2_classifier_uses_only_callback_attempts_and_solver_reach() -> None:
    module = load()
    outcome = object()
    assert module.classify(
        {"proposer_diagnostics": {"proposal_attempts": 0, "solver_reached": 0}},
        outcome,
    ) == "NO_REAL_PROPOSAL_ATTEMPT"
    assert module.classify(
        {"proposer_diagnostics": {"proposal_attempts": 3, "solver_reached": 0}},
        outcome,
    ) == "PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH"
    assert module.classify(
        {"proposer_diagnostics": {"proposal_attempts": 3, "solver_reached": 1}},
        outcome,
    ) == "LOCAL_EMPIRICAL_PATH_CONFIRMED"


def test_v2_preflight_passes_without_calling_api() -> None:
    module = load()
    result = module.base.preflight()
    assert result["gate"] == "PASS"
    assert result["api_calls"] == 0
    assert result["validation_calls"] == 0
    assert result["test_calls"] == 0
