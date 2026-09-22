from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reports" / "final_pre_experiment_freeze_20260922"
IDS = (
    "gepa_layer2_real_canary_v1",
    "sequential_symmetry_breaking_online_pilot_v1",
    "gepa_saturation_comparison_v1",
    "mars_layer2_real_canary_v1",
)
REQUIRED = {
    "PROTOCOL.md",
    "manifest.json",
    "claim_registry.json",
    "cost_envelope.json",
    "provider_model_freeze.json",
    "data_freeze.json",
    "stopping_semantics.json",
    "authorization_gate.json",
    "preflight_report.json",
    "sha256_manifest.json",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_preregistration_roots_are_complete_and_hash_valid():
    for experiment_id in IDS:
        root = BUNDLE / experiment_id
        assert {path.name for path in root.iterdir()} == REQUIRED
        manifest = read(root / "sha256_manifest.json")["files"]
        for name, expected in manifest.items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


def test_provider_models_patience_and_access_are_frozen_without_secrets():
    for experiment_id in IDS:
        root = BUNDLE / experiment_id
        provider = read(root / "provider_model_freeze.json")
        stopping = read(root / "stopping_semantics.json")
        gate = read(root / "authorization_gate.json")
        assert provider["provider_profile"] == "lwj"
        assert provider["models"]["solver"] == "qwen3-8b"
        assert provider["models"]["teacher"] == "qwen3.7-flash"
        assert provider["endpoint_identity"]["raw_endpoint_persisted"] is False
        assert stopping["selected_local_no_update_patience"] == 3
        assert stopping["selected_team_no_update_patience"] == 2
        assert gate["authorized"] is False
        assert gate["ready_to_run"] is False
        assert read(root / "manifest.json")["validation50_calls"] == 0
        assert read(root / "manifest.json")["test50_calls"] == 0
        serialized = "\n".join(path.read_text(encoding="utf-8") for path in root.iterdir())
        assert "compatible-mode" not in serialized
        assert "sk-" not in serialized.lower()


def test_formal_pair_has_prospective_three_seed_saturation_design():
    root = BUNDLE / "gepa_saturation_comparison_v1"
    manifest = read(root / "manifest.json")
    stopping = read(root / "stopping_semantics.json")
    claims = read(root / "claim_registry.json")
    assert manifest["seeds"] == [80, 81, 82]
    assert manifest["execution_regime"] == "saturation"
    assert stopping["scientific_hard_budget"] == "DISABLED"
    assert "GEPA_NATIVE_SATURATION" in manifest["arms"]
    assert "GEPA_LAYER2_SATURATION" in manifest["arms"]
    assert claims["cross_optimizer_claims"] is False


def test_prep_fails_closed_instead_of_claiming_execution_readiness():
    state = read(BUNDLE / "completion_state.json")
    assert state["real_provider_attempts_during_prep"] == 0
    assert state["ready_for_authorization"] is False
    assert all(read(BUNDLE / experiment_id / "preflight_report.json")["blockers"] for experiment_id in IDS)
