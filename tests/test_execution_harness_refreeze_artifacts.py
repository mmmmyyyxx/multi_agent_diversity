from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/execution_harness_refreeze_20260922"
OLD = ROOT / "reports/final_pre_experiment_freeze_20260922"


def test_historical_preregistration_root_is_byte_identical():
    assert hashlib.sha256((OLD / "sha256_manifest.json").read_bytes()).hexdigest() == (
        "a229f8688067b0514b6ce8397094ce998b0bfcc07e87e71456e413998cc9735e"
    )
    supersession = json.loads((REPORT / "supersession_metadata.json").read_text())
    assert supersession["old_files_modified"] == 0
    assert supersession["old_status"] == "PREREGISTERED_NOT_EXECUTABLE"


def test_new_manifests_explicitly_bind_lwj_and_exact_models():
    for experiment_id in (
        "gepa_layer2_real_canary_v2",
        "sequential_symmetry_breaking_online_pilot_v2",
        "gepa_saturation_comparison_v2",
    ):
        manifest = yaml.safe_load(
            (ROOT / "experiments/manifests" / f"{experiment_id}.yaml").read_text()
        )
        provider = manifest["execution_freeze"]["provider"]
        assert manifest["model"]["provider_profile"] == "lwj"
        assert provider["provider_profile"] == "lwj"
        assert provider["implicit_fallback_allowed"] is False
        assert provider["models"]["solver"]["model"] == "qwen3-8b"
        assert provider["models"]["evaluator"]["model"] == "qwen3.7-flash"
        assert provider["raw_endpoint_persisted"] is False
        assert provider["api_key_persisted"] is False


def test_fresh_initialization_and_formal_gate_are_frozen():
    sequential = yaml.safe_load(
        (ROOT / "experiments/manifests/sequential_symmetry_breaking_online_pilot_v2.yaml").read_text()
    )
    assert sequential["execution_freeze"]["initialization_policy"] == (
        "FRESH_DETERMINISTIC_INITIALIZATION_V1"
    )
    assert sequential["api_authorization"]["authorized"] is False

    formal = yaml.safe_load(
        (ROOT / "experiments/manifests/gepa_saturation_comparison_v2.yaml").read_text()
    )
    assert formal["seeds"] == [80, 81, 82]
    assert formal["execution_gate"]["status"] == "EXECUTION_GATED"
    assert formal["budget"]["limit"]["local_no_update_patience"] == 3
    assert formal["budget"]["limit"]["team_no_update_patience"] == 2
    assert formal["execution_gate"]["observed_dependencies"] == {
        "canary": "PENDING",
        "sequential": "PENDING",
    }


def test_report_contains_no_raw_endpoint_or_secret_names():
    forbidden = ("compatible-mode/v1", "sk-", "ws-e9q", "ws-tbe")
    for path in REPORT.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            assert not any(token.lower() in text for token in forbidden)
