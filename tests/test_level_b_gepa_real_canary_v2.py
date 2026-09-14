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


def test_v2_freeze_is_narrow_and_not_api_authorized() -> None:
    module = load()
    protocol = module.protocol_document()
    assert protocol["experiment_id"] == "level_b_gepa_real_canary_v2"
    assert protocol["attempt_id"].endswith("pending_authorization")
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


def test_v2_preflight_passes_without_calling_api() -> None:
    module = load()
    result = module.base.preflight()
    assert result["gate"] == "PASS"
    assert result["api_calls"] == 0
    assert result["validation_calls"] == 0
    assert result["test_calls"] == 0
