from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


support = load("m2f_feedback_support", "scripts/m2f_probe_support.py")
analyzer = load("m2f_feedback_analyzer", "scripts/analyze_v16_m2f_feedback_necessity_probe.py")


def fixture_case():
    return {
        "parent_prompt": "parent",
        "source_candidate_prompt": "source",
        "repair_evidence": [{"question_hash": "repair-hash", "question": "repair example"}],
        "loss_evidence": [{"question_hash": "secret-loss-hash", "question": "secret loss example"}],
        "numeric_summary": {
            "responsibility_gain_count": 2,
            "all_loss_count": 3,
            "pivotal_loss_count": 1,
        },
    }


def payload(request: str) -> dict:
    return json.loads(request.split("RepairInput:\n", 1)[1])


def test_f1_excludes_actual_collateral_evidence():
    request = support.build_causal_control_request(fixture_case(), "F1_LOSS_BLIND")
    body = payload(request)
    assert "candidate_specific_competence_losses" not in body
    assert body["numeric_summary"] == {"responsibility_gain_count": 2}
    assert "secret-loss-hash" not in request
    assert "secret loss example" not in request


def test_f2_diff_is_candidate_specific_collateral_evidence():
    case = fixture_case()
    first = payload(support.build_causal_control_request(case, "F1_LOSS_BLIND"))
    second = payload(support.build_causal_control_request(case, "F2_CANDIDATE_FEEDBACK"))
    assert first["parent_member_prompt"] == second["parent_member_prompt"]
    assert first["source_m20_candidate_prompt"] == second["source_m20_candidate_prompt"]
    assert first["successful_assigned_responsibility_repairs"] == second["successful_assigned_responsibility_repairs"]
    assert second["candidate_specific_competence_losses"] == case["loss_evidence"]
    assert second["numeric_summary"] == case["numeric_summary"]


def test_frozen_design_has_seven_paired_one_call_cases():
    spec = json.loads((ROOT / "experiments/v16_m2f_feedback_necessity_fixed_candidate_20260813/analysis_spec.json").read_text())
    assert spec["source_candidate_count"] == 7
    assert spec["cell_count"] == 14
    assert spec["calls_per_cell"] == 1
    assert spec["analysis_only_metrics"] == ["critical_net", "oracle_delta", "vote_net"]


def test_summarize_keeps_exchange_metrics_analysis_only():
    row = {
        "retained_source_repairs": 2, "source_responsibility_repairs": 2,
        "nonresponsibility_loss": 3, "loss_decomposition": {"recovered": 1, "new": 2},
        "common_safe": True, "critical_gain": 4, "critical_loss": 1,
        "critical_net": 3, "oracle_delta": 2, "vote_net": 1, "target_gain": 2,
    }
    value = analyzer.summarize([row])
    assert value["critical_net"] == 3
    assert value["oracle_delta"] == 2
    assert value["vote_net"] == 1


def test_unknown_arm_is_rejected():
    try:
        support.build_causal_control_request(fixture_case(), "F3")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown causal-control arm accepted")


def test_unchanged_is_valid_only_for_causal_control():
    case = fixture_case()
    response = '{"repaired_prompt":"source"}'
    with __import__("pytest").raises(ValueError, match="unchanged"):
        support.parse_repair_output(response, case)
    assert support.parse_repair_output(
        response, case, allow_unchanged=True
    ) == "source"
