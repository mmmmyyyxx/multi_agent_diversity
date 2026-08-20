from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "build_v17_failure_decomposition_report",
    SCRIPTS / "build_v17_failure_decomposition_report.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_zero_api_is_a_hard_constant() -> None:
    assert MODULE.ZERO_API_REQUIRED is True


def test_generalization_gap_formula() -> None:
    rows = []
    for seed in MODULE.SEEDS:
        for arm_index, arm in enumerate(MODULE.ARMS):
            for split_index, split in enumerate(MODULE.PUBLIC_SPLITS):
                rows.append({"seed": seed, "arm": arm, "split": split, "vote_accuracy": arm_index + split_index / 10})
    gaps = MODULE.generalization_rows(rows)
    item = next(row for row in gaps if row["seed"] == 56 and row["contrast"] == "S1_to_S2")
    assert item["train_delta"] == pytest.approx(1.0)
    assert item["train_to_test_gap"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"oracle_after": 0}, "ORACLE_LOST"),
        ({"H_after": 3}, "WRONG_COALITION_STRENGTHENED"),
        ({"M_after": 0}, "TIE_OR_MARGIN_FAILURE"),
        ({"M_after": 1}, "COVERAGE_REMAINS_BUT_VOTE_LOST"),
    ],
)
def test_vote_loss_taxonomy_is_deterministic(patch: dict[str, int], expected: str) -> None:
    row = {"vote_before": 1, "vote_after": 0, "oracle_before": 1, "oracle_after": 1, "H_before": 2, "H_after": 2, "M_after": -1}
    row.update(patch)
    assert MODULE.vote_loss_taxonomy(row) == expected


def test_conversion_accounting() -> None:
    rows = [
        {"oracle_before": 0, "oracle_after": 1, "vote_before": 0, "vote_after": 1, "pivotal_agents_before": [], "pivotal_agents_after": [2], "unique_agents_before": [], "unique_agents_after": [2]},
        {"oracle_before": 0, "oracle_after": 1, "vote_before": 0, "vote_after": 0, "pivotal_agents_before": [], "pivotal_agents_after": [], "unique_agents_before": [], "unique_agents_after": [3]},
    ]
    result = MODULE.conversion_row(rows, seed=56, split="test", contrast="S1_to_S2")
    assert result["oracle_gain_count"] == 2
    assert result["oracle_gain_and_vote_gain"] == 1
    assert result["oracle_gain_without_vote_gain"] == 1
    assert result["new_pivotal_and_vote_gain"] == 1


def test_recovery_profile_does_not_claim_causality() -> None:
    rows = [{"vote_before": 0, "vote_after": 1, "oracle_before": 1, "oracle_after": 1, "H_before": 3, "H_after": 2, "G_before": 2, "G_after": 3, "M_before": -1, "M_after": 1}]
    result = MODULE.recovery_profile(rows)
    assert result["converted_existing_coverage"] == 1
    assert "cause" not in result["profile"].lower()


def test_hypothesis_classifier_boundaries() -> None:
    from v17_failure_decomposition_support import classify_hypotheses

    rows = []
    for seed in (56, 57, 58):
        rows.append({
            "seed": seed,
            "s2_s1_train_delta": 0.2,
            "s2_s1_test_delta": -0.1,
            "s2_s1_gap": 0.3,
            "s2_oracle_minus_s1": 0.1,
            "s2_vote_minus_s1": -0.1,
            "s2_oracle_vote_gap_minus_s1": 0.2,
            "s2_coverage_delta": 0.1,
            "s2_entropy_minus_s1": -0.1,
            "high_target_agent_test_delta": -0.2,
            "low_target_agent_test_delta": -0.1,
            "s1_accepted_updates": 5,
            "s2_accepted_updates": 3,
            "s1_test_gain_per_commit": 1.0,
            "s2_test_gain_per_commit": 0.0,
            "s2_specialization_train_minus_s1": 2,
            "s2_specialization_test_minus_s1": 0,
            "specialization_measure_count": 2,
        })
    statuses = {row["hypothesis_id"]: row["status"] for row in classify_hypotheses(rows)}
    assert all(statuses[key] == "SUPPORTED" for key in ("H1", "H2", "H3", "H4A", "H4B", "H5"))


PRIVATE = ROOT / "runs" / "v17_failure_decomposition_20260820"
REPORT = ROOT / "reports" / "v17_failure_decomposition_20260820"


@pytest.mark.skipif(not (PRIVATE / "evidence_inventory.json").is_file(), reason="local V17 evidence is not present")
def test_local_frozen_reconstruction_contract() -> None:
    inventory = json.loads((PRIVATE / "evidence_inventory.json").read_text(encoding="utf-8"))
    analysis = json.loads((PRIVATE / "analysis_metrics_private.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (PRIVATE / "reconstructed_row_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert inventory["evidence_complete"] is True
    assert len(inventory["cells"]) == 45
    assert {item["seed"] for item in inventory["cells"]} == {56, 57, 58}
    assert inventory["validation_test_final_state_hashes_match"] is True
    assert all(row["M"] == row["G"] - row["H"] for row in rows)
    assert all(row["oracle_correct"] == int(row["correct_agent_count"] > 0) for row in rows)
    assert all(len(row["unique_agents"]) in {0, 1} for row in rows)
    assert len(analysis["split_metrics"]) == 45


@pytest.mark.skipif(not (REPORT / "protocol_gate.json").is_file(), reason="local V17 report is not present")
def test_public_report_is_sanitized_and_accounted() -> None:
    protocol = json.loads((REPORT / "protocol_gate.json").read_text(encoding="utf-8"))
    diagnosis = json.loads((REPORT / "diagnosis_summary.json").read_text(encoding="utf-8"))
    assert protocol["reconstructed_cells"] == 45
    assert all(value == 0 for value in protocol["api_model_solver_optimizer_evaluator_call_counts"].values())
    assert diagnosis["S1_S2_test_transition_net"] == diagnosis["S1_S2_test_vote_net"]
    for path in REPORT.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".png":
            lowered = path.read_text(encoding="utf-8").lower()
            assert "d:\\" not in lowered
            assert "c:\\" not in lowered
            assert '"prompt_text"' not in lowered
            assert '"raw_response"' not in lowered


def test_zero_api_scripts_have_no_provider_client_construction() -> None:
    for name in ("audit_v17_failure_decomposition.py", "build_v17_failure_decomposition_report.py", "v17_failure_decomposition_support.py"):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "OpenAI(" not in source
        assert "DashScope" not in source
        assert "requests.post" not in source
