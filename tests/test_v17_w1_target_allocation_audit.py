from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/v17_w1_target_allocation_mechanism_audit_20260820"


def _json(name: str) -> dict:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def _csv(name: str) -> list[dict[str, str]]:
    with (REPORT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_reconstruction_gate_and_matrix_are_complete_without_api() -> None:
    gate = _json("reconstruction_gate.json")
    assert gate["gate"] == "PASS"
    assert gate["zero_api"] is True
    assert gate["new_test_calls"] == 0
    assert (gate["parent_count"], gate["cell_count"], gate["branch_count"]) == (6, 24, 48)
    assert gate["w1_score_reconstruction_mismatch"] == 0
    assert gate["branch_target_identity_mismatch"] == 0
    assert gate["validation_metric_reconstruction_mismatch"] == 0
    assert gate["cache_sha256_unchanged"] is True


def test_branch_rows_preserve_fixed_w1_formula_and_cell_inventory() -> None:
    rows = _csv("branch_level.csv")
    assert len(rows) == 48
    assert {(row["case_id"], row["cell"]) for row in rows} == {
        (case_id, cell)
        for case_id in {row["case_id"] for row in rows}
        for cell in "ABCD"
    }
    assert len({row["case_id"] for row in rows}) == 6
    assert all(sum(row["case_id"] == case and row["cell"] == cell for row in rows) == 2
               for case, cell in {(row["case_id"], row["cell"]) for row in rows})
    for row in rows:
        reconstructed = (
            float(row["base_opportunity_B"]) + 0.05 * float(row["normalized_wait"])
        ) * float(row["failure_discount"])
        assert abs(reconstructed - float(row["w1_score"])) <= 1e-8


def test_funnel_and_classifier_match_reconstructed_evidence() -> None:
    summary = _json("summary.json")
    assert summary["realized_by_cell"] == {
        "A": {"validation_vote_delta": 1, "validation_oracle_delta": 2, "would_commit_count": 3},
        "B": {"validation_vote_delta": 0, "validation_oracle_delta": 0, "would_commit_count": 1},
        "C": {"validation_vote_delta": 2, "validation_oracle_delta": -3, "would_commit_count": 3},
        "D": {"validation_vote_delta": 0, "validation_oracle_delta": 2, "would_commit_count": 2},
    }
    rr, w1 = summary["funnel"]["RR"], summary["funnel"]["W1"]
    assert (rr["valid_source_branch_count"], rr["feasible_branch_count"], rr["cell_winner_count"]) == (10, 7, 6)
    assert (w1["valid_source_branch_count"], w1["feasible_branch_count"], w1["cell_winner_count"]) == (9, 3, 3)
    classifier = _json("classifier.json")
    assert classifier["primary_mechanism_diagnosis"] == "MULTIPLE_TARGET_ALLOCATION_FAILURES"
    assert classifier["repeated_target_diminishing_returns"] is False
    assert set(classifier["supported_component_labels"]) == {
        "TARGET_COVERAGE_EXPLORATION_FAILURE",
        "TARGET_VALUE_ESTIMATION_FAILURE",
        "BRANCH_REALIZABILITY_MISMATCH",
    }


def test_candidate_level_is_not_fabricated() -> None:
    status = _json("candidate_level_status.json")
    assert status["status"] == "NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS"
    assert status["candidate_level_csv_created"] is False
    assert not (REPORT / "candidate_level.csv").exists()
