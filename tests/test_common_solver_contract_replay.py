from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analyze_common_solver_contract_replay import audit, coalition_decomposition


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/common_solver_contract_v1_replay_20260906"
REPORT = ROOT / "reports/common_solver_contract_v1_replay_20260906"
SEED77_RUN = ROOT / "runs/common_solver_contract_v1_seed77_replay_20260907"
SEED77_PREP = ROOT / "runs/common_solver_contract_v1_seed77_prep_20260907/private_replay_registry.json"
SEED77_MANIFEST = ROOT / "reports/common_solver_contract_v1_seed77_prep_20260907/contract_manifest.json"
SEED77_REPORT = ROOT / "reports/common_solver_contract_v1_seed77_replay_20260907"


def read_json(name: str) -> dict:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_common_replay_official_audit_passes() -> None:
    result, predictions = audit(ROOT, RUN)
    assert result["gate"] == "PASS"
    assert result["errors"] == []
    assert result["source_freeze"] == "PASS"
    assert result["test50_accessed"] is False
    assert result["optimization_rerun"] is False
    assert result["prediction_row_count"] == 200
    assert len(predictions) == 200
    assert result["accounting"]["logical_calls"] == 600
    assert result["accounting"]["successful_provider_calls"] == 400
    assert result["accounting"]["failed_provider_attempts"] == 0
    assert result["accounting"]["cache_hits"] == 200


def test_common_replay_metrics_and_classifier_are_factual() -> None:
    with (REPORT / "per_state_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["state_id"]: row for row in csv.DictReader(handle)}
    assert rows["P0_COMMON"]["vote_correct"] == "30"
    assert rows["MARS_SEED76_FINAL"]["vote_correct"] == "35"
    assert rows["GEPA_SEED76_FINAL"]["vote_correct"] == "37"
    assert rows["DIVERSITY_SEED76_P1_FINAL"]["vote_correct"] == "33"
    assert rows["DIVERSITY_SEED76_P1_FINAL"]["oracle_correct"] == "46"
    classifier = read_json("classifier.json")
    assert classifier["common_contract_replay"] == "PASS"
    assert classifier["evaluation_contract_parity"] == "CONFIRMED_FOR_THIS_REPLAY"
    assert classifier["end_to_end_optimization_parity"] == "NOT_ESTABLISHED"
    assert classifier["formal_cross_method_raw_ranking"] == "NOT_YET_ELIGIBLE"
    assert (
        classifier["full_parity_rerun_decision"]
        == "JUSTIFIED_IF_FORMAL_CROSS_METHOD_RANKING_IS_REQUIRED"
    )


def test_common_replay_public_report_is_sanitized_and_test_locked() -> None:
    facts = read_json("fact_assertions.json")
    sanitization = read_json("sanitization_manifest.json")
    assert facts["gate"] == "PASS"
    assert facts["test50_accessed"] is False
    assert facts["optimization_rerun"] is False
    assert facts["provider_calls"] == 400
    assert facts["failed_provider_attempts"] == 0
    assert sanitization["status"] == "PASS"
    assert sanitization["finding_count"] == 0


def test_coalition_decomposition_accounts_for_all_aligned_cases() -> None:
    cases = [{"position": position, "gold": "A"} for position in range(50)]
    predictions = []
    for position in range(50):
        predictions.extend(
            [
                {
                    "state_id": "P0_COMMON",
                    "case_position": position,
                    "member_labels": ["B"] * 5,
                    "member_valid": [True] * 5,
                    "vote_correct": False,
                    "oracle_correct": False,
                },
                {
                    "state_id": "DIVERSITY_SEED77_P1_FINAL",
                    "case_position": position,
                    "member_labels": ["A", "B", "B", "B", "B"],
                    "member_valid": [True] * 5,
                    "vote_correct": False,
                    "oracle_correct": True,
                },
            ]
        )
    summary, matrix = coalition_decomposition(
        predictions,
        cases,
        "DIVERSITY_SEED77_P1_FINAL",
    )
    assert sum(row["case_count"] for row in matrix) == 50
    assert summary["new_coverage_cases"] == 50
    assert summary["new_correct_member_votes"] == 50
    assert summary["oracle_gain_case_count"] == 50
    assert summary["oracle_gain_to_vote_gain_count"] == 0


def test_seed77_common_replay_and_coalition_report_are_factual() -> None:
    result, predictions = audit(
        ROOT,
        SEED77_RUN,
        prep=SEED77_PREP,
        manifest_path=SEED77_MANIFEST,
    )
    assert result["gate"] == "PASS"
    assert result["accounting"]["logical_calls"] == 600
    assert result["accounting"]["successful_provider_calls"] == 350
    assert result["accounting"]["cache_hits"] == 250
    assert len(predictions) == 200

    with (SEED77_REPORT / "per_state_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["state_id"]: row for row in csv.DictReader(handle)}
    assert rows["P0_COMMON"]["vote_correct"] == "31"
    assert rows["MARS_SEED77_FINAL"]["vote_correct"] == "31"
    assert rows["GEPA_SEED77_FINAL"]["vote_correct"] == "35"
    assert rows["DIVERSITY_SEED77_P1_FINAL"]["vote_correct"] == "29"
    assert rows["DIVERSITY_SEED77_P1_FINAL"]["oracle_correct"] == "43"

    coalition = json.loads(
        (SEED77_REPORT / "diversity_coalition_decomposition.json").read_text(encoding="utf-8")
    )
    assert coalition["new_coverage_cases"] == 14
    assert coalition["lost_coverage_cases"] == 2
    assert coalition["new_correct_member_votes"] == 32
    assert coalition["lost_correct_member_votes"] == 42
    assert coalition["oracle_gain_to_vote_gain_count"] == 5
    assert coalition["p0_correct_to_final_majority_preserved_count"] == 24
