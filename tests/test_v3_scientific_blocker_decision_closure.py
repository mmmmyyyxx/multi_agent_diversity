"""Zero-API regression evidence for V3 scientific decision closure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.team_search.schemas import TeamEvidenceCase
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder
from scripts.analyze_v3_scientific_blockers import (
    cross_product_grid,
    initial_state_capacity,
    quota_matrix,
)


def test_v3_initial_state_exact_schedule_arithmetic_and_replay() -> None:
    first = initial_state_capacity()
    second = initial_state_capacity()
    assert first == second
    assert first["responsibility_source"] == "raw_legal_pre_routing_v1"
    assert first["selected_member"] == 0
    assert (first["D"], first["N"], first["C"], first["V"]) == (0, 0, 50, 50)
    assert first["f"] == 0 and first["target_score"] == 50.0
    assert first["overlapping_residual_count"] == 50
    assert (first["aligned_responsibility_rows"], first["focus_rows"], first["anchor_rows"]) == (50, 0, 0)
    assert first["search_packet_item_count"] == 50
    assert (first["ordered_schedule_batches"], first["batch_size"], first["schedule_delivery_slots"]) == (12, 3, 36)
    assert first["minimum_metric_budget_for_coverage"] == 51
    assert first["local_eval_ids"] == 12
    assert (first["gepa_seed_evaluation_calls"], first["gepa_proposal_minibatch_cost"],
            first["gepa_accepted_full_validation_cost"]) == (12, 6, 12)
    assert first["max_rejected_proposal_batches_at_budget_36"] == 4
    assert first["max_distinct_deliveries_if_every_proposal_rejected"] == 12
    assert first["metric_budget_for_17_rejected_proposal_batches"] == 114
    assert first["assignment_responsibility_value"] == 0.0
    assert first["current_failure"] == "packet schedule must cover every selected search example"
    assert first["real_api_calls"] == first["validation50_calls"] == first["test50_calls"] == 0


def test_quota_matrix_covers_all_three_lanes_and_other_groups() -> None:
    matrix = quota_matrix()
    assert len(matrix["lane_cases"]) == 21
    for row in matrix["lane_cases"]:
        n = row["lane_count"]
        if n == 0:
            assert row["team_minibatch_status"] == "NO_ALIGNED_RESPONSIBILITY_TARGET_NOT_REACHED"
        elif n < 4:
            assert "requires exactly 4 unique repair examples" in row["team_minibatch_status"]
        else:
            assert row["team_minibatch_status"] == "CONSTRUCTIBLE"
    assert len(matrix["other_quota_cases"]) == 10
    assert all((row["status"] == "CONSTRUCTIBLE") == (row["available_extra_rows"] == 4)
               for row in matrix["other_quota_cases"])


def test_cross_product_counts_are_partitioned_and_deterministic() -> None:
    first = cross_product_grid()
    assert first == cross_product_grid()
    counts = first["counts"]
    assert counts["valid_grid_states"] == 128544
    assert counts["raw_positive_states"] == 127776
    assert counts["same_target"] + counts["different_target"] + counts["no_feasible_target"] == counts["raw_positive_states"]
    assert counts["selected_target_truncated"] + counts["selected_target_not_truncated"] == counts["same_target"] + counts["different_target"]
    assert sum(part["valid_grid_states"] for part in first["by_overlap_degree"].values()) == counts["valid_grid_states"]
    assert first["evidence_type"] == "SYNTHETIC_STRUCTURAL_COUNTERFACTUAL_ONLY"


def test_published_decision_counts_match_offline_analysis() -> None:
    report = Path(__file__).resolve().parents[1] / "reports/v3_scientific_blocker_decision_closure_20260927"
    schedule = json.loads((report / "schedule_capacity_analysis.json").read_text(encoding="utf-8"))
    quota = json.loads((report / "team_minibatch_feasibility_analysis.json").read_text(encoding="utf-8"))
    cross = json.loads((report / "cross_product_analysis.json").read_text(encoding="utf-8"))
    facts = json.loads((report / "fact_assertions.json").read_text(encoding="utf-8"))
    observed = initial_state_capacity()
    assert schedule["initial_state"]["packet_item_count"] == observed["search_packet_item_count"]
    assert schedule["initial_state"]["nominal_schedule_slots"] == observed["schedule_delivery_slots"]
    assert schedule["capacity_arithmetic"]["minimum_budget_for_nominal_root_packet_coverage"] == observed["minimum_metric_budget_for_coverage"]
    assert len(quota["lane_cases"]) == len(quota_matrix()["lane_cases"])
    assert cross["counts"]["valid_grid_states"] == cross_product_grid()["counts"]["valid_grid_states"]
    assert sum(facts["assertions"]["synthetic_target_partition"]) == facts["assertions"]["synthetic_positive_raw_target_states"]
    assert facts["final_gate"] == "DECISION_REQUIRED"


@pytest.mark.parametrize("primary, preservation, extra_hard", (
    (3, 4, 4), (4, 3, 4), (4, 4, 3), (4, 4, 4), (5, 4, 3),
))
def test_synthetic_quota_predicate_matches_real_selector(
    primary: int, preservation: int, extra_hard: int,
) -> None:
    def row(index: int, group: str) -> TeamEvidenceCase:
        return TeamEvidenceCase(
            example_id=f"synthetic-{index}", input_payload="synthetic",
            gold="A", target_output="B", feedback=None,
            evidence_group=group,
            tags=("repair", "direct_flip", "team_hard") if group == "repair" else (group,),
        )

    evidence = tuple(row(index, "repair") for index in range(primary))
    evidence += tuple(row(20 + index, "preservation") for index in range(preservation))
    evidence += tuple(row(40 + index, "team_hard") for index in range(extra_hard))
    expected = primary >= 4 and preservation >= 4 and primary + extra_hard >= 8
    try:
        selected = Layer2EvidenceRequestBuilder().select_team_minibatch(
            evidence, primary_responsibility_lane="direct_flip",
        )
    except ValueError:
        assert not expected
    else:
        assert expected and len(selected) == 12
