"""Zero-provider probes for V3 pre-execution method feasibility."""

from __future__ import annotations

import pytest

from multi_dataset_diverse_rl.production_transfer_diagnostic import _diagnostic_stop_reason
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    build_primary_responsibility_summary_from_counts,
    select_primary_responsibility_targets,
)
from multi_dataset_diverse_rl.team_search.schemas import TeamEvidenceCase
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder
from scripts.prepare_online_transfer_diagnostic_v3 import frozen_payload


def _row(index: int, group: str, lane: str = "direct_flip") -> TeamEvidenceCase:
    return TeamEvidenceCase(
        example_id=f"synthetic-{index}",
        input_payload="synthetic input", gold="A", target_output="B",
        feedback=None, evidence_group=group,
        tags=("repair", lane, "team_hard") if group == "repair" else (group,),
    )


@pytest.mark.parametrize("lane", ("direct_flip", "near_margin", "coverage"))
@pytest.mark.parametrize("repair_count", (1, 2, 3))
def test_selected_primary_lane_with_fewer_than_four_repairs_aborts(
    lane: str, repair_count: int,
) -> None:
    evidence = tuple(_row(index, "repair", lane) for index in range(repair_count))
    evidence += tuple(_row(10 + index, "preservation") for index in range(4))
    evidence += tuple(_row(20 + index, "team_hard") for index in range(4))
    with pytest.raises(ValueError, match="requires exactly 4 unique repair examples"):
        Layer2EvidenceRequestBuilder().select_team_minibatch(
            evidence, primary_responsibility_lane=lane,
        )


@pytest.mark.parametrize("lane", ("direct_flip", "near_margin", "coverage"))
def test_four_repairs_are_sufficient_if_other_quotas_exist(lane: str) -> None:
    evidence = tuple(_row(index, "repair", lane) for index in range(4))
    evidence += tuple(_row(10 + index, "preservation") for index in range(4))
    evidence += tuple(_row(20 + index, "team_hard") for index in range(4))
    selected = Layer2EvidenceRequestBuilder().select_team_minibatch(
        evidence, primary_responsibility_lane=lane,
    )
    assert len(selected) == len({row.example_id for row in selected}) == 12
    assert [sum(row.evidence_group == group for row in selected)
            for group in ("repair", "preservation", "team_hard")] == [4, 4, 4]


@pytest.mark.parametrize("lane, counts", (
    ("direct_flip", (1, 0, 0)),
    ("near_margin", (0, 1, 0)),
    ("coverage", (0, 0, 1)),
))
def test_scheduler_can_select_a_lane_that_cannot_supply_quota(
    lane: str, counts: tuple[int, int, int],
) -> None:
    summaries = tuple(build_primary_responsibility_summary_from_counts(
        member_id=member, direct_count=counts[0] if member == 2 else 0,
        near_margin_count=counts[1] if member == 2 else 0,
        coverage_count=counts[2] if member == 2 else 0,
        failure_count=0,
    ) for member in range(5))
    decision = select_primary_responsibility_targets(
        summaries, seed=81, update_index=1, target_count=1,
    )
    assert decision.selected_member_ids == (2,)
    assert summaries[2].primary_lane == lane
    assert summaries[2].primary_score > 0
    evidence = (_row(0, "repair", lane),)
    evidence += tuple(_row(10 + index, "preservation") for index in range(4))
    evidence += tuple(_row(20 + index, "team_hard") for index in range(4))
    with pytest.raises(ValueError, match="requires exactly 4 unique repair examples"):
        Layer2EvidenceRequestBuilder().select_team_minibatch(
            evidence, primary_responsibility_lane=lane,
        )


@pytest.mark.parametrize("proposal_count", (15, 16, 17, 18, 19, 20))
def test_proposal_preopportunity_boundary(proposal_count: int) -> None:
    reason = _diagnostic_stop_reason(0, proposal_count)
    if proposal_count <= 16:
        assert reason is None
    elif proposal_count < 20:
        assert reason == "REFLECTION_PROPOSAL_PREOPPORTUNITY_GUARD"
    else:
        assert reason == "REFLECTION_PROPOSAL_CEILING_REACHED"


def test_v3_frozen_source_inventory_covers_online_dependency_paths() -> None:
    manifest, _ = frozen_payload(execution_source_sha="a" * 40)
    paths = set(manifest["execution"]["source_paths"])
    for name in (
        "system_runtime.py", "task_builder.py", "controller.py",
        "production_transfer_diagnostic.py", "execution_runtime.py",
        "native_feed.py", "gepa_native.py", "candidate_selection.py",
        "system.py",
    ):
        assert any(path.endswith("/" + name) for path in paths), name
    assert any(path.startswith("infrastructure/common_solver_contract_v1/") for path in paths)
