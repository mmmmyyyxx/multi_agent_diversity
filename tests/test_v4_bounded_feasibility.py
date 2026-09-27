"""Zero-provider V4 responsibility, quota and nominal-capacity invariants."""

from __future__ import annotations

import hashlib
import json

import pytest

from multi_dataset_diverse_rl.native_feed import CandidateTransitionAudit
from multi_dataset_diverse_rl.team_search.feasibility import (
    Layer2EvidenceInfeasible, Layer2FeasibilityReason,
)
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    PrimaryResponsibilityPersistentRealizabilityScheduler,
    PrimaryResponsibilitySummary,
    select_primary_responsibility_targets,
)
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase, TeamSearchAssignment, TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder
from multi_dataset_diverse_rl.versions import LAYER2_EVIDENCE_PACKET_V4_VERSION


PARENT = "Decide using the available context."
PARENT_HASH = hashlib.sha256(PARENT.encode("utf-8")).hexdigest()
REQUEST = TeamSearchRequest(81, 0, "parent-team", 36, "solver", "output")


def rows(*, lane: str = "direct_flip", repair: int = 5,
         preservation: int = 4, team_hard: int = 4) -> tuple[TeamEvidenceCase, ...]:
    return tuple(
        TeamEvidenceCase(f"{group}-{index:02d}", f"case {group} {index}",
                         "A", "B", None, group, (lane,) if group == "repair" else ())
        for group, count in (("repair", repair), ("preservation", preservation),
                             ("team_hard", team_hard))
        for index in range(count)
    )


def assignment(evidence: tuple[TeamEvidenceCase, ...], *,
               focus: int = 0, anchor: int = 0,
               lane: str = "direct_flip") -> TeamSearchAssignment:
    builder = Layer2EvidenceRequestBuilder(bounded_search_view=True)
    mini = builder.select_team_minibatch(evidence, primary_responsibility_lane=lane)
    transition = None
    if focus or anchor:
        ids = tuple(row.example_id for row in evidence)
        assert focus + anchor <= len(ids)
        parent = tuple((item, index < focus)
                       for index, item in enumerate(ids))
        child = tuple((item, False if index < focus else True if index < focus + anchor
                       else parent[index][1]) for index, item in enumerate(ids))
        transition = CandidateTransitionAudit(
            parent_candidate_hash="previous", child_candidate_hash=PARENT_HASH,
            parent_correctness=parent, child_correctness=child,
        )
    return TeamSearchAssignment(
        0, PARENT, evidence, "bounded search", "v4", tuple(row.example_id for row in mini),
        lane, 50.0, transition,
    )


@pytest.mark.parametrize("lane", ("direct_flip", "near_margin", "coverage"))
@pytest.mark.parametrize("count", range(6))
def test_primary_quota_is_exact_and_never_filled_from_other_lanes(lane: str, count: int) -> None:
    builder = Layer2EvidenceRequestBuilder(bounded_search_view=True)
    evidence = rows(lane=lane, repair=count)
    if count < 4:
        with pytest.raises(ValueError, match="requires exactly 4 unique repair examples"):
            builder.select_team_minibatch(evidence, primary_responsibility_lane=lane)
    else:
        selected = builder.select_team_minibatch(evidence, primary_responsibility_lane=lane)
        assert [row.evidence_group for row in selected] == (
            ["repair"] * 4 + ["preservation"] * 4 + ["team_hard"] * 4
        )
        assert len({row.example_id for row in selected}) == 12


@pytest.mark.parametrize("group", ("preservation", "team_hard"))
@pytest.mark.parametrize("count", range(6))
def test_global_quota_requires_four_additional_unique_ids(group: str, count: int) -> None:
    builder = Layer2EvidenceRequestBuilder(bounded_search_view=True)
    evidence = rows(**{group: count})
    if count < 4:
        with pytest.raises(ValueError, match=f"requires exactly 4 unique {group} examples"):
            builder.select_team_minibatch(evidence, primary_responsibility_lane="direct_flip")
    else:
        assert len(builder.select_team_minibatch(
            evidence, primary_responsibility_lane="direct_flip",
        )) == 12


@pytest.mark.parametrize("focus,anchor,expected", (
    (0, 0, 36), (10, 10, 16), (16, 16, 4), (17, 16, None),
))
def test_exact_transition_reservation_and_bounded_repair(
    focus: int, anchor: int, expected: int | None,
) -> None:
    evidence = rows(repair=50, preservation=20, team_hard=20)
    frozen = assignment(evidence, focus=focus, anchor=anchor)
    builder = Layer2EvidenceRequestBuilder(bounded_search_view=True)
    if expected is None:
        with pytest.raises(Layer2EvidenceInfeasible) as failure:
            builder.build(REQUEST, frozen)
        assert failure.value.reason is Layer2FeasibilityReason.INSUFFICIENT_PACKET_REPAIR_CAPACITY
        return
    packet = builder.build(REQUEST, frozen).packet
    assert packet.packet_version == LAYER2_EVIDENCE_PACKET_V4_VERSION
    assert len(packet.responsibility_examples) == expected
    assert len(packet.focus_examples) == focus
    assert len(packet.anchor_examples) == anchor
    assert len(packet.ordered_batch_schedule) == 12
    assert sum(map(len, packet.ordered_batch_schedule)) == 36
    assert {row.example_id for row in packet.local_eval_examples} == set(frozen.local_validation_example_ids)
    provenance = dict(packet.provenance)
    assert provenance["responsibility_universe_count"] == "50"
    full_ids = json.loads(provenance["responsibility_universe_ids_json"])
    scheduled_ids = json.loads(provenance["responsibility_scheduled_ids_json"])
    assert len(full_ids) == 50 and scheduled_ids == full_ids[:expected]


def test_feasibility_mask_preserves_raw_scores_and_skipped_failure_count() -> None:
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler(
        version="primary_responsibility_feasibility_constrained_v1"
    )
    summaries = tuple(PrimaryResponsibilitySummary(
        member_id=index, direct_count=0, near_margin_count=0,
        coverage_count=0, direct_score=0, near_margin_score=0,
        coverage_score=0, primary_lane="direct_flip", primary_score=score,
        runner_up_lane="near_margin", runner_up_score=0, failure_count=0,
        realizability=1.0, target_score=float(score),
    ) for index, score in enumerate((10, 8, 0, 0, 0)))
    choice = select_primary_responsibility_targets(
        summaries, seed=81, update_index=0, target_count=1,
        eligible_member_ids=(1,), scheduler_version=scheduler.version,
    )
    assert choice.selected_member_ids == (1,)
    assert [row.target_score for row in choice.summaries] == [10, 8, 0, 0, 0]
    scheduler.record_outcome(
        decision=choice, update_index=0, committed_member_id=None,
        valid_outcome=True,
    )
    assert scheduler.state.failure_count_by_member[0] == 0
    assert scheduler.state.failure_count_by_member[1] == 1
