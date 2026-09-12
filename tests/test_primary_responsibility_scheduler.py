from __future__ import annotations

from dataclasses import asdict

import pytest

from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPAOptimizerConfig
from multi_dataset_diverse_rl.responsibility import MemberAwareRepairOpportunity
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    COVERAGE,
    DIRECT_FLIP,
    FALLBACK,
    NEAR_MARGIN,
    PrimaryResponsibilityPersistentRealizabilityScheduler,
    build_primary_responsibility_summaries,
    select_primary_responsibility_targets,
)
from multi_dataset_diverse_rl.team_search.protocol import TeamSearchContract
from multi_dataset_diverse_rl.team_search.schemas import TeamEvidenceCase, TeamSearchAssignment, TeamSearchRequest
from multi_dataset_diverse_rl.team_search.task_builder import LocalTaskBuilder


def opportunity(kind: str, index: int, agent: int = 0) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(
        agent_id=agent,
        question_hash=f"q-{agent}-{kind}-{index}",
        vote_flip_gain=1 if kind == DIRECT_FLIP else 0,
        margin_gain=1 if kind == NEAR_MARGIN else 0,
        member_error=True,
        coverage_opportunity=kind == COVERAGE,
        conversion_opportunity=False,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def assigned(d: int, n: int, c: int, agent: int = 0):
    rows = [opportunity(DIRECT_FLIP, i, agent) for i in range(d)]
    rows += [opportunity(NEAR_MARGIN, i, agent) for i in range(n)]
    rows += [opportunity(COVERAGE, i, agent) for i in range(c)]
    margins = {row.question_hash: (-1 if row.margin_gain else -2) for row in rows}
    return {agent: tuple(rows)}, margins


def summary(d: int, n: int, c: int, failures: int = 0):
    rows, margins = assigned(d, n, c)
    return build_primary_responsibility_summaries(
        assigned=rows,
        current_margin_by_question=margins,
        failure_count_by_member={0: failures},
        member_ids=(0,),
    )[0]


def test_weighting_and_primary_lane() -> None:
    row = summary(2, 3, 5)
    assert (row.direct_score, row.near_margin_score, row.coverage_score) == (8, 6, 5)
    assert (row.primary_lane, row.primary_score) == (DIRECT_FLIP, 8)


def test_lower_lane_can_beat_isolated_direct() -> None:
    a = summary(1, 0, 0)
    rows, margins = assigned(0, 3, 0, 1)
    b = build_primary_responsibility_summaries(
        assigned=rows,
        current_margin_by_question=margins,
        failure_count_by_member={},
        member_ids=(1,),
    )[0]
    assert b.primary_score == 6 > a.primary_score == 4


def test_primary_is_max_not_sum_and_tie_prefers_direct() -> None:
    row = summary(1, 2, 3)
    assert (row.direct_score, row.near_margin_score, row.coverage_score) == (4, 4, 3)
    assert row.primary_score == 4
    assert row.primary_lane == DIRECT_FLIP
    assert row.runner_up_lane == NEAR_MARGIN


def test_target_score_applies_persistent_failure_discount() -> None:
    row = summary(3, 0, 0, failures=2)
    assert row.primary_score == 12
    assert row.realizability == pytest.approx(1 / 3)
    assert row.target_score == pytest.approx(4)


def test_selected_failure_persists_across_teammate_commit_and_own_commit_resets() -> None:
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler(member_ids=(0, 1, 2))
    assignments: dict[int, tuple[MemberAwareRepairOpportunity, ...]] = {}
    margins: dict[str, int] = {}
    for member in (0, 1, 2):
        rows, row_margins = assigned(1, 0, 0, member)
        assignments.update(rows)
        margins.update(row_margins)

    first = scheduler.select(
        assigned=assignments, current_margin_by_question=margins, seed=1, update_index=0
    )
    assert len(first.selected_member_ids) == 2
    failed_member, teammate = first.selected_member_ids
    scheduler.record_outcome(
        decision=first,
        update_index=0,
        committed_member_id=teammate,
        valid_outcome=True,
    )
    assert scheduler.state.failure_count_by_member[failed_member] == 1
    assert scheduler.state.failure_count_by_member[teammate] == 0

    second = scheduler.select(
        assigned=assignments, current_margin_by_question=margins, seed=1, update_index=1
    )
    # Force a frozen target decision containing the failed member to isolate reset semantics.
    second = type(second)(
        second.scheduler_version,
        second.summaries,
        (failed_member, teammate),
        second.rr_cursor_before,
        second.rr_cursor_after,
        second.fallback_used,
    )
    scheduler.record_outcome(
        decision=second,
        update_index=1,
        committed_member_id=failed_member,
        valid_outcome=True,
    )
    assert scheduler.state.failure_count_by_member[failed_member] == 0
    assert scheduler.state.failure_count_by_member[teammate] == 1


def test_unselected_and_operational_abort_are_unchanged() -> None:
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler(member_ids=(0, 1, 2))
    scheduler.state.failure_count_by_member.update({0: 2, 1: 0, 2: 3})
    assignments, margins = assigned(1, 0, 0, 0)
    decision = scheduler.select(
        assigned=assignments, current_margin_by_question=margins, seed=0, update_index=0
    )
    before = dict(scheduler.state.failure_count_by_member)
    scheduler.record_outcome(
        decision=decision,
        update_index=0,
        committed_member_id=None,
        valid_outcome=False,
    )
    assert scheduler.state.failure_count_by_member == before


def test_all_zero_uses_existing_deterministic_fallback_order() -> None:
    summaries = build_primary_responsibility_summaries(
        assigned={}, current_margin_by_question={}, failure_count_by_member={}
    )
    first = select_primary_responsibility_targets(
        summaries, seed=3, update_index=4, rr_cursor=9
    )
    replay = select_primary_responsibility_targets(
        summaries, seed=3, update_index=4, rr_cursor=9
    )
    assert first == replay
    assert first.fallback_used
    assert first.selected_member_ids == (1, 2)  # (seed + 2*update) % 5
    assert all(row.primary_lane == FALLBACK for row in summaries)


def test_primary_lane_filters_only_responsibility_evidence() -> None:
    evidence = (
        TeamEvidenceCase("d", "payload", "A", None, None, "responsibility", (DIRECT_FLIP,)),
        TeamEvidenceCase("n", "payload", "A", None, None, "responsibility", (NEAR_MARGIN,)),
        TeamEvidenceCase("p", "payload", "A", None, None, "preservation", ()),
        TeamEvidenceCase("c", "payload", "A", None, None, "coalition", ()),
    )
    assignment = TeamSearchAssignment(
        0,
        "parent",
        evidence,
        "context",
        "r",
        local_validation_example_ids=("d", "n", "p", "c"),
        primary_responsibility_lane=NEAR_MARGIN,
    )
    request = TeamSearchRequest(1, 2, "team", 10, "solver", "output")
    task = LocalTaskBuilder().build(request, assignment)
    assert {row.example_id for row in task.search_examples} == {"n", "p", "c"}
    assert {row.example_id for row in task.local_validation_examples} == {"n", "p", "c"}
    assert "primary_responsibility_lane=near_margin" in task.optimization_context


def test_only_team_contract_changes_for_opt_in_scheduler() -> None:
    local_hash = GEPAOptimizerConfig().identity()
    current = TeamSearchContract()
    experimental = TeamSearchContract(
        target_policy="primary_responsibility_persistent_realizability_v1"
    )
    assert local_hash == "3c83562ab77a18fa621d853195007f2068cdb5ec7beeea7ff86bc7fcaef1e046"
    assert current.identity() == "5fb0dc2858ea237e51ada48fa2556fbcd4171a51749696a7d5d8e99a509728d4"
    assert experimental.identity() != current.identity()
    assert asdict(current) | {"target_policy": experimental.target_policy} == asdict(experimental)
