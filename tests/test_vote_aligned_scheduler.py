from __future__ import annotations

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.identity import config_fingerprint
from multi_dataset_diverse_rl.responsibility import MemberAwareRepairOpportunity
from multi_dataset_diverse_rl.vote_aligned_scheduler import (
    DIRECT_FLIP,
    NEAR_MARGIN,
    PURE_COVERAGE,
    classify_opportunity_lane,
    select_vote_aligned_targets,
)


def opportunity(
    agent: int,
    question: str,
    *,
    flip: int = 0,
    margin_gain: int = 1,
    coverage: bool = False,
) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(
        agent_id=agent,
        question_hash=question,
        vote_flip_gain=flip,
        margin_gain=margin_gain,
        member_error=True,
        coverage_opportunity=coverage,
        conversion_opportunity=not coverage,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def test_direct_flip_precedes_near_margin_and_coverage() -> None:
    result = select_vote_aligned_targets(
        assigned={
            0: [opportunity(0, "coverage", coverage=True)],
            1: [opportunity(1, "near")],
            2: [opportunity(2, "flip", flip=1)],
        },
        current_margin_by_question={"coverage": -2, "near": -1, "flip": 0},
        seed=75,
        update_index=0,
    )
    assert result.targets == (2, 1)
    assert [row["lane_selected"] for row in result.slot_decisions] == [
        DIRECT_FLIP,
        NEAR_MARGIN,
    ]


def test_near_margin_is_exactly_nonflip_post_correction_m_zero() -> None:
    result = select_vote_aligned_targets(
        assigned={
            0: [opportunity(0, "far", margin_gain=1)],
            1: [opportunity(1, "near", margin_gain=2)],
            2: [opportunity(2, "coverage", coverage=True)],
        },
        current_margin_by_question={"far": -2, "near": -2, "coverage": -3},
        seed=0,
        update_index=0,
    )
    assert result.member_lane_counts[0][NEAR_MARGIN] == 0
    assert result.member_lane_counts[1][NEAR_MARGIN] == 1
    assert result.targets[0] == 1


def test_pure_coverage_precedes_canonical_fallback() -> None:
    result = select_vote_aligned_targets(
        assigned={
            0: [opportunity(0, "far", margin_gain=1)],
            3: [opportunity(3, "coverage", coverage=True)],
        },
        current_margin_by_question={"far": -3, "coverage": -3},
        seed=0,
        update_index=0,
    )
    assert result.targets == (3, 0)
    assert result.slot_decisions[0]["lane_selected"] == PURE_COVERAGE
    assert result.slot_decisions[1]["lane_selected"] == "fallback_rr"


def test_lane_rr_is_stateful_distinct_and_deterministically_replayable() -> None:
    assigned = {
        agent: [opportunity(agent, f"flip-{agent}", flip=1)]
        for agent in range(4)
    }
    margins = {f"flip-{agent}": 0 for agent in range(4)}
    first = select_vote_aligned_targets(
        assigned=assigned,
        current_margin_by_question=margins,
        seed=75,
        update_index=0,
    )
    replay = select_vote_aligned_targets(
        assigned=assigned,
        current_margin_by_question=margins,
        seed=75,
        update_index=0,
    )
    second = select_vote_aligned_targets(
        assigned=assigned,
        current_margin_by_question=margins,
        seed=75,
        update_index=1,
        cursor_before=first.cursor_after,
    )
    assert first == replay
    assert len(set(first.targets)) == 2
    assert first.cursor_after[DIRECT_FLIP] == 2
    assert second.cursor_after[DIRECT_FLIP] == 4
    assert first.targets != second.targets


def test_target_scheduler_enters_run_fingerprint() -> None:
    baseline = Config.from_flat(target_scheduler="rr_generic")
    treatment = Config.from_flat(target_scheduler="vote_aligned_rr")
    assert config_fingerprint(baseline) != config_fingerprint(treatment)


def test_lane_telemetry_is_mutually_exclusive() -> None:
    margins = {"direct": -1, "near": -1, "pure": -2, "fallback": -3}
    assert classify_opportunity_lane(
        opportunity(0, "direct", flip=1, coverage=True), margins
    ) == DIRECT_FLIP
    assert classify_opportunity_lane(
        opportunity(0, "near", margin_gain=1, coverage=True), margins
    ) == NEAR_MARGIN
    assert classify_opportunity_lane(
        opportunity(0, "pure", margin_gain=1, coverage=True), margins
    ) == PURE_COVERAGE
    assert classify_opportunity_lane(
        opportunity(0, "fallback", margin_gain=1, coverage=False), margins
    ) is None


def test_exclusive_coverage_telemetry_preserves_hierarchy_selection() -> None:
    result = select_vote_aligned_targets(
        assigned={
            0: [opportunity(0, "direct-coverage", flip=1, coverage=True)],
            1: [opportunity(1, "near-coverage", coverage=True)],
            2: [opportunity(2, "pure-coverage", coverage=True)],
        },
        current_margin_by_question={
            "direct-coverage": -1,
            "near-coverage": -1,
            "pure-coverage": -3,
        },
        seed=75,
        update_index=0,
    )
    assert result.targets == (0, 1)
    assert [row["lane_selected"] for row in result.slot_decisions] == [
        DIRECT_FLIP,
        NEAR_MARGIN,
    ]
    assert result.member_lane_counts[0] == {
        DIRECT_FLIP: 1,
        NEAR_MARGIN: 0,
        PURE_COVERAGE: 0,
    }
    assert result.member_lane_counts[1] == {
        DIRECT_FLIP: 0,
        NEAR_MARGIN: 1,
        PURE_COVERAGE: 0,
    }
    assert result.member_lane_counts[2][PURE_COVERAGE] == 1
