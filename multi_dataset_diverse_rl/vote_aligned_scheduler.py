from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .responsibility import MemberAwareRepairOpportunity


VOTE_ALIGNED_SCHEDULER_VERSION = "hierarchical_lane_rr_direct_near_coverage_v1"
RR_GENERIC_SCHEDULER = "rr_generic"
VOTE_ALIGNED_RR_SCHEDULER = "vote_aligned_rr"

DIRECT_FLIP = "direct_flip"
NEAR_MARGIN = "near_margin"
PURE_COVERAGE = "pure_coverage"
FALLBACK_RR = "fallback_rr"
LANE_PRIORITY = (DIRECT_FLIP, NEAR_MARGIN, PURE_COVERAGE)


@dataclass(frozen=True)
class VoteAlignedTargetSelection:
    targets: tuple[int, ...]
    slot_decisions: tuple[dict[str, object], ...]
    member_lane_counts: dict[int, dict[str, int]]
    cursor_after: dict[str, int]


def _near_margin(
    row: MemberAwareRepairOpportunity,
    current_margin_by_question: Mapping[str, int],
) -> bool:
    """A non-flipping correction that lands exactly on the M=0 boundary."""
    if row.vote_flip_gain or row.margin_gain <= 0:
        return False
    if row.question_hash not in current_margin_by_question:
        raise KeyError(f"missing current plurality margin: {row.question_hash}")
    return int(current_margin_by_question[row.question_hash]) + int(row.margin_gain) == 0


def classify_opportunity_lane(
    row: MemberAwareRepairOpportunity,
    current_margin_by_question: Mapping[str, int],
) -> str | None:
    """Return the mutually exclusive telemetry lane used by the hierarchy."""
    if int(row.vote_flip_gain) > 0:
        return DIRECT_FLIP
    if _near_margin(row, current_margin_by_question):
        return NEAR_MARGIN
    if bool(row.coverage_opportunity):
        return PURE_COVERAGE
    return None


def member_lane_counts(
    assigned: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
    current_margin_by_question: Mapping[str, int],
) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for agent_id, rows in assigned.items():
        if not rows:
            continue
        lanes = [
            classify_opportunity_lane(row, current_margin_by_question)
            for row in rows
        ]
        result[int(agent_id)] = {
            DIRECT_FLIP: lanes.count(DIRECT_FLIP),
            NEAR_MARGIN: lanes.count(NEAR_MARGIN),
            PURE_COVERAGE: lanes.count(PURE_COVERAGE),
        }
    return result


def _canonical_rr_order(
    actionable: Sequence[int], *, seed: int, update_index: int, agent_count: int
) -> list[int]:
    pool = set(map(int, actionable))
    start = (int(seed) + 2 * int(update_index)) % int(agent_count)
    return [
        candidate
        for offset in range(agent_count)
        if (candidate := (start + offset) % agent_count) in pool
    ]


def select_vote_aligned_targets(
    *,
    assigned: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
    current_margin_by_question: Mapping[str, int],
    seed: int,
    update_index: int,
    cursor_before: Mapping[str, int] | None = None,
    target_count: int = 2,
    agent_count: int = 5,
) -> VoteAlignedTargetSelection:
    """Select distinct targets using lane hierarchy and lane-local RR only."""
    counts = member_lane_counts(assigned, current_margin_by_question)
    actionable = sorted(counts)
    cursors = {lane: int((cursor_before or {}).get(lane, 0)) for lane in LANE_PRIORITY}
    selected: list[int] = []
    decisions: list[dict[str, object]] = []

    for slot in range(min(int(target_count), len(actionable))):
        remaining = [agent for agent in actionable if agent not in selected]
        selected_lane = FALLBACK_RR
        candidates: list[int] = []
        for lane in LANE_PRIORITY:
            candidates = [agent for agent in remaining if counts[agent][lane] > 0]
            if candidates:
                selected_lane = lane
                break

        if selected_lane == FALLBACK_RR:
            ordered = _canonical_rr_order(
                remaining,
                seed=seed,
                update_index=update_index,
                agent_count=agent_count,
            )
            chosen = ordered[0]
            cursor_start = None
            cursor_end = None
            fallback_reason = "no_direct_flip_near_margin_or_pure_coverage"
            candidates = ordered
        else:
            candidates = sorted(candidates)
            cursor_start = cursors[selected_lane]
            chosen = candidates[(int(seed) + cursor_start) % len(candidates)]
            cursors[selected_lane] = cursor_start + 1
            cursor_end = cursors[selected_lane]
            fallback_reason = ""

        selected.append(chosen)
        decisions.append({
            "slot": slot + 1,
            "lane_selected": selected_lane,
            "candidate_member_ids": list(candidates),
            "selected_target_id": chosen,
            "rr_cursor_before": cursor_start,
            "rr_cursor_after": cursor_end,
            "fallback_reason": fallback_reason,
        })

    return VoteAlignedTargetSelection(
        targets=tuple(selected),
        slot_decisions=tuple(decisions),
        member_lane_counts=counts,
        cursor_after=cursors,
    )
