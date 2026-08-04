from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from .peer_state import PeerVoteContext, TeamVoteState, soft_vote_utility


@dataclass(frozen=True)
class MemberAwareRepairOpportunity:
    """A counterfactual repair opportunity for one member and one example."""

    agent_id: int
    question_hash: str
    vote_flip_gain: int
    margin_gain: int
    member_error: bool

    # Diagnostic and proposal-context fields only.
    coverage_opportunity: bool
    conversion_opportunity: bool
    dominant_wrong_member: bool
    unique_correct: bool
    pivotal_correct: bool
    oracle_soft_utility_gain: float


class RepairLane(str, Enum):
    COVERAGE = "coverage"
    DIRECT_FLIP = "direct_flip"
    MARGIN_SUPPORT = "margin_support"


@dataclass(frozen=True)
class ResidualServiceAssignment:
    question_hash: str
    repair_lane: RepairLane
    eligible_agent_ids: tuple[int, ...]
    active_eligible_agent_ids: tuple[int, ...]
    service_agent_id: int | None
    service_blocked_by_freeze: bool
    anchor_match_level: int
    lane_load_before: int
    total_load_before: int
    routing_seed_rank: int


@dataclass(frozen=True)
class ServiceRoutingState:
    assignments_by_question: dict[str, ResidualServiceAssignment]
    repair_lane_by_question: dict[str, RepairLane]
    service_portfolios: dict[int, tuple[MemberAwareRepairOpportunity, ...]]
    active_lane_by_agent: dict[int, RepairLane | None]
    active_slices: dict[int, tuple[MemberAwareRepairOpportunity, ...]]


@dataclass(frozen=True)
class CandidateMarginalContribution:
    vote_gain_count: int
    vote_loss_count: int
    net_vote_delta: int
    soft_utility_delta: float
    coverage_gain_count: int
    coverage_loss_count: int
    dominant_wrong_exit_count: int
    dominant_wrong_join_count: int
    assigned_residual_repair_count: int
    assigned_residual_utility_delta: float


@dataclass(frozen=True)
class ProtectionContribution:
    unique_correct_loss_count: int
    pivotal_correct_loss_count: int
    unique_correct_gain_count: int = 0
    pivotal_correct_gain_count: int = 0


@dataclass(frozen=True)
class MemberResponsibilityPortfolio:
    agent_id: int
    residuals: tuple[MemberAwareRepairOpportunity, ...]
    direct_fix_count: int
    margin_gain_sum: int

    # Diagnostic fields only.
    residual_count: int
    coverage_count: int
    conversion_count: int
    dominant_wrong_count: int


@dataclass(frozen=True)
class ResponsibilityTargetPriority:
    agent_id: int
    direct_fix_count: int
    margin_gain_sum: int
    member_gain: int
    maximum_member_gain: int
    uplift_deficit: int
    updates_since_selected: int
    frozen: bool
    target_pareto_front: int
    seeded_rank: str
    legal_portfolio_size: int = 0
    service_portfolio_size: int = 0
    active_lane: str | None = None
    active_lane_size: int = 0
    anchor: str | None = None

    def target_values(self) -> tuple[float, float, float]:
        return (
            float(self.direct_fix_count),
            float(self.margin_gain_sum),
            float(self.uplift_deficit),
        )


@dataclass(frozen=True)
class TargetSelectionDecision:
    selected_agent_id: int | None
    selection_pool_stage: str
    update_lane: str
    eligible_agent_ids: tuple[int, ...]
    frozen_agent_ids: tuple[int, ...]
    active_candidate_agent_ids: tuple[int, ...]
    target_pareto_fronts: dict[int, int]
    target_frontier_agent_ids: tuple[int, ...]
    selected_direct_fix_count: int | None = None
    selected_margin_gain_sum: int | None = None
    selected_uplift_deficit: int | None = None
    selected_updates_since_selected: int | None = None
    no_actionable_reason: str = ""


@dataclass
class ResponsibilityState:
    eligible_agents_by_question: dict[str, tuple[int, ...]] = field(
        default_factory=dict
    )
    updates_since_selected_by_agent: dict[int, int] = field(default_factory=dict)
    accepted_updates_by_agent: dict[int, int] = field(default_factory=dict)
    seeded_rank_by_agent: dict[int, str] = field(default_factory=dict)
    target_attempt_count_by_agent: dict[int, int] = field(default_factory=dict)
    consecutive_failed_updates_by_agent: dict[int, int] = field(default_factory=dict)
    last_failed_portfolio_signature_by_agent: dict[int, str] = field(default_factory=dict)
    frozen_by_agent: dict[int, bool] = field(default_factory=dict)
    frozen_portfolio_signature_by_agent: dict[int, str] = field(default_factory=dict)
    frozen_residual_hashes_by_agent: dict[int, tuple[str, ...]] = field(default_factory=dict)
    frozen_direct_fix_count_by_agent: dict[int, int] = field(default_factory=dict)
    frozen_margin_gain_sum_by_agent: dict[int, int] = field(default_factory=dict)
    other_accepted_updates_since_freeze_by_agent: dict[int, int] = field(default_factory=dict)
    freeze_count_by_agent: dict[int, int] = field(default_factory=dict)
    specialization_anchor_by_agent: dict[int, RepairLane | None] = field(
        default_factory=dict
    )


FREEZE_FAILURE_THRESHOLD = 2
FREEZE_OTHER_ACCEPT_THRESHOLD = 2
FREEZE_PORTFOLIO_OVERLAP_THRESHOLD = 0.8


def initialize_repairability_state(
    state: ResponsibilityState,
    agent_ids: Sequence[int],
) -> None:
    for agent_id in map(int, agent_ids):
        state.consecutive_failed_updates_by_agent.setdefault(agent_id, 0)
        state.last_failed_portfolio_signature_by_agent.setdefault(agent_id, "")
        state.frozen_by_agent.setdefault(agent_id, False)
        state.frozen_portfolio_signature_by_agent.setdefault(agent_id, "")
        state.frozen_residual_hashes_by_agent.setdefault(agent_id, ())
        state.frozen_direct_fix_count_by_agent.setdefault(agent_id, 0)
        state.frozen_margin_gain_sum_by_agent.setdefault(agent_id, 0)
        state.other_accepted_updates_since_freeze_by_agent.setdefault(agent_id, 0)
        state.freeze_count_by_agent.setdefault(agent_id, 0)
        state.specialization_anchor_by_agent.setdefault(agent_id, None)


def compute_member_aware_repair_opportunity(
    *,
    team_state: TeamVoteState,
    peer_context: PeerVoteContext,
    tau: float = 1.0,
) -> MemberAwareRepairOpportunity:
    target = peer_context.target_agent_id
    if peer_context.question_hash != team_state.question_hash:
        raise ValueError("team and peer context question hashes differ")

    current_correct = bool(team_state.team_correctness[target])
    fixed_gold_count = peer_context.peer_gold_vote_count + 1
    fixed_margin = fixed_gold_count - peer_context.peer_largest_wrong_vote_count
    fixed_vote_correct = fixed_margin > 0
    target_answer = team_state.team_answers[target]

    return MemberAwareRepairOpportunity(
        agent_id=target,
        question_hash=team_state.question_hash,
        vote_flip_gain=int(fixed_vote_correct) - int(team_state.vote_correct),
        margin_gain=int(fixed_margin - team_state.plurality_margin),
        member_error=not current_correct,
        coverage_opportunity=bool(
            not current_correct and peer_context.peer_gold_vote_count == 0
        ),
        conversion_opportunity=bool(
            not current_correct
            and not team_state.vote_correct
            and team_state.gold_vote_count > 0
        ),
        dominant_wrong_member=bool(
            not current_correct
            and bool(target_answer)
            and target_answer in team_state.dominant_wrong_answers
        ),
        unique_correct=bool(
            current_correct and peer_context.peer_gold_vote_count == 0
        ),
        pivotal_correct=bool(
            current_correct
            and team_state.vote_correct
            and peer_context.peer_margin <= 0
        ),
        oracle_soft_utility_gain=(
            soft_vote_utility(fixed_gold_count, fixed_margin, tau)
            - soft_vote_utility(
                team_state.gold_vote_count,
                team_state.plurality_margin,
                tau,
            )
        ),
    )


def repair_eligibility_key(
    row: MemberAwareRepairOpportunity,
) -> tuple[int, int]:
    return (int(row.vote_flip_gain), int(row.margin_gain))


def _seeded_hash(seed: int, question_hash: str, agent_id: int) -> str:
    return hashlib.sha256(
        f"{seed}:{question_hash}:{agent_id}".encode()
    ).hexdigest()


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(
        a >= b for a, b in zip(left, right, strict=True)
    ) and any(a > b for a, b in zip(left, right, strict=True))


def _pareto_front_numbers(
    identifiers: Sequence[int],
    values: Mapping[int, Sequence[float]],
) -> dict[int, int]:
    remaining = set(map(int, identifiers))
    fronts: dict[int, int] = {}
    number = 1
    while remaining:
        frontier = [
            item
            for item in sorted(remaining)
            if not any(
                other != item and _dominates(values[other], values[item])
                for other in remaining
            )
        ]
        if not frontier:
            raise AssertionError("Pareto front construction made no progress")
        for item in frontier:
            fronts[item] = number
            remaining.remove(item)
        number += 1
    return fronts


def compute_repair_eligibility_sets(
    *,
    team_states: Mapping[str, TeamVoteState],
    opportunities: Mapping[
        str, Sequence[MemberAwareRepairOpportunity]
    ],
    state: ResponsibilityState,
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[int, list[MemberAwareRepairOpportunity]],
    dict[str, dict[str, Any]],
]:
    """Choose per-residual members by lexicographic (vote flip, margin gain)."""

    agents = sorted(state.updates_since_selected_by_agent)
    eligible_by_question: dict[str, tuple[int, ...]] = {}
    portfolios = {agent: [] for agent in agents}
    audits: dict[str, dict[str, Any]] = {}

    for question_hash in sorted(team_states):
        team_state = team_states[question_hash]
        if team_state.vote_correct:
            continue
        wrong_rows = [
            row
            for row in opportunities.get(question_hash, ())
            if row.member_error
        ]
        if not wrong_rows:
            continue

        best_key = max(repair_eligibility_key(row) for row in wrong_rows)
        eligible_agents = tuple(
            sorted(
                row.agent_id
                for row in wrong_rows
                if repair_eligibility_key(row) == best_key
            )
        )
        eligible_by_question[question_hash] = eligible_agents
        eligible_set = set(eligible_agents)
        for row in wrong_rows:
            if row.agent_id in eligible_set:
                portfolios[row.agent_id].append(row)

        audits[question_hash] = {
            "vote_correct": False,
            "candidate_counterfactual_values": {
                str(row.agent_id): {
                    "vote_flip_gain": int(row.vote_flip_gain),
                    "margin_gain": int(row.margin_gain),
                }
                for row in sorted(wrong_rows, key=lambda item: item.agent_id)
            },
            "eligible_agent_ids": list(eligible_agents),
            "eligibility_tie_count": len(eligible_agents),
            "coverage_failure": team_state.gold_vote_count == 0,
            "conversion_failure": (
                not team_state.vote_correct
                and team_state.gold_vote_count > 0
            ),
        }

    state.eligible_agents_by_question = eligible_by_question
    return eligible_by_question, portfolios, audits


def repair_lane_for(
    team_state: TeamVoteState,
    opportunity: MemberAwareRepairOpportunity,
) -> RepairLane:
    """Assign exactly one program-defined lane to a vote-wrong residual."""

    if team_state.vote_correct:
        raise ValueError("repair lanes are defined only for vote-wrong residuals")
    if not opportunity.member_error:
        raise ValueError("repair lane opportunity must belong to a wrong member")
    if team_state.gold_vote_count == 0:
        return RepairLane.COVERAGE
    if opportunity.vote_flip_gain == 1:
        return RepairLane.DIRECT_FLIP
    return RepairLane.MARGIN_SUPPORT


def _anchor_match_level(
    anchor: RepairLane | None,
    lane: RepairLane,
) -> int:
    if anchor == lane:
        return 2
    if anchor is None:
        return 1
    return 0


def _routing_rank(
    seed: int,
    eligible_agent_ids: Sequence[int],
    state: ResponsibilityState,
) -> dict[int, int]:
    for agent_id in map(int, eligible_agent_ids):
        state.seeded_rank_by_agent.setdefault(
            agent_id, _seeded_hash(seed, "target", agent_id)
        )
    ordered = sorted(
        map(int, eligible_agent_ids),
        key=lambda agent_id: state.seeded_rank_by_agent[agent_id],
    )
    return {agent_id: index for index, agent_id in enumerate(ordered)}


def build_service_routing(
    *,
    team_states: Mapping[str, TeamVoteState],
    opportunities: Mapping[str, Sequence[MemberAwareRepairOpportunity]],
    eligible_agents_by_question: Mapping[str, Sequence[int]],
    state: ResponsibilityState,
    seed: int,
) -> ServiceRoutingState:
    """Route each serviceable residual to one legal, unfrozen member."""

    agent_ids = tuple(sorted(state.updates_since_selected_by_agent))
    initialize_repairability_state(state, agent_ids)
    lane_load: dict[tuple[int, RepairLane], int] = {}
    total_load = {agent_id: 0 for agent_id in agent_ids}
    portfolios: dict[int, list[MemberAwareRepairOpportunity]] = {
        agent_id: [] for agent_id in agent_ids
    }
    assignments: dict[str, ResidualServiceAssignment] = {}
    lanes: dict[str, RepairLane] = {}

    for question_hash in sorted(eligible_agents_by_question):
        eligible = tuple(sorted(map(int, eligible_agents_by_question[question_hash])))
        if not eligible:
            continue
        row_by_agent = {
            int(row.agent_id): row
            for row in opportunities.get(question_hash, ())
        }
        representative = row_by_agent[eligible[0]]
        lane = repair_lane_for(team_states[question_hash], representative)
        lanes[question_hash] = lane
        active = tuple(
            agent_id
            for agent_id in eligible
            if not state.frozen_by_agent.get(agent_id, False)
        )
        ranks = _routing_rank(seed, eligible, state)
        if not active:
            assignments[question_hash] = ResidualServiceAssignment(
                question_hash=question_hash,
                repair_lane=lane,
                eligible_agent_ids=eligible,
                active_eligible_agent_ids=(),
                service_agent_id=None,
                service_blocked_by_freeze=True,
                anchor_match_level=-1,
                lane_load_before=-1,
                total_load_before=-1,
                routing_seed_rank=-1,
            )
            continue

        service_agent_id = min(
            active,
            key=lambda agent_id: (
                -_anchor_match_level(
                    state.specialization_anchor_by_agent[agent_id], lane
                ),
                lane_load.get((agent_id, lane), 0),
                total_load[agent_id],
                ranks[agent_id],
            ),
        )
        assignment = ResidualServiceAssignment(
            question_hash=question_hash,
            repair_lane=lane,
            eligible_agent_ids=eligible,
            active_eligible_agent_ids=active,
            service_agent_id=service_agent_id,
            service_blocked_by_freeze=False,
            anchor_match_level=_anchor_match_level(
                state.specialization_anchor_by_agent[service_agent_id], lane
            ),
            lane_load_before=lane_load.get((service_agent_id, lane), 0),
            total_load_before=total_load[service_agent_id],
            routing_seed_rank=ranks[service_agent_id],
        )
        assignments[question_hash] = assignment
        portfolios[service_agent_id].append(row_by_agent[service_agent_id])
        lane_load[(service_agent_id, lane)] = assignment.lane_load_before + 1
        total_load[service_agent_id] += 1

    active_lanes: dict[int, RepairLane | None] = {}
    active_slices: dict[int, tuple[MemberAwareRepairOpportunity, ...]] = {}
    lane_rank = {
        RepairLane.DIRECT_FLIP: 2,
        RepairLane.COVERAGE: 1,
        RepairLane.MARGIN_SUPPORT: 0,
    }
    for agent_id in agent_ids:
        rows = tuple(portfolios[agent_id])
        grouped: dict[RepairLane, list[MemberAwareRepairOpportunity]] = {}
        for row in rows:
            lane = lanes[row.question_hash]
            grouped.setdefault(lane, []).append(row)
        anchor = state.specialization_anchor_by_agent[agent_id]
        if state.frozen_by_agent.get(agent_id, False) or not grouped:
            active_lane = None
        elif anchor is not None and grouped.get(anchor):
            active_lane = anchor
        else:
            active_lane = max(
                grouped,
                key=lambda item: (
                    sum(row.vote_flip_gain == 1 for row in grouped[item]),
                    sum(row.margin_gain for row in grouped[item]),
                    len(grouped[item]),
                    lane_rank[item],
                ),
            )
        active_lanes[agent_id] = active_lane
        active_slices[agent_id] = tuple(
            grouped.get(active_lane, ()) if active_lane is not None else ()
        )

    return ServiceRoutingState(
        assignments_by_question=assignments,
        repair_lane_by_question=lanes,
        service_portfolios={
            agent_id: tuple(rows) for agent_id, rows in portfolios.items()
        },
        active_lane_by_agent=active_lanes,
        active_slices=active_slices,
    )


def responsibility_portfolios(
    *,
    assignments: Mapping[
        int, Sequence[MemberAwareRepairOpportunity]
    ],
    state: ResponsibilityState,
) -> dict[int, MemberResponsibilityPortfolio]:
    result: dict[int, MemberResponsibilityPortfolio] = {}
    for agent_id in sorted(state.updates_since_selected_by_agent):
        rows = tuple(assignments.get(agent_id, ()))
        result[agent_id] = MemberResponsibilityPortfolio(
            agent_id=agent_id,
            residuals=rows,
            direct_fix_count=sum(
                int(row.vote_flip_gain > 0) for row in rows
            ),
            margin_gain_sum=sum(int(row.margin_gain) for row in rows),
            residual_count=len(rows),
            coverage_count=sum(
                int(row.coverage_opportunity) for row in rows
            ),
            conversion_count=sum(
                int(row.conversion_opportunity) for row in rows
            ),
            dominant_wrong_count=sum(
                int(row.dominant_wrong_member) for row in rows
            ),
        )
    return result


def responsibility_portfolio_signature(
    portfolio: MemberResponsibilityPortfolio,
) -> str:
    payload = {
        "residuals": sorted(
            (
                row.question_hash,
                int(row.vote_flip_gain),
                int(row.margin_gain),
            )
            for row in portfolio.residuals
        ),
        "direct_fix_count": int(portfolio.direct_fix_count),
        "margin_gain_sum": int(portfolio.margin_gain_sum),
        "residual_count": int(portfolio.residual_count),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def responsibility_portfolio_overlap(
    frozen_residual_hashes: Sequence[str],
    current_residual_hashes: Sequence[str],
) -> float:
    frozen = set(map(str, frozen_residual_hashes))
    current = set(map(str, current_residual_hashes))
    union = frozen | current
    if not union:
        return 1.0
    return len(frozen & current) / len(union)


def record_target_update_failure(
    *,
    state: ResponsibilityState,
    portfolio: MemberResponsibilityPortfolio,
    update_index: int,
) -> dict[str, Any] | None:
    """Record one complete semantic search failure and freeze on repetition."""

    agent_id = int(portfolio.agent_id)
    initialize_repairability_state(state, (agent_id,))
    signature = responsibility_portfolio_signature(portfolio)
    if state.last_failed_portfolio_signature_by_agent[agent_id] == signature:
        streak = state.consecutive_failed_updates_by_agent[agent_id] + 1
    else:
        streak = 1
    state.consecutive_failed_updates_by_agent[agent_id] = streak
    state.last_failed_portfolio_signature_by_agent[agent_id] = signature
    if streak < FREEZE_FAILURE_THRESHOLD or state.frozen_by_agent[agent_id]:
        return None

    residual_hashes = tuple(sorted(row.question_hash for row in portfolio.residuals))
    state.frozen_by_agent[agent_id] = True
    state.frozen_portfolio_signature_by_agent[agent_id] = signature
    state.frozen_residual_hashes_by_agent[agent_id] = residual_hashes
    state.frozen_direct_fix_count_by_agent[agent_id] = int(portfolio.direct_fix_count)
    state.frozen_margin_gain_sum_by_agent[agent_id] = int(portfolio.margin_gain_sum)
    state.other_accepted_updates_since_freeze_by_agent[agent_id] = 0
    state.freeze_count_by_agent[agent_id] += 1
    return {
        "artifact_schema_version": "repairability_freeze_event_v1",
        "agent_id": agent_id,
        "update_index": int(update_index),
        "failure_streak": streak,
        "portfolio_signature_hash": signature,
        "D": int(portfolio.direct_fix_count),
        "S": int(portfolio.margin_gain_sum),
        "residual_count": int(portfolio.residual_count),
        "freeze_reason": "same_responsibility_state_complete_failure_threshold",
    }


def record_target_update_acceptance(
    *,
    state: ResponsibilityState,
    accepted_agent_id: int,
) -> None:
    accepted_agent_id = int(accepted_agent_id)
    initialize_repairability_state(
        state, state.updates_since_selected_by_agent
    )
    state.consecutive_failed_updates_by_agent[accepted_agent_id] = 0
    state.last_failed_portfolio_signature_by_agent[accepted_agent_id] = ""
    for agent_id, frozen in state.frozen_by_agent.items():
        if frozen and agent_id != accepted_agent_id:
            state.other_accepted_updates_since_freeze_by_agent[agent_id] += 1


def record_specialization_anchor_acceptance(
    *,
    state: ResponsibilityState,
    accepted_agent_id: int,
    active_lane: RepairLane,
    update_index: int,
) -> dict[str, Any]:
    """Set or switch the target anchor only after an accepted update."""

    accepted_agent_id = int(accepted_agent_id)
    initialize_repairability_state(state, (accepted_agent_id,))
    old_anchor = state.specialization_anchor_by_agent[accepted_agent_id]
    state.specialization_anchor_by_agent[accepted_agent_id] = active_lane
    return {
        "artifact_schema_version": "specialization_anchor_event_v1",
        "update_index": int(update_index),
        "agent_id": accepted_agent_id,
        "old_anchor": old_anchor.value if old_anchor is not None else None,
        "active_lane": active_lane.value,
        "new_anchor": active_lane.value,
        "event": (
            "accepted_anchor_set"
            if old_anchor is None or old_anchor == active_lane
            else "accepted_anchor_switch"
        ),
    }


def refresh_frozen_member_states(
    *,
    state: ResponsibilityState,
    assignments: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
    update_index: int,
) -> list[dict[str, Any]]:
    """Unfreeze only after other accepted updates and material portfolio change."""

    initialize_repairability_state(
        state, state.updates_since_selected_by_agent
    )
    portfolios = responsibility_portfolios(assignments=assignments, state=state)
    events: list[dict[str, Any]] = []
    for agent_id in sorted(state.frozen_by_agent):
        if not state.frozen_by_agent[agent_id]:
            continue
        portfolio = portfolios[agent_id]
        current_hashes = tuple(
            sorted(row.question_hash for row in portfolio.residuals)
        )
        overlap = responsibility_portfolio_overlap(
            state.frozen_residual_hashes_by_agent[agent_id], current_hashes
        )
        accepted = state.other_accepted_updates_since_freeze_by_agent[agent_id]
        old_direct = state.frozen_direct_fix_count_by_agent[agent_id]
        material_change = (
            overlap < FREEZE_PORTFOLIO_OVERLAP_THRESHOLD
            or portfolio.direct_fix_count != old_direct
        )
        if accepted < FREEZE_OTHER_ACCEPT_THRESHOLD or not material_change:
            continue
        events.append({
            "artifact_schema_version": "repairability_unfreeze_event_v1",
            "agent_id": agent_id,
            "update_index": int(update_index),
            "other_accepted_updates": int(accepted),
            "portfolio_jaccard": float(overlap),
            "D_before": int(old_direct),
            "D_after": int(portfolio.direct_fix_count),
            "unfreeze_reason": "other_accepts_and_material_portfolio_change",
            "old_anchor": (
                state.specialization_anchor_by_agent[agent_id].value
                if state.specialization_anchor_by_agent[agent_id] is not None
                else None
            ),
            "anchor_cleared": True,
        })
        state.frozen_by_agent[agent_id] = False
        state.consecutive_failed_updates_by_agent[agent_id] = 0
        state.last_failed_portfolio_signature_by_agent[agent_id] = ""
        state.frozen_portfolio_signature_by_agent[agent_id] = ""
        state.frozen_residual_hashes_by_agent[agent_id] = ()
        state.frozen_direct_fix_count_by_agent[agent_id] = 0
        state.frozen_margin_gain_sum_by_agent[agent_id] = 0
        state.other_accepted_updates_since_freeze_by_agent[agent_id] = 0
        state.specialization_anchor_by_agent[agent_id] = None
    return events


def target_priorities(
    *,
    assignments: Mapping[
        int, Sequence[MemberAwareRepairOpportunity]
    ],
    state: ResponsibilityState,
    seed: int,
    current_member_correct_counts: Sequence[int],
    initial_member_correct_counts: Sequence[int],
    member_uplift_tolerance: int,
    legal_assignments: Mapping[
        int, Sequence[MemberAwareRepairOpportunity]
    ] | None = None,
    service_portfolios: Mapping[
        int, Sequence[MemberAwareRepairOpportunity]
    ] | None = None,
    active_lane_by_agent: Mapping[int, RepairLane | None] | None = None,
) -> tuple[ResponsibilityTargetPriority, ...]:
    if member_uplift_tolerance < 0:
        raise ValueError("member_uplift_tolerance cannot be negative")
    if len(current_member_correct_counts) != len(
        initial_member_correct_counts
    ):
        raise ValueError("member count vectors differ")

    gains = [
        int(current) - int(initial)
        for current, initial in zip(
            current_member_correct_counts,
            initial_member_correct_counts,
            strict=True,
        )
    ]
    maximum = max(gains, default=0)
    active_portfolios = responsibility_portfolios(
        assignments=assignments,
        state=state,
    )
    legal_portfolios = responsibility_portfolios(
        assignments=legal_assignments or assignments,
        state=state,
    )
    service_portfolio_rows = service_portfolios or assignments
    initialize_repairability_state(state, legal_portfolios)
    rows: list[ResponsibilityTargetPriority] = []
    for agent_id, legal_portfolio in legal_portfolios.items():
        if not legal_portfolio.residual_count:
            continue
        portfolio = active_portfolios[agent_id]
        service_size = len(service_portfolio_rows.get(agent_id, ()))
        active_lane = (
            active_lane_by_agent.get(agent_id)
            if active_lane_by_agent is not None
            else None
        )
        wait = state.updates_since_selected_by_agent[agent_id]
        rank = state.seeded_rank_by_agent.setdefault(
            agent_id,
            _seeded_hash(seed, "target", agent_id),
        )
        rows.append(
            ResponsibilityTargetPriority(
                agent_id=agent_id,
                direct_fix_count=portfolio.direct_fix_count,
                margin_gain_sum=portfolio.margin_gain_sum,
                member_gain=gains[agent_id],
                maximum_member_gain=maximum,
                uplift_deficit=max(
                    0,
                    maximum
                    - gains[agent_id]
                    - int(member_uplift_tolerance),
                ),
                updates_since_selected=wait,
                frozen=state.frozen_by_agent[agent_id],
                target_pareto_front=0,
                seeded_rank=rank,
                legal_portfolio_size=legal_portfolio.residual_count,
                service_portfolio_size=service_size,
                active_lane=(active_lane.value if active_lane else None),
                active_lane_size=portfolio.residual_count,
                anchor=(
                    state.specialization_anchor_by_agent[agent_id].value
                    if state.specialization_anchor_by_agent[agent_id]
                    else None
                ),
            )
        )

    active_rows = [
        row for row in rows
        if not row.frozen and row.service_portfolio_size and row.active_lane_size
    ]
    fronts = (
        _pareto_front_numbers(
            [row.agent_id for row in active_rows],
            {row.agent_id: row.target_values() for row in active_rows},
        )
        if active_rows
        else {}
    )
    return tuple(
        replace(
            row,
            target_pareto_front=fronts.get(row.agent_id, 0),
        )
        for row in rows
    )


def build_target_selection_decision(
    priorities: Sequence[ResponsibilityTargetPriority],
) -> TargetSelectionDecision:
    eligible = tuple(priorities)
    eligible_ids = tuple(row.agent_id for row in eligible)
    frozen_ids = tuple(row.agent_id for row in eligible if row.frozen)
    active = tuple(
        row for row in eligible
        if not row.frozen and row.service_portfolio_size and row.active_lane_size
    )
    active_ids = tuple(row.agent_id for row in active)
    if active:
        frontier = tuple(
            row for row in active if row.target_pareto_front == 1
        )
        selected_row = min(
            frontier,
            key=lambda row: (
                -row.updates_since_selected,
                row.seeded_rank,
            ),
        )
        return TargetSelectionDecision(
            selected_agent_id=selected_row.agent_id,
            selection_pool_stage="responsibility_joint_pareto",
            update_lane="responsibility_conditioned",
            eligible_agent_ids=eligible_ids,
            frozen_agent_ids=frozen_ids,
            active_candidate_agent_ids=active_ids,
            target_pareto_fronts={
                row.agent_id: row.target_pareto_front for row in active
            },
            target_frontier_agent_ids=tuple(
                row.agent_id for row in frontier
            ),
            selected_direct_fix_count=selected_row.direct_fix_count,
            selected_margin_gain_sum=selected_row.margin_gain_sum,
            selected_uplift_deficit=selected_row.uplift_deficit,
            selected_updates_since_selected=selected_row.updates_since_selected,
        )
    if eligible:
        return TargetSelectionDecision(
            selected_agent_id=None,
            selection_pool_stage="no_actionable_repairability",
            update_lane="no_actionable_repairability",
            eligible_agent_ids=eligible_ids,
            frozen_agent_ids=frozen_ids,
            active_candidate_agent_ids=(),
            target_pareto_fronts={},
            target_frontier_agent_ids=(),
            no_actionable_reason="no_actionable_repairability",
        )

    return TargetSelectionDecision(
        selected_agent_id=None,
        selection_pool_stage="no_actionable_responsibility",
        update_lane="no_actionable_responsibility",
        eligible_agent_ids=(),
        frozen_agent_ids=(),
        active_candidate_agent_ids=(),
        target_pareto_fronts={},
        target_frontier_agent_ids=(),
        no_actionable_reason="no_actionable_responsibility",
    )


def select_target_agent(
    priorities: Sequence[ResponsibilityTargetPriority],
) -> int | None:
    frontier = tuple(
        row for row in priorities
        if (
            not row.frozen
            and row.service_portfolio_size
            and row.active_lane_size
            and row.target_pareto_front == 1
        )
    )
    return min(
        frontier,
        key=lambda row: (
            -row.updates_since_selected,
            row.seeded_rank,
        ),
        default=None,
    ).agent_id if frontier else None
