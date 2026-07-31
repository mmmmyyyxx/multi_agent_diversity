from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
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
    overdue: bool
    target_pareto_front: int
    seeded_rank: str

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
    overdue_agent_ids: tuple[int, ...]
    catchup_eligible_agent_ids: tuple[int, ...]
    actual_candidate_agent_ids: tuple[int, ...]
    target_pareto_fronts: dict[int, int]
    target_frontier_agent_ids: tuple[int, ...]
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


def target_priorities(
    *,
    assignments: Mapping[
        int, Sequence[MemberAwareRepairOpportunity]
    ],
    state: ResponsibilityState,
    seed: int,
    max_wait_updates: int,
    current_member_correct_counts: Sequence[int],
    initial_member_correct_counts: Sequence[int],
    member_uplift_tolerance: int,
) -> tuple[ResponsibilityTargetPriority, ...]:
    if max_wait_updates <= 0:
        raise ValueError("max_wait_updates must be positive")
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
    portfolios = responsibility_portfolios(
        assignments=assignments,
        state=state,
    )
    rows: list[ResponsibilityTargetPriority] = []
    for agent_id, portfolio in portfolios.items():
        if not portfolio.residual_count:
            continue
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
                overdue=wait >= max_wait_updates,
                target_pareto_front=0,
                seeded_rank=rank,
            )
        )

    fronts = (
        _pareto_front_numbers(
            [row.agent_id for row in rows],
            {row.agent_id: row.target_values() for row in rows},
        )
        if rows
        else {}
    )
    return tuple(
        replace(
            row,
            target_pareto_front=fronts[row.agent_id],
        )
        for row in rows
    )


def build_target_selection_decision(
    priorities: Sequence[ResponsibilityTargetPriority],
    *,
    all_member_gains: Sequence[int],
    state: ResponsibilityState,
    max_wait_updates: int,
    member_uplift_tolerance: int,
    member_catchup_mode: str,
) -> TargetSelectionDecision:
    if member_catchup_mode not in {"off", "fallback_v1"}:
        raise ValueError(
            "member_catchup_mode must be 'off' or 'fallback_v1'"
        )

    eligible = tuple(priorities)
    overdue = tuple(row for row in eligible if row.overdue)
    eligible_ids = tuple(row.agent_id for row in eligible)
    overdue_ids = tuple(row.agent_id for row in overdue)

    if overdue:
        selected_row = min(
            overdue,
            key=lambda row: (
                -row.updates_since_selected,
                -row.direct_fix_count,
                -row.margin_gain_sum,
                -row.uplift_deficit,
                row.seeded_rank,
            ),
        )
        return TargetSelectionDecision(
            selected_agent_id=selected_row.agent_id,
            selection_pool_stage="responsibility_max_wait_override",
            update_lane="responsibility_conditioned",
            eligible_agent_ids=eligible_ids,
            overdue_agent_ids=overdue_ids,
            catchup_eligible_agent_ids=(),
            actual_candidate_agent_ids=overdue_ids,
            target_pareto_fronts={},
            target_frontier_agent_ids=(),
        )

    if eligible:
        frontier = tuple(
            row for row in eligible if row.target_pareto_front == 1
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
            overdue_agent_ids=(),
            catchup_eligible_agent_ids=(),
            actual_candidate_agent_ids=eligible_ids,
            target_pareto_fronts={
                row.agent_id: row.target_pareto_front for row in eligible
            },
            target_frontier_agent_ids=tuple(
                row.agent_id for row in frontier
            ),
        )

    gains = list(map(int, all_member_gains))
    maximum = max(gains, default=0)
    catchup = tuple(
        agent_id
        for agent_id in sorted(state.updates_since_selected_by_agent)
        if member_catchup_mode == "fallback_v1"
        and max(
            0,
            maximum
            - gains[agent_id]
            - int(member_uplift_tolerance),
        )
        > 0
        and state.updates_since_selected_by_agent[agent_id]
        >= max_wait_updates
    )
    if catchup:
        selected = min(
            catchup,
            key=lambda agent_id: (
                -state.updates_since_selected_by_agent[agent_id],
                -max(
                    0,
                    maximum
                    - gains[agent_id]
                    - int(member_uplift_tolerance),
                ),
                state.seeded_rank_by_agent.setdefault(
                    agent_id,
                    _seeded_hash(0, "target", agent_id),
                ),
            ),
        )
        return TargetSelectionDecision(
            selected_agent_id=selected,
            selection_pool_stage="no_actionable_responsibility",
            update_lane="generic_member_catchup",
            eligible_agent_ids=(),
            overdue_agent_ids=(),
            catchup_eligible_agent_ids=catchup,
            actual_candidate_agent_ids=(),
            target_pareto_fronts={},
            target_frontier_agent_ids=(),
            no_actionable_reason=(
                "catchup_extension_selected_after_"
                "no_actionable_responsibility"
            ),
        )

    return TargetSelectionDecision(
        selected_agent_id=None,
        selection_pool_stage="no_actionable_responsibility",
        update_lane="no_actionable_responsibility",
        eligible_agent_ids=(),
        overdue_agent_ids=(),
        catchup_eligible_agent_ids=(),
        actual_candidate_agent_ids=(),
        target_pareto_fronts={},
        target_frontier_agent_ids=(),
        no_actionable_reason="no_actionable_responsibility",
    )


def select_target_agent(
    priorities: Sequence[ResponsibilityTargetPriority],
    **kwargs: Any,
) -> int | None:
    if kwargs:
        return build_target_selection_decision(
            priorities,
            **kwargs,
        ).selected_agent_id
    overdue = tuple(row for row in priorities if row.overdue)
    if overdue:
        return min(
            overdue,
            key=lambda row: (
                -row.updates_since_selected,
                -row.direct_fix_count,
                -row.margin_gain_sum,
                -row.uplift_deficit,
                row.seeded_rank,
            ),
        ).agent_id
    frontier = tuple(
        row for row in priorities if row.target_pareto_front == 1
    )
    return min(
        frontier,
        key=lambda row: (
            -row.updates_since_selected,
            row.seeded_rank,
        ),
        default=None,
    ).agent_id if frontier else None
