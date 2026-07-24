from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .peer_state import PeerVoteContext, TeamVoteState, soft_vote_utility


@dataclass(frozen=True)
class MemberAwareRepairOpportunity:
    agent_id: int
    question_hash: str
    current_correct: bool
    current_invalid: bool
    direct_vote_fix: bool
    oracle_soft_utility_gain: float
    coverage_opportunity: bool
    dominant_wrong_member: bool
    unique_correct: bool
    pivotal_correct: bool

    initial_correct_count: int
    current_correct_count: int
    gain_count: int
    improvement_need: int
    unique_correct_count: int
    pivotal_correct_count: int

    member_error: bool
    protection_need_count: int


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


@dataclass(frozen=True)
class AgentTargetPriority:
    agent_id: int
    individual_error_count: int
    assigned_load: int

    direct_vote_fix_count: int
    oracle_soft_utility_gain_sum: float
    coverage_opportunity_count: int
    dominant_wrong_count: int

    gain_count: int
    current_correct_count: int
    best_current_correct_count: int
    headroom_to_best: int
    unimproved: bool
    improvement_need: int
    unique_correct_count: int
    pivotal_correct_count: int

    updates_since_selected: int
    overdue: bool
    pareto_front: int
    seeded_rank: str
    best_observed_target_gain: int = 0
    no_positive_candidate_streak: int = 0
    next_regular_eligible_update: int = 0
    cooling_down: bool = False
    target_attempt_count: int = 0

    def pareto_values(self) -> tuple[float, ...]:
        return (
            float(self.headroom_to_best),
            float(max(0, self.best_observed_target_gain)),
            float(self.improvement_need),
            float(self.direct_vote_fix_count),
            float(self.oracle_soft_utility_gain_sum),
            float(self.coverage_opportunity_count),
            float(-self.no_positive_candidate_streak),
        )

    @property
    def protection_risk(self) -> int:
        return self.unique_correct_count + self.pivotal_correct_count


@dataclass
class ResponsibilityState:
    primary_owner_by_question: dict[str, int] = field(default_factory=dict)
    owner_age_by_question: dict[str, int] = field(default_factory=dict)
    assigned_load_by_agent: dict[int, int] = field(default_factory=dict)
    updates_since_selected_by_agent: dict[int, int] = field(default_factory=dict)
    accepted_updates_by_agent: dict[int, int] = field(default_factory=dict)
    seeded_rank_by_agent: dict[int, str] = field(default_factory=dict)
    best_observed_target_gain_by_agent: dict[int, int] = field(default_factory=dict)
    no_positive_candidate_streak_by_agent: dict[int, int] = field(default_factory=dict)
    next_regular_eligible_update_by_agent: dict[int, int] = field(default_factory=dict)
    target_attempt_count_by_agent: dict[int, int] = field(default_factory=dict)


def compute_member_aware_repair_opportunity(
    *,
    team_state: TeamVoteState,
    peer_context: PeerVoteContext,
    initial_correct_counts: Sequence[int],
    member_correct_counts: Sequence[int],
    member_gains_from_initial: Sequence[int],
    unique_correct_counts: Sequence[int] | None = None,
    pivotal_correct_counts: Sequence[int] | None = None,
    tau: float = 1.0,
) -> MemberAwareRepairOpportunity:
    target = peer_context.target_agent_id
    if peer_context.question_hash != team_state.question_hash:
        raise ValueError("team and peer context question hashes differ")
    member_count = len(team_state.team_correctness)
    if len(initial_correct_counts) != member_count:
        raise ValueError("initial member count vector does not match team size")
    if len(member_correct_counts) != member_count:
        raise ValueError("member count vector does not match team size")
    if len(member_gains_from_initial) != len(member_correct_counts):
        raise ValueError("member gain vector does not match member count vector")
    if unique_correct_counts is None:
        unique_correct_counts = (0,) * member_count
    if pivotal_correct_counts is None:
        pivotal_correct_counts = (0,) * member_count
    if (
        len(unique_correct_counts) != member_count
        or len(pivotal_correct_counts) != member_count
    ):
        raise ValueError("member protection count vector does not match team size")
    current_correct = bool(team_state.team_correctness[target])
    current_invalid = not bool(team_state.team_validity[target])
    fixed_gold = peer_context.peer_gold_vote_count + 1
    fixed_margin = fixed_gold - peer_context.peer_largest_wrong_vote_count
    target_answer = team_state.team_answers[target]
    total_gain = sum(int(value) for value in member_gains_from_initial)
    member_gain = int(member_gains_from_initial[target])
    current_count = int(member_correct_counts[target])
    member_count_n = len(member_correct_counts)
    return MemberAwareRepairOpportunity(
        agent_id=target,
        question_hash=team_state.question_hash,
        current_correct=current_correct,
        current_invalid=current_invalid,
        direct_vote_fix=bool(not team_state.vote_correct and fixed_margin > 0),
        oracle_soft_utility_gain=(
            soft_vote_utility(fixed_gold, fixed_margin, tau)
            - soft_vote_utility(team_state.gold_vote_count, team_state.plurality_margin, tau)
        ),
        coverage_opportunity=bool(not current_correct and peer_context.peer_gold_vote_count == 0),
        dominant_wrong_member=bool(
            not current_correct
            and bool(target_answer)
            and target_answer in team_state.dominant_wrong_answers
        ),
        unique_correct=bool(current_correct and peer_context.peer_gold_vote_count == 0),
        pivotal_correct=bool(
            current_correct and team_state.vote_correct and peer_context.peer_margin <= 0
        ),
        initial_correct_count=int(initial_correct_counts[target]),
        current_correct_count=current_count,
        gain_count=member_gain,
        improvement_need=max(0, total_gain - member_count_n * member_gain),
        unique_correct_count=int(unique_correct_counts[target]),
        pivotal_correct_count=int(pivotal_correct_counts[target]),
        member_error=not current_correct,
        protection_need_count=(
            int(unique_correct_counts[target]) + int(pivotal_correct_counts[target])
        ),
    )


def _seeded_hash(seed: int, question_hash: str, agent_id: int) -> str:
    return hashlib.sha256(f"{seed}:{question_hash}:{agent_id}".encode("utf-8")).hexdigest()


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def _pareto_front_numbers(
    identifiers: Sequence[int],
    values: Mapping[int, Sequence[float]],
) -> dict[int, int]:
    remaining = set(int(identifier) for identifier in identifiers)
    fronts: dict[int, int] = {}
    front_number = 1
    while remaining:
        current = [
            identifier
            for identifier in sorted(remaining)
            if not any(
                other != identifier
                and _dominates(values[other], values[identifier])
                for other in remaining
            )
        ]
        if not current:
            raise AssertionError("Pareto front construction made no progress")
        for identifier in current:
            fronts[identifier] = front_number
            remaining.remove(identifier)
        front_number += 1
    return fronts


def assign_primary_responsibilities(
    *,
    team_states: Mapping[str, TeamVoteState],
    opportunities: Mapping[str, Sequence[MemberAwareRepairOpportunity]],
    state: ResponsibilityState,
    seed: int,
    responsibility_switch_margin: float,
) -> tuple[
    dict[str, int],
    dict[int, list[MemberAwareRepairOpportunity]],
    dict[str, dict[str, Any]],
]:
    agent_ids = sorted(state.updates_since_selected_by_agent)
    if not agent_ids:
        raise ValueError("responsibility state has no agents")
    old_owners = dict(state.primary_owner_by_question)
    old_ages = dict(state.owner_age_by_question)
    owners: dict[str, int] = {}
    assigned = {agent_id: [] for agent_id in agent_ids}
    loads = {agent_id: 0 for agent_id in agent_ids}
    audits: dict[str, dict[str, Any]] = {}

    for question_hash in sorted(team_states):
        if team_states[question_hash].vote_correct:
            continue
        eligible = [
            row for row in opportunities.get(question_hash, ())
            if row.member_error
        ]
        if not eligible:
            continue
        values = {
            row.agent_id: (
                float(int(row.direct_vote_fix)),
                float(row.oracle_soft_utility_gain),
                float(row.improvement_need),
                float(int(row.coverage_opportunity)),
                float(int(row.dominant_wrong_member)),
            )
            for row in eligible
        }
        front_numbers = _pareto_front_numbers(
            [row.agent_id for row in eligible],
            values,
        )
        frontier = [
            row for row in eligible
            if front_numbers[row.agent_id] == 1
        ]
        preferred = min(
            frontier,
            key=lambda row: (
                -row.improvement_need,
                -int(row.direct_vote_fix),
                -row.oracle_soft_utility_gain,
                -int(row.coverage_opportunity),
                -int(row.dominant_wrong_member),
                loads[row.agent_id],
                -state.updates_since_selected_by_agent[row.agent_id],
                _seeded_hash(seed, question_hash, row.agent_id),
            ),
        )
        previous_id = old_owners.get(question_hash)
        previous = next((row for row in frontier if row.agent_id == previous_id), None)
        inertia_allowed = bool(
            previous is not None
            and preferred.improvement_need <= previous.improvement_need
            and int(preferred.direct_vote_fix) <= int(previous.direct_vote_fix)
            and (
                preferred.oracle_soft_utility_gain
                - previous.oracle_soft_utility_gain
                <= float(responsibility_switch_margin)
            )
        )
        if inertia_allowed:
            owner = previous
            chosen_reason = "previous_owner_inertia_within_member_and_soft_margin"
        else:
            owner = preferred
            chosen_reason = "member_aware_pareto_preference"
        owners[question_hash] = owner.agent_id
        assigned[owner.agent_id].append(owner)
        loads[owner.agent_id] += 1
        audits[question_hash] = {
            "vote_correct": False,
            "eligible_agent_ids": [row.agent_id for row in eligible],
            "candidate_pareto_fronts": {
                str(agent_id): front_numbers[agent_id]
                for agent_id in sorted(front_numbers)
            },
            "candidate_vectors": {
                str(agent_id): list(values[agent_id])
                for agent_id in sorted(values)
            },
            "previous_owner": previous_id,
            "chosen_owner": owner.agent_id,
            "chosen_reason": chosen_reason,
        }

    state.primary_owner_by_question = dict(owners)
    state.owner_age_by_question = {
        question_hash: old_ages.get(question_hash, 0) + 1
        if old_owners.get(question_hash) == owner else 0
        for question_hash, owner in owners.items()
    }
    state.assigned_load_by_agent = loads
    return owners, assigned, audits


def target_priorities(
    *,
    opportunities: Mapping[str, Sequence[MemberAwareRepairOpportunity]],
    assignments: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
    state: ResponsibilityState,
    seed: int,
    max_wait_updates: int,
    update_index: int = 0,
) -> tuple[AgentTargetPriority, ...]:
    rows_by_agent: dict[int, list[MemberAwareRepairOpportunity]] = {
        agent_id: [] for agent_id in state.updates_since_selected_by_agent
    }
    for rows in opportunities.values():
        for row in rows:
            rows_by_agent[row.agent_id].append(row)
    priorities = []
    current_counts = {
        agent_id: max((row.current_correct_count for row in rows), default=0)
        for agent_id, rows in rows_by_agent.items()
    }
    best_current_count = max(current_counts.values(), default=0)
    for agent_id, rows in sorted(rows_by_agent.items()):
        errors = [row for row in rows if row.member_error]
        seeded_rank = state.seeded_rank_by_agent.setdefault(
            agent_id, _seeded_hash(seed, "target", agent_id)
        )
        reference = rows[0] if rows else None
        priorities.append(AgentTargetPriority(
            agent_id=agent_id,
            individual_error_count=len(errors),
            assigned_load=len(assignments.get(agent_id, ())),
            direct_vote_fix_count=sum(row.direct_vote_fix for row in errors),
            oracle_soft_utility_gain_sum=sum(
                row.oracle_soft_utility_gain for row in errors
            ),
            coverage_opportunity_count=sum(
                row.coverage_opportunity for row in errors
            ),
            dominant_wrong_count=sum(
                row.dominant_wrong_member for row in errors
            ),
            gain_count=reference.gain_count if reference is not None else 0,
            current_correct_count=current_counts[agent_id],
            best_current_correct_count=best_current_count,
            headroom_to_best=best_current_count - current_counts[agent_id],
            unimproved=(reference.gain_count <= 0 if reference is not None else True),
            improvement_need=max((row.improvement_need for row in errors), default=0),
            unique_correct_count=(
                reference.unique_correct_count if reference is not None else 0
            ),
            pivotal_correct_count=(
                reference.pivotal_correct_count if reference is not None else 0
            ),
            updates_since_selected=state.updates_since_selected_by_agent[agent_id],
            overdue=bool(
                errors
                and state.updates_since_selected_by_agent[agent_id]
                >= max_wait_updates
            ),
            pareto_front=0,
            seeded_rank=seeded_rank,
            best_observed_target_gain=state.best_observed_target_gain_by_agent.get(agent_id, 0),
            no_positive_candidate_streak=state.no_positive_candidate_streak_by_agent.get(agent_id, 0),
            next_regular_eligible_update=state.next_regular_eligible_update_by_agent.get(agent_id, 0),
            cooling_down=update_index < state.next_regular_eligible_update_by_agent.get(agent_id, 0),
            target_attempt_count=state.target_attempt_count_by_agent.get(agent_id, 0),
        ))
    eligible = [row for row in priorities if row.individual_error_count > 0]
    values = {row.agent_id: row.pareto_values() for row in eligible}
    fronts = _pareto_front_numbers([row.agent_id for row in eligible], values)
    return tuple(
        replace(row, pareto_front=fronts.get(row.agent_id, 0))
        for row in priorities
    )


def select_target_agent(
    priorities: Sequence[AgentTargetPriority],
) -> int:
    if not priorities:
        raise ValueError("no erroneous agents are available for member-aware selection")
    eligible = [row for row in priorities if row.individual_error_count > 0]
    if not eligible:
        raise ValueError("no erroneous agents are available for member-aware selection")
    overdue = [row for row in eligible if row.overdue]
    if overdue:
        candidates = overdue
    else:
        regular = [row for row in eligible if not row.cooling_down] or eligible
        candidates = [row for row in regular if row.unimproved] or regular
    candidate_values = {row.agent_id: row.pareto_values() for row in candidates}
    candidate_fronts = _pareto_front_numbers(
        [row.agent_id for row in candidates],
        candidate_values,
    )
    frontier = [row for row in candidates if candidate_fronts[row.agent_id] == 1]
    return min(
        frontier,
        key=lambda row: (
            -row.headroom_to_best,
            -row.best_observed_target_gain,
            -row.improvement_need,
            -row.direct_vote_fix_count,
            -row.oracle_soft_utility_gain_sum,
            -row.assigned_load,
            -row.updates_since_selected,
            row.protection_risk,
            row.seeded_rank,
        ),
    ).agent_id
