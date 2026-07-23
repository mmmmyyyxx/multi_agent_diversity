from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

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
    member_correct_count: int
    team_correct_count_sum: int
    improvement_need: int
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
    overdue: bool
    improvement_need: int
    member_error_count: int
    assigned_count: int
    direct_vote_fix_count: int
    protection_need_count: int
    updates_since_selected: int
    seeded_rank: str

    def pareto_values(self) -> tuple[int, ...]:
        return (
            self.improvement_need,
            self.member_error_count,
            self.assigned_count,
            self.direct_vote_fix_count,
        )


@dataclass
class ResponsibilityState:
    primary_owner_by_question: dict[str, int] = field(default_factory=dict)
    owner_age_by_question: dict[str, int] = field(default_factory=dict)
    assigned_load_by_agent: dict[int, int] = field(default_factory=dict)
    updates_since_selected_by_agent: dict[int, int] = field(default_factory=dict)
    accepted_updates_by_agent: dict[int, int] = field(default_factory=dict)
    seeded_rank_by_agent: dict[int, str] = field(default_factory=dict)


def compute_member_aware_repair_opportunity(
    *,
    team_state: TeamVoteState,
    peer_context: PeerVoteContext,
    member_correct_counts: Sequence[int],
    member_gains_from_initial: Sequence[int],
    tau: float = 1.0,
) -> MemberAwareRepairOpportunity:
    target = peer_context.target_agent_id
    if peer_context.question_hash != team_state.question_hash:
        raise ValueError("team and peer context question hashes differ")
    if len(member_correct_counts) != len(team_state.team_correctness):
        raise ValueError("member count vector does not match team size")
    if len(member_gains_from_initial) != len(member_correct_counts):
        raise ValueError("member gain vector does not match member count vector")
    current_correct = bool(team_state.team_correctness[target])
    current_invalid = not bool(team_state.team_validity[target])
    fixed_gold = peer_context.peer_gold_vote_count + 1
    fixed_margin = fixed_gold - peer_context.peer_largest_wrong_vote_count
    target_answer = team_state.team_answers[target]
    total = sum(int(value) for value in member_correct_counts)
    total_gain = sum(int(value) for value in member_gains_from_initial)
    member_gain = int(member_gains_from_initial[target])
    member_count = int(member_correct_counts[target])
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
        member_correct_count=member_count,
        team_correct_count_sum=total,
        improvement_need=max(0, total_gain - member_count_n * member_gain),
        member_error=not current_correct,
        protection_need_count=int(current_correct) + int(
            current_correct and team_state.vote_correct and peer_context.peer_margin <= 0
        ),
    )


def _seeded_hash(seed: int, question_hash: str, agent_id: int) -> str:
    return hashlib.sha256(f"{seed}:{question_hash}:{agent_id}".encode("utf-8")).hexdigest()


def _dominates(left: Sequence[int], right: Sequence[int]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def assign_primary_responsibilities(
    *,
    team_states: Mapping[str, TeamVoteState],
    opportunities: Mapping[str, Sequence[MemberAwareRepairOpportunity]],
    state: ResponsibilityState,
    seed: int,
) -> tuple[dict[str, int], dict[int, list[MemberAwareRepairOpportunity]]]:
    agent_ids = sorted(state.updates_since_selected_by_agent)
    if not agent_ids:
        raise ValueError("responsibility state has no agents")
    old_owners = dict(state.primary_owner_by_question)
    old_ages = dict(state.owner_age_by_question)
    owners: dict[str, int] = {}
    assigned = {agent_id: [] for agent_id in agent_ids}
    loads = {agent_id: 0 for agent_id in agent_ids}

    for question_hash in sorted(team_states):
        eligible = [
            row for row in opportunities.get(question_hash, ())
            if row.member_error
        ]
        if not eligible:
            continue
        values = {
            row.agent_id: (
                int(row.direct_vote_fix),
                row.improvement_need,
                int(row.coverage_opportunity),
                int(row.dominant_wrong_member),
            )
            for row in eligible
        }
        frontier = [
            row for row in eligible
            if not any(
                other.agent_id != row.agent_id
                and _dominates(values[other.agent_id], values[row.agent_id])
                for other in eligible
            )
        ]
        previous_id = old_owners.get(question_hash)
        previous = next((row for row in frontier if row.agent_id == previous_id), None)
        if previous is not None:
            owner = previous
        else:
            owner = min(
                frontier,
                key=lambda row: (
                    loads[row.agent_id],
                    -state.updates_since_selected_by_agent[row.agent_id],
                    state.accepted_updates_by_agent.get(row.agent_id, 0),
                    _seeded_hash(seed, question_hash, row.agent_id),
                ),
            )
        owners[question_hash] = owner.agent_id
        assigned[owner.agent_id].append(owner)
        loads[owner.agent_id] += 1

    state.primary_owner_by_question = dict(owners)
    state.owner_age_by_question = {
        question_hash: old_ages.get(question_hash, 0) + 1
        if old_owners.get(question_hash) == owner else 0
        for question_hash, owner in owners.items()
    }
    state.assigned_load_by_agent = loads
    return owners, assigned


def target_priorities(
    *,
    opportunities: Mapping[str, Sequence[MemberAwareRepairOpportunity]],
    assignments: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
    state: ResponsibilityState,
    seed: int,
    max_wait_updates: int,
) -> tuple[AgentTargetPriority, ...]:
    rows_by_agent: dict[int, list[MemberAwareRepairOpportunity]] = {
        agent_id: [] for agent_id in state.updates_since_selected_by_agent
    }
    for rows in opportunities.values():
        for row in rows:
            rows_by_agent[row.agent_id].append(row)
    priorities = []
    for agent_id, rows in sorted(rows_by_agent.items()):
        errors = [row for row in rows if row.member_error]
        if not errors:
            continue
        seeded_rank = state.seeded_rank_by_agent.setdefault(
            agent_id, _seeded_hash(seed, "target", agent_id)
        )
        priorities.append(AgentTargetPriority(
            agent_id=agent_id,
            overdue=state.updates_since_selected_by_agent[agent_id] >= max_wait_updates,
            improvement_need=max((row.improvement_need for row in errors), default=0),
            member_error_count=len(errors),
            assigned_count=len(assignments.get(agent_id, ())),
            direct_vote_fix_count=sum(row.direct_vote_fix for row in errors),
            protection_need_count=sum(row.protection_need_count for row in rows),
            updates_since_selected=state.updates_since_selected_by_agent[agent_id],
            seeded_rank=seeded_rank,
        ))
    return tuple(priorities)


def select_target_agent(
    priorities: Sequence[AgentTargetPriority],
) -> int:
    if not priorities:
        raise ValueError("no erroneous agents are available for member-aware selection")
    candidates = [row for row in priorities if row.overdue] or list(priorities)
    frontier = [
        row for row in candidates
        if not any(
            other.agent_id != row.agent_id
            and _dominates(other.pareto_values(), row.pareto_values())
            for other in candidates
        )
    ]
    return min(
        frontier,
        key=lambda row: (
            -row.updates_since_selected,
            -row.improvement_need,
            -row.member_error_count,
            row.seeded_rank,
        ),
    ).agent_id
