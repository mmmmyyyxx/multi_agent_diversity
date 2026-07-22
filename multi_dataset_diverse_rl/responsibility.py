from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .peer_state import PeerVoteContext, TeamVoteState, soft_vote_utility


@dataclass(frozen=True)
class OracleRepairOpportunity:
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


@dataclass
class ResponsibilityState:
    primary_owner_by_question: dict[str, int] = field(default_factory=dict)
    owner_age_by_question: dict[str, int] = field(default_factory=dict)
    assigned_load_by_agent: dict[int, int] = field(default_factory=dict)
    updates_since_selected_by_agent: dict[int, int] = field(default_factory=dict)


def compute_oracle_repair_opportunity(
    *,
    team_state: TeamVoteState,
    peer_context: PeerVoteContext,
    tau: float = 1.0,
) -> OracleRepairOpportunity:
    target = peer_context.target_agent_id
    if peer_context.question_hash != team_state.question_hash:
        raise ValueError("team and peer context question hashes differ")
    if peer_context.gold_answer != team_state.gold_answer:
        raise ValueError("team and peer context gold answers differ")
    current_correct = bool(team_state.team_correctness[target])
    current_invalid = not bool(team_state.team_validity[target])
    fixed_gold = peer_context.peer_gold_vote_count + 1
    fixed_wrong = peer_context.peer_largest_wrong_vote_count
    fixed_margin = fixed_gold - fixed_wrong
    target_answer = team_state.team_answers[target]
    return OracleRepairOpportunity(
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
        pivotal_correct=bool(current_correct and team_state.vote_correct and peer_context.peer_margin <= 0),
    )


def _stable_hash(question_hash: str, agent_id: int) -> str:
    return hashlib.sha256(f"{question_hash}:{agent_id}".encode("utf-8")).hexdigest()


def assign_primary_responsibilities(
    *,
    team_states: Mapping[str, TeamVoteState],
    peer_contexts: Mapping[str, Mapping[int, PeerVoteContext]],
    opportunities: Mapping[str, Sequence[OracleRepairOpportunity]],
    state: ResponsibilityState,
    switch_margin: float = 0.05,
) -> tuple[dict[str, int], dict[int, list[OracleRepairOpportunity]]]:
    agent_ids = sorted(state.updates_since_selected_by_agent)
    if not agent_ids:
        raise ValueError("responsibility state has no agents")
    owners: dict[str, int] = {}
    assigned = {agent_id: [] for agent_id in agent_ids}
    loads = {agent_id: 0 for agent_id in agent_ids}
    old_owners = dict(state.primary_owner_by_question)
    old_ages = dict(state.owner_age_by_question)

    for question_hash in sorted(team_states):
        team_state = team_states[question_hash]
        if team_state.vote_correct:
            continue
        question_opportunities = list(opportunities.get(question_hash, ()))
        contexts = peer_contexts.get(question_hash)
        if contexts is None:
            raise KeyError(f"missing peer contexts for {question_hash}")
        eligible = [
            opportunity
            for opportunity in question_opportunities
            if not opportunity.current_correct and opportunity.agent_id in contexts
        ]
        if not eligible:
            continue
        previous_id = old_owners.get(question_hash)
        previous = next((row for row in eligible if row.agent_id == previous_id), None)
        all_c0 = team_state.gold_vote_count == 0

        if all_c0:
            best_gain = max(row.oracle_soft_utility_gain for row in eligible)
            gain_tolerance = max(float(switch_margin), 1e-9)
            competitive = [
                row for row in eligible
                if best_gain - row.oracle_soft_utility_gain <= gain_tolerance
            ]
            dominant_exit = max(int(row.dominant_wrong_member) for row in competitive)
            competitive = [
                row for row in competitive if int(row.dominant_wrong_member) == dominant_exit
            ]
            if previous is not None and previous in competitive:
                best = previous
            else:
                best = min(
                    competitive,
                    key=lambda row: (
                        loads[row.agent_id],
                        -state.updates_since_selected_by_agent[row.agent_id],
                        _stable_hash(question_hash, row.agent_id),
                        row.agent_id,
                    ),
                )
        else:
            best = max(
                eligible,
                key=lambda row: (
                    int(row.direct_vote_fix),
                    row.oracle_soft_utility_gain,
                    int(row.dominant_wrong_member),
                    int(previous_id == row.agent_id),
                    -loads[row.agent_id],
                    state.updates_since_selected_by_agent[row.agent_id],
                    tuple(-ord(char) for char in _stable_hash(question_hash, row.agent_id)),
                    -row.agent_id,
                ),
            )
            if previous is not None and best.agent_id != previous.agent_id:
                owner_age = old_ages.get(question_hash, 0)
                stability_margin = float(switch_margin) + min(0.05, 0.005 * owner_age)
                materially_better = (
                    best.direct_vote_fix and not previous.direct_vote_fix
                ) or (
                    best.direct_vote_fix == previous.direct_vote_fix
                    and best.oracle_soft_utility_gain
                    > previous.oracle_soft_utility_gain + stability_margin
                )
                if not materially_better:
                    best = previous

        owners[question_hash] = best.agent_id
        assigned[best.agent_id].append(best)
        loads[best.agent_id] += 1

    state.primary_owner_by_question = dict(owners)
    state.owner_age_by_question = {
        question_hash: old_ages.get(question_hash, 0) + 1
        if old_owners.get(question_hash) == owner
        else 0
        for question_hash, owner in owners.items()
    }
    state.assigned_load_by_agent = dict(loads)
    return owners, assigned


def select_target_agent(
    assignments: Mapping[int, Sequence[OracleRepairOpportunity]],
    state: ResponsibilityState,
    max_wait_updates: int = 8,
) -> int:
    agent_ids = sorted(state.updates_since_selected_by_agent)
    if not agent_ids:
        raise ValueError("no agents are available for responsibility selection")
    overdue = [
        agent_id
        for agent_id in agent_ids
        if state.updates_since_selected_by_agent[agent_id] >= max_wait_updates
    ]
    candidates = overdue or agent_ids

    def key(agent_id: int) -> tuple[float | int, ...]:
        rows = tuple(assignments.get(agent_id, ()))
        return (
            sum(int(row.direct_vote_fix) for row in rows),
            sum(row.oracle_soft_utility_gain for row in rows),
            sum(int(row.coverage_opportunity) for row in rows),
            sum(int(row.dominant_wrong_member) for row in rows),
            state.updates_since_selected_by_agent[agent_id],
            -agent_id,
        )

    return max(candidates, key=key)
