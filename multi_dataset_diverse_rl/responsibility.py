from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .peer_state import PeerVoteContext, TeamVoteState, soft_vote_utility


@dataclass(frozen=True)
class MemberAwareRepairOpportunity:
    """A per-residual repair value; member-level fairness never enters it."""

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
    member_error: bool

    def repair_vector(self) -> tuple[float, float, float, float]:
        return (
            float(int(self.direct_vote_fix)), float(self.oracle_soft_utility_gain),
            float(int(self.coverage_opportunity)), float(int(self.dominant_wrong_member)),
        )


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
    soft_utility_gain_sum: float
    coverage_count: int
    dominant_wrong_count: int
    residual_count: int
    oldest_responsibility_age: int


@dataclass(frozen=True)
class ResponsibilityTargetPriority:
    agent_id: int
    responsibility_count: int
    direct_fix_count: int
    soft_utility_gain_sum: float
    coverage_count: int
    dominant_wrong_count: int
    member_gain: int
    maximum_member_gain: int
    uplift_deficit: int
    oldest_responsibility_age: int
    updates_since_selected: int
    overdue: bool
    responsibility_pareto_front: int
    joint_target_pareto_front: int
    seeded_rank: str

    def responsibility_values(self) -> tuple[float, ...]:
        return (float(self.direct_fix_count), float(self.soft_utility_gain_sum),
                float(self.coverage_count), float(self.responsibility_count),
                float(self.oldest_responsibility_age))

    def joint_values(self) -> tuple[float, ...]:
        return (float(self.direct_fix_count), float(self.soft_utility_gain_sum),
                float(self.coverage_count), float(self.uplift_deficit),
                float(self.oldest_responsibility_age))


@dataclass(frozen=True)
class TargetSelectionDecision:
    selected_agent_id: int | None
    selection_pool_stage: str
    update_lane: str
    eligible_agent_ids: tuple[int, ...]
    overdue_agent_ids: tuple[int, ...]
    catchup_eligible_agent_ids: tuple[int, ...]
    actual_candidate_agent_ids: tuple[int, ...]
    actual_candidate_pareto_fronts: dict[int, int]
    actual_frontier_agent_ids: tuple[int, ...]
    no_actionable_reason: str = ""


@dataclass
class ResponsibilityState:
    eligible_agents_by_question: dict[str, tuple[int, ...]] = field(default_factory=dict)
    responsibility_first_seen_update: dict[str, int] = field(default_factory=dict)
    updates_since_selected_by_agent: dict[int, int] = field(default_factory=dict)
    accepted_updates_by_agent: dict[int, int] = field(default_factory=dict)
    seeded_rank_by_agent: dict[int, str] = field(default_factory=dict)
    target_attempt_count_by_agent: dict[int, int] = field(default_factory=dict)


def compute_member_aware_repair_opportunity(*, team_state: TeamVoteState, peer_context: PeerVoteContext,
                                             tau: float = 1.0) -> MemberAwareRepairOpportunity:
    target = peer_context.target_agent_id
    if peer_context.question_hash != team_state.question_hash:
        raise ValueError("team and peer context question hashes differ")
    current_correct = bool(team_state.team_correctness[target])
    fixed_gold = peer_context.peer_gold_vote_count + 1
    fixed_margin = fixed_gold - peer_context.peer_largest_wrong_vote_count
    target_answer = team_state.team_answers[target]
    return MemberAwareRepairOpportunity(
        agent_id=target, question_hash=team_state.question_hash, current_correct=current_correct,
        current_invalid=not bool(team_state.team_validity[target]),
        direct_vote_fix=bool(not team_state.vote_correct and fixed_margin > 0),
        oracle_soft_utility_gain=(soft_vote_utility(fixed_gold, fixed_margin, tau)
                                  - soft_vote_utility(team_state.gold_vote_count, team_state.plurality_margin, tau)),
        coverage_opportunity=bool(not current_correct and peer_context.peer_gold_vote_count == 0),
        dominant_wrong_member=bool(not current_correct and bool(target_answer)
                                   and target_answer in team_state.dominant_wrong_answers),
        unique_correct=bool(current_correct and peer_context.peer_gold_vote_count == 0),
        pivotal_correct=bool(current_correct and team_state.vote_correct and peer_context.peer_margin <= 0),
        member_error=not current_correct,
    )


def _seeded_hash(seed: int, question_hash: str, agent_id: int) -> str:
    return hashlib.sha256(f"{seed}:{question_hash}:{agent_id}".encode()).hexdigest()


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(a > b for a, b in zip(left, right, strict=True))


def _pareto_front_numbers(identifiers: Sequence[int], values: Mapping[int, Sequence[float]]) -> dict[int, int]:
    remaining, fronts, number = set(map(int, identifiers)), {}, 1
    while remaining:
        frontier = [item for item in sorted(remaining) if not any(
            other != item and _dominates(values[other], values[item]) for other in remaining)]
        if not frontier:
            raise AssertionError("Pareto front construction made no progress")
        for item in frontier:
            fronts[item] = number
            remaining.remove(item)
        number += 1
    return fronts


def _pair_key(agent_id: int, question_hash: str) -> str:
    return f"{agent_id}:{question_hash}"


def compute_repair_eligibility_frontiers(*, team_states: Mapping[str, TeamVoteState],
                                         opportunities: Mapping[str, Sequence[MemberAwareRepairOpportunity]],
                                         state: ResponsibilityState, current_update_index: int) -> tuple[
                                             dict[str, tuple[int, ...]],
                                             dict[int, list[MemberAwareRepairOpportunity]],
                                             dict[str, dict[str, Any]],
                                         ]:
    """Keep every non-dominated wrong-member repair option for each residual."""
    agents = sorted(state.updates_since_selected_by_agent)
    eligible_by_question: dict[str, tuple[int, ...]] = {}
    portfolios = {agent: [] for agent in agents}
    audits: dict[str, dict[str, Any]] = {}
    active_pairs: set[str] = set()
    for question_hash in sorted(team_states):
        team_state = team_states[question_hash]
        if team_state.vote_correct:
            continue
        candidates = [row for row in opportunities.get(question_hash, ()) if row.member_error]
        if not candidates:
            continue
        values = {row.agent_id: row.repair_vector() for row in candidates}
        fronts = _pareto_front_numbers([row.agent_id for row in candidates], values)
        frontier = tuple(sorted(row.agent_id for row in candidates if fronts[row.agent_id] == 1))
        eligible_by_question[question_hash] = frontier
        for row in candidates:
            if row.agent_id in frontier:
                portfolios[row.agent_id].append(row)
                key = _pair_key(row.agent_id, question_hash)
                active_pairs.add(key)
                state.responsibility_first_seen_update.setdefault(key, int(current_update_index))
        audits[question_hash] = {
            "vote_correct": False,
            "eligible_agent_ids": list(frontier),
            "candidate_pareto_fronts": {str(agent): fronts[agent] for agent in sorted(fronts)},
            "candidate_repair_vectors": {str(agent): list(values[agent]) for agent in sorted(values)},
        }
    state.eligible_agents_by_question = eligible_by_question
    state.responsibility_first_seen_update = {
        key: seen for key, seen in state.responsibility_first_seen_update.items() if key in active_pairs
    }
    return eligible_by_question, portfolios, audits


def responsibility_portfolios(*, assignments: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
                              state: ResponsibilityState, current_update_index: int) -> dict[int, MemberResponsibilityPortfolio]:
    result = {}
    for agent_id in sorted(state.updates_since_selected_by_agent):
        rows = tuple(assignments.get(agent_id, ()))
        ages = [max(0, int(current_update_index) - state.responsibility_first_seen_update.get(
            _pair_key(agent_id, row.question_hash), int(current_update_index))) for row in rows]
        result[agent_id] = MemberResponsibilityPortfolio(
            agent_id=agent_id, residuals=rows,
            direct_fix_count=sum(int(row.direct_vote_fix) for row in rows),
            soft_utility_gain_sum=sum(row.oracle_soft_utility_gain for row in rows),
            coverage_count=sum(int(row.coverage_opportunity) for row in rows),
            dominant_wrong_count=sum(int(row.dominant_wrong_member) for row in rows),
            residual_count=len(rows), oldest_responsibility_age=max(ages, default=0),
        )
    return result


def target_priorities(*, assignments: Mapping[int, Sequence[MemberAwareRepairOpportunity]], state: ResponsibilityState,
                      seed: int, max_wait_updates: int, current_member_correct_counts: Sequence[int],
                      initial_member_correct_counts: Sequence[int], current_update_index: int,
                      member_uplift_tolerance: int) -> tuple[ResponsibilityTargetPriority, ...]:
    if max_wait_updates <= 0:
        raise ValueError("max_wait_updates must be positive")
    if len(current_member_correct_counts) != len(initial_member_correct_counts):
        raise ValueError("member count vectors differ")
    gains = [int(current) - int(initial) for current, initial in zip(current_member_correct_counts, initial_member_correct_counts, strict=True)]
    maximum = max(gains, default=0)
    portfolios = responsibility_portfolios(assignments=assignments, state=state, current_update_index=current_update_index)
    rows = []
    for agent_id, portfolio in portfolios.items():
        if not portfolio.residual_count:
            continue
        wait = state.updates_since_selected_by_agent[agent_id]
        rank = state.seeded_rank_by_agent.setdefault(agent_id, _seeded_hash(seed, "target", agent_id))
        rows.append(ResponsibilityTargetPriority(
            agent_id=agent_id, responsibility_count=portfolio.residual_count,
            direct_fix_count=portfolio.direct_fix_count, soft_utility_gain_sum=portfolio.soft_utility_gain_sum,
            coverage_count=portfolio.coverage_count, dominant_wrong_count=portfolio.dominant_wrong_count,
            member_gain=gains[agent_id], maximum_member_gain=maximum,
            uplift_deficit=max(0, maximum - gains[agent_id] - int(member_uplift_tolerance)),
            oldest_responsibility_age=portfolio.oldest_responsibility_age, updates_since_selected=wait,
            overdue=wait >= max_wait_updates, responsibility_pareto_front=0, joint_target_pareto_front=0,
            seeded_rank=rank,
        ))
    responsibility_fronts = _pareto_front_numbers([row.agent_id for row in rows], {row.agent_id: row.responsibility_values() for row in rows}) if rows else {}
    joint_fronts = _pareto_front_numbers([row.agent_id for row in rows], {row.agent_id: row.joint_values() for row in rows}) if rows else {}
    return tuple(replace(row, responsibility_pareto_front=responsibility_fronts[row.agent_id],
                         joint_target_pareto_front=joint_fronts[row.agent_id]) for row in rows)


def build_target_selection_decision(priorities: Sequence[ResponsibilityTargetPriority], *,
                                    all_member_gains: Sequence[int], state: ResponsibilityState,
                                    max_wait_updates: int, member_uplift_tolerance: int,
                                    member_catchup_mode: str) -> TargetSelectionDecision:
    eligible = tuple(priorities)
    overdue = tuple(row for row in eligible if row.overdue)
    gains, maximum = list(map(int, all_member_gains)), max(all_member_gains, default=0)
    responsible_ids = {row.agent_id for row in eligible}
    catchup = tuple(agent for agent in sorted(state.updates_since_selected_by_agent)
                    if member_catchup_mode == "fallback_v1" and agent not in responsible_ids
                    and max(0, maximum - gains[agent] - int(member_uplift_tolerance)) > 0
                    and state.updates_since_selected_by_agent[agent] >= max_wait_updates)
    if overdue:
        pool, stage, lane = overdue, "responsibility_overdue_frontier", "responsibility_conditioned"
        values = {row.agent_id: row.responsibility_values() for row in pool}
    elif catchup:
        pool, stage, lane = (), "generic_member_catchup_frontier", "generic_member_catchup"
        values = {agent: (float(max(0, maximum - gains[agent] - int(member_uplift_tolerance))),
                          float(state.updates_since_selected_by_agent[agent])) for agent in catchup}
    elif eligible:
        pool, stage, lane = eligible, "responsibility_joint_frontier", "responsibility_conditioned"
        values = {row.agent_id: row.joint_values() for row in pool}
    else:
        return TargetSelectionDecision(None, "no_actionable_responsibility", "no_actionable_responsibility", (), (), catchup, (), {}, (), "no_actionable_responsibility")
    ids = tuple(row.agent_id for row in pool) if pool else catchup
    fronts = _pareto_front_numbers(ids, values)
    frontier = tuple(agent for agent in ids if fronts[agent] == 1)
    selected = min(frontier, key=lambda agent: (-state.updates_since_selected_by_agent[agent],
                                                  state.seeded_rank_by_agent.setdefault(agent, _seeded_hash(0, "target", agent))))
    return TargetSelectionDecision(selected, stage, lane, tuple(row.agent_id for row in eligible),
                                   tuple(row.agent_id for row in overdue), catchup, ids, fronts, frontier)


def mark_responsibilities_serviced(*, state: ResponsibilityState, agent_id: int,
                                  assignments: Mapping[int, Sequence[MemberAwareRepairOpportunity]], update_index: int) -> None:
    for row in assignments.get(agent_id, ()):
        state.responsibility_first_seen_update[_pair_key(agent_id, row.question_hash)] = int(update_index)


def select_target_agent(priorities: Sequence[ResponsibilityTargetPriority], **kwargs: Any) -> int | None:
    if not kwargs:
        return min((row.agent_id for row in priorities if row.joint_target_pareto_front == 1), default=None)
    return build_target_selection_decision(priorities, **kwargs).selected_agent_id
