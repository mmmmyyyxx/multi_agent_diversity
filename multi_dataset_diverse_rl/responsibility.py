from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .peer_state import PeerVoteContext, TeamVoteState, soft_vote_utility


@dataclass(frozen=True)
class MemberAwareRepairOpportunity:
    """A target-local repair opportunity, deliberately free of competence rank."""

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
            float(int(self.direct_vote_fix)),
            float(self.oracle_soft_utility_gain),
            float(int(self.coverage_opportunity)),
            float(int(self.dominant_wrong_member)),
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
class ResponsibilityTargetPriority:
    """A priority calculated solely from the target's owned residual portfolio."""

    agent_id: int
    assigned_load: int
    owned_direct_vote_fix_count: int
    owned_oracle_soft_utility_gain_sum: float
    owned_coverage_opportunity_count: int
    owned_dominant_wrong_count: int
    oldest_owned_responsibility_age: int
    updates_since_selected: int
    overdue: bool
    pareto_front: int
    seeded_rank: str

    def selection_pareto_values(self) -> tuple[float, ...]:
        return (
            float(self.owned_direct_vote_fix_count),
            float(self.owned_oracle_soft_utility_gain_sum),
            float(self.owned_coverage_opportunity_count),
            float(self.owned_dominant_wrong_count),
            float(self.assigned_load),
            float(self.oldest_owned_responsibility_age),
        )


@dataclass(frozen=True)
class TargetSelectionDecision:
    selected_agent_id: int | None
    selection_pool_stage: str
    eligible_agent_ids: tuple[int, ...]
    overdue_agent_ids: tuple[int, ...]
    actual_candidate_agent_ids: tuple[int, ...]
    actual_candidate_pareto_fronts: dict[int, int]
    actual_frontier_agent_ids: tuple[int, ...]
    no_actionable_reason: str = ""


@dataclass
class ResponsibilityState:
    primary_owner_by_question: dict[str, int] = field(default_factory=dict)
    owner_age_by_question: dict[str, int] = field(default_factory=dict)
    assigned_load_by_agent: dict[int, int] = field(default_factory=dict)
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
    """Compute the four repair dimensions and target-local preservation evidence."""
    target = peer_context.target_agent_id
    if peer_context.question_hash != team_state.question_hash:
        raise ValueError("team and peer context question hashes differ")
    current_correct = bool(team_state.team_correctness[target])
    current_invalid = not bool(team_state.team_validity[target])
    fixed_gold = peer_context.peer_gold_vote_count + 1
    fixed_margin = fixed_gold - peer_context.peer_largest_wrong_vote_count
    target_answer = team_state.team_answers[target]
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
        coverage_opportunity=bool(
            not current_correct and peer_context.peer_gold_vote_count == 0
        ),
        dominant_wrong_member=bool(
            not current_correct
            and bool(target_answer)
            and target_answer in team_state.dominant_wrong_answers
        ),
        unique_correct=bool(current_correct and peer_context.peer_gold_vote_count == 0),
        pivotal_correct=bool(
            current_correct and team_state.vote_correct and peer_context.peer_margin <= 0
        ),
        member_error=not current_correct,
    )


def _seeded_hash(seed: int, question_hash: str, agent_id: int) -> str:
    return hashlib.sha256(f"{seed}:{question_hash}:{agent_id}".encode("utf-8")).hexdigest()


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def _pareto_front_numbers(
    identifiers: Sequence[int], values: Mapping[int, Sequence[float]]
) -> dict[int, int]:
    remaining = set(int(identifier) for identifier in identifiers)
    fronts: dict[int, int] = {}
    front_number = 1
    while remaining:
        frontier = [
            identifier for identifier in sorted(remaining)
            if not any(
                other != identifier and _dominates(values[other], values[identifier])
                for other in remaining
            )
        ]
        if not frontier:
            raise AssertionError("Pareto front construction made no progress")
        for identifier in frontier:
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
    """Assign every vote-wrong residual from its repair vector alone.

    Preservation facts are not ownership dimensions.  The previous owner can stay only
    if it remains Pareto-frontier competitive and the preferred repair advantage is not
    material under the immutable repair-only inertia rule.
    """
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
            row for row in opportunities.get(question_hash, ()) if row.member_error
        ]
        if not eligible:
            continue
        values = {row.agent_id: row.repair_vector() for row in eligible}
        front_numbers = _pareto_front_numbers([row.agent_id for row in eligible], values)
        frontier = [row for row in eligible if front_numbers[row.agent_id] == 1]
        preferred = min(
            frontier,
            key=lambda row: (
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
            and int(previous.direct_vote_fix) >= int(preferred.direct_vote_fix)
            and int(previous.coverage_opportunity) >= int(preferred.coverage_opportunity)
            and int(previous.dominant_wrong_member) >= int(preferred.dominant_wrong_member)
            and (
                preferred.oracle_soft_utility_gain - previous.oracle_soft_utility_gain
                <= float(responsibility_switch_margin)
            )
        )
        owner = previous if inertia_allowed else preferred
        chosen_reason = (
            "previous_owner_repair_only_inertia" if inertia_allowed
            else "repair_only_pareto_preference"
        )
        assert owner is not None
        owners[question_hash] = owner.agent_id
        assigned[owner.agent_id].append(owner)
        loads[owner.agent_id] += 1
        audits[question_hash] = {
            "vote_correct": False,
            "eligible_agent_ids": [row.agent_id for row in eligible],
            "candidate_pareto_fronts": {
                str(agent_id): front_numbers[agent_id] for agent_id in sorted(front_numbers)
            },
            "candidate_vectors": {
                str(agent_id): list(values[agent_id]) for agent_id in sorted(values)
            },
            "previous_owner": previous_id,
            "chosen_owner": owner.agent_id,
            "chosen_reason": chosen_reason,
            "previous_owner_age": old_ages.get(question_hash, 0),
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
    assignments: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
    state: ResponsibilityState,
    seed: int,
    max_wait_updates: int,
) -> tuple[ResponsibilityTargetPriority, ...]:
    """Return only agents that own at least one actionable residual."""
    if max_wait_updates <= 0:
        raise ValueError("max_wait_updates must be positive")
    priorities: list[ResponsibilityTargetPriority] = []
    for agent_id in sorted(state.updates_since_selected_by_agent):
        rows = tuple(assignments.get(agent_id, ()))
        if not rows:
            continue
        seeded_rank = state.seeded_rank_by_agent.setdefault(
            agent_id, _seeded_hash(seed, "target", agent_id)
        )
        priorities.append(ResponsibilityTargetPriority(
            agent_id=agent_id,
            assigned_load=len(rows),
            owned_direct_vote_fix_count=sum(int(row.direct_vote_fix) for row in rows),
            owned_oracle_soft_utility_gain_sum=sum(
                row.oracle_soft_utility_gain for row in rows
            ),
            owned_coverage_opportunity_count=sum(
                int(row.coverage_opportunity) for row in rows
            ),
            owned_dominant_wrong_count=sum(
                int(row.dominant_wrong_member) for row in rows
            ),
            oldest_owned_responsibility_age=max(
                (state.owner_age_by_question.get(row.question_hash, 0) for row in rows),
                default=0,
            ),
            updates_since_selected=state.updates_since_selected_by_agent[agent_id],
            overdue=state.updates_since_selected_by_agent[agent_id] >= max_wait_updates,
            pareto_front=0,
            seeded_rank=seeded_rank,
        ))
    values = {row.agent_id: row.selection_pareto_values() for row in priorities}
    fronts = _pareto_front_numbers([row.agent_id for row in priorities], values)
    return tuple(replace(row, pareto_front=fronts[row.agent_id]) for row in priorities)


def build_target_selection_decision(
    priorities: Sequence[ResponsibilityTargetPriority],
) -> TargetSelectionDecision:
    eligible = tuple(priorities)
    if not eligible:
        return TargetSelectionDecision(
            selected_agent_id=None,
            selection_pool_stage="no_actionable_responsibility",
            eligible_agent_ids=(),
            overdue_agent_ids=(),
            actual_candidate_agent_ids=(),
            actual_candidate_pareto_fronts={},
            actual_frontier_agent_ids=(),
            no_actionable_reason="no_actionable_responsibility",
        )
    overdue = tuple(row for row in eligible if row.overdue)
    candidates = overdue or eligible
    fronts = _pareto_front_numbers(
        [row.agent_id for row in candidates],
        {row.agent_id: row.selection_pareto_values() for row in candidates},
    )
    frontier = tuple(row for row in candidates if fronts[row.agent_id] == 1)
    selected = min(
        frontier,
        key=lambda row: (
            -row.oldest_owned_responsibility_age,
            -row.updates_since_selected,
            -row.assigned_load,
            row.seeded_rank,
        ),
    )
    return TargetSelectionDecision(
        selected_agent_id=selected.agent_id,
        selection_pool_stage=("assigned_owner_max_wait" if overdue else "assigned_owner_portfolio"),
        eligible_agent_ids=tuple(row.agent_id for row in eligible),
        overdue_agent_ids=tuple(row.agent_id for row in overdue),
        actual_candidate_agent_ids=tuple(row.agent_id for row in candidates),
        actual_candidate_pareto_fronts=fronts,
        actual_frontier_agent_ids=tuple(row.agent_id for row in frontier),
    )


def select_target_agent(
    priorities: Sequence[ResponsibilityTargetPriority],
) -> int | None:
    return build_target_selection_decision(priorities).selected_agent_id
