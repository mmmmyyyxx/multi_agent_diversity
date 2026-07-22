from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .peer_state import PeerVoteState, build_leave_one_out_peer_state, build_peer_vote_state, soft_vote_utility


@dataclass(frozen=True)
class AgentExampleCredit:
    agent_id: int
    question_hash: str
    current_correct: bool
    current_invalid: bool
    direct_vote_fix: int
    fix_soft_utility_gain: float
    coverage_opportunity: bool
    dominant_wrong_member: bool
    unique_correct: bool
    pivotal_vote_correct: bool


@dataclass
class ResponsibilityState:
    previous_primary_owner_by_question: dict[str, int] = field(default_factory=dict)
    responsibility_age_by_question: dict[str, int] = field(default_factory=dict)
    agent_updates_since_last_selected: dict[int, int] = field(default_factory=dict)
    assigned_load_per_agent: dict[int, int] = field(default_factory=dict)


def counterfactual_credit(
    *,
    agent_id: int,
    current_state: PeerVoteState,
    gold_answer: str,
    normalize_answer: Callable[[str], str] | None = None,
    match_answer: Callable[[str, str], bool] | None = None,
    tie_break: str = "random",
    seed: int = 0,
    tau: float = 1.0,
) -> AgentExampleCredit:
    peer_state = build_leave_one_out_peer_state(
        agent_id=agent_id,
        question_hash=current_state.question_hash,
        gold_answer=gold_answer,
        answers=current_state.normalized_answers,
        valid_vector=current_state.valid_vector,
        normalize_answer=normalize_answer,
        match_answer=match_answer,
        tie_break=tie_break,
        seed=seed,
    )
    fixed_answers = list(current_state.normalized_answers)
    fixed_valid = list(current_state.valid_vector)
    fixed_answers[agent_id] = gold_answer
    fixed_valid[agent_id] = True
    fixed_state = build_peer_vote_state(
        question_hash=current_state.question_hash,
        gold_answer=gold_answer,
        answers=fixed_answers,
        valid_vector=fixed_valid,
        normalize_answer=normalize_answer,
        match_answer=match_answer,
        tie_break=tie_break,
        seed=seed,
    )
    current_correct = bool(current_state.correctness_vector[agent_id])
    current_invalid = not bool(current_state.valid_vector[agent_id])
    target_answer = current_state.normalized_answers[agent_id]
    return AgentExampleCredit(
        agent_id=int(agent_id),
        question_hash=current_state.question_hash,
        current_correct=current_correct,
        current_invalid=current_invalid,
        direct_vote_fix=int(not current_state.vote_correct and fixed_state.vote_correct),
        fix_soft_utility_gain=(
            soft_vote_utility(fixed_state.gold_vote_count, fixed_state.plurality_margin, tau)
            - soft_vote_utility(current_state.gold_vote_count, current_state.plurality_margin, tau)
        ),
        coverage_opportunity=bool(not current_correct and peer_state.gold_vote_count == 0),
        dominant_wrong_member=bool(
            current_state.gold_vote_count > 0
            and not current_correct
            and target_answer in current_state.dominant_wrong_answers
        ),
        unique_correct=bool(current_correct and peer_state.gold_vote_count == 0),
        pivotal_vote_correct=bool(current_state.vote_correct and not peer_state.vote_correct),
    )


def _stable_hash(question_hash: str, agent_id: int) -> str:
    return hashlib.sha256(f"{question_hash}:{agent_id}".encode("utf-8")).hexdigest()


def assign_primary_responsibilities(
    credits_by_question: Mapping[str, Sequence[AgentExampleCredit]],
    state: ResponsibilityState,
    switch_margin: float = 0.05,
) -> tuple[dict[str, int], dict[int, list[AgentExampleCredit]]]:
    owners: dict[str, int] = {}
    assigned: dict[int, list[AgentExampleCredit]] = {
        int(agent_id): [] for agent_id in state.agent_updates_since_last_selected
    }
    loads = {int(agent_id): 0 for agent_id in state.agent_updates_since_last_selected}
    for question_hash in sorted(credits_by_question):
        eligible = [credit for credit in credits_by_question[question_hash] if not credit.current_correct]
        if not eligible:
            continue
        previous_id = state.previous_primary_owner_by_question.get(question_hash)
        previous = next((credit for credit in eligible if credit.agent_id == previous_id), None)

        if all(credit.coverage_opportunity for credit in eligible):
            if previous is not None:
                best = previous
            else:
                best = min(eligible, key=lambda credit: (
                    int(loads.get(credit.agent_id, 0)),
                    -int(state.agent_updates_since_last_selected.get(credit.agent_id, 0)),
                    _stable_hash(question_hash, credit.agent_id),
                    int(credit.agent_id),
                ))
            owners[question_hash] = best.agent_id
            assigned.setdefault(best.agent_id, []).append(best)
            loads[best.agent_id] = loads.get(best.agent_id, 0) + 1
            continue

        def rank(credit: AgentExampleCredit) -> tuple:
            return (
                int(credit.direct_vote_fix),
                float(credit.fix_soft_utility_gain),
                int(credit.dominant_wrong_member),
                int(previous_id == credit.agent_id),
                -int(loads.get(credit.agent_id, 0)),
                int(state.agent_updates_since_last_selected.get(credit.agent_id, 0)),
                -int(credit.agent_id),
            )

        best = max(eligible, key=rank)
        if previous is not None and best.agent_id != previous.agent_id:
            can_switch = (
                (best.direct_vote_fix > previous.direct_vote_fix)
                or best.fix_soft_utility_gain > previous.fix_soft_utility_gain + float(switch_margin)
            )
            if not can_switch:
                best = previous
        owners[question_hash] = best.agent_id
        assigned.setdefault(best.agent_id, []).append(best)
        loads[best.agent_id] = loads.get(best.agent_id, 0) + 1

    old_owners = dict(state.previous_primary_owner_by_question)
    state.previous_primary_owner_by_question = dict(owners)
    state.assigned_load_per_agent = dict(loads)
    state.responsibility_age_by_question = {
        question_hash: int(state.responsibility_age_by_question.get(question_hash, 0) + 1)
        if old_owners.get(question_hash) == owner else 0
        for question_hash, owner in owners.items()
    }
    return owners, assigned


def select_target_agent(
    assignments: Mapping[int, Sequence[AgentExampleCredit]],
    updates_since_last_selected: Mapping[int, int],
    max_wait_updates: int = 8,
) -> int:
    agent_ids = sorted(set(updates_since_last_selected) | set(assignments))
    if not agent_ids:
        raise ValueError("no agents are available for responsibility selection")
    overdue = [agent_id for agent_id in agent_ids if updates_since_last_selected.get(agent_id, 0) >= max_wait_updates]
    candidates = overdue or agent_ids

    def key(agent_id: int) -> tuple:
        rows = list(assignments.get(agent_id, ()))
        return (
            sum(row.direct_vote_fix for row in rows),
            sum(row.fix_soft_utility_gain for row in rows),
            sum(row.coverage_opportunity for row in rows),
            sum(row.dominant_wrong_member for row in rows),
            int(updates_since_last_selected.get(agent_id, 0)),
            -int(agent_id),
        )

    return max(candidates, key=key)
