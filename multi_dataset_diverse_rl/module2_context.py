from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

from .evaluation.fixed_probe import ProbeExample
from .peer_state import TeamVoteState, build_team_vote_state
from .versions import EXPERIMENTAL_MODULE2_VERSION


C0_CURRENT_V15 = "c0_current_v15"
C2_BOUNDARY_PLUS_PRESERVATION = "c2_boundary_plus_preservation"
C3_COALITION_AWARE_PRESERVATION = "c3_coalition_aware_preservation"
MODULE2_CONTEXT_VARIANTS = (
    C0_CURRENT_V15,
    C2_BOUNDARY_PLUS_PRESERVATION,
    C3_COALITION_AWARE_PRESERVATION,
)
REPAIR_SET_MAX = 6
PRESERVATION_SET_MAX = 6


@dataclass(frozen=True)
class RepairContextItem:
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    tier: str
    gold_vote_count: int
    repair_distance: int
    boundary_class: str
    target_role: str


@dataclass(frozen=True)
class PreservationContextItem:
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    tier: str
    observed_correct_state_count: int
    parent_margin: int
    removed_margin: int


@dataclass(frozen=True)
class Module2ContextSets:
    repair: tuple[RepairContextItem, ...]
    preservation: tuple[PreservationContextItem, ...]


def exact_repair_distance(
    state: TeamVoteState,
    *,
    normalize_answer: Callable[[str], str],
    match_answer: Callable[[str, str], bool],
    tie_break: str,
    seed: int,
) -> int:
    if state.vote_correct:
        return 0
    wrong = [
        agent_id
        for agent_id, correct in enumerate(state.team_correctness)
        if not correct
    ]
    for count in range(1, len(wrong) + 1):
        for subset in itertools.combinations(wrong, count):
            answers = list(state.team_answers)
            validity = list(state.team_validity)
            for agent_id in subset:
                answers[agent_id] = state.gold_answer
                validity[agent_id] = True
            repaired = build_team_vote_state(
                question_hash=state.question_hash,
                gold_answer=state.gold_answer,
                answers=answers,
                valid_vector=validity,
                normalize_answer=normalize_answer,
                match_answer=match_answer,
                tie_break=tie_break,
                seed=seed,
            )
            if repaired.vote_correct:
                return count
    raise AssertionError("full repair did not produce a correct plurality vote")


def _repair_tier(distance: int, gold_vote_count: int) -> str:
    if distance == 1 and gold_vote_count > 0:
        return "R1_ONE_REPAIR_AWAY"
    if gold_vote_count == 1:
        return "R2_SINGLETON_FRAGMENTED"
    if distance == 2:
        return "R3_TWO_REPAIRS_AWAY"
    return "R4_OTHER_ASSIGNED"


def _boundary_class(distance: int, gold_vote_count: int) -> str:
    if distance == 1:
        return "one_repair_away"
    if gold_vote_count == 1:
        return "singleton_fragmented"
    return "fragmented"


def _target_role(distance: int, gold_vote_count: int) -> str:
    if gold_vote_count == 0:
        return "discoverer"
    if distance == 1:
        return "boundary_closing_member"
    return "reinforcing_member"


def build_module2_context_sets(
    *,
    examples: Sequence[ProbeExample],
    states: Sequence[TeamVoteState],
    target_agent_id: int,
    assigned_question_hashes: set[str],
    stable_correct_question_hashes: set[str],
    accepted_state_count: int,
    normalize_answer: Callable[[str], str],
    match_answer: Callable[[str, str], bool],
    tie_break: str,
    seed: int,
) -> Module2ContextSets:
    if len(examples) != len(states):
        raise ValueError("examples and team states must have identical length")
    target = int(target_agent_id)
    repair: list[RepairContextItem] = []
    preservation: list[PreservationContextItem] = []
    for example, state in zip(examples, states, strict=True):
        if example.question_hash != state.question_hash:
            raise ValueError("example/team-state question hash mismatch")
        target_correct = bool(state.team_correctness[target])
        if (
            example.question_hash in assigned_question_hashes
            and not target_correct
            and not state.vote_correct
        ):
            distance = exact_repair_distance(
                state,
                normalize_answer=normalize_answer,
                match_answer=match_answer,
                tie_break=tie_break,
                seed=seed,
            )
            repair.append(RepairContextItem(
                question_hash=example.question_hash,
                question=example.question,
                gold_answer=example.gold_answer,
                target_current_answer=state.team_answers[target],
                tier=_repair_tier(distance, state.gold_vote_count),
                gold_vote_count=state.gold_vote_count,
                repair_distance=distance,
                boundary_class=_boundary_class(distance, state.gold_vote_count),
                target_role=_target_role(distance, state.gold_vote_count),
            ))
        if not target_correct:
            continue
        validity = list(state.team_validity)
        validity[target] = False
        removed = build_team_vote_state(
            question_hash=state.question_hash,
            gold_answer=state.gold_answer,
            answers=state.team_answers,
            valid_vector=validity,
            normalize_answer=normalize_answer,
            match_answer=match_answer,
            tie_break=tie_break,
            seed=seed,
        )
        if state.vote_correct and not removed.vote_correct:
            tier = "P1_VOTE_CRITICAL"
        elif (
            state.vote_correct
            and removed.vote_correct
            and removed.plurality_margin < state.plurality_margin
        ):
            tier = "P2_COALITION_SUPPORT"
        elif example.question_hash in stable_correct_question_hashes:
            tier = "P3_STABLE_COMPETENCE"
        else:
            continue
        preservation.append(PreservationContextItem(
            question_hash=example.question_hash,
            question=example.question,
            gold_answer=example.gold_answer,
            target_current_answer=state.team_answers[target],
            tier=tier,
            observed_correct_state_count=int(accepted_state_count),
            parent_margin=state.plurality_margin,
            removed_margin=removed.plurality_margin,
        ))
    repair_rank = {
        "R1_ONE_REPAIR_AWAY": 0,
        "R2_SINGLETON_FRAGMENTED": 1,
        "R3_TWO_REPAIRS_AWAY": 2,
        "R4_OTHER_ASSIGNED": 3,
    }
    preserve_rank = {
        "P1_VOTE_CRITICAL": 0,
        "P2_COALITION_SUPPORT": 1,
        "P3_STABLE_COMPETENCE": 2,
    }
    repair.sort(key=lambda row: (repair_rank[row.tier], row.question_hash))
    preservation.sort(key=lambda row: (
        preserve_rank[row.tier],
        -row.observed_correct_state_count
        if row.tier == "P3_STABLE_COMPETENCE" else 0,
        row.question_hash,
    ))
    return Module2ContextSets(
        repair=tuple(repair[:REPAIR_SET_MAX]),
        preservation=tuple(preservation[:PRESERVATION_SET_MAX]),
    )
