from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .utils import plurality_vote_with_diagnostics


TEAM_SIZE = 5


@dataclass(frozen=True)
class TeamVoteState:
    question_hash: str
    gold_answer: str
    team_answers: tuple[str, ...]
    team_validity: tuple[bool, ...]
    team_correctness: tuple[bool, ...]
    gold_vote_count: int
    wrong_vote_histogram: tuple[tuple[str, int], ...]
    largest_wrong_vote_count: int
    dominant_wrong_answers: tuple[str, ...]
    plurality_margin: int
    vote_answer: str
    vote_correct: bool
    top_tie: bool


@dataclass(frozen=True)
class PeerVoteContext:
    question_hash: str
    target_agent_id: int
    gold_answer: str
    peer_answers: tuple[str, ...]
    peer_validity: tuple[bool, ...]
    peer_correctness: tuple[bool, ...]
    peer_gold_vote_count: int
    peer_wrong_vote_histogram: tuple[tuple[str, int], ...]
    peer_largest_wrong_vote_count: int
    peer_dominant_wrong_answers: tuple[str, ...]
    peer_margin: int


def soft_vote_utility(gold_vote_count: int, plurality_margin: int, tau: float = 1.0) -> float:
    if gold_vote_count <= 0:
        return 0.0
    if tau <= 0:
        raise ValueError("soft_vote_tau must be positive")
    value = float(plurality_margin) / float(tau)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _normalize_team(
    answers: Sequence[str],
    valid_vector: Sequence[bool] | None,
    normalize_answer: Callable[[str], str] | None,
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    if len(answers) != TEAM_SIZE:
        raise ValueError(f"TeamVoteState requires exactly {TEAM_SIZE} agent answers")
    normalize = normalize_answer or (lambda value: str(value or "").strip())
    normalized = tuple(normalize(str(answer or "")) for answer in answers)
    supplied_validity = tuple(valid_vector) if valid_vector is not None else (True,) * TEAM_SIZE
    if len(supplied_validity) != TEAM_SIZE:
        raise ValueError("team validity length must match the five answers")
    validity = tuple(bool(value) and bool(normalized[index]) for index, value in enumerate(supplied_validity))
    return normalized, validity


def _wrong_histogram(
    answers: Sequence[str], validity: Sequence[bool], correctness: Sequence[bool]
) -> tuple[tuple[str, int], ...]:
    counts = Counter(
        answer
        for answer, valid, correct in zip(answers, validity, correctness, strict=True)
        if valid and answer and not correct
    )
    return tuple(sorted((str(answer), int(count)) for answer, count in counts.items()))


def build_team_vote_state(
    *,
    question_hash: str,
    gold_answer: str,
    answers: Sequence[str],
    valid_vector: Sequence[bool] | None = None,
    normalize_answer: Callable[[str], str] | None = None,
    match_answer: Callable[[str, str], bool] | None = None,
    tie_break: str = "abstain",
    seed: int = 0,
) -> TeamVoteState:
    matcher = match_answer or (lambda prediction, gold: prediction == gold)
    normalized, validity = _normalize_team(answers, valid_vector, normalize_answer)
    correctness = tuple(
        bool(valid and matcher(answer, gold_answer))
        for answer, valid in zip(normalized, validity, strict=True)
    )
    histogram = _wrong_histogram(normalized, validity, correctness)
    largest_wrong = max((count for _, count in histogram), default=0)
    dominant_wrong = tuple(answer for answer, count in histogram if count == largest_wrong and largest_wrong > 0)
    gold_count = sum(correctness)
    vote = plurality_vote_with_diagnostics(
        [answer if validity[index] else "" for index, answer in enumerate(normalized)],
        tie_break_method=tie_break,
        seed=seed,
        question_hash=question_hash,
    )
    vote_answer = str(vote["vote_answer"])
    vote_correct = bool(vote_answer and matcher(vote_answer, gold_answer))
    margin = int(gold_count - largest_wrong)
    if tie_break == "abstain" and vote_correct != (margin > 0):
        raise AssertionError("tie-as-abstain vote correctness must be equivalent to M > 0")
    return TeamVoteState(
        question_hash=str(question_hash),
        gold_answer=str(gold_answer),
        team_answers=normalized,
        team_validity=validity,
        team_correctness=correctness,
        gold_vote_count=int(gold_count),
        wrong_vote_histogram=histogram,
        largest_wrong_vote_count=int(largest_wrong),
        dominant_wrong_answers=dominant_wrong,
        plurality_margin=margin,
        vote_answer=vote_answer,
        vote_correct=vote_correct,
        top_tie=bool(vote["vote_tie"]),
    )


def build_peer_vote_context(team_state: TeamVoteState, target_agent_id: int) -> PeerVoteContext:
    if len(team_state.team_answers) != TEAM_SIZE:
        raise ValueError("TeamVoteState does not contain five agents")
    target = int(target_agent_id)
    if target < 0 or target >= TEAM_SIZE:
        raise IndexError("target_agent_id is outside the five-agent team")
    peer_answers = tuple(answer for index, answer in enumerate(team_state.team_answers) if index != target)
    peer_validity = tuple(value for index, value in enumerate(team_state.team_validity) if index != target)
    peer_correctness = tuple(value for index, value in enumerate(team_state.team_correctness) if index != target)
    if len(peer_answers) != TEAM_SIZE - 1:
        raise AssertionError("PeerVoteContext must contain exactly four peers")
    histogram = _wrong_histogram(peer_answers, peer_validity, peer_correctness)
    largest_wrong = max((count for _, count in histogram), default=0)
    dominant_wrong = tuple(answer for answer, count in histogram if count == largest_wrong and largest_wrong > 0)
    peer_gold = sum(peer_correctness)
    return PeerVoteContext(
        question_hash=team_state.question_hash,
        target_agent_id=target,
        gold_answer=team_state.gold_answer,
        peer_answers=peer_answers,
        peer_validity=peer_validity,
        peer_correctness=peer_correctness,
        peer_gold_vote_count=int(peer_gold),
        peer_wrong_vote_histogram=histogram,
        peer_largest_wrong_vote_count=int(largest_wrong),
        peer_dominant_wrong_answers=dominant_wrong,
        peer_margin=int(peer_gold - largest_wrong),
    )
