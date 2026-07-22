from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .utils import plurality_vote_with_diagnostics


@dataclass(frozen=True)
class PeerVoteState:
    question_hash: str
    gold_answer: str
    normalized_answers: tuple[str, ...]
    valid_vector: tuple[bool, ...]
    correctness_vector: tuple[bool, ...]
    gold_vote_count: int
    wrong_vote_counts: tuple[tuple[str, int], ...]
    largest_wrong_vote_count: int
    dominant_wrong_answers: tuple[str, ...]
    plurality_margin: int
    vote_answer: str
    vote_correct: bool
    top_tie: bool


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


def build_peer_vote_state(
    *,
    question_hash: str,
    gold_answer: str,
    answers: Sequence[str],
    valid_vector: Sequence[bool] | None = None,
    normalize_answer: Callable[[str], str] | None = None,
    match_answer: Callable[[str, str], bool] | None = None,
    tie_break: str = "random",
    seed: int = 0,
) -> PeerVoteState:
    normalize = normalize_answer or (lambda value: str(value or "").strip())
    matcher = match_answer or (lambda prediction, gold: prediction == gold)
    normalized = tuple(normalize(str(answer or "")) for answer in answers)
    valid = tuple(
        bool(value) and bool(normalized[index])
        for index, value in enumerate(valid_vector or [True] * len(normalized))
    )
    if len(valid) != len(normalized):
        raise ValueError("valid_vector length must match answers")
    voting_answers = [answer if valid[index] else "" for index, answer in enumerate(normalized)]
    correctness = tuple(
        bool(valid[index] and matcher(answer, gold_answer))
        for index, answer in enumerate(normalized)
    )
    gold_count = sum(correctness)
    wrong_counts = Counter(
        answer for index, answer in enumerate(normalized)
        if valid[index] and answer and not correctness[index]
    )
    largest_wrong = max(wrong_counts.values(), default=0)
    dominant_wrong = tuple(sorted(answer for answer, count in wrong_counts.items() if count == largest_wrong))
    vote = plurality_vote_with_diagnostics(
        voting_answers, tie_break_method=tie_break, seed=seed, question_hash=question_hash,
    )
    vote_answer = str(vote.get("vote_answer", ""))
    return PeerVoteState(
        question_hash=str(question_hash),
        gold_answer=str(gold_answer),
        normalized_answers=normalized,
        valid_vector=valid,
        correctness_vector=correctness,
        gold_vote_count=int(gold_count),
        wrong_vote_counts=tuple(sorted(wrong_counts.items())),
        largest_wrong_vote_count=int(largest_wrong),
        dominant_wrong_answers=dominant_wrong,
        plurality_margin=int(gold_count - largest_wrong),
        vote_answer=vote_answer,
        vote_correct=bool(vote_answer and matcher(vote_answer, gold_answer)),
        top_tie=bool(vote.get("vote_tie", False)),
    )


def build_team_peer_states(**kwargs) -> PeerVoteState:
    return build_peer_vote_state(**kwargs)


def build_leave_one_out_peer_state(
    *, agent_id: int, answers: Sequence[str], valid_vector: Sequence[bool] | None = None, **kwargs,
) -> PeerVoteState:
    if not 0 <= int(agent_id) < len(answers):
        raise IndexError("agent_id is outside the team")
    peers = [answer for index, answer in enumerate(answers) if index != int(agent_id)]
    peer_valid = None
    if valid_vector is not None:
        peer_valid = [value for index, value in enumerate(valid_vector) if index != int(agent_id)]
    return build_peer_vote_state(answers=peers, valid_vector=peer_valid, **kwargs)
