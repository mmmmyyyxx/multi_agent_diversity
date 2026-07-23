from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MemberGainMetrics:
    initial_correct_counts: tuple[int, ...]
    candidate_correct_counts: tuple[int, ...]
    gains: tuple[int, ...]
    minimum_gain: int
    total_gain: int
    improved_member_count: int
    regressed_member_count: int
    all_members_improved: bool


@dataclass(frozen=True, order=True)
class TeamObjectiveVector:
    vote_correct_count: int
    minimum_member_gain: int
    total_member_gain: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (
            self.vote_correct_count,
            self.minimum_member_gain,
            self.total_member_gain,
        )


def member_gain_metrics(
    initial_correct_counts: Sequence[int],
    candidate_correct_counts: Sequence[int],
) -> MemberGainMetrics:
    initial = tuple(int(value) for value in initial_correct_counts)
    candidate = tuple(int(value) for value in candidate_correct_counts)
    if not initial or len(initial) != len(candidate):
        raise ValueError("initial and candidate member counts must have equal non-zero length")
    gains = tuple(after - before for before, after in zip(initial, candidate, strict=True))
    return MemberGainMetrics(
        initial_correct_counts=initial,
        candidate_correct_counts=candidate,
        gains=gains,
        minimum_gain=min(gains),
        total_gain=sum(gains),
        improved_member_count=sum(value > 0 for value in gains),
        regressed_member_count=sum(value < 0 for value in gains),
        all_members_improved=all(value > 0 for value in gains),
    )


def team_objective_vector(
    vote_correct_count: int,
    member_gain: MemberGainMetrics,
) -> TeamObjectiveVector:
    return TeamObjectiveVector(
        vote_correct_count=int(vote_correct_count),
        minimum_member_gain=member_gain.minimum_gain,
        total_member_gain=member_gain.total_gain,
    )


def pareto_dominates(
    left: TeamObjectiveVector,
    right: TeamObjectiveVector,
) -> bool:
    left_values = left.as_tuple()
    right_values = right.as_tuple()
    return (
        all(a >= b for a, b in zip(left_values, right_values, strict=True))
        and any(a > b for a, b in zip(left_values, right_values, strict=True))
    )


def pareto_front(
    vectors: Sequence[TeamObjectiveVector],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, vector in enumerate(vectors)
        if not any(
            other_index != index and pareto_dominates(other, vector)
            for other_index, other in enumerate(vectors)
        )
    )
