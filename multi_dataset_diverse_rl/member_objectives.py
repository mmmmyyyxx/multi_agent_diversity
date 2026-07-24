from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MemberGainMetrics:
    initial_correct_counts: tuple[int, ...]
    incumbent_correct_counts: tuple[int, ...]
    candidate_correct_counts: tuple[int, ...]

    gain_counts: tuple[int, ...]
    minimum_gain_count: int
    total_gain_count: int
    mean_gain: float

    improved_agent_count: int
    regressed_agent_count: int
    all_members_non_regressed: bool
    all_members_improved: bool

    target_gain_vs_initial: int
    target_gain_vs_incumbent: int


@dataclass(frozen=True)
class TeamObjectiveVector:
    vote_correct_count: int
    minimum_member_gain_count: int
    total_member_gain_count: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (
            self.vote_correct_count,
            self.minimum_member_gain_count,
            self.total_member_gain_count,
        )


def member_gain_metrics(
    initial_correct_counts: Sequence[int],
    incumbent_correct_counts: Sequence[int],
    candidate_correct_counts: Sequence[int],
    target_agent_id: int,
) -> MemberGainMetrics:
    initial = tuple(int(value) for value in initial_correct_counts)
    incumbent = tuple(int(value) for value in incumbent_correct_counts)
    candidate = tuple(int(value) for value in candidate_correct_counts)
    if (
        not initial
        or len(initial) != len(incumbent)
        or len(initial) != len(candidate)
    ):
        raise ValueError(
            "initial, incumbent, and candidate member counts must have equal non-zero length"
        )
    if not 0 <= int(target_agent_id) < len(initial):
        raise ValueError("target_agent_id is outside the member count vector")
    gains = tuple(after - before for before, after in zip(initial, candidate, strict=True))
    return MemberGainMetrics(
        initial_correct_counts=initial,
        incumbent_correct_counts=incumbent,
        candidate_correct_counts=candidate,
        gain_counts=gains,
        minimum_gain_count=min(gains),
        total_gain_count=sum(gains),
        mean_gain=sum(gains) / len(gains),
        improved_agent_count=sum(value > 0 for value in gains),
        regressed_agent_count=sum(value < 0 for value in gains),
        all_members_non_regressed=all(value >= 0 for value in gains),
        all_members_improved=all(value > 0 for value in gains),
        target_gain_vs_initial=(
            candidate[int(target_agent_id)] - initial[int(target_agent_id)]
        ),
        target_gain_vs_incumbent=(
            candidate[int(target_agent_id)] - incumbent[int(target_agent_id)]
        ),
    )


def team_objective_vector(
    vote_correct_count: int,
    member_gain: MemberGainMetrics,
) -> TeamObjectiveVector:
    return TeamObjectiveVector(
        vote_correct_count=int(vote_correct_count),
        minimum_member_gain_count=member_gain.minimum_gain_count,
        total_member_gain_count=member_gain.total_gain_count,
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
