"""Prospective trajectory endpoints for safe plurality symmetry breaking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SymmetryBreakingEvent:
    """One completed online opportunity and its realized post-opportunity state."""

    opportunity: int
    committed: bool
    commit_index: int
    vote_correct: int
    p_i: tuple[int, int, int, int, int]

    @property
    def total_pivotal_opportunities(self) -> int:
        return sum(self.p_i)


class SymmetryBreakingTrajectory:
    """Track frozen time-to-event endpoints without influencing target selection."""

    def __init__(self, *, baseline_vote_correct: int, baseline_p_i: Mapping[str, int]) -> None:
        ordered = tuple(int(baseline_p_i[str(member)]) for member in range(5))
        if any(value < 0 for value in ordered):
            raise ValueError("P_i counts must be non-negative")
        self.baseline_vote_correct = int(baseline_vote_correct)
        self.baseline_p_i = ordered
        self.events: list[SymmetryBreakingEvent] = []
        self.commit_count = 0
        self.no_commit_streak = 0
        self.t_commit_opportunity: int | None = None
        self.t_pivotal_opportunity: int | None = None
        self.t_pivotal_commit_index: int | None = None
        self.t_vote_opportunity: int | None = None
        self.t_vote_commit_index: int | None = None

    def observe(
        self,
        *,
        committed: bool,
        vote_correct: int,
        p_i: Mapping[str, int],
    ) -> SymmetryBreakingEvent:
        if committed:
            self.commit_count += 1
            self.no_commit_streak = 0
        else:
            self.no_commit_streak += 1
        opportunity = len(self.events) + 1
        ordered = tuple(int(p_i[str(member)]) for member in range(5))
        if any(value < 0 for value in ordered):
            raise ValueError("P_i counts must be non-negative")
        event = SymmetryBreakingEvent(
            opportunity=opportunity,
            committed=bool(committed),
            commit_index=self.commit_count,
            vote_correct=int(vote_correct),
            p_i=ordered,
        )
        self.events.append(event)
        if committed and self.t_commit_opportunity is None:
            self.t_commit_opportunity = opportunity
        if sum(ordered) > 0 and self.t_pivotal_opportunity is None:
            self.t_pivotal_opportunity = opportunity
            self.t_pivotal_commit_index = self.commit_count
        if vote_correct > self.baseline_vote_correct and self.t_vote_opportunity is None:
            self.t_vote_opportunity = opportunity
            self.t_vote_commit_index = self.commit_count
        return event

    def stop_reason(
        self,
        *,
        max_opportunities: int,
        max_commits: int,
        no_commit_patience: int,
    ) -> str | None:
        if self.t_vote_opportunity is not None:
            return "first_vote_improvement"
        if self.commit_count >= max_commits:
            return "max_safe_commits"
        if self.no_commit_streak >= no_commit_patience:
            return "no_commit_patience"
        if len(self.events) >= max_opportunities:
            return "max_opportunities"
        return None

    def payload(self) -> dict[str, Any]:
        return {
            "baseline_vote_correct": self.baseline_vote_correct,
            "baseline_p_i": list(self.baseline_p_i),
            "opportunities": len(self.events),
            "safe_commits": self.commit_count,
            "no_commit_streak": self.no_commit_streak,
            "T_commit_opportunity": self.t_commit_opportunity,
            "T_pivotal_opportunity": self.t_pivotal_opportunity,
            "T_pivotal_commit_index": self.t_pivotal_commit_index,
            "T_vote_opportunity": self.t_vote_opportunity,
            "T_vote_commit_index": self.t_vote_commit_index,
            "events": [asdict(event) for event in self.events],
        }


def classify_symmetry_breaking_trajectory(
    trajectory: Mapping[str, Any], *, integrity_passed: bool
) -> str:
    """Apply the preregistered descriptive mechanism classifier."""
    baseline = tuple(int(value) for value in trajectory.get("baseline_p_i", ()))
    if not integrity_passed or len(baseline) != 5 or sum(baseline) != 0:
        return "HOLD_BASELINE_OR_INTEGRITY_FAILURE"
    t_commit = trajectory.get("T_commit_opportunity")
    t_pivotal = trajectory.get("T_pivotal_opportunity")
    t_vote = trajectory.get("T_vote_opportunity")
    if t_vote is not None:
        if t_commit is None or t_pivotal is None or not (t_commit <= t_pivotal <= t_vote):
            return "HOLD_ENDPOINT_ORDER_VIOLATION"
        return "ALGORITHMIC_SYMMETRY_BREAKING_WITH_VOTE_GAIN_OBSERVED"
    if t_pivotal is not None:
        return "PLURALITY_RESPONSIVE_STATE_REACHED_WITHOUT_VOTE_GAIN"
    if int(trajectory.get("safe_commits", 0)) > 0:
        return "SAFE_COMMITS_WITHOUT_PLURALITY_RESPONSIVENESS"
    return "NO_SAFE_COMMIT_OBSERVED"
