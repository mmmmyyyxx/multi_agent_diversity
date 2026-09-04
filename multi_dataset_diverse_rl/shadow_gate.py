"""Winner-only shadow write-back gate for anti-overfitting experiments.

This module deliberately contains no search or ranking logic.  It receives one
already-frozen Optimize winner and returns a write-back decision from aggregate
Shadow metrics only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


SHADOW_GATE_VERSION = "winner_only_shadow_gate_v1"
SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT = 2
MAX_NO_SHADOW_APPROVED_COMMIT_STREAK = 6


@dataclass(frozen=True)
class ShadowGateMetrics:
    incumbent_vote_correct: int
    candidate_vote_correct: int
    incumbent_target_correct: int
    candidate_target_correct: int
    row_count: int

    @property
    def vote_delta(self) -> int:
        return self.candidate_vote_correct - self.incumbent_vote_correct

    @property
    def target_delta(self) -> int:
        return self.candidate_target_correct - self.incumbent_target_correct


@dataclass(frozen=True)
class ShadowGateDecision:
    passed: bool
    reasons: tuple[str, ...]
    metrics: ShadowGateMetrics
    gate_version: str = SHADOW_GATE_VERSION
    catastrophic_target_loss_count: int = SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT

    def sanitized(self) -> dict[str, Any]:
        value = asdict(self)
        value["metrics"]["vote_delta"] = self.metrics.vote_delta
        value["metrics"]["target_delta"] = self.metrics.target_delta
        return value


def evaluate_shadow_gate(
    metrics: ShadowGateMetrics,
    *,
    catastrophic_target_loss_count: int = SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT,
) -> ShadowGateDecision:
    """Apply the frozen v1 gate without scores, retries, or feedback."""
    if metrics.row_count != 50:
        raise ValueError("shadow gate requires exactly 50 rows")
    if catastrophic_target_loss_count < 0:
        raise ValueError("catastrophic threshold must be non-negative")
    reasons: list[str] = []
    if metrics.vote_delta < 0:
        reasons.append("shadow_vote_regression")
    if metrics.target_delta < -catastrophic_target_loss_count:
        reasons.append("catastrophic_target_member_regression")
    return ShadowGateDecision(not reasons, tuple(reasons), metrics,
                              catastrophic_target_loss_count=catastrophic_target_loss_count)


def shadow_false_positive(decision: ShadowGateDecision) -> bool:
    """Optimize Common-Safe winners with negative Shadow Vote transfer."""
    return decision.metrics.vote_delta < 0


def advance_no_commit_streak(current: int, *, committed: bool) -> tuple[int, bool]:
    updated = 0 if committed else current + 1
    return updated, updated >= MAX_NO_SHADOW_APPROVED_COMMIT_STREAK


def assert_winner_only_event(event: Mapping[str, Any]) -> None:
    """Fail closed on telemetry that suggests adaptive Shadow use."""
    if int(event.get("shadow_candidate_count", -1)) != 1:
        raise AssertionError("shadow evaluation must receive exactly one winner")
    forbidden = (
        "shadow_rank", "shadow_retry_count", "shadow_teacher_feedback",
        "shadow_revision", "shadow_selected_candidate",
    )
    if any(event.get(key) not in (None, False, 0, "", []) for key in forbidden):
        raise AssertionError("shadow evidence entered adaptive search")
