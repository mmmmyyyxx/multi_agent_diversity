"""Opt-in Layer-2 primary-responsibility scheduler.

The scheduler deliberately knows nothing about LocalPromptOptimizer backends.
Its persistent failure counter is member-local and is never reset by a team
hash change or by a teammate's successful write-back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Mapping, Sequence

from ..responsibility import MemberAwareRepairOpportunity
from ..versions import (
    PRIMARY_RESPONSIBILITY_COVERAGE_WEIGHT,
    PRIMARY_RESPONSIBILITY_DIRECT_WEIGHT,
    PRIMARY_RESPONSIBILITY_NEAR_MARGIN_WEIGHT,
    PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
)
from ..vote_aligned_scheduler import classify_opportunity_lane


DIRECT_FLIP = "direct_flip"
NEAR_MARGIN = "near_margin"
COVERAGE = "coverage"
FALLBACK = "fallback"
LANE_TIE_ORDER = (DIRECT_FLIP, NEAR_MARGIN, COVERAGE)


@dataclass(frozen=True)
class PrimaryResponsibilitySummary:
    member_id: int
    direct_count: int
    near_margin_count: int
    coverage_count: int
    direct_score: int
    near_margin_score: int
    coverage_score: int
    primary_lane: str
    primary_score: int
    runner_up_lane: str
    runner_up_score: int
    failure_count: int
    realizability: float
    target_score: float


@dataclass(frozen=True)
class PrimaryTargetSelection:
    scheduler_version: str
    summaries: tuple[PrimaryResponsibilitySummary, ...]
    selected_member_ids: tuple[int, ...]
    rr_cursor_before: int
    rr_cursor_after: int
    fallback_used: bool


@dataclass(frozen=True)
class RealizabilityTransition:
    update_index: int
    member_id: int
    selected: bool
    committed: bool
    valid_outcome: bool
    failure_count_before: int
    realizability_before: float
    failure_count_after: int
    realizability_after: float
    primary_lane: str


@dataclass
class PersistentRealizabilityState:
    failure_count_by_member: dict[int, int] = field(default_factory=dict)
    rr_cursor: int = 0
    target_count_by_member: dict[int, int] = field(default_factory=dict)
    commit_count_by_member: dict[int, int] = field(default_factory=dict)
    primary_lane_target_counts: dict[str, int] = field(default_factory=dict)
    primary_lane_commit_counts: dict[str, int] = field(default_factory=dict)
    transitions: list[RealizabilityTransition] = field(default_factory=list)

    def initialize(self, member_ids: Sequence[int]) -> None:
        for member_id in map(int, member_ids):
            self.failure_count_by_member.setdefault(member_id, 0)
            self.target_count_by_member.setdefault(member_id, 0)
            self.commit_count_by_member.setdefault(member_id, 0)
        for lane in (*LANE_TIE_ORDER, FALLBACK):
            self.primary_lane_target_counts.setdefault(lane, 0)
            self.primary_lane_commit_counts.setdefault(lane, 0)

    def checkpoint_payload(self) -> dict[str, object]:
        """Return a JSON-safe resume payload with no team-hash coupling."""
        return {
            "schema_version": "persistent_member_realizability_state_v1",
            "scheduler_version": PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
            "failure_count_by_member": dict(self.failure_count_by_member),
            "rr_cursor": int(self.rr_cursor),
            "target_count_by_member": dict(self.target_count_by_member),
            "commit_count_by_member": dict(self.commit_count_by_member),
            "primary_lane_target_counts": dict(self.primary_lane_target_counts),
            "primary_lane_commit_counts": dict(self.primary_lane_commit_counts),
        }

    @classmethod
    def from_checkpoint_payload(
        cls, payload: Mapping[str, object], *, member_ids: Sequence[int]
    ) -> "PersistentRealizabilityState":
        if payload.get("schema_version") != "persistent_member_realizability_state_v1":
            raise ValueError("persistent realizability checkpoint schema mismatch")
        if payload.get("scheduler_version") != PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION:
            raise ValueError("persistent realizability scheduler version mismatch")

        def member_counts(key: str) -> dict[int, int]:
            value = payload.get(key)
            if not isinstance(value, Mapping):
                raise ValueError(f"missing persistent realizability field: {key}")
            result = {int(member): int(count) for member, count in value.items()}
            if any(count < 0 for count in result.values()):
                raise ValueError(f"negative persistent realizability field: {key}")
            return result

        def lane_counts(key: str) -> dict[str, int]:
            value = payload.get(key)
            if not isinstance(value, Mapping):
                raise ValueError(f"missing persistent realizability field: {key}")
            result = {str(lane): int(count) for lane, count in value.items()}
            if set(result) != {*LANE_TIE_ORDER, FALLBACK} or any(count < 0 for count in result.values()):
                raise ValueError(f"invalid persistent realizability lane field: {key}")
            return result

        state = cls(
            failure_count_by_member=member_counts("failure_count_by_member"),
            rr_cursor=int(payload.get("rr_cursor", -1)),
            target_count_by_member=member_counts("target_count_by_member"),
            commit_count_by_member=member_counts("commit_count_by_member"),
            primary_lane_target_counts=lane_counts("primary_lane_target_counts"),
            primary_lane_commit_counts=lane_counts("primary_lane_commit_counts"),
        )
        if state.rr_cursor < 0:
            raise ValueError("persistent realizability rr cursor cannot be negative")
        expected = set(map(int, member_ids))
        for key, value in (
            ("failure", state.failure_count_by_member),
            ("target", state.target_count_by_member),
            ("commit", state.commit_count_by_member),
        ):
            if set(value) != expected:
                raise ValueError(f"persistent realizability {key} member identity mismatch")
        return state


def _primary_lane_and_runner_up(scores: Mapping[str, int]) -> tuple[str, int, str, int]:
    ordered = sorted(scores, key=lambda lane: (-int(scores[lane]), LANE_TIE_ORDER.index(lane)))
    primary, runner_up = ordered[:2]
    return primary, int(scores[primary]), runner_up, int(scores[runner_up])


def build_primary_responsibility_summary_from_counts(
    *,
    member_id: int,
    direct_count: int,
    near_margin_count: int,
    coverage_count: int,
    failure_count: int,
) -> PrimaryResponsibilitySummary:
    counts = {
        DIRECT_FLIP: int(direct_count),
        NEAR_MARGIN: int(near_margin_count),
        COVERAGE: int(coverage_count),
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("responsibility counts cannot be negative")
    scores = {
        DIRECT_FLIP: PRIMARY_RESPONSIBILITY_DIRECT_WEIGHT * counts[DIRECT_FLIP],
        NEAR_MARGIN: PRIMARY_RESPONSIBILITY_NEAR_MARGIN_WEIGHT * counts[NEAR_MARGIN],
        COVERAGE: PRIMARY_RESPONSIBILITY_COVERAGE_WEIGHT * counts[COVERAGE],
    }
    primary, primary_score, runner_up, runner_up_score = _primary_lane_and_runner_up(scores)
    if primary_score == 0:
        primary = FALLBACK
    failures = int(failure_count)
    if failures < 0:
        raise ValueError("realizability failure count cannot be negative")
    realizability = 1.0 / (1.0 + failures)
    return PrimaryResponsibilitySummary(
        member_id=int(member_id),
        direct_count=counts[DIRECT_FLIP],
        near_margin_count=counts[NEAR_MARGIN],
        coverage_count=counts[COVERAGE],
        direct_score=scores[DIRECT_FLIP],
        near_margin_score=scores[NEAR_MARGIN],
        coverage_score=scores[COVERAGE],
        primary_lane=primary,
        primary_score=primary_score,
        runner_up_lane=runner_up,
        runner_up_score=runner_up_score,
        failure_count=failures,
        realizability=realizability,
        target_score=primary_score * realizability,
    )


def build_primary_responsibility_summaries(
    *,
    assigned: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
    current_margin_by_question: Mapping[str, int],
    failure_count_by_member: Mapping[int, int],
    member_ids: Sequence[int] = (0, 1, 2, 3, 4),
) -> tuple[PrimaryResponsibilitySummary, ...]:
    """Construct every member summary once from the frozen parent state."""
    summaries: list[PrimaryResponsibilitySummary] = []
    for member_id in map(int, member_ids):
        counts = {DIRECT_FLIP: 0, NEAR_MARGIN: 0, COVERAGE: 0}
        for row in assigned.get(member_id, ()):
            lane = classify_opportunity_lane(row, current_margin_by_question)
            if lane == "pure_coverage":
                lane = COVERAGE
            if lane in counts:
                counts[lane] += 1
        summaries.append(build_primary_responsibility_summary_from_counts(
            member_id=member_id,
            direct_count=counts[DIRECT_FLIP],
            near_margin_count=counts[NEAR_MARGIN],
            coverage_count=counts[COVERAGE],
            failure_count=int(failure_count_by_member.get(member_id, 0)),
        ))
    return tuple(summaries)


def select_primary_responsibility_targets(
    summaries: Sequence[PrimaryResponsibilitySummary],
    *,
    seed: int,
    update_index: int,
    rr_cursor: int,
    target_count: int = 2,
) -> PrimaryTargetSelection:
    """Select Top-2 by score; equal-score groups use deterministic stateful RR."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    rows = tuple(summaries)
    if len({row.member_id for row in rows}) != len(rows):
        raise ValueError("member summaries must be unique")
    fallback = not any(row.target_score > 0 for row in rows)
    start = (
        (int(seed) + 2 * int(update_index))
        if fallback
        else (int(seed) + int(update_index) + int(rr_cursor))
    ) % max(1, len(rows))
    cyclic_rank = {
        rows[(start + offset) % len(rows)].member_id: offset
        for offset in range(len(rows))
    } if rows else {}
    ordered = sorted(
        rows,
        key=(
            (lambda row: (cyclic_rank[row.member_id], row.member_id))
            if fallback
            else (lambda row: (-row.target_score, cyclic_rank[row.member_id], row.member_id))
        ),
    )
    chosen = tuple(row.member_id for row in ordered[: min(target_count, len(ordered))])
    return PrimaryTargetSelection(
        scheduler_version=PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
        summaries=rows,
        selected_member_ids=chosen,
        rr_cursor_before=int(rr_cursor),
        rr_cursor_after=int(rr_cursor) + 1,
        fallback_used=fallback,
    )


class PrimaryResponsibilityPersistentRealizabilityScheduler:
    """Stateful Layer-2 policy with member-local, persistent realizability."""

    version = PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION

    def __init__(
        self,
        *,
        member_ids: Sequence[int] = (0, 1, 2, 3, 4),
        state: PersistentRealizabilityState | None = None,
    ) -> None:
        self.member_ids = tuple(map(int, member_ids))
        self.state = state or PersistentRealizabilityState()
        self.state.initialize(self.member_ids)

    def select(
        self,
        *,
        assigned: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
        current_margin_by_question: Mapping[str, int],
        seed: int,
        update_index: int,
        target_count: int = 2,
    ) -> PrimaryTargetSelection:
        summaries = build_primary_responsibility_summaries(
            assigned=assigned,
            current_margin_by_question=current_margin_by_question,
            failure_count_by_member=self.state.failure_count_by_member,
            member_ids=self.member_ids,
        )
        decision = select_primary_responsibility_targets(
            summaries,
            seed=seed,
            update_index=update_index,
            rr_cursor=self.state.rr_cursor,
            target_count=target_count,
        )
        self.state.rr_cursor = decision.rr_cursor_after
        return decision

    def record_outcome(
        self,
        *,
        decision: PrimaryTargetSelection,
        update_index: int,
        committed_member_id: int | None,
        valid_outcome: bool,
    ) -> tuple[RealizabilityTransition, ...]:
        """Apply only a valid completed optimization outcome.

        A selected member resets only on its own successful commit.  A teammate
        commit does not reset it; an operational abort changes no counter.
        """
        by_member = {row.member_id: row for row in decision.summaries}
        selected = set(decision.selected_member_ids)
        if committed_member_id is not None and committed_member_id not in selected:
            raise ValueError("committed member must be one of the frozen targets")
        if committed_member_id is not None and not valid_outcome:
            raise ValueError("operational abort cannot contain a commit")
        emitted: list[RealizabilityTransition] = []
        for member_id in self.member_ids:
            before = self.state.failure_count_by_member[member_id]
            committed = valid_outcome and member_id == committed_member_id
            is_selected = member_id in selected
            after = before
            if valid_outcome and is_selected:
                after = 0 if committed else before + 1
                self.state.target_count_by_member[member_id] += 1
                lane = by_member[member_id].primary_lane
                self.state.primary_lane_target_counts[lane] += 1
                if committed:
                    self.state.commit_count_by_member[member_id] += 1
                    self.state.primary_lane_commit_counts[lane] += 1
            self.state.failure_count_by_member[member_id] = after
            transition = RealizabilityTransition(
                update_index=int(update_index),
                member_id=member_id,
                selected=is_selected,
                committed=committed,
                valid_outcome=bool(valid_outcome),
                failure_count_before=before,
                realizability_before=1.0 / (1.0 + before),
                failure_count_after=after,
                realizability_after=1.0 / (1.0 + after),
                primary_lane=by_member[member_id].primary_lane,
            )
            self.state.transitions.append(transition)
            emitted.append(transition)
        return tuple(emitted)

    def telemetry_summary(self) -> dict[str, object]:
        targets = dict(self.state.target_count_by_member)
        commits = dict(self.state.commit_count_by_member)
        total = sum(targets.values())
        values = list(targets.values())
        entropy = 0.0
        if total:
            entropy = -sum((value / total) * log(value / total) for value in values if value)
        gini = 0.0
        if total and values:
            gini = sum(abs(a - b) for a in values for b in values) / (2 * len(values) * total)
        return {
            "scheduler_version": self.version,
            "target_count_by_member": targets,
            "commit_count_by_member": commits,
            "target_to_commit_rate_by_member": {
                member: (commits[member] / count if count else None)
                for member, count in targets.items()
            },
            "primary_lane_target_counts": dict(self.state.primary_lane_target_counts),
            "primary_lane_commit_counts": dict(self.state.primary_lane_commit_counts),
            "target_entropy": entropy,
            "target_gini": gini,
        }
