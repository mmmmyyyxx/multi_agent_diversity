"""Shared V4 feasibility-aware target materialization.

This is the single production selector used by the diagnostic and formal
Layer-2 composition. It does not perform provider calls or alter GEPA search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .feasibility import Layer2EvidenceInfeasible, Layer2FeasibilityReason
from .primary_responsibility_scheduler import (
    PrimaryResponsibilityPersistentRealizabilityScheduler,
    PrimaryTargetSelection,
)
from .schemas import TeamSearchAssignment, TeamSearchRequest
from .system_runtime import FrozenResponsibilitySnapshot, SystemResponsibilityAssignmentFactory
from .task_builder import Layer2EvidenceRequestBuilder
from ..versions import PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION


@dataclass(frozen=True)
class V4FeasibleOpportunity:
    decision: PrimaryTargetSelection
    assignment: TeamSearchAssignment
    packet: Any
    eligible_member_ids: tuple[int, ...]
    feasibility_reasons: tuple[tuple[int, str], ...]
    raw_decision: PrimaryTargetSelection


@dataclass(frozen=True)
class V4NoFeasibleOpportunity:
    raw_decision: PrimaryTargetSelection
    feasibility_reasons: tuple[tuple[int, str], ...]


def select_v4_feasible_opportunity(
    *, snapshot: FrozenResponsibilitySnapshot,
    scheduler: PrimaryResponsibilityPersistentRealizabilityScheduler,
    factory: SystemResponsibilityAssignmentFactory,
    task_builder: Layer2EvidenceRequestBuilder,
    request: TeamSearchRequest,
    raw_decision: PrimaryTargetSelection | None = None,
) -> V4FeasibleOpportunity | V4NoFeasibleOpportunity:
    """Mask by positive raw V and complete evidence feasibility, then rank.

    A ``V4NoFeasibleOpportunity`` result means E(s) is empty. The caller must stop before an
    opportunity begins, leaving f, epoch and provider accounting unchanged.
    """

    if not task_builder.bounded_search_view:
        raise ValueError("V4 requires bounded Layer-2 search evidence")
    raw = raw_decision or scheduler.select(
        assigned=snapshot.assigned,
        current_margin_by_question=snapshot.current_margin_by_question,
        seed=request.seed, update_index=request.update_index, target_count=1,
    )
    feasible: dict[int, tuple[TeamSearchAssignment, Any]] = {}
    reasons: dict[int, str] = {}
    for row in raw.summaries:
        if row.primary_score <= 0:
            reasons[row.member_id] = Layer2FeasibilityReason.NO_POSITIVE_RESPONSIBILITY.value
            continue
        try:
            assignment = factory.build_from_member(
                request=request, member_id=row.member_id,
                primary_lane=row.primary_lane,
                responsibility_identity=PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
                responsibility_value=row.primary_score,
            )
            packet = task_builder.build(request, assignment).packet
        except Layer2EvidenceInfeasible as exc:
            reasons[row.member_id] = exc.reason.value
        else:
            feasible[row.member_id] = (assignment, packet)
            reasons[row.member_id] = Layer2FeasibilityReason.FEASIBLE.value
    if not feasible:
        return V4NoFeasibleOpportunity(
            raw_decision=raw,
            feasibility_reasons=tuple(sorted(reasons.items())),
        )
    decision = scheduler.select(
        assigned=snapshot.assigned,
        current_margin_by_question=snapshot.current_margin_by_question,
        seed=request.seed, update_index=request.update_index, target_count=1,
        eligible_member_ids=tuple(feasible),
    )
    if len(decision.selected_member_ids) != 1:
        raise RuntimeError("V4 requires one selected feasible target")
    target = decision.selected_member_ids[0]
    assignment, packet = feasible[target]
    summary = next(row for row in decision.summaries if row.member_id == target)
    if assignment.responsibility_value != summary.primary_score or packet.responsibility_value != summary.primary_score:
        raise RuntimeError("V4 raw responsibility V mismatch")
    return V4FeasibleOpportunity(
        decision=decision, assignment=assignment, packet=packet,
        eligible_member_ids=tuple(sorted(feasible)),
        feasibility_reasons=tuple(sorted(reasons.items())),
        raw_decision=raw,
    )
