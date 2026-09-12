"""Opt-in production binding for primary-responsibility target scheduling.

This module owns orchestration only.  It does not alter Local GEPA, candidate
evaluation, Common-Safe ranking, Shadow, or commit semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from ..responsibility import MemberAwareRepairOpportunity
from .controller import TeamSearchController
from .primary_responsibility_scheduler import (
    PrimaryResponsibilityPersistentRealizabilityScheduler,
    PrimaryResponsibilitySummary,
    PrimaryTargetSelection,
    RealizabilityTransition,
)
from .schemas import TeamSearchAssignment, TeamSearchOutcome, TeamSearchRequest


class PrimaryResponsibilityAssignmentFactory(Protocol):
    """Translate one frozen scheduler summary into one target assignment."""

    def build(
        self,
        *,
        request: TeamSearchRequest,
        summary: PrimaryResponsibilitySummary,
    ) -> TeamSearchAssignment:
        ...


@dataclass(frozen=True)
class PrimaryResponsibilityOpportunityOutcome:
    decision: PrimaryTargetSelection
    assignments: tuple[TeamSearchAssignment, ...]
    team_outcome: TeamSearchOutcome
    realizability_transitions: tuple[RealizabilityTransition, ...]


class PrimaryResponsibilityOnlineBinding:
    """Wire scheduler selection and durable outcome recording around Layer 2."""

    def __init__(
        self,
        *,
        scheduler: PrimaryResponsibilityPersistentRealizabilityScheduler,
        assignment_factory: PrimaryResponsibilityAssignmentFactory,
        controller: TeamSearchController,
    ) -> None:
        self.scheduler = scheduler
        self.assignment_factory = assignment_factory
        self.controller = controller

    async def run_opportunity(
        self,
        request: TeamSearchRequest,
        *,
        assigned: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
        current_margin_by_question: Mapping[str, int],
    ) -> PrimaryResponsibilityOpportunityOutcome:
        if request.update_index in self.scheduler.state.completed_update_indices:
            raise ValueError("optimization opportunity outcome was already recorded")
        decision = self.scheduler.select(
            assigned=assigned,
            current_margin_by_question=current_margin_by_question,
            seed=request.seed,
            update_index=request.update_index,
            target_count=2,
        )
        if len(decision.selected_member_ids) != 2:
            raise ValueError("online binding requires exactly two frozen target branches")
        by_member = {row.member_id: row for row in decision.summaries}
        assignments: tuple[TeamSearchAssignment, ...] = ()
        try:
            assignments = tuple(
                self.assignment_factory.build(
                    request=request,
                    summary=by_member[member_id],
                )
                for member_id in decision.selected_member_ids
            )
            if tuple(row.target_member for row in assignments) != decision.selected_member_ids:
                raise ValueError("assignment targets do not match the frozen scheduler decision")
            if any(
                row.primary_responsibility_lane != by_member[row.target_member].primary_lane
                for row in assignments
            ):
                raise ValueError("assignment primary lane does not match the frozen summary")
            team_outcome = await self.controller.run_frozen_opportunity(request, assignments)
        except Exception:
            self.scheduler.record_outcome(
                decision=decision,
                update_index=request.update_index,
                committed_member_id=None,
                valid_outcome=False,
            )
            raise

        committed_member = (
            int(team_outcome.assignment.target_member)
            if team_outcome.committed_candidate_id is not None
            else None
        )
        if team_outcome.audit_metadata.get("committed_member_id") != committed_member:
            raise ValueError("team outcome commit identity is inconsistent")
        transitions = self.scheduler.record_outcome(
            decision=decision,
            update_index=request.update_index,
            committed_member_id=committed_member,
            valid_outcome=True,
        )
        return PrimaryResponsibilityOpportunityOutcome(
            decision=decision,
            assignments=assignments,
            team_outcome=team_outcome,
            realizability_transitions=transitions,
        )
