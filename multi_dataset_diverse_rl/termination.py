"""Deterministic trajectory termination semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


COMPLETED_BY_BUDGET = "completed_by_budget"
COMPLETED_BY_EARLY_STOP = "completed_by_early_stop"
INCOMPLETE = "incomplete"
USER_ABORTED_INCOMPLETE = "user_aborted_incomplete"
OPERATIONAL_FAILURE = "operational_failure"

_SHADOW_STOP = re.compile(r"^no_shadow_approved_commit_streak_(\d+)$")


@dataclass(frozen=True)
class TerminationAssessment:
    status: str
    planned_update_opportunities: int
    executed_update_opportunities: int
    remaining_unexecuted: int
    terminal_update_index: int | None
    terminal_update_ordinal: int | None
    final_no_commit_streak: int
    early_stop_reason: str
    errors: tuple[str, ...] = ()

    @property
    def training_completed(self) -> bool:
        return self.status in {COMPLETED_BY_BUDGET, COMPLETED_BY_EARLY_STOP}

    def to_dict(self) -> dict[str, Any]:
        return {
            "termination_status": self.status,
            "training_completed": self.training_completed,
            "planned_update_opportunities": self.planned_update_opportunities,
            "executed_update_opportunities": self.executed_update_opportunities,
            "remaining_unexecuted": self.remaining_unexecuted,
            "terminal_update_index": self.terminal_update_index,
            "terminal_update_ordinal": self.terminal_update_ordinal,
            "final_no_commit_streak": self.final_no_commit_streak,
            "early_stop_reason": self.early_stop_reason,
            "errors": list(self.errors),
        }


def assess_trajectory_termination(
    *,
    planned_update_opportunities: int,
    executed_update_records: Sequence[Mapping[str, Any]],
    stored_early_stop_reason: str,
    completed_update_count: int | None = None,
    user_aborted: bool = False,
    operational_failure: bool = False,
) -> TerminationAssessment:
    """Reconstruct completion from ordered update records without trusting a label."""
    planned = int(planned_update_opportunities)
    executed = (
        len(executed_update_records)
        if completed_update_count is None
        else int(completed_update_count)
    )
    reason = str(stored_early_stop_reason or "")
    errors: list[str] = []
    if planned < 0 or executed < 0 or executed > planned:
        errors.append("invalid_update_counts")
    indices: list[int] = []
    approvals: list[bool] = []
    for ordinal, row in enumerate(executed_update_records):
        try:
            indices.append(int(row["update_index"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_update_index:{ordinal}")
        value = row.get("writeback_approved")
        if not isinstance(value, bool):
            errors.append(f"invalid_writeback_outcome:{ordinal}")
        else:
            approvals.append(value)
    if len(executed_update_records) != executed:
        errors.append("executed_record_count_mismatch")
    if indices != list(range(executed)):
        errors.append("executed_update_indices_not_exact_prefix")

    streak = 0
    for approved in approvals:
        streak = 0 if approved else streak + 1
    terminal_index = executed - 1 if executed else None
    terminal_ordinal = executed if executed else None

    if not errors and executed == planned:
        status = COMPLETED_BY_BUDGET
    else:
        match = _SHADOW_STOP.fullmatch(reason)
        required_streak = int(match.group(1)) if match else None
        if (
            not errors
            and executed < planned
            and required_streak is not None
            and required_streak > 0
            and streak == required_streak
        ):
            status = COMPLETED_BY_EARLY_STOP
        elif operational_failure:
            status = OPERATIONAL_FAILURE
        elif user_aborted:
            status = USER_ABORTED_INCOMPLETE
        else:
            status = INCOMPLETE
        if executed < planned and reason and required_streak is None:
            errors.append("unsupported_early_stop_reason")
        elif (
            executed < planned
            and required_streak is not None
            and streak != required_streak
        ):
            errors.append("early_stop_streak_mismatch")

    return TerminationAssessment(
        status=status,
        planned_update_opportunities=planned,
        executed_update_opportunities=executed,
        remaining_unexecuted=max(0, planned - executed),
        terminal_update_index=terminal_index,
        terminal_update_ordinal=terminal_ordinal,
        final_no_commit_streak=streak,
        early_stop_reason=reason,
        errors=tuple(errors),
    )
