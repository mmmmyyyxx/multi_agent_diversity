"""Compatibility wrapper around the unchanged Common-Safe team authority."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..candidate_selection import CandidateEvaluation, common_monotone_safe_key, evaluate_constraints
from .schemas import TeamCandidateRecord


class CommonSafeTeamCandidateSelector:
    def annotate(
        self,
        records: Sequence[TeamCandidateRecord],
        *,
        active: CandidateEvaluation,
    ) -> tuple[TeamCandidateRecord, ...]:
        annotated: list[TeamCandidateRecord] = []
        for row in records:
            if not row.promoted or row.full_evaluation is None:
                annotated.append(row)
                continue
            annotated.append(replace(row, constraint=evaluate_constraints(row.full_evaluation, active)))
        return tuple(annotated)

    def select(self, records: Sequence[TeamCandidateRecord]) -> TeamCandidateRecord | None:
        feasible = [
            row
            for row in records
            if row.promoted
            and row.full_evaluation is not None
            and row.constraint is not None
            and row.constraint.passed
        ]
        return max(
            feasible,
            key=lambda row: common_monotone_safe_key(
                row.full_evaluation, row.local_candidate.generation  # type: ignore[arg-type]
            ),
            default=None,
        )
