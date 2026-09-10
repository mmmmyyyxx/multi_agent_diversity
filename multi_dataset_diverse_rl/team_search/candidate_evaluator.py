"""Layer-2-only interfaces for fixed-peer team evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..candidate_selection import CandidateEvaluation
from ..local_optimizers.schemas import LocalPromptCandidate
from ..shadow_gate import ShadowGateDecision
from .schemas import TeamEvidenceCase, TeamMiniBatchMetrics, TeamSearchAssignment


@dataclass(frozen=True)
class EvaluationCost:
    solver_calls: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TeamCandidateEvaluator(Protocol):
    def active_evaluation(self, assignment: TeamSearchAssignment) -> CandidateEvaluation:
        ...

    def evaluate_minibatch(
        self,
        assignment: TeamSearchAssignment,
        candidate: LocalPromptCandidate,
        minibatch: Sequence[TeamEvidenceCase],
    ) -> tuple[TeamMiniBatchMetrics, EvaluationCost]:
        ...

    def evaluate_full(
        self, assignment: TeamSearchAssignment, candidate: LocalPromptCandidate
    ) -> tuple[CandidateEvaluation, EvaluationCost]:
        ...

    def evaluate_shadow(
        self, assignment: TeamSearchAssignment, candidate: LocalPromptCandidate
    ) -> tuple[ShadowGateDecision, EvaluationCost]:
        ...


class TeamCommitter(Protocol):
    def commit(
        self,
        *,
        assignment: TeamSearchAssignment,
        candidate: LocalPromptCandidate,
        evaluation: CandidateEvaluation,
    ) -> None:
        ...
