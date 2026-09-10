"""Backend-agnostic Layer 2 orchestration facade."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from ..local_optimizers.base import LocalPromptOptimizer
from .candidate_evaluator import TeamCandidateEvaluator, TeamCommitter
from .candidate_selector import CommonSafeTeamCandidateSelector
from .progressive_evaluation import promote_team_candidates
from .schemas import (
    TeamCandidateRecord,
    TeamCostAccounting,
    TeamSearchAssignment,
    TeamSearchOutcome,
    TeamSearchRequest,
)
from .task_builder import LocalTaskBuilder


class ResponsibilityAssignmentProvider(Protocol):
    def assign(self, request: TeamSearchRequest) -> TeamSearchAssignment:
        ...


class TeamSearchController:
    """Local optimizer proposes; this controller evaluates, selects, and commits."""

    def __init__(
        self,
        *,
        responsibility: ResponsibilityAssignmentProvider,
        task_builder: LocalTaskBuilder,
        local_optimizer: LocalPromptOptimizer,
        evaluator: TeamCandidateEvaluator,
        selector: CommonSafeTeamCandidateSelector,
        committer: TeamCommitter,
    ) -> None:
        self.responsibility = responsibility
        self.task_builder = task_builder
        self.local_optimizer = local_optimizer
        self.evaluator = evaluator
        self.selector = selector
        self.committer = committer

    async def run_opportunity(self, request: TeamSearchRequest) -> TeamSearchOutcome:
        assignment = self.responsibility.assign(request)
        task = self.task_builder.build(request, assignment)
        team_minibatch = self.task_builder.select_team_minibatch(assignment.evidence)
        local = await self.local_optimizer.optimize(task)
        records: list[TeamCandidateRecord] = []
        minibatch_calls = minibatch_tokens = 0
        for candidate in local.candidates:
            metrics, cost = self.evaluator.evaluate_minibatch(assignment, candidate, team_minibatch)
            records.append(TeamCandidateRecord(candidate, minibatch_metrics=metrics))
            minibatch_calls += cost.solver_calls
            minibatch_tokens += cost.total_tokens
        promoted = list(promote_team_candidates(records, max_promoted=2))
        full_calls = full_tokens = 0
        for index, row in enumerate(promoted):
            if not row.promoted:
                continue
            evaluation, cost = self.evaluator.evaluate_full(assignment, row.local_candidate)
            promoted[index] = replace(row, full_evaluation=evaluation)
            full_calls += cost.solver_calls
            full_tokens += cost.total_tokens
        active = self.evaluator.active_evaluation(assignment)
        annotated = self.selector.annotate(promoted, active=active)
        winner = self.selector.select(annotated)
        committed_id: str | None = None
        shadow_calls = shadow_tokens = 0
        if winner is not None and winner.full_evaluation is not None:
            shadow, cost = self.evaluator.evaluate_shadow(assignment, winner.local_candidate)
            shadow_calls += cost.solver_calls
            shadow_tokens += cost.total_tokens
            if shadow.passed:
                self.committer.commit(
                    assignment=assignment,
                    candidate=winner.local_candidate,
                    evaluation=winner.full_evaluation,
                )
                committed_id = winner.local_candidate.candidate_id
        cost = TeamCostAccounting(
            local_optimizer_solver_calls=local.solver_calls,
            local_optimizer_meta_calls=local.optimizer_calls,
            local_optimizer_tokens=local.total_tokens,
            team_minibatch_solver_calls=minibatch_calls,
            team_full_solver_calls=full_calls,
            team_shadow_solver_calls=shadow_calls,
            team_tokens=minibatch_tokens + full_tokens + shadow_tokens,
        )
        feasible = sum(row.constraint is not None and row.constraint.passed for row in annotated)
        return TeamSearchOutcome(
            assignment=assignment,
            candidates=annotated,
            committed_candidate_id=committed_id,
            termination_reason="committed" if committed_id else "no_team_safe_winner",
            cost=cost,
            funnel={
                "local_candidates": len(local.candidates),
                "team_minibatch_survivors": sum(row.promoted for row in annotated),
                "full_team_evaluated_candidates": sum(row.full_evaluation is not None for row in annotated),
                "feasible_candidates": feasible,
                "committed_candidates": int(committed_id is not None),
            },
            audit_metadata={
                "local_optimizer_backend": local.backend_name,
                "local_optimizer_version": local.backend_version,
                "local_termination_reason": local.termination_reason,
                "team_minibatch_example_ids": [row.example_id for row in team_minibatch],
            },
        )
