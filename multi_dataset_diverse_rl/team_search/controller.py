"""Backend-agnostic Layer 2 orchestration facade."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence

from ..local_optimizers.base import (
    Layer2EvidencePromptOptimizer,
    LocalPromptOptimizer,
    NativeFeedPromptOptimizer,
)
from ..local_optimizers.backend_registry import Layer1Backend
from ..local_optimizers.schemas import LocalOptimizationResult, LocalOptimizationTask
from ..native_feed import Layer2OptimizationRequest, NativeOptimizationRequest
from ..candidate_selection import evaluate_constraints
from .candidate_evaluator import TeamCandidateEvaluator, TeamCommitter
from .candidate_selector import CommonSafeTeamCandidateSelector
from .progressive_evaluation import (
    has_promotion_signal, is_catastrophic, promote_team_candidates,
)
from .schemas import (
    TeamCandidateRecord,
    TeamCostAccounting,
    TeamSearchAssignment,
    TeamSearchOutcome,
    TeamSearchRequest,
)
from .task_builder import (
    Layer2EvidenceRequestBuilder,
    LocalTaskBuilder,
    NativeFeedRequestBuilder,
)


class ResponsibilityAssignmentProvider(Protocol):
    def assign(self, request: TeamSearchRequest) -> TeamSearchAssignment:
        ...


@dataclass(frozen=True)
class _EvaluatedBranch:
    assignment: TeamSearchAssignment
    candidates: tuple[TeamCandidateRecord, ...]
    cost: TeamCostAccounting
    audit_metadata: dict[str, Any]


class TeamSearchController:
    """Local optimizer proposes; this controller evaluates, selects, and commits."""

    def __init__(
        self,
        *,
        responsibility: ResponsibilityAssignmentProvider,
        task_builder: LocalTaskBuilder | NativeFeedRequestBuilder | Layer2EvidenceRequestBuilder,
        local_optimizer: (
            LocalPromptOptimizer | NativeFeedPromptOptimizer | Layer2EvidencePromptOptimizer
            | Layer1Backend
        ),
        evaluator: TeamCandidateEvaluator,
        selector: CommonSafeTeamCandidateSelector,
        committer: TeamCommitter,
        diagnostic_full_for_local_accepts: bool = False,
    ) -> None:
        self.responsibility = responsibility
        self.task_builder = task_builder
        self.local_optimizer = local_optimizer
        self.evaluator = evaluator
        self.selector = selector
        self.committer = committer
        self.diagnostic_full_for_local_accepts = diagnostic_full_for_local_accepts

    async def _run_local_optimizer(
        self,
        task: LocalOptimizationTask | NativeOptimizationRequest | Layer2OptimizationRequest,
    ) -> LocalOptimizationResult:
        if isinstance(self.local_optimizer, Layer1Backend):
            expected = (
                "layer2" if isinstance(task, Layer2OptimizationRequest)
                else "native" if isinstance(task, NativeOptimizationRequest)
                else None
            )
            if expected is None:
                raise TypeError("unified Layer1 backend does not accept legacy local tasks")
            if self.local_optimizer.optimization_mode != expected:
                raise TypeError(
                    "configured optimization_mode does not match the Layer-2 task builder"
                )
            return await self.local_optimizer.optimize(task)
        if isinstance(task, Layer2OptimizationRequest):
            if not isinstance(self.local_optimizer, Layer2EvidencePromptOptimizer):
                raise TypeError(
                    "Layer-2 evidence request requires a Layer2EvidencePromptOptimizer"
                )
            before = task.packet.packet_hash
            result = await self.local_optimizer.optimize_layer2(task)
            if task.packet.packet_hash != before:
                raise RuntimeError("Layer-1 backend mutated the Layer-2 evidence packet")
            return result
        if isinstance(task, NativeOptimizationRequest):
            if not isinstance(self.local_optimizer, NativeFeedPromptOptimizer):
                raise TypeError(
                    "native-feed request requires a NativeFeedPromptOptimizer"
                )
            return await self.local_optimizer.optimize_native(task)
        if not isinstance(self.local_optimizer, LocalPromptOptimizer):
            raise TypeError("legacy local task requires a LocalPromptOptimizer")
        return await self.local_optimizer.optimize(task)

    async def run_opportunity(self, request: TeamSearchRequest) -> TeamSearchOutcome:
        assignment = self.responsibility.assign(request)
        return await self.run_frozen_opportunity(request, (assignment,))

    async def run_local_empirical_canary(self, request: TeamSearchRequest) -> TeamSearchOutcome:
        """Technical stage boundary: stop before any team evaluation/write-back.

        This does not alter the normal search path. It permits a preregistered
        one-opportunity integration canary to observe the real local empirical
        funnel without turning it into an online team experiment.
        """

        assignment = self.responsibility.assign(request)
        task = self.task_builder.build(request, assignment)
        local = await self._run_local_optimizer(task)
        telemetry = (
            dict(local.optimizer_state.payload.get("telemetry", {}))
            if local.optimizer_state is not None else {}
        )
        if self.diagnostic_full_for_local_accepts:
            # The frozen 36-call GEPA search can accept at most one child.  Its
            # returned frontier must represent that child exactly; otherwise
            # this diagnostic cannot claim an uncensored accepted-mutation sample.
            accepted = int(telemetry.get("accepted_mutations", -1))
            if accepted != len(local.candidates) or accepted > 1:
                raise RuntimeError("diagnostic local-acceptance/frontier mismatch")
        telemetry = (
            dict(local.optimizer_state.payload.get("telemetry", {}))
            if local.optimizer_state is not None else {}
        )
        return TeamSearchOutcome(
            assignment=assignment,
            candidates=tuple(TeamCandidateRecord(row) for row in local.candidates),
            committed_candidate_id=None,
            termination_reason="LOCAL_EMPIRICAL_CANARY_BOUNDARY",
            cost=TeamCostAccounting(
                local_optimizer_solver_calls=local.solver_calls,
                local_optimizer_meta_calls=local.optimizer_calls,
                local_optimizer_tokens=local.total_tokens,
            ),
            funnel={
                "local_candidates": len(local.candidates),
                "team_minibatch_survivors": 0,
                "full_team_evaluated_candidates": 0,
                "committed_candidates": 0,
            },
            audit_metadata={
                "local_optimizer_backend": local.backend_name,
                "local_optimizer_version": local.backend_version,
                "local_termination_reason": local.termination_reason,
                "local_optimizer_telemetry": telemetry,
                "responsibility_packet_hash": (
                    task.packet.packet_hash
                    if isinstance(task, Layer2OptimizationRequest) else None
                ),
                "team_evaluation_reached": False,
            },
        )

    async def _evaluate_frozen_branch(
        self,
        request: TeamSearchRequest,
        assignment: TeamSearchAssignment,
    ) -> _EvaluatedBranch:
        task = self.task_builder.build(request, assignment)
        team_minibatch = self.task_builder.select_team_minibatch(
            assignment.evidence,
            primary_responsibility_lane=assignment.primary_responsibility_lane,
        )
        minibatch_telemetry = self.task_builder.team_minibatch_telemetry(team_minibatch)
        local = await self._run_local_optimizer(task)
        records: list[TeamCandidateRecord] = []
        minibatch_calls = minibatch_tokens = 0
        for candidate in local.candidates:
            metrics, cost = await asyncio.to_thread(
                self.evaluator.evaluate_minibatch,
                assignment,
                candidate,
                team_minibatch,
            )
            records.append(TeamCandidateRecord(candidate, minibatch_metrics=metrics))
            minibatch_calls += cost.solver_calls
            minibatch_tokens += cost.total_tokens
        promoted = list(promote_team_candidates(records, max_promoted=2))
        full_calls = full_tokens = 0
        for index, row in enumerate(promoted):
            if not row.promoted:
                continue
            evaluation, cost = await asyncio.to_thread(
                self.evaluator.evaluate_full,
                assignment,
                row.local_candidate,
            )
            promoted[index] = replace(row, full_evaluation=evaluation)
            full_calls += cost.solver_calls
            full_tokens += cost.total_tokens
        active = self.evaluator.active_evaluation(assignment)
        diagnostic_rows: list[dict[str, Any]] = []
        if self.diagnostic_full_for_local_accepts:
            diagnostic_evaluate = getattr(self.evaluator, "evaluate_diagnostic_full", None)
            responsiveness = getattr(self.evaluator, "parent_vote_responsiveness", None)
            if not callable(diagnostic_evaluate) or not callable(responsiveness):
                raise TypeError("diagnostic evaluator lacks read-only Full or parent responsiveness")
            parent_response = responsiveness(assignment)
            for row in promoted:
                local_acceptance_delta = row.local_candidate.backend_metadata.get("local_acceptance_delta")
                if local_acceptance_delta is None or float(local_acceptance_delta) <= 0:
                    raise RuntimeError("diagnostic candidate lacks strict local acceptance evidence")
                if row.minibatch_metrics is None:
                    raise RuntimeError("diagnostic candidate lacks TeamMiniBatch result")
                if row.promoted:
                    if row.full_evaluation is None:
                        raise RuntimeError("promoted candidate lacks ordinary Full result")
                    full = row.full_evaluation
                    diagnostic_only = False
                else:
                    full, diagnostic_cost = await asyncio.to_thread(
                        diagnostic_evaluate, assignment, row.local_candidate,
                    )
                    full_calls += diagnostic_cost.solver_calls
                    full_tokens += diagnostic_cost.total_tokens
                    diagnostic_only = True
                constraint = evaluate_constraints(full, active)
                metrics = row.minibatch_metrics
                vote_delta = (
                    full.team_outcome.vote_correct_count
                    - active.team_outcome.vote_correct_count
                )
                oracle_delta = (
                    full.marginal.coverage_gain_count
                    - full.marginal.coverage_loss_count
                )
                diagnostic_rows.append({
                    "candidate_id": row.local_candidate.candidate_id,
                    "update_index": request.update_index,
                    "parent_team_hash": request.team_state_hash,
                    "target_member": assignment.target_member,
                    "primary_lane": assignment.primary_responsibility_lane,
                    "parent_responsiveness": dict(parent_response),
                    "local_parent_score": row.local_candidate.backend_metadata.get("local_parent_score"),
                    "local_candidate_score": row.local_candidate.local_score,
                    "local_full_validation_delta": row.local_candidate.backend_metadata.get("local_full_validation_delta"),
                    "local_acceptance_delta": row.local_candidate.backend_metadata.get("local_acceptance_delta"),
                    "local_newly_fixed": row.local_candidate.backend_metadata.get("local_newly_fixed"),
                    "local_newly_broken": row.local_candidate.backend_metadata.get("local_newly_broken"),
                    "local_preservation_loss": row.local_candidate.backend_metadata.get("local_preservation_loss"),
                    "team_minibatch": {
                        "invalid_delta": metrics.invalid_delta,
                        "vote_delta": metrics.vote_delta,
                        "target_delta": metrics.target_delta,
                        "coalition_delta": metrics.coalition_delta,
                        "responsibility_delta": metrics.responsibility_delta,
                        "broad_delta": metrics.broad_delta,
                        "oracle_delta": metrics.oracle_delta,
                        "passed": row.promoted,
                        "reason": (
                            "CATASTROPHIC" if is_catastrophic(metrics)
                            else "NO_POSITIVE_SIGNAL" if not has_promotion_signal(metrics)
                            else "PROMOTED" if row.promoted
                            else "RANK_BUDGET"
                        ),
                    },
                    "full": {
                        "diagnostic_only": diagnostic_only,
                        "target_delta": full.competence.correct_count - active.competence.correct_count,
                        "vote_delta": vote_delta,
                        "oracle_delta": oracle_delta,
                        "vote_gain_count": full.marginal.vote_gain_count,
                        "vote_loss_count": full.marginal.vote_loss_count,
                        "pivotal_gain_count": full.protection.pivotal_correct_gain_count,
                        "pivotal_loss_count": full.protection.pivotal_correct_loss_count,
                        "diagnostic_common_safe_passed": constraint.passed,
                    },
                    "labels": [
                        *(["LOCAL_POSITIVE_BUT_TEAM_NEUTRAL"] if vote_delta == 0 else []),
                        *(["STRUCTURALLY_VOTE_CENSORED"] if vote_delta == 0 and not parent_response["single_member_vote_changeable_cases"] else []),
                        *(["RESPONSIVE_BUT_VOTE_NEUTRAL"] if vote_delta == 0 and parent_response["single_member_vote_changeable_cases"] else []),
                        *(["LOCAL_POSITIVE_COVERAGE_GAIN_ONLY"] if oracle_delta > 0 and vote_delta == 0 else []),
                        *(["LOCAL_POSITIVE_TEAM_NEGATIVE"] if vote_delta < 0 else []),
                        *(["LOCAL_POSITIVE_PASSES_MINIBATCH"] if row.promoted else []),
                        *(["LOCAL_POSITIVE_BREAKS_PEER_SUPPORT"] if full.protection.pivotal_correct_loss_count > 0 else []),
                    ],
                })
        annotated = self.selector.annotate(promoted, active=active)
        return _EvaluatedBranch(
            assignment=assignment,
            candidates=annotated,
            cost=TeamCostAccounting(
                local_optimizer_solver_calls=local.solver_calls,
                local_optimizer_meta_calls=local.optimizer_calls,
                local_optimizer_tokens=local.total_tokens,
                team_minibatch_solver_calls=minibatch_calls,
                team_full_solver_calls=full_calls,
                team_shadow_solver_calls=0,
                team_tokens=minibatch_tokens + full_tokens,
            ),
            audit_metadata={
                "local_optimizer_backend": local.backend_name,
                "local_optimizer_version": local.backend_version,
                "local_termination_reason": local.termination_reason,
                "team_minibatch_example_ids": [row.example_id for row in team_minibatch],
                "team_minibatch": minibatch_telemetry,
                "primary_responsibility_lane": assignment.primary_responsibility_lane,
                "local_optimizer_telemetry": (
                    dict(local.optimizer_state.payload.get("telemetry", {}))
                    if local.optimizer_state is not None
                    else {}
                ),
                "local_accepted_update_count": len(local.candidates),
                **({"transfer_diagnostic": diagnostic_rows} if self.diagnostic_full_for_local_accepts else {}),
            },
        )

    async def run_frozen_opportunity(
        self,
        request: TeamSearchRequest,
        assignments: Sequence[TeamSearchAssignment],
    ) -> TeamSearchOutcome:
        """Evaluate frozen target branches and write back at most one winner.

        All branches are fully evaluated before any shadow evaluation or state
        mutation.  Existing single-target callers retain the same behavior by
        entering through :meth:`run_opportunity`.
        """
        frozen = tuple(assignments)
        if not frozen or len(frozen) > 2:
            raise ValueError("a frozen opportunity requires one or two target branches")
        target_ids = tuple(row.target_member for row in frozen)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("frozen target branches must address distinct members")
        evaluated: list[_EvaluatedBranch] = []
        for assignment in frozen:
            evaluated.append(await self._evaluate_frozen_branch(request, assignment))
        branches = tuple(evaluated)
        branch_winners = tuple(
            winner
            for branch in branches
            if (winner := self.selector.select(branch.candidates)) is not None
        )
        winner = self.selector.select(branch_winners)
        committed_id: str | None = None
        committed_member_id: int | None = None
        shadow_calls = shadow_tokens = 0
        if winner is not None and winner.full_evaluation is not None:
            owner = next(
                branch for branch in branches if any(row is winner for row in branch.candidates)
            )
            shadow, shadow_cost = await asyncio.to_thread(
                self.evaluator.evaluate_shadow,
                owner.assignment,
                winner.local_candidate,
            )
            shadow_calls = shadow_cost.solver_calls
            shadow_tokens = shadow_cost.total_tokens
            if shadow.passed:
                self.committer.commit(
                    assignment=owner.assignment,
                    candidate=winner.local_candidate,
                    evaluation=winner.full_evaluation,
                )
                committed_id = winner.local_candidate.candidate_id
                committed_member_id = owner.assignment.target_member
        cost = TeamCostAccounting(
            local_optimizer_solver_calls=sum(
                branch.cost.local_optimizer_solver_calls for branch in branches
            ),
            local_optimizer_meta_calls=sum(
                branch.cost.local_optimizer_meta_calls for branch in branches
            ),
            local_optimizer_tokens=sum(branch.cost.local_optimizer_tokens for branch in branches),
            team_minibatch_solver_calls=sum(
                branch.cost.team_minibatch_solver_calls for branch in branches
            ),
            team_full_solver_calls=sum(branch.cost.team_full_solver_calls for branch in branches),
            team_shadow_solver_calls=shadow_calls,
            team_tokens=sum(branch.cost.team_tokens for branch in branches) + shadow_tokens,
        )
        annotated = tuple(row for branch in branches for row in branch.candidates)
        feasible = sum(row.constraint is not None and row.constraint.passed for row in annotated)
        local_metadata = tuple(branch.audit_metadata for branch in branches)
        if self.diagnostic_full_for_local_accepts:
            for branch in branches:
                for row in branch.audit_metadata["transfer_diagnostic"]:
                    candidate_id = row["candidate_id"]
                    ordinary = next(
                        item for item in branch.candidates
                        if item.local_candidate.candidate_id == candidate_id
                    )
                    row["ordinary_common_safe"] = (
                        "PASS" if ordinary.constraint.passed else "FAIL"
                    ) if ordinary.constraint is not None else "NOT_REACHED"
                    row["ordinary_shadow"] = (
                        "PASS" if candidate_id == committed_id else
                        "FAIL" if winner is not None
                        and winner.local_candidate.candidate_id == candidate_id else
                        "NOT_REACHED"
                    )
                    row["committed"] = candidate_id == committed_id
        primary_metadata = local_metadata[0]
        return TeamSearchOutcome(
            assignment=(
                next(
                    branch.assignment
                    for branch in branches
                    if branch.assignment.target_member == committed_member_id
                )
                if committed_member_id is not None
                else frozen[0]
            ),
            candidates=annotated,
            committed_candidate_id=committed_id,
            termination_reason="committed" if committed_id else "no_team_safe_winner",
            cost=cost,
            funnel={
                "target_branches": len(branches),
                "local_candidates": len(annotated),
                "team_minibatch_survivors": sum(row.promoted for row in annotated),
                "full_team_evaluated_candidates": sum(row.full_evaluation is not None for row in annotated),
                **({"diagnostic_full_evaluated_candidates": sum(
                    len(branch.audit_metadata["transfer_diagnostic"]) for branch in branches
                )} if self.diagnostic_full_for_local_accepts else {}),
                "feasible_candidates": feasible,
                "committed_candidates": int(committed_id is not None),
            },
            audit_metadata={
                **primary_metadata,
                "selected_target_ids": target_ids,
                "committed_member_id": committed_member_id,
                "branch_metadata": local_metadata,
            },
        )
