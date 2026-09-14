"""Production adapters from the canonical five-member system into Layer 2.

The adapters keep the official local optimizer backend isolated from team
selection.  They intentionally contain no target scheduler: an experiment
runner supplies frozen assignments and may therefore compare schedulers while
holding every downstream operation fixed.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeVar

from ..candidate_selection import CandidateEvaluation
from ..evaluation.fixed_probe import FixedProbeEvaluator, evaluate_candidate_profile, subset_profiles
from ..evaluation.mutable_prompt_contract import validate_mutable_decision_procedure
from ..local_optimizers.schemas import LocalPromptCandidate
from ..peer_state import TeamVoteState
from ..responsibility import MemberAwareRepairOpportunity
from ..shadow_gate import ShadowGateDecision, ShadowGateMetrics, evaluate_shadow_gate
from ..system import PromptEnsembleOptimizationSystem
from ..vote_aligned_scheduler import classify_opportunity_lane
from .candidate_evaluator import EvaluationCost
from .schemas import TeamEvidenceCase, TeamMiniBatchMetrics, TeamSearchAssignment, TeamSearchRequest
from .task_builder import LocalTaskBuilder


T = TypeVar("T")


def _run_on_loop(loop: asyncio.AbstractEventLoop, coroutine: Awaitable[T]) -> T:
    """Run one system coroutine from a Layer-2 worker thread."""

    return asyncio.run_coroutine_threadsafe(coroutine, loop).result()


def _evaluation(
    system: PromptEnsembleOptimizationSystem,
    *,
    target: int,
    prompt: str,
    candidate_profile: Sequence[Any],
    assigned_hashes: set[str],
    indices: Sequence[int] | None = None,
) -> CandidateEvaluation:
    if system.fixed_probe is None:
        raise RuntimeError("fixed Optimize probe is not initialized")
    examples = system.fixed_probe.examples
    active = system.active_profiles
    initial = system.initial_profiles
    if indices is not None:
        examples, active = subset_profiles(examples, active, indices)
        _, initial = subset_profiles(system.fixed_probe.examples, initial, indices)
    return evaluate_candidate_profile(
        prompt=prompt,
        prompt_hash=system.prompt_hash(prompt),
        examples=examples,
        active_profiles=active,
        initial_profiles=initial,
        candidate_profile=candidate_profile,
        target_agent_id=target,
        assigned_question_hashes=assigned_hashes,
        normalize_answer=system.normalize_answer,
        match_answer=system.match_answer,
        tie_break=system.protocol.tie_policy,
        seed=system.cfg.training.seed,
        tau=system.cfg.peer_state.soft_vote_tau,
    )


@dataclass(frozen=True)
class FrozenResponsibilitySnapshot:
    assigned: Mapping[int, tuple[MemberAwareRepairOpportunity, ...]]
    state_by_question: Mapping[str, TeamVoteState]
    current_margin_by_question: Mapping[str, int]


def freeze_current_responsibility(
    system: PromptEnsembleOptimizationSystem,
    *,
    update_index: int,
) -> FrozenResponsibilitySnapshot:
    """Materialize the one parent-state responsibility snapshot used by both branches."""

    _, assigned = system.assign_responsibilities(update_index=update_index)
    states, _, _ = system.current_states_and_opportunities()
    state_by_question = {row.question_hash: row for row in states}
    return FrozenResponsibilitySnapshot(
        assigned={member: tuple(rows) for member, rows in assigned.items()},
        state_by_question=state_by_question,
        current_margin_by_question={
            row.question_hash: int(row.plurality_margin) for row in states
        },
    )


class SystemResponsibilityAssignmentFactory:
    """Build disjoint Optimize-only evidence from one frozen parent state."""

    def __init__(
        self,
        *,
        system: PromptEnsembleOptimizationSystem,
        snapshot_reader: Callable[[], FrozenResponsibilitySnapshot],
        task_builder: LocalTaskBuilder,
    ) -> None:
        self.system = system
        self.snapshot_reader = snapshot_reader
        self.task_builder = task_builder

    def build_from_member(
        self,
        *,
        request: TeamSearchRequest,
        member_id: int,
        primary_lane: str | None,
        responsibility_identity: str,
    ) -> TeamSearchAssignment:
        del request
        if self.system.fixed_probe is None:
            raise RuntimeError("fixed Optimize probe is not initialized")
        target = int(member_id)
        snapshot = self.snapshot_reader()
        assigned_rows = tuple(snapshot.assigned.get(target, ()))
        assigned_by_hash = {row.question_hash: row for row in assigned_rows}
        evidence: list[TeamEvidenceCase] = []
        for index, example in enumerate(self.system.fixed_probe.examples):
            state = snapshot.state_by_question[example.question_hash]
            opportunity = assigned_by_hash.get(example.question_hash)
            target_answer = self.system.active_profiles[target][index]
            if opportunity is not None:
                lane = classify_opportunity_lane(
                    opportunity, snapshot.current_margin_by_question
                ) or "coverage"
                group = "responsibility"
                tags = ("responsibility", lane)
                feedback = (
                    "Improve the general decision procedure for this assigned residual "
                    f"while preserving unrelated competence; lane={lane}."
                )
            elif not state.vote_correct and not state.team_correctness[target]:
                group = "coalition"
                tags = ("coalition", "vote_wrong")
                feedback = (
                    "This is an unassigned coalition diagnostic. Improve only through a "
                    "general rule; do not specialize to this item."
                )
            else:
                group = "preservation"
                tags = (
                    "preservation",
                    "target_correct" if state.team_correctness[target] else "broad_context",
                )
                feedback = "Preserve broad correct behavior and the immutable output contract."
            evidence.append(
                TeamEvidenceCase(
                    example_id=example.question_hash,
                    input_payload=example.question,
                    gold=example.gold_answer,
                    target_output=target_answer.answer if target_answer.valid else None,
                    feedback=feedback,
                    evidence_group=group,
                    tags=tags,
                )
            )
        provisional = TeamSearchAssignment(
            target_member=target,
            parent_prompt=self.system.agents[target].current_prompt,
            evidence=tuple(evidence),
            optimization_context=(
                "Use only general reasoning rules. Preserve broad competence, do not copy "
                "examples, and do not modify or discuss the output interface."
            ),
            responsibility_identity=responsibility_identity,
            primary_responsibility_lane=primary_lane,
        )
        # Local validation and TeamMiniBatch use the same primary-lane-aligned
        # responsibility quota while coalition and preservation remain global.
        minibatch = self.task_builder.select_team_minibatch(
            provisional.evidence,
            primary_responsibility_lane=primary_lane,
        )
        return TeamSearchAssignment(
            **{
                **provisional.__dict__,
                "local_validation_example_ids": tuple(row.example_id for row in minibatch),
            }
        )

    def build(self, *, request: TeamSearchRequest, summary: Any) -> TeamSearchAssignment:
        return self.build_from_member(
            request=request,
            member_id=int(summary.member_id),
            primary_lane=str(summary.primary_lane),
            responsibility_identity="primary_responsibility_persistent_realizability_v1",
        )


class SystemLocalSolverEvaluator:
    """Single-member evaluator used only inside official local GEPA."""

    def __init__(
        self,
        *,
        system: PromptEnsembleOptimizationSystem,
        loop: asyncio.AbstractEventLoop,
        stage: Callable[[Mapping[str, Any] | None], None],
        accounting: Callable[[], Mapping[str, int]],
        solver_contract_id: str,
        output_contract_id: str,
    ) -> None:
        self.system = system
        self.loop = loop
        self.stage = stage
        self.accounting = accounting
        self.solver_contract_id = solver_contract_id
        self.output_contract_id = output_contract_id

    def evaluate(self, decision_procedure: str, example: Any) -> Any:
        from ..local_optimizers.gepa_adapter import LocalSolverObservation

        if self.system.fixed_probe is None:
            raise RuntimeError("fixed Optimize probe is not initialized")
        context = getattr(self, "task_context", None)
        if not isinstance(context, Mapping):
            raise RuntimeError("local GEPA Solver call lacks task attribution")
        target = int(context["target_member"])
        index_by_id = {
            row.question_hash: index
            for index, row in enumerate(self.system.fixed_probe.examples)
        }
        index = index_by_id[example.example_id]
        before = dict(self.accounting())

        async def run() -> Any:
            self.stage({
                **dict(context),
                "candidate_id": "local_gepa",
                "evaluation_stage": "local_optimizer_solver_eval",
            })
            try:
                result = await self.system.fixed_probe.evaluate_prompt_indices(
                    target,
                    decision_procedure,
                    self.system.prompt_hash(decision_procedure),
                    (index,),
                    self.system.solve,
                )
                return result[index]
            finally:
                self.stage(None)

        answer = _run_on_loop(self.loop, run())
        after = dict(self.accounting())
        return LocalSolverObservation(
            parsed_answer=answer.answer if answer.valid else None,
            raw_output=answer.trace,
            correct=bool(answer.valid and self.system.match_answer(answer.answer, example.gold)),
            valid=bool(answer.valid),
            failure_reason=None if answer.valid else answer.validity_status,
            input_tokens=int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0)),
            output_tokens=int(after.get("completion_tokens", 0)) - int(before.get("completion_tokens", 0)),
            provider_called=(
                int(after.get("successful_provider_calls", 0))
                > int(before.get("successful_provider_calls", 0))
            ),
        )


class SystemTeamCandidateEvaluator:
    """Fixed-peer TeamMiniBatch12, full Optimize, and winner-only Shadow."""

    def __init__(
        self,
        *,
        system: PromptEnsembleOptimizationSystem,
        shadow_probe: FixedProbeEvaluator,
        loop: asyncio.AbstractEventLoop,
        stage: Callable[[Mapping[str, Any] | None], None],
        accounting: Callable[[], Mapping[str, int]],
        update_index_reader: Callable[[], int],
    ) -> None:
        self.system = system
        self.shadow_probe = shadow_probe
        self.loop = loop
        self.stage = stage
        self.accounting = accounting
        self.update_index_reader = update_index_reader
        self.full_profiles: dict[tuple[int, str, str], tuple[Any, ...]] = {}
        self.shadow_events: list[dict[str, Any]] = []

    @staticmethod
    def _assigned(assignment: TeamSearchAssignment) -> set[str]:
        return {
            row.example_id
            for row in assignment.evidence
            if row.evidence_group == "responsibility"
        }

    def active_evaluation(self, assignment: TeamSearchAssignment) -> CandidateEvaluation:
        target = assignment.target_member
        return _evaluation(
            self.system,
            target=target,
            prompt=self.system.agents[target].current_prompt,
            candidate_profile=self.system.active_profiles[target],
            assigned_hashes=self._assigned(assignment),
        )

    def _profile(
        self,
        assignment: TeamSearchAssignment,
        candidate: LocalPromptCandidate,
        *,
        indices: Sequence[int] | None,
        evaluation_stage: str,
    ) -> tuple[tuple[Any, ...], EvaluationCost]:
        if self.system.fixed_probe is None:
            raise RuntimeError("fixed Optimize probe is not initialized")
        target = assignment.target_member
        before = dict(self.accounting())

        async def run() -> tuple[Any, ...]:
            self.stage({
                "seed": self.system.cfg.training.seed,
                "parent_id": f"seed78_update{self.update_index_reader()}",
                "update_index": self.update_index_reader(),
                "target_member": target,
                "candidate_id": candidate.candidate_id,
                "proposal_engine": "official_gepa",
                "evaluation_stage": evaluation_stage,
            })
            try:
                if indices is None:
                    return await self.system.fixed_probe.evaluate_prompt(
                        target,
                        candidate.prompt,
                        self.system.prompt_hash(candidate.prompt),
                        self.system.solve,
                    )
                partial = await self.system.fixed_probe.evaluate_prompt_indices(
                    target,
                    candidate.prompt,
                    self.system.prompt_hash(candidate.prompt),
                    indices,
                    self.system.solve,
                )
                return tuple(partial[index] for index in indices)
            finally:
                self.stage(None)

        profile = _run_on_loop(self.loop, run())
        after = dict(self.accounting())
        return profile, EvaluationCost(
            solver_calls=(
                int(after.get("successful_provider_calls", 0))
                - int(before.get("successful_provider_calls", 0))
            ),
            input_tokens=int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0)),
            output_tokens=int(after.get("completion_tokens", 0)) - int(before.get("completion_tokens", 0)),
        )

    def evaluate_minibatch(
        self,
        assignment: TeamSearchAssignment,
        candidate: LocalPromptCandidate,
        minibatch: Sequence[TeamEvidenceCase],
    ) -> tuple[TeamMiniBatchMetrics, EvaluationCost]:
        if self.system.fixed_probe is None:
            raise RuntimeError("fixed Optimize probe is not initialized")
        by_id = {
            row.question_hash: index
            for index, row in enumerate(self.system.fixed_probe.examples)
        }
        indices = tuple(by_id[row.example_id] for row in minibatch)
        profile, cost = self._profile(
            assignment,
            candidate,
            indices=indices,
            evaluation_stage="team_minibatch_eval",
        )
        candidate_eval = _evaluation(
            self.system,
            target=assignment.target_member,
            prompt=candidate.prompt,
            candidate_profile=profile,
            assigned_hashes=self._assigned(assignment),
            indices=indices,
        )
        active_examples, active_profiles = subset_profiles(
            self.system.fixed_probe.examples, self.system.active_profiles, indices
        )
        _, initial_profiles = subset_profiles(
            self.system.fixed_probe.examples, self.system.initial_profiles, indices
        )
        active_eval = evaluate_candidate_profile(
            prompt=self.system.agents[assignment.target_member].current_prompt,
            prompt_hash=self.system.prompt_hash(
                self.system.agents[assignment.target_member].current_prompt
            ),
            examples=active_examples,
            active_profiles=active_profiles,
            initial_profiles=initial_profiles,
            candidate_profile=active_profiles[assignment.target_member],
            target_agent_id=assignment.target_member,
            assigned_question_hashes=self._assigned(assignment),
            normalize_answer=self.system.normalize_answer,
            match_answer=self.system.match_answer,
            tie_break=self.system.protocol.tie_policy,
            seed=self.system.cfg.training.seed,
            tau=self.system.cfg.peer_state.soft_vote_tau,
        )
        return TeamMiniBatchMetrics(
            invalid_delta=(
                candidate_eval.competence.invalid_count
                - active_eval.competence.invalid_count
            ),
            vote_delta=(
                candidate_eval.team_outcome.vote_correct_count
                - active_eval.team_outcome.vote_correct_count
            ),
            target_delta=(
                candidate_eval.competence.correct_count
                - active_eval.competence.correct_count
            ),
            coalition_delta=candidate_eval.marginal.net_vote_delta,
            responsibility_delta=candidate_eval.marginal.assigned_residual_repair_count,
            broad_delta=(
                candidate_eval.member_gain.total_gain_count
                - active_eval.member_gain.total_gain_count
            ),
        ), cost

    def evaluate_full(
        self,
        assignment: TeamSearchAssignment,
        candidate: LocalPromptCandidate,
    ) -> tuple[CandidateEvaluation, EvaluationCost]:
        profile, cost = self._profile(
            assignment,
            candidate,
            indices=None,
            evaluation_stage="team_full_eval",
        )
        evaluation = _evaluation(
            self.system,
            target=assignment.target_member,
            prompt=candidate.prompt,
            candidate_profile=profile,
            assigned_hashes=self._assigned(assignment),
        )
        key = (
            assignment.target_member,
            candidate.candidate_id,
            self.system.team_prompt_state_hash(),
        )
        self.full_profiles[key] = profile
        return evaluation, cost

    def evaluate_shadow(
        self,
        assignment: TeamSearchAssignment,
        candidate: LocalPromptCandidate,
    ) -> tuple[ShadowGateDecision, EvaluationCost]:
        target = assignment.target_member
        parent_hash = self.system.team_prompt_state_hash()
        before = dict(self.accounting())

        async def run() -> tuple[list[tuple[Any, ...]], tuple[Any, ...]]:
            self.stage({
                "seed": self.system.cfg.training.seed,
                "parent_id": f"seed78_update{self.update_index_reader()}",
                "update_index": self.update_index_reader(),
                "target_member": target,
                "candidate_id": candidate.candidate_id,
                "proposal_engine": "official_gepa",
                "evaluation_stage": "team_shadow_eval",
            })
            try:
                active = list(await asyncio.gather(*(
                    self.shadow_probe.evaluate_prompt(
                        member,
                        agent.current_prompt,
                        self.system.prompt_hash(agent.current_prompt),
                        self.system.solve,
                    )
                    for member, agent in enumerate(self.system.agents)
                )))
                proposed = await self.shadow_probe.evaluate_prompt(
                    target,
                    candidate.prompt,
                    self.system.prompt_hash(candidate.prompt),
                    self.system.solve,
                )
                return active, proposed
            finally:
                self.stage(None)

        active, proposed_target = _run_on_loop(self.loop, run())
        after = dict(self.accounting())
        incumbent = self.system._dataset_metrics_from_profiles(
            self.shadow_probe.examples, active
        )
        proposed = list(active)
        proposed[target] = proposed_target
        candidate_metrics = self.system._dataset_metrics_from_profiles(
            self.shadow_probe.examples, proposed
        )
        decision = evaluate_shadow_gate(
            ShadowGateMetrics(
                incumbent_vote_correct=incumbent.vote_correct_count,
                candidate_vote_correct=candidate_metrics.vote_correct_count,
                incumbent_target_correct=incumbent.per_agent_correct_counts[target],
                candidate_target_correct=candidate_metrics.per_agent_correct_counts[target],
                row_count=len(self.shadow_probe.examples),
            )
        )
        self.shadow_events.append({
            "update_index": self.update_index_reader(),
            "parent_team_hash": parent_hash,
            "candidate_id": candidate.candidate_id,
            "target_member": target,
            **decision.sanitized(),
        })
        return decision, EvaluationCost(
            solver_calls=(
                int(after.get("successful_provider_calls", 0))
                - int(before.get("successful_provider_calls", 0))
            ),
            input_tokens=int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0)),
            output_tokens=int(after.get("completion_tokens", 0)) - int(before.get("completion_tokens", 0)),
        )


class SystemTeamCommitter:
    """Apply exactly one already-evaluated candidate to the active team."""

    def __init__(
        self,
        *,
        system: PromptEnsembleOptimizationSystem,
        evaluator: SystemTeamCandidateEvaluator,
        update_index_reader: Callable[[], int],
    ) -> None:
        self.system = system
        self.evaluator = evaluator
        self.update_index_reader = update_index_reader

    def commit(
        self,
        *,
        assignment: TeamSearchAssignment,
        candidate: LocalPromptCandidate,
        evaluation: CandidateEvaluation,
    ) -> None:
        target = assignment.target_member
        parent_hash = self.system.team_prompt_state_hash()
        key = (target, candidate.candidate_id, parent_hash)
        if key not in self.evaluator.full_profiles:
            raise RuntimeError("committed candidate lacks a full Optimize profile")
        if evaluation.prompt_hash != self.system.prompt_hash(candidate.prompt):
            raise ValueError("commit evaluation/prompt identity mismatch")
        agent = self.system.agents[target]
        old_prompt = agent.current_prompt
        old_previous_prompt = agent.previous_active_prompt
        old_profile = self.system.active_profiles[target]
        old_stable = deepcopy(self.system.stable_correct_question_hashes_by_agent)
        old_accepted_state_count = self.system.accepted_state_count
        old_responsibility_state = deepcopy(self.system.responsibility_state)
        old_team_state_version = self.system.team_state_version
        old_responsibility_state_version = self.system.responsibility_state_version
        old_refresh_count = self.system.responsibility_refresh_count
        old_cached = (
            deepcopy(self.system.cached_responsibility_eligibility),
            deepcopy(self.system.cached_responsibility_assignments),
            deepcopy(self.system.cached_service_assignments),
            deepcopy(self.system.cached_repair_lane_by_question),
            deepcopy(self.system.cached_service_portfolios),
            deepcopy(self.system.cached_active_lane_by_agent),
            deepcopy(self.system.cached_active_responsibility_assignments),
            deepcopy(self.system.cached_member_opportunities),
        )
        old_lengths = (
            len(self.system.peer_state_history),
            len(self.system.responsibility_assignments),
            len(self.system.service_routing_audit),
            len(self.system.specialization_anchor_trajectory),
            len(self.system.responsibility_portfolio_trajectory),
            len(self.system.member_opportunities),
        )
        try:
            validate_mutable_decision_procedure(candidate.prompt)
            agent.previous_active_prompt = old_prompt
            agent.current_prompt = candidate.prompt
            self.system.active_profiles[target] = self.evaluator.full_profiles[key]
            self.system._record_accepted_state_stability()
            self.system.responsibility_state.accepted_updates_by_agent[target] = (
                self.system.responsibility_state.accepted_updates_by_agent.get(target, 0) + 1
            )
            self.system.refresh_responsibility_after_commit(
                update_index=self.update_index_reader()
            )
        except Exception:
            agent.current_prompt = old_prompt
            agent.previous_active_prompt = old_previous_prompt
            self.system.active_profiles[target] = old_profile
            self.system.stable_correct_question_hashes_by_agent = old_stable
            self.system.accepted_state_count = old_accepted_state_count
            self.system.responsibility_state = old_responsibility_state
            self.system.team_state_version = old_team_state_version
            self.system.responsibility_state_version = old_responsibility_state_version
            self.system.responsibility_refresh_count = old_refresh_count
            (
                self.system.cached_responsibility_eligibility,
                self.system.cached_responsibility_assignments,
                self.system.cached_service_assignments,
                self.system.cached_repair_lane_by_question,
                self.system.cached_service_portfolios,
                self.system.cached_active_lane_by_agent,
                self.system.cached_active_responsibility_assignments,
                self.system.cached_member_opportunities,
            ) = old_cached
            histories = (
                self.system.peer_state_history,
                self.system.responsibility_assignments,
                self.system.service_routing_audit,
                self.system.specialization_anchor_trajectory,
                self.system.responsibility_portfolio_trajectory,
                self.system.member_opportunities,
            )
            for history, length in zip(histories, old_lengths):
                del history[length:]
            raise
