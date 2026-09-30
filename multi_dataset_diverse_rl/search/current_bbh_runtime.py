"""Current BBH physics at the unified ports; no historical controller or selector.

The old system evaluator and atomic system committer remain exact replay
authorities. Their record shapes are translated only at this boundary.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..candidate_selection import common_monotone_safe_key, evaluate_constraints
from ..native_feed import CandidateTransitionAudit
from ..local_optimizers.gepa_native import GEPALayer2EvidenceOptimizer
from ..team_search.schemas import TeamEvidenceCase, TeamSearchAssignment
from .evaluation import PromotionPolicy
from .aggregation import PluralityAggregation
from .current_bbh import (
    CurrentBBHEvidenceSource, CurrentBBHStateSource,
    PluralityResponsibilityAnalyzer,
)
from .engines import GEPASearchEngine
from .evaluation import CandidateEvaluationPipeline
from .gepa import CurrentGEPABridge, CurrentGEPAPacketAdapter, GEPADerivedConfig
from .history import HistoryState, MemoryProvider, NullMemoryProvider
from .orchestrator import OpportunityBuilder, StateAnalyzer, UnifiedSearchOrchestrator
from .policies import (
    CurrentEvidenceFeasibility, CurrentRoleEvidencePolicy,
    GlobalStopPolicy, TargetPolicyV1,
)
from .schemas import (
    EvaluatedCandidate, EvidenceItem, OptimizationOpportunity, SearchCandidate,
    SearchContractError, SearchMethodConfig, TeamEvaluation, TransitionDecision,
    TransitionRecord,
)


def _case(row: EvidenceItem) -> TeamEvidenceCase:
    data = row.signals
    return TeamEvidenceCase(
        example_id=row.example_id,
        input_payload=str(data["input_payload"]),
        gold=str(data["gold"]),
        target_output=data.get("target_output"),
        feedback=data.get("feedback"),
        evidence_group=str(data["legacy_group"]),
        tags=tuple(data["legacy_tags"]),
        team_disagreement=int(data["team_disagreement"]),
        residual_frequency=int(data["residual_frequency"]),
        team_margin=int(data["team_margin"]),
        mutation_sensitive=bool(data["mutation_sensitive"]),
    )


class CurrentBBHAssignmentAdapter:
    """Translate a frozen opportunity to the low-level BBH measurement API."""

    def __init__(self, history: HistoryState) -> None:
        self.history = history
        self.packet_adapter = CurrentGEPAPacketAdapter()

    def build(self, opportunity: OptimizationOpportunity) -> TeamSearchAssignment:
        universe = tuple(opportunity.evaluation_plan["evidence_universe"])
        member = opportunity.target_member
        signal = opportunity.diagnosis.responsibility[member]
        latest = self.history.latest_member_transition(member)
        transition = None
        if latest is not None:
            if not latest.parent_correctness or not latest.child_correctness:
                raise SearchContractError("current BBH transition lacks correctness profiles")
            transition = CandidateTransitionAudit(
                latest.parent_prompt_hash, latest.child_prompt_hash,
                latest.parent_correctness, latest.child_correctness,
            )
        return TeamSearchAssignment(
            target_member=member,
            parent_prompt=opportunity.parent_prompt,
            evidence=tuple(_case(row) for row in universe),
            optimization_context=self.packet_adapter.responsibility_context,
            responsibility_identity="unified_raw_overlapping_responsibility_v1",
            local_validation_example_ids=tuple(
                row.example_id for row in opportunity.evidence.search_validation_evidence
            ),
            primary_responsibility_lane=signal.primary_lane,
            responsibility_value=float(opportunity.objective["raw_responsibility"]),
            latest_transition=transition,
        )


class CurrentBBHTeamEvaluationProvider:
    def __init__(self, assignment: CurrentBBHAssignmentAdapter,
                 bridge: CurrentGEPABridge, evaluator: Any) -> None:
        self.assignment = assignment
        self.bridge = bridge
        self.evaluator = evaluator
        self.costs: list[tuple[str, Any]] = []

    def _raw(self, candidate: SearchCandidate) -> Any:
        try:
            return self.bridge.raw_candidates[candidate.candidate_id]
        except KeyError as exc:
            raise SearchContractError("candidate absent from GEPA return") from exc

    async def active(self, opportunity: OptimizationOpportunity) -> TeamEvaluation:
        raw = self.evaluator.active_evaluation(self.assignment.build(opportunity))
        return TeamEvaluation(0.0, None, (0.0,) * 5,
                              aggregation_diagnostics={"bbh_full": raw})

    async def team_probe(self, opportunity: OptimizationOpportunity,
                         candidate: SearchCandidate) -> TeamEvaluation:
        assigned = self.assignment.build(opportunity)
        rows = tuple(_case(row) for row in opportunity.evidence.team_probe_evidence)
        metrics, cost = self.evaluator.evaluate_minibatch(
            assigned, self._raw(candidate), rows,
        )
        self.costs.append(("team_probe", cost))
        return TeamEvaluation(float(metrics.vote_delta), None,
                              (float(metrics.target_delta),) * 5,
                              aggregation_diagnostics={"bbh_probe": metrics})

    async def full(self, opportunity: OptimizationOpportunity,
                   candidate: SearchCandidate) -> TeamEvaluation:
        evaluation, cost = self.evaluator.evaluate_full(
            self.assignment.build(opportunity), self._raw(candidate),
        )
        self.costs.append(("full", cost))
        return TeamEvaluation(float(evaluation.team_outcome.vote_correct_count), None,
                              tuple(float(value) for value in
                                    evaluation.member_gain.candidate_correct_counts),
                              aggregation_diagnostics={"bbh_full": evaluation})


class CurrentBBHPromotion(PromotionPolicy):
    """Exact V4 catastrophe guard and promotion key over frozen TeamProbe rows."""

    def select(self, rows: Sequence[tuple[SearchCandidate, TeamEvaluation]]) -> tuple[str, ...]:
        eligible = []
        for candidate, evaluation in rows:
            metrics = evaluation.aggregation_diagnostics["bbh_probe"]
            if (metrics.invalid_delta > 0 or metrics.vote_delta <= -2 or
                    metrics.team_net_vote_delta <= -3):
                continue
            if not any(value > 0 for value in (
                metrics.responsibility_delta, metrics.target_delta,
                metrics.vote_delta, metrics.broad_delta,
                metrics.team_net_vote_delta,
            )):
                continue
            key = (metrics.vote_delta, metrics.team_net_vote_delta,
                   metrics.responsibility_delta, metrics.target_delta,
                   metrics.broad_delta, candidate.candidate_id)
            eligible.append((key, candidate.candidate_id))
        return tuple(row[1] for row in sorted(eligible, reverse=True)[:2])


class CurrentBBHTransitionPolicy:
    """Frozen Common-Safe constraints and key, after Full and before Shadow."""

    def __init__(self, bridge: CurrentGEPABridge) -> None:
        self.bridge = bridge

    def select(self, parent: TeamEvaluation,
               candidates: Sequence[EvaluatedCandidate]) -> TransitionDecision:
        active = parent.aggregation_diagnostics["bbh_full"]
        feasible = []
        for row in candidates:
            if not row.promoted or row.full is None:
                continue
            raw = row.full.aggregation_diagnostics["bbh_full"]
            if evaluate_constraints(raw, active).passed:
                generation = self.bridge.raw_candidates[row.candidate.candidate_id].generation
                feasible.append((common_monotone_safe_key(raw, generation), row))
        if not feasible:
            return TransitionDecision(None, "NO_COMMON_SAFE_WINNER")
        return TransitionDecision(max(feasible, key=lambda item: item[0])[1],
                                  "COMMON_SAFE_WINNER")


class CurrentBBHShadowGate:
    def __init__(self, assignment: CurrentBBHAssignmentAdapter,
                 bridge: CurrentGEPABridge, evaluator: Any) -> None:
        self.assignment = assignment
        self.bridge = bridge
        self.evaluator = evaluator
        self.costs: list[Any] = []

    async def check(self, opportunity: OptimizationOpportunity,
                    candidate: EvaluatedCandidate) -> bool:
        raw = self.bridge.raw_candidates[candidate.candidate.candidate_id]
        decision, cost = self.evaluator.evaluate_shadow(
            self.assignment.build(opportunity), raw,
        )
        self.costs.append(cost)
        return bool(decision.passed)


class CurrentBBHCommitter:
    def __init__(self, state: Any, assignment: CurrentBBHAssignmentAdapter,
                 bridge: CurrentGEPABridge, evaluator: Any, committer: Any) -> None:
        self.state = state
        self.assignment = assignment
        self.bridge = bridge
        self.evaluator = evaluator
        self.committer = committer

    def commit(self, opportunity: OptimizationOpportunity,
               decision: TransitionDecision, history: HistoryState,
               memory: MemoryProvider) -> TransitionRecord:
        selected = decision.candidate
        if selected is None or selected.full is None:
            raise SearchContractError("BBH commit requires selected Full candidate")
        before = self.state.snapshot()
        candidate = self.bridge.raw_candidates[selected.candidate.candidate_id]
        key = (opportunity.target_member, candidate.candidate_id, before.team_state_id)
        transition = self.evaluator.transition_audits.get(key)
        if transition is None:
            raise SearchContractError("BBH commit lacks candidate transition audit")
        record = TransitionRecord(
            opportunity.opportunity_id, before.team_state_id, "",
            opportunity.target_member, candidate.candidate_id,
            newly_fixed_ids=transition.newly_fixed_ids,
            newly_broken_ids=transition.newly_broken_ids,
            parent_correctness=transition.parent_correctness,
            child_correctness=transition.child_correctness,
            parent_prompt_hash=transition.parent_candidate_hash,
            child_prompt_hash=transition.child_candidate_hash,
        )
        # Current runtime installs NullMemoryProvider. A future stateful memory
        # provider needs an explicit transactional hook before enablement.
        if type(memory) is not NullMemoryProvider:
            raise SearchContractError("stateful memory commit requires transactional integration")
        self.committer.commit(
            assignment=self.assignment.build(opportunity), candidate=candidate,
            evaluation=selected.full.aggregation_diagnostics["bbh_full"],
        )
        after = self.state.snapshot()
        if after.team_state_id == before.team_state_id:
            raise SearchContractError("BBH commit did not change state")
        from dataclasses import replace
        record = replace(record, child_state_id=after.team_state_id)
        history.observe_transition(record)
        memory.observe_transition(record)
        return record


def build_current_bbh_orchestrator(
    *, system: Any, benchmark: Any, optimizer: Any, evaluator: Any,
    committer: Any, seed: int, solver_contract_id: str,
    output_contract_id: str, method: SearchMethodConfig | None = None,
    history: HistoryState | None = None,
    provider_call_reader: Callable[[], int] | None = None,
) -> UnifiedSearchOrchestrator:
    """Compose one current BBH search graph from explicit runtime services."""
    selected = method or SearchMethodConfig()
    expected = SearchMethodConfig()
    if selected.identity() != expected.identity():
        raise SearchContractError("current BBH composition requires frozen component identities")
    if not benchmark.capabilities.supports_current_responsibility:
        raise SearchContractError("BBH responsibility capability required")
    if not benchmark.capabilities.supports_plurality:
        raise SearchContractError("current BBH composition requires plurality")
    if isinstance(optimizer, GEPALayer2EvidenceOptimizer):
        if provider_call_reader is None:
            raise SearchContractError("real GEPA composition requires provider usage reader")
        if not isinstance(optimizer.engine.config, GEPADerivedConfig):
            raise SearchContractError("unified GEPA must use versioned derived config")
        if optimizer.engine.config.identity() != GEPADerivedConfig().identity():
            raise SearchContractError("current method GEPA config differs from frozen v1 identity")
        if optimizer.saturation_config is None or not optimizer.saturation_config.enabled:
            raise SearchContractError("current BBH GEPA requires saturation stopping")
        if optimizer.saturation_config.local_no_update_patience != 3:
            raise SearchContractError("current BBH GEPA requires local patience 3")
    state = CurrentBBHStateSource(system)
    structured = history if history is not None else HistoryState()
    assignment = CurrentBBHAssignmentAdapter(structured)
    bridge = CurrentGEPABridge(
        optimizer=optimizer, history=structured, seed=seed,
        solver_contract_id=solver_contract_id,
        output_contract_id=output_contract_id,
    )
    return UnifiedSearchOrchestrator(
        method=selected, benchmark=benchmark,
        aggregation=PluralityAggregation(), state=state,
        analyzer=StateAnalyzer(PluralityResponsibilityAnalyzer(system)),
        opportunities=OpportunityBuilder(
            source=CurrentBBHEvidenceSource(system, structured),
            feasibility=CurrentEvidenceFeasibility(), target=TargetPolicyV1(),
            evidence=CurrentRoleEvidencePolicy(),
        ),
        engine=GEPASearchEngine(bridge),
        evaluation=CandidateEvaluationPipeline(
            CurrentBBHTeamEvaluationProvider(assignment, bridge, evaluator),
            CurrentBBHPromotion(),
        ),
        transition=CurrentBBHTransitionPolicy(bridge),
        gate=CurrentBBHShadowGate(assignment, bridge, evaluator),
        committer=CurrentBBHCommitter(state, assignment, bridge, evaluator, committer),
        history=structured, memory=NullMemoryProvider(),
        stop=GlobalStopPolicy(selected.global_stop.no_commit_patience),
        provider_call_reader=provider_call_reader,
    )
