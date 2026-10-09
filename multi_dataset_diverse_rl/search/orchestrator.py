"""The single control flow for a new team prompt search method."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from .. import versions
from .benchmark import BenchmarkAdapter
from .search_ports import SearchEngine
from .evaluation import AdaptiveValidationGate, CandidateEvaluationPipeline
from .history import HistoryState, MemoryProvider, NullMemoryProvider, NullPatternAnalyzer, PatternAnalyzer
from .policies import (
    EvidencePolicy, GlobalStopPolicy, OpportunityFeasibilityPolicy,
    ResponsibilityAnalyzer, TargetPolicy,
)
from .schemas import (
    Diagnosis, EvidenceItem, OptimizationOpportunity, SearchContractError,
    SearchMethodConfig, TeamStateSnapshot, TransitionRecord,
)
from .transition import TeamStateCommitter, TeamStateStore, TransitionPolicy


class EvidenceSource(Protocol):
    def for_member(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis, member_id: int,
    ) -> Sequence[EvidenceItem]: ...


@dataclass(frozen=True)
class UnifiedSearchContext:
    benchmark: BenchmarkAdapter
    aggregation: Any
    history: HistoryState
    pattern_view: Mapping[str, Any]
    memory_view: Mapping[str, Any]


@dataclass(frozen=True)
class OpportunityTrace:
    opportunity_id: str
    parent_state_id: str
    target_member: int
    candidate_ids: tuple[str, ...]
    promoted_ids: tuple[str, ...]
    selected_candidate_id: str | None
    committed_candidate_id: str | None
    child_state_id: str
    stop_reason: str | None
    proposal_exposed: int = 0
    local_search_survival_update: int = 0
    evidence_audit: Mapping[str, Any] = field(default_factory=dict)
    memory_audit: Mapping[str, Any] = field(default_factory=dict)
    allocation_audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedSearchResult:
    initial_state_id: str
    final_state_id: str
    stop_reason: str
    trace: tuple[OpportunityTrace, ...]
    transitions: tuple[TransitionRecord, ...]
    method_identity: str


class StateAnalyzer:
    def __init__(
        self, responsibility: ResponsibilityAnalyzer,
        patterns: PatternAnalyzer | None = None,
    ) -> None:
        self.responsibility = responsibility
        self.patterns = patterns or NullPatternAnalyzer()

    def analyze(self, state: TeamStateSnapshot, history: HistoryState) -> Diagnosis:
        diagnosed = self.responsibility.analyze(state, history)
        return replace(diagnosed, patterns=dict(self.patterns.analyze(state)))


class OpportunityBuilder:
    """Feasibility precedes target ranking; evidence roles remain distinct."""

    def __init__(
        self, *, source: EvidenceSource,
        feasibility: OpportunityFeasibilityPolicy,
        target: TargetPolicy,
        evidence: EvidencePolicy,
        search_metric_budget: int = 42,
    ) -> None:
        self.source = source
        self.feasibility = feasibility
        self.target = target
        self.evidence = evidence
        self.search_metric_budget = search_metric_budget

    def build(self,**kwargs):
        raise SearchContractError('CURRENT_OPPORTUNITY_BUILDER_REQUIRED')


class UnifiedSearchOrchestrator:
    """Snapshot → diagnose → opportunity → search → evaluation → commit → stop."""

    def __init__(
        self, *, method: SearchMethodConfig, benchmark: BenchmarkAdapter,
        aggregation: Any, state: TeamStateStore, analyzer: StateAnalyzer,
        opportunities: OpportunityBuilder, engine: SearchEngine,
        evaluation: CandidateEvaluationPipeline, transition: TransitionPolicy,
        gate: AdaptiveValidationGate, committer: TeamStateCommitter,
        history: HistoryState | None = None,
        memory: MemoryProvider | None = None,
        stop: GlobalStopPolicy | None = None,
        provider_call_reader: Callable[[], int] | None = None,
        runtime_readiness: Callable[[], Sequence[str]] | None = None,
        execution_observer: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        from .current_policy import CURRENT_POLICY_BUNDLE
        CURRENT_POLICY_BUNDLE.validate_method(method)
        if memory is None:
            raise SearchContractError('CURRENT_MEMORY_REQUIRED')
        self.method = method
        self.benchmark = benchmark
        self.aggregation = aggregation
        self.state = state
        self.analyzer = analyzer
        self.opportunities = opportunities
        self.engine = engine
        self.evaluation = evaluation
        self.transition = transition
        self.gate = gate
        self.committer = committer
        self.history = history if history is not None else HistoryState()
        self.memory = memory
        self.stop = stop if stop is not None else GlobalStopPolicy(
            method.global_stop.no_commit_patience,
        )
        self.provider_call_reader = provider_call_reader
        self.runtime_readiness = runtime_readiness
        self.execution_observer = execution_observer
        self.observation_observer = None

    def _emergency_provider_limit_reached(self) -> bool:
        if self.provider_call_reader is None:
            return False
        calls = int(self.provider_call_reader())
        if calls < 0:
            raise SearchContractError("negative provider usage")
        return calls >= self.method.global_stop.emergency_max_provider_calls

    async def run(self, *, max_opportunities: int) -> UnifiedSearchResult:
        bound=getattr(self,'operational_bound',None)
        if bound is not None and max_opportunities!=bound['max_opportunities']:
            raise SearchContractError('FROZEN_OPERATIONAL_HORIZON_MISMATCH')
        if max_opportunities <= 0:
            raise SearchContractError("positive operational opportunity ceiling required")
        benchmark_id = getattr(self.benchmark, "benchmark_id", None)
        if benchmark_id is not None:
            from ..benchmarks.registry import benchmark_spec
            blockers = (self.runtime_readiness() if self.runtime_readiness is not None
                        else benchmark_spec(benchmark_id).blockers())
            if blockers:
                raise SearchContractError("HOLD_PRE_PROVIDER: " + ",".join(blockers))
        if (self.method.diagnosis_policy == versions.UNIFIED_PLURALITY_RESPONSIBILITY_VERSION
                and self.method.aggregation_policy != versions.UNIFIED_PLURALITY_AGGREGATION_VERSION):
            raise SearchContractError("SCIENTIFIC_DECISION_REQUIRED: LLM aggregation responsibility")
        if not self.benchmark.capabilities.supports_current_responsibility:
            raise SearchContractError("SCIENTIFIC_DECISION_REQUIRED: aggregation-aware responsibility")
        if getattr(self.aggregation, "identity", None) != self.method.aggregation_policy:
            raise SearchContractError("aggregation implementation/method identity mismatch")
        expected_acceptance='layer1_local_guidance_team_admission_v1'
        if self.method.search_acceptance_policy != expected_acceptance:
            raise SearchContractError("CURRENT_TEAM_ADMISSION_POLICY_MISMATCH")
        if (self.opportunities.search_metric_budget != 42 or
                self.opportunities.evidence.metric_budget != self.opportunities.search_metric_budget or
                self.opportunities.evidence.minimum != 3):
            raise SearchContractError("CURRENT_LAYER1_BUDGET_MISMATCH")
        checks = ((self.engine, self.method.search_engine), (self.memory, self.method.memory_policy),
                  (self.opportunities.patterns, self.method.pattern_policy),
                  (self.opportunities.evidence, self.method.evidence_policy),
                  (self.opportunities.feasibility, self.method.feasibility_policy))
        if any(getattr(obj, "identity", None) != expected for obj, expected in checks):
            raise SearchContractError("CURRENT_COMPONENT_IDENTITY_MISMATCH")
        if getattr(self.transition, "identity", None) != self.method.transition_policy:
            raise SearchContractError("TRANSITION_POLICY_IDENTITY_MISMATCH")
        if self.method.memory_policy != versions.UNIFIED_NULL_MEMORY_VERSION and self.method.mechanism_config.get("memory") != self.memory.limits:
            raise SearchContractError("memory limits must enter explicit method identity")
        if self.method.pattern_policy != versions.UNIFIED_NULL_PATTERN_VERSION and not self.method.mechanism_config.get("pattern_provider_binding"):
            raise SearchContractError("PATTERN_PROVIDER_NOT_BOUND")
        if not self.memory.competence:
            self.memory.bootstrap(self.state)
        initial = self.state.snapshot().team_state_id
        scores = getattr(self.state, "initial_member_scores", None)
        identity = getattr(self.state, "initial_state_id", None)
        if scores is None or identity is None:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        self.transition.bind_initial(scores, identity)
        trace: list[OpportunityTrace] = []
        reason = "OPERATIONAL_OPPORTUNITY_CEILING"
        for index in range(max_opportunities):
            if self._emergency_provider_limit_reached():
                reason = "EMERGENCY_PROVIDER_CALL_CEILING"
                break
            parent = self.state.snapshot()
            diagnosis = self.analyzer.analyze(parent, self.history)
            opportunity = self.opportunities.build(
                state=parent, diagnosis=diagnosis,
                history=self.history, update_index=index,
            )
            if opportunity is None:
                reason = getattr(self.opportunities,'last_reason','NO_REPAIR_SIGNAL')
                break
            if self.observation_observer:
                self.observation_observer('MEMORY_BEFORE_OPPORTUNITY',dict(
                    opportunity_id=opportunity.opportunity_id,target_member=opportunity.target_member))
            opportunity = replace(
                opportunity,
                memory_view=dict(self.memory.read_for_opportunity(opportunity)),
            )
            if self.execution_observer:
                self.execution_observer("OPPORTUNITY", {"parent": parent, "opportunity": opportunity})
            context = UnifiedSearchContext(
                self.benchmark, self.aggregation, self.history,
                opportunity.pattern_context, opportunity.memory_view,
            )
            searched = await self.engine.search(opportunity, context)
            evaluated = await self.evaluation.evaluate(opportunity, searched)
            if self.execution_observer:
                self.execution_observer("EVALUATION", {"opportunity_id": opportunity.opportunity_id, "search": searched, "evaluated": evaluated})
            active = await self.evaluation.provider.active(opportunity)
            decision = self.transition.select(active, evaluated)
            selected = decision.candidate.candidate.candidate_id if decision.candidate else None
            committed: str | None = None
            gate_passed = (await self.gate.check(opportunity, decision.candidate)
                           if decision.candidate is not None else None)
            memory_delta = None
            from .memory_records import OpportunityOutcome
            progress = {}
            if (True
                    and decision.candidate is not None and gate_passed):
                from .target_or_team_transition import progress_path
                team_gain = decision.candidate.full.aggregate_score - active.aggregate_score
                target_gain = (decision.candidate.full.member_scores[opportunity.target_member]
                               - active.member_scores[opportunity.target_member])
                progress = dict(realized_team_gain=team_gain, realized_target_gain=target_gain,
                                progress_path=progress_path(team_gain, target_gain))
            memory_delta = self.memory.prepare_outcome(OpportunityOutcome(
                opportunity, tuple(evaluated), selected,
                bool(decision.candidate is not None and gate_passed), gate_passed, index,
                operational_failure=(bool(searched.search_state.get("operational_failure")) or
                                     any(r.diagnostics.get("operational_failure") for r in evaluated) or
                                     bool(decision.candidate is not None and
                                          getattr(self.gate, "operational_failure", False))), **progress))
            self.memory.validate_delta(memory_delta)
            if decision.candidate is not None and gate_passed:
                record = self.committer.commit(
                    opportunity, decision, self.history, NullMemoryProvider(),
                )
                committed = record.candidate_id
                self.history.observe_opportunity(opportunity.target_member, committed=True)
            else:
                self.history.observe_opportunity(opportunity.target_member, committed=False)
            child = self.state.snapshot()
            if (committed is None and child.team_state_id != parent.team_state_id
                    or committed is not None and child.team_state_id == parent.team_state_id):
                raise SearchContractError("state changed without matching atomic commit")
            self.memory.apply_outcome(memory_delta)
            if self.observation_observer:
                self.observation_observer('MEMORY_AFTER_OPPORTUNITY',dict(
                    opportunity_id=opportunity.opportunity_id,target_member=opportunity.target_member,
                    committed=committed,selected=selected,gate_passed=gate_passed))
            if self.execution_observer:
                self.execution_observer("TRANSITION", {"opportunity_id": opportunity.opportunity_id, "selected": selected,
                    "gate_passed": gate_passed, "committed": committed, "child": child})
            stopped = self.stop.observe_opportunity(
                parent_state_id=parent.team_state_id,
                eligible_members=tuple(opportunity.objective.get(
                    "eligible_members", (opportunity.target_member,),
                )),
                selected_member=opportunity.target_member,
                local_update=(searched.local_survival_update_count > 0),
                committed=committed is not None,
            )
            allocation = {}
            gain = (decision.candidate.full.aggregate_score - active.aggregate_score
                    if committed is not None and decision.candidate is not None else 0.0)
            # Observation only: this record has no scheduler read point.
            allocation = dict(raw_values={str(m): s.raw_value for m, s in diagnosis.responsibility.items()},
                DNC={str(m):dict(D=s.direct_count, N=s.near_margin_count, C=s.coverage_count)
                     for m, s in diagnosis.responsibility.items()},
                failure_counts=opportunity.evaluation_plan.get("allocation_failure_counts", {}),
                prior_opportunity_counts=opportunity.evaluation_plan.get("prior_opportunity_counts", {}),
                prior_exposure_counts=opportunity.evaluation_plan.get("prior_exposure_counts", {}),
                exposure_definition="one responsibility observation per member per opportunity",
                target_scores=opportunity.objective.get("target_scores", {}),
                eligible_members=opportunity.objective.get("eligible_members", ()),
                feasibility_by_member=opportunity.objective.get("feasibility_by_member", {}),
                target_count=self.history.target_counts.get(opportunity.target_member, 0),
                parent_team_score=active.aggregate_score,
                child_team_score=(decision.candidate.full.aggregate_score if committed is not None else active.aggregate_score),
                evaluation_support_identity=parent.diagnostics.get("evaluation_support_identity"),
                member_metric=parent.diagnostics.get("member_metric"),
                evaluator_identity=parent.diagnostics.get("evaluator_identity"),
                realized_team_gain=gain, committed=committed is not None,
                initial_member_scores=self.transition.initial_scores,
                incumbent_member_scores=parent.member_scores, child_member_scores=child.member_scores,
                inference_scope="DESCRIPTIVE_NOT_COUNTERFACTUAL_OR_COMPONENT_CAUSAL")
            from .target_or_team_transition import progress_path
            target_gain = (child.member_scores[opportunity.target_member]
                           - parent.member_scores[opportunity.target_member]) if committed else 0.0
            allocation.update(realized_target_gain=target_gain,
                realized_progress_path=progress_path(gain, target_gain))
            trace.append(OpportunityTrace(
                opportunity.opportunity_id, parent.team_state_id,
                opportunity.target_member,
                tuple(row.candidate_id for row in searched.candidates),
                tuple(row.candidate.candidate_id for row in evaluated if row.promoted),
                selected, committed, child.team_state_id, stopped,
                searched.team_candidate_count, searched.local_survival_update_count,
                opportunity.evaluation_plan.get("evidence_audit", {}),
                self.memory.audit() if hasattr(self.memory, "audit") else {},
                allocation,
            ))
            if stopped is not None:
                reason = stopped
                break
            if self._emergency_provider_limit_reached():
                reason = "EMERGENCY_PROVIDER_CALL_CEILING"
                break
        return UnifiedSearchResult(
            initial, self.state.snapshot().team_state_id, reason,
            tuple(trace), tuple(self.history.transitions), self.method.identity(),
        )
