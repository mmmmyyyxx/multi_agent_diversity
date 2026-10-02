"""V2 composition and opportunity diagnostic stage on the single Unified graph."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .. import versions
from .aggregation import PluralityAggregation
from .current_bbh import CurrentBBHEvidenceSource, CurrentBBHStateSource, PluralityResponsibilityAnalyzer
from .current_bbh_runtime import (CurrentBBHAssignmentAdapter, CurrentBBHCommitter, CurrentBBHPromotion,
                                  CurrentBBHShadowGate, CurrentBBHTeamEvaluationProvider, CurrentBBHTransitionPolicy)
from .evaluation import CandidateEvaluationPipeline
from .gepa_v2 import GEPATeamCandidateExposureEngine, GEPATeamExposureOptimizer, GEPATeamExposureConfig, V2GEPABridge
from .history import HistoryState, NullMemoryProvider, NullPatternAnalyzer
from .memory import StructuredLongTermMemoryProviderV1
from .orchestrator import OpportunityBuilder, StateAnalyzer, UnifiedSearchOrchestrator
from .patterns import PatternDiagnosticV1
from .policies import TargetPolicyV1
from .schemas import OptimizationOpportunity, SearchContractError, SearchMethodConfig
from .variable_evidence import PatternCapableVariableEvidencePolicyV1, VariableEvidenceFeasibilityV1


class V2OpportunityBuilder(OpportunityBuilder):
    def __init__(self, *, patterns=None, **kwargs):
        super().__init__(**kwargs)
        self.patterns = patterns if patterns is not None else NullPatternAnalyzer()

    def build(self, *, state, diagnosis, history, update_index):
        # Freeze all feasibility and ranking before optional target diagnostic.
        rows_by_member = {m:tuple(self.source.for_member(state, diagnosis, m)) for m in sorted(diagnosis.responsibility)}
        feasible = tuple(m for m,rows in rows_by_member.items() if self.feasibility.feasible(state, diagnosis, m, rows))
        target = self.target.select(state, diagnosis, feasible, history)
        member = target.selected_member
        if member is None: return None
        if member >= len(state.member_prompts): raise SearchContractError("target outside team")
        rows = rows_by_member[member]
        pattern = ({} if self.patterns.identity == versions.UNIFIED_NULL_PATTERN_VERSION else
                   self.patterns.analyze(state, diagnosis, member, rows, history))
        view, audit = self.evidence.compose(state, diagnosis, member, rows, pattern)
        signal = diagnosis.responsibility[member]
        return OptimizationOpportunity(f"{state.team_state_id}:{update_index}:{member}",
            state.team_state_id, member, state.member_prompts[member],
            {"raw_responsibility":signal.raw_value, "target_score":target.target_scores[member],
             "eligible_members":target.eligible_members,
             **({"target_scores":dict(target.target_scores),
                 "feasibility_by_member":{m:m in feasible for m in rows_by_member}}
                if self.evidence.identity == versions.UNIFIED_FOCUSED_EVIDENCE_VERSION else {})}, diagnosis, view, pattern_context=pattern,
            search_budget={"metric_calls":self.search_metric_budget},
            evaluation_plan={"max_promoted":2, "evidence_universe":rows,
                             "evidence_audit":audit, "v2_candidate_contract":True,
                             **({"allocation_failure_counts":{m:history.failure_counts.get(m, 0) for m in rows_by_member}}
                                if self.evidence.identity == versions.UNIFIED_FOCUSED_EVIDENCE_VERSION else {})})


def build_v2_bbh_orchestrator(*, system, benchmark, optimizer, evaluator, committer,
                              seed, solver_contract_id, output_contract_id,
                              method=None, history=None, provider_call_reader=None,
                              pattern_provider=None, memory=None):
    selected = method or SearchMethodConfig.v2()
    # Only the explicit dormant mechanisms may vary in this composition.
    expected = replace(SearchMethodConfig.v2(), pattern_policy=selected.pattern_policy,
                       memory_policy=selected.memory_policy, mechanism_config=selected.mechanism_config)
    if selected.identity() != expected.identity():
        raise SearchContractError("V2 composition requires versioned component identities")
    if not isinstance(optimizer, GEPATeamExposureOptimizer):
        raise SearchContractError("V2 requires evaluated-proposal GEPA export")
    if optimizer.config.identity() != GEPATeamExposureConfig().identity():
        raise SearchContractError("V2 GEPA config differs from frozen component identity")
    if not benchmark.capabilities.binary_plurality_responsibility:
        raise SearchContractError("V2 BBH requires binary plurality capabilities")
    structured = history if history is not None else HistoryState()
    state = CurrentBBHStateSource(system)
    patterns = (NullPatternAnalyzer() if selected.pattern_policy == versions.UNIFIED_NULL_PATTERN_VERSION
                else PatternDiagnosticV1(pattern_provider))
    if selected.pattern_policy not in {versions.UNIFIED_NULL_PATTERN_VERSION, versions.UNIFIED_PATTERN_DIAGNOSTIC_VERSION}:
        raise SearchContractError("unsupported pattern identity")
    if selected.memory_policy == versions.UNIFIED_NULL_MEMORY_VERSION:
        if memory is not None: raise SearchContractError("non-null memory with null method identity")
        memory = NullMemoryProvider()
    elif selected.memory_policy == versions.UNIFIED_STRUCTURED_MEMORY_VERSION:
        limits = selected.mechanism_config.get("memory")
        if not limits: raise SearchContractError("MEMORY_LIMITS_NOT_FROZEN")
        memory = memory if memory is not None else StructuredLongTermMemoryProviderV1(**limits)
    else: raise SearchContractError("unsupported memory identity")
    assignment = CurrentBBHAssignmentAdapter(structured)
    bridge = V2GEPABridge(optimizer=optimizer, history=structured, seed=seed,
                         solver_contract_id=solver_contract_id, output_contract_id=output_contract_id)
    return UnifiedSearchOrchestrator(method=selected, benchmark=benchmark,
        aggregation=PluralityAggregation(), state=state,
        analyzer=StateAnalyzer(PluralityResponsibilityAnalyzer(system)),
        opportunities=V2OpportunityBuilder(source=CurrentBBHEvidenceSource(system, structured),
            feasibility=VariableEvidenceFeasibilityV1(), target=TargetPolicyV1(),
            evidence=PatternCapableVariableEvidencePolicyV1(), patterns=patterns),
        engine=GEPATeamCandidateExposureEngine(bridge),
        evaluation=CandidateEvaluationPipeline(CurrentBBHTeamEvaluationProvider(assignment, bridge, evaluator), CurrentBBHPromotion()),
        transition=CurrentBBHTransitionPolicy(bridge), gate=CurrentBBHShadowGate(assignment, bridge, evaluator),
        committer=CurrentBBHCommitter(state, assignment, bridge, evaluator, committer),
        history=structured, memory=memory, provider_call_reader=provider_call_reader)
