"""Current production composition for any frozen binary-plurality adapter."""
from .binary_responsibility import BinaryPluralityResponsibilityAnalyzer
from .binary_runtime import BinaryTeamStateStore, BinaryEvidenceSource, FixedPeerTeamEvaluationProvider, FixedPeerPromotion, FixedPeerCommonSafe
from .evaluation import CandidateEvaluationPipeline
from .gepa_v2 import GEPATeamExposureOptimizer, GEPATeamExposureConfig, GEPATeamCandidateExposureEngine, V2GEPABridge
from .history import HistoryState, NullMemoryProvider, NullPatternAnalyzer
from .memory import StructuredLongTermMemoryProviderV1
from .patterns import PatternDiagnosticV1
from .orchestrator import StateAnalyzer, UnifiedSearchOrchestrator
from .policies import TargetPolicyV1, GlobalStopPolicy
from .runtime_v2 import V2OpportunityBuilder
from .private_gate import private_binary_gate
from .transition import TeamStateCommitter
from .variable_evidence import PatternCapableVariableEvidencePolicyV1, VariableEvidenceFeasibilityV1
from .schemas import SearchContractError, GlobalStopConfig
from .semantic_contract import FocusedPatternDiagnosticV2, FocusedEvidencePolicyV2, InitialCompetenceTransitionV2
from .experience import StrategyExperienceMemoryV2
from .. import versions


class FirstParentEpochStop(GlobalStopPolicy):
    """Canary phase boundary; frozen scientific patience still equals two."""
    def observe_opportunity(self, **kwargs):
        reason = super().observe_opportunity(**kwargs)
        if reason is not None:
            return reason
        if kwargs["committed"] or self.no_commit_epochs == 1:
            return "CANARY_PARENT_EPOCH_COMPLETE"
        return None


def build_binary_orchestrator(*, benchmark, aggregation, examples, prompts, solver, optimizer,
        method, seed, shadow_loader, shadow_count, runtime_readiness, pattern_provider=None,
        first_parent_epoch=False, provider_call_reader=None):
    if not isinstance(optimizer, GEPATeamExposureOptimizer) or optimizer.config.identity() != GEPATeamExposureConfig().identity():
        raise SearchContractError("FROZEN_GEPA_EXPOSURE_REQUIRED")
    if method.diagnosis_policy != versions.BINARY_PLURALITY_RESPONSIBILITY_VERSION:
        raise SearchContractError("BINARY_RESPONSIBILITY_IDENTITY_MISMATCH")
    prediction_invalidity = bool(getattr(benchmark, "invalid_predictions_are_incorrect", False))
    current = method.method == versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION
    if method.method not in {versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_VERSION,
                             versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION}:
        raise SearchContractError("UNSUPPORTED_BINARY_METHOD")
    pattern_id = versions.UNIFIED_FOCUSED_PATTERN_VERSION if current else versions.UNIFIED_PATTERN_DIAGNOSTIC_VERSION
    memory_id = versions.UNIFIED_EXPERIENCE_MEMORY_VERSION if current else versions.UNIFIED_STRUCTURED_MEMORY_VERSION
    from .schemas import SearchMethodConfig
    expected = (SearchMethodConfig.v2_1 if current else SearchMethodConfig.v2)(
        diagnosis_policy=versions.BINARY_PLURALITY_RESPONSIBILITY_VERSION,
        aggregation_policy=aggregation.identity, pattern_policy=method.pattern_policy,
        memory_policy=method.memory_policy, mechanism_config=method.mechanism_config,
        global_stop=GlobalStopConfig(emergency_max_provider_calls=method.global_stop.emergency_max_provider_calls))
    if current and method.identity() != expected.identity():
        raise SearchContractError("BINARY_COMPONENT_IDENTITIES_NOT_BOUND")
    if method.pattern_policy not in {versions.UNIFIED_NULL_PATTERN_VERSION, pattern_id} or method.memory_policy not in {versions.UNIFIED_NULL_MEMORY_VERSION, memory_id}:
        raise SearchContractError("UNSUPPORTED_MECHANISM_POLICY")
    blockers = tuple(runtime_readiness())
    if blockers:
        raise SearchContractError("HOLD_PRE_PROVIDER: " + ",".join(blockers))
    history = HistoryState()
    store = BinaryTeamStateStore(benchmark=benchmark, examples=examples, prompts=prompts, solver=solver,
                                aggregation=aggregation, freeze_initial_competence=current)
    patterns = (NullPatternAnalyzer() if method.pattern_policy == versions.UNIFIED_NULL_PATTERN_VERSION
                else (FocusedPatternDiagnosticV2 if current else PatternDiagnosticV1)(pattern_provider))
    memory = (NullMemoryProvider() if method.memory_policy == versions.UNIFIED_NULL_MEMORY_VERSION
              else (StrategyExperienceMemoryV2 if current else StructuredLongTermMemoryProviderV1)(**method.mechanism_config["memory"]))
    bridge = V2GEPABridge(optimizer=optimizer, history=history, seed=seed,
        solver_contract_id=solver.solver_contract_id, output_contract_id=solver.output_contract_id)
    transition = InitialCompetenceTransitionV2(invalid_predictions_are_incorrect=prediction_invalidity) if current else FixedPeerCommonSafe()
    provider = FixedPeerTeamEvaluationProvider(store, transition if current else None)
    gate = private_binary_gate(benchmark=benchmark, load_examples=shadow_loader,
        expected_count=shadow_count, solver=solver.for_gate() if hasattr(solver, "for_gate") else solver, store=store)
    composed = UnifiedSearchOrchestrator(method=method, benchmark=benchmark, aggregation=aggregation, state=store,
        analyzer=StateAnalyzer(BinaryPluralityResponsibilityAnalyzer(benchmark.capabilities)),
        opportunities=V2OpportunityBuilder(source=BinaryEvidenceSource(store, history),
            feasibility=VariableEvidenceFeasibilityV1(), target=TargetPolicyV1(),
            evidence=(FocusedEvidencePolicyV2 if current else PatternCapableVariableEvidencePolicyV1)(), patterns=patterns),
        engine=GEPATeamCandidateExposureEngine(bridge), evaluation=CandidateEvaluationPipeline(provider, FixedPeerPromotion(invalid_predictions_are_incorrect=prediction_invalidity)),
        transition=transition, gate=gate, committer=TeamStateCommitter(store), history=history, memory=memory,
        stop=FirstParentEpochStop(2) if first_parent_epoch else GlobalStopPolicy(2),
        runtime_readiness=runtime_readiness, provider_call_reader=provider_call_reader)
    # Identity checks must finish before the caller initializes provider state.
    return composed
