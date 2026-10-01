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
from .schemas import SearchContractError
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
    blockers = tuple(runtime_readiness())
    if blockers:
        raise SearchContractError("HOLD_PRE_PROVIDER: " + ",".join(blockers))
    history = HistoryState()
    store = BinaryTeamStateStore(benchmark=benchmark, examples=examples, prompts=prompts, solver=solver, aggregation=aggregation)
    patterns = NullPatternAnalyzer() if method.pattern_policy == versions.UNIFIED_NULL_PATTERN_VERSION else PatternDiagnosticV1(pattern_provider)
    memory = NullMemoryProvider() if method.memory_policy == versions.UNIFIED_NULL_MEMORY_VERSION else StructuredLongTermMemoryProviderV1(**method.mechanism_config["memory"])
    if method.pattern_policy not in {versions.UNIFIED_NULL_PATTERN_VERSION, versions.UNIFIED_PATTERN_DIAGNOSTIC_VERSION} or method.memory_policy not in {versions.UNIFIED_NULL_MEMORY_VERSION, versions.UNIFIED_STRUCTURED_MEMORY_VERSION}:
        raise SearchContractError("UNSUPPORTED_MECHANISM_POLICY")
    bridge = V2GEPABridge(optimizer=optimizer, history=history, seed=seed,
        solver_contract_id=solver.solver_contract_id, output_contract_id=solver.output_contract_id)
    provider = FixedPeerTeamEvaluationProvider(store)
    gate = private_binary_gate(benchmark=benchmark, load_examples=shadow_loader,
        expected_count=shadow_count, solver=solver.for_gate() if hasattr(solver, "for_gate") else solver, store=store)
    composed = UnifiedSearchOrchestrator(method=method, benchmark=benchmark, aggregation=aggregation, state=store,
        analyzer=StateAnalyzer(BinaryPluralityResponsibilityAnalyzer(benchmark.capabilities)),
        opportunities=V2OpportunityBuilder(source=BinaryEvidenceSource(store, history),
            feasibility=VariableEvidenceFeasibilityV1(), target=TargetPolicyV1(),
            evidence=PatternCapableVariableEvidencePolicyV1(), patterns=patterns),
        engine=GEPATeamCandidateExposureEngine(bridge), evaluation=CandidateEvaluationPipeline(provider, FixedPeerPromotion()),
        transition=FixedPeerCommonSafe(), gate=gate, committer=TeamStateCommitter(store), history=history, memory=memory,
        stop=FirstParentEpochStop(2) if first_parent_epoch else GlobalStopPolicy(2),
        runtime_readiness=runtime_readiness, provider_call_reader=provider_call_reader)
    # Identity checks must finish before the caller initializes provider state.
    return composed
