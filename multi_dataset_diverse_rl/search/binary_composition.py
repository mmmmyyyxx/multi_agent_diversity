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
from dataclasses import replace
from .layer1_responsibility import ResponsibilityConditionedOptimizer, ResponsibilityConditionedEngine, Layer1Config
from .layer1_memory import MemoryConditionedOptimizer, MemoryConditionedEngine, MemoryLayer1Config, PreservationAnchorEvidenceV3
from .action_memory import StructuredActionMemoryV3, LIMITS as ACTION_MEMORY_LIMITS
from .rolling_risk_memory import StructuredRollingRiskMemoryV4, POLICY as SHARED_RISK_POLICY
from .pattern_responsibility import ResponsibilityPatternDiscoveryV3, PatternConditionedEvidenceV4, POLICY as PATTERN_POLICY
from .pattern_layer1 import PatternMemoryOptimizer, PatternMemoryEngine, PatternLayer1Config
from .pattern_layer1 import GradientPatternMemoryOptimizer, GradientPatternMemoryEngine, GradientPatternLayer1Config
from .textual_gradients import GradientPatternDiscovery, GradientExtractor, GradientPatternConditionedEvidence, POLICY as GRADIENT_PATTERN_POLICY


class FirstParentEpochStop(GlobalStopPolicy):
    """Canary phase boundary; frozen scientific patience still equals two."""
    def observe_opportunity(self, **kwargs):
        reason = super().observe_opportunity(**kwargs)
        if reason is not None:
            return reason
        if kwargs["committed"] or self.no_commit_epochs == 1:
            return "CANARY_PARENT_EPOCH_COMPLETE"
        return None


class OneProductionOpportunityStop(GlobalStopPolicy):
    """Versioned low-cost Canary boundary after one fully completed opportunity."""
    def observe_opportunity(self, **kwargs):
        super().observe_opportunity(**kwargs)
        return 'CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'


def build_binary_orchestrator(*, benchmark, aggregation, examples, prompts, solver, optimizer,
        method, seed, shadow_loader, shadow_count, runtime_readiness, pattern_provider=None,
        first_parent_epoch=False, provider_call_reader=None, one_production_opportunity=False):
    layer1=isinstance(optimizer,ResponsibilityConditionedOptimizer)
    action_memory=isinstance(optimizer,MemoryConditionedOptimizer)
    pattern_aware=isinstance(optimizer,PatternMemoryOptimizer)
    gradient_aware=isinstance(optimizer,GradientPatternMemoryOptimizer)
    if not layer1 and (not isinstance(optimizer, GEPATeamExposureOptimizer) or optimizer.config.identity() != GEPATeamExposureConfig().identity()):
        raise SearchContractError("FROZEN_GEPA_EXPOSURE_REQUIRED")
    if layer1 and optimizer.config.identity()!=(GradientPatternLayer1Config() if gradient_aware else PatternLayer1Config() if pattern_aware else MemoryLayer1Config() if action_memory else Layer1Config()).identity():
        raise SearchContractError('FROZEN_LAYER1_SEARCH_REQUIRED')
    if method.diagnosis_policy != versions.BINARY_PLURALITY_RESPONSIBILITY_VERSION:
        raise SearchContractError("BINARY_RESPONSIBILITY_IDENTITY_MISMATCH")
    prediction_invalidity = bool(getattr(benchmark, "invalid_predictions_are_incorrect", False))
    current = method.method == versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION
    if method.method not in {versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_VERSION,
                             versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION}:
        raise SearchContractError("UNSUPPORTED_BINARY_METHOD")
    pattern_id = versions.UNIFIED_FOCUSED_PATTERN_VERSION if current else versions.UNIFIED_PATTERN_DIAGNOSTIC_VERSION
    if pattern_aware:pattern_id=versions.PATTERN_AWARE_DISCOVERY_VERSION
    if gradient_aware:pattern_id=versions.GRADIENT_PATTERN_DISCOVERY_VERSION
    memory_id = versions.UNIFIED_EXPERIENCE_MEMORY_VERSION if current else versions.UNIFIED_STRUCTURED_MEMORY_VERSION
    if action_memory:
        memory_id=method.memory_policy
        if memory_id not in {versions.STRUCTURED_ACTION_MEMORY_VERSION,versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION}:
            raise SearchContractError('MEMORY_MAINLINE_MECHANISMS_MISMATCH')
    from .schemas import SearchMethodConfig
    expected = (SearchMethodConfig.v2_1 if current else SearchMethodConfig.v2)(
        diagnosis_policy=versions.BINARY_PLURALITY_RESPONSIBILITY_VERSION,
        aggregation_policy=aggregation.identity, pattern_policy=method.pattern_policy,
        memory_policy=method.memory_policy, mechanism_config=method.mechanism_config,
        global_stop=GlobalStopConfig(emergency_max_provider_calls=method.global_stop.emergency_max_provider_calls))
    if layer1:
        if not current:raise SearchContractError('LAYER1_REQUIRES_V2_1')
        expected=replace(expected,search_engine=versions.LAYER1_RESPONSIBILITY_SEARCH_VERSION,
            search_acceptance_policy='layer1_local_guidance_team_admission_v1')
        if action_memory:
            expected=replace(expected,search_engine=versions.LAYER1_FEEDBACK_SEARCH_VERSION,
                evidence_policy=versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION if gradient_aware else versions.PATTERN_CONDITIONED_EVIDENCE_VERSION if pattern_aware else versions.LAYER1_ANCHOR_EVIDENCE_VERSION)
            if method.memory_policy!=memory_id or method.pattern_policy!=(pattern_id if pattern_aware else versions.UNIFIED_NULL_PATTERN_VERSION):
                raise SearchContractError('MEMORY_MAINLINE_MECHANISMS_MISMATCH')
            mechanism_config={'memory':ACTION_MEMORY_LIMITS,
                    'optimizer_input_schema':versions.LAYER1_INPUT_SCHEMA_VERSION,
                    'panel_policy':versions.LAYER1_ANCHOR_EVIDENCE_VERSION}
            if memory_id==versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION:
                mechanism_config['shared_risk_policy']=SHARED_RISK_POLICY
            if pattern_aware:
                if memory_id!=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION:raise SearchContractError('PATTERN_REQUIRES_FROZEN_ROLLING_MEMORY')
                mechanism_config.update(optimizer_input_schema=versions.GRADIENT_OPTIMIZER_INPUT_VERSION if gradient_aware else versions.PATTERN_OPTIMIZER_INPUT_VERSION,
                    panel_policy=versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION if gradient_aware else versions.PATTERN_CONDITIONED_EVIDENCE_VERSION,
                    pattern_policy=GRADIENT_PATTERN_POLICY if gradient_aware else PATTERN_POLICY,
                    pattern_provider_binding=method.mechanism_config.get('pattern_provider_binding'))
                if not mechanism_config['pattern_provider_binding']:raise SearchContractError('PATTERN_PROVIDER_NOT_BOUND')
                transport=method.mechanism_config.get('pattern_support_id_transport')
                if transport is not None:
                    if transport!=versions.PATTERN_SUPPORT_ID_ALIAS_VERSION or getattr(pattern_provider,'support_id_transport',None)!=transport:
                        raise SearchContractError('PATTERN_ID_TRANSPORT_NOT_BOUND')
                    mechanism_config['pattern_support_id_transport']=transport
                guard=method.mechanism_config.get('pattern_abstraction_guard')
                if guard is not None:
                    if guard not in (versions.PATTERN_ABSTRACTION_GUARD_VERSION, versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION):
                        raise SearchContractError('PATTERN_ABSTRACTION_GUARD_NOT_BOUND')
                    mechanism_config['pattern_abstraction_guard']=guard
            if method.mechanism_config!=mechanism_config:
                raise SearchContractError('MEMORY_CONTEXT_POLICIES_NOT_BOUND')
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
                else (ResponsibilityPatternDiscoveryV3 if pattern_aware else FocusedPatternDiagnosticV2 if current else PatternDiagnosticV1)(pattern_provider))
    if pattern_aware:
        patterns=ResponsibilityPatternDiscoveryV3(pattern_provider,abstraction_guard_version=method.mechanism_config.get('pattern_abstraction_guard'))
    if gradient_aware:
        if (method.mechanism_config.get('pattern_abstraction_guard')!=versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
                or method.mechanism_config.get('pattern_support_id_transport')!=versions.PATTERN_SUPPORT_ID_ALIAS_VERSION):
            raise SearchContractError('PATTERN_GRADIENT_PROVIDER_POLICY_NOT_BOUND')
        patterns=GradientPatternDiscovery(GradientExtractor(pattern_provider.gradient_provider),pattern_provider)
    if action_memory and memory_id==versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION:
        memory=StructuredRollingRiskMemoryV4(risk_policy=method.mechanism_config['shared_risk_policy'],**method.mechanism_config['memory'])
    else:
        memory = (NullMemoryProvider() if method.memory_policy == versions.UNIFIED_NULL_MEMORY_VERSION
                  else (StructuredActionMemoryV3 if action_memory else StrategyExperienceMemoryV2 if current else StructuredLongTermMemoryProviderV1)(**method.mechanism_config["memory"]))
    if action_memory:optimizer.memory=memory
    bridge = None if layer1 else V2GEPABridge(optimizer=optimizer, history=history, seed=seed,
        solver_contract_id=solver.solver_contract_id, output_contract_id=solver.output_contract_id)
    transition = InitialCompetenceTransitionV2(invalid_predictions_are_incorrect=prediction_invalidity) if current else FixedPeerCommonSafe()
    provider = FixedPeerTeamEvaluationProvider(store, transition if current else None)
    gate = private_binary_gate(benchmark=benchmark, load_examples=shadow_loader,
        expected_count=shadow_count, solver=solver.for_gate() if hasattr(solver, "for_gate") else solver, store=store)
    composed = UnifiedSearchOrchestrator(method=method, benchmark=benchmark, aggregation=aggregation, state=store,
        analyzer=StateAnalyzer(BinaryPluralityResponsibilityAnalyzer(benchmark.capabilities)),
        opportunities=V2OpportunityBuilder(source=BinaryEvidenceSource(store, history),
            feasibility=VariableEvidenceFeasibilityV1(), target=TargetPolicyV1(),
            evidence=(GradientPatternConditionedEvidence if gradient_aware else PatternConditionedEvidenceV4 if pattern_aware else PreservationAnchorEvidenceV3 if action_memory else FocusedEvidencePolicyV2 if current else PatternCapableVariableEvidencePolicyV1)(), patterns=patterns),
        engine=(GradientPatternMemoryEngine if gradient_aware else PatternMemoryEngine if pattern_aware else MemoryConditionedEngine if action_memory else ResponsibilityConditionedEngine)(optimizer,seed) if layer1 else GEPATeamCandidateExposureEngine(bridge), evaluation=CandidateEvaluationPipeline(provider, FixedPeerPromotion(invalid_predictions_are_incorrect=prediction_invalidity)),
        transition=transition, gate=gate, committer=TeamStateCommitter(store), history=history, memory=memory,
        stop=OneProductionOpportunityStop(2) if one_production_opportunity else FirstParentEpochStop(2) if first_parent_epoch else GlobalStopPolicy(2),
        runtime_readiness=runtime_readiness, provider_call_reader=provider_call_reader)
    # Identity checks must finish before the caller initializes provider state.
    return composed
