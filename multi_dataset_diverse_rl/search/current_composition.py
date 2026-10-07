"""The current scientific graph: Responsibility → Gradient → Memory → Layer1 → team."""
from .binary_responsibility import BinaryPluralityResponsibilityAnalyzer
from .binary_runtime import BinaryTeamStateStore, BinaryEvidenceSource, FixedPeerTeamEvaluationProvider, FixedPeerPromotion
from .evaluation import CandidateEvaluationPipeline
from .history import HistoryState
from .orchestrator import StateAnalyzer, UnifiedSearchOrchestrator
from .policies import TargetPolicyV1, GlobalStopPolicy
from .private_gate import private_binary_gate
from .transition import TeamStateCommitter
from .variable_evidence import VariableEvidenceFeasibilityV1
from .target_or_team_transition import InitialCompetenceTargetOrTeamProgressV3
from .current_layer1 import CurrentEngine, CurrentLayer1Config
from .current_opportunity import CurrentOpportunityBuilder
from .current_policy import CURRENT_POLICY_BUNDLE
from .textual_gradients import GradientPatternDiscovery, GradientExtractor, GradientPatternConditionedEvidence
from .rolling_risk_memory import StructuredRollingRiskMemoryV4
from .schemas import SearchContractError


class OneProductionOpportunityStop(GlobalStopPolicy):
    """Frozen phase boundary; scientific no-commit patience remains two."""
    def observe_opportunity(self, **kwargs):
        super().observe_opportunity(**kwargs)
        return 'CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'


def build_current_team_prompt_search(*, benchmark, aggregation, examples, prompts, solver,
        optimizer, method, seed, shadow_loader, shadow_count, runtime_readiness,
        pattern_provider, provider_call_reader=None, policy_bundle=CURRENT_POLICY_BUNDLE,
        execution_phase='canary'):
    if execution_phase not in {'canary', 'pilot'}:
        raise SearchContractError('CURRENT_EXECUTION_PHASE_UNBOUND')
    if execution_phase == 'pilot':
        raise SearchContractError('TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN')
    if policy_bundle != CURRENT_POLICY_BUNDLE:
        raise SearchContractError('CURRENT_POLICY_MISMATCH')
    policy_bundle.validate_method(method)
    if getattr(pattern_provider,'partition_completion_policy',None) != method.mechanism_config.get('partition_completion_policy'):
        raise SearchContractError('PATTERN_PARTITION_COMPLETION_POLICY_MISMATCH')
    if getattr(getattr(pattern_provider,'gradient_provider',None),'recovery_policy',None) != method.mechanism_config.get('gradient_recovery_policy'):
        raise SearchContractError('GRADIENT_CONTRACT_RECOVERY_POLICY_MISMATCH')
    if optimizer.config != CurrentLayer1Config():
        raise SearchContractError('CURRENT_LAYER1_POLICY_MISMATCH')
    if (pattern_provider is None or getattr(pattern_provider,'gradient_provider',None) is None
            or getattr(pattern_provider,'support_id_transport',None) != method.mechanism_config['pattern_support_id_transport']):
        raise SearchContractError('HOLD_PRE_PROVIDER: CURRENT_GRADIENT_PROVIDER_NOT_BOUND')
    blockers=tuple(runtime_readiness())
    if blockers:
        raise SearchContractError('HOLD_PRE_PROVIDER: '+','.join(blockers))
    if aggregation.identity != method.aggregation_policy:
        raise SearchContractError('CURRENT_POLICY_MISMATCH')
    history=HistoryState()
    store=BinaryTeamStateStore(benchmark=benchmark,examples=examples,prompts=prompts,solver=solver,
        aggregation=aggregation,freeze_initial_competence=True)
    patterns=GradientPatternDiscovery(GradientExtractor(pattern_provider.gradient_provider),pattern_provider)
    memory=StructuredRollingRiskMemoryV4(risk_policy=method.mechanism_config['shared_risk_policy'],
        **method.mechanism_config['memory'])
    if memory is None or memory.identity != policy_bundle.memory:
        raise SearchContractError('HOLD_PRE_PROVIDER: CURRENT_MEMORY_PROVIDER_NOT_BOUND')
    optimizer.memory=memory
    invalidity=bool(getattr(benchmark,'invalid_predictions_are_incorrect',False))
    transition=InitialCompetenceTargetOrTeamProgressV3(invalid_predictions_are_incorrect=invalidity)
    provider=FixedPeerTeamEvaluationProvider(store,transition)
    gate=private_binary_gate(benchmark=benchmark,load_examples=shadow_loader,expected_count=shadow_count,
        solver=solver.for_gate() if hasattr(solver,'for_gate') else solver,store=store)
    return UnifiedSearchOrchestrator(method=method,benchmark=benchmark,aggregation=aggregation,state=store,
        analyzer=StateAnalyzer(BinaryPluralityResponsibilityAnalyzer(benchmark.capabilities)),
        opportunities=CurrentOpportunityBuilder(source=BinaryEvidenceSource(store,history),
            feasibility=VariableEvidenceFeasibilityV1(),target=TargetPolicyV1(),
            evidence=GradientPatternConditionedEvidence(),patterns=patterns),
        engine=CurrentEngine(optimizer,seed),evaluation=CandidateEvaluationPipeline(provider,
            FixedPeerPromotion(invalid_predictions_are_incorrect=invalidity)),
        transition=transition,gate=gate,committer=TeamStateCommitter(store),history=history,memory=memory,
        stop=(OneProductionOpportunityStop(2) if execution_phase == 'canary' else GlobalStopPolicy(2)),runtime_readiness=runtime_readiness,
        provider_call_reader=provider_call_reader)
