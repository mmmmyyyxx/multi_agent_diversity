"""One complete scientific bundle; older schemas convey no execution eligibility."""
from dataclasses import asdict, dataclass
from copy import deepcopy
from .. import current_contract as identities
from .schemas import SearchContractError, SearchMethodConfig, GlobalStopConfig
from .textual_gradients import POLICY as PATTERN_POLICY
from .rolling_risk_memory import POLICY as MEMORY_POLICY
from .private_action_memory import LIMITS
from .current_layer1 import CurrentLayer1Config


@dataclass(frozen=True)
class CurrentPolicyBundle:
    responsibility: str = identities.BINARY_PLURALITY_RESPONSIBILITY_VERSION
    gradient: str = identities.PER_EXAMPLE_GRADIENT_VERSION
    abstraction_guard: str = identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
    clustering: str = identities.GRADIENT_PATTERN_DISCOVERY_VERSION
    partition_completion: str = identities.GRADIENT_PARTITION_COMPLETION_VERSION
    gradient_recovery: str = identities.GRADIENT_CONTRACT_RECOVERY_VERSION
    pattern_responsibility: str = identities.PATTERN_RESPONSIBILITY_VERSION
    evidence: str = identities.GRADIENT_CONDITIONED_EVIDENCE_VERSION
    memory: str = identities.STRUCTURED_ROLLING_RISK_MEMORY_VERSION
    layer1: str = identities.LAYER1_FEEDBACK_SEARCH_VERSION
    transition: str = identities.UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION

    def __post_init__(self):
        expected = dict(responsibility=identities.BINARY_PLURALITY_RESPONSIBILITY_VERSION,
            gradient=identities.PER_EXAMPLE_GRADIENT_VERSION,
            abstraction_guard=identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION,
            clustering=identities.GRADIENT_PATTERN_DISCOVERY_VERSION,
            partition_completion=identities.GRADIENT_PARTITION_COMPLETION_VERSION,
            gradient_recovery=identities.GRADIENT_CONTRACT_RECOVERY_VERSION,
            pattern_responsibility=identities.PATTERN_RESPONSIBILITY_VERSION,
            evidence=identities.GRADIENT_CONDITIONED_EVIDENCE_VERSION,
            memory=identities.STRUCTURED_ROLLING_RISK_MEMORY_VERSION,
            layer1=identities.LAYER1_FEEDBACK_SEARCH_VERSION,
            transition=identities.UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION)
        if asdict(self) != expected:
            raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')

    def method(self, *, aggregation, provider_binding, successful_provider_calls,
            partition_completion_policy=None,gradient_recovery_policy=None,pattern_cluster_generation_policy=None):
        from .gradient_recovery import validate_policy
        validate_policy(gradient_recovery_policy)
        if not isinstance(provider_binding, str) or len(provider_binding) != 64:
            raise SearchContractError('CURRENT_GRADIENT_POLICY_MISMATCH')
        # The compatibility dataclass preserves its exact frozen identity payload.
        if partition_completion_policy not in (None, self.partition_completion):
            raise SearchContractError('PATTERN_PARTITION_COMPLETION_POLICY_MISMATCH')
        method=SearchMethodConfig(method=identities.UNIFIED_TEAM_PROMPT_SEARCH_V2_2_VERSION,
            search_engine=self.layer1, diagnosis_policy=self.responsibility,
            aggregation_policy=aggregation, evidence_policy=self.evidence,
            feasibility_policy='variable_evidence_feasibility_v1',
            transition_policy=self.transition, pattern_policy=self.clustering,
            memory_policy=self.memory, search_acceptance_policy='layer1_local_guidance_team_admission_v1',
            global_stop=GlobalStopConfig(emergency_max_provider_calls=successful_provider_calls),
            mechanism_config=dict(memory=deepcopy(LIMITS),shared_risk_policy=deepcopy(MEMORY_POLICY),
                optimizer_input_schema=identities.GRADIENT_OPTIMIZER_INPUT_VERSION,
                panel_policy=self.evidence,pattern_policy=deepcopy(PATTERN_POLICY),
                pattern_provider_binding=provider_binding,
                pattern_support_id_transport=identities.PATTERN_SUPPORT_ID_ALIAS_VERSION,
                pattern_abstraction_guard=identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION))
        if partition_completion_policy is not None:
            method.mechanism_config['partition_completion_policy']=partition_completion_policy
        if gradient_recovery_policy is not None:
            method.mechanism_config['gradient_recovery_policy']=deepcopy(gradient_recovery_policy)
        if pattern_cluster_generation_policy is not None:
            from ..benchmarks.math_optimizer_generation import pattern_cluster_generation_contract
            if pattern_cluster_generation_policy != pattern_cluster_generation_contract():
                raise SearchContractError('PATTERN_CLUSTER_GENERATION_POLICY_BINDING_MISMATCH')
            method.mechanism_config['pattern_cluster_generation_policy']=deepcopy(pattern_cluster_generation_policy)
        return method

    def validate_method(self, method):
        expected=self.method(aggregation=method.aggregation_policy,
            provider_binding=method.mechanism_config.get('pattern_provider_binding'),
            successful_provider_calls=method.global_stop.emergency_max_provider_calls,
            partition_completion_policy=method.mechanism_config.get('partition_completion_policy'),
            gradient_recovery_policy=method.mechanism_config.get('gradient_recovery_policy'),
            pattern_cluster_generation_policy=method.mechanism_config.get('pattern_cluster_generation_policy'))
        if method.identity() != expected.identity():
            raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')

    def validate_contract(self, contract):
        from ..benchmarks.math_optimizer_generation import frozen_cluster_policy
        frozen_cluster_policy(contract)
        from .gradient_recovery import validate_policy
        validate_policy(contract.get('gradient_recovery_policy'))
        if contract.get('partition_completion_policy') not in (None, self.partition_completion):
            raise SearchContractError('PATTERN_PARTITION_COMPLETION_POLICY_MISMATCH')
        expected=dict(pattern_policy=PATTERN_POLICY,memory_policy_identity=self.memory,
            shared_risk_policy=MEMORY_POLICY,layer1_search_policy=asdict(CurrentLayer1Config()),
            optimizer_input_schema=identities.GRADIENT_OPTIMIZER_INPUT_VERSION,
            panel_evidence_policy=self.evidence,candidate_contract_identity=identities.SEMANTIC_MUTABLE_CONTRACT_VERSION,
            responsibility=self.responsibility,transition_policy=self.transition,
            pattern_support_id_transport=identities.PATTERN_SUPPORT_ID_ALIAS_VERSION,
            pattern_abstraction_guard=identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)
        if any(contract.get(key) != value for key,value in expected.items()):
            raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')


CURRENT_POLICY_BUNDLE=CurrentPolicyBundle()
