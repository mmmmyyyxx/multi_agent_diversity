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
    clustering: str = identities.GRADIENT_PATTERN_DISCOVERY_VERSION
    pattern_responsibility: str = identities.PATTERN_RESPONSIBILITY_VERSION
    evidence: str = identities.GRADIENT_CONDITIONED_EVIDENCE_VERSION
    memory: str = identities.STRUCTURED_ROLLING_RISK_MEMORY_VERSION
    layer1: str = identities.LAYER1_FEEDBACK_SEARCH_VERSION
    transition: str = identities.UNIFIED_COMPETENCE_TRANSITION_VERSION

    def __post_init__(self):
        expected = dict(responsibility=identities.BINARY_PLURALITY_RESPONSIBILITY_VERSION,
            gradient=identities.PER_EXAMPLE_GRADIENT_VERSION,
            clustering=identities.GRADIENT_PATTERN_DISCOVERY_VERSION,
            pattern_responsibility=identities.PATTERN_RESPONSIBILITY_VERSION,
            evidence=identities.GRADIENT_CONDITIONED_EVIDENCE_VERSION,
            memory=identities.STRUCTURED_ROLLING_RISK_MEMORY_VERSION,
            layer1=identities.LAYER1_FEEDBACK_SEARCH_VERSION,
            transition=identities.UNIFIED_COMPETENCE_TRANSITION_VERSION)
        if asdict(self) != expected:
            raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')

    def method(self, *, aggregation, provider_binding, successful_provider_calls):
        if not isinstance(provider_binding, str) or len(provider_binding) != 64:
            raise SearchContractError('CURRENT_GRADIENT_POLICY_MISMATCH')
        # The compatibility dataclass preserves its exact frozen identity payload.
        return SearchMethodConfig(method=identities.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION,
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

    def validate_method(self, method):
        expected=self.method(aggregation=method.aggregation_policy,
            provider_binding=method.mechanism_config.get('pattern_provider_binding'),
            successful_provider_calls=method.global_stop.emergency_max_provider_calls)
        if method.identity() != expected.identity():
            raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')

    def validate_contract(self, contract):
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
