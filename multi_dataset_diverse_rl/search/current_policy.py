"""Closed V2.4 scientific bundle; no historical execution defaults."""
from dataclasses import asdict, dataclass
from copy import deepcopy
from .. import current_contract as identities
from .schemas import SearchContractError, SearchMethodConfig, GlobalStopConfig
from .textual_gradients import pattern_policy_for_trajectory
from .rolling_risk_memory import POLICY as MEMORY_POLICY
from .private_action_memory import LIMITS
from .current_layer1 import EvidenceLayer1Config
from .optimization_evidence import frozen_policy, METHOD, LAYER1, EVIDENCE, MEMORY, INPUT, PROBE_POLICY


def require_current_contract(contract):
    """Check identity before data, provider, cache or parent construction."""
    from ..provider_routing import frozen_routing
    from ..benchmarks.math_solver_decoding import frozen_solver_policy
    frozen_routing(contract)
    frozen_solver_policy(contract)
    from ..benchmarks.math_response_evidence import trajectory_policy
    from .optimization_evidence import BINDING
    if (contract.get('method_identity') != METHOD or contract.get('identity') != BINDING
            or contract.get('paired_realization_policy') is not None
            or contract.get('gradient_recovery_policy') is not None
            or contract.get('repair_probe_policy') != PROBE_POLICY):
        raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')
    frozen_policy(contract.get('optimization_evidence_policy'))
    from .generation_failures import frozen_recovery
    frozen_recovery(contract.get('generated_output_recovery_policy'))
    from ..benchmarks.math_structured_interface import system_prompt_policy, system_interface_contract
    from ..benchmarks.math_flexible_answer import POLICY as extraction
    if (contract.get('system_prompt_policy') != system_prompt_policy(contract.get('initial_team_version'))
            or contract.get('answer_extraction_policy') != extraction
            or contract.get('solver_output_interface') != system_interface_contract(contract.get('initial_team_version'))
            or contract.get('cache_policy') != identities.CURRENT_CACHE_POLICY
            or any(contract.get(k) is not None for k in ('initial_evidence_reuse_policy',
                'initial_evidence_reuse_manifest_path','initial_evidence_reuse_manifest_sha256'))):
        raise SearchContractError('CURRENT_STRUCTURED_SYSTEM_POLICY_REQUIRED')
    if contract.get('solver_trajectory_policy') != trajectory_policy():
        raise SearchContractError('CURRENT_VISIBLE_TRAJECTORY_REQUIRED')


@dataclass(frozen=True)
class CurrentPolicyBundle:
    responsibility: str = identities.BINARY_PLURALITY_RESPONSIBILITY_VERSION
    gradient: str = identities.PER_EXAMPLE_GRADIENT_VERSION
    abstraction_guard: str = identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
    clustering: str = identities.GRADIENT_PATTERN_DISCOVERY_VERSION
    partition_completion: str = identities.GRADIENT_PARTITION_COMPLETION_VERSION
    pattern_responsibility: str = identities.PATTERN_RESPONSIBILITY_VERSION
    evidence: str = EVIDENCE
    memory: str = MEMORY
    layer1: str = LAYER1
    transition: str = identities.UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION

    def __post_init__(self):
        expected = {name: field.default for name, field in self.__dataclass_fields__.items()}
        if asdict(self) != expected:
            raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')

    def method(self, *, aggregation, provider_binding, successful_provider_calls,
            partition_completion_policy=None, gradient_recovery_policy=None,
            pattern_cluster_generation_policy=None, solver_trajectory_policy=None,
            optimization_evidence_policy=None):
        revised = frozen_policy(optimization_evidence_policy)
        if gradient_recovery_policy is not None:
            raise SearchContractError('REFERENCE_GRADIENT_RECOVERY_UNBOUND')
        pattern_policy = pattern_policy_for_trajectory(solver_trajectory_policy, revised)
        from .generation_failures import POLICY as recovery
        if not isinstance(provider_binding, str) or len(provider_binding) != 64:
            raise SearchContractError('CURRENT_GRADIENT_POLICY_MISMATCH')
        if partition_completion_policy not in (None, self.partition_completion):
            raise SearchContractError('PATTERN_PARTITION_COMPLETION_POLICY_MISMATCH')
        method = SearchMethodConfig(method=METHOD, search_engine=self.layer1,
            diagnosis_policy=self.responsibility, aggregation_policy=aggregation,
            evidence_policy=self.evidence, feasibility_policy=identities.UNIFIED_VARIABLE_FEASIBILITY_VERSION,
            transition_policy=self.transition, pattern_policy=self.clustering,
            memory_policy=self.memory, search_acceptance_policy='layer1_local_guidance_team_admission_v1',
            global_stop=GlobalStopConfig(emergency_max_provider_calls=successful_provider_calls),
            mechanism_config=dict(memory=deepcopy(LIMITS), shared_risk_policy=deepcopy(MEMORY_POLICY),
                optimizer_input_schema=INPUT, panel_policy=self.evidence, pattern_policy=pattern_policy,
                repair_probe_policy=deepcopy(PROBE_POLICY),
                pattern_provider_binding=provider_binding,generated_output_recovery_policy=deepcopy(recovery),
                pattern_support_id_transport=identities.PATTERN_SUPPORT_ID_ALIAS_VERSION,
                pattern_abstraction_guard=identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION))
        if partition_completion_policy is not None:
            method.mechanism_config['partition_completion_policy'] = partition_completion_policy
        if pattern_cluster_generation_policy is not None:
            from ..benchmarks.math_optimizer_generation import pattern_cluster_generation_contract
            if pattern_cluster_generation_policy != pattern_cluster_generation_contract():
                raise SearchContractError('PATTERN_CLUSTER_GENERATION_POLICY_BINDING_MISMATCH')
            method.mechanism_config['pattern_cluster_generation_policy'] = deepcopy(pattern_cluster_generation_policy)
        method.mechanism_config['solver_trajectory_policy'] = deepcopy(solver_trajectory_policy)
        method.mechanism_config['optimization_evidence_policy'] = revised
        return method

    def validate_method(self, method):
        expected = self.method(aggregation=method.aggregation_policy,
            provider_binding=method.mechanism_config.get('pattern_provider_binding'),
            successful_provider_calls=method.global_stop.emergency_max_provider_calls,
            partition_completion_policy=method.mechanism_config.get('partition_completion_policy'),
            gradient_recovery_policy=method.mechanism_config.get('gradient_recovery_policy'),
            pattern_cluster_generation_policy=method.mechanism_config.get('pattern_cluster_generation_policy'),
            solver_trajectory_policy=method.mechanism_config.get('solver_trajectory_policy'),
            optimization_evidence_policy=method.mechanism_config.get('optimization_evidence_policy'))
        if method.identity() != expected.identity():
            raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')

    def validate_contract(self, contract):
        require_current_contract(contract)
        from .solver_execution import frozen_execution_policy
        execution=frozen_execution_policy(contract)
        if execution and contract.get('accounting_scope_policy')!='FRESH_V25_REPAIR_SINGLE_ARM_2M_V2':
            raise SearchContractError('V24_ACCOUNTING_SCOPE_MISMATCH')
        from ..benchmarks.math_optimizer_generation import frozen_cluster_policy
        frozen_cluster_policy(contract)
        from ..benchmarks.math_response_evidence import frozen_trajectory_policy
        trajectory = frozen_trajectory_policy(contract)
        revised = frozen_policy(contract.get('optimization_evidence_policy'))
        if contract.get('partition_completion_policy') not in (None, self.partition_completion):
            raise SearchContractError('PATTERN_PARTITION_COMPLETION_POLICY_MISMATCH')
        expected = dict(pattern_policy=pattern_policy_for_trajectory(trajectory, revised),
            memory_policy_identity=self.memory, shared_risk_policy=MEMORY_POLICY,
            layer1_search_policy=asdict(EvidenceLayer1Config()), optimizer_input_schema=INPUT,
            panel_evidence_policy=self.evidence, candidate_contract_identity=identities.SEMANTIC_MUTABLE_CONTRACT_VERSION,
            responsibility=self.responsibility, transition_policy=self.transition,
            pattern_support_id_transport=identities.PATTERN_SUPPORT_ID_ALIAS_VERSION,
            pattern_abstraction_guard=identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)
        if any(contract.get(key) != value for key, value in expected.items()):
            raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')


CURRENT_POLICY_BUNDLE = CurrentPolicyBundle()
