"""Explicit Qwen reflection/Pattern wire and measurement contract, opt-in only."""
from .. import versions
from ..current_contract import MATH_STRUCTURED_SYSTEM_BINDING_VERSION
from ..search.schemas import SearchContractError


def optimizer_generation_contract(identity=versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION):
    if identity!=versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION:
        raise SearchContractError('OPTIMIZER_GENERATION_POLICY_IDENTITY_UNKNOWN')
    return dict(identity=identity,model='qwen3.7-flash',roles=['reflection','pattern'],
        enable_thinking=False,temperature=0.7,max_completion_tokens=1800,max_tokens='NOT_SENT',
        output_token_scope='reasoning_plus_answer',measurement_tolerance_tokens=10,
        accounting_output_ceiling=1810,truncation='existing_operational_output_truncation',
        top_p=0.8,top_k=20,presence_penalty=1.5,frequency_penalty=0)

def frozen_optimizer_policy(contract):
    frozen_cluster_policy(contract)
    if contract.get('identity')!=MATH_STRUCTURED_SYSTEM_BINDING_VERSION:
        raise SearchContractError('OPTIMIZER_GENERATION_POLICY_REQUIRES_FRESH_BINDING')
    policy=contract.get('optimizer_generation_policy');expected=optimizer_generation_contract()
    if (policy!=expected or any(type(policy[k]) is not type(v) for k,v in expected.items())
            or any(contract['models'][k]!=expected['model'] for k in ('optimizer_reflection','pattern'))):
        raise SearchContractError('OPTIMIZER_GENERATION_POLICY_BINDING_MISMATCH')
    return dict(policy)


def pattern_cluster_generation_contract():
    """Fresh cluster-only ceiling; reflection and per-example policy stay V3."""
    policy = optimizer_generation_contract(versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION)
    policy.update(identity=versions.PATTERN_CLUSTER_GENERATION_POLICY_VERSION,
        roles=['pattern_cluster'], max_completion_tokens=8192, accounting_output_ceiling=8202)
    return policy


def frozen_cluster_policy(contract):
    policy = contract.get('pattern_cluster_generation_policy')
    if policy is None:
        return None
    expected = pattern_cluster_generation_contract()
    if (contract.get('identity')!=MATH_STRUCTURED_SYSTEM_BINDING_VERSION
            or contract.get('execution_phase') not in {'pilot','canary'}
            or policy != expected
            or any(type(policy.get(k)) is not type(v) for k, v in expected.items())):
        raise SearchContractError('PATTERN_CLUSTER_GENERATION_POLICY_BINDING_MISMATCH')
    return dict(policy)


def frozen_optimizer_role_policy(contract, role):
    optimizer = frozen_optimizer_policy(contract)
    cluster = frozen_cluster_policy(contract)
    return cluster if role == 'pattern_cluster' and cluster is not None else optimizer
