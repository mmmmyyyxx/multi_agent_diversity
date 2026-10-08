"""Explicit Qwen reflection/Pattern wire and measurement contract, opt-in only."""
from .. import versions
from ..search.schemas import SearchContractError


def optimizer_generation_contract(identity=versions.MATH_OPTIMIZER_GENERATION_POLICY_VERSION):
    if identity not in {versions.MATH_OPTIMIZER_GENERATION_POLICY_V1_VERSION,
            versions.MATH_OPTIMIZER_GENERATION_POLICY_VERSION, versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION}:
        raise SearchContractError('OPTIMIZER_GENERATION_POLICY_IDENTITY_UNKNOWN')
    policy=dict(identity=identity,
        model='qwen3.7-flash', roles=['reflection', 'pattern'],
        enable_thinking=identity==versions.MATH_OPTIMIZER_GENERATION_POLICY_V1_VERSION,
        temperature=0.0, max_completion_tokens=1800, max_tokens='NOT_SENT',
        output_token_scope='reasoning_plus_answer', measurement_tolerance_tokens=10,
        accounting_output_ceiling=1810, truncation='existing_operational_output_truncation')
    if identity==versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION:
        policy.update(temperature=0.7,top_p=0.8,top_k=20,presence_penalty=1.5,frequency_penalty=0)
    return policy


def frozen_optimizer_policy(contract):
    frozen_cluster_policy(contract)
    policy = contract.get('optimizer_generation_policy')
    if contract.get('identity') in {versions.MATH_LOW_COST_OPTIMIZER_EXECUTION_BINDING_VERSION, versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION, versions.MATH_V2_2_EXECUTION_BINDING_VERSION}:
        if not isinstance(policy,dict):
            raise SearchContractError('OPTIMIZER_GENERATION_POLICY_BINDING_MISMATCH')
        expected = optimizer_generation_contract(policy.get('identity'))
        if (contract['identity'] in {versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION, versions.MATH_V2_2_EXECUTION_BINDING_VERSION}) != (expected['identity']==versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION):
            raise SearchContractError('OPTIMIZER_GENERATION_POLICY_BINDING_MISMATCH')
        if (policy != expected or any(type(policy[k]) is not type(v) for k,v in expected.items())
                or any(contract['models'][k] != expected['model'] for k in ('optimizer_reflection','pattern'))):
            raise SearchContractError('OPTIMIZER_GENERATION_POLICY_BINDING_MISMATCH')
        return dict(policy)
    if policy is not None:
        raise SearchContractError('OPTIMIZER_GENERATION_POLICY_REQUIRES_FRESH_BINDING')
    return None


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
    if (contract.get('identity') != versions.MATH_V2_2_EXECUTION_BINDING_VERSION
            or contract.get('execution_phase') != 'pilot'
            or policy != expected
            or any(type(policy.get(k)) is not type(v) for k, v in expected.items())):
        raise SearchContractError('PATTERN_CLUSTER_GENERATION_POLICY_BINDING_MISMATCH')
    return dict(policy)


def frozen_optimizer_role_policy(contract, role):
    optimizer = frozen_optimizer_policy(contract)
    cluster = frozen_cluster_policy(contract)
    return cluster if role == 'pattern_cluster' and cluster is not None else optimizer
