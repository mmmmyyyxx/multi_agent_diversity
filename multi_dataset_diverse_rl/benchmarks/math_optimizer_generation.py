"""Explicit Qwen reflection/Pattern wire and measurement contract, opt-in only."""
from .. import versions
from ..search.schemas import SearchContractError


def optimizer_generation_contract():
    return dict(identity=versions.MATH_OPTIMIZER_GENERATION_POLICY_VERSION,
        model='qwen3.7-flash', roles=['reflection', 'pattern'], enable_thinking=True,
        temperature=0.0, max_completion_tokens=1800, max_tokens='NOT_SENT',
        output_token_scope='reasoning_plus_answer', measurement_tolerance_tokens=10,
        accounting_output_ceiling=1810, truncation='existing_operational_output_truncation')


def frozen_optimizer_policy(contract):
    policy = contract.get('optimizer_generation_policy')
    if contract.get('identity') == versions.MATH_LOW_COST_OPTIMIZER_EXECUTION_BINDING_VERSION:
        expected = optimizer_generation_contract()
        if (policy != expected or any(type(policy[k]) is not type(v) for k,v in expected.items())
                or any(contract['models'][k] != expected['model'] for k in ('optimizer_reflection','pattern'))):
            raise SearchContractError('OPTIMIZER_GENERATION_POLICY_BINDING_MISMATCH')
        return dict(policy)
    if policy is not None:
        raise SearchContractError('OPTIMIZER_GENERATION_POLICY_REQUIRES_FRESH_BINDING')
    return None
