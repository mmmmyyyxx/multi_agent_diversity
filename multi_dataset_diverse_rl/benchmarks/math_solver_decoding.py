"""Closed current sampling; optimizer generation has its own frozen policy."""
from .. import versions
from ..current_contract import MATH_STRUCTURED_SYSTEM_BINDING_VERSION
from ..search.schemas import SearchContractError

def solver_decoding_contract(identity=versions.MATH_SOLVER_DECODING_POLICY_V2_VERSION):
    if identity!=versions.MATH_SOLVER_DECODING_POLICY_V2_VERSION:
        raise SearchContractError('SOLVER_DECODING_POLICY_IDENTITY_UNKNOWN')
    return dict(identity=identity,enable_thinking=False,temperature=0.2,top_p=0.8,top_k=20,min_p=0,
        presence_penalty=0,frequency_penalty=0,max_output_tokens=3600)

def frozen_solver_policy(contract):
    if contract.get('identity')!=MATH_STRUCTURED_SYSTEM_BINDING_VERSION:
        raise SearchContractError('SOLVER_DECODING_POLICY_REQUIRES_FRESH_BINDING')
    policy=contract.get('solver_decoding_policy');expected=solver_decoding_contract()
    if policy!=expected or any(type(policy[k]) is not type(v) for k,v in expected.items()):
        raise SearchContractError('SOLVER_DECODING_POLICY_BINDING_MISMATCH')
    return dict(policy)

def generation_request_fields(contract,role):
    policy=frozen_solver_policy(contract)
    from .math_optimizer_generation import frozen_optimizer_role_policy
    if role in {'reflection','pattern_gradient','pattern_cluster'}:
        optimizer=frozen_optimizer_role_policy(contract,role)
        if not optimizer:raise SearchContractError('CURRENT_OPTIMIZER_GENERATION_REQUIRED')
        return dict(temperature=optimizer['temperature'],max_completion_tokens=optimizer['max_completion_tokens'],
            top_p=optimizer['top_p'],presence_penalty=optimizer['presence_penalty'],
            frequency_penalty=optimizer['frequency_penalty'],
            extra_body=dict(enable_thinking=optimizer['enable_thinking'],top_k=optimizer['top_k']))
    if role=='solver':
        return dict(temperature=policy['temperature'],top_p=policy['top_p'],
            presence_penalty=policy['presence_penalty'],frequency_penalty=policy['frequency_penalty'],
            max_tokens=policy['max_output_tokens'],
            extra_body={k:policy[k] for k in ('enable_thinking','top_k','min_p')})
    raise SearchContractError('CURRENT_PROVIDER_ROLE_FORBIDDEN')
