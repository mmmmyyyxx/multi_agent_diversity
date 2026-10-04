"""Frozen Solver sampling, isolated from optimizer generation and old bindings."""
from .. import versions
from ..search.schemas import SearchContractError


def solver_decoding_contract(identity=versions.MATH_SOLVER_DECODING_POLICY_VERSION):
    if identity not in {versions.MATH_SOLVER_DECODING_POLICY_VERSION, versions.MATH_SOLVER_DECODING_POLICY_V2_VERSION}:
        raise SearchContractError('SOLVER_DECODING_POLICY_IDENTITY_UNKNOWN')
    return dict(identity=identity,
        enable_thinking=False, temperature=0.2 if identity==versions.MATH_SOLVER_DECODING_POLICY_V2_VERSION else 0.7, top_p=0.8, top_k=20, min_p=0,
        presence_penalty=0, frequency_penalty=0, max_output_tokens=3600)


def frozen_solver_policy(contract):
    policy = contract.get("solver_decoding_policy")
    if contract.get("identity") in {versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION, *versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS}:
        expected = solver_decoding_contract(versions.MATH_SOLVER_DECODING_POLICY_V2_VERSION if contract.get('identity') in {versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION} else versions.MATH_SOLVER_DECODING_POLICY_VERSION)
        if (policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items())):
            raise SearchContractError("SOLVER_DECODING_POLICY_BINDING_MISMATCH")
        return dict(policy)
    if policy is not None:
        raise SearchContractError("SOLVER_DECODING_POLICY_REQUIRES_FRESH_BINDING")
    return None


def generation_request_fields(contract, role):
    policy = frozen_solver_policy(contract)
    decoding = contract["decoding"]
    from .math_optimizer_generation import frozen_optimizer_policy
    optimizer = frozen_optimizer_policy(contract)
    if role in {'reflection','pattern'} and optimizer is not None:
        fields=dict(temperature=optimizer['temperature'],
            max_completion_tokens=optimizer['max_completion_tokens'],
            extra_body={'enable_thinking':optimizer['enable_thinking']})
        if optimizer['identity']==versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION:
            fields.update(top_p=optimizer['top_p'],presence_penalty=optimizer['presence_penalty'],
                frequency_penalty=optimizer['frequency_penalty'])
            fields['extra_body']['top_k']=optimizer['top_k']
        return fields
    if role == "solver" and policy is not None:
        # Chat Completions' max_tokens is the exact wire alias for the frozen
        # max_output_tokens cap. Provider extensions are flattened by transport.
        return dict(temperature=policy["temperature"], top_p=policy["top_p"],
            presence_penalty=policy["presence_penalty"], frequency_penalty=policy["frequency_penalty"],
            max_tokens=policy["max_output_tokens"],
            extra_body={k: policy[k] for k in ("enable_thinking", "top_k", "min_p")})
    return dict(temperature=decoding["temperature"],
        max_tokens=decoding.get("solver_max_output_tokens", decoding["max_output_tokens"])
            if role == "solver" else decoding["max_output_tokens"],
        extra_body={"enable_thinking": False} if role == "solver" else {})
