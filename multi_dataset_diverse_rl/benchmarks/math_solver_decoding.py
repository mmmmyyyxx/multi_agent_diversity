"""Frozen Solver sampling, isolated from optimizer generation and old bindings."""
from .. import versions
from ..search.schemas import SearchContractError


def solver_decoding_contract():
    return dict(identity=versions.MATH_SOLVER_DECODING_POLICY_VERSION,
        enable_thinking=False, temperature=0.7, top_p=0.8, top_k=20, min_p=0,
        presence_penalty=0, frequency_penalty=0, max_output_tokens=3600)


def frozen_solver_policy(contract):
    policy = contract.get("solver_decoding_policy")
    if contract.get("identity") == versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION:
        expected = solver_decoding_contract()
        if (policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items())):
            raise SearchContractError("SOLVER_DECODING_POLICY_BINDING_MISMATCH")
        return dict(policy)
    if policy is not None:
        raise SearchContractError("SOLVER_DECODING_POLICY_REQUIRES_FRESH_BINDING")
    return None


def generation_request_fields(contract, role):
    policy = frozen_solver_policy(contract)
    decoding = contract["decoding"]
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
