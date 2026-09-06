"""Frozen shared Solver evaluation contract for cross-repository replay."""

from .contract import (
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
    CommonSolverContractError,
    ParsedSolverOutput,
    canonical_json_bytes,
    canonical_question_payload,
    contract_identity,
    parse_solver_output,
    request_identity,
    serialize_solver_request,
    solver_system_prompt,
)
from .evaluator import CommonSolverEvaluator, EvaluationResult, TransportResponse

__all__ = [
    "COMMON_SOLVER_CONTRACT_ID",
    "CONTRACT_SPEC",
    "CommonSolverContractError",
    "CommonSolverEvaluator",
    "EvaluationResult",
    "ParsedSolverOutput",
    "canonical_json_bytes",
    "canonical_question_payload",
    "contract_identity",
    "parse_solver_output",
    "request_identity",
    "serialize_solver_request",
    "solver_system_prompt",
    "TransportResponse",
]
