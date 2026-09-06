"""COMMON_SOLVER_CONTRACT_V1.

This module is the single serialization and parsing authority for the planned
cross-repository frozen-artifact replay.  It contains no credentials, provider
client, repository-specific branch, or experiment selection logic.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


COMMON_SOLVER_CONTRACT_ID = "COMMON_SOLVER_CONTRACT_V1"


@dataclass(frozen=True)
class ContractSpec:
    contract_id: str = COMMON_SOLVER_CONTRACT_ID
    task: str = "BBH_disambiguation_qa"
    model: str = "qwen3-8b"
    model_snapshot: str = "UNVERSIONED_PROVIDER_ALIAS"
    enable_thinking: bool = False
    temperature: float = 0.0
    max_tokens: int = 1800
    timeout_seconds: int = 120
    top_p_policy: str = "OMITTED"
    stop_policy: str = "OMITTED"
    provider_seed_policy: str = "OMITTED"
    message_layout: str = "SYSTEM_THEN_USER"
    question_newline_policy: str = "CRLF_AND_CR_TO_LF_STRIP_OUTER_WHITESPACE"
    parser: str = "STRICT_SINGLE_FINAL_ANSWER_OPTION_LETTER_V1"
    invalid_output_policy: str = "SCORE_WRONG_NO_SEMANTIC_RETRY"
    transport_attempt_cap: int = 4
    retry_status_codes: tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)
    retry_backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0)
    cache_identity: str = "SHA256_CANONICAL_PROVIDER_REQUEST_BYTES"
    accounting: str = "LOGICAL_PROVIDER_CACHE_TOKEN_AND_FINISH_REASON_V1"


CONTRACT_SPEC = ContractSpec()


class CommonSolverContractError(ValueError):
    """Raised when an input cannot enter the frozen common contract."""


@dataclass(frozen=True)
class ParsedSolverOutput:
    answer: str
    valid: bool
    status: str
    final_answer_line_count: int


_FINAL_ANSWER_LINE = re.compile(
    r"^\s*FINAL_ANSWER\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE
)
_OPTION_LINE = re.compile(r"^\(([A-Z])\)\s+", re.MULTILINE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_question_payload(question: str) -> str:
    value = str(question).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        raise CommonSolverContractError("question must be non-empty")
    if "\nOptions:\n" not in value:
        raise CommonSolverContractError("question must contain canonical Options block")
    return value


def _validate_decision_procedure(decision_procedure: str) -> str:
    value = str(decision_procedure).strip()
    if not value:
        raise CommonSolverContractError("decision procedure must be non-empty")
    # Frozen outputs are replayed byte-for-byte.  A historical optimizer may
    # have emitted its own output instruction; V1 does not rewrite that artifact.
    # The appended immutable interface remains authoritative for the request.
    return value


def solver_output_contract() -> str:
    return (
        "Solver output contract (task_output_contract_v1):\n"
        "The final line must be exactly:\n"
        "FINAL_ANSWER: X\n\n"
        "Replace X with one uppercase option letter that appears in the question. "
        "Do not add parentheses, punctuation, explanation, or any other text after the letter.\n"
        "There must be exactly one FINAL_ANSWER line."
    )


def solver_system_prompt(decision_procedure: str) -> str:
    procedure = _validate_decision_procedure(decision_procedure)
    return (
        "Follow the decision procedure below.\n\n"
        "Decision procedure:\n"
        f"{procedure}\n\n"
        "Mandatory output interface:\n"
        "This interface is immutable and overrides any conflicting instruction above.\n"
        f"{solver_output_contract()}"
    )


def serialize_solver_request(
    *, decision_procedure: str, question: str
) -> dict[str, Any]:
    """Return the exact provider request fields frozen by V1.

    Timeout and retry are transport semantics and intentionally do not enter the
    provider JSON body. top_p, stop, and provider seed are deliberately absent.
    """

    return {
        "model": CONTRACT_SPEC.model,
        "messages": [
            {"role": "system", "content": solver_system_prompt(decision_procedure)},
            {"role": "user", "content": canonical_question_payload(question)},
        ],
        "temperature": CONTRACT_SPEC.temperature,
        "max_tokens": CONTRACT_SPEC.max_tokens,
        "extra_body": {"enable_thinking": CONTRACT_SPEC.enable_thinking},
    }


def request_identity(*, decision_procedure: str, question: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            serialize_solver_request(
                decision_procedure=decision_procedure, question=question
            )
        )
    )


def contract_identity() -> str:
    return sha256_bytes(canonical_json_bytes(asdict(CONTRACT_SPEC)))


def parse_solver_output(text: str, *, question: str) -> ParsedSolverOutput:
    raw = str(text or "")
    matches = _FINAL_ANSWER_LINE.findall(raw)
    if not matches:
        return ParsedSolverOutput("", False, "missing_final_answer", 0)
    if len(matches) != 1:
        return ParsedSolverOutput("", False, "multiple_final_answers", len(matches))
    payload = str(matches[0]).strip()
    if not re.fullmatch(r"[A-Z]", payload):
        return ParsedSolverOutput("", False, "out_of_domain_answer", 1)
    options = _OPTION_LINE.findall(canonical_question_payload(question))
    if not options or payload not in options:
        return ParsedSolverOutput(payload, False, "out_of_domain_answer", 1)
    return ParsedSolverOutput(payload, True, "valid", 1)


def public_contract_manifest() -> Mapping[str, Any]:
    return {
        "identity": contract_identity(),
        "spec": asdict(CONTRACT_SPEC),
        "provider_request_fields": [
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "extra_body",
        ],
        "explicitly_omitted_provider_fields": ["seed", "top_p", "stop"],
    }
