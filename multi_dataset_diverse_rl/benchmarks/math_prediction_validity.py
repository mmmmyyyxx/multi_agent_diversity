"""Frozen prediction invalidity, separate from reference and transport integrity."""
from dataclasses import dataclass
from functools import lru_cache
import json
import subprocess
import sys

from .. import versions
from ..current_contract import MATH_STRUCTURED_SYSTEM_BINDING_VERSION
from ..search.schemas import ParsedOutput, SearchContractError
from .math_domain_v2 import SETTINGS


def frozen_prediction_policy(contract):
    if contract.get('identity') == MATH_STRUCTURED_SYSTEM_BINDING_VERSION:
        from .math_flexible_answer import validity_contract
        expected = validity_contract()
        if contract.get('prediction_validity_policy') != expected:
            raise SearchContractError('MATH_STRUCTURED_VALIDITY_BINDING_MISMATCH')
        return expected
    raise SearchContractError('MATH_PREDICTION_VALIDITY_REQUIRES_FRESH_BINDING')


@dataclass(frozen=True)
class MATHPrediction:
    """Lossless private result. Metadata survives state/cache/JSON persistence."""
    text: str | None
    finish_reason: str | None
    answer: str
    prediction_valid: bool
    invalid_reason: str | None

    def parsed(self):
        return ParsedOutput(self.answer, self.prediction_valid)

    def identity_bytes(self):
        from dataclasses import asdict
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()


@lru_cache(maxsize=2048)
def classify_payload(expression):
    if len(expression) > SETTINGS["maximum_expression_characters"]:
        return "PAYLOAD_UNSUPPORTED"
    try:
        result = subprocess.run([sys.executable, "-m",
            "multi_dataset_diverse_rl.benchmarks.math_prediction_worker"],
            input=json.dumps(dict(expression=expression)), capture_output=True,
            text=True, timeout=SETTINGS["process_deadline"], check=False)
    except subprocess.TimeoutExpired:
        return "PAYLOAD_PARSE_FAILURE"
    except OSError as exc:
        raise SearchContractError("MATH_PREDICTION_WORKER_UNAVAILABLE") from exc
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        raise SearchContractError("MATH_PREDICTION_WORKER_RESPONSE_INVALID") from exc
    if result.returncode or set(data) != {"invalid_reason"} or data["invalid_reason"] not in (
            None, "PAYLOAD_PARSE_FAILURE", "PAYLOAD_UNSUPPORTED"):
        raise SearchContractError("MATH_PREDICTION_WORKER_FAILURE")
    return data["invalid_reason"]


def invalid_recovery_contract():
    return dict(identity=versions.MATH_INVALID_RECOVERY_VERSION, solver_invalid_max_retries=3,
        max_semantic_attempts=4, same_request_bytes=True, first_valid_stops=True,
        transport_failures_consume_semantic_attempt=False, terminal_requires_completed_invalid_attempts=4,
        terminal_invalid='incorrect_continue_no_vote', logical_evaluation_count=1,
        cache='resolved_valid_or_exhausted_terminal_only', invalidity_weight='none')


def frozen_recovery_policy(contract):
    if contract.get('identity')!=MATH_STRUCTURED_SYSTEM_BINDING_VERSION:
        raise SearchContractError('MATH_INVALID_RECOVERY_REQUIRES_FRESH_BINDING')
    expected=structured_recovery_contract()
    policy=contract.get('invalid_recovery_policy')
    if policy!=expected or any(type(policy[k]) is not type(v) for k,v in expected.items()):
        raise SearchContractError('MATH_INVALID_RECOVERY_BINDING_MISMATCH')
    return dict(policy)


def structured_recovery_contract():
    from .math_flexible_answer import IDENTITY
    return {**invalid_recovery_contract(), 'identity':'MATH_FLEXIBLE_ANSWER_INVALID_RECOVERY_V5',
        'same_request_bytes':False,'same_messages_and_sampling':True,
        'capacity_policy':'STRUCTURED_SOLVER_CAPACITY_EXECUTION_V2',
        'answer_extraction':IDENTITY,'retry_trigger':'extraction_or_frozen_invalidity_never_correctness'}


@dataclass(frozen=True)
class MATHResolvedPrediction(MATHPrediction):
    semantic_attempt_count: int
    raw_invalid_count: int
    recovered_invalid: bool
    terminal_invalid: bool
    transport_retry_attempts: int
    original_predictions: tuple[MATHPrediction, ...]


def resolve_predictions(attempts, transport_retry_attempts=0):
    attempts=tuple(attempts)
    if (not 1 <= len(attempts) <= 4 or any(p.prediction_valid for p in attempts[:-1])
            or not attempts[-1].prediction_valid and len(attempts) != 4
            or type(transport_retry_attempts) is not int or transport_retry_attempts < 0):
        raise SearchContractError('MATH_UNRESOLVED_PREDICTION_CANNOT_BE_CACHED')
    from dataclasses import asdict
    last=attempts[-1]
    return MATHResolvedPrediction(**asdict(last),semantic_attempt_count=len(attempts),
        raw_invalid_count=sum(not p.prediction_valid for p in attempts),
        recovered_invalid=last.prediction_valid and len(attempts)>1,
        terminal_invalid=not last.prediction_valid,transport_retry_attempts=transport_retry_attempts,
        original_predictions=attempts)
