"""Frozen prediction invalidity, separate from reference and transport integrity."""
from dataclasses import dataclass
from functools import lru_cache
import json
import subprocess
import sys

from .. import versions
from ..search.schemas import ParsedOutput, SearchContractError
from .math_domain_v2 import SETTINGS, final_payload


def prediction_validity_contract():
    return dict(identity=versions.MATH_PREDICTION_VALIDITY_VERSION,
        reference_failure="hard_contract_failure", prediction_failure="invalid_incorrect_continue",
        truncated_even_with_marker="invalid_incorrect", invalid_response_retries=0,
        fallback_extraction=False, invalid_vote="excluded_from_equivalence_classes",
        denominator="all_frozen_examples", invalidity_telemetry="observation_only")


def frozen_prediction_policy(contract):
    policy = contract.get("prediction_validity_policy")
    if contract.get("identity") in {versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION,
                                   *versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS, versions.MATH_V2_2_EXECUTION_BINDING_VERSION, versions.MATH_VISIBLE_TRAJECTORY_BINDING_VERSION, versions.MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION}:
        expected = prediction_validity_contract()
        if contract['identity'] in {*versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS, versions.MATH_V2_2_EXECUTION_BINDING_VERSION, versions.MATH_VISIBLE_TRAJECTORY_BINDING_VERSION, versions.MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION}:
            expected.update(identity=versions.MATH_RECOVERY_PREDICTION_VERSION, invalid_response_retries=3)
        if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
            raise SearchContractError("MATH_PREDICTION_VALIDITY_BINDING_MISMATCH")
        return dict(policy)
    if policy is not None:
        raise SearchContractError("MATH_PREDICTION_VALIDITY_REQUIRES_FRESH_BINDING")
    return None


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


def classify_prediction(text, finish_reason="stop"):
    if text is not None and not isinstance(text, str):
        raise SearchContractError("MATH_PREDICTION_SERIALIZATION_INVALID")
    reason = None
    expression = ""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    markers = [line for line in lines if line.startswith("FINAL_ANSWER:")]
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        reason = "OUTPUT_TRUNCATED"
    elif finish_reason != "stop":
        reason = "OTHER_PREDICTION_CONTRACT_FAILURE"
    elif not markers:
        reason = "MISSING_FINAL_MARKER"
    elif len(markers) > 1:
        reason = "MULTIPLE_FINAL_MARKERS"
    elif not markers[0].removeprefix("FINAL_ANSWER:").strip():
        reason = "EMPTY_FINAL_PAYLOAD"
    elif final_payload(text) is None:
        reason = "OTHER_PREDICTION_CONTRACT_FAILURE"
    else:
        expression = final_payload(text)
        reason = classify_payload(expression)
    return MATHPrediction(text, finish_reason, expression if reason is None else "",
        reason is None, reason)


def prediction_from_persisted(value):
    if isinstance(value, MATHResolvedPrediction) or (isinstance(value, dict)
            and set(value) == set(MATHResolvedPrediction.__dataclass_fields__)):
        from dataclasses import asdict
        data = asdict(value) if isinstance(value, MATHResolvedPrediction) else dict(value)
        attempts = tuple(prediction_from_persisted(v) for v in data['original_predictions'])
        expected = resolve_predictions(attempts, data['transport_retry_attempts'])
        if (json.dumps(data,sort_keys=True) != json.dumps(asdict(expected),sort_keys=True)
                or any(type(data[k]) is not bool for k in ('prediction_valid','recovered_invalid','terminal_invalid'))
                or any(type(data[k]) is not int for k in ('semantic_attempt_count','raw_invalid_count','transport_retry_attempts'))):
            raise SearchContractError('MATH_RESOLVED_PREDICTION_STATE_CORRUPTION')
        return expected
    if isinstance(value, MATHPrediction):
        expected = classify_prediction(value.text, value.finish_reason)
        if value != expected or type(value.prediction_valid) is not bool:
            raise SearchContractError("MATH_PREDICTION_STATE_CORRUPTION")
        return expected
    if isinstance(value, dict) and set(value) == set(MATHPrediction.__dataclass_fields__):
        expected = classify_prediction(value["text"], value["finish_reason"])
        if value != expected.__dict__ or type(value["prediction_valid"]) is not bool:
            raise SearchContractError("MATH_PREDICTION_STATE_CORRUPTION")
        return expected
    raise SearchContractError("MATH_PREDICTION_STATE_INVALID")


def invalid_recovery_contract():
    return dict(identity=versions.MATH_INVALID_RECOVERY_VERSION, solver_invalid_max_retries=3,
        max_semantic_attempts=4, same_request_bytes=True, first_valid_stops=True,
        transport_failures_consume_semantic_attempt=False, terminal_requires_completed_invalid_attempts=4,
        terminal_invalid='incorrect_continue_no_vote', logical_evaluation_count=1,
        cache='resolved_valid_or_exhausted_terminal_only', invalidity_weight='none')


def frozen_recovery_policy(contract):
    policy=contract.get('invalid_recovery_policy')
    if contract.get('identity') in {*versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS, versions.MATH_V2_2_EXECUTION_BINDING_VERSION, versions.MATH_VISIBLE_TRAJECTORY_BINDING_VERSION, versions.MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION}:
        expected=invalid_recovery_contract()
        if policy != expected or any(type(policy[k]) is not type(v) for k,v in expected.items()):
            raise SearchContractError('MATH_INVALID_RECOVERY_BINDING_MISMATCH')
        return dict(policy)
    if policy is not None:
        raise SearchContractError('MATH_INVALID_RECOVERY_REQUIRES_FRESH_BINDING')
    return None


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
