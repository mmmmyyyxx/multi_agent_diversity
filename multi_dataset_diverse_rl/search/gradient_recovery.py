"""Opt-in bounded first-contract-valid draws; no semantic quality selection."""
from copy import deepcopy
import hashlib
import json

from ..current_contract import GRADIENT_CONTRACT_RECOVERY_VERSION
from .schemas import SearchContractError

POLICY = dict(identity=GRADIENT_CONTRACT_RECOVERY_VERSION,
    max_physical_attempts_per_wrong=3, selection='first_contract_valid',
    logical_gradients_per_wrong=1, accepted_gradients_per_wrong=1,
    previous_invalid_output_visible=False, same_provider_visible_request=True,
    semantic_quality_selection=False, llm_judge=False, output_cache=False,
    infrastructure_retry=False)


class GradientOutputContractError(SearchContractError):
    """Raised only after successful generation, never for broker/input errors."""
    def __init__(self, category):
        super().__init__('PATTERN_GRADIENT_EXTRACTION_INVALID')
        self.failure_category = category


def validate_policy(policy):
    if policy is not None and (not isinstance(policy,dict) or digest(policy)!=digest(POLICY)):
        raise SearchContractError('GRADIENT_CONTRACT_RECOVERY_POLICY_MISMATCH')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode()).hexdigest()


def accepted_output(value, row):
    from .textual_gradients import validate_gradient
    from .numeric_provenance import numeric_guard_result
    if not isinstance(value, dict) or set(value) != {'gradient'}:
        raise GradientOutputContractError('SCHEMA_REJECTION')
    text = value['gradient']
    if not isinstance(text, str) or not text.strip():
        raise GradientOutputContractError('EMPTY_OR_NONSTRING_GRADIENT')
    if len(text) > 400:
        raise GradientOutputContractError('LENGTH_REJECTION')
    # The frozen validator, rather than telemetry or usefulness, decides admission.
    try:
        return validate_gradient(text, (row,))
    except SearchContractError as exc:
        if str(exc) != 'PATTERN_GRADIENT_EXTRACTION_INVALID':
            raise
        numeric = numeric_guard_result(text, (row,))
        category = ('STRONG_NUMERIC_PROVENANCE' if numeric['hard_reject']
            else 'FROZEN_GRADIENT_STRUCTURAL_OR_CONTENT_REJECTION')
        raise GradientOutputContractError(category) from None


def extract_first_valid(extractor, procedure, rows):
    from .textual_gradients import GRADIENT_POLICY
    from .pattern_primitives import single_failure_example
    from .numeric_provenance import numeric_guard_result
    provider = extractor.provider
    output = []
    extractor.recovery_batch += 1
    for row in rows:
        payload = dict(schema=GRADIENT_POLICY['schema'],
            current_member_procedure=procedure, example=single_failure_example(row))
        payload_hash = digest(payload)
        logical_id = digest(dict(batch=extractor.recovery_batch,
            example_id=row.example_id, scientific_payload_sha256=payload_hash))
        for attempt in range(1, POLICY['max_physical_attempts_per_wrong'] + 1):
            value = None
            failure = None
            # Only this typed output error crosses the retry boundary. Input,
            # context, transport, accounting and arbitrary exceptions propagate.
            try:
                value = provider.extract(deepcopy(payload))
                text = accepted_output(value, row)
            except GradientOutputContractError as exc:
                failure = exc.failure_category
            generated = value.get('gradient') if isinstance(value, dict) else None
            numeric = numeric_guard_result(generated, (row,)) if isinstance(generated, str) else None
            response = getattr(provider, 'last_response', None)
            raw_hash = (hashlib.sha256(response['text'].encode()).hexdigest()
                if response is not None else digest(value))
            audit = dict(logical_gradient_id=logical_id, example_id=row.example_id,
                physical_attempt_no=attempt, scientific_payload_sha256=payload_hash,
                response_sha256=raw_hash, contract_valid=failure is None,
                failure_category=failure, length=len(generated) if isinstance(generated, str) else None,
                gradient_sha256=hashlib.sha256(generated.encode()).hexdigest() if isinstance(generated, str) else None,
                numeric_guard_result=numeric,
                provider_tokens=({k:response[k] for k in ('input_tokens','output_tokens')}
                    if response is not None else None),
                provider_request_sha256=response.get('request_sha256') if response is not None else None,
                accepted_attempt_no=attempt if failure is None else None,
                accepted_gradient_sha256=hashlib.sha256(text.encode()).hexdigest() if failure is None else None,
                prior_invalid_attempt_count=attempt-1)
            writer = getattr(provider, 'recovery_writer', None)
            if writer is not None:
                writer(deepcopy(audit))  # Failure here forbids another draw.
            extractor.recovery_audit.append(deepcopy(audit))
            if isinstance(generated, str):
                numeric_audit = dict(example_id=row.example_id, gradient_sha256=audit['gradient_sha256'],
                    numeric_guard_result=numeric, logical_gradient_id=logical_id, physical_attempt_no=attempt)
                writer = getattr(provider, 'numeric_guard_writer', None)
                if writer is not None:
                    writer(numeric_audit)
                extractor.numeric_guard_audit.append(numeric_audit)
            if failure is None:
                output.append(dict(example_id=row.example_id, gradient=text))
                break
        else:
            exc = SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID')
            exc.gradient_recovery_exhausted = True
            raise exc
    return tuple(output)


def statistics(events):
    groups = {}
    for event in events:
        groups.setdefault(event['logical_gradient_id'], []).append(event)
    accepted = [g[-1] for g in groups.values() if g[-1]['contract_valid']]
    retries = sum(e['physical_attempt_no'] > 1 for e in accepted)
    exhausted = sum(len(g) == 3 and not g[-1]['contract_valid'] for g in groups.values())
    return dict(logical_gradient_count=len(groups), accepted_gradient_count=len(accepted),
        physical_gradient_calls=len(events), gradient_first_pass_valid=sum(e['physical_attempt_no']==1 for e in accepted),
        gradient_retry_once=sum(e['physical_attempt_no']==2 for e in accepted),
        gradient_retry_twice=sum(e['physical_attempt_no']==3 for e in accepted), gradient_three_fail=exhausted,
        strong_leakage_rejections=sum(e['failure_category']=='STRONG_NUMERIC_PROVENANCE' for e in events),
        numeric_warnings=sum(bool((e['numeric_guard_result'] or {}).get('warning_reasons')) for e in events),
        schema_rejections=sum(e['failure_category'] in ('MALFORMED_JSON','SCHEMA_REJECTION') for e in events),
        length_rejections=sum(e['failure_category']=='LENGTH_REJECTION' for e in events),
        gradient_contract_recovery_rate=(sum(len(g)>1 for g in groups.values())/len(groups) if groups else 0),
        successful_recovery_rate=(retries/sum(len(g)>1 for g in groups.values())
            if any(len(g)>1 for g in groups.values()) else None))
