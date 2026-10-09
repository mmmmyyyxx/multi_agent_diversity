"""Gold-blind frozen final-answer boundaries for structured System Prompt V2.4.

Terminal declarations or a whole standalone expression are eligible. Every explicit marker, including
intermediate boxes, must agree textually; conservative false invalids are allowed.
The existing pinned mathematical payload parser and equivalence scorer are reused.
"""
from dataclasses import asdict
import json
import re

from ..search.schemas import SearchContractError
from .math_prediction_validity import MATHPrediction, MATHResolvedPrediction, classify_payload, resolve_predictions

IDENTITY = 'MATH_EXPLICIT_FINAL_ANSWER_EXTRACTION_V1'
VALIDITY = 'MATH_STRUCTURED_PREDICTION_VALIDITY_V1'
POLICY = dict(identity=IDENTITY, terminal_forms=['markdown_hash3', 'balanced_boxed', 'final_answer_label','standalone_expression'],
    terminal_line='last_nonempty', boxed_suffix='punctuation_only',
    conflicts='all_explicit_markers_must_have_identical_normalized_payload',
    normalization='strip_outer_math_delimiters_and_whitespace_only',
    unmarked_expression='whole_response_single_line_math_syntax_only', gold_access=False, intermediate_selection=False,
    truncated='invalid_even_with_answer', payload_parser='math_verify_no_fallback_v1')


def normalize_payload(value):
    value = value.strip()
    for left, right in (('$$', '$$'), ('\\(', '\\)'), ('\\[', '\\]'), ('$', '$')):
        if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
            value = value[len(left):-len(right)].strip()
            break
    return value


def boundaries(text):
    """Return explicit (payload, start, end, terminal) records without a reference."""
    records = []
    nonempty = [(m.start(), m.group()) for m in re.finditer(r'[^\r\n]+', text) if m.group().strip()]
    last_start = nonempty[-1][0] if nonempty else -1
    for start, line in nonempty:
        match = re.fullmatch(r'\s*(?:###\s+|Final answer:\s*)(.+?)\s*', line, re.I)
        if match:
            payload = match[1]
            if '###' in payload or re.search(r'\bFinal answer:',payload,re.I):
                return None
            # A boxed payload inside a label is unwrapped by the box scanner.
            box=re.search(r'\\boxed\s*\{',payload)
            if box and normalize_payload(payload[:box.start()]).strip('$'):
                return None
            if not box:
                records.append((normalize_payload(payload), start, start + len(line), start == last_start))
    for match in re.finditer(r'\\boxed\s*\{', text):
        depth = 1; cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == '{' and (cursor == 0 or text[cursor-1] != '\\'): depth += 1
            elif text[cursor] == '}' and (cursor == 0 or text[cursor-1] != '\\'): depth -= 1
            cursor += 1
        if depth:
            return None
        end = cursor
        suffix = text[end:].strip()
        terminal = match.start() >= last_start and re.fullmatch(r'[.$\s]*|\\[)\]]\s*[.]*', suffix) is not None
        records.append((normalize_payload(text[match.end():end-1]), match.start(), end, terminal))
    return records


def extract_answer(text):
    if not isinstance(text, str) or not text.strip():
        return '', 'MISSING_EXPLICIT_FINAL_ANSWER'
    records = boundaries(text)
    if records is None:
        return '', 'MALFORMED_FINAL_BOUNDARY'
    if not records or not any(r[3] for r in records):
        # Accept a complete standalone expression, never a number picked out of prose.
        standalone=normalize_payload(text)
        if (not records and '\n' not in standalone and '\r' not in standalone
                and re.fullmatch(r'[\d\s+*/^=.,{}()\[\]<>|;:!%\-\\a-zA-Z]+', standalone)
                and not re.search(r'\b[A-Za-z]{4,}\b', re.sub(r'\\[A-Za-z]+', '', standalone))
                and any(c.isdigit() for c in standalone)):
            return standalone, None
        return '', 'MISSING_EXPLICIT_FINAL_ANSWER'
    if any(not r[0] for r in records):
        return '', 'EMPTY_FINAL_PAYLOAD'
    if len({r[0] for r in records}) != 1:
        return '', 'CONFLICTING_FINAL_ANSWERS'
    payload=next(r[0] for r in records if r[3])
    # The native mathematical parser must not extract a convenient subexpression
    # from prose inside a final declaration. Mathematical commands remain native.
    without_commands=re.sub(r'\\[A-Za-z]+','',payload)
    words=re.findall(r'[A-Za-z]{2,}',without_commands)
    if any(w.casefold() not in {'sqrt','sin','cos','tan','log','ln','pi','oo','inf','cm','mm','kg'} for w in words):
        return '', 'AMBIGUOUS_FINAL_PAYLOAD'
    return payload, None


def classify_prediction(text, finish_reason='stop'):
    if text is not None and not isinstance(text, str):
        raise SearchContractError('MATH_PREDICTION_SERIALIZATION_INVALID')
    expression, reason = extract_answer(text)
    if finish_reason in {'length', 'max_tokens', 'max_output_tokens'}:
        reason = 'OUTPUT_TRUNCATED'
    elif finish_reason != 'stop':
        reason = 'OTHER_PREDICTION_CONTRACT_FAILURE'
    elif reason is None:
        reason = classify_payload(expression)
    return MATHPrediction(text, finish_reason, expression if reason is None else '', reason is None, reason)


def prediction_from_persisted(value):
    data = asdict(value) if isinstance(value, (MATHPrediction, MATHResolvedPrediction)) else value
    if not isinstance(data, dict):
        raise SearchContractError('MATH_STRUCTURED_PREDICTION_STATE_INVALID')
    if set(data) == set(MATHResolvedPrediction.__dataclass_fields__):
        attempts = tuple(prediction_from_persisted(p) for p in data['original_predictions'])
        expected = resolve_predictions(attempts, data['transport_retry_attempts'])
    elif set(data) == set(MATHPrediction.__dataclass_fields__):
        expected = classify_prediction(data['text'], data['finish_reason'])
    else:
        raise SearchContractError('MATH_STRUCTURED_PREDICTION_STATE_INVALID')
    if (json.dumps(data, sort_keys=True) != json.dumps(asdict(expected), sort_keys=True)
            or type(data['prediction_valid']) is not bool):
        raise SearchContractError('MATH_STRUCTURED_PREDICTION_STATE_CORRUPTION')
    if isinstance(expected, MATHResolvedPrediction) and (
            any(type(data[k]) is not bool for k in ('recovered_invalid', 'terminal_invalid'))
            or any(type(data[k]) is not int for k in ('semantic_attempt_count', 'raw_invalid_count', 'transport_retry_attempts'))):
        raise SearchContractError('MATH_STRUCTURED_PREDICTION_STATE_CORRUPTION')
    return expected


def validity_contract():
    return dict(identity=VALIDITY, reference_failure='hard_contract_failure',
        prediction_failure='invalid_incorrect_continue', truncated_even_with_marker='invalid_incorrect',
        invalid_response_retries=3, fallback_extraction=False,
        invalid_vote='excluded_from_equivalence_classes', denominator='all_frozen_examples',
        invalidity_telemetry='observation_only', answer_extraction=POLICY)
