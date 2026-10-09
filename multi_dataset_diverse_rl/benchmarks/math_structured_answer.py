"""Gold-blind explicit terminal declarations; mathematical scoring stays pinned."""
from dataclasses import asdict
import json
import re
from ..search.schemas import SearchContractError
from .math_prediction_validity import MATHPrediction, MATHResolvedPrediction, classify_payload, resolve_predictions

IDENTITY = 'MATH_EXPLICIT_FINAL_ANSWER_EXTRACTION_V2'
VALIDITY = 'MATH_EXPLICIT_FINAL_PREDICTION_VALIDITY_V2'
POLICY = dict(identity=IDENTITY, terminal_forms=['final_answer_label','underscore_final_answer_label','labeled_balanced_boxed'],
    terminal_line='last_nonempty', boxed_suffix='punctuation_only',
    conflicts='explicit_declarations_must_be_mathematically_equivalent_gold_blind',
    normalization='strip_outer_math_delimiters_and_whitespace_only',
    unmarked_expression='forbidden', gold_access=False, intermediate_selection=False,
    truncated='invalid_even_with_answer', payload_parser='math_verify_no_fallback_v1')
LABEL = re.compile(r'^\s*Final[ _]+answer\s*:\s*(.*?)\s*$', re.I)


def normalize_payload(value):
    value = value.strip()
    for left, right in (('$$', '$$'), ('\\(', '\\)'), ('\\[', '\\]'), ('$', '$')):
        if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
            value = value[len(left):-len(right)].strip()
            break
    return value


def _label_payload(value):
    value = normalize_payload(value)
    match = re.match(r'\\boxed\s*\{', value)
    if not match:
        return value if '\\boxed' not in value else None
    depth = 1
    cursor = match.end()
    while cursor < len(value) and depth:
        if value[cursor] == '{' and value[cursor-1] != '\\': depth += 1
        elif value[cursor] == '}' and value[cursor-1] != '\\': depth -= 1
        cursor += 1
    if depth or not re.fullmatch(r'[.\s]*', value[cursor:]):
        return None
    return normalize_payload(value[match.end():cursor-1])


def boundaries(text):
    """Only labeled declarations. Headings and intermediate boxes are ordinary text."""
    lines = [(m.start(), m.group()) for m in re.finditer(r'[^\r\n]+', text) if m.group().strip()]
    last = lines[-1][0] if lines else -1
    records = []
    for start, line in lines:
        match = LABEL.fullmatch(line)
        if match:
            payload = _label_payload(match[1])
            if payload is None: return None
            records.append((payload, start, start + len(line), start == last))
    return records


def extract_answer(text):
    if not isinstance(text, str) or not text.strip():
        return '', 'MISSING_EXPLICIT_FINAL_ANSWER'
    records = boundaries(text)
    if records is None: return '', 'MALFORMED_FINAL_BOUNDARY'
    if not records or not records[-1][3]: return '', 'MISSING_EXPLICIT_FINAL_ANSWER'
    if any(not r[0] for r in records): return '', 'EMPTY_FINAL_PAYLOAD'
    for payload, *_ in records:
        without_commands = re.sub(r'\\[A-Za-z]+', '', payload)
        words = re.findall(r'[A-Za-z]{2,}', without_commands)
        if any(w.casefold() not in {'sqrt','sin','cos','tan','log','ln','pi','oo','inf','cm','mm','kg'} for w in words):
            return '', 'AMBIGUOUS_FINAL_PAYLOAD'
    final = records[-1][0]
    if len({r[0] for r in records}) > 1:
        from .math_domain_v2 import domain_matrix
        # No gold is available here. Compare declarations to one another using
        # the same pinned relation, instead of treating equivalent syntax as conflict.
        for other, *_ in records[:-1]:
            if other == final: continue
            relation = domain_matrix((other, final))
            if not all(relation['valid']): return '', 'PAYLOAD_PARSE_FAILURE'
            if not relation['equivalence'][0][1]: return '', 'CONFLICTING_FINAL_ANSWERS'
    return final, None


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
