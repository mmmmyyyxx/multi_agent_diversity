"""Gold-blind, read-only baseline diagnosis; never a production scoring policy.

No provider client or optimizer is imported. Diagnostic extraction is separate
from the frozen native parser and cannot be used as an arm's primary result.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import re

from ..benchmarks.math_structured_answer import LABEL, normalize_payload, classify_prediction
from ..benchmarks.math_prediction_validity import classify_payload
from ..benchmarks.math_domain_v2 import domain_matrix

DIAGNOSTIC_ID = 'MATH_BASELINE_COMMON_DIAGNOSTIC_V1'
BOXED_ID = 'MATH_BASELINE_BOXED_TERMINAL_V1'
NATIVE_ID = 'MATH_EXPLICIT_FINAL_ANSWER_EXTRACTION_V2'
TRUNCATED = {'length', 'max_tokens', 'max_output_tokens'}
REASONS = ('MISSING_FINAL_LABEL', 'MARKDOWN_WRAPPED_LABEL', 'FINAL_LABEL_NOT_LAST',
    'BOXED_ONLY', 'MALFORMED_FINAL_PAYLOAD', 'CONFLICTING_FINAL_DECLARATIONS',
    'MATH_PAYLOAD_PARSE_FAILURE', 'OUTPUT_TRUNCATED',
    'OTHER_OUTPUT_CONTRACT_FAILURE', 'INSUFFICIENT_EVIDENCE')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def balanced_box(value):
    """Exactly one whole expression, without prose or trailing punctuation."""
    value=normalize_payload(value)
    match=re.match(r'\\boxed\s*\{',value)
    if not match:return None
    depth=1;cursor=match.end()
    while cursor<len(value) and depth:
        escaped=(len(value[:cursor])-len(value[:cursor].rstrip('\\'))) % 2
        if not escaped:
            if value[cursor]=='{':depth+=1
            elif value[cursor]=='}':depth-=1
        cursor+=1
    if depth or value[cursor:].strip():return None
    payload=normalize_payload(value[match.end():cursor-1])
    return payload if payload else None


def undecorate_label(line):
    value=line.strip()
    value=re.sub(r'^(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)', '', value)
    if value.startswith(('**','__')):
        marker=value[:2]
        if value.endswith(marker):value=value[2:-2].strip()
        else:value=re.sub(r'^(?:\*\*|__)(Final[ _]+answer\s*:)(?:\*\*|__)\s*',r'\1 ',value,flags=re.I)
    return value


def diagnostic_payload(value):
    value=value.strip()
    for marker in ('**','__','`'):
        if value.startswith(marker) and value.endswith(marker) and len(value)>2*len(marker):
            value=value[len(marker):-len(marker)].strip()
    return normalize_payload(value)


def declarations(text):
    """Explicit assertions only; no search through arbitrary intermediate math."""
    lines=[line.strip() for line in (text or '').splitlines() if line.strip()]
    items=[]
    for index,line in enumerate(lines):
        cleaned=undecorate_label(line)
        match=LABEL.fullmatch(cleaned)
        if match:
            value=diagnostic_payload(match[1]);boxed=balanced_box(value)
            if not value and index+1<len(lines):
                next_line=diagnostic_payload(lines[index+1])
                if lines[index+1] in {'\\[','$$'}:
                    closing='\\]' if lines[index+1]=='\\[' else '$$'
                    end=next((j for j in range(index+2,len(lines)) if lines[j]==closing),None)
                    if end is not None:
                        next_line=diagnostic_payload(' '.join(lines[index+2:end]))
                if not re.search(r'[A-Za-z]{2,}',re.sub(r'\\[A-Za-z]+','',next_line)):
                    value=next_line;boxed=balanced_box(value)
            items.append(dict(payload=boxed if boxed is not None else value,
                form='wrapped_label' if cleaned!=line else 'plain_label',terminal=index==len(lines)-1))
            continue
        match=re.fullmatch(r'(?:Thus[, ]+|Therefore[, ]+|Hence[, ]+)?(?:the\s+)?(?:final\s+)?answer\s+(?:is|=)\s+(.+?)\.?',cleaned,re.I)
        if match:
            value=diagnostic_payload(match[1]);boxed=balanced_box(value)
            items.append(dict(payload=boxed if boxed is not None else value,
                form='natural_language_final',terminal=index==len(lines)-1))
    if lines:
        last=diagnostic_payload(lines[-1])
        boxed=balanced_box(last)
        if boxed is None:
            # A concluding prose prefix is diagnostic only. All suffix content
            # still has to be one complete box; intermediate boxes are ignored.
            match=re.search(r'\\boxed\s*\{',last)
            if match and re.fullmatch(r'(?:Thus|Therefore|Hence|So|The (?:final )?answer is)[,:\s]*',last[:match.start()],re.I):
                boxed=balanced_box(last[match.start():])
        if boxed is not None:
            items.append(dict(payload=boxed,form='terminal_boxed',terminal=True))
        elif last in {'\\]','$$'}:
            raw=(text or '').rstrip()
            closing=last;opening='\\[' if closing=='\\]' else '$$'
            start=raw.rfind(opening,0,len(raw)-len(closing))
            if start>=0:
                payload=raw[start+len(opening):len(raw)-len(closing)].strip()
                boxed=balanced_box(payload)
                if boxed is not None:
                    items.append(dict(payload=boxed,form='terminal_boxed_display_block',terminal=True))
    return lines,items


@dataclass(frozen=True)
class DiagnosticExtraction:
    policy_identity: str
    usage_scope: str
    status: str
    payload: str | None
    forms: tuple[str,...]
    candidate_count: int
    mathematical_parse_reason: str | None
    truncated: bool


def diagnostic_extract(text,finish_reason='stop'):
    """No reference argument or gold-dependent selection is possible here."""
    truncated=finish_reason in TRUNCATED
    lines,items=declarations(text)
    if not items and lines and not truncated:
        value=normalize_payload(lines[-1])
        # Whole final-line syntax only; prose or intermediate equalities are
        # never searched for a favorable value. A final equality stays intact.
        if value and not re.search(r'[A-Za-z]{2,}',re.sub(r'\\[A-Za-z]+','',value)):
            items=[dict(payload=value,form='standalone_terminal_expression',terminal=True)]
    forms=tuple(sorted({i['form'] for i in items}))
    values=list(dict.fromkeys(i['payload'] for i in items))
    def result(status,payload=None,reason=None):
        return DiagnosticExtraction(DIAGNOSTIC_ID,'DIAGNOSTIC_ONLY',status,payload,forms,len(values),reason,truncated)
    if not values:return result('UNKNOWN_NO_EXPLICIT_RESULT')
    reasons=[classify_payload(v) if v else 'EMPTY_PAYLOAD' for v in values]
    if any(reasons):return result('UNPARSEABLE_CANDIDATE',reason=next(r for r in reasons if r))
    for value in values[1:]:
        relation=domain_matrix((values[0],value))
        if not all(relation['valid']):return result('UNPARSEABLE_CANDIDATE',reason='PAYLOAD_PARSE_FAILURE')
        if not relation['equivalence'][0][1]:return result('AMBIGUOUS_MULTIPLE_RESULTS')
    return result('EXPLICIT_RESULT_TRUNCATED' if truncated else 'EXPLICIT_PARSEABLE_RESULT',values[-1])


def grade_diagnostic(extraction,reference):
    """Reference is consulted only after immutable, gold-blind extraction."""
    if extraction.policy_identity!=DIAGNOSTIC_ID or extraction.usage_scope!='DIAGNOSTIC_ONLY':
        raise ValueError('DIAGNOSTIC_IDENTITY_REQUIRED')
    if extraction.payload is None:return None
    relation=domain_matrix((reference,extraction.payload))
    if not relation['valid'][0]:raise ValueError('REFERENCE_UNSCORABLE')
    return bool(relation['valid'][1] and relation['equivalence'][0][1])


def primary_reason(text,finish_reason,native):
    if text is None:return 'INSUFFICIENT_EVIDENCE'
    if finish_reason in TRUNCATED:return 'OUTPUT_TRUNCATED'
    if native.prediction_valid:return 'VALID'
    reason=native.invalid_reason
    if reason=='OTHER_PREDICTION_CONTRACT_FAILURE':return 'OTHER_OUTPUT_CONTRACT_FAILURE'
    if reason=='CONFLICTING_FINAL_ANSWERS':return 'CONFLICTING_FINAL_DECLARATIONS'
    if reason in {'PAYLOAD_PARSE_FAILURE','PAYLOAD_UNSUPPORTED'}:return 'MATH_PAYLOAD_PARSE_FAILURE'
    if reason in {'MALFORMED_FINAL_BOUNDARY','EMPTY_FINAL_PAYLOAD','AMBIGUOUS_FINAL_PAYLOAD'}:
        return 'MALFORMED_FINAL_PAYLOAD'
    lines,items=declarations(text)
    if any(i['form']=='wrapped_label' and i['terminal'] for i in items):return 'MARKDOWN_WRAPPED_LABEL'
    if any(i['form'] in {'plain_label','wrapped_label'} for i in items):return 'FINAL_LABEL_NOT_LAST'
    if any(i['form'].startswith('terminal_boxed') for i in items):return 'BOXED_ONLY'
    if reason=='MISSING_EXPLICIT_FINAL_ANSWER':return 'MISSING_FINAL_LABEL'
    return 'OTHER_OUTPUT_CONTRACT_FAILURE'


def secondary_flags(text,finish_reason):
    lines,items=declarations(text)
    return dict(contains_boxed='\\boxed' in (text or ''),
        contains_explicit_natural_final=any(i['form']=='natural_language_final' for i in items),
        contains_markdown_heading=any(re.match(r'^#{1,6}\s',line) for line in lines),
        contains_explicit_declaration=bool(items),output_truncated=finish_reason in TRUNCATED,
        multiple_nonempty_lines=len(lines)>1,
        written_work_status='OBSERVED_MULTILINE_CONTENT_NOT_PROOF_OF_VALID_REASONING' if len(lines)>1 else 'NOT_OBSERVED')


def boxed_native(text,finish_reason='stop'):
    lines=[line.strip() for line in (text or '').splitlines() if line.strip()]
    payload=balanced_box(lines[-1]) if lines else None
    reason=('OUTPUT_TRUNCATED' if finish_reason in TRUNCATED else
        'OTHER_OUTPUT_CONTRACT_FAILURE' if finish_reason!='stop' else
        'MISSING_TERMINAL_BOXED' if payload is None else classify_payload(payload))
    return dict(policy_identity=BOXED_ID,usage_scope='ARM_NATIVE',valid=reason is None,
        payload=payload if reason is None else None,invalid_reason=reason)


def native_extract(arm,text,finish_reason='stop'):
    if arm in ('A','B'):
        prediction=classify_prediction(text,finish_reason)
        return dict(policy_identity=NATIVE_ID,usage_scope='ARM_NATIVE',valid=prediction.prediction_valid,
            payload=prediction.answer if prediction.prediction_valid else None,invalid_reason=prediction.invalid_reason)
    if arm=='C':return boxed_native(text,finish_reason)
    raise ValueError('CALIBRATION_ARM_UNKNOWN')


def primary_correctness(native,reference):
    if (not isinstance(native,dict) or native.get('usage_scope')!='ARM_NATIVE'
            or native.get('policy_identity') not in {NATIVE_ID,BOXED_ID}):
        raise ValueError('DIAGNOSTIC_CANNOT_SUPPLY_PRIMARY_SCORE')
    if not native['valid']:return False
    relation=domain_matrix((reference,native['payload']))
    if not relation['valid'][0]:raise ValueError('REFERENCE_UNSCORABLE')
    return bool(relation['valid'][1] and relation['equivalence'][0][1])


def summarize_rounds(logicals,attempts,errors):
    rounds=[]
    for ordinal in range(1,5):
        observed=[a for a in attempts if a['semantic_attempt_index']==ordinal]
        failures=[e for e in errors if e['semantic_attempt_no']==ordinal]
        rounds.append(dict(semantic_attempt_index=ordinal,eligible_logical_examples=len(observed),
            original_logical_denominator=len(logicals),successful_responses=len(observed),
            became_valid=sum(a['native_valid'] for a in observed),
            became_correct=sum(a['official_correctness_under_frozen_rule'] for a in observed),
            conditional_valid_rate=sum(a['native_valid'] for a in observed)/len(observed) if observed else None,
            input_tokens=sum(a['input_tokens'] for a in observed),
            output_tokens=sum(a['output_tokens'] for a in observed),
            reported_tokens=sum(a['input_tokens']+a['output_tokens'] for a in observed),
            transport_errors=len(failures),
            unknown_usage_reservation=sum(e.get('conservative_charge',0) for e in failures),
            output_truncated=sum(a['finish_reason'] in TRUNCATED for a in observed)))
    return rounds


def paired_comparison(left,right):
    """Complete matching IDs only; incomplete panels are never efficacy results."""
    if set(left)!=set(right):raise ValueError('PAIRED_MEMBERSHIP_MISMATCH')
    n=len(left)
    fields=('first_valid','first_correct','recovered_valid','recovered_correct')
    result={}
    for field in fields:
        improved=sum(not left[k][field] and right[k][field] for k in left)
        regressed=sum(left[k][field] and not right[k][field] for k in left)
        result[field]=dict(improved=improved,regressed=regressed,paired_delta=(improved-regressed)/n if n else None)
    return dict(example_count=n,metrics=result)
