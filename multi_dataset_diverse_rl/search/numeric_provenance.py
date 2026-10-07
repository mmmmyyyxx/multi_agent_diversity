"""V6 bounded provenance evidence, with preserved V5 replay behavior.

Numeric coincidence, including distinctive values, supplies warnings only.
Explicit answers, copied expressions and source-bound contexts remain terminal.
This deterministic evidence threshold is not a comprehensive semantic oracle.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import re

from .abstraction_content import normalize

_DIGIT = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+'
_DIGITS = re.compile(r'(?<![\w.])(?:' + _DIGIT + r')(?:e[+-]?\d+)?(?!\w|\.\d)')
_SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven '
    'twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(), range(20)))
_TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10)))
_SCALES = {'hundred':100,'thousand':1000,'million':1000000}
_CARDINAL = '|'.join((*_SMALL,*_TENS,*_SCALES))
_WORDS = re.compile(r'\b(?:'+_CARDINAL+r')(?:[ -]+(?:and[ -]+)?(?:'+_CARDINAL+r'))*\b')
_RATIO = re.compile(r'(?<!\w)(\d+)\s*/\s*(\d+)(?!\w)|\\(?:d?frac)\s*\{(\d+)\}\s*\{(\d+)\}')
_UNITS = re.compile(r'^\s*(?:%|(?:dollars?|cents?|euros?|yuan|meters?|metres?|'
    r'centimeters?|kilometers?|miles?|feet|inches|seconds?|minutes?|hours?|days?|'
    r'years?|degrees?|liters?|litres?|kilograms?|grams?|students?|participants?|'
    r'people|players?|balls?|coins?|apples?)\b)')
_INSTANCE = re.compile(r'\b(?:given|specified|provided|stated|observed|supplied)'
    r'(?:\s+[a-z]+){0,3}\s*(?:is|equals|of|:|=)?\s*$')
_ANSWER = re.compile(r'\b(?:answer\s+(?:is|equals)|return|output|report\s+(?:the\s+)?answer(?:\s+as)?)\s*[:=]?\s*$')
_EXPRESSION = re.compile(r'(?<!\w)[a-z0-9]+(?:\s*[=+*/^<>-]\s*[a-z0-9]+)+(?!\w)')


def numeric_answer_literal(text):
    """Recognize bounded numeric literals before symbolic-answer substring checks."""
    return bool(re.fullmatch(r'[\d\s.,+*/^{}()\\-]+',text)
        or _WORDS.fullmatch(text) or _RATIO.fullmatch(text))


@dataclass(frozen=True)
class Number:
    value: Fraction
    start: int
    end: int
    distinctive: bool


def _word_value(words):
    total=current=0
    for word in re.split(r'[ -]+',words):
        if word=='and':continue
        if word in _SMALL:current+=_SMALL[word]
        elif word in _TENS:current+=_TENS[word]
        elif word=='hundred':current=max(1,current)*100
        else:total+=max(1,current)*_SCALES[word];current=0
    return total+current


def numbers(text):
    """Normalize digit/cardinal aliases and simple ratios without a math parser.

    Cardinal English, NFKC digits, decimals, grouped integers, scientific
    notation and integer slash/LaTeX fractions are bounded supported forms.
    Arbitrary verbal/LaTeX algebra and semantic paraphrases are not covered.
    """
    result=[]
    for match in _DIGITS.finditer(text):
        surface=match.group().replace(',','')
        try:
            decimal=Decimal(surface)
            if abs(decimal.as_tuple().exponent)>600:continue
            value=Fraction(decimal)
        except (InvalidOperation,ValueError,OverflowError):continue
        # A numeric-looking provider string is tiny under the 400/600 limit;
        # cap exponent expansion before Fraction conversion for hostile inputs.
        significant=decimal.normalize().as_tuple().digits
        result.append(Number(value,match.start(),match.end(),abs(value)>=100 or len(significant)>=3))
    for match in _WORDS.finditer(text):
        value=Fraction(_word_value(match.group()))
        result.append(Number(value,match.start(),match.end(),abs(value)>=100))
    for match in _RATIO.finditer(text):
        numerator,denominator=(match.group(1),match.group(2)) if match.group(1) is not None else (match.group(3),match.group(4))
        if int(denominator):
            value=Fraction(int(numerator),int(denominator))
            result.append(Number(value,match.start(),match.end(),value not in {0,1,2,Fraction(1,2)}))
    return tuple(result)


def _unit(text, number):
    match=_UNITS.match(text[number.end:])
    if not match:return None
    return match.group().strip().rstrip('s')


def numeric_content_leaked_v5(text, rows):
    text=normalize(text);output=numbers(text)
    for row in rows:
        sources=tuple(normalize(row.signals.get(key,'')) for key in ('input_payload','gold','target_output'))
        for source in sources:
            for match in _EXPRESSION.finditer(source):
                fragment=match.group()
                if (len(re.sub(r'\s','',fragment))>=5
                        and any(n.value not in {0,1,2} for n in numbers(fragment))
                        and re.search(r'(?<!\w)'+r'\s*'.join(re.escape(p) for p in
                            re.findall(r'[a-z0-9]+|[^a-z0-9\s]',fragment))+r'(?!\w)',text)):
                    return True
        source_numbers=tuple((source,n) for source in sources for n in numbers(source))
        answers={n.value for source in sources[1:] for n in numbers(source)}
        for number in output:
            matching=tuple((source,n) for source,n in source_numbers if n.value==number.value)
            prefix=text[max(0,number.start-64):number.start]
            if _ANSWER.search(prefix) and number.value in answers:return True
            if not matching:continue
            if number.distinctive or any(n.distinctive for _,n in matching):return True
            if _INSTANCE.search(prefix):return True
            unit=_unit(text,number)
            if unit is not None and any(_unit(source,n)==unit for source,n in matching):return True
    return False


STRONG_REASONS = (
    'ANSWER_LITERAL_MATCH', 'COPIED_NUMERIC_EXPRESSION', 'VALUE_AND_UNIT_MATCH',
    'INSTANCE_CUED_VALUE_MATCH', 'VALUE_WITH_ENTITY_CONTEXT',
    'VALUE_WITH_LONG_SOURCE_FRAGMENT',
)
WARNING_REASONS = ('BARE_DISTINCTIVE_VALUE_MATCH', 'BARE_NON_DISTINCTIVE_VALUE_MATCH')


def _copied_expressions(text, source):
    """Retain the bounded literal-expression threshold, including subfragments.

    No token-frequency, magnitude threshold, semantic similarity or whitelist
    decides whether an expression is instance-specific.
    """
    from .abstraction_content import _MATH_SPAN,contains_literal
    for match in _MATH_SPAN.finditer(source):
        literal=next(g for g in match.groups() if g is not None)
        if (len(literal)>=5 and re.search(r'[=+*/^\\-]',literal)
                and any(n.value not in {0,1,2} for n in numbers(literal))
                and contains_literal(text,literal)):
            yield match
    for match in _EXPRESSION.finditer(source):
        fragment=match.group()
        # Scientific notation is one numeric literal, not a copied algebraic
        # subtraction expression formed by the exponent's sign.
        if any(n.start()<=match.start() and n.end()>=match.end() for n in _DIGITS.finditer(source)):
            continue
        if (len(re.sub(r'\s','',fragment))>=5
                and any(n.value not in {0,1,2} for n in numbers(fragment))
                and re.search(r'(?<!\w)'+r'\s*'.join(re.escape(p) for p in
                    re.findall(r'[a-z0-9]+|[^a-z0-9\s]',fragment))+r'(?!\w)',text)):
            yield match


def numeric_guard_result(text, rows):
    """Return stable decisions and local provenance; callers sanitize publication.

    Spans index normalized source fields, and their hashes bind exact normalized
    surfaces. Values and source IDs are private audit information. Warning fields
    never participate in Pattern ranking, Memory, search or stopping.
    """
    from .abstraction_content import (contains_literal, _ENTITY_CONTEXTS,
        _COORDINATED_CONSTRAINT, _QUOTED)
    from ..local_optimizers.example_text import contains_supplied_example_text
    from ..local_optimizers.schemas import LocalEvidenceExample
    text=normalize(text);output=numbers(text);matches=[];strong=set();warnings=set()
    expression_matches=[]
    for row in rows:
        sources=tuple((key,normalize(row.signals.get(key,'')))
            for key in ('input_payload','gold','target_output'))
        source_numbers=tuple((key,source,n) for key,source in sources for n in numbers(source))
        answers={n.value for key,_,n in source_numbers if key in ('gold','target_output')}
        for key,source in sources:
            for match in _copied_expressions(text,source):
                strong.add('COPIED_NUMERIC_EXPRESSION')
                expression_matches.append(dict(example_id=row.example_id,source_field=key,
                    source_span=[match.start(),match.end()],
                    source_span_sha256=hashlib.sha256(match.group().encode()).hexdigest()))
        raw_source=str(row.signals.get('input_payload',''))
        entities=[m.group('name') for pattern in _ENTITY_CONTEXTS for m in pattern.finditer(raw_source)]
        entities.extend(m.group(key) for m in _COORDINATED_CONSTRAINT.finditer(raw_source)
            for key in ('first','second'))
        entities.extend(next(g for g in m.groups() if g is not None)
            for m in _QUOTED.finditer(raw_source)
            if len(next(g for g in m.groups() if g is not None).split())>=3)
        entity=any(contains_literal(text,normalize(e)) for e in entities)
        long_fragment=contains_supplied_example_text(text,(LocalEvidenceExample(
            row.example_id,row.signals.get('input_payload',''),row.signals.get('gold','')),))
        for number in output:
            matching=tuple((key,source,n) for key,source,n in source_numbers if n.value==number.value)
            if not matching:continue
            prefix=text[max(0,number.start-64):number.start];reasons=set()
            # An entire numeric answer is an answer assertion even without prose.
            if number.value in answers and (_ANSWER.search(prefix)
                    or numeric_answer_literal(text) and any(
                        text==source for key,source in sources if key in ('gold','target_output'))):
                reasons.add('ANSWER_LITERAL_MATCH')
            if _INSTANCE.search(prefix):reasons.add('INSTANCE_CUED_VALUE_MATCH')
            unit=_unit(text,number)
            if unit is not None and any(_unit(source,n)==unit for _,source,n in matching):
                reasons.add('VALUE_AND_UNIT_MATCH')
            if entity:reasons.add('VALUE_WITH_ENTITY_CONTEXT')
            if long_fragment:reasons.add('VALUE_WITH_LONG_SOURCE_FRAGMENT')
            strong.update(reasons)
            # A copied expression can independently establish a hard rejection;
            # bare matched values still get their own truthful warning evidence.
            warning=None
            if not reasons:
                warning=('BARE_DISTINCTIVE_VALUE_MATCH' if number.distinctive
                    or any(n.distinctive for _,_,n in matching) else 'BARE_NON_DISTINCTIVE_VALUE_MATCH')
                warnings.add(warning)
            for key,source,n in matching:
                matches.append(dict(value=str(number.value),example_id=row.example_id,
                    source_field=key,gradient_span=[number.start,number.end],
                    source_span=[n.start,n.end],
                    source_span_sha256=hashlib.sha256(source[n.start:n.end].encode()).hexdigest(),
                    strong_reasons=[r for r in STRONG_REASONS if r in reasons],warning_reason=warning))
    return dict(hard_reject=bool(strong),strong_reasons=[r for r in STRONG_REASONS if r in strong],
        warning_reasons=[r for r in WARNING_REASONS if r in warnings],
        matched_values=sorted({m['value'] for m in matches}),matches=matches,
        expression_matches=expression_matches)


def numeric_content_leaked(text, rows):
    return numeric_guard_result(text,rows)['hard_reject']
