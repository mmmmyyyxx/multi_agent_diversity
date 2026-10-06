"""V5 bounded numeric provenance heuristic, not a semantic provenance oracle.

Coinciding numbers alone do not establish copying. Strong signals are explicit
answer assertions, distinctive source values/ratios, or source quantities with
matching units or an explicit instance-bound cue. Ordinary small structural
values remain legal without those signals. No model call or regeneration occurs.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
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


def numeric_content_leaked(text, rows):
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
