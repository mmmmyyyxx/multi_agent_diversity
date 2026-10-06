"""Deterministic provenance checks for the opt-in Pattern abstraction boundary.

Capitalization or an ordinary word shared with a problem is not evidence of
an entity. Entity candidates require an explicit naming or actor context.
This is a bounded content detector, not a general semantic/NER oracle.
"""
import re
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


_NAME = r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*"
_ENTITY_CONTEXTS = tuple(re.compile(pattern) for pattern in (
    # Explicit names, honorifics and human roles provide evidence beyond case.
    rf'\b(?:named|called)\s+[\"\']?(?P<name>{_NAME})\b',
    rf'\b(?:Mr|Mrs|Ms|Dr|Professor)\.?\s+(?P<name>{_NAME})\b',
    rf'\b(?:student|person|player|customer|worker|teacher|friend)\s+(?P<name>{_NAME})\b',
    # These inflected actor actions are distinct from mathematical imperatives.
    rf'\b(?P<name>{_NAME})\s+(?:calculates|buys|sells|owns|spends|earns|pays|travels|walks|runs|collects|chooses|thinks|says)\b',
    rf'\b(?P<name>{_NAME})\s+and\s+{_NAME}\s+(?:each|both|have|buy|sell|own|share|play)\b',
    rf'\b{_NAME}\s+and\s+(?P<name>{_NAME})\s+(?:each|both|have|buy|sell|own|share|play)\b',
))
_QUOTED = re.compile(r'"([^"\n]+)"|“([^”\n]+)”|\'([^\'\n]+)\'')
_MATH_SPAN = re.compile(r'\$([^$\n]+)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]')

# A relational participation constraint supplies actor provenance even when
# neither name is individually followed by an action. The optional comma is
# part of the source grammar, not evidence that ordinary capitals are names.
_COORDINATED_CONSTRAINT = re.compile(
    rf'\b(?P<first>{_NAME})\s+and\s+(?P<second>{_NAME})\s*,?\s*'
    r'(?:refuse|decline)\s+to\s+(?:play|work|travel|sit|participate)\s+'
    r'(?:together|with\s+each\s+other)\b')


def contains_literal(text, literal):
    """Match a complete normalized literal, never a substring of another word."""
    return bool(literal and re.search(r'(?<!\w)' + re.escape(literal) + r'(?!\w)', text))


def specific_content_leaked(text, rows, *, numeric_gold=True):
    normalized = normalize(text)
    for row in rows:
        source = unicodedata.normalize('NFKC', str(row.signals['input_payload']))
        gold = normalize(row.signals['gold'])
        numeric_answer = bool(re.fullmatch(r'[\d\s.,+*/^{}()\\-]+', gold))
        if gold and (numeric_gold or not numeric_answer) and (normalized == gold or len(gold) >= 3 and contains_literal(normalized, gold)):
            return True
        # Short symbolic answers are prohibited when supplied as an answer,
        # while a variable in a general reasoning instruction remains legal.
        if gold and re.search(r'\b(?:answer\s+is|answer\s+equals|return|output)\s+' +
                re.escape(gold) + r'(?!\w)', normalized):
            return True
        for pattern in _ENTITY_CONTEXTS:
            for match in pattern.finditer(source):
                if contains_literal(normalized, normalize(match.group('name'))):
                    return True
        # Quoted multiword labels and explicit mathematical expressions are
        # stronger provenance evidence than ordinary vocabulary overlap.
        for match in _QUOTED.finditer(source):
            literal = normalize(next(group for group in match.groups() if group is not None))
            if len(literal.split()) >= 3 and contains_literal(normalized, literal):
                return True
        for match in _MATH_SPAN.finditer(source):
            literal = normalize(next(group for group in match.groups() if group is not None))
            if len(literal) >= 5 and re.search(r'[=+*/^\\-]', literal) and contains_literal(normalized, literal):
                return True
    return False


def current_specific_content_leaked(text, rows, *, numeric_gold=True):
    """V4 implementation repair; the original V3 helper remains replayable."""
    rows = tuple(rows)
    if specific_content_leaked(text, rows, numeric_gold=numeric_gold):
        return True
    normalized = normalize(text)
    for row in rows:
        source = unicodedata.normalize('NFKC', str(row.signals['input_payload']))
        for match in _COORDINATED_CONSTRAINT.finditer(source):
            if any(contains_literal(normalized, normalize(match.group(key)))
                    for key in ('first', 'second')):
                return True
    return False
