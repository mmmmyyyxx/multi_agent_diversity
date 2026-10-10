"""Bounded source/context evidence for generated abstractions, not a semantic oracle.

Genericity needs both a parameterized expression and an explicit general-method
context. Literal overlap alone is ambiguous and rejects a draw conservatively.
No expression from an observed incident is whitelisted. Numeric provenance,
answer intent, entities and long reference copying remain independent checks.
"""
import hashlib
import re
from .abstraction_content import (normalize, contains_literal, _MATH_SPAN,
    _ENTITY_CONTEXTS, _COORDINATED_CONSTRAINT, _QUOTED)
from .numeric_provenance import numeric_guard_result, numbers, numeric_answer_literal
from ..local_optimizers.example_text import contains_supplied_example_text
from ..local_optimizers.schemas import LocalEvidenceExample

IDENTITY = 'SOURCE_CONTEXT_SYMBOLIC_ABSTRACTION_GUARD_V1'
_GENERAL = re.compile(r'\b(?:general(?:ized)?|generic|symbolic|identity|identities|'
    r'formula|definition|normalize|normalized|normalization|for any|for arbitrary)\b')
_INSTANCE = re.compile(r'\b(?:this (?:problem|example|question)|given (?:number|value|constant)s?|'
    r'provided (?:number|value|constant)s?|answer lookup|memorized answer|reference answer|gold answer)\b')
_FIXED = re.compile(r'\b(?:answer\s+(?:is|equals)|return|output|report\s+(?:the\s+)?answer)\s*[:=]?\s*$')
_REFERENCE = re.compile(r'\b(?:copy|return|output|use)\s+(?:the\s+)?(?:gold|ground\s*truth|reference)\s+(?:answer|solution)\b')
_COMMANDS = frozenset(('frac','dfrac','tfrac','sqrt','sum','prod','log','ln',
    'sin','cos','tan','pi','delta','pm','cdot','times','left','right','le','ge','neq'))
_FIELDS = ('input_payload','gold','target_output','reference_solution')

def generic_expression(text, literal):
    """Conservative structural admission, without evaluating model-supplied code.

    Small structural coefficients alone are insufficient. Need multiple symbolic
    parameters (or a nontrivial one-variable identity), general-method context,
    no answer/instance cue, and closed mathematical syntax. Other coincidences
    remain ambiguous, not established leaks and not accepted abstractions.
    """
    text=normalize(text);literal=normalize(literal)
    occurrences=tuple(re.finditer(r'(?<!\w)'+re.escape(literal)+r'(?!\w)',text))
    if not occurrences:return False
    commands=re.findall(r'\\([a-z]+)',literal)
    if any(c not in _COMMANDS for c in commands):return False
    syntax=re.sub(r'\\[a-z]+',' ',literal)
    if re.search(r'[^a-z0-9\s=+*/^_{}()[\].,<>!|-]',syntax):return False
    # A closed elementary coefficient grammar is a necessary structural bound,
    # never sufficient evidence. Magnitude alone never admits a formula.
    if any(n.value not in {0,1,2,3,4} for n in numbers(syntax)):return False
    # Small case totals are still supplied values. Only zero/unit equations can
    # qualify as structural constraints; a genericity claim cannot admit a
    # copied equation fixed to another numeric total, on either side.
    if '=' in syntax:
        for side in syntax.split('='):
            scalar=side.strip().strip('{}()').strip()
            if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',scalar) and any(n.value not in {0,1} for n in numbers(scalar)):
                return False
    parameters=set(re.findall('[a-z]',syntax))
    operators=re.findall(r'[=+*/^-]',syntax)
    parameterized=len(parameters)>=2 and len(operators)>=2
    for occurrence in occurrences:
        context=text[max(0,occurrence.start()-100):min(len(text),occurrence.end()+100)]
        if (not _GENERAL.search(context) or _INSTANCE.search(context)
                or _FIXED.search(text[max(0,occurrence.start()-64):occurrence.start()])):
            return False
        identity=(len(parameters)==1 and '=' in syntax and len(operators)>=4
            and re.search(r'\bidentit(?:y|ies)\b',context))
        if not (parameterized or identity):return False
    return True

def abstraction_result(text, rows):
    text=normalize(text);rows=tuple(rows);matches=[];warnings=[]
    def record(row,field,source,start,end,category,generic=False):
        value=dict(example_id=row.example_id,source_field=field,source_span=[start,end],
            source_span_sha256=hashlib.sha256(source[start:end].encode()).hexdigest(),
            category=category,generic_structure_and_context=generic)
        (warnings if generic else matches).append(value)
    if _REFERENCE.search(text):
        matches.append(dict(category='EXPLICIT_REFERENCE_COPY_INTENT'))
    for row in rows:
        gold=normalize(row.signals['gold'])
        if gold and (text==gold or len(gold)>=3 and not numeric_answer_literal(gold) and contains_literal(text,gold)
                or re.search(r'\b(?:answer\s+is|answer\s+equals|return|output)\s+'+re.escape(gold)+r'(?!\w)',text)):
            record(row,'gold',gold,0,len(gold),'REFERENCE_ANSWER_LITERAL')
        source=str(row.signals['input_payload'])
        entities=[m.group('name') for p in _ENTITY_CONTEXTS for m in p.finditer(source)]
        entities.extend(m.group(k) for m in _COORDINATED_CONSTRAINT.finditer(source) for k in ('first','second'))
        entities.extend(next(g for g in m.groups() if g is not None) for m in _QUOTED.finditer(source)
            if len(next(g for g in m.groups() if g is not None).split())>=3)
        for name in entities:
            if contains_literal(text,normalize(name)):
                matches.append(dict(example_id=row.example_id,source_field='input_payload',category='SOURCE_ENTITY_OR_LABEL'))
        for field in _FIELDS:
            source=normalize(row.signals.get(field,''))
            if not source:continue
            if contains_supplied_example_text(text,(LocalEvidenceExample(row.example_id,source,gold),)):
                record(row,field,source,0,len(source),'LONG_SOURCE_FRAGMENT')
            for match in _MATH_SPAN.finditer(source):
                group=next(i for i,g in enumerate(match.groups(),1) if g is not None)
                captured=match.group(group);literal=normalize(captured)
                if len(literal)>=5 and re.search(r'[=+*/^\\-]',literal) and contains_literal(text,literal):
                    generic=field!='gold' and generic_expression(text,literal)
                    a,b=match.span(group)
                    a+=len(captured)-len(captured.lstrip());b-=len(captured)-len(captured.rstrip())
                    record(row,field,source,a,b,'SOURCE_MATH_SPAN',generic)
    numeric=numeric_guard_result(text,rows)
    # V6 numeric-expression overlap is reconsidered using the same bounded
    # generic criterion. All other strong numeric provenance is preserved.
    retained=[];generic_numeric=[]
    byid={row.example_id:row for row in rows}
    for match in numeric['expression_matches']:
        row=byid[match['example_id']];source=normalize(row.signals.get(match['source_field'],''))
        a,b=match['source_span'];literal=source[a:b]
        if match['source_field']!='gold' and generic_expression(text,literal):generic_numeric.append(match)
        else:retained.append(match)
    strong=[r for r in numeric['strong_reasons'] if r!='COPIED_NUMERIC_EXPRESSION' or retained]
    numeric={**numeric,'expression_matches':retained,'generic_expression_warnings':generic_numeric,
        'strong_reasons':strong,'hard_reject':bool(strong)}
    return dict(identity=IDENTITY,hard_reject=bool(matches or numeric['hard_reject']),
        source_matches=matches,generic_overlap_warnings=warnings,numeric=numeric,
        semantic_leakage_claim='BOUNDED_PROVENANCE_NOT_SEMANTIC_PROOF')
