"""Opt-in mathematical reasoning boundary; legacy lexical guards are untouched."""
import re
import unicodedata

from .mutable_prompt_contract import _FIXED_ANSWER_PAYLOAD
from ..local_optimizers.example_text import contains_supplied_example_text

IDENTITY = 'SEMANTIC_MUTABLE_REASONING_CONTRACT_V2'


def normalized_procedure(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


_PRESENTATION = tuple(re.compile(p) for p in (
    r'(?<!\w)final[_-]+answer(?!\w)',
    r'\b(?:mandatory\s+output\s+interface|solver\s+output\s+contract)\b',
    r'\b(?:end|begin|start)\s+(?:your\s+|the\s+)?(?:response|output|reply)\s+with\b',
    r'\b(?:reply|respond|return|output|print)\s+(?:with\s+)?(?:only|exactly)\b',
    r'\b(?:final\s+)?(?:response|output)\s+(?:must|should|shall|has\s+to|needs\s+to|is\s+to|will)\b',
    r'\b(?:reply|respond|response|output|answer|return|print)\b.{0,60}\b(?:one|single|two|three|\d+)\s+(?:nonempty\s+)?lines?\b',
    r'\b(?:last|first|final|answer)\s+line\b|\b(?:line\s+count|newlines?|literal\s+prefix|parser\s+(?:marker|token))\b',
    r'\b(?:markdown|code\s+fences?|triple[- ]backticks?|json|xml)\b|```',
    r'\b(?:append|add|attach)\b.{0,100}\b(?:to|in)\s+(?:the\s+)?(?:final\s+)?(?:output|response|answer)\b',
    r'\bconfidence\s+(?:indicator|label|score|tag)\b',
    r'\b(?:output|return|respond\s+with)\s+["\'`]?none["\'`]?(?!\w)',
    r'\b(?:no|without|omit|exclude)\s+(?:any\s+)?(?:extra\s+)?(?:text|commentary|explanation|reasoning|words)\b',
    r'\b(?:do\s+not|never|avoid)\s+(?:include|show|print|provide)\b.{0,80}(?:commentary|intermediate\s+(?:reasoning|calculations)|extraneous\s+(?:information|text)).{0,80}\b(?:final\s+)?(?:output|response|answer(?:\s+block)?)\b',
))


def semantic_violation_reasons(prompt):
    text=normalized_procedure(prompt)
    reasons=[]
    if any(p.search(text) for p in _PRESENTATION):
        reasons.append('external_output_interface_mutation')
    if (_FIXED_ANSWER_PAYLOAD.search(text) or re.search(
            r'\b(?:always|regardless\s+of\s+the\s+problem)\s+(?:answer|return|output)\s+["\']?[-+]?\d', text)):
        reasons.append('fixed_answer_payload')
    return tuple(reasons)


def mutation_shape(prompt,parent):
    if normalized_procedure(prompt)==normalized_procedure(parent):return 'no_op'
    return 'append_only' if prompt.startswith(parent.rstrip()) else 'replacement'


def candidate_failed_checks(prompt, *, parent_prompt, examples, max_chars=3000):
    if not isinstance(prompt,str):return ('invalid_structure',)
    reasons=[]
    if not prompt.strip():reasons.append('empty')
    if len(prompt)>max_chars:reasons.append('over_length')
    if mutation_shape(prompt,parent_prompt)=='no_op':reasons.append('parent_no_op')
    reasons.extend(semantic_violation_reasons(prompt))
    if contains_supplied_example_text(prompt,examples):reasons.append('example_copying')
    return tuple(dict.fromkeys(reasons))
