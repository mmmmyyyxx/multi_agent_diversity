"""Shared bounded fixed-answer checks; no answer-presentation restrictions."""
import re
import unicodedata

_FIXED_ANSWER_PAYLOAD = re.compile(
    r'(?i)\b(?:always|regardless\s+of\s+(?:the\s+)?(?:problem|question))\s+'
    r'(?:answer|return|output|print)\s+["\']?(?:[-+]?\d|yes\b|no\b|true\b|false\b)'
    r'|\b(?:fixed\s+answer|answer\s+lookup|memorized\s+answers?)\b')

IDENTITY = 'STRUCTURED_SYSTEM_BLOCK_MUTATION_CONTRACT_V1'


def normalized_text(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def semantic_violation_reasons(prompt):
    text=normalized_text(prompt)
    reasons=[]
    if (_FIXED_ANSWER_PAYLOAD.search(text) or re.search(
            r'\b(?:always|regardless\s+of\s+the\s+problem)\s+(?:answer|return|output)\s+["\']?[-+]?\d', text)):
        reasons.append('fixed_answer_payload')
    return tuple(reasons)
