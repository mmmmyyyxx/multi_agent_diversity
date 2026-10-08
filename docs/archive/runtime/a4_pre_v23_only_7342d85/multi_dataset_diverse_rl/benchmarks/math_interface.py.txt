"""Immutable response formatting, independent of every mutable search prompt."""
from __future__ import annotations

import hashlib

from .. import versions


MATH_SOLVER_INTERFACE_V2 = (
    "Reasoning may precede the answer. Your response must end with exactly one "
    "final-answer line:\nFINAL_ANSWER: <answer>\n"
    "The answer payload must be nonempty. Do not include any other final-answer lines."
)


def solver_interface_contract():
    return dict(identity=versions.MATH_SOLVER_INTERFACE_VERSION,
                sha256=hashlib.sha256(MATH_SOLVER_INTERFACE_V2.encode()).hexdigest(),
                parser_identity="math_verify_no_fallback_v1")
