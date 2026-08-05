from __future__ import annotations

import re
import unicodedata


OUTPUT_CONTRACT_CONTAMINATION = "output_contract_contamination"

_FINAL_ANSWER_MARKER = re.compile(r"(?<!\w)final[\s_-]*answer(?!\w)")
_COPIED_INTERFACE_MARKERS = (
    re.compile(r"(?<!\w)mandatory\s+output\s+interface(?!\w)"),
    re.compile(r"(?<!\w)solver\s+output\s+contract(?!\w)"),
    re.compile(
        r"(?<!\w)there\s+must\s+be\s+exactly\s+one\s+"
        r"final[\s_-]*answer\s+line(?!\w)"
    ),
)
_COPIED_FORMATTING_DIRECTIVES = (
    re.compile(
        r"(?<!\w)(?:output|return|format)\s+(?:the\s+)?(?:final\s+)?"
        r"answer\s+(?:as|in)(?!\w)"
    ),
    re.compile(
        r"(?<!\w)end\s+(?:your|the)\s+(?:response|output)\s+with(?!\w)"
    ),
)
_FIXED_ANSWER_PAYLOAD = re.compile(
    r"(?i)(?<!\w)(?:[\"']answer[\"']|answer|label)\s*(?::|=)\s*"
    r"(?:[\"']?(?:[a-dx]|yes|no|true|false|valid|invalid)[\"']?|<[^>]+>)"
    r"(?!\w)|(?<!\w)(?:answer|label)\s+is\s+"
    r"(?:[a-dx]|yes|no|true|false)(?!\w)"
)


def _normalized_for_contract_check(prompt: str) -> str:
    text = unicodedata.normalize("NFKC", str(prompt or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.casefold()


def mutable_prompt_violation_reasons(prompt: str) -> tuple[str, ...]:
    """Return stable internal categories for immutable-interface contamination."""

    normalized = _normalized_for_contract_check(prompt)
    reasons: list[str] = []
    if _FINAL_ANSWER_MARKER.search(normalized):
        reasons.append("forbidden_final_answer_marker")
    if any(pattern.search(normalized) for pattern in _COPIED_INTERFACE_MARKERS):
        reasons.append("copied_solver_interface")
    if any(pattern.search(normalized) for pattern in _COPIED_FORMATTING_DIRECTIVES):
        reasons.append("copied_solver_interface")
    if _FIXED_ANSWER_PAYLOAD.search(normalized):
        reasons.append("fixed_answer_payload")
    return tuple(dict.fromkeys(reasons))


def validate_mutable_decision_procedure(prompt: str) -> None:
    """Fail closed when mutable prompt text contains the Solver output interface."""

    if mutable_prompt_violation_reasons(prompt):
        raise ValueError(
            "mutable_prompt_contract_violation: "
            f"{OUTPUT_CONTRACT_CONTAMINATION}"
        )
