"""Shared supplied-example text guard, without optimizer runtime imports."""
import re
from typing import Sequence
from .schemas import LocalEvidenceExample

def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))

def contains_supplied_example_text(prompt: str, examples: Sequence[LocalEvidenceExample]) -> bool:
    """Reject long verbatim example fragments without importing team TCS code."""

    normalized_prompt = " ".join(_normalized_tokens(prompt))
    for example in examples:
        source = " ".join(_normalized_tokens(example.input_payload))
        words = source.split()
        for width in (12, 10):
            if len(words) >= width and any(
                " ".join(words[index : index + width]) in normalized_prompt
                for index in range(len(words) - width + 1)
            ):
                return True
    return False
