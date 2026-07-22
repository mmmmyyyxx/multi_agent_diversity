from __future__ import annotations

import re

from ..tasks import TaskSpec
from .prompt_question import PromptAnswer


FINAL_ANSWER_LINE = re.compile(
    r"^\s*FINAL_ANSWER\s*:\s*(.*?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _in_domain(answer: str, question: str, task_spec: TaskSpec, answer_format: str) -> bool:
    value = str(answer or "").strip()
    if not value:
        return False
    fmt = str(answer_format or "").strip().lower()
    lowered = value.lower()
    if fmt == "option_letter":
        if not re.fullmatch(r"[A-Z]", value.upper()):
            return False
        count = task_spec.option_count(question) if task_spec.option_count else 0
        return not count or ord(value.upper()) - ord("A") < count
    if fmt == "boolean":
        return lowered in {"true", "false", "yes", "no"}
    if fmt == "yes_no":
        return lowered in {"yes", "no"}
    if fmt == "valid_invalid":
        return lowered in {"valid", "invalid"}
    if fmt == "numeric":
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.replace(",", "")))
    if fmt == "free_text":
        return True

    if task_spec.name == "mmlu":
        return value.upper() in {"A", "B", "C", "D"}
    if task_spec.name == "gsm8k":
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.replace(",", "")))
    count = task_spec.option_count(question) if task_spec.option_count else 0
    if count and re.fullmatch(r"[A-Z]", value.upper()):
        return ord(value.upper()) - ord("A") < count
    return True


def _raw_payload_in_domain(payload: str, question: str, task_spec: TaskSpec, answer_format: str) -> bool:
    """Reject trailing prose that a permissive task parser could silently discard."""
    value = str(payload or "").strip()
    fmt = str(answer_format or "").strip().lower()
    if fmt == "option_letter" or (not fmt and task_spec.name == "mmlu"):
        if not re.fullmatch(r"(?:\(\s*)?[A-Z](?:\s*\))?", value, flags=re.IGNORECASE):
            return False
        count = task_spec.option_count(question) if task_spec.option_count else 0
        letter = re.search(r"[A-Z]", value, flags=re.IGNORECASE)
        return bool(letter) and (not count or ord(letter.group(0).upper()) - ord("A") < count)
    if fmt == "boolean":
        return value.lower() in {"true", "false", "yes", "no"}
    if fmt == "yes_no":
        return value.lower() in {"yes", "no"}
    if fmt == "valid_invalid":
        return value.lower() in {"valid", "invalid"}
    if fmt == "numeric" or (not fmt and task_spec.name == "gsm8k"):
        return bool(re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", value))
    return bool(value)


def parse_solver_output(
    text: str,
    *,
    question: str,
    task_spec: TaskSpec,
    answer_format: str,
) -> PromptAnswer:
    raw = str(text or "")
    matches = FINAL_ANSWER_LINE.findall(raw)
    if not matches:
        return PromptAnswer("", raw, False, "missing_final_answer")
    if len(matches) != 1:
        return PromptAnswer("", raw, False, "multiple_final_answers")
    if not matches[0].strip():
        return PromptAnswer("", raw, False, "unparseable_final_answer")
    raw_payload = matches[0].strip()
    if not _raw_payload_in_domain(raw_payload, question, task_spec, answer_format):
        return PromptAnswer("", raw, False, "out_of_domain_answer")
    final_line = f"FINAL_ANSWER: {raw_payload}"
    answer = task_spec.extract_pred(final_line, question)
    if not answer:
        return PromptAnswer("", raw, False, "unparseable_final_answer")
    if not _in_domain(answer, question, task_spec, answer_format):
        return PromptAnswer(answer, raw, False, "out_of_domain_answer")
    return PromptAnswer(answer, raw, True, "valid")
