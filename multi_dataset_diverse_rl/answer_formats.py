import json
import re
from typing import Any, List


SUPPORTED_ANSWER_FORMATS = {
    "option_letter",
    "boolean",
    "yes_no",
    "valid_invalid",
    "numeric",
    "free_text",
}


def _strip_prefix(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*(?:FINAL_ANSWER|Final answer|final answer|Answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*The answer is\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def _last_marked_or_line(raw_output: Any) -> str:
    text = str(raw_output or "").strip()
    if not text:
        return ""
    matches = re.findall(r"(?:FINAL_ANSWER|Final answer|final answer|Answer|answer)\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def _normalize_number(value: str) -> str:
    text = str(value or "").replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return re.sub(r"\s+", " ", text).strip().lower()
    raw = match.group(0)
    try:
        number = float(raw)
        if abs(number - int(number)) < 1e-9:
            return str(int(number))
        return ("%f" % number).rstrip("0").rstrip(".")
    except Exception:
        return raw


def _parse_aliases(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if not isinstance(value, str):
        return []
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []
    return []


def canonical_answer(value: Any, answer_format: str) -> str:
    fmt = str(answer_format or "").strip().lower()
    text = _strip_prefix(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.。]\s*$", "", text).strip()

    if fmt == "option_letter":
        match = re.search(r"(?:^|\b)(?:option|choice)\s*([A-Z])(?:\b|$)", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"^\(?\s*([A-Z])\s*\)?(?:[.)])?$", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"\b([A-Z])\b", text.upper())
        return match.group(1).upper() if match else text.upper()

    lowered = text.lower()
    if fmt == "boolean":
        if re.search(r"\btrue\b", lowered):
            return "true"
        if re.search(r"\bfalse\b", lowered):
            return "false"
        return lowered

    if fmt == "yes_no":
        if re.search(r"\byes\b", lowered):
            return "yes"
        if re.search(r"\bno\b", lowered):
            return "no"
        return lowered

    if fmt == "valid_invalid":
        if re.search(r"\binvalid\b", lowered):
            return "invalid"
        if re.search(r"\bvalid\b", lowered):
            return "valid"
        return lowered

    if fmt == "numeric":
        return _normalize_number(text)

    return lowered


def extract_prediction(raw_output: Any, answer_format: str) -> str:
    return canonical_answer(_last_marked_or_line(raw_output), answer_format)


def match_answer(pred: Any, gold: Any, answer_format: str) -> bool:
    fmt = str(answer_format or "").strip().lower()
    pred_norm = canonical_answer(pred, fmt)
    gold_aliases = _parse_aliases(gold)
    if gold_aliases:
        return any(pred_norm == canonical_answer(alias, fmt) for alias in gold_aliases)
    gold_norm = canonical_answer(gold, fmt)
    if fmt == "numeric":
        try:
            return abs(float(pred_norm) - float(gold_norm)) < 1e-9
        except Exception:
            return pred_norm == gold_norm
    return pred_norm == gold_norm
