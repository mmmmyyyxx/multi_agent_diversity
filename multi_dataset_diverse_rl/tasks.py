import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class TaskSpec:
    name: str
    parse_gold: Callable[[Any, Optional[str]], str]
    extract_pred: Callable[[Optional[str], Optional[str]], str]
    match_answer: Callable[[str, str], bool]
    format_question: Optional[Callable[[dict], str]] = None


def normalize_spaces(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_all_numbers(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text))


def canonical_number_str(value: Any) -> str:
    raw = str(value or "").replace(",", "").strip()
    try:
        number = float(raw)
        if abs(number - int(number)) < 1e-9:
            return str(int(number))
        return ("%f" % number).rstrip("0").rstrip(".")
    except Exception:
        return normalize_spaces(value)


def numeric_equal(left: str, right: str) -> bool:
    try:
        return abs(float(str(left).replace(",", "")) - float(str(right).replace(",", ""))) < 1e-9
    except Exception:
        return False


def parse_gsm8k_gold(answer: Any, question: Optional[str] = None) -> str:
    text = str(answer or "")
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return canonical_number_str(match.group(1))
    numbers = extract_all_numbers(text)
    if numbers:
        return canonical_number_str(numbers[-1])
    return normalize_spaces(text)


def extract_pred_answer_gsm8k(text: Optional[str], question: Optional[str] = None) -> str:
    if text is None:
        return ""
    raw = str(text).replace(",", "")
    patterns = [
        r"FINAL_ANSWER\s*:\s*([-+]?\d+(?:\.\d+)?)",
        r"Answer\s*:\s*([-+]?\d+(?:\.\d+)?)",
        r"The answer is\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return canonical_number_str(match.group(1))
    numbers = extract_all_numbers(raw)
    if numbers:
        return canonical_number_str(numbers[-1])
    return normalize_spaces(text)


def match_gsm8k_answer(pred: str, gold: str) -> bool:
    pred_norm = canonical_number_str(pred)
    gold_norm = canonical_number_str(gold)
    return pred_norm == gold_norm or numeric_equal(pred_norm, gold_norm)


def parse_mmlu_gold(answer: Any, question: Optional[str] = None) -> str:
    text = normalize_spaces(answer).upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([0-3])\b", text)
    if match:
        return ["A", "B", "C", "D"][int(match.group(1))]
    return text


def extract_pred_answer_mmlu(text: Optional[str], question: Optional[str] = None) -> str:
    if text is None:
        return ""
    raw = str(text)
    patterns = [
        r"FINAL_ANSWER\s*:\s*\(?([ABCD])\)?\b",
        r"Answer\s*:\s*\(?([ABCD])\)?\b",
        r"The answer is\s*\(?([ABCD])\)?\b",
        r"\boption\s*([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    tokens = re.findall(r"\b([ABCD])\b", raw.upper())
    if tokens:
        return tokens[-1]
    return normalize_spaces(raw).upper()


def match_mmlu_answer(pred: str, gold: str) -> bool:
    return parse_mmlu_gold(pred) == parse_mmlu_gold(gold)


def _strip_answer_prefix(text: str) -> str:
    value = normalize_spaces(text)
    prefix_patterns = [
        r"^(?:FINAL_ANSWER|Final answer)\s*:\s*",
        r"^Answer\s*:\s*",
        r"^The answer is\s+",
    ]
    for pattern in prefix_patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
    return value


def normalize_bbh_answer(text: Any) -> str:
    value = _strip_answer_prefix(str(text or ""))
    value = value.strip()
    value = re.sub(r"[。\.]\s*$", "", value).strip()
    value = normalize_spaces(value).lower()

    option_match = re.fullmatch(r"\(?\s*([a-z])\s*\)?", value)
    if option_match:
        return option_match.group(1)
    option_match = re.fullmatch(r"(?:option|choice)\s+([a-z])", value)
    if option_match:
        return option_match.group(1)
    option_match = re.fullmatch(r"([a-z])\s*[\).]", value)
    if option_match:
        return option_match.group(1)

    aliases = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "correct": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "incorrect": "no",
    }
    if value in aliases:
        return aliases[value]

    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.0+)?", value):
        return canonical_number_str(value)
    if re.fullmatch(r"[-+]?\d[\d,]*\.\d+", value):
        return canonical_number_str(value)
    return value


def _parse_alias_list(value: Any) -> Optional[List[Any]]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(stripped)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except Exception:
            continue
    return None


def parse_gold_bbh(answer: Any, question: Optional[str] = None) -> str:
    aliases = _parse_alias_list(answer)
    if aliases is not None:
        normalized = [normalize_bbh_answer(item) for item in aliases]
        normalized = [item for item in normalized if item]
        return json.dumps(normalized, ensure_ascii=False)
    return normalize_bbh_answer(answer)


def _bbh_gold_aliases(gold: Any) -> List[str]:
    aliases = _parse_alias_list(gold)
    if aliases is not None:
        return [normalize_bbh_answer(item) for item in aliases if normalize_bbh_answer(item)]
    normalized = normalize_bbh_answer(gold)
    return [normalized] if normalized else []


def extract_pred_answer_bbh(text: Optional[str], question: Optional[str] = None) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    final_matches = re.findall(r"FINAL_ANSWER\s*:\s*(.+)", raw, flags=re.IGNORECASE)
    if final_matches:
        return normalize_bbh_answer(final_matches[-1][:200])
    answer_matches = re.findall(r"(?:^|\n)\s*Answer\s*:\s*(.+)", raw, flags=re.IGNORECASE)
    if answer_matches:
        return normalize_bbh_answer(answer_matches[-1][:200])
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    return normalize_bbh_answer(lines[-1][:200])


def match_bbh_answer(pred: str, gold: str) -> bool:
    pred_norm = normalize_bbh_answer(pred)
    gold_aliases = _bbh_gold_aliases(gold)
    if pred_norm in gold_aliases:
        return True
    yes_aliases = {"yes", "true"}
    no_aliases = {"no", "false"}
    if pred_norm in yes_aliases and any(alias in yes_aliases for alias in gold_aliases):
        return True
    if pred_norm in no_aliases and any(alias in no_aliases for alias in gold_aliases):
        return True
    return any(numeric_equal(pred_norm, alias) for alias in gold_aliases)


def infer_task_type(task_type: str = "auto", question: Optional[str] = None, answer: Optional[Any] = None) -> str:
    declared = str(task_type or "auto").strip().lower()
    if declared in {"gsm8k", "mmlu", "bbh"}:
        return declared

    question_text = normalize_spaces(question).upper()
    answer_text = normalize_spaces(answer).upper()
    if answer_text in {"A", "B", "C", "D"}:
        return "mmlu"
    if re.search(r"\bOPTIONS\b", question_text) and re.search(r"\bA\.|\bB\.|\bC\.|\bD\.", question_text):
        return "mmlu"
    if "####" in str(answer or ""):
        return "gsm8k"
    if answer_text and re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", answer_text):
        return "gsm8k"
    if any(marker in question_text for marker in ["BIG-BENCH", "BBH", "BOOLEAN EXPRESSIONS", "DATE UNDERSTANDING"]):
        return "bbh"
    return "bbh"


def _auto_parse_gold(answer: Any, question: Optional[str] = None) -> str:
    task = infer_task_type("auto", question=question, answer=answer)
    return get_task_spec(task).parse_gold(answer, question)


def _auto_extract_pred(text: Optional[str], question: Optional[str] = None) -> str:
    task = infer_task_type("auto", question=question, answer=None)
    return get_task_spec(task).extract_pred(text, question)


def _auto_match_answer(pred: str, gold: str) -> bool:
    if match_mmlu_answer(pred, gold):
        return True
    if match_gsm8k_answer(pred, gold):
        return True
    return match_bbh_answer(pred, gold)


TASK_SPECS: Dict[str, TaskSpec] = {
    "mmlu": TaskSpec("mmlu", parse_mmlu_gold, extract_pred_answer_mmlu, match_mmlu_answer),
    "gsm8k": TaskSpec("gsm8k", parse_gsm8k_gold, extract_pred_answer_gsm8k, match_gsm8k_answer),
    "bbh": TaskSpec("bbh", parse_gold_bbh, extract_pred_answer_bbh, match_bbh_answer),
    "auto": TaskSpec("auto", _auto_parse_gold, _auto_extract_pred, _auto_match_answer),
}


def get_task_spec(task_type: str = "auto") -> TaskSpec:
    key = str(task_type or "auto").strip().lower()
    if key not in TASK_SPECS:
        key = "auto"
    return TASK_SPECS[key]
