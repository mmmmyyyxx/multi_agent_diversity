from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def load_jsonl(path: str, limit: int = -1) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
                if limit > 0 and len(rows) >= limit:
                    break
        return rows
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_prompt_text(value: str) -> str:
    """Canonicalize transport whitespace without flattening prompt structure."""
    lines = str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.rstrip() for line in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_json_obj(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    candidates = [value] if value.startswith("{") and value.endswith("}") else []
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", value, flags=re.DOTALL | re.IGNORECASE))
    match = re.search(r"(\{.*\})", value, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def plurality_vote_with_diagnostics(
    answers: list[str], *, tie_break_method: str = "abstain", seed: int = 0, question_hash: str = "",
) -> dict[str, Any]:
    cleaned = [str(answer) for answer in answers if str(answer).strip()]
    method = str(tie_break_method).lower()
    if method not in {"first", "random", "abstain"}:
        raise ValueError(f"Unknown plurality tie-break method: {tie_break_method}")
    if not cleaned:
        return {
            "vote_answer": "", "vote_tie": False, "tie_candidates": [],
            "vote_counts": {}, "tie_break_method": method,
        }
    counts = Counter(cleaned)
    top_count = max(counts.values())
    candidates = sorted(answer for answer, count in counts.items() if count == top_count)
    tied = len(candidates) > 1
    if not tied:
        vote_answer = candidates[0]
    elif method == "abstain":
        vote_answer = ""
    elif method == "first":
        vote_answer = next(answer for answer in cleaned if answer in candidates)
    else:
        vote_answer = random.Random(f"{seed}:{question_hash}:{'|'.join(candidates)}").choice(candidates)
    return {
        "vote_answer": vote_answer,
        "vote_tie": tied,
        "tie_candidates": candidates,
        "vote_counts": dict(counts),
        "tie_break_method": method,
    }
