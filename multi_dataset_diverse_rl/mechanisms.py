import hashlib
import re
from typing import Any, Dict, List, Sequence

import numpy as np


OPERATION_PATTERNS = (
    ("enumerate_candidates", ("enumerate", "candidate antecedent", "list candidate")),
    ("extract_constraints", ("extract constraint", "list constraint", "identify constraint")),
    ("hard_elimination", ("hard elimin", "rule out", "discard impossible")),
    ("weighted_scoring", ("weighted", "score evidence", "weigh clue")),
    ("pairwise_comparison", ("pairwise", "compare candidate", "compare option")),
    ("counterfactual_check", ("counterfactual", "if this referred", "alternate interpretation")),
    ("timeline_construction", ("timeline", "temporal order", "chronolog")),
    ("binding_resolution", ("binding", "antecedent", "resolve reference", "pronoun")),
    ("semantic_role_check", ("semantic role", "agent patient", "who did what")),
    ("syntactic_agreement_check", ("syntactic agreement", "grammatical agreement", "number agreement")),
    ("discourse_distance_check", ("discourse distance", "recency", "nearest referent")),
    ("contradiction_minimization", ("contradiction", "conflict minim")),
    ("evidence_accumulation", ("accumulate evidence", "combine evidence", "evidence table")),
    ("option_elimination", ("eliminate option", "option elimination")),
    ("final_consistency_check", ("final consistency", "consistency check", "verify conclusion")),
)

GENERIC_PROMPT_PATTERNS = (
    r"you are (?:a|an) [^.]+solver\. ?",
    r"produce (?:a )?compact[^.]*trace[.;]? ?",
    r"make your decision procedure visible[.;]? ?",
    r"verify (?:your answer|key logic)[.;]? ?",
    r"give exactly one final answer[.;]? ?",
    r"final_answer\s*:\s*<[^>]+>",
)


def _clean_step(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^\s*(?:step\s*)?\d+[.):\-]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:-")


def normalize_operation(step: Any) -> str:
    text = _clean_step(step)
    for operation, needles in OPERATION_PATTERNS:
        if any(needle in text for needle in needles):
            return operation
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_mechanism_representation(prompt: str, mechanism_steps: Sequence[Any]) -> Dict[str, Any]:
    cleaned_steps = [_clean_step(step) for step in mechanism_steps if _clean_step(step)]
    operations = [normalize_operation(step) for step in cleaned_steps]
    operations = [operation for operation in operations if operation]
    if not operations:
        residual = str(prompt or "").lower()
        for pattern in GENERIC_PROMPT_PATTERNS:
            residual = re.sub(pattern, " ", residual, flags=re.IGNORECASE)
        residual = re.sub(r"\s+", " ", residual).strip(" .;:-")
        if residual:
            operations = [normalize_operation(residual)]
            cleaned_steps = [residual]
    embedding_text = " ; ".join(operations)
    return {
        "normalized_operations": list(operations),
        "normalized_operation_sequence": list(operations),
        "normalized_mechanism_text": embedding_text,
        "mechanism_embedding_text": embedding_text,
        "mechanism_hash": hashlib.sha256(embedding_text.encode("utf-8")).hexdigest(),
    }


def levenshtein_sequence_distance(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = list(left), list(right)
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, av in enumerate(a, 1):
        current = [i]
        for j, bv in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + int(av != bv)))
        previous = current
    return float(previous[-1]) / float(max(len(a), len(b), 1))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 1.0 if list(left) == list(right) else 0.0
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def mechanism_distance(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    sequence_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> Dict[str, float]:
    sequence = levenshtein_sequence_distance(
        left.get("normalized_operation_sequence", []), right.get("normalized_operation_sequence", [])
    )
    embedding = float(np.clip(1.0 - cosine_similarity(
        left.get("mechanism_embedding", []), right.get("mechanism_embedding", [])
    ), 0.0, 1.0))
    total_weight = max(float(sequence_weight) + float(embedding_weight), 1e-12)
    combined = (float(sequence_weight) * sequence + float(embedding_weight) * embedding) / total_weight
    return {
        "sequence_distance": float(np.clip(sequence, 0.0, 1.0)),
        "embedding_distance": embedding,
        "mechanism_distance": float(np.clip(combined, 0.0, 1.0)),
    }


def mechanism_niche_key(representation: Dict[str, Any]) -> tuple:
    sequence = tuple(representation.get("normalized_operation_sequence", [])[:4])
    family = sequence[0] if sequence else "unknown"
    return family, sequence


def mechanisms_are_near_duplicate(left: Dict[str, Any], right: Dict[str, Any], threshold: float = 0.97) -> bool:
    left_sequence = list(left.get("normalized_operation_sequence", []))
    right_sequence = list(right.get("normalized_operation_sequence", []))
    return left_sequence == right_sequence and cosine_similarity(
        left.get("mechanism_embedding", []), right.get("mechanism_embedding", [])
    ) >= float(threshold)
