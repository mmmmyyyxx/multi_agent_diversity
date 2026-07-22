from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .candidate_selection import vote_first_key


PROMPT_MEMORY_SLOTS = (
    "active", "competence_best", "ensemble_best", "responsibility_best", "rollback",
)


def contribution_signature(
    *, fixed_probe_hash: str, question_hashes: Sequence[str], answer_hashes: Sequence[str],
    correctness_vector: Sequence[bool], invalid_vector: Sequence[bool], vote_contribution_vector: Sequence[int],
    coverage_contribution_vector: Sequence[int], unique_correct_vector: Sequence[bool],
    pivotal_correct_vector: Sequence[bool], dominant_wrong_membership_vector: Sequence[bool],
) -> str:
    payload = {
        "fixed_probe_hash": fixed_probe_hash,
        "question_hashes": list(question_hashes),
        "answer_hashes": list(answer_hashes),
        "correctness_vector": list(correctness_vector),
        "invalid_vector": list(invalid_vector),
        "vote_contribution_vector": list(vote_contribution_vector),
        "coverage_contribution_vector": list(coverage_contribution_vector),
        "unique_correct_vector": list(unique_correct_vector),
        "pivotal_correct_vector": list(pivotal_correct_vector),
        "dominant_wrong_membership_vector": list(dominant_wrong_membership_vector),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def rebuild_prompt_memory(
    items: Sequence[Mapping[str, Any]], *, active_prompt_hash: str,
    previous_active_item: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    unique = {str(item.get("prompt_hash", "")): dict(item) for item in items if item.get("prompt_hash")}
    active = unique.get(str(active_prompt_hash))
    if active is None:
        raise ValueError("prompt memory requires the active prompt")
    feasible = [item for item in unique.values() if bool(item.get("metrics", {}).get("constraints_passed", False))]
    slots: list[tuple[str, Mapping[str, Any] | None]] = [
        ("active", active),
        ("competence_best", max(feasible, key=lambda item: (
            int(item.get("metrics", {}).get("candidate_target_correct_count", 0)), str(item.get("prompt_hash", "")),
        ), default=None)),
        ("ensemble_best", max(feasible, key=vote_first_key, default=None)),
        ("responsibility_best", max(feasible, key=lambda item: (
            float(item.get("metrics", {}).get("assigned_residual_utility_delta", 0.0)), str(item.get("prompt_hash", "")),
        ), default=None)),
        ("rollback", previous_active_item),
    ]
    memory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slot, item in slots:
        if item is None:
            continue
        prompt_hash = str(item.get("prompt_hash", ""))
        if not prompt_hash or prompt_hash in seen:
            continue
        row = dict(item)
        row["prompt_memory_slot"] = slot
        memory.append(row)
        seen.add(prompt_hash)
    return memory


def select_generation_parents(memory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(item) for item in memory
        if (
            str(item.get("prompt_memory_slot", "")) == "active"
            or bool(item.get("metrics", {}).get("constraints_passed", False))
        )
    ]
