"""Post-search materialization of the frozen Formal five-member team."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .persistence.durable_io import atomic_write_json, read_json
from .utils import normalize_prompt_text


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt_text(prompt).encode("utf-8")).hexdigest()


def team_hash(prompts: Sequence[str]) -> str:
    if len(prompts) != 5 or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("Formal final team requires exactly five nonempty prompts")
    prompt_hashes = [_prompt_hash(prompt) for prompt in prompts]
    return hashlib.sha256(json.dumps(
        prompt_hashes, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def persist_final_team(
    path: Path, *, mode: str, initial_prompts: Sequence[str],
    final_prompts: Sequence[str], candidate_id: str | None,
    initial_team_hash: str, search_final_identity: str,
) -> dict[str, Any]:
    """Write private reconstruction evidence after search; never feed search."""
    if mode not in {"GEPA_NATIVE", "GEPA_LAYER2_V4"}:
        raise ValueError("unsupported Formal final-team mode")
    if team_hash(initial_prompts) != initial_team_hash:
        raise ValueError("Formal initial team hash mismatch")
    if mode == "GEPA_NATIVE":
        if len(set(initial_prompts)) != 1:
            raise ValueError("Native control requires homogeneous initial team")
        if candidate_id is None:
            if tuple(final_prompts) != tuple(initial_prompts) or search_final_identity != initial_team_hash:
                raise ValueError("Native no-candidate final state mismatch")
        elif (len(set(final_prompts)) != 1 or search_final_identity != candidate_id):
            raise ValueError("Native selected candidate/final team mismatch")
    elif candidate_id is not None or team_hash(final_prompts) != search_final_identity:
        raise ValueError("Layer2 final team differs from committed state")
    payload = {
        "schema_version": "formal_v3_final_team_materialization_v1",
        "mode_id": mode,
        "rule": ("replicate_first_returned_native_candidate_to_all_five"
                 if mode == "GEPA_NATIVE" else "preserve_final_committed_five_member_team"),
        "initial_team_hash": initial_team_hash,
        "search_final_identity": search_final_identity,
        "selected_native_candidate_id": candidate_id,
        "final_team_hash": team_hash(final_prompts),
        "prompt_hashes": [_prompt_hash(prompt) for prompt in final_prompts],
        "prompts": list(final_prompts),
    }
    atomic_write_json(path, payload)
    if read_json(path) != payload:
        raise RuntimeError("Formal final team read-back mismatch")
    return {key: value for key, value in payload.items() if key != "prompts"}
