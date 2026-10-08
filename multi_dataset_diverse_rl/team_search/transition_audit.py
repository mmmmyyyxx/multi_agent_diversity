"""Shared correctness-transition value type; no historical search imports."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Mapping
import hashlib,json
def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CandidateTransitionAudit:
    """Sanitized one-step parent-to-child correctness transition."""

    parent_candidate_hash: str
    child_candidate_hash: str
    parent_correctness: tuple[tuple[str, bool], ...]
    child_correctness: tuple[tuple[str, bool], ...]
    newly_fixed_ids: tuple[str, ...] = field(init=False)
    newly_broken_ids: tuple[str, ...] = field(init=False)
    unchanged_correct_count: int = field(init=False)
    unchanged_wrong_count: int = field(init=False)
    parent_profile_hash: str = field(init=False)
    child_profile_hash: str = field(init=False)
    transition_effect_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.parent_candidate_hash or not self.child_candidate_hash:
            raise ValueError("transition candidate hashes are required")
        parent = dict(self.parent_correctness)
        child = dict(self.child_correctness)
        if (
            not parent
            or set(parent) != set(child)
            or len(parent) != len(self.parent_correctness)
            or len(child) != len(self.child_correctness)
        ):
            raise ValueError("transition profiles require identical unique example ids")
        ordered_ids = sorted(parent)
        fixed = tuple(row_id for row_id in ordered_ids if not parent[row_id] and child[row_id])
        broken = tuple(row_id for row_id in ordered_ids if parent[row_id] and not child[row_id])
        unchanged_correct = sum(parent[row_id] and child[row_id] for row_id in ordered_ids)
        unchanged_wrong = sum(not parent[row_id] and not child[row_id] for row_id in ordered_ids)
        parent_payload = [[row_id, bool(parent[row_id])] for row_id in ordered_ids]
        child_payload = [[row_id, bool(child[row_id])] for row_id in ordered_ids]
        parent_hash = _stable_hash(parent_payload)
        child_hash = _stable_hash(child_payload)
        effect_payload = {
            "parent_candidate_hash": self.parent_candidate_hash,
            "child_candidate_hash": self.child_candidate_hash,
            "parent_profile_hash": parent_hash,
            "child_profile_hash": child_hash,
            "newly_fixed_ids": list(fixed),
            "newly_broken_ids": list(broken),
            "unchanged_correct_count": unchanged_correct,
            "unchanged_wrong_count": unchanged_wrong,
        }
        object.__setattr__(self, "newly_fixed_ids", fixed)
        object.__setattr__(self, "newly_broken_ids", broken)
        object.__setattr__(self, "unchanged_correct_count", unchanged_correct)
        object.__setattr__(self, "unchanged_wrong_count", unchanged_wrong)
        object.__setattr__(self, "parent_profile_hash", parent_hash)
        object.__setattr__(self, "child_profile_hash", child_hash)
        object.__setattr__(self, "transition_effect_hash", _stable_hash(effect_payload))

    def sanitized_payload(self) -> Mapping[str, Any]:
        return {
            "parent_candidate_hash": self.parent_candidate_hash,
            "child_candidate_hash": self.child_candidate_hash,
            "newly_fixed_ids": list(self.newly_fixed_ids),
            "newly_broken_ids": list(self.newly_broken_ids),
            "unchanged_correct_count": self.unchanged_correct_count,
            "unchanged_wrong_count": self.unchanged_wrong_count,
            "parent_profile_hash": self.parent_profile_hash,
            "child_profile_hash": self.child_profile_hash,
            "transition_effect_hash": self.transition_effect_hash,
        }
