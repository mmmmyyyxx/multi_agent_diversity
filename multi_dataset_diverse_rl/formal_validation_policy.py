"""Frozen, post-search Formal Validation50 identity; never used by search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .governance.freeze_hash import source_freeze_sha256
from .governance.startup_identity import canonical_json_bytes


SPLIT_MANIFEST = "experiments/anti_overfitting_split_v1/split_manifest.json"
FOLD_ASSIGNMENT = "experiments/anti_overfitting_split_v1/fold_assignment.json"
VALIDATION_IDENTITY = "anti_overfitting_split_v1/validation"


def _question_hash_set_sha256(values: set[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(sorted(values)) + b"\n").hexdigest()


def formal_validation50_policy(root: Path) -> dict[str, Any]:
    """Bind the independent validation split and prove four-way disjointness."""
    manifest = json.loads((root / SPLIT_MANIFEST).read_text(encoding="utf-8"))
    folds = json.loads((root / FOLD_ASSIGNMENT).read_text(encoding="utf-8"))["folds"]
    if manifest.get("schema_version") != "anti_overfitting_split_manifest_v1":
        raise ValueError("Formal Validation50 split-manifest identity mismatch")
    groups = {
        "optimize100": [*folds["fold_a"], *folds["fold_b"]],
        "shadow50": list(folds["fold_c"]),
        "validation50": list(manifest["question_hashes"]["validation"]),
        "test50": list(manifest["question_hashes"]["test"]),
    }
    expected_counts = {"optimize100": 100, "shadow50": 50,
                       "validation50": 50, "test50": 50}
    sets: dict[str, set[str]] = {}
    for name, values in groups.items():
        if (len(values) != expected_counts[name] or len(set(values)) != len(values)
                or any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                       for value in values)):
            raise ValueError(f"Formal {name} question-hash inventory mismatch")
        sets[name] = set(values)
    if (sets["optimize100"] | sets["shadow50"]
            != set(manifest["question_hashes"]["train_dev"])
            or manifest.get("counts") != {"train_dev": 150, "validation": 50, "test": 50}):
        raise ValueError("Formal search split does not match split manifest")
    if any(sets[left] & sets[right] for index, left in enumerate(sets)
           for right in list(sets)[index + 1:]):
        raise ValueError("Formal Validation50, Shadow50, Optimize100, and Test50 must be disjoint")
    return {
        "schema_version": "formal_post_freeze_validation50_policy_v2",
        "split_identity": VALIDATION_IDENTITY,
        "split_manifest": SPLIT_MANIFEST,
        "split_manifest_schema": manifest["schema_version"],
        "split_manifest_sha256": source_freeze_sha256(root / SPLIT_MANIFEST),
        "fold_assignment_sha256": source_freeze_sha256(root / FOLD_ASSIGNMENT),
        "count": 50,
        "question_hash_set_sha256": _question_hash_set_sha256(sets["validation50"]),
        "question_hash_set_hash_semantics": "sorted_unique_canonical_json_utf8_lf_v1",
        "optimize100_question_hash_set_sha256": _question_hash_set_sha256(sets["optimize100"]),
        "shadow50_question_hash_set_sha256": _question_hash_set_sha256(sets["shadow50"]),
        "test50_question_hash_set_sha256": _question_hash_set_sha256(sets["test50"]),
        "shadow50_identity": "anti_overfitting_split_v1/fold_c",
        "search_access": "FORBIDDEN_ZERO_CALLS",
        "evaluation_phase": "POST_FREEZE_ONLY_SEPARATE_AUTHORIZATION",
        "test50_stage0": "SEALED_ZERO_CALLS",
    }


def verify_formal_validation50_policy(root: Path, candidate: Mapping[str, Any]) -> None:
    """Fail closed if a preregistration points Validation50 at Shadow50."""
    if dict(candidate) != formal_validation50_policy(root):
        raise ValueError("Formal post-freeze Validation50 policy mismatch")
