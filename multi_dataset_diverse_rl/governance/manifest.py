"""Schema and lifecycle validation for experiment manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import yaml

from multi_dataset_diverse_rl import versions


MANIFEST_STATUSES = (
    "DRAFT",
    "PREREGISTERED",
    "IMPLEMENTED",
    "PREFLIGHT_PASS",
    "RUNNING",
    "TRAIN_FROZEN",
    "VALIDATED",
    "COMPLETED_PRE_TEST",
    "TESTED",
    "COMPLETED",
)

_ALLOWED_TRANSITIONS = {
    "DRAFT": {"PREREGISTERED"},
    "PREREGISTERED": {"IMPLEMENTED"},
    "IMPLEMENTED": {"PREFLIGHT_PASS"},
    "PREFLIGHT_PASS": {"RUNNING"},
    "RUNNING": {"TRAIN_FROZEN"},
    "TRAIN_FROZEN": {"VALIDATED", "COMPLETED_PRE_TEST", "COMPLETED"},
    "VALIDATED": {"COMPLETED_PRE_TEST", "COMPLETED"},
    "COMPLETED_PRE_TEST": {"TESTED"},
    "TESTED": {"COMPLETED"},
    "COMPLETED": set(),
}


def load_manifest(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("experiment manifest must be a mapping")
    return data


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def preregistration_hash(value: Mapping[str, Any]) -> str:
    """Hash frozen design facts independently from mutable result fields."""
    frozen = {
        key: value.get(key)
        for key in (
            "experiment_id",
            "lineage",
            "scientific_question",
            "hypotheses",
            "design",
            "method_identity",
            "runtime_version",
            "data",
            "model",
            "seeds",
            "budget",
            "api_authorization",
            "selection",
        )
    }
    return hashlib.sha256(canonical_json(frozen)).hexdigest()


def _validate_lifecycle(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    history = manifest.get("lifecycle_history", [])
    if manifest.get("legacy_index"):
        return errors

    method_identity = manifest.get("method_identity")
    if method_identity is not None and method_identity != versions.METHOD_VERSION:
        errors.append("runtime method identity does not match versions.py")
    if not isinstance(history, list) or not history:
        return ["lifecycle_history must be non-empty"]
    statuses = [row.get("status") for row in history if isinstance(row, dict)]
    if len(statuses) != len(history):
        errors.append("every lifecycle_history row must be a mapping with status")
        return errors
    if statuses[0] != "DRAFT":
        errors.append("lifecycle must start at DRAFT")
    if statuses[-1] != manifest.get("status"):
        errors.append("lifecycle final status must equal manifest status")
    for current, following in zip(statuses, statuses[1:]):
        if following not in _ALLOWED_TRANSITIONS.get(current, set()):
            errors.append(f"illegal lifecycle transition: {current} -> {following}")
    return errors


def validate_manifest(
    manifest: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[str]:
    errors = [
        error.message
        for error in sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(manifest),
            key=lambda item: list(item.path),
        )
    ]
    errors.extend(_validate_lifecycle(manifest))

    for key in ("design_commit", "implementation_commit", "result_commit"):
        value = manifest.get("git", {}).get(key)
        if value is not None and not _is_commit(value):
            errors.append(f"git.{key} must be a full 40-character commit or null")

    if manifest.get("legacy_index"):
        return errors

    status = manifest.get("status")
    status_index = MANIFEST_STATUSES.index(status) if status in MANIFEST_STATUSES else -1
    running_index = MANIFEST_STATUSES.index("RUNNING")
    if status_index >= running_index and manifest.get("api_authorization", {}).get("authorized"):
        prereg = manifest.get("artifacts", {}).get("preregistration", {})
        if not isinstance(prereg, dict) or not _is_sha256(prereg.get("sha256")):
            errors.append("API run at RUNNING or later requires preregistration sha256")
        if not _is_commit(manifest.get("git", {}).get("implementation_commit")):
            errors.append("API run at RUNNING or later requires implementation commit")
        if manifest.get("budget", {}).get("frozen_before_run") is not True:
            errors.append("API run at RUNNING or later requires frozen budget")

    if manifest.get("data", {}).get("formal"):
        split_hashes = manifest.get("data", {}).get("split_hashes")
        if not isinstance(split_hashes, dict) or not split_hashes:
            errors.append("formal experiment requires split hashes")

    prereg = manifest.get("artifacts", {}).get("preregistration")
    if isinstance(prereg, dict) and prereg.get("sha256"):
        if prereg.get("sha256") != preregistration_hash(manifest):
            errors.append("frozen preregistration hash does not match manifest design facts")

    for key in ("validation_policy", "test_policy"):
        if not manifest.get("data", {}).get(key):
            errors.append(f"data.{key} must be explicit")
    if not manifest.get("selection", {}).get("frozen_rule"):
        errors.append("selection.frozen_rule must be explicit")
    if manifest.get("selection", {}).get("test_used_for_selection") is not False:
        errors.append("test data cannot be used for selection")
    return errors


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )
