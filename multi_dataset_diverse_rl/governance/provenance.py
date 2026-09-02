"""Sanitized, host-independent provenance construction."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from multi_dataset_diverse_rl import versions

from .manifest import preregistration_hash


DEFAULT_SOURCE_ROOTS = (
    "multi_dataset_diverse_rl",
    "scripts",
    "configs",
    "requirements.txt",
)


def _git(workspace: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=workspace, text=True, encoding="utf-8"
    ).strip()


def source_tree_hash(
    workspace: str | Path,
    source_roots: Iterable[str] = DEFAULT_SOURCE_ROOTS,
) -> str:
    root = Path(workspace).resolve()
    allowed = tuple(item.replace("\\", "/").rstrip("/") for item in source_roots)
    tracked = _git(root, "ls-files").splitlines()
    digest = hashlib.sha256()
    for relative in sorted(tracked):
        normalized = relative.replace("\\", "/")
        if not any(
            normalized == prefix or normalized.startswith(prefix + "/")
            for prefix in allowed
        ):
            continue
        path = root / relative
        digest.update(normalized.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def build_provenance(
    workspace: str | Path,
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    config_hashes: Mapping[str, str],
    timestamp: str | None = None,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    model = manifest.get("model", {})
    budget = manifest.get("budget", {})
    return {
        "schema_version": "experiment_provenance_v1",
        "git_head": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(status),
        "run_id": run_id,
        "method_version": versions.METHOD_VERSION,
        "checkpoint_version": versions.CHECKPOINT_VERSION,
        "config_hashes": dict(sorted(config_hashes.items())),
        "manifest_hash": preregistration_hash(manifest),
        "split_hashes": manifest.get("data", {}).get("split_hashes", {}),
        "seeds": manifest.get("seeds", []),
        "model_identifiers": {
            "solver": model.get("solver"),
            "optimizer_roles": model.get("optimizer_roles", {}),
        },
        "thinking": model.get("thinking"),
        "temperatures": model.get("temperatures", {}),
        "max_tokens": model.get("max_tokens", {}),
        "candidate_counts": budget.get("candidate_counts", {}),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": source_tree_hash(root),
    }
