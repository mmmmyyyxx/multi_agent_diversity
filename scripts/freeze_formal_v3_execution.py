"""Durably freeze a completed Formal search cell before read-only derivation."""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.persistence.durable_io import (  # noqa: E402
    atomic_replace, io_path, read_json,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary  # noqa: E402


SCHEMA = "formal_v3_execution_evidence_freeze_v1"
FREEZE_NAME = "execution_evidence_freeze.json"
DERIVED_NAME = "formal_trajectory_trace.jsonl"
REQUIRED_ARTIFACTS = frozenset({
    "execution_summary.json", "run_lifecycle.json", "final_team_materialization.json",
    "ledger.jsonl",
})


def _digest(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with open(io_path(path), "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _run_files(run_root: Path) -> dict[str, Path]:
    """Enumerate all run artifacts through Windows-safe OS paths, rejecting links."""

    root = Path(io_path(run_root))
    if os.path.islink(io_path(run_root)):
        raise ValueError("Formal run root cannot be a symlink")
    found: dict[str, Path] = {}
    for directory, dirs, files in os.walk(io_path(run_root), followlinks=False):
        for name in dirs:
            if os.path.islink(os.path.join(directory, name)):
                raise ValueError("Formal run artifact cannot be a symlink")
        for name in files:
            path = Path(directory) / name
            if os.path.islink(io_path(path)):
                raise ValueError("Formal run artifact cannot be a symlink")
            relative = path.relative_to(root).as_posix()
            if relative in found:
                raise ValueError("Duplicate Formal run artifact path")
            found[relative] = path
    return found


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ntpath.splitdrive(value)[0]:
        raise ValueError("Unsafe Formal freeze path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in value.split("/")) or path.as_posix() != value:
        raise ValueError("Unsafe Formal freeze path")
    return value


def verify_freeze(run_root: Path, *, allow_derived: bool = True) -> dict[str, object]:
    """Replay the complete frozen inventory against every physical run artifact."""

    inventory = read_json(run_root / FREEZE_NAME)
    if not isinstance(inventory, dict) or inventory.get("schema_version") != SCHEMA:
        raise ValueError("Formal freeze schema mismatch")
    summary = read_json(run_root / "execution_summary.json")
    if inventory.get("attempt_id") != summary.get("experiment_id"):
        raise ValueError("Formal freeze attempt identity mismatch")
    rows = inventory.get("files")
    if (not isinstance(rows, list) or type(inventory.get("artifact_count")) is not int
            or inventory["artifact_count"] != len(rows)):
        raise ValueError("Formal freeze artifact count mismatch")
    listed: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed Formal freeze artifact")
        relative = _safe_relative(row.get("path"))
        sha = row.get("sha256")
        if (relative in listed or relative == FREEZE_NAME or relative == DERIVED_NAME
                or type(row.get("size_bytes")) is not int or row["size_bytes"] < 0
                or not isinstance(sha, str) or len(sha) != 64
                or any(char not in "0123456789abcdef" for char in sha)):
            raise ValueError("Malformed or duplicate Formal freeze artifact")
        listed[relative] = row
    if not REQUIRED_ARTIFACTS <= listed.keys():
        raise ValueError("Formal freeze lacks required artifact")
    actual = _run_files(run_root)
    permitted = set(listed) | {FREEZE_NAME}
    if allow_derived:
        permitted.add(DERIVED_NAME)
    if set(actual) != permitted and not (allow_derived and set(actual) == permitted - {DERIVED_NAME}):
        raise ValueError("Formal freeze inventory differs from physical run artifacts")
    for relative, row in listed.items():
        size, sha = _digest(actual[relative])
        if size != row["size_bytes"] or sha != row["sha256"]:
            raise ValueError(f"Formal freeze artifact hash mismatch: {relative}")
    return inventory


def freeze(run_root: Path) -> dict[str, object]:
    destination = run_root / FREEZE_NAME
    if os.path.lexists(io_path(destination)):
        raise FileExistsError("Formal execution is already frozen")
    lifecycle = read_json(run_root / "run_lifecycle.json")
    if lifecycle.get("status") != "EXECUTION_COMPLETE":
        raise ValueError("Formal execution must complete before raw freeze")
    summary = read_json(run_root / "execution_summary.json")
    if (lifecycle.get("attempt_id") != summary.get("experiment_id")
            or summary.get("validation50_calls") != 0 or summary.get("test50_calls") != 0
            or summary.get("mode_id") not in {"GEPA_NATIVE", "GEPA_LAYER2_V4"}):
        raise ValueError("Formal execution identity or search-only split mismatch")
    physical_ledger = ledger_summary(run_root / "ledger.jsonl")
    if (summary.get("ledger") != physical_ledger
            or lifecycle.get("provider_attempts") != physical_ledger["provider_attempts"]
            or lifecycle.get("provider_successes") != physical_ledger["successful_provider_calls"]
            or lifecycle.get("provider_failures") != physical_ledger["failed_provider_attempts"]):
        raise ValueError("Formal execution ledger/lifecycle mismatch")
    final_team = read_json(run_root / "final_team_materialization.json")
    if (final_team.get("mode_id") != summary["mode_id"]
            or final_team.get("initial_team_hash") != summary.get("initial_team_hash")
            or final_team.get("final_team_hash") != summary.get("final_team_materialization", {}).get("final_team_hash")
            or {key: value for key, value in final_team.items() if key != "prompts"}
            != summary.get("final_team_materialization")):
        raise ValueError("Formal final-team materialization mismatch")
    physical = _run_files(run_root)
    if FREEZE_NAME in physical or DERIVED_NAME in physical or not REQUIRED_ARTIFACTS <= physical.keys():
        raise ValueError("Formal raw freeze requires only pre-derived artifacts")
    files = []
    for relative, path in sorted(physical.items()):
        _safe_relative(relative)
        size, sha = _digest(path)
        files.append({"path": relative, "size_bytes": size, "sha256": sha})
    payload = {"schema_version": SCHEMA, "attempt_id": summary["experiment_id"],
               "files": files, "artifact_count": len(files)}
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    published = False
    try:
        with open(io_path(temporary), "x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(io_path(destination)):
            raise FileExistsError("Formal execution is already frozen")
        atomic_replace(temporary, destination)
        published = True
        if verify_freeze(run_root, allow_derived=False) != payload:
            raise ValueError("Formal freeze read-back mismatch")
        return {"attempt_id": payload["attempt_id"], "artifact_count": len(files),
                "freeze_sha256": _digest(destination)[1]}
    except BaseException:
        if published:
            os.unlink(io_path(destination))
        raise
    finally:
        if os.path.lexists(io_path(temporary)):
            os.unlink(io_path(temporary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args().run_root), sort_keys=True))
