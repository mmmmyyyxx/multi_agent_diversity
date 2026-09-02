"""Shared report-contract, sanitization, and SHA256 utilities."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


STANDARD_REPORT_FILES = {
    "README.md",
    "manifest_snapshot.yaml",
    "provenance.json",
    "summary.json",
    "classifier.json",
    "fact_assertions.json",
    "api_ledger_summary.json",
    "evaluation_access_summary.json",
    "sanitization_manifest.json",
    "sha256_manifest.json",
}

FORBIDDEN_KEY_FRAGMENTS = {
    "api_key",
    "credential",
    "endpoint_secret",
    "raw_prompt",
    "prompt_text",
    "question_text",
    "gold_answer",
    "model_answer",
    "raw_response",
    "sqlite_content",
    "checkpoint_content",
}

_WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[\s\"'=])(?:[a-z]:[\\/])")
_POSIX_USER_ABSOLUTE = re.compile(r"(?:^|[\s\"'=])/(?:home|Users|root)/")
_SECRET_VALUE = re.compile(r"(?i)(?:api[_-]?key|credential|secret)\s*[:=]\s*[\"']?[^\s\"']{8,}")
_RAW_LABEL = re.compile(
    r"(?im)^\s*(?:raw_prompt|prompt_text|question_text|gold_answer|model_answer|raw_response)\s*[:=,]"
)


def _walk_values(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield from _walk_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_values(child, f"{path}[{index}]")
    else:
        yield path, value


def scan_sanitized_artifacts(root: str | Path) -> list[dict[str, str]]:
    directory = Path(root)
    findings: list[dict[str, str]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if path.suffix.lower() in {".sqlite", ".db", ".ckpt", ".checkpoint"}:
            findings.append({"file": relative, "reason": "forbidden_binary_artifact"})
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"file": relative, "reason": "non_utf8_or_binary"})
            continue
        if _WINDOWS_ABSOLUTE.search(text) or _POSIX_USER_ABSOLUTE.search(text):
            findings.append({"file": relative, "reason": "absolute_path"})
        if _SECRET_VALUE.search(text):
            findings.append({"file": relative, "reason": "possible_secret_value"})
        if _RAW_LABEL.search(text):
            findings.append({"file": relative, "reason": "raw_content_label"})
        if path.suffix.lower() in {".json", ".jsonl"}:
            records = []
            try:
                records = (
                    [json.loads(line) for line in text.splitlines() if line.strip()]
                    if path.suffix.lower() == ".jsonl"
                    else [json.loads(text)]
                )
            except json.JSONDecodeError:
                findings.append({"file": relative, "reason": "invalid_json"})
                continue
            for record in records:
                for field_path, value in _walk_values(record):
                    if isinstance(value, str) and field_path.rsplit(".", 1)[-1].lower() in FORBIDDEN_KEY_FRAGMENTS:
                        findings.append({"file": relative, "reason": f"forbidden_field:{field_path}"})
    return sorted(findings, key=lambda row: (row["file"], row["reason"]))


def build_sha256_manifest(root: str | Path) -> dict[str, Any]:
    directory = Path(root)
    files = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative == "sha256_manifest.json" or relative.startswith("runs/"):
            continue
        files.append(
            {
                "file": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {"algorithm": "sha256", "files": files}


def snapshot_tree_hashes(root: str | Path) -> dict[str, str]:
    """Snapshot an evidence directory so later tooling can prove it was read-only."""
    manifest = build_sha256_manifest(root)
    return {row["file"]: row["sha256"] for row in manifest["files"]}


def compare_tree_snapshot(root: str | Path, frozen: dict[str, str]) -> list[str]:
    current = snapshot_tree_hashes(root)
    changed = sorted(
        path for path in set(frozen) | set(current) if frozen.get(path) != current.get(path)
    )
    return changed


def validate_report_contract(
    root: str | Path,
    *,
    optimization: bool = False,
    validation: bool = False,
) -> list[str]:
    directory = Path(root)
    required = set(STANDARD_REPORT_FILES)
    if optimization:
        required.add("funnel_summary.json")
    if validation:
        if not any(directory.glob("validation_results.*")):
            required.add("validation_results.*")
    present = {path.name for path in directory.iterdir()} if directory.exists() else set()
    return sorted(required - present)


def validate_fact_assertions(assertions: dict[str, Any]) -> list[str]:
    """Return the names of explicit machine assertions that are not true."""
    return sorted(key for key, value in assertions.items() if value is not True)
