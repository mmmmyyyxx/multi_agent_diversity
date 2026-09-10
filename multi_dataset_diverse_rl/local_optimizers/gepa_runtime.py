"""Resolve and verify the single frozen official GEPA checkout."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


GEPA_VERSION = "v0.1.1"
GEPA_COMMIT = "b4dbb55b7601dac448cdb836d5a401ca7d9eb920"
GEPA_SOURCE_SHA256 = "84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8"
GEPA_ROOT_ENV = "RG_GEPA_VENDOR_ROOT"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def frozen_gepa_root() -> Path:
    configured = os.environ.get(GEPA_ROOT_ENV)
    if configured:
        return Path(configured).resolve()
    return (_repository_root().parent / "independent_gepa_repro" / "vendor" / "gepa").resolve()


def source_bundle_sha256(root: Path) -> str:
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--", "pyproject.toml", "src/gepa"]
    )
    relative_paths = [value.decode("utf-8") for value in tracked.split(b"\0") if value]
    digest = hashlib.sha256()
    for relative_text in sorted(relative_paths):
        path = root / relative_text
        relative = relative_text.encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def verify_frozen_gepa(root: Path | None = None) -> dict[str, str]:
    checkout = (root or frozen_gepa_root()).resolve()
    source = checkout / "src" / "gepa"
    if not source.is_dir():
        raise RuntimeError("frozen GEPA source directory is missing")
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    if commit != GEPA_COMMIT:
        raise RuntimeError("frozen GEPA commit mismatch")
    source_hash = source_bundle_sha256(checkout)
    if source_hash != GEPA_SOURCE_SHA256:
        raise RuntimeError("frozen GEPA source bundle mismatch")
    return {"version": GEPA_VERSION, "commit": commit, "source_sha256": source_hash}


def import_frozen_gepa() -> Any:
    checkout = frozen_gepa_root()
    verify_frozen_gepa(checkout)
    source = str((checkout / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("gepa")
    resolved = Path(module.__file__).resolve()
    if (checkout / "src").resolve() not in resolved.parents:
        raise RuntimeError("GEPA resolved outside the frozen checkout")
    return module
