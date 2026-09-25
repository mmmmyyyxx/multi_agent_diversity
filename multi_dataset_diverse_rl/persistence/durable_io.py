"""Windows-safe, durable primitives for current production artifacts.

The extended-length prefix is applied only at the OS boundary; logical paths
and persisted identities remain platform-independent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
from typing import Any


def io_path(path: str | Path) -> str:
    """Return a native path that does not depend on Windows MAX_PATH policy."""

    value = str(path)
    if os.name != "nt":
        return value
    if value.startswith("\\\\?\\"):
        return value
    absolute = str(Path(value).resolve())
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def ensure_directory(path: str | Path) -> None:
    os.makedirs(io_path(path), exist_ok=True)


def atomic_replace(source: str | Path, destination: str | Path) -> None:
    os.replace(io_path(source), io_path(destination))


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write, fsync, and atomically replace one JSON artifact."""

    target = Path(path)
    ensure_directory(target.parent)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(io_path(temporary), "x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(temporary, target)
    except BaseException:
        try:
            os.unlink(io_path(temporary))
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path) -> Any:
    with open(io_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    with open(io_path(target), "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
