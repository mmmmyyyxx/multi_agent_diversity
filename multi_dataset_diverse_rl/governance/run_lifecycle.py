"""Durable terminal-state handling for experiment run ledgers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


async def run_with_lifecycle(
    path: Path,
    *,
    identity: dict[str, Any],
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Run one operation and always replace RUNNING with a terminal marker."""

    if path.exists():
        raise FileExistsError("run lifecycle path must be fresh")
    started = {**identity, "status": "RUNNING", "history": [
        {"status": "RUNNING", "timestamp": _timestamp()}
    ]}
    _atomic_write(path, started)
    try:
        result = await operation()
    except BaseException as exc:
        _atomic_write(path, {
            **identity,
            "status": "EXECUTION_ABORTED",
            "error_type": type(exc).__name__,
            "history": [*started["history"], {
                "status": "EXECUTION_ABORTED", "timestamp": _timestamp(),
                "error_type": type(exc).__name__,
            }],
        })
        raise
    _atomic_write(path, {
        **identity,
        "status": "EXECUTION_COMPLETE",
        "history": [*started["history"], {
            "status": "EXECUTION_COMPLETE", "timestamp": _timestamp(),
        }],
    })
    return result
