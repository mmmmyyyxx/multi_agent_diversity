"""Strict, atomic artifact persistence used by all run histories."""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .durable_io import atomic_replace, ensure_directory, io_path


def _json_text(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False)


class ArtifactWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        ensure_directory(self.root)

    def append_jsonl(self, filename: str, rows: Iterable[Mapping[str, Any]]) -> None:
        materialized = [dict(row) for row in rows]
        if not materialized:
            return
        path = self.root / filename
        ensure_directory(path.parent)
        with open(io_path(path), "a", encoding="utf-8", newline="\n") as handle:
            for row in materialized:
                handle.write(_json_text(row) + "\n")

    def write_jsonl(self, filename: str, rows: Iterable[Mapping[str, Any]]) -> None:
        path = self.root / filename
        ensure_directory(path.parent)
        temporary = path.with_name(f".{uuid.uuid4().hex[:12]}.tmp")
        with open(io_path(temporary), "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_json_text(dict(row)) + "\n")
        atomic_replace(temporary, path)

    def write_json(self, filename: str, payload: Any) -> None:
        path = self.root / filename
        ensure_directory(path.parent)
        temporary = path.with_name(f".{uuid.uuid4().hex[:12]}.tmp")
        for attempt in range(3):
            try:
                with open(io_path(temporary), "w", encoding="utf-8") as handle:
                    handle.write(_json_text(payload, indent=2))
                atomic_replace(temporary, path)
                return
            except OSError:
                try:
                    os.unlink(io_path(temporary))
                except OSError:
                    pass
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def write_csv(self, filename: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
        path = self.root / filename
        ensure_directory(path.parent)
        temporary = path.with_name(f".{uuid.uuid4().hex[:12]}.tmp")
        with open(io_path(temporary), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        atomic_replace(temporary, path)
