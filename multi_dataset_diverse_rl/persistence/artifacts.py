"""Strict, atomic artifact persistence used by all run histories."""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSIONS = {
    "candidate_history_schema": 2,
    "archive_history_schema": 2,
    "joint_history_schema": 2,
    "lineage_history_schema": 2,
}


def _json_text(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False)


class ArtifactWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append_jsonl(self, filename: str, rows: Iterable[Mapping[str, Any]]) -> None:
        materialized = [dict(row) for row in rows]
        if not materialized:
            return
        path = self.root / filename
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in materialized:
                handle.write(_json_text(row) + "\n")

    def write_json(self, filename: str, payload: Any) -> None:
        path = self.root / filename
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        for attempt in range(3):
            try:
                temporary.write_text(_json_text(payload, indent=2), encoding="utf-8")
                os.replace(temporary, path)
                return
            except OSError:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def write_csv(self, filename: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
        path = self.root / filename
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    def record(self, filename: str, event: Mapping[str, Any], schema_name: str) -> None:
        if schema_name not in SCHEMA_VERSIONS:
            raise ValueError(f"Unknown artifact schema: {schema_name}")
        self.append_jsonl(filename, [{"schema_version": SCHEMA_VERSIONS[schema_name], **dict(event)}])
