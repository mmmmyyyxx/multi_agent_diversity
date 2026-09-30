"""Deterministic local materialization and explicit, never auto-frozen splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..search.schemas import SearchContractError


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MaterializedDataset:
    benchmark_id: str
    upstream_identity: str
    upstream_split: str
    source_sha256: str
    ordered_ids: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]


def load_jsonl(path: Path, *, benchmark_id: str, upstream_identity: str,
               upstream_split: str, expected_sha256: str, id_field: str) -> MaterializedDataset:
    if not all((benchmark_id, upstream_identity, upstream_split, expected_sha256, id_field)):
        raise SearchContractError("BENCHMARK_PROVENANCE_NOT_FROZEN")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise SearchContractError("BENCHMARK_SOURCE_HASH_MISMATCH")
    rows = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip())
    if any(not isinstance(row, dict) or id_field not in row for row in rows):
        raise SearchContractError("BENCHMARK_ROW_SCHEMA_MISMATCH")
    ids = tuple(str(row[id_field]) for row in rows)
    if len(ids) != len(set(ids)):
        raise SearchContractError("BENCHMARK_DUPLICATE_EXAMPLE_ID")
    return MaterializedDataset(benchmark_id, upstream_identity, upstream_split,
                               digest, ids, rows)


@dataclass(frozen=True)
class SplitManifest:
    benchmark_id: str
    upstream_identity: str
    upstream_split: str
    source_sha256: str
    selection_rule: str
    seed: int | None
    ordered_ids: tuple[str, ...]
    search_ids: tuple[str, ...]
    shadow_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    status: str = "PROPOSED"

    def __post_init__(self) -> None:
        if self.status not in {"PROPOSED", "FROZEN"}:
            raise SearchContractError("BENCHMARK_SPLIT_STATUS_INVALID")
        groups = (self.search_ids, self.shadow_ids, self.validation_ids, self.test_ids)
        if len(self.ordered_ids) != len(set(self.ordered_ids)):
            raise SearchContractError("BENCHMARK_DUPLICATE_EXAMPLE_ID")
        if any(len(group) != len(set(group)) for group in groups):
            raise SearchContractError("BENCHMARK_DUPLICATE_SPLIT_ID")
        if any(set(a) & set(b) for i, a in enumerate(groups) for b in groups[i + 1:]):
            raise SearchContractError("BENCHMARK_SPLIT_OVERLAP")
        if set().union(*(set(group) for group in groups)) != set(self.ordered_ids):
            raise SearchContractError("BENCHMARK_SPLIT_COVERAGE_MISMATCH")
        if not all((self.benchmark_id, self.upstream_identity, self.upstream_split,
                    self.source_sha256, self.selection_rule)):
            raise SearchContractError("BENCHMARK_SPLIT_IDENTITY_INCOMPLETE")

    def identity(self) -> str:
        from dataclasses import asdict
        return _digest(asdict(self))

    def group_hashes(self) -> dict[str, str]:
        return {name: _digest(getattr(self, f"{name}_ids"))
                for name in ("search", "shadow", "validation", "test")}

    def require_frozen(self) -> None:
        if self.status != "FROZEN":
            raise SearchContractError("BENCHMARK_SPLIT_NOT_FROZEN")
