"""Role-specific data access for frozen benchmark splits."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

from .data_freeze import GROUPS, PROTOCOL, SOURCE_PINS, file_hash, read_jsonl
from ..search.schemas import SearchContractError


class DataPurpose(str, Enum):
    MUTATION = "mutation"
    REFLECTION = "reflection"
    EVIDENCE = "evidence"
    PATTERN = "pattern"
    MEMORY = "memory"
    SEARCH_EVALUATION = "search_evaluation"
    ADAPTIVE_GATE = "adaptive_gate"


@dataclass(frozen=True)
class SplitAccessPolicy:
    phase: str = "search"

    def require(self, split: str, purpose: DataPurpose, *, manifest_only: bool = False) -> None:
        if split not in GROUPS or self.phase != "search" or not isinstance(purpose, DataPurpose):
            raise SearchContractError("SPLIT_ACCESS_POLICY_INVALID")
        if manifest_only:
            return
        if split in {"test", "validation"}:
            raise SearchContractError("HELDOUT_SEARCH_ACCESS_FORBIDDEN")
        if split == "shadow" and purpose != DataPurpose.ADAPTIVE_GATE:
            raise SearchContractError("SHADOW_CONTENT_SEARCH_VISIBILITY_FORBIDDEN")


@dataclass(frozen=True)
class AdaptiveGateFeedback:
    """Only this aggregate value may cross from gate to search state."""
    aggregate_score: float
    passed: bool

    def __post_init__(self):
        import math
        if type(self.aggregate_score) not in (int, float) or not math.isfinite(self.aggregate_score) or type(self.passed) is not bool:
            raise SearchContractError("ADAPTIVE_GATE_FEEDBACK_INVALID")


class FrozenSplitReader:
    """Memberships come from the manifest; no online resampling entrypoint."""
    def __init__(self, root: Path, benchmark: str, *, expected_manifest_sha256: str,
                 policy: SplitAccessPolicy | None = None):
        self.root = root
        self.policy = policy or SplitAccessPolicy()
        if benchmark not in SOURCE_PINS:
            raise SearchContractError("BENCHMARK_UNKNOWN")
        manifest_path = root / "manifests" / (benchmark + ".json")
        if not expected_manifest_sha256 or file_hash(manifest_path) != expected_manifest_sha256:
            raise SearchContractError("BENCHMARK_MANIFEST_IDENTITY_MISMATCH")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (self.manifest["status"] != "FROZEN" or self.manifest["benchmark_id"] != benchmark
                or self.manifest["protocol"] != PROTOCOL
                or any(self.manifest["source"].get(k) != v for k, v in SOURCE_PINS[benchmark].items())):
            raise SearchContractError("BENCHMARK_SPLIT_NOT_FROZEN")
        self.membership_path = root / "manifests" / (benchmark + ".ids.jsonl")
        if (self.manifest["membership_file"] != self.membership_path.name
                or file_hash(self.membership_path) != self.manifest["membership_sha256"]):
            raise SearchContractError("BENCHMARK_MEMBERSHIP_IDENTITY_MISMATCH")

    def rows(self, split: str, purpose: DataPurpose) -> tuple[dict, ...]:
        self.policy.require(split, purpose)
        item = self.manifest["materialized"][split]
        path = (self.root / item["path"]).resolve()
        if not path.is_relative_to(self.root.resolve() / "materialized") or file_hash(path) != item["sha256"]:
            raise SearchContractError("BENCHMARK_SPLIT_HASH_MISMATCH")
        rows = tuple(read_jsonl(path))
        members = [r for r in read_jsonl(self.membership_path) if r["project_split"] == split]
        if ([r["stable_example_id"] for r in rows] != [r["stable_example_id"] for r in members]
                or len(rows) != self.manifest["counts"][split]):
            raise SearchContractError("BENCHMARK_SPLIT_MEMBERSHIP_MISMATCH")
        return rows
