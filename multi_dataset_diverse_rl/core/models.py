from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .enums import ArchiveBucket, CandidateType


@dataclass
class MechanismRepresentation:
    canonical_operations: list[str] = field(default_factory=list)
    semantic_residual_text: str = ""
    embedding: list[float] = field(default_factory=list)
    family_kind: str = "unknown"
    family_id: str = "unknown"
    specificity_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "normalized_operations": list(self.canonical_operations),
            "normalized_operation_sequence": list(self.canonical_operations),
            "mechanism_embedding": list(self.embedding),
        })
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MechanismRepresentation":
        operations = payload.get("canonical_operations", payload.get("normalized_operation_sequence", []))
        return cls(
            canonical_operations=[str(value) for value in operations or []],
            semantic_residual_text=str(payload.get("semantic_residual_text", "") or ""),
            embedding=[float(value) for value in payload.get("embedding", payload.get("mechanism_embedding", [])) or []],
            family_kind=str(payload.get("family_kind", "unknown") or "unknown"),
            family_id=str(payload.get("family_id", "unknown") or "unknown"),
            specificity_score=float(payload.get("specificity_score", 0.0) or 0.0),
        )


@dataclass
class QualityCounts:
    vote: int = 0
    total_agent_correct: int = 0
    bottom2_correct: int = 0
    c1: int = 0
    c2: int = 0
    per_agent_correct: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QualityCounts":
        return cls(
            vote=int(payload.get("vote", payload.get("vote_correct_count", 0)) or 0),
            total_agent_correct=int(payload.get("total_agent_correct", payload.get("total_agent_correct_count", 0)) or 0),
            bottom2_correct=int(payload.get("bottom2_correct", payload.get("bottom2_correct_count", 0)) or 0),
            c1=int(payload.get("c1", payload.get("coverage_depth_c1_correct_count", 0)) or 0),
            c2=int(payload.get("c2", payload.get("coverage_depth_c2_correct_count", 0)) or 0),
            per_agent_correct=[int(value) for value in payload.get("per_agent_correct", payload.get("per_agent_correct_count", [])) or []],
        )


@dataclass
class CandidateMetrics:
    candidate_type: CandidateType = CandidateType.TASK_REPAIR
    target_accuracy: float = 0.0
    accuracy_delta: float = 0.0
    reward: float = 0.0
    quality: QualityCounts = field(default_factory=QualityCounts)
    mechanism: MechanismRepresentation = field(default_factory=MechanismRepresentation)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "candidate_type": self.candidate_type.value,
            "candidate_target_accuracy": self.target_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "reward": self.reward,
            "quality_counts": self.quality.to_dict(),
            "mechanism_representation": self.mechanism.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateMetrics":
        known = {"candidate_type", "candidate_target_accuracy", "target_agent_accuracy", "accuracy_delta", "reward", "quality_counts", "mechanism_representation"}
        return cls(
            candidate_type=CandidateType(str(payload.get("candidate_type", CandidateType.TASK_REPAIR.value))),
            target_accuracy=float(payload.get("candidate_target_accuracy", payload.get("target_agent_accuracy", 0.0)) or 0.0),
            accuracy_delta=float(payload.get("accuracy_delta", 0.0) or 0.0),
            reward=float(payload.get("reward", 0.0) or 0.0),
            quality=QualityCounts.from_dict(payload.get("quality_counts", payload)),
            mechanism=MechanismRepresentation.from_dict(payload.get("mechanism_representation", {})),
            extra={key: value for key, value in payload.items() if key not in known},
        )


@dataclass
class CandidateRecord:
    candidate_id: str
    prompt: str
    prompt_hash: str
    metrics: CandidateMetrics = field(default_factory=CandidateMetrics)
    archive_bucket: ArchiveBucket | None = None
    parent_id: str = ""
    proposal: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra, "candidate_id": self.candidate_id, "prompt": self.prompt,
            "prompt_hash": self.prompt_hash, "parent_id": self.parent_id,
            "proposal": dict(self.proposal), "metrics": self.metrics.to_dict(),
            "archive_bucket": self.archive_bucket.value if self.archive_bucket else "",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateRecord":
        bucket = str(payload.get("archive_bucket", "") or "")
        known = {"candidate_id", "id", "prompt", "prompt_hash", "parent_id", "proposal", "metrics", "archive_bucket"}
        return cls(
            candidate_id=str(payload.get("candidate_id", payload.get("id", ""))),
            prompt=str(payload.get("prompt", "")), prompt_hash=str(payload.get("prompt_hash", "")),
            parent_id=str(payload.get("parent_id", "") or ""), proposal=dict(payload.get("proposal", {}) or {}),
            metrics=CandidateMetrics.from_dict(payload.get("metrics", {})),
            archive_bucket=ArchiveBucket(bucket) if bucket else None,
            extra={key: value for key, value in payload.items() if key not in known},
        )


@dataclass
class QualityAnchor:
    anchor_id: str
    epoch: int
    prompt_hashes: list[str]
    counts: QualityCounts
    created_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id, "epoch": self.epoch,
            "prompt_hashes": list(self.prompt_hashes), "counts": self.counts.to_dict(),
            "created_order": self.created_order,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QualityAnchor":
        return cls(
            anchor_id=str(payload.get("anchor_id", "")), epoch=int(payload.get("epoch", 0) or 0),
            prompt_hashes=[str(value) for value in payload.get("prompt_hashes", []) or []],
            counts=QualityCounts.from_dict(payload.get("counts", payload)),
            created_order=int(payload.get("created_order", 0) or 0),
        )


@dataclass
class JointSelectionResult:
    selected_prompt_hashes: list[str]
    quality_feasible_count: int
    anchor_feasible_count: int
    fallback_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JointSelectionResult":
        return cls(
            selected_prompt_hashes=[str(value) for value in payload.get("selected_prompt_hashes", []) or []],
            quality_feasible_count=int(payload.get("quality_feasible_count", 0) or 0),
            anchor_feasible_count=int(payload.get("anchor_feasible_count", 0) or 0),
            fallback_reason=str(payload.get("fallback_reason", "") or ""),
            diagnostics=dict(payload.get("diagnostics", {}) or {}),
        )
