"""Canonical configuration sections and the flat compatibility registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping


@dataclass
class DataConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateGenerationConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateEvaluationConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGuardConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchiveConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class JointSelectionConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodIdentity:
    values: dict[str, Any] = field(default_factory=dict)


SECTION_TYPES = {
    "data": DataConfig, "models": ModelConfig, "runtime": RuntimeConfig,
    "generation": CandidateGenerationConfig, "evaluation": CandidateEvaluationConfig,
    "quality": QualityGuardConfig, "archive": ArchiveConfig, "joint": JointSelectionConfig,
    "lineage": LineageConfig, "output": OutputConfig, "identity": MethodIdentity,
}


def section_for_field(name: str) -> str:
    if name.startswith(("task_", "dataset_", "train_path", "val_path", "test_path", "train_size", "val_size", "test_size", "answer_format", "benchmark", "comparison_task_id", "split_integrity")):
        return "data"
    if name.endswith("_model") or name.endswith("_api_key_env") or name.endswith("_base_url_env"):
        return "models"
    if name.startswith(("teacher_", "critic_", "student_", "optimizer_architecture", "optimizer_fallback", "num_candidates", "optimizer_parent")):
        return "generation"
    if name.startswith(("candidate_eval_", "candidate_reuse_", "solver_rollout_", "eval_solver_")):
        return "evaluation"
    if name.startswith(("candidate_refill_", "probation_", "qd_")) or name in {"beam_size", "joint_representative_beam_size"}:
        return "archive"
    if name.startswith("joint_") or name.startswith("team_diversity_") or name.startswith("peer_collapse_"):
        return "joint"
    if name.startswith("lineage_"):
        return "lineage"
    if name.endswith("_version") or name in {"method_version", "target_selector_mode", "search_mode", "reward_mode", "candidate_selection_mode", "best_state_selection_mode", "experiment_setting"}:
        return "identity"
    if name.startswith(("reward_", "accuracy_guard_", "invalid_guard_", "competence_", "behavior_", "residual_", "mechanism_", "soft_guard_", "catastrophic_", "pivotal_", "shared_error_")):
        return "quality"
    if name.startswith(("out_", "llm_call_", "trace_embedding", "embedding_", "diversity_metric", "invalid_binary", "use_joint_trace")):
        return "output"
    return "runtime"


def split_flat_config(values: Mapping[str, Any]) -> dict[str, Any]:
    sections = {name: section_type() for name, section_type in SECTION_TYPES.items()}
    for name, value in values.items():
        sections[section_for_field(name)].values[name] = value
    return sections


def flatten_sections(sections: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in sections.values():
        result.update(dict(getattr(section, "values", {}) or {}))
    return result


def canonical_config_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "sections"):
        return {
            name: dict(section.values)
            for name, section in config.sections().items()
        }
    values = config.to_flat_dict() if hasattr(config, "to_flat_dict") else asdict(config)
    return {
        name: dict(section.values)
        for name, section in split_flat_config(values).items()
    }


def canonical_field_registry(config_type: type) -> dict[str, str]:
    if hasattr(config_type, "flat_field_registry"):
        return dict(config_type.flat_field_registry())
    return {item.name: f"{section_for_field(item.name)}.{item.name}" for item in fields(config_type)}
