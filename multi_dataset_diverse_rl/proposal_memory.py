"""Agent-isolated, state-local proposal-search memory.

The store is deliberately keyed by the complete active decision state.  It
contains only hashes and numeric candidate outcomes; prompts, questions,
answers, and model responses never enter this layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def assigned_residual_set_hash(question_hashes: Sequence[str]) -> str:
    """Hash a deduplicated, sorted owned-residual set with a versioned encoding."""
    ordered = tuple(sorted({str(value) for value in question_hashes}))
    payload = {
        "schema": "assigned_residual_set_v1",
        "question_hashes": list(ordered),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProposalMemoryKey:
    run_id: str
    team_state_version: int
    target_agent_id: int
    target_prompt_hash: str
    assigned_residual_set_hash: str

    def key_hash(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SanitizedCandidateSummary:
    prompt_hash: str
    target_gain: int
    vote_gain_count: int
    vote_loss_count: int
    vote_net_gain: int
    assigned_residual_repair_count: int
    assigned_residual_utility_delta: float
    coverage_gain_count: int
    coverage_loss_count: int
    unique_correct_gain_count: int
    unique_correct_loss_count: int
    pivotal_correct_gain_count: int
    pivotal_correct_loss_count: int
    rejection_reasons: tuple[str, ...]


@dataclass
class ProposalMemoryEntry:
    key: ProposalMemoryKey
    assigned_question_hashes: tuple[str, ...]
    attempt_count: int = 0
    previous_evidence_bundle_hashes: tuple[str, ...] = ()
    previous_repair_plan_hashes: tuple[str, ...] = ()
    last_failure_stage: str = ""
    last_rejection_reason_histogram: dict[str, int] = field(default_factory=dict)
    candidate_summaries: tuple[SanitizedCandidateSummary, ...] = ()
    max_target_gain: int | None = None
    max_vote_net_gain: int | None = None
    max_assigned_residual_repair_count: int | None = None
    rotation_cursor: int = 0
    immediate_tabu_bundle_hash: str | None = None
    rotation_exhausted: bool = False


@dataclass(frozen=True)
class ProposalFailureFeedback:
    memory_key_hash: str
    attempt_count: int
    failure_stage: str
    max_target_gain: int | None
    max_vote_net_gain: int | None
    max_assigned_residual_repair_count: int | None
    rejection_reason_histogram: dict[str, int]
    previous_evidence_bundle_hash: str | None
    previous_repair_plan_hash: str | None
    required_revision_mode: str
    rotation_level: str
    rotation_exhausted: bool


def feedback_for(entry: ProposalMemoryEntry) -> ProposalFailureFeedback:
    failure_stage = entry.last_failure_stage
    if failure_stage == "critic":
        revision_mode, rotation_level = "critic_plan_revision", "none"
    elif failure_stage == "zero_repair_behavior":
        revision_mode, rotation_level = "change_repair_mechanism", "pattern"
    elif failure_stage == "regressive_progress":
        revision_mode, rotation_level = "preservation_localization", "preservation"
    else:
        revision_mode, rotation_level = "new_state_local_hypothesis", "representative"
    return ProposalFailureFeedback(
        memory_key_hash=entry.key.key_hash(),
        attempt_count=entry.attempt_count,
        failure_stage=failure_stage,
        max_target_gain=entry.max_target_gain,
        max_vote_net_gain=entry.max_vote_net_gain,
        max_assigned_residual_repair_count=entry.max_assigned_residual_repair_count,
        rejection_reason_histogram=dict(entry.last_rejection_reason_histogram),
        previous_evidence_bundle_hash=(
            entry.previous_evidence_bundle_hashes[-1]
            if entry.previous_evidence_bundle_hashes else None
        ),
        previous_repair_plan_hash=(
            entry.previous_repair_plan_hashes[-1]
            if entry.previous_repair_plan_hashes else None
        ),
        required_revision_mode=revision_mode,
        rotation_level=rotation_level,
        rotation_exhausted=entry.rotation_exhausted,
    )


def entry_from_dict(payload: Mapping[str, Any]) -> ProposalMemoryEntry:
    raw = dict(payload)
    raw["key"] = ProposalMemoryKey(**raw["key"])
    raw["assigned_question_hashes"] = tuple(raw["assigned_question_hashes"])
    raw["previous_evidence_bundle_hashes"] = tuple(raw.get("previous_evidence_bundle_hashes", ()))
    raw["previous_repair_plan_hashes"] = tuple(raw.get("previous_repair_plan_hashes", ()))
    raw["candidate_summaries"] = tuple(
        SanitizedCandidateSummary(
            **{**row, "rejection_reasons": tuple(row.get("rejection_reasons", ()))}
        )
        for row in raw.get("candidate_summaries", ())
    )
    raw["last_rejection_reason_histogram"] = dict(
        raw.get("last_rejection_reason_histogram", {})
    )
    return ProposalMemoryEntry(**raw)


def entry_to_dict(entry: ProposalMemoryEntry) -> dict[str, Any]:
    return asdict(entry)
