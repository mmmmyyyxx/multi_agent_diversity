"""Schemas owned by Layer 2; no local-optimizer implementation details."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..candidate_selection import CandidateEvaluation, ConstraintDecision
from ..local_optimizers.schemas import LocalPromptCandidate
from ..native_feed import CandidateTransitionAudit


@dataclass(frozen=True)
class TeamEvidenceCase:
    example_id: str
    input_payload: str
    gold: str
    target_output: str | None
    feedback: str | None
    evidence_group: str
    tags: tuple[str, ...]
    source_split: str = "optimize"
    team_disagreement: int = 0
    residual_frequency: int = 0
    team_margin: int = 0
    mutation_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.evidence_group not in {"repair", "preservation", "team_hard"}:
            raise ValueError("unknown team evidence group")
        if self.source_split != "optimize":
            raise ValueError("local optimizer evidence must be Optimize-derived")


@dataclass(frozen=True)
class TeamSearchRequest:
    seed: int
    update_index: int
    team_state_hash: str
    local_metric_budget: int
    solver_contract_id: str
    output_contract_id: str
    optimize_universe_id: str = "optimize_only_universe"


@dataclass(frozen=True)
class TeamSearchAssignment:
    target_member: int
    parent_prompt: str
    evidence: tuple[TeamEvidenceCase, ...]
    optimization_context: str
    responsibility_identity: str
    local_validation_example_ids: tuple[str, ...] = ()
    primary_responsibility_lane: str | None = None
    responsibility_value: float = 0.0
    latest_transition: CandidateTransitionAudit | None = None


@dataclass(frozen=True)
class TeamMiniBatchMetrics:
    invalid_delta: int = 0
    vote_delta: int = 0
    target_delta: int = 0
    team_net_vote_delta: int = 0
    responsibility_delta: int = 0
    broad_delta: int = 0
    oracle_delta: int = 0  # diagnostic only; never enters promotion


@dataclass(frozen=True)
class TeamCandidateRecord:
    local_candidate: LocalPromptCandidate
    minibatch_metrics: TeamMiniBatchMetrics | None = None
    promoted: bool = False
    full_evaluation: CandidateEvaluation | None = None
    constraint: ConstraintDecision | None = None


@dataclass(frozen=True)
class TeamCostAccounting:
    local_optimizer_solver_calls: int = 0
    local_optimizer_meta_calls: int = 0
    local_optimizer_tokens: int = 0
    team_minibatch_solver_calls: int = 0
    team_full_solver_calls: int = 0
    team_shadow_solver_calls: int = 0
    team_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.local_optimizer_tokens + self.team_tokens


@dataclass(frozen=True)
class TeamSearchOutcome:
    assignment: TeamSearchAssignment
    candidates: tuple[TeamCandidateRecord, ...]
    committed_candidate_id: str | None
    termination_reason: str
    cost: TeamCostAccounting
    funnel: Mapping[str, int] = field(default_factory=dict)
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)
