"""Common top-level result envelope shared by every backend/controller mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class UnifiedRunRecord:
    code_sha: str
    config_hash: str
    backend: str
    optimization_mode: str
    backend_fidelity: str
    initial_state: Mapping[str, Any]
    optimization_opportunities: Sequence[Mapping[str, Any]]
    candidate_funnel: Mapping[str, int]
    team_evaluation_funnel: Mapping[str, int]
    commits: Sequence[Mapping[str, Any]]
    vote_trajectory: Sequence[Mapping[str, Any]]
    oracle_trajectory: Sequence[Mapping[str, Any]]
    member_accuracy_trajectory: Sequence[Mapping[str, Any]]
    pivotality_trajectory: Sequence[Mapping[str, Any]]
    categorical_profile_hashes: Sequence[str]
    cost_accounting: Mapping[str, int]
    final_state: Mapping[str, Any]
    backend_details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.code_sha) != 40 or any(c not in "0123456789abcdef" for c in self.code_sha):
            raise ValueError("code_sha must be a lowercase Git SHA-1")
        if len(self.config_hash) != 64 or any(c not in "0123456789abcdef" for c in self.config_hash):
            raise ValueError("config_hash must be a lowercase SHA-256")
        if self.backend not in {"gepa", "mars"}:
            raise ValueError("unknown backend")
        if self.optimization_mode not in {"native", "layer2"}:
            raise ValueError("unknown optimization mode")
        if not self.backend_fidelity:
            raise ValueError("backend fidelity is required")

    def sanitized(self) -> Mapping[str, Any]:
        return {
            "code_sha": self.code_sha,
            "config_hash": self.config_hash,
            "backend": self.backend,
            "optimization_mode": self.optimization_mode,
            "backend_fidelity": self.backend_fidelity,
            "initial_state": dict(self.initial_state),
            "optimization_opportunities": [dict(row) for row in self.optimization_opportunities],
            "candidate_funnel": dict(self.candidate_funnel),
            "team_evaluation_funnel": dict(self.team_evaluation_funnel),
            "commits": [dict(row) for row in self.commits],
            "vote_trajectory": [dict(row) for row in self.vote_trajectory],
            "oracle_trajectory": [dict(row) for row in self.oracle_trajectory],
            "member_accuracy_trajectory": [dict(row) for row in self.member_accuracy_trajectory],
            "pivotality_trajectory": [dict(row) for row in self.pivotality_trajectory],
            "categorical_profile_hashes": list(self.categorical_profile_hashes),
            "cost_accounting": dict(self.cost_accounting),
            "final_state": dict(self.final_state),
            "backend_details": dict(self.backend_details),
        }
