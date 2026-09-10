"""Layered ledger contract for cost and lineage attribution."""

from __future__ import annotations

from dataclasses import dataclass


PHASES = {
    "local_optimizer_reflection",
    "local_optimizer_solver_eval",
    "team_minibatch_eval",
    "team_full_eval",
    "team_shadow_eval",
    "validation_replay",
}


@dataclass(frozen=True)
class TwoLayerLedgerRecord:
    seed: int
    update_index: int
    target_member: int
    local_optimizer_backend: str
    local_search_id: str
    local_candidate_id: str | None
    local_parent_ids: tuple[str, ...]
    local_generation: int | None
    team_candidate_id: str | None
    phase: str
    evaluation_stage: str
    logical_role: str
    client_role: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    logical_call_id: str
    provider_attempt_id: str
    cache_hit: bool

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError("unknown two-layer ledger phase")
        if self.input_tokens + self.output_tokens != self.total_tokens:
            raise ValueError("two-layer ledger token arithmetic mismatch")
        if not self.logical_call_id or not self.provider_attempt_id:
            raise ValueError("two-layer ledger call identities are required")
