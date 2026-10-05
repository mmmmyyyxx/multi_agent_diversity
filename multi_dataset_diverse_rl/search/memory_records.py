"""Immutable outcome transactions shared by current and replay memory."""
from __future__ import annotations
from dataclasses import dataclass

RISK_CODES = frozenset({"TEAM_PROBE_REJECTION", "COMMON_SAFE_REJECTION", "SHADOW_REJECTION"})

@dataclass(frozen=True)
class MemoryEntry:
    memory_id: str
    scope: str
    owner_member: int | None
    kind: str
    pattern_id: str | None
    principle: str
    risk_code: str | None
    source_opportunity_id: str
    source_candidate_hash: str
    source_transition_id: str | None
    fixed_count: int
    broken_count: int
    support_count: int
    created_update: int
    lane: str

@dataclass(frozen=True)
class OpportunityOutcome:
    opportunity: object
    evaluated: tuple
    selected_candidate_id: str | None
    committed: bool
    gate_passed: bool | None
    update_index: int
    complete: bool = True
    operational_failure: bool = False

@dataclass(frozen=True)
class MemoryDelta:
    revision: int
    private: tuple[MemoryEntry, ...]
    shared: tuple[MemoryEntry, ...]
