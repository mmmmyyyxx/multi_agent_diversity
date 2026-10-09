"""Replaceable diagnosis, target, feasibility, evidence and stop policies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Protocol, Sequence

from .history import HistoryState
from .responsibility_value import responsibility_value
from .schemas import (
    Diagnosis, EvidenceItem, EvidenceView, SearchContractError, TeamStateSnapshot,
)


@dataclass(frozen=True)
class ResponsibilitySignal:
    member_id: int
    direct_count: int
    near_margin_count: int
    coverage_count: int
    primary_lane: str

    @property
    def raw_value(self) -> int:
        return responsibility_value(self.direct_count, self.near_margin_count,
                                    self.coverage_count)


class ResponsibilityAnalyzer(Protocol):
    def analyze(self, state: TeamStateSnapshot, history: HistoryState) -> Diagnosis: ...


@dataclass(frozen=True)
class TargetDecision:
    selected_member: int | None
    eligible_members: tuple[int, ...]
    target_scores: Mapping[int, float]
    reason: str


class OpportunityFeasibilityPolicy(Protocol):
    def feasible(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis,
        member_id: int, evidence: Sequence[EvidenceItem],
    ) -> bool: ...


class CurrentEvidenceFeasibility:
    """Current 4/4/4 plus bounded packet capacity, before target ranking."""

    identity = "v4_exact_evidence_feasibility_v1"

    def feasible(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis,
        member_id: int, evidence: Sequence[EvidenceItem],
    ) -> bool:
        signal = diagnosis.responsibility.get(member_id)
        if not isinstance(signal, ResponsibilitySignal) or signal.raw_value <= 0:
            return False
        focus = sum("TRANSITION_FOCUS" in row.roles for row in evidence)
        anchor = sum("TRANSITION_ANCHOR" in row.roles for row in evidence)
        if 36 - focus - anchor < 4:
            return False
        try:
            CurrentRoleEvidencePolicy().build(state, diagnosis, member_id, evidence)
        except SearchContractError:
            return False
        return True


class TargetPolicy(Protocol):
    def select(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis,
        feasible_members: Sequence[int], history: HistoryState,
    ) -> TargetDecision: ...


class TargetPolicyV1:
    from ..current_contract import UNIFIED_TARGET_POLICY_VERSION as identity

    def __init__(self, seed=81):
        self.seed = seed

    def select(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis,
        feasible_members: Sequence[int], history: HistoryState,
    ) -> TargetDecision:
        eligible = tuple(sorted(set(feasible_members)))
        scores: dict[int, float] = {}
        for member in eligible:
            signal = diagnosis.responsibility.get(member)
            if not isinstance(signal, ResponsibilitySignal):
                raise SearchContractError("target lacks responsibility diagnosis")
            scores[member] = signal.raw_value / (1 + history.failure_counts.get(member, 0))
        positive = tuple(m for m in eligible if scores[m] > 0)
        if positive:
            selected = min(positive, key=lambda member: (-scores[member], member))
            reason = 'POSITIVE_RESPONSIBILITY'
        elif eligible:
            import random
            ordinal = sum(history.target_counts.values())
            selected = random.Random(f'{self.seed}:{state.team_state_id}:{ordinal}:member-fallback-v2').choice(eligible)
            reason = 'SEEDED_ZERO_RESPONSIBILITY_FALLBACK'
        else:
            selected = None
            reason = 'NO_REPAIR_SIGNAL'
        return TargetDecision(selected, eligible, scores, reason)


class EvidencePolicy(Protocol):
    def build(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis,
        member_id: int, rows: Sequence[EvidenceItem],
    ) -> EvidenceView: ...


def _stable_key(row: EvidenceItem) -> tuple[str, str]:
    return hashlib.sha256(row.example_id.encode("utf-8")).hexdigest(), row.example_id






class GlobalStopPolicy:
    identity = "team_epoch_no_commit_v1"

    def __init__(self, no_commit_patience: int = 2) -> None:
        if no_commit_patience <= 0:
            raise SearchContractError("global patience must be positive")
        self.no_commit_patience = no_commit_patience
        self.no_commit_epochs = 0
        self.parent_state_id: str | None = None
        self.eligible_members: tuple[int, ...] = ()
        self.seen_members: set[int] = set()
        self.any_local_update = False

    def observe_opportunity(
        self, *, parent_state_id: str, eligible_members: Sequence[int],
        selected_member: int, local_update: bool, committed: bool,
    ) -> str | None:
        eligible = tuple(sorted(set(eligible_members)))
        if not eligible or selected_member not in eligible:
            raise SearchContractError("selected member outside frozen feasible epoch")
        if self.parent_state_id is None:
            self.parent_state_id = parent_state_id
            self.eligible_members = eligible
        elif self.parent_state_id != parent_state_id or self.eligible_members != eligible:
            raise SearchContractError("epoch parent or eligibility changed without commit")
        self.seen_members.add(selected_member)
        self.any_local_update = self.any_local_update or local_update
        if committed:
            self.no_commit_epochs = 0
            self.parent_state_id = None
            self.eligible_members = ()
            self.seen_members.clear()
            self.any_local_update = False
            return None
        if set(eligible) <= self.seen_members:
            self.no_commit_epochs += 1
            self.parent_state_id = None
            self.eligible_members = ()
            self.seen_members.clear()
            self.any_local_update = False
            if self.no_commit_epochs >= self.no_commit_patience:
                return "SATURATION_REACHED"
        return None

    def observe_epoch(self, *, committed: bool) -> str | None:
        """Compatibility helper for callers that already froze a full epoch."""
        self.no_commit_epochs = 0 if committed else self.no_commit_epochs + 1
        return "SATURATION_REACHED" if self.no_commit_epochs >= self.no_commit_patience else None
