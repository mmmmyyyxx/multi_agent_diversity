"""Replaceable diagnosis, target, feasibility, evidence and stop policies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Protocol, Sequence

from .history import HistoryState
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
        return max(4 * self.direct_count, 2 * self.near_margin_count,
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
    identity = "responsibility_failure_discount_v1"

    def select(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis,
        feasible_members: Sequence[int], history: HistoryState,
    ) -> TargetDecision:
        del state
        eligible = tuple(sorted(set(feasible_members)))
        scores: dict[int, float] = {}
        for member in eligible:
            signal = diagnosis.responsibility.get(member)
            if not isinstance(signal, ResponsibilitySignal):
                raise SearchContractError("target lacks responsibility diagnosis")
            scores[member] = signal.raw_value / (1 + history.failure_counts.get(member, 0))
        selected = min(eligible, key=lambda member: (-scores[member], member)) if eligible else None
        return TargetDecision(selected, eligible, scores,
                              "selected" if selected is not None else "NO_FEASIBLE_OPPORTUNITY")


class EvidencePolicy(Protocol):
    def build(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis,
        member_id: int, rows: Sequence[EvidenceItem],
    ) -> EvidenceView: ...


def _stable_key(row: EvidenceItem) -> tuple[str, str]:
    return hashlib.sha256(row.example_id.encode("utf-8")).hexdigest(), row.example_id


class CurrentRoleEvidencePolicy:
    """Current quotas while exposing search validation and team probe separately."""

    identity = "role_view_4_4_4_v1"

    def build(
        self, state: TeamStateSnapshot, diagnosis: Diagnosis,
        member_id: int, rows: Sequence[EvidenceItem],
    ) -> EvidenceView:
        del state
        signal = diagnosis.responsibility.get(member_id)
        if not isinstance(signal, ResponsibilitySignal):
            raise SearchContractError("evidence requires responsibility signal")
        chosen: list[EvidenceItem] = []
        chosen_groups: list[str] = []
        used: set[str] = set()
        for role in ("REPAIR", "PRESERVATION", "TEAM_HARD"):
            candidates = [row for row in rows if role in row.roles]
            if role == "REPAIR" and signal.primary_lane != "fallback":
                candidates = [row for row in candidates if
                              signal.primary_lane in row.roles or
                              row.signals.get("lane") == signal.primary_lane or
                              (signal.primary_lane == "coverage" and
                               "pure_coverage" in row.roles)]
            if role == "REPAIR":
                lane_order = {"direct_flip": 0, "near_margin": 1, "coverage": 2}
                key = lambda row: (
                    lane_order.get(str(row.signals.get("lane", "")), 3), *_stable_key(row),
                )
            elif role == "PRESERVATION":
                key = lambda row: (
                    -int(bool(row.signals.get("mutation_sensitive", False))),
                    int(row.signals.get("team_margin", 0)),
                    -int(row.signals.get("team_disagreement", 0)), *_stable_key(row),
                )
            else:
                key = lambda row: (
                    -int(row.signals.get("team_disagreement", 0)),
                    -int(row.signals.get("residual_frequency", 0)), *_stable_key(row),
                )
            candidates.sort(key=key)
            selected = [row for row in candidates if row.example_id not in used][:4]
            if len(selected) != 4:
                raise SearchContractError(f"incomplete {role} evidence quota")
            chosen.extend(selected)
            chosen_groups.extend([role.lower()] * len(selected))
            used.update(row.example_id for row in selected)
        focus = tuple(sorted((row for row in rows if "TRANSITION_FOCUS" in row.roles),
                             key=lambda row: row.example_id))
        anchor = tuple(sorted((row for row in rows if "TRANSITION_ANCHOR" in row.roles),
                              key=lambda row: row.example_id))
        capacity = 36 - len(focus) - len(anchor)
        if capacity < 4:
            raise SearchContractError("insufficient bounded repair capacity")
        lane_order = {"direct_flip": 0, "near_margin": 1, "coverage": 2}
        repairs = sorted((row for row in rows if "REPAIR" in row.roles
                          and (signal.primary_lane == "fallback"
                               or signal.primary_lane in row.roles
                               or row.signals.get("lane") == signal.primary_lane
                               or (signal.primary_lane == "coverage" and
                                   "pure_coverage" in row.roles))),
                         key=lambda row: (
                             lane_order.get(str(row.signals.get("lane", "")), 3),
                             *_stable_key(row),
                         ))[:capacity]
        mutation = (
            *(EvidenceItem(row.example_id, row.source_split,
                           frozenset({"RESPONSIBILITY"}), row.signals)
              for row in repairs),
            *(EvidenceItem(row.example_id, row.source_split,
                           frozenset({"TRANSITION_FOCUS"}), row.signals)
              for row in focus),
            *(EvidenceItem(row.example_id, row.source_split,
                           frozenset({"TRANSITION_ANCHOR"}), row.signals)
              for row in anchor),
        )
        # The same IDs are intentionally permitted, but the two purposes have
        # independent role-bearing immutable values.
        validation = tuple(EvidenceItem(
            row.example_id, row.source_split,
            row.roles | {"SEARCH_VALIDATION"},
            row.signals,
        ) for row, group in zip(chosen, chosen_groups, strict=True))
        probe = tuple(EvidenceItem(
            row.example_id, row.source_split,
            row.roles | {"TEAM_PROBE"},
            {**row.signals, "legacy_group": group},
        ) for row, group in zip(chosen, chosen_groups, strict=True))
        return EvidenceView(mutation, validation, probe,
                            "optimize_full", "shadow_adaptive")


class SearchStopPolicy:
    identity = "gepa_strict_local_saturation_v1"

    def __init__(self, no_update_patience: int = 3) -> None:
        if no_update_patience <= 0:
            raise SearchContractError("search patience must be positive")
        self.no_update_patience = no_update_patience


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
