"""Structured optimizer history and deliberately empty LLM memory."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .schemas import OptimizationOpportunity, TransitionRecord


@dataclass
class HistoryState:
    target_counts: dict[int, int] = field(default_factory=dict)
    failure_counts: dict[int, int] = field(default_factory=dict)
    commit_counts: dict[int, int] = field(default_factory=dict)
    latest_by_member: dict[int, TransitionRecord] = field(default_factory=dict)
    transitions: list[TransitionRecord] = field(default_factory=list)
    candidate_lineage: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    pattern_statistics: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> "HistoryState":
        return deepcopy(self)

    def restore(self, snapshot: "HistoryState") -> None:
        self.__dict__.update(deepcopy(snapshot.__dict__))

    def latest_member_transition(self, member_id: int) -> TransitionRecord | None:
        return self.latest_by_member.get(member_id)

    def observe_opportunity(self, member_id: int, *, committed: bool) -> None:
        self.target_counts[member_id] = self.target_counts.get(member_id, 0) + 1
        if committed:
            self.commit_counts[member_id] = self.commit_counts.get(member_id, 0) + 1
            self.failure_counts[member_id] = 0
        else:
            self.failure_counts[member_id] = self.failure_counts.get(member_id, 0) + 1

    def observe_transition(self, transition: TransitionRecord) -> None:
        self.latest_by_member[transition.target_member] = transition
        self.transitions.append(transition)


class MemoryProvider(Protocol):
    def read_for_opportunity(
        self, opportunity: OptimizationOpportunity,
    ) -> Mapping[str, Any]: ...

    def observe_transition(self, transition: TransitionRecord) -> None: ...


class NullMemoryProvider:
    identity = "null_memory_v1"

    def read_for_opportunity(
        self, opportunity: OptimizationOpportunity,
    ) -> Mapping[str, Any]:
        del opportunity
        return {}

    def observe_transition(self, transition: TransitionRecord) -> None:
        del transition

    def prepare_outcome(self, outcome):
        return None

    def validate_delta(self, delta):
        return None

    def apply_outcome(self, delta):
        return None


class PatternAnalyzer(Protocol):
    def analyze(self, state: Any) -> Mapping[str, Any]: ...


class NullPatternAnalyzer:
    identity = "null_pattern_v1"

    def analyze(self, state: Any) -> Mapping[str, Any]:
        del state
        return {}
