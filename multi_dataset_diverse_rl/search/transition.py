"""Team transition policy and atomic write-back, apart from candidate search."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Protocol, Sequence

from .history import HistoryState, MemoryProvider
from .schemas import (
    EvaluatedCandidate, OptimizationOpportunity, SearchContractError,
    TeamEvaluation, TeamStateSnapshot, TransitionDecision, TransitionRecord,
)


class TransitionPolicy(Protocol):
    def select(
        self, parent: TeamEvaluation, candidates: Sequence[EvaluatedCandidate],
    ) -> TransitionDecision: ...


class CommonSafeTransitionPolicy:
    """Current monotone target/team/invalid guards, independent of engine."""

    identity = "common_safe_v1"

    def select(
        self, parent: TeamEvaluation, candidates: Sequence[EvaluatedCandidate],
    ) -> TransitionDecision:
        feasible = []
        for row in candidates:
            full = row.full
            if not row.promoted or full is None:
                continue
            target = row.diagnostics.get("target_member")
            if not isinstance(target, int):
                raise SearchContractError("transition requires target member")
            current = parent.member_scores[target]
            proposed = full.member_scores[target]
            invalid_delta = int(full.aggregation_diagnostics.get("terminal_invalid_delta", 0))
            if (proposed >= current and full.aggregate_score >= parent.aggregate_score
                    and (proposed > current or full.aggregate_score > parent.aggregate_score)
                    and invalid_delta <= 0):
                feasible.append(row)
        if not feasible:
            return TransitionDecision(None, "NO_COMMON_SAFE_WINNER")
        winner = max(feasible, key=lambda row: (
            row.full.aggregate_score - parent.aggregate_score,  # type: ignore[union-attr]
            min(row.full.member_scores) - min(parent.member_scores),  # type: ignore[union-attr]
            sum(row.full.member_scores) - sum(parent.member_scores),  # type: ignore[union-attr]
            row.candidate.candidate_id,
        ))
        return TransitionDecision(winner, "COMMON_SAFE_WINNER")


class TeamStateStore(Protocol):
    def snapshot(self) -> TeamStateSnapshot: ...

    def restore(self, snapshot: TeamStateSnapshot) -> None: ...

    def replace_member(self, member_id: int, prompt: str) -> TeamStateSnapshot: ...


class TeamStateCommitter:
    """One member transition, with rollback of state and structured history."""

    def __init__(self, store: TeamStateStore) -> None:
        self.store = store

    def commit(
        self, opportunity: OptimizationOpportunity,
        decision: TransitionDecision,
        history: HistoryState,
        memory: MemoryProvider,
    ) -> TransitionRecord:
        if decision.candidate is None or decision.candidate.full is None:
            raise SearchContractError("cannot commit an unevaluated candidate")
        parent = self.store.snapshot()
        history_before = history.snapshot()
        if parent.team_state_id != opportunity.parent_state_id:
            raise SearchContractError("parent state changed before commit")
        try:
            child = self.store.replace_member(
                opportunity.target_member, decision.candidate.candidate.prompt,
            )
            if child.team_state_id == parent.team_state_id:
                raise SearchContractError("commit did not change team state")
            record = TransitionRecord(
                opportunity.opportunity_id, parent.team_state_id, child.team_state_id,
                opportunity.target_member, decision.candidate.candidate.candidate_id,
                parent_prompt_hash=hashlib.sha256(opportunity.parent_prompt.encode("utf-8")).hexdigest(),
                child_prompt_hash=hashlib.sha256(decision.candidate.candidate.prompt.encode("utf-8")).hexdigest(),
                **(self.store.transition_fields(parent, child, opportunity.target_member)
                   if hasattr(self.store, "transition_fields") else {}),
            )
            history.observe_transition(record)
            memory.observe_transition(record)
            return record
        except BaseException:
            self.store.restore(parent)
            history.restore(history_before)
            raise
