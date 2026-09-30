"""BBH migration comparator; historical task/record types stop at this edge.

This module is for deterministic replay of the frozen graph. The unified core
does not import it, and new method manifests do not select its old scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..team_search.progressive_evaluation import promote_team_candidates
from ..team_search.schemas import TeamCandidateRecord
from .evaluation import CandidateEvaluationProvider, PromotionPolicy
from .schemas import (
    Diagnosis, EvidenceItem, EvidenceView, EvaluatedCandidate,
    OptimizationOpportunity, SearchCandidate, TeamEvaluation,
    TransitionDecision, TransitionRecord,
)


def _item(row: Any, extra: str | None = None) -> EvidenceItem:
    roles = {str(row.evidence_group).upper(), *map(str, row.tags)}
    if extra:
        roles.add(extra)
    return EvidenceItem(
        row.example_id, row.source_split, frozenset(roles),
        {"team_disagreement": row.team_disagreement,
         "residual_frequency": row.residual_frequency,
         "team_margin": row.team_margin,
         "mutation_sensitive": row.mutation_sensitive},
    )


class FrozenBBHOpportunityBuilder:
    """Convert one frozen V4 assignment to role-separated unified evidence."""

    def __init__(self, request: Any, assignment: Any, task_builder: Any) -> None:
        self.request = request
        self.assignment = assignment
        self.task_builder = task_builder
        self.task = task_builder.build(request, assignment)
        self.team_probe = task_builder.select_team_minibatch(
            assignment.evidence,
            primary_responsibility_lane=assignment.primary_responsibility_lane,
        )

    def build(self, *, state, diagnosis, history, update_index):
        del history, update_index
        if state.team_state_id != self.request.team_state_hash:
            raise RuntimeError("frozen BBH parent mismatch")
        if state.member_prompts[self.assignment.target_member] != self.assignment.parent_prompt:
            raise RuntimeError("frozen BBH prompt mismatch")
        by_id = {row.example_id: row for row in self.assignment.evidence}
        ids = self.assignment.local_validation_example_ids or tuple(
            row.example_id for row in self.team_probe
        )
        search_validation = tuple(_item(by_id[row_id], "SEARCH_VALIDATION")
                                  for row_id in ids)
        probe = tuple(_item(row, "TEAM_PROBE") for row in self.team_probe)
        view = EvidenceView(
            mutation_evidence=tuple(_item(row) for row in self.assignment.evidence),
            search_validation_evidence=search_validation,
            team_probe_evidence=probe,
            full_evaluation_scope="optimize_full",
            adaptive_gate_scope="shadow_adaptive",
        )
        return OptimizationOpportunity(
            opportunity_id=f"bbh:{self.request.team_state_hash}:{self.request.update_index}",
            parent_state_id=state.team_state_id,
            target_member=self.assignment.target_member,
            parent_prompt=self.assignment.parent_prompt,
            objective={"raw_responsibility": self.assignment.responsibility_value},
            diagnosis=diagnosis, evidence=view,
            search_budget={"metric_calls": self.request.local_metric_budget},
            evaluation_plan={"max_promoted": 2},
        )


class FrozenBBHGEPARequestBridge:
    def __init__(self, opportunity: FrozenBBHOpportunityBuilder, optimizer: Any) -> None:
        self.opportunity = opportunity
        self.optimizer = optimizer
        self.raw_candidates: dict[str, Any] = {}

    def make_task(self, opportunity, context):
        del context
        if opportunity.target_member != self.opportunity.assignment.target_member:
            raise RuntimeError("frozen BBH target mismatch")
        return self.opportunity.task

    async def optimize(self, task):
        if hasattr(self.optimizer, "optimize_layer2") and hasattr(task, "packet"):
            result = await self.optimizer.optimize_layer2(task)
        else:
            result = await self.optimizer.optimize(task)
        self.raw_candidates = {row.candidate_id: row for row in result.candidates}
        return result


class FrozenBBHTeamEvaluationProvider:
    def __init__(self, opportunity: FrozenBBHOpportunityBuilder,
                 bridge: FrozenBBHGEPARequestBridge, evaluator: Any) -> None:
        self.opportunity = opportunity
        self.bridge = bridge
        self.evaluator = evaluator
        self.events: list[str] = []

    async def active(self, opportunity):
        del opportunity
        raw = self.evaluator.active_evaluation(self.opportunity.assignment)
        return TeamEvaluation(0.0, None, (0.0,) * 5,
                              aggregation_diagnostics={"legacy_raw": raw})

    async def team_probe(self, opportunity, candidate):
        del opportunity
        raw, _cost = self.evaluator.evaluate_minibatch(
            self.opportunity.assignment,
            self.bridge.raw_candidates[candidate.candidate_id],
            self.opportunity.team_probe,
        )
        self.events.append("team_probe")
        return TeamEvaluation(float(raw.vote_delta), None, (float(raw.target_delta),) * 5,
                              aggregation_diagnostics={"legacy_metrics": raw})

    async def full(self, opportunity, candidate):
        del opportunity
        raw, _cost = self.evaluator.evaluate_full(
            self.opportunity.assignment,
            self.bridge.raw_candidates[candidate.candidate_id],
        )
        self.events.append("full")
        return TeamEvaluation(0.0, None, (0.0,) * 5,
                              aggregation_diagnostics={"legacy_raw": raw})


class FrozenBBHPromotion:
    def select(self, rows):
        records = tuple(TeamCandidateRecord(
            row, minibatch_metrics=evaluation.aggregation_diagnostics["legacy_metrics"],
        ) for row, evaluation in rows)
        selected = promote_team_candidates(records, max_promoted=2)
        return tuple(row.local_candidate.candidate_id for row in selected if row.promoted)


class FrozenBBHTransitionPolicy:
    def __init__(self, bridge: FrozenBBHGEPARequestBridge, selector: Any) -> None:
        self.bridge = bridge
        self.selector = selector

    def select(self, parent, candidates):
        records = tuple(TeamCandidateRecord(
            self.bridge.raw_candidates[row.candidate.candidate_id],
            minibatch_metrics=(row.team_probe.aggregation_diagnostics["legacy_metrics"]
                               if row.team_probe is not None else None),
            promoted=row.promoted,
            full_evaluation=(row.full.aggregation_diagnostics["legacy_raw"]
                             if row.full is not None else None),
        ) for row in candidates)
        annotated = self.selector.annotate(
            records, active=parent.aggregation_diagnostics["legacy_raw"],
        )
        winner = self.selector.select(annotated)
        if winner is None:
            return TransitionDecision(None, "NO_COMMON_SAFE_WINNER")
        matching = next(row for row in candidates
                        if row.candidate.candidate_id == winner.local_candidate.candidate_id)
        return TransitionDecision(matching, "COMMON_SAFE_WINNER")


class FrozenBBHGate:
    def __init__(self, opportunity: FrozenBBHOpportunityBuilder,
                 bridge: FrozenBBHGEPARequestBridge, evaluator: Any) -> None:
        self.opportunity = opportunity
        self.bridge = bridge
        self.evaluator = evaluator
        self.calls = 0

    async def check(self, opportunity, candidate):
        del opportunity
        self.calls += 1
        decision, _cost = self.evaluator.evaluate_shadow(
            self.opportunity.assignment,
            self.bridge.raw_candidates[candidate.candidate.candidate_id],
        )
        return bool(decision.passed)


class FrozenBBHCommitter:
    def __init__(self, opportunity: FrozenBBHOpportunityBuilder,
                 bridge: FrozenBBHGEPARequestBridge,
                 committer: Any, state: Any) -> None:
        self.opportunity = opportunity
        self.bridge = bridge
        self.committer = committer
        self.state = state

    def commit(self, opportunity, decision, history, memory):
        if decision.candidate is None or decision.candidate.full is None:
            raise RuntimeError("frozen BBH winner requires Full")
        before = self.state.snapshot()
        candidate = self.bridge.raw_candidates[decision.candidate.candidate.candidate_id]
        self.committer.commit(
            assignment=self.opportunity.assignment,
            candidate=candidate,
            evaluation=decision.candidate.full.aggregation_diagnostics["legacy_raw"],
        )
        after = self.state.snapshot()
        record = TransitionRecord(
            opportunity.opportunity_id, before.team_state_id,
            after.team_state_id, opportunity.target_member, candidate.candidate_id,
        )
        history.observe_transition(record)
        memory.observe_transition(record)
        return record
