"""Progressive team evaluation and separate adaptive validation gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .aggregation import AggregationPolicy
from .benchmark import BenchmarkAdapter, BenchmarkInput
from .schemas import (
    EvaluatedCandidate, OptimizationOpportunity, SearchCandidate, SearchContractError,
    SearchResult, TeamEvaluation,
)


@dataclass(frozen=True)
class ScoredTeamRow:
    input_id: str
    member_scores: tuple[float, ...]
    aggregate_score: float
    aggregate_valid: bool
    aggregation_diagnostics: dict[str, object]
    aggregation_calls: int = 0
    aggregation_tokens: int = 0
    aggregation_logical_evaluations: int = 0
    aggregation_cache_hits: int = 0


class TeamEvaluator:
    """Gold is used only after aggregation has returned an inference output."""

    def __init__(self, benchmark: BenchmarkAdapter, aggregation: AggregationPolicy) -> None:
        self.benchmark = benchmark
        self.aggregation = aggregation

    async def evaluate_row(
        self, *, item: BenchmarkInput, member_outputs: tuple[str, ...], gold: object,
    ) -> ScoredTeamRow:
        if len(member_outputs) == 0:
            raise SearchContractError("team evaluation requires member outputs")
        # The aggregation call has no gold parameter or evaluation object.
        aggregated = await self.aggregation.aggregate(
            item=item, member_outputs=member_outputs, benchmark=self.benchmark,
        )
        members = tuple(self.benchmark.parse_member_output(raw, item)
                        for raw in member_outputs)
        member_scores = tuple(self.benchmark.score_member_output(row, gold)
                              for row in members)
        aggregate_score = self.benchmark.score_member_output(
            aggregated.parsed_output, gold,
        )
        return ScoredTeamRow(
            item.input_id, member_scores, aggregate_score,
            aggregated.parsed_output.valid, dict(aggregated.diagnostics),
            aggregated.calls, aggregated.tokens,
            aggregated.logical_evaluations, aggregated.cache_hits,
        )


class CandidateEvaluationProvider(Protocol):
    async def active(
        self, opportunity: OptimizationOpportunity,
    ) -> TeamEvaluation: ...

    async def team_probe(
        self, opportunity: OptimizationOpportunity, candidate: SearchCandidate,
    ) -> TeamEvaluation: ...

    async def full(
        self, opportunity: OptimizationOpportunity, candidate: SearchCandidate,
    ) -> TeamEvaluation: ...


class PromotionPolicy(Protocol):
    def select(
        self, rows: Sequence[tuple[SearchCandidate, TeamEvaluation]],
    ) -> tuple[str, ...]: ...


class CandidateEvaluationPipeline:
    """Search candidates → team probe → bounded Full; no write-back here."""

    def __init__(
        self, provider: CandidateEvaluationProvider, promotion: PromotionPolicy,
        *, max_promoted: int = 2,
    ) -> None:
        if max_promoted != 2:
            raise SearchContractError("current promotion budget must remain two")
        self.provider = provider
        self.promotion = promotion
        self.max_promoted = max_promoted

    async def evaluate(
        self, opportunity: OptimizationOpportunity, search: SearchResult,
    ) -> tuple[EvaluatedCandidate, ...]:
        evaluated_probe: list[tuple[SearchCandidate, TeamEvaluation]] = []
        for candidate in search.candidates:
            evaluated_probe.append((
                candidate, await self.provider.team_probe(opportunity, candidate),
            ))
        rows = tuple(evaluated_probe)
        selected = self.promotion.select(rows)
        if len(selected) > self.max_promoted or len(set(selected)) != len(selected):
            raise SearchContractError("invalid promotion selection")
        candidate_ids = {candidate.candidate_id for candidate, _ in rows}
        if not set(selected) <= candidate_ids:
            raise SearchContractError("promotion selected an unknown candidate")
        output: list[EvaluatedCandidate] = []
        for candidate, probe in rows:
            promoted = candidate.candidate_id in selected
            full = await self.provider.full(opportunity, candidate) if promoted else None
            output.append(EvaluatedCandidate(
                candidate, probe, full, promoted, False,
                {"target_member": opportunity.target_member},
            ))
        return tuple(output)


class AdaptiveValidationGate(Protocol):
    async def check(
        self, opportunity: OptimizationOpportunity, candidate: EvaluatedCandidate,
    ) -> bool: ...
