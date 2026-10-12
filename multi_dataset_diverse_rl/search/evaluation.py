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

    async def full(
        self, opportunity: OptimizationOpportunity, candidate: SearchCandidate,
    ) -> TeamEvaluation: ...


class PromotionPolicy(Protocol):
    def select(
        self, rows: Sequence[tuple[SearchCandidate, TeamEvaluation]],
    ) -> tuple[str, ...]: ...






class AdaptiveValidationGate(Protocol):
    async def check(
        self, opportunity: OptimizationOpportunity, candidate: EvaluatedCandidate,
    ) -> bool: ...
