"""Candidate search owns proposal/search state, independent of team commit."""

from __future__ import annotations

from typing import Any, Protocol

from .schemas import OptimizationOpportunity, SearchCandidate, SearchResult


class SearchContext(Protocol):
    benchmark: Any
    aggregation: Any
    history: Any
    pattern_view: Any
    memory_view: Any


class SearchEngine(Protocol):
    async def search(
        self, opportunity: OptimizationOpportunity, context: SearchContext,
    ) -> SearchResult: ...


class GEPARequestBridge(Protocol):
    """The only place a current GEPA task representation is needed."""

    def make_task(
        self, opportunity: OptimizationOpportunity, context: SearchContext,
    ) -> Any: ...

    async def optimize(self, task: Any) -> Any: ...


class GEPASearchEngine:
    identity = "gepa_derived_v1"

    def __init__(self, bridge: GEPARequestBridge) -> None:
        self.bridge = bridge

    async def search(
        self, opportunity: OptimizationOpportunity, context: SearchContext,
    ) -> SearchResult:
        result = await self.bridge.optimize(self.bridge.make_task(opportunity, context))
        return SearchResult(
            candidates=tuple(SearchCandidate(
                candidate_id=row.candidate_id,
                prompt=row.prompt,
                search_score=row.local_score,
                lineage=dict(row.backend_metadata),
                backend_details=dict(row.backend_metadata),
            ) for row in result.candidates),
            stop_reason=result.termination_reason,
            search_state=(dict(result.optimizer_state.payload)
                          if result.optimizer_state is not None else {}),
            solver_calls=result.solver_calls,
            search_meta_calls=result.optimizer_calls,
            solver_tokens=result.total_tokens,
        )
