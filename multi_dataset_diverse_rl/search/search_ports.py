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
