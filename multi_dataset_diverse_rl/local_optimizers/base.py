"""The only public interface between team search and local optimization."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schemas import LocalOptimizationResult, LocalOptimizationTask


@runtime_checkable
class LocalPromptOptimizer(Protocol):
    async def optimize(self, task: LocalOptimizationTask) -> LocalOptimizationResult:
        """Return a bounded candidate set; never make a team write-back decision."""
        ...


@runtime_checkable
class OptimizationContextProvider(Protocol):
    def build_context(self, *, task_id: str) -> str:
        ...


class NoOpContextProvider:
    def build_context(self, *, task_id: str) -> str:
        del task_id
        return ""
