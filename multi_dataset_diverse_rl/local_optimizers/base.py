"""The only public interface between team search and local optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .schemas import LocalEvidenceExample, LocalOptimizationResult, LocalOptimizationTask
from ..native_feed import Layer2OptimizationRequest, NativeOptimizationRequest

if TYPE_CHECKING:
    from ..experiment import LocalOptimizationRequest, RuntimeContext


@dataclass(frozen=True)
class LocalSolverObservation:
    """Backend-neutral result of one decision-procedure/example evaluation."""

    parsed_answer: str | None
    raw_output: str
    correct: bool
    valid: bool
    failure_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    provider_called: bool = True


class LocalSolverEvaluator(Protocol):
    solver_contract_id: str
    output_contract_id: str

    def evaluate(
        self, decision_procedure: str, example: LocalEvidenceExample
    ) -> LocalSolverObservation:
        ...


@runtime_checkable
class LocalPromptOptimizer(Protocol):
    async def optimize(self, task: LocalOptimizationTask) -> LocalOptimizationResult:
        """Return a bounded candidate set; never make a team write-back decision."""
        ...


@runtime_checkable
class LocalOptimizerBackend(Protocol):
    """Production Layer-1 boundary shared by GEPA and MARS.

    The request describes the local problem. The context contains explicit
    execution identity. Neither object exposes the historical monolithic
    ``Config`` to Layer 1.
    """

    name: str
    fidelity: str

    async def optimize(
        self,
        request: "LocalOptimizationRequest",
        context: "RuntimeContext",
    ) -> LocalOptimizationResult:
        ...


@runtime_checkable
class NativeFeedPromptOptimizer(Protocol):
    async def optimize_native(
        self, request: NativeOptimizationRequest
    ) -> LocalOptimizationResult:
        """Consume a backend-owned native data feed and return bounded candidates."""
        ...


@runtime_checkable
class Layer2EvidencePromptOptimizer(Protocol):
    async def optimize_layer2(
        self, request: Layer2OptimizationRequest
    ) -> LocalOptimizationResult:
        """Search only over the immutable evidence packet supplied by Layer 2."""
        ...


@runtime_checkable
class OptimizationContextProvider(Protocol):
    def build_context(self, *, task_id: str) -> str:
        ...


class NoOpContextProvider:
    def build_context(self, *, task_id: str) -> str:
        del task_id
        return ""
