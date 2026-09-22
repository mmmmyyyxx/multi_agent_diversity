"""The two production Layer-1 adapters.

GEPA and MARS keep their native search implementations. These adapters only
translate the one production request into the already-frozen native or
Layer-2 entry seam. They never own target selection, team policy or write-back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..experiment import LocalOptimizationRequest, RuntimeContext
from ..native_feed import Layer2OptimizationRequest, NativeOptimizationRequest
from .base import Layer2EvidencePromptOptimizer, NativeFeedPromptOptimizer
from .schemas import LocalOptimizationResult


class BackendContractError(ValueError):
    """A production Layer-1 request is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class _BackendAdapter:
    name: Literal["gepa", "mars"]
    fidelity: str
    native: NativeFeedPromptOptimizer
    layer2: Layer2EvidencePromptOptimizer

    async def optimize(
        self,
        request: LocalOptimizationRequest,
        context: RuntimeContext,
    ) -> LocalOptimizationResult:
        request.validate_against(context)
        if request.optimization_scope == "native":
            if not isinstance(request.problem, NativeOptimizationRequest):
                raise BackendContractError("native request requires NativeOptimizationRequest")
            return await self.native.optimize_native(request.problem)
        if not isinstance(request.problem, Layer2OptimizationRequest):
            raise BackendContractError("Layer2 request requires Layer2OptimizationRequest")
        packet_hash = request.problem.packet.packet_hash
        result = await self.layer2.optimize_layer2(request.problem)
        if request.problem.packet.packet_hash != packet_hash:
            raise BackendContractError("Layer-1 backend mutated Layer2 evidence")
        return result


class GEPABackend(_BackendAdapter):
    def __init__(
        self,
        *,
        native: NativeFeedPromptOptimizer,
        layer2: Layer2EvidencePromptOptimizer,
        fidelity: str = "LEVEL_B_API_COMPATIBLE_ADAPTATION",
    ) -> None:
        super().__init__("gepa", fidelity, native, layer2)


class MARSBackend(_BackendAdapter):
    def __init__(
        self,
        *,
        native: NativeFeedPromptOptimizer,
        layer2: Layer2EvidencePromptOptimizer,
        fidelity: str = "MARS_NATIVE_SEARCH_WITH_LAYER2_EVIDENCE_ADAPTER",
    ) -> None:
        super().__init__("mars", fidelity, native, layer2)


def production_backend(
    name: str,
    *,
    native: NativeFeedPromptOptimizer,
    layer2: Layer2EvidencePromptOptimizer,
):
    """Construct one of the two supported adapters without a four-mode fork."""

    if name == "gepa":
        return GEPABackend(native=native, layer2=layer2)
    if name == "mars":
        return MARSBackend(native=native, layer2=layer2)
    raise BackendContractError(f"unknown production backend: {name}")
