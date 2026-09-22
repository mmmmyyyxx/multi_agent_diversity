"""One runtime registry for replaceable Layer-1 backends and controller modes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, Protocol, runtime_checkable

from ..native_feed import Layer2OptimizationRequest, NativeOptimizationRequest
from .base import Layer2EvidencePromptOptimizer, NativeFeedPromptOptimizer
from .schemas import LocalOptimizationResult


OptimizerBackendName = Literal["gepa", "mars"]
OptimizationMode = Literal["native", "layer2"]
OptimizationProblem = NativeOptimizationRequest | Layer2OptimizationRequest
BackendFactory = Callable[..., NativeFeedPromptOptimizer | Layer2EvidencePromptOptimizer]


@dataclass(frozen=True)
class BackendRuntimeConfig:
    """Backend/controller choice and the experiment identity fields it affects."""

    optimizer_backend: OptimizerBackendName
    optimization_mode: OptimizationMode
    seed: int
    solver_model: str
    optimizer_model: str
    budget: str
    data_split_manifest: str
    initial_state: str
    run_mode: Literal["fixed_budget", "saturation"] = "fixed_budget"
    saturation_config_identity: str = "none"

    def __post_init__(self) -> None:
        if self.optimizer_backend not in {"gepa", "mars"}:
            raise ValueError("optimizer_backend must be gepa or mars")
        if self.optimization_mode not in {"native", "layer2"}:
            raise ValueError("optimization_mode must be native or layer2")
        if self.run_mode not in {"fixed_budget", "saturation"}:
            raise ValueError("run_mode must be fixed_budget or saturation")
        if self.run_mode == "saturation" and self.saturation_config_identity == "none":
            raise ValueError("saturation run identity requires saturation config identity")
        if not all((
            self.solver_model, self.optimizer_model, self.budget,
            self.data_split_manifest, self.initial_state,
        )):
            raise ValueError("unified backend configuration fields must be non-empty")

    @property
    def mode_id(self) -> str:
        return f"{self.optimizer_backend}_{self.optimization_mode}".upper()

    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@runtime_checkable
class Layer1Backend(Protocol):
    optimizer_backend: str
    optimization_mode: str
    backend_fidelity: str

    async def optimize(self, problem: OptimizationProblem) -> LocalOptimizationResult:
        """Optimize one native or Layer-2-defined problem without team write-back."""
        ...


@dataclass(frozen=True)
class ConfiguredLayer1Backend:
    optimizer_backend: str
    optimization_mode: str
    backend_fidelity: str
    implementation: NativeFeedPromptOptimizer | Layer2EvidencePromptOptimizer

    async def optimize(self, problem: OptimizationProblem) -> LocalOptimizationResult:
        if self.optimization_mode == "native":
            if not isinstance(problem, NativeOptimizationRequest):
                raise TypeError("native backend requires NativeOptimizationRequest")
            if not isinstance(self.implementation, NativeFeedPromptOptimizer):
                raise TypeError("native backend implementation lacks optimize_native")
            return await self.implementation.optimize_native(problem)
        if not isinstance(problem, Layer2OptimizationRequest):
            raise TypeError("layer2 backend requires Layer2OptimizationRequest")
        if not isinstance(self.implementation, Layer2EvidencePromptOptimizer):
            raise TypeError("layer2 backend implementation lacks optimize_layer2")
        before = problem.packet.packet_hash
        result = await self.implementation.optimize_layer2(problem)
        if problem.packet.packet_hash != before:
            raise RuntimeError("Layer-1 backend mutated the Layer-2 evidence packet")
        return result


class Layer1BackendRegistry:
    """Register backend implementations by capability, never by Git branch."""

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], tuple[BackendFactory, str]] = {}

    def register(
        self, *, backend: str, mode: str, factory: BackendFactory, fidelity: str
    ) -> None:
        key = (backend, mode)
        if backend in {"sepo", "espo"}:
            raise ValueError(f"reserved backend is not implemented: {backend}")
        if not backend or mode not in {"native", "layer2"} or not fidelity:
            raise ValueError("invalid backend registration")
        if key in self._factories:
            raise ValueError(f"backend mode already registered: {backend}/{mode}")
        self._factories[key] = (factory, fidelity)

    def create(
        self, config: BackendRuntimeConfig, **kwargs: Any
    ) -> ConfiguredLayer1Backend:
        key = (config.optimizer_backend, config.optimization_mode)
        try:
            factory, fidelity = self._factories[key]
        except KeyError as exc:
            raise KeyError(f"unknown backend mode: {key[0]}/{key[1]}") from exc
        return ConfiguredLayer1Backend(
            optimizer_backend=key[0],
            optimization_mode=key[1],
            backend_fidelity=fidelity,
            implementation=factory(**kwargs),
        )

    @property
    def registered_modes(self) -> tuple[str, ...]:
        return tuple(sorted(f"{backend}_{mode}".upper() for backend, mode in self._factories))


def default_backend_registry() -> Layer1BackendRegistry:
    from .gepa_native import GEPALayer2EvidenceOptimizer, GEPANativeFeedOptimizer
    from .mars_native import MARSLayer2EvidenceOptimizer, MARSNativeFeedOptimizer

    registry = Layer1BackendRegistry()
    registry.register(
        backend="gepa", mode="native", factory=GEPANativeFeedOptimizer,
        fidelity="NATIVE_GEPA_CONTROL",
    )
    registry.register(
        backend="gepa", mode="layer2", factory=GEPALayer2EvidenceOptimizer,
        fidelity="LEVEL_B_GEPA_SEARCH_CORE_WITH_LAYER2_EVIDENCE",
    )
    registry.register(
        backend="mars", mode="native", factory=MARSNativeFeedOptimizer,
        fidelity="OFFICIAL_CODE_STYLE_MARS_CONTROL",
    )
    registry.register(
        backend="mars", mode="layer2", factory=MARSLayer2EvidenceOptimizer,
        fidelity="MARS_TCS_SEARCH_CORE_WITH_LAYER2_EVIDENCE",
    )
    return registry
