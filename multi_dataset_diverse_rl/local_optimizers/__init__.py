"""Replaceable single-member prompt-optimization backends."""

from .base import (
    LocalOptimizerBackend,
    LocalPromptOptimizer,
    Layer2EvidencePromptOptimizer,
    LocalSolverEvaluator,
    LocalSolverObservation,
    NativeFeedPromptOptimizer,
    OptimizationContextProvider,
)
from .backend_registry import (
    BackendRuntimeConfig,
    ConfiguredLayer1Backend,
    Layer1Backend,
    Layer1BackendRegistry,
    default_backend_registry,
)
from .schemas import (
    LocalEvidenceExample,
    LocalOptimizationResult,
    LocalOptimizationTask,
    LocalOptimizerBudget,
    LocalPromptCandidate,
    OpaqueOptimizerState,
)
from ..saturation import RunMode, SaturationConfig, StopReason

__all__ = [
    "LocalEvidenceExample",
    "BackendRuntimeConfig",
    "ConfiguredLayer1Backend",
    "Layer1Backend",
    "Layer1BackendRegistry",
    "LocalOptimizationResult",
    "LocalOptimizationTask",
    "LocalOptimizerBudget",
    "LocalPromptCandidate",
    "LocalPromptOptimizer",
    "LocalOptimizerBackend",
    "Layer2EvidencePromptOptimizer",
    "LocalSolverEvaluator",
    "LocalSolverObservation",
    "NativeFeedPromptOptimizer",
    "OpaqueOptimizerState",
    "OptimizationContextProvider",
    "default_backend_registry",
    "RunMode",
    "SaturationConfig",
    "StopReason",
]
