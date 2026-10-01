"""Public compatibility exports, loaded only when explicitly requested.

Offline benchmark/data tooling must not require a provider SDK merely to
import a package submodule. Public export names and implementations are retained.
"""
from importlib import import_module


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = ".config" if name == "Config" else (
        ".system" if name == "PromptEnsembleOptimizationSystem" else ".experiment")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "Config",
    "ExperimentEngine",
    "ExperimentInputs",
    "ExperimentServices",
    "ExperimentSpec",
    "LocalOptimizationRequest",
    "OptimizationScope",
    "OptimizerBackend",
    "PromptEnsembleOptimizationSystem",
    "RuntimeContext",
    "StoppingRegime",
    "run_experiment",
]
