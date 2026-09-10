"""Backend registry; unimplemented optimizers never silently fall back."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import LocalPromptOptimizer


OptimizerFactory = Callable[..., LocalPromptOptimizer]


class LocalOptimizerRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, OptimizerFactory] = {}

    def register(self, name: str, factory: OptimizerFactory) -> None:
        if not name or name in {"sepo", "espo"}:
            raise ValueError("invalid or reserved optimizer backend")
        if name in self._factories:
            raise ValueError(f"optimizer backend already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> LocalPromptOptimizer:
        if name in {"sepo", "espo"}:
            raise NotImplementedError(f"optimizer backend is reserved but not implemented: {name}")
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"unknown optimizer backend: {name}") from exc
        return factory(**kwargs)

    @property
    def registered(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_registry() -> LocalOptimizerRegistry:
    from .gepa_optimizer import GEPALocalPromptOptimizer
    from .legacy_tcs import LegacyTCSLocalOptimizer

    registry = LocalOptimizerRegistry()
    registry.register("legacy_tcs", LegacyTCSLocalOptimizer)
    registry.register("gepa", GEPALocalPromptOptimizer)
    return registry
