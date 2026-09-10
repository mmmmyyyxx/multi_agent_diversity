"""Compatibility facade for the frozen Teacher-Critic-Student proposer.

The facade deliberately owns no TCS behavior. A caller supplies the existing
proposal entry point, and this module only checks/converts the neutral result.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from .schemas import LocalOptimizationResult, LocalOptimizationTask


LegacyTCSBridge = Callable[[LocalOptimizationTask], Awaitable[LocalOptimizationResult]]


class LegacyTCSLocalOptimizer:
    backend_name = "legacy_tcs"

    def __init__(self, bridge: LegacyTCSBridge, *, backend_version: str):
        self._bridge = bridge
        self.backend_version = backend_version

    async def optimize(self, task: LocalOptimizationTask) -> LocalOptimizationResult:
        result = await self._bridge(task)
        if result.backend_name != self.backend_name:
            raise ValueError("legacy TCS bridge returned the wrong backend identity")
        if result.backend_version != self.backend_version:
            raise ValueError("legacy TCS bridge returned the wrong backend version")
        return result
