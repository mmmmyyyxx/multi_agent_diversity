"""Local-backend configuration kept outside team-search policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LocalOptimizerSelection:
    backend: str = "gepa"
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in {"gepa", "legacy_tcs", "sepo", "espo"}:
            raise ValueError("unknown local optimizer selection")
