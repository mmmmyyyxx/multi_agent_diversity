"""The pinned official GEPA control remains a distinct paper baseline."""

from __future__ import annotations

from ..local_optimizers.gepa_optimizer import (
    GEPAOptimizerConfig as LegacyFrozenGEPAConfig,
    verify_frozen_gepa_engine_contract,
)

OFFICIAL_GEPA_BASELINE_ID = "official_gepa_v0_1_1_pinned_baseline"

__all__ = [
    "LegacyFrozenGEPAConfig", "OFFICIAL_GEPA_BASELINE_ID",
    "verify_frozen_gepa_engine_contract",
]
