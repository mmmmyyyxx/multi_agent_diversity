from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_gepa_candidate_breadth",
    ROOT / "scripts" / "audit_gepa_candidate_breadth.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pool_row(candidate_hash: str, rank_value: int) -> dict:
    return {
        "source_candidate_hash": candidate_hash,
        "source_metrics": {"ranking_key": [rank_value, 0.0, candidate_hash]},
    }


def test_frontier_prefers_best_in_class_residual_coverage_before_quality() -> None:
    high_quality = pool_row("high", 2)
    broad = pool_row("broad", 1)
    signatures = {
        "high": {"r1": (0, 1, 1), "r2": (0, 0, 0)},
        "broad": {"r1": (0, 1, 1), "r2": (1, 2, 1)},
    }
    choice, diagnostics = MODULE._frontier_choice(
        [high_quality, broad], signatures
    )
    assert choice == "broad"
    assert diagnostics["broad"]["best_in_class_residual_count"] == 2
    assert diagnostics["broad"]["on_local_frontier"] is True
    assert diagnostics["high"]["on_local_frontier"] is False


def test_frontier_uses_historical_quality_only_as_tie_break() -> None:
    high_quality = pool_row("high", 2)
    low_quality = pool_row("low", 1)
    signatures = {
        "high": {"r1": (1, 1, 1), "r2": (0, 0, 0)},
        "low": {"r1": (0, 0, 0), "r2": (1, 1, 1)},
    }
    choice, diagnostics = MODULE._frontier_choice(
        [high_quality, low_quality], signatures
    )
    assert choice == "high"
    assert diagnostics["high"]["on_local_frontier"] is True
    assert diagnostics["low"]["on_local_frontier"] is True


def test_classifier_does_not_invent_phase_b_label() -> None:
    source = (ROOT / "scripts" / "audit_gepa_candidate_breadth.py").read_text(
        encoding="utf-8"
    )
    assert '"proposal_breadth_label": None' in source
    assert '"phase_b_status": "NOT_EVALUATED_PHASE_A_STOP"' in source
    assert "validation_used_for_frontier\": False" in source


def test_report_contains_no_api_or_test_execution_path() -> None:
    source = (ROOT / "scripts" / "audit_gepa_candidate_breadth.py").read_text(
        encoding="utf-8"
    )
    assert "await " not in source
    assert "DASHSCOPE" not in source
    assert "test_path" not in source
    assert "new_api_calls\": 0" in source


def test_source_validation_uses_registry_historical_cache_namespace() -> None:
    source = (ROOT / "scripts" / "audit_gepa_candidate_breadth.py").read_text(
        encoding="utf-8"
    )
    assert 'Path(str(representative["historical_cache_path"]))' in source
    assert "historical_cache.get(frontier_hash, validation_hashes)" in source
    assert "validation_cache.get" not in source
