from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_v18_writeback_quality.py"
SPEC = importlib.util.spec_from_file_location("v18_writeback_quality", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_input_allowlist_excludes_test_artifacts() -> None:
    assert all("test" not in name.lower() for name in MODULE.ALLOWED_PRIVATE_INPUTS)


def test_frozen_train_side_signals() -> None:
    row = {
        "train_vote_loss_count": 2,
        "train_pivotal_loss_count": 1,
        "train_unique_loss_count": 0,
        "train_coverage_loss_count": 0,
        "train_vote_gain_count": 4,
        "train_soft_utility_delta": 0.1,
        "train_target_gain": 6,
        "train_vote_net_gain": 2,
        "candidate_stage": "loss_blind_generic_revision",
        "assigned_residual_repair_count": 1,
    }
    result = MODULE._signals(row)
    assert result["train_vote_loss_positive"] is True
    assert result["train_vote_gain_and_loss_cooccur"] is True
    assert result["train_pivotal_loss_positive"] is True
    assert result["train_target_only_progress"] is False
    assert tuple(result) == MODULE.SIGNALS


def test_signal_table_does_not_fit_thresholds() -> None:
    rows = [
        {"arm": "HYBRID_BASE", "validation_net_delta": -1, "validation_loss_count": 3,
         **{signal: signal == "train_vote_loss_positive" for signal in MODULE.SIGNALS}},
        {"arm": "HYBRID_BASE", "validation_net_delta": 2, "validation_loss_count": 0,
         **{signal: False for signal in MODULE.SIGNALS}},
    ]
    table = MODULE.signal_table(rows, "HYBRID_BASE")
    vote_loss = next(row for row in table if row["signal"] == "train_vote_loss_positive")
    assert vote_loss["negative_net_precision"] == 1.0
    assert vote_loss["negative_net_sensitivity"] == 1.0
    assert vote_loss["false_positive_count"] == 0


def test_published_writeback_facts_if_report_exists() -> None:
    report = ROOT / "reports" / "v18_writeback_quality_diagnostic_20260824"
    if not report.is_dir() or not (report / "fact_assertions.json").is_file():
        return
    import json

    facts = json.loads((report / "fact_assertions.json").read_text(encoding="utf-8"))
    classifier = json.loads((report / "classifier.json").read_text(encoding="utf-8"))
    assert facts["hybrid_collateral_loss_event_count"] == 7
    assert facts["hybrid_collateral_loss_commit_count"] == 2
    assert facts["hybrid_gain_event_count"] == 5
    assert facts["harmful_pool_zero_loss_feasible_candidate_count"] == 0
    assert classifier["ranking_misselection_supported"] is False
    assert classifier["m2f_compatibility_signal_available_in_v18"] is False
