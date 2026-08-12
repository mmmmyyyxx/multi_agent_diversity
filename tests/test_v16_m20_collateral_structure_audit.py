from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / "scripts/audit_v16_m20_collateral_structure.py"
    spec = importlib.util.spec_from_file_location("collateral_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collateral_classifier_precedence():
    module = load()
    base = {
        "loss_count": 1, "nonresponsibility_loss_count": 1,
        "responsibility_gain_count": 1,
        "nonresponsibility_unique_loss_count": 0,
        "nonresponsibility_pivotal_loss_count": 0,
    }
    assert module.collateral_class({**base, "loss_count": 0}) == "NONE"
    assert module.collateral_class(base) == "LOCAL_ACCIDENTAL"
    assert module.collateral_class({**base, "nonresponsibility_unique_loss_count": 1}) == "SPECIALIZATION_OVERWRITE"
    assert module.collateral_class({**base, "loss_count": 4, "nonresponsibility_loss_count": 4}) == "BROAD_DEGRADATION"
    assert module.collateral_class({**base, "loss_count": 2, "nonresponsibility_loss_count": 2}) == "SKILL_TRADEOFF"


def test_real_audit_reconstructs_published_counts_without_api():
    module = load()
    registry = module.json.loads((ROOT / "runs/v16_responsibility_coherence_generic_m20_retry1_prep/probe_registry_private.json").read_text(encoding="utf-8"))
    run = ROOT / "runs/v16_generic_m20_fixed_parent_probe_retry2"
    summary = module.json.loads((run / "probe_summary.json").read_text(encoding="utf-8"))
    candidates, examples, report = module.audit(registry, summary, run / "_shared_solver_cache.sqlite")
    assert len(candidates) == 14
    assert report["target_regression_candidate_count"] == 7
    assert report["responsibility_gain_count"] == 31
    assert report["regression_responsibility_gain_count"] == 13
    assert report["regression_nonresponsibility_loss_count"] == 77
    assert report["regression_pivotal_loss_count"] == 10
    assert report["regression_stable_loss_count"] == 67
    assert report["api_calls"] == report["model_calls"] == 0
    assert report["cache_open_mode"] == "read_only_immutable"
    assert all(row["question_hash"] and len(row["candidate_hash"]) == 64 for row in examples)
    assert all(row["parent_competence_role"] in {"not_parent_correct", "unique", "pivotal", "stable", "fragile"} for row in examples)
