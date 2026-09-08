from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "audit_diversity_writeback_cost_mechanism.py"
    spec = importlib.util.spec_from_file_location("writeback_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_commit_classifier_order() -> None:
    audit = _module()
    base = {"mean_member_delta": -0.01, "vote_delta": 0.0, "oracle_delta": 0.01, "net_correct_member_votes": -1}
    assert audit.commit_class(base) == "C4_COVERAGE_WITH_COLLATERAL"
    assert audit.commit_class({**base, "vote_delta": 0.01}) == "C2_USEFUL_SPECIALIZATION"


def test_depth_bucket() -> None:
    audit = _module()
    assert audit.depth_bucket(0) == "G=0"
    assert audit.depth_bucket(2) == "G=2"
    assert audit.depth_bucket(3) == "G>=3"
