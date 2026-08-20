from __future__ import annotations

import importlib.util
import shutil
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry():
    builder = load(
        "v17_module1_registry", "scripts/build_v17_module1_2x2_registry.py"
    )
    return builder.build_registry("0" * 40)


@pytest.fixture
def scratch():
    path = ROOT / "runs" / f"_test_v17_module1_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_six_distinct_reconstructible_parent_strata(registry):
    assert registry["case_count"] == 6
    assert len({row["parent_team_hash"] for row in registry["cases"]}) == 6
    assert [row["stratum"] for row in registry["cases"]].count(
        "concentration_witness"
    ) == 2
    assert [row["stratum"] for row in registry["cases"]].count(
        "throughput_witness"
    ) == 2
    assert [row["stratum"] for row in registry["cases"]].count(
        "neutral_control"
    ) == 2


def test_w1_and_round_robin_are_independently_replayed(registry):
    builder = load(
        "v17_module1_registry_selectors",
        "scripts/build_v17_module1_2x2_registry.py",
    )
    for case in registry["cases"]:
        assert case["w1_target_ids"] == case["w1_independent_replay_target_ids"]
        assert case["round_robin_target_ids"] == builder.rr_targets(
            case["source_seed"], case["source_update_index"]
        )
        assert len(set(case["round_robin_target_ids"])) == 2
        assert len(set(case["w1_target_ids"])) == 2


def test_phase_a_preflight_is_zero_api_and_context_isolated(registry, scratch):
    module = load(
        "v17_module1_preflight", "scripts/preflight_v17_module1_2x2.py"
    )
    result = module.preflight(registry, scratch)
    assert result["status"] == "PASS", result["errors"]
    assert result["api_calls"] == result["model_calls"] == 0
    assert result["validation_calls"] == result["test_calls"] == 0
    assert len(result["context_checks"]) == 48
    for row in result["context_checks"]:
        if row["cell"] in {"A", "B"}:
            assert row["context_type"] == "AccuracyDiagnosisContext"
            assert row["assigned_hash_count"] == 0
        else:
            assert row["context_type"] == "PeerStateDiagnosisContext"
            assert row["assigned_hash_count"] > 0


def test_budget_and_isolation_contract_is_exact(registry):
    assert registry["cell_count"] == 24
    assert registry["branch_count"] == 48
    assert registry["source_candidate_budget"] == 96
    assert registry["source_candidates_per_target"] == 2
    assert registry["loss_blind_revision_per_valid_source"] == 1
    assert registry["commit_enabled"] is False
    assert registry["trajectory_mutation_enabled"] is False
    assert registry["final_test_enabled"] is False
    assert registry["proposal_memory_mode"] == "off"


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [
        ({"A":{"vote":0,"oracle":0},"B":{"vote":-2,"oracle":0},"C":{"vote":0,"oracle":0},"D":{"vote":-2,"oracle":0}}, "TARGET_ALLOCATION_DOMINANT"),
        ({"A":{"vote":0,"oracle":0},"B":{"vote":0,"oracle":0},"C":{"vote":-2,"oracle":0},"D":{"vote":-2,"oracle":0}}, "RESIDUAL_CONTEXT_DOMINANT"),
        ({"A":{"vote":0,"oracle":0},"B":{"vote":-1,"oracle":0},"C":{"vote":-1,"oracle":0},"D":{"vote":-2,"oracle":0}}, "BOTH_CONTRIBUTE"),
        ({"A":{"vote":0,"oracle":0},"B":{"vote":1,"oracle":0},"C":{"vote":1,"oracle":0},"D":{"vote":-4,"oracle":0}}, "NEGATIVE_INTERACTION_DOMINANT"),
        ({"A":{"vote":0,"oracle":0},"B":{"vote":1,"oracle":0},"C":{"vote":1,"oracle":0},"D":{"vote":2,"oracle":0}}, "NO_CLEAR_LOCAL_CAUSAL_SOURCE"),
    ],
)
def test_classifier_has_only_five_frozen_outcomes(aggregate, expected):
    support = load(
        "v17_module1_support_classifier", "scripts/v17_module1_2x2_support.py"
    )
    assert support.classify(aggregate)["conclusion"] == expected


def test_would_commit_realized_delta_contract():
    support = load(
        "v17_module1_support_realized", "scripts/v17_module1_2x2_support.py"
    )
    assert support.realized_delta(False, 40, 47) == 0
    assert support.realized_delta(True, 40, 47) == 7


def test_would_commit_uses_one_unchanged_cross_branch_winner():
    support = load(
        "v17_module1_support_winner", "scripts/v17_module1_2x2_support.py"
    )
    class Evaluator:
        @staticmethod
        def _cross_branch_key(row):
            return row.key
    class Branch:
        def __init__(self, key, accepted=True):
            self.key = key
            self.accepted = object() if accepted else None
    low, high = Branch((1, 0)), Branch((2, 0))
    assert support.choose_would_commit(Evaluator(), [low, high]) is high
    assert support.choose_would_commit(
        Evaluator(), [None, Branch((3, 0), accepted=False)]
    ) is None


def test_phase_b_runner_requires_freeze_and_has_no_test_evaluator():
    source = (ROOT / "scripts/run_v17_module1_2x2_probe.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--source_freeze"' in source
    assert "evaluate_final_test" not in source
    assert "test_path" not in source
