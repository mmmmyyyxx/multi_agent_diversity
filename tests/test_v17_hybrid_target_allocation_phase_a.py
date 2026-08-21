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
def builder():
    return load("v17_hybrid_builder", "scripts/build_v17_hybrid_target_allocation_registry.py")


@pytest.fixture(scope="module")
def registry(builder):
    return builder.build_registry("0" * 40)


@pytest.fixture
def scratch():
    path = ROOT / "runs" / f"_test_v17_hybrid_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_prospective_parent_selection_is_balanced_deterministic_and_excludes_old(builder):
    first = builder.select_parent_specs()
    second = builder.select_parent_specs()
    assert first == second
    assert len(first) == 6
    assert sorted(row["seed"] for row in first) == [56, 56, 57, 57, 58, 58]
    assert all((row["seed"], row["update_index"]) not in builder.old_parent_pairs() for row in first)
    assert all(row["selection_outcomes_used"] is False for row in first)


def test_hybrid_selector_is_exact_legal_and_rr_constrained(registry):
    support = load("v17_hybrid_support_select", "scripts/v17_hybrid_target_allocation_support.py")
    for case in registry["cases"]:
        arms = support.arm_specs(case)
        eligible = set(case["responsibility_eligible_ids"])
        assert arms[support.HYBRID][0]["target_member"] == int(case["w1_priority_rows"][0]["agent_id"])
        rr_order = support.rr_eligible_order(case["source_seed"], case["source_update_index"], eligible)
        assert arms[support.HYBRID][1]["target_member"] == next(
            agent for agent in rr_order if agent != arms[support.HYBRID][0]["target_member"]
        )
        for rows in arms.values():
            targets = [row["target_member"] for row in rows]
            assert len(targets) == len(set(targets)) == 2
            assert set(targets).issubset(eligible)


def test_branch_sharing_preserves_36_conceptual_branches(registry):
    assert registry["cell_count"] == 18
    assert registry["conceptual_branch_count"] == 36
    assert registry["conceptual_source_slot_count"] == 72
    assert registry["deduplicated_branch_count"] == 20
    assert registry["actual_source_slot_budget"] == 40
    assert registry["source_candidates_per_target"] == 2
    assert registry["loss_blind_revision_per_valid_source"] == 1


def test_phase_a_preflight_is_zero_api_and_member_aware(registry, scratch):
    module = load("v17_hybrid_preflight", "scripts/preflight_v17_hybrid_target_allocation.py")
    result = module.preflight(registry, scratch)
    assert result["status"] == "PASS", result["errors"]
    assert result["prospective_parents_frozen"] == 6
    assert result["old_diagnostic_parents_excluded_from_primary"] is True
    assert result["api_calls"] == result["model_calls"] == 0
    assert result["validation_calls"] == result["test_calls"] == 0
    assert len(result["context_checks"]) == 20
    assert all(row["context_type"] == "PeerStateDiagnosisContext" for row in result["context_checks"])
    assert all(row["assigned_hash_count"] > 0 for row in result["context_checks"])
    assert all(result["selector_invariants"].values())


def test_frozen_classifier_rules_cover_allowed_outcomes():
    support = load("v17_hybrid_support_classifier", "scripts/v17_hybrid_target_allocation_support.py")

    def parents(hybrid_vote, w1_vote, hybrid_oracle=None, w1_oracle=None):
        hybrid_oracle = hybrid_vote if hybrid_oracle is None else hybrid_oracle
        w1_oracle = w1_vote if w1_oracle is None else w1_oracle
        return [
            {
                support.W1: {"validation_vote_delta": w1_vote[i], "validation_oracle_delta": w1_oracle[i]},
                support.RR: {"validation_vote_delta": 0, "validation_oracle_delta": 0},
                support.HYBRID: {"validation_vote_delta": hybrid_vote[i], "validation_oracle_delta": hybrid_oracle[i]},
            }
            for i in range(3)
        ]

    assert support.classify(parents([1, 1, 0], [0, 0, 0]), {support.W1: 1, support.RR: 3, support.HYBRID: 2})["final_pilot_diagnosis"] == "HYBRID_RECOVERY_SUPPORTED"
    assert support.classify(parents([0, 0, 0], [0, 0, 0]), {support.W1: 1, support.RR: 3, support.HYBRID: 2})["final_pilot_diagnosis"] == "HYBRID_THROUGHPUT_ONLY"
    assert support.classify(parents([1, 1, 0], [0, 0, 0]), {support.W1: 2, support.RR: 3, support.HYBRID: 2})["final_pilot_diagnosis"] == "HYBRID_VALUE_ONLY"
    assert support.classify(parents([-1, -1, 0], [0, 0, 0]), {support.W1: 1, support.RR: 3, support.HYBRID: 1})["final_pilot_diagnosis"] == "HYBRID_HARMFUL"


def test_runner_freezes_decisions_before_validation_and_has_no_test_path():
    source = (ROOT / "scripts/run_v17_hybrid_target_allocation_pilot.py").read_text(encoding="utf-8")
    assert "decisions[arm]" in source
    assert source.index("decisions[arm]") < source.index("parent_validation, validation_by_candidate")
    assert '"decision_frozen_before_validation": True' in source
    assert "evaluate_final_test" not in source
    assert "test_path" not in source
    assert '"team_prompt_commit_count": 0' in source
    assert '"trajectory_mutation_count": 0' in source


def test_candidate_logging_schema_is_explicit_and_sanitized():
    source = (ROOT / "scripts/run_v17_hybrid_target_allocation_pilot.py").read_text(encoding="utf-8")
    for field in (
        "candidate_id", "candidate_stage", "valid", "feasible",
        "common_safe_outcome", "sanitized_rejection_reasons", "train",
        "validation", "branch_rank", "cell_rank", "selected_as_cell_winner",
        "would_commit_contribution",
    ):
        assert f'"{field}"' in source
    assert '"prompt": row.prompt' not in source
    assert '"question"' not in source[source.index("candidates.append({"):source.index("payload = {")]


def test_no_commit_realized_delta_is_zero():
    support = load("v17_hybrid_support_delta", "scripts/v17_hybrid_target_allocation_support.py")
    assert support.realized_delta(False, 10, 15) == 0
    assert support.realized_delta(True, 10, 15) == 5
