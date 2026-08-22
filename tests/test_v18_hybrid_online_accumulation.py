from __future__ import annotations

import inspect
from pathlib import Path

from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.build_v18_hybrid_online_accumulation_registry import build_registry
from scripts.preflight_v18_hybrid_online_accumulation import preflight
from scripts.run_v18_hybrid_online_accumulation import (
    HybridOnlineSystem,
    _candidate_rows,
)
from scripts.v18_hybrid_online_accumulation_support import (
    HYBRID,
    W1,
    build_residual_lineage,
    classify,
    generation_key,
    hybrid_targets,
    summarize_trajectory,
)


def _example(g, h, members, *, vote=None):
    return {
        "example_id_hash": "e",
        "G": g,
        "H": h,
        "M": g - h,
        "vote_correct": (g - h > 0) if vote is None else vote,
        "oracle_covered": g > 0,
        "correct_member_ids": members,
        "correct_member_count": g,
        "responsibility_eligible_member_ids": [1, 2],
    }


def _state(index, g, h, members):
    return {
        "state_index": index,
        "after_update_index": index - 1,
        "metrics": {
            "vote_correct_count": int(g - h > 0),
            "plurality_vote_acc": float(g - h > 0),
            "oracle_correct_count": int(g > 0),
            "oracle_acc": float(g > 0),
        },
        "examples": [_example(g, h, members)],
    }


def test_registry_recovers_canonical_horizon_and_unused_consecutive_seeds():
    registry = build_registry()
    assert registry["seeds"] == [59, 60, 61]
    assert registry["seed_freeze"]["gate"] == "PASS"
    assert registry["update_opportunities_per_trajectory"] == 8
    assert registry["source_candidates_per_target"] == 2
    assert registry["loss_blind_revision_per_valid_source"] == 1
    assert registry["new_test_calls"] == 0


def test_all_phase_a_gates_pass_without_api():
    gate = preflight(build_registry())
    assert gate["gate"] == "PASS"
    assert set(gate["gates"].values()) == {"PASS"}
    assert gate["api_calls"] == gate["model_calls"] == 0


def test_hybrid_selector_is_exact_w1_rank1_plus_frozen_rr():
    assert hybrid_targets(
        seed=59,
        update_index=0,
        w1_order=[4, 1, 3, 2],
        responsibility_eligible=[1, 2, 3, 4],
    ) == (4, 1)
    source = inspect.getsource(HybridOnlineSystem.select_targets)
    assert "super().select_targets" in source
    assert "hybrid_targets" in source
    assert "breadth" not in source.lower()
    assert "direct_flip" not in source.lower()
    assert PromptEnsembleOptimizationSystem.select_targets is not HybridOnlineSystem.select_targets


def test_generation_key_is_deterministic_and_parent_specific():
    args = dict(
        experiment_seed=59,
        update_index=2,
        target_member=3,
        source_slot=1,
        candidate_stage="source",
        parent_team_hash="a" * 64,
    )
    assert generation_key(**args) == generation_key(**args)
    assert generation_key(**args) != generation_key(**{**args, "parent_team_hash": "b" * 64})


def test_longitudinal_recovery_deepening_cross_member_and_vote_conversion():
    states = [
        _state(0, 0, 3, []),
        _state(1, 1, 3, [0]),
        _state(2, 2, 1, [0, 2]),
    ]
    lineage = build_residual_lineage(states)
    assert len(lineage) == 1
    row = lineage[0]
    assert row["first_0_to_1_state"] == 1
    assert row["later_1_to_2_state"] == 2
    assert row["cross_member_accumulation"] is True
    assert row["first_vote_conversion_state"] == 2
    assert row["persistent_singleton"] is False
    summary = summarize_trajectory(
        states=states,
        accepted_commit_count=2,
        feasible_branch_count=2,
        feasible_candidate_count=3,
        update_opportunities=8,
    )
    assert summary["recovered_singleton_count"] == 1
    assert summary["longitudinal_deepened_coverage_count"] == 1
    assert summary["recovered_coverage_to_vote_count"] == 1
    assert summary["support_transitions"]["0_to_1"] == 1
    assert summary["support_transitions"]["1_to_2"] == 1


def test_persistent_singleton_and_undefined_rate_are_not_fabricated():
    states = [_state(0, 0, 3, []), _state(1, 1, 3, [1]), _state(2, 1, 2, [1])]
    row = build_residual_lineage(states)[0]
    assert row["persistent_singleton"] is True
    summary = summarize_trajectory(
        states=states,
        accepted_commit_count=2,
        feasible_branch_count=2,
        feasible_candidate_count=2,
        update_opportunities=8,
    )
    assert summary["persistent_singleton_count"] == 1
    assert summary["deepening_rate"] == 0.0
    no_recovery = summarize_trajectory(
        states=[_state(0, 2, 1, [0, 1])],
        accepted_commit_count=0,
        feasible_branch_count=0,
        feasible_candidate_count=0,
        update_opportunities=8,
    )
    assert no_recovery["deepening_rate"] is None


def test_frozen_classifier_uses_only_three_matched_seed_differences():
    rows = []
    for seed in (59, 60, 61):
        base = {
            "seed": seed,
            "longitudinal_deepened_coverage_count": 0,
            "recovered_coverage_to_vote_count": 0,
            "accepted_commit_count": 1,
            "feasible_branch_count": 1,
            "deepening_rate": None,
            "deepening_rate_support": 0,
            "final_validation_vote_acc": 0.5,
            "final_validation_oracle_acc": 0.7,
        }
        rows.append({**base, "arm": W1})
        rows.append({
            **base,
            "arm": HYBRID,
            "longitudinal_deepened_coverage_count": 1,
            "recovered_coverage_to_vote_count": 1,
            "accepted_commit_count": 2,
            "feasible_branch_count": 2,
            "final_validation_vote_acc": 0.52,
            "final_validation_oracle_acc": 0.72,
        })
    result = classify(rows)
    assert result["online_accumulation_supported"] is True
    assert result["online_vote_conversion_signal"] is True
    assert result["final_diagnosis"] == "LONGITUDINAL_ACCUMULATION_WITH_VOTE_CONVERSION"


def test_candidate_logging_is_sanitized_and_preserves_common_safe_outcome():
    decision = {
        "parent_team_hash": "a" * 64,
        "accepted_prompt_hash": "p1",
        "branches": [{"target_agent_id": 1, "branch_winner_hash": "p1"}],
        "candidates": [{
            "target_agent_id": 1,
            "target_selection_rank": 1,
            "prompt_hash": "p1",
            "candidate_stage": "m20_source",
            "evaluation": {"prompt": "private"},
            "constraint": {
                "passed": True,
                "rejection_reasons": [],
                "target_gain": 1,
                "vote_gain_count": 0,
                "vote_loss_count": 0,
                "vote_net_gain": 0,
            },
        }],
    }
    rows = _candidate_rows(seed=59, arm=W1, update_index=0, decision=decision)
    assert rows[0]["common_safe_outcome"] == "passed"
    assert rows[0]["winner"] is True
    forbidden = {"prompt", "question", "answer", "response", "endpoint"}
    assert not forbidden.intersection(rows[0])


def test_runner_freezes_validation_schedule_and_forbids_test_evaluation():
    source = Path("scripts/run_v18_hybrid_online_accumulation.py").read_text(encoding="utf-8")
    assert "if accepted:" in source
    assert "_validation_snapshot" in source
    assert "validation_used_for_selection\": False" in source
    assert "system.test_evaluation_count != 0" in source
    assert "evaluate_final_test" not in source
    assert "responsibility_refresh_count" in source
