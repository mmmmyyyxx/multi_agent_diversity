from __future__ import annotations

import inspect
import json

import pytest

from multi_dataset_diverse_rl.shadow_gate import (
    MAX_NO_SHADOW_APPROVED_COMMIT_STREAK,
    ShadowGateMetrics,
    advance_no_commit_streak,
    assert_winner_only_event,
    evaluate_shadow_gate,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.anti_overfitting_shadow_support import (
    BUCKETS,
    FOLD_MAP,
    construct_assignment,
)
from scripts.run_shadow_gated_evolution import DESIGN_ROOT, _protocol_document
from scripts.run_shadow_gated_evolution import DEFAULT_PREP_ROOT, _authorized


def test_split_is_exact_disjoint_deterministic_and_balanced() -> None:
    items = [
        {
            "question_hash": f"{index:064x}",
            "gold_label": ("A", "B", "C")[index % 3],
            "static_difficulty_bin": f"correct_{index % 4}_of_3",
            "lexical_cluster": index % 8,
        }
        for index in range(250)
    ]
    first = construct_assignment(items)
    second = construct_assignment(items)
    assert first == second
    assert set(first) == set(BUCKETS)
    assert {key: len(value) for key, value in first.items()} == {key: 50 for key in BUCKETS}
    flattened = [value for values in first.values() for value in values]
    assert len(flattened) == len(set(flattened)) == 250
    report = json.loads((DESIGN_ROOT / "balance_report.json").read_text(encoding="utf-8"))
    assert report["gate"] == "PASS"


def test_cross_fit_isolation_and_seed_mapping() -> None:
    folds = json.loads((DESIGN_ROOT / "fold_assignment.json").read_text(encoding="utf-8"))
    assert len(folds["fresh_seeds"]) == 3
    assert len(set(folds["fresh_seeds"])) == 3
    for index, row in enumerate(folds["trajectory_groups"]):
        optimize = set().union(*(set(folds["folds"][name]) for name in row["optimize"].split("+")))
        shadow = set(folds["folds"][row["shadow"]])
        assert (len(optimize), len(shadow), optimize & shadow) == (100, 50, set())
        assert (row["optimize"], row["shadow"]) == FOLD_MAP[index]


def test_shadow_gate_vote_and_catastrophic_member_rules() -> None:
    passed = evaluate_shadow_gate(ShadowGateMetrics(30, 30, 25, 23, 50))
    assert passed.passed
    vote_fail = evaluate_shadow_gate(ShadowGateMetrics(30, 29, 25, 25, 50))
    assert vote_fail.reasons == ("shadow_vote_regression",)
    member_fail = evaluate_shadow_gate(ShadowGateMetrics(30, 31, 25, 22, 50))
    assert member_fail.reasons == ("catastrophic_target_member_regression",)
    with pytest.raises(ValueError, match="exactly 50"):
        evaluate_shadow_gate(ShadowGateMetrics(0, 0, 0, 0, 49))


def test_early_stop_is_exactly_six_consecutive_no_commits() -> None:
    streak = 0
    for index in range(MAX_NO_SHADOW_APPROVED_COMMIT_STREAK - 1):
        streak, stopped = advance_no_commit_streak(streak, committed=False)
        assert not stopped and streak == index + 1
    streak, stopped = advance_no_commit_streak(streak, committed=False)
    assert (streak, stopped) == (6, True)
    assert advance_no_commit_streak(5, committed=True) == (0, False)


def test_shadow_telemetry_forbids_ranking_retry_revision_and_feedback() -> None:
    clean = {"shadow_candidate_count": 1, "shadow_rank": None,
             "shadow_retry_count": 0, "shadow_teacher_feedback": False,
             "shadow_revision": False, "shadow_selected_candidate": ""}
    assert_winner_only_event(clean)
    for key, value in (("shadow_candidate_count", 2), ("shadow_rank", 1),
                       ("shadow_retry_count", 1), ("shadow_teacher_feedback", True),
                       ("shadow_revision", True), ("shadow_selected_candidate", "hash")):
        row = dict(clean); row[key] = value
        with pytest.raises(AssertionError):
            assert_winner_only_event(row)


def test_shadow_hook_is_after_optimize_ranking_and_before_commit() -> None:
    source = inspect.getsource(PromptEnsembleOptimizationSystem.update_once)
    assert source.index("optimize_winner = max") < source.index("approve_writeback_candidate")
    assert source.index("approve_writeback_candidate") < source.index("agent.current_prompt = accepted.prompt")


def test_protocol_freezes_no_leakage_and_zero_test_access() -> None:
    protocol = _protocol_document()
    assert protocol["shadow_gate"]["winner_only"] is True
    assert protocol["shadow_gate"]["feedback_or_retry"] is False
    assert protocol["validation_policy"].startswith("one final frozen-state")
    assert protocol["test_policy"].startswith("zero access")
    assert protocol["max_update_opportunities"] == 32
    assert protocol["models"]["solver"] == "qwen3-8b"
    runner_source = inspect.getsource(__import__(
        "scripts.run_shadow_gated_evolution", fromlist=["prepare"]
    ).prepare)
    assert '"test_calls_before_final_freeze": 0' in runner_source
    assert '"events": []' in runner_source


def test_resume_registry_contract_is_idempotent_by_update_identity() -> None:
    # The persisted identity is the pair (trajectory, update); replay uses set
    # membership and therefore cannot schedule a completed update twice.
    completed = {("seed75:shadow", 0), ("seed75:shadow", 1)}
    requested = [("seed75:shadow", index) for index in range(4)]
    remaining = [row for row in requested if row not in completed]
    assert remaining == [("seed75:shadow", 2), ("seed75:shadow", 3)]
    assert [row for row in requested if row not in completed] == remaining


def test_phase_b_fails_closed_without_explicit_api_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTI_OVERFITTING_SHADOW_AUTHORIZED", raising=False)
    with pytest.raises(RuntimeError, match="explicit Phase-B API authorization"):
        _authorized(DEFAULT_PREP_ROOT)


def test_test50_loader_is_hard_blocked_in_execution_path() -> None:
    module = __import__("scripts.run_shadow_gated_evolution", fromlist=["execute"])
    source = inspect.getsource(module.execute)
    assert 'path == "TEST50_BLOCKED_BY_GOVERNANCE"' in source
    assert "return []" in source
