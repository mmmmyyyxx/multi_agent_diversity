from scripts.v18_no_semantic_critic_online import (
    ARMS, NEUTRAL_TOTAL_CORRECT_TOLERANCE, SEEDS, UPDATES, _funnel_counts, classify,
)
from scripts.v18_teacher_critic_pipeline_support import deterministic_hard_gate


def test_frozen_design():
    assert ARMS == ("A_CANONICAL", "C_NO_SEMANTIC_CRITIC")
    assert SEEDS == (68,)
    assert UPDATES == 8
    assert NEUTRAL_TOTAL_CORRECT_TOLERANCE == 1


def test_extension_seeds_are_disjoint_from_seed68():
    assert {69, 70}.isdisjoint(SEEDS)


def test_classifier_precedence_and_boundaries():
    assert classify(commits_a=8, commits_c=10, vote_correct_a=90, vote_correct_c=93, wins=2, losses=1) == "ONLINE_THROUGHPUT_AND_VOTE_SUPPORTED"
    assert classify(commits_a=8, commits_c=10, vote_correct_a=90, vote_correct_c=91, wins=1, losses=1) == "ONLINE_THROUGHPUT_ONLY"
    assert classify(commits_a=8, commits_c=10, vote_correct_a=90, vote_correct_c=88, wins=0, losses=2) == "ONLINE_THROUGHPUT_WITH_TRANSFER_REGRESSION"
    assert classify(commits_a=8, commits_c=8, vote_correct_a=90, vote_correct_c=95, wins=3, losses=0) == "NO_CLEAR_ONLINE_ADVANTAGE"


def test_deterministic_gate_stays_hard_safety_only():
    safe = {"failure_pattern": "Confuses agent and patient roles.", "repair_rule": "Track grammatical roles before deciding.", "preservation_rule": "Preserve explicit ambiguity checks."}
    unsafe = {**safe, "repair_rule": "Rewrite the output format and omit FINAL_ANSWER."}
    assert deterministic_hard_gate(safe)["pass"] is True
    assert deterministic_hard_gate(unsafe)["pass"] is False


def test_runtime_constraint_feasible_is_the_feasible_counter():
    decision = {"branches": [{"funnel": {
        "critic_calls": 1, "critic_approved": 1, "constraint_feasible": 3,
        "stage_b_evaluated": 4,
    }}]}
    assert _funnel_counts(decision)["feasible_candidates"] == 3
