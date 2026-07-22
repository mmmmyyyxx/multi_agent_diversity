from copy import deepcopy

from multi_dataset_diverse_rl.peer_state import build_peer_vote_state
from multi_dataset_diverse_rl.responsibility import (
    ResponsibilityState, assign_primary_responsibilities, counterfactual_credit, select_target_agent,
)


def team(answers, gold="A", question_hash="q"):
    return build_peer_vote_state(
        question_hash=question_hash, gold_answer=gold, answers=answers,
        normalize_answer=str.upper, match_answer=lambda left, right: left == right,
        tie_break="first", seed=42,
    )


def credit(answers, agent_id, question_hash="q"):
    return counterfactual_credit(
        agent_id=agent_id, current_state=team(answers, question_hash=question_hash), gold_answer="A",
        normalize_answer=str.upper, match_answer=lambda left, right: left == right,
        tie_break="first", seed=42,
    )


def test_direct_fix_soft_gain_and_dominant_wrong_membership():
    row = credit(["A", "A", "B", "B", "B"], 2)
    assert row.direct_vote_fix == 1
    assert row.fix_soft_utility_gain > 0
    assert row.dominant_wrong_member is True


def test_same_gold_count_different_wrong_cluster_changes_counterfactual_credit():
    concentrated = credit(["A", "A", "B", "B", "B"], 2)
    dispersed = credit(["A", "A", "B", "C", "D"], 2)
    assert concentrated.fix_soft_utility_gain != dispersed.fix_soft_utility_gain


def test_coverage_unique_correct_and_leave_one_out_pivotal():
    coverage = credit(["B", "B", "C", "C", "D"], 0)
    unique = credit(["A", "B", "B", "C", "C"], 0)
    pivotal = credit(["A", "A", "B", "B", "C"], 0)
    assert coverage.coverage_opportunity is True
    assert coverage.fix_soft_utility_gain > 0
    assert unique.unique_correct is True
    assert pivotal.pivotal_vote_correct is True


def test_assignment_is_deterministic_and_inertial():
    rows = {"q": [credit(["B", "B", "C", "C", "D"], agent, "q") for agent in range(5)]}
    state = ResponsibilityState(agent_updates_since_last_selected={agent: 0 for agent in range(5)})
    first, _ = assign_primary_responsibilities(rows, state)
    copied = deepcopy(state)
    second, _ = assign_primary_responsibilities(rows, copied)
    assert second == first
    assert copied.responsibility_age_by_question["q"] == 1


def test_owner_switch_requires_direct_fix_or_margin():
    rows = {"q": [credit(["A", "A", "B", "B", "B"], agent, "q") for agent in (2, 3, 4)]}
    state = ResponsibilityState(
        previous_primary_owner_by_question={"q": 3},
        agent_updates_since_last_selected={agent: 0 for agent in range(5)},
    )
    owners, _ = assign_primary_responsibilities(rows, state, switch_margin=0.05)
    assert owners["q"] == 3


def test_c0_keeps_legal_previous_owner_before_counterfactual_ties():
    rows = {"q": [credit(["B", "B", "C", "C", "D"], agent, "q") for agent in range(5)]}
    state = ResponsibilityState(
        previous_primary_owner_by_question={"q": 4},
        agent_updates_since_last_selected={agent: agent for agent in range(5)},
    )
    owners, _ = assign_primary_responsibilities(rows, state)
    assert owners["q"] == 4


def test_max_wait_fairness_and_resume_state_are_exact():
    assignments = {0: [credit(["A", "A", "B", "B", "B"], 2)], 1: []}
    waits = {0: 0, 1: 8}
    assert select_target_agent(assignments, waits, max_wait_updates=8) == 1
    restored = ResponsibilityState(**deepcopy(ResponsibilityState(
        previous_primary_owner_by_question={"q": 2}, responsibility_age_by_question={"q": 3},
        agent_updates_since_last_selected={0: 1, 1: 4}, assigned_load_per_agent={0: 0, 1: 1},
    ).__dict__))
    assert restored.previous_primary_owner_by_question == {"q": 2}
    assert restored.responsibility_age_by_question == {"q": 3}
