from dataclasses import replace

from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.responsibility import (
    ResponsibilityState,
    assign_primary_responsibilities,
    build_target_selection_decision,
    compute_member_aware_repair_opportunity,
    select_target_agent,
    target_priorities,
)


def team(answers, question_hash="q"):
    return build_team_vote_state(
        question_hash=question_hash, gold_answer="A", answers=answers,
        normalize_answer=str.upper, match_answer=lambda left, right: left == right,
        tie_break="abstain",
    )


def opportunities(answers, question_hash="q"):
    current = team(answers, question_hash)
    return current, tuple(
        compute_member_aware_repair_opportunity(
            team_state=current, peer_context=build_peer_vote_context(current, agent)
        ) for agent in range(5)
    )


def state(**overrides):
    values = {
        "assigned_load_by_agent": {agent: 0 for agent in range(5)},
        "updates_since_selected_by_agent": {agent: 0 for agent in range(5)},
        "accepted_updates_by_agent": {agent: 0 for agent in range(5)},
        "target_attempt_count_by_agent": {agent: 0 for agent in range(5)},
    }
    values.update(overrides)
    return ResponsibilityState(**values)


def assign(current, rows, current_state=None, seed=43, margin=0.05):
    return assign_primary_responsibilities(
        team_states={current.question_hash: current},
        opportunities={current.question_hash: rows}, state=current_state or state(),
        seed=seed, responsibility_switch_margin=margin,
    )


def test_opportunity_contains_only_repair_and_local_preservation_fields():
    _, rows = opportunities(["A", "A", "B", "B", "B"])
    assert rows[2].repair_vector() == (
        float(rows[2].direct_vote_fix), rows[2].oracle_soft_utility_gain,
        float(rows[2].coverage_opportunity), float(rows[2].dominant_wrong_member),
    )
    assert not any(name in {"improvement_need", "gain_count", "relative_rank"} for name in rows[2].__dict__)


def test_assignment_only_covers_vote_wrong_samples_and_wrong_agents():
    wrong_state, wrong_rows = opportunities(["A", "A", "B", "B", "B"])
    owners, assigned, audits = assign(wrong_state, wrong_rows)
    assert owners["q"] in {2, 3, 4}
    assert not assigned[0] and not assigned[1]
    assert audits["q"]["vote_correct"] is False
    correct_state, correct_rows = opportunities(["A", "A", "A", "B", "B"])
    owners, assigned, audits = assign(correct_state, correct_rows)
    assert owners == {} and all(not rows for rows in assigned.values()) and audits == {}


def test_owner_front_and_lexical_priority_are_repair_only():
    current, rows = opportunities(["A", "A", "B", "B", "B"])
    by_id = {row.agent_id: row for row in rows}
    by_id[2] = replace(by_id[2], direct_vote_fix=True, oracle_soft_utility_gain=0.2)
    by_id[3] = replace(by_id[3], direct_vote_fix=False, oracle_soft_utility_gain=0.1)
    by_id[4] = replace(by_id[4], direct_vote_fix=False, oracle_soft_utility_gain=0.0)
    owners, _, audits = assign(current, tuple(by_id[index] for index in range(5)))
    assert audits["q"]["candidate_pareto_fronts"]["2"] == 1
    assert audits["q"]["candidate_pareto_fronts"]["4"] > 1
    assert owners["q"] == 2


def test_owner_inertia_requires_frontier_and_repair_nonweakness():
    current, rows = opportunities(["A", "A", "B", "B", "B"])
    by_id = {row.agent_id: row for row in rows}
    by_id[2] = replace(by_id[2], direct_vote_fix=True, coverage_opportunity=True, oracle_soft_utility_gain=1.0)
    by_id[3] = replace(by_id[3], direct_vote_fix=False, coverage_opportunity=False, oracle_soft_utility_gain=0.0)
    owners, _, audit = assign(
        current, tuple(by_id[index] for index in range(5)),
        current_state=state(primary_owner_by_question={"q": 3}, owner_age_by_question={"q": 4}),
    )
    assert owners["q"] == 2
    assert audit["q"]["chosen_reason"] == "repair_only_pareto_preference"


def test_only_assigned_owners_are_eligible_and_empty_portfolio_is_noop():
    _, rows = opportunities(["B", "B", "B", "B", "B"])
    priorities = target_priorities(assignments={agent: [] for agent in range(5)}, state=state(), seed=42, max_wait_updates=8)
    decision = build_target_selection_decision(priorities)
    assert priorities == () and decision.selected_agent_id is None
    assert decision.no_actionable_reason == "no_actionable_responsibility"
    priorities = target_priorities(assignments={2: (rows[2],), 4: (rows[4],)}, state=state(), seed=42, max_wait_updates=8)
    assert {row.agent_id for row in priorities} == {2, 4}


def test_target_portfolio_front_and_max_wait_apply_only_to_owners():
    _, rows = opportunities(["B", "B", "B", "B", "B"])
    current_state = state(updates_since_selected_by_agent={0: 8, 1: 0, 2: 0, 3: 0, 4: 0})
    priorities = target_priorities(assignments={0: (), 2: (rows[2],), 4: (rows[4],)}, state=current_state, seed=42, max_wait_updates=8)
    assert not any(row.agent_id == 0 for row in priorities)
    assert select_target_agent(priorities) in {2, 4}
    overdue_state = state(updates_since_selected_by_agent={0: 0, 1: 0, 2: 8, 3: 0, 4: 0})
    overdue = target_priorities(assignments={2: (rows[2],), 4: (rows[4],)}, state=overdue_state, seed=42, max_wait_updates=8)
    assert select_target_agent(overdue) == 2


def test_owner_and_target_selection_are_invariant_to_removed_competence_state():
    current, rows = opportunities(["B", "B", "B", "B", "B"])
    first = assign(current, rows, current_state=state())[0]
    second = assign(current, rows, current_state=state(accepted_updates_by_agent={agent: 99 for agent in range(5)}))[0]
    assert first == second
    priorities_a = target_priorities(assignments={2: (rows[2],), 4: (rows[4],)}, state=state(), seed=42, max_wait_updates=8)
    priorities_b = target_priorities(assignments={2: (rows[2],), 4: (rows[4],)}, state=state(accepted_updates_by_agent={agent: 99 for agent in range(5)}), seed=42, max_wait_updates=8)
    assert priorities_a == priorities_b


def test_wait_eight_becomes_overdue_only_at_eight_updates():
    _, rows = opportunities(["B", "B", "B", "B", "B"])
    at_seven = target_priorities(assignments={0: (rows[0],)}, state=state(updates_since_selected_by_agent={0: 7, 1: 0, 2: 0, 3: 0, 4: 0}), seed=42, max_wait_updates=8)
    at_eight = target_priorities(assignments={0: (rows[0],)}, state=state(updates_since_selected_by_agent={0: 8, 1: 0, 2: 0, 3: 0, 4: 0}), seed=42, max_wait_updates=8)
    assert not at_seven[0].overdue and at_eight[0].overdue
