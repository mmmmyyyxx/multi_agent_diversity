from dataclasses import replace

from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.responsibility import (
    ResponsibilityState,
    build_target_selection_decision,
    compute_member_aware_repair_opportunity,
    compute_repair_eligibility_frontiers,
    compute_unique_owner_control_portfolios,
    target_priorities,
)


def team(answers, question_hash="q"):
    return build_team_vote_state(question_hash=question_hash, gold_answer="A", answers=answers,
                                 normalize_answer=str.upper, match_answer=lambda left, right: left == right,
                                 tie_break="abstain")


def opportunities(state):
    return tuple(compute_member_aware_repair_opportunity(
        team_state=state, peer_context=build_peer_vote_context(state, agent)) for agent in range(5))


def runtime(**overrides):
    value = {
        "updates_since_selected_by_agent": {agent: 0 for agent in range(5)},
        "accepted_updates_by_agent": {agent: 0 for agent in range(5)},
        "target_attempt_count_by_agent": {agent: 0 for agent in range(5)},
    }
    value.update(overrides)
    return ResponsibilityState(**value)


def test_repair_vector_excludes_member_fairness_and_history():
    row = opportunities(team(["A", "A", "B", "B", "B"]))[2]
    assert row.repair_vector() == (float(row.direct_vote_fix), row.oracle_soft_utility_gain,
                                   float(row.coverage_opportunity), float(row.dominant_wrong_member))
    assert not any("gain" in name or "wait" in name for name in row.__dict__ if name != "oracle_soft_utility_gain")


def test_frontier_keeps_all_nondominated_wrong_members_and_excludes_correct_members():
    state = team(["A", "A", "B", "B", "B"])
    rows = list(opportunities(state))
    rows[2] = replace(rows[2], direct_vote_fix=True, oracle_soft_utility_gain=0.2)
    rows[3] = replace(rows[3], direct_vote_fix=False, coverage_opportunity=True, oracle_soft_utility_gain=0.3)
    rows[4] = replace(rows[4], oracle_soft_utility_gain=0.0)
    eligible, portfolios, audit = compute_repair_eligibility_frontiers(
        team_states={"q": state}, opportunities={"q": rows}, state=runtime(), current_update_index=0)
    assert eligible["q"] == (2, 3)
    assert [row.question_hash for row in portfolios[2]] == ["q"]
    assert [row.question_hash for row in portfolios[3]] == ["q"]
    assert not portfolios[0] and not portfolios[1] and not portfolios[4]
    assert audit["q"]["candidate_pareto_fronts"]["4"] > 1


def test_frontier_is_invariant_to_member_fairness_and_wait_state():
    state = team(["B", "B", "B", "B", "B"])
    rows = opportunities(state)
    first, _, _ = compute_repair_eligibility_frontiers(
        team_states={"q": state}, opportunities={"q": rows}, state=runtime(), current_update_index=0)
    second, _, _ = compute_repair_eligibility_frontiers(
        team_states={"q": state}, opportunities={"q": rows},
        state=runtime(updates_since_selected_by_agent={agent: 99 for agent in range(5)},
                      accepted_updates_by_agent={agent: 99 for agent in range(5)}), current_update_index=0)
    assert first == second


def test_unique_owner_control_selects_one_legal_frontier_member_per_residual():
    state = team(["A", "A", "B", "B", "B"])
    rows = list(opportunities(state))
    rows[2] = replace(rows[2], direct_vote_fix=True, oracle_soft_utility_gain=0.2)
    rows[3] = replace(rows[3], coverage_opportunity=True, oracle_soft_utility_gain=0.3)
    eligible, portfolios, audit = compute_unique_owner_control_portfolios(
        team_states={"q": state}, opportunities={"q": rows}, state=runtime(),
        seed=42, current_update_index=0,
    )
    assert len(eligible["q"]) == 1
    selected = eligible["q"][0]
    assert audit["q"]["candidate_pareto_fronts"][str(selected)] == 1
    assert [row.question_hash for row in portfolios[selected]] == ["q"]
    assert audit["q"]["control_selected_agent_id"] == selected


def test_unique_owner_control_scheduler_ignores_uplift_deficit():
    state = team(["B", "B", "B", "B", "B"])
    rows = opportunities(state)
    assignments = {0: [rows[0]], 1: [rows[1]], 2: [], 3: [], 4: []}
    runtime_state = runtime(updates_since_selected_by_agent={0: 0, 1: 0, 2: 9, 3: 0, 4: 0})
    priorities = target_priorities(
        assignments=assignments, state=runtime_state, seed=42, max_wait_updates=8,
        current_member_correct_counts=(0, 10, 0, 0, 0),
        initial_member_correct_counts=(0, 0, 0, 0, 0), current_update_index=0,
        member_uplift_tolerance=10**9, responsibility_mode="unique_owner_v6",
    )
    decision = build_target_selection_decision(
        priorities, all_member_gains=(0, 10, 0, 0, 0), state=runtime_state,
        max_wait_updates=8, member_uplift_tolerance=5, member_catchup_mode="off",
        responsibility_mode="unique_owner_v6",
    )
    assert decision.update_lane == "responsibility_conditioned"
    assert set(decision.actual_candidate_agent_ids) == {0, 1}


def test_scheduler_joint_frontier_and_overdue_responsibility_precedence():
    state = team(["B", "B", "B", "B", "B"])
    rows = opportunities(state)
    assignments = {0: [rows[0]], 1: [rows[1]], 2: [], 3: [], 4: []}
    runtime_state = runtime(updates_since_selected_by_agent={0: 8, 1: 0, 2: 9, 3: 0, 4: 0})
    priorities = target_priorities(assignments=assignments, state=runtime_state, seed=42, max_wait_updates=8,
                                  current_member_correct_counts=(3, 7, 0, 7, 7),
                                  initial_member_correct_counts=(0, 0, 0, 0, 0), current_update_index=8,
                                  member_uplift_tolerance=5)
    decision = build_target_selection_decision(priorities, all_member_gains=(3, 7, 0, 7, 7),
                                                state=runtime_state, max_wait_updates=8,
                                                member_uplift_tolerance=5, member_catchup_mode="fallback_v1")
    assert decision.update_lane == "responsibility_conditioned"
    assert decision.selection_pool_stage == "responsibility_overdue_frontier"
    assert decision.selected_agent_id == 0
    assert 2 not in decision.actual_candidate_agent_ids


def test_catchup_is_separate_and_only_when_no_responsibility_portfolio_exists():
    state = runtime(updates_since_selected_by_agent={0: 8, 1: 8, 2: 0, 3: 0, 4: 0})
    decision = build_target_selection_decision((), all_member_gains=(0, 10, 10, 10, 10), state=state,
                                                max_wait_updates=8, member_uplift_tolerance=5,
                                                member_catchup_mode="fallback_v1")
    assert decision.update_lane == "generic_member_catchup"
    assert decision.selected_agent_id == 0
