from copy import deepcopy

from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    OracleRepairOpportunity,
    ResponsibilityState,
    assign_primary_responsibilities,
    compute_oracle_repair_opportunity,
    select_target_agent,
)


def team(answers, question_hash="q"):
    return build_team_vote_state(
        question_hash=question_hash,
        gold_answer="A",
        answers=answers,
        normalize_answer=str.upper,
        match_answer=lambda left, right: left == right,
        tie_break="abstain",
    )


def opportunities(answers, question_hash="q"):
    current = team(answers, question_hash)
    contexts = {agent: build_peer_vote_context(current, agent) for agent in range(5)}
    rows = tuple(
        compute_oracle_repair_opportunity(team_state=current, peer_context=contexts[agent])
        for agent in range(5)
    )
    return current, contexts, rows


def assignment_inputs(answers, question_hash="q"):
    current, contexts, rows = opportunities(answers, question_hash)
    return {question_hash: current}, {question_hash: contexts}, {question_hash: rows}


def test_oracle_opportunity_is_not_candidate_marginal_contribution():
    _, _, rows = opportunities(["A", "A", "B", "B", "B"])
    assert isinstance(rows[2], OracleRepairOpportunity)
    assert not isinstance(rows[2], CandidateMarginalContribution)
    assert rows[2].direct_vote_fix is True
    assert rows[2].oracle_soft_utility_gain > 0
    assert rows[2].dominant_wrong_member is True


def test_c0_coverage_unique_and_pivotal_opportunities():
    _, _, c0 = opportunities(["B", "B", "C", "C", "D"])
    _, _, unique = opportunities(["A", "B", "B", "C", "C"])
    _, _, pivotal = opportunities(["A", "A", "A", "B", "B"])
    assert c0[0].coverage_opportunity is True
    assert unique[0].unique_correct is True
    assert pivotal[0].pivotal_correct is True


def test_responsibility_is_assigned_only_to_currently_wrong_agents():
    teams, contexts, rows = assignment_inputs(["A", "A", "B", "B", "B"])
    state = ResponsibilityState(
        assigned_load_by_agent={agent: 0 for agent in range(5)},
        updates_since_selected_by_agent={agent: 0 for agent in range(5)},
    )
    owners, assigned = assign_primary_responsibilities(
        team_states=teams, peer_contexts=contexts, opportunities=rows, state=state,
    )
    assert owners["q"] in {2, 3, 4}
    assert not assigned[0] and not assigned[1]


def test_c0_balancing_is_deterministic_and_keeps_legal_owner():
    teams, contexts, rows = assignment_inputs(["B", "B", "C", "C", "D"])
    state = ResponsibilityState(
        primary_owner_by_question={"q": 4},
        owner_age_by_question={"q": 3},
        assigned_load_by_agent={agent: 0 for agent in range(5)},
        updates_since_selected_by_agent={agent: agent for agent in range(5)},
    )
    owners, _ = assign_primary_responsibilities(
        team_states=teams, peer_contexts=contexts, opportunities=rows, state=state,
    )
    assert owners == {"q": 4}
    assert state.owner_age_by_question["q"] == 4
    copied = deepcopy(state)
    again, _ = assign_primary_responsibilities(
        team_states=teams, peer_contexts=contexts, opportunities=rows, state=copied,
    )
    assert again == owners
    assert copied.owner_age_by_question["q"] == 5


def test_owner_inertia_and_owner_age_raise_switch_threshold():
    teams, contexts, rows = assignment_inputs(["A", "A", "B", "B", "B"])
    state = ResponsibilityState(
        primary_owner_by_question={"q": 3},
        owner_age_by_question={"q": 8},
        assigned_load_by_agent={agent: 0 for agent in range(5)},
        updates_since_selected_by_agent={agent: 0 for agent in range(5)},
    )
    owners, _ = assign_primary_responsibilities(
        team_states=teams,
        peer_contexts=contexts,
        opportunities=rows,
        state=state,
        switch_margin=0.05,
    )
    assert owners["q"] == 3
    assert state.owner_age_by_question["q"] == 9


def test_max_wait_fairness_overrides_residual_pressure():
    _, _, rows = opportunities(["A", "A", "B", "B", "B"])
    state = ResponsibilityState(
        assigned_load_by_agent={agent: 0 for agent in range(5)},
        updates_since_selected_by_agent={0: 0, 1: 8, 2: 0, 3: 0, 4: 0},
    )
    assert select_target_agent({2: [rows[2]], 1: []}, state, max_wait_updates=8) == 1
