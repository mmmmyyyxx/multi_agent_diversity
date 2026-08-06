from dataclasses import replace

from multi_dataset_diverse_rl.peer_state import (
    build_peer_vote_context,
    build_team_vote_state,
)
from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity,
    RepairLane,
    ResponsibilityState,
    build_target_selection_decision,
    compute_member_aware_repair_opportunity,
    compute_repair_eligibility_sets,
    initialize_legacy_freeze_state,
    record_target_update_acceptance,
    record_target_update_failure,
    repair_eligibility_key,
    refresh_frozen_member_states,
    responsibility_portfolios,
    target_priorities,
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


def runtime(**overrides):
    values = {
        "eligible_agents_by_question": {},
        "updates_since_selected_by_agent": {
            agent: 0 for agent in range(5)
        },
        "accepted_updates_by_agent": {agent: 0 for agent in range(5)},
        "seeded_rank_by_agent": {},
        "target_attempt_count_by_agent": {
            agent: 0 for agent in range(5)
        },
    }
    values.update(overrides)
    return ResponsibilityState(**values)


def opportunity(
    agent_id,
    *,
    vote_flip_gain=0,
    margin_gain=0,
    member_error=True,
    coverage=False,
    conversion=False,
    dominant=False,
    utility=0.0,
    question_hash="q",
):
    return MemberAwareRepairOpportunity(
        agent_id=agent_id,
        question_hash=question_hash,
        vote_flip_gain=vote_flip_gain,
        margin_gain=margin_gain,
        member_error=member_error,
        coverage_opportunity=coverage,
        conversion_opportunity=conversion,
        dominant_wrong_member=dominant,
        unique_correct=not member_error,
        pivotal_correct=False,
        oracle_soft_utility_gain=utility,
    )


def eligibility(rows, state=None):
    vote_wrong = state or team(["B", "B", "C", "C", "D"])
    return compute_repair_eligibility_sets(
        team_states={"q": vote_wrong},
        opportunities={"q": tuple(rows)},
        state=runtime(),
    )


def test_counterfactual_opportunity_reports_vote_and_margin_gains():
    state = team(["A", "B", "B", "C", "D"])
    row = compute_member_aware_repair_opportunity(
        team_state=state,
        peer_context=build_peer_vote_context(state, 1),
    )
    assert row.member_error
    assert row.vote_flip_gain == 1
    assert row.margin_gain == 2
    assert repair_eligibility_key(row) == (1, 2)


def test_direct_vote_flip_precedes_larger_nonflip_margin_gain():
    rows = [
        opportunity(0, vote_flip_gain=1, margin_gain=1),
        opportunity(1, vote_flip_gain=0, margin_gain=9),
    ]
    eligible, portfolios, audit = eligibility(rows)
    assert eligible["q"] == (0,)
    assert [row.agent_id for row in portfolios[0]] == [0]
    assert audit["q"]["candidate_counterfactual_values"]["1"] == {
        "vote_flip_gain": 0,
        "margin_gain": 9,
    }


def test_margin_gain_breaks_equal_vote_flip_and_exact_ties_are_retained():
    rows = [
        opportunity(0, vote_flip_gain=1, margin_gain=1),
        opportunity(1, vote_flip_gain=1, margin_gain=2),
        opportunity(2, vote_flip_gain=1, margin_gain=2),
    ]
    eligible, _, audit = eligibility(rows)
    assert eligible["q"] == (1, 2)
    assert audit["q"]["eligibility_tie_count"] == 2


def test_diagnostic_labels_do_not_change_eligibility():
    rows = [
        opportunity(
            0,
            vote_flip_gain=1,
            margin_gain=2,
            coverage=False,
            dominant=False,
            utility=0.0,
        ),
        opportunity(
            1,
            vote_flip_gain=1,
            margin_gain=2,
            coverage=True,
            conversion=True,
            dominant=True,
            utility=100.0,
        ),
    ]
    eligible, _, _ = eligibility(rows)
    assert eligible["q"] == (0, 1)


def test_member_state_and_history_do_not_change_eligibility():
    rows = [
        opportunity(0, vote_flip_gain=0, margin_gain=1),
        opportunity(1, vote_flip_gain=0, margin_gain=1),
    ]
    first_state = runtime()
    second_state = runtime(
        updates_since_selected_by_agent={
            0: 99,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        },
        accepted_updates_by_agent={
            0: 10,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        },
    )
    first, _, _ = compute_repair_eligibility_sets(
        team_states={"q": team(["B", "B", "C", "C", "D"])},
        opportunities={"q": rows},
        state=first_state,
    )
    second, _, _ = compute_repair_eligibility_sets(
        team_states={"q": team(["B", "B", "C", "C", "D"])},
        opportunities={"q": rows},
        state=second_state,
    )
    assert first == second == {"q": (0, 1)}


def test_correct_member_and_vote_correct_example_do_not_generate_responsibility():
    wrong_state = team(["B", "B", "C", "C", "D"])
    eligible, _, _ = compute_repair_eligibility_sets(
        team_states={"q": wrong_state},
        opportunities={
            "q": [
                opportunity(
                    0,
                    vote_flip_gain=9,
                    margin_gain=9,
                    member_error=False,
                ),
                opportunity(1, vote_flip_gain=0, margin_gain=1),
            ]
        },
        state=runtime(),
    )
    assert eligible == {"q": (1,)}

    correct_state = team(["A", "A", "A", "B", "C"])
    eligible, portfolios, audit = compute_repair_eligibility_sets(
        team_states={"q": correct_state},
        opportunities={
            "q": [opportunity(agent) for agent in range(5)]
        },
        state=runtime(),
    )
    assert eligible == {}
    assert all(not rows for rows in portfolios.values())
    assert audit == {}


def test_portfolio_formal_aggregates_are_only_direct_fix_and_margin():
    rows = [
        opportunity(
            0,
            vote_flip_gain=1,
            margin_gain=2,
            coverage=True,
            conversion=True,
            dominant=True,
            utility=100.0,
            question_hash="q1",
        ),
        opportunity(
            0,
            vote_flip_gain=0,
            margin_gain=3,
            question_hash="q2",
        ),
    ]
    portfolio = responsibility_portfolios(
        assignments={0: rows},
        state=runtime(),
    )[0]
    assert portfolio.direct_fix_count == 1
    assert portfolio.margin_gain_sum == 5
    assert portfolio.residual_count == 2
    assert portfolio.coverage_count == 1
    assert portfolio.conversion_count == 1
    assert portfolio.dominant_wrong_count == 1


def priorities_for(
    assignments,
    *,
    waits=None,
    current=(10, 10, 10, 10, 10),
    initial=(10, 10, 10, 10, 10),
):
    state = runtime(
        updates_since_selected_by_agent=waits
        or {agent: 0 for agent in range(5)}
    )
    rows = target_priorities(
        assignments=assignments,
        state=state,
        seed=7,
        current_member_correct_counts=current,
        initial_member_correct_counts=initial,
        member_uplift_tolerance=5,
    )
    return state, rows


def test_target_vector_is_exactly_direct_margin_and_uplift_deficit():
    assignments = {
        0: [opportunity(0, vote_flip_gain=1, margin_gain=2)],
        1: [opportunity(1, vote_flip_gain=0, margin_gain=4)],
    }
    _, rows = priorities_for(
        assignments,
        current=(10, 16, 10, 10, 10),
    )
    by_agent = {row.agent_id: row for row in rows}
    assert by_agent[0].target_values() == (1.0, 2.0, 1.0)
    assert by_agent[1].target_values() == (0.0, 4.0, 0.0)
    assert {row.target_pareto_front for row in rows} == {1}


def test_diagnostic_portfolio_fields_do_not_change_target_front():
    base = opportunity(0, vote_flip_gain=1, margin_gain=2)
    diagnostic = replace(
        base,
        coverage_opportunity=True,
        conversion_opportunity=True,
        dominant_wrong_member=True,
        oracle_soft_utility_gain=999.0,
    )
    _, first = priorities_for({0: [base], 1: [
        opportunity(1, vote_flip_gain=1, margin_gain=2)
    ]})
    _, second = priorities_for({0: [diagnostic], 1: [
        opportunity(1, vote_flip_gain=1, margin_gain=2)
    ]})
    assert [
        (row.agent_id, row.target_values(), row.target_pareto_front)
        for row in first
    ] == [
        (row.agent_id, row.target_values(), row.target_pareto_front)
        for row in second
    ]


def test_frontier_tie_break_uses_wait_then_seeded_rank_only():
    assignments = {
        0: [opportunity(0, vote_flip_gain=1, margin_gain=2)],
        1: [opportunity(1, vote_flip_gain=1, margin_gain=2)],
    }
    _, rows = priorities_for(
        assignments,
        waits={0: 2, 1: 4, 2: 0, 3: 0, 4: 0},
    )
    decision = build_target_selection_decision(rows)
    assert decision.selection_pool_stage == "responsibility_joint_pareto"
    assert decision.selected_agent_id == 1
    assert decision.target_frontier_agent_ids == (0, 1)


def test_arbitrarily_large_wait_cannot_override_first_frontier():
    assignments = {
        0: [opportunity(0, vote_flip_gain=1, margin_gain=2)],
        1: [opportunity(1, vote_flip_gain=0, margin_gain=1)],
    }
    _, rows = priorities_for(
        assignments,
        waits={0: 0, 1: 999, 2: 0, 3: 0, 4: 0},
        current=(10, 10, 10, 10, 10),
    )
    decision = build_target_selection_decision(rows)
    assert decision.selection_pool_stage == "responsibility_joint_pareto"
    assert decision.selected_agent_id == 0
    assert decision.target_pareto_fronts == {0: 1, 1: 2}


def test_no_portfolio_member_cannot_enter_responsibility_lane():
    _, rows = priorities_for({})
    decision = build_target_selection_decision(rows)
    assert decision.selected_agent_id is None
    assert decision.selection_pool_stage == "no_actionable_responsibility"
    assert decision.update_lane == "no_actionable_responsibility"


def test_frozen_member_is_excluded_even_with_largest_vector_and_wait():
    assignments = {
        0: [opportunity(0, vote_flip_gain=1, margin_gain=9)],
        1: [opportunity(1, vote_flip_gain=0, margin_gain=1)],
    }
    state = runtime(
        updates_since_selected_by_agent={0: 999, 1: 0, 2: 0, 3: 0, 4: 0}
    )
    initialize_legacy_freeze_state(state, range(5))
    state.frozen_by_agent[0] = True
    rows = target_priorities(
        assignments=assignments,
        state=state,
        seed=7,
        current_member_correct_counts=(0, 10, 10, 10, 10),
        initial_member_correct_counts=(0, 10, 10, 10, 10),
        member_uplift_tolerance=5,
    )
    decision = build_target_selection_decision(rows)
    assert decision.selected_agent_id == 1
    assert decision.frozen_agent_ids == (0,)
    assert decision.active_candidate_agent_ids == (1,)


def test_same_portfolio_second_complete_failure_freezes_and_changed_state_resets():
    state = runtime()
    initialize_legacy_freeze_state(state, range(5))
    first = responsibility_portfolios(
        assignments={0: [opportunity(0, vote_flip_gain=1, margin_gain=2)]},
        state=state,
    )[0]
    assert record_target_update_failure(
        state=state, portfolio=first, update_index=0
    ) is None
    assert state.consecutive_failed_updates_by_agent[0] == 1
    event = record_target_update_failure(
        state=state, portfolio=first, update_index=1
    )
    assert event is not None
    assert state.frozen_by_agent[0]

    changed_state = runtime()
    initialize_legacy_freeze_state(changed_state, range(5))
    record_target_update_failure(
        state=changed_state, portfolio=first, update_index=0
    )
    changed = responsibility_portfolios(
        assignments={0: [opportunity(
            0, vote_flip_gain=0, margin_gain=4, question_hash="q2"
        )]},
        state=changed_state,
    )[0]
    assert record_target_update_failure(
        state=changed_state, portfolio=changed, update_index=1
    ) is None
    assert changed_state.consecutive_failed_updates_by_agent[0] == 1
    assert not changed_state.frozen_by_agent[0]


def test_unfreeze_requires_both_other_accepts_and_material_portfolio_change():
    state = runtime()
    initialize_legacy_freeze_state(state, range(5))
    original_assignments = {
        0: [opportunity(0, vote_flip_gain=1, margin_gain=2, question_hash="q1")]
    }
    portfolio = responsibility_portfolios(
        assignments=original_assignments, state=state
    )[0]
    record_target_update_failure(state=state, portfolio=portfolio, update_index=0)
    record_target_update_failure(state=state, portfolio=portfolio, update_index=1)
    state.specialization_anchor_by_agent[0] = RepairLane.COVERAGE

    changed_assignments = {
        0: [opportunity(0, vote_flip_gain=0, margin_gain=3, question_hash="q2")]
    }
    assert not refresh_frozen_member_states(
        state=state, assignments=changed_assignments, update_index=2
    )
    record_target_update_acceptance(state=state, accepted_agent_id=1)
    record_target_update_acceptance(state=state, accepted_agent_id=2)
    s_only_change = {
        0: [opportunity(0, vote_flip_gain=1, margin_gain=3, question_hash="q1")]
    }
    assert not refresh_frozen_member_states(
        state=state, assignments=s_only_change, update_index=3
    )
    assert not refresh_frozen_member_states(
        state=state, assignments=original_assignments, update_index=3
    )
    events = refresh_frozen_member_states(
        state=state, assignments=changed_assignments, update_index=4
    )
    assert len(events) == 1
    assert not state.frozen_by_agent[0]
    assert state.consecutive_failed_updates_by_agent[0] == 0
    assert state.specialization_anchor_by_agent[0] is None


def test_accepted_target_resets_failure_streak_and_signature():
    state = runtime()
    initialize_legacy_freeze_state(state, range(5))
    state.consecutive_failed_updates_by_agent[1] = 1
    state.last_failed_portfolio_signature_by_agent[1] = "previous"
    record_target_update_acceptance(state=state, accepted_agent_id=1)
    assert state.consecutive_failed_updates_by_agent[1] == 0
    assert state.last_failed_portfolio_signature_by_agent[1] == ""


def test_all_nonempty_portfolios_frozen_has_no_actionable_repairability():
    assignments = {0: [opportunity(0, vote_flip_gain=1, margin_gain=2)]}
    state = runtime()
    initialize_legacy_freeze_state(state, range(5))
    state.frozen_by_agent[0] = True
    rows = target_priorities(
        assignments=assignments,
        state=state,
        seed=7,
        current_member_correct_counts=(10,) * 5,
        initial_member_correct_counts=(10,) * 5,
        member_uplift_tolerance=5,
    )
    decision = build_target_selection_decision(rows)
    assert decision.selected_agent_id is None
    assert decision.no_actionable_reason == "no_actionable_repairability"
