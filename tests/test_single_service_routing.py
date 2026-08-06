from multi_dataset_diverse_rl.peer_state import (
    build_peer_vote_context,
    build_team_vote_state,
)
from multi_dataset_diverse_rl.protocol import (
    candidate_budget_contract,
    experiment_protocol,
)
from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity,
    RepairLane,
    ResponsibilityState,
    build_service_routing,
    compute_member_aware_repair_opportunity,
    compute_repair_eligibility_sets,
    record_specialization_anchor_acceptance,
    repair_lane_for,
    target_priorities,
)


def team(question_hash, answers):
    return build_team_vote_state(
        question_hash=question_hash,
        gold_answer="A",
        answers=answers,
        normalize_answer=str.upper,
        match_answer=lambda left, right: left == right,
        tie_break="abstain",
    )


def opportunity(
    agent_id,
    question_hash,
    *,
    vote_flip_gain=0,
    margin_gain=1,
    dominant=False,
):
    return MemberAwareRepairOpportunity(
        agent_id=agent_id,
        question_hash=question_hash,
        vote_flip_gain=vote_flip_gain,
        margin_gain=margin_gain,
        member_error=True,
        coverage_opportunity=False,
        conversion_opportunity=False,
        dominant_wrong_member=dominant,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def runtime():
    return ResponsibilityState(
        updates_since_selected_by_agent={agent: 0 for agent in range(5)},
        accepted_updates_by_agent={agent: 0 for agent in range(5)},
        target_attempt_count_by_agent={agent: 0 for agent in range(5)},
    )


def test_each_vote_wrong_residual_has_exactly_one_program_lane():
    scenarios = (
        (team("coverage", ["B", "B", "C", "C", "D"]), 0, RepairLane.COVERAGE),
        (team("direct", ["A", "A", "B", "B", "B"]), 2, RepairLane.DIRECT_FLIP),
        (team("margin", ["A", "B", "B", "C", "C"]), 1, RepairLane.MARGIN_SUPPORT),
    )
    observed = []
    for state, agent_id, expected in scenarios:
        row = compute_member_aware_repair_opportunity(
            team_state=state,
            peer_context=build_peer_vote_context(state, agent_id),
        )
        lane = repair_lane_for(state, row)
        assert lane is expected
        observed.append(lane)
    assert set(observed) == set(RepairLane)


def test_exact_tie_legal_eligibility_is_unchanged():
    state = team("q", ["B", "C", "D", "E", "F"])
    rows = [
        opportunity(agent, "q", vote_flip_gain=0, margin_gain=(2 if agent in {0, 2, 4} else 1))
        for agent in range(5)
    ]
    eligible, _, _ = compute_repair_eligibility_sets(
        team_states={"q": state}, opportunities={"q": rows}, state=runtime()
    )
    assert eligible["q"] == (0, 2, 4)


def route(states, rows, eligible, state=None, seed=17):
    return build_service_routing(
        team_states=states,
        opportunities=rows,
        eligible_agents_by_question=eligible,
        state=state or runtime(),
        seed=seed,
    )


def test_service_portfolios_are_unique_complete_and_deterministic():
    states = {
        f"q{index}": team(f"q{index}", ["B", "B", "C", "C", "D"])
        for index in range(8)
    }
    rows = {
        question_hash: (
            opportunity(0, question_hash),
            opportunity(1, question_hash),
        )
        for question_hash in states
    }
    eligible = {question_hash: (0, 1) for question_hash in states}
    first = route(states, rows, eligible)
    second = route(
        dict(reversed(tuple(states.items()))),
        dict(reversed(tuple(rows.items()))),
        dict(reversed(tuple(eligible.items()))),
    )
    assert first.assignments_by_question == second.assignments_by_question
    assigned = [
        row.question_hash
        for portfolio in first.service_portfolios.values()
        for row in portfolio
    ]
    assert sorted(assigned) == sorted(states)
    assert len(assigned) == len(set(assigned))
    loads = [len(first.service_portfolios[agent]) for agent in (0, 1)]
    assert max(loads) - min(loads) <= 1


def test_anchor_match_then_unanchored_then_load_control_routing():
    state = runtime()
    state.specialization_anchor_by_agent = {
        0: RepairLane.COVERAGE,
        1: None,
        2: RepairLane.DIRECT_FLIP,
        3: None,
        4: None,
    }
    coverage = team("q", ["B", "B", "C", "C", "D"])
    routed = route(
        {"q": coverage},
        {"q": tuple(opportunity(agent, "q") for agent in (0, 1, 2))},
        {"q": (0, 1, 2)},
        state,
    )
    assert routed.assignments_by_question["q"].service_agent_id == 0

    state.specialization_anchor_by_agent[0] = RepairLane.MARGIN_SUPPORT
    routed = route(
        {"q": coverage},
        {"q": tuple(opportunity(agent, "q") for agent in (0, 1))},
        {"q": (0, 1)},
        state,
    )
    assert routed.assignments_by_question["q"].service_agent_id == 1


def test_v13_service_routing_ignores_legacy_frozen_state():
    state = runtime()
    state.frozen_by_agent = {agent: False for agent in range(5)}
    state.frozen_by_agent[0] = True
    coverage = team("q", ["B", "B", "C", "C", "D"])
    rows = {"q": (opportunity(0, "q"), opportunity(1, "q"))}
    routed = route({"q": coverage}, rows, {"q": (0, 1)}, state)
    assert routed.assignments_by_question["q"].service_agent_id in {0, 1}

    state.frozen_by_agent[1] = True
    rerouted = route({"q": coverage}, rows, {"q": (0, 1)}, state)
    decision = rerouted.assignments_by_question["q"]
    assert decision.service_agent_id in {0, 1}
    assert not decision.service_blocked_by_freeze


def test_anchor_retains_active_lane_and_scheduler_uses_only_that_slice():
    state = runtime()
    state.specialization_anchor_by_agent = {
        0: RepairLane.COVERAGE,
        1: None,
        2: None,
        3: None,
        4: None,
    }
    coverage = team("coverage", ["B", "B", "C", "C", "D"])
    direct = team("direct", ["A", "A", "B", "B", "B"])
    rows = {
        "coverage": (opportunity(0, "coverage", margin_gain=1),),
        "direct": (
            opportunity(0, "direct", vote_flip_gain=1, margin_gain=9),
        ),
    }
    routed = route(
        {"coverage": coverage, "direct": direct},
        rows,
        {"coverage": (0,), "direct": (0,)},
        state,
    )
    assert routed.active_lane_by_agent[0] is RepairLane.COVERAGE
    assert [row.question_hash for row in routed.active_slices[0]] == ["coverage"]
    priorities = target_priorities(
        assignments=routed.active_slices,
        legal_assignments={0: list(rows["coverage"] + rows["direct"])},
        service_portfolios=routed.service_portfolios,
        active_lane_by_agent=routed.active_lane_by_agent,
        state=state,
        seed=17,
        current_member_correct_counts=(10,) * 5,
        initial_member_correct_counts=(10,) * 5,
        member_uplift_tolerance=5,
    )
    row = next(item for item in priorities if item.agent_id == 0)
    assert (row.direct_fix_count, row.margin_gain_sum) == (0, 1)
    assert row.service_portfolio_size == 2
    assert row.active_lane_size == 1


def test_anchor_is_created_or_switched_only_by_acceptance():
    state = runtime()
    first = record_specialization_anchor_acceptance(
        state=state,
        accepted_agent_id=0,
        active_lane=RepairLane.COVERAGE,
        update_index=1,
    )
    assert state.specialization_anchor_by_agent[0] is RepairLane.COVERAGE
    assert first["event"] == "accepted_anchor_set"
    before = dict(state.specialization_anchor_by_agent)
    assert dict(state.specialization_anchor_by_agent) == before
    switched = record_specialization_anchor_acceptance(
        state=state,
        accepted_agent_id=0,
        active_lane=RepairLane.DIRECT_FLIP,
        update_index=2,
    )
    assert state.specialization_anchor_by_agent[0] is RepairLane.DIRECT_FLIP
    assert switched["event"] == "accepted_anchor_switch"


def test_setting_isolation_enables_service_flow_from_s1():
    names = (
        "shared_static_reference",
        "shared_generic_evolution",
        "shared_member_aware_dual_target",
        "shared_responsibility_conditioned_dual_target",
    )
    protocols = {
        name: experiment_protocol(
            name,
            initialization_mode="shared_identical",
            tie_policy="abstain",
            candidate_budget_contract=candidate_budget_contract(
                name,
                candidates_per_target_branch=2,
                stage_b_budget_per_branch=2,
                stage_a_channel_top_k=2,
                representative_size=3,
                coverage_size=1,
                conversion_size=1,
                preservation_size=1,
            ),
        )
        for name in names
    }
    for name in tuple(protocols)[:2]:
        assert not protocols[name].service_routing_enabled
        assert not protocols[name].repairability_freeze_enabled
    assert protocols["shared_member_aware_dual_target"].service_routing_enabled
    assert (
        protocols["shared_member_aware_dual_target"].tcs_context_policy
        == "generic_peer_state"
    )
    assert protocols["shared_responsibility_conditioned_dual_target"].tcs_context_policy == "member_aware_responsibility_conditioned"
