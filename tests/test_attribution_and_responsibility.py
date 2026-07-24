from dataclasses import replace

from multi_dataset_diverse_rl.peer_state import (
    build_peer_vote_context,
    build_team_vote_state,
)
from multi_dataset_diverse_rl.responsibility import (
    ResponsibilityState,
    assign_primary_responsibilities,
    compute_member_aware_repair_opportunity,
    select_target_agent,
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


def opportunities(
    answers,
    counts=(8, 7, 6, 5, 4),
    gains=(0, 1, 0, -1, -2),
    question_hash="q",
):
    current = team(answers, question_hash)
    initial = tuple(
        count - gain for count, gain in zip(counts, gains, strict=True)
    )
    rows = tuple(
        compute_member_aware_repair_opportunity(
            team_state=current,
            peer_context=build_peer_vote_context(current, agent),
            initial_correct_counts=initial,
            member_correct_counts=counts,
            member_gains_from_initial=gains,
            unique_correct_counts=(1, 0, 0, 0, 0),
            pivotal_correct_counts=(1, 0, 0, 0, 0),
        )
        for agent in range(5)
    )
    return current, rows


def state(**overrides):
    values = dict(
        assigned_load_by_agent={agent: 0 for agent in range(5)},
        updates_since_selected_by_agent={agent: 0 for agent in range(5)},
        accepted_updates_by_agent={agent: 0 for agent in range(5)},
    )
    values.update(overrides)
    return ResponsibilityState(**values)


def assign(current, rows, current_state=None, seed=43, margin=0.05):
    return assign_primary_responsibilities(
        team_states={current.question_hash: current},
        opportunities={current.question_hash: rows},
        state=current_state or state(),
        seed=seed,
        responsibility_switch_margin=margin,
    )


def test_opportunity_has_complete_member_state_and_exact_need_formula():
    _, rows = opportunities(["A", "A", "B", "B", "B"])
    assert rows[4].initial_correct_count == 6
    assert rows[4].current_correct_count == 4
    assert rows[4].gain_count == -2
    assert rows[4].improvement_need == sum((0, 1, 0, -1, -2)) - 5 * -2
    assert rows[0].improvement_need == 0
    assert rows[0].unique_correct_count == 1
    assert rows[0].pivotal_correct_count == 1


def test_assignment_only_covers_vote_wrong_samples_and_wrong_agents():
    wrong_state, wrong_rows = opportunities(["A", "A", "B", "B", "B"])
    owners, assigned, audits = assign(wrong_state, wrong_rows)
    assert owners["q"] in {2, 3, 4}
    assert not assigned[0] and not assigned[1]
    assert audits["q"]["vote_correct"] is False

    correct_state, correct_rows = opportunities(["A", "A", "A", "B", "B"])
    owners, assigned, audits = assign(correct_state, correct_rows)
    assert owners == {}
    assert all(not rows for rows in assigned.values())
    assert audits == {}


def test_owner_front_uses_all_five_dimensions_and_member_first_preference():
    current, rows = opportunities(["A", "A", "B", "B", "B"])
    by_id = {row.agent_id: row for row in rows}
    # Agent 2 has direct vote leverage; agent 3 has the larger member need.
    by_id[2] = replace(
        by_id[2],
        direct_vote_fix=True,
        oracle_soft_utility_gain=0.2,
        improvement_need=1,
        coverage_opportunity=False,
        dominant_wrong_member=False,
    )
    by_id[3] = replace(
        by_id[3],
        direct_vote_fix=False,
        oracle_soft_utility_gain=0.1,
        improvement_need=5,
        coverage_opportunity=False,
        dominant_wrong_member=False,
    )
    # Agent 4 is strictly dominated by agent 3.
    by_id[4] = replace(
        by_id[4],
        direct_vote_fix=False,
        oracle_soft_utility_gain=0.0,
        improvement_need=1,
        coverage_opportunity=False,
        dominant_wrong_member=False,
    )
    owners, _, audits = assign(current, tuple(by_id[index] for index in range(5)))
    assert audits["q"]["candidate_pareto_fronts"]["2"] == 1
    assert audits["q"]["candidate_pareto_fronts"]["3"] == 1
    assert audits["q"]["candidate_pareto_fronts"]["4"] > 1
    assert owners["q"] == 3


def test_dominated_previous_owner_switches_despite_inertia():
    current, rows = opportunities(["A", "A", "B", "B", "B"])
    by_id = {row.agent_id: row for row in rows}
    by_id[2] = replace(
        by_id[2],
        direct_vote_fix=True,
        oracle_soft_utility_gain=1.0,
        improvement_need=10,
        coverage_opportunity=True,
        dominant_wrong_member=True,
    )
    by_id[3] = replace(
        by_id[3],
        direct_vote_fix=False,
        oracle_soft_utility_gain=0.0,
        improvement_need=0,
        coverage_opportunity=False,
        dominant_wrong_member=False,
    )
    current_state = state(
        primary_owner_by_question={"q": 3},
        owner_age_by_question={"q": 4},
    )
    owners, _, audit = assign(
        current,
        tuple(by_id[index] for index in range(5)),
        current_state=current_state,
    )
    assert owners["q"] == 2
    assert audit["q"]["chosen_reason"] == "member_aware_pareto_preference"


def test_all_erroneous_agents_are_target_eligible_without_assignments():
    _, rows = opportunities(["B", "B", "B", "B", "B"])
    priorities = target_priorities(
        opportunities={"q": rows},
        assignments={agent: [] for agent in range(5)},
        state=state(),
        seed=42,
        max_wait_updates=4,
    )
    assert {row.agent_id for row in priorities} == set(range(5))
    assert all(row.assigned_load == 0 for row in priorities)
    assert select_target_agent(priorities) in range(5)


def test_target_priority_uses_five_axis_front_and_member_first_preference():
    _, rows = opportunities(["B", "B", "B", "B", "B"])
    priorities = target_priorities(
        opportunities={"q": rows},
        assignments={agent: [] for agent in range(5)},
        state=state(),
        seed=42,
        max_wait_updates=4,
    )
    selected = select_target_agent(priorities)
    front = {row.agent_id for row in priorities if row.pareto_front == 1}
    assert selected in front
    selected_row = next(row for row in priorities if row.agent_id == selected)
    assert selected_row.individual_error_count == 1
    assert isinstance(selected_row.oracle_soft_utility_gain_sum, float)


def test_overdue_after_four_misses_is_selected_first():
    _, rows = opportunities(["B", "B", "B", "B", "B"])
    current_state = state(
        updates_since_selected_by_agent={0: 0, 1: 4, 2: 0, 3: 0, 4: 0}
    )
    priorities = target_priorities(
        opportunities={"q": rows},
        assignments={agent: [] for agent in range(5)},
        state=current_state,
        seed=42,
        max_wait_updates=4,
    )
    assert select_target_agent(priorities) == 1


def test_seed_changes_only_exact_symmetric_ties():
    current, rows = opportunities(
        ["B", "B", "B", "B", "B"],
        counts=(5, 5, 5, 5, 5),
        gains=(0, 0, 0, 0, 0),
    )
    winners = set()
    for seed in range(20):
        owners, _, _ = assign(current, rows, seed=seed)
        winners.add(owners["q"])
    assert len(winners) > 1
    owners_a, _, _ = assign(current, rows, seed=42)
    owners_b, _, _ = assign(current, rows, seed=42)
    assert owners_a == owners_b
