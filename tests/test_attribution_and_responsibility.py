from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
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


def opportunities(answers, counts=(8, 7, 6, 5, 4), question_hash="q"):
    current = team(answers, question_hash)
    rows = tuple(
        compute_member_aware_repair_opportunity(
            team_state=current,
            peer_context=build_peer_vote_context(current, agent),
            member_correct_counts=counts,
            member_gains_from_initial=(0, 1, 0, -1, -2),
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


def test_improvement_need_uses_gain_sum_minus_k_member_gain():
    _, rows = opportunities(["A", "A", "B", "B", "B"])
    assert rows[4].improvement_need == sum((0, 1, 0, -1, -2)) - 5 * -2
    assert rows[0].improvement_need == 0


def test_assignment_uses_only_wrong_agents_and_is_seed_deterministic():
    current, rows = opportunities(["A", "A", "B", "B", "B"])
    kwargs = dict(
        team_states={"q": current},
        opportunities={"q": rows},
        seed=43,
    )
    owners_a, assigned_a = assign_primary_responsibilities(state=state(), **kwargs)
    owners_b, _ = assign_primary_responsibilities(state=state(), **kwargs)
    assert owners_a == owners_b
    assert owners_a["q"] in {2, 3, 4}
    assert not assigned_a[0] and not assigned_a[1]


def test_all_agents_with_member_errors_are_target_eligible_without_assignments():
    current, rows = opportunities(["B", "B", "B", "B", "B"])
    priorities = target_priorities(
        opportunities={"q": rows},
        assignments={agent: [] for agent in range(5)},
        state=state(),
        seed=42,
        max_wait_updates=4,
    )
    assert {row.agent_id for row in priorities} == set(range(5))
    assert select_target_agent(priorities) in range(5)


def test_overdue_target_is_selected_before_non_overdue_frontier_member():
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
