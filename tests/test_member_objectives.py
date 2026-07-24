from multi_dataset_diverse_rl.member_objectives import (
    member_gain_metrics,
    pareto_dominates,
    pareto_front,
    team_member_gain_state,
    team_objective_vector,
)


def metrics(candidate, incumbent=(5, 5, 5, 5, 5), target=0):
    return member_gain_metrics(
        (5, 5, 5, 5, 5),
        incumbent,
        candidate,
        target,
    )


def test_member_gain_reports_complete_integer_count_state():
    result = member_gain_metrics(
        (4, 5, 6, 7, 8),
        (4, 6, 6, 7, 9),
        (5, 5, 8, 6, 10),
        1,
    )
    assert result.initial_correct_counts == (4, 5, 6, 7, 8)
    assert result.incumbent_correct_counts == (4, 6, 6, 7, 9)
    assert result.candidate_correct_counts == (5, 5, 8, 6, 10)
    assert result.gain_counts == (1, 0, 2, -1, 2)
    assert result.minimum_gain_count == -1
    assert result.total_gain_count == 4
    assert result.mean_gain == 0.8
    assert result.improved_agent_count == 3
    assert result.regressed_agent_count == 1
    assert result.all_members_non_regressed is False
    assert result.all_members_improved is False
    assert result.target_gain_vs_initial == 0
    assert result.target_gain_vs_incumbent == -1


def test_team_member_gain_state_is_target_free():
    state = team_member_gain_state((10, 10, 10), (12, 9, 10))
    assert state.initial_correct_counts == (10, 10, 10)
    assert state.current_correct_counts == (12, 9, 10)
    assert state.gain_counts == (2, -1, 0)
    assert state.minimum_gain_count == -1
    assert state.total_gain_count == 1
    assert state.regressed_agent_count == 1


def test_zero_one_and_all_member_uplift_cases():
    zero = metrics((5, 5, 5, 5, 5))
    one = metrics((6, 5, 5, 5, 5))
    all_up = metrics((6, 6, 6, 6, 6))
    assert zero.gain_counts == (0, 0, 0, 0, 0)
    assert one.total_gain_count == 1
    assert one.improved_agent_count == 1
    assert one.minimum_gain_count == 0
    assert all_up.minimum_gain_count == 1
    assert all_up.all_members_non_regressed is True
    assert all_up.all_members_improved is True


def test_pareto_equality_dominance_and_tradeoff():
    baseline = team_objective_vector(10, metrics((5, 5, 5, 5, 5)))
    vote_only = team_objective_vector(11, metrics((5, 5, 5, 5, 5)))
    members_only = team_objective_vector(10, metrics((6, 6, 6, 6, 6)))
    tradeoff = team_objective_vector(9, metrics((7, 7, 7, 7, 7)))
    assert not pareto_dominates(baseline, baseline)
    assert pareto_dominates(vote_only, baseline)
    assert pareto_dominates(members_only, baseline)
    assert not pareto_dominates(tradeoff, vote_only)
    assert not pareto_dominates(vote_only, tradeoff)


def test_pareto_front_keeps_only_nondominated_vectors():
    vectors = (
        team_objective_vector(11, metrics((5, 5, 5, 5, 5))),
        team_objective_vector(10, metrics((6, 6, 6, 6, 6))),
        team_objective_vector(9, metrics((4, 4, 4, 4, 4))),
    )
    assert pareto_front(vectors) == (0, 1)
