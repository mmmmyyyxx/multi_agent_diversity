from multi_dataset_diverse_rl.member_objectives import (
    member_gain_metrics,
    pareto_dominates,
    pareto_front,
    team_objective_vector,
)


def test_member_gain_is_integer_count_based():
    metrics = member_gain_metrics((4, 5, 6, 7, 8), (5, 5, 8, 6, 10))
    assert metrics.gains == (1, 0, 2, -1, 2)
    assert metrics.minimum_gain == -1
    assert metrics.total_gain == 4
    assert metrics.improved_member_count == 3
    assert metrics.regressed_member_count == 1


def test_pareto_requires_no_worse_dimension_and_one_strict_gain():
    baseline = team_objective_vector(
        10, member_gain_metrics((5,) * 5, (5,) * 5)
    )
    vote_only_with_regression = team_objective_vector(
        11, member_gain_metrics((5,) * 5, (4, 5, 5, 5, 7))
    )
    member_only = team_objective_vector(
        10, member_gain_metrics((5,) * 5, (6, 6, 6, 6, 6))
    )
    assert not pareto_dominates(vote_only_with_regression, baseline)
    assert pareto_dominates(member_only, baseline)


def test_pareto_front_keeps_incomparable_vectors():
    vectors = (
        team_objective_vector(11, member_gain_metrics((5,) * 5, (5,) * 5)),
        team_objective_vector(10, member_gain_metrics((5,) * 5, (6,) * 5)),
        team_objective_vector(9, member_gain_metrics((5,) * 5, (4,) * 5)),
    )
    assert pareto_front(vectors) == (0, 1)
