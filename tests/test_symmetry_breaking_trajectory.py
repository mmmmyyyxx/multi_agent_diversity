from multi_dataset_diverse_rl.team_search.symmetry_breaking import (
    SymmetryBreakingTrajectory,
    classify_symmetry_breaking_trajectory,
)


ZERO = {str(member): 0 for member in range(5)}


def test_tracks_commit_pivotal_and_vote_times_in_both_clocks() -> None:
    trajectory = SymmetryBreakingTrajectory(
        baseline_vote_correct=40,
        baseline_p_i=ZERO,
    )
    trajectory.observe(committed=False, vote_correct=40, p_i=ZERO)
    trajectory.observe(committed=True, vote_correct=40, p_i=ZERO)
    pivotal = dict(ZERO)
    pivotal["3"] = 2
    trajectory.observe(committed=True, vote_correct=40, p_i=pivotal)
    trajectory.observe(committed=True, vote_correct=41, p_i=pivotal)

    payload = trajectory.payload()
    assert payload["T_commit_opportunity"] == 2
    assert payload["T_pivotal_opportunity"] == 3
    assert payload["T_pivotal_commit_index"] == 2
    assert payload["T_vote_opportunity"] == 4
    assert payload["T_vote_commit_index"] == 3
    assert classify_symmetry_breaking_trajectory(payload, integrity_passed=True) == (
        "ALGORITHMIC_SYMMETRY_BREAKING_WITH_VOTE_GAIN_OBSERVED"
    )


def test_frozen_stopping_priority_and_non_success_classifiers() -> None:
    trajectory = SymmetryBreakingTrajectory(
        baseline_vote_correct=40,
        baseline_p_i=ZERO,
    )
    trajectory.observe(committed=True, vote_correct=40, p_i=ZERO)
    assert trajectory.stop_reason(
        max_opportunities=8, max_commits=1, no_commit_patience=6
    ) == "max_safe_commits"
    assert classify_symmetry_breaking_trajectory(
        trajectory.payload(), integrity_passed=True
    ) == "SAFE_COMMITS_WITHOUT_PLURALITY_RESPONSIVENESS"

    no_commit = SymmetryBreakingTrajectory(
        baseline_vote_correct=40,
        baseline_p_i=ZERO,
    )
    for _ in range(6):
        no_commit.observe(committed=False, vote_correct=40, p_i=ZERO)
    assert no_commit.stop_reason(
        max_opportunities=8, max_commits=4, no_commit_patience=6
    ) == "no_commit_patience"
    assert classify_symmetry_breaking_trajectory(
        no_commit.payload(), integrity_passed=True
    ) == "NO_SAFE_COMMIT_OBSERVED"


def test_invalid_baseline_and_endpoint_order_fail_closed() -> None:
    invalid = {**ZERO, "0": 1}
    trajectory = SymmetryBreakingTrajectory(
        baseline_vote_correct=40,
        baseline_p_i=invalid,
    )
    assert classify_symmetry_breaking_trajectory(
        trajectory.payload(), integrity_passed=True
    ) == "HOLD_BASELINE_OR_INTEGRITY_FAILURE"

    impossible = {
        "baseline_p_i": [0, 0, 0, 0, 0],
        "safe_commits": 1,
        "T_commit_opportunity": 3,
        "T_pivotal_opportunity": 2,
        "T_vote_opportunity": 2,
    }
    assert classify_symmetry_breaking_trajectory(
        impossible, integrity_passed=True
    ) == "HOLD_ENDPOINT_ORDER_VIOLATION"
