from multi_dataset_diverse_rl.candidate_selection import (
    ConstraintLimits, candidate_is_acceptable, candidate_is_feasible, vote_first_key,
)


def item(name="candidate", **values):
    metrics = {
        "candidate_target_correct_count": 8, "candidate_invalid_count": 0,
        "net_vote_delta": 0, "vote_loss_count": 0, "soft_vote_utility_delta": 0.0,
        "coverage_gain_count": 0, "assigned_residual_utility_delta": 0.0,
        "unique_correct_loss_count": 0, "pivotal_vote_correct_loss_count": 0,
    }
    metrics.update(values)
    return {"prompt_hash": name, "generation": 1, "metrics": metrics}


def test_accuracy_guard_blocks_vote_gain():
    candidate = item(candidate_target_correct_count=7, net_vote_delta=3)
    active = item("active")
    assert not candidate_is_feasible(candidate["metrics"], active["metrics"], active["metrics"], ConstraintLimits())


def test_vote_first_under_feasibility():
    active = item("active")
    stronger = item("stronger", net_vote_delta=1)
    assert vote_first_key(stronger) > vote_first_key(active)
    assert candidate_is_acceptable(stronger, active, ConstraintLimits())


def test_dense_c0_to_c1_and_accuracy_only_paths_are_accepted():
    active = item("active")
    dense = item("dense", soft_vote_utility_delta=0.01, coverage_gain_count=1)
    accuracy = item("accuracy", candidate_target_correct_count=9)
    assert candidate_is_acceptable(dense, active, ConstraintLimits())
    assert candidate_is_acceptable(accuracy, active, ConstraintLimits())


def test_wrong_label_dispersion_and_team_losses_are_rejected():
    active = item("active")
    wrong_to_wrong = item("wrong-to-wrong")
    vote_loss = item("vote-loss", net_vote_delta=-1, vote_loss_count=1)
    unique_loss = item("unique-loss", net_vote_delta=1, unique_correct_loss_count=1)
    pivotal_loss = item("pivotal-loss", net_vote_delta=1, pivotal_vote_correct_loss_count=1)
    limits = ConstraintLimits()
    assert not candidate_is_acceptable(wrong_to_wrong, active, limits)
    assert not candidate_is_feasible(vote_loss["metrics"], active["metrics"], active["metrics"], limits)
    assert not candidate_is_feasible(unique_loss["metrics"], active["metrics"], active["metrics"], limits)
    assert not candidate_is_feasible(pivotal_loss["metrics"], active["metrics"], active["metrics"], limits)
