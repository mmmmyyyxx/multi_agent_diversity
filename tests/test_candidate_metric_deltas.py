import pytest

from multi_dataset_diverse_rl.system import compute_candidate_metric_deltas


@pytest.mark.parametrize(
    "baseline,candidate",
    [(0.2, 0.8), (0.8, 0.2), (0.5, 0.5)],
)
def test_candidate_metric_deltas_are_canonical_for_positive_negative_and_zero(baseline, candidate):
    values = compute_candidate_metric_deltas(
        baseline_target_accuracy=baseline,
        candidate_target_accuracy=candidate,
        baseline_team_accuracy=baseline,
        candidate_team_accuracy=candidate,
        baseline_oracle_accuracy=baseline,
        candidate_oracle_accuracy=candidate,
        baseline_embedding_diversity=baseline,
        candidate_embedding_diversity=candidate,
        baseline_invalid_rate=baseline,
        candidate_invalid_rate=candidate,
    )

    assert values["accuracy_delta"] == pytest.approx(candidate - baseline)
    assert values["vote_delta"] == pytest.approx(candidate - baseline)
    assert values["coverage_delta"] == pytest.approx(candidate - baseline)
    assert values["diversity_delta"] == pytest.approx(candidate - baseline)
    assert values["invalid_delta"] == pytest.approx(candidate - baseline)
