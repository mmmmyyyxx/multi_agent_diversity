import pytest

from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    ConstraintLimits,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    candidate_is_acceptable,
    evaluate_constraints,
    member_first_key,
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
)


def item(
    name="candidate",
    *,
    correct=8,
    invalid=0,
    vote_count=8,
    member_counts=(10, 10, 10, 10, 10),
    vote_loss=0,
    unique_loss=0,
    pivotal_loss=0,
):
    gains = member_gain_metrics((10, 10, 10, 10, 10), member_counts)
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(correct, correct / 10, invalid, invalid / 10),
        team_outcome=TeamOutcomeMetrics(
            (), vote_count, vote_count / 10, (), (), (), 0.0
        ),
        marginal=CandidateMarginalContribution(
            vote_gain_count=max(0, vote_count - 8),
            vote_loss_count=vote_loss,
            net_vote_delta=vote_count - 8,
            soft_utility_delta=0.0,
            coverage_gain_count=0,
            coverage_loss_count=0,
            dominant_wrong_exit_count=0,
            dominant_wrong_join_count=0,
            assigned_residual_repair_count=0,
            assigned_residual_utility_delta=0.0,
        ),
        protection=ProtectionContribution(unique_loss, pivotal_loss),
        member_gain=gains,
    )


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (item(correct=7), "local_accuracy"),
        (item(invalid=1), "invalid"),
        (item(vote_loss=1), "vote_loss"),
        (item(unique_loss=1), "unique_correct"),
        (item(pivotal_loss=1), "pivotal_correct"),
    ],
)
def test_each_guard_rejects_explicitly(candidate, reason):
    active = item("active")
    decision = evaluate_constraints(candidate, active, active, ConstraintLimits())
    assert not decision.passed
    assert reason in decision.rejection_reasons


def test_vote_positive_member_regression_is_not_formally_acceptable():
    active = item("active")
    candidate = item("vote-up", vote_count=9, member_counts=(9, 10, 10, 10, 12))
    assert not candidate_is_acceptable(candidate, active)


def test_vote_neutral_worst_member_gain_is_acceptable():
    active = item("active")
    candidate = item("member-up", member_counts=(11, 11, 11, 11, 11))
    assert candidate_is_acceptable(candidate, active)
    assert member_first_key(candidate) > member_first_key(active)


def test_typed_metrics_require_member_gain():
    with pytest.raises(TypeError):
        CandidateEvaluation(
            prompt="p",
            prompt_hash="h",
            competence=PromptCompetenceMetrics(1, 1.0, 0, 0.0),
            team_outcome=TeamOutcomeMetrics((), 1, 1.0, (), (), (), 0.0),
            marginal=CandidateMarginalContribution(0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0),
            protection=ProtectionContribution(0, 0),
        )
