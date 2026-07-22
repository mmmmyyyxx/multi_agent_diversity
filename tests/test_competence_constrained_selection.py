import pytest

from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    ConstraintLimits,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    candidate_is_acceptable,
    evaluate_constraints,
    vote_first_key,
)
from multi_dataset_diverse_rl.responsibility import CandidateMarginalContribution, ProtectionContribution


def item(
    name="candidate",
    *,
    correct=8,
    invalid=0,
    net_vote=0,
    vote_loss=0,
    soft=0.0,
    coverage=0,
    residual=0.0,
    unique_loss=0,
    pivotal_loss=0,
):
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(correct, correct / 10, invalid, invalid / 10),
        team_outcome=TeamOutcomeMetrics((), 0.0, (), (), (), 0.0),
        marginal=CandidateMarginalContribution(
            vote_gain_count=max(0, net_vote + vote_loss),
            vote_loss_count=vote_loss,
            net_vote_delta=net_vote,
            soft_utility_delta=soft,
            coverage_gain_count=coverage,
            coverage_loss_count=0,
            dominant_wrong_exit_count=0,
            dominant_wrong_join_count=0,
            assigned_residual_repair_count=coverage,
            assigned_residual_utility_delta=residual,
        ),
        protection=ProtectionContribution(unique_loss, pivotal_loss),
    )


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (item(correct=7, net_vote=3), "local_accuracy"),
        (item(invalid=1), "invalid"),
        (item(net_vote=-1, vote_loss=1), "vote_loss"),
        (item(net_vote=1, unique_loss=1), "unique_correct"),
        (item(net_vote=1, pivotal_loss=1), "pivotal_correct"),
    ],
)
def test_each_guard_rejects_explicitly(candidate, reason):
    active = item("active")
    decision = evaluate_constraints(candidate, active, active, ConstraintLimits())
    assert decision.passed is False
    assert reason in decision.rejection_reasons


def test_global_initial_accuracy_guard_is_distinct_from_local_guard():
    active = item("active", correct=7)
    initial = item("initial", correct=8)
    candidate = item(correct=7)
    decision = evaluate_constraints(candidate, active, initial, ConstraintLimits())
    assert decision.local_accuracy_passed is True
    assert decision.initial_accuracy_passed is False


def test_vote_gain_precedes_soft_gain_and_accuracy_in_vote_first_key():
    vote_gain = item("vote", correct=8, net_vote=1)
    soft = item("soft", correct=10, soft=1.0)
    assert vote_first_key(vote_gain) > vote_first_key(soft)
    assert candidate_is_acceptable(vote_gain, item("active"), ConstraintLimits())


def test_c0_wrong_to_wrong_is_not_accepted_but_c0_to_c1_is():
    active = item("active")
    wrong_to_wrong = item("wrong-to-wrong")
    coverage = item("coverage", correct=9, soft=0.05, coverage=1)
    assert candidate_is_acceptable(wrong_to_wrong, active, ConstraintLimits()) is False
    assert candidate_is_acceptable(coverage, active, ConstraintLimits()) is True


def test_typed_metrics_cannot_silently_default_missing_fields():
    with pytest.raises(TypeError):
        CandidateEvaluation(
            prompt="p",
            prompt_hash="h",
            competence=PromptCompetenceMetrics(1, 1.0, 0, 0.0),
            team_outcome=TeamOutcomeMetrics((), 0.0, (), (), (), 0.0),
            protection=ProtectionContribution(0, 0),
        )
