import pytest

from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    candidate_is_acceptable,
    evaluate_constraints,
    member_aware_pareto_front,
    member_first_key,
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
)
from multi_dataset_diverse_rl.versions import CANDIDATE_ACCEPTANCE_VERSION


def item(
    name="candidate",
    *,
    correct=10,
    terminal_invalid=0,
    vote_count=8,
    member_counts=None,
    vote_gain=0,
    vote_loss=0,
    unique_gain=0,
    unique_loss=0,
    pivotal_gain=0,
    pivotal_loss=0,
    soft_utility=0.0,
):
    counts = tuple(member_counts or (correct, 10, 10, 10, 10))
    gains = member_gain_metrics(
        (10, 10, 10, 10, 10),
        (10, 10, 10, 10, 10),
        counts,
        0,
    )
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(
            correct,
            correct / 20,
            terminal_invalid,
            terminal_invalid / 20,
            terminal_invalid,
        ),
        team_outcome=TeamOutcomeMetrics(
            (), vote_count, vote_count / 20, (), (), (), soft_utility
        ),
        marginal=CandidateMarginalContribution(
            vote_gain_count=vote_gain,
            vote_loss_count=vote_loss,
            net_vote_delta=vote_gain - vote_loss,
            soft_utility_delta=0.0,
            coverage_gain_count=0,
            coverage_loss_count=0,
            dominant_wrong_exit_count=0,
            dominant_wrong_join_count=0,
            assigned_residual_repair_count=0,
            assigned_residual_utility_delta=0.0,
        ),
        protection=ProtectionContribution(
            unique_correct_loss_count=unique_loss,
            pivotal_correct_loss_count=pivotal_loss,
            unique_correct_gain_count=unique_gain,
            pivotal_correct_gain_count=pivotal_gain,
        ),
        member_gain=gains,
    )


def test_aggregate_guard_allows_local_vote_and_pivotal_losses_with_net_gain():
    active = item("active")
    candidate = item(
        "aggregate-up",
        correct=13,
        vote_count=11,
        vote_gain=4,
        vote_loss=1,
        pivotal_gain=4,
        pivotal_loss=1,
    )
    decision = evaluate_constraints(candidate, active)
    assert decision.hard_feasible
    assert decision.rejection_reasons == ()
    assert decision.vote_net_gain == 3
    assert decision.pivotal_correct_loss_count == 1


def test_vote_neutral_target_improvement_is_accepted():
    active = item("active")
    candidate = item("member-up", correct=11, vote_count=8)
    decision = evaluate_constraints(candidate, active)
    assert decision.hard_feasible
    assert candidate_is_acceptable(candidate, active)


def test_net_vote_regression_is_rejected():
    active = item("active")
    candidate = item(
        "vote-down", correct=11, vote_count=7, vote_gain=1, vote_loss=2
    )
    decision = evaluate_constraints(candidate, active)
    assert not decision.hard_feasible
    assert "team_vote_regression" in decision.rejection_reasons


def test_unique_and_pivotal_losses_are_diagnostic_only():
    active = item("active")
    candidate = item(
        "protected-loss",
        correct=11,
        unique_loss=1,
        pivotal_loss=1,
    )
    decision = evaluate_constraints(candidate, active)
    assert decision.hard_feasible
    assert decision.unique_correct_loss_count == 1
    assert decision.pivotal_correct_loss_count == 1


def test_vote_improvement_can_accept_a_target_neutral_candidate():
    assert (
        CANDIDATE_ACCEPTANCE_VERSION
        == "target_or_vote_strict_progress_v1"
    )
    active = item("active")
    candidate = item("vote-only", correct=10, vote_count=9, vote_gain=1)
    decision = evaluate_constraints(candidate, active)
    assert decision.hard_feasible
    assert not decision.target_strict_improvement
    assert decision.target_nonregression_passed
    assert decision.target_or_vote_progress_passed
    assert decision.pareto_dominates_incumbent
    assert candidate_is_acceptable(candidate, active)


def test_neutral_target_and_vote_is_not_an_accepted_update():
    active = item("active")
    candidate = item("neutral", correct=10, vote_count=8)
    decision = evaluate_constraints(candidate, active)
    assert not decision.hard_feasible
    assert "no_target_or_vote_progress" in decision.rejection_reasons


def test_target_cannot_regress_even_when_vote_improves():
    active = item("active")
    candidate = item("target-down", correct=9, vote_count=9, vote_gain=1)
    decision = evaluate_constraints(candidate, active)
    assert not decision.hard_feasible
    assert "target_regression" in decision.rejection_reasons


def test_terminal_invalid_cannot_increase():
    active = item("active")
    candidate = item("invalid-up", correct=11, terminal_invalid=1)
    decision = evaluate_constraints(candidate, active)
    assert not decision.hard_feasible
    assert "terminal_invalid_regression" in decision.rejection_reasons


def test_member_objective_must_pareto_dominate_incumbent():
    active = item("active")
    candidate = item(
        "member-regression",
        correct=11,
        member_counts=(11, 9, 10, 10, 10),
    )
    decision = evaluate_constraints(candidate, active)
    assert not decision.hard_feasible
    assert "member_objective_regression" in decision.rejection_reasons


def test_internal_candidate_front_and_member_first_preference():
    dominated = item("dominated", correct=11, vote_count=9)
    dominant = item("dominant", correct=12, vote_count=9)
    tradeoff = item(
        "tradeoff",
        correct=11,
        vote_count=8,
        member_counts=(11, 11, 11, 11, 11),
    )
    assert set(member_aware_pareto_front((dominated, dominant, tradeoff))) == {
        "dominant",
        "tradeoff",
    }
    assert member_first_key(tradeoff) > member_first_key(dominant)


def test_typed_metrics_require_member_gain():
    with pytest.raises(TypeError):
        CandidateEvaluation(
            prompt="p",
            prompt_hash="h",
            competence=PromptCompetenceMetrics(1, 1.0, 0, 0.0),
            team_outcome=TeamOutcomeMetrics((), 1, 1.0, (), (), (), 0.0),
            marginal=CandidateMarginalContribution(
                0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0
            ),
            protection=ProtectionContribution(0, 0),
        )
