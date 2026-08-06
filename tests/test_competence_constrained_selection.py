from dataclasses import replace

import pytest

from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    candidate_is_acceptable,
    common_monotone_safe_key,
    evaluate_constraints,
    member_first_safe_key,
    vote_first_key,
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
)
from multi_dataset_diverse_rl.versions import CANDIDATE_ACCEPTANCE_VERSION
from multi_dataset_diverse_rl.protocol import (
    MAIN_ABLATION_SETTINGS,
    candidate_budget_contract,
    experiment_protocol,
)


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
    initial_counts=(10, 10, 10, 10, 10),
    incumbent_counts=(10, 10, 10, 10, 10),
    target_id=0,
):
    if member_counts is None:
        counts_list = list(incumbent_counts)
        counts_list[target_id] = correct
        counts = tuple(counts_list)
    else:
        counts = tuple(member_counts)
    gains = member_gain_metrics(
        initial_counts,
        incumbent_counts,
        counts,
        target_id,
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
        == "fixed_peer_monotone_target_or_vote_v2"
    )
    active = item("active")
    candidate = item("vote-only", correct=10, vote_count=9, vote_gain=1)
    decision = evaluate_constraints(candidate, active)
    assert decision.hard_feasible
    assert not decision.target_strict_improvement
    assert decision.target_nonregression_passed
    assert decision.target_or_vote_progress_passed
    assert decision.derived_team_pareto_passed
    assert decision.objective_invariant_checked
    assert decision.pareto_dominates_incumbent
    assert candidate_is_acceptable(candidate, active)


def test_s1_through_s4_have_identical_feasible_sets_and_ranking():
    active = item("active")
    candidates = (
        item("vote-only", correct=10, vote_count=9, vote_gain=1),
        item("target-only", correct=11, vote_count=8),
        item("regression", correct=11, vote_count=7, vote_loss=1),
    )
    protocols = [
        experiment_protocol(
            setting,
            initialization_mode="shared_identical",
            tie_policy="abstain",
            candidate_budget_contract=candidate_budget_contract(
                setting,
                candidates_per_target_branch=2,
                stage_b_budget_per_branch=2,
                stage_a_channel_top_k=3,
                representative_size=12,
                coverage_size=6,
                conversion_size=6,
                preservation_size=4,
            ),
        )
        for setting in MAIN_ABLATION_SETTINGS[1:4]
    ]
    feasible_sets = []
    selected_hashes = []
    for protocol in protocols:
        assert protocol.candidate_acceptance_policy == (
            "fixed_peer_monotone_target_or_vote"
        )
        assert protocol.candidate_ranking_policy == "common_monotone_safe"
        feasible = [
            candidate
            for candidate in candidates
            if evaluate_constraints(candidate, active).passed
        ]
        feasible_sets.append(tuple(row.prompt_hash for row in feasible))
        selected_hashes.append(
            max(feasible, key=common_monotone_safe_key).prompt_hash
        )
    assert len(set(feasible_sets)) == 1
    assert len(set(selected_hashes)) == 1


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
    assert decision.derived_team_pareto_passed
    assert decision.objective_invariant_checked
    assert "terminal_invalid_regression" in decision.rejection_reasons


def test_pareto_is_diagnostic_and_never_an_active_rejection_reason():
    active = item("active")
    candidate = item("member-up", correct=11)
    decision = evaluate_constraints(candidate, active)
    assert decision.hard_feasible
    assert "member_objective_regression" not in decision.rejection_reasons
    assert decision.member_objective_dominance_passed
    assert decision.member_objective_dominance_passed == decision.derived_team_pareto_passed


@pytest.mark.parametrize("target_delta", [-1, 0, 3])
def test_total_gain_delta_equals_target_gain(target_delta):
    active = item("active")
    candidate = item("candidate", correct=10 + target_delta)
    decision = evaluate_constraints(candidate, active)
    assert decision.target_gain == target_delta
    assert decision.total_gain_delta == target_delta


def test_non_weak_target_does_not_change_minimum_gain():
    initial = (10, 10, 10, 10, 10)
    incumbent = (13, 15, 17, 18, 19)
    active = item(
        "active", correct=17, initial_counts=initial,
        incumbent_counts=incumbent, member_counts=incumbent, target_id=2,
    )
    candidate = item(
        "candidate", correct=18, initial_counts=initial,
        incumbent_counts=incumbent, target_id=2,
    )
    decision = evaluate_constraints(candidate, active)
    assert not decision.target_is_unique_weakest
    assert not decision.target_is_tied_weakest
    assert decision.minimum_gain_delta == 0


def test_tied_weakest_target_does_not_change_minimum_gain():
    initial = (10, 10, 10, 10, 10)
    incumbent = (13, 13, 17, 18, 19)
    active = item(
        "active", correct=13, initial_counts=initial,
        incumbent_counts=incumbent, member_counts=incumbent,
    )
    candidate = item(
        "candidate", correct=16, initial_counts=initial,
        incumbent_counts=incumbent,
    )
    decision = evaluate_constraints(candidate, active)
    assert decision.target_is_tied_weakest
    assert not decision.target_is_unique_weakest
    assert decision.minimum_gain_delta == 0


@pytest.mark.parametrize(
    ("candidate_correct", "expected_minimum_delta"),
    [(14, 2), (16, 3)],
)
def test_unique_weakest_target_minimum_gain_is_capped_by_next_member(
    candidate_correct, expected_minimum_delta
):
    initial = (10, 10, 10, 10, 10)
    incumbent = (12, 15, 17, 18, 19)
    active = item(
        "active", correct=12, initial_counts=initial,
        incumbent_counts=incumbent, member_counts=incumbent,
    )
    candidate = item(
        "candidate", correct=candidate_correct, initial_counts=initial,
        incumbent_counts=incumbent,
    )
    decision = evaluate_constraints(candidate, active)
    assert decision.target_is_unique_weakest
    assert not decision.target_is_tied_weakest
    assert decision.minimum_gain_delta == expected_minimum_delta


def test_monotone_core_guards_imply_derived_team_pareto_property_style():
    initial = (10, 10, 10, 10, 10)
    for gains in ((0, 0, 0, 0, 0), (2, 5, 7, 8, 9), (-2, 1, 3, 3, 6)):
        incumbent = tuple(base + gain for base, gain in zip(initial, gains, strict=True))
        for target_id in range(5):
            active = item(
                "active", correct=incumbent[target_id], vote_count=8,
                initial_counts=initial, incumbent_counts=incumbent,
                member_counts=incumbent, target_id=target_id,
            )
            for target_delta, vote_delta in ((0, 1), (1, 0), (2, 3)):
                candidate = item(
                    "candidate", correct=incumbent[target_id] + target_delta,
                    vote_count=8 + vote_delta, vote_gain=vote_delta,
                    initial_counts=initial, incumbent_counts=incumbent,
                    target_id=target_id,
                )
                decision = evaluate_constraints(candidate, active)
                assert decision.objective_invariant_checked
                assert decision.derived_team_pareto_passed


def test_corrupted_total_gain_metrics_fail_fast():
    active = item("active")
    candidate = item("candidate", correct=11)
    corrupted = replace(
        candidate,
        member_gain=replace(candidate.member_gain, total_gain_count=99),
    )
    with pytest.raises(AssertionError, match="single_target_total_gain_delta_mismatch"):
        evaluate_constraints(corrupted, active)


def test_corrupted_minimum_gain_metrics_break_objective_invariant():
    active = item("active")
    candidate = item("candidate", correct=11)
    corrupted = replace(
        candidate,
        member_gain=replace(candidate.member_gain, minimum_gain_count=-1),
    )
    with pytest.raises(
        AssertionError,
        match="fixed_peer_single_target_objective_invariant_broken",
    ):
        evaluate_constraints(corrupted, active)


def test_member_first_safe_key_removes_redundant_total_gain_dimension():
    candidates = [
        item("gain-1", correct=11, vote_count=9),
        item("gain-2", correct=12, vote_count=9),
        item("vote-2", correct=11, vote_count=10, vote_gain=2),
    ]
    old_key = lambda row: (
        row.member_gain.minimum_gain_count,
        row.team_outcome.vote_correct_count,
        row.member_gain.total_gain_count,
        row.member_gain.target_gain_vs_incumbent,
    )
    new_key = lambda row: member_first_safe_key(row)[:3]
    assert max(candidates, key=old_key).prompt_hash == max(
        candidates, key=new_key
    ).prompt_hash


def test_vote_first_and_member_first_share_feasible_set_but_can_rank_differently():
    initial = (10, 10, 10, 10, 10)
    incumbent = (12, 15, 17, 18, 19)
    active = item(
        "active", correct=12, vote_count=8, initial_counts=initial,
        incumbent_counts=incumbent, member_counts=incumbent,
    )
    member_candidate = item(
        "member", correct=15, vote_count=9, vote_gain=1,
        initial_counts=initial, incumbent_counts=incumbent,
    )
    vote_candidate = item(
        "vote", correct=13, vote_count=10, vote_gain=2,
        initial_counts=initial, incumbent_counts=incumbent,
    )
    candidates = (member_candidate, vote_candidate)
    feasible_for_s2 = {
        row.prompt_hash for row in candidates if evaluate_constraints(row, active).passed
    }
    feasible_for_s3 = {
        row.prompt_hash for row in candidates if evaluate_constraints(row, active).passed
    }
    assert feasible_for_s2 == feasible_for_s3 == {"member", "vote"}
    assert max(candidates, key=vote_first_key).prompt_hash == "vote"
    assert max(candidates, key=member_first_safe_key).prompt_hash == "member"


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
