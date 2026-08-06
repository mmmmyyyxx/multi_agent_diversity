from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    stage_a_rcru_shortlist,
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
)
from multi_dataset_diverse_rl.responsibility_contribution import (
    CoalitionContributionMetrics,
    PromptEditMetrics,
    ResponsibilityContributionMetrics,
    ResponsibilityUtilityKind,
    ResponsibilityUtilityMetrics,
    RobustSupportMetrics,
    coalition_contribution_metrics,
    deterministic_paired_bootstrap,
    evaluate_robust_contribution_constraints,
    prompt_edit_metrics,
    responsibility_contribution_pareto_front,
    responsibility_contribution_pareto_front_numbers,
    responsibility_utility,
    responsibility_utility_metrics,
    robust_contribution_key,
)


def rcru_metrics(
    *,
    utility_delta=0,
    utility_total=5,
    deltas=(),
    contribution=0,
    contribution_delta=0,
    lcb=0.0,
    edit_tokens=3,
    edit_ratio=0.1,
    growth=2,
):
    deltas = tuple(deltas)
    return ResponsibilityContributionMetrics(
        utility=ResponsibilityUtilityMetrics(
            repair_lane="direct_flip",
            active_residual_count=max(1, len(deltas)),
            utility_total=utility_total,
            incumbent_utility_total=utility_total - utility_delta,
            utility_delta=utility_delta,
            positive_support_count=sum(value > 0 for value in deltas),
            negative_support_count=sum(value < 0 for value in deltas),
            unchanged_support_count=sum(value == 0 for value in deltas),
            per_example_deltas=deltas,
            per_example_question_hashes=tuple(
                f"q{index}" for index in range(len(deltas))
            ),
        ),
        coalition=CoalitionContributionMetrics(
            positive_pivotal_count=max(contribution, 0),
            negative_pivotal_count=max(-contribution, 0),
            net_contribution=contribution,
            incumbent_positive_pivotal_count=max(
                contribution - contribution_delta, 0
            ),
            incumbent_negative_pivotal_count=max(
                -(contribution - contribution_delta), 0
            ),
            incumbent_net_contribution=contribution - contribution_delta,
            positive_pivotal_delta=0,
            negative_pivotal_delta=0,
            net_contribution_delta=contribution_delta,
        ),
        robust_support=RobustSupportMetrics(
            bootstrap_replicates=1000,
            bootstrap_lower_quantile=0.10,
            bootstrap_mean_delta=float(utility_delta),
            bootstrap_lcb=lcb,
            deterministic_seed_hash="seed",
        ),
        edit=PromptEditMetrics(
            parent_character_count=10,
            candidate_character_count=10 + growth,
            character_growth=growth,
            parent_token_count=5,
            candidate_token_count=5,
            inserted_token_count=edit_tokens,
            deleted_token_count=0,
            replaced_token_count=0,
            total_edit_token_count=edit_tokens,
            normalized_edit_ratio=edit_ratio,
        ),
    )


def candidate(
    name,
    *,
    target=10,
    vote=8,
    terminal_invalid=0,
    metrics=None,
):
    incumbent_counts = (10, 10, 10, 10, 10)
    candidate_counts = (target, 10, 10, 10, 10)
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(
            target, target / 20, terminal_invalid, terminal_invalid / 20,
            terminal_invalid,
        ),
        team_outcome=TeamOutcomeMetrics(
            (), vote, vote / 20, (), (), (), 0.0
        ),
        marginal=CandidateMarginalContribution(
            vote_gain_count=max(0, vote - 8),
            vote_loss_count=max(0, 8 - vote),
            net_vote_delta=vote - 8,
            soft_utility_delta=0.0,
            coverage_gain_count=0,
            coverage_loss_count=0,
            dominant_wrong_exit_count=0,
            dominant_wrong_join_count=0,
            assigned_residual_repair_count=0,
            assigned_residual_utility_delta=0.0,
        ),
        protection=ProtectionContribution(0, 0),
        member_gain=member_gain_metrics(
            incumbent_counts, incumbent_counts, candidate_counts, 0
        ),
        responsibility_contribution=metrics,
    )


def test_three_lane_utility_definitions():
    assert responsibility_utility(
        ResponsibilityUtilityKind.COVERAGE,
        target_correct=True,
        team_vote_correct=False,
        plurality_margin=-2,
    ) == 1
    assert responsibility_utility(
        "coverage",
        target_correct=False,
        team_vote_correct=True,
        plurality_margin=1,
    ) == 0
    assert responsibility_utility(
        "direct_flip",
        target_correct=False,
        team_vote_correct=True,
        plurality_margin=1,
    ) == 1
    assert responsibility_utility(
        "direct_flip",
        target_correct=True,
        team_vote_correct=False,
        plurality_margin=-1,
    ) == 0
    assert (
        responsibility_utility(
            "margin_support",
            target_correct=False,
            team_vote_correct=False,
            plurality_margin=-2,
        )
        - responsibility_utility(
            "margin_support",
            target_correct=False,
            team_vote_correct=False,
            plurality_margin=-4,
        )
        == 2
    )
    assert min(-3, 0) - min(-2, 0) == -1
    assert min(1, 0) - min(-1, 0) == 1
    assert min(3, 0) - min(1, 0) == 0


def test_active_slice_isolation_excludes_other_lane_and_outside_portfolio():
    incumbent = {
        "active": SimpleNamespace(
            team_correctness=(False,), vote_correct=False, plurality_margin=-2
        ),
        "other-lane": SimpleNamespace(
            team_correctness=(False,), vote_correct=False, plurality_margin=-3
        ),
        "outside": SimpleNamespace(
            team_correctness=(True,), vote_correct=True, plurality_margin=2
        ),
    }
    candidate_states = {
        "active": SimpleNamespace(
            team_correctness=(True,), vote_correct=False, plurality_margin=-1
        ),
        "other-lane": SimpleNamespace(
            team_correctness=(False,), vote_correct=True, plurality_margin=1
        ),
        "outside": SimpleNamespace(
            team_correctness=(False,), vote_correct=False, plurality_margin=-4
        ),
    }
    metrics = responsibility_utility_metrics(
        repair_lane="coverage",
        active_question_hashes=("active",),
        incumbent_states=incumbent,
        candidate_states=candidate_states,
        target_agent_id=0,
    )
    assert metrics.active_residual_count == 1
    assert metrics.utility_delta == 1
    assert metrics.per_example_question_hashes == ("active",)


def test_leave_one_out_contribution_and_fixed_peer_invariant():
    metrics = coalition_contribution_metrics(
        incumbent_full_vote_vector=(True, False, True, False),
        candidate_full_vote_vector=(True, False, True, False),
        incumbent_peer_vote_vector=(False, True, True, False),
        candidate_peer_vote_vector=(False, True, True, False),
    )
    assert metrics.positive_pivotal_count == 1
    assert metrics.negative_pivotal_count == 1
    assert metrics.net_contribution == 0
    with pytest.raises(
        AssertionError, match="fixed_peer_leave_one_out_state_changed"
    ):
        coalition_contribution_metrics(
            incumbent_full_vote_vector=(True,),
            candidate_full_vote_vector=(True,),
            incumbent_peer_vote_vector=(False,),
            candidate_peer_vote_vector=(True,),
        )


def test_responsibility_swap_passes_but_outside_lane_gain_does_not():
    incumbent = candidate("incumbent")
    swap = candidate(
        "swap",
        metrics=rcru_metrics(
            utility_delta=2, utility_total=7, deltas=(1, 1, 0, 0), lcb=0.0
        ),
    )
    decision = evaluate_robust_contribution_constraints(swap, incumbent)
    assert decision.passed
    outside = candidate(
        "outside",
        target=13,
        metrics=rcru_metrics(
            utility_delta=0, utility_total=5, deltas=(0, 0)
        ),
    )
    rejected = evaluate_robust_contribution_constraints(outside, incumbent)
    assert not rejected.passed
    assert "no_vote_or_lane_progress" in rejected.rejection_reasons


def test_lane_regression_rejects_even_with_vote_gain():
    incumbent = candidate("incumbent")
    row = candidate(
        "vote-up-lane-down",
        vote=9,
        metrics=rcru_metrics(
            utility_delta=-1, utility_total=4, deltas=(-1, 0), lcb=-1
        ),
    )
    decision = evaluate_robust_contribution_constraints(row, incumbent)
    assert "active_lane_regression" in decision.rejection_reasons


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"target": 9}, "target_regression"),
        ({"vote": 7}, "team_vote_regression"),
        ({"terminal_invalid": 1}, "terminal_invalid_regression"),
    ),
)
def test_layer_one_guards_are_independent(kwargs, reason):
    incumbent = candidate("incumbent")
    row = candidate(
        "unsafe",
        metrics=rcru_metrics(
            utility_delta=2, utility_total=7, deltas=(1, 1), lcb=0
        ),
        **kwargs,
    )
    decision = evaluate_robust_contribution_constraints(row, incumbent)
    assert reason in decision.rejection_reasons


def test_role_only_support_and_vote_gain_rules():
    incumbent = candidate("incumbent")
    one = candidate(
        "one",
        metrics=rcru_metrics(
            utility_delta=1, utility_total=6, deltas=(1, 0, 0), lcb=0
        ),
    )
    assert "insufficient_lane_support" in (
        evaluate_robust_contribution_constraints(one, incumbent).rejection_reasons
    )
    two = candidate(
        "two",
        metrics=rcru_metrics(
            utility_delta=2, utility_total=7, deltas=(1, 1, 0), lcb=0
        ),
    )
    assert evaluate_robust_contribution_constraints(two, incumbent).passed
    vote_gain = candidate(
        "vote",
        vote=9,
        metrics=rcru_metrics(
            utility_delta=1, utility_total=6, deltas=(1, 0), lcb=-1
        ),
    )
    assert evaluate_robust_contribution_constraints(vote_gain, incumbent).passed


def test_bootstrap_is_reproducible_and_can_reject_role_only_gain():
    kwargs = dict(
        run_seed=42,
        team_state_version=3,
        update_index=7,
        target_agent_id=2,
        candidate_prompt_hash="candidate",
        active_lane="margin_support",
        active_question_hashes=("q1", "q2", "q3", "q4"),
    )
    first = deterministic_paired_bootstrap((-5, 2, 2, 2), **kwargs)
    second = deterministic_paired_bootstrap((-5, 2, 2, 2), **kwargs)
    assert first == second
    assert first.bootstrap_lcb < 0
    incumbent = candidate("incumbent")
    row = candidate(
        "unstable",
        metrics=rcru_metrics(
            utility_delta=1,
            utility_total=6,
            deltas=(-5, 2, 2, 2),
            lcb=first.bootstrap_lcb,
        ),
    )
    decision = evaluate_robust_contribution_constraints(row, incumbent)
    assert "negative_lane_bootstrap_lcb" in decision.rejection_reasons


def test_contribution_pareto_front_and_minimal_edit_tie_break():
    a = candidate(
        "a",
        vote=10,
        metrics=rcru_metrics(
            utility_delta=1, utility_total=6, deltas=(1, 0), contribution=0
        ),
    )
    b = candidate(
        "b",
        vote=9,
        metrics=rcru_metrics(
            utility_delta=3, utility_total=8, deltas=(1, 1), contribution=1
        ),
    )
    c = candidate(
        "c",
        vote=8,
        metrics=rcru_metrics(
            utility_delta=1, utility_total=6, deltas=(1, 1), contribution=0
        ),
    )
    assert {row.prompt_hash for row in responsibility_contribution_pareto_front((a, b, c))} == {
        "a",
        "b",
    }
    assert responsibility_contribution_pareto_front_numbers((a, b, c)) == {
        "a": 1,
        "b": 1,
        "c": 2,
    }
    large = replace(
        b,
        prompt_hash="large",
        responsibility_contribution=replace(
            b.responsibility_contribution,
            edit=replace(
                b.responsibility_contribution.edit,
                total_edit_token_count=10,
                normalized_edit_ratio=0.5,
            ),
        ),
    )
    small = replace(
        b,
        prompt_hash="small",
        responsibility_contribution=replace(
            b.responsibility_contribution,
            edit=replace(
                b.responsibility_contribution.edit,
                total_edit_token_count=2,
                normalized_edit_ratio=0.1,
            ),
        ),
    )
    assert max((large, small), key=robust_contribution_key).prompt_hash == "small"


def test_programmatic_edit_metric_and_rcru_stage_a_channels():
    edit = prompt_edit_metrics("use rule one", "use careful rule one")
    assert edit.inserted_token_count == 1
    assert edit.total_edit_token_count == 1
    rows = [
        candidate(
            "vote",
            vote=10,
            metrics=rcru_metrics(
                utility_delta=1, utility_total=6, deltas=(1,), contribution=0
            ),
        ),
        candidate(
            "lane",
            vote=8,
            metrics=rcru_metrics(
                utility_delta=3, utility_total=8, deltas=(1, 1), contribution=1
            ),
        ),
        candidate(
            "coalition",
            vote=8,
            metrics=rcru_metrics(
                utility_delta=1, utility_total=6, deltas=(1,), contribution=3
            ),
        ),
    ]
    selected, decisions = stage_a_rcru_shortlist(
        rows, channel_top_k=1, total_budget=3
    )
    assert len(selected) == 3
    assert "team_vote" in decisions["vote"].selected_by_channels
    assert "lane_fulfillment" in decisions["lane"].selected_by_channels
    assert (
        "coalition_contribution"
        in decisions["coalition"].selected_by_channels
    )


def test_missing_metrics_fails_fast():
    with pytest.raises(ValueError, match="rcru_metrics_missing_for_candidate"):
        evaluate_robust_contribution_constraints(
            candidate("missing"), candidate("incumbent")
        )
