from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    stage_a_multichannel_shortlist,
)
from multi_dataset_diverse_rl.responsibility import CandidateMarginalContribution, ProtectionContribution


def candidate(name, accuracy=0, vote=0, soft=0.0, coverage=0, residual=0.0):
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(accuracy, accuracy / 10, 0, 0.0),
        team_outcome=TeamOutcomeMetrics((), 0.0, (), (), (), 0.0),
        marginal=CandidateMarginalContribution(
            vote_gain_count=max(0, vote),
            vote_loss_count=max(0, -vote),
            net_vote_delta=vote,
            soft_utility_delta=soft,
            coverage_gain_count=coverage,
            coverage_loss_count=0,
            dominant_wrong_exit_count=0,
            dominant_wrong_join_count=0,
            assigned_residual_repair_count=coverage,
            assigned_residual_utility_delta=residual,
        ),
        protection=ProtectionContribution(0, 0),
    )


def test_three_channels_union_deduplicates_and_obeys_budget():
    rows = [
        candidate("accuracy", accuracy=9),
        candidate("vote", vote=3),
        candidate("coverage", coverage=4),
        candidate("shared", accuracy=10, vote=4, coverage=5),
        candidate("other", accuracy=1),
    ]
    selected, decisions = stage_a_multichannel_shortlist(rows, channel_top_k=2, total_budget=4)
    hashes = {row.prompt_hash for row in selected}
    assert len(hashes) == 4
    assert {"accuracy", "vote", "coverage", "shared"} == hashes
    assert "accuracy" in decisions["accuracy"].selected_by_channels
    assert "vote" in decisions["vote"].selected_by_channels
    assert "responsibility" in decisions["coverage"].selected_by_channels


def test_pareto_rank_fill_is_not_prompt_hash_fill():
    rows = [
        candidate("accuracy", accuracy=9),
        candidate("vote", vote=2),
        candidate("coverage", residual=2.0),
        candidate("aaa_dominated", accuracy=0, vote=-2),
        candidate("zzz_balanced", accuracy=4, vote=1, coverage=1),
    ]
    selected, decisions = stage_a_multichannel_shortlist(rows, channel_top_k=1, total_budget=4)
    hashes = {row.prompt_hash for row in selected}
    assert "zzz_balanced" in hashes
    assert "aaa_dominated" not in hashes
    assert decisions["zzz_balanced"].pareto_front < decisions["aaa_dominated"].pareto_front
