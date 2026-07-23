from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    stage_a_multichannel_shortlist,
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
)


def candidate(name, vote=5, member_counts=(10, 10, 10, 10, 10)):
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(member_counts[0], 0.5, 0, 0.0),
        team_outcome=TeamOutcomeMetrics((), vote, 0.5, (), (), (), 0.0),
        marginal=CandidateMarginalContribution(
            max(0, vote - 5), max(0, 5 - vote), vote - 5, 0.0,
            0, 0, 0, 0, 0, 0.0,
        ),
        protection=ProtectionContribution(0, 0),
        member_gain=member_gain_metrics((10, 10, 10, 10, 10), member_counts),
    )


def test_member_aware_channels_cover_vote_worst_and_mean_objectives():
    rows = [
        candidate("vote", vote=8),
        candidate("worst", member_counts=(12, 12, 12, 12, 12)),
        candidate("mean", member_counts=(10, 10, 10, 10, 25)),
        candidate("balanced", vote=7, member_counts=(11, 11, 11, 11, 12)),
    ]
    selected, decisions = stage_a_multichannel_shortlist(
        rows, channel_top_k=1, total_budget=4
    )
    assert len(selected) == 4
    assert "team_vote" in decisions["vote"].selected_by_channels
    assert "worst_member" in decisions["worst"].selected_by_channels
    assert "mean_member" in decisions["mean"].selected_by_channels


def test_pareto_front_rejects_strictly_dominated_member_vector():
    rows = [
        candidate("strong", vote=7, member_counts=(11, 11, 11, 11, 12)),
        candidate("dominated", vote=5, member_counts=(9, 10, 10, 10, 10)),
    ]
    _, decisions = stage_a_multichannel_shortlist(rows, channel_top_k=1, total_budget=1)
    assert decisions["strong"].pareto_front < decisions["dominated"].pareto_front
