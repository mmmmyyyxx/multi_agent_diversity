from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    stage_a_multichannel_shortlist,
    stage_a_scores,
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
)


def candidate(
    name,
    vote=5,
    member_counts=(10, 10, 10, 10, 10),
    incumbent_counts=(10, 10, 10, 10, 10),
    invalid=0,
    assigned_repair=0,
    soft_utility=0.0,
):
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(member_counts[0], 0.5, invalid, 0.0),
        team_outcome=TeamOutcomeMetrics((), vote, 0.5, (), (), (), soft_utility),
        marginal=CandidateMarginalContribution(
            max(0, vote - 5), max(0, 5 - vote), vote - 5, 0.0,
            0, 0, 0, 0, assigned_repair, 0.0,
        ),
        protection=ProtectionContribution(0, 0),
        member_gain=member_gain_metrics(
            (10, 10, 10, 10, 10),
            incumbent_counts,
            member_counts,
            0,
        ),
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


def test_stage_a_keys_include_required_tie_break_signals():
    row = candidate(
        "signals",
        vote=7,
        member_counts=(12, 11, 11, 11, 11),
        incumbent_counts=(11, 10, 10, 10, 10),
        invalid=1,
        assigned_repair=2,
        soft_utility=0.75,
    )
    scores = stage_a_scores(row)
    assert scores.team_vote_key == (7, 2, 0, 0.75, 2)
    assert scores.worst_member_key == (1, 1, 5, 1, -1)
    assert scores.mean_member_key == (6, 1, 5, 2, -1)
