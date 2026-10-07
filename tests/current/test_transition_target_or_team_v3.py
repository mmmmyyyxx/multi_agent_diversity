"""Zero-API reachability proofs for the V2.2 transition amendment."""

from multi_dataset_diverse_rl.search.initial_competence_transition import (
    InitialCompetenceTargetOrTeamTransitionV3,
    InitialCompetenceTransitionV2,
)
from multi_dataset_diverse_rl.search.schemas import (
    EvaluatedCandidate,
    SearchCandidate,
    TeamEvaluation,
)


def evaluation(target, vote, *, terminal_invalid_delta=0, broken=0, soft=0.0):
    return TeamEvaluation(
        float(vote),
        None,
        (float(target), 22.0, 22.0, 22.0, 22.0),
        aggregation_diagnostics={
            "terminal_invalid_delta": terminal_invalid_delta,
            "team_newly_broken_count": broken,
            "mean_soft_vote_utility": soft,
            "target_invalid_count": 0,
        },
    )


def row(candidate_id, target, vote, **kwargs):
    return EvaluatedCandidate(
        SearchCandidate(candidate_id, f"Solve carefully. {candidate_id}"),
        None,
        evaluation(target, vote, **kwargs),
        True,
        False,
        {"target_member": 0},
    )


def bound(policy):
    policy.bind_initial((22.0,) * 5, "initial")
    return policy


def test_v3_breaks_symmetric_start_deadlock_with_target_only_progress():
    parent = evaluation(22, 22)
    v2 = bound(InitialCompetenceTransitionV2())
    v3 = bound(InitialCompetenceTargetOrTeamTransitionV3())
    additive = row("additive", 25, 22)
    assert v2.select(parent, (additive,)).candidate is None
    assert v3.select(parent, (additive,)).candidate is additive


def test_v3_accepts_either_safe_strict_progress_path():
    policy = bound(InitialCompetenceTargetOrTeamTransitionV3())
    parent = evaluation(24, 22)
    target_only = row("target", 25, 22)
    vote_only = row("vote", 22, 23)
    assert policy.select(parent, (target_only,)).candidate is target_only
    assert policy.select(parent, (vote_only,)).candidate is vote_only


def test_v3_rejects_floor_break_vote_regression_and_no_progress():
    policy = bound(InitialCompetenceTargetOrTeamTransitionV3())
    parent = evaluation(24, 22)
    assert policy.select(parent, (row("below_floor", 21, 23),)).candidate is None
    assert policy.select(parent, (row("vote_regression", 25, 21),)).candidate is None
    assert policy.select(parent, (row("neutral", 24, 22),)).candidate is None
    assert policy.select(parent, (row("invalid", 25, 22, terminal_invalid_delta=1),)).candidate is None


def test_v3_target_score_participates_in_vote_neutral_winner_ranking():
    policy = bound(InitialCompetenceTargetOrTeamTransitionV3())
    parent = evaluation(22, 22)
    weaker = row("weaker", 23, 22, soft=1.0)
    stronger = row("stronger", 25, 22, soft=0.0)
    assert policy.select(parent, (weaker, stronger)).candidate is stronger
