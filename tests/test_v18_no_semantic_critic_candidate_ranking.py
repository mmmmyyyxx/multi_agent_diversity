from scripts.audit_v18_no_semantic_critic_candidate_ranking import (
    ARM,
    SEEDS,
    _train_pareto_dominates,
)


def test_scope_is_three_seed_c_only():
    assert SEEDS == (68, 69, 70)
    assert ARM == "C_NO_SEMANTIC_CRITIC"


def test_train_pareto_dominance_requires_no_worse_all_dimensions():
    winner = {"target_gain": 3, "vote_net_gain": 2, "vote_loss_count": 1}
    assert _train_pareto_dominates(
        {"target_gain": 4, "vote_net_gain": 2, "vote_loss_count": 1}, winner
    )
    assert not _train_pareto_dominates(
        {"target_gain": 4, "vote_net_gain": 1, "vote_loss_count": 0}, winner
    )
