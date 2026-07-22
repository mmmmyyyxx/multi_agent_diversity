from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


def metrics(vote, mean, agents=None, invalid=0.0, utility=0.5, c0=1):
    return {
        "plurality_vote_acc": vote, "mean_individual_acc": mean,
        "min_individual_acc": min(agents or [mean]), "per_agent_acc": agents or [mean] * 5,
        "mean_invalid_rate": invalid, "mean_soft_vote_utility": utility, "c0_count": c0,
        "rows": [
            {"question_hash": "q1", "vote_correct": vote >= 0.6},
            {"question_hash": "q2", "vote_correct": vote >= 0.7},
        ],
    }


def test_validation_filters_competence_then_ranks_vote_first():
    system = PromptEnsembleOptimizationSystem(Config())
    initial = metrics(0.5, 0.7)
    infeasible = metrics(0.9, 0.69, [0.69] * 5)
    lower_vote_higher_mean = metrics(0.6, 0.9, [0.9] * 5)
    higher_vote = metrics(0.7, 0.7)
    assert system.validation_key(infeasible, initial, 1) is None
    assert (
        system.validation_key(higher_vote, initial, 1)
        > system.validation_key(lower_vote_higher_mean, initial, 1)
    )
