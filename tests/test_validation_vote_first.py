import asyncio

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample, PromptAnswer
from multi_dataset_diverse_rl.evaluation.validation import (
    DatasetEvaluationRow,
    DatasetMetrics,
    ValidationProbeEvaluator,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


def metrics(vote, mean, agents=None, invalid=0.0, utility=0.5, c0=1):
    values = tuple(agents or [mean] * 5)
    return DatasetMetrics(
        plurality_vote_acc=vote,
        vote_acc=vote,
        mean_individual_acc=mean,
        min_individual_acc=min(values),
        per_agent_acc=values,
        mean_soft_vote_utility=utility,
        c0_count=c0,
        mean_invalid_rate=invalid,
        tie_count=0,
        tie_rate=0.0,
        rows=(
            DatasetEvaluationRow("q1", vote >= 0.6, False, 1, 2, -1),
            DatasetEvaluationRow("q2", vote >= 0.7, False, 1, 2, -1),
        ),
    )


def test_validation_filters_competence_then_ranks_vote_first():
    system = PromptEnsembleOptimizationSystem(Config())
    initial = metrics(0.5, 0.7)
    infeasible = metrics(0.9, 0.69, [0.69] * 5)
    lower_vote_higher_mean = metrics(0.6, 0.9, [0.9] * 5)
    higher_vote = metrics(0.7, 0.7)
    assert system.validation_key(infeasible, initial, 1) is None
    assert system.validation_key(higher_vote, initial, 1) > system.validation_key(lower_vote_higher_mean, initial, 1)


def test_validation_prompt_question_cache_is_shared_across_agents():
    calls = []

    async def solve(question, agent_id, prompt):
        calls.append((question, agent_id, prompt))
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    evaluator = ValidationProbeEvaluator(
        [ProbeExample("q1", "h1", "A"), ProbeExample("q2", "h2", "A")],
        model_identity="model",
        parser_version="parser",
        temperature=0.0,
        seed=42,
    )

    async def run():
        await asyncio.gather(*(
            evaluator.evaluate_prompt(agent, "shared", "same-hash", solve) for agent in range(5)
        ))

    asyncio.run(run())
    assert len(calls) == 2
    assert evaluator.cache_misses == 2
    assert evaluator.cache_hits == 8
