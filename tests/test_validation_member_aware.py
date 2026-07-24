import asyncio

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample, PromptAnswer
from multi_dataset_diverse_rl.evaluation.validation import (
    DatasetEvaluationRow,
    DatasetMetrics,
    ValidationProbeEvaluator,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


def metrics(vote_count, counts=None, invalid=0.0, utility=0.5, c0=1):
    values = tuple(counts or [7] * 5)
    size = 10
    return DatasetMetrics(
        vote_correct_count=vote_count,
        per_agent_correct_counts=values,
        plurality_vote_acc=vote_count / size,
        vote_acc=vote_count / size,
        mean_individual_acc=sum(values) / (size * 5),
        min_individual_acc=min(values) / size,
        per_agent_acc=tuple(value / size for value in values),
        mean_soft_vote_utility=utility,
        c0_count=c0,
        mean_invalid_rate=invalid,
        tie_count=0,
        tie_rate=0.0,
        rows=(
            DatasetEvaluationRow("q1", vote_count >= 6, False, 1, 2, -1),
            DatasetEvaluationRow("q2", vote_count >= 7, False, 1, 2, -1),
        ),
    )


def test_validation_filters_initial_member_feasibility_then_ranks_member_objective():
    system = PromptEnsembleOptimizationSystem(Config())
    initial = metrics(5)
    infeasible = metrics(9, [6, 9, 9, 9, 9])
    lower_min_gain = metrics(7, [7, 9, 9, 9, 9])
    higher_min_gain = metrics(6, [8, 8, 8, 8, 8])
    assert system.validation_key(infeasible, initial, 1) is None
    assert system.validation_key(higher_min_gain, initial, 1) > system.validation_key(
        lower_min_gain, initial, 1
    )


def test_validation_uses_complete_declared_preference_order():
    system = PromptEnsembleOptimizationSystem(Config())
    initial = metrics(5, [7] * 5, invalid=0.2, utility=0.5, c0=2)

    def key(
        *,
        vote=6,
        counts=(8, 8, 7, 7, 7),
        invalid=0.1,
        utility=0.6,
        c0=1,
        epoch=1,
    ):
        return system.validation_key(
            metrics(vote, counts, invalid=invalid, utility=utility, c0=c0),
            initial,
            epoch,
        )

    assert key(counts=(8, 8, 8, 8, 8)) > key(counts=(9, 9, 9, 9, 7))
    assert key(vote=7) > key(vote=6)
    assert key(counts=(9, 8, 7, 7, 7)) > key(counts=(8, 8, 7, 7, 7))
    assert key(counts=(8, 8, 7, 7, 7)) > key(counts=(9, 7, 7, 7, 7))
    assert key(utility=0.7) > key(utility=0.6)
    assert key(c0=0) > key(c0=1)
    assert key(invalid=0.05) > key(invalid=0.1)
    assert key(epoch=1) > key(epoch=2)


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
