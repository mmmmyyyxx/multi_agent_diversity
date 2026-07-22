import asyncio

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptQuestionEvaluator
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.utils import normalize_prompt_text


def test_same_prompt_question_across_agents_calls_solver_once():
    evaluator = PromptQuestionEvaluator(
        model_request_identity="model-request",
        parser_version="parser",
        temperature=0.0,
        decoding_seed=42,
    )
    calls = []

    async def solve(question, agent_id, prompt):
        calls.append((question, agent_id, prompt))
        return PromptAnswer("A", "reason\nFINAL_ANSWER: A", True)

    async def run():
        return await asyncio.gather(*(
            evaluator.evaluate(
                question="question",
                question_hash="question-hash",
                prompt="same prompt",
                prompt_hash="prompt-hash",
                agent_id=agent_id,
                solve=solve,
            )
            for agent_id in range(5)
        ))

    outputs = asyncio.run(run())
    assert len(calls) == 1
    assert len(outputs) == 5
    assert evaluator.cache_misses == 1
    assert evaluator.cache_hits == 4


def test_optimization_validation_and_test_share_sampling_semantics(tmp_path):
    calls = []

    async def solve(question, agent_id, prompt):
        calls.append((question, agent_id, prompt))
        return PromptAnswer("A", "reason\nFINAL_ANSWER: A", True)

    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(out_dir=str(tmp_path), answer_format="option_letter"),
        solver=solve,
    )
    data = [{"question": "same question", "answer": "A"}]

    async def run():
        await system.initialize_fixed_probe(data)
        system.validation_probe = system.build_validation_probe(data)
        await system.evaluate_dataset(data, validation=True)
        await system.evaluate_dataset(data)

    asyncio.run(run())
    assert len(calls) == 1
    assert system.prompt_question_evaluator.cache_misses == 1


def test_multiline_prompt_structure_is_preserved_in_hash_and_rollout(tmp_path):
    raw = "\r\nStep 1: inspect evidence   \r\n  - keep indentation\r\nStep 2: verify\r\n\r\n"
    canonical = "Step 1: inspect evidence\n  - keep indentation\nStep 2: verify"
    flattened = "Step 1: inspect evidence - keep indentation Step 2: verify"
    assert normalize_prompt_text(raw) == canonical

    seen = []

    async def solve(_question, _agent_id, prompt):
        seen.append(prompt)
        return PromptAnswer("A", "reason\nFINAL_ANSWER: A", True)

    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)), solver=solve)
    assert system.prompt_hash(raw) == system.prompt_hash(canonical)
    assert system.prompt_hash(canonical) != system.prompt_hash(flattened)

    evaluator = system.prompt_question_evaluator
    asyncio.run(evaluator.evaluate(
        question="q",
        question_hash="q-hash",
        prompt=canonical,
        prompt_hash=system.prompt_hash(canonical),
        agent_id=0,
        solve=solve,
    ))
    assert seen == [canonical]
