import asyncio
import sqlite3

import pytest

from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import PersistentSolverCache
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer, PromptQuestionEvaluator


def metadata():
    return {
        "solver_model": "solver",
        "endpoint_identity": "endpoint",
        "output_contract_version": "contract",
        "max_tokens": 100,
    }


def evaluator(path, seed):
    return PromptQuestionEvaluator(
        model_request_identity="request",
        parser_version="parser",
        temperature=0.0,
        decoding_seed=seed,
        cache_metadata=metadata(),
        shared_cache=PersistentSolverCache(path),
    )


def evaluate(instance, solve):
    return instance.evaluate(
        question="question",
        question_hash="question-hash",
        prompt="prompt",
        prompt_hash="prompt-hash",
        agent_id=0,
        solve=solve,
    )


def test_persistent_cache_shares_observation_across_evaluators_and_isolates_seed(tmp_path):
    path = tmp_path / "shared.sqlite"
    calls = []

    async def solve_a(*_args):
        calls.append("a")
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    async def solve_b(*_args):
        calls.append("b")
        return PromptAnswer("B", "FINAL_ANSWER: B", True)

    first = asyncio.run(evaluate(evaluator(path, 42), solve_a))
    second = asyncio.run(evaluate(evaluator(path, 42), solve_b))
    third = asyncio.run(evaluate(evaluator(path, 43), solve_b))

    assert first.answer == second.answer == "A"
    assert third.answer == "B"
    assert calls == ["a", "b"]
    assert PersistentSolverCache(path).ready_entry_count() == 2


def test_persistent_cache_claim_prevents_concurrent_duplicate_calls(tmp_path):
    path = tmp_path / "shared.sqlite"
    left = evaluator(path, 42)
    right = evaluator(path, 42)
    calls = 0

    async def solve(*_args):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.1)
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    async def run():
        return await asyncio.gather(evaluate(left, solve), evaluate(right, solve))

    answers = asyncio.run(run())
    assert [answer.answer for answer in answers] == ["A", "A"]
    assert calls == 1
    assert left.shared_cache.misses + right.shared_cache.misses == 1
    assert left.shared_cache.hits + right.shared_cache.hits == 1


def test_dead_cache_owner_is_taken_over_without_stale_timeout(tmp_path):
    path = tmp_path / "shared.sqlite"
    abandoned = PersistentSolverCache(path, stale_after_seconds=9999)
    cache_key = evaluator(path, 42).key("prompt-hash", "question-hash")
    state, _ = abandoned._claim_or_read(
        cache_key,
        {
            **metadata(),
            "model_request_identity": "request",
            "parser_version": "parser",
            "temperature": 0.0,
            "evaluation_replica_seed": 42,
            "prompt_hash": "prompt-hash",
            "question_hash": "question-hash",
        },
    )
    assert state == "owner"
    with abandoned._connect() as connection:
        connection.execute(
            "UPDATE solver_cache SET owner_id = ? WHERE cache_key = ?",
            ("99999999:dead", cache_key),
        )

    calls = 0

    async def solve(*_args):
        nonlocal calls
        calls += 1
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    answer = asyncio.run(evaluate(evaluator(path, 42), solve))
    assert answer.answer == "A"
    assert calls == 1


def test_incompatible_persistent_cache_schema_fails_explicitly(tmp_path):
    path = tmp_path / "shared.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE cache_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO cache_metadata(key, value) VALUES ('schema_version', 'legacy')"
        )
    with pytest.raises(ValueError, match="schema mismatch"):
        PersistentSolverCache(path)
