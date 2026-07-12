import asyncio

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system(tmp_path):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(out_dir=str(tmp_path), agents=1, candidate_reuse_recorded_rollouts=True, solver_rollout_singleflight=True)
    system.agents = [AgentState("prompt")]
    system.solver_rollout_cache = {}
    system.solver_rollout_inflight = {}
    system.solver_rollout_inflight_lock = asyncio.Lock()
    system.solver_call_semaphore = asyncio.Semaphore(20)
    return system


def test_singleflight_coalesces_concurrent_rollouts(tmp_path):
    system = _system(tmp_path)
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {"trace": "trace", "answer": "A"}

    async def run_all():
        return await asyncio.gather(*[
            system.get_or_create_solver_rollout(cache_key="same", lookup=lambda: None, call_factory=factory)
            for _ in range(10)
        ])

    results = asyncio.run(run_all())
    assert calls == 1
    assert sum(origin == "api_call" for _, origin in results) == 1
    assert sum(origin == "inflight_reuse" for _, origin in results) == 9
    assert all(row == {"trace": "trace", "answer": "A"} for row, _ in results)
    assert system.solver_rollout_inflight == {}


def test_singleflight_propagates_failure_and_cleans_up(tmp_path):
    system = _system(tmp_path)
    calls = 0

    async def failing_factory():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        raise RuntimeError("solver failed")

    async def run_all():
        outcomes = await asyncio.gather(*[
            system.get_or_create_solver_rollout(cache_key="same", lookup=lambda: None, call_factory=failing_factory)
            for _ in range(3)
        ], return_exceptions=True)
        return outcomes

    outcomes = asyncio.run(run_all())
    assert calls == 1
    assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)
    assert system.solver_rollout_inflight == {}

    async def successful_factory():
        nonlocal calls
        calls += 1
        return {"trace": "recovered", "answer": "B"}

    row, origin = asyncio.run(system.get_or_create_solver_rollout(cache_key="same", lookup=lambda: None, call_factory=successful_factory))
    assert calls == 2
    assert origin == "api_call"
    assert row["answer"] == "B"
