import asyncio

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system(tmp_path):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(
        out_dir=str(tmp_path), agents=5, candidate_reuse_recorded_rollouts=True,
        candidate_eval_execution_mode="factorized_cached", solver_rollout_singleflight=True,
    )
    system.agents = [AgentState(f"base-{idx}") for idx in range(5)]
    system.solver_rollout_cache = {}
    system.solver_rollout_inflight = {}
    system.solver_rollout_inflight_lock = asyncio.Lock()
    system.solver_call_semaphore = asyncio.Semaphore(50)
    return system


def test_factorized_cold_and_warm_cache_call_counts(tmp_path):
    system = _system(tmp_path)
    calls = []

    async def fake_solve(question, agent_id, prompt):
        calls.append((question, agent_id, prompt))
        return f"reasoning {agent_id} {prompt} FINAL_ANSWER: A", "A"

    system.solve_once = fake_solve
    batch = [{"question": f"q{i}", "answer": "A"} for i in range(4)]
    pool = [
        {"prompt": prompt, "parent_id": parent, "tcs_call_group_id": group}
        for prompt, parent, group in [
            ("base-0", "p0", "g0"), ("p1", "p1", "g1"), ("p2", "p2", "g2"),
            ("p3", "p3", "g3"), ("p1", "p4", "g4"), ("p2", "p5", "g5"),
        ]
    ]
    stats = asyncio.run(system._prewarm_factorized_candidate_rollouts(agent_id=0, eval_batch=batch, peer_prompts=[f"base-{idx}" for idx in range(5)], candidate_pool=pool))
    assert len(calls) == (5 - 1 + 4) * 4
    assert stats["candidate_eval_naive_rollout_request_count"] == 6 * 5 * 4
    assert stats["candidate_eval_factorized_rollout_request_count"] == (5 - 1 + 4) * 4
    assert stats["candidate_eval_unique_target_prompt_count"] == 4
    assert stats["candidate_eval_duplicate_target_prompt_count"] == 2
    assert stats["candidate_eval_solver_api_call_count"] == 32

    warm_stats = asyncio.run(system._prewarm_factorized_candidate_rollouts(agent_id=0, eval_batch=batch, peer_prompts=[f"base-{idx}" for idx in range(5)], candidate_pool=pool))
    assert len(calls) == 32
    assert warm_stats["candidate_eval_solver_api_call_count"] == 0
    assert warm_stats["candidate_eval_memory_cache_hit_count"] == 32


def test_duplicate_candidate_objects_keep_distinct_provenance(tmp_path):
    system = _system(tmp_path)
    calls = []

    async def fake_solve(question, agent_id, prompt):
        calls.append((question, agent_id, prompt))
        return "reasoning FINAL_ANSWER: A", "A"

    system.solve_once = fake_solve
    batch = [{"question": "q", "answer": "A"}]
    pool = [
        {"candidate_id": "candidate-a", "prompt": "same", "parent_id": "parent-a", "tcs_call_group_id": "group-a"},
        {"candidate_id": "candidate-b", "prompt": "same", "parent_id": "parent-b", "tcs_call_group_id": "group-b"},
    ]
    stats = asyncio.run(system._prewarm_factorized_candidate_rollouts(agent_id=0, eval_batch=batch, peer_prompts=[f"base-{idx}" for idx in range(5)], candidate_pool=pool))
    assert stats["candidate_eval_candidate_object_count"] == 2
    assert stats["candidate_eval_unique_target_prompt_count"] == 2  # active target is additionally evaluated as baseline
    assert [row["parent_id"] for row in pool] == ["parent-a", "parent-b"]
    assert [row["tcs_call_group_id"] for row in pool] == ["group-a", "group-b"]
