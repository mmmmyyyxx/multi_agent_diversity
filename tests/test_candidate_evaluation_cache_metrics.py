import asyncio
import copy

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.tasks import get_task_spec


def _system(tmp_path, mode):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(
        out_dir=str(tmp_path), agents=3, task_type="bbh", answer_format="option_letter",
        reward_mode="vote_useful_diversity", candidate_selection_mode="vote_pareto", candidate_eval_execution_mode=mode,
        candidate_reuse_recorded_rollouts=True, solver_rollout_singleflight=True,
    )
    system.task_spec = get_task_spec("bbh")
    system.agents = [AgentState(f"base-{idx}") for idx in range(3)]
    system.solver_rollout_cache = {}
    system.solver_rollout_inflight = {}
    system.solver_rollout_inflight_lock = asyncio.Lock()
    system.solver_call_semaphore = asyncio.Semaphore(20)
    system.joint_diversity_cache = {}
    system.embedding_cache = {}
    system._encode_trace_document = lambda text: [float((sum(map(ord, text)) % 7) + 1), 1.0]
    return system


def test_factorized_cache_preserves_candidate_metrics(tmp_path):
    calls = {"legacy": 0, "factorized": 0}

    def install_solver(system, label):
        async def fake_solve(question, agent_id, prompt):
            calls[label] += 1
            answer = "B" if "candidate-b" in prompt else "A"
            return f"reasoning procedure uses several explicit checks before deciding. FINAL_ANSWER: {answer}", answer
        system.solve_once = fake_solve

    batch = [{"question": "Which option is correct?", "answer": "A"}, {"question": "Choose the right answer.", "answer": "A"}]
    prompts = ["candidate-a", "candidate-b"]
    legacy = _system(tmp_path / "legacy", "legacy")
    factorized = _system(tmp_path / "factorized", "factorized_cached")
    install_solver(legacy, "legacy")
    install_solver(factorized, "factorized")

    legacy_metrics = [asyncio.run(legacy.evaluate_candidate_prompt(0, prompt, ["base-0", "base-1", "base-2"], batch)) for prompt in prompts]
    pool = [{"prompt": prompt} for prompt in prompts]
    asyncio.run(factorized._prewarm_factorized_candidate_rollouts(agent_id=0, eval_batch=batch, peer_prompts=["base-0", "base-1", "base-2"], candidate_pool=pool))
    factorized_metrics = [asyncio.run(factorized.evaluate_candidate_prompt(0, prompt, ["base-0", "base-1", "base-2"], batch)) for prompt in prompts]

    keys = [
        "target_agent_accuracy", "team_accuracy", "baseline_oracle_acc", "candidate_oracle_acc",
        "coverage_gain_count", "coverage_gain_rate", "coverage_loss_count", "coverage_loss_rate",
        "net_coverage_delta", "invalid_rate", "embedding_diversity", "useful_diversity",
        "reward", "invalid_guard_passed",
    ]
    for left, right in zip(legacy_metrics, factorized_metrics):
        for key in keys:
            if isinstance(left[key], bool):
                assert left[key] is right[key]
            else:
                assert left[key] == pytest.approx(right[key])
    legacy_candidates = [
        {"candidate_id": f"c{index}", "prompt": prompt, "metrics": copy.deepcopy(metrics), "reward": metrics["reward"]}
        for index, (prompt, metrics) in enumerate(zip(prompts, legacy_metrics))
    ]
    factorized_candidates = [
        {"candidate_id": f"c{index}", "prompt": prompt, "metrics": copy.deepcopy(metrics), "reward": metrics["reward"]}
        for index, (prompt, metrics) in enumerate(zip(prompts, factorized_metrics))
    ]
    legacy_selected, _ = legacy._select_vote_pareto_beam(legacy_candidates, beam_size=2, current_prompt="candidate-a")
    factorized_selected, _ = factorized._select_vote_pareto_beam(factorized_candidates, beam_size=2, current_prompt="candidate-a")
    assert [row["candidate_id"] for row in legacy_selected] == [row["candidate_id"] for row in factorized_selected]
    assert legacy_selected[0]["prompt"] == factorized_selected[0]["prompt"]
    assert calls["factorized"] <= calls["legacy"]
