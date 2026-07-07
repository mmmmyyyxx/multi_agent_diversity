import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.tasks import get_task_spec


def _system(cfg=None, prompts=None):
    cfg = cfg or Config()
    prompts = prompts or ["same prompt" for _ in range(int(cfg.agents))]
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.task_spec = get_task_spec("mmlu")
    system.agents = [AgentState(prompt) for prompt in prompts]
    system.update_logs = []
    system.recent_window_records = []
    return system


def test_phase_adaptive_early_phase_gives_higher_diversity_weight():
    cfg = Config(agents=5, reward_schedule_mode="phase_adaptive")
    system = _system(cfg, prompts=["shared"] * 5)

    weights = system._effective_reward_weights()

    assert weights["div_delta"] > cfg.reward_weight_div_delta_late
    assert abs(weights["div_delta"] - cfg.reward_weight_div_delta_early) < abs(weights["div_delta"] - cfg.reward_weight_div_delta_late)
    assert weights["diversity_need"] > 0.5


def test_phase_adaptive_late_phase_lowers_diversity_weight():
    cfg = Config(agents=5, reward_schedule_mode="phase_adaptive", reward_diversity_warmup_updates=10)
    system = _system(cfg, prompts=[f"prompt {i}" for i in range(5)])
    for agent in system.agents:
        agent.accept_count = 2

    late = system._effective_reward_weights()
    early = _system(cfg, prompts=["shared"] * 5)._effective_reward_weights()

    assert late["div_delta"] <= cfg.reward_weight_div_delta_late + 1e-9
    assert late["diversity_need"] <= 1e-9
    assert late["target_accuracy"] >= early["target_accuracy"]


def test_guarded_reward_uses_effective_weights():
    system = _system(Config(reward_mode="guarded_diversity", reward_schedule_mode="phase_adaptive"))
    result = system._candidate_reward_guarded(
        baseline_team_accuracy=0.5,
        candidate_team_accuracy=0.5,
        baseline_target_accuracy=0.4,
        candidate_target_accuracy=0.5,
        baseline_embedding_diversity=0.1,
        candidate_embedding_diversity=0.3,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
    )

    assert "effective_weight_target_accuracy" in result
    assert "effective_weight_div_delta" in result
    assert "reward_phase_progress" in result
    assert "reward_diversity_need" in result


def test_coverage_useful_reward_uses_effective_weights():
    cfg = Config(
        reward_mode="coverage_useful_diversity",
        reward_schedule_mode="static",
        reward_weight_coverage=0.3,
        reward_weight_useful_diversity=0.2,
    )
    system = _system(cfg)
    result = system._candidate_reward_coverage_useful_diversity(
        baseline_team_accuracy=0.0,
        candidate_team_accuracy=0.0,
        baseline_target_accuracy=0.0,
        candidate_target_accuracy=0.5,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
        rescue_rate=0.0,
        coverage_delta=0.25,
        useful_diversity=0.5,
    )

    assert result["effective_weight_coverage"] == 0.3
    assert result["effective_weight_useful_diversity"] == 0.2
    assert round(result["reward"], 6) == round(0.5 + 0.3 * 0.25 + 0.2 * 0.5, 6)


async def _fake_empty_chat(**kwargs):
    return json.dumps({"candidates": []})


def test_fallback_disabled_returns_fewer_candidates():
    system = _system(Config(optimizer_fallback_mode="none"))
    system._chat = _fake_empty_chat
    candidates = asyncio.run(
        system.propose_candidates(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0], "per_agent_invalid_rate": [0.0]},
            num_candidates=2,
            generation_batches=[{"batch_type": "target_error_repair", "cases": [{"case_id": "c1"}]}],
        )
    )
    assert candidates == []


def test_fallback_template_mode_preserves_old_behavior():
    system = _system(Config(optimizer_fallback_mode="template"))
    system._chat = _fake_empty_chat
    candidates = asyncio.run(
        system.propose_candidates(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0], "per_agent_invalid_rate": [0.0]},
            num_candidates=2,
            generation_batches=[{"batch_type": "target_error_repair", "cases": [{"case_id": "c1"}]}],
        )
    )
    assert len(candidates) == 2
    assert all("fallback" in c["candidate_source"] for c in candidates)


def test_update_still_safe_when_optimizer_returns_zero_candidates():
    cfg = Config(optimizer_fallback_mode="none", beam_size=1, num_candidates_per_parent=2, agents=2)
    system = _system(cfg, prompts=["p0", "p1"])
    system.joint_diversity_cache = {}
    system.solver_rollout_cache = {}
    system.agents[0].prompt_beam = [system._make_beam_item("p0", None, {}, None, 0)]

    async def fake_propose_candidates(**kwargs):
        return []

    async def fake_prewarm(**kwargs):
        return {"enabled": False}

    async def fake_eval(**kwargs):
        return {"reward": 0.5, "target_agent_accuracy": 0.5, "num_eval_samples": 1}

    system.propose_candidates = fake_propose_candidates
    system.ensure_recorded_rollouts_for_prompts = fake_prewarm
    system.evaluate_candidate_prompt = fake_eval
    system._append_prompt_history_event = lambda *args, **kwargs: None
    changed, summary = asyncio.run(
        system.update_prompt_with_beam(
            agent_id=0,
            overlap_diagnosis={"homogeneous_cases": []},
            eval_batch=[{"question": "q", "answer": "A"}],
            step_id=1,
            epoch_id=1,
        )
    )

    assert changed is False
    assert summary["candidate_count"] == 1
    assert summary["optimizer_underfilled"] is True
    assert summary["num_optimizer_candidates"] == 0
    assert summary["num_fallback_candidates"] == 0
