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
    system.execution_session_id = "testsession"
    system.task_spec = get_task_spec("mmlu")
    system.agents = [AgentState(prompt) for prompt in prompts]
    system.update_logs = []
    system.recent_window_records = []
    system.optimizer_generation_diagnostics = {}
    system.no_effective_evolution_counter = 0
    system.no_effective_evolution_stopped = False
    system.no_effective_evolution_reason = ""
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


def test_vote_useful_reward_uses_effective_weights():
    cfg = Config(
        reward_mode="vote_useful_diversity",
        reward_schedule_mode="static",
        reward_weight_vote_delta=0.3,
        reward_weight_vote_margin=0.2,
        reward_weight_boundary_diversity=0.1,
    )
    system = _system(cfg)
    result = system._candidate_reward_vote_useful_diversity(
        baseline_team_accuracy=0.0,
        candidate_team_accuracy=0.25,
        baseline_target_accuracy=0.0,
        candidate_target_accuracy=0.5,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
        baseline_mean_vote_margin=-0.2,
        candidate_mean_vote_margin=0.0,
        baseline_boundary_useful_diversity=0.0,
        candidate_boundary_useful_diversity=0.5,
    )

    assert result["effective_weight_vote_delta"] == 0.3
    assert result["effective_weight_vote_margin"] == 0.2
    assert result["effective_weight_boundary_diversity"] == 0.1
    assert round(result["reward"], 6) == round(0.5 + 0.3 * 0.25 + 0.2 * 0.2 + 0.1 * 0.5, 6)


async def _fake_empty_chat(**kwargs):
    return json.dumps({"candidates": []})


def test_fallback_disabled_returns_fewer_candidates():
    system = _system(Config(optimizer_architecture="one_shot", optimizer_fallback_mode="none"))
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
    system = _system(Config(optimizer_architecture="one_shot", optimizer_fallback_mode="template"))
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
    cfg = Config(optimizer_architecture="one_shot", optimizer_fallback_mode="none", beam_size=1, num_candidates_per_parent=2, agents=2)
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
    assert summary["num_existing_beam_candidates"] == 1


def test_update_limits_optimizer_parent_concurrency():
    cfg = Config(
        optimizer_architecture="one_shot",
        optimizer_fallback_mode="none",
        beam_size=3,
        num_candidates_per_parent=1,
        optimizer_parent_concurrency=2,
        agents=2,
    )
    system = _system(cfg, prompts=["p0", "p1"])
    system.joint_diversity_cache = {}
    system.solver_rollout_cache = {}
    system.agents[0].prompt_beam = [
        system._make_beam_item("p0", None, {}, None, 0),
        system._make_beam_item("p0 parent 1", None, {}, None, 0),
        system._make_beam_item("p0 parent 2", None, {}, None, 0),
    ]
    active = 0
    max_active = 0
    calls = []

    async def fake_propose_candidates(parent_prompt, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        calls.append(parent_prompt)
        await asyncio.sleep(0.01)
        active -= 1
        return [
            {
                "candidate_prompt": f"{parent_prompt} improved",
                "candidate_source": "optimizer",
                "optimizer_generation_diagnostics": {
                    "optimizer_raw_candidate_count": 1,
                    "optimizer_final_candidate_count": 1,
                    "optimizer_underfilled": False,
                },
            }
        ]

    async def fake_prewarm(**kwargs):
        return {"enabled": False}

    async def fake_eval(**kwargs):
        prompt = str(kwargs.get("candidate_prompt", ""))
        reward = 1.0 if "improved" in prompt else 0.1
        return {"reward": reward, "target_agent_accuracy": reward, "num_eval_samples": 1}

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

    assert changed is True
    assert len(calls) == 3
    assert max_active == 2
    assert summary["optimizer_parent_concurrency"] == 2


async def _fake_blank_chat(**kwargs):
    return ""


async def _fake_bad_json_chat(**kwargs):
    return "not json at all"


def test_optimizer_generation_diagnostics_empty_response():
    system = _system(Config(optimizer_architecture="one_shot", optimizer_fallback_mode="none"))
    system._chat = _fake_blank_chat
    candidates = asyncio.run(
        system.propose_candidates(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0], "per_agent_invalid_rate": [0.0]},
            num_candidates=2,
            generation_batches=[{"batch_type": "target_error_repair", "cases": [{"case_id": "c1"}]}],
        )
    )

    diagnostics = system._optimizer_generation_diagnostics_for_parent(0, "parent")
    assert candidates == []
    assert diagnostics["optimizer_raw_response_empty"] == 1
    assert diagnostics["optimizer_json_parse_failed"] == 0
    assert diagnostics["optimizer_final_candidate_count"] == 0
    assert diagnostics["optimizer_underfilled"] is True


def test_optimizer_generation_diagnostics_json_parse_failed():
    system = _system(Config(optimizer_architecture="one_shot", optimizer_fallback_mode="none"))
    system._chat = _fake_bad_json_chat
    candidates = asyncio.run(
        system.propose_candidates(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0], "per_agent_invalid_rate": [0.0]},
            num_candidates=1,
            generation_batches=[{"batch_type": "target_error_repair", "cases": [{"case_id": "c1"}]}],
        )
    )

    diagnostics = system._optimizer_generation_diagnostics_for_parent(0, "parent")
    assert candidates == []
    assert diagnostics["optimizer_raw_response_empty"] == 0
    assert diagnostics["optimizer_json_parse_failed"] == 1
    assert diagnostics["optimizer_underfilled"] is True


def test_update_logs_split_top_beam_from_active_change():
    cfg = Config(optimizer_architecture="one_shot", optimizer_fallback_mode="none", beam_size=1, num_candidates_per_parent=1, agents=2)
    system = _system(cfg, prompts=["p0", "p1"])
    system.joint_diversity_cache = {}
    system.solver_rollout_cache = {}
    system.agents[0].prompt_beam = [system._make_beam_item("p0", None, {}, None, 0)]

    async def fake_propose_candidates(**kwargs):
        return [
            {
                "candidate_prompt": "new useful prompt",
                "candidate_source": "optimizer",
                "optimizer_generation_diagnostics": {
                    "optimizer_raw_candidate_count": 1,
                    "optimizer_final_candidate_count": 1,
                    "optimizer_underfilled": False,
                },
            }
        ]

    async def fake_prewarm(**kwargs):
        return {"enabled": False}

    async def fake_eval(agent_id, candidate_prompt, **kwargs):
        reward = 1.0 if candidate_prompt == "new useful prompt" else 0.1
        return {"reward": reward, "target_agent_accuracy": reward, "num_eval_samples": 1}

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

    assert changed is True
    assert summary["active_prompt_changed"] is True
    assert summary["top1_candidate_source"] == "optimizer"
    assert summary["execution_session_id"] == "testsession"
    assert summary["update_attempt_id"] == "testsession_e1_s1_a0"
    top1 = next(row for row in system.update_logs if row.get("is_top1"))
    assert top1["in_top_beam"] is True
    assert top1["active_prompt_changed"] is True
    assert top1["top1_candidate_source"] == "optimizer"
    beam_summary = next(row for row in system.update_logs if row.get("event") == "beam_update_summary")
    assert beam_summary["execution_session_id"] == "testsession"
    assert beam_summary["update_attempt_id"] == "testsession_e1_s1_a0"


def test_no_effective_evolution_tracking_stops_after_patience():
    cfg = Config(
        no_effective_evolution_patience=2,
        no_effective_evolution_min_optimizer_candidates=1,
        no_effective_evolution_stop_enabled=True,
    )
    system = _system(cfg)
    summary = {
        "update_requested": True,
        "update_ready": True,
        "num_optimizer_candidates": 0,
        "active_prompt_changed_count": 0,
        "updated_agent_ids": [],
    }

    first = system._apply_no_effective_evolution_tracking(dict(summary))
    second = system._apply_no_effective_evolution_tracking(dict(summary))

    assert first["no_effective_evolution_counter"] == 1
    assert first["no_effective_evolution_stopped"] is False
    assert second["no_effective_evolution_counter"] == 2
    assert second["no_effective_evolution_stopped"] is True
    assert "num_optimizer_candidates<1" in second["no_effective_evolution_reason"]
