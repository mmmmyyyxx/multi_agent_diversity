import asyncio

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.tasks import get_task_spec


def _system_without_init(cfg: Config):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.task_spec = get_task_spec("mmlu")
    return system


def test_vote_useful_reward_uses_vote_delta_margin_and_boundary_diversity():
    system = _system_without_init(
        Config(
            reward_mode="vote_useful_diversity",
            reward_schedule_mode="static",
            reward_weight_vote_delta=0.3,
            reward_weight_vote_margin=0.2,
            reward_weight_boundary_diversity=0.1,
        )
    )
    result = system._candidate_reward_vote_useful_diversity(
        baseline_team_accuracy=0.25,
        candidate_team_accuracy=0.50,
        baseline_target_accuracy=0.4,
        candidate_target_accuracy=0.5,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
        baseline_mean_vote_margin=-0.2,
        candidate_mean_vote_margin=0.0,
        baseline_boundary_useful_diversity=0.1,
        candidate_boundary_useful_diversity=0.5,
    )
    assert result["accuracy_guard_passed"] is True
    assert result["invalid_guard_passed"] is True
    assert result["vote_delta"] == 0.25
    assert result["vote_margin_delta"] == 0.2
    assert result["boundary_useful_diversity_delta"] == 0.4
    assert result["boundary_diversity_gain"] == 0.4
    assert result["reward_total"] == result["reward"]
    assert result["reward_component_boundary_diversity"] == 0.1 * 0.4
    assert round(sum(value for key, value in result.items() if key.startswith("reward_component_")), 6) == round(result["reward"], 6)
    assert round(result["reward"], 6) == round(0.5 + 0.3 * 0.25 + 0.2 * 0.2 + 0.1 * 0.4, 6)


def test_vote_useful_reward_rejects_target_accuracy_or_invalid_regression():
    system = _system_without_init(Config(reward_mode="vote_useful_diversity", accuracy_guard_epsilon=0.02, invalid_guard_epsilon=0.05))
    target_drop = system._candidate_reward_vote_useful_diversity(
        baseline_team_accuracy=0.3, candidate_team_accuracy=0.8,
        baseline_target_accuracy=0.8, candidate_target_accuracy=0.7,
        baseline_invalid_rate=0.0, candidate_invalid_rate=0.0,
        baseline_mean_vote_margin=-0.3, candidate_mean_vote_margin=0.4,
        baseline_boundary_useful_diversity=0.0, candidate_boundary_useful_diversity=1.0,
    )
    invalid = system._candidate_reward_vote_useful_diversity(
        baseline_team_accuracy=0.3, candidate_team_accuracy=0.8,
        baseline_target_accuracy=0.8, candidate_target_accuracy=0.8,
        baseline_invalid_rate=0.0, candidate_invalid_rate=0.2,
        baseline_mean_vote_margin=-0.3, candidate_mean_vote_margin=0.4,
        baseline_boundary_useful_diversity=0.0, candidate_boundary_useful_diversity=1.0,
    )
    assert target_drop["accuracy_guard_passed"] is False
    assert target_drop["reward"] == -1.0
    assert invalid["invalid_guard_passed"] is False
    assert invalid["reward"] == -1.0


def test_vote_useful_reward_does_not_penalize_leaving_the_vote_boundary():
    system = _system_without_init(
        Config(
            reward_mode="vote_useful_diversity",
            reward_schedule_mode="static",
            reward_weight_vote_delta=0.3,
            reward_weight_vote_margin=0.2,
            reward_weight_boundary_diversity=0.4,
        )
    )
    result = system._candidate_reward_vote_useful_diversity(
        baseline_team_accuracy=0.0,
        candidate_team_accuracy=1.0,
        baseline_target_accuracy=0.4,
        candidate_target_accuracy=0.6,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
        baseline_mean_vote_margin=0.0,
        candidate_mean_vote_margin=0.4,
        baseline_boundary_useful_diversity=1.0 / 3.0,
        candidate_boundary_useful_diversity=0.0,
    )
    assert result["boundary_useful_diversity_delta"] == -1.0 / 3.0
    assert result["boundary_diversity_gain"] == 0.0
    assert round(result["reward"], 6) == round(0.6 + 0.3 + 0.2 * 0.4, 6)


def test_vote_useful_diversity_mode_detection_and_reward_agent_selection():
    system = _system_without_init(Config(reward_mode="vote_useful_diversity", agents=3))
    system.agents = [object(), object(), object()]
    assert system._is_vote_useful_diversity_mode()
    diagnosis = {
        "per_agent_error_count": [0, 1, 0],
        "per_agent_team_wrong_error_count": [0, 2, 0],
        "per_agent_invalid_rate": [0.0, 0.0, 0.0],
        "per_agent_pivotal_fix_count": [0, 2, 0],
        "per_agent_dominant_wrong_redundancy_count": [0, 1, 0],
    }
    assert system.select_reward_agents_for_update(diagnosis, metrics={})[0] == 1


def test_accuracy_only_reward_uses_target_agent_accuracy():
    system = _system_without_init(Config(reward_mode="accuracy_only", agents=5))
    system.agents = [object() for _ in range(5)]
    system._active_prompt_list = lambda: [f"p{i}" for i in range(5)]

    async def fake_solve(question, prompts, source=""):
        target_answer = "A" if prompts[0] == "candidate" else "B"
        return ["t0", "t1", "t2", "t3", "t4"], [target_answer, "B", "B", "B", "B"], {"solver_reuse_hits": 0, "solver_reuse_misses": 0, "solver_calls": 0, "solver_reuse_total": 0}

    system.solve_with_prompts_reusing_records = fake_solve
    system.compute_rollout_metrics = lambda *args, **kwargs: {
        "vote_correct": 0, "vote_answer": "B", "vote_tie": False, "tie_candidates": [],
        "vote_counts": {"B": 4, "A": 1}, "tie_break_method": "first",
        "majority_vote_answer": "B", "weighted_vote_answer": "B", "majority_vote_correct": 0,
        "weighted_vote_correct": 0, "aggregation_mode": "majority",
    }
    system._hash = lambda value: "hash"
    result = asyncio.run(system._evaluate_candidate_prompt_accuracy_only(
        agent_id=0, candidate_prompt="candidate", peer_prompts=[f"p{i}" for i in range(5)],
        eval_batch=[{"question": f"q{i}", "answer": "A"} for i in range(5)],
    ))
    assert result["team_accuracy"] == 0.0
    assert result["target_agent_accuracy"] == 1.0
    assert result["reward"] == 1.0
    assert result["accuracy_only_reward_basis"] == "target_agent_accuracy"
    assert result["baseline_target_accuracy"] == 0.0
    assert result["candidate_target_accuracy"] == 1.0
    assert result["accuracy_delta"] == 1.0
    assert result["vote_delta"] == result["candidate_team_accuracy"] - result["baseline_team_accuracy"]
    assert result["vote_delta"] == result["vote_gain_rate"] - result["vote_loss_rate"]
    assert result["reward_total"] == result["reward_component_target_accuracy"]


def test_dataset_summary_includes_vote_margin_and_boundary_diversity():
    system = _system_without_init(Config())
    summary = system._summarize_rollout_rows([
        {"vote_correct": 0, "individual_correct": [0, 1, 0], "vote_counts": {"A": 2, "B": 1}, "normalized_vote_margin": -1 / 3, "boundary_useful_diversity": 0.0},
        {"vote_correct": 1, "individual_correct": [1, 0, 0], "vote_counts": {"C": 3}, "normalized_vote_margin": 1 / 3, "boundary_useful_diversity": 0.5},
    ])
    assert summary["vote_acc"] == 0.5
    assert summary["mean_vote_margin"] == 0.0
    assert summary["mean_boundary_useful_diversity"] == 0.25
