from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system_without_init(cfg: Config):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    return system


def test_guarded_reward_penalizes_accuracy_drop():
    cfg = Config(reward_mode="guarded_diversity", accuracy_guard_epsilon=0.02)
    system = _system_without_init(cfg)
    result = system._candidate_reward_guarded(
        baseline_team_accuracy=0.8,
        candidate_team_accuracy=0.8,
        baseline_target_accuracy=0.8,
        candidate_target_accuracy=0.7,
        baseline_embedding_diversity=0.2,
        candidate_embedding_diversity=0.9,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.1,
    )
    assert result["accuracy_guard_passed"] is False
    assert result["reward"] < -1.0
    assert result["accuracy_delta"] == -0.10000000000000009


def test_guarded_reward_uses_target_agent_accuracy_not_team_accuracy():
    cfg = Config(
        reward_mode="guarded_diversity",
        reward_schedule_mode="static",
        accuracy_guard_epsilon=0.02,
        reward_weight_div_delta=0.3,
        reward_weight_invalid_delta=0.5,
    )
    system = _system_without_init(cfg)
    result = system._candidate_reward_guarded(
        baseline_team_accuracy=0.8,
        candidate_team_accuracy=0.2,
        baseline_target_accuracy=0.8,
        candidate_target_accuracy=0.79,
        baseline_embedding_diversity=0.2,
        candidate_embedding_diversity=0.4,
        baseline_invalid_rate=0.1,
        candidate_invalid_rate=0.1,
    )
    assert result["accuracy_guard_passed"] is True
    assert round(result["reward"], 6) == round(0.79 + 0.3 * 0.2, 6)
    assert round(result["accuracy_delta"], 6) == round(-0.01, 6)
    assert round(result["vote_delta"], 6) == round(-0.6, 6)
    assert round(result["diversity_delta"], 6) == 0.2
