from multi_dataset_diverse_rl.cli import (
    build_training_checkpoint,
    checkpoint_incompatibility_reasons,
    restore_system_state,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState, BehaviorStateSummary
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system(tmp_path):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(out_dir=str(tmp_path), agents=1, train_size=1, epochs=1, emergent_specialization_enabled=True)
    system.agents = [AgentState("prompt")]
    return system


def _checkpoint(system):
    return build_training_checkpoint(
        system.cfg, system, epoch_index=0, cursor=0, order=[0], train_accumulators={},
        best_score=0.0, best_epoch=0, epochs_without_improvement=0, stopped_early=False,
        no_effective_evolution_counter=0, no_effective_evolution_stopped=False,
        no_effective_evolution_reason="",
    )


def test_trajectory_state_round_trip_and_next_ema_matches_continuous(tmp_path):
    source = _system(tmp_path / "source")
    context = next(iter(source.agents[0].specialization_profile))
    source.update_specialization_profile(source.agents[0], {context: 1.0}, {context: 2})
    source.agents[0].accepted_behavior_archive = [BehaviorStateSummary("s", 1, "hash", {}, {context: 1.0}, dict(source.agents[0].specialization_profile), 1.0, 1.0, 0.2, [])]
    payload = _checkpoint(source)
    resumed = _system(tmp_path / "resumed")
    restore_system_state(resumed, payload["state"])
    source.update_specialization_profile(source.agents[0], {context: 0.5}, {context: 1})
    resumed.update_specialization_profile(resumed.agents[0], {context: 0.5}, {context: 1})
    assert resumed.agents[0].specialization_profile == source.agents[0].specialization_profile
    assert resumed.agents[0].accepted_behavior_archive[0].state_id == "s"
    candidate = {"prompt": "different text", "parent_prompt": "prompt", "source": "optimizer", "metrics": {
        "behavior_fingerprint": {}, "vote_delta": 0.0, "accuracy_delta": 0.0, "vote_margin_delta": 0.0,
    }}
    resumed.agents[0].history.append("historic prompt")
    candidate["prompt"] = "historic prompt"
    assert resumed._candidate_trajectory_feasibility(resumed.agents[0], candidate)["rejection_reason"] == "exact_prompt_cycle"


def test_emergent_config_change_and_old_checkpoint_version_are_incompatible(tmp_path):
    system = _system(tmp_path)
    payload = _checkpoint(system)
    changed = Config(out_dir=str(tmp_path), agents=1, train_size=1, epochs=1, emergent_specialization_enabled=True, behavior_archive_size=8)
    reasons = checkpoint_incompatibility_reasons(payload, changed, [None])
    assert any("behavior_archive_size" in reason for reason in reasons)
    payload["version"] = 2
    assert any("version" in reason for reason in checkpoint_incompatibility_reasons(payload, system.cfg, [None]))
