import json

import pytest

from multi_dataset_diverse_rl.cli import (
    abort_incompatible_checkpoint,
    build_training_checkpoint,
    checkpoint_compatible,
    checkpoint_incompatibility_reasons,
    checkpoint_behavior_config_fingerprint,
    restore_cost_summary,
    restore_system_state,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system(tmp_path, agents=2):
    cfg = Config(out_dir=str(tmp_path), agents=agents, train_size=4, epochs=2, seed=123)
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.agents = [AgentState(f"prompt {i}") for i in range(agents)]
    for i, agent in enumerate(system.agents):
        agent.current_prompt = f"current {i}"
        agent.prompt_beam = [system._make_beam_item(agent.current_prompt, 0.5, {"rank": i}, None, 0)]
        agent.history = [agent.initial_prompt, agent.current_prompt]
        agent.accept_count = i + 1
        agent.reject_count = i
    return system


def _checkpoint_kwargs():
    return {
        "epoch_index": 0,
        "cursor": 2,
        "order": [3, 2, 1, 0],
        "train_accumulators": {"train_vote_correct": [1, 0]},
        "best_score": 0.25,
        "best_epoch": 0,
        "epochs_without_improvement": 1,
        "stopped_early": False,
        "no_effective_evolution_counter": 0,
        "no_effective_evolution_stopped": False,
        "no_effective_evolution_reason": "",
    }


def test_training_checkpoint_compatible_for_training_stage(tmp_path):
    system = _system(tmp_path)
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())

    assert payload["stage"] == "training"
    assert checkpoint_compatible(payload, system.cfg, [None, None, None, None]) is True

    payload["cursor"] = 5
    assert checkpoint_compatible(payload, system.cfg, [None, None, None, None]) is False


def test_training_checkpoint_rejects_changed_config_signature(tmp_path):
    system = _system(tmp_path)
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())
    changed_cfg = Config(out_dir=str(tmp_path), agents=2, train_size=4, epochs=2, seed=123, reward_mode="accuracy_only")

    assert checkpoint_compatible(payload, changed_cfg, [None, None, None, None]) is False


def test_training_checkpoint_reports_changed_config_signature(tmp_path):
    system = _system(tmp_path)
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())
    changed_cfg = Config(out_dir=str(tmp_path), agents=2, train_size=4, epochs=2, seed=123, reward_mode="accuracy_only")

    reasons = checkpoint_incompatibility_reasons(payload, changed_cfg, [None, None, None, None])

    assert any("reward_mode" in reason for reason in reasons)


def test_training_checkpoint_rejects_changed_reward_and_eval_behavior(tmp_path):
    system = _system(tmp_path)
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())
    changed_cfg = Config(
        out_dir=str(tmp_path),
        agents=2,
        train_size=4,
        epochs=2,
        seed=123,
        candidate_eval_repeats=2,
        reward_weight_vote_margin=0.9,
        vote_tie_break="abstain",
    )

    reasons = checkpoint_incompatibility_reasons(payload, changed_cfg, [None, None, None, None])

    assert payload["behavior_config_fingerprint"] == checkpoint_behavior_config_fingerprint(system.cfg)
    assert any("behavior_config_fingerprint" in reason for reason in reasons)
    assert any("candidate_eval_repeats" in reason for reason in reasons)
    assert any("reward_weight_vote_margin" in reason for reason in reasons)
    assert any("vote_tie_break" in reason for reason in reasons)


def test_abort_incompatible_checkpoint_exits_nonzero(tmp_path, capsys):
    cfg = Config(out_dir=str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        abort_incompatible_checkpoint(cfg, ["reward_mode: checkpoint='guarded_diversity' current='accuracy_only'"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Incompatible training_checkpoint.json" in captured.out
    assert "reward_mode" in captured.out


def test_training_checkpoint_compatible_for_epoch_boundary(tmp_path):
    system = _system(tmp_path)
    kwargs = _checkpoint_kwargs()
    kwargs.update({"epoch_index": 1, "cursor": 0, "order": [], "train_accumulators": {}, "stage": "between_epochs"})
    payload = build_training_checkpoint(system.cfg, system, **kwargs)

    assert checkpoint_compatible(payload, system.cfg, [None, None, None, None]) is True


def test_epoch_evaluated_checkpoint_preserves_epoch_record(tmp_path):
    system = _system(tmp_path)
    epoch_record = {"epoch": 1, "train": {"vote_acc": 0.5}, "val": {"vote_acc": 0.75}}
    kwargs = _checkpoint_kwargs()
    kwargs.update({"stage": "epoch_evaluated", "epoch_record": epoch_record})
    payload = build_training_checkpoint(system.cfg, system, **kwargs)

    assert checkpoint_compatible(payload, system.cfg, [None, None, None, None]) is True
    assert payload["epoch_record"] == epoch_record


def test_restore_system_state_restores_prompts_beams_and_counts(tmp_path):
    source = _system(tmp_path / "source")
    payload = build_training_checkpoint(source.cfg, source, **_checkpoint_kwargs())
    target = _system(tmp_path / "target")
    for agent in target.agents:
        agent.current_prompt = "stale"
        agent.prompt_beam = []
        agent.accept_count = 0
        agent.reject_count = 0

    restore_system_state(target, payload["state"])

    assert [agent.current_prompt for agent in target.agents] == ["current 0", "current 1"]
    assert [agent.prompt_beam[0]["prompt"] for agent in target.agents] == ["current 0", "current 1"]
    assert [agent.accept_count for agent in target.agents] == [1, 2]
    assert [agent.reject_count for agent in target.agents] == [0, 1]


def test_restore_cost_summary_keeps_previous_counts(tmp_path):
    system = _system(tmp_path)
    (tmp_path / "cost_summary.json").write_text(
        json.dumps({"solver_calls": 7, "total_llm_calls": 9, "total_tokens": 1234}),
        encoding="utf-8",
    )

    restore_cost_summary(system)

    assert system.cost_summary["solver_calls"] == 7
    assert system.cost_summary["total_llm_calls"] == 9
    assert system.cost_summary["total_tokens"] == 1234
    assert "optimizer_calls" in system.cost_summary
