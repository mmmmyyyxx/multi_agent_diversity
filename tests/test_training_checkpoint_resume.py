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
    write_json_atomic,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system(tmp_path, agents=2):
    cfg = Config(out_dir=str(tmp_path), agents=agents, train_size=4, epochs=2, seed=123)
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.agents = [AgentState(f"prompt {i}") for i in range(agents)]
    system.recent_window_records = [
        {"question_hash": "window-1"},
        {"question_hash": "window-2"},
    ]
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


def test_checkpoint_atomic_write_retries_transient_replace_failure(tmp_path, monkeypatch):
    from multi_dataset_diverse_rl import cli as cli_module

    path = tmp_path / "training_checkpoint.json"
    real_replace = cli_module.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "transient sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr(cli_module.os, "replace", flaky_replace)
    monkeypatch.setattr(cli_module.time, "sleep", lambda _seconds: None)

    write_json_atomic(str(path), {"version": 2, "cursor": 10})

    assert attempts == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2, "cursor": 10}
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_atomic_write_raises_after_persistent_replace_failure(tmp_path, monkeypatch):
    from multi_dataset_diverse_rl import cli as cli_module

    path = tmp_path / "training_checkpoint.json"
    attempts = 0

    def failing_replace(_source, _destination):
        nonlocal attempts
        attempts += 1
        raise PermissionError(5, "persistent sharing violation")

    monkeypatch.setattr(cli_module.os, "replace", failing_replace)
    monkeypatch.setattr(cli_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError):
        write_json_atomic(str(path), {"version": 2})

    assert attempts == 3
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_training_checkpoint_compatible_for_training_stage(tmp_path):
    system = _system(tmp_path)
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())

    assert payload["stage"] == "training"
    assert checkpoint_compatible(payload, system.cfg, [None, None, None, None]) is True

    payload["cursor"] = 5
    assert checkpoint_compatible(payload, system.cfg, [None, None, None, None]) is False


def test_training_checkpoint_requires_window_state_inside_update_window(tmp_path):
    system = _system(tmp_path)
    system.recent_window_records = [{"question_hash": "q1"}, {"question_hash": "q2"}]
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())
    assert checkpoint_compatible(payload, system.cfg, [None, None, None, None]) is True

    payload["state"].pop("recent_window_records")
    reasons = checkpoint_incompatibility_reasons(payload, system.cfg, [None, None, None, None])

    assert any("stopped inside an update window" in reason for reason in reasons)


def test_training_checkpoint_rejects_skipped_update_boundary(tmp_path):
    system = _system(tmp_path)
    system.cfg.update_every = 2
    system.recent_window_records = []
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())
    (tmp_path / "train_step_logs.jsonl").write_text(
        json.dumps({
            "epoch": 1,
            "step": 2,
            "update_summary": {"skipped_reason": "window_not_ready"},
        }) + "\n",
        encoding="utf-8",
    )

    reasons = checkpoint_incompatibility_reasons(payload, system.cfg, [None, None, None, None])

    assert any("cannot be resumed faithfully" in reason for reason in reasons)


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


def test_training_checkpoint_reports_changed_derived_behavior_field(tmp_path):
    system = _system(tmp_path)
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())
    payload["behavior_config"]["effective_aggregation_mode"] = "weighted_vote"
    payload["behavior_config_fingerprint"] = "changed-derived-behavior"

    reasons = checkpoint_incompatibility_reasons(payload, system.cfg, [None, None, None, None])

    assert any("effective_aggregation_mode" in reason for reason in reasons)


def test_training_checkpoint_fingerprint_normalizes_integer_boolean_flags():
    bool_cfg = Config(
        competence_depth_enabled=True,
        competence_depth2_aux_enabled=True,
        competence_progressive_residual_enabled=True,
    )
    int_cfg = Config(
        competence_depth_enabled=1,
        competence_depth2_aux_enabled=1,
        competence_progressive_residual_enabled=1,
    )

    assert checkpoint_behavior_config_fingerprint(bool_cfg) == checkpoint_behavior_config_fingerprint(int_cfg)


def test_training_checkpoint_accepts_legacy_integer_boolean_payload(tmp_path):
    system = _system(tmp_path)
    system.cfg.competence_depth_enabled = True
    system.cfg.competence_depth2_aux_enabled = True
    system.cfg.competence_progressive_residual_enabled = True
    payload = build_training_checkpoint(system.cfg, system, **_checkpoint_kwargs())
    for field in (
        "competence_depth_enabled",
        "competence_depth2_aux_enabled",
        "competence_progressive_residual_enabled",
    ):
        payload["behavior_config"][field] = 1
    payload["behavior_config_fingerprint"] = "legacy-integer-boolean-fingerprint"

    assert checkpoint_compatible(payload, system.cfg, [None, None, None, None]) is True


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
    source.recent_window_records = [
        {"question_hash": "q1", "answers": ["A", "B"], "gold": "A"},
        {"question_hash": "q2", "answers": ["C", "C"], "gold": "B"},
    ]
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
    assert target.recent_window_records == source.recent_window_records


def test_restore_legacy_checkpoint_without_window_uses_empty_window(tmp_path):
    source = _system(tmp_path / "source")
    payload = build_training_checkpoint(source.cfg, source, **_checkpoint_kwargs())
    payload["state"].pop("recent_window_records")
    target = _system(tmp_path / "target")
    target.recent_window_records = [{"question_hash": "stale"}]

    restore_system_state(target, payload["state"])

    assert target.recent_window_records == []


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
