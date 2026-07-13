import json

from multi_dataset_diverse_rl.cli import restore_agent_prompts, restore_prompt_history
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system(tmp_path):
    cfg = Config(out_dir=str(tmp_path), agents=2, beam_size=2)
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.agents = [AgentState("initial prompt") for _ in range(2)]
    for agent in system.agents:
        agent.prompt_beam = [system._make_beam_item(agent.current_prompt, 0.1, {}, None, 0)]
    system.prompt_history = system._init_prompt_history()
    return system


def test_restore_agent_prompts_syncs_prompt_history(tmp_path):
    system = _system(tmp_path)

    restore_agent_prompts(system, ["best prompt A", "best prompt B"], selected_epoch=2)
    system.flush_prompt_history()

    prompt_history = json.loads((tmp_path / "prompt_history.json").read_text(encoding="utf-8"))
    assert prompt_history["0"]["current_prompt"] == "best prompt A"
    assert prompt_history["1"]["current_prompt"] == "best prompt B"
    assert prompt_history["0"]["events"][-1]["decision"] == "restore_best_prompts"
    assert prompt_history["0"]["events"][-1]["selected_epoch"] == 2


def test_prompt_history_snapshot_retries_transient_replace_failure(tmp_path, monkeypatch):
    from multi_dataset_diverse_rl import system as system_module

    system = _system(tmp_path)
    real_replace = system_module.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(22, "transient invalid argument")
        return real_replace(source, destination)

    monkeypatch.setattr(system_module.os, "replace", flaky_replace)
    system.flush_prompt_history()

    assert attempts == 2
    assert json.loads((tmp_path / "prompt_history.json").read_text(encoding="utf-8"))["0"]["initial_prompt"] == "initial prompt"


def test_restore_prompt_history_keeps_existing_events(tmp_path):
    system = _system(tmp_path)
    (tmp_path / "prompt_history.json").write_text(
        json.dumps(
            {
                "0": {
                    "initial_prompt": "initial prompt",
                    "initial_prompt_hash": system._hash("initial prompt"),
                    "current_prompt": "old prompt",
                    "current_prompt_hash": system._hash("old prompt"),
                    "prompt_beam": [],
                    "events": [{"decision": "beam_accept"}],
                },
                "1": {
                    "initial_prompt": "initial prompt",
                    "initial_prompt_hash": system._hash("initial prompt"),
                    "current_prompt": "old prompt",
                    "current_prompt_hash": system._hash("old prompt"),
                    "prompt_beam": [],
                    "events": [],
                },
            }
        ),
        encoding="utf-8",
    )
    system.agents[0].current_prompt = "resumed prompt"

    restore_prompt_history(system)

    assert system.prompt_history["0"]["events"][0]["decision"] == "beam_accept"
    assert system.prompt_history["0"]["events"][-1]["decision"] == "checkpoint_resume"
    assert system.prompt_history["0"]["current_prompt"] == "resumed prompt"


def test_system_init_does_not_overwrite_prompt_history_when_resuming(tmp_path, monkeypatch):
    from multi_dataset_diverse_rl import system as system_module

    existing = {"0": {"events": [{"decision": "old"}]}}
    (tmp_path / "prompt_history.json").write_text(json.dumps(existing), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(system_module.TraceBeamSearchSystem, "_load_recorded_solver_rollouts", lambda self: None)

    TraceBeamSearchSystem(Config(out_dir=str(tmp_path), agents=1, resume_from_checkpoint=True))

    prompt_history = json.loads((tmp_path / "prompt_history.json").read_text(encoding="utf-8"))
    assert prompt_history == existing
