import json

from multi_dataset_diverse_rl.cli import restore_agent_prompts
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
