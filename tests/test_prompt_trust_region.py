from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system(enabled=True):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(
        residual_cycle_guard_enabled=enabled,
        prompt_max_change_ratio=0.1,
        prompt_large_shift_warmup_accepts=2,
        prompt_large_shift_min_vote_delta=0.02,
        baseline_allowed_vote_loss=0.0,
    )
    system.agents = [AgentState("short parent")]
    return system


def _item(**metrics):
    values = {
        "behavior_fingerprint": {}, "vote_delta": 0.0, "accuracy_delta": 0.0,
        "vote_margin_delta": 0.0, "vote_loss_rate": 0.0,
    }
    values.update(metrics)
    return {
        "prompt": "a completely different and substantially longer candidate mechanism",
        "parent_prompt": "short parent",
        "candidate_pool_source": "optimizer",
        "candidate_source": "teacher_critic_student",
        "metrics": values,
    }


def test_warmup_allows_large_shift_and_post_warmup_rejects_unsupported_shift():
    system = _system()
    agent = system.agents[0]
    assert system._candidate_trajectory_feasibility(agent, _item())["rejection_reason"] == ""
    agent.accept_count = 2
    assert system._candidate_trajectory_feasibility(agent, _item())["rejection_reason"] == "unsupported_large_prompt_shift"


def test_supported_large_shift_and_small_edit_pass():
    system = _system()
    agent = system.agents[0]
    agent.accept_count = 2
    assert system._candidate_trajectory_feasibility(agent, _item(vote_delta=0.02))["rejection_reason"] == ""
    local = {
        "prompt": "short parent verify",
        "parent_prompt": "short parent",
        "candidate_pool_source": "optimizer",
        "candidate_source": "teacher_critic_student",
        "metrics": _item()["metrics"],
    }
    system.cfg.prompt_max_change_ratio = 0.5
    assert system._candidate_trajectory_feasibility(agent, local)["rejection_reason"] == ""


def test_v7_trajectory_guards_off_skip_prompt_trust_region():
    system = _system(enabled=False)
    system.agents[0].accept_count = 10
    result = system._candidate_trajectory_feasibility(system.agents[0], _item())
    assert result["rejection_reason"] == ""
