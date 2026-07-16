from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState, BehaviorFingerprintEntry, BehaviorStateSummary
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _fingerprint(count=16, answer="answer"):
    return {
        f"q{index}": BehaviorFingerprintEntry(True, answer, True, 1, "target_correct_robust")
        for index in range(count)
    }


def _system(**overrides):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(emergent_specialization_enabled=True, **overrides)
    system.agents = [AgentState("current prompt")]
    return system


def _state(prompt_hash, count=16):
    return BehaviorStateSummary("state-1", 1, prompt_hash, _fingerprint(count), {}, {}, 0.8, 0.8, 0.2, [])


def _candidate(system, prompt="different text", count=16, source="optimizer", **metrics):
    values = {
        "behavior_fingerprint": {key: value.__dict__ for key, value in _fingerprint(count).items()},
        "vote_delta": 0.0,
        "accuracy_delta": 0.0,
        "vote_margin_delta": 0.0,
    }
    values.update(metrics)
    return {"prompt": prompt, "parent_prompt": "current prompt", "source": source, "metrics": values}


def test_exact_historic_prompt_cycle_rejected_but_current_fallback_allowed():
    system = _system()
    agent = system.agents[0]
    agent.history.append("historic prompt")
    rejected = system._candidate_trajectory_feasibility(agent, _candidate(system, "historic prompt"))
    current = system._candidate_trajectory_feasibility(agent, _candidate(system, "current prompt", source="current_active_fallback"))
    assert rejected["rejection_reason"] == "exact_prompt_cycle"
    assert current["rejection_reason"] == ""


def test_textually_different_behavior_cycle_rejected_with_sufficient_overlap():
    system = _system()
    agent = system.agents[0]
    agent.accepted_behavior_archive = [_state("old-hash")]
    result = system._candidate_trajectory_feasibility(agent, _candidate(system))
    assert result["rejection_reason"] == "behavior_cycle"
    assert result["max_behavior_cycle_similarity"] == 1.0


def test_insufficient_overlap_only_records_and_meaningful_improvement_passes():
    system = _system()
    agent = system.agents[0]
    agent.accepted_behavior_archive = [_state("old-hash", 8)]
    low_overlap = system._candidate_trajectory_feasibility(agent, _candidate(system, count=8))
    assert low_overlap["rejection_reason"] == ""
    agent.accepted_behavior_archive = [_state("old-hash", 16)]
    improved = system._candidate_trajectory_feasibility(agent, _candidate(system, vote_delta=0.02))
    assert improved["rejection_reason"] == ""


def test_behavior_archive_fifo_limit():
    system = _system(behavior_archive_size=2)
    archive = []
    for index in range(3):
        system._append_bounded_archive(archive, index)
    assert archive == [1, 2]
