from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState, BehaviorFingerprintEntry, BehaviorStateSummary
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _fingerprint(count=16, answer="answer", target_correct=True, team_correct=True, margin=1):
    return {
        f"q{index}": BehaviorFingerprintEntry(
            target_correct,
            answer,
            team_correct,
            margin,
            "target_correct_robust" if target_correct else "team_wrong_nonpivotal",
        )
        for index in range(count)
    }


def _system(**overrides):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(residual_cycle_guard_enabled=True, **overrides)
    system.agents = [AgentState("current prompt")]
    return system


def _state(prompt_hash, count=16, **fingerprint_kwargs):
    return BehaviorStateSummary(
        state_id="state-1",
        epoch=1,
        prompt_hash=prompt_hash,
        behavior_fingerprint=_fingerprint(count, **fingerprint_kwargs),
        transition_vector={},
        target_accuracy=0.8,
        team_vote_accuracy=0.8,
        mean_vote_margin=0.2,
        preserved_mechanisms=[],
    )


def _candidate(system, prompt="different text", count=16, source="optimizer", fingerprint_kwargs=None, **metrics):
    values = {
        "behavior_fingerprint": {
            key: value.__dict__
            for key, value in _fingerprint(count, **(fingerprint_kwargs or {})).items()
        },
        "vote_delta": 0.0,
        "accuracy_delta": 0.0,
        "vote_margin_delta": 0.0,
    }
    values.update(metrics)
    return {
        "prompt": prompt,
        "parent_prompt": "current prompt",
        "candidate_pool_source": source,
        "candidate_source": "teacher_critic_student" if source == "optimizer" else source,
        "metrics": values,
    }


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
    assert result["rejection_reason"] == "accepted_state_cycle"
    assert result["max_behavior_cycle_similarity"] == 1.0


def test_insufficient_overlap_only_records_and_meaningful_improvement_passes():
    system = _system()
    agent = system.agents[0]
    agent.accepted_behavior_archive = [_state("old-hash", 8)]
    low_overlap = system._candidate_trajectory_feasibility(agent, _candidate(system, count=8))
    assert low_overlap["rejection_reason"] == ""
    agent.accepted_behavior_archive = [_state("old-hash", 16, target_correct=False, team_correct=False, margin=-2)]
    improved = system._candidate_trajectory_feasibility(agent, _candidate(system))
    assert improved["rejection_reason"] == ""


def test_behavior_archive_fifo_limit():
    system = _system(behavior_archive_size=2)
    archive = []
    for index in range(3):
        system._append_bounded_archive(archive, index)
    assert archive == [1, 2]


def test_legacy_source_alias_remains_supported():
    system = _system()
    agent = system.agents[0]
    agent.history.append("historic prompt")
    item = _candidate(system, "historic prompt")
    item["source"] = item.pop("candidate_pool_source")

    result = system._candidate_trajectory_feasibility(agent, item)

    assert result["rejection_reason"] == "exact_prompt_cycle"
