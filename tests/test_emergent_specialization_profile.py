import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import BEHAVIOR_CONTEXT_NAMES, AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system(**overrides):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(emergent_specialization_enabled=True, **overrides)
    system.agents = [AgentState("shared") for _ in range(2)]
    return system


def test_uniform_initialization_has_no_agent_id_role():
    system = _system()
    assert system.agents[0].specialization_profile == system.agents[1].specialization_profile
    assert sum(system.agents[0].specialization_profile.values()) == pytest.approx(1.0)


def test_accepted_positive_transitions_diverge_profiles_with_correct_ema():
    system = _system(specialization_ema=0.2, specialization_smoothing=0.0)
    first, second = BEHAVIOR_CONTEXT_NAMES[:2]
    old = system.agents[0].specialization_profile[first]
    assert system.update_specialization_profile(system.agents[0], {first: 1.0}, {first: 4})
    assert system.update_specialization_profile(system.agents[1], {second: 1.0}, {second: 4})
    assert system.agents[0].specialization_profile[first] == pytest.approx(0.8 * old + 0.2)
    assert system.agents[0].specialization_profile != system.agents[1].specialization_profile


def test_zero_or_unsupported_transition_does_not_create_profile_direction():
    system = _system()
    agent = system.agents[0]
    before = dict(agent.specialization_profile)
    assert not system.update_specialization_profile(agent, {BEHAVIOR_CONTEXT_NAMES[0]: 1.0}, {BEHAVIOR_CONTEXT_NAMES[0]: 1})
    assert agent.specialization_profile == before
    assert agent.specialization_update_count == 0


def test_exploration_floor_preserves_non_primary_context_mass():
    system = _system(specialization_exploration_floor=0.2)
    agent = system.agents[0]
    agent.specialization_profile = {key: float(index == 0) for index, key in enumerate(BEHAVIOR_CONTEXT_NAMES)}
    effective = system.effective_specialization_profile(agent)
    assert all(value > 0.0 for value in effective.values())
    assert sum(effective.values()) == pytest.approx(1.0)


def test_affinity_bonus_cannot_override_clear_base_pressure(monkeypatch):
    system = _system(specialization_affinity_weight=0.5)
    system.agents[0].specialization_update_count = 1
    system.agents[1].specialization_update_count = 1
    diagnosis = {
        "per_agent_error_count": [0, 2],
        "per_agent_team_wrong_error_count": [0, 1],
        "per_agent_invalid_rate": [0, 0],
        "per_agent_pivotal_fix_count": [0, 0],
        "per_agent_dominant_wrong_redundancy_count": [0, 0],
        "per_agent_context_pressure": [
            {BEHAVIOR_CONTEXT_NAMES[0]: 10},
            {BEHAVIOR_CONTEXT_NAMES[1]: 1},
        ],
    }
    monkeypatch.setattr("multi_dataset_diverse_rl.system.random.shuffle", lambda values: None)
    assert system.select_reward_agents_for_update(diagnosis, {})[0] == 1
