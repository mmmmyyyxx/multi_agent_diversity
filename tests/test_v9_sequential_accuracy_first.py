from copy import deepcopy

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.sequential_state import (
    accuracy_first_key,
    candidate_strictly_beats_incumbent,
    epoch_agent_order,
    full_probe_constraints,
    question_state,
    rebuild_prompt_memory,
    safe_trace_diversity_c4c5,
    state_histogram,
    state_vote_reward,
)


def _item(prompt_hash, correct, reward=0.0, invalid=0, slack=0.0, generation=1):
    return {
        "prompt_hash": prompt_hash,
        "prompt": prompt_hash,
        "generation": generation,
        "outcome_signature": f"outcome-{prompt_hash}",
        "metrics": {
            "candidate_target_correct_count": correct,
            "candidate_target_accuracy": correct / 10,
            "state_vote_reward": reward,
            "candidate_invalid_count": invalid,
            "diversity_constraint_slack": slack,
        },
    }


def test_c0_through_c5_and_compatibility_summary():
    assert [question_state(value) for value in range(6)] == [
        "C0", "C1", "C2", "C3", "C4", "C5"
    ]
    histogram = state_histogram(range(6))
    assert all(histogram[f"c{value}_count"] == 1 for value in range(6))
    assert histogram["c3plus_count"] == 3


def test_state_potential_rewards_expected_transitions():
    cfg = Config(state_vote_reward_enabled=False, state_reward_bottom2_weight=0.0)
    rewards = []
    for before, after in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        active = [[1] * before + [0] * (5 - before)]
        candidate = [[1] * after + [0] * (5 - after)]
        rewards.append(state_vote_reward(active, candidate, [0], [0], cfg)["state_reward_total"])
    assert rewards == pytest.approx([1.0, 0.75, 1.5, 0.35, 0.15])


def test_wrong_answer_labels_and_option_count_have_zero_reward_value():
    cfg = Config()
    active_correctness = [[1, 1, 0, 0, 0]]
    candidate_correctness = deepcopy(active_correctness)
    result = state_vote_reward(
        active_correctness,
        candidate_correctness,
        [0],
        [1],  # A raw tie outcome may change when B becomes C.
        cfg,
    )
    assert result["state_reward_total"] == 0.0
    assert result["vote_accuracy_delta"] == 0.0
    assert result["diagnostic_raw_vote_accuracy_delta"] == 1.0


def test_accuracy_is_primary_over_reward_invalid_and_diversity():
    cfg = Config()
    accurate = _item("accurate", 9, reward=-10.0, invalid=2, slack=-1.0)
    tempting = _item("tempting", 8, reward=100.0, invalid=0, slack=100.0)
    assert accuracy_first_key(accurate) > accuracy_first_key(tempting)
    assert candidate_strictly_beats_incumbent(accurate, tempting, cfg)
    assert not candidate_strictly_beats_incumbent(tempting, accurate, cfg)


def test_equal_accuracy_requires_secondary_gain():
    cfg = Config(state_min_secondary_reward_gain=0.1)
    incumbent = _item("incumbent", 8, reward=1.0)
    assert not candidate_strictly_beats_incumbent(_item("small", 8, reward=1.05), incumbent, cfg)
    assert candidate_strictly_beats_incumbent(_item("large", 8, reward=1.2), incumbent, cfg)


def test_full_probe_accuracy_constraints_use_discrete_counts():
    cfg = Config()
    active = {"candidate_target_correct_count": 8, "candidate_invalid_count": 0}
    initial = {"candidate_target_correct_count": 8}
    candidate = {
        "full_probe_size": 10,
        "candidate_target_correct_count": 7,
        "candidate_invalid_count": 0,
        "correct_set_diversity_mean": 1.0,
        "correct_set_diversity_min": 1.0,
        "safe_trace_constraint_available": False,
    }
    result = full_probe_constraints(candidate, active, initial, cfg)
    assert result["local_accuracy_loss_count"] == 1
    assert result["global_accuracy_loss_count"] == 1
    assert not result["sequential_constraints_passed"]


def test_safe_trace_uses_only_correct_valid_c4_c5_pairs():
    cfg = Config()
    profiles = []
    correctness = [1, 1, 1, 1, 0]
    for agent_id in range(5):
        profiles.append({
            "correctness_vector": [correctness[agent_id], 1],
            "invalid_vector": [0, int(agent_id == 1)],
            "trace_embedding_vector_per_question": [
                [1.0, 0.0] if agent_id == 0 else [0.0, 1.0],
                [1.0, 0.0] if agent_id == 0 else [0.0, 1.0],
            ],
        })
    result = safe_trace_diversity_c4c5(profiles, 0, cfg)
    # C4 has three valid correct peers; C5 excludes the invalid peer.
    assert result["safe_trace_pair_count"] == 6
    assert result["safe_trace_constraint_available"] is True
    wrong_target = deepcopy(profiles)
    wrong_target[0]["correctness_vector"] = [0, 0]
    unavailable = safe_trace_diversity_c4c5(wrong_target, 0, cfg)
    assert unavailable["safe_trace_constraint_available"] is False


def test_prompt_memory_capacity_slots_and_dedup():
    items = [
        _item("active", 8, reward=0.0),
        _item("accuracy", 10, reward=-1.0),
        _item("reward", 8, reward=5.0),
        _item("diverse", 8, reward=0.0, slack=3.0),
        _item("recent", 8, reward=0.5),
        _item("extra", 7, reward=9.0),
    ]
    for index, item in enumerate(items):
        item["accepted_update_index"] = index
    memory = rebuild_prompt_memory(items, "active", capacity=5)
    assert len(memory) <= 5
    assert memory[0]["prompt_memory_slot"] == "active"
    assert len({item["prompt_hash"] for item in memory}) == len(memory)
    assert "c2_split" not in {item["prompt_memory_slot"] for item in memory}


def test_safe_trace_variant_can_fill_diversity_slot():
    active = _item("active", 8)
    active["outcome_signature_hash"] = "same-outcome"
    active["safe_trace_signature_hash"] = "trace-a"
    variant = _item("variant", 8, slack=2.0)
    variant["outcome_signature_hash"] = "same-outcome"
    variant["safe_trace_signature_hash"] = "trace-b"
    memory = rebuild_prompt_memory([active, variant], "active", capacity=5)
    assert {item["prompt_hash"] for item in memory} == {"active", "variant"}
    assert next(item for item in memory if item["prompt_hash"] == "variant")["prompt_memory_slot"] == "safe_diversity_parent"


def test_rotating_order_is_deterministic_and_resume_indexable():
    assert epoch_agent_order(0) == [0, 1, 2, 3, 4]
    assert epoch_agent_order(1) == [1, 2, 3, 4, 0]
    assert epoch_agent_order(2) == [2, 3, 4, 0, 1]
    order = epoch_agent_order(1)
    assert order[3] == 4
