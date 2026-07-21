from copy import deepcopy
import json

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.cli import rollout_method_metadata
from multi_dataset_diverse_rl.optimization.candidate_generator import CandidateGeneratorMixin
from multi_dataset_diverse_rl.persistence.checkpoint import checkpoint_behavior_config_fingerprint
from multi_dataset_diverse_rl.rollout_diversity import (
    is_fixed_acceptance_probe_method,
    is_rollout_qd_method,
)
from multi_dataset_diverse_rl.sequential_state import (
    accuracy_first_key,
    candidate_strictly_beats_incumbent,
    epoch_agent_order,
    full_probe_constraints,
    outcome_signature,
    paired_safe_trace_diversity_c4c5,
    question_state,
    rebuild_prompt_memory,
    safe_trace_diversity_c4c5,
    safe_trace_signature,
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


def test_v9_is_fixed_probe_but_not_rollout_qd_and_v8_route_is_unchanged():
    assert not is_rollout_qd_method("v9_state_conditioned_error")
    assert is_fixed_acceptance_probe_method("v9_state_conditioned_error")
    for method in ("v8_accuracy_rollout_embedding", "v8_rollout_qd_vote_ready"):
        assert is_rollout_qd_method(method)
        assert is_fixed_acceptance_probe_method(method)


def test_v9_optimizer_context_excludes_wrong_cluster_and_boundary_fields():
    holder = CandidateGeneratorMixin()
    holder.state_search_diagnostics = {}
    context = holder._build_v9_sequential_teacher_context({
        "target_agent_id": 2,
        "parent_prompt_preview": "repair target errors",
        "peer_role_specs": [{"summary": "generic procedural redundancy"}],
        "validity_constraints": {"required_final_answer_line": True},
        "optimization_routes": ["general_accuracy", "vote_conversion"],
        "generation_batches": [{
            "purpose": "make the target correct on vote failures",
            "dominant_wrong": 4,
            "cases": [{"wrong_cluster": "B", "repair_hint": "preserve correct cases"}],
        }],
        "diagnostic_focus": {
            "problem_type": "bbh",
            "answer_format": "option_letter",
            "target_error_patterns": ["ambiguity handling"],
            "invalid_output_patterns": ["missing final answer"],
            "prompt_redundancy_summary": "generic procedural redundancy",
            "peer_behavior_summary": ["different validation order"],
            "target_error_summary": "target_error_count=3; target_dominant_wrong_redundancy_count=2",
            "invalid_output_summary": "target_invalid_rate=0.1",
            "boundary_useful_diversity": 0.5,
        },
    })
    serialized = json.dumps(context, sort_keys=True).lower()
    for token in (
        "dominant_wrong", "wrong_cluster", "wrong_split", "dispersion",
        "boundary_useful_diversity", "diversity",
    ):
        assert token not in serialized
    assert "target_error_patterns" in serialized
    assert "vote_conversion" in serialized
    assert holder.state_search_diagnostics["optimizer_context_wrong_cluster_field_count"] == 0


def test_v9_ablation_presets_are_single_variable():
    from scripts.experiment_config import select_settings

    names = (
        "shared_v9_sequential_accuracy,shared_v9_sequential_accuracy_state,"
        "shared_v9_sequential_accuracy_state_vote,"
        "shared_v9_sequential_accuracy_state_vote_diversity"
    )
    presets = [setting.resolved_overrides() for setting in select_settings(names)]

    def differences(left, right):
        return {
            key for key in set(left) | set(right)
            if left.get(key) != right.get(key)
        }

    assert differences(presets[0], presets[1]) == {"state_distribution_reward_enabled"}
    assert differences(presets[1], presets[2]) == {"state_vote_reward_enabled"}
    assert differences(presets[2], presets[3]) == {"state_diversity_constraints_enabled"}
    assert all(preset["state_bottom2_reward_enabled"] is False for preset in presets)
    assert all(preset["state_c2_wrong_split_enabled"] is False for preset in presets)
    assert all(preset["state_rollout_exploration_enabled"] is False for preset in presets)


def test_v9_metadata_and_fingerprint_ignore_deprecated_archive_controls():
    base = Config(
        method_version="v9_state_conditioned_error",
        state_joint_total_correct_slack_rate=0.01,
        state_representative_capacity=4,
    )
    changed_legacy = Config(
        method_version="v9_state_conditioned_error",
        state_joint_total_correct_slack_rate=9.0,
        state_representative_capacity=99,
    )
    assert checkpoint_behavior_config_fingerprint(base) == checkpoint_behavior_config_fingerprint(changed_legacy)
    changed_active = Config(
        method_version="v9_state_conditioned_error",
        state_bottom2_reward_enabled=True,
    )
    assert checkpoint_behavior_config_fingerprint(base) != checkpoint_behavior_config_fingerprint(changed_active)
    metadata = rollout_method_metadata(base)
    assert metadata["rollout_qd_method"] is False
    assert metadata["fixed_acceptance_probe_enabled"] is True
    assert metadata["rollout_archive_enabled"] is False
    assert metadata["true_plurality_vote_delta_used"] is True
    assert metadata["wrong_answer_dispersion_used_for_generation"] is False
    assert "state_joint_total_correct_slack_rate" not in metadata


def test_state_potential_rewards_expected_transitions():
    cfg = Config(state_vote_reward_enabled=False, state_reward_bottom2_weight=0.0)
    rewards = []
    for before, after in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        active = [[1] * before + [0] * (5 - before)]
        candidate = [[1] * after + [0] * (5 - after)]
        rewards.append(state_vote_reward(active, candidate, [0], [0], cfg)["state_reward_total"])
    assert rewards == pytest.approx([1.0, 0.75, 1.5, 0.35, 0.15])


@pytest.mark.parametrize(
    "distribution_enabled,vote_enabled",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_bottom2_component_is_independent_and_disabled_by_default(
    distribution_enabled, vote_enabled
):
    cfg = Config(
        state_distribution_reward_enabled=distribution_enabled,
        state_vote_reward_enabled=vote_enabled,
        state_bottom2_reward_enabled=False,
    )
    result = state_vote_reward(
        [[1, 1, 0, 0, 0]],
        [[1, 1, 1, 0, 0]],
        [0],
        [1],
        cfg,
    )
    assert result["state_reward_bottom2_component"] == 0.0


def test_true_vote_gain_and_loss_are_used_even_when_g_is_unchanged():
    cfg = Config(state_distribution_reward_enabled=False, state_vote_reward_enabled=True)
    active_correctness = [[1, 1, 0, 0, 0]]
    candidate_correctness = deepcopy(active_correctness)
    gain = state_vote_reward(
        active_correctness,
        candidate_correctness,
        [0],
        [1],
        cfg,
    )
    loss = state_vote_reward(candidate_correctness, active_correctness, [1], [0], cfg)
    assert gain["vote_accuracy_delta"] == 1.0
    assert gain["state_reward_vote_component"] == 2.0
    assert gain["vote_gain_count"] == 1
    assert loss["vote_accuracy_delta"] == -1.0
    assert loss["state_reward_vote_component"] == -2.0
    assert loss["vote_loss_count"] == 1


def test_wrong_label_or_h_change_without_vote_change_has_zero_reward_value():
    cfg = Config(state_distribution_reward_enabled=False, state_vote_reward_enabled=True)
    correctness = [[1, 1, 0, 0, 0]]
    result = state_vote_reward(correctness, deepcopy(correctness), [0], [0], cfg)
    assert result["state_reward_total"] == 0.0
    assert result["vote_accuracy_delta"] == 0.0


def test_accuracy_is_primary_over_reward_invalid_and_diversity():
    cfg = Config()
    accurate = _item("accurate", 9, reward=-10.0, invalid=2, slack=-1.0)
    tempting = _item("tempting", 8, reward=100.0, invalid=0, slack=100.0)
    assert accuracy_first_key(accurate, cfg) > accuracy_first_key(tempting, cfg)
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


def test_full_probe_safe_trace_constraint_uses_paired_support_and_skips_zero_support():
    cfg = Config(state_diversity_constraints_enabled=True, state_safe_trace_local_epsilon=0.05)
    active = {
        "candidate_target_correct_count": 8,
        "candidate_invalid_count": 0,
        "correct_set_diversity_mean": 0.4,
        "correct_set_diversity_min": 0.2,
    }
    candidate = {
        "full_probe_size": 10,
        "candidate_target_correct_count": 8,
        "candidate_invalid_count": 0,
        "correct_set_diversity_mean": 0.4,
        "correct_set_diversity_min": 0.2,
        "active_paired_safe_trace_diversity": 0.8,
        "candidate_paired_safe_trace_diversity": 0.6,
        "paired_safe_trace_pair_count": 4,
        "paired_safe_trace_constraint_available": True,
    }
    rejected = full_probe_constraints(candidate, active, active, cfg)
    assert rejected["safe_trace_constraint_passed"] is False
    assert rejected["safe_trace_constraint_rejected"] is True
    no_support = {**candidate, "paired_safe_trace_constraint_available": False, "paired_safe_trace_pair_count": 0}
    passed = full_probe_constraints(no_support, active, active, cfg)
    assert passed["safe_trace_constraint_passed"] is True


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
    memory = rebuild_prompt_memory(items, "active", capacity=5, config=Config())
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
    accuracy = _item("accuracy", 10)
    reward = _item("reward", 9, reward=3.0)
    memory = rebuild_prompt_memory(
        [active, variant, accuracy, reward], "active", capacity=5,
        config=Config(state_diversity_constraints_enabled=True),
    )
    assert {"active", "variant"}.issubset({item["prompt_hash"] for item in memory})
    assert next(item for item in memory if item["prompt_hash"] == "variant")["prompt_memory_slot"] == "safe_diversity_parent"


def test_diversity_slack_affects_only_enabled_ablation_key():
    low = _item("low", 8, reward=1.0, slack=0.0)
    high = _item("high", 8, reward=1.0, slack=2.0)
    disabled = Config(state_diversity_constraints_enabled=False)
    enabled = Config(state_diversity_constraints_enabled=True)
    assert accuracy_first_key(low, disabled)[:-2] == accuracy_first_key(high, disabled)[:-2]
    assert accuracy_first_key(high, enabled) > accuracy_first_key(low, enabled)


def test_memory_fills_to_capacity_and_previous_active_is_rollback():
    items = [_item(f"p{index}", 10 - index, reward=float(index)) for index in range(6)]
    memory, diagnostics = rebuild_prompt_memory(
        items,
        "p0",
        capacity=5,
        config=Config(state_diversity_constraints_enabled=False),
        previous_active_item=items[1],
        return_diagnostics=True,
    )
    assert len(memory) == 5
    assert diagnostics["memory_underfilled"] is False
    rollback = next(item for item in memory if item["prompt_memory_slot"] == "rollback_or_recent_success")
    assert rollback["prompt_hash"] == "p1"
    assert diagnostics["rollback_prompt_hash"] == "p1"
    assert not any(item["prompt_memory_slot"] == "safe_diversity_parent" for item in memory)


def test_previous_active_rollback_survives_matching_outcome_signature():
    new_active = _item("new", 8)
    previous_active = _item("old", 8)
    new_active["outcome_signature_hash"] = "same-outcome"
    previous_active["outcome_signature_hash"] = "same-outcome"
    memory = rebuild_prompt_memory(
        [new_active, previous_active],
        "new",
        config=Config(state_diversity_constraints_enabled=False),
        previous_active_item=previous_active,
    )
    rollback = next(item for item in memory if item["prompt_memory_slot"] == "rollback_or_recent_success")
    assert rollback["prompt_hash"] == "old"


def test_memory_underfill_reports_reason():
    memory, diagnostics = rebuild_prompt_memory(
        [_item("active", 8)],
        "active",
        capacity=5,
        config=Config(state_diversity_constraints_enabled=False),
        return_diagnostics=True,
    )
    assert len(memory) == 1
    assert diagnostics["memory_underfilled"] is True
    assert diagnostics["memory_underfilled_reason"] == "insufficient_distinct_safe_prompts"


def test_outcome_signature_scopes_probe_and_order_but_ignores_answer_labels():
    profile = {
        "question_hashes": ["q0", "q1"],
        "correctness_vector": [1, 0],
        "invalid_vector": [0, 0],
        "answer_vector": ["A", "B"],
    }
    changed_label = {**profile, "answer_vector": ["A", "C"]}
    base = outcome_signature(profile, "v2", "probe-a", ["q0", "q1"])
    assert base == outcome_signature(changed_label, "v2", "probe-a", ["q0", "q1"])
    assert base != outcome_signature(profile, "v2", "probe-b", ["q0", "q1"])
    assert base != outcome_signature(profile, "v2", "probe-a", ["q1", "q0"])


def _team_profiles(target_embedding=(1.0, 0.0)):
    profiles = []
    for agent_id in range(5):
        profiles.append({
            "question_hashes": ["q0", "q1"],
            "correctness_vector": [1, int(agent_id < 3)],
            "invalid_vector": [0, 0],
            "trace_embedding_vector_per_question": [
                list(target_embedding) if agent_id == 0 else [0.0, 1.0],
                [1.0, 0.0],
            ],
            "fixed_probe_hash": "probe",
        })
    return profiles


def test_safe_trace_signature_uses_only_valid_correct_c4_c5_pairs_and_is_stable():
    profiles = _team_profiles((1.0, 0.0))
    signature = safe_trace_signature(profiles, 0, "v2", "probe", ["q0", "q1"])
    perturbed = _team_profiles((1.00001, 0.00001))
    assert signature == safe_trace_signature(perturbed, 0, "v2", "probe", ["q0", "q1"])
    invalid = deepcopy(profiles)
    invalid[1]["invalid_vector"][0] = 1
    assert signature != safe_trace_signature(invalid, 0, "v2", "probe", ["q0", "q1"])


def test_paired_safe_trace_uses_identical_support_and_zero_support_skips():
    cfg = Config()
    active = _team_profiles()
    candidate = deepcopy(active)
    candidate[0]["trace_embedding_vector_per_question"][0] = [0.0, 1.0]
    paired = paired_safe_trace_diversity_c4c5(active, candidate, 0, cfg)
    assert paired["paired_safe_trace_pair_count"] == 4
    assert paired["paired_safe_trace_constraint_available"] is True
    assert paired["paired_safe_trace_delta"] < 0.0
    candidate[0]["correctness_vector"][0] = 0
    unavailable = paired_safe_trace_diversity_c4c5(active, candidate, 0, cfg)
    assert unavailable["paired_safe_trace_pair_count"] == 0
    assert unavailable["paired_safe_trace_constraint_available"] is False


def test_rotating_order_is_deterministic_and_resume_indexable():
    assert epoch_agent_order(0) == [0, 1, 2, 3, 4]
    assert epoch_agent_order(1) == [1, 2, 3, 4, 0]
    assert epoch_agent_order(2) == [2, 3, 4, 0, 1]
    order = epoch_agent_order(1)
    assert order[3] == 4
