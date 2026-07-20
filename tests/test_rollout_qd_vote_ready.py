import json
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.cli import rollout_vote_first_validation_key, write_selected_prompts
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.checkpoint import checkpoint_behavior_config
from multi_dataset_diverse_rl.rollout_diversity import (
    candidate_reward,
    candidate_transition_metrics,
    quality_guard,
    rollout_distance,
    rollout_signature,
    rollout_team_key,
    select_rollout_archive,
    vote_ready_candidate_key,
    wrong_diversity_is_useful,
)
from scripts.experiment_config import select_settings


def profile(answers, correctness, invalid=None, embeddings=None):
    result = {
        "answer_vector": list(answers),
        "correctness_vector": list(correctness),
        "invalid_vector": list(invalid or [0] * len(answers)),
        "trace_embedding_vector_per_question": list(embeddings or [[1.0, 0.0]] * len(answers)),
        "wrong_diversity_useful_vector": [1] * len(answers),
    }
    result["rollout_signature_hash"] = rollout_signature(result)
    return result


def item(name, rollout_profile, *, accuracy=0.8, reward=0.8, guarded=True, mechanism=None):
    return {
        "prompt": f"Prompt {name}", "prompt_hash": name, "generation": 1, "reward": reward,
        "proposal": {"mechanism_steps": mechanism or []},
        "metrics": {
            "rollout_profile": rollout_profile,
            "candidate_target_accuracy": accuracy,
            "candidate_invalid_rate": 0.0,
            "rollout_quality_guard_passed": guarded,
            "candidate_rollout_diversity": 0.2,
        },
    }


def test_rollout_settings_disable_mechanism_and_capability_decisions():
    settings = {setting.name: setting for setting in select_settings("all")}
    assert settings["shared_accuracy_rollout_embedding_tcs"].method_version == "v8_accuracy_rollout_embedding"
    assert settings["shared_vote_ready_rollout_diversity_tcs"].method_version == "v8_rollout_qd_vote_ready"
    cfg = Config(method_version="v8_rollout_qd_vote_ready", reward_mode="rollout_vote_ready")
    fingerprint = checkpoint_behavior_config(cfg)
    assert fingerprint["method_version"] == "v8_rollout_qd_vote_ready"
    assert cfg.residual_specialization_enabled is False
    assert cfg.competence_depth_enabled is False


def test_rollout_schema_does_not_require_mechanism_metadata():
    from multi_dataset_diverse_rl.optimization.candidate_schema import CandidateSchemaMixin

    holder = object.__new__(CandidateSchemaMixin)
    holder.cfg = Config(method_version="v8_rollout_qd_vote_ready")
    minimum = {
        "candidate_prompt": "Solve with an explicit consistency check.",
        "target_error_pattern": "missed qualifier",
        "accuracy_repair_rule": "verify every qualifier",
        "expected_accuracy_effect": "fewer qualifier mistakes",
        "rollout_diversity_intent": "change valid solver traces only when useful",
    }
    assert holder._missing_optimizer_fields(minimum, "teacher_critic_student") == []


def test_text_difference_does_not_create_rollout_niches():
    cfg = Config(method_version="v8_rollout_qd_vote_ready")
    same = profile(["A", "B"], [1, 0])
    incumbent = item("incumbent", same)
    paraphrase = item("wildly-different-text", dict(same), mechanism=["claimed novelty"])
    better = item("better", profile(["A", "A"], [1, 1]), accuracy=1.0, reward=1.0)
    archive = select_rollout_archive([incumbent, paraphrase, better], "incumbent", 6, cfg, vote_ready=True)
    hashes = {row["prompt_hash"] for row in archive}
    assert "better" in hashes
    assert hashes & {"incumbent", "wildly-different-text"} == {"incumbent"}


def test_similar_text_with_different_rollouts_can_coexist():
    cfg = Config(method_version="v8_rollout_qd_vote_ready")
    left = item("prompt-v1", profile(["A", "B"], [1, 0]))
    right = item("prompt-v2", profile(["A", "A"], [1, 1]), accuracy=1.0)
    archive = select_rollout_archive([left, right], "prompt-v1", 6, cfg, vote_ready=True)
    assert {row["prompt_hash"] for row in archive} == {"prompt-v1", "prompt-v2"}


def test_invalid_or_low_accuracy_diversity_cannot_win():
    cfg = Config(method_version="v8_rollout_qd_vote_ready")
    metrics = {
        "baseline_target_accuracy": 0.8, "candidate_target_accuracy": 0.2,
        "baseline_invalid_rate": 0.0, "candidate_invalid_rate": 0.5,
        "c3_to_c2_count": 0, "vote_loss_count": 0,
        "rollout_diversity_delta": 1.0,
    }
    assert quality_guard(metrics, cfg)["rollout_quality_guard_passed"] is False
    good = {**metrics, "candidate_target_accuracy": 0.8, "candidate_invalid_rate": 0.0, "rollout_diversity_delta": 0.1}
    assert candidate_reward(good, cfg, vote_ready=True) > candidate_reward(metrics, cfg, vote_ready=True)
    invalid_profile = profile(["", ""], [0, 0], invalid=[1, 1], embeddings=[[1, 0], [0, 1]])
    assert rollout_distance(invalid_profile, invalid_profile)["rollout_trace_embedding_distance"] == 0.0


def test_correctness_distance_ignores_invalid_correctness_flags():
    valid = profile(["A", "B"], [1, 0], invalid=[0, 0])
    invalid_flagged_correct = profile(["A", "B"], [1, 0], invalid=[1, 0])
    empty = profile(["A", "B"], [0, 0], invalid=[0, 0])
    assert rollout_distance(invalid_flagged_correct, empty)["correct_set_rollout_distance"] == 0.0
    assert rollout_distance(valid, empty)["correct_set_rollout_distance"] == 1.0


def test_trace_distance_includes_identical_valid_pairs_in_mean():
    left = profile(["A", "A"], [1, 1], embeddings=[[1, 0], [1, 0]])
    right = profile(["A", "A"], [1, 1], embeddings=[[1, 0], [0, 1]])
    assert rollout_distance(left, right)["rollout_trace_embedding_distance"] == pytest.approx(0.5)


def test_useful_wrong_distance_is_symmetric():
    left = profile(["B"], [0])
    right = profile(["C"], [0])
    left["wrong_diversity_useful_vector"] = [0]
    assert rollout_distance(left, right)["useful_wrong_answer_dispersion"] == 1.0
    assert rollout_distance(right, left)["useful_wrong_answer_dispersion"] == 1.0


def test_baseline_wrong_diversity_mask_does_not_use_candidate_improvement():
    row = {
        "baseline_gold_vote_count": 1,
        "baseline_largest_wrong_vote_count": 4,
        "candidate_largest_wrong_vote_count": 2,
        "baseline_plurality_margin_votes": -3,
        "candidate_plurality_margin_votes": 1,
        "baseline_vote_correct": False,
        "candidate_vote_correct": True,
    }
    assert wrong_diversity_is_useful(row, candidate=False) is False
    assert wrong_diversity_is_useful(row, candidate=True) is True


def transition_row(before, after, *, before_vote=False, after_vote=False):
    return {
        "baseline_individual_correct": [1] * before + [0] * (5 - before),
        "candidate_individual_correct": [1] * after + [0] * (5 - after),
        "baseline_vote_correct": before_vote, "candidate_vote_correct": after_vote,
        "baseline_plurality_margin_votes": before - (5 - before),
        "candidate_plurality_margin_votes": after - (5 - after),
        "baseline_largest_wrong_vote_count": 5 - before,
        "candidate_largest_wrong_vote_count": 5 - after,
    }


def test_c2_to_c3_is_prioritized_over_c0_to_c1():
    c0_c1 = candidate_transition_metrics([transition_row(0, 1)])
    c2_c3 = candidate_transition_metrics([transition_row(2, 3, after_vote=True)])
    common = {"candidate_target_accuracy": 0.8, "candidate_invalid_rate": 0.0, "candidate_rollout_diversity": 0.1}
    left = {"prompt_hash": "a", "metrics": {**common, **c0_c1}}
    right = {"prompt_hash": "b", "metrics": {**common, **c2_c3}}
    assert vote_ready_candidate_key(right) > vote_ready_candidate_key(left)


def test_c3_or_vote_regression_fails_zero_loss_guard():
    cfg = Config(method_version="v8_rollout_qd_vote_ready")
    c3_loss = candidate_transition_metrics([transition_row(3, 2, before_vote=True)])
    metrics = {
        **c3_loss,
        "baseline_target_accuracy": 0.8, "candidate_target_accuracy": 0.8,
        "baseline_invalid_rate": 0.0, "candidate_invalid_rate": 0.0,
    }
    guard = quality_guard(metrics, cfg)
    assert guard["c3_loss_guard_passed"] is False
    assert guard["vote_loss_guard_passed"] is False


def test_joint_selector_keeps_vote_and_c3_above_diversity():
    team_a = {
        "vote_correct_count": 70, "c3_correct_count": 70, "total_agent_correct_count": 350,
        "bottom2_correct_count": 120, "mean_gold_plurality_margin": 0.3,
        "dominant_wrong_concentration": 0.5, "rollout_diversity_score": 0.2,
        "coverage_depth_c2": 0.8, "coverage_depth_c1": 0.9,
    }
    team_b = {**team_a, "vote_correct_count": 68, "c3_correct_count": 62, "rollout_diversity_score": 0.8}
    assert rollout_team_key(team_a) > rollout_team_key(team_b)
    team_c = {**team_a, "rollout_diversity_score": 0.8}
    assert rollout_team_key(team_c) > rollout_team_key(team_a)


def test_rollout_validation_key_is_vote_c3_first():
    high_vote = {"epoch": 2, "val": {"vote_acc": 0.70, "coverage_depth_c3": 0.70, "mean_individual_acc": 0.6}}
    high_div = {"epoch": 1, "val": {"vote_acc": 0.68, "coverage_depth_c3": 0.62, "mean_individual_acc": 0.9, "rollout_embedding_diversity": 1.0}}
    assert rollout_vote_first_validation_key(high_vote) < rollout_vote_first_validation_key(high_div)


def test_rollout_best_state_serializes_selection_key_and_method_metadata(tmp_path):
    cfg = Config(
        method_version="v8_rollout_qd_vote_ready",
        beam_policy_version="rollout_archive_v1",
        active_team_selector_version="vote_ready_rollout_joint_v1",
    )
    system = SimpleNamespace(
        cfg=cfg,
        agents=[SimpleNamespace(current_prompt="Prompt A", lineage_state={})],
        latest_joint_team_metrics={"vote_correct_count": 7},
        mechanism_based_decision_count=0,
        _hash=lambda value: "hash-a",
    )
    epoch_record = {"epoch": 1, "val": {"vote_acc": 0.7, "coverage_depth_c3": 0.6}}
    path = tmp_path / "best_prompts.json"
    write_selected_prompts(path, system, 1, "rollout_vote_first", 0.7, "rollout_vote_first", epoch_record)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["best_state_selection_key"] == list(rollout_vote_first_validation_key(epoch_record))
    assert payload["method_version"] == "v8_rollout_qd_vote_ready"
    assert payload["mechanism_diversity_enabled"] is False
    assert payload["joint_team_metrics"] == {"vote_correct_count": 7}
