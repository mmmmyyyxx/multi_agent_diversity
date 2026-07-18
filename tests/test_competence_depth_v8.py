from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.cli import (
    build_training_checkpoint,
    checkpoint_behavior_config,
    is_better_validation_state,
    restore_system_state,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import (
    TraceBeamSearchSystem,
    competence_depth_dominates,
    competence_non_dominated_sort,
    competence_specialization_strength,
    compute_coverage_depth_transitions,
)
from scripts.experiment_config import select_settings


def rows(before, after):
    return compute_coverage_depth_transitions([before], [after], max_depth=5)


@pytest.mark.parametrize(
    "before,after,gain,loss",
    [
        ([0, 0, 0, 0, 0], [1, 0, 0, 0, 0], 1, None),
        ([1, 0, 0, 0, 0], [1, 1, 0, 0, 0], 2, None),
        ([1, 1, 0, 0, 0], [1, 1, 1, 0, 0], 3, None),
        ([1, 1, 1, 0, 0], [1, 1, 0, 0, 0], None, 3),
        ([1, 1, 0, 0, 0], [1, 0, 0, 0, 0], None, 2),
    ],
)
def test_coverage_depth_single_threshold_transition(before, after, gain, loss):
    result = rows(before, after)
    if gain:
        assert result[f"depth{gain}_gain_count"] == 1
    if loss:
        assert result[f"depth{loss}_loss_count"] == 1


def test_depth1_and_depth3_match_oracle_and_majority_deltas():
    before = [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0]]
    after = [[1, 0, 0, 0, 0], [1, 1, 1, 0, 0]]
    result = compute_coverage_depth_transitions(before, after)
    assert result["depth1_net_delta"] == pytest.approx(0.5)
    assert result["depth3_net_delta"] == pytest.approx(0.5)


@pytest.mark.parametrize("bottom2,expected", [(0.50, 0.0), (0.60, 0.5), (0.70, 1.0)])
def test_competence_schedule(bottom2, expected):
    assert competence_specialization_strength(bottom2, 0.55, 0.65) == pytest.approx(expected)


def candidate(cid, gain=0, loss=0, acc=0.5, aux=0):
    return {
        "candidate_id": cid,
        "metrics": {
            "vote_gain_rate": gain,
            "vote_loss_rate": loss,
            "candidate_target_accuracy": acc,
            "stage_aux_objective": aux,
        },
    }


def test_competence_depth_pareto_uses_fourth_objective():
    depth2 = candidate("depth2", aux=1.0)
    neutral = candidate("neutral", aux=0.0)
    assert competence_depth_dominates(depth2, neutral)
    assert competence_non_dominated_sort([neutral, depth2])[0] == [1]


def test_v8_settings_are_opt_in_and_v7_fingerprint_payload_has_no_v8_fields():
    names = {setting.name: setting for setting in select_settings("all")}
    full = names["shared_vote_tcs_competence_depth2_progressive_residual"]
    assert full.candidate_selection_mode == "competence_depth_pareto"
    assert full.competence_progressive_residual_enabled is True
    assert "competence_depth_enabled" not in checkpoint_behavior_config(Config())


def test_weak_agent_selector_bonus_prefers_larger_competence_deficit(monkeypatch):
    cfg = Config(
        competence_depth_enabled=True,
        boundary_selector_enabled=True,
        competence_selector_weight=10.0,
        capability_affinity_weight=0.0,
        capability_coverage_gap_weight=0.0,
    )
    system = make_system(cfg)
    system.previous_epoch_per_agent_acc = [0.8, 0.3, 0.8, 0.8, 0.8]
    diagnosis = {
        "per_agent_pivotal_fix_rate": [0.1] * 5,
        "per_agent_near_boundary_error_rate": [0.0] * 5,
        "per_agent_dominant_wrong_rate": [0.0] * 5,
        "per_agent_shared_error_rate": [0.0] * 5,
        "per_agent_general_error_rate": [0.0] * 5,
        "per_agent_invalid_rate": [0.0] * 5,
    }
    assert system._select_boundary_reward_agents(diagnosis)[0] == 1


def test_vote_competence_first_prefers_bottom_agents_over_higher_mean():
    a = {"epoch": 1, "val": {"vote_acc": 0.62, "bottom2_mean_acc": 0.58, "coverage_depth_c2": 0.60, "best_minus_bottom2_gap": 0.04, "mean_individual_acc": 0.59}}
    b = {"epoch": 2, "val": {"vote_acc": 0.62, "bottom2_mean_acc": 0.45, "coverage_depth_c2": 0.56, "best_minus_bottom2_gap": 0.31, "mean_individual_acc": 0.62}}
    assert is_better_validation_state(a, b, 0, "competence_depth_schedule", "vote_competence_first")


def make_system(cfg=None):
    system = TraceBeamSearchSystem.__new__(TraceBeamSearchSystem)
    system.cfg = cfg or Config(competence_depth_enabled=True)
    system.prompt_overlength_rejection_count = 0
    system.truncated_prompt_count = 0
    system.specialization_strength = 0.0
    system.previous_epoch_per_agent_acc = []
    system.previous_epoch_bottom2_mean_acc = 0.0
    system.competence_phase_epoch = 1
    system.competence_schedule_version = "competence_depth_v1"
    system.specialization_strength_history = [0.0]
    system.recent_window_records = []
    system.agents = [AgentState("prompt") for _ in range(5)]
    return system


def test_v8_prompt_over_hard_limit_is_rejected_without_truncation():
    system = make_system(Config(competence_depth_enabled=True, student_candidate_prompt_soft_max_chars=10, student_candidate_prompt_hard_max_chars=20))
    original = "x" * 21 + "."
    prepared, audit = system._prepare_v8_candidate_text_fields({"candidate_prompt": original})
    assert prepared is None
    assert audit["candidate_prompt_overlength_rejected"]
    assert original == "x" * 21 + "."
    assert system.truncated_prompt_count == 0


def test_v8_prompt_requires_complete_sentence_and_accepts_soft_overage():
    system = make_system(Config(competence_depth_enabled=True, student_candidate_prompt_soft_max_chars=10, student_candidate_prompt_hard_max_chars=30))
    prepared, audit = system._prepare_v8_candidate_text_fields({"candidate_prompt": "Use two checks."})
    assert prepared["candidate_prompt"] == "Use two checks."
    assert audit["candidate_prompt_over_soft_limit"]
    incomplete, incomplete_audit = system._prepare_v8_candidate_text_fields({"candidate_prompt": "Use two checks"})
    assert incomplete is None
    assert incomplete_audit["candidate_prompt_incomplete_rejected"]


def test_progressive_weights_and_shrinkage():
    system = make_system(Config(competence_depth_enabled=True, competence_progressive_residual_enabled=True, specialization_support_shrinkage=3, competence_extra_support_shrinkage=3))
    system.specialization_strength = 0.0
    assert system._effective_progressive_weight(0.25) == 0
    assert system._effective_support_shrinkage() == 6
    system.specialization_strength = 0.5
    assert system._effective_progressive_weight(0.25) == pytest.approx(0.125)
    assert system._effective_support_shrinkage() == pytest.approx(4.5)
    system.specialization_strength = 1.0
    assert system._effective_progressive_weight(0.25) == pytest.approx(0.25)
    assert system._effective_support_shrinkage() == pytest.approx(3)


def test_checkpoint_restores_competence_schedule(tmp_path):
    cfg = Config(out_dir=str(tmp_path), competence_depth_enabled=True)
    system = make_system(cfg)
    system.specialization_strength = 0.5
    system.previous_epoch_per_agent_acc = [0.5, 0.6, 0.7, 0.8, 0.9]
    system.previous_epoch_bottom2_mean_acc = 0.55
    system.competence_phase_epoch = 2
    payload = build_training_checkpoint(
        cfg, system, epoch_index=1, cursor=0, order=[], train_accumulators={}, best_score=0,
        best_epoch=0, epochs_without_improvement=0, stopped_early=False,
        no_effective_evolution_counter=0, no_effective_evolution_stopped=False,
        no_effective_evolution_reason="", stage="between_epochs",
    )
    restored = make_system(cfg)
    restore_system_state(restored, payload["state"])
    assert restored.specialization_strength == 0.5
    assert restored.previous_epoch_per_agent_acc == system.previous_epoch_per_agent_acc
    assert restored.competence_phase_epoch == 2
