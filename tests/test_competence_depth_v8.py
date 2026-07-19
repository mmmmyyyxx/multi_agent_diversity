import asyncio
import copy
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.cli import (
    build_training_checkpoint,
    checkpoint_behavior_config,
    checkpoint_incompatibility_reasons,
    is_better_validation_state,
    restore_system_state,
    select_competence_probe_indices,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import (
    TraceBeamSearchSystem,
    competence_depth_dominates,
    competence_non_dominated_sort,
    competence_specialization_strength,
    competence_relative_specialization_strength,
    compute_coverage_depth_transitions,
    mechanism_signature_distance,
    normalize_mechanism_signature,
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


def test_depth1_and_depth3_track_independent_support_thresholds():
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


def test_stable_specialization_is_only_final_validation_tie_break():
    common = {
        "plurality_vote_acc": 0.6, "mean_individual_acc": 0.6,
        "bottom2_mean_acc": 0.5, "coverage_depth_c1": 0.8,
        "coverage_depth_c2": 0.6, "mean_normalized_plurality_margin": 0.1,
        "mean_invalid_rate": 0.0,
    }
    diverse = {"epoch": 2, "method_version": "v8_stable_qd_lineage", "val": {**common, "stable_specialization_score": 0.7}}
    flat = {"epoch": 1, "method_version": "v8_stable_qd_lineage", "val": {**common, "stable_specialization_score": 0.1}}
    assert is_better_validation_state(diverse, flat, 0, "competence_depth_schedule", "vote_generalization_first")
    weaker_vote = {"epoch": 3, "method_version": "v8_stable_qd_lineage", "val": {**common, "plurality_vote_acc": 0.59, "stable_specialization_score": 1.0}}
    assert not is_better_validation_state(weaker_vote, flat, 0, "competence_depth_schedule", "vote_generalization_first")


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
    system.competence_probe_indices = []
    system.competence_probe_question_hashes = []
    system.initial_competence_probe_metrics = {}
    system.latest_competence_probe_metrics = {}
    system.competence_probe_history = []
    system.initial_active_prompt_hashes = []
    system.first_nonzero_specialization_epoch = None
    system.effective_specialization_epoch_count = 0
    system.depth1_guard_rejection_count = 0
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


def probe_metrics(bottom2=0.40, mean=0.60, c1=0.80, c2=0.55, size=75):
    return {
        "probe_size": size,
        "bottom2_mean_acc": bottom2,
        "mean_individual_acc": mean,
        "coverage_depth_c1": c1,
        "coverage_depth_c2": c2,
        "per_agent_acc": [bottom2] * 2 + [mean] * 3,
    }


def relative_schedule(initial=None, snapshot=None, current=0.0, size=75, **kwargs):
    return competence_relative_specialization_strength(
        initial_metrics=initial or probe_metrics(size=size),
        snapshot_metrics=snapshot or probe_metrics(size=size),
        probe_size=size,
        current_strength=current,
        **kwargs,
    )


def test_v81_dynamic_probe_thresholds_and_relative_gain():
    record = relative_schedule(snapshot=probe_metrics(bottom2=0.46), size=75)
    assert record["effective_low_delta"] == pytest.approx(1 / 75)
    assert record["effective_high_delta"] == pytest.approx(0.06)
    assert record["bottom2_gain"] == pytest.approx(0.06)
    assert record["raw_specialization_strength"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "snapshot,reason",
    [
        (probe_metrics(bottom2=0.46, mean=0.58), "mean_regression"),
        (probe_metrics(bottom2=0.46, c1=0.78), "c1_regression"),
        (probe_metrics(bottom2=0.46, c2=0.53), "c2_regression"),
    ],
)
def test_v81_competence_preservation_gate(snapshot, reason):
    record = relative_schedule(snapshot=snapshot)
    assert record["raw_specialization_strength"] > 0
    assert record["gated_raw_specialization_strength"] == 0
    assert reason in record["gate_failure_reasons"]


def test_v81_schedule_ema_max_step_and_monotonicity():
    first = relative_schedule(snapshot=probe_metrics(bottom2=0.50), max_step=0.35, ema=0.5)
    assert first["next_specialization_strength"] == pytest.approx(0.35)
    second = relative_schedule(snapshot=probe_metrics(bottom2=0.50), current=0.35, max_step=0.35, ema=0.5)
    assert second["next_specialization_strength"] > 0.35
    regressed = relative_schedule(snapshot=probe_metrics(bottom2=0.40), current=0.6, monotonic=True)
    assert regressed["next_specialization_strength"] == pytest.approx(0.6)


def test_v81_complete_epoch_uses_static_snapshot_not_online_metrics():
    cfg = Config(
        competence_depth_enabled=True,
        competence_schedule_mode="baseline_relative_opt_snapshot",
        competence_schedule_version="competence_depth_v2_opt_snapshot_c1_guard",
    )
    system = make_system(cfg)
    system.initial_competence_probe_metrics = probe_metrics(bottom2=0.40)
    system.competence_probe_indices = list(range(75))
    record = system.complete_competence_epoch(
        per_agent_acc=[0.2] * 5,
        snapshot_metrics=probe_metrics(bottom2=0.46),
        epoch=1,
    )
    assert record["bottom2_gain"] == pytest.approx(0.06)
    assert system.previous_epoch_bottom2_mean_acc == pytest.approx(0.46)


def test_v81_fixed_probe_selection_is_deterministic_and_restorable():
    data = [{"question": f"q{i}", "answer": "A"} for i in range(20)]
    cfg = Config(seed=42, competence_probe_size=7, competence_probe_seed_offset=7000)
    first = select_competence_probe_indices(data, cfg)
    second = select_competence_probe_indices(data, cfg)
    restored = select_competence_probe_indices(data, cfg, first)
    assert first == second == restored
    assert len(first) == 7


@pytest.mark.parametrize("gain,loss,passed", [(0, 1, False), (1, 1, True), (2, 1, True)])
def test_v81_candidate_c1_guard(gain, loss, passed):
    cfg = Config(competence_depth1_candidate_guard_enabled=True)
    system = make_system(cfg)
    metrics = {"depth1_net_delta": (gain - loss) / 10}
    assert system._apply_competence_depth1_candidate_guard(metrics) is passed
    assert metrics["competence_depth1_guard_passed"] is passed
    assert (metrics.get("rejection_reason") == "competence_depth1_guard") is (not passed)


def test_v81_checkpoint_round_trip_preserves_probe_and_stage_state(tmp_path):
    cfg = Config(
        out_dir=str(tmp_path), competence_depth_enabled=True,
        competence_schedule_mode="baseline_relative_opt_snapshot",
        competence_schedule_version="competence_depth_v2_opt_snapshot_c1_guard",
    )
    system = make_system(cfg)
    system.competence_probe_indices = [3, 1]
    system.competence_probe_question_hashes = ["h3", "h1"]
    system.initial_competence_probe_metrics = probe_metrics()
    system.latest_competence_probe_metrics = probe_metrics(bottom2=0.46)
    system.competence_probe_history = [{"epoch": 0}, {"epoch": 1}]
    system.specialization_strength = 0.35
    system.specialization_strength_history = [0.0]
    payload = build_training_checkpoint(
        cfg, system, epoch_index=1, cursor=0, order=[], train_accumulators={}, best_score=0,
        best_epoch=0, epochs_without_improvement=0, stopped_early=False,
        no_effective_evolution_counter=0, no_effective_evolution_stopped=False,
        no_effective_evolution_reason="", stage="between_epochs",
    )
    restored = make_system(cfg)
    restore_system_state(restored, payload["state"])
    assert restored.competence_probe_indices == [3, 1]
    assert restored.competence_probe_question_hashes == ["h3", "h1"]
    assert restored.initial_competence_probe_metrics == system.initial_competence_probe_metrics
    assert restored.latest_competence_probe_metrics == system.latest_competence_probe_metrics
    assert restored.specialization_strength == pytest.approx(0.35)


def test_v7_fingerprint_ignores_all_v81_schedule_fields():
    baseline = checkpoint_behavior_config(Config())
    changed = checkpoint_behavior_config(Config(
        competence_schedule_mode="baseline_relative_opt_snapshot",
        competence_schedule_version="different",
        competence_probe_size=17,
        competence_depth1_candidate_guard_enabled=True,
    ))
    assert baseline == changed


def test_v81_checkpoint_rejects_old_schedule_with_explicit_version_reason(tmp_path):
    old_cfg = Config(out_dir=str(tmp_path), competence_depth_enabled=True)
    payload = build_training_checkpoint(
        old_cfg, make_system(old_cfg), epoch_index=0, cursor=0, order=[], train_accumulators={},
        best_score=0, best_epoch=0, epochs_without_improvement=0, stopped_early=False,
        no_effective_evolution_counter=0, no_effective_evolution_stopped=False,
        no_effective_evolution_reason="", stage="between_epochs",
    )
    new_cfg = Config(
        out_dir=str(tmp_path), competence_depth_enabled=True,
        competence_schedule_mode="baseline_relative_opt_snapshot",
        competence_schedule_version="competence_depth_v2_opt_snapshot_c1_guard",
    )
    reasons = checkpoint_incompatibility_reasons(payload, new_cfg, [])
    assert any("competence_schedule_version mismatch" in reason for reason in reasons)


def test_v81_probe_does_not_mutate_training_state(tmp_path):
    cfg = Config(out_dir=str(tmp_path), competence_depth_enabled=True)
    system = make_system(cfg)
    system.task_spec = SimpleNamespace(parse_gold=lambda answer, question: str(answer))
    system.competence_probe_history = []
    system.recent_window_records = [{"case": 1}]
    system.specialization_strength = 0.25

    async def fake_solve(question, prompts):
        return ["trace"] * 5, ["A"] * 5

    system.solve_with_prompts = fake_solve
    system._record_solver_rollouts = lambda *args, **kwargs: None
    system.compute_rollout_metrics = lambda *args, **kwargs: {
        "individual_correct": [1, 1, 1, 1, 1], "vote_correct": 1,
        "plurality_vote_correct": 1, "majority_vote_correct": 1,
    }
    before = {
        "prompts": [agent.current_prompt for agent in system.agents],
        "beams": copy.deepcopy([agent.prompt_beam for agent in system.agents]),
        "counts": [(agent.accept_count, agent.reject_count) for agent in system.agents],
        "profiles": copy.deepcopy([agent.capability_profile for agent in system.agents]),
        "windows": copy.deepcopy(system.recent_window_records),
        "strength": system.specialization_strength,
    }
    metrics = asyncio.run(system.evaluate_competence_probe(
        [{"question": "q", "answer": "A"}], probe_name="test", epoch=0
    ))
    assert metrics["coverage_depth_c1"] == 1.0
    assert before["prompts"] == [agent.current_prompt for agent in system.agents]
    assert before["beams"] == [agent.prompt_beam for agent in system.agents]
    assert before["counts"] == [(agent.accept_count, agent.reject_count) for agent in system.agents]
    assert before["profiles"] == [agent.capability_profile for agent in system.agents]
    assert before["windows"] == system.recent_window_records
    assert before["strength"] == system.specialization_strength


def test_v81_progressive_stage_requires_nonzero_strength_used_by_epoch():
    system = make_system(Config(competence_depth_enabled=True, competence_min_effective_specialization_epochs=1))
    system.specialization_strength = 0.35
    system.effective_specialization_epoch_count = 0
    not_exercised = system._summarize_rollout_rows([])
    assert not_exercised["progressive_stage_exercised"] is False
    assert not_exercised["progressive_stage_not_exercised_reason"] == "activation_after_final_epoch"
    system.effective_specialization_epoch_count = 1
    exercised = system._summarize_rollout_rows([])
    assert exercised["progressive_stage_exercised"] is True


def hybrid_config(**kwargs):
    values = dict(
        method_version="v8_2_hybrid_progressive",
        target_selector_mode="hybrid_competence_boundary",
        competence_depth_enabled=True,
        competence_depth2_aux_enabled=True,
        competence_progressive_residual_enabled=True,
        boundary_selector_enabled=True,
        candidate_selection_mode="competence_depth_pareto",
        competence_depth1_candidate_guard_enabled=True,
    )
    values.update(kwargs)
    return Config(**values)


def test_v82_setting_is_opt_in_and_legacy_progressive_stays_v1():
    settings = {setting.name: setting for setting in select_settings("all")}
    legacy = settings["shared_vote_tcs_competence_depth2_progressive_residual"]
    hybrid = settings["shared_vote_tcs_competence_depth2_progressive_residual_hybrid"]
    assert legacy.competence_schedule_mode in (None, "")
    assert legacy.method_version in (None, "")
    assert hybrid.method_version == "v8_stable_qd_lineage"
    assert hybrid.target_selector_mode == "hybrid_competence_boundary"
    assert hybrid.beam_policy_version == "quality_diversity_archive_v1"
    assert hybrid.active_team_selector_version == "joint_quality_diversity_v1"


def test_stable_qd_early_large_shift_has_no_self_drift_penalty():
    system = make_system(hybrid_config(method_version="v8_stable_qd_lineage"))
    metrics = {
        "reward": 0.5,
        "rejection_reason": "unsupported_large_prompt_shift",
        "prompt_change_ratio": 1.0,
        "accuracy_delta": 0.0,
    }
    result = system._apply_hybrid_soft_guards(metrics)
    assert result["rejection_reason"] == ""
    assert result["soft_mechanism_shift_penalty"] == 0.0
    assert result["soft_cycle_penalty"] == 0.0


def test_stable_qd_keeps_mechanism_contract_as_hard_guard():
    system = make_system(hybrid_config(method_version="v8_stable_qd_lineage"))
    result = system._apply_hybrid_soft_guards({"reward": 0.5, "rejection_reason": "mechanism_contract_missing"})
    assert result["rejection_reason"] == "mechanism_contract_missing"
    assert result["hard_guard_passed"] is False


def test_v82_hybrid_selector_keeps_c1_c2_errors_early(monkeypatch):
    system = make_system(hybrid_config())
    system.latest_competence_probe_metrics = {
        "per_agent_acc": [0.6] * 5, "mean_individual_acc": 0.6,
    }
    diagnosis = {
        "per_agent_general_error_rate": [0.1, 0.5, 0.1, 0.1, 0.1],
        "per_agent_c1_creation_opportunity": [0.0, 0.8, 0.0, 0.0, 0.0],
        "per_agent_c2_creation_opportunity": [0.0, 0.5, 0.0, 0.0, 0.0],
        "per_agent_plurality_pivotal_fix_rate": [0.6, 0.0, 0.0, 0.0, 0.0],
        "per_agent_dominant_wrong_rate": [0.0] * 5,
        "per_agent_shared_error_rate": [0.0] * 5,
    }
    assert system._select_hybrid_reward_agents(diagnosis)[0] == 1
    assert diagnosis["hybrid_selector_diagnostics"][1]["hybrid_target_score"] > 0


def test_v82_hybrid_selector_increases_boundary_weight_late():
    system = make_system(hybrid_config())
    diagnosis = {
        "per_agent_general_error_rate": [0.2] * 5,
        "per_agent_c1_creation_opportunity": [0.0] * 5,
        "per_agent_c2_creation_opportunity": [0.0] * 5,
        "per_agent_plurality_pivotal_fix_rate": [0.8, 0.0, 0.0, 0.0, 0.0],
        "per_agent_dominant_wrong_rate": [0.0] * 5,
        "per_agent_shared_error_rate": [0.0] * 5,
    }
    system.specialization_strength = 0.0
    system._select_hybrid_reward_agents(diagnosis)
    early = diagnosis["hybrid_selector_diagnostics"][0]["hybrid_target_score"]
    system.specialization_strength = 1.0
    system._select_hybrid_reward_agents(diagnosis)
    late = diagnosis["hybrid_selector_diagnostics"][0]["hybrid_target_score"]
    assert late > early
    assert diagnosis["hybrid_selector_weights"]["individual_error_rate"] > 0


def test_v82_case_batches_keep_general_creation_boundary_and_residual_evidence():
    system = make_system(hybrid_config(max_homogeneous_cases_per_agent=4, random_window_cases_per_agent=2))
    system.specialization_strength = 0.5
    cases = [
        {"case_id": "g1", "target_agent_id": 0, "case_type": "target_wrong_peer_correct_nonboundary", "baseline_correct_count": 3},
        {"case_id": "g2", "target_agent_id": 0, "case_type": "target_wrong_vote_already_correct", "baseline_correct_count": 2},
        {"case_id": "c1", "target_agent_id": 0, "case_type": "target_wrong_shared_error", "baseline_correct_count": 0},
        {"case_id": "c2", "target_agent_id": 0, "case_type": "target_wrong_shared_error", "baseline_correct_count": 1},
        {"case_id": "b", "target_agent_id": 0, "case_type": "target_wrong_pivotal_vote_fix", "baseline_correct_count": 2},
        {"case_id": "r", "target_agent_id": 0, "case_type": "target_wrong_dominant_wrong_cluster", "baseline_correct_count": 2},
    ]
    batches = system._build_case_generation_batches(0, {"target_error_cases": cases})
    assert {batch["batch_type"] for batch in batches} == {
        "general_error", "c1_c2_creation", "actual_plurality_boundary", "residual_shared_error",
    }
    assert sum(len(batch["cases"]) for batch in batches) <= 6
    system.specialization_strength = 0.0
    early_batches = system._build_case_generation_batches(0, {"target_error_cases": cases})
    assert "residual_shared_error" not in {batch["batch_type"] for batch in early_batches}


def test_v82_mechanism_alternative_remains_valid_when_repair_was_filtered():
    system = make_system(hybrid_config())
    assert system._hybrid_candidate_type_rejection_reason("mechanism_alternative", set()) == ""
    assert system._hybrid_candidate_type_rejection_reason(
        "mechanism_alternative", {"mechanism_alternative"}
    ) == "duplicate_candidate_type:mechanism_alternative"


def test_v82_substantive_parent_extension_is_not_treated_as_fallback_redundancy():
    system = make_system(hybrid_config())
    parent = "Use explicit reasoning and give exactly one final answer."
    extension = parent + " Enumerate each pronoun candidate and eliminate inconsistent antecedents."

    assert system._is_redundant_candidate_prompt(parent, extension) is True
    assert system._is_redundant_candidate_prompt(
        parent,
        extension,
        allow_substantive_parent_extension=True,
    ) is False
    assert system._is_redundant_candidate_prompt(
        parent,
        parent,
        allow_substantive_parent_extension=True,
    ) is True


def test_v82_tcs_parser_keeps_alternative_after_redundant_repair():
    system = make_system(hybrid_config())
    system.execution_session_id = "test"

    async def approved(**kwargs):
        return {
            "approved": True,
            "teacher_question": {"socratic_guiding_question": "Which concrete decision step should change?"},
            "critic_reviews": [{"score": 0.9}],
            "teacher_critic_rounds": 1,
            "teacher_rewrite_count": 0,
        }

    def student_item(candidate_type, prompt, steps):
        return {
            "candidate_prompt": prompt,
            "student_interpretation_of_question": "Change one decision step.",
            "target_error_pattern": "binding error",
            "accuracy_repair_rule": "Resolve references before choosing.",
            "diversity_contribution": "Use a different decision operation.",
            "error_correlation_reduction": "Avoid the shared wrong path.",
            "task_alignment_rule": "Apply reference resolution.",
            "peer_redundancy_avoidance": "Use a distinct operation.",
            "expected_accuracy_effect": "Preserve or improve accuracy.",
            "expected_diversity_effect": "Change useful behavior.",
            "risk_control": "Keep the answer format.",
            "rationale": "The operation addresses the failure.",
            "preserved_mechanisms": ["final answer format"],
            "modified_mechanism": "reference comparison",
            "change_summary": "Change reference comparison.",
            "target_residual_family": "entity_binding",
            "expected_shared_error_effect": "Reduce shared binding errors.",
            "candidate_type": candidate_type,
            "mechanism_steps": steps,
            "target_failure_buckets": ["general_error"],
            "expected_effect": "Improve reference decisions.",
        }

    async def students(**kwargs):
        return {
            "candidates": [
                student_item("task_specific_repair", "Parent prompt.", ["extract constraints"]),
                student_item("mechanism_alternative", "Compare candidate explanations before the final answer.", ["pairwise compare", "counterfactual check"]),
            ],
            "diagnostics": {},
        }

    system.generate_approved_teacher_question = approved
    system.generate_student_candidates = students
    proposals = asyncio.run(system.propose_candidates_teacher_critic_student(
        agent_id=0,
        parent_prompt="Parent prompt.",
        overlap_diagnosis={},
        num_candidates=2,
        generation_batches=[{"batch_type": "general_error", "cases": [], "purpose": "repair"}],
    ))
    assert [proposal["candidate_type"] for proposal in proposals] == ["mechanism_alternative"]


def test_v82_competence_reward_has_c1_and_residual_floor():
    system = make_system(hybrid_config())
    system.specialization_strength = 1.0
    result = system._candidate_reward_competence_depth({
        "accuracy_delta": 0.0, "depth1_gain_rate": 0.5, "depth1_loss_rate": 0.0,
        "depth2_gain_rate": 0.0, "depth2_loss_rate": 0.0,
        "vote_gain_rate": 0.0, "vote_loss_rate": 0.0,
        "depth1_net_delta": 0.5, "depth2_net_delta": 0.0,
        "boundary_shared_error_net_gain": 0.0,
    }, 0.0)
    assert result["competence_mix"] == pytest.approx(0.30)
    assert result["specialization_mix"] == pytest.approx(0.70)
    assert result["competence_reward_component"] == pytest.approx(0.4)


def test_v82_soft_cycle_guard_penalizes_without_hard_rejection():
    system = make_system(hybrid_config())
    metrics = {
        "reward": 1.0, "accuracy_delta": -0.01,
        "behavior_cycle_overlap": 20, "max_behavior_cycle_similarity": 0.98,
        "rejection_reason": "accepted_state_cycle",
    }
    result = system._apply_hybrid_soft_guards(metrics)
    assert result["rejection_reason"] == ""
    assert result["soft_guard_penalty"] > 0
    assert result["penalized_reward"] < result["raw_reward"]


def test_v82_safe_exploit_explore_beam_slots():
    system = make_system(hybrid_config())
    def item(cid, prompt, reward, source, distance):
        return {
            "candidate_id": cid, "prompt": prompt, "candidate_pool_source": source,
            "reward": reward, "metrics": {
                "reward": reward, "penalized_reward": reward,
                "baseline_target_accuracy": 0.5, "candidate_target_accuracy": 0.5,
                "baseline_invalid_rate": 0.0, "candidate_invalid_rate": 0.0,
                "vote_gain_rate": max(0.0, reward), "vote_loss_rate": 0.0,
                "stage_aux_objective": distance, "mechanism_signature_distance": distance,
                "accuracy_delta": 0.0, "depth1_net_delta": 0.0, "depth2_net_delta": 0.0,
            },
        }
    rows = [
        item("safe", "prompt", 0.1, "existing_beam", 0.0),
        item("exploit", "repair prompt", 2.0, "optimizer", 0.1),
        item("explore", "alternative prompt", 1.0, "optimizer", 0.8),
    ]
    selected, summary = system._select_hybrid_beam(rows, 3, "prompt")
    slots = {row["beam_slot"] for row in selected}
    assert slots == {"safe", "exploit", "explore"}
    assert selected[0]["beam_slot"] != "explore"
    assert summary["explore_slot_occupancy"] == 1


def test_v82_tcs_schema_and_mechanism_signature():
    system = make_system(hybrid_config())
    schema = system._student_candidate_schema_json()
    assert "candidate_type" in schema
    assert "mechanism_steps" in schema
    repair = normalize_mechanism_signature(["Enumerate candidate antecedents", "weighted score clues"])
    alternative = normalize_mechanism_signature(["Compare explanations", "counterfactual coherence"])
    assert repair != alternative
    assert mechanism_signature_distance(repair, alternative) > 0


def test_v82_vote_generalization_selector_prefers_higher_mean_not_flatter_lower_mean():
    stronger = {"epoch": 1, "val": {"plurality_vote_acc": 0.6, "mean_individual_acc": 0.62, "bottom2_mean_acc": 0.48, "coverage_depth_c1": 0.8, "coverage_depth_c2": 0.6}}
    flatter = {"epoch": 2, "val": {"plurality_vote_acc": 0.6, "mean_individual_acc": 0.58, "bottom2_mean_acc": 0.54, "coverage_depth_c1": 0.8, "coverage_depth_c2": 0.6}}
    assert is_better_validation_state(stronger, flatter, 0, "competence_depth_schedule", "vote_generalization_first")


def test_v82_checkpoint_preserves_mechanism_and_beam_slot_state(tmp_path):
    cfg = hybrid_config(out_dir=str(tmp_path))
    system = make_system(cfg)
    system.mechanism_signature_history = [{"agent_id": 1, "retained": [["weighted_scoring"]]}]
    system.mechanism_signature_by_prompt_hash = {"abc": ["weighted_scoring"]}
    system.hybrid_selector_history = [{"epoch": 1, "agents": []}]
    system.beam_slot_state = {"1": ["exploit", "safe", "explore"]}
    system.exploration_slot_candidates = [{"agent_id": 1, "candidate_id": "c"}]
    payload = build_training_checkpoint(
        cfg, system, epoch_index=1, cursor=0, order=[], train_accumulators={}, best_score=0,
        best_epoch=0, epochs_without_improvement=0, stopped_early=False,
        no_effective_evolution_counter=0, no_effective_evolution_stopped=False,
        no_effective_evolution_reason="", stage="between_epochs",
    )
    restored = make_system(cfg)
    restore_system_state(restored, payload["state"])
    assert restored.mechanism_signature_by_prompt_hash == system.mechanism_signature_by_prompt_hash
    assert restored.beam_slot_state == system.beam_slot_state
    assert restored.exploration_slot_candidates == system.exploration_slot_candidates


def test_stable_qd_checkpoint_preserves_archive_probe_and_lineage_state(tmp_path):
    cfg = hybrid_config(
        out_dir=str(tmp_path), method_version="v8_stable_qd_lineage",
        beam_policy_version="quality_diversity_archive_v1",
        active_team_selector_version="joint_quality_diversity_v1",
        lineage_policy_version="stable_lineage_anchor_v1",
        mechanism_distance_version="mechanism_sequence_embedding_v1",
    )
    system = make_system(cfg)
    system.mechanism_embedding_cache = {"mechanism": [1.0, 0.0]}
    system.prompt_probe_cache = {"probe": {"answer": "A"}}
    system.behavior_profile_by_prompt_hash = {"prompt": {"correctness_vector": [1]}}
    system.joint_team_selection_history = [{"epoch": 1, "combination_count": 243}]
    system.lineage_history = [{"epoch": 1, "agent_id": 0}]
    system.quality_diversity_archive_history = [{"agent_id": 0, "niche_count": 2}]
    system.behavior_profile_history = [{"epoch": 1, "profiles": []}]
    system.latest_joint_team_metrics = {"combination_count": 243}
    system.total_agent_update_count = 2
    system.task_repair_niche_occupancy_count = 1
    system.mechanism_niche_occupancy_count = 1
    system.peer_collapse_soft_count = 1
    system.peer_collapse_hard_rejection_count = 0
    system.agents[0].lineage_state["lineage_status"] = "provisional"
    payload = build_training_checkpoint(
        cfg, system, epoch_index=1, cursor=0, order=[], train_accumulators={}, best_score=0,
        best_epoch=0, epochs_without_improvement=0, stopped_early=False,
        no_effective_evolution_counter=0, no_effective_evolution_stopped=False,
        no_effective_evolution_reason="", stage="between_epochs",
    )
    restored = make_system(cfg)
    restore_system_state(restored, payload["state"])
    assert restored.prompt_probe_cache == system.prompt_probe_cache
    assert restored.quality_diversity_archive_history == system.quality_diversity_archive_history
    assert restored.agents[0].lineage_state["lineage_status"] == "provisional"
    assert restored.latest_joint_team_metrics["combination_count"] == 243
