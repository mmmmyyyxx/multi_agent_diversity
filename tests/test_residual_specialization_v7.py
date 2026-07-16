import asyncio
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.cli import (
    build_training_checkpoint,
    checkpoint_incompatibility_reasons,
    restore_system_state,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import (
    AgentState,
    BehaviorContext,
    BehaviorFingerprintEntry,
    BehaviorStateSummary,
    CapabilityResidualFamily,
    RejectedBehaviorSummary,
)
from multi_dataset_diverse_rl.system import (
    TraceBeamSearchSystem,
    error_pareto_dominates,
    pareto_dominates,
)


def _system(**kwargs):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(**kwargs)
    system.agents = [AgentState("shared") for _ in range(system.cfg.agents)]
    system.task_spec = SimpleNamespace(match_answer=lambda pred, gold: str(pred) == str(gold))
    return system


def _row(*, context, family, vote_gain=False, margin_delta=0.0, fix=False):
    return {
        "behavior_context": context,
        "capability_residual_family": family,
        "baseline_vote_correct": False,
        "candidate_vote_correct": bool(vote_gain),
        "baseline_mean_vote_margin": 0.0,
        "candidate_mean_vote_margin": margin_delta,
        "baseline_target_correct": False,
        "candidate_target_correct": bool(fix),
    }


def test_rare_pivotal_evidence_is_shrunk_but_not_discarded():
    system = _system(residual_specialization_enabled=True)
    pivotal = system._candidate_residual_metrics([
        _row(
            context=BehaviorContext.TEAM_WRONG_PIVOTAL_FIX.value,
            family=CapabilityResidualFamily.RELATION_TRACKING.value,
            vote_gain=True,
            fix=True,
        )
    ])
    common = system._candidate_residual_metrics([
        _row(
            context=BehaviorContext.TEAM_WRONG_NONPIVOTAL.value,
            family=CapabilityResidualFamily.OPTION_COMPARISON.value,
            margin_delta=0.01,
        )
        for _ in range(20)
    ])
    rare_value = pivotal["capability_shrunk_transition"][CapabilityResidualFamily.RELATION_TRACKING.value]
    common_value = common["capability_shrunk_transition"][CapabilityResidualFamily.OPTION_COMPARISON.value]
    assert rare_value > 0.0
    assert pivotal["capability_support_reliability"][CapabilityResidualFamily.RELATION_TRACKING.value] == pytest.approx(0.25)
    assert rare_value > common_value
    assert set(pivotal["capability_evidence_rows"][0]) >= {
        "raw_transition_value", "vote_context_weight", "support_reliability",
        "shrunk_transition_value", "capability_residual_family",
    }


def test_boundary_selector_prefers_more_pivotal_agent_over_more_total_errors():
    system = _system(boundary_selector_enabled=True, capability_affinity_weight=0.25)
    diagnosis = {
        "per_agent_pivotal_fix_rate": [0.05, 0.50, 0.0, 0.0, 0.0],
        "per_agent_near_boundary_error_rate": [0.0] * 5,
        "per_agent_dominant_wrong_rate": [0.0] * 5,
        "per_agent_shared_error_rate": [0.0] * 5,
        "per_agent_general_error_rate": [0.90, 0.20, 0.0, 0.0, 0.0],
        "per_agent_invalid_rate": [0.0] * 5,
        "per_agent_capability_pressure": [{} for _ in range(5)],
        "capability_coverage_gap": {},
    }
    assert system.select_reward_agents_for_update(diagnosis, {})[0] == 1


def test_boundary_case_order_places_pivotal_before_peer_correct():
    system = _system(boundary_selector_enabled=True)
    diagnosis = {
        "target_error_cases": [
            {"target_agent_id": 0, "case_type": "target_wrong_peer_correct_nonboundary", "window_index": 0},
            {"target_agent_id": 0, "case_type": "target_wrong_pivotal_vote_fix", "window_index": 1},
        ]
    }
    cases = system._target_error_cases_for_agent(diagnosis, 0)
    assert [case["case_type"] for case in cases] == [
        "target_wrong_pivotal_vote_fix",
        "target_wrong_peer_correct_nonboundary",
    ]


def test_v7_teacher_and_critic_prompts_use_shared_error_and_pivotal_language():
    system = _system(boundary_selector_enabled=True)
    captured = []

    async def fake_chat(**kwargs):
        captured.append(kwargs["system_prompt"])
        if kwargs["stage"].startswith("teacher_critic"):
            return '{"passed":true,"score":1.0}'
        return '{"socratic_guiding_question":"Which local mechanism should change?"}'

    system._chat = fake_chat
    async def run():
        await system.propose_teacher_question(0, "parent", {}, 1)
        await system.critique_teacher_question(
            0, {"socratic_guiding_question": "Which local mechanism should change?"}, {}
        )

    asyncio.run(run())
    joined = "\n".join(captured)
    assert "Do not optimize for voting failure in this step." not in joined
    assert "focused on voting failure" not in joined
    assert "shared-error mechanism" in joined
    assert "pivotal correction" in joined
    assert "preserve pivotal-correct behavior" in joined or "preserves pivotal-correct behavior" in joined


def test_v7_reward_schedule_does_not_depend_on_prompt_uniqueness():
    first = _system(boundary_selector_enabled=True, reward_schedule_mode="phase_adaptive")
    second = _system(boundary_selector_enabled=True, reward_schedule_mode="phase_adaptive")
    for idx, agent in enumerate(second.agents):
        agent.current_prompt = f"unique {idx}"
    first_weights = first._effective_reward_weights()
    second_weights = second._effective_reward_weights()
    for key in (
        "target_accuracy", "div_delta", "vote_delta", "vote_margin",
        "boundary_diversity", "invalid_delta", "accuracy_guard_epsilon",
        "phase_progress", "diversity_need",
    ):
        assert first_weights[key] == second_weights[key]


def test_capability_profiles_start_equal_and_diverge_only_after_accepted_evidence():
    system = _system(residual_specialization_enabled=True, specialization_update_period=2)
    assert system.agents[0].capability_profile == system.agents[1].capability_profile
    metrics_a = {
        "capability_transition_support": {CapabilityResidualFamily.RELATION_TRACKING.value: 1},
        "capability_weighted_gain": {CapabilityResidualFamily.RELATION_TRACKING.value: 4.0},
        "capability_weighted_loss": {},
    }
    metrics_b = {
        "capability_transition_support": {CapabilityResidualFamily.OPTION_COMPARISON.value: 1},
        "capability_weighted_gain": {CapabilityResidualFamily.OPTION_COMPARISON.value: 3.0},
        "capability_weighted_loss": {},
    }
    system._accumulate_capability_evidence(system.agents[0], metrics_a, 1)
    system._accumulate_capability_evidence(system.agents[1], metrics_b, 1)
    assert not system._flush_capability_profile(system.agents[0], 1, force=False)
    system._flush_capability_profile(system.agents[0], 1, force=True)
    system._flush_capability_profile(system.agents[1], 1, force=True)
    assert system.agents[0].capability_profile != system.agents[1].capability_profile
    unchanged = dict(system.agents[2].capability_profile)
    assert not system._flush_capability_profile(system.agents[2], 1, force=True)
    assert system.agents[2].capability_profile == unchanged


def test_capability_evidence_and_pending_slow_state_round_trip():
    source = AgentState("prompt")
    family = CapabilityResidualFamily.ENTITY_BINDING.value
    source.capability_profile[family] = 1.0
    source.capability_evidence[family].support = 1
    source.capability_evidence[family].weighted_gain = 4.0
    source.capability_evidence[family].posterior_value = 1.0
    source.pending_capability_evidence = [{"epoch": 2, "support": {family: 1}}]
    source.pending_capability_update_count = 1
    restored = AgentState("prompt")
    restored.restore_trajectory_state(source.trajectory_state_dict())
    assert restored.capability_profile == source.capability_profile
    assert restored.capability_evidence[family].support == 1
    assert restored.pending_capability_evidence == source.pending_capability_evidence


def _training_checkpoint(system):
    return build_training_checkpoint(
        system.cfg,
        system,
        epoch_index=0,
        cursor=0,
        order=[0],
        train_accumulators={},
        best_score=0.0,
        best_epoch=0,
        epochs_without_improvement=0,
        stopped_early=False,
        no_effective_evolution_counter=0,
        no_effective_evolution_stopped=False,
        no_effective_evolution_reason="",
    )


def test_v7_checkpoint_round_trip_and_incompatibility_guard(tmp_path):
    source = _system(
        out_dir=str(tmp_path / "source"),
        agents=1,
        train_size=1,
        epochs=1,
        residual_specialization_enabled=True,
    )
    family = CapabilityResidualFamily.RELATION_TRACKING.value
    source.agents[0].capability_profile[family] = 1.0
    source.agents[0].capability_evidence[family].support = 2
    source.agents[0].capability_profile_update_count = 1
    payload = _training_checkpoint(source)

    resumed = _system(
        out_dir=str(tmp_path / "resumed"),
        agents=1,
        train_size=1,
        epochs=1,
        residual_specialization_enabled=True,
    )
    restore_system_state(resumed, payload["state"])
    assert resumed.agents[0].capability_profile == source.agents[0].capability_profile
    assert resumed.agents[0].capability_evidence[family].support == 2

    changed = Config(
        out_dir=str(tmp_path),
        agents=1,
        train_size=1,
        epochs=1,
        residual_specialization_enabled=True,
        behavior_archive_size=8,
    )
    assert any("behavior_archive_size" in reason for reason in checkpoint_incompatibility_reasons(payload, changed, [None]))
    payload["version"] = 3
    assert any("version" in reason for reason in checkpoint_incompatibility_reasons(payload, source.cfg, [None]))


def test_boundary_shared_error_metrics_reward_rescue_and_penalize_creation():
    system = _system(shared_error_metrics_enabled=True)
    rescue = system._candidate_boundary_error_metrics([{
        "baseline_target_correct": False,
        "candidate_target_correct": True,
        "baseline_vote_correct": False,
        "candidate_vote_correct": True,
        "peer_wrong_count": 4,
        "baseline_target_in_dominant_wrong_cluster": True,
        "candidate_target_in_dominant_wrong_cluster": False,
    }])
    creation = system._candidate_boundary_error_metrics([{
        "baseline_target_correct": True,
        "candidate_target_correct": False,
        "baseline_vote_correct": True,
        "candidate_vote_correct": False,
        "peer_wrong_count": 4,
        "baseline_target_in_dominant_wrong_cluster": False,
        "candidate_target_in_dominant_wrong_cluster": True,
    }])
    assert rescue["boundary_shared_error_net_gain"] > 0.0
    assert creation["boundary_shared_error_net_gain"] < 0.0
    assert rescue["pivotal_rescue_rate"] == 1.0
    assert creation["pivotal_loss_rate"] == 1.0


def test_v7_candidate_log_fields_preserve_selection_evidence():
    metrics = {
        "pivotal_rescue_count": 2,
        "pivotal_rescue_rate": 0.5,
        "shared_error_rescue_score": 0.25,
        "boundary_shared_error_net_gain": 1.25,
        "error_dependence_guard_passed": False,
        "paired_boundary_transition_rows": [{"question_hash": "q1"}],
        "capability_transition_support": {CapabilityResidualFamily.ENTITY_BINDING.value: 1},
        "capability_shrunk_transition": {CapabilityResidualFamily.ENTITY_BINDING.value: 0.4},
        "capability_evidence_rows": [{"question_hash": "q1", "support_reliability": 0.25}],
    }

    fields = TraceBeamSearchSystem._candidate_v7_log_fields(metrics)

    assert fields["pivotal_rescue_count"] == 2
    assert fields["boundary_shared_error_net_gain"] == pytest.approx(1.25)
    assert fields["error_dependence_guard_passed"] is False
    assert fields["paired_boundary_transition_rows"] == [{"question_hash": "q1"}]
    assert fields["capability_transition_support"][CapabilityResidualFamily.ENTITY_BINDING.value] == 1
    assert fields["capability_evidence_rows"][0]["support_reliability"] == pytest.approx(0.25)


def test_vote_error_pareto_adds_boundary_objective_without_changing_legacy_dominance():
    a = {"vote_gain_rate": 0.2, "vote_loss_rate": 0.0, "candidate_target_accuracy": 0.8, "boundary_shared_error_net_gain": 0.4}
    b = {"vote_gain_rate": 0.2, "vote_loss_rate": 0.0, "candidate_target_accuracy": 0.8, "boundary_shared_error_net_gain": 0.0}
    assert not pareto_dominates(a, b)
    assert error_pareto_dominates(a, b)


def _fingerprint(target_correct=False, team_correct=False, margin=0):
    return {
        "q": BehaviorFingerprintEntry(
            target_correct=target_correct,
            target_answer_hash="same",
            team_vote_correct=team_correct,
            vote_margin_bucket=margin,
            behavior_context=BehaviorContext.TEAM_WRONG_NONPIVOTAL.value,
        )
    }


def test_residual_cycle_guard_checks_rejected_failures_and_allows_higher_utility():
    system = _system(
        candidate_selection_mode="vote_error_pareto",
        residual_cycle_guard_enabled=True,
        behavior_cycle_min_overlap=1,
        behavior_cycle_similarity_threshold=0.9,
        mechanism_trust_region_enabled=False,
        reward_schedule_mode="static",
    )
    rejected_fp = _fingerprint(False, False, -2)
    system.agents[0].rejected_behavior_archive.append(RejectedBehaviorSummary(
        state_id="rejected",
        epoch=0,
        prompt_hash="old",
        parent_prompt_hash="parent",
        rejection_reason="not_selected",
        prompt_change_ratio=0.1,
        max_behavior_cycle_similarity=1.0,
        behavior_cycle_overlap=1,
        transition_vector={},
        behavior_fingerprint=rejected_fp,
        paired_behavior_utility=system.behavior_fingerprint_utility(rejected_fp),
        failure_signature="not_selected",
    ))
    base_metrics = {
        "baseline_target_accuracy": 0.5,
        "candidate_target_accuracy": 0.5,
        "baseline_invalid_rate": 0.0,
        "candidate_invalid_rate": 0.0,
        "behavior_fingerprint": {"q": vars(rejected_fp["q"])},
    }
    repeated = {"prompt": "new text", "parent_prompt": "parent", "source": "optimizer", "metrics": dict(base_metrics)}
    assert system._candidate_trajectory_feasibility(system.agents[0], repeated)["rejection_reason"] == "rejected_failure_cycle"

    improved_fp = _fingerprint(True, True, 2)
    improved = {
        "prompt": "better text",
        "parent_prompt": "parent",
        "source": "optimizer",
        "metrics": {**base_metrics, "behavior_fingerprint": {"q": vars(improved_fp["q"])}},
    }
    assert system._candidate_trajectory_feasibility(system.agents[0], improved)["rejection_reason"] == ""


def test_mechanism_contract_is_required_from_first_v7_update():
    system = _system(
        candidate_selection_mode="vote_error_pareto",
        mechanism_trust_region_enabled=True,
        reward_schedule_mode="static",
    )
    item = {
        "prompt": "new",
        "parent_prompt": "parent",
        "source": "optimizer",
        "proposal": {"preserved_mechanisms": [], "modified_mechanism": "check", "change_summary": "change"},
        "metrics": {
            "baseline_target_accuracy": 0.5,
            "candidate_target_accuracy": 0.5,
            "baseline_invalid_rate": 0.0,
            "candidate_invalid_rate": 0.0,
        },
    }
    result = system._candidate_trajectory_feasibility(system.agents[0], item)
    assert result["rejection_reason"] == "mechanism_contract_missing"


def test_v7_template_fallback_uses_mechanism_not_persona_role():
    system = _system(boundary_selector_enabled=True)
    fallback = system._structured_fallback_role(0, 0)
    assert "mechanism_name" in fallback
    assert "role_name" not in fallback
    assert not fallback["candidate_prompt"].lower().startswith("you are")


def test_high_order_error_dependence_metrics_are_exported():
    system = _system()
    rows = [
        {"individual_correct": [1, 1, 0, 0, 0], "vote_correct": 0, "vote_counts": {"A": 2, "B": 3}, "gold_vote_count": 2, "largest_wrong_vote_count": 3, "normalized_vote_margin": -0.2},
        {"individual_correct": [1, 1, 1, 1, 1], "vote_correct": 1, "vote_counts": {"A": 5}, "gold_vote_count": 5, "largest_wrong_vote_count": 0, "normalized_vote_margin": 1.0},
    ]
    summary = system._summarize_rollout_rows(rows)
    for key in (
        "mean_pairwise_double_fault", "mean_pairwise_error_covariance", "same_wrong_pair_rate",
        "triple_joint_error_rate", "majority_failure_tail_rate", "coverage_depth_c1",
        "coverage_depth_c5", "mean_boundary_conditional_error", "dominant_wrong_cluster_size",
        "gold_vs_largest_wrong_margin",
    ):
        assert key in summary
    assert summary["triple_joint_error_rate"] == 0.5
