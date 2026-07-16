import asyncio

from multi_dataset_diverse_rl.cli import build_training_checkpoint, restore_system_state
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import (
    BEHAVIOR_CONTEXT_NAMES,
    AgentState,
    BehaviorFingerprintEntry,
    BehaviorStateSummary,
)
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.tasks import get_task_spec


def _fingerprint(answer: str, count: int = 4):
    return {
        f"q{index}": {
            "target_correct": True,
            "target_answer_hash": answer,
            "team_vote_correct": True,
            "vote_margin_bucket": 1,
            "behavior_context": "target_correct_robust",
        }
        for index in range(count)
    }


def _system(tmp_path):
    cfg = Config(
        out_dir=str(tmp_path),
        agents=1,
        epochs=2,
        train_size=2,
        optimizer_architecture="one_shot",
        optimizer_fallback_mode="none",
        reward_mode="vote_useful_diversity",
        candidate_selection_mode="vote_pareto",
        candidate_eval_execution_mode="legacy",
        candidate_eval_concurrency=1,
        optimizer_parent_concurrency=1,
        beam_size=1,
        num_candidates_per_parent=2,
        emergent_specialization_enabled=True,
        behavior_cycle_min_overlap=4,
        behavior_cycle_similarity_threshold=0.95,
        prompt_large_shift_warmup_accepts=99,
        specialization_min_context_support=2,
    )
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.execution_session_id = "integration-session"
    system.task_spec = get_task_spec("mmlu")
    system.agents = [AgentState("current prompt")]
    system.agents[0].prompt_beam = [system._make_beam_item("current prompt", None, {}, None, 0)]
    system.update_logs = []
    system.trajectory_events = []
    system.recent_window_records = []
    system.optimizer_generation_diagnostics = {}
    system.no_effective_evolution_counter = 0
    system.no_effective_evolution_stopped = False
    system.no_effective_evolution_reason = ""
    system.joint_diversity_cache = {}
    system.solver_rollout_cache = {}
    system._append_prompt_history_event = lambda *args, **kwargs: None

    async def fake_prewarm(**kwargs):
        return {"solver_reuse_hits": 0, "solver_calls": 0, "solver_reuse_hit_rate": 0.0}

    system.ensure_recorded_rollouts_for_prompts = fake_prewarm
    return system


def _install_update_fakes(system, *, accepted_prompt, accepted_answer, cycle_prompt, context_index):
    context = BEHAVIOR_CONTEXT_NAMES[context_index]

    async def fake_propose(**kwargs):
        diagnostics = {
            "optimizer_architecture": "one_shot",
            "optimizer_raw_candidate_count": 2,
            "optimizer_final_candidate_count": 2,
            "optimizer_underfilled": False,
        }
        return [
            {
                "candidate_prompt": accepted_prompt,
                "candidate_source": "optimizer",
                "optimizer_generation_diagnostics": diagnostics,
            },
            {
                "candidate_prompt": cycle_prompt,
                "candidate_source": "optimizer",
                "optimizer_generation_diagnostics": diagnostics,
            },
        ]

    async def fake_evaluate(candidate_prompt, **kwargs):
        is_accepted = candidate_prompt == accepted_prompt
        is_cycle = candidate_prompt == cycle_prompt
        target_accuracy = 0.7 if is_accepted else 0.5
        return {
            "reward": 1.0 if is_cycle else (0.8 if is_accepted else 0.1),
            "baseline_target_accuracy": 0.5,
            "candidate_target_accuracy": target_accuracy,
            "target_agent_accuracy": target_accuracy,
            "accuracy_delta": target_accuracy - 0.5,
            "baseline_invalid_rate": 0.0,
            "candidate_invalid_rate": 0.0,
            "invalid_rate": 0.0,
            "vote_gain_rate": 0.2 if is_accepted else 0.0,
            "vote_loss_rate": 0.0,
            "vote_delta": 0.2 if is_accepted else 0.0,
            "vote_margin_delta": 0.2 if is_accepted else 0.0,
            "boundary_useful_diversity_delta": 0.0,
            "candidate_team_accuracy": 0.7 if is_accepted else 0.5,
            "candidate_mean_vote_margin": 0.2 if is_accepted else 0.0,
            "behavior_fingerprint": _fingerprint("archive-answer" if is_cycle else accepted_answer),
            "candidate_transition_vector": {context: 1.0} if is_accepted else {},
            "candidate_transition_support": {context: 2} if is_accepted else {},
            "trajectory_alignment": 0.0,
            "num_eval_samples": 1,
        }

    system.propose_candidates = fake_propose
    system.evaluate_candidate_prompt = fake_evaluate


def _checkpoint(system):
    return build_training_checkpoint(
        system.cfg,
        system,
        epoch_index=0,
        cursor=1,
        order=[0, 1],
        train_accumulators={},
        best_score=0.0,
        best_epoch=0,
        epochs_without_improvement=0,
        stopped_early=False,
        no_effective_evolution_counter=0,
        no_effective_evolution_stopped=False,
        no_effective_evolution_reason="",
    )


def _install_single_candidate_fakes(system, candidate_prompt, metrics_by_prompt):
    async def fake_propose(**kwargs):
        return [
            {
                "candidate_prompt": candidate_prompt,
                "candidate_source": "optimizer",
                "optimizer_generation_diagnostics": {
                    "optimizer_architecture": "one_shot",
                    "optimizer_raw_candidate_count": 1,
                    "optimizer_final_candidate_count": 1,
                    "optimizer_underfilled": False,
                },
            }
        ]

    async def fake_evaluate(candidate_prompt, **kwargs):
        return dict(metrics_by_prompt[candidate_prompt])

    system.propose_candidates = fake_propose
    system.evaluate_candidate_prompt = fake_evaluate


def _candidate_metrics(
    *,
    target_accuracy=0.5,
    vote_gain=0.0,
    vote_delta=0.0,
    fingerprint_answer="new-answer",
    context_index=None,
):
    transition = {}
    support = {}
    if context_index is not None:
        context = BEHAVIOR_CONTEXT_NAMES[context_index]
        transition = {context: 1.0}
        support = {context: 2}
    return {
        "reward": target_accuracy + vote_delta,
        "baseline_target_accuracy": 0.5,
        "candidate_target_accuracy": target_accuracy,
        "target_agent_accuracy": target_accuracy,
        "accuracy_delta": target_accuracy - 0.5,
        "baseline_invalid_rate": 0.0,
        "candidate_invalid_rate": 0.0,
        "invalid_rate": 0.0,
        "vote_gain_rate": vote_gain,
        "vote_loss_rate": 0.0,
        "vote_delta": vote_delta,
        "vote_margin_delta": vote_delta,
        "boundary_useful_diversity_delta": 0.0,
        "candidate_team_accuracy": 0.5 + vote_delta,
        "candidate_mean_vote_margin": vote_delta,
        "behavior_fingerprint": _fingerprint(fingerprint_answer),
        "candidate_transition_vector": transition,
        "candidate_transition_support": support,
        "trajectory_alignment": 0.0,
        "num_eval_samples": 1,
    }


def test_emergent_specialization_full_update_and_checkpoint_continuation(tmp_path):
    system = _system(tmp_path / "source")
    archive_fingerprint = {
        key: BehaviorFingerprintEntry.from_dict(value)
        for key, value in _fingerprint("archive-answer").items()
    }
    system.agents[0].accepted_behavior_archive = [
        BehaviorStateSummary(
            "initial-state",
            0,
            "initial-hash",
            archive_fingerprint,
            {},
            dict(system.agents[0].specialization_profile),
            0.5,
            0.5,
            0.0,
            [],
        )
    ]
    _install_update_fakes(
        system,
        accepted_prompt="accepted specialized prompt one",
        accepted_answer="new-answer-one",
        cycle_prompt="textually new but behaviorally cyclic prompt one",
        context_index=0,
    )

    changed, summary = asyncio.run(
        system.update_prompt_with_beam(
            agent_id=0,
            overlap_diagnosis={"homogeneous_cases": []},
            eval_batch=[{"question": "q", "answer": "A"}],
            step_id=1,
            epoch_id=1,
        )
    )

    agent = system.agents[0]
    assert changed is True
    assert agent.current_prompt == "accepted specialized prompt one"
    assert agent.specialization_update_count == 1
    assert agent.specialization_profile[BEHAVIOR_CONTEXT_NAMES[0]] > agent.specialization_profile[BEHAVIOR_CONTEXT_NAMES[1]]
    assert len(agent.accepted_behavior_archive) == 2
    assert agent.cycle_reject_count == 1
    assert agent.rejected_behavior_archive[-1].rejection_reason == "behavior_cycle"
    assert summary["top1_candidate_pool_source"] == "optimizer"
    assert summary["top1_candidate_source"] == "optimizer"

    optimizer_rows = [row for row in system.update_logs if row.get("event") == "candidate_evaluated" and row.get("candidate_pool_source") == "optimizer"]
    cycle_row = next(row for row in optimizer_rows if row.get("rejection_reason") == "behavior_cycle")
    assert cycle_row["pareto_selected"] is False
    assert all(row["candidate_source"] == "optimizer" for row in optimizer_rows)
    assert {event["candidate_pool_source"] for event in system.trajectory_events} == {"optimizer"}
    assert {event["candidate_source"] for event in system.trajectory_events} == {"optimizer"}

    payload = _checkpoint(system)
    resumed = _system(tmp_path / "resumed")
    restore_system_state(resumed, payload["state"])
    assert resumed.agents[0].specialization_profile == agent.specialization_profile
    assert len(resumed.agents[0].accepted_behavior_archive) == 2
    assert resumed.agents[0].cycle_reject_count == 1

    _install_update_fakes(
        resumed,
        accepted_prompt="accepted specialized prompt two",
        accepted_answer="new-answer-two",
        cycle_prompt="textually new but behaviorally cyclic prompt two",
        context_index=1,
    )
    resumed_changed, _ = asyncio.run(
        resumed.update_prompt_with_beam(
            agent_id=0,
            overlap_diagnosis={"homogeneous_cases": []},
            eval_batch=[{"question": "q2", "answer": "B"}],
            step_id=2,
            epoch_id=1,
        )
    )

    assert resumed_changed is True
    assert resumed.agents[0].current_prompt == "accepted specialized prompt two"
    assert resumed.agents[0].specialization_update_count == 2
    assert len(resumed.agents[0].accepted_behavior_archive) == 3
    assert resumed.agents[0].cycle_reject_count == 2


def test_cycle_rejection_preserves_profile_and_current_active_fallback(tmp_path):
    system = _system(tmp_path)
    agent = system.agents[0]
    agent.prompt_beam = [system._make_beam_item("stale beam prompt", None, {}, None, 0)]
    initial_profile = dict(agent.specialization_profile)
    agent.accepted_behavior_archive = [
        BehaviorStateSummary(
            "initial-state",
            0,
            "initial-hash",
            {key: BehaviorFingerprintEntry.from_dict(value) for key, value in _fingerprint("archive-answer").items()},
            {},
            initial_profile,
            0.5,
            0.5,
            0.0,
            [],
        )
    ]
    cycle_prompt = "new text with archived behavior"
    _install_single_candidate_fakes(
        system,
        cycle_prompt,
        {
            cycle_prompt: _candidate_metrics(fingerprint_answer="archive-answer", context_index=0),
            "stale beam prompt": _candidate_metrics(target_accuracy=0.4, fingerprint_answer="stale"),
            "current prompt": _candidate_metrics(
                target_accuracy=0.6,
                vote_gain=0.1,
                vote_delta=0.1,
                fingerprint_answer="archive-answer",
            ),
        },
    )

    changed, summary = asyncio.run(
        system.update_prompt_with_beam(
            agent_id=0,
            overlap_diagnosis={"homogeneous_cases": []},
            eval_batch=[{"question": "q", "answer": "A"}],
            step_id=1,
            epoch_id=1,
        )
    )

    assert changed is False
    assert agent.current_prompt == "current prompt"
    assert summary["top1_candidate_pool_source"] == "current_active_fallback"
    assert agent.specialization_update_count == 0
    assert agent.specialization_profile == initial_profile
    assert len(agent.accepted_behavior_archive) == 1
    assert agent.cycle_reject_count == 1
    cycle_row = next(row for row in system.update_logs if row.get("prompt_preview") == cycle_prompt)
    assert cycle_row["pareto_feasible"] is False
    assert cycle_row["pareto_selected"] is False
    assert cycle_row["rejection_reason"] == "behavior_cycle"


def test_retained_inactive_candidate_does_not_update_profile_or_archive(tmp_path):
    system = _system(tmp_path)
    system.cfg.beam_size = 2
    system.cfg.num_candidates_per_parent = 1
    candidate_prompt = "feasible retained specialist prompt"
    initial_profile = dict(system.agents[0].specialization_profile)
    _install_single_candidate_fakes(
        system,
        candidate_prompt,
        {
            "current prompt": _candidate_metrics(target_accuracy=0.5, vote_gain=0.2, vote_delta=0.2),
            candidate_prompt: _candidate_metrics(target_accuracy=0.6, context_index=0),
        },
    )

    changed, _ = asyncio.run(
        system.update_prompt_with_beam(
            agent_id=0,
            overlap_diagnosis={"homogeneous_cases": []},
            eval_batch=[{"question": "q", "answer": "A"}],
            step_id=1,
            epoch_id=1,
        )
    )

    agent = system.agents[0]
    assert changed is False
    assert agent.specialization_update_count == 0
    assert agent.specialization_profile == initial_profile
    assert agent.accepted_behavior_archive == []
    assert agent.rejected_behavior_archive == []
    candidate_row = next(row for row in system.update_logs if row.get("prompt_preview") == candidate_prompt)
    assert candidate_row["pareto_selected"] is True
    assert candidate_row["in_top_beam"] is True
    assert candidate_row["is_top1"] is False
    event = next(row for row in system.trajectory_events if row.get("candidate_id") == candidate_row["candidate_id"])
    assert event["accepted"] is False
    assert event["decision"] == "retained_beam_inactive"


def test_emergent_disabled_preserves_legacy_selection_behavior(tmp_path):
    system = _system(tmp_path)
    system.cfg.emergent_specialization_enabled = False
    candidate_prompt = "historic candidate selected by legacy pareto"
    system.agents[0].history.append(candidate_prompt)
    _install_single_candidate_fakes(
        system,
        candidate_prompt,
        {
            "current prompt": _candidate_metrics(target_accuracy=0.5),
            candidate_prompt: _candidate_metrics(target_accuracy=0.8, vote_gain=0.2, vote_delta=0.2, context_index=0),
        },
    )

    changed, summary = asyncio.run(
        system.update_prompt_with_beam(
            agent_id=0,
            overlap_diagnosis={"homogeneous_cases": []},
            eval_batch=[{"question": "q", "answer": "A"}],
            step_id=1,
            epoch_id=1,
        )
    )

    assert changed is True
    assert system.agents[0].current_prompt == candidate_prompt
    assert summary["top1_candidate_pool_source"] == "optimizer"
    assert system.agents[0].specialization_update_count == 0
    assert system.agents[0].accepted_behavior_archive == []
    assert system.trajectory_events == []
