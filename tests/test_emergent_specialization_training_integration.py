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
