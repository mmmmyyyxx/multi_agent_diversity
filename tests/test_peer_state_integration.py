import asyncio
import json
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.system import (
    CandidateFunnel,
    PromptEnsembleOptimizationSystem,
)


QUESTIONS = {"q0": "A", "q1": "A", "q2": "A"}


async def fake_solver(question, agent_id, prompt):
    if "repair-q0" in prompt:
        answer = "A"
    elif question == "q1" and agent_id in {0, 1, 2}:
        answer = "A"
    elif question == "q2" and agent_id in {0, 1}:
        answer = "A"
    else:
        answer = "B"
    return PromptAnswer(answer, f"check FINAL_ANSWER: {answer}", True)


TEACHER = {
    "failure_pattern": "the solver commits before checking explicit constraints",
    "repair_rule": (
        "Check each explicit constraint before committing and abstain when the "
        "remaining evidence does not distinguish the viable options."
    ),
    "preservation_rule": "Keep conclusions that continue to pass every explicit check.",
}
APPROVED = {"failed_checks": [], "risk_case_ids": [], "feedback": ""}


async def fake_optimizer(system_prompt, _user_prompt, _temperature, _max_tokens):
    if "Check only explicit hard blockers" in system_prompt:
        return json.dumps(APPROVED)
    if system_prompt.startswith("Return strict JSON only."):
        return json.dumps({"candidate_prompts": ["repair-q0"]})
    return json.dumps(TEACHER)


def build_system(tmp_path, optimizer=fake_optimizer, **overrides):
    values = {
        "out_dir": str(tmp_path),
        "answer_format": "option_letter",
        "num_candidates_per_parent": 2,
        "stage_a_channel_top_k": 1,
        "stage_b_candidate_budget": 2,
        "experiment_setting": "shared_responsibility_conditioned_dual_target",
    }
    values.update(overrides)
    cfg = Config.from_flat(**values)
    return PromptEnsembleOptimizationSystem(
        cfg, solver=fake_solver, optimizer_chat=optimizer,
    )


async def initialize(system):
    await system.initialize_fixed_probe(
        [{"question": question, "answer": gold} for question, gold in QUESTIONS.items()]
    )


def routed_proposal(system):
    _, assignments = system.ensure_responsibility_current()
    target = next(
        agent_id for agent_id, rows in assignments.items() if rows
    )
    return target, {row.question_hash for row in assignments[target]}


def test_full_aggregated_chain_accepts_and_refreshes_once_per_transition(tmp_path):
    system = build_system(tmp_path)

    async def run():
        await initialize(system)
        system.ensure_responsibility_current()
        before = system.responsibility_refresh_count
        changed = await system.update_once(0)
        return before, changed

    before, changed = asyncio.run(run())
    assert changed
    assert system.responsibility_refresh_count == before + 1
    audit = system.tcs_context_history[-1]
    assert audit["context_type"] == "SingleLaneDiagnosisContext"
    assert audit["full_probe_case_count"] == 3
    assert audit["selected_pattern_count"] == 1
    assert audit["selected_case_count"] <= 3
    assert audit["forbidden_field_violations"] == []
    assert [row["role"] for row in system.tcs_rounds] == [
        "teacher", "critic", "student",
    ]
    assert system.candidate_decisions[-1]["funnel"]["accepted_candidate"]
    assert system.candidate_decisions[-1]["candidates"][0]["repair_plan_hash"]
    target = system.candidate_decisions[-1]["target_agent_id"]
    assert system.responsibility_state.specialization_anchor_by_agent[target]
    assert any(
        row["event"] == "accepted_anchor_set" and row["agent_id"] == target
        for row in system.specialization_anchor_trajectory
    )


def test_service_refresh_failure_rolls_back_anchor_routing_and_team_state(tmp_path):
    system = build_system(tmp_path)

    async def run():
        await initialize(system)
        system.ensure_responsibility_current()
        before = {
            "prompts": [agent.current_prompt for agent in system.agents],
            "profiles": list(system.active_profiles),
            "anchors": dict(
                system.responsibility_state.specialization_anchor_by_agent
            ),
            "routing": dict(system.cached_service_assignments),
            "active": dict(system.cached_active_lane_by_agent),
            "team_version": system.team_state_version,
            "responsibility_version": system.responsibility_state_version,
            "routing_audit": len(system.service_routing_audit),
            "anchor_audit": len(system.specialization_anchor_trajectory),
        }

        def fail_refresh(*, update_index):
            system.team_state_version += 1
            system.cached_service_assignments = {}
            system.specialization_anchor_trajectory.append({"event": "bad"})
            raise RuntimeError("offline routing refresh failure")

        system.refresh_responsibility_after_commit = fail_refresh
        with pytest.raises(RuntimeError, match="routing refresh failure"):
            await system.update_once(0)
        return before

    before = asyncio.run(run())
    assert [agent.current_prompt for agent in system.agents] == before["prompts"]
    assert system.active_profiles == before["profiles"]
    assert system.responsibility_state.specialization_anchor_by_agent == before["anchors"]
    assert system.cached_service_assignments == before["routing"]
    assert system.cached_active_lane_by_agent == before["active"]
    assert system.team_state_version == before["team_version"]
    assert system.responsibility_state_version == before["responsibility_version"]
    assert len(system.service_routing_audit) == before["routing_audit"]
    assert len(system.specialization_anchor_trajectory) == before["anchor_audit"]


def test_generic_context_isolation_for_accuracy_and_peer_state(tmp_path):
    async def inspect(setting):
        system = build_system(
            tmp_path / setting,
            experiment_setting=setting,
        )
        await initialize(system)
        if setting == "shared_member_aware_dual_target":
            _, assignments = system.ensure_responsibility_current()
            target, _ = system.select_target(assignments, 0)
            assert target is not None
            hashes = {row.question_hash for row in assignments[target]}
        else:
            target, hashes = 0, set()
        await system.propose_candidates(target, hashes, CandidateFunnel())
        return system.tcs_context_history[-1]

    async def run_all():
        return await asyncio.gather(
            inspect("shared_generic_evolution"),
            inspect("shared_member_aware_dual_target"),
        )

    accuracy, peer = asyncio.run(run_all())
    assert accuracy["context_type"] == "AccuracyDiagnosisContext"
    assert peer["context_type"] == "PeerStateDiagnosisContext"
    assert accuracy["forbidden_field_violations"] == []
    assert peer["forbidden_field_violations"] == []
    assert not any(
        "assigned" in path or "member_gain" in path
        for path in peer["serialized_recursive_field_paths"]
    )


def test_s1_and_s2_share_routing_scheduler_but_isolate_context(tmp_path):
    async def inspect(setting):
        system = build_system(
            tmp_path / setting,
            experiment_setting=setting,
        )
        await initialize(system)
        _, assignments = system.ensure_responsibility_current()
        target, _ = system.select_target(assignments, 0)
        assert target is not None
        hashes = {row.question_hash for row in assignments[target]}
        await system.propose_candidates(target, hashes, CandidateFunnel())
        return system, target, system.tcs_context_history[-1]

    async def run():
        return await asyncio.gather(
            inspect("shared_member_aware_dual_target"),
            inspect("shared_responsibility_conditioned_dual_target"),
        )

    (s4, s4_target, s4_audit), (s5, s5_target, s5_audit) = asyncio.run(run())
    assert s4.cached_service_assignments == s5.cached_service_assignments
    assert s4.cached_active_lane_by_agent == s5.cached_active_lane_by_agent
    assert s4_target == s5_target
    assert s4_audit["context_type"] == "PeerStateDiagnosisContext"
    assert s5_audit["context_type"] == "SingleLaneDiagnosisContext"


def test_s6_shares_s5_target_slice_context_and_candidate_generation(tmp_path):
    async def inspect(setting):
        system = build_system(
            tmp_path / setting,
            experiment_setting=setting,
        )
        await initialize(system)
        _, assignments = system.ensure_responsibility_current()
        target, _ = system.select_target(assignments, 0)
        assert target is not None
        hashes = {row.question_hash for row in assignments[target]}
        candidates = await system.propose_candidates(
            target, hashes, CandidateFunnel(), update_index=0
        )
        return system, target, hashes, candidates

    async def run():
        return await asyncio.gather(
            inspect("shared_responsibility_conditioned_dual_target"),
            inspect("shared_full_dual_target_rcru"),
        )

    (s5, target5, hashes5, candidates5), (
        s6, target6, hashes6, candidates6,
    ) = asyncio.run(run())
    assert target5 == target6
    assert hashes5 == hashes6
    assert s5.cached_service_assignments == s6.cached_service_assignments
    assert s5.cached_active_lane_by_agent == s6.cached_active_lane_by_agent
    assert s5.tcs_context_history[-1]["proposal_context_hash"] == (
        s6.tcs_context_history[-1]["proposal_context_hash"]
    )
    assert [row.prompt_hash for row in candidates5] == [
        row.prompt_hash for row in candidates6
    ]
    assert s5.protocol.candidate_acceptance_policy == (
        "fixed_peer_monotone_target_or_vote"
    )
    assert s6.protocol.candidate_acceptance_policy == (
        "responsibility_robust_contribution"
    )


def test_s6_writes_hash_only_rcru_candidate_audit(tmp_path):
    system = build_system(
        tmp_path,
        experiment_setting="shared_full_dual_target_rcru",
    )

    async def run():
        await initialize(system)
        return await system.update_once(0)

    asyncio.run(run())
    assert system.rcru_candidate_decisions
    forbidden = {
        "question",
        "gold",
        "answer",
        "prompt",
        "raw",
    }
    for row in system.rcru_candidate_decisions:
        assert row["artifact_schema_version"] == "rcru_candidate_decision_v2"
        assert "candidate_prompt_hash" in row
        assert "positive_support_guard_required" in row
        assert "positive_support_guard_passed" in row
        assert "no_negative_support_guard_required" in row
        assert "no_negative_support_guard_passed" in row
        assert "bootstrap_guard_required" in row
        assert "bootstrap_guard_passed" in row
        assert not any(
            token in key.lower() and key != "candidate_prompt_hash"
            for key in row
            for token in forbidden
        )


@pytest.mark.parametrize(
    "setting",
    (
        "shared_generic_evolution",
    ),
)
def test_s0_creates_no_service_anchor_or_freeze_state(tmp_path, setting):
    system = build_system(tmp_path / setting, experiment_setting=setting)
    asyncio.run(initialize(system))
    target, _ = system.select_target({agent: [] for agent in range(5)}, 0)
    assert target == 0
    assert system.cached_service_assignments == {}
    assert system.cached_service_portfolios == {}
    assert system.cached_active_lane_by_agent == {}
    assert system.responsibility_state.specialization_anchor_by_agent == {}
    assert system.responsibility_state.frozen_by_agent == {}


def test_only_valid_critic_rejection_consumes_semantic_revision(tmp_path):
    teacher_calls = critic_calls = 0
    teacher_system_requests = []
    teacher_user_requests = []

    async def optimizer(system_prompt, user_prompt, _temperature, _max_tokens):
        nonlocal teacher_calls, critic_calls
        if "Check only explicit hard blockers" in system_prompt:
            critic_calls += 1
            if critic_calls == 1:
                return json.dumps({
                    "failed_checks": ["actionable_specificity"],
                    "risk_case_ids": [],
                    "feedback": "Specify the executable verification order.",
                })
            return json.dumps(APPROVED)
        if system_prompt.startswith("Return strict JSON only."):
            return json.dumps({"candidate_prompts": ["repair-q0"]})
        teacher_calls += 1
        teacher_system_requests.append(system_prompt)
        teacher_user_requests.append(user_prompt)
        if teacher_calls == 2:
            assert "PreviousTeacherRepairPlan:" in user_prompt
            assert '"failure_pattern": "the solver commits before checking explicit constraints"' in user_prompt
            assert '"failed_checks": ["actionable_specificity"]' in user_prompt
            assert '"risk_case_ids": []' in user_prompt
            assert "Specify the executable verification order." in user_prompt
            return json.dumps({
                **TEACHER,
                "repair_rule": (
                    "Check each explicit constraint in order, record the first "
                    "observable conflict, and abstain only when viable options remain tied."
                ),
            })
        return json.dumps(TEACHER)

    system = build_system(tmp_path, optimizer)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        target, hashes = routed_proposal(system)
        candidates = await system.propose_candidates(target, hashes, funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert len(candidates) == 1
    assert teacher_calls == 2 and critic_calls == 2
    assert funnel.critic_semantic_rejections == 1
    assert teacher_system_requests[0] == teacher_system_requests[1]
    assert "DiagnosisContext:" in teacher_system_requests[1]
    assert "DiagnosisContextHash:" in teacher_user_requests[1]
    teacher_rows = [row for row in system.tcs_rounds if row["role"] == "teacher"]
    critic_rows = [row for row in system.tcs_rounds if row["role"] == "critic"]
    assert teacher_rows[0]["previous_plan_hash"] == ""
    assert teacher_rows[1]["previous_plan_hash"] == teacher_rows[0]["teacher_plan_hash"]
    assert teacher_rows[1]["revision_critic_hash"] == critic_rows[0]["critic_decision_hash"]
    assert teacher_rows[1]["revision_changed_fields"] == ["repair_rule"]


def test_teacher_revision_preserves_cumulative_hard_check_constraints(tmp_path):
    teacher_calls = critic_calls = 0

    async def optimizer(system_prompt, user_prompt, _temperature, _max_tokens):
        nonlocal teacher_calls, critic_calls
        if "Check only explicit hard blockers" in system_prompt:
            critic_calls += 1
            if critic_calls == 1:
                return json.dumps({
                    "failed_checks": ["evidence_mismatch"],
                    "risk_case_ids": [],
                    "feedback": "Use only checks observable in the task input.",
                })
            return json.dumps({
                "failed_checks": ["actionable_specificity"],
                "risk_case_ids": [],
                "feedback": "The revised rule is still vague.",
            })
        if system_prompt.startswith("Return strict JSON only."):
            return json.dumps({"candidate_prompts": ["repair-q0"]})
        teacher_calls += 1
        if teacher_calls == 2:
            assert '"failed_checks": ["evidence_mismatch"]' in user_prompt
            assert '"risk_case_ids": []' in user_prompt
            assert "all four hard checks cumulatively" in user_prompt
            return json.dumps({
                **TEACHER,
                "repair_rule": "Use contextual evidence to choose the best option.",
            })
        return json.dumps(TEACHER)

    system = build_system(tmp_path, optimizer)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        target, hashes = routed_proposal(system)
        candidates = await system.propose_candidates(target, hashes, funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    assert teacher_calls == 2 and critic_calls == 2
    assert funnel.critic_semantic_rejections == 2
    assert funnel.terminal_failure_class == "critic_semantic_rejection_exhausted"


def test_critic_invalid_json_retries_same_request_without_teacher_revision(tmp_path):
    calls = []

    async def optimizer(system_prompt, user_prompt, _temperature, _max_tokens):
        calls.append((system_prompt, user_prompt))
        if "Check only explicit hard blockers" in system_prompt:
            return "{"
        return json.dumps(TEACHER)

    system = build_system(tmp_path, optimizer)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        target, hashes = routed_proposal(system)
        candidates = await system.propose_candidates(target, hashes, funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    teacher_requests = [row for row in calls if "Propose one task-general" in row[0]]
    critic_requests = [row for row in calls if "Check only explicit hard blockers" in row[0]]
    assert len(teacher_requests) == 1
    assert len(critic_requests) == 2
    assert critic_requests[0] == critic_requests[1]
    assert funnel.critic_invalid_responses == 2


def result(
    text: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 1,
) -> LLMCallResult:
    return LLMCallResult(
        text=text,
        prompt_tokens=1,
        completion_tokens=completion_tokens,
        total_tokens=completion_tokens + 1,
        latency_seconds=0.0,
        finish_reason=finish_reason,
    )


def test_teacher_truncation_retries_identical_request_without_semantic_round_use(tmp_path):
    system = build_system(tmp_path)
    captured = []

    async def chat(_model, system_prompt, user_prompt, _temperature, max_tokens, role):
        captured.append((role, system_prompt, user_prompt, max_tokens))
        if len(captured) == 1:
            return result("{", finish_reason="length")
        if "Check only explicit hard blockers" in system_prompt:
            return result(json.dumps(APPROVED))
        if system_prompt.startswith("Return strict JSON only."):
            return result(json.dumps({"candidate_prompts": ["repair-q0"]}))
        return result(json.dumps(TEACHER))

    system._chat = chat

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        target, hashes = routed_proposal(system)
        candidates = await system.propose_candidates(target, hashes, funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert len(candidates) == 1
    assert captured[0][1:3] == captured[1][1:3]
    assert funnel.teacher_truncated_responses == 1
    teacher_rows = [row for row in system.tcs_rounds if row["role"] == "teacher"]
    assert [row["semantic_round"] for row in teacher_rows] == [1, 1]


def test_critic_truncation_never_triggers_teacher_revision(tmp_path):
    system = build_system(tmp_path)
    role_calls = []

    async def chat(_model, system_prompt, _user_prompt, _temperature, max_tokens, _role):
        if "Check only explicit hard blockers" in system_prompt:
            role_calls.append("critic")
            return result("{", finish_reason="length")
        role_calls.append("teacher")
        return result(json.dumps(TEACHER))

    system._chat = chat

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        target, hashes = routed_proposal(system)
        candidates = await system.propose_candidates(target, hashes, funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    assert role_calls == ["teacher", "critic", "critic"]
    assert funnel.critic_truncated_responses == 2
    assert funnel.terminal_failure_class == "critic_provider_truncation"
    assert funnel.terminal_failure_role == "critic"
    assert funnel.infrastructure_failed_updates == 1


@pytest.mark.parametrize(
    ("scenario", "expected_class", "expected_role", "infrastructure"),
    [
        ("teacher_schema", "teacher_schema_exhausted", "teacher", 0),
        ("teacher_truncation", "teacher_provider_truncation", "teacher", 1),
        ("critic_schema", "critic_schema_exhausted", "critic", 0),
        (
            "critic_rejection",
            "critic_semantic_rejection_exhausted",
            "critic",
            0,
        ),
        (
            "student_schema",
            "student_invalid_exhausted_after_upstream_regeneration",
            "student",
            0,
        ),
        (
            "student_truncation",
            "student_invalid_exhausted_after_upstream_regeneration",
            "student",
            0,
        ),
        (
            "student_zero",
            "student_invalid_exhausted_after_upstream_regeneration",
            "student",
            0,
        ),
        ("transport", "transport_failure", "teacher", 1),
    ],
)
def test_terminal_failure_taxonomy(
    tmp_path, scenario, expected_class, expected_role, infrastructure
):
    system = build_system(tmp_path)

    async def chat(_model, system_prompt, _user_prompt, _temperature, _max_tokens, _role):
        is_critic = "Check only explicit hard blockers" in system_prompt
        is_student = system_prompt.startswith("Return strict JSON only.")
        if scenario == "transport" and not is_critic and not is_student:
            raise ConnectionError("offline transport fault")
        if scenario.startswith("teacher_") and not is_critic and not is_student:
            return result(
                "{",
                finish_reason=(
                    "length" if scenario == "teacher_truncation" else "stop"
                ),
            )
        if is_critic:
            if scenario == "critic_schema":
                return result("{")
            if scenario == "critic_rejection":
                return result(json.dumps({
                    "failed_checks": ["actionable_specificity"],
                    "risk_case_ids": [],
                    "feedback": "Specify the executable verification order.",
                }))
            return result(json.dumps(APPROVED))
        if is_student:
            if scenario == "student_schema":
                return result(json.dumps({"candidate_prompts": "invalid"}))
            if scenario == "student_truncation":
                return result("{", finish_reason="length")
            if scenario == "student_zero":
                return result(json.dumps({"candidate_prompts": []}))
            return result(json.dumps({"candidate_prompts": ["repair-q0"]}))
        return result(json.dumps(TEACHER))

    system._chat = chat

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        target, hashes = routed_proposal(system)
        candidates = await system.propose_candidates(target, hashes, funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    assert funnel.terminal_failure_class == expected_class
    assert funnel.terminal_failure_role == expected_role
    assert funnel.infrastructure_failed_updates == infrastructure


def test_pipeline_failure_does_not_masquerade_as_rollout_rejection(tmp_path):
    system = build_system(tmp_path)

    async def no_candidates(
        _target_agent_id, _assigned_hashes, funnel, update_index=-1
    ):
        funnel.parents_considered = 1
        funnel.terminal_failure_class = "transport_failure"
        funnel.terminal_failure_role = "teacher"
        funnel.infrastructure_failed_updates = 1
        return []

    system.propose_candidates = no_candidates

    async def run():
        await initialize(system)
        changed = await system.update_once(0)
        target = system.candidate_decisions[-1]["target_agent_id"]
        return changed, system.previous_update_outcomes[target]

    changed, outcome = asyncio.run(run())
    assert changed is False
    assert outcome.attempted is True
    assert outcome.empirical_evaluation_completed is False
    assert outcome.accepted is False
    assert outcome.rejection_reasons == ()


def test_pipeline_failure_has_no_scheduler_search_state(tmp_path):
    system = build_system(tmp_path)
    async def run():
        await initialize(system)
        system.ensure_responsibility_current()
        async def no_candidates(*_args, **_kwargs):
            return []
        system.propose_candidates = no_candidates
        await system.update_once(0)
        return system.candidate_decisions[-1]
    decision = asyncio.run(run())
    assert decision["candidate_search_outcome_updated"] is False


def test_observed_candidate_gain_is_audit_only(tmp_path):
    system = build_system(tmp_path)
    candidate = SimpleNamespace(
        member_gain=SimpleNamespace(target_gain_vs_incumbent=4)
    )
    gain, cooldown = system._update_candidate_search_outcome(
        target=0,
        update_index=3,
        stage_b_evaluations=[candidate],
    )
    assert gain == 4
    assert cooldown == 0
    assert not any("candidate_search" in name for name in system.responsibility_state.__dict__)


def test_nonpositive_observation_creates_no_cooldown(tmp_path):
    system = build_system(tmp_path)
    candidate = SimpleNamespace(
        member_gain=SimpleNamespace(target_gain_vs_incumbent=0)
    )
    gain, cooldown = system._update_candidate_search_outcome(
        target=0,
        update_index=3,
        stage_b_evaluations=[candidate],
    )
    assert gain == 0
    assert cooldown == 0
    assert not any("cooldown" in name for name in system.responsibility_state.__dict__)


def test_student_partial_validity_keeps_valid_candidate_without_retry(tmp_path):
    student_calls = 0

    async def optimizer(system_prompt, _user_prompt, _temperature, _max_tokens):
        nonlocal student_calls
        if "Check only explicit hard blockers" in system_prompt:
            return json.dumps(APPROVED)
        if system_prompt.startswith("Return strict JSON only."):
            student_calls += 1
            return json.dumps({"candidate_prompts": ["parent", "repair-q0"]})
        return json.dumps(TEACHER)

    system = build_system(
        tmp_path,
        optimizer,
        shared_prompt="parent",
        num_candidates_per_parent=2,
        stage_b_candidate_budget=2,
    )

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        target, hashes = routed_proposal(system)
        candidates = await system.propose_candidates(target, hashes, funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert student_calls == 1
    assert [row.prompt for row in candidates] == ["repair-q0"]
    assert funnel.student_partially_valid_responses == 1


def test_main_stage_a_passes_every_valid_generated_candidate_to_stage_b(tmp_path):
    async def optimizer(system_prompt, _user_prompt, _temperature, _max_tokens):
        if "Check only explicit hard blockers" in system_prompt:
            return json.dumps(APPROVED)
        if system_prompt.startswith("Return strict JSON only."):
            return json.dumps({
                "candidate_prompts": ["repair-q0", "unhelpful-candidate"]
            })
        return json.dumps(TEACHER)

    system = build_system(
        tmp_path,
        optimizer,
        experiment_setting="shared_responsibility_conditioned_dual_target",
        num_candidates_per_parent=2,
        stage_b_candidate_budget=2,
    )

    async def run():
        await initialize(system)
        target, hashes = routed_proposal(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(target, hashes, funnel)
        _, _, evaluated = await system.evaluate_candidates(
            target, candidates, hashes, funnel
        )
        return funnel, evaluated

    funnel, evaluated = asyncio.run(run())
    assert len(evaluated) == 2
    assert funnel.stage_a_evaluated == 2
    assert funnel.stage_b_evaluated == 2
    assert all(row.stage_a_decision.selected for row in evaluated)
    assert all(
        row.stage_a_decision.selected_by_channels == ("matched_all_generated",)
        for row in evaluated
    )
