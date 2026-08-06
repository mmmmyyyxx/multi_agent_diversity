import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.system import (
    CandidateFunnel,
    PromptEnsembleOptimizationSystem,
)


TEACHER = {
    "failure_pattern": "premature commitment",
    "repair_rule": "Check every explicit condition before committing.",
    "preservation_rule": "Keep conclusions that still pass the checks.",
}
REGENERATED = {
    **TEACHER,
    "repair_rule": "Compare every option in order before committing.",
}
APPROVED = {"failed_checks": [], "risk_case_ids": [], "feedback": ""}


async def solver(_question, _agent_id, _prompt):
    return PromptAnswer("A", "FINAL_ANSWER: A", True)


def system_for(tmp_path, chat, **overrides):
    values = {
        "out_dir": str(tmp_path),
        "answer_format": "option_letter",
        "experiment_setting": "shared_peer_state_member_first_safe",
        "num_candidates_per_parent": 2,
        "stage_a_channel_top_k": 1,
        "stage_b_candidate_budget": 1,
    }
    values.update(overrides)
    return PromptEnsembleOptimizationSystem(
        Config.from_flat(**values),
        solver=solver,
        optimizer_chat=chat,
    )


async def initialize(system):
    rows = [{"question": "q", "answer": "A"}]
    system.validation_probe = system.build_validation_probe(rows)
    await system.initialize_fixed_probe(rows)


def role(system_prompt):
    if system_prompt.startswith("Return strict JSON only."):
        return "student"
    if "Check only explicit hard blockers" in system_prompt:
        return "critic"
    return "teacher"


def test_invalid_invalid_valid_recovers_on_third_student_call(tmp_path):
    student_calls = 0
    student_requests = []

    async def chat(system_prompt, user_prompt, _temperature, _max_tokens):
        nonlocal student_calls
        current = role(system_prompt)
        if current == "critic":
            return json.dumps(APPROVED)
        if current == "teacher":
            return json.dumps(TEACHER)
        student_calls += 1
        student_requests.append(user_prompt)
        if student_calls < 3:
            return json.dumps({"candidate_prompts": [None, ""]})
        return json.dumps({"candidate_prompts": ["valid repair"]})

    system = system_for(tmp_path, chat)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel, 0)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert len(candidates) == 1
    assert student_calls == 3
    assert funnel.student_retry_count == 2
    assert funnel.student_recovered is True
    assert funnel.upstream_regeneration_count == 0
    assert "StudentRecoveryFeedback:" in student_requests[1]
    assert "empty_or_non_string" in student_requests[1]
    assert "valid repair" not in student_requests[1]


def test_partial_valid_stops_without_retry(tmp_path):
    student_calls = 0

    async def chat(system_prompt, _user_prompt, _temperature, _max_tokens):
        nonlocal student_calls
        current = role(system_prompt)
        if current == "critic":
            return json.dumps(APPROVED)
        if current == "teacher":
            return json.dumps(TEACHER)
        student_calls += 1
        return json.dumps({"candidate_prompts": [None, "valid repair"]})

    system = system_for(tmp_path, chat)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel, 0)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert len(candidates) == 1
    assert student_calls == 1
    assert funnel.student_partially_valid_responses == 1
    assert funnel.student_retry_triggered is False


def test_exhausted_cycle_regenerates_teacher_and_second_cycle_recovers(tmp_path):
    counts = {"teacher": 0, "critic": 0, "student": 0}

    async def chat(system_prompt, user_prompt, _temperature, _max_tokens):
        current = role(system_prompt)
        counts[current] += 1
        if current == "critic":
            return json.dumps(APPROVED)
        if current == "teacher":
            return json.dumps(
                REGENERATED
                if "student_upstream_regeneration" in user_prompt
                else TEACHER
            )
        if counts["student"] <= 4:
            return json.dumps({"candidate_prompts": [None, ""]})
        return json.dumps({"candidate_prompts": ["recovered upstream"]})

    system = system_for(tmp_path, chat)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel, 0)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert len(candidates) == 1
    assert counts == {"teacher": 2, "critic": 2, "student": 5}
    assert funnel.student_cycle_exhausted is True
    assert funnel.upstream_regeneration_triggered is True
    assert funnel.upstream_regeneration_count == 1
    assert funnel.student_recovered is True
    upstream_teacher = [
        row for row in system.tcs_rounds
        if row["role"] == "teacher"
        and row.get("student_generation_cycle_index") == 1
    ]
    assert upstream_teacher[-1]["upstream_plan_changed"] is True


def test_two_exhausted_cycles_have_distinct_terminal_failure(tmp_path):
    async def chat(system_prompt, user_prompt, _temperature, _max_tokens):
        current = role(system_prompt)
        if current == "critic":
            return json.dumps(APPROVED)
        if current == "teacher":
            return json.dumps(
                REGENERATED
                if "student_upstream_regeneration" in user_prompt
                else TEACHER
            )
        return json.dumps({"candidate_prompts": [None, ""]})

    system = system_for(tmp_path, chat)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel, 0)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    assert funnel.student_calls == 8
    assert funnel.upstream_regeneration_count == 1
    assert funnel.terminal_failure_class == (
        "student_invalid_exhausted_after_upstream_regeneration"
    )
    assert funnel.terminal_student_failure_class == funnel.terminal_failure_class


def test_contract_contamination_exhaustion_is_protocol_failure_without_rollout_or_freeze(
    tmp_path,
):
    student_requests = []

    async def chat(system_prompt, user_prompt, _temperature, _max_tokens):
        current = role(system_prompt)
        if current == "critic":
            return json.dumps(APPROVED)
        if current == "teacher":
            return json.dumps(
                REGENERATED
                if "student_upstream_regeneration" in user_prompt
                else TEACHER
            )
        student_requests.append(user_prompt)
        return json.dumps({
            "candidate_prompts": [
                "Check every condition.\nFINAL_ANSWER: A",
                "Compare all options.\nFINAL ANSWER: B",
            ]
        })

    system = system_for(
        tmp_path,
        chat,
        experiment_setting="shared_member_aware_full",
    )

    async def run():
        rows = [{"question": "q", "answer": "B"}]
        system.validation_probe = system.build_validation_probe(rows)
        await system.initialize_fixed_probe(rows)
        solver_calls_before = sum(
            row["role"] == "solver" for row in system.llm.calls
        )
        failures_before = dict(
            system.responsibility_state.consecutive_failed_updates_by_agent
        )
        accepted = await system.update_once(0)
        solver_calls_after = sum(
            row["role"] == "solver" for row in system.llm.calls
        )
        return accepted, solver_calls_before, solver_calls_after, failures_before

    accepted, calls_before, calls_after, failures_before = asyncio.run(run())
    decision = system.candidate_decisions[-1]
    funnel = decision["funnel"]
    assert accepted is False
    assert calls_after == calls_before
    assert funnel["stage_a_evaluated"] == 0
    assert funnel["output_contract_contamination_count"] == 16
    assert funnel["terminal_failure_class"] == "proposal_protocol_failure"
    assert system.previous_update_outcomes[decision["target_agent_id"]].rejection_reasons == (
        "proposal_protocol_failure",
    )
    assert (
        system.responsibility_state.consecutive_failed_updates_by_agent
        == failures_before
    )
    assert not system.repairability_freeze_events
    assert "A candidate included the immutable solver output interface." in (
        student_requests[1]
    )
    student_audits = [
        row for row in system.tcs_rounds if row.get("role") == "student"
    ]
    assert all(row["response_excerpt"] == "" for row in student_audits)
    assert all(row["protocol_rejections"] for row in student_audits)
    assert all(
        "candidate_hash" in rejection
        and "detected_marker_type" in rejection
        and "candidate_prompt" not in rejection
        for row in student_audits
        for rejection in row["protocol_rejections"]
    )


def test_unchanged_upstream_plan_is_audited_and_recriticized(tmp_path):
    counts = {"teacher": 0, "critic": 0, "student": 0}

    async def chat(system_prompt, _user_prompt, _temperature, _max_tokens):
        current = role(system_prompt)
        counts[current] += 1
        if current == "critic":
            return json.dumps(APPROVED)
        if current == "teacher":
            return json.dumps(TEACHER)
        if counts["student"] <= 4:
            return json.dumps({"candidate_prompts": []})
        return json.dumps({"candidate_prompts": ["valid repair"]})

    system = system_for(tmp_path, chat)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        await system.propose_candidates(0, set(), funnel, 0)
        return funnel

    funnel = asyncio.run(run())
    upstream_teacher = [
        row for row in system.tcs_rounds
        if row["role"] == "teacher"
        and row.get("student_generation_cycle_index") == 1
    ]
    assert upstream_teacher[-1]["upstream_plan_changed"] is False
    assert counts["critic"] == 2
    assert funnel.student_recovered is True


def test_upstream_critic_rejection_never_enters_second_student_cycle(tmp_path):
    counts = {"teacher": 0, "critic": 0, "student": 0}

    async def chat(system_prompt, user_prompt, _temperature, _max_tokens):
        current = role(system_prompt)
        counts[current] += 1
        if current == "critic":
            if counts["critic"] == 1:
                return json.dumps(APPROVED)
            return json.dumps({
                "failed_checks": ["actionable_specificity"],
                "risk_case_ids": [],
                "feedback": "Make the regenerated rule executable.",
            })
        if current == "teacher":
            return json.dumps(
                REGENERATED
                if (
                    "student_upstream_regeneration" in user_prompt
                    or "PreviousTeacherRepairPlan:" in user_prompt
                )
                else TEACHER
            )
        return json.dumps({"candidate_prompts": [None, ""]})

    system = system_for(tmp_path, chat)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel, 0)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    assert counts["student"] == 4
    assert counts["critic"] == 3
    assert funnel.terminal_failure_class == (
        "upstream_critic_semantic_rejection_exhausted"
    )


def test_upstream_teacher_invalid_has_distinct_terminal_failure(tmp_path):
    counts = {"teacher": 0, "critic": 0, "student": 0}

    async def chat(system_prompt, _user_prompt, _temperature, _max_tokens):
        current = role(system_prompt)
        counts[current] += 1
        if current == "critic":
            return json.dumps(APPROVED)
        if current == "teacher":
            return json.dumps(TEACHER) if counts["teacher"] == 1 else "{"
        return json.dumps({"candidate_prompts": []})

    system = system_for(tmp_path, chat)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel, 0)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    assert counts["student"] == 4
    assert funnel.terminal_failure_class == "upstream_teacher_invalid_exhausted"


def test_student_exhaustion_never_updates_potential_or_stage_b(tmp_path):
    async def chat(system_prompt, _user_prompt, _temperature, _max_tokens):
        current = role(system_prompt)
        if current == "critic":
            return json.dumps(APPROVED)
        if current == "teacher":
            return json.dumps(TEACHER)
        return json.dumps({"candidate_prompts": [None, ""]})

    system = system_for(
        tmp_path,
        chat,
        experiment_setting="shared_independent_accuracy",
    )

    async def run():
        await initialize(system)
        await system.update_once(0)
        return system.candidate_decisions[-1]

    decision = asyncio.run(run())
    assert decision["funnel"]["stage_a_evaluated"] == 0
    assert decision["funnel"]["stage_b_evaluated"] == 0
    assert decision["candidate_search_outcome_updated"] is False
    assert not any("candidate_search" in name for name in system.responsibility_state.__dict__)
