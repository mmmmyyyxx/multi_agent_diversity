import asyncio
import json

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
    if system_prompt == "Return strict JSON only.":
        return json.dumps({"candidate_prompts": ["repair-q0"]})
    return json.dumps(TEACHER)


def build_system(tmp_path, optimizer=fake_optimizer, **overrides):
    values = {
        "out_dir": str(tmp_path),
        "answer_format": "option_letter",
        "num_candidates_per_parent": 1,
        "stage_a_channel_top_k": 1,
        "stage_b_candidate_budget": 1,
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
    assert audit["context_type"] == "MemberAwareDiagnosisContext"
    assert audit["full_probe_case_count"] == 3
    assert audit["selected_pattern_count"] <= 3
    assert audit["selected_case_count"] <= 3
    assert audit["forbidden_field_violations"] == []
    assert [row["role"] for row in system.tcs_rounds] == [
        "teacher", "critic", "student",
    ]
    assert system.candidate_decisions[-1]["funnel"]["accepted_candidate"]
    assert system.candidate_decisions[-1]["candidates"][0]["repair_plan_hash"]


def test_generic_context_isolation_for_accuracy_and_peer_state(tmp_path):
    async def inspect(setting):
        system = build_system(
            tmp_path / setting,
            experiment_setting=setting,
        )
        await initialize(system)
        await system.propose_candidates(0, set(), CandidateFunnel())
        return system.tcs_context_history[-1]

    async def run_all():
        return await asyncio.gather(
            inspect("shared_independent_accuracy"),
            inspect("shared_peer_state_vote_first"),
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


def test_only_valid_critic_rejection_consumes_semantic_revision(tmp_path):
    teacher_calls = critic_calls = 0

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
        if system_prompt == "Return strict JSON only.":
            return json.dumps({"candidate_prompts": ["repair-q0"]})
        teacher_calls += 1
        if teacher_calls == 2:
            assert "Specify the executable verification order." in user_prompt
        return json.dumps(TEACHER)

    system = build_system(tmp_path, optimizer)

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert len(candidates) == 1
    assert teacher_calls == 2 and critic_calls == 2
    assert funnel.critic_semantic_rejections == 1


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
        candidates = await system.propose_candidates(0, set(), funnel)
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
    limit: int = 300,
) -> LLMCallResult:
    return LLMCallResult(
        text=text,
        prompt_tokens=1,
        completion_tokens=completion_tokens,
        total_tokens=completion_tokens + 1,
        latency_seconds=0.0,
        finish_reason=finish_reason,
        completion_token_limit=limit,
        hit_completion_limit=finish_reason == "length" or completion_tokens >= limit,
    )


def test_teacher_truncation_retries_identical_request_without_semantic_round_use(tmp_path):
    system = build_system(tmp_path)
    captured = []

    async def chat(_model, system_prompt, user_prompt, _temperature, max_tokens, role):
        captured.append((role, system_prompt, user_prompt, max_tokens))
        if len(captured) == 1:
            return result("{", finish_reason="length", completion_tokens=max_tokens, limit=max_tokens)
        if "Check only explicit hard blockers" in system_prompt:
            return result(json.dumps(APPROVED), limit=max_tokens)
        if system_prompt == "Return strict JSON only.":
            return result(json.dumps({"candidate_prompts": ["repair-q0"]}), limit=max_tokens)
        return result(json.dumps(TEACHER), limit=max_tokens)

    system._chat = chat

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel)
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
            return result("{", finish_reason="length", completion_tokens=max_tokens, limit=max_tokens)
        role_calls.append("teacher")
        return result(json.dumps(TEACHER), limit=max_tokens)

    system._chat = chat

    async def run():
        await initialize(system)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    assert role_calls == ["teacher", "critic", "critic"]
    assert funnel.critic_truncated_responses == 2


def test_student_partial_validity_keeps_valid_candidate_without_retry(tmp_path):
    student_calls = 0

    async def optimizer(system_prompt, _user_prompt, _temperature, _max_tokens):
        nonlocal student_calls
        if "Check only explicit hard blockers" in system_prompt:
            return json.dumps(APPROVED)
        if system_prompt == "Return strict JSON only.":
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
        candidates = await system.propose_candidates(0, set(), funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert student_calls == 1
    assert [row.prompt for row in candidates] == ["repair-q0"]
    assert funnel.student_partially_valid_responses == 1
