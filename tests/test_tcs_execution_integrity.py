import asyncio
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import (
    TCS_AUDIT_CONTEXT,
    TraceBeamSearchSystem,
    tcs_metadata_applicable,
    validate_tcs_candidate_metadata,
)


def _metadata(**overrides):
    base = {
        "optimizer_architecture": "teacher_critic_student",
        "candidate_source": "teacher_critic_student",
        "candidate_pool_source": "optimizer",
        "teacher_question": "Which independent verification path repairs the observed error?",
        "teacher_question_approved": True,
        "teacher_question_forced_best_score": False,
        "teacher_critic_rounds": 1,
        "teacher_rewrite_count": 0,
        "student_candidate_count_raw": 2,
        "student_candidate_count_final": 2,
    }
    base.update(overrides)
    return base


def test_tcs_metadata_accepts_first_round_rewrite_and_forced_best():
    assert validate_tcs_candidate_metadata(_metadata()) == []
    assert validate_tcs_candidate_metadata(_metadata(teacher_critic_rounds=2, teacher_rewrite_count=1)) == []
    assert validate_tcs_candidate_metadata(
        _metadata(teacher_question_approved=False, teacher_question_forced_best_score=True, teacher_critic_rounds=3)
    ) == []


def test_tcs_metadata_rejects_missing_provenance_but_excludes_existing_beam():
    invalid = _metadata(teacher_question="", teacher_critic_rounds=0)
    assert tcs_metadata_applicable(invalid)
    assert set(validate_tcs_candidate_metadata(invalid)) >= {"missing_teacher_question", "zero_teacher_critic_rounds"}

    existing = _metadata(candidate_source="existing_beam", candidate_pool_source="existing_beam", teacher_question="", teacher_critic_rounds=0)
    assert not tcs_metadata_applicable(existing)
    assert validate_tcs_candidate_metadata(existing) == []


def test_tcs_metadata_checks_student_counts():
    assert "missing_student_raw_count" in validate_tcs_candidate_metadata(_metadata(student_candidate_count_raw=0))
    assert "missing_student_final_count" in validate_tcs_candidate_metadata(_metadata(student_candidate_count_final=0))
    assert "inconsistent_student_counts" in validate_tcs_candidate_metadata(
        _metadata(student_candidate_count_raw=1, student_candidate_count_final=2)
    )


def _bare_system(cfg=None):
    cfg = cfg or Config(optimizer_architecture="teacher_critic_student", optimizer_fallback_mode="none", agents=1)
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.agents = [AgentState("parent one")]
    system.update_logs = []
    system.optimizer_generation_diagnostics = {}
    system.llm_call_logs = []
    system.cost_summary = system._empty_cost_summary()
    return system


class _CompletionClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        system_prompt = kwargs["messages"][0]["content"]
        if "You are the Teacher in" in system_prompt:
            content = '{"socratic_guiding_question":"Which independent check repairs the observed error?"}'
        elif "You are the Critic in" in system_prompt:
            content = '{"passed":true,"score":0.9,"quality_critique":"grounded"}'
        else:
            content = '{"candidates":[{"candidate_prompt":"Check the decisive constraint, compare alternatives, then verify one final answer.","student_interpretation_of_question":"add an independent check","target_error_pattern":"missed constraint","accuracy_repair_rule":"check constraints","diversity_contribution":"independent verification","error_correlation_reduction":"avoid shared omission","task_alignment_rule":"respect output format","peer_redundancy_avoidance":"use a separate check","expected_accuracy_effect":"fewer omissions","expected_diversity_effect":"different valid route","risk_control":"stay concise","rationale":"repairs the observed error"}]}'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )


def test_tcs_llm_call_stages_recorded_in_execution_order():
    system = _bare_system()
    client = _CompletionClient()
    system.evaluator_client = client
    system.solver_client = client

    async def run_tcs():
        token = TCS_AUDIT_CONTEXT.set({"epoch": 1, "step": 10, "agent_id": 0, "parent_id": "beam-parent"})
        try:
            return await system.propose_candidates_teacher_critic_student(
                agent_id=0,
                parent_prompt="parent one",
                overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0]},
                num_candidates=1,
                generation_batches=[{"batch_type": "window_update_diagnosis", "cases": []}],
            )
        finally:
            TCS_AUDIT_CONTEXT.reset(token)

    candidates = asyncio.run(run_tcs())

    assert len(candidates) == 1
    calls = system.llm_call_logs
    assert [row["llm_call_stage"] for row in calls] == ["teacher", "critic", "student"]
    assert [row["model_role"] for row in calls] == ["optimizer", "evaluator", "optimizer"]
    assert all(row["call_succeeded"] for row in calls)
    assert all(row["optimizer_architecture"] == "teacher_critic_student" for row in calls)
    assert all(row["epoch"] == 1 and row["step"] == 10 and row["parent_id"] == "beam-parent" for row in calls)


def test_concurrent_parent_provenance_does_not_cross_contaminate():
    async def make_candidate(parent_id, question):
        token = TCS_AUDIT_CONTEXT.set({"parent_id": parent_id, "agent_id": 0, "epoch": 1, "step": 10})
        try:
            await asyncio.sleep(0)
            context = dict(TCS_AUDIT_CONTEXT.get())
            return {
                "parent_id": context["parent_id"],
                "teacher_question": question,
                "teacher_critic_rounds": 1,
                "student_candidate_count_raw": 1,
                "student_candidate_count_final": 1,
            }
        finally:
            TCS_AUDIT_CONTEXT.reset(token)

    async def run_both():
        return await asyncio.gather(make_candidate("parent-a", "question-a"), make_candidate("parent-b", "question-b"))

    first, second = asyncio.run(run_both())
    assert first["parent_id"] == "parent-a"
    assert first["teacher_question"] == "question-a"
    assert second["parent_id"] == "parent-b"
    assert second["teacher_question"] == "question-b"


def test_invalid_tcs_candidate_fails_before_evaluation():
    system = _bare_system()
    system.agents[0].prompt_beam = [{"id": "parent", "prompt": "parent one", "score": 0.0, "metrics": {}, "generation": 0}]
    system._build_case_generation_batches = lambda *_: [{"batch_type": "window_update_diagnosis", "cases": []}]

    async def invalid_proposals(**_kwargs):
        return [{
            "candidate_prompt": "Candidate with broken provenance.",
            "candidate_source": "teacher_critic_student",
            "optimizer_generation_diagnostics": {
                "optimizer_architecture": "teacher_critic_student",
                "teacher_question": "",
                "teacher_critic_rounds": 0,
                "teacher_question_forced_best_score": False,
                "student_candidate_count_raw": 1,
                "student_candidate_count_final": 1,
            },
        }]

    system.propose_candidates = invalid_proposals
    with pytest.raises(RuntimeError, match=r"Invalid Teacher-Critic-Student candidate metadata: agent_id=0 epoch=1 step=10 parent_id=parent"):
        asyncio.run(system.update_prompt_with_beam(0, {}, [], step_id=10, epoch_id=1))
