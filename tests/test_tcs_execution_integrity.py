import asyncio
import json
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
from scripts.audit_tcs_run import audit_run


def _metadata(**overrides):
    base = {
        "optimizer_architecture": "teacher_critic_student",
        "candidate_source": "teacher_critic_student",
        "candidate_pool_source": "optimizer",
        "teacher_question": "Which independent verification path repairs the observed error?",
        "teacher_question_approved": True,
        "teacher_question_forced_best_score": False,
        "tcs_call_group_id": "e1_s10_a0_pparent_hash",
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
        _metadata(teacher_question_approved=False, teacher_question_forced_best_score=True, teacher_critic_rounds=3, teacher_question_forced_best_round=2)
    ) == []


def test_tcs_metadata_rejects_missing_provenance_but_excludes_existing_beam():
    invalid = _metadata(teacher_question="", teacher_critic_rounds=0)
    assert tcs_metadata_applicable(invalid)
    assert set(validate_tcs_candidate_metadata(invalid)) >= {"missing_teacher_question", "zero_teacher_critic_rounds"}

    existing = _metadata(candidate_source="existing_beam", candidate_pool_source="existing_beam", teacher_question="", teacher_critic_rounds=0)
    assert not tcs_metadata_applicable(existing)
    assert validate_tcs_candidate_metadata(existing) == []

    open_candidate = _metadata(
        optimizer_architecture="open_mechanism_exploration",
        candidate_source="open_mechanism_exploration",
        teacher_question="",
        teacher_critic_rounds=0,
        tcs_call_group_id="",
    )
    assert not tcs_metadata_applicable(open_candidate)
    assert validate_tcs_candidate_metadata(open_candidate) == []


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
    system.execution_session_id = "testsession01"
    system.agents = [AgentState("parent one")]
    system.update_logs = []
    system.optimizer_generation_diagnostics = {}
    system.llm_call_logs = []
    system.cost_summary = system._empty_cost_summary()
    return system


def _audit_candidate(metadata, candidate_id="candidate"):
    row = _metadata(**metadata)
    row.update({
        "event": "candidate_evaluated", "candidate_id": candidate_id,
        "accuracy_delta": 0.0, "diversity_delta": 0.0, "invalid_delta": 0.0,
        "vote_delta": 0.0, "coverage_delta": 0.0, "net_coverage_delta": 0.0,
        "candidate_target_accuracy": 0.5, "baseline_target_accuracy": 0.5,
        "candidate_embedding_diversity": 0.5, "baseline_embedding_diversity": 0.5,
        "candidate_invalid_rate": 0.0, "baseline_invalid_rate": 0.0,
        "candidate_team_accuracy": 0.5, "baseline_team_accuracy": 0.5,
        "candidate_oracle_acc": 0.5, "baseline_oracle_acc": 0.5,
    })
    return row


def _audit_call(stage, group, session, attempt, *, parent_id="parent", rounds=1):
    return {
        "llm_call_stage": stage, "tcs_call_group_id": group,
        "execution_session_id": session, "update_attempt_id": attempt,
        "parent_id": parent_id, "agent_id": 0, "epoch": 1, "step": 10,
        "call_succeeded": True, "response_empty": False,
    }


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
        token = TCS_AUDIT_CONTEXT.set({
            "epoch": 1, "step": 10, "agent_id": 0, "parent_id": "beam-parent",
            "execution_session_id": "testsession01", "update_attempt_id": "testsession01_e1_s10_a0",
            "tcs_call_group_id": "testsession01_e1_s10_a0_pbeam_parent",
        })
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
    assert {row["execution_session_id"] for row in calls} == {"testsession01"}
    assert {row["update_attempt_id"] for row in calls} == {"testsession01_e1_s10_a0"}
    assert {row["tcs_call_group_id"] for row in calls} == {"testsession01_e1_s10_a0_pbeam_parent"}
    assert candidates[0]["execution_session_id"] == "testsession01"
    assert candidates[0]["update_attempt_id"] == "testsession01_e1_s10_a0"


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


def test_session_scoped_ids_share_update_but_separate_parents():
    system = _bare_system()
    update_id = system._update_attempt_id(epoch_id=1, step_id=10, agent_id=0)
    first = system._tcs_call_group_id(update_id, "parent-a", "prompt-a")
    second = system._tcs_call_group_id(update_id, "parent-b", "prompt-b")
    assert update_id == "testsession01_e1_s10_a0"
    assert first != second
    assert first.startswith(update_id + "_p")


def test_tcs_group_ids_separate_initial_and_refill_generation_rounds():
    system = _bare_system()
    update_id = system._update_attempt_id(epoch_id=1, step_id=10, agent_id=0)
    initial = system._tcs_call_group_id(update_id, "parent", "prompt", generation_round=0)
    refill = system._tcs_call_group_id(update_id, "parent", "prompt", generation_round=1)
    assert initial != refill
    assert initial.endswith("_r0")
    assert refill.endswith("_r1")


def test_refill_candidate_preserves_tcs_provenance():
    system = _bare_system()
    metadata = _metadata(
        execution_session_id="testsession01",
        update_attempt_id="testsession01_e1_s10_a0",
    )
    proposal = {
        "candidate_source": "teacher_critic_student",
        "optimizer_architecture": "teacher_critic_student",
        "tcs_call_group_id": metadata["tcs_call_group_id"],
        "execution_session_id": metadata["execution_session_id"],
        "update_attempt_id": metadata["update_attempt_id"],
        "optimizer_generation_diagnostics": metadata,
    }
    candidate = system._make_refill_candidate(
        proposal=proposal,
        prompt="repair prompt",
        parent_id="parent",
        parent_prompt="parent prompt",
        agent_id=0,
        candidate_index=0,
        refill_round=1,
        generation=1,
    )
    audit_metadata = {
        "optimizer_architecture": candidate["optimizer_architecture"],
        "candidate_source": candidate["candidate_source"],
        "candidate_pool_source": candidate["candidate_pool_source"],
        "tcs_call_group_id": candidate["tcs_call_group_id"],
        **candidate["optimizer_generation_diagnostics"],
    }
    assert validate_tcs_candidate_metadata(audit_metadata) == []


def test_open_exploration_call_has_no_tcs_group_or_teacher_critic_calls():
    system = _bare_system(Config(
        method_version="v8_stable_qd_lineage",
        optimizer_architecture="teacher_critic_student",
        optimizer_fallback_mode="none",
        agents=1,
    ))
    client = _CompletionClient()
    system.evaluator_client = client
    system.solver_client = client
    system.truncated_prompt_count = 0
    system.prompt_overlength_rejection_count = 0
    system.open_exploration_generation_count = 0
    system.open_exploration_candidate_count = 0

    token = TCS_AUDIT_CONTEXT.set({
        "epoch": 1,
        "step": 10,
        "agent_id": 0,
        "parent_id": "parent",
        "execution_session_id": "testsession01",
        "update_attempt_id": "testsession01_e1_s10_a0",
        "tcs_call_group_id": "must-not-leak",
    })
    try:
        asyncio.run(system.propose_candidates_teacher_critic_student(
            agent_id=0,
            parent_prompt="parent prompt",
            overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0]},
            num_candidates=1,
            generation_batches=[{"batch_type": "window_update_diagnosis", "cases": []}],
            generation_channel="open_mechanism_exploration",
        ))
    finally:
        TCS_AUDIT_CONTEXT.reset(token)

    assert len(system.llm_call_logs) == 1
    assert system.llm_call_logs[0]["llm_call_stage"] == "open_mechanism_exploration_agent_0"
    assert system.llm_call_logs[0]["optimizer_architecture"] == "open_mechanism_exploration"
    assert system.llm_call_logs[0]["tcs_call_group_id"] == ""
    assert system.cost_summary["calls_saved_by_tcs_round_reduction"] == 0


def test_v8_tcs_round_reduction_reports_saved_calls():
    system = _bare_system(Config(
        method_version="v8_stable_qd_lineage",
        optimizer_architecture="teacher_critic_student",
        optimizer_fallback_mode="none",
        agents=1,
    ))
    client = _CompletionClient()
    system.evaluator_client = client
    system.solver_client = client
    system.truncated_prompt_count = 0
    system.prompt_overlength_rejection_count = 0
    system.tcs_repair_generation_count = 0
    system.tcs_repair_candidate_count = 0

    asyncio.run(system.propose_candidates_teacher_critic_student(
        agent_id=0,
        parent_prompt="parent prompt",
        overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0]},
        num_candidates=1,
        generation_batches=[{"batch_type": "target_error_repair", "cases": [{"case_id": "c1"}]}],
        generation_channel="tcs_repair",
    ))

    assert system.cost_summary["tcs_critic_calls"] == 1
    assert system.cost_summary["tcs_rewrite_calls"] == 0
    assert system.cost_summary["calls_saved_by_tcs_round_reduction"] == 4


def test_repeated_step_in_new_session_has_distinct_provenance_ids():
    first = _bare_system()
    second = _bare_system()
    first.execution_session_id = "sessionaaaa"
    second.execution_session_id = "sessionbbbb"
    first_update = first._update_attempt_id(1, 10, 0)
    second_update = second._update_attempt_id(1, 10, 0)
    assert first_update != second_update
    assert first._tcs_call_group_id(first_update, "parent", "prompt") != second._tcs_call_group_id(second_update, "parent", "prompt")


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


def test_group_audit_requires_per_parent_call_evidence(tmp_path):
    group = "e1_s10_a0_pparent_hash"
    candidate = _metadata(
        parent_id="parent", agent_id=0, epoch=1, step=10, tcs_call_group_id=group,
        teacher_question_forced_best_score=False, teacher_question_approved=True,
    )
    candidate.update({"event": "candidate_evaluated", "candidate_id": "candidate", "accuracy_delta": 0.0, "diversity_delta": 0.0, "invalid_delta": 0.0, "vote_delta": 0.0, "coverage_delta": 0.0, "net_coverage_delta": 0.0, "candidate_target_accuracy": 0.5, "baseline_target_accuracy": 0.5, "candidate_embedding_diversity": 0.5, "baseline_embedding_diversity": 0.5, "candidate_invalid_rate": 0.0, "baseline_invalid_rate": 0.0, "candidate_team_accuracy": 0.5, "baseline_team_accuracy": 0.5, "candidate_oracle_acc": 0.5, "baseline_oracle_acc": 0.5})
    calls = [
        {"llm_call_stage": stage, "tcs_call_group_id": group, "parent_id": "parent", "agent_id": 0, "epoch": 1, "step": 10, "call_succeeded": True, "response_empty": False}
        for stage in ("teacher", "critic", "student")
    ]
    (tmp_path / "update_logs.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    (tmp_path / "llm_calls.jsonl").write_text("\n".join(json.dumps(row) for row in calls) + "\n", encoding="utf-8")
    report = audit_run(tmp_path)
    assert report["problems"] is False
    assert report["completed_tcs_group_count"] == 1
    assert report["legacy_group_id_count"] == 1

    calls[1]["tcs_call_group_id"] = "other-group"
    (tmp_path / "llm_calls.jsonl").write_text("\n".join(json.dumps(row) for row in calls) + "\n", encoding="utf-8")
    report = audit_run(tmp_path)
    assert report["problems"] is True
    assert report["unexplained_incomplete_tcs_group_count"] == 1


def test_audit_separates_replayed_step_by_execution_session(tmp_path):
    rows = []
    calls = []
    for session, rounds in (("sessionaaaa", 3), ("sessionbbbb", 1)):
        attempt = f"{session}_e1_s10_a0"
        group = f"{attempt}_pparent_prompt"
        rows.append(_audit_candidate({
            "parent_id": "parent", "agent_id": 0, "epoch": 1, "step": 10,
            "tcs_call_group_id": group, "execution_session_id": session,
            "update_attempt_id": attempt, "teacher_critic_rounds": rounds,
            "teacher_rewrite_count": rounds - 1,
        }, candidate_id=session))
        calls.append(_audit_call("teacher", group, session, attempt))
        calls.extend(_audit_call("critic", group, session, attempt) for _ in range(rounds))
        calls.extend(_audit_call("teacher_rewrite", group, session, attempt) for _ in range(rounds - 1))
        calls.append(_audit_call("student", group, session, attempt))
    (tmp_path / "update_logs.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (tmp_path / "llm_calls.jsonl").write_text("\n".join(json.dumps(row) for row in calls) + "\n", encoding="utf-8")
    report = audit_run(tmp_path)
    assert report["problems"] is False
    assert report["completed_tcs_group_count"] == 2
    assert report["legacy_group_id_count"] == 0


def test_audit_still_detects_real_critic_round_mismatch(tmp_path):
    session = "sessionaaaa"
    attempt = f"{session}_e1_s10_a0"
    group = f"{attempt}_pparent_prompt"
    candidate = _audit_candidate({
        "parent_id": "parent", "agent_id": 0, "epoch": 1, "step": 10,
        "tcs_call_group_id": group, "execution_session_id": session,
        "update_attempt_id": attempt, "teacher_critic_rounds": 2,
        "teacher_rewrite_count": 0,
    })
    calls = [_audit_call(stage, group, session, attempt) for stage in ("teacher", "critic", "student")]
    (tmp_path / "update_logs.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    (tmp_path / "llm_calls.jsonl").write_text("\n".join(json.dumps(row) for row in calls) + "\n", encoding="utf-8")
    report = audit_run(tmp_path)
    assert report["problems"] is True
    assert "critic_round_count_mismatch" in report["incomplete_tcs_groups"][group]


def test_audit_reports_malformed_jsonl(tmp_path):
    (tmp_path / "update_logs.jsonl").write_text("{broken\n", encoding="utf-8")
    (tmp_path / "llm_calls.jsonl").write_text("", encoding="utf-8")
    report = audit_run(tmp_path)
    assert report["problems"] is True
    assert report["malformed_jsonl_count"] == 1
    assert report["malformed_locations"][0]["line_number"] == 1
