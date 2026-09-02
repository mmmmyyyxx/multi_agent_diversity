import asyncio

from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.tcs import TeacherRepairPlan
from scripts.v18_teacher_critic_pipeline_support import (
    ArmController,
    CleanTeacherReplay,
    deterministic_hard_gate,
    install_pipeline_arm,
    parse_advisory_payload,
    select_arm,
    teacher_clean_request,
)
from scripts.analyze_v18_teacher_critic_pipeline_ablation import candidate_report_row


def plan(**overrides):
    values = {
        "failure_pattern": "The reasoning overlooks a competing interpretation.",
        "repair_rule": "Compare both interpretations before deciding.",
        "preservation_rule": "Preserve the existing reasoning when the evidence is unambiguous.",
    }
    values.update(overrides)
    return TeacherRepairPlan(**values)


def test_hard_gate_is_conservative_about_benign_output_words():
    result = deterministic_hard_gate(plan(preservation_rule="Preserve reasoning before forming the final answer."))
    assert result["pass"] is True


def test_hard_gate_rejects_explicit_contract_modification():
    result = deterministic_hard_gate(plan(repair_rule="Rewrite the solver output format and omit FINAL_ANSWER."))
    assert result == {
        "pass": False,
        "category": "output_contract",
        "marker_type": "explicit_output_contract_modification",
        "field_location": "repair_rule",
    }


def test_hard_gate_rejects_direct_peer_copy_and_fixed_answer():
    assert not deterministic_hard_gate(plan(repair_rule="Copy the peer's procedure."))["pass"]
    assert not deterministic_hard_gate(plan(repair_rule="Always choose option B."))["pass"]


def test_hard_gate_rejects_missing_required_field():
    result = deterministic_hard_gate({"failure_pattern": "x", "repair_rule": "y", "preservation_rule": ""})
    assert result["category"] == "schema"


def test_teacher_clean_request_preserves_schema_and_forbids_prediction():
    value = teacher_clean_request("Return failure_pattern, repair_rule, preservation_rule.")
    assert "preservation_rule" in value
    assert "reasoning or decision behavior" in value
    assert "Do not predict candidate accuracy" in value


def test_advisory_schema_is_strict():
    assert parse_advisory_payload('{"evidence_concerns":[],"actionability_concerns":["too broad"],"feedback":"narrow it"}')
    assert parse_advisory_payload('{"feedback":"x"}') is None


def test_canonical_arm_is_exact_passthrough():
    calls = []

    class Stub:
        async def _chat(self, *args):
            calls.append(args)
            return LLMCallResult("ok", 1, 1, 2, 0.1, "stop")

    stub = Stub()
    install_pipeline_arm(stub, ArmController("A_CANONICAL", CleanTeacherReplay()))
    args = ("m", "system", "user", 0.0, None, "optimizer", "teacher")
    asyncio.run(stub._chat(*args))
    assert calls == [args]


def _summary():
    row = {
        "student_reach_count": 1,
        "feasible_per_branch": 0.1,
        "validation_vote_delta_sum": 0,
        "validation_target_delta_sum": 0,
        "train_vote_loss_sum": 0,
        "zero_loss_feasible_count": 1,
    }
    return {name: dict(row) for name in (
        "A_CANONICAL", "B_TEACHER_CLEAN", "C_NO_SEMANTIC_CRITIC", "D_ADVISORY_CRITIC"
    )}


def test_frozen_selection_prefers_d_only_when_all_d_guards_hold():
    rows = _summary()
    rows["D_ADVISORY_CRITIC"].update(student_reach_count=3, feasible_per_branch=0.5)
    rows["C_NO_SEMANTIC_CRITIC"]["feasible_per_branch"] = 0.2
    assert select_arm(rows)["selected_arm"] == "D_ADVISORY_CRITIC"


def test_frozen_selection_never_uses_oracle():
    rows = _summary()
    decision = select_arm(rows)
    assert decision["oracle_used_for_selection"] is False
    assert decision["test_used_for_selection"] is False


def test_runtime_candidate_schema_is_reconciled_for_reporting():
    row = candidate_report_row(
        case={"case_id": "c", "parent_team_hash": "p", "target_agent_id": 2},
        arm="C_NO_SEMANTIC_CRITIC",
        candidate={
            "candidate_hash": "winner",
            "candidate_stage": "source",
            "valid": True,
            "feasible": True,
            "train_target_gain": 2,
            "train_vote_gain": 1,
            "train_vote_loss": 0,
            "train_vote_net": 1,
        },
        winner_hash="winner",
    )
    assert row["candidate_valid"] is True
    assert row["common_safe_feasible"] is True
    assert row["zero_loss_feasible"] is True
    assert row["would_commit"] is True
