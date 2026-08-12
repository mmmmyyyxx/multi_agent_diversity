from __future__ import annotations

from dataclasses import replace

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.protocol import experiment_protocol, candidate_budget_contract
from multi_dataset_diverse_rl.persistence.checkpoint import build_checkpoint, validate_checkpoint
from multi_dataset_diverse_rl.tcs import (
    M20_CURRENT_V15,
    M2A_RESIDUAL_DIAGNOSIS,
    M2B_DIAGNOSIS_MINIMAL_EDIT,
    M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC,
    M2D_RAW_RESPONSIBILITY_MINIMAL_EDIT,
    M2E_SCOPED_BEHAVIORAL_PATCH,
    MINIMAL_RESPONSIBILITY_EDIT_INSTRUCTION,
    TeacherRepairPlan,
    build_critic_request,
    build_student_request,
    build_teacher_request,
    build_teacher_revision_request,
    build_teacher_regeneration_request,
    teacher_repair_plan_hash,
    parse_critic_decision,
    parse_teacher_repair_plan,
    parse_student_candidates,
    construct_scoped_prompt,
)
from tests.test_responsibility_conditioned_tcs import single_lane_context


def context():
    return single_lane_context()


def diagnosis_payload():
    return {
        "failure_pattern": "commits before comparing alternatives",
        "repair_rule": "compare two plausible interpretations before deciding",
        "preservation_rule": "retain decisive grammatical constraints",
        "diagnosis_primary_failure_mode": "premature interpretation commitment",
        "diagnosis_evidence_patterns": ["assigned failures omit alternative comparison"],
        "diagnosis_peer_contrast": "successful reasoning contrasts plausible interpretations",
        "diagnosis_desired_behavior_changes": ["explicitly compare alternatives"],
        "edit_plan": ["add one comparison step"],
    }


def test_m20_teacher_request_is_backward_compatible():
    assert build_teacher_request(context()) == build_teacher_request(
        context(), evolution_variant=M20_CURRENT_V15
    )


def test_m20_plan_wire_payload_and_hash_remain_legacy_compatible():
    import hashlib
    import json
    plan = TeacherRepairPlan("failure", "repair", "preserve")
    legacy = {
        "failure_pattern": "failure",
        "repair_rule": "repair",
        "preservation_rule": "preserve",
    }
    encoded = json.dumps(
        legacy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert teacher_repair_plan_hash(plan) == hashlib.sha256(encoded.encode()).hexdigest()
    student = build_student_request(
        parent_prompt="parent", approved_plan=plan,
        answer_format="multiple_choice", candidate_count=2,
        candidate_prompt_max_chars=3000,
    )
    assert "diagnosis_primary_failure_mode" not in student


def test_diagnosis_contract_survives_revision_and_upstream_regeneration():
    from multi_dataset_diverse_rl.tcs import CriticDecision
    plan = parse_teacher_repair_plan(
        diagnosis_payload(), evolution_variant=M2A_RESIDUAL_DIAGNOSIS
    )
    decision = CriticDecision(False, ("evidence_mismatch",), (), "revise")
    revision = build_teacher_revision_request(
        context=context(), previous_plan=plan, critic_decision=decision,
        evolution_variant=M2A_RESIDUAL_DIAGNOSIS,
    )
    regeneration = build_teacher_regeneration_request(
        previous_plan_hash="h", student_rejection_classes=("invalid_json",),
        evolution_variant=M2A_RESIDUAL_DIAGNOSIS,
    )
    assert "all eight required fields" in revision
    assert "five residual-diagnosis/edit-plan fields" in regeneration


def test_diagnosis_schema_and_limits():
    plan = parse_teacher_repair_plan(
        diagnosis_payload(), evolution_variant=M2A_RESIDUAL_DIAGNOSIS
    )
    assert plan.diagnosis_primary_failure_mode
    bad = diagnosis_payload()
    bad["edit_plan"] = ["a", "b", "c"]
    with pytest.raises(ValueError, match="one or two"):
        parse_teacher_repair_plan(bad, evolution_variant=M2A_RESIDUAL_DIAGNOSIS)


def test_generic_diagnosis_is_rejected():
    bad = diagnosis_payload()
    bad["diagnosis_primary_failure_mode"] = "improve accuracy"
    with pytest.raises(ValueError, match="generic"):
        parse_teacher_repair_plan(bad, evolution_variant=M2A_RESIDUAL_DIAGNOSIS)


def test_minimal_edit_and_critic_isolation():
    plan = parse_teacher_repair_plan(
        diagnosis_payload(), evolution_variant=M2B_DIAGNOSIS_MINIMAL_EDIT
    )
    common = dict(
        parent_prompt="parent role", approved_plan=plan,
        answer_format="multiple_choice", candidate_count=2,
        candidate_prompt_max_chars=3000,
    )
    a = build_student_request(**common, evolution_variant=M2A_RESIDUAL_DIAGNOSIS)
    b = build_student_request(**common, evolution_variant=M2B_DIAGNOSIS_MINIMAL_EDIT)
    c = build_student_request(
        **common,
        evolution_variant=M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC,
    )
    assert "smallest role-consistent change" not in a
    assert b == c and "smallest role-consistent change" in b
    critic_b = build_critic_request(context(), plan, evolution_variant=M2B_DIAGNOSIS_MINIMAL_EDIT)
    critic_c = build_critic_request(
        context(), plan,
        evolution_variant=M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC,
    )
    assert critic_b != critic_c
    assert "preservation_or_output_risk" not in critic_c


def test_m2d_is_exact_raw_responsibility_minimal_edit_hybrid():
    raw_plan = TeacherRepairPlan(
        "premature commitment", "compare alternatives", "preserve valid rules"
    )
    assert build_teacher_request(
        context(), evolution_variant=M20_CURRENT_V15
    ) == build_teacher_request(
        context(), evolution_variant=M2D_RAW_RESPONSIBILITY_MINIMAL_EDIT
    )
    assert build_critic_request(
        context(), raw_plan, evolution_variant=M20_CURRENT_V15
    ) == build_critic_request(
        context(), raw_plan,
        evolution_variant=M2D_RAW_RESPONSIBILITY_MINIMAL_EDIT,
    )
    common = dict(
        parent_prompt="parent role", approved_plan=raw_plan,
        answer_format="multiple_choice", candidate_count=2,
        candidate_prompt_max_chars=3000,
    )
    m20 = build_student_request(**common, evolution_variant=M20_CURRENT_V15)
    m2d = build_student_request(
        **common, evolution_variant=M2D_RAW_RESPONSIBILITY_MINIMAL_EDIT
    )
    diagnosis_plan = parse_teacher_repair_plan(
        diagnosis_payload(), evolution_variant=M2B_DIAGNOSIS_MINIMAL_EDIT
    )
    m2b = build_student_request(
        **{**common, "approved_plan": diagnosis_plan},
        evolution_variant=M2B_DIAGNOSIS_MINIMAL_EDIT,
    )
    assert MINIMAL_RESPONSIBILITY_EDIT_INSTRUCTION.strip() not in m20
    assert MINIMAL_RESPONSIBILITY_EDIT_INSTRUCTION in m2d
    assert MINIMAL_RESPONSIBILITY_EDIT_INSTRUCTION in m2b
    assert "diagnosis_primary_failure_mode" not in m2d
    assert "relevance-focused" not in build_critic_request(
        context(), raw_plan,
        evolution_variant=M2D_RAW_RESPONSIBILITY_MINIMAL_EDIT,
    )


def test_relevance_critic_schema_rejects_legacy_risk_reason():
    payload = {
        "failed_checks": ["preservation_or_output_risk"],
        "risk_case_ids": [], "feedback": "speculative risk",
    }
    with pytest.raises(ValueError, match="unknown"):
        parse_critic_decision(
            payload, allowed_case_ids=set(),
            evolution_variant=M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC,
        )


@pytest.mark.parametrize("suffix,variant", [
    ("m20_current_v15", M20_CURRENT_V15),
    ("m2a_residual_diagnosis", M2A_RESIDUAL_DIAGNOSIS),
    ("m2b_diagnosis_minimal_edit", M2B_DIAGNOSIS_MINIMAL_EDIT),
    ("m2c_diagnosis_minimal_edit_relevance_critic", M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC),
    ("m2d_raw_responsibility_minimal_edit", M2D_RAW_RESPONSIBILITY_MINIMAL_EDIT),
    ("m2e_scoped_behavioral_patch", M2E_SCOPED_BEHAVIORAL_PATCH),
])
def test_protocol_variants_hold_module1_and_common_safe(suffix, variant):
    cfg = Config.from_flat(
        experiment_setting=f"experimental_v16_{suffix}",
        module2_context_variant="c0_current_v15",
        module2_evolution_variant=variant,
    )
    from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
    system = PromptEnsembleOptimizationSystem(cfg)
    assert system.protocol.module2_evolution_variant == variant
    assert system.protocol.module2_context_variant == "c0_current_v15"
    assert system.protocol.target_selection_policy == "repairability_adjusted_responsibility"
    assert system.protocol.candidate_acceptance_policy == "fixed_peer_monotone_target_or_vote"
    assert system.protocol.candidates_per_target_branch == 2
    assert system.cfg.tcs.teacher_critic_max_rounds == Config().tcs.teacher_critic_max_rounds
    assert system.cfg.tcs.num_candidates_per_parent == 2


def test_checkpoint_evolution_variant_mismatch_fails(tmp_path):
    from tests.test_checkpoint_peer_state import build_system
    source = build_system(tmp_path / "source")
    payload = build_checkpoint(source, epoch_index=0, update_index=0, training_state={})
    payload["module2_evolution_variant"] = M2A_RESIDUAL_DIAGNOSIS
    target = build_system(tmp_path / "target")
    with pytest.raises(ValueError, match="evolution variant"):
        validate_checkpoint(payload, target)


def test_new_variants_do_not_add_preservation_set_or_boundary_context():
    request = build_teacher_request(
        context(), evolution_variant=M2A_RESIDUAL_DIAGNOSIS
    )
    assert "Preservation responsibilities" not in request
    assert "repair_distance" not in request
    assert "boundary" not in request.lower()


def test_m2e_scoped_patch_is_deterministic_append_only():
    teacher = parse_teacher_repair_plan(
        {
            "trigger_condition": "When two interpretations remain semantically plausible",
            "localized_behavior": "Compare the decisive evidence for those interpretations before committing.",
        },
        evolution_variant=M2E_SCOPED_BEHAVIORAL_PATCH,
    )
    parent = "Keep the original procedure byte-for-byte."
    prompt = construct_scoped_prompt(
        parent, teacher.trigger_condition, teacher.localized_behavior
    )
    assert prompt[: len(parent)] == parent
    parsed = parse_student_candidates(
        {"scoped_patches": [{
            "trigger_condition": teacher.trigger_condition,
            "localized_behavior": teacher.localized_behavior,
        }]},
        parent_prompt=parent,
        context=context(),
        expected_count=1,
        evolution_variant=M2E_SCOPED_BEHAVIORAL_PATCH,
    )
    assert parsed.candidates[0].candidate_prompt == prompt
    assert parsed.candidates[0].trigger_condition == teacher.trigger_condition


def test_m2e_rejects_unconditional_or_answer_specific_trigger():
    with pytest.raises(ValueError, match="unconditional"):
        parse_teacher_repair_plan(
            {
                "trigger_condition": "Always before every answer",
                "localized_behavior": "Compare answer choice A with B.",
            },
            evolution_variant=M2E_SCOPED_BEHAVIORAL_PATCH,
        )
