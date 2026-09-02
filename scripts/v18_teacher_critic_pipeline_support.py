from __future__ import annotations

import hashlib
import json
import re
import types
from dataclasses import asdict, dataclass, field
from typing import Any

from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.tcs import (
    STUDENT_SYSTEM_PROMPT,
    TeacherRepairPlan,
    extract_json_obj,
    teacher_repair_plan_hash,
)
from scripts.v18_safety_only_critic_pilot_support import (
    _plan_from_critic_prompt,
    canonical_hash,
)


AUTH_ENV = "V18_TEACHER_CRITIC_PIPELINE_AUTHORIZED"
ARMS = ("A_CANONICAL", "B_TEACHER_CLEAN", "C_NO_SEMANTIC_CRITIC", "D_ADVISORY_CRITIC")

TEACHER_CLEAN_SUFFIX = """

Teacher-Clean experimental constraint:
- failure_pattern diagnoses one reasoning or decision failure.
- repair_rule gives one executable reasoning or decision correction.
- preservation_rule describes only reasoning or decision behavior that must remain unchanged.
- In every field, do not discuss or modify the solver output format, FINAL_ANSWER, answer labels, or the immutable output interface.
- Do not hard-code answers or copy a specific peer prompt, procedure, reasoning trace, or steps.
- Do not predict candidate accuracy, collateral loss, validation transfer, or whether a candidate should be committed.
Return the existing strict TeacherRepairPlan schema unchanged.
""".strip()

_GOLD = re.compile(
    r"\b(?:gold|correct)\s+(?:answer|label)\s*(?:is|=|:)\s*(?:option\s*)?[A-Z0-9]\b",
    re.IGNORECASE,
)
_SAMPLE_ID = re.compile(
    r"\b(?:case|sample|question)[ _-]?(?:id|hash)\s*(?:is|=|:)\s*[0-9a-f]{6,}\b",
    re.IGNORECASE,
)
_FIXED_ANSWER = re.compile(
    r"\b(?:always|default(?:\s+to)?|regardless[^.;]{0,24})\s+"
    r"(?:choose|select|return|answer)\s+(?:(?:option|label|answer)\s*)?[A-Z]\b",
    re.IGNORECASE,
)
_PEER_COPY = re.compile(
    r"\b(?:copy|imitate|reproduce|follow)\s+(?:the\s+)?(?:specific\s+)?"
    r"(?:peer|member|agent)(?:'s)?\s+(?:prompt|procedure|reasoning|trace|steps)\b",
    re.IGNORECASE,
)
_OUTPUT_MODIFY = re.compile(
    r"\b(?:change|modify|rewrite|replace|ignore|remove|omit|alter)\s+(?:the\s+)?"
    r"(?:(?:solver|response)\s+)?(?:output\s+)?(?:format|interface|contract)\b|"
    r"\b(?:emit|include|remove|omit|rewrite)\s+(?:the\s+)?FINAL[_ ]ANSWER\b",
    re.IGNORECASE,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def teacher_clean_request(request: str) -> str:
    return request.rstrip() + "\n\n" + TEACHER_CLEAN_SUFFIX


def deterministic_hard_gate(plan: TeacherRepairPlan | dict[str, Any]) -> dict[str, Any]:
    payload = asdict(plan) if isinstance(plan, TeacherRepairPlan) else dict(plan)
    required = ("failure_pattern", "repair_rule", "preservation_rule")
    if any(not isinstance(payload.get(name), str) or not payload[name].strip() for name in required):
        return {
            "pass": False,
            "category": "schema",
            "marker_type": "required_field_missing_or_empty",
            "field_location": next(
                name for name in required
                if not isinstance(payload.get(name), str) or not str(payload.get(name, "")).strip()
            ),
        }
    checks = (
        ("anti_cheating", "explicit_gold_answer_leakage", _GOLD),
        ("anti_cheating", "explicit_sample_id_memorization", _SAMPLE_ID),
        ("anti_cheating", "explicit_fixed_answer_hard_coding", _FIXED_ANSWER),
        ("anti_cheating", "detectable_direct_peer_procedure_copying", _PEER_COPY),
        ("output_contract", "explicit_output_contract_modification", _OUTPUT_MODIFY),
    )
    for field_name in required:
        text = payload[field_name]
        for category, marker, pattern in checks:
            if pattern.search(text):
                return {
                    "pass": False,
                    "category": category,
                    "marker_type": marker,
                    "field_location": field_name,
                }
    return {"pass": True, "category": "none", "marker_type": "none", "field_location": "none"}


def advisory_critic_request(canonical_request: str) -> str:
    marker = "DiagnosisContext:\n"
    evidence = canonical_request.split(marker, 1)[1] if marker in canonical_request else canonical_request
    schema = {
        "evidence_concerns": [],
        "actionability_concerns": [],
        "feedback": "",
    }
    return (
        "Act only as a non-blocking semantic advisor. Identify evidence-grounding or "
        "actionability concerns and provide one concise revision suggestion. Do not "
        "predict preservation loss, collateral risk, candidate performance, validation "
        "transfer, or final acceptance. Your decision never blocks Student execution. "
        "Return strict JSON with exactly these fields: "
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
        + "\nDiagnosisContext:\n"
        + evidence
    )


def parse_advisory_payload(text: str) -> dict[str, Any] | None:
    payload = extract_json_obj(text)
    if not isinstance(payload, dict) or set(payload) != {
        "evidence_concerns", "actionability_concerns", "feedback"
    }:
        return None
    for key in ("evidence_concerns", "actionability_concerns"):
        if not isinstance(payload[key], list) or any(not isinstance(item, str) for item in payload[key]):
            return None
    if not isinstance(payload["feedback"], str):
        return None
    return {
        "evidence_concerns": tuple(item.strip() for item in payload["evidence_concerns"] if item.strip()),
        "actionability_concerns": tuple(item.strip() for item in payload["actionability_concerns"] if item.strip()),
        "feedback": payload["feedback"].strip(),
    }


def _role(system_prompt: str, client_role: str, logical_role: str | None) -> str:
    if logical_role:
        return logical_role
    if client_role == "solver":
        return "solver"
    if client_role == "evaluator":
        return "critic"
    if system_prompt in {STUDENT_SYSTEM_PROMPT, "Return strict JSON only."}:
        return "student"
    return "teacher"


@dataclass
class CleanTeacherReplay:
    results: dict[str, LLMCallResult] = field(default_factory=dict)
    replay_hits: int = 0


@dataclass
class ArmController:
    arm: str
    clean_replay: CleanTeacherReplay
    hard_gate_decisions: list[dict[str, Any]] = field(default_factory=list)
    advisory_decisions: list[dict[str, Any]] = field(default_factory=list)
    latest_advisory_feedback: str = ""
    clean_teacher_replay_hits: int = 0
    synthetic_hard_gate_calls: int = 0
    advisory_api_calls: int = 0


def install_pipeline_arm(system: Any, controller: ArmController) -> None:
    if controller.arm not in ARMS:
        raise ValueError(controller.arm)
    original = system._chat

    async def arm_chat(
        self: Any,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
        client_role: str,
        logical_role: str | None = None,
    ) -> LLMCallResult:
        role = _role(system_prompt, client_role, logical_role)
        clean = controller.arm != "A_CANONICAL"
        if role == "teacher" and clean:
            adjusted_system = teacher_clean_request(system_prompt)
            key = canonical_hash([
                model, adjusted_system, user_prompt, temperature, max_tokens,
                client_role, role,
            ])
            if key in controller.clean_replay.results:
                controller.clean_replay.replay_hits += 1
                controller.clean_teacher_replay_hits += 1
                return controller.clean_replay.results[key]
            result = await original(
                model, adjusted_system, user_prompt, temperature, max_tokens,
                client_role, logical_role,
            )
            controller.clean_replay.results[key] = result
            return result

        if role == "critic" and controller.arm in {
            "C_NO_SEMANTIC_CRITIC", "D_ADVISORY_CRITIC"
        }:
            plan = _plan_from_critic_prompt(system_prompt)
            gate = deterministic_hard_gate(plan)
            gate_row = {
                "teacher_plan_hash": teacher_repair_plan_hash(plan),
                **gate,
            }
            controller.hard_gate_decisions.append(gate_row)
            controller.synthetic_hard_gate_calls += 1
            if not gate["pass"]:
                failed_check = (
                    "shortcut_or_copying"
                    if gate["category"] == "anti_cheating"
                    else "preservation_or_output_risk"
                )
                return LLMCallResult(
                    json.dumps({
                        "failed_checks": [failed_check],
                        "risk_case_ids": [],
                        "feedback": f"deterministic hard gate: {gate['marker_type']}",
                    }, separators=(",", ":")),
                    0, 0, 0, 0.0, "stop",
                )
            if controller.arm == "C_NO_SEMANTIC_CRITIC":
                return LLMCallResult(
                    '{"failed_checks":[],"risk_case_ids":[],"feedback":""}',
                    0, 0, 0, 0.0, "stop",
                )
            advisory_result = await original(
                model,
                advisory_critic_request(system_prompt),
                "Provide non-blocking semantic advice.",
                temperature,
                max_tokens,
                client_role,
                logical_role,
            )
            controller.advisory_api_calls += 1
            advisory = parse_advisory_payload(advisory_result.text)
            if advisory is None:
                return advisory_result
            feedback = advisory["feedback"]
            controller.latest_advisory_feedback = feedback
            controller.advisory_decisions.append({
                "teacher_plan_hash": teacher_repair_plan_hash(plan),
                "evidence_concern_count": len(advisory["evidence_concerns"]),
                "actionability_concern_count": len(advisory["actionability_concerns"]),
                "feedback_hash": sha256_text(feedback) if feedback else "",
                "effective_block": False,
            })
            return LLMCallResult(
                json.dumps({
                    "failed_checks": [],
                    "risk_case_ids": [],
                    "feedback": feedback,
                }, separators=(",", ":")),
                advisory_result.prompt_tokens,
                advisory_result.completion_tokens,
                advisory_result.total_tokens,
                advisory_result.latency_seconds,
                advisory_result.finish_reason,
            )

        if role == "student" and controller.arm == "D_ADVISORY_CRITIC" and controller.latest_advisory_feedback:
            user_prompt = (
                user_prompt
                + "\n\nNonBindingAdvisoryCriticFeedback:\n"
                + controller.latest_advisory_feedback
            )
        return await original(
            model, system_prompt, user_prompt, temperature, max_tokens,
            client_role, logical_role,
        )

    system._chat = types.MethodType(arm_chat, system)


def select_arm(summary: dict[str, dict[str, float]]) -> dict[str, Any]:
    a, b = summary["A_CANONICAL"], summary["B_TEACHER_CLEAN"]
    c, d = summary["C_NO_SEMANTIC_CRITIC"], summary["D_ADVISORY_CRITIC"]
    prefer_d = (
        d["student_reach_count"] > max(a["student_reach_count"], b["student_reach_count"])
        and d["feasible_per_branch"] > max(a["feasible_per_branch"], b["feasible_per_branch"])
        and d["validation_vote_delta_sum"] >= c["validation_vote_delta_sum"]
        and d["train_vote_loss_sum"] <= c["train_vote_loss_sum"]
        and d["validation_target_delta_sum"] >= c["validation_target_delta_sum"]
    )
    d_quality_advantage = (
        d["feasible_per_branch"] > c["feasible_per_branch"]
        and d["zero_loss_feasible_count"] > c["zero_loss_feasible_count"]
        and d["validation_target_delta_sum"] >= c["validation_target_delta_sum"]
    )
    prefer_c = (
        c["validation_vote_delta_sum"] >= d["validation_vote_delta_sum"]
        and not d_quality_advantage
    )
    removal_collapses = (
        c["validation_vote_delta_sum"] <= b["validation_vote_delta_sum"] - 2
        and d["validation_vote_delta_sum"] <= b["validation_vote_delta_sum"] - 2
    )
    prefer_b = (
        b["student_reach_count"] > a["student_reach_count"]
        and b["feasible_per_branch"] >= a["feasible_per_branch"]
        and b["validation_vote_delta_sum"] >= a["validation_vote_delta_sum"]
        and removal_collapses
    )
    if prefer_d:
        selected, reason = "D_ADVISORY_CRITIC", "advisory feedback adds throughput and quality without veto"
    elif prefer_c:
        selected, reason = "C_NO_SEMANTIC_CRITIC", "advisory feedback lacks stable quality benefit"
    elif prefer_b:
        selected, reason = "B_TEACHER_CLEAN", "Teacher cleanup helps while veto removal collapses validation quality"
    else:
        selected, reason = "A_CANONICAL", "new arms do not meet frozen replacement criteria"
    return {
        "selected_arm": selected,
        "reason": reason,
        "prefer_d": prefer_d,
        "prefer_c": prefer_c,
        "prefer_b": prefer_b,
        "oracle_used_for_selection": False,
        "test_used_for_selection": False,
    }
