from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config, add_config_arguments, config_from_args
from multi_dataset_diverse_rl.evaluation.output_contract import solver_output_contract
from multi_dataset_diverse_rl.evaluation.solver_output import parse_solver_output
from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient
from multi_dataset_diverse_rl.persistence.artifacts import ArtifactWriter
from multi_dataset_diverse_rl.tasks import get_task_spec
from multi_dataset_diverse_rl.tcs import (
    AccuracyDiagnosisContext,
    PreviousUpdateOutcome,
    TeacherRepairPlan,
    build_critic_request,
    build_student_request,
    build_teacher_request,
    parse_critic_decision,
    parse_student_candidates,
    parse_teacher_repair_plan,
)
from multi_dataset_diverse_rl.utils import extract_json_obj


def _audit(raw: str, parsed: object | None, error: str = "") -> dict:
    excerpt = str(raw or "").strip()
    if len(excerpt) > 600:
        excerpt = excerpt[:300] + "\n...[truncated]...\n" + excerpt[-300:]
    return {
        "response_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "json_extracted": parsed is not None,
        "schema_valid": not error,
        "parse_error": error,
        "response_excerpt": excerpt,
    }


async def run(cfg: Config, client: RoleAwareLLMClient | None = None) -> dict:
    client = client or RoleAwareLLMClient(cfg)
    question = "Which option names the first letter of the alphabet?\n(A) A\n(B) B"
    solver_result = await client.chat_result(
        cfg.models.agent_model,
        solver_output_contract("option_letter"),
        question,
        0.0,
        min(cfg.models.solver_max_tokens, 200),
        "solver",
    )
    solver_answer = parse_solver_output(
        solver_result.text,
        question=question,
        task_spec=get_task_spec("mmlu"),
        answer_format="option_letter",
    )

    context = AccuracyDiagnosisContext(
        target_agent_id=0,
        parent_prompt="Check each option against the question and return one option letter.",
        parent_prompt_hash=hashlib.sha256(b"transport-parent").hexdigest(),
        target_correct_count=0,
        target_error_count=1,
        target_invalid_count=0,
        patterns=(),
        evidence_cases=(),
        previous_outcome=PreviousUpdateOutcome(),
    )
    teacher_raw = await client.chat(
        cfg.models.optimizer_model,
        build_teacher_request(context),
        "Produce the repair proposal.",
        cfg.tcs.teacher_temperature,
        cfg.tcs.teacher_max_tokens,
        "optimizer",
    )
    teacher_json = extract_json_obj(teacher_raw)
    teacher_error = ""
    teacher = None
    try:
        if teacher_json is None:
            raise ValueError("teacher response is not JSON")
        teacher = parse_teacher_repair_plan(
            teacher_json,
            field_max_chars=cfg.tcs.teacher_field_max_chars,
        )
    except (KeyError, TypeError, ValueError) as exc:
        teacher_error = str(exc)

    transport_teacher = teacher or TeacherRepairPlan(
        failure_pattern="The solver commits before comparing every option.",
        repair_rule=(
            "Compare every option, eliminate direct contradictions, verify the selected "
            "letter, and retain viable options when evidence is insufficient."
        ),
        preservation_rule="Keep answers that pass the option-by-option verification.",
    )
    critic_error = ""
    critic = None
    student_error = ""
    candidates = ()
    critic_raw = await client.chat(
        cfg.models.evaluator_model,
        build_critic_request(context, transport_teacher),
        "Audit the proposal.",
        cfg.tcs.critic_temperature,
        cfg.tcs.critic_max_tokens,
        "evaluator",
    )
    critic_json = extract_json_obj(critic_raw)
    try:
        if critic_json is None:
            raise ValueError("critic response is not JSON")
        critic = parse_critic_decision(
            critic_json,
            allowed_case_ids=set(),
            feedback_max_chars=cfg.tcs.critic_feedback_max_chars,
        )
    except (KeyError, TypeError, ValueError) as exc:
        critic_error = str(exc)

    student_raw = await client.chat(
        cfg.models.optimizer_model,
        "Return strict JSON only.",
        build_student_request(
            parent_prompt=context.parent_prompt,
            approved_plan=transport_teacher,
            answer_format="option_letter",
            candidate_count=cfg.tcs.num_candidates_per_parent,
            candidate_prompt_max_chars=cfg.tcs.candidate_prompt_max_chars,
        ),
        cfg.tcs.student_temperature,
        cfg.tcs.student_max_tokens,
        "optimizer",
    )
    student_json = extract_json_obj(student_raw)
    try:
        if student_json is None:
            raise ValueError("student response is not JSON")
        parsed_candidates = parse_student_candidates(
            student_json,
            parent_prompt=context.parent_prompt,
            context=context,
            candidate_prompt_max_chars=cfg.tcs.candidate_prompt_max_chars,
        )
        candidates = parsed_candidates.candidates
    except (KeyError, TypeError, ValueError) as exc:
        student_error = str(exc)

    report = {
        "ok": bool(
            solver_answer.valid
            and teacher is not None
            and critic is not None
            and len(candidates) == cfg.tcs.num_candidates_per_parent
        ),
        "solver": {
            "valid": solver_answer.valid,
            "validity_status": solver_answer.validity_status,
            "raw_final_answer_payload": solver_answer.raw_final_answer_payload,
            "response_hash": solver_answer.response_hash,
        },
        "teacher": {
            **_audit(teacher_raw, teacher_json, teacher_error),
            "proposal": asdict(teacher) if teacher is not None else None,
        },
        "critic": {
            **_audit(critic_raw, critic_json, critic_error),
            "teacher_input_source": "live" if teacher is not None else "fixed_transport_fixture",
            "decision": asdict(critic) if critic is not None else None,
        },
        "student": {
            **_audit(student_raw, student_json, student_error),
            "teacher_input_source": "live" if teacher is not None else "fixed_transport_fixture",
            "requested_count": cfg.tcs.num_candidates_per_parent,
            "schema_valid_count": len(candidates),
        },
        "cost": client.cost_summary(),
    }
    ArtifactWriter(cfg.persistence.out_dir).write_json("role_transport_smoke.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Test solver, Teacher, Critic, and Student transport.")
    return add_config_arguments(value)


def main() -> int:
    cfg = config_from_args(parser().parse_args())
    report = asyncio.run(run(cfg))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
