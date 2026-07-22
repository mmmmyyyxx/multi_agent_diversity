from __future__ import annotations

from typing import Any, Mapping, Sequence


STUDENT_REQUIRED_FIELDS = (
    "candidate_prompt", "target_failure_mechanism", "repair_procedure",
    "preservation_rule", "expected_responsibility_effect",
)


def build_responsibility_context(
    *, target_agent_id: int, assigned_cases: Sequence[Mapping[str, Any]],
    preservation_cases: Sequence[Mapping[str, Any]], representative_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "target_agent_id": int(target_agent_id),
        "assigned_coverage_cases": [dict(row) for row in assigned_cases if int(row.get("peer_gold_vote_count", 0)) == 0],
        "assigned_conversion_cases": [dict(row) for row in assigned_cases if int(row.get("peer_gold_vote_count", 0)) > 0],
        "preservation_unique_correct_cases": [dict(row) for row in preservation_cases if row.get("unique_correct")],
        "preservation_pivotal_vote_cases": [dict(row) for row in preservation_cases if row.get("pivotal_vote_correct")],
        "representative_cases": [dict(row) for row in representative_cases],
    }


def teacher_instruction(
    context: Mapping[str, Any], *, responsibility_conditioned: bool, accuracy_only: bool,
) -> str:
    if accuracy_only:
        objective = (
            "Analyze the target Student's individual errors. Identify a generalizable reasoning failure and propose "
            "an executable decision-procedure repair that improves target accuracy without increasing invalid output."
        )
    elif responsibility_conditioned:
        objective = (
            "Analyze this Student's assigned residual team responsibilities using peer answer histograms, G, H, M, "
            "direct vote-fix value, and soft-utility gain. Explicitly preserve unique-correct and pivotal-correct abilities."
        )
    else:
        objective = (
            "Analyze the target Student's errors under the supplied peer voting state and propose an executable "
            "decision-procedure repair without assigning residual ownership. Preserve supplied protected cases."
        )
    return (
        objective + " Do not assign a preset role, manufacture wording differences, memorize answers, or merely add "
        "longer chain-of-thought.\n\nContext:\n" + str(dict(context))
    )


def critic_instruction() -> str:
    return (
        "Audit whether the proposal targets the assigned residual mechanism, avoids generic chain-of-thought, "
        "does not copy an obvious peer procedure, protects preservation cases, avoids answer memorization, and "
        "specifies an executable task-general reasoning procedure. Return JSON with approved, score, feedback."
    )


def student_instruction(parent_prompt: str, teacher_feedback: str, count: int) -> str:
    return (
        f"Rewrite the parent prompt into {count} candidates that implement the approved repair while preserving "
        "protected abilities. Return strict JSON: {\"candidates\": [{\"candidate_prompt\": \"...\", "
        "\"target_failure_mechanism\": \"...\", \"repair_procedure\": \"...\", "
        "\"preservation_rule\": \"...\", \"expected_responsibility_effect\": \"...\"}]}. "
        f"Parent prompt: {parent_prompt}\nTeacher feedback: {teacher_feedback}"
    )


def validate_student_candidate(candidate: Mapping[str, Any]) -> list[str]:
    return [field for field in STUDENT_REQUIRED_FIELDS if not str(candidate.get(field, "")).strip()]


def critic_rejects_surface_rewrite(proposal: str) -> bool:
    text = str(proposal or "").lower()
    generic = ("think step by step", "be careful", "reason more", "use chain of thought")
    executable = ("if ", "when ", "verify", "compare", "derive", "check")
    return any(marker in text for marker in generic) and not any(marker in text for marker in executable)
