"""Common sanitized API, funnel, and evaluation-access accounting."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


FUNNEL_STAGES = (
    "TARGET_SELECTED",
    "TEACHER_PLAN_CREATED",
    "HARD_GATE_PASS",
    "HARD_GATE_REJECT",
    "SEMANTIC_CRITIC_PASS",
    "SEMANTIC_CRITIC_REJECT",
    "STUDENT_REACHED",
    "STUDENT_CANDIDATE_VALID",
    "STUDENT_CANDIDATE_INVALID",
    "ROLLOUT_COMPLETED",
    "COMMON_SAFE_FEASIBLE",
    "COMMON_SAFE_REJECT",
    "BRANCH_WINNER",
    "WOULD_COMMIT",
    "COMMIT",
    "VALIDATION_EVALUATED",
    "TEST_EVALUATED",
    "NOT_APPLICABLE",
)


def summarize_funnel(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(events)
    unknown = sorted({row.get("stage") for row in rows} - set(FUNNEL_STAGES))
    if unknown:
        raise ValueError(f"unknown funnel stages: {unknown}")
    active = [row for row in rows if row.get("stage") != "NOT_APPLICABLE"]
    counts = Counter(row["stage"] for row in active)
    branches = {row.get("branch_id") for row in active if row.get("branch_id")}
    updated = {
        row.get("target_member")
        for row in active
        if row.get("stage") == "COMMIT" and row.get("target_member") is not None
    }
    attempted = len(branches)
    student = counts["STUDENT_REACHED"]
    valid = counts["STUDENT_CANDIDATE_VALID"]
    feasible = counts["COMMON_SAFE_FEASIBLE"]
    would_commit = counts["WOULD_COMMIT"]
    commits = counts["COMMIT"]

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "schema_version": "optimization_funnel_summary_v1",
        "branches_attempted": attempted,
        "student_reach": student,
        "valid_candidates": valid,
        "feasible_candidates": feasible,
        "would_commit": would_commit,
        "commits": commits,
        "distinct_members_updated": len(updated),
        "rates": {
            "reach_per_branch": ratio(student, attempted),
            "valid_per_student": ratio(valid, student),
            "feasible_per_student": ratio(feasible, student),
            "feasible_per_branch": ratio(feasible, attempted),
            "would_commit_per_branch": ratio(would_commit, attempted),
            "commit_per_branch": ratio(commits, attempted),
        },
        "stage_counts": dict(sorted(counts.items())),
        "not_applicable_event_count": len(rows) - len(active),
    }


def summarize_api_ledger(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate_count: int = 0,
    feasible_candidate_count: int = 0,
    accepted_update_count: int = 0,
) -> dict[str, Any]:
    entries = list(rows)
    role_counts = Counter(str(row.get("role")) for row in entries)
    failure_counts = Counter(
        str(row.get("failure_category"))
        for row in entries
        if not row.get("success")
    )
    total_tokens = sum(int(row.get("total_tokens", 0)) for row in entries)
    by_role = {
        role: {
            "calls": count,
            "input_tokens": sum(
                int(row.get("input_tokens", 0))
                for row in entries
                if str(row.get("role")) == role
            ),
            "output_tokens": sum(
                int(row.get("output_tokens", 0))
                for row in entries
                if str(row.get("role")) == role
            ),
            "total_tokens": sum(
                int(row.get("total_tokens", 0))
                for row in entries
                if str(row.get("role")) == role
            ),
        }
        for role, count in sorted(role_counts.items())
    }
    def per_unit(denominator: int) -> float | None:
        return total_tokens / denominator if denominator else None

    return {
        "schema_version": "api_ledger_summary_v1",
        "call_count": len(entries),
        "by_role": by_role,
        "failure_categories": dict(sorted(failure_counts.items())),
        "total_tokens": total_tokens,
        "tokens_per_candidate": per_unit(candidate_count),
        "tokens_per_feasible_candidate": per_unit(feasible_candidate_count),
        "tokens_per_accepted_update": per_unit(accepted_update_count),
    }


def validate_evaluation_access(
    rows: Iterable[Mapping[str, Any]], *, final_freeze_phase: str = "FINAL_FROZEN"
) -> dict[str, Any]:
    entries = list(rows)
    errors: list[str] = []
    test_rows = [row for row in entries if row.get("split") == "test"]
    validation_rows = [row for row in entries if row.get("split") == "validation"]
    for index, row in enumerate(entries):
        if row.get("split") not in {"train", "validation", "test"}:
            errors.append(f"row {index}: invalid split")
        if not row.get("purpose") or not row.get("phase"):
            errors.append(f"row {index}: purpose and phase are required")
        if int(row.get("row_count", -1)) < 0:
            errors.append(f"row {index}: row_count must be non-negative")
        if row.get("split") == "validation" and row.get("selection_frozen_before_access") is not True:
            errors.append(f"row {index}: validation accessed before selection freeze")
        if row.get("split") == "test":
            if row.get("selection_frozen_before_access") is not True:
                errors.append(f"row {index}: test accessed before final freeze")
            if row.get("phase") != final_freeze_phase:
                errors.append(f"row {index}: test phase must be {final_freeze_phase}")
    return {
        "schema_version": "evaluation_access_summary_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "validation_access_count": len(validation_rows),
        "test_access_count": len(test_rows),
        "test_calls_zero": not test_rows,
    }
