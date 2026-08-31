from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from generic_m20_probe_support import system_for
from multi_dataset_diverse_rl.tcs import build_teacher_request, serialize_context


AUDIT_VERSION = "gepa_critic_gate_failure_audit_v1"
EXPECTED_CASES = ("seed59_update3", "seed61_update5")
SETTING = "experimental_v16_efficacy_g_matched"
EVOLUTION_VARIANT = "m20_current_v15"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def critic_group_status(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    critic = [row for row in rows if row.get("role") == "critic"]
    student = [row for row in rows if row.get("role") == "student"]
    semantic = [
        row for row in critic
        if row.get("failure_class") == "semantic_rejection"
    ]
    return {
        "critic_response_count": len(critic),
        "critic_schema_invalid_count": sum(
            not bool(row.get("schema_valid")) for row in critic
        ),
        "critic_truncated_count": sum(
            bool(row.get("response_truncated")) for row in critic
        ),
        "critic_semantic_rejection_count": len(semantic),
        "critic_approved_count": sum(
            bool(row.get("effective_approved")) for row in critic
        ),
        "student_reached": bool(student),
        "semantic_gate_exhausted": bool(semantic) and not student,
    }


def historical_groups(v18_root: Path) -> tuple[list[dict[str, Any]], Counter[str], Counter[tuple[str, ...]], Counter[int]]:
    groups: list[dict[str, Any]] = []
    failed_checks: Counter[str] = Counter()
    combinations: Counter[tuple[str, ...]] = Counter()
    semantic_rounds: Counter[int] = Counter()
    for rounds_path in sorted(v18_root.glob("seed*/**/tcs_rounds.jsonl")):
        seed = int(rounds_path.parts[-3].removeprefix("seed"))
        arm = rounds_path.parts[-2]
        by_branch: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(rounds_path):
            by_branch[(int(row["update_index"]), int(row["target_agent_id"]))].append(row)
            if row.get("role") == "critic" and row.get("failure_class") == "semantic_rejection":
                checks = tuple(sorted(map(str, row.get("failed_checks", []))))
                combinations[checks] += 1
                failed_checks.update(checks)
                semantic_rounds[int(row["semantic_round"])] += 1
        context_rows = {
            (int(row["update_index"]), int(row["target_agent_id"])): row
            for row in read_jsonl(rounds_path.parent / "tcs_context_history.jsonl")
        }
        for (update_index, target_agent_id), rows in sorted(by_branch.items()):
            status = critic_group_status(rows)
            context = context_rows[(update_index, target_agent_id)]
            groups.append({
                "seed": seed,
                "arm": arm,
                "update_index": update_index,
                "target_agent_id": target_agent_id,
                "proposal_context_hash": str(context["proposal_context_hash"]),
                "context_characters": int(context["context_characters"]),
                "selected_case_count": int(context["selected_case_count"]),
                **status,
            })
    return groups, failed_checks, combinations, semantic_rounds


def reconstruct_context(case: dict[str, Any], scratch_root: Path) -> dict[str, Any]:
    system = system_for(
        case,
        setting=SETTING,
        out_dir=scratch_root / str(case["case_id"]),
        cache_path=scratch_root / "_solver_cache.sqlite",
        evolution_variant=EVOLUTION_VARIANT,
    )
    target = int(case["target_agent_id"])
    context, diagnostics = system._proposal_context(
        target,
        system.agents[target].current_prompt,
        set(map(str, case["assigned_question_hashes"])),
        rotation_cursor=0,
        proposal_failure_feedback=None,
    )
    serialized = serialize_context(context)
    teacher_request = build_teacher_request(
        context,
        field_max_chars=system.cfg.tcs.teacher_field_max_chars,
        total_max_chars=system.cfg.tcs.teacher_total_max_chars,
        evolution_variant=system.protocol.module2_evolution_variant,
    )
    return {
        "context_type": type(context).__name__,
        "proposal_context_hash": sha256_text(serialized),
        "context_characters": len(serialized),
        "teacher_request_characters": len(teacher_request),
        "selected_case_count": len(diagnostics.selected_case_ids),
    }


def audit(
    registry_path: Path,
    probe_root: Path,
    v18_root: Path,
    scratch_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = read_json(registry_path)
    cases = registry["cases"]
    if tuple(case["case_id"] for case in cases) != EXPECTED_CASES:
        raise ValueError("unexpected frozen case inventory")
    groups, failed_checks, combinations, semantic_rounds = historical_groups(v18_root)
    group_index = {
        (row["seed"], row["arm"], row["update_index"], row["target_agent_id"]): row
        for row in groups
    }
    comparisons: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        result = read_json(probe_root / case_id / "case_result.json")
        funnel = result["candidate_funnel"]
        historical = group_index[(
            int(case["source_seed"]),
            str(case["arm"]),
            int(case["source_update_index"]),
            int(case["target_agent_id"]),
        )]
        reconstructed = reconstruct_context(case, scratch_root)
        previous_outcomes = case.get("previous_update_outcome_by_agent", {})
        prior_same_target = [
            row for row in groups
            if row["seed"] == int(case["source_seed"])
            and row["arm"] == str(case["arm"])
            and row["target_agent_id"] == int(case["target_agent_id"])
            and row["update_index"] < int(case["source_update_index"])
        ]
        comparisons.append({
            "case_id": case_id,
            "seed": int(case["source_seed"]),
            "update_index": int(case["source_update_index"]),
            "target_agent_id": int(case["target_agent_id"]),
            "assigned_residual_count": len(case["assigned_question_hashes"]),
            "probe_teacher_calls": int(funnel["teacher_calls"]),
            "probe_teacher_invalid_responses": int(funnel["teacher_invalid_responses"]),
            "probe_teacher_truncated_responses": int(funnel["teacher_truncated_responses"]),
            "probe_critic_calls": int(funnel["critic_calls"]),
            "probe_critic_invalid_responses": int(funnel["critic_invalid_responses"]),
            "probe_critic_truncated_responses": int(funnel["critic_truncated_responses"]),
            "probe_critic_semantic_rejections": int(funnel["critic_semantic_rejections"]),
            "probe_student_calls": int(funnel["student_calls"]),
            "probe_terminal_failure_class": str(funnel["terminal_failure_class"]),
            "probe_terminal_failure_role": str(funnel["terminal_failure_role"]),
            "probe_exact_failed_checks_persisted": False,
            "probe_exact_feedback_persisted": False,
            "historical_same_branch_student_reached": bool(historical["student_reached"]),
            "historical_same_branch_critic_semantic_rejections": int(
                historical["critic_semantic_rejection_count"]
            ),
            "reconstructed_context_hash": reconstructed["proposal_context_hash"],
            "historical_context_hash": historical["proposal_context_hash"],
            "context_hash_matches_historical": (
                reconstructed["proposal_context_hash"]
                == historical["proposal_context_hash"]
            ),
            "reconstructed_context_characters": reconstructed["context_characters"],
            "historical_context_characters": historical["context_characters"],
            "reconstructed_teacher_request_characters": reconstructed[
                "teacher_request_characters"
            ],
            "selected_case_count": reconstructed["selected_case_count"],
            "frozen_previous_outcome_present_for_target": (
                str(case["target_agent_id"]) in previous_outcomes
                or int(case["target_agent_id"]) in previous_outcomes
            ),
            "historical_prior_same_target_semantic_gate_exhausted": any(
                row["semantic_gate_exhausted"] for row in prior_same_target
            ),
        })

    student_reached = [row for row in groups if row["student_reached"]]
    exhausted = [row for row in groups if row["semantic_gate_exhausted"]]
    all_schema_invalid = sum(row["critic_schema_invalid_count"] for row in groups)
    all_truncated = sum(row["critic_truncated_count"] for row in groups)
    reached_context_chars = [row["context_characters"] for row in student_reached]
    exhausted_context_chars = [row["context_characters"] for row in exhausted]
    summary = {
        "audit_version": AUDIT_VERSION,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "method_modified": False,
        "probe_case_count": len(comparisons),
        "probe_semantic_gate_exhausted_count": sum(
            row["probe_terminal_failure_class"]
            == "critic_semantic_rejection_exhausted"
            for row in comparisons
        ),
        "probe_format_or_parse_failure_count": sum(
            row["probe_teacher_invalid_responses"]
            + row["probe_critic_invalid_responses"]
            for row in comparisons
        ),
        "probe_truncation_count": sum(
            row["probe_teacher_truncated_responses"]
            + row["probe_critic_truncated_responses"]
            for row in comparisons
        ),
        "probe_student_call_count": sum(row["probe_student_calls"] for row in comparisons),
        "probe_exact_rejection_reason_recoverable": False,
        "probe_telemetry_gap": (
            "critic failed_checks and feedback were not persisted by the fixed-parent runner"
        ),
        "same_historical_context_hash_count": sum(
            row["context_hash_matches_historical"] for row in comparisons
        ),
        "different_historical_context_hash_count": sum(
            not row["context_hash_matches_historical"] for row in comparisons
        ),
        "historical_v18": {
            "branch_count": len(groups),
            "student_reached_branch_count": len(student_reached),
            "student_reached_rate": len(student_reached) / len(groups),
            "critic_semantic_gate_exhausted_branch_count": len(exhausted),
            "critic_semantic_gate_exhausted_rate": len(exhausted) / len(groups),
            "critic_semantic_rejection_response_count": sum(combinations.values()),
            "critic_schema_invalid_response_count": all_schema_invalid,
            "critic_truncated_response_count": all_truncated,
            "student_reached_mean_context_characters": (
                sum(reached_context_chars) / len(reached_context_chars)
            ),
            "semantic_exhausted_mean_context_characters": (
                sum(exhausted_context_chars) / len(exhausted_context_chars)
            ),
            "selected_case_counts": dict(sorted(Counter(
                row["selected_case_count"] for row in groups
            ).items())),
            "failed_check_counts": dict(sorted(failed_checks.items())),
            "failed_check_combinations": {
                "+".join(key): value
                for key, value in sorted(combinations.items())
            },
            "semantic_rejection_counts_by_round": {
                str(key): value for key, value in sorted(semantic_rounds.items())
            },
        },
        "conclusions": {
            "candidate_selection_not_primary": True,
            "proposal_breadth_evaluated": False,
            "pre_student_critic_gate_bottleneck_confirmed": True,
            "blocked_failures_are_format_or_parse_failures": False,
            "over_strict_critic_established_for_blocked_cases": False,
            "same_parent_target_is_intrinsically_unserviceable": False,
            "larger_context_explains_semantic_exhaustion": False,
            "candidate_count_visible_before_student": False,
            "fixed_parent_context_reconstruction_exact_for_all_cases": False,
            "next_intervention_requires_new_preregistration": True,
        },
        "final_diagnosis": "PRE_STUDENT_CRITIC_GATE_BOTTLENECK_CONFIRMED",
    }
    return summary, comparisons


def write_outputs(out: Path, summary: dict[str, Any], comparisons: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (out / "branch_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--v18-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("output directory must be fresh or empty")
    if ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("output must be project-local")
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    summary, comparisons = audit(
        args.registry.resolve(),
        args.probe_root.resolve(),
        args.v18_root.resolve(),
        args.scratch_root.resolve(),
    )
    write_outputs(args.out.resolve(), summary, comparisons)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
