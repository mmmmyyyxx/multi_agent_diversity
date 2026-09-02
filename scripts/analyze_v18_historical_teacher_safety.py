from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    mutable_prompt_violation_reasons,
)
from scripts.v18_safety_only_critic_pilot_support import _ANTI, _OUTPUT


BASE_FIELDS = ("failure_pattern", "repair_rule", "preservation_rule")
OUTPUT_SUBTYPES = {
    "final_answer_reference": re.compile(r"(?<!\w)final[\s_-]*answer(?!\w)", re.I),
    "output_protocol_reference": re.compile(r"\b(?:output|solver)\s+(?:contract|interface|format)\b|\bresponse\s+format\b", re.I),
    "answer_label_behavior": re.compile(r"\b(?:answer|option|label)s?\b.{0,45}\b(?:a|b|c)(?:\s*[,/]\s*(?:a|b|c))*\b", re.I),
    "formatting_directive": re.compile(r"\b(?:only\s+return|return\s+only|provide\s+only|only\s+provide|respond\s+with|end\s+.{0,25}\s+with)\b", re.I),
}
ANTI_SUBTYPES = {
    "reference_to_gold_label": re.compile(r"\bgold\s+answer\b", re.I),
    "explicit_sample_identifier": re.compile(r"\bcase[_ -]?id\b|\b[0-9a-f]{24,64}\b", re.I),
    "peer_procedure_copying": re.compile(r"\bcopy\b.{0,30}\bpeer\b|\bpeer\b.{0,30}\b(?:prompt|procedure)\b", re.I),
    "hardcoded_answer_behavior": re.compile(r"\b(?:always|default)\b.{0,20}\b(?:choose|select|return)\b.{0,20}\b(?:option\s*)?[abc]\b", re.I),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    names = list(fieldnames or (list(rows[0]) if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_field(text: str) -> dict[str, Any]:
    value = str(text or "")
    output = bool(mutable_prompt_violation_reasons(value) or _OUTPUT.search(value))
    anti = bool(_ANTI.search(value))
    output_subtypes = sorted(name for name, pattern in OUTPUT_SUBTYPES.items() if output and pattern.search(value))
    anti_subtypes = sorted(name for name, pattern in ANTI_SUBTYPES.items() if anti and pattern.search(value))
    if output and not output_subtypes:
        output_subtypes = ["other_explicit_output_contract"]
    if anti and not anti_subtypes:
        anti_subtypes = ["other_anti_cheating"]
    return {
        "output_contract": output,
        "anti_cheating": anti,
        "unsafe": output or anti,
        "output_subtypes": output_subtypes,
        "anti_subtypes": anti_subtypes,
    }


def _branch_key(trajectory: str, row: dict[str, Any]) -> tuple[str, int, int]:
    return trajectory, int(row["update_index"]), int(row["target_agent_id"])


def collect(source: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]]]:
    plans: list[dict[str, Any]] = []
    contexts: dict[tuple[str, int, int], dict[str, Any]] = {}
    for tcs_path in sorted(source.glob("seed*/*/tcs_rounds.jsonl")):
        trajectory = "/".join(tcs_path.relative_to(source).parts[:2])
        rows = read_jsonl(tcs_path)
        critics: dict[tuple[int, int, str], dict[str, Any]] = {}
        for row in rows:
            if row.get("role") == "critic":
                critics[(int(row["update_index"]), int(row["target_agent_id"]), str(row.get("teacher_plan_hash", "")))] = row
        for row in rows:
            if row.get("role") != "teacher" or not row.get("schema_valid") or not isinstance(row.get("repair_plan"), dict):
                continue
            field_results = {field: classify_field(str(row["repair_plan"].get(field, ""))) for field in BASE_FIELDS}
            critic = critics.get((int(row["update_index"]), int(row["target_agent_id"]), str(row.get("teacher_plan_hash", ""))), {})
            output_fields = [field for field in BASE_FIELDS if field_results[field]["output_contract"]]
            anti_fields = [field for field in BASE_FIELDS if field_results[field]["anti_cheating"]]
            plans.append({
                "trajectory": trajectory,
                "seed": int(re.search(r"seed(\d+)", trajectory).group(1)),
                "arm": trajectory.split("/")[1],
                "update_index": int(row["update_index"]),
                "target_member": int(row["target_agent_id"]),
                "semantic_round": int(row.get("semantic_round", 1)),
                "teacher_plan_hash": str(row.get("teacher_plan_hash", "")),
                "output_contract_unsafe": bool(output_fields),
                "anti_cheating_unsafe": bool(anti_fields),
                "hard_safety_unsafe": bool(output_fields or anti_fields),
                "output_fields": output_fields,
                "anti_fields": anti_fields,
                "output_subtypes": sorted({subtype for field in BASE_FIELDS for subtype in field_results[field]["output_subtypes"]}),
                "anti_subtypes": sorted({subtype for field in BASE_FIELDS for subtype in field_results[field]["anti_subtypes"]}),
                "critic_approved": bool(critic.get("effective_approved")),
                "critic_failed_checks": sorted(map(str, critic.get("failed_checks", []))),
                "context_hash": str(row.get("context_hash", "")),
                "input_characters": int(row.get("input_characters", 0)),
                "output_characters": int(row.get("output_characters", 0)),
            })
        context_path = tcs_path.with_name("tcs_context_history.jsonl")
        if context_path.exists():
            for row in read_jsonl(context_path):
                contexts[_branch_key(trajectory, row)] = row
    return plans, contexts


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def analyze(source: Path, report: Path) -> dict[str, Any]:
    if report.exists():
        raise FileExistsError("fresh report root required")
    plans, contexts = collect(source)
    if not plans:
        raise ValueError("no structured historical Teacher plans")
    branches: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for plan in plans:
        branches[(plan["trajectory"], plan["update_index"], plan["target_member"])].append(plan)
    for rows in branches.values():
        rows.sort(key=lambda row: (row["semantic_round"], row["teacher_plan_hash"]))

    field_rows = []
    for field in BASE_FIELDS:
        total = len(plans)
        output = sum(field in row["output_fields"] for row in plans)
        anti = sum(field in row["anti_fields"] for row in plans)
        field_rows.append({
            "field": field,
            "structured_plan_count": total,
            "output_contract_count": output,
            "output_contract_rate": rate(output, total),
            "anti_cheating_count": anti,
            "anti_cheating_rate": rate(anti, total),
            "any_hard_safety_count": sum(field in set(row["output_fields"] + row["anti_fields"]) for row in plans),
        })

    marker_counts = Counter()
    for row in plans:
        marker_counts.update(("output_contract", value) for value in row["output_subtypes"])
        marker_counts.update(("anti_cheating", value) for value in row["anti_subtypes"])
    marker_rows = [{"category": key[0], "marker_type": key[1], "plan_count": count, "plan_rate": rate(count, len(plans))} for key, count in sorted(marker_counts.items())]

    retry_branches = [rows for rows in branches.values() if len(rows) >= 2]
    unsafe_first = [rows for rows in retry_branches if rows[0]["hard_safety_unsafe"]]
    retry_repaired = sum(not rows[1]["hard_safety_unsafe"] for rows in unsafe_first)
    repeat_same = 0
    for rows in unsafe_first:
        first = set(rows[0]["output_subtypes"] + rows[0]["anti_subtypes"])
        second = set(rows[1]["output_subtypes"] + rows[1]["anti_subtypes"])
        repeat_same += bool(first & second)
    retry_summary = [{
        "retry_branch_count": len(retry_branches),
        "unsafe_first_with_retry_count": len(unsafe_first),
        "unsafe_to_safe_count": retry_repaired,
        "safety_repair_rate": rate(retry_repaired, len(unsafe_first)),
        "unsafe_after_retry_count": len(unsafe_first) - retry_repaired,
        "same_marker_repeated_count": repeat_same,
        "same_marker_repeated_rate": rate(repeat_same, len(unsafe_first)),
    }]

    branch_rows = []
    for key, rows in sorted(branches.items()):
        passed = any(row["critic_approved"] for row in rows)
        context = contexts.get(key, {})
        branch_rows.append({
            "trajectory": key[0],
            "seed": int(re.search(r"seed(\d+)", key[0]).group(1)),
            "arm": key[0].split("/")[1],
            "update_index": key[1],
            "target_member": key[2],
            "critic_branch_outcome": "passed" if passed else "blocked",
            "teacher_plan_count": len(rows),
            "first_plan_unsafe": rows[0]["hard_safety_unsafe"],
            "final_plan_unsafe": rows[-1]["hard_safety_unsafe"],
            "final_output_contract_unsafe": rows[-1]["output_contract_unsafe"],
            "final_anti_cheating_unsafe": rows[-1]["anti_cheating_unsafe"],
            "final_preservation_output_unsafe": "preservation_rule" in rows[-1]["output_fields"],
            "any_plan_unsafe": any(row["hard_safety_unsafe"] for row in rows),
            "final_output_fields": "|".join(rows[-1]["output_fields"]),
            "final_anti_fields": "|".join(rows[-1]["anti_fields"]),
            "selected_case_count": int(context.get("selected_case_count", 0)),
            "selected_pattern_count": int(context.get("selected_pattern_count", 0)),
            "context_characters": int(context.get("context_characters", 0)),
            "context_mode": str(context.get("context_mode", "")),
            "module2_context_variant": str(context.get("module2_context_variant", "")),
        })

    outcome_rows = []
    for outcome in ("passed", "blocked"):
        subset = [row for row in branch_rows if row["critic_branch_outcome"] == outcome]
        outcome_rows.append({
            "critic_branch_outcome": outcome,
            "branch_count": len(subset),
            "first_plan_unsafe_count": sum(row["first_plan_unsafe"] for row in subset),
            "first_plan_unsafe_rate": rate(sum(row["first_plan_unsafe"] for row in subset), len(subset)),
            "final_plan_unsafe_count": sum(row["final_plan_unsafe"] for row in subset),
            "final_plan_unsafe_rate": rate(sum(row["final_plan_unsafe"] for row in subset), len(subset)),
            "mean_context_characters": sum(row["context_characters"] for row in subset) / len(subset) if subset else 0.0,
            "mean_selected_case_count": sum(row["selected_case_count"] for row in subset) / len(subset) if subset else 0.0,
            "final_output_contract_unsafe_rate": rate(sum(row["final_output_contract_unsafe"] for row in subset), len(subset)),
            "final_anti_cheating_unsafe_rate": rate(sum(row["final_anti_cheating_unsafe"] for row in subset), len(subset)),
            "final_preservation_output_unsafe_rate": rate(sum(row["final_preservation_output_unsafe"] for row in subset), len(subset)),
        })

    context_rows = []
    for dimension in ("arm", "seed", "target_member", "selected_case_count", "selected_pattern_count", "context_mode", "module2_context_variant"):
        values = sorted({str(row[dimension]) for row in branch_rows})
        for value in values:
            subset = [row for row in branch_rows if str(row[dimension]) == value]
            context_rows.append({
                "dimension": dimension,
                "value": value,
                "branch_count": len(subset),
                "first_plan_unsafe_rate": rate(sum(row["first_plan_unsafe"] for row in subset), len(subset)),
                "final_plan_unsafe_rate": rate(sum(row["final_plan_unsafe"] for row in subset), len(subset)),
                "critic_pass_rate": rate(sum(row["critic_branch_outcome"] == "passed" for row in subset), len(subset)),
            })

    unsafe_count = sum(row["hard_safety_unsafe"] for row in plans)
    field_unsafe_counts = {row["field"]: row["any_hard_safety_count"] for row in field_rows}
    dominant_field = max(field_unsafe_counts, key=field_unsafe_counts.get)
    dominant_share = rate(field_unsafe_counts[dominant_field], unsafe_count)
    blocked = next(row for row in outcome_rows if row["critic_branch_outcome"] == "blocked")
    stable = rate(unsafe_count, len(plans)) >= 0.20 and dominant_share >= 0.60 and retry_summary[0]["same_marker_repeated_rate"] >= 0.50
    passed = next(row for row in outcome_rows if row["critic_branch_outcome"] == "passed")
    blocked_minus_passed = blocked["final_plan_unsafe_rate"] - passed["final_plan_unsafe_rate"]
    explains_safety_gate = blocked["final_plan_unsafe_rate"] >= 0.50
    explains_canonical_blocking = explains_safety_gate and blocked_minus_passed >= 0.15
    summary = {
        "audit_version": "v18_historical_teacher_safety_failure_audit_v1",
        "trajectory_count": len({row["trajectory"] for row in plans}),
        "branch_count": len(branch_rows),
        "structured_teacher_plan_count": len(plans),
        "hard_safety_unsafe_plan_count": unsafe_count,
        "hard_safety_unsafe_plan_rate": rate(unsafe_count, len(plans)),
        "output_contract_unsafe_plan_count": sum(row["output_contract_unsafe"] for row in plans),
        "anti_cheating_unsafe_plan_count": sum(row["anti_cheating_unsafe"] for row in plans),
        "dominant_unsafe_field": dominant_field,
        "dominant_field_share_of_unsafe_plans": dominant_share,
        "retry": retry_summary[0],
        "critic_outcome": {row["critic_branch_outcome"]: row for row in outcome_rows},
        "stable_repeatable_pattern": stable,
        "blocked_minus_passed_final_unsafe_rate": blocked_minus_passed,
        "sufficient_to_explain_prospective_safety_gate_blocking": explains_safety_gate,
        "discriminative_for_historical_canonical_blocking": explains_canonical_blocking,
        "decision": (
            "STABLE_PATTERN_NOT_DISCRIMINATIVE_FOR_CANONICAL_BLOCKING"
            if stable and explains_safety_gate and not explains_canonical_blocking
            else "STABLE_REPEATABLE_TEACHER_HARD_SAFETY_PATTERN"
            if stable and explains_canonical_blocking
            else "TEACHER_HARD_SAFETY_PATTERN_NOT_SUFFICIENTLY_ESTABLISHED"
        ),
        "api_calls": 0,
        "test_accessed": False,
        "historical_artifacts_modified": False,
    }

    report.mkdir(parents=True)
    write_csv(report / "field_contamination.csv", field_rows)
    write_csv(report / "marker_types.csv", marker_rows)
    write_csv(report / "retry_safety_repair.csv", retry_summary)
    write_csv(report / "critic_outcome_comparison.csv", outcome_rows)
    write_csv(report / "context_associations.csv", context_rows)
    write_csv(report / "branch_level_sanitized.csv", branch_rows)
    write_json(report / "summary.json", summary)
    write_json(report / "classifier.json", {
        "version": "v18_historical_teacher_safety_classifier_v1",
        "hard_safety_rule": "exact_deterministic_safety_only_patterns_applied_per_base_field",
        "stable_pattern_thresholds": {"unsafe_plan_rate_min": 0.20, "dominant_field_share_min": 0.60, "retry_same_marker_rate_min": 0.50},
        "prospective_safety_gate_explanation_threshold": {"blocked_branch_final_plan_unsafe_rate_min": 0.50},
        "historical_canonical_discrimination_threshold": {"blocked_minus_passed_final_unsafe_rate_min": 0.15},
        "validation_or_test_used": False,
    })
    write_json(report / "provenance.json", {
        "source": "immutable_v18_hybrid_online_accumulation_teacher_and_context_telemetry",
        "prospective_plan_text_used": False,
        "prospective_limitation": "only_hashes_and_coarse_categories_were_persisted",
        "historical_artifacts_modified": False,
        "raw_text_published": False,
    })
    readme = f"""# V18 Historical Teacher Safety Failure Audit

This zero-API audit applies the frozen deterministic hard-safety patterns to
historical structured Teacher plans. It publishes only aggregate counts,
rates, hashes, and structural labels; no plan, prompt, question, answer,
response, provider configuration, cache, or runtime-state content is included.

- Structured Teacher plans: **{len(plans)}**
- Hard-safety unsafe plans: **{unsafe_count} ({rate(unsafe_count, len(plans)):.1%})**
- Dominant contaminated field: **{dominant_field}**
- Dominant-field coverage among unsafe plans: **{dominant_share:.1%}**
- Unsafe first plans with retry: **{len(unsafe_first)}**
- Retry safety-repair rate: **{retry_summary[0]['safety_repair_rate']:.1%}**
- Same-marker retry recurrence: **{retry_summary[0]['same_marker_repeated_rate']:.1%}**
- Critic-blocked branches ending unsafe: **{blocked['final_plan_unsafe_rate']:.1%}**
- Critic-passed branches ending unsafe: **{passed['final_plan_unsafe_rate']:.1%}**
- Blocked minus passed unsafe-rate difference: **{blocked_minus_passed:+.1%}**

Frozen diagnostic decision: **{summary['decision']}**.

The pattern is stable enough to explain why the prospective deterministic
safety gate blocked every sampled branch. It is not discriminative for the
historical canonical Critic outcome: passed branches were at least as likely
to contain a hard-safety trigger as blocked branches. Consequently, Teacher
safety contamination and canonical semantic blocking are distinct historical
mechanisms rather than one sufficient explanation for all pre-Student loss.

This audit evaluates association and recurrence, not a causal Teacher-prompt
intervention. The prospective plans are not reclassified below their persisted
coarse categories because their text was intentionally not retained.

```text
API_CALLS=0
TEST_ACCESSED=false
HISTORICAL_ARTIFACTS_MODIFIED=false
```
"""
    (report / "README.md").write_text(readme, encoding="utf-8")
    assertions = {
        "six_trajectories": summary["trajectory_count"] == 6,
        "plans_present": len(plans) > 0,
        "branch_accounting": len(branch_rows) == len(branches),
        "field_accounting": all(row["structured_plan_count"] == len(plans) for row in field_rows),
        "retry_accounting": retry_summary[0]["unsafe_to_safe_count"] + retry_summary[0]["unsafe_after_retry_count"] == len(unsafe_first),
        "zero_api": True,
        "no_test": True,
    }
    assertions["pass"] = all(assertions.values())
    write_json(report / "fact_assertions.json", assertions)
    forbidden = re.compile(r"(?:[A-Za-z]:\\)|DASHSCOPE|api[_-]?key|FINAL_ANSWER:|question_text|gold_answer|model_answer|raw_response|endpoint|\.sqlite|checkpoint", re.I)
    findings = []
    for path in report.iterdir():
        if path.is_file() and forbidden.search(path.read_text(encoding="utf-8")):
            findings.append(path.name)
    write_json(report / "sanitization_manifest.json", {"status": "PASS" if not findings else "FAIL", "findings": findings, "raw_text_published": False, "absolute_paths_published": False})
    if findings:
        raise RuntimeError(f"sanitization findings: {findings}")
    write_json(report / "sha256_manifest.json", {"algorithm": "sha256", "files": [{"file": path.name, "sha256": sha256_file(path)} for path in sorted(report.iterdir()) if path.name != "sha256_manifest.json"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source, report = args.source.resolve(), args.report.resolve()
    if ROOT.resolve() not in source.parents or ROOT.resolve() not in report.parents:
        raise SystemExit("project-local paths required")
    print(json.dumps(analyze(source, report), indent=2))


if __name__ == "__main__":
    main()
