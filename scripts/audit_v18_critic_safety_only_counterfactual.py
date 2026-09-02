from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs" / "v18_hybrid_online_accumulation_pilot_20260822"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "v18_critic_safety_only_counterfactual_audit_20260902"

ANTI_CHEATING = "ANTI_CHEATING"
SCHEMA_OR_FORMAT = "SCHEMA_OR_FORMAT"
OUTPUT_CONTRACT = "OUTPUT_CONTRACT"
SEMANTIC_QUALITY_ONLY = "SEMANTIC_QUALITY_ONLY"
AMBIGUOUS = "AMBIGUOUS_OUTPUT_VS_PRESERVATION"

OUTPUT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\boutput\s+(?:contract|interface|format)\b",
        r"\bresponse\s+format\b",
        r"\bfinal\s+(?:answer|output|response)\b",
        r"\banswer\s+label\b",
        r"\bfixed\s+answer\b",
        r"\bhard[- ]?cod(?:e|ed|ing)\b",
        r"\bspecializ(?:e|ed|ing)\b.{0,40}\boutput\b",
        r"\b(?:emit|return|respond|provide)\b.{0,50}\b(?:answer|option|label)\b",
        r"\b(?:only\s+return|return\s+only|provide\s+only|only\s+provide)\b",
        r"\bwithout\s+(?:any\s+)?additional\s+(?:commentary|explanation|reasoning|text)\b",
        r"\bselect\s+one\s+of\s+the\b",
        r"\bprovided\s+(?:options|answer\s+choices)\b",
    )
)

PRESERVATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bpreserv(?:e|ed|ing)\b.{0,50}\b(?:correct|answer|selection|response|decision|behavior)\b",
        r"\bretain\b.{0,50}\b(?:correct|answer|selection|response|decision)\b",
        r"\bmaintain\b.{0,50}\b(?:existing|previous|correct)\b",
        r"\b(?:not|never|do\s+not)\s+override\b",
        r"\balready\s+(?:correct|selected)\b",
        r"\bpreviously\s+correct\b",
        r"\bexisting\s+correct\b",
        r"\b(?:unique|pivotal)\s+(?:correct|answer|response)\b",
        r"\bgold\s+answer\b",
        r"\bprior\s+(?:answer|selection|decision|response)\b",
    )
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_signal(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_rejection(row: dict[str, Any], teacher_plan: dict[str, Any] | None) -> tuple[str, str, bool, bool]:
    if not row.get("schema_valid", False) or row.get("response_truncated", False) or row.get("parse_error"):
        return SCHEMA_OR_FORMAT, "critic_schema_or_transport_surface", False, False

    checks = set(row.get("failed_checks") or [])
    if "shortcut_or_copying" in checks:
        return ANTI_CHEATING, "shortcut_or_copying", False, False

    if "preservation_or_output_risk" in checks:
        plan_text = " ".join(str(value) for value in (teacher_plan or {}).values())
        output_signal = has_signal(OUTPUT_PATTERNS, plan_text)
        preservation_signal = has_signal(PRESERVATION_PATTERNS, plan_text)
        if output_signal and not preservation_signal:
            return OUTPUT_CONTRACT, "direct_output_surface_signal", True, False
        if preservation_signal and not output_signal:
            return SEMANTIC_QUALITY_ONLY, "direct_preservation_only_signal", False, True
        return AMBIGUOUS, "mixed_or_insufficient_output_preservation_signal", output_signal, preservation_signal

    if checks and checks <= {"evidence_mismatch", "actionable_specificity"}:
        return SEMANTIC_QUALITY_ONLY, "semantic_grounding_or_actionability", False, False

    return SCHEMA_OR_FORMAT, "unknown_or_unmapped_structured_check", False, False


def discover_runs(run_root: Path) -> list[tuple[int, str, Path]]:
    found: list[tuple[int, str, Path]] = []
    for seed_dir in sorted(run_root.glob("seed*")):
        if not seed_dir.is_dir() or not seed_dir.name[4:].isdigit():
            continue
        seed = int(seed_dir.name[4:])
        for arm_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            if (arm_dir / "tcs_rounds.jsonl").exists():
                found.append((seed, arm_dir.name, arm_dir))
    return found


def build_audit(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    branch_rounds: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    run_inventory = discover_runs(run_root)

    for seed, arm, run_dir in run_inventory:
        rows = read_jsonl(run_dir / "tcs_rounds.jsonl")
        teachers = {
            row.get("teacher_plan_hash"): row.get("repair_plan")
            for row in rows
            if row.get("role") == "teacher" and row.get("teacher_plan_hash") and row.get("repair_plan")
        }
        for row in rows:
            if row.get("role") != "critic":
                continue
            branch_key = (seed, arm, int(row["update_index"]), int(row["target_agent_id"]))
            compact = {
                "effective_approved": bool(row.get("effective_approved")),
                "semantic_round": int(row.get("semantic_round", 0)),
                "critic_decision_hash": str(row.get("critic_decision_hash", "")),
            }
            if not compact["effective_approved"]:
                category, reason, output_signal, preservation_signal = classify_rejection(
                    row, teachers.get(row.get("teacher_plan_hash"))
                )
                decision = {
                    "seed": seed,
                    "arm": arm,
                    "update_index": branch_key[2],
                    "target_agent_id": branch_key[3],
                    "semantic_round": compact["semantic_round"],
                    "teacher_plan_hash": str(row.get("teacher_plan_hash", "")),
                    "critic_decision_hash": compact["critic_decision_hash"],
                    "failed_checks": "+".join(row.get("failed_checks") or []),
                    "category": category,
                    "classification_reason": reason,
                    "direct_output_signal": output_signal,
                    "direct_preservation_signal": preservation_signal,
                }
                decisions.append(decision)
                compact["category"] = category
            branch_rounds[branch_key].append(compact)

    branches: list[dict[str, Any]] = []
    for key in sorted(branch_rounds):
        rounds = branch_rounds[key]
        original = any(row["effective_approved"] for row in rounds)
        clear_semantic = any(row.get("category") == SEMANTIC_QUALITY_ONLY for row in rounds)
        ambiguous = any(row.get("category") == AMBIGUOUS for row in rounds)
        lower = original or clear_semantic
        upper = lower or ambiguous
        branches.append(
            {
                "seed": key[0],
                "arm": key[1],
                "update_index": key[2],
                "target_agent_id": key[3],
                "critic_response_count": len(rounds),
                "original_student_reached": original,
                "safety_only_student_reach_lower_bound": lower,
                "safety_only_student_reach_upper_bound": upper,
                "branch_category_set": "+".join(sorted({row["category"] for row in rounds if "category" in row})),
            }
        )

    counts = Counter(row["category"] for row in decisions)
    original_count = sum(row["original_student_reached"] for row in branches)
    lower_count = sum(row["safety_only_student_reach_lower_bound"] for row in branches)
    upper_count = sum(row["safety_only_student_reach_upper_bound"] for row in branches)
    branch_count = len(branches)
    total_critic = sum(row["critic_response_count"] for row in branches)
    by_arm_seed: list[dict[str, Any]] = []
    for (seed, arm), group in sorted(_group(branches, lambda row: (row["seed"], row["arm"]))):
        n = len(group)
        by_arm_seed.append(
            {
                "seed": seed,
                "arm": arm,
                "branch_count": n,
                "original_reach_count": sum(row["original_student_reached"] for row in group),
                "lower_bound_reach_count": sum(row["safety_only_student_reach_lower_bound"] for row in group),
                "upper_bound_reach_count": sum(row["safety_only_student_reach_upper_bound"] for row in group),
            }
        )

    summary = {
        "audit_version": "v18_critic_safety_only_counterfactual_v1",
        "run_count": len(run_inventory),
        "branch_count": branch_count,
        "critic_response_count": total_critic,
        "critic_approval_count": total_critic - len(decisions),
        "critic_rejection_count": len(decisions),
        "rejection_category_counts": {category: counts.get(category, 0) for category in (ANTI_CHEATING, SCHEMA_OR_FORMAT, OUTPUT_CONTRACT, SEMANTIC_QUALITY_ONLY, AMBIGUOUS)},
        "student_reach": {
            "original_count": original_count,
            "original_rate": original_count / branch_count,
            "safety_only_lower_bound_count": lower_count,
            "safety_only_lower_bound_rate": lower_count / branch_count,
            "safety_only_upper_bound_count": upper_count,
            "safety_only_upper_bound_rate": upper_count / branch_count,
            "lower_bound_absolute_increase": lower_count - original_count,
            "upper_bound_absolute_increase": upper_count - original_count,
        },
        "by_seed_arm": by_arm_seed,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "method_modified": False,
        "historical_artifacts_modified": False,
        "interpretation_scope": "student_reach_only_not_candidate_quality_or_efficacy",
    }
    return summary, decisions, branches


def _group(rows: list[dict[str, Any]], key: Any) -> list[tuple[Any, list[dict[str, Any]]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return list(grouped.items())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report_root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    if args.report_root.exists():
        raise SystemExit("fresh report root required")

    summary, decisions, branches = build_audit(args.run_root)
    assertions = {
        "run_count_is_6": summary["run_count"] == 6,
        "branch_count_is_96": summary["branch_count"] == 96,
        "critic_response_count_is_175": summary["critic_response_count"] == 175,
        "critic_rejection_count_is_148": summary["critic_rejection_count"] == 148,
        "critic_approval_count_is_27": summary["critic_approval_count"] == 27,
        "original_student_reach_is_27": summary["student_reach"]["original_count"] == 27,
        "all_rejections_classified_once": sum(summary["rejection_category_counts"].values()) == 148,
        "bounds_are_monotone": summary["student_reach"]["original_count"] <= summary["student_reach"]["safety_only_lower_bound_count"] <= summary["student_reach"]["safety_only_upper_bound_count"] <= 96,
        "zero_api": summary["api_calls"] == summary["validation_calls"] == summary["test_calls"] == 0,
    }
    assertions["fact_assertions_pass"] = all(assertions.values())
    if not assertions["fact_assertions_pass"]:
        raise RuntimeError(assertions)

    args.report_root.mkdir(parents=True)
    write_json(args.report_root / "summary.json", summary)
    write_json(
        args.report_root / "classifier.json",
        {
            "version": "v18_critic_safety_only_classifier_v1",
            "category_precedence": [SCHEMA_OR_FORMAT, ANTI_CHEATING, OUTPUT_CONTRACT, SEMANTIC_QUALITY_ONLY, AMBIGUOUS],
            "ambiguous_bound_policy": {"lower_bound": "kept_as_blocker", "upper_bound": "treated_as_non_safety"},
            "uses_validation_or_test": False,
            "uses_raw_model_text_in_published_artifacts": False,
        },
    )
    write_json(args.report_root / "fact_assertions.json", assertions)
    write_csv(args.report_root / "decision_classification.csv", decisions)
    write_csv(args.report_root / "branch_reach_bounds.csv", branches)

    reach = summary["student_reach"]
    categories = summary["rejection_category_counts"]
    readme = f"""# V18 Safety-Only Critic Counterfactual Audit

This zero-API audit enumerates all 96 frozen V18 proposal branches and all 175
Critic responses. Of those responses, 27 were approvals and 148 were semantic
rejections. No method, historical artifact, validation state, or test state was
modified.

## Rejection mapping

| Category | Decisions |
|---|---:|
| Anti-cheating | {categories[ANTI_CHEATING]} |
| Schema/format | {categories[SCHEMA_OR_FORMAT]} |
| Output-contract | {categories[OUTPUT_CONTRACT]} |
| Semantic-quality-only | {categories[SEMANTIC_QUALITY_ONLY]} |
| Ambiguous output vs preservation | {categories[AMBIGUOUS]} |

## Student reach counterfactual

| Estimate | Branches | Rate |
|---|---:|---:|
| Historical | {reach['original_count']}/96 | {reach['original_rate']:.3%} |
| Safety-only lower bound | {reach['safety_only_lower_bound_count']}/96 | {reach['safety_only_lower_bound_rate']:.3%} |
| Safety-only upper bound | {reach['safety_only_upper_bound_count']}/96 | {reach['safety_only_upper_bound_rate']:.3%} |

The lower bound continues to block every ambiguous
`preservation_or_output_risk` decision. The upper bound treats those ambiguous
decisions as non-safety. Both estimates retain anti-cheating, schema/format, and
direct output-contract blockers.

This counterfactual estimates only whether Student would be reached. It does
not show that a bypassed plan would yield a strict-valid, feasible, safe, or
useful candidate. A prospective shadow-bypass experiment would still be needed
to estimate candidate quality conditional on a Critic rejection.

```text
API_CALLS=0
VALIDATION_CALLS=0
TEST_CALLS=0
METHOD_MODIFIED=false
HISTORICAL_ARTIFACTS_MODIFIED=false
```
"""
    (args.report_root / "README.md").write_text(readme, encoding="utf-8")

    forbidden = re.compile(r"(?:[A-Za-z]:\\|FINAL_ANSWER:|DASHSCOPE|api[_-]?key|raw_response|question_text|gold_answer|model_answer)", re.IGNORECASE)
    findings = []
    for path in sorted(args.report_root.iterdir()):
        if path.name in {"sanitization_manifest.json", "sha256_manifest.json"}:
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden.search(text):
            findings.append(path.name)
    sanitization = {"status": "PASS" if not findings else "FAIL", "forbidden_finding_count": len(findings), "forbidden_files": findings, "raw_prompts_published": False, "raw_feedback_published": False, "absolute_paths_published": False}
    write_json(args.report_root / "sanitization_manifest.json", sanitization)
    if findings:
        raise RuntimeError(sanitization)

    manifest = {
        "algorithm": "sha256",
        "files": [
            {"file": path.name, "sha256": sha256_file(path)}
            for path in sorted(args.report_root.iterdir())
            if path.name != "sha256_manifest.json"
        ],
    }
    write_json(args.report_root / "sha256_manifest.json", manifest)
    print(json.dumps({"status": "PASS", "student_reach": reach, "categories": categories}, indent=2))


if __name__ == "__main__":
    main()
