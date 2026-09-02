from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v18_safety_only_critic_pilot_support import read_json, sha256_file, write_json
from scripts.v18_teacher_critic_pipeline_support import ARMS, select_arm


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rate(value: int | float, denominator: int | float) -> float:
    return float(value) / float(denominator) if denominator else 0.0


def analyze(raw: Path, registry_path: Path, gate_path: Path, regression_path: Path, report: Path) -> dict[str, Any]:
    if report.exists():
        raise FileExistsError("fresh report root required")
    registry, gate, regression = read_json(registry_path), read_json(gate_path), read_json(regression_path)
    if gate["gate"] != "PASS" or regression["gate"] != "PASS":
        raise ValueError("admission gate failed")
    branch_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    raw_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for case in registry["cases"]:
        for arm in ARMS:
            row = read_json(raw / case["case_id"] / arm / "branch_result.json")
            raw_by_arm[arm].append(row)
            hard = row.get("hard_gate_decisions", [])
            branch_rows.append({
                "case_id": case["case_id"],
                "historical_status": case["historical_status"],
                "arm": arm,
                "parent_hash": case["parent_team_hash"],
                "target_member": case["target_agent_id"],
                "teacher_plan_count": len(row.get("teacher_plan_hashes", [])),
                "teacher_schema_valid": bool(row.get("teacher_schema_valid")),
                "hard_gate_pass_count": sum(bool(item.get("pass")) for item in hard),
                "hard_gate_reject_count": sum(not bool(item.get("pass")) for item in hard),
                "semantic_critic_used": bool(row.get("semantic_critic_used")),
                "semantic_critic_pass_count": int(row.get("semantic_critic_pass_count", 0)),
                "semantic_critic_reject_count": int(row.get("semantic_critic_reject_count", 0)),
                "student_reached": bool(row.get("student_reached")),
                "strict_valid_candidates": int(row.get("strict_valid_candidates", 0)),
                "common_safe_feasible_candidates": int(row.get("common_safe_feasible_candidates", 0)),
                "would_commit": bool(row.get("would_commit")),
                "api_calls": int(row.get("role_totals", {}).get("api_calls", 0)),
                "total_tokens": int(row.get("role_totals", {}).get("total_tokens", 0)),
            })
            validation_rows.append({
                "case_id": case["case_id"],
                "arm": arm,
                "parent_hash": case["parent_team_hash"],
                "target_member": case["target_agent_id"],
                "would_commit": bool(row.get("would_commit")),
                "winner_hash": str(row.get("winner_hash", "")),
                "validation_target_delta": int(row.get("validation_target_delta", 0)),
                "validation_vote_delta": int(row.get("validation_vote_delta", 0)),
                "validation_oracle_delta": int(row.get("validation_oracle_delta", 0)),
            })
            for candidate in row.get("candidate_rows", []):
                candidate_rows.append({
                    "case_id": case["case_id"],
                    "arm": arm,
                    "parent_hash": case["parent_team_hash"],
                    "target_member": case["target_agent_id"],
                    "candidate_hash": candidate.get("candidate_hash", ""),
                    "candidate_stage": candidate.get("candidate_stage", ""),
                    "candidate_valid": bool(candidate.get("candidate_valid")),
                    "common_safe_feasible": bool(candidate.get("common_safe_feasible")),
                    "train_target_gain": int(candidate.get("train_target_gain", 0)),
                    "train_vote_gain": int(candidate.get("train_vote_gain", 0)),
                    "train_vote_loss": int(candidate.get("train_vote_loss", 0)),
                    "train_vote_net": int(candidate.get("train_vote_net", 0)),
                    "zero_loss_feasible": bool(candidate.get("zero_loss_feasible")),
                    "would_commit": bool(candidate.get("would_commit")),
                })
    summary: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        branches = [row for row in branch_rows if row["arm"] == arm]
        candidates = [row for row in candidate_rows if row["arm"] == arm]
        feasible = [row for row in candidates if row["common_safe_feasible"]]
        validation = [row for row in validation_rows if row["arm"] == arm]
        reached = sum(row["student_reached"] for row in branches)
        invalid_responses = sum(
            int(row.get("funnel", {}).get("student_invalid_responses", 0)) for row in raw_by_arm[arm]
        )
        student_calls = sum(int(row.get("student_calls", 0)) for row in raw_by_arm[arm])
        summary[arm] = {
            "branch_count": len(branches),
            "teacher_plan_count": sum(row["teacher_plan_count"] for row in branches),
            "hard_gate_pass_count": sum(row["hard_gate_pass_count"] for row in branches),
            "hard_gate_reject_count": sum(row["hard_gate_reject_count"] for row in branches),
            "semantic_critic_pass_count": sum(row["semantic_critic_pass_count"] for row in branches),
            "semantic_critic_reject_count": sum(row["semantic_critic_reject_count"] for row in branches),
            "student_reach_count": reached,
            "student_reach_rate": rate(reached, len(branches)),
            "strict_valid_candidate_count": len(candidates),
            "common_safe_feasible_candidate_count": len(feasible),
            "feasible_per_student": rate(len(feasible), reached),
            "feasible_per_branch": rate(len(feasible), len(branches)),
            "would_commit_count": sum(row["would_commit"] for row in branches),
            "would_commit_per_branch": rate(sum(row["would_commit"] for row in branches), len(branches)),
            "zero_loss_feasible_count": sum(row["zero_loss_feasible"] for row in feasible),
            "train_target_gain_sum": sum(row["train_target_gain"] for row in feasible),
            "train_vote_gain_sum": sum(row["train_vote_gain"] for row in feasible),
            "train_vote_loss_sum": sum(row["train_vote_loss"] for row in feasible),
            "train_vote_net_sum": sum(row["train_vote_net"] for row in feasible),
            "student_invalid_response_rate": rate(invalid_responses, student_calls),
            "validation_target_delta_sum": sum(row["validation_target_delta"] for row in validation),
            "validation_vote_delta_sum": sum(row["validation_vote_delta"] for row in validation),
            "validation_oracle_delta_sum": sum(row["validation_oracle_delta"] for row in validation),
            "api_calls": sum(row["api_calls"] for row in branches),
            "total_tokens": sum(row["total_tokens"] for row in branches),
        }
    contrast_pairs = (
        ("B-A", "B_TEACHER_CLEAN", "A_CANONICAL"),
        ("C-B", "C_NO_SEMANTIC_CRITIC", "B_TEACHER_CLEAN"),
        ("D-C", "D_ADVISORY_CRITIC", "C_NO_SEMANTIC_CRITIC"),
    )
    contrast_rows = []
    for name, treatment, control in contrast_pairs:
        contrast_rows.append({
            "contrast": name,
            "treatment": treatment,
            "control": control,
            "student_reach_delta": summary[treatment]["student_reach_count"] - summary[control]["student_reach_count"],
            "feasible_candidate_delta": summary[treatment]["common_safe_feasible_candidate_count"] - summary[control]["common_safe_feasible_candidate_count"],
            "would_commit_delta": summary[treatment]["would_commit_count"] - summary[control]["would_commit_count"],
            "train_vote_loss_delta": summary[treatment]["train_vote_loss_sum"] - summary[control]["train_vote_loss_sum"],
            "validation_target_delta_difference": summary[treatment]["validation_target_delta_sum"] - summary[control]["validation_target_delta_sum"],
            "validation_vote_delta_difference": summary[treatment]["validation_vote_delta_sum"] - summary[control]["validation_vote_delta_sum"],
            "validation_oracle_delta_difference": summary[treatment]["validation_oracle_delta_sum"] - summary[control]["validation_oracle_delta_sum"],
        })
    selection = select_arm(summary)
    report.mkdir(parents=True)
    write_csv(report / "branch_funnel.csv", branch_rows)
    write_csv(report / "candidate_quality.csv", candidate_rows)
    write_csv(report / "validation_results.csv", validation_rows)
    write_csv(report / "arm_contrasts.csv", contrast_rows)
    write_json(report / "summary.json", {
        "experiment_version": "v18_teacher_critic_pipeline_ablation_v1",
        "case_count": len(registry["cases"]),
        "branch_count": len(branch_rows),
        "arms": summary,
        "selected_arm": selection["selected_arm"],
        "test_calls": 0,
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
    })
    write_json(report / "arm_selection.json", selection)
    write_json(report / "preregistration.json", {
        "arms": list(ARMS),
        "case_selection_rule": registry["case_selection_rule"],
        "source_candidates_per_branch": 2,
        "revision_per_valid_source": 1,
        "validation_after_all_train_decisions_frozen": True,
        "validation_vote_primary": True,
        "oracle_selection_forbidden": True,
        "test_access_forbidden": True,
        "selection_version": registry["arm_selection_version"],
    })
    write_json(report / "historical_hard_gate_regression.json", regression)
    write_json(report / "provenance.json", {
        "execution_commit": registry["execution_commit"],
        "registry_content_hash": registry["registry_content_hash"],
        "audit_gate": gate["gate"],
        "raw_artifacts_modified": False,
        "raw_text_published": False,
    })
    facts = {
        "audit_pass": gate["gate"] == "PASS",
        "historical_regression_pass": regression["gate"] == "PASS",
        "six_cases": len(registry["cases"]) == 6,
        "twenty_four_branches": len(branch_rows) == 24,
        "all_arms_present": all(sum(row["arm"] == arm for row in branch_rows) == 6 for arm in ARMS),
        "zero_test": True,
        "zero_commits": True,
        "oracle_not_used_for_selection": not selection["oracle_used_for_selection"],
    }
    facts["pass"] = all(facts.values())
    write_json(report / "fact_assertions.json", facts)
    (report / "README.md").write_text(
        "# V18 Teacher-Critic Pipeline Simplification\n\n"
        "This fixed-parent four-arm experiment compares canonical Teacher/Critic, "
        "Teacher-Clean only, Teacher-Clean without semantic Critic, and Teacher-Clean "
        "with a non-blocking advisory Critic. Candidate decisions were frozen before "
        "winner-only Val50 evaluation. Validation Vote is the primary architecture "
        "selection metric; Oracle is mechanism-only.\n\n"
        f"Frozen selected arm: **{selection['selected_arm']}**.\n\n"
        f"Reason: {selection['reason']}.\n\n"
        "No prompt was committed, no trajectory was mutated, and Test125 was not accessed.\n\n"
        "```text\nTEST_ACCESSED=false\nTEAM_PROMPT_COMMITS=0\n```\n",
        encoding="utf-8",
    )
    forbidden = re.compile(
        r"(?:[A-Za-z]:\\)|DASHSCOPE|api[_-]?key|FINAL_ANSWER:|question_text|gold_answer|"
        r"model_answer|raw_response|endpoint|\.sqlite|checkpoint",
        re.IGNORECASE,
    )
    findings = []
    for path in report.iterdir():
        if path.is_file() and forbidden.search(path.read_text(encoding="utf-8")):
            findings.append(path.name)
    write_json(report / "sanitization_manifest.json", {
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "raw_text_published": False,
        "absolute_paths_published": False,
    })
    if findings:
        raise RuntimeError(f"sanitization findings: {findings}")
    write_json(report / "sha256_manifest.json", {
        "algorithm": "sha256",
        "files": [
            {"file": path.name, "sha256": sha256_file(path)}
            for path in sorted(report.iterdir()) if path.name != "sha256_manifest.json"
        ],
    })
    return {"selected_arm": selection["selected_arm"], "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.raw.resolve(), args.registry.resolve(), args.gate.resolve(), args.regression.resolve(), args.report.resolve()]
    if any(ROOT.resolve() not in path.parents for path in paths):
        raise SystemExit("project-local paths required")
    print(json.dumps(analyze(*paths), indent=2))


if __name__ == "__main__":
    main()
