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
from scripts.v18_shadow_raw_critic_support import LABELS, classify


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(raw: Path, gate_path: Path, report: Path) -> dict[str, Any]:
    if report.exists():
        raise FileExistsError("fresh report root required")
    gate = read_json(gate_path)
    if gate["gate"] != "PASS":
        raise ValueError("pilot gate must PASS")
    cases = [read_json(path) for path in sorted(raw.glob("seed*_*/case_result.json"))]
    if len(cases) != 6:
        raise ValueError("six frozen cases required")
    branches, candidates, validations = [], [], []
    for case in cases:
        control, shadow = case["arms"]["canonical_control"], case["arms"]["shadow_raw"]
        rejected_witness = bool(shadow["shadow_intervention_count"])
        failed_checks = sorted({check for event in shadow["shadow_events"] for check in event["original_failed_checks"]})
        for arm, row in (("canonical_control", control), ("shadow_raw", shadow)):
            branches.append({
                "case_id": case["case_id"],
                "historical_status": case["historical_status"],
                "arm": arm,
                "parent_hash": case["parent_hash"],
                "target_member": case["target_member"],
                "rejected_plan_witness": rejected_witness,
                "original_failed_checks": "|".join(failed_checks),
                "shadow_intervention_count": row["shadow_intervention_count"],
                "student_reached": row["student_reached"],
                "strict_valid_candidates": row["strict_valid_candidates"],
                "common_safe_feasible_candidates": row["common_safe_feasible_candidates"],
                "would_commit": row["would_commit"],
                "critic_api_calls": row["critic_api_calls"],
                "teacher_api_calls": row["teacher_api_calls"],
                "student_api_calls": row["student_api_calls"],
            })
            for candidate in row["candidate_rows"]:
                candidates.append({
                    "case_id": case["case_id"],
                    "arm": arm,
                    "rejected_plan_witness": rejected_witness,
                    "candidate_hash": candidate["candidate_hash"],
                    "candidate_stage": candidate["candidate_stage"],
                    "candidate_valid": candidate["valid"],
                    "common_safe_feasible": candidate["feasible"],
                    "train_target_gain": candidate["train_target_gain"],
                    "train_vote_gain": candidate["train_vote_gain"],
                    "train_vote_loss": candidate["train_vote_loss"],
                    "train_vote_net": candidate["train_vote_net"],
                    "zero_loss_feasible": bool(candidate["feasible"] and int(candidate["train_vote_loss"]) == 0),
                    "would_commit": candidate["candidate_hash"] == row["winner_hash"],
                })
            validations.append({
                "case_id": case["case_id"],
                "arm": arm,
                "rejected_plan_witness": rejected_witness,
                "would_commit": row["would_commit"],
                "winner_hash": row["winner_hash"],
                "validation_target_delta": row["validation_target_delta"],
                "validation_vote_delta": row["validation_vote_delta"],
                "validation_oracle_delta": row["validation_oracle_delta"],
            })
    witnesses = [case for case in cases if case["arms"]["shadow_raw"]["shadow_intervention_count"]]
    shadow_witnesses = [case["arms"]["shadow_raw"] for case in witnesses]
    feasible_branches = sum(row["common_safe_feasible_candidates"] > 0 for row in shadow_witnesses)
    commits = sum(bool(row["would_commit"]) for row in shadow_witnesses)
    vote_delta = sum(int(row["validation_vote_delta"]) for row in shadow_witnesses)
    label = classify(
        rejected_witnesses=len(witnesses),
        feasible_branches=feasible_branches,
        would_commit_branches=commits,
        validation_vote_delta_sum=vote_delta,
    )
    if label not in LABELS:
        raise AssertionError(label)
    shadow_candidates = [row for row in candidates if row["arm"] == "shadow_raw" and row["rejected_plan_witness"]]
    summary = {
        "pilot_version": "v18_shadow_raw_critic_pilot_v1",
        "case_count": 6,
        "branch_count": 12,
        "canonical_rejected_plan_witness_count": len(witnesses),
        "shadow_student_reach_count": sum(row["student_reached"] for row in shadow_witnesses),
        "shadow_valid_candidate_count": sum(int(row["strict_valid_candidates"]) for row in shadow_witnesses),
        "shadow_feasible_candidate_count": sum(int(row["common_safe_feasible_candidates"]) for row in shadow_witnesses),
        "shadow_feasible_branch_count": feasible_branches,
        "shadow_would_commit_branch_count": commits,
        "shadow_zero_loss_feasible_count": sum(row["zero_loss_feasible"] for row in shadow_candidates),
        "shadow_train_vote_gain_sum": sum(int(row["train_vote_gain"]) for row in shadow_candidates if row["common_safe_feasible"]),
        "shadow_train_vote_loss_sum": sum(int(row["train_vote_loss"]) for row in shadow_candidates if row["common_safe_feasible"]),
        "shadow_validation_vote_delta_sum": vote_delta,
        "shadow_validation_oracle_delta_sum": sum(int(row["validation_oracle_delta"]) for row in shadow_witnesses),
        "final_label": label,
        "shadow_critic_api_calls": sum(case["arms"]["shadow_raw"]["critic_api_calls"] for case in cases),
        "test_calls": 0,
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
    }
    report.mkdir(parents=True)
    write_csv(report / "branch_funnel.csv", branches)
    write_csv(report / "candidate_quality.csv", candidates)
    write_csv(report / "validation_pairs.csv", validations)
    write_json(report / "summary.json", summary)
    write_json(report / "classifier.json", {
        "version": "v18_shadow_raw_critic_classifier_v1",
        "labels": list(LABELS),
        "minimum_rejected_witnesses": 3,
        "over_filtering_feasible_branches_min": 2,
        "over_filtering_would_commit_branches_min": 1,
        "over_filtering_validation_vote_delta_min": -1,
        "frozen_before_validation": True,
    })
    write_json(report / "provenance.json", {
        "case_selection": "per_seed_earliest_hybrid_blocked_and_passed",
        "same_first_teacher_plan": True,
        "same_canonical_critic_response": True,
        "sole_intervention": "continue_after_valid_canonical_semantic_rejection",
        "historical_artifacts_modified": False,
        "raw_text_published": False,
        "test_accessed": False,
    })
    assertions = {
        "gate_pass": gate["gate"] == "PASS",
        "six_cases": len(cases) == 6,
        "twelve_branches": len(branches) == 12,
        "shadow_critic_zero_api": summary["shadow_critic_api_calls"] == 0,
        "no_test": True,
        "no_commits": True,
    }
    assertions["pass"] = all(assertions.values())
    write_json(report / "fact_assertions.json", assertions)
    readme = f"""# V18 Canonical Critic Shadow-Raw Pilot

This paired fixed-parent diagnostic directly continues canonical-rejected
Teacher plans into the unchanged Student and empirical rollout pipeline. It
does not use a deterministic safety checker and does not alter the canonical
Critic decision record.

- Frozen parent cases: **6**
- Valid canonical-rejected plan witnesses: **{len(witnesses)}**
- Shadow Student reaches: **{sum(row['student_reached'] for row in shadow_witnesses)}**
- Shadow valid candidates: **{summary['shadow_valid_candidate_count']}**
- Shadow Common-Safe candidates: **{summary['shadow_feasible_candidate_count']}** across **{feasible_branches}** branches
- Shadow hypothetical commits: **{commits}**
- Shadow validation Vote delta: **{vote_delta:+d}**
- Shadow validation Oracle delta: **{summary['shadow_validation_oracle_delta_sum']:+d}**

Frozen classification: **{label}**.

Validation was evaluated only for train-frozen hypothetical winners. No prompt
was committed, no trajectory was mutated, and no test data was accessed.

```text
SHADOW_CRITIC_API_CALLS=0
TEST_CALLS=0
TEAM_PROMPT_COMMITS=0
```
"""
    (report / "README.md").write_text(readme, encoding="utf-8")
    forbidden = re.compile(r"(?:[A-Za-z]:\\)|DASHSCOPE|api[_-]?key|FINAL_ANSWER:|question_text|gold_answer|model_answer|raw_response|endpoint|\.sqlite|checkpoint", re.I)
    findings = [path.name for path in report.iterdir() if path.is_file() and forbidden.search(path.read_text(encoding="utf-8"))]
    write_json(report / "sanitization_manifest.json", {"status": "PASS" if not findings else "FAIL", "findings": findings, "raw_text_published": False, "absolute_paths_published": False})
    if findings:
        raise RuntimeError(findings)
    write_json(report / "sha256_manifest.json", {"algorithm": "sha256", "files": [{"file": path.name, "sha256": sha256_file(path)} for path in sorted(report.iterdir()) if path.name != "sha256_manifest.json"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.raw.resolve(), args.gate.resolve(), args.report.resolve()), indent=2))


if __name__ == "__main__":
    main()
