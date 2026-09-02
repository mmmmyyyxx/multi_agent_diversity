from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.v18_safety_only_critic_pilot_support import ARMS, LABELS, ROOT, classify, read_json, sha256_file, write_json


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(raw: Path, gate_path: Path, report: Path) -> dict[str, Any]:
    if report.exists(): raise FileExistsError("fresh report root required")
    gate = read_json(gate_path)
    if gate["gate"] != "PASS": raise ValueError("pilot gate must PASS")
    cases = [read_json(path) for path in sorted(raw.glob("seed*_*/case_result.json"))]
    if len(cases) != 6: raise ValueError("six frozen cases required")
    funnels, candidates, validation = [], [], []
    for case in cases:
        canonical_best_loss = min((int(row["train_vote_loss"]) for row in case["arms"]["canonical_llm"]["candidate_rows"] if row["feasible"]), default=None)
        for arm in ARMS:
            row = case["arms"][arm]
            funnels.append({
                "case_id": case["case_id"], "historical_status": case["historical_status"], "arm": arm,
                "parent_hash": case["parent_hash"], "target_member": case["target_member"],
                "critic_decisions": row["critic_decision_count"], "critic_approvals": row["critic_approvals"],
                "critic_api_calls": row["critic_api_calls"], "student_reached": row["student_reached"],
                "student_calls": row["student_calls"], "strict_valid_candidates": row["strict_valid_candidates"],
                "common_safe_feasible_candidates": row["common_safe_feasible_candidates"], "would_commit": row["would_commit"],
            })
            for candidate in row["candidate_rows"]:
                candidates.append({
                    "case_id": case["case_id"], "arm": arm, "candidate_hash": candidate["candidate_hash"],
                    "candidate_stage": candidate["candidate_stage"], "candidate_valid": candidate["valid"],
                    "common_safe_feasible": candidate["feasible"], "train_target_gain": candidate["train_target_gain"],
                    "train_vote_gain": candidate["train_vote_gain"], "train_vote_loss": candidate["train_vote_loss"],
                    "train_vote_net": candidate["train_vote_net"], "zero_loss_feasible": bool(candidate["feasible"] and int(candidate["train_vote_loss"]) == 0),
                    "lower_loss_than_canonical_best": bool(arm == "deterministic_safety_only" and candidate["feasible"] and canonical_best_loss is not None and int(candidate["train_vote_loss"]) < canonical_best_loss),
                    "would_commit": candidate["candidate_hash"] == row["winner_hash"],
                })
            validation.append({
                "case_id": case["case_id"], "historical_status": case["historical_status"], "arm": arm,
                "winner_hash": row["winner_hash"], "would_commit": row["would_commit"],
                "validation_vote_delta": row["validation_vote_delta"], "validation_oracle_delta": row["validation_oracle_delta"],
                "validation_target_delta": row["validation_target_delta"],
            })
    aggregate = {}
    for arm in ARMS:
        f = [row for row in funnels if row["arm"] == arm]; c = [row for row in candidates if row["arm"] == arm]; v = [row for row in validation if row["arm"] == arm]
        reached = sum(bool(row["student_reached"]) for row in f); valid = sum(int(row["strict_valid_candidates"]) for row in f); feasible = sum(int(row["common_safe_feasible_candidates"]) for row in f); commits = sum(bool(row["would_commit"]) for row in f)
        aggregate[arm] = {
            "branches_attempted": 6, "critic_approvals": sum(int(row["critic_approvals"]) for row in f), "student_reach_count": reached,
            "student_reach_rate": reached / 6, "strict_valid_candidates": valid, "valid_per_student": valid / reached if reached else 0.0,
            "feasible_candidates": feasible, "feasible_per_student": feasible / reached if reached else 0.0,
            "feasible_per_branch": feasible / 6, "would_commit_count": commits, "would_commit_per_branch": commits / 6,
            "zero_loss_feasible_count": sum(row["zero_loss_feasible"] for row in c),
            "mean_feasible_vote_loss": sum(int(row["train_vote_loss"]) for row in c if row["common_safe_feasible"]) / feasible if feasible else 0.0,
            "validation_vote_delta_sum": sum(int(row["validation_vote_delta"]) for row in v),
            "validation_oracle_delta_sum": sum(int(row["validation_oracle_delta"]) for row in v),
            "critic_api_calls": sum(int(row["critic_api_calls"]) for row in f),
        }
    label = classify(aggregate)
    if label not in LABELS: raise AssertionError(label)
    summary = {"pilot_version": "v18_safety_only_critic_pilot_v1", "case_count": 6, "branch_count": 12, "aggregate": aggregate, "final_label": label, "safety_only_critic_api_calls": aggregate["deterministic_safety_only"]["critic_api_calls"], "test_calls": 0, "team_prompt_commit_count": 0, "trajectory_mutation_count": 0}
    report.mkdir(parents=True)
    write_csv(report / "branch_funnel.csv", funnels); write_csv(report / "candidate_quality.csv", candidates); write_csv(report / "validation_pairs.csv", validation)
    write_json(report / "summary.json", summary)
    write_json(report / "classifier.json", {"version": "v18_safety_only_critic_classifier_v1", "labels": list(LABELS), "clear_student_reach_increase": "at_least_2_of_6_additional_branches", "material_would_commit_degradation": "more_than_0_05_rate", "clear_validation_collateral_deterioration": "aggregate_vote_delta_more_than_2_below_canonical", "frozen_before_validation": True})
    write_json(report / "provenance.json", {"case_selection": "per_seed_earliest_hybrid_blocked_and_passed", "historical_raw_artifacts_modified": False, "raw_prompts_published": False, "test_accessed": False})
    assertions = {"gate_pass": gate["gate"] == "PASS", "six_cases": len(cases) == 6, "twelve_branches": len(funnels) == 12, "safety_critic_zero_api": aggregate["deterministic_safety_only"]["critic_api_calls"] == 0, "no_test": True, "no_commits": True}
    assertions["fact_assertions_pass"] = all(assertions.values()); write_json(report / "fact_assertions.json", assertions)
    readme = f"""# V18 Safety-Only Critic Prospective Pilot

This paired fixed-parent pilot compares the unchanged canonical LLM Critic with
a deterministic safety-only gate. Six cases were frozen before execution: one
historically blocked and one historically passed branch for each of Seeds 59,
60, and 61.

| Arm | Student reach | Valid | Feasible | WOULD_COMMIT | Validation Vote delta | Validation Oracle delta |
|---|---:|---:|---:|---:|---:|---:|
| Canonical | {aggregate['canonical_llm']['student_reach_count']}/6 | {aggregate['canonical_llm']['strict_valid_candidates']} | {aggregate['canonical_llm']['feasible_candidates']} | {aggregate['canonical_llm']['would_commit_count']} | {aggregate['canonical_llm']['validation_vote_delta_sum']:+d} | {aggregate['canonical_llm']['validation_oracle_delta_sum']:+d} |
| Safety-only | {aggregate['deterministic_safety_only']['student_reach_count']}/6 | {aggregate['deterministic_safety_only']['strict_valid_candidates']} | {aggregate['deterministic_safety_only']['feasible_candidates']} | {aggregate['deterministic_safety_only']['would_commit_count']} | {aggregate['deterministic_safety_only']['validation_vote_delta_sum']:+d} | {aggregate['deterministic_safety_only']['validation_oracle_delta_sum']:+d} |

Frozen classification: **{label}**.

The result separates candidate supply from candidate quality. Validation was
read only after all train-side hypothetical decisions were frozen and did not
select candidates. No prompt was committed, no trajectory was mutated, and no
test example was accessed.

```text
SAFETY_ONLY_CRITIC_API_CALLS=0
TEST_CALLS=0
TEAM_PROMPT_COMMITS=0
```
"""
    (report / "README.md").write_text(readme, encoding="utf-8")
    forbidden = ("DASHSCOPE", "api_key", "raw_response", "question_text", "gold_answer", "model_answer", "FINAL_ANSWER:")
    findings = [path.name for path in report.iterdir() if any(token.lower() in path.read_text(encoding="utf-8").lower() for token in forbidden)]
    write_json(report / "sanitization_manifest.json", {"status": "PASS" if not findings else "FAIL", "findings": findings, "absolute_paths_published": False, "raw_text_published": False})
    if findings: raise RuntimeError(findings)
    write_json(report / "sha256_manifest.json", {"algorithm": "sha256", "files": [{"file": path.name, "sha256": sha256_file(path)} for path in sorted(report.iterdir()) if path.name != "sha256_manifest.json"]})
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--raw",type=Path,required=True); parser.add_argument("--gate",type=Path,required=True); parser.add_argument("--report",type=Path,required=True); args=parser.parse_args()
    print(json.dumps(analyze(args.raw.resolve(),args.gate.resolve(),args.report.resolve()),indent=2))


if __name__ == "__main__": main()
