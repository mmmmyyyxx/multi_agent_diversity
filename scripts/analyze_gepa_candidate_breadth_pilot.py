from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gepa_candidate_breadth_support import ROOT, classify, read_json, write_json


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(raw: Path, gate_path: Path, report: Path) -> dict[str, Any]:
    gate = read_json(gate_path)
    if gate["gate"] != "PASS" or gate["blocker_count"]:
        raise ValueError("canonical gate must PASS before analysis")
    cases = [
        read_json(path) for path in sorted(raw.glob("seed*_update*/case_result.json"))
    ]
    if len(cases) != 2:
        raise ValueError("two frozen cases required")
    classifier = classify(cases)
    rows = []
    for case in cases:
        n2_hashes = set(case["n2"]["pool_candidate_hashes"])
        for row in case["candidate_rows"]:
            rows.append({
                "parent_id": case["case_id"],
                "candidate_hash": row["candidate_hash"],
                "candidate_stage": row["candidate_stage"],
                "source_slot": row["source_slot"],
                "in_n2_pool": row["candidate_hash"] in n2_hashes,
                "in_n4_pool": True,
                "n4_only": row["candidate_hash"] not in n2_hashes,
                "valid": row["valid"],
                "feasible": row["feasible"],
                "train_target_gain": row["train_target_gain"],
                "train_vote_gain": row["train_vote_gain"],
                "train_vote_loss": row["train_vote_loss"],
                "train_vote_net": row["train_vote_net"],
                "n2_would_commit": row["candidate_hash"] == case["n2"]["winner_hash"],
                "n4_would_commit": row["candidate_hash"] == case["n4"]["winner_hash"],
                "validation_vote_delta_if_selected": (
                    case["n2"]["validation_vote_delta"]
                    if row["candidate_hash"] == case["n2"]["winner_hash"] else
                    case["n4"]["validation_vote_delta"]
                    if row["candidate_hash"] == case["n4"]["winner_hash"] else "NA"
                ),
                "validation_oracle_delta_if_selected": (
                    case["n2"]["validation_oracle_delta"]
                    if row["candidate_hash"] == case["n2"]["winner_hash"] else
                    case["n4"]["validation_oracle_delta"]
                    if row["candidate_hash"] == case["n4"]["winner_hash"] else "NA"
                ),
            })
    write_csv(report / "phase_b_candidate_pool.csv", rows)
    summary = {
        "audit_version": "gepa_candidate_breadth_phase_a_and_b_v1",
        "phase_a_diagnosis": "CANDIDATE_SELECTION_NOT_PRIMARY",
        "phase_a_winner_changed_parent_count": 0,
        "phase_b_gate": "PASS",
        "phase_b_case_count": 2,
        "n2": {
            "valid_candidate_count": sum(int(row["n2"]["valid_candidate_count"]) for row in cases),
            "feasible_candidate_count": sum(int(row["n2"]["feasible_candidate_count"]) for row in cases),
            "zero_loss_feasible_count": sum(int(row["n2"]["zero_loss_feasible_count"]) for row in cases),
            "validation_vote_delta": sum(int(row["n2"]["validation_vote_delta"]) for row in cases),
            "validation_oracle_delta": sum(int(row["n2"]["validation_oracle_delta"]) for row in cases),
        },
        "n4": {
            "valid_candidate_count": sum(int(row["n4"]["valid_candidate_count"]) for row in cases),
            "feasible_candidate_count": sum(int(row["n4"]["feasible_candidate_count"]) for row in cases),
            "zero_loss_feasible_count": sum(int(row["n4"]["zero_loss_feasible_count"]) for row in cases),
            "n4_only_feasible_candidate_count": sum(int(row["n4"]["n4_only_feasible_candidate_count"]) for row in cases),
            "n4_only_zero_loss_feasible_count": sum(int(row["n4"]["n4_only_zero_loss_feasible_count"]) for row in cases),
            "n4_only_lower_than_n2_best_loss_feasible_count": sum(int(row["n4"]["n4_only_lower_than_n2_best_loss_feasible_count"]) for row in cases),
            "validation_vote_delta": sum(int(row["n4"]["validation_vote_delta"]) for row in cases),
            "validation_oracle_delta": sum(int(row["n4"]["validation_oracle_delta"]) for row in cases),
        },
        "classifier": classifier,
        "new_test_calls": 0,
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
    }
    write_json(report / "summary.json", summary)
    write_json(report / "classifier.json", classifier)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.raw, args.gate, args.report):
        resolved = path.resolve()
        if resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
            raise SystemExit("all paths must be project-local")
    print(json.dumps(analyze(args.raw.resolve(), args.gate.resolve(), args.report.resolve()), indent=2))


if __name__ == "__main__":
    main()
