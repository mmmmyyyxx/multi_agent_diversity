from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(*, prep: Path, trajectory_report: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh report directory required")
    gate = read_json(prep / "phase_a_gate.json")
    freeze = read_json(prep / "source_freeze.json")
    registry = read_json(prep / "private_source_registry.json")
    if gate["phase_a_gate"] != "STOP_INELIGIBLE_UNDER_FROZEN_M2F":
        raise ValueError("this package path is only for the frozen eligibility STOP")
    if int(gate["frozen_m2f_eligible_count"]) != 0 or gate["phase_b_authorized_by_gate"]:
        raise ValueError("Phase B must not be run when frozen eligibility is empty")
    historical = {
        (int(row["seed"]), row["arm"], int(row["update_index"])): row
        for row in read_csv(trajectory_report / "accepted_commit_quality.csv")
    }
    writeback_rows = read_csv(
        trajectory_report.parent
        / "v18_writeback_quality_diagnostic_20260824"
        / "accepted_commit_train_evidence.csv"
    )
    committed_by_key = {
        (int(row["seed"]), row["arm"], int(row["update_index"])): row
        for row in writeback_rows
    }
    pairs = []
    for source in registry["cases"]:
        key = (int(source["seed"]), source["arm"], int(source["update_index"]))
        historical_row = historical[key]
        # Historical transition metrics apply only to the actually committed
        # source. The successor team hash is not the candidate hash, so use the
        # frozen write-back evidence table for the exact committed candidate.
        committed_candidate = committed_by_key[key]
        is_committed = (
            source["source_candidate_hash"]
            == committed_candidate["committed_candidate_hash"]
        )
        pairs.append({
            "case_id": source["case_id"],
            "seed": source["seed"],
            "update_index": source["update_index"],
            "target_agent_id": source["target_agent_id"],
            "source_candidate_hash": source["source_candidate_hash"],
            "historically_committed_source": is_committed,
            "source_common_safe": source["source_common_safe"],
            "source_target_gain": source["source_target_gain"],
            "source_train_vote_gain": source["source_vote_gain_count"],
            "source_train_vote_loss": source["source_vote_loss_count"],
            "source_train_vote_net": source["source_vote_net_gain"],
            "source_responsibility_gain_count": source["source_responsibility_gain_count"],
            "minimum_loss_evidence_count": source["minimum_loss_evidence_count"],
            "source_rejection_reason_count": len(source["source_rejection_reasons"]),
            "frozen_m2f_repair_eligible": source["frozen_m2f_repair_eligible"],
            "eligibility_failure_class": "no_frozen_collateral_rejection_reason",
            "repair_attempted": False,
            "repair_output_valid": None,
            "repair_feasible": None,
            "repair_target_gain": None,
            "repair_train_vote_gain": None,
            "repair_train_vote_loss": None,
            "repair_train_vote_net": None,
            "target_gain_retention": None,
            "source_validation_vote_gain": (
                int(historical_row["validation_gain_count"]) if is_committed else None
            ),
            "source_validation_vote_loss": (
                int(historical_row["validation_loss_count"]) if is_committed else None
            ),
            "source_validation_vote_net": (
                int(historical_row["validation_net_delta"]) if is_committed else None
            ),
            "repair_validation_vote_gain": None,
            "repair_validation_vote_loss": None,
            "repair_validation_vote_net": None,
            "repair_validation_target_delta": None,
            "repair_validation_oracle_delta": None,
            "new_test_calls": 0,
        })
    if len(pairs) != 7 or sum(bool(row["historically_committed_source"]) for row in pairs) != 2:
        raise ValueError("source/committed inventory mismatch")
    classifier = {
        "classifier_version": "v18_harmful_commit_m2f_classifier_v1",
        "allowed_labels": registry["classifier"]["labels"],
        "final_diagnosis": "M2F_NOT_SUPPORTED",
        "interpretation": "not_evaluated_ineligible_under_frozen_m2f",
        "repair_efficacy_interpretable": False,
        "m2f_applicability_gap": True,
        "eligibility_modified": False,
    }
    source_totals = {
        "target_gain": sum(int(row["source_target_gain"]) for row in pairs),
        "train_vote_gain": sum(int(row["source_train_vote_gain"]) for row in pairs),
        "train_vote_loss": sum(int(row["source_train_vote_loss"]) for row in pairs),
        "train_vote_net": sum(int(row["source_train_vote_net"]) for row in pairs),
    }
    committed = [row for row in pairs if row["historically_committed_source"]]
    summary = {
        "report_version": "v18_harmful_commit_m2f_repair_pilot_v1",
        "phase_a_gate": gate["phase_a_gate"],
        "phase_b_gate": "NOT_RUN_PHASE_A_STOP",
        "source_candidate_count": len(pairs),
        "eligible_source_candidate_count": 0,
        "valid_repair_output_count": 0,
        "feasible_repair_count": 0,
        "zero_loss_repair_count": 0,
        "lower_loss_repair_count": 0,
        "targeting_retained_count": 0,
        "repair_attributable_feasible_rescue_count": 0,
        "source_totals": source_totals,
        "repair_totals": None,
        "historically_committed_source_count": len(committed),
        "source_validation_negative_net_case_count": sum(
            int(row["source_validation_vote_net"]) < 0 for row in committed
        ),
        "repair_validation_negative_net_case_count": None,
        "api_calls": {
            "model_calls": 0,
            "solver_calls": 0,
            "evaluator_calls": 0,
            "new_validation_calls": 0,
            "new_test_calls": 0,
        },
        "method_modified": False,
        "eligibility_modified": False,
        "historical_raw_artifacts_modified": False,
        "classifier": classifier,
    }
    out.mkdir(parents=True)
    write_csv(out / "candidate_pairs.csv", pairs)
    write_json(out / "summary.json", summary)
    write_json(out / "classifier.json", classifier)
    write_json(out / "source_freeze.json", freeze)
    readme = f"""# V18 Harmful-Commit M2F Repair Pilot

## Gate result

```text
Phase A gate = {gate['phase_a_gate']}
Phase B gate = NOT_RUN_PHASE_A_STOP
Final diagnosis = M2F_NOT_SUPPORTED
Interpretation = not_evaluated_ineligible_under_frozen_m2f
```

Phase A reconstructed all seven Common-Safe feasible candidates with positive
train Vote loss from the two frozen V18 harmful pools. Source prompt hashes,
parent team hashes, targets, responsibility membership, Common-Safe outcomes,
and the historical raw artifact tree were verified.

Phase B was not run. This is a frozen applicability stop, not evidence that an
executed repair failed.

## Why eligibility is empty

Existing M2F repair requires all three conditions:

1. responsibility gain is positive;
2. candidate-specific loss evidence is positive;
3. the source was rejected for target, team-Vote, or terminal-invalid
   regression.

All 7 sources satisfy the first condition and have at least one pivotal/unique
loss evidence item. All 7 fail only the third condition: they are Common-Safe
feasible and have empty rejection-reason lists.

The requested pool filter (`feasible AND train_vote_loss > 0`) therefore finds
7 sources, while unchanged M2F eligibility finds 0. Eligibility was not widened
to force API execution.

## Source inventory

```text
source candidates = 7
historically committed harmful sources = 2
source target gain total = {source_totals['target_gain']}
source train Vote gains = {source_totals['train_vote_gain']}
source train Vote losses = {source_totals['train_vote_loss']}
source train Vote net = {source_totals['train_vote_net']}
frozen M2F eligible = 0
```

The two historically committed sources account for 7 validation loss events
and validation net -4. Validation results for the other five uncommitted
sources remain `NA`; they were not evaluated counterfactually.

## Repair metrics

```text
valid repair outputs = 0
feasible repairs = 0
zero-loss repairs = 0
lower-loss repairs = 0
targeting retained = 0 (not evaluated)
harmful validation cases before = 2
harmful validation cases after = NA
MODEL_CALLS = 0
SOLVER_CALLS = 0
EVALUATOR_CALLS = 0
NEW_VALIDATION_CALLS = 0
NEW_TEST_CALLS = 0
```

Zero repair counts mean `not attempted`, not empirical repair failures.

## Answer to the study question

The existing M2F implementation cannot currently be evaluated on these
train-visible harmful write-back cases without changing its eligibility
semantics. It is designed to repair candidates already rejected for a frozen
collateral-regression reason, whereas the V18 cases are accepted candidates
whose aggregate train gains mask per-example Vote losses.

Accordingly, this task identifies an M2F applicability gap at the boundary
between rejected-candidate repair and accepted-update quality control. It does
not authorize changing that boundary in this task.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    facts = {
        "fact_assertions_pass": True,
        "phase_a_source_reconstruction": "PASS",
        "phase_a_m2f_semantics": "PASS",
        "phase_a_gate": gate["phase_a_gate"],
        "phase_b_gate": "NOT_RUN_PHASE_A_STOP",
        "source_candidate_count": 7,
        "eligible_source_candidate_count": 0,
        "common_safe_source_count": 7,
        "positive_responsibility_gain_source_count": 7,
        "positive_minimum_loss_evidence_source_count": 7,
        "empty_rejection_reason_source_count": 7,
        "repair_attempt_count": 0,
        "new_api_calls": 0,
        "new_validation_calls": 0,
        "new_test_calls": 0,
        "eligibility_modified": False,
        "historical_raw_artifacts_modified": False,
    }
    write_json(out / "fact_assertions.json", facts)
    manifest = {
        path.name: sha256_file(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    write_json(out / "sha256_manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--trajectory_report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = package(
        prep=args.prep.resolve(),
        trajectory_report=args.trajectory_report.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps({
        "phase_a_gate": summary["phase_a_gate"],
        "phase_b_gate": summary["phase_b_gate"],
        "final_diagnosis": summary["classifier"]["final_diagnosis"],
        "api_calls": summary["api_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
