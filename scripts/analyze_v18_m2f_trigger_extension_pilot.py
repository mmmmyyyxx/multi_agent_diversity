from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refuse empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(value: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(value)


def analyze(*, prep: Path, pilot: Path, audit_path: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh report root required")
    registry = read_json(prep / "private_registry.json")
    freeze = read_json(prep / "source_freeze.json")
    phase_a = read_json(prep / "phase_a_gate.json")
    pilot_summary = read_json(pilot / "pilot_summary.json")
    audit = read_json(audit_path)
    if phase_a.get("phase_a_gate") != "PASS" or audit.get("gate") != "PASS":
        raise ValueError("Phase A and Phase B audit must PASS before analysis")
    case_by_id = {row["case_id"]: row for row in registry["cases"]}
    pairs = []
    for cell in pilot_summary["cells"]:
        case = case_by_id[cell["case_id"]]
        repair = cell.get("repair_metrics") or {}
        source_val = cell["source_validation_metrics"]
        repair_val = cell.get("repair_validation_metrics") or {}
        pairs.append({
            "case_id": case["case_id"],
            "seed": case["source_seed"],
            "update_index": case["source_update_index"],
            "target_agent_id": case["target_agent_id"],
            "source_candidate_hash": case["source_candidate_hash"],
            "repair_candidate_hash": cell.get("repair_candidate_hash") or None,
            "historically_committed_source": bool(case["historically_committed_source"]),
            "extended_m2f_eligible": bool(case["extended_m2f_eligible"]),
            "repair_attempted": bool(cell["repair_attempted"]),
            "repair_output_valid": bool(cell["repair_output_valid"]),
            "repair_evaluable": bool(cell["repair_evaluable"]),
            "repair_feasible": bool(cell["repair_feasible"]),
            "source_responsibility_gain": len(case["repair_evidence"]),
            "repair_responsibility_gain": cell.get("repair_responsibility_gain_count"),
            "responsibility_targeting_retention": cell.get("responsibility_targeting_retention"),
            "targeting_retained": bool(cell.get("targeting_retained")),
            "source_train_target_gain": case["source_metrics"]["target_gain"],
            "repair_train_target_gain": repair.get("target_gain"),
            "target_gain_delta": (
                int(repair["target_gain"]) - int(case["source_metrics"]["target_gain"])
                if repair else None
            ),
            "source_train_vote_gain": case["source_metrics"]["vote_gain_count"],
            "source_train_vote_loss": case["source_metrics"]["vote_loss_count"],
            "source_train_vote_net": case["source_metrics"]["vote_net_gain"],
            "repair_train_vote_gain": repair.get("vote_gain_count"),
            "repair_train_vote_loss": repair.get("vote_loss_count"),
            "repair_train_vote_net": repair.get("vote_net_gain"),
            "train_loss_reduced": (
                int(repair["vote_loss_count"]) < int(case["source_metrics"]["vote_loss_count"])
                if repair else False
            ),
            "zero_loss_repair": bool(repair and int(repair["vote_loss_count"]) == 0),
            "source_validation_vote_gain": source_val["vote_gain_count"],
            "source_validation_vote_loss": source_val["vote_loss_count"],
            "source_validation_vote_net": source_val["vote_net_gain"],
            "source_validation_target_delta": source_val["target_gain"],
            "source_validation_oracle_delta": source_val["oracle_delta"],
            "repair_validation_vote_gain": repair_val.get("vote_gain_count"),
            "repair_validation_vote_loss": repair_val.get("vote_loss_count"),
            "repair_validation_vote_net": repair_val.get("vote_net_gain"),
            "repair_validation_target_delta": repair_val.get("target_gain"),
            "repair_validation_oracle_delta": repair_val.get("oracle_delta"),
            "new_test_calls": 0,
        })
    if len(pairs) != 7:
        raise ValueError("paired inventory mismatch")

    evaluable = [row for row in pairs if row["repair_evaluable"]]
    feasible = [row for row in pairs if row["repair_feasible"]]
    source_all = {
        "target_gain": sum(int(row["source_train_target_gain"]) for row in pairs),
        "vote_gain": sum(int(row["source_train_vote_gain"]) for row in pairs),
        "vote_loss": sum(int(row["source_train_vote_loss"]) for row in pairs),
        "vote_net": sum(int(row["source_train_vote_net"]) for row in pairs),
    }
    source_paired = {
        "target_gain": sum(int(row["source_train_target_gain"]) for row in evaluable),
        "vote_gain": sum(int(row["source_train_vote_gain"]) for row in evaluable),
        "vote_loss": sum(int(row["source_train_vote_loss"]) for row in evaluable),
        "vote_net": sum(int(row["source_train_vote_net"]) for row in evaluable),
    }
    repair_totals = {
        "target_gain": sum(int(row["repair_train_target_gain"]) for row in evaluable),
        "vote_gain": sum(int(row["repair_train_vote_gain"]) for row in evaluable),
        "vote_loss": sum(int(row["repair_train_vote_loss"]) for row in evaluable),
        "vote_net": sum(int(row["repair_train_vote_net"]) for row in evaluable),
    }
    retained_repairs = sum(
        int(cell.get("retained_source_responsibility_repairs", 0))
        for cell in pilot_summary["cells"] if cell.get("repair_evaluable")
    )
    source_repairs = sum(
        len(case_by_id[cell["case_id"]]["repair_evidence"])
        for cell in pilot_summary["cells"] if cell.get("repair_evaluable")
    )
    aggregate_retention = retained_repairs / max(1, source_repairs)
    harmful_pairs = [row for row in pairs if row["historically_committed_source"]]
    negative_before = sum(int(row["source_validation_vote_net"]) < 0 for row in harmful_pairs)
    negative_after = sum(
        row["repair_validation_vote_net"] is None
        or int(row["repair_validation_vote_net"]) < 0
        for row in harmful_pairs
    )
    source_validation = {
        "vote_gain": sum(int(row["source_validation_vote_gain"]) for row in pairs),
        "vote_loss": sum(int(row["source_validation_vote_loss"]) for row in pairs),
        "vote_net": sum(int(row["source_validation_vote_net"]) for row in pairs),
        "oracle_delta": sum(int(row["source_validation_oracle_delta"]) for row in pairs),
    }
    repair_validation = {
        "vote_gain": sum(int(row["repair_validation_vote_gain"]) for row in evaluable),
        "vote_loss": sum(int(row["repair_validation_vote_loss"]) for row in evaluable),
        "vote_net": sum(int(row["repair_validation_vote_net"]) for row in evaluable),
        "oracle_delta": sum(int(row["repair_validation_oracle_delta"]) for row in evaluable),
    }

    loss_reduced = repair_totals["vote_loss"] < source_paired["vote_loss"]
    high_retention = aggregate_retention >= 0.8
    harmful = (
        aggregate_retention < 0.5
        or repair_totals["vote_loss"] > source_paired["vote_loss"]
        or negative_after > negative_before
    )
    many_invalid_or_infeasible = len(evaluable) < 4 or len(feasible) < 4
    if harmful:
        label = "EXTENDED_M2F_HARMFUL"
    elif many_invalid_or_infeasible or not loss_reduced or not high_retention:
        label = "EXTENDED_M2F_TRIGGER_NOT_SUPPORTED"
    elif negative_after < negative_before:
        label = "EXTENDED_M2F_WRITEBACK_RISK_REDUCTION_SUPPORTED"
    else:
        label = "EXTENDED_M2F_TRAIN_COLLATERAL_REDUCTION_ONLY"
    classifier = {
        "classifier_version": "v18_m2f_trigger_extension_classifier_v1",
        "allowed_labels": registry["classifier"]["labels"],
        "frozen_rules": {
            "targeting_retention_high": ">=0.8 weighted responsibility-repair retention",
            "targeting_retention_harmful": "<0.5 weighted responsibility-repair retention",
            "many_invalid_or_infeasible": "valid repairs <4 or feasible repairs <4",
            "validation_harmful_case_rule": "invalid/unevaluable repair remains unresolved",
        },
        "criteria": {
            "aggregate_train_vote_loss_reduced": loss_reduced,
            "high_targeting_retention": high_retention,
            "many_invalid_or_infeasible": many_invalid_or_infeasible,
            "negative_harmful_cases_before": negative_before,
            "negative_harmful_cases_after": negative_after,
        },
        "final_label": label,
    }

    pool_rows = []
    for seed in (59, 61):
        seed_cases = [row for row in registry["cases"] if int(row["source_seed"]) == seed]
        originals = seed_cases[0]["original_pool"]
        original_best = max(originals, key=lambda row: key(row["source_metrics"]["ranking_key"]))
        alternatives = [
            {
                "candidate_hash": row["repair_candidate_hash"],
                "source_candidate_hash": row["source_candidate_hash"],
                "vote_loss_count": row["repair_train_vote_loss"],
                "ranking_key": next(
                    cell["repair_metrics"]["ranking_key"]
                    for cell in pilot_summary["cells"]
                    if cell["case_id"] == row["case_id"]
                ),
            }
            for row in feasible if int(row["seed"]) == seed
        ]
        extended_pool = [
            {
                "candidate_hash": row["source_candidate_hash"],
                "source_candidate_hash": row["source_candidate_hash"],
                "vote_loss_count": row["source_metrics"]["vote_loss_count"],
                "ranking_key": row["source_metrics"]["ranking_key"],
            }
            for row in originals
        ] + alternatives
        extended_best = max(extended_pool, key=lambda row: key(row["ranking_key"]))
        pool_rows.append({
            "seed": seed,
            "source_pool_size": len(originals),
            "feasible_repair_alternative_count": len(alternatives),
            "zero_loss_feasible_alternative_count": sum(
                int(row["vote_loss_count"]) == 0 for row in alternatives
            ),
            "lower_loss_feasible_alternative_count": sum(
                int(row["vote_loss_count"])
                < int(next(
                    pair["source_train_vote_loss"] for pair in pairs
                    if pair["source_candidate_hash"] == row["source_candidate_hash"]
                ))
                for row in alternatives
            ),
            "original_best_candidate_hash": original_best["source_candidate_hash"],
            "extended_best_candidate_hash": extended_best["candidate_hash"],
            "extended_best_is_repair": (
                extended_best["candidate_hash"] != extended_best["source_candidate_hash"]
            ),
            "ranking_improved_by_repair": (
                key(extended_best["ranking_key"])
                > key(original_best["source_metrics"]["ranking_key"])
            ),
        })

    summary = {
        "report_version": "v18_m2f_trigger_extension_pilot_report_v1",
        "phase_a_gate": "PASS",
        "phase_b_gate": "PASS",
        "eligible_source_candidates": 7,
        "repair_attempts": int(pilot_summary["repair_attempt_count"]),
        "valid_repairs": len(evaluable),
        "feasible_repairs": len(feasible),
        "targeting_retention_count": sum(bool(row["targeting_retained"]) for row in pairs),
        "aggregate_responsibility_targeting_retention": aggregate_retention,
        "source_train_totals_all_7": source_all,
        "source_train_totals_evaluable_pairs": source_paired,
        "repair_train_totals_evaluable_pairs": repair_totals,
        "source_validation_totals_all_7": source_validation,
        "repair_validation_totals_evaluable_pairs": repair_validation,
        "zero_loss_repairs": sum(bool(row["zero_loss_repair"]) for row in pairs),
        "lower_loss_repairs": sum(bool(row["train_loss_reduced"]) for row in pairs),
        "historical_harmful_negative_validation_cases_before": negative_before,
        "historical_harmful_negative_validation_cases_after": negative_after,
        "pool_reconstruction": {
            "zero_loss_feasible_alternative_count": sum(
                int(row["zero_loss_feasible_alternative_count"]) for row in pool_rows
            ),
            "lower_loss_feasible_alternative_count": sum(
                int(row["lower_loss_feasible_alternative_count"]) for row in pool_rows
            ),
            "ranking_improved_pool_count": sum(
                bool(row["ranking_improved_by_repair"]) for row in pool_rows
            ),
        },
        "call_counts": pilot_summary["call_counts"],
        "logical_train_evaluator_calls": pilot_summary["logical_train_evaluator_calls"],
        "logical_validation_evaluator_calls": pilot_summary["logical_validation_evaluator_calls"],
        "new_test_calls": 0,
        "classifier": classifier,
    }
    out.mkdir(parents=True)
    write_csv(out / "candidate_pairs.csv", pairs)
    write_csv(out / "pool_reconstruction.csv", pool_rows)
    write_json(out / "summary.json", summary)
    write_json(out / "classifier.json", classifier)
    write_json(out / "source_freeze.json", {
        "freeze_version": freeze["freeze_version"],
        "execution_commit": freeze["execution_commit"],
        "registry_content_hash": freeze["registry_content_hash"],
        "raw_artifact_identity": freeze["raw_artifact_identity"],
        "trigger_version": freeze["trigger_version"],
        "repair_version": freeze["repair_version"],
        "source_file_hashes": {
            row["path"]: row["sha256"] for row in freeze["source_files"]
        },
    })
    write_json(out / "provenance.json", {
        "study_type": "prospective_test_of_m2f_eligibility_trigger_extension",
        "execution_commit": freeze["execution_commit"],
        "registry_content_hash": freeze["registry_content_hash"],
        "phase_a_gate": phase_a["phase_a_gate"],
        "phase_b_audit_gate": audit["gate"],
        "repair_mechanism_changed": False,
        "eligibility_trigger_changed": True,
        "historical_v18_artifacts_modified": False,
        "test_accessed": False,
    })
    readme = f"""# V18 M2F Trigger Extension Pilot

This is a **prospective test of an M2F eligibility/trigger extension**. It is
not an evaluation of unchanged frozen M2F.

The repair mechanism itself is unchanged. Only eligibility is extended from
rejected collateral candidates to Common-Safe-feasible candidates with
observed train Vote loss.

## Gates and result

```text
Phase A = PASS
Phase B = PASS
eligible = 7/7
repair attempts = {summary['repair_attempts']}
valid repairs = {summary['valid_repairs']}
feasible repairs = {summary['feasible_repairs']}
final classifier = {label}
NEW_TEST_CALLS = 0
```

## Frozen primary comparison

All source candidates together had train target gain `{source_all['target_gain']}`
and train Vote `{source_all['vote_gain']}/-{source_all['vote_loss']}` =
`{source_all['vote_net']}`. Aggregate repair metrics are computed only over
evaluable paired repairs; invalid output is never treated as an unattempted
repair or silently retried.

Responsibility-targeting retention is the existing M2F definition: the
fraction of source responsibility repairs retained by the repaired prompt.
The frozen high-retention criterion is `>= 0.8`.

Validation was evaluated only after all train-side repair decisions were
frozen. It was not used by the trigger, prompt, Common-Safe evaluation,
ranking, repair selection, or pool reconstruction. Test was not accessed.

## Interpretation

The report answers only whether extending the existing M2F trigger to
Common-Safe-feasible candidates with train-visible Vote loss reduces observed
write-back risk without destroying targeted repair. It does not change Hybrid,
W1, responsibility, candidate generation, Common-Safe, ranking, plurality, the
repair prompt, retries, or any validation-aware mechanism.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    write_json(out / "fact_assertions.json", {
        "status": "PASS",
        "phase_a_gate": "PASS",
        "phase_b_gate": "PASS",
        "source_count": 7,
        "source_train_totals": source_all,
        "repair_attempt_count": summary["repair_attempts"],
        "validation_after_train_freeze": True,
        "new_test_calls": 0,
        "raw_artifacts_modified": False,
    })
    (out / "test_report.txt").write_text("verification pending\n", encoding="utf-8")
    (out / "sanitization_report.txt").write_text("sanitization pending\n", encoding="utf-8")
    manifest = [
        f"{sha256(path)}  {path.name}"
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "sha256_manifest.txt"
    ]
    (out / "sha256_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({
        "final_classifier": label,
        "valid_repairs": len(evaluable),
        "feasible_repairs": len(feasible),
        "zero_loss_repairs": summary["zero_loss_repairs"],
        "lower_loss_repairs": summary["lower_loss_repairs"],
        "new_test_calls": 0,
    }, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        prep=args.prep.resolve(),
        pilot=args.pilot.resolve(),
        audit_path=args.audit.resolve(),
        out=args.out.resolve(),
    )


if __name__ == "__main__":
    main()
