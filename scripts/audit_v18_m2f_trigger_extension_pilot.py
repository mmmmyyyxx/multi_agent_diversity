from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.admit_v18_hybrid_online_scientific_analysis import artifact_tree_identity
from scripts.prepare_v18_m2f_trigger_extension_pilot import canonical_hash
from scripts.run_v18_m2f_trigger_extension_pilot import freeze_errors


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit(
    *, prep: Path, pilot: Path, historical_root: Path,
    admission_path: Path, out: Path,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh audit root required")
    registry = read_json(prep / "private_registry.json")
    freeze = read_json(prep / "source_freeze.json")
    phase_a = read_json(prep / "phase_a_gate.json")
    summary = read_json(pilot / "pilot_summary.json")
    train_freeze = read_json(pilot / "train_decisions_frozen.json")
    admission = read_json(admission_path)
    errors = list(freeze_errors(registry, freeze, phase_a))
    if artifact_tree_identity(historical_root) != admission["raw_artifact_identity"]:
        errors.append("historical_raw_artifact_identity")
    if freeze["raw_artifact_identity"] != admission["raw_artifact_identity"]:
        errors.append("frozen_raw_artifact_identity")
    if summary.get("registry_content_hash") != registry.get("registry_content_hash"):
        errors.append("pilot_registry_hash")
    if int(summary.get("eligible_source_count", -1)) != 7:
        errors.append("eligible_source_count")
    if int(summary.get("repair_attempt_count", -1)) != 7:
        errors.append("repair_attempt_count")
    if int(summary.get("new_test_calls", -1)) != 0:
        errors.append("new_test_calls")
    if int(summary.get("team_prompt_commits", -1)) != 0:
        errors.append("team_prompt_commits")
    if int(summary.get("trajectory_mutations", -1)) != 0:
        errors.append("trajectory_mutations")
    if not train_freeze.get("train_decisions_frozen"):
        errors.append("train_decisions_not_frozen")
    freeze_without_hash = dict(train_freeze)
    recorded_freeze_hash = str(freeze_without_hash.pop("freeze_hash", ""))
    if recorded_freeze_hash != canonical_hash(freeze_without_hash):
        errors.append("train_decisions_freeze_hash")
    if recorded_freeze_hash != summary.get("train_decisions_freeze_hash"):
        errors.append("summary_train_freeze_hash")
    cells = summary.get("cells", [])
    if len(cells) != 7 or len({row.get("case_id") for row in cells}) != 7:
        errors.append("cell_inventory")
    expected_cases = {row["case_id"]: row for row in registry["cases"]}
    for index, row in enumerate(cells, start=1):
        case_id = str(row.get("case_id", ""))
        case = expected_cases.get(case_id)
        if case is None:
            errors.append(f"unknown_case:{case_id}")
            continue
        if not row.get("extended_m2f_eligible") or not row.get("repair_attempted"):
            errors.append(f"trigger_or_attempt:{case_id}")
        if bool(row.get("repair_output_valid")) != bool(row.get("repair_evaluable")):
            errors.append(f"valid_evaluable_semantics:{case_id}")
        if row.get("repair_feasible") and not row.get("repair_evaluable"):
            errors.append(f"feasible_without_evaluation:{case_id}")
        if int(row.get("team_prompt_commits", -1)) != 0:
            errors.append(f"cell_commit:{case_id}")
        if int(row.get("trajectory_mutations", -1)) != 0:
            errors.append(f"cell_mutation:{case_id}")
        if int(row.get("test_calls", -1)) != 0:
            errors.append(f"cell_test:{case_id}")
        if int(row.get("validation_calls_before_train_freeze", -1)) != 0:
            errors.append(f"validation_leakage:{case_id}")
        cell_dir = pilot / f"cell_{index}_{case['source_candidate_hash'][:12]}"
        train_result = read_json(cell_dir / "train_result.json")
        validation_result = read_json(cell_dir / "validation_result.json")
        if canonical_hash(train_result) != train_freeze["train_result_hashes"].get(case_id):
            errors.append(f"frozen_train_result:{case_id}")
        if bool(validation_result.get("repair_validation_evaluated")) != bool(
            train_result.get("repair_output_valid")
        ):
            errors.append(f"validation_inventory:{case_id}")
        if (cell_dir / "validation_result.json").stat().st_mtime_ns < (
            pilot / "train_decisions_frozen.json"
        ).stat().st_mtime_ns:
            errors.append(f"validation_before_train_freeze:{case_id}")
    valid = int(summary.get("valid_repair_count", 0))
    if int(summary.get("logical_train_evaluator_calls", -1)) != valid:
        errors.append("train_evaluator_call_count")
    if int(summary.get("logical_validation_evaluator_calls", -1)) != 7 + valid:
        errors.append("validation_evaluator_call_count")

    source_target = sum(int(row["source_metrics"]["target_gain"]) for row in cells)
    source_gain = sum(int(row["source_metrics"]["vote_gain_count"]) for row in cells)
    source_loss = sum(int(row["source_metrics"]["vote_loss_count"]) for row in cells)
    if (source_target, source_gain, source_loss) != (39, 43, 15):
        errors.append("source_metric_totals")
    harmful_expected = {
        (59, 3): (2, 3, -1),
        (61, 5): (1, 4, -3),
    }
    for row in cells:
        case = expected_cases.get(str(row.get("case_id")))
        if case and case.get("historically_committed_source"):
            expected = harmful_expected[(int(case["source_seed"]), int(case["source_update_index"]))]
            metrics = row["source_validation_metrics"]
            observed = (
                int(metrics["vote_gain_count"]),
                int(metrics["vote_loss_count"]),
                int(metrics["vote_net_gain"]),
            )
            if observed != expected:
                errors.append(f"historical_harmful_validation:{case['case_id']}")
    result = {
        "audit_version": "v18_m2f_trigger_extension_audit_v1",
        "gate": "PASS" if not errors else "FAIL",
        "blocker_count": len(errors),
        "blockers": errors,
        "source_count": 7,
        "eligible_count": 7,
        "repair_attempt_count": int(summary.get("repair_attempt_count", 0)),
        "valid_repair_count": valid,
        "feasible_repair_count": int(summary.get("feasible_repair_count", 0)),
        "validation_evaluator_calls": int(
            summary.get("logical_validation_evaluator_calls", 0)
        ),
        "new_test_calls": 0,
        "historical_raw_artifact_identity": freeze["raw_artifact_identity"],
        "historical_raw_artifacts_modified": False,
        "call_counts": summary.get("call_counts", {}),
    }
    out.mkdir(parents=True)
    write_json(out / "audit_summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--historical_root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        prep=args.prep.resolve(),
        pilot=args.pilot.resolve(),
        historical_root=args.historical_root.resolve(),
        admission_path=args.admission.resolve(),
        out=args.out.resolve(),
    )
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
