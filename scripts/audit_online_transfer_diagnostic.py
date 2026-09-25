"""Read-only, zero-API integrity and mechanism summary for a completed attempt."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary  # noqa: E402
from multi_dataset_diverse_rl.persistence.durable_io import io_path, read_json  # noqa: E402
from multi_dataset_diverse_rl.production_transfer_diagnostic import (  # noqa: E402
    _candidate_stage_costs, _opportunity_costs, _usage,
)


ALLOWED_LEDGER_PHASES = {
    "initialization", "local_optimizer_solver_eval", "local_optimizer_reflection",
    "team_minibatch_eval", "team_full_eval", "diagnostic_full_eval",
    "team_shadow_eval",
}


def _audit_complete(run_root: Path) -> dict[str, object]:
    summary = read_json(run_root / "execution_summary.json")
    lifecycle = read_json(run_root / "run_lifecycle.json")
    ledger_path = run_root / "ledger.jsonl"
    with open(io_path(ledger_path), encoding="utf-8") as handle:
        ledger_rows = [json.loads(line) for line in handle if line.strip()]
    usage = ledger_summary(ledger_path)
    rows = summary["candidate_diagnostics"]
    failures: list[str] = []
    if summary.get("experiment_id") not in {
        "gepa_layer2_local_to_team_transfer_diagnostic_v1",
        "gepa_layer2_local_to_team_transfer_diagnostic_v2",
    }:
        failures.append("experiment_identity")
    v2 = summary.get("experiment_id") == "gepa_layer2_local_to_team_transfer_diagnostic_v2"
    if lifecycle.get("status") != "EXECUTION_COMPLETE":
        failures.append("lifecycle")
    if summary.get("seed") != 81 or summary.get("validation50_calls") != 0 or summary.get("test50_calls") != 0:
        failures.append("seed_or_split_access")
    if summary.get("ledger") != usage:
        failures.append("ledger_reconciliation")
    if len({str(row["record_id"]) for row in ledger_rows}) != len(ledger_rows):
        failures.append("duplicate_ledger_record")
    if {str(row["phase"]) for row in ledger_rows} - ALLOWED_LEDGER_PHASES:
        failures.append("unexpected_provider_phase")
    if any(row.get("postprocess_failed") or row.get("error_type") == "postprocess_failed"
           for row in ledger_rows):
        failures.append("provider_success_postprocess_failure")
    if usage["successful_provider_calls"] > 1200 or usage["provider_attempts"] > 4800:
        failures.append("emergency_ceiling")
    opportunities = int(summary["opportunities"])
    accepted = int(summary["accepted_mutations"])
    proposals = int(summary["reflection_proposals"])
    if not (0 <= opportunities <= 10 and 0 <= accepted <= 5 and 0 <= proposals <= 20):
        failures.append("sample_or_search_ceiling")
    parents = summary["parent_sequence"]
    if len(parents) != opportunities or len(rows) != accepted:
        failures.append("parent_or_mandatory_full_count")
    if v2:
        initialization_rows = [
            row for row in ledger_rows
            if row.get("phase") == "initialization"
            and row.get("record_kind") == "solver_logical_completion"
        ]
        if len(initialization_rows) != 500:
            failures.append("initialization_500_logical_rows")
        profile_dir = run_root / "system" / "team_full_categorical_profiles"
        with os.scandir(io_path(profile_dir)) as entries:
            baseline_profiles = [
                read_json(entry.path) for entry in entries if entry.name.endswith(".json")
            ]
        baseline_profiles = [
            row for row in baseline_profiles
            if row.get("evaluation_stage") == "fixed_probe_initialization"
        ]
        if (
            len(baseline_profiles) != 5
            or {row.get("target_member") for row in baseline_profiles} != set(range(5))
            or any(row.get("row_count") != 100 or len(row.get("rows", ())) != 100
                   for row in baseline_profiles)
        ):
            failures.append("initialization_five_profiles")
        expected_stage = {
            "global": _usage(ledger_rows, scope="global"),
            "initialization": _usage(
                [row for row in ledger_rows if int(row.get("update_index", -2)) == -1],
                scope="global_initialization",
            ),
            "opportunities": [
                _opportunity_costs(ledger_rows, index) for index in range(opportunities)
            ],
        }
        if summary.get("stage_accounting") != expected_stage or any(
            int(row.get("update_index", -2)) not in {-1, *range(opportunities)}
            for row in ledger_rows
        ):
            failures.append("stage_ledger_reconciliation")
    by_update = {int(row["update_index"]): row for row in rows}
    if len(by_update) != len(rows):
        failures.append("multiple_local_accepts_per_opportunity")
    for index in range(max(0, len(parents) - 1)):
        candidate = by_update.get(index)
        committed = bool(candidate and candidate.get("committed"))
        if (parents[index + 1] != parents[index]) != committed:
            failures.append("online_parent_transition")
    for row in rows:
        minibatch = row.get("team_minibatch", {})
        full = row.get("full", {})
        if (
            row.get("parent_team_hash") != parents[int(row["update_index"])]
            or row.get("local_acceptance_delta") is None
            or float(row["local_acceptance_delta"]) <= 0
            or not isinstance(full.get("vote_delta"), int)
            or bool(full.get("diagnostic_only")) == bool(minibatch.get("passed"))
        ):
            failures.append("candidate_evidence_incomplete")
        if not minibatch.get("passed") and (
            row.get("ordinary_common_safe") != "NOT_REACHED"
            or row.get("ordinary_shadow") != "NOT_REACHED"
            or row.get("committed")
        ):
            failures.append("diagnostic_feedback_leakage")
        if v2:
            try:
                expected_cost = _candidate_stage_costs(
                    ledger_rows, update_index=int(row["update_index"]),
                    candidate_id=str(row["candidate_id"]),
                    diagnostic_only=bool(full["diagnostic_only"]),
                    shadow_reached=row.get("ordinary_shadow") != "NOT_REACHED",
                )
                if row.get("stage_costs") != expected_cost:
                    failures.append("candidate_stage_cost_reconciliation")
                if not isinstance(row.get("candidate_hash"), str) or len(row["candidate_hash"]) != 64:
                    failures.append("candidate_hash_missing")
            except (KeyError, TypeError, ValueError):
                failures.append("candidate_stage_cost_incomplete")
    if summary.get("target_status") != (
        "TARGET_REACHED" if accepted == 5 else "TARGET_NOT_REACHED"
    ):
        failures.append("target_status")
    vote_counts = Counter(
        "positive" if int(row["full"]["vote_delta"]) > 0
        else "negative" if int(row["full"]["vote_delta"]) < 0
        else "neutral"
        for row in rows
    )
    false_negative = sum(
        not row["team_minibatch"]["passed"] and int(row["full"]["vote_delta"]) > 0
        for row in rows
    )
    return {
        "gate": "PASS" if not failures else "HOLD",
        "failed_checks": sorted(set(failures)),
        "opportunities": opportunities,
        "accepted_mutations": accepted,
        "reflection_proposals": proposals,
        "target_status": summary.get("target_status"),
        "distinct_parent_count": len(set(parents)),
        "full_vote_direction_counts": {
            key: vote_counts[key] for key in ("positive", "neutral", "negative")
        },
        "minibatch_false_negative_full_vote_positive": false_negative,
        "provider_attempts": usage["provider_attempts"],
        "successful_provider_calls": usage["successful_provider_calls"],
        "validation50_calls": 0,
        "test50_calls": 0,
    }


def audit(run_root: Path) -> dict[str, object]:
    """Malformed or incomplete evidence is a deterministic HOLD, never PASS."""

    try:
        return _audit_complete(run_root)
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
        return {
            "gate": "HOLD",
            "failed_checks": ["incomplete_or_malformed_evidence", type(exc).__name__],
            "validation50_calls": None,
            "test50_calls": None,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root), sort_keys=True, indent=2))
