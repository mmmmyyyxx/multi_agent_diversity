"""Read-only, zero-API integrity and mechanism summary for a completed attempt."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary  # noqa: E402


ALLOWED_LEDGER_PHASES = {
    "initialization", "local_optimizer_solver_eval", "local_optimizer_reflection",
    "team_minibatch_eval", "team_full_eval", "diagnostic_full_eval",
    "team_shadow_eval",
}


def audit(run_root: Path) -> dict[str, object]:
    summary = json.loads((run_root / "execution_summary.json").read_text(encoding="utf-8"))
    lifecycle = json.loads((run_root / "run_lifecycle.json").read_text(encoding="utf-8"))
    ledger_path = run_root / "ledger.jsonl"
    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    usage = ledger_summary(ledger_path)
    rows = summary["candidate_diagnostics"]
    failures: list[str] = []
    if summary.get("experiment_id") != "gepa_layer2_local_to_team_transfer_diagnostic_v1":
        failures.append("experiment_identity")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root), sort_keys=True, indent=2))
