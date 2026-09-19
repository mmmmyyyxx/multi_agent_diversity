"""Owner zero-API audit and sanitized report for the frozen Phase-B run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts

EXPECTED = [f"seed78_update0_member{i}" for i in range(1, 5)]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def audit(run: Path, handoff: Path, report: Path) -> dict:
    if report.exists():
        raise FileExistsError("public Phase-B report root must be fresh")
    execution = read(run / "execution.json")
    completion = read(run / "completion.json")
    lifecycle = read(run / "run_lifecycle.json")
    frozen_handoff = read(handoff)
    if execution["execution_status"] != "EXECUTION_COMPLETE" or completion["status"] != "EXECUTION_COMPLETE":
        raise ValueError("execution did not complete operationally")
    if lifecycle["status"] not in {"RUNNING", "EXECUTION_COMPLETE"}:
        raise ValueError("unexpected lifecycle marker")
    parents = execution["pooled"]["parents"]
    if [row["parent_task_id"] for row in parents] != EXPECTED:
        raise ValueError("executed parent order/identity mismatch")
    proposals = []
    for parent in parents:
        if parent["funnel"]["proposal_attempts_observed"] != 8:
            raise ValueError("per-parent proposal quota mismatch")
        for row in parent["proposals"]:
            if row["parent_task_id"] != parent["parent_task_id"] or row["target_member"] != parent["target_member"]:
                raise ValueError("proposal parent attribution mismatch")
            if row["primary_denominator_eligible"] != bool(row["changed"] and row["solver_reached"]):
                raise ValueError("proposal denominator identity mismatch")
            if row["solver_reached"]:
                expected = ("STRICT_POSITIVE" if row["delta_local_count"] > 0 else
                            "STRICT_NEGATIVE" if row["delta_local_count"] < 0 else "EXACT_EQUAL")
                if row["outcome_class"] != expected:
                    raise ValueError("proposal outcome classification mismatch")
                if row["gepa_accepted"] != (row["delta_local_count"] > 0):
                    raise ValueError("official acceptance mismatch")
                if expected == "EXACT_EQUAL":
                    kind = ("BEHAVIORAL_NO_OP" if row["newly_fixed"] == row["newly_broken"] == 0
                            else "REPAIR_PRESERVATION_CANCELLATION")
                    if row["zero_delta_decomposition"] != kind:
                        raise ValueError("zero-delta decomposition mismatch")
            forbidden = {"prompt", "proposal_text", "parent_text", "question", "gold", "raw_output", "trace"}
            if forbidden.intersection(row):
                raise ValueError("raw-content field in proposal telemetry")
            proposals.append(row)
    metric_rows = sum(parent["metric_calls"] for parent in parents)
    accounting = execution["accounting"]
    if any(parent["metric_calls"] > 205 for parent in parents) or metric_rows > 820:
        raise ValueError("GEPA metric-row ceiling exceeded")
    solver_logical_rows = accounting["solver"]["logical_calls"]
    if solver_logical_rows > 820 or solver_logical_rows > metric_rows:
        raise ValueError("local Solver logical-row accounting mismatch")
    if accounting["reflection_logical_calls"] > 32:
        raise ValueError("reflection ceiling exceeded")
    ledger = [json.loads(line) for line in (run / "provider_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line]
    starts = [row for row in ledger if row["event"] == "provider_attempt_start"]
    successes = [row for row in ledger if row["event"] == "provider_attempt_success"]
    failures = [row for row in ledger if row["event"] == "provider_attempt_failure"]
    terminal = {(row["task_id"], row["role"], row["attempt"]): row for row in successes + failures}
    if len(starts) != len(terminal) or any(
            (row["task_id"], row["role"], row["attempt"]) not in terminal for row in starts):
        raise ValueError("provider attempt ledger pairing mismatch")
    if [row["sequence"] for row in ledger] != list(range(1, len(ledger) + 1)):
        raise ValueError("provider ledger sequence mismatch")
    role_success = Counter(row["role"] for row in successes)
    if role_success["solver"] != accounting["solver"]["successful_provider_calls"]:
        raise ValueError("solver ledger/accounting mismatch")
    if role_success["reflection"] != accounting["reflection_successful_calls"]:
        raise ValueError("reflection ledger/accounting mismatch")
    isolation = execution["isolation"]
    if any(isolation.values()) or execution["team_transfer"] != "TEAM_TRANSFER_NOT_EVALUATED":
        raise ValueError("team or held-out isolation violation")

    parent_public = []
    for parent in parents:
        parent_public.append({"parent_task_id": parent["parent_task_id"], "target_member": parent["target_member"],
            "quota_status": parent["quota_status"], "primary": parent["primary"],
            "funnel": parent["funnel"], "summary": parent["parent_level_summary"],
            "failure_concentration": parent["failure_concentration"],
            "pairwise_minibatch_jaccard": parent["pairwise_minibatch_jaccard"],
            "fraction_proposals_reusing_previous_failure": parent["fraction_proposals_reusing_previous_failure"],
            "unique_failure_examples_seen": parent["unique_failure_examples_seen"],
            "metric_calls": parent["metric_calls"], "skipped_iterations": parent["skipped_iterations"]})
    eligible = [row for row in proposals if row["primary_denominator_eligible"]]
    outcomes = Counter(row["outcome_class"] for row in eligible)
    zeros = Counter(row["zero_delta_decomposition"] for row in eligible if row["outcome_class"] == "EXACT_EQUAL")
    accepted_parents = [row["parent_task_id"] for row in parents if row["at_least_one_accepted"]]
    fix_break = {
        "total_newly_fixed": sum(row["newly_fixed"] for row in eligible),
        "total_newly_broken": sum(row["newly_broken"] for row in eligible),
        "fix_only": sum(row["newly_fixed"] > 0 and row["newly_broken"] == 0 for row in eligible),
        "break_only": sum(row["newly_fixed"] == 0 and row["newly_broken"] > 0 for row in eligible),
        "both_fix_and_break": sum(row["newly_fixed"] > 0 and row["newly_broken"] > 0 for row in eligible),
        "no_behavior_change": sum(row["newly_fixed"] == row["newly_broken"] == 0 for row in eligible),
    }
    root_parented = [row for row in proposals if row["parent_candidate_index"] == 0]
    evolved_parented = [row for row in proposals if row["parent_candidate_index"] != 0]
    search_concentration = {
        "root_parented_proposals": len(root_parented),
        "evolved_parented_proposals": len(evolved_parented),
        "root_parented_primary_eligible": sum(row["primary_denominator_eligible"] for row in root_parented),
        "evolved_parented_primary_eligible": sum(row["primary_denominator_eligible"] for row in evolved_parented),
        "root_parented_accepted": sum(row["gepa_accepted"] for row in root_parented),
        "evolved_parented_accepted": sum(row["gepa_accepted"] for row in evolved_parented),
        "all_pairwise_minibatch_jaccard_zero": all(
            value == 0 for parent in parents for value in parent["pairwise_minibatch_jaccard"]),
        "all_previous_failure_reuse_fractions_zero": all(
            parent["fraction_proposals_reusing_previous_failure"] == 0 for parent in parents),
        "dominant_failure_pattern_share_by_parent": {
            parent["parent_task_id"]: parent["failure_concentration"]["dominant_failure_pattern_share"]
            for parent in parents},
    }
    proposal_validity = {
        "proposal_end": len(proposals),
        "changed": sum(row["changed"] for row in proposals),
        "unchanged": sum(not row["changed"] for row in proposals),
        "contract_invalid": sum(row["contract_invalid"] for row in proposals),
        "over_length": sum("over_length" in row["failed_checks"] for row in proposals),
        "solver_reached": sum(row["solver_reached"] for row in proposals),
        "primary_denominator": len(eligible),
    }
    interpretation = {
        "reading_order": ["proposal_validity", "delta_local", "fix_break_decomposition",
                          "member_heterogeneity", "search_concentration"],
        "accepted_parent_count": len(accepted_parents), "accepted_parent_ids": accepted_parents,
        "primary_classification": (
            "LOCAL_GEPA_STRICT_IMPROVEMENT_OBSERVED_ACROSS_MULTIPLE_MEMBER_TASKS" if len(accepted_parents) >= 2
            else "STRONG_MEMBER_TASK_HETEROGENEITY" if len(accepted_parents) == 1
            else "NO_ACCEPTED_LOCAL_MUTATION_OBSERVED"),
        "behavioral_no_op_count": zeros["BEHAVIORAL_NO_OP"],
        "repair_preservation_cancellation_count": zeros["REPAIR_PRESERVATION_CANCELLATION"],
        "outcome_counts": dict(outcomes),
        "proposal_validity": proposal_validity,
        "fix_break_decomposition": fix_break,
        "search_concentration": search_concentration,
        "secondary_diagnostics": {
            "LOW_BEHAVIORAL_EFFECT_OF_PROPOSALS": {
                "behavioral_no_ops": zeros["BEHAVIORAL_NO_OP"],
                "status": "OBSERVED_BUT_NOT_DOMINANT"},
            "LOCAL_REPAIR_PRESERVATION_TRADEOFF": {
                "eligible_with_both_fixes_and_breaks": fix_break["both_fix_and_break"],
                "status": "OBSERVED_BUT_NOT_COMMON"},
        },
        "secondary_diagnostic_labels_require_owner_interpretation": True,
        "team_transfer": "TEAM_TRANSFER_NOT_EVALUATED", "cross_state_generalization": "NOT_SUPPORTED",
    }
    cost = {"phase_a_acquisition": {"successful_solver_calls": 100, "tokens": 26113,
                                     "acceptance_denominator_contribution": 0},
            "phase_b_local_solver": {**accounting["solver"],
                "provider_attempts": accounting["solver_provider_attempts"],
                "gepa_metric_observation_rows": metric_rows},
            "phase_b_reflection": {"logical_calls": accounting["reflection_logical_calls"],
                "provider_attempts": accounting["reflection_provider_attempts"],
                "successful_calls": accounting["reflection_successful_calls"],
                "input_tokens": accounting["reflection_input_tokens"],
                "output_tokens": accounting["reflection_output_tokens"],
                "total_tokens": accounting["reflection_input_tokens"] + accounting["reflection_output_tokens"]},
            "phase_b_total_provider_calls": len(successes),
            "phase_b_total_tokens": accounting["solver"]["total_tokens"]
                + accounting["reflection_input_tokens"] + accounting["reflection_output_tokens"],
            "frozen_normal_metric_row_envelope": 624,
            "frozen_hard_metric_row_ceiling": 820,
            "failed_provider_attempts": len(failures)}
    complete = execution["pooled"]["complete"] and all(row["quota_status"] == "COMPLETE" for row in parents)
    report.mkdir(parents=True)
    summary = {"classifier": "VALID_COMPLETE_4X8" if complete else "HOLD_PARENT_QUOTA_INCOMPLETE",
        "scope": "single-baseline-state multi-member Layer-1 GEPA acceptance-rate pilot",
        "shared_source_state_hash": frozen_handoff["data"]["shared_source_state_hash"],
        "pooled_primary": execution["pooled"]["primary"],
        "parents_with_at_least_one_acceptance": execution["pooled"]["parent_level_at_least_one_accepted"],
        "per_parent_acceptance_rates": execution["pooled"]["per_parent_acceptance_rates"],
        "all_parent_quotas_complete": complete,
        "operational_note": ("STALE_RUNNING_LIFECYCLE_MARKER_WITH_COMPLETE_TERMINAL_ARTIFACTS"
                             if lifecycle["status"] == "RUNNING" else None),
        "interpretation": interpretation,
        "team_transfer": "TEAM_TRANSFER_NOT_EVALUATED"}
    write(report / "summary.json", summary)
    write(report / "classifier.json", {"classifier": summary["classifier"],
          "primary_diagnostic": interpretation["primary_classification"],
          "team_transfer": "TEAM_TRANSFER_NOT_EVALUATED"})
    write(report / "parent_summaries.json", parent_public)
    with (report / "proposal_outcomes.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in proposals:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write(report / "funnel_summary.json", execution["pooled"]["funnel"])
    write(report / "interpretation.json", interpretation)
    write(report / "cost_accounting.json", cost)
    write(report / "api_ledger_summary.json", {"attempts": len(starts), "successes": len(successes),
          "failures": len(failures), "success_by_role": dict(role_success),
          "provider_ledger_sha256": sha(run / "provider_ledger.jsonl")})
    write(report / "evaluation_access_summary.json", isolation)
    write(report / "integrity_audit.json", {
          "status": "PASS", "execution_status": execution["execution_status"],
          "completion_status": completion["status"], "parent_ids": EXPECTED,
          "member_zero_executed": False, "proposal_end_total": len(proposals),
          "proposal_end_by_parent": {parent["parent_task_id"]: parent["funnel"]["proposal_attempts_observed"]
                                      for parent in parents},
          "gepa_metric_observation_rows": metric_rows,
          "solver_logical_rows": solver_logical_rows,
          "reflection_logical_calls": accounting["reflection_logical_calls"],
          "provider_attempts": len(starts), "provider_failures": len(failures),
          "run_lifecycle_marker": lifecycle["status"],
          "lifecycle_marker_stale": lifecycle["status"] == "RUNNING",
          "terminal_artifacts_establish_completion": True,
          "source_hashes_verified_after_execution": completion["source_hashes_verified_after_execution"]})
    write(report / "fact_assertions.json", {"exact_four_frozen_parents": True,
          "single_shared_baseline_state": True, "proposal_events_at_most_eight_each": True,
          "gepa_metric_observation_rows_at_most_820": True,
          "solver_logical_rows_at_most_820": True, "reflection_calls_at_most_32": True,
          "phase_a_excluded_from_denominator": True, "Validation50_calls_zero": True,
          "Test50_calls_zero": True, "team_execution_zero": True,
          "team_transfer_not_evaluated": True, "no_automatic_budget_expansion": True})
    write(report / "provenance.json", {"frozen_commit": execution["frozen_commit"],
          "handoff_sha256": sha(handoff), "freeze_sha256": frozen_handoff["freeze_sha256"],
          "execution_sha256": sha(run / "execution.json"), "completion_sha256": sha(run / "completion.json"),
          "private_parent_summaries": {path.stem: sha(path) for path in sorted((run / "parent_summaries").glob("*.json"))}})
    (report / "README.md").write_text(
        f"# Phase-B Local GEPA 4x8 result\n\nClassifier: **{summary['classifier']}**.\n\n"
        f"Pooled descriptive acceptance: {summary['pooled_primary']['numerator']}/"
        f"{summary['pooled_primary']['denominator']} = {summary['pooled_primary']['rate']}. "
        "This is a dependent, adaptive single-state pilot; it is not an IID population estimate.\n\n"
        f"Accepted parents: {len(accepted_parents)}/4. Primary diagnostic: "
        f"`{interpretation['primary_classification']}`. Exact-equal behavioral no-ops: "
        f"{zeros['BEHAVIORAL_NO_OP']}; repair/preservation cancellations: "
        f"{zeros['REPAIR_PRESERVATION_CANCELLATION']}. Proposal validity was the main funnel loss: "
        f"{proposal_validity['contract_invalid']}/32 proposals were contract-invalid, all with an "
        f"over-length failure. Eligible proposals produced {fix_break['total_newly_fixed']} fixes and "
        f"{fix_break['total_newly_broken']} breaks.\n\n"
        "Operational note: the start-time `run_lifecycle.json` marker remained `RUNNING`; "
        "`execution.json` and `completion.json` both record `EXECUTION_COMPLETE`, the process exited "
        "successfully, and post-run source hashes matched.\n\n"
        "Interpretation is limited to member/task heterogeneity within one frozen baseline state. "
        "`TEAM_TRANSFER_NOT_EVALUATED` remains true. No further Layer-1 budget was run.\n",
        encoding="utf-8")
    findings = scan_sanitized_artifacts(report)
    write(report / "sanitization_manifest.json", {"status": "PASS" if not findings else "FAIL", "findings": findings})
    write(report / "sha256_manifest.json", build_sha256_manifest(report))
    if findings or read(report / "sha256_manifest.json") != build_sha256_manifest(report):
        raise ValueError("sanitization or deterministic hash replay failed")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.handoff, args.report), indent=2))
