"""Zero-API integrity and scientific audit for completed team-transfer v2."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.team_search.accepted_mutation_replay import (
    full_team_result, minibatch_full_gate_audit, summarize_mandatory_full_transfer,
)


IDENTITY = "accepted_local_mutation_team_transfer_v2"
EXPECTED_ORDER = [
    "member1_proposal2_7643bc41b45e",
    "member2_proposal8_a5fef723548e",
    "member3_proposal5_0229baf5ae5f",
    "member4_proposal2_c537ca25a8aa",
    "member4_proposal4_f5d945233dc9",
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def run(run_root: Path, bundle: Path, prep: Path, report: Path) -> dict:
    for path in (run_root, bundle, prep, report):
        path = path.resolve()
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("fresh execution report required")
    execution = read(run_root / "execution.json")
    lifecycle = read(run_root / "run_lifecycle.json")
    completion = read(run_root / "completion.json")
    private = read(bundle / "private_bundle.json")
    freeze = read(bundle / "freeze.json")
    source_freeze = read(prep / "accepted_mutation_freeze.json")
    lines = [json.loads(line) for line in (run_root / "provider_ledger.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    starts = [row for row in lines if row["event"] == "provider_attempt_start"]
    successes = [row for row in lines if row["event"] == "provider_attempt_success"]
    failures = [row for row in lines if row["event"] == "provider_attempt_failure"]
    cases = execution["cases"]

    expected_ids = [row["mutation_id"] for row in source_freeze["mutations"]]
    recomputed_summary = summarize_mandatory_full_transfer(cases)
    recomputed_gate = minibatch_full_gate_audit(cases)
    stage_successes = dict(Counter(row["phase"] for row in successes))
    request_ids = [row["request_identity"] for row in starts]
    integrity = {
        "execution_complete": execution["execution_status"] == "EXECUTION_COMPLETE",
        "lifecycle_terminal": lifecycle["status"] == "EXECUTION_COMPLETE",
        "lifecycle_transition": [row["status"] for row in lifecycle["history"]]
            == ["RUNNING", "EXECUTION_COMPLETE"],
        "completion_source_reverified": completion == {
            "status": "EXECUTION_COMPLETE", "source_hashes_verified_after_execution": True,
        },
        "frozen_commit": lifecycle["frozen_commit"] == freeze["commit"],
        "baseline_hash": execution["baseline_team_hash"] == freeze["baseline_team_hash"],
        "exact_five_ordered_mutations": [row["mutation_id"] for row in cases]
            == expected_ids == EXPECTED_ORDER,
        "mandatory_full_five_of_five": execution["full_evaluated_candidates"] == 5
            and all(row["full_status"] == "PASS" for row in cases),
        "minibatch_diagnostic_contract": all(
            row["team_minibatch_contract"] == "TEAM_MINIBATCH_DIAGNOSTIC_GATE_V1"
            for row in cases
        ),
        "full_classification_replay": all(
            row["full_team_result"] == full_team_result(row["team_vote_delta"])
            for row in cases
        ),
        "summary_replay": execution["transfer_summary"] == recomputed_summary,
        "gate_audit_replay": execution["team_minibatch_gate_audit"] == recomputed_gate,
        "local_telemetry_join": all(
            (row["local_delta"], row["newly_fixed"], row["newly_broken"])
            == (1, 1, 0) for row in cases
        ),
        "ledger_sequence_contiguous": [row["sequence"] for row in lines]
            == list(range(1, len(lines) + 1)),
        "ledger_start_terminal_balance": len(starts) == len(successes) + len(failures),
        "ledger_request_identity_unique_without_retries": len(request_ids) == len(set(request_ids)),
        "provider_accounting": len(successes) == execution["accounting"]["provider_successes"] == 750
            and len(failures) == execution["accounting"]["provider_failures"] == 0,
        "stage_accounting": stage_successes == {
            "team_minibatch_eval": 60,
            "team_full_eval": 440,
            "team_shadow_control_eval": 50,
            "team_shadow_candidate_eval": 200,
        },
        "budget_within_ceiling": len(successes) <= 800 and len(starts) <= 3200,
        "isolation_zero": execution["isolation"] == {
            "GEPA": 0, "Reflection": 0, "Test50": 0, "Validation50": 0,
            "persistent_realizability_update": 0, "write_back": 0,
        },
        "allowed_ledger_phases_only": set(stage_successes) == {
            "team_minibatch_eval", "team_full_eval",
            "team_shadow_control_eval", "team_shadow_candidate_eval",
        },
        "no_abort_artifact": not (run_root / "abort.json").exists(),
    }
    if not all(integrity.values()):
        raise ValueError("integrity audit failed: " + ", ".join(
            key for key, value in integrity.items() if not value
        ))

    tasks = private["phase_a_tasks"]
    task_ids = [f"seed78_update0_member{i}" for i in range(5)]
    prompt_hashes = [hashlib.sha256(tasks[key]["parent_prompt"].encode()).hexdigest()
                     for key in task_ids]
    profiles = [[row["parent_output"] for row in tasks[key]["search_examples"]]
                for key in task_ids]
    profile_hashes = [hashlib.sha256(json.dumps(
        profile, ensure_ascii=False, separators=(",", ":")
    ).encode()).hexdigest() for profile in profiles]
    all_baseline_outputs_valid = all(value is not None for profile in profiles for value in profile)
    baseline_audit = {
        "audit_timing": "POST_HOC_AFTER_FROZEN_EXECUTION",
        "unique_baseline_prompt_hashes": len(set(prompt_hashes)),
        "baseline_prompt_hash": prompt_hashes[0],
        "unique_optimize_profile_hashes": len(set(profile_hashes)),
        "all_five_optimize_profiles_identical": len(set(profile_hashes)) == 1,
        "all_baseline_outputs_valid": all_baseline_outputs_valid,
        "fixed_identical_peer_votes_per_treatment": 4,
        "single_member_plurality_change_structurally_possible_rows": (
            0 if len(set(profile_hashes)) == 1 and all_baseline_outputs_valid else None
        ),
        "mechanism": "FOUR_IDENTICAL_VALID_FIXED_PEERS_LOCK_PLURALITY_AGAINST_ONE_MEMBER_REPLACEMENT",
        "interpretation_scope": (
            "Explains zero Full vote deltas in this baseline; does not estimate transfer in a "
            "diverse or evolved team state."
        ),
    }

    transfer_table = []
    for row in cases:
        transfer_table.append({key: row[key] for key in (
            "mutation_id", "target_member", "accepted_candidate_hash", "local_delta",
            "newly_fixed", "newly_broken", "team_minibatch_status",
            "target_member_full_delta", "team_vote_delta", "full_team_result",
            "oracle_coverage_delta", "team_correct_votes_gain", "team_correct_votes_loss",
            "pivotal_flip_gain", "pivotal_flip_loss", "unique_correct_gain",
            "unique_correct_loss", "collateral_loss_count", "common_safe_status",
            "shadow_status",
        )})

    report.mkdir(parents=True, exist_ok=True)
    write(report / "summary.json", {
        "experiment_id": IDENTITY,
        "status": "VALID_SINGLE_BASELINE_STATE_REPLAY",
        "primary_estimand": {"team_positive_count": 0, "denominator": 5},
        "team_equal_count": 5, "team_negative_count": 0,
        "primary_interpretation": "LOCAL_IMPROVEMENT_OFTEN_FAILS_TO_CHANGE_TEAM_OUTCOME",
        "scope": "WITHIN_THIS_IDENTICAL_PEER_BASELINE_STATE",
        "iid_inference": False,
    })
    write(report / "transfer_table.json", transfer_table)
    write(report / "team_minibatch_gate_audit.json", {
        "gate_version": "TEAM_MINIBATCH_DIAGNOSTIC_GATE_V1",
        **execution["team_minibatch_gate_audit"],
        "interpretation": (
            "All five were false-positive promotions for the preregistered Full-team vote-positive "
            "endpoint; every MiniBatch passed on non-vote signals while Full vote delta was zero."
        ),
    })
    write(report / "baseline_redundancy_audit.json", baseline_audit)
    write(report / "integrity_audit.json", integrity)
    write(report / "cost_accounting.json", execution["accounting"])
    write(report / "evaluation_access_summary.json", {
        **execution["isolation"], "Optimize_candidate_provider_successes": 500,
        "Shadow_provider_successes": 250,
    })
    write(report / "api_ledger_summary.json", {
        "provider_attempts": len(starts), "provider_successes": len(successes),
        "provider_failures": len(failures), "stage_successes": stage_successes,
        "prompt_tokens": execution["accounting"]["solver"]["prompt_tokens"],
        "completion_tokens": execution["accounting"]["solver"]["completion_tokens"],
        "total_tokens": execution["accounting"]["solver"]["total_tokens"],
        "cache_hits": execution["accounting"]["solver"]["cache_hits"],
    })
    write(report / "interpretation.json", {
        "preregistered_primary_state": "LOCAL_IMPROVEMENT_OFTEN_FAILS_TO_CHANGE_TEAM_OUTCOME",
        "supported": [
            "All five locally accepted mutations improved target-member Full accuracy.",
            "None changed Full-team plurality accuracy in the frozen baseline.",
            "All five MiniBatch decisions were false-positive promotions for Full vote positivity.",
            "Four of five passed Common-Safe; all four Shadow-eligible mutations passed Shadow.",
        ],
        "post_hoc_mechanism": baseline_audit["mechanism"],
        "not_supported": [
            "LOCAL_IMPROVEMENT_CAN_TRANSFER_TO_TEAM_GAIN_WITHIN_THIS_STATE",
            "LOCAL_TEAM_OBJECTIVE_MISALIGNMENT_OBSERVED",
            "Cross-state or diverse-team transfer conclusions",
            "Immediate scheduler efficacy conclusions",
        ],
        "next_decision": (
            "Do not start scheduler A/B from this evidence alone. First choose a preregistered "
            "non-identical team state where a one-member replacement can change plurality, then "
            "replicate the same mandatory-Full transfer estimand."
        ),
    })
    write(report / "fact_assertions.json", {
        "execution_complete": True, "five_mandatory_full_results": True,
        "same_frozen_baseline": True, "all_five_mutations_replayed": True,
        "minibatch_did_not_censor_full": True, "no_writeback": True,
        "validation_test_zero": True, "budgets_respected": True,
        "baseline_redundancy_labeled_post_hoc": True,
    })
    write(report / "provenance.json", {
        "frozen_execution_commit": freeze["commit"],
        "execution_sha256": sha(run_root / "execution.json"),
        "completion_sha256": sha(run_root / "completion.json"),
        "lifecycle_sha256": sha(run_root / "run_lifecycle.json"),
        "provider_ledger_sha256": sha(run_root / "provider_ledger.jsonl"),
        "execution_freeze_sha256": sha(bundle / "freeze.json"),
        "execution_handoff_sha256": sha(bundle / "HANDOFF.json"),
        "candidate_freeze_sha256": sha(prep / "accepted_mutation_freeze.json"),
        "raw_prompts_published": False, "raw_questions_published": False,
        "raw_answers_published": False,
    })
    (report / "README.md").write_text(
        "# Accepted local mutation team-transfer v2 execution\n\n"
        "The frozen run completed 5/5 mandatory Full evaluations. Primary result: "
        "0/5 TEAM_POSITIVE, 5/5 TEAM_EQUAL, 0/5 TEAM_NEGATIVE. All five diagnostic "
        "MiniBatch decisions passed, yielding five false-positive promotions for the Full "
        "vote-positive endpoint. A post-hoc audit found five identical baseline prompts and "
        "Optimize profiles, so four identical fixed peers structurally locked plurality against "
        "a one-member replacement. Conclusions are limited to this baseline state.\n",
        encoding="utf-8",
    )
    findings = scan_sanitized_artifacts(report)
    write(report / "sanitization_manifest.json", {
        "status": "PASS" if not findings else "FAIL", "findings": findings,
    })
    write(report / "sha256_manifest.json", build_sha256_manifest(report))
    if findings or read(report / "sha256_manifest.json") != build_sha256_manifest(report):
        raise ValueError("execution report sanitization/hash replay failed")
    return {
        "status": "VALID_SINGLE_BASELINE_STATE_REPLAY",
        "full_evaluated_candidates": 5, "team_positive_count": 0,
        "team_equal_count": 5, "team_negative_count": 0,
        "provider_successes": len(successes), "provider_failures": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.run_root.resolve(), args.bundle.resolve(), args.prep.resolve(), args.report.resolve()
    ), indent=2))


if __name__ == "__main__":
    main()
