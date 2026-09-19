"""Zero-API v2 freeze removing TeamMiniBatch censoring from team replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from multi_dataset_diverse_rl.governance.artifacts import (
    build_sha256_manifest, scan_sanitized_artifacts,
)
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.versions import (
    ACCEPTED_LOCAL_MUTATION_TEAM_TRANSFER_V2_VERSION,
    METHOD_VERSION,
    TEAM_MINIBATCH_DIAGNOSTIC_GATE_VERSION,
    TEAM_TRANSFER_DECOMPOSITION_V2_VERSION,
)


IDENTITY = ACCEPTED_LOCAL_MUTATION_TEAM_TRANSFER_V2_VERSION
V1_IDENTITY = "accepted_local_mutation_team_transfer_v1"
BASELINE = "faa0fc81ebe71a686554355b9c5946a1765fa3913e026ab65e59f07ae01052dd"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def run(args) -> dict:
    for name in ("v1_private_bundle", "v1_report", "v1_manifest", "private_bundle", "report"):
        setattr(args, name, getattr(args, name).resolve())
    protocol = ROOT / "experiments" / IDENTITY / "PROTOCOL.md"
    manifest_path = ROOT / "experiments" / "manifests" / f"{IDENTITY}.yaml"
    v1_protocol = ROOT / "experiments" / V1_IDENTITY / "PROTOCOL.md"
    if args.private_bundle.exists() or manifest_path.exists():
        raise FileExistsError("fresh v2 private bundle and manifest required")
    if args.report.exists() and any(args.report.iterdir()):
        raise FileExistsError("fresh v2 report required")

    v1_manifest = yaml.safe_load(args.v1_manifest.read_text(encoding="utf-8"))
    v1_freeze = read(args.v1_report / "accepted_mutation_freeze.json")
    v1_private = read(args.v1_private_bundle)
    if v1_manifest["experiment_id"] != V1_IDENTITY:
        raise ValueError("v1 manifest identity mismatch")
    if v1_manifest["api_authorization"]["authorized"] is not False:
        raise ValueError("v1 must remain unauthorized")
    if v1_manifest["result"]["classifier"] != "PREREGISTERED_NOT_EXECUTED":
        raise ValueError("v1 must remain unexecuted")
    if (ROOT / v1_manifest["design"]["formal_run_root"]).exists():
        raise ValueError("v1 formal run must remain absent")
    if read(args.v1_report / "sha256_manifest.json") != build_sha256_manifest(args.v1_report):
        raise ValueError("v1 report hash replay failed")
    if sha(args.v1_private_bundle) != v1_freeze["private_bundle_sha256"]:
        raise ValueError("v1 private bundle hash mismatch")
    mutations = v1_private["accepted_mutations"]
    if v1_private["baseline_team_hash"] != BASELINE or len(mutations) != 5:
        raise ValueError("v1 baseline or mutation count mismatch")
    expected_order = [(1, 2), (2, 8), (3, 5), (4, 2), (4, 4)]
    if [(row["target_member"], row["source_proposal_index"]) for row in mutations] != expected_order:
        raise ValueError("accepted mutation order mismatch")
    for row in mutations:
        if hashlib.sha256(row["accepted_candidate_prompt"].encode()).hexdigest() != row["accepted_candidate_hash"]:
            raise ValueError("accepted candidate prompt hash mismatch")

    private = dict(v1_private)
    private["schema_version"] = "accepted_local_mutation_team_transfer_private_v2"
    private["experiment_id"] = IDENTITY
    private["evaluation_semantics"] = {
        "mandatory_full_candidates": 5,
        "team_minibatch_role": TEAM_MINIBATCH_DIAGNOSTIC_GATE_VERSION,
        "no_sequential_state_changes": True,
    }
    write(args.private_bundle, private)

    args.report.mkdir(parents=True, exist_ok=True)
    public_mutations = v1_freeze["mutations"]
    write(args.report / "accepted_mutation_freeze.json", {
        "schema_version": "accepted_local_mutation_freeze_v2",
        "experiment_id": IDENTITY,
        "source_freeze_sha256": sha(args.v1_report / "accepted_mutation_freeze.json"),
        "selection_rule": "all five v1-frozen mutations byte-identical; no subset or regeneration",
        "baseline_team_hash": BASELINE,
        "mutation_count": 5,
        "mutations": public_mutations,
        "private_bundle_sha256": sha(args.private_bundle),
    })
    write(args.report / "evaluation_semantics_freeze.json", {
        "full_evaluation": "MANDATORY_5_OF_5",
        "full_conditioned_on_team_minibatch": False,
        "team_minibatch_role": TEAM_MINIBATCH_DIAGNOSTIC_GATE_VERSION,
        "common_safe_role": "SECONDARY_DIAGNOSTIC_AFTER_MANDATORY_FULL",
        "shadow_role": "SECONDARY_DIAGNOSTIC_WITH_EXISTING_ELIGIBILITY",
        "commit": False,
        "primary_estimand": "FullTeamVoteDelta_i=Vote(T_i)-Vote(T_0)",
        "primary_denominator": 5,
        "iid_inference": False,
        "single_baseline_state": True,
    })
    write(args.report / "scientific_state_freeze.json", {
        "LOCAL_EMPIRICAL_PATH_CONFIRMED": True,
        "LOCAL_STRICT_IMPROVEMENT_CONFIRMED_WITHIN_SINGLE_STATE": True,
        "TEAM_TRANSFER_NOT_EVALUATED": True,
        "cross_state_generalization": "PROHIBITED",
    })
    write(args.report / "v1_supersession.json", {
        "status": "SUPERSEDED_BEFORE_EXECUTION_BY_V2_DUE_TO_TEAMMINIBATCH_CENSORING",
        "v1_experiment_id": V1_IDENTITY,
        "v1_protocol_sha256": sha(v1_protocol),
        "v1_manifest_sha256": sha(args.v1_manifest),
        "v1_private_bundle_sha256": sha(args.v1_private_bundle),
        "v1_report_tree": build_sha256_manifest(args.v1_report),
        "v1_files_modified": False,
    })
    write(args.report / "cost_envelope.json", {
        "mandatory_logical_solver_rows": 560,
        "team_minibatch_logical_rows": 60,
        "full_logical_rows": 500,
        "mandatory_exact_cache_hits": 60,
        "mandatory_successful_solver_provider_calls": 500,
        "baseline_optimize_provider_calls": 0,
        "baseline_optimize_reused_rows": 500,
        "shadow_control_successful_solver_calls_min": 0,
        "shadow_control_successful_solver_calls_max": 50,
        "shadow_candidate_successful_solver_calls_min": 0,
        "shadow_candidate_successful_solver_calls_max": 250,
        "total_successful_solver_provider_floor": 500,
        "total_successful_solver_provider_ceiling": 800,
        "transport_attempt_ceiling": 3200,
        "solver_max_completion_tokens_per_call": 1800,
        "hard_completion_token_ceiling": 1440000,
        "prompt_token_estimate": "UNAVAILABLE_NO_FROZEN_TOKENIZER",
        "currency_cost_estimate": "UNAVAILABLE_NO_FROZEN_PROVIDER_PRICE",
        "Reflection": 0, "GEPA": 0, "writeback": 0,
        "persistent_realizability": 0, "Validation50": 0, "Test50": 0,
    })
    write(args.report / "evaluation_access_summary.json", {
        "preparation_api_calls": 0, "Optimize100_preparation_calls": 0,
        "Shadow50_preparation_calls": 0, "Validation50_calls": 0,
        "Test50_calls": 0, "GEPA_calls": 0, "Reflection_calls": 0,
        "writebacks": 0, "scheduler_updates": 0,
        "persistent_realizability_updates": 0,
    })
    write(args.report / "api_ledger_summary.json", {
        "provider_calls": 0, "preparation_only": True,
    })
    write(args.report / "fact_assertions.json", {
        "exact_v1_mutation_freeze_preserved": True,
        "all_five_mutations_included": True,
        "single_common_baseline": True,
        "mandatory_full_five_of_five": True,
        "minibatch_is_diagnostic": True,
        "no_candidate_generation_or_selection": True,
        "no_commit_scheduler_or_realizability": True,
        "validation_test_zero": True,
        "v1_artifacts_unmodified": True,
        "zero_api_preparation": True,
    })

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "experiment_manifest_v1", "experiment_id": IDENTITY,
        "title": "Mandatory-Full accepted local mutation team-transfer replay",
        "status": "IMPLEMENTED", "legacy_index": False,
        "lifecycle_history": [{"status": status, "timestamp": now}
                              for status in ("DRAFT", "PREREGISTERED", "IMPLEMENTED")],
        "lineage": {"parents": [V1_IDENTITY], "derives_from": V1_IDENTITY},
        "scientific_question": "Do all five strict local improvements transfer to the fixed team's Full performance?",
        "hypotheses": [
            "Multiple accepted local mutations may yield positive Full-team vote deltas.",
            "MiniBatch and mandatory Full outcomes may reveal diagnostic gate errors.",
        ],
        "method_identity": METHOD_VERSION, "runtime_version": IDENTITY,
        "data": v1_manifest["data"], "model": v1_manifest["model"], "seeds": v1_manifest["seeds"],
        "design": {
            "changed": [
                "Full Optimize100 is mandatory for all five candidates",
                "TeamMiniBatch12 is a diagnostic simulated gate",
            ],
            "unchanged": [
                "all five frozen mutations", "common baseline team", "candidate hashes",
                "local telemetry", "five-member plurality", "tie-as-abstain",
                "TeamMiniBatch12 composition and threshold", "Common-Safe", "Shadow gate",
                "solver model and contract", "splits",
            ],
            "forbidden_changes": [
                "candidate generation", "candidate subset selection", "GEPA", "Reflection",
                "scheduler", "writeback", "persistent realizability", "Validation50", "Test50",
            ],
            "supersession_status": "SUPERSEDED_BEFORE_EXECUTION_BY_V2_DUE_TO_TEAMMINIBATCH_CENSORING",
            "baseline_team_hash": BASELINE, "candidate_count": 5,
            "mandatory_full_candidate_count": 5,
            "team_minibatch_role": TEAM_MINIBATCH_DIAGNOSTIC_GATE_VERSION,
            "candidate_freeze_sha256": sha(args.report / "accepted_mutation_freeze.json"),
            "private_bundle_sha256": sha(args.private_bundle),
            "decomposition_version": TEAM_TRANSFER_DECOMPOSITION_V2_VERSION,
            "formal_run_root": "runs/accepted_local_mutation_team_transfer_v2_attempt1",
            "protocol_sha256": sha(protocol),
        },
        "api_authorization": {
            "authorized": False,
            "authorization_scope": "pending explicit mandatory-Full v2 authorization",
            "allowed_roles": [], "allowed_phases": [],
        },
        "budget": {
            "type": "mandatory_full_read_only_five_pair_replay", "frozen_before_run": True,
            "limit": {
                "mutation_pairs": 5, "mandatory_full_candidates": 5,
                "mandatory_logical_solver_rows": 560,
                "successful_solver_provider_calls": 800,
                "transport_attempts": 3200, "per_request_attempts": 4,
                "solver_concurrency": 8, "Reflection": 0, "GEPA": 0,
                "writeback": 0, "persistent_realizability": 0,
                "Validation50": 0, "Test50": 0,
            },
        },
        "selection": {
            "primary_metric": "team_positive_count_over_five_mandatory_full_replays",
            "frozen_rule": "include all five accepted mutations; mandatory Full independent of MiniBatch",
            "validation_used_for_selection": False, "test_used_for_selection": False,
        },
        "artifacts": {
            "preregistration": {"path": protocol.relative_to(ROOT).as_posix()},
            "report": args.report.relative_to(ROOT).as_posix(),
            "provenance": (args.report / "provenance.json").relative_to(ROOT).as_posix(),
        },
        "git": {
            "design_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "implementation_commit": None, "result_commit": None,
        },
        "result": {
            "classifier": "PREREGISTERED_NOT_EXECUTED",
            "conclusion": "Team transfer remains unevaluated pending separate v2 API authorization.",
            "evidence_type": "not_yet_available",
        },
    }
    manifest["artifacts"]["preregistration"]["sha256"] = preregistration_hash(manifest)
    schema = read(ROOT / "infrastructure/experiment_manifest.schema.json")
    errors = validate_manifest(manifest, schema)
    if errors:
        raise ValueError("manifest invalid: " + "; ".join(errors))
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)

    write(args.report / "provenance.json", {
        "source_v1_protocol_sha256": sha(v1_protocol),
        "source_v1_manifest_sha256": sha(args.v1_manifest),
        "source_v1_private_bundle_sha256": sha(args.v1_private_bundle),
        "source_v1_candidate_freeze_sha256": sha(args.v1_report / "accepted_mutation_freeze.json"),
        "protocol_sha256": sha(protocol), "manifest_sha256": sha(manifest_path),
        "raw_prompts_published": False, "raw_questions_published": False,
        "raw_answers_published": False,
    })
    write(args.report / "EXPERIMENT_HANDOFF.json", {
        "schema_version": "sol_luna_experiment_handoff_v1", "experiment_id": IDENTITY,
        "READY_TO_RUN": False, "authorization": "PENDING_EXPLICIT_API_AUTHORIZATION",
        "exact_runner_command": None, "private_bundle_sha256": sha(args.private_bundle),
        "candidate_freeze_sha256": sha(args.report / "accepted_mutation_freeze.json"),
        "baseline_team_hash": BASELINE,
        "blockers": ["api_authorization_pending", "clean_commit_and_source_hash_freeze_pending"],
    })
    (args.report / "README.md").write_text(
        "# Accepted local mutation team-transfer v2 preparation\n\n"
        "Status: **PREREGISTERED_NOT_EXECUTED**. V2 preserves all five v1 mutations "
        "and makes Full evaluation mandatory for 5/5. TeamMiniBatch12 is a diagnostic "
        "simulated gate. Preparation made zero API calls; team transfer remains unevaluated.\n",
        encoding="utf-8",
    )
    findings = scan_sanitized_artifacts(args.report)
    write(args.report / "sanitization_manifest.json", {
        "status": "PASS" if not findings else "FAIL", "findings": findings,
    })
    write(args.report / "sha256_manifest.json", build_sha256_manifest(args.report))
    if findings or read(args.report / "sha256_manifest.json") != build_sha256_manifest(args.report):
        raise ValueError("v2 report sanitization/hash replay failed")
    return {
        "status": "PREREGISTERED_NOT_EXECUTED", "mutation_count": 5,
        "mandatory_full_candidates": 5, "api_calls": 0, "READY_TO_RUN": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-private-bundle", type=Path, required=True)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--v1-manifest", type=Path, required=True)
    parser.add_argument("--private-bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
