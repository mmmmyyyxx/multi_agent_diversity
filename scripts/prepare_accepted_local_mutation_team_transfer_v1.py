"""Zero-API freeze of all five accepted Phase-B mutations for team replay."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import validate_complete_compact_prompt
from multi_dataset_diverse_rl.parent_acquisition import digest, text_hash
from multi_dataset_diverse_rl.versions import (
    ACCEPTED_LOCAL_MUTATION_TEAM_TRANSFER_VERSION,
    METHOD_VERSION,
    TEAM_MINIBATCH_CONTRACT_VERSION,
    TEAM_TRANSFER_DECOMPOSITION_VERSION,
)


IDENTITY = ACCEPTED_LOCAL_MUTATION_TEAM_TRANSFER_VERSION
BASELINE_STATE = "faa0fc81ebe71a686554355b9c5946a1765fa3913e026ab65e59f07ae01052dd"
PHASE_B_ID = "local_gepa_acceptance_rate_pilot_phase_b_v2"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical_hash(values) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def group(tags: list[str]) -> str:
    found = [name for name in ("responsibility", "coalition", "preservation") if name in tags]
    if len(found) != 1:
        raise ValueError("each Optimize example must have one evidence group tag")
    return found[0]


def run(args) -> dict:
    args.phase_a_tasks = args.phase_a_tasks.resolve()
    args.phase_b_run = args.phase_b_run.resolve()
    args.phase_b_bundle = args.phase_b_bundle.resolve()
    args.shadow_csv = args.shadow_csv.resolve()
    args.private_bundle = args.private_bundle.resolve()
    args.report = args.report.resolve()
    if args.private_bundle.exists():
        raise FileExistsError(f"fresh output required: {args.private_bundle.name}")
    if args.report.exists() and any(args.report.iterdir()):
        raise FileExistsError(f"fresh output required: {args.report.name}")
    manifest_path = ROOT / "experiments/manifests" / f"{IDENTITY}.yaml"
    if manifest_path.exists():
        raise FileExistsError("fresh experiment manifest required")
    protocol = ROOT / "experiments" / IDENTITY / "PROTOCOL.md"

    phase_a_tasks = read(args.phase_a_tasks)
    phase_b_tasks = read(args.phase_b_bundle / "selected_parent_tasks_private.json")
    execution = read(args.phase_b_run / "execution.json")
    if execution["execution_status"] != "EXECUTION_COMPLETE" or not execution["pooled"]["complete"]:
        raise ValueError("complete Phase-B source required")
    expected_tasks = [f"seed78_update0_member{i}" for i in range(5)]
    if sorted(phase_a_tasks) != expected_tasks:
        raise ValueError("complete five-member Phase-A task catalog required")
    if sorted(phase_b_tasks) != expected_tasks[1:]:
        raise ValueError("exact Phase-B parent set required")
    for task_id in expected_tasks[1:]:
        if digest(phase_a_tasks[task_id]) != digest(phase_b_tasks[task_id]):
            raise ValueError("Phase-A/Phase-B task identity mismatch")

    ordered_ids = [row["example_id"] for row in phase_a_tasks[expected_tasks[0]]["search_examples"]]
    if len(ordered_ids) != 100 or len(set(ordered_ids)) != 100:
        raise ValueError("Optimize100 identity required")
    for task_id in expected_tasks:
        task = phase_a_tasks[task_id]
        if [row["example_id"] for row in task["search_examples"]] != ordered_ids:
            raise ValueError("all baseline tasks must share ordered Optimize100")
        if text_hash(task["parent_prompt"]) != "549bc93c03f703faf5aa1bd56b557135fb6e65d0cf6c055b8fad6a15e7c87a63":
            raise ValueError("unexpected root parent prompt")

    accepted_private = []
    accepted_public = []
    for member in range(1, 5):
        task_id = f"seed78_update0_member{member}"
        summary_path = args.phase_b_run / "parent_summaries" / f"{task_id}.json"
        summary = read(summary_path)
        candidates_path = args.phase_b_run / "local_gepa" / task_id / "candidates.json"
        candidates = read(candidates_path)
        prompts = {text_hash(row["decision_procedure"]): row["decision_procedure"] for row in candidates}
        for proposal in summary["proposals"]:
            if not proposal["gepa_accepted"]:
                continue
            candidate_hash = proposal["proposal_hash"]
            prompt = prompts.get(candidate_hash)
            if prompt is None:
                raise ValueError("accepted candidate prompt is missing")
            validate_complete_compact_prompt(
                prompt, parent_prompt=phase_a_tasks[task_id]["parent_prompt"], examples=(), max_chars=3000,
            )
            validation = phase_a_tasks[task_id]["local_validation_examples"]
            groups = {name: sum(group(row["tags"]) == name for row in validation)
                      for name in ("responsibility", "coalition", "preservation")}
            if groups != {"responsibility": 4, "coalition": 4, "preservation": 4}:
                raise ValueError("TeamMiniBatch12 group mismatch")
            mutation_id = f"member{member}_proposal{proposal['proposal_index']}_{candidate_hash[:12]}"
            source = {
                "mutation_id": mutation_id,
                "baseline_team_hash": BASELINE_STATE,
                "target_member": member,
                "root_parent_hash": proposal["parent_hash"],
                "accepted_candidate_hash": candidate_hash,
                "local_delta": proposal["delta_local_count"],
                "newly_fixed": proposal["newly_fixed"],
                "newly_broken": proposal["newly_broken"],
                "source_proposal_index": proposal["proposal_index"],
                "source_parent_task_id": task_id,
                "source_phase_b_run_identity": "local_gepa_acceptance_rate_pilot_phase_b_v2_attempt1",
                "source_phase_b_experiment_id": PHASE_B_ID,
                "team_minibatch_contract": TEAM_MINIBATCH_CONTRACT_VERSION,
                "team_minibatch_example_ids": [row["example_id"] for row in validation],
                "team_minibatch_group_counts": groups,
            }
            accepted_public.append(source)
            accepted_private.append({**source, "accepted_candidate_prompt": prompt})
    accepted_public.sort(key=lambda row: (
        row["target_member"], row["source_proposal_index"], row["accepted_candidate_hash"]
    ))
    by_id = {row["mutation_id"]: row for row in accepted_private}
    accepted_private = [{**row, "accepted_candidate_prompt": by_id[row["mutation_id"]]["accepted_candidate_prompt"]}
                        for row in accepted_public]
    if len(accepted_public) != 5 or any(row["local_delta"] <= 0 for row in accepted_public):
        raise ValueError("exactly five strict-positive accepted mutations required")

    with args.shadow_csv.open(encoding="utf-8-sig", newline="") as handle:
        shadow_rows = list(csv.DictReader(handle))
    shadow_ids = [text_hash(row["question"]) for row in shadow_rows]
    expected_shadow = read(ROOT / "experiments/anti_overfitting_split_v1/fold_assignment.json")["folds"]["fold_c"]
    if len(shadow_rows) != 50 or sorted(shadow_ids) != sorted(expected_shadow):
        raise ValueError("frozen Shadow50 identity mismatch")

    private = {
        "schema_version": "accepted_local_mutation_team_transfer_private_v1",
        "experiment_id": IDENTITY,
        "baseline_team_hash": BASELINE_STATE,
        "phase_a_tasks": phase_a_tasks,
        "accepted_mutations": accepted_private,
        "shadow50": shadow_rows,
    }
    write(args.private_bundle, private)

    args.report.mkdir(parents=True, exist_ok=True)
    write(args.report / "accepted_mutation_freeze.json", {
        "schema_version": "accepted_local_mutation_freeze_v1",
        "experiment_id": IDENTITY,
        "selection_rule": "all Phase-B gepa_accepted mutations; no subset or regeneration",
        "baseline_team_hash": BASELINE_STATE,
        "mutation_count": 5,
        "mutations": accepted_public,
        "private_bundle_sha256": sha(args.private_bundle),
    })
    write(args.report / "scientific_state_freeze.json", {
        "LOCAL_EMPIRICAL_PATH_CONFIRMED": True,
        "LOCAL_STRICT_IMPROVEMENT_CONFIRMED_WITHIN_SINGLE_STATE": True,
        "TEAM_TRANSFER_NOT_EVALUATED": True,
        "pooled_acceptance": {"numerator": 5, "denominator": 10,
                              "estimand": "DESCRIPTIVE_NON_IID_SINGLE_STATE"},
        "cross_state_generalization": "PROHIBITED",
    })
    write(args.report / "cost_envelope.json", {
        "optimize_candidate_profile_successful_solver_ceiling": 500,
        "shadow_control_shared_successful_solver_ceiling": 50,
        "shadow_candidate_profile_successful_solver_ceiling": 250,
        "total_successful_solver_provider_ceiling": 800,
        "transport_attempt_ceiling": 3200,
        "reflection_calls": 0, "gepa_calls": 0,
        "Validation50_calls": 0, "Test50_calls": 0,
    })
    write(args.report / "evaluation_access_summary.json", {
        "preparation_api_calls": 0, "Optimize100_preparation_calls": 0,
        "Shadow50_preparation_calls": 0, "Validation50_calls": 0, "Test50_calls": 0,
        "gepa_proposals": 0, "reflection_calls": 0, "writebacks": 0,
        "persistent_realizability_updates": 0,
    })
    write(args.report / "api_ledger_summary.json", {"provider_calls": 0, "preparation_only": True})
    write(args.report / "fact_assertions.json", {
        "exactly_five_source_accepted_mutations": True,
        "all_source_accepted_mutations_included": True,
        "single_baseline_team": True,
        "member_zero_not_substituted": True,
        "candidate_text_not_published": True,
        "read_only_no_writeback": True,
        "validation_test_zero": True,
        "zero_api_preparation": True,
    })

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "experiment_manifest_v1", "experiment_id": IDENTITY,
        "title": "Accepted local mutation read-only team-transfer replay", "status": "IMPLEMENTED",
        "legacy_index": False,
        "lifecycle_history": [{"status": status, "timestamp": now}
                              for status in ("DRAFT", "PREREGISTERED", "IMPLEMENTED")],
        "lineage": {"parents": [PHASE_B_ID], "derives_from": PHASE_B_ID},
        "scientific_question": "Do all five accepted local mutations transfer to team-level gains in the same fixed baseline state?",
        "hypotheses": [
            "Multiple accepted local mutations may yield positive full-team vote deltas.",
            "If progressive team transfer is absent, local and team objectives may be misaligned.",
        ],
        "method_identity": METHOD_VERSION, "runtime_version": IDENTITY,
        "data": {"task": "BBH disambiguation_qa", "formal": False,
                 "split_ids": {"optimize100": "anti_overfitting_split_v1_fold_a_plus_b",
                               "shadow50": "anti_overfitting_split_v1_fold_c"},
                 "split_hashes": {"optimize100_ordered_ids_sha256": digest(ordered_ids),
                                  "shadow50_ids_sha256": logical_hash(shadow_ids)},
                 "validation_policy": "prohibited; Validation50 calls=0",
                 "test_policy": "prohibited; Test50 calls=0"},
        "model": {"solver": "qwen3-8b", "optimizer_roles": {}, "thinking": False,
                  "temperatures": {"solver": 0.0}, "max_tokens": {"solver": 1800}},
        "seeds": [78],
        "design": {
            "changed": ["read-only replay of all five frozen accepted mutations"],
            "unchanged": ["five-member plurality", "tie-as-abstain", "TeamMiniBatch12",
                          "Common-Safe", "Shadow gate", "COMMON_SOLVER_CONTRACT_V1"],
            "forbidden_changes": ["candidate generation", "candidate subset selection", "GEPA",
                                  "Reflection", "scheduler", "writeback", "persistent realizability",
                                  "Validation50", "Test50"],
            "baseline_team_hash": BASELINE_STATE, "candidate_count": 5,
            "candidate_freeze_sha256": sha(args.report / "accepted_mutation_freeze.json"),
            "private_bundle_sha256": sha(args.private_bundle),
            "decomposition_version": TEAM_TRANSFER_DECOMPOSITION_VERSION,
            "formal_run_root": "runs/accepted_local_mutation_team_transfer_v1_attempt1",
            "protocol_sha256": sha(protocol),
        },
        "api_authorization": {"authorized": False, "authorization_scope": "pending explicit team-replay authorization",
                              "allowed_roles": [], "allowed_phases": []},
        "budget": {"type": "progressive_read_only_five_pair_replay", "frozen_before_run": True,
                   "limit": {"mutation_pairs": 5, "successful_solver_provider_calls": 800,
                             "transport_attempts": 3200, "per_request_attempts": 4,
                             "solver_concurrency": 8,
                             "Reflection": 0, "GEPA": 0, "writeback": 0,
                             "persistent_realizability": 0, "Validation50": 0, "Test50": 0}},
        "selection": {"primary_metric": "five_case_local_to_team_transfer_mapping",
                      "frozen_rule": "include all five accepted mutations in deterministic order; no ranking or subset",
                      "validation_used_for_selection": False, "test_used_for_selection": False},
        "artifacts": {"preregistration": {"path": protocol.relative_to(ROOT).as_posix()},
                      "report": args.report.relative_to(ROOT).as_posix(),
                      "provenance": (args.report / "provenance.json").relative_to(ROOT).as_posix()},
        "git": {"design_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "implementation_commit": None, "result_commit": None},
        "result": {"classifier": "PREREGISTERED_NOT_EXECUTED",
                   "conclusion": "Team transfer remains unevaluated pending separate API authorization.",
                   "evidence_type": "not_yet_available"},
    }
    manifest["artifacts"]["preregistration"]["sha256"] = preregistration_hash(manifest)
    schema = read(ROOT / "infrastructure/experiment_manifest.schema.json")
    errors = validate_manifest(manifest, schema)
    if errors:
        raise ValueError("manifest invalid: " + "; ".join(errors))
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)

    write(args.report / "provenance.json", {
        "source_phase_a_tasks_sha256": sha(args.phase_a_tasks),
        "source_phase_b_execution_sha256": sha(args.phase_b_run / "execution.json"),
        "source_phase_b_completion_sha256": sha(args.phase_b_run / "completion.json"),
        "source_phase_b_provider_ledger_sha256": sha(args.phase_b_run / "provider_ledger.jsonl"),
        "source_phase_b_bundle_sha256": sha(args.phase_b_bundle / "selected_parent_tasks_private.json"),
        "shadow50_csv_sha256": sha(args.shadow_csv),
        "protocol_sha256": sha(protocol), "manifest_sha256": sha(manifest_path),
        "raw_prompts_published": False, "raw_questions_published": False,
        "raw_answers_published": False,
    })
    write(args.report / "EXPERIMENT_HANDOFF.json", {
        "schema_version": "sol_luna_experiment_handoff_v1", "experiment_id": IDENTITY,
        "READY_TO_RUN": False, "authorization": "PENDING_EXPLICIT_API_AUTHORIZATION",
        "exact_runner_command": None, "private_bundle_sha256": sha(args.private_bundle),
        "candidate_freeze_sha256": sha(args.report / "accepted_mutation_freeze.json"),
        "baseline_team_hash": BASELINE_STATE,
        "blockers": ["api_authorization_pending", "clean_commit_and_source_hash_freeze_pending"],
    })
    (args.report / "README.md").write_text(
        "# Accepted local mutation team-transfer v1 preparation\n\n"
        "Status: **PREREGISTERED_NOT_EXECUTED**. All five Phase-B accepted mutations "
        "are frozen for independent read-only replay against the same baseline team. "
        "Preparation made zero API calls. Team transfer remains unevaluated.\n",
        encoding="utf-8",
    )
    findings = scan_sanitized_artifacts(args.report)
    write(args.report / "sanitization_manifest.json", {
        "status": "PASS" if not findings else "FAIL", "findings": findings,
    })
    write(args.report / "sha256_manifest.json", build_sha256_manifest(args.report))
    if findings or read(args.report / "sha256_manifest.json") != build_sha256_manifest(args.report):
        raise ValueError("public report sanitization/hash replay failed")
    return {"status": "PREREGISTERED_NOT_EXECUTED", "mutation_count": 5,
            "api_calls": 0, "READY_TO_RUN": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-tasks", type=Path, required=True)
    parser.add_argument("--phase-b-run", type=Path, required=True)
    parser.add_argument("--phase-b-bundle", type=Path, required=True)
    parser.add_argument("--shadow-csv", type=Path, required=True)
    parser.add_argument("--private-bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
