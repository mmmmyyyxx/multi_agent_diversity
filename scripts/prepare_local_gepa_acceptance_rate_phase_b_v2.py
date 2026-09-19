"""Finalize the zero-API Phase-B 4x8 preregistration from frozen Phase-A tasks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from infrastructure.common_solver_contract_v1.contract import CONTRACT_SPEC
from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPAOptimizerConfig
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import verify_frozen_gepa
from multi_dataset_diverse_rl.local_optimizers.proposal_telemetry import cost_envelope, evaluation_schema
from multi_dataset_diverse_rl.parent_acquisition import digest, restore_task, select_parents, task_integrity
from multi_dataset_diverse_rl.versions import LOCAL_GEPA_PHASE_B_TELEMETRY_VERSION, METHOD_VERSION

IDENTITY = "local_gepa_acceptance_rate_pilot_phase_b_v2"
EXPERIMENT = ROOT / "experiments" / IDENTITY
PROTOCOL = EXPERIMENT / "PROTOCOL.md"
MANIFEST = ROOT / "experiments/manifests" / f"{IDENTITY}.yaml"
REPORT = ROOT / "reports" / f"{IDENTITY}_prep_20260918"
PHASE_A = ROOT / "reports/local_gepa_parent_acquisition_v1"
FORMAL_RUN = ROOT / "runs" / f"{IDENTITY}_attempt1"
PRIVATE_FREEZE = ROOT / "runs" / f"{IDENTITY}_freeze"
EXPECTED_SELECTED = [f"seed78_update0_member{i}" for i in range(1, 5)]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def sanitized_parent(row: dict) -> dict:
    keys = (
        "parent_task_id", "source_state_hash", "source_seed", "update_index",
        "target_member", "primary_responsibility_lane", "parent_prompt_hash",
        "task_payload_sha256", "local_problem_sha256", "search_example_identity_hash",
        "search_example_ids", "search_payload_sha256", "local_validation_identity_hash",
        "local_validation_example_ids",
        "local_validation_payload_sha256", "optimization_context_sha256", "task_seed",
    )
    return {key: row[key] for key in keys}


def prepare(private_tasks_path: Path, *, refresh_existing: bool = False) -> dict:
    if FORMAL_RUN.exists():
        raise ValueError("formal_run_root_must_be_absent")
    if not refresh_existing and (REPORT.exists() or MANIFEST.exists() or PRIVATE_FREEZE.exists()):
        raise FileExistsError("Phase-B freeze outputs must be fresh")
    if refresh_existing:
        if not MANIFEST.exists() or yaml.safe_load(MANIFEST.read_text()).get("experiment_id") != IDENTITY:
            raise ValueError("refresh target identity mismatch")
    phase_a_hashes = read(PHASE_A / "sha256_manifest.json")
    if phase_a_hashes != build_sha256_manifest(PHASE_A):
        raise ValueError("Phase-A public artifact hash replay failed")
    phase_a_cost = read(PHASE_A / "acquisition_cost.json")
    if (phase_a_cost["successful_solver_requests"], phase_a_cost["total_tokens"]) != (100, 26113):
        raise ValueError("Phase-A cost identity mismatch")
    catalog = read(PHASE_A / "eligible_parent_catalog.json")
    selected_ids = read(PHASE_A / "selected_parent_ids.json")
    if selected_ids != EXPECTED_SELECTED or select_parents(catalog) != EXPECTED_SELECTED:
        raise ValueError("frozen deterministic parent selection mismatch")
    if len(catalog) != 5 or len({row["source_state_hash"] for row in catalog}) != 1:
        raise ValueError("single-state five-parent acquisition required")
    tasks = read(private_tasks_path)
    if set(tasks) != {row["parent_task_id"] for row in catalog}:
        raise ValueError("private Phase-A task catalog mismatch")
    selected_rows = [row for row in catalog if row["parent_task_id"] in selected_ids]
    unselected_rows = [row for row in catalog if row["parent_task_id"] not in selected_ids]
    for row in catalog:
        payload = tasks[row["parent_task_id"]]
        if digest(payload) != row["task_payload_sha256"]:
            raise ValueError("parent task payload hash mismatch")
        task = restore_task(payload)
        if task.task_id != row["parent_task_id"] or task.seed != row["task_seed"]:
            raise ValueError("parent task identity mismatch")
        examples = [{"example_id": item.example_id, "question": item.input_payload, "gold": item.gold}
                    for item in task.search_examples]
        task_integrity(task, examples, row["target_member"], row["primary_responsibility_lane"])
    PRIVATE_FREEZE.mkdir(parents=True, exist_ok=refresh_existing)
    private_bundle = PRIVATE_FREEZE / "selected_parent_tasks_private.json"
    write(private_bundle, {identity: tasks[identity] for identity in selected_ids})

    subset_audit = []
    for group in combinations(catalog, 4):
        ids = sorted(row["parent_task_id"] for row in group)
        subset_audit.append({
            "parent_task_ids": ids,
            "coverage": {"source_states": len({r["source_state_hash"] for r in group}),
                         "targets": len({r["target_member"] for r in group}),
                         "primary_lanes": len({r["primary_responsibility_lane"] for r in group})},
            "stable_subset_sha256": digest(ids),
            "selected": ids == selected_ids,
        })
    source_state = selected_rows[0]["source_state_hash"]
    parent_freeze = {
        "schema_version": "phase_b_parent_freeze_v1", "experiment_id": IDENTITY,
        "source_acquisition_artifact": "reports/local_gepa_parent_acquisition_v1",
        "source_acquisition_manifest_sha256": file_hash(PHASE_A / "sha256_manifest.json"),
        "source_acquisition_private_tasks_sha256": file_hash(private_tasks_path),
        "private_selected_task_bundle_sha256": file_hash(private_bundle),
        "shared_baseline_state_hash": source_state, "shared_baseline_state_count": 1,
        "selected_parents": [sanitized_parent(row) for row in selected_rows],
        "unselected_eligible_parent": {**sanitized_parent(unselected_rows[0]),
            "exclusion_reason": "stable_subset_sha256_tiebreak_after_equal_source_target_lane_coverage",
            "future_use": "separately_preregistered_replication_only"},
        "selection_rule": "exact maximum source-state/target/lane coverage then minimum stable subset SHA256",
        "selection_used_phase_b_outcomes": False, "mismatch_policy": "FAIL_CLOSED",
    }

    REPORT.mkdir(parents=True, exist_ok=refresh_existing)
    write(REPORT / "PARENT_FREEZE.json", parent_freeze)
    write(REPORT / "parent_selection_audit.json", {
        "eligible_count": 5, "selected_count": 4, "selected_parent_ids": selected_ids,
        "unselected_parent_id": unselected_rows[0]["parent_task_id"],
        "unselected_reason": parent_freeze["unselected_eligible_parent"]["exclusion_reason"],
        "subsets": sorted(subset_audit, key=lambda row: row["parent_task_ids"]),
        "deterministic_replay": "PASS", "outcome_blind": True,
    })

    envelope = cost_envelope(8, 12, 3, 16)
    phase_b_cost = {
        "phase_a_acquisition_cost": {"successful_solver_requests": 100, "prompt_tokens": 19665,
                                     "completion_tokens": 6448, "total_tokens": 26113,
                                     "acceptance_denominator_contribution": 0},
        "phase_b_reflection_cost": {"logical_call_ceiling": 32, "output_token_ceiling": 57600,
                                    "model": "qwen3.7-flash"},
        "phase_b_local_solver_cost": {"metric_row_ceiling": 820,
                                      "transport_attempt_ceiling": 820 * CONTRACT_SPEC.transport_attempt_cap,
                                      "successful_output_token_ceiling": 820 * CONTRACT_SPEC.max_tokens,
                                      "model": CONTRACT_SPEC.model},
        "per_parent": envelope, "parent_count": 4, "proposal_event_ceiling": 32,
        "expected_normal": {"proposal_events": 32, "reflection_logical_calls": 32,
                            "local_solver_metric_rows": 624, "nominal_transport_attempts": 624,
                            "solver_output_token_ceiling": 1123200,
                            "reflection_output_token_ceiling": 57600},
        "worst_case_hard_ceiling": {"proposal_events": 32, "reflection_logical_calls": 32,
                                    "local_solver_metric_rows": 820, "transport_attempts": 3280,
                                    "solver_output_tokens": 1476000,
                                    "reflection_output_tokens": 57600},
        "input_token_ceiling": None, "currency_ceiling": None,
        "unreported_failed_attempt_usage_policy": "disclose_if_provider_does_not_return_usage",
        "accounting_categories_must_remain_separate": True,
    }
    write(REPORT / "cost_envelope.json", phase_b_cost)
    write(REPORT / "proposal_quota_freeze.json", {
        "parent_count": 4, "actual_proposal_end_per_parent": 8,
        "planned_actual_proposal_end_total": 32, "skip_allowance_per_parent": 16,
        "max_metric_calls_per_parent": 205, "counter": "official on_proposal_end",
        "per_parent_invariant": "actual_proposal_end <= 8",
        "normal_completion": "actual_proposal_end == 8",
        "shortfall_status": "PARENT_PROPOSAL_QUOTA_INCOMPLETE", "resume_or_replacement": "PROHIBITED",
    })
    write(REPORT / "telemetry_contract.json", {
        "version": LOCAL_GEPA_PHASE_B_TELEMETRY_VERSION,
        "evaluation_batch_schema": evaluation_schema(),
        "proposal_outcomes": ["STRICT_POSITIVE", "EXACT_EQUAL", "STRICT_NEGATIVE"],
        "zero_delta_decomposition": ["BEHAVIORAL_NO_OP", "REPAIR_PRESERVATION_CANCELLATION"],
        "primary_estimand": "pooled accepted / contract-valid changed Solver-reached; DESCRIPTIVE_NON_IID",
        "required_parent_rates": selected_ids,
        "required_parent_level_statistic": "parents_with_at_least_one_acceptance / 4",
        "iid_inference_permitted": False, "optional_interval_label": "NAIVE_IID_REFERENCE_ONLY",
        "raw_text_publication": "PROHIBITED",
    })
    write(REPORT / "interpretation_matrix.json", {
        "acceptance_multiple_parents": "LOCAL_GEPA_STRICT_IMPROVEMENT_OBSERVED_ACROSS_MULTIPLE_MEMBER_TASKS",
        "acceptance_one_parent": "STRONG_MEMBER_TASK_HETEROGENEITY",
        "mostly_equal_no_op": "LOW_BEHAVIORAL_EFFECT_OF_PROPOSALS",
        "fix_and_break_common": "LOCAL_REPAIR_PRESERVATION_TRADEOFF",
        "similar_edits_concentrated_patterns": "SEARCH_EXPLORATION_CONCENTRATION",
        "diverse_edits_patterns_near_zero": "LOCAL_OBJECTIVE_OR_FEEDBACK_LIMITATION",
        "always": "TEAM_TRANSFER_NOT_EVALUATED", "automatic_method_change": False,
    })
    write(REPORT / "final_report_template.json", {
        "scope": "single-baseline-state multi-member Layer-1 GEPA acceptance-rate pilot",
        "shared_source_state_hash": source_state, "parent_summaries_required": selected_ids,
        "pooled": {"accepted": None, "denominator": None, "descriptive_rate": None},
        "parents_with_at_least_one_acceptance": {"numerator": None, "denominator": 4, "rate": None},
        "per_parent_required_fields": ["proposal_attempts", "contract_valid", "solver_reached",
            "positive", "equal", "negative", "accepted", "total_newly_fixed", "total_newly_broken",
            "preservation_losses", "mean_edit_similarity", "median_edit_similarity",
            "unique_failure_patterns", "dominant_failure_pattern_concentration",
            "max_lineage_depth", "accepted_generation_count"],
        "same_state_limitation_required": True, "team_transfer": "TEAM_TRANSFER_NOT_EVALUATED",
        "cross_state_generalization": "PROHIBITED", "inferential_ci_main_conclusion": "PROHIBITED",
    })

    now = datetime.now(timezone.utc).isoformat()
    config = asdict(GEPAOptimizerConfig())
    manifest = {
        "schema_version": "experiment_manifest_v1", "experiment_id": IDENTITY,
        "title": "Single-state multi-member Layer-1 GEPA 4x8 descriptive pilot",
        "status": "PREFLIGHT_PASS", "legacy_index": False,
        "lifecycle_history": [{"status": status, "timestamp": now}
                              for status in ("DRAFT", "PREREGISTERED", "IMPLEMENTED", "PREFLIGHT_PASS")],
        "lineage": {"parents": ["local_gepa_parent_acquisition_v1"],
                    "derives_from": "local_gepa_parent_acquisition_v1"},
        "scientific_question": "How does official GEPA local search behave across four member-specific tasks in one fixed baseline state?",
        "hypotheses": ["Local strict improvement and failure modes may differ by member task within the fixed state."],
        "method_identity": METHOD_VERSION, "runtime_version": IDENTITY,
        "data": {"task": "BBH disambiguation_qa", "formal": False,
                 "split_ids": {"search_and_local_validation": "Optimize100 only", "validation50": "not_accessed", "test50": "not_accessed"},
                 "split_hashes": {"source_acquisition_manifest_sha256": parent_freeze["source_acquisition_manifest_sha256"]},
                 "validation_policy": "Optimize-derived local validation only; heldout Validation50 calls=0",
                 "test_policy": "Test50 calls=0"},
        "model": {"solver": CONTRACT_SPEC.model, "optimizer_roles": {"reflection": "qwen3.7-flash"},
                  "thinking": False, "temperatures": {"solver": 0.0, "reflection": 0.0},
                  "max_tokens": {"solver": 1800, "reflection": 1800}},
        "seeds": [row["task_seed"] for row in selected_rows],
        "design": {"changed": ["single-state four-member descriptive scope", "non-IID outcome reporting"],
                   "unchanged": ["official GEPA core", "Level-B adapter", "reflection template and dataset", "local tasks"],
                   "forbidden_changes": ["parent reacquisition or reselection", "GEPA core", "reflection template", "local validation size", "scheduler", "TeamMiniBatch", "Common-Safe", "Shadow", "team write-back", "persistent realizability"],
                   "attempt_id": IDENTITY + "_attempt1", "formal_run_root": "runs/" + IDENTITY + "_attempt1",
                   "scope": "single_baseline_state_multi_member_layer1_gepa",
                   "shared_baseline_state_hash": source_state,
                   "selected_parents": [sanitized_parent(row) for row in selected_rows],
                   "unselected_eligible_parent": sanitized_parent(unselected_rows[0]),
                   "parent_freeze_sha256": file_hash(REPORT / "PARENT_FREEZE.json"),
                   "proposal_quota_per_parent": 8, "skipped_iteration_allowance": 16,
                   "official_gepa": config, "protocol_sha256": file_hash(PROTOCOL)},
        "api_authorization": {"authorized": False, "authorization_scope": "pending new explicit Phase-B authorization",
                              "allowed_roles": [], "allowed_phases": []},
        "budget": {"type": "frozen_actual_proposal_event_quota_with_hard_metric_ceiling", "frozen_before_run": True,
                   "limit": {"parents": 4, "proposals_per_parent": 8, "total_proposals": 32,
                             "max_metric_calls_per_parent": 205, "total_metric_rows": 820,
                             "reflection_calls": 32, "skip_allowance_per_parent": 16,
                             "heldout_Validation50": 0, "Test50": 0}},
        "selection": {"primary_metric": "pooled_strict_improvement_rate_descriptive_non_iid",
                      "frozen_rule": "official accepted / contract-valid changed Solver-reached, plus four parent rates and parents-with-acceptance/4; no main inferential CI",
                      "validation_used_for_selection": False, "test_used_for_selection": False},
        "artifacts": {"preregistration": {"path": PROTOCOL.relative_to(ROOT).as_posix()},
                      "report": REPORT.relative_to(ROOT).as_posix(),
                      "provenance": (REPORT / "provenance.json").relative_to(ROOT).as_posix()},
        "git": {"design_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "implementation_commit": None, "result_commit": None},
        "result": {"classifier": "PREREGISTERED_NOT_EXECUTED", "conclusion": "Phase-B API authorization pending; team transfer not evaluated.", "evidence_type": "not_yet_available"},
    }
    manifest["artifacts"]["preregistration"]["sha256"] = preregistration_hash(manifest)
    errors = validate_manifest(manifest, json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text()))
    if errors:
        raise ValueError(errors)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    (REPORT / "manifest_snapshot.yaml").write_bytes(MANIFEST.read_bytes())
    write(REPORT / "summary.json", {"status": "PHASE_B_PREREGISTERED_AUTHORIZATION_PENDING",
          "scope": manifest["design"]["scope"], "selected_parent_ids": selected_ids,
          "shared_baseline_state_count": 1, "planned_proposal_events": 32,
          "formal_run_root_absent": True, "phase_b_api_calls": 0,
          "Validation50_calls": 0, "Test50_calls": 0, "team_transfer": "TEAM_TRANSFER_NOT_EVALUATED"})
    write(REPORT / "classifier.json", {"classifier": "PREREGISTERED_NOT_EXECUTED",
          "parent_acquisition": "PARENT_ACQUISITION_COMPLETE", "phase_b_execution": "NOT_AUTHORIZED",
          "scientific_evidence": "NOT_YET_AVAILABLE"})
    isolation = {"phase_b_api_calls_zero": True, "formal_run_root_absent": True,
        "Validation50_calls_zero": True, "Test50_calls_zero": True, "TeamMiniBatch_calls_zero": True,
        "full_team_calls_zero": True, "Common_Safe_calls_zero": True, "Shadow_calls_zero": True,
        "team_writebacks_zero": True, "persistent_realizability_updates_zero": True}
    write(REPORT / "evaluation_access_summary.json", isolation)
    write(REPORT / "api_ledger_summary.json", {"phase_b_provider_calls": 0, "phase_b_tokens": 0,
        "reflection_calls": 0, "local_solver_calls": 0, "authorization": "PENDING",
        "phase_a_cost_reference": {"successful_calls": 100, "tokens": 26113}})
    write(REPORT / "fact_assertions.json", {**isolation,
        "exact_four_parent_identity_replay": True, "single_shared_baseline_state": True,
        "fifth_parent_recorded": True, "selection_outcome_blind": True,
        "proposal_quota_frozen": True, "cost_ceiling_frozen": True,
        "iid_inference_prohibited": True, "team_transfer_not_evaluated": True,
        "official_gepa_core_unchanged": True, "READY_TO_RUN_false": True})
    source_paths = [
        ROOT / "multi_dataset_diverse_rl/local_optimizers/proposal_telemetry.py",
        ROOT / "multi_dataset_diverse_rl/local_optimizers/gepa_adapter.py",
        ROOT / "multi_dataset_diverse_rl/local_optimizers/gepa_optimizer.py",
        ROOT / "multi_dataset_diverse_rl/versions.py",
        ROOT / "scripts/run_local_gepa_acceptance_rate_pilot.py",
        ROOT / "scripts/prepare_local_gepa_acceptance_rate_phase_b_v2.py",
        ROOT / "scripts/preflight_local_gepa_acceptance_rate_phase_b_v2.py",
        ROOT / "tests/test_local_gepa_acceptance_rate_preparation.py",
        ROOT / "tests/test_local_gepa_phase_b_v2_freeze.py",
        PROTOCOL,
    ]
    write(REPORT / "provenance.json", {"base_commit": manifest["git"]["design_commit"],
        "phase_a_source_commit": read(PHASE_A / "provenance.json")["source_commit"],
        "phase_a_report_manifest_sha256": parent_freeze["source_acquisition_manifest_sha256"],
        "phase_a_private_tasks_sha256": parent_freeze["source_acquisition_private_tasks_sha256"],
        "phase_b_private_selected_tasks_sha256": parent_freeze["private_selected_task_bundle_sha256"],
        "protocol_sha256": file_hash(PROTOCOL), "manifest_sha256": file_hash(MANIFEST),
        "official_gepa": verify_frozen_gepa(),
        "source_files": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in source_paths},
        "historical_phase_a_modified": False,
        "historical_phase_b_v1_preparation_modified": False})
    handoff = {"schema_version": "sol_luna_experiment_handoff_v1", "experiment_id": IDENTITY,
        "source": {"repository_commit": None, "tracked_worktree_status": "freeze changes not committed"},
        "protocol": {"name": IDENTITY, "version": 2, "protocol_sha256": file_hash(PROTOCOL),
                     "manifest_path": MANIFEST.relative_to(ROOT).as_posix(), "manifest_sha256": file_hash(MANIFEST),
                     "preregistration_path": PROTOCOL.relative_to(ROOT).as_posix(),
                     "preregistration_sha256": preregistration_hash(manifest)},
        "data": {"dataset": "BBH disambiguation_qa", "split_identity": "Optimize100 only",
                 "split_sha256": parent_freeze["source_acquisition_manifest_sha256"]},
        "models": {"solver": CONTRACT_SPEC.model, "optimizer_or_reflection": "qwen3.7-flash",
                   "thinking": False, "local_optimizer_backend": "official GEPA Level-B",
                   "local_optimizer_version_or_sha256": verify_frozen_gepa()["source_sha256"]},
        "execution": {"seeds": manifest["seeds"], "opportunity_budget": manifest["budget"]["limit"],
                      "early_stop_rule": "8 actual proposal_end or frozen terminal condition per parent",
                      "exact_runner_command": None, "expected_output_directory": manifest["design"]["formal_run_root"],
                      "expected_checkpoint_path": None, "expected_ledger_path": None,
                      "allowed_retries": "COMMON_SOLVER_CONTRACT_V1 only; no experiment resume"},
        "authorization": {"api_scope": "PENDING", "validation_access_policy": "NO_ACCESS",
                          "test_access_policy": "NO_ACCESS"},
        "fail_closed_conditions": ["parent/source/hash mismatch", "proposal quota or budget mismatch",
            "missing clean implementation commit", "missing new explicit authorization", "heldout access",
            "need to change method or parent selection"], "READY_TO_RUN": False}
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    write(EXPERIMENT / "EXPERIMENT_HANDOFF.json", handoff)
    (REPORT / "README.md").write_text(
        "# Phase-B 4x8 zero-API freeze\n\n"
        "**PHASE_B_PREREGISTERED_AUTHORIZATION_PENDING. READY_TO_RUN=false.**\n\n"
        "Exact members 1-4 are frozen from one shared baseline state. This design can describe "
        "member/task heterogeneity within that state; it cannot support cross-state generalization, "
        "population acceptance inference, scheduler claims or team transfer. The fifth eligible "
        "member-0 task is retained in the selection audit and cannot replace a selected parent.\n\n"
        "The primary pooled rate and four parent rates are descriptive. Proposal outcomes are "
        "adaptive and dependent; no inferential CI belongs in the main conclusion. Exact equality, "
        "degradation, behavioral no-op and repair/preservation cancellation remain distinct.\n\n"
        "Phase-A acquisition cost is preserved separately. Phase-B has made zero API calls and its "
        "formal run root does not exist. Validation50, Test50 and every team-level stage remain zero.\n",
        encoding="utf-8")
    findings = scan_sanitized_artifacts(REPORT)
    write(REPORT / "sanitization_manifest.json", {"status": "PASS" if not findings else "FAIL", "findings": findings})
    write(REPORT / "sha256_manifest.json", build_sha256_manifest(REPORT))
    if findings:
        raise ValueError("Phase-B report sanitization failed")
    return {"status": "PHASE_B_PREREGISTERED_AUTHORIZATION_PENDING", "selected_parent_ids": selected_ids,
            "shared_state_hash": source_state, "phase_b_api_calls": 0, "READY_TO_RUN": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-tasks", type=Path, required=True)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.private_tasks, refresh_existing=args.refresh_existing), indent=2))
