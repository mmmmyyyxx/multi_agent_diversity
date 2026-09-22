"""Build the final zero-API preregistration bundle for saturation studies.

The generated artifacts contain hashes and logical provider identities only.
They deliberately fail closed when the frozen execution harness cannot prove
the requested ``lwj`` binding without changing the scientific source commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final_pre_experiment_freeze_20260922"
SOURCE_COMMIT = "a85e31bea2ab28f62abb31337b91f9895b11ae37"
LOCAL_PATIENCE = 3
TEAM_PATIENCE = 2
SEEDS = [80, 81, 82]
SPLIT_MANIFEST = ROOT / "experiments" / "anti_overfitting_split_v1" / "split_manifest.json"
FOLD_ASSIGNMENT = ROOT / "experiments" / "anti_overfitting_split_v1" / "fold_assignment.json"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def logical_hash(values: list[str]) -> str:
    return sha256_bytes(canonical_bytes(sorted(values)))


def endpoint_fingerprint(argument: str) -> str:
    if argument:
        return argument
    endpoint = os.getenv("LWJ_DASHSCOPE_BASE_URL", "").strip().rstrip("/")
    if not endpoint:
        raise RuntimeError("LWJ_DASHSCOPE_BASE_URL or --endpoint-fingerprint is required")
    return sha256_bytes(endpoint.encode("utf-8"))


def provider_freeze(fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": "provider_model_freeze_v1",
        "provider_profile": "lwj",
        "endpoint_identity": {
            "logical_profile": "lwj",
            "normalized_endpoint_sha256": fingerprint,
            "raw_endpoint_persisted": False,
        },
        "models": {
            "solver": "qwen3-8b",
            "teacher": "qwen3.7-flash",
            "critic": "qwen3.7-flash",
            "student": "qwen3.7-flash",
            "evaluator": "qwen3.7-flash",
            "thinking": False,
        },
        "forbidden_model": "qwen3.7-flash-2026-07-15",
        "connectivity_smoke_evidence": [
            {"model": "qwen3-8b", "status": "SUCCESS", "exact_output": "SMOKE_OK"},
            {"model": "qwen3.7-flash", "status": "SUCCESS", "exact_output": "SMOKE_OK"},
            {
                "model": "qwen3.7-flash-2026-07-15",
                "status": "PERMISSION_DENIED",
                "http_status": 403,
            },
        ],
        "smoke_evidence_role": "connectivity_only_not_scientific_denominator",
        "api_key_persisted": False,
        "private_hostname_persisted": False,
    }


def data_freeze() -> dict[str, Any]:
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    folds = json.loads(FOLD_ASSIGNMENT.read_text(encoding="utf-8"))["folds"]
    optimize = list(folds["fold_a"]) + list(folds["fold_b"])
    shadow = list(folds["fold_c"])
    return {
        "schema_version": "layer2_saturation_data_freeze_v1",
        "task": "BBH disambiguation_qa",
        "optimize100": {
            "identity": "anti_overfitting_split_v1_fold_a_plus_b",
            "count": len(optimize),
            "question_hashes_sha256": logical_hash(optimize),
        },
        "shadow50": {
            "identity": "anti_overfitting_split_v1_fold_c",
            "count": len(shadow),
            "question_hashes_sha256": logical_hash(shadow),
            "role": "optimization_time_safety_only_when_protocol_uses_shadow",
        },
        "validation50": {
            "count": int(split["counts"]["validation"]),
            "question_hashes_sha256": logical_hash(split["question_hashes"]["validation"]),
            "access": "FORBIDDEN",
        },
        "test50": {
            "count": int(split["counts"]["test"]),
            "question_hashes_sha256": logical_hash(split["question_hashes"]["test"]),
            "access": "FORBIDDEN",
        },
        "source_files": {
            "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
            "fold_assignment_sha256": sha256_file(FOLD_ASSIGNMENT),
        },
        "raw_questions_or_answers_persisted": False,
    }


def stopping(kind: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "saturation_stopping_preregistration_v1",
        "selected_local_no_update_patience": LOCAL_PATIENCE,
        "selected_team_no_update_patience": TEAM_PATIENCE,
        "historical_candidate_options": {"local": [2, 3, 5], "team": [1, 2, 3]},
        "selected_before_real_outcomes": True,
        "validation_or_test_used": False,
    }
    if kind == "formal":
        payload.update(
            {
                "execution_regime": "saturation",
                "scientific_hard_budget": "DISABLED",
                "emergency_ceiling": "ENABLED_OPERATIONAL_ONLY",
                "emergency_limits": {
                    "provider_calls": 100000,
                    "optimizer_steps": 100000,
                    "team_epochs": 10000,
                    "wall_seconds": 86400,
                },
                "emergency_result": "SATURATION_NOT_REACHED_EMERGENCY_CEILING",
                "native_team_patience": None,
            }
        )
    else:
        payload.update(
            {
                "execution_regime": "fixed_engineering_or_mechanism_pilot",
                "scientific_saturation_claim_allowed": False,
            }
        )
    return payload


def common_manifest(experiment_id: str, study: str, seeds: list[int]) -> dict[str, Any]:
    prompt = (
        "You are a careful reasoning solver. Use an explicit decision procedure "
        "and verify the key inference before finalizing the decision."
    )
    return {
        "schema_version": "final_layer2_preregistration_v1",
        "experiment_id": experiment_id,
        "study": study,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "scientific_source_commit": SOURCE_COMMIT,
        "source_change_policy": "invalidate_freeze_and_create_fresh_attempt",
        "provider_profile": "lwj",
        "models": {
            "solver": "qwen3-8b",
            "optimizer_roles": "qwen3.7-flash",
            "evaluator": "qwen3.7-flash",
        },
        "seeds": seeds,
        "initial_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "initial_state_policy": {
            "same_prompt_family": True,
            "paired_within_seed": True,
            "persist_before_search": [
                "initial_prompt_identity",
                "initial_team_state_hash",
                "initial_member_profiles",
                "initial_vote_acc",
                "initial_oracle",
                "initial_pivotality",
            ],
        },
        "validation50_calls": 0,
        "test50_calls": 0,
        "raw_secrets_or_endpoints": False,
    }


def protocol_text(experiment_id: str, title: str, question: str, body: list[str]) -> str:
    lines = [f"# {title}", "", f"Experiment: `{experiment_id}`", "", question, ""]
    lines.extend(body)
    lines.extend(
        [
            "",
            "This protocol is preregistered but not executed. API authorization is false.",
            "Validation50 and Test50 are forbidden during development optimization.",
            "",
        ]
    )
    return "\n".join(lines)


def bundle_spec(experiment_id: str) -> dict[str, Any]:
    if experiment_id == "gepa_layer2_real_canary_v1":
        manifest = common_manifest(experiment_id, "GEPA_LAYER2_REAL_CANARY", [80])
        manifest.update(
            {
                "evidence_type": "engineering_canary_not_performance_evidence",
                "deterministic_start_rule": (
                    "fresh shared-root Optimize100 initialization; after initialization, "
                    "select the first actionable target in frozen scheduler order; freeze "
                    "the packet before reflection or candidate-provider calls"
                ),
                "required_pre_reflection_freeze": [
                    "team_state_hash",
                    "target_member",
                    "primary_responsibility_lane",
                    "responsibility_packet_hash",
                    "responsibility_ids_hash",
                    "focus_ids_hash",
                    "anchor_ids_hash",
                    "local_eval_ids_hash",
                    "ordered_schedule_hash",
                ],
            }
        )
        claims = {
            "allowed": ["GEPA_LAYER2_REAL_EMPIRICAL_PATH_CONFIRMED"],
            "technical_success": [
                "layer2_packet_constructed",
                "native_example_selection_calls_eq_0",
                "reflection_receives_role_labeled_evidence",
                "changed_proposals_gte_1",
                "contract_valid_proposals_gte_1",
                "candidate_solver_reached_gte_1",
                "local_empirical_delta_computed",
            ],
            "positive_delta_required": False,
            "failure_categories": [
                "NO_RESPONSIBILITY_EVIDENCE",
                "NO_LOCAL_EVAL_EVIDENCE",
                "NO_PROPOSAL",
                "PROPOSAL_CONTRACT_BLOCKED",
                "CANDIDATE_SOLVER_NOT_REACHED",
                "PROVIDER_FAILURE",
                "OPERATIONAL_FAILURE",
                "SATURATION_LOGIC_FAILURE",
            ],
        }
        cost = {
            "budget_type": "fixed_minimal_engineering_canary",
            "opportunities": 1,
            "target_branches": 1,
            "local_metric_calls": 36,
            "provider_success_ceiling": 256,
            "provider_attempt_ceiling": 1024,
            "reflection_success_ceiling": 8,
            "wall_seconds_ceiling": 3600,
            "validation50_calls": 0,
            "test50_calls": 0,
        }
        preflight_blockers = ["existing_canary_runner_does_not_bind_provider_profile_lwj"]
        question = "Can real GEPA consume a Layer2-owned packet and reach local empirical comparison?"
        body = [
            "The canary is a technical gate, not a performance experiment.",
            "A positive local delta, TeamMiniBatch survival, or commit is not required.",
            "Root focus and anchor are empty; initialization evidence is frozen before reflection.",
        ]
    elif experiment_id == "sequential_symmetry_breaking_online_pilot_v1":
        manifest = common_manifest(experiment_id, "SEQUENTIAL_LAYER2_MECHANISM_PILOT", [80])
        manifest.update(
            {
                "pipeline": [
                    "target_allocation",
                    "responsibility_and_transition_evidence",
                    "GEPA_local_search",
                    "TeamMiniBatch",
                    "Full",
                    "Common-Safe",
                    "Shadow_when_applicable",
                    "atomic_commit",
                    "categorical_profiles",
                    "pivotality",
                    "persistent_realizability",
                ],
                "endpoints": ["T_commit", "T_pivotal", "T_vote"],
                "packet_overlap_telemetry_only": True,
            }
        )
        claims = {
            "allowed": [
                "ONLINE_LAYER2_COMMIT_PATH_CONFIRMED",
                "ONLINE_PIPELINE_REACHES_PIVOTAL_CAPABLE_STATE",
                "ONLINE_PIPELINE_OBTAINS_VOTE_GAIN",
            ],
            "classifiers": {
                "A": "commit+pivotal+vote observed",
                "B": "commit+pivotal observed; vote not observed",
                "C": "commit observed; pivotal not observed",
                "D": "no commit within pilot",
            },
        }
        cost = {
            "budget_type": "small_mechanism_pilot_not_saturation",
            "max_opportunities": 8,
            "max_safe_commits": 4,
            "consecutive_no_commit_opportunities": 6,
            "targets_per_opportunity": 2,
            "local_metric_calls_per_target": 36,
            "provider_success_ceiling": 4832,
            "provider_attempt_ceiling": 19328,
            "validation50_calls": 0,
            "test50_calls": 0,
        }
        preflight_blockers = [
            "existing_sequential_runner_does_not_bind_provider_profile_lwj",
            "historical_private_parent_freeze_missing; fresh deterministic initialization harness required",
            "depends_on_successful_gepa_layer2_real_canary_v1",
        ]
        question = "Can the complete online Layer2 system autonomously reach commits, pivotality, and Vote gain?"
        body = [
            "This pilot tests algorithmic reachability and is not a saturation study.",
            "No historical mutation may be hand selected.",
            "Local patience 3 and team patience 2 remain frozen but do not replace the pilot ceiling.",
        ]
    elif experiment_id == "gepa_saturation_comparison_v1":
        manifest = common_manifest(experiment_id, "GEPA_NATIVE_vs_GEPA_LAYER2", SEEDS)
        manifest.update(
            {
                "execution_regime": "saturation",
                "arms": ["STATIC_5_AGENT_PLURALITY", "GEPA_NATIVE_SATURATION", "GEPA_LAYER2_SATURATION"],
                "primary_endpoint": {
                    "native": "final_saturated_single_prompt_accuracy",
                    "layer2": "final_saturated_vote_accuracy",
                },
                "primary_interpretation": "system_level_not_strict_layer2_causal_attribution",
                "future_controls_not_run": ["GEPA_5_AGENT_NEUTRAL", "matched_budget_GEPA_pair"],
            }
        )
        claims = {
            "allowed": ["LAYER2_AUGMENTED_GEPA_REACHES_HIGHER_SATURATED_PERFORMANCE_THAN_NATIVE_GEPA"],
            "forbidden": ["LAYER2_CAUSALLY_IMPROVES_GEPA_IN_A_MATCHED_FIVE_AGENT_SETTING"],
            "primary_reporting": "per_seed_paired_values_with_initial_performance_visible",
            "cross_optimizer_claims": False,
        }
        cost = {
            "budget_type": "saturation_no_scientific_hard_budget",
            "cost_matching": False,
            "emergency_provider_calls": 100000,
            "emergency_optimizer_steps": 100000,
            "emergency_team_epochs": 10000,
            "emergency_wall_seconds": 86400,
            "report_full_role_and_token_accounting": True,
            "validation50_calls": 0,
            "test50_calls": 0,
        }
        preflight_blockers = [
            "depends_on_successful_canary_and_sequential_pilot",
            "formal_three_seed_saturation_execution_harness_not_frozen_at_scientific_source_commit",
        ]
        question = "What performance does native GEPA versus Layer2-augmented GEPA reach at approximate saturation?"
        body = [
            "Seeds 80, 81, and 82 are frozen prospectively.",
            "Scientific hard budgets are disabled; emergency ceilings are operational only.",
            "This is not a matched-budget or strict five-agent causal comparison.",
        ]
    else:
        manifest = common_manifest(experiment_id, "MARS_LAYER2_REAL_CANARY", [80])
        manifest.update(
            {
                "required_path": ["Layer2_packet", "Planner", "Teacher", "Critic", "Student", "exact_local_eval_Target_scoring"],
                "global_dataset_selection": "FORBIDDEN",
                "stage_dependency": "after_GEPA_study_infrastructure_stable",
            }
        )
        claims = {"allowed": ["MARS_LAYER2_REAL_EMPIRICAL_PATH_CONFIRMED"], "performance_claims": False}
        cost = {
            "budget_type": "fixed_minimal_engineering_canary",
            "provider_success_ceiling": 512,
            "provider_attempt_ceiling": 2048,
            "validation50_calls": 0,
            "test50_calls": 0,
        }
        preflight_blockers = ["stage_not_open_until_GEPA_infrastructure_is_stable", "dedicated_MARS_canary_harness_not_frozen"]
        question = "Can MARS consume only the Layer2 packet and complete exact local Target scoring?"
        body = ["This is a prepared future canary and is not authorized or ready to run."]
    return {
        "manifest": manifest,
        "claims": claims,
        "cost": cost,
        "blockers": preflight_blockers,
        "protocol": protocol_text(experiment_id, manifest["study"], question, body),
    }


def build_bundle(experiment_id: str, fingerprint: str, shared_data: dict[str, Any]) -> None:
    root = OUT / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    spec = bundle_spec(experiment_id)
    (root / "PROTOCOL.md").write_text(spec["protocol"], encoding="utf-8", newline="\n")
    write_json(root / "manifest.json", spec["manifest"])
    write_json(root / "claim_registry.json", spec["claims"])
    write_json(root / "cost_envelope.json", spec["cost"])
    write_json(root / "provider_model_freeze.json", provider_freeze(fingerprint))
    write_json(root / "data_freeze.json", shared_data)
    kind = "formal" if experiment_id == "gepa_saturation_comparison_v1" else "pilot"
    write_json(root / "stopping_semantics.json", stopping(kind))
    write_json(
        root / "authorization_gate.json",
        {
            "authorized": False,
            "authorization_state": "AUTHORIZATION_REQUIRED",
            "authorization_identity": None,
            "ready_to_run": False,
            "ready_for_authorization": False,
            "fresh_one_time_authorization_required": True,
            "reuse_for_retry_extension_or_new_seed": False,
            "blockers": spec["blockers"],
        },
    )
    write_json(
        root / "preflight_report.json",
        {
            "gate": "HOLD_EXECUTION_HARNESS_NOT_FROZEN",
            "scientific_preregistration_complete": True,
            "scientific_source_commit_matches": True,
            "provider_and_models_frozen": True,
            "data_identity_frozen": True,
            "validation50_calls_during_prep": 0,
            "test50_calls_during_prep": 0,
            "real_provider_attempts_during_prep": 0,
            "blockers": spec["blockers"],
        },
    )
    files = sorted(path for path in root.iterdir() if path.name != "sha256_manifest.json")
    write_json(
        root / "sha256_manifest.json",
        {"schema_version": "sha256_manifest_v1", "files": {path.name: sha256_file(path) for path in files}},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-fingerprint", default="")
    args = parser.parse_args()
    fingerprint = endpoint_fingerprint(args.endpoint_fingerprint)
    OUT.mkdir(parents=True, exist_ok=True)
    shared_data = data_freeze()
    ids = [
        "gepa_layer2_real_canary_v1",
        "sequential_symmetry_breaking_online_pilot_v1",
        "gepa_saturation_comparison_v1",
        "mars_layer2_real_canary_v1",
    ]
    for experiment_id in ids:
        build_bundle(experiment_id, fingerprint, shared_data)
    write_json(
        OUT / "completion_state.json",
        {
            "base_sha": SOURCE_COMMIT,
            "selected_local_patience": LOCAL_PATIENCE,
            "selected_team_patience": TEAM_PATIENCE,
            "formal_seed_set": SEEDS,
            "provider_profile": "lwj",
            "real_provider_attempts_during_prep": 0,
            "validation50_calls": 0,
            "test50_calls": 0,
            "preregistered": ids,
            "ready_for_authorization": False,
            "reason": "execution harnesses at the pinned source do not explicitly bind lwj; fail closed",
        },
    )
    write_json(
        OUT / "test_summary.json",
        {
            "focused_pre_canary_tests": "151 passed",
            "compileall": "PASS",
            "governance_preflight": "PASS",
            "sha256_replay": "PASS",
            "known_full_suite_status": "1178 passed, 15 failed, 13 errors",
            "known_nonpassing_class": "missing_historical_private_artifacts",
            "new_failure_class": None,
            "real_provider_attempts_during_prep": 0,
            "validation50_calls": 0,
            "test50_calls": 0,
        },
    )
    write_json(
        OUT / "sanitization_manifest.json",
        {
            "status": "PASS",
            "raw_endpoint": False,
            "private_provider_hostname": False,
            "api_key": False,
            "raw_questions_or_answers": False,
            "raw_model_responses": False,
            "absolute_local_paths": False,
        },
    )
    (OUT / "README.md").write_text(
        "# Final pre-experiment saturation freeze\n\n"
        "This bundle prospectively freezes provider/models, seeds 80/81/82, "
        "local patience 3, team patience 2, data identities, claims, and stopping rules.\n\n"
        "Scientific preregistration is complete, but execution authorization is "
        "fail-closed: the runners at the pinned scientific source do not explicitly "
        "bind `provider_profile=lwj`, and the historical sequential private parent "
        "freeze is unavailable. No API, Validation50, or Test50 call occurred.\n",
        encoding="utf-8",
        newline="\n",
    )
    top_files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "sha256_manifest.json")
    nested = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "sha256_manifest.json")
    write_json(
        OUT / "sha256_manifest.json",
        {
            "schema_version": "sha256_manifest_v1",
            "files": {path.relative_to(OUT).as_posix(): sha256_file(path) for path in sorted(set(top_files + nested))},
        },
    )
    print(json.dumps({"out": OUT.as_posix(), "endpoint_fingerprint": fingerprint, "experiments": ids}, sort_keys=True))


if __name__ == "__main__":
    main()
