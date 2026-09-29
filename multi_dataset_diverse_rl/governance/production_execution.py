"""One-time admission for the unified production entry.

Scientific preparation is separate from execution. Identity serialization and
authorization replay are delegated to ``startup_identity``; this module only
binds those checks to source, dependency, provider, and lifecycle boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
from typing import Any, Mapping

from .execution_harness_v2 import (
    INITIALIZATION_POLICY, PROVIDER_PROFILE, ROLE_MODEL, SOLVER_MODEL,
    endpoint_fingerprint_from_environment,
)
from .freeze_hash import source_freeze_sha256
from .startup_identity import (
    StartupIdentityError, build_startup_bundle, read_bundle,
    validate_startup_bundle,
)
from ..experiment import ExperimentSpec, RuntimeContext
from ..local_optimizers.gepa_runtime import verify_frozen_gepa
from ..provider_credentials import LWJ_DASHSCOPE_API_KEY_ENV
from ..persistence.durable_io import (
    atomic_replace, atomic_write_json, ensure_directory, io_path, read_json,
)


LEGACY_FREEZE_ENV_VARS = (
    "V17_FORMAL_SOURCE_FREEZE", "V16_M2F_ONLINE_SOURCE_FREEZE",
)

FORMAL_ATTEMPT2_PREREQUISITE = {
    "attempt_id": "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt3",
    "execution_source_sha": "66e1762ad7ea8d85f71de3b604d0e04fb8b9c7d0",
    "audit_commit": "766c8c6cbcd5dfcbf7b7445f3c41b2cdecad821f",
    "scientific_validity": "VALID",
    "efficacy": "NOT_EVALUABLE",
    "pilot_status": "CLOSED_VALID_INCONCLUSIVE",
    "raw_evidence_freeze_sha256": "a907c155c720f2d218f393e46594bac7261595a54e7953d3628e01c4a47e979b",
}

FORMAL_ATTEMPT3_REPAIR = {
    "superseded_campaign": "Formal V3 attempt2",
    "retry_justification": "INVALID_EXECUTION_CONFORMANCE",
    "incident_commit": "c29bdceca25d63b93edba9eaf57bff33b9ad51c3",
    "failure_class": "POST_SEARCH_JSON_ROUNDTRIP_TYPE_NORMALIZATION",
    "scientific_method_changed": False,
}


def formal_attempt3_incident(root: Path) -> dict[str, Any]:
    """Bind the published aborted-attempt evidence without reading private runs."""

    prefix = "reports/formal_v3_attempt2_seed80_native_abort_20260929/"
    commit = FORMAL_ATTEMPT3_REPAIR["incident_commit"]
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{commit}:{prefix}scientific_validity_audit.json"],
        )
        manifest = json.loads(subprocess.check_output(
            ["git", "-C", str(root), "show", f"{commit}:{prefix}sha256_manifest.json"],
            text=True, encoding="utf-8",
        ))
        audit = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: Formal attempt2 incident unavailable") from exc
    if (manifest.get("scientific_validity_audit.json") != hashlib.sha256(raw).hexdigest()
            or audit.get("attempt_id") != "gepa_saturation_comparison_v3_seed80_native_attempt2"
            or audit.get("execution_source_sha") != "9737626373790aeb55a8ab6b99937b6d3085eace"
            or audit.get("lifecycle") != "ABORTED"
            or audit.get("scientific_validity") != "INVALID_EXECUTION_CONFORMANCE"
            or audit.get("efficacy") != "NOT_ASSESSED"
            or audit.get("failure_boundary") != "POST_SEARCH_EXECUTION_SUMMARY_JSON_READBACK"
            or audit.get("successful_provider_calls") != 1137
            or audit.get("remaining_five_attempts") != "UNAUTHORIZED_UNCONSUMED_UNEXECUTED"
            or audit.get("validation50_calls") != 0
            or audit.get("test50_calls") != 0):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: Formal attempt2 incident mismatch")
    return dict(FORMAL_ATTEMPT3_REPAIR)


def formal_attempt2_prerequisite(root: Path) -> dict[str, Any]:
    closure = _read(root / "experiments/gepa_saturation_comparison_v3/PILOT_CLOSURE.json")
    if (closure.get("schema_version") != "formal_v3_pilot_closure_v1"
            or closure.get("PILOT_STATUS") != "CLOSED_VALID_INCONCLUSIVE"
            or closure.get("FORMAL_PREREQUISITE") != "SATISFIED"
            or closure.get("ADDITIONAL_DIAGNOSTIC_REQUIRED") != "NO"
            or closure.get("diagnostic_prerequisite") != FORMAL_ATTEMPT2_PREREQUISITE
            or closure.get("attempt2_status") != "INVALID"
            or closure.get("sole_valid_diagnostic") != FORMAL_ATTEMPT2_PREREQUISITE["attempt_id"]
            or closure.get("prospective_returned_candidates_observed") != 1
            or closure.get("attempt4_planned_or_required") is not False
            or closure.get("efficacy_may_change_formal_scientific_semantics") is not False
            or closure.get("old_formal_attempt1_status") != "SUPERSEDED_BEFORE_EXECUTION"):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: Formal pilot closure mismatch")
    audit_path = "reports/v4_seed81_attempt3_execution_20260928/scientific_validity_audit.json"
    try:
        frozen_audit = json.loads(subprocess.check_output(
            ["git", "-C", str(root), "show",
             f"{FORMAL_ATTEMPT2_PREREQUISITE['audit_commit']}:{audit_path}"],
            text=True, encoding="utf-8",
        ))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: Formal pilot audit commit unavailable") from exc
    if any(frozen_audit.get(key) != FORMAL_ATTEMPT2_PREREQUISITE[key] for key in (
        "attempt_id", "execution_source_sha", "scientific_validity", "efficacy",
        "raw_evidence_freeze_sha256",
    )) or (frozen_audit.get("audit_gate") != "PASS"
           or frozen_audit.get("returned_candidate_count") != 1
           or frozen_audit.get("lifecycle") != "EXECUTION_COMPLETE"):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: Formal pilot audit evidence mismatch")
    return dict(FORMAL_ATTEMPT2_PREREQUISITE)


def formal_v3_contract() -> dict[str, Any]:
    """The active formal treatment binding, shared by freeze and admission."""

    from ..versions import (
        LAYER2_EVIDENCE_PACKET_V4_VERSION,
        LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
        LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
        LAYER2_RESPONSIBILITY_SOURCE_VERSION,
        LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
        PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
        SATURATION_STOPPING_CONTRACT_VERSION,
        TEAM_EPOCH_SEMANTICS_V4_VERSION,
        TEAM_MINIBATCH_CONTRACT_VERSION,
    )

    return {
        "native_method": "GEPA_NATIVE",
        "treatment_method": "GEPA_LAYER2_V4",
        "layer2_protocol": LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
        "responsibility_source": LAYER2_RESPONSIBILITY_SOURCE_VERSION,
        "scheduler": PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
        "target_feasibility": LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
        "evidence_packet": LAYER2_EVIDENCE_PACKET_V4_VERSION,
        "evidence_selection": LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
        "team_minibatch": TEAM_MINIBATCH_CONTRACT_VERSION,
        "team_epoch": TEAM_EPOCH_SEMANTICS_V4_VERSION,
        "saturation": SATURATION_STOPPING_CONTRACT_VERSION,
        "bounded_search_view": True,
        "local_eval_identity": "same_frozen_team_minibatch12_ids",
        "validation50_calls": 0,
        "test50_calls": 0,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    with open(io_path(path), "x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: frozen artifact is not an object")
    return value


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True, encoding="utf-8",
    ).strip()


@dataclass(frozen=True)
class ValidatedExecutionContext:
    """Created only after canonical replay; permits one provider composition."""

    experiment_id: str
    attempt_id: str
    execution_source_sha: str
    preregistration_sha256: str
    run_identity_sha256: str
    provider_profile: str
    endpoint_fingerprint: str
    allowed_phase: str
    allowed_roles: tuple[str, ...]
    prep_root: Path
    run_root: Path | None = None

    @property
    def admitted(self) -> bool:
        return self.run_root is not None


def _expected_bundle(
    *, root: Path, manifest: Mapping[str, Any], protocol: Mapping[str, Any],
    execution_source_sha: str, prep: Path,
) -> dict[str, Any]:
    source_files = []
    for relative in manifest["execution"]["source_paths"]:
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
            raise StartupIdentityError("ABORT_PRE_PROVIDER: invalid frozen source path")
        source_files.append({
            "path": path.as_posix(), "sha256": source_freeze_sha256(root / path),
        })
    split_root = prep / "splits_private"
    split_hashes = {
        path.name: source_freeze_sha256(path)
        for path in sorted(split_root.glob("*.csv"))
    }
    if set(split_hashes) != {"optimize100.csv", "shadow50.csv"}:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: frozen split inventory mismatch")
    return build_startup_bundle(
        manifest=manifest,
        protocol=protocol,
        experiment_id=str(manifest["experiment_id"]),
        attempt_id=str(manifest["attempt_id"]),
        scientific_method_anchor_sha=str(manifest["execution"]["scientific_method_anchor_sha"]),
        execution_source_sha=execution_source_sha,
        provider_profile=str(manifest["runtime"]["provider_profile"]),
        endpoint_fingerprint=str(manifest["runtime"]["endpoint_fingerprint"]),
        models=manifest["models"],
        data_hashes=split_hashes,
        initialization={"policy": INITIALIZATION_POLICY},
        seeds=[int(manifest["runtime"]["seed"])],
        local_patience=int(manifest["scientific"]["local_no_update_patience"]),
        team_patience=int(manifest["scientific"]["team_no_update_patience"]),
        saturation_mode=(
            "formal_gepa_saturation_v3"
            if str(manifest["experiment_id"]).startswith("gepa_saturation_comparison_v3_")
            else
            "online_local_to_team_transfer_diagnostic_v1"
            if manifest["experiment_id"] in {
                "gepa_layer2_local_to_team_transfer_diagnostic_v1",
                "gepa_layer2_local_to_team_transfer_diagnostic_v2",
                "gepa_layer2_local_to_team_transfer_diagnostic_v3",
                "gepa_layer2_local_to_team_transfer_diagnostic_v4",
            }
            else "single_opportunity_engineering_canary"
        ),
        source_files=source_files,
    )


def validate_execution(
    *, root: Path, prep: Path, require_authorized: bool,
) -> ValidatedExecutionContext:
    """Replay disk facts, source bytes, dependency, and binding pre-provider."""

    if any(os.environ.get(name, "").strip() for name in LEGACY_FREEZE_ENV_VARS):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: legacy source-freeze environment")

    manifest = _read(prep / "manifest.json")
    protocol = _read(prep / "protocol.json")
    formal_match = re.fullmatch(
        r"gepa_saturation_comparison_v3_seed(80|81|82)_(native|layer2)_attempt(1|2|3)",
        str(manifest.get("experiment_id", "")),
    )
    formal = formal_match is not None
    formal_attempt2 = formal and formal_match.group(3) == "2"
    formal_attempt3 = formal and formal_match.group(3) == "3"
    if not formal and manifest.get("experiment_id") not in {
        "gepa_layer2_real_canary_post_refactor_v1",
        "gepa_layer2_real_canary_post_refactor_v2",
        "gepa_layer2_local_to_team_transfer_diagnostic_v1",
        "gepa_layer2_local_to_team_transfer_diagnostic_v2",
        "gepa_layer2_local_to_team_transfer_diagnostic_v3",
        "gepa_layer2_local_to_team_transfer_diagnostic_v4",
    }:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: unsupported experiment")
    diagnostic = manifest["experiment_id"] in {
        "gepa_layer2_local_to_team_transfer_diagnostic_v1",
        "gepa_layer2_local_to_team_transfer_diagnostic_v2",
        "gepa_layer2_local_to_team_transfer_diagnostic_v3",
        "gepa_layer2_local_to_team_transfer_diagnostic_v4",
    }
    if manifest["experiment_id"] == "gepa_layer2_local_to_team_transfer_diagnostic_v3":
        from ..versions import (
            LAYER2_TEAM_SEARCH_PROTOCOL_VERSION,
            TEAM_MINIBATCH_CONTRACT_VERSION,
            PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
            LAYER2_RESPONSIBILITY_SOURCE_VERSION,
        )
        semantic = {
            "layer2_protocol": LAYER2_TEAM_SEARCH_PROTOCOL_VERSION,
            "team_minibatch": TEAM_MINIBATCH_CONTRACT_VERSION,
            "scheduler": PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
            "responsibility_source": LAYER2_RESPONSIBILITY_SOURCE_VERSION,
            "local_eval_identity": "same_frozen_team_minibatch12_ids",
            "local_acceptance": "target_member_only_no_team_feedback",
        }
        if (manifest.get("semantic_contract") != semantic
                or protocol.get("semantic_contract") != semantic
                or manifest.get("schema_version") != "online_transfer_diagnostic_freeze_v3"
                or protocol.get("schema_version") != "online_local_to_team_transfer_diagnostic_v3"):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: v3 semantic contract mismatch")
    if manifest["experiment_id"] == "gepa_layer2_local_to_team_transfer_diagnostic_v4":
        from ..versions import (
            LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
            TEAM_MINIBATCH_CONTRACT_VERSION,
            PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
            LAYER2_RESPONSIBILITY_SOURCE_VERSION,
            LAYER2_EVIDENCE_PACKET_V4_VERSION,
            LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
            LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
        )
        semantic = {
            "layer2_protocol": LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
            "responsibility_source": LAYER2_RESPONSIBILITY_SOURCE_VERSION,
            "scheduler": PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
            "feasibility_policy": LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
            "packet_version": LAYER2_EVIDENCE_PACKET_V4_VERSION,
            "evidence_selection_policy": LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
            "team_minibatch": TEAM_MINIBATCH_CONTRACT_VERSION,
            "local_eval_identity": "same_frozen_team_minibatch12_ids",
            "local_acceptance": "target_member_only_no_team_feedback",
            "responsibility_value": "raw_V_not_discounted_target_score",
            "no_feasible_stop": "NO_FEASIBLE_LAYER2_OPPORTUNITY",
        }
        if (manifest.get("semantic_contract") != semantic
                or protocol.get("semantic_contract") != semantic
                or manifest.get("schema_version") != "online_transfer_diagnostic_freeze_v4"
                or protocol.get("schema_version") != "online_local_to_team_transfer_diagnostic_v4"
                or protocol.get("packet_capacity") != {
                    "local_metric_budget": 36, "batch_size": 3,
                    "nominal_role_item_slots": 36,
                    "focus_anchor": "exact_latest_transition_never_truncated",
                    "repair": "first_min_Rfull_capacity_existing_deterministic_order",
                    "delivery": "separately_observed_not_implied_by_schedule",
                }):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: v4 semantic contract mismatch")
    v4_attempt2 = (
        manifest.get("experiment_id") == "gepa_layer2_local_to_team_transfer_diagnostic_v4"
        and manifest.get("attempt_id")
        == "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2"
    )
    v4_attempt3 = (
        manifest.get("experiment_id") == "gepa_layer2_local_to_team_transfer_diagnostic_v4"
        and manifest.get("attempt_id")
        == "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt3"
    )
    if v4_attempt2:
        from .v4_source_closure import active_v4_source_paths

        if manifest.get("execution_refresh") != {
            "scientific_method_changed": False,
            "execution_implementation_refreshed": True,
            "supersedes_for_execution": "gepa_layer2_local_to_team_transfer_diagnostic_v4",
            "reason": "current_shared_v4_selector_and_complete_source_identity_closure",
        } or manifest.get("execution", {}).get("source_paths") != list(
            active_v4_source_paths(root)
        ):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: V4 source closure mismatch")
    elif v4_attempt3:
        from .v4_source_closure import active_v4_source_paths

        if manifest.get("execution_refresh") != {
            "scientific_method_changed": False,
            "execution_implementation_refreshed": True,
            "supersedes_for_execution": "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2",
            "reason": "diagnostic_returned_candidate_sampling_integrity_repair",
        } or manifest.get("execution", {}).get("scientific_method_anchor_sha") != (
            "85812a7d891e6a2c3bfdca00a1cb4d14074735a4"
        ) or manifest.get("diagnostic_protocol_repair") != {
            "prior_attempt": "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2",
            "prior_validity": "INVALID_NOT_EVALUABLE_DIAGNOSTIC_SAMPLING_INTEGRITY",
            "scientific_method_changed": False,
            "sampled_boundary": "Layer1_to_Layer2_returned_candidates",
            "prior_provider_calls_excluded": True,
        } or manifest.get("execution", {}).get("source_paths") != list(
            active_v4_source_paths(root, attempt3=True)
        ):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: V4 attempt3 source closure mismatch")
    elif manifest.get("attempt_id") != manifest.get("experiment_id"):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: attempt identity mismatch")
    if manifest.get("scientific", {}).get("backend") != "gepa" or (
        manifest.get("scientific", {}).get("optimization_scope") not in
        ({"native", "layer2"} if formal else {"layer2"})
    ):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: unsupported mode")
    spec = ExperimentSpec(**_scientific_kwargs(manifest["scientific"]))
    if formal:
        match = formal_match
        assert match is not None
        expected_scope = match.group(2)
        if (
            manifest.get("schema_version") != "formal_gepa_saturation_freeze_v3"
            or protocol.get("schema_version") != "formal_gepa_saturation_protocol_v3"
            or manifest["scientific"].get("stopping_regime") != "saturation"
            or spec.optimization_scope.value != expected_scope
            or spec.local_no_update_patience != 3
            or spec.team_no_update_patience != 2
            or (expected_scope == "layer2" and not spec.is_v4_layer2)
            or (expected_scope == "native" and spec.layer2_protocol_version is not None)
            or protocol.get("seed") != int(match.group(1))
            or manifest.get("runtime", {}).get("seed") != int(match.group(1))
            or protocol.get("arm") != ("GEPA_NATIVE" if expected_scope == "native" else "GEPA_LAYER2_V4")
            or protocol.get("optimize_rows") != 100
            or protocol.get("shadow_rows") != 50
            or protocol.get("scientific_budget") != "none_saturation_only"
            or protocol.get("local_no_update_patience") != 3
            or protocol.get("team_no_update_patience") != 2
            or protocol.get("emergency_ceiling") != {
                key: manifest["scientific"][key] for key in (
                    "emergency_max_provider_calls", "emergency_max_optimizer_steps",
                    "emergency_max_team_epochs", "emergency_max_wall_seconds",
                )
            }
            or protocol.get("data_identity") != manifest["scientific"].get("data_identity")
            or protocol.get("initialization_policy") != INITIALIZATION_POLICY
            or protocol.get("validation50_calls") != 0
            or protocol.get("test50_calls") != 0
            or manifest.get("formal_v3_contract") != formal_v3_contract()
            or protocol.get("formal_v3_contract") != formal_v3_contract()
        ):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: formal V3 contract mismatch")
        if formal_attempt2 or formal_attempt3:
            from ..formal_validation_policy import formal_validation50_policy

            expected_prerequisite = formal_attempt2_prerequisite(root)
            try:
                expected_validation_policy = formal_validation50_policy(root)
            except (ValueError, KeyError, TypeError) as exc:
                raise StartupIdentityError(
                    "ABORT_PRE_PROVIDER: Formal Validation50 split integrity failure"
                ) from exc
            expected_refreeze = {
                "superseded_execution_source_sha": "504dadb4024a0c69cc2f8aac6e6d400f77f3a680",
                "reason": "held_out_validation50_was_mislabeled_shadow50_fold_c",
                "formal_real_calls_observed": 0,
                "scientific_search_changed": False,
            }
            required_source_paths = {
                path.relative_to(root).as_posix()
                for directory in ("multi_dataset_diverse_rl", "infrastructure/common_solver_contract_v1")
                for path in (root / directory).rglob("*.py")
            } | {
                "scripts/run_experiment.py",
                "scripts/prepare_gepa_saturation_comparison_v3.py",
                "scripts/audit_formal_v3_offline_freeze.py",
                "scripts/freeze_formal_v3_execution.py",
                "scripts/derive_formal_trajectory_trace.py",
                "experiments/gepa_saturation_comparison_v3/PROTOCOL.md",
                "experiments/gepa_saturation_comparison_v3/PILOT_CLOSURE.json",
                "experiments/gepa_saturation_comparison_v3/POST_FREEZE_VALIDATION50_EVALUATION.md",
                "experiments/anti_overfitting_split_v1/split_manifest.json",
            }
            if (manifest.get("execution_gate") != {"real_v4_diagnostic": "SCIENTIFICALLY_VALID"}
                    or manifest.get("diagnostic_prerequisite") != expected_prerequisite
                    or protocol.get("diagnostic_prerequisite") != expected_prerequisite
                    or manifest.get("post_freeze_validation50") != expected_validation_policy
                    or protocol.get("post_freeze_validation50") != expected_validation_policy
                    or manifest.get("preexecution_refreeze") != expected_refreeze
                    or protocol.get("preexecution_refreeze") != expected_refreeze
                    or protocol.get("post_search_final_team") != {
                        "native": "replicate_first_returned_native_candidate_to_all_five_else_initial_team",
                        "layer2": "preserve_final_committed_five_member_team",
                        "validation50": "POST_FREEZE_VALIDATION50_EVALUATION_separate_authorization",
                        "test50": "SEALED_ZERO_CALLS_STAGE0",
                    }
                    or not required_source_paths <= set(
                        manifest.get("execution", {}).get("source_paths", ())
                    )):
                raise StartupIdentityError("ABORT_PRE_PROVIDER: formal V3 diagnostic prerequisite mismatch")
            if formal_attempt3:
                incident = formal_attempt3_incident(root)
                if (manifest.get("execution_repair") != incident
                        or protocol.get("execution_repair") != incident
                        or manifest.get("execution", {}).get("scientific_method_anchor_sha")
                        != "9737626373790aeb55a8ab6b99937b6d3085eace"):
                    raise StartupIdentityError("ABORT_PRE_PROVIDER: formal V3 attempt3 repair identity mismatch")
            elif manifest.get("execution_repair") is not None or protocol.get("execution_repair") is not None:
                raise StartupIdentityError("ABORT_PRE_PROVIDER: historical formal attempt2 repair identity mismatch")
        elif (manifest.get("execution_gate") != {"real_v4_diagnostic": "PENDING_SCIENTIFIC_VALIDITY"}
              or manifest.get("diagnostic_prerequisite") is not None):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: historical formal attempt1 identity mismatch")
        if require_authorized and formal_attempt2:
            raise StartupIdentityError("ABORT_PRE_PROVIDER: formal attempt2 campaign closed before further execution")
        if require_authorized and not (formal_attempt2 or formal_attempt3):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: formal attempt1 superseded before execution")
    if diagnostic:
        from .v4_attempt3_contract import diagnostic_contract, resource_upper_bounds
        expected_diagnostic_contract = (
            diagnostic_contract() if v4_attempt3 else {
                "accepted_mutation_target": 5,
                "max_opportunities": 10,
                "reflection_proposal_ceiling": 20,
                "successful_provider_ceiling": 1200,
                "transport_attempt_ceiling": 4800,
                "mandatory_full": True,
                "diagnostic_full_is_admission_inert": True,
            }
        )
        if (
            manifest["scientific"].get("stopping_regime") != "fixed_budget"
            or manifest["scientific"].get("fixed_budget_units") != 10
            or manifest["scientific"].get("data_identity") != "anti_overfitting_split_v1_fold_a+b_to_c"
            or manifest.get("diagnostic_contract") != expected_diagnostic_contract
        ):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: diagnostic budget/policy mismatch")
    if manifest.get("method_identity") != spec.method_identity or manifest.get("spec_identity") != spec.identity():
        raise StartupIdentityError("ABORT_PRE_PROVIDER: production method identity mismatch")
    runtime = manifest["runtime"]
    if diagnostic and (
        runtime.get("seed") != 81
        or protocol.get("seed") != 81
        or (protocol.get("returned_candidate_target") if v4_attempt3
            else protocol.get("accepted_mutation_target")) != 5
        or protocol.get("reflection_proposal_ceiling") != 20
        or protocol.get("successful_provider_ceiling") != (7000 if v4_attempt3 else 1200)
        or protocol.get("transport_attempt_ceiling") != (150000 if v4_attempt3 else 4800)
        or (v4_attempt3 and any(
            protocol.get(key) != value for key, value in expected_diagnostic_contract.items()
        ))
        or (v4_attempt3 and protocol.get("resource_upper_bounds") != resource_upper_bounds())
        or (v4_attempt3 and protocol.get("within_opportunity_target_crossing")
            != "retain_all_returned_candidates_then_stop")
        or (v4_attempt3 and protocol.get("diagnostic_protocol_repair_only") is not True)
        or (v4_attempt3 and protocol.get("scientific_method_change") is not False)
        or (v4_attempt3 and protocol.get("diagnostic_full_policy")
            != "mandatory_all_returned_strict_positive_admission_inert")
        or (v4_attempt3 and protocol.get("full_rows_per_returned_candidate") != 100)
        or (v4_attempt3 and protocol.get("proposal_ceiling_guard")
            != "stop_before_next_opportunity_if_12_proposals_could_overshoot")
        or protocol.get("diagnostic_full_is_admission_inert") is not True
        or protocol.get("validation50_calls") != 0
        or protocol.get("test50_calls") != 0
    ):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: diagnostic protocol mismatch")
    if runtime.get("provider_profile") != PROVIDER_PROFILE:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: provider profile mismatch")
    if runtime.get("solver_model") != SOLVER_MODEL or runtime.get("optimizer_model") != ROLE_MODEL or runtime.get("evaluator_model") != ROLE_MODEL:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: model binding mismatch")
    if manifest.get("models") != {
        "solver": {"model": SOLVER_MODEL, "thinking": False},
        "reflection": {"model": ROLE_MODEL},
    }:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: model-role contract mismatch")
    if runtime.get("endpoint_fingerprint") != endpoint_fingerprint_from_environment():
        raise StartupIdentityError("ABORT_PRE_PROVIDER: endpoint fingerprint mismatch")
    if require_authorized and not os.environ.get(LWJ_DASHSCOPE_API_KEY_ENV):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: lwj credential unavailable")
    if manifest.get("access") != {"validation50_calls": 0, "test50_calls": 0}:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: split access mismatch")
    if manifest.get("execution", {}).get("initialization_policy") != INITIALIZATION_POLICY:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: initialization policy mismatch")
    dependency = verify_frozen_gepa()
    if manifest.get("dependency") != {"gepa": dependency}:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: GEPA dependency mismatch")
    source = str(manifest["execution"]["execution_source_sha"])
    if _git_head(root) != source:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: execution source mismatch")
    expected = _expected_bundle(
        root=root, manifest=manifest, protocol=protocol,
        execution_source_sha=source, prep=prep,
    )
    stored = read_bundle(prep / "startup_identity")
    allowed_phase = "formal" if formal else "diagnostic" if diagnostic else "canary"
    result = validate_startup_bundle(
        stored=stored, expected=expected, require_authorized=require_authorized,
        phase=allowed_phase, roles=("solver", "reflection"),
    )
    if (prep / "authorization_consumed.json").exists():
        raise StartupIdentityError("ABORT_PRE_PROVIDER: authorization already consumed")
    authorization = stored["authorization"]
    if authorization.get("allowed_roles") != ["reflection", "solver"] or authorization.get("allowed_phases") != [allowed_phase]:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: authorization scope mismatch")
    if require_authorized and authorization.get("authorization_scope") != manifest["attempt_id"]:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: stale authorization scope")
    return ValidatedExecutionContext(
        experiment_id=str(manifest["experiment_id"]),
        attempt_id=str(manifest["attempt_id"]),
        execution_source_sha=source,
        preregistration_sha256=result["preregistration_sha256"],
        run_identity_sha256=result["run_identity_sha256"],
        provider_profile=PROVIDER_PROFILE,
        endpoint_fingerprint=str(runtime["endpoint_fingerprint"]),
        allowed_phase=allowed_phase, allowed_roles=("solver", "reflection"),
        prep_root=prep,
    )


def _scientific_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ..experiment import OptimizerBackend, OptimizationScope, StoppingRegime

    return {
        "backend": OptimizerBackend(str(payload["backend"])),
        "optimization_scope": OptimizationScope(str(payload["optimization_scope"])),
        "stopping_regime": StoppingRegime(str(payload["stopping_regime"])),
        "task_identity": str(payload["task_identity"]),
        "data_identity": str(payload["data_identity"]),
        "fixed_budget_units": int(payload["fixed_budget_units"]),
        "local_no_update_patience": int(payload["local_no_update_patience"]),
        "team_no_update_patience": int(payload["team_no_update_patience"]),
        "layer2_protocol_version": payload.get("layer2_protocol_version"),
        "emergency_max_provider_calls": int(payload.get("emergency_max_provider_calls", 100_000)),
        "emergency_max_optimizer_steps": int(payload.get("emergency_max_optimizer_steps", 100_000)),
        "emergency_max_team_epochs": int(payload.get("emergency_max_team_epochs", 10_000)),
        "emergency_max_wall_seconds": (
            int(payload["emergency_max_wall_seconds"])
            if payload.get("emergency_max_wall_seconds") is not None else 86_400
        ),
    }


def validate_local_readiness(run_root: Path) -> None:
    """Rehearse local writes before consuming the one-time authorization."""

    root = run_root.resolve()
    staging = root.with_name(f".{root.name}.starting")
    if os.path.exists(io_path(root)) or os.path.exists(io_path(staging)):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: stale run or staging root")
    ensure_directory(root.parent)
    if any(root.parent.glob(f".{root.name}.*.tmp")):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: stale finalization temp")
    rehearsal = root.with_name(f".{root.name}.readiness-{uuid.uuid4().hex[:8]}")
    if rehearsal.resolve().parent != root.parent:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: invalid rehearsal target")
    from ..persistence.artifacts import ArtifactWriter
    from ..local_optimizers.gepa_callbacks import GEPALineageCallback
    from ..team_search.execution_runtime import CappedDurableLedger, ledger_summary

    try:
        ensure_directory(rehearsal)
        writer = ArtifactWriter(rehearsal / "system")
        writer.write_json("team_full_categorical_profiles/" + "a" * 64 + ".json", {"ok": True})
        writer.write_jsonl("nested/profiles.jsonl", [{"ok": True}])
        writer.append_jsonl("nested/profiles.jsonl", [{"also": True}])
        writer.write_csv("nested/profiles.csv", [{"ok": 1}], ["ok"])
        atomic_write_json(rehearsal / "execution_summary.json", {"ok": True})
        atomic_write_json(rehearsal / "run_lifecycle.json", {"status": "RUNNING"})
        ledger = CappedDurableLedger(
            rehearsal / "ledger.jsonl", successful_ceiling=2, attempt_ceiling=2,
        )
        ledger.reserve_provider_attempt("solver")
        ledger.append({
            "record_id": "readiness", "phase": "initialization", "logical_role": "solver",
            "client_role": "solver", "provider_attempts": 1,
            "successful_provider_calls": 1, "cache_hit": False,
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
            "seed": 0, "arm": "readiness", "update_index": -1, "target_member": -1,
        })
        lineage = GEPALineageCallback(rehearsal / "local_gepa" / "g-test.lineage.jsonl")
        lineage.on_optimization_start({"trainset_size": 1, "valset_size": 1,
                                       "seed_candidate": {"decision_procedure": "test"}})
        if (read_json(rehearsal / "execution_summary.json") != {"ok": True}
                or ledger_summary(rehearsal / "ledger.jsonl")["successful_provider_calls"] != 1):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: readiness read-back mismatch")
        moved = rehearsal.with_name(rehearsal.name + ".renamed")
        atomic_replace(rehearsal, moved)
        rehearsal = moved
    finally:
        if os.path.exists(io_path(rehearsal)):
            shutil.rmtree(io_path(rehearsal))


def admit_execution(permit: ValidatedExecutionContext, run_root: Path) -> ValidatedExecutionContext:
    """Consume once, then publish a fresh run-local RUNNING fact atomically."""

    if permit.admitted:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: run already started")
    validate_local_readiness(run_root)
    staging = run_root.with_name(f".{run_root.name}.starting")
    if os.path.exists(io_path(run_root)) or os.path.exists(io_path(staging)):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: stale run or staging root")
    consumed = permit.prep_root / "authorization_consumed.json"
    _write_new(consumed, {
        "attempt_id": permit.attempt_id,
        "run_identity_sha256": permit.run_identity_sha256,
        "status": "CONSUMED",
    })
    try:
        ensure_directory(staging)
        _write_new(staging / "run_lifecycle.json", {
            "attempt_id": permit.attempt_id,
            "status": "RUNNING",
            "run_identity_sha256": permit.run_identity_sha256,
            "provider_boundary_reached": False,
            "provider_client_constructed": False,
            "provider_attempts": 0,
            "provider_successes": 0,
            "provider_failures": 0,
            "events": [{"status": "RUNNING", "timestamp": _utc_now()}],
        })
        atomic_replace(staging, run_root)
    except BaseException:
        # The consumed marker is intentionally never rolled back. A failed
        # launch cannot silently reuse the same one-time authorization.
        raise
    from dataclasses import replace
    return replace(permit, run_root=run_root)


def terminal_lifecycle(permit: ValidatedExecutionContext, *, status: str, provider_attempts: int = 0, provider_successes: int = 0, provider_failures: int = 0) -> None:
    if not permit.admitted or status not in {"EXECUTION_COMPLETE", "FAILED_START", "ABORTED"}:
        raise StartupIdentityError("invalid terminal lifecycle transition")
    path = permit.run_root / "run_lifecycle.json"  # type: ignore[operator]
    current = _read(path)
    if current.get("status") != "RUNNING":
        raise StartupIdentityError("run lifecycle is not RUNNING")
    current.update({
        "status": status,
        "provider_boundary_reached": provider_attempts > 0 or current.get("provider_boundary_reached", False),
        "provider_attempts": provider_attempts,
        "provider_successes": provider_successes,
        "provider_failures": provider_failures,
        "events": [*current["events"], {"status": status, "timestamp": _utc_now()}],
    })
    atomic_write_json(path, current)
    if _read(path).get("status") != status:
        raise StartupIdentityError("terminal lifecycle read-back mismatch")


def mark_provider_client_constructed(permit: ValidatedExecutionContext) -> None:
    """Record construction, distinct from the first physical provider attempt."""

    if not permit.admitted or permit.run_root is None:
        raise StartupIdentityError("provider construction requires admission")
    path = permit.run_root / "run_lifecycle.json"
    current = _read(path)
    if current.get("status") != "RUNNING" or current.get("provider_client_constructed"):
        raise StartupIdentityError("invalid provider construction boundary")
    current["provider_client_constructed"] = True
    atomic_write_json(path, current)
    if not _read(path).get("provider_client_constructed"):
        raise StartupIdentityError("provider construction read-back mismatch")
