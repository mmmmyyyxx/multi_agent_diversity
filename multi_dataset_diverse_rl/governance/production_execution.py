"""One-time admission for the unified production entry.

Scientific preparation is separate from execution. Identity serialization and
authorization replay are delegated to ``startup_identity``; this module only
binds those checks to source, dependency, provider, and lifecycle boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
            "online_local_to_team_transfer_diagnostic_v1"
            if manifest["experiment_id"] in {
                "gepa_layer2_local_to_team_transfer_diagnostic_v1",
                "gepa_layer2_local_to_team_transfer_diagnostic_v2",
                "gepa_layer2_local_to_team_transfer_diagnostic_v3",
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
    if manifest.get("experiment_id") not in {
        "gepa_layer2_real_canary_post_refactor_v1",
        "gepa_layer2_real_canary_post_refactor_v2",
        "gepa_layer2_local_to_team_transfer_diagnostic_v1",
        "gepa_layer2_local_to_team_transfer_diagnostic_v2",
        "gepa_layer2_local_to_team_transfer_diagnostic_v3",
    }:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: unsupported experiment")
    diagnostic = manifest["experiment_id"] in {
        "gepa_layer2_local_to_team_transfer_diagnostic_v1",
        "gepa_layer2_local_to_team_transfer_diagnostic_v2",
        "gepa_layer2_local_to_team_transfer_diagnostic_v3",
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
    if manifest.get("attempt_id") != manifest.get("experiment_id"):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: attempt identity mismatch")
    if manifest.get("scientific", {}).get("backend") != "gepa" or manifest.get("scientific", {}).get("optimization_scope") != "layer2":
        raise StartupIdentityError("ABORT_PRE_PROVIDER: unsupported mode")
    spec = ExperimentSpec(**_scientific_kwargs(manifest["scientific"]))
    if diagnostic:
        if (
            manifest["scientific"].get("stopping_regime") != "fixed_budget"
            or manifest["scientific"].get("fixed_budget_units") != 10
            or manifest["scientific"].get("data_identity") != "anti_overfitting_split_v1_fold_a+b_to_c"
            or manifest.get("diagnostic_contract") != {
                "accepted_mutation_target": 5,
                "max_opportunities": 10,
                "reflection_proposal_ceiling": 20,
                "successful_provider_ceiling": 1200,
                "transport_attempt_ceiling": 4800,
                "mandatory_full": True,
                "diagnostic_full_is_admission_inert": True,
            }
        ):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: diagnostic budget/policy mismatch")
    if manifest.get("method_identity") != spec.method_identity or manifest.get("spec_identity") != spec.identity():
        raise StartupIdentityError("ABORT_PRE_PROVIDER: production method identity mismatch")
    runtime = manifest["runtime"]
    if diagnostic and (
        runtime.get("seed") != 81
        or protocol.get("seed") != 81
        or protocol.get("accepted_mutation_target") != 5
        or protocol.get("reflection_proposal_ceiling") != 20
        or protocol.get("successful_provider_ceiling") != 1200
        or protocol.get("transport_attempt_ceiling") != 4800
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
    allowed_phase = "diagnostic" if diagnostic else "canary"
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
