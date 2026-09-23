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
import subprocess
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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
        saturation_mode="single_opportunity_engineering_canary",
        source_files=source_files,
    )


def validate_execution(
    *, root: Path, prep: Path, require_authorized: bool,
) -> ValidatedExecutionContext:
    """Replay disk facts, source bytes, dependency, and binding pre-provider."""

    manifest = _read(prep / "manifest.json")
    protocol = _read(prep / "protocol.json")
    if manifest.get("experiment_id") != "gepa_layer2_real_canary_post_refactor_v1":
        raise StartupIdentityError("ABORT_PRE_PROVIDER: unsupported experiment")
    if manifest.get("attempt_id") != manifest.get("experiment_id"):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: attempt identity mismatch")
    if manifest.get("scientific", {}).get("backend") != "gepa" or manifest.get("scientific", {}).get("optimization_scope") != "layer2":
        raise StartupIdentityError("ABORT_PRE_PROVIDER: unsupported mode")
    spec = ExperimentSpec(**_scientific_kwargs(manifest["scientific"]))
    if manifest.get("method_identity") != spec.method_identity or manifest.get("spec_identity") != spec.identity():
        raise StartupIdentityError("ABORT_PRE_PROVIDER: production method identity mismatch")
    runtime = manifest["runtime"]
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
    result = validate_startup_bundle(
        stored=stored, expected=expected, require_authorized=require_authorized,
        phase="canary", roles=("solver", "reflection"),
    )
    if (prep / "authorization_consumed.json").exists():
        raise StartupIdentityError("ABORT_PRE_PROVIDER: authorization already consumed")
    authorization = stored["authorization"]
    if authorization.get("allowed_roles") != ["reflection", "solver"] or authorization.get("allowed_phases") != ["canary"]:
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
        allowed_phase="canary", allowed_roles=("solver", "reflection"),
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


def admit_execution(permit: ValidatedExecutionContext, run_root: Path) -> ValidatedExecutionContext:
    """Consume once, then publish a fresh run-local RUNNING fact atomically."""

    if permit.admitted or run_root.exists():
        raise StartupIdentityError("ABORT_PRE_PROVIDER: run already started")
    consumed = permit.prep_root / "authorization_consumed.json"
    _write_new(consumed, {
        "attempt_id": permit.attempt_id,
        "run_identity_sha256": permit.run_identity_sha256,
        "status": "CONSUMED",
    })
    staging = run_root.with_name(f".{run_root.name}.starting")
    if staging.exists():
        raise StartupIdentityError("ABORT_PRE_PROVIDER: launch staging already exists")
    try:
        staging.mkdir(parents=True)
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
        os.replace(staging, run_root)
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
    temporary = path.with_name(".run_lifecycle.terminal.tmp")
    _write_new(temporary, current)
    os.replace(temporary, path)


def mark_provider_client_constructed(permit: ValidatedExecutionContext) -> None:
    """Record construction, distinct from the first physical provider attempt."""

    if not permit.admitted or permit.run_root is None:
        raise StartupIdentityError("provider construction requires admission")
    path = permit.run_root / "run_lifecycle.json"
    current = _read(path)
    if current.get("status") != "RUNNING" or current.get("provider_client_constructed"):
        raise StartupIdentityError("invalid provider construction boundary")
    current["provider_client_constructed"] = True
    temporary = path.with_name(".run_lifecycle.constructed.tmp")
    _write_new(temporary, current)
    os.replace(temporary, path)
