"""Canonical, zero-network startup identity and authorization contract.

This module is the only implementation allowed to construct or validate the
startup identity used by the 2026-09-22 execution-harness family.  Scientific
facts and attempt-local authorization are deliberately separate: changing an
authorization/lifecycle fact cannot change the preregistered scientific hash.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CANONICAL_SERIALIZATION = "canonical_json_utf8_sorted_compact_v1"
STARTUP_IDENTITY_SCHEMA = "startup_scientific_identity_v1"
RUN_IDENTITY_SCHEMA = "startup_run_identity_v1"
AUTHORIZATION_SCHEMA = "startup_authorization_binding_v1"

_OPERATIONAL_MANIFEST_KEYS = frozenset(
    {"status", "lifecycle_history", "result", "result_commit"}
)
_OPERATIONAL_AUTHORIZATION_KEYS = frozenset(
    {"authorized", "authorization_scope", "authorization_state", "consumed"}
)
_OPERATIONAL_GIT_KEYS = frozenset(
    {"execution_source_sha", "implementation_commit", "result_commit"}
)


class StartupIdentityError(RuntimeError):
    """Raised before provider construction when startup identity is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON without platform- or pretty-print dependence."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalized_path(value: str) -> str:
    return value.replace("\\", "/")


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip through canonical JSON to remove mapping implementation state."""

    return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))


def scientific_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable design facts, excluding lifecycle/authorization state.

    Allowed roles and phases are part of the scientific execution policy.
    Whether a user has authorized one attempt is operational metadata and is
    intentionally excluded.
    """

    payload = deepcopy(dict(manifest))
    for key in _OPERATIONAL_MANIFEST_KEYS:
        payload.pop(key, None)
    artifacts = payload.pop("artifacts", None)
    del artifacts  # hash fields and output paths are metadata, never self-inputs
    authorization = payload.get("api_authorization")
    if isinstance(authorization, dict):
        payload["api_authorization"] = {
            key: value
            for key, value in authorization.items()
            if key not in _OPERATIONAL_AUTHORIZATION_KEYS
        }
    git = payload.get("git")
    if isinstance(git, dict):
        payload["git"] = {
            key: value for key, value in git.items() if key not in _OPERATIONAL_GIT_KEYS
        }
    execution_freeze = payload.get("execution_freeze")
    if isinstance(execution_freeze, dict):
        execution_freeze = dict(execution_freeze)
        execution_freeze.pop("execution_source_sha", None)
        payload["execution_freeze"] = execution_freeze
    return _canonical_mapping(payload)


def build_startup_bundle(
    *,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    experiment_id: str,
    attempt_id: str,
    scientific_method_anchor_sha: str,
    execution_source_sha: str,
    provider_profile: str,
    endpoint_fingerprint: str,
    models: Mapping[str, Any],
    data_hashes: Mapping[str, str],
    initialization: Mapping[str, Any],
    seeds: Sequence[int],
    local_patience: int,
    team_patience: int,
    saturation_mode: str,
    source_files: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Construct scientific, run, and authorization identities once."""

    if provider_profile != "lwj":
        raise StartupIdentityError("ABORT_PRE_PROVIDER: provider_profile must be lwj")
    if not endpoint_fingerprint:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: endpoint fingerprint missing")
    normalized_sources = sorted(
        (
            {
                "path": _normalized_path(str(row["path"])),
                "sha256": str(row["sha256"]),
            }
            for row in source_files
        ),
        key=lambda row: row["path"],
    )
    scientific_payload = {
        "schema_version": STARTUP_IDENTITY_SCHEMA,
        "experiment_id": experiment_id,
        "scientific_method_anchor_sha": scientific_method_anchor_sha,
        "execution_source_sha": execution_source_sha,
        "protocol_sha256": canonical_sha256(protocol),
        "manifest_sha256": canonical_sha256(scientific_manifest_payload(manifest)),
        "provider": {
            "profile": provider_profile,
            "endpoint_fingerprint": endpoint_fingerprint,
        },
        "models": _canonical_mapping(models),
        "data_hashes": dict(sorted((str(k), str(v)) for k, v in data_hashes.items())),
        "initialization": _canonical_mapping(initialization),
        "seeds": [int(seed) for seed in seeds],
        "stopping": {
            "local_no_update_patience": int(local_patience),
            "team_no_update_patience": int(team_patience),
            "saturation_mode": saturation_mode,
        },
        "source_files": normalized_sources,
        "serialization": CANONICAL_SERIALIZATION,
    }
    preregistration_sha256 = canonical_sha256(scientific_payload)
    scientific_identity = {
        "payload": scientific_payload,
        "preregistration_sha256": preregistration_sha256,
    }
    run_payload = {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "experiment_id": experiment_id,
        "attempt_id": attempt_id,
        "preregistration_sha256": preregistration_sha256,
        "execution_source_sha": execution_source_sha,
    }
    run_identity = {
        "payload": run_payload,
        "run_identity_sha256": canonical_sha256(run_payload),
    }
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "experiment_id": experiment_id,
        "attempt_id": attempt_id,
        "run_identity_sha256": run_identity["run_identity_sha256"],
        "authorized": False,
        "authorization_state": "AUTHORIZATION_REQUIRED",
        "authorization_scope": None,
        "allowed_roles": sorted(
            str(role)
            for role in manifest.get("api_authorization", {}).get("allowed_roles", [])
        ),
        "allowed_phases": sorted(
            str(phase)
            for phase in manifest.get("api_authorization", {}).get("allowed_phases", [])
        ),
    }
    return {
        "scientific_identity": scientific_identity,
        "run_identity": run_identity,
        "authorization": authorization,
    }


def authorized_artifact(
    bundle: Mapping[str, Any], *, scope: str, explicit_user_authorized: bool
) -> dict[str, Any]:
    """Create an attempt-local grant without changing scientific identity."""

    if not explicit_user_authorized:
        raise StartupIdentityError("explicit user API authorization is required")
    authorization = deepcopy(dict(bundle["authorization"]))
    authorization.update(
        {
            "authorized": True,
            "authorization_state": "AUTHORIZED",
            "authorization_scope": scope,
        }
    )
    return authorization


def validate_startup_bundle(
    *,
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
    require_authorized: bool,
    phase: str | None = None,
    roles: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate disk-round-tripped startup facts before provider construction."""

    for key in ("scientific_identity", "run_identity"):
        if canonical_json_bytes(stored.get(key)) != canonical_json_bytes(expected.get(key)):
            raise StartupIdentityError(f"ABORT_PRE_PROVIDER: {key} mismatch")
    scientific = stored["scientific_identity"]
    if scientific.get("preregistration_sha256") != canonical_sha256(
        scientific.get("payload")
    ):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: preregistration hash mismatch")
    run_identity = stored["run_identity"]
    if run_identity.get("run_identity_sha256") != canonical_sha256(
        run_identity.get("payload")
    ):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: run identity hash mismatch")
    if run_identity["payload"].get("preregistration_sha256") != scientific.get(
        "preregistration_sha256"
    ):
        raise StartupIdentityError("ABORT_PRE_PROVIDER: run/preregistration mismatch")
    authorization = stored.get("authorization", {})
    expected_authorization = expected.get("authorization", {})
    for key in (
        "schema_version",
        "experiment_id",
        "attempt_id",
        "run_identity_sha256",
        "allowed_roles",
        "allowed_phases",
    ):
        if authorization.get(key) != expected_authorization.get(key):
            raise StartupIdentityError(f"ABORT_PRE_PROVIDER: authorization {key} mismatch")
    if require_authorized:
        if authorization.get("authorized") is not True:
            raise StartupIdentityError("ABORT_PRE_PROVIDER: authorization required")
        if authorization.get("authorization_state") != "AUTHORIZED":
            raise StartupIdentityError("ABORT_PRE_PROVIDER: authorization state mismatch")
        if not authorization.get("authorization_scope"):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: authorization scope missing")
        if phase not in authorization.get("allowed_phases", []):
            raise StartupIdentityError("ABORT_PRE_PROVIDER: phase is not authorized")
        missing = set(roles) - set(authorization.get("allowed_roles", []))
        if missing:
            raise StartupIdentityError(
                "ABORT_PRE_PROVIDER: roles are not authorized: " + ",".join(sorted(missing))
            )
    return {
        "status": "PRE_PROVIDER_AUTHORIZATION_VALID",
        "preregistration_sha256": scientific["preregistration_sha256"],
        "run_identity_sha256": run_identity["run_identity_sha256"],
        "provider_boundary_reached": False,
    }


def write_bundle(root: Path, bundle: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("scientific_identity", "run_identity", "authorization"):
        path = root / f"{name}.json"
        path.write_bytes(canonical_json_bytes(bundle[name]) + b"\n")


def read_bundle(root: Path) -> dict[str, Any]:
    return {
        name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("scientific_identity", "run_identity", "authorization")
    }
