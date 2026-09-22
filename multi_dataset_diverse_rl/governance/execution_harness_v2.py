"""Fail-closed execution contract for the 2026-09-22 fresh freezes.

This module owns provider/configuration and run-identity checks only.  It must
not import or alter responsibility, optimizer, team-evaluation, or write-back
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Mapping

from openai import AsyncOpenAI


from ..config import Config
from ..provider_credentials import (
    LWJ_DASHSCOPE_API_KEY_ENV,
    LWJ_DASHSCOPE_BASE_URL_ENV,
    resolve_api_key,
    resolve_base_url,
)
from ..provider_factory import ProviderClientFactory


EXECUTION_FREEZE_VERSION = "fresh_execution_freeze_v2"
INITIALIZATION_POLICY = "FRESH_DETERMINISTIC_INITIALIZATION_V1"
PROVIDER_PROFILE = "lwj"
SOLVER_MODEL = "qwen3-8b"
ROLE_MODEL = "qwen3.7-flash"
FORBIDDEN_ROLE_MODEL = "qwen3.7-flash-2026-07-15"
LOCAL_NO_UPDATE_PATIENCE = 3
TEAM_NO_UPDATE_PATIENCE = 2
FORMAL_SEEDS = (80, 81, 82)
SCIENTIFIC_METHOD_ANCHOR_SHA = "a85e31bea2ab28f62abb31337b91f9895b11ae37"
SCIENTIFIC_PREREGISTRATION_SHA = "80442264a8086f3fe407e1551ae2ecc8448f8676"


class ExecutionBindingError(RuntimeError):
    """Raised before provider access when a frozen execution identity drifts."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_endpoint_fingerprint(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    if not normalized:
        raise ExecutionBindingError("LWJ_DASHSCOPE_BASE_URL is required")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def endpoint_fingerprint_from_environment(
    environment: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if environment is None else environment
    return normalized_endpoint_fingerprint(
        str(values.get(LWJ_DASHSCOPE_BASE_URL_ENV, ""))
    )


def frozen_model_identities(*, critic_used: bool, teacher_used: bool,
                            student_used: bool, evaluator_used: bool) -> dict[str, Any]:
    def role(used: bool) -> dict[str, Any]:
        return {"model": ROLE_MODEL if used else None, "status": "USED" if used else "UNUSED"}

    return {
        "solver": {"model": SOLVER_MODEL, "status": "USED", "thinking": False},
        "teacher": role(teacher_used),
        "critic": role(critic_used),
        "student": role(student_used),
        "evaluator": role(evaluator_used),
    }


def assert_config_binding(cfg: Config, frozen: Mapping[str, Any]) -> None:
    if cfg.models.provider_profile != PROVIDER_PROFILE:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: provider_profile must be lwj")
    if str(frozen.get("provider_profile", "")) != PROVIDER_PROFILE:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: manifest provider_profile missing or changed")
    expected_models = frozen.get("models", {})
    if cfg.models.agent_model != SOLVER_MODEL:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: solver model mismatch")
    if cfg.models.optimizer_model != ROLE_MODEL or cfg.models.evaluator_model != ROLE_MODEL:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: role model mismatch")
    if expected_models.get("solver", {}).get("model") != SOLVER_MODEL:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: frozen solver identity mismatch")
    for role in ("teacher", "critic", "student", "evaluator"):
        row = expected_models.get(role, {})
        if row.get("status") == "USED" and row.get("model") != ROLE_MODEL:
            raise ExecutionBindingError(f"ABORT_PRE_PROVIDER: frozen {role} identity mismatch")


@dataclass(frozen=True)
class ProviderPreflight:
    provider_profile: str
    endpoint_fingerprint: str
    solver_model: str
    optimizer_model: str
    evaluator_model: str
    client_constructed: bool
    raw_endpoint_persisted: bool = False
    api_key_persisted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_profile": self.provider_profile,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "solver_model": self.solver_model,
            "optimizer_model": self.optimizer_model,
            "evaluator_model": self.evaluator_model,
            "client_constructed": self.client_constructed,
            "raw_endpoint_persisted": self.raw_endpoint_persisted,
            "api_key_persisted": self.api_key_persisted,
        }


def preflight_provider_binding(
    cfg: Config,
    frozen: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    construct_client: bool = True,
) -> ProviderPreflight:
    """Validate configuration without sending a request.

    ``AsyncOpenAI`` construction performs no network I/O.  The object is closed
    by garbage collection; no request method is called here.
    """

    assert_config_binding(cfg, frozen)
    values = os.environ if environment is None else environment
    api_key = str(values.get(LWJ_DASHSCOPE_API_KEY_ENV, ""))
    endpoint = str(values.get(LWJ_DASHSCOPE_BASE_URL_ENV, ""))
    if not api_key:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: LWJ_DASHSCOPE_API_KEY is required")
    fingerprint = normalized_endpoint_fingerprint(endpoint)
    frozen_fingerprint = str(frozen.get("endpoint_fingerprint", ""))
    if not frozen_fingerprint or fingerprint != frozen_fingerprint:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: endpoint fingerprint mismatch")
    if construct_client:
        ProviderClientFactory.create(
            api_key=api_key, base_url=endpoint, client_type=AsyncOpenAI
        )
    return ProviderPreflight(
        provider_profile=PROVIDER_PROFILE,
        endpoint_fingerprint=fingerprint,
        solver_model=cfg.models.agent_model,
        optimizer_model=cfg.models.optimizer_model,
        evaluator_model=cfg.models.evaluator_model,
        client_constructed=construct_client,
    )


def require_runtime_environment(cfg: Config) -> dict[str, str]:
    """Resolve exact lwj credentials; never fall back to historical myx."""

    if cfg.models.provider_profile != PROVIDER_PROFILE:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: provider_profile must be lwj")
    key_env, key = resolve_api_key("", PROVIDER_PROFILE)
    base_env, base = resolve_base_url("", PROVIDER_PROFILE)
    if key_env != LWJ_DASHSCOPE_API_KEY_ENV or not key:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: lwj API key unavailable")
    if base_env != LWJ_DASHSCOPE_BASE_URL_ENV or not base:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: lwj endpoint unavailable")
    return {"endpoint_fingerprint": normalized_endpoint_fingerprint(base)}


def execution_identity(
    *, experiment_id: str, execution_source_sha: str, protocol_sha: str,
    manifest_sha: str, endpoint_fingerprint: str, seed: int,
    models: Mapping[str, Any], initialization_policy: str = INITIALIZATION_POLICY,
) -> dict[str, Any]:
    return {
        "schema_version": "fresh_execution_identity_v1",
        "experiment_id": experiment_id,
        "scientific_method_anchor_sha": SCIENTIFIC_METHOD_ANCHOR_SHA,
        "scientific_preregistration_sha": SCIENTIFIC_PREREGISTRATION_SHA,
        "execution_source_sha": execution_source_sha,
        "protocol_sha": protocol_sha,
        "manifest_sha": manifest_sha,
        "provider_profile": PROVIDER_PROFILE,
        "endpoint_fingerprint": endpoint_fingerprint,
        "models": dict(models),
        "seed": seed,
        "local_no_update_patience": LOCAL_NO_UPDATE_PATIENCE,
        "team_no_update_patience": TEAM_NO_UPDATE_PATIENCE,
        "initialization_policy": initialization_policy,
        "execution_freeze_version": EXECUTION_FREEZE_VERSION,
    }


def authorization_binding(identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(identity)
    return {
        "schema_version": "one_time_execution_authorization_binding_v1",
        "authorized": False,
        "authorization_state": "AUTHORIZATION_REQUIRED",
        "identity_sha256": canonical_json_sha256(payload),
        "identity": payload,
    }


def fresh_initialization_identity(
    *, seed: int, canonical_prompt: str, question_hashes: list[str],
    provider_profile: str, endpoint_fingerprint: str, solver_model: str,
    member_profile_hashes: list[str], team_state_hash: str,
) -> dict[str, Any]:
    """Canonical structural identity emitted after empirical initialization."""

    if provider_profile != PROVIDER_PROFILE:
        raise ExecutionBindingError("fresh initialization requires lwj")
    if solver_model != SOLVER_MODEL:
        raise ExecutionBindingError("fresh initialization solver model mismatch")
    if len(member_profile_hashes) != 5:
        raise ExecutionBindingError("fresh initialization requires five member profiles")
    prompt_hash = hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()
    payload = {
        "schema_version": "fresh_deterministic_initialization_v1",
        "policy": INITIALIZATION_POLICY,
        "seed": seed,
        "provider_profile": provider_profile,
        "endpoint_fingerprint": endpoint_fingerprint,
        "solver_model": solver_model,
        "canonical_prompt_sha256": prompt_hash,
        "member_prompt_hashes": [prompt_hash] * 5,
        "question_hashes_sha256": canonical_json_sha256(sorted(question_hashes)),
        "member_profile_hashes": list(member_profile_hashes),
        "team_state_hash": team_state_hash,
        "focus": [],
        "anchor": [],
    }
    payload["identity_sha256"] = canonical_json_sha256(payload)
    return payload


def assert_authorized(
    authorization: Mapping[str, Any], identity: Mapping[str, Any],
) -> None:
    if authorization.get("authorized") is not True:
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: explicit authorization required")
    if authorization.get("identity_sha256") != canonical_json_sha256(dict(identity)):
        raise ExecutionBindingError("ABORT_PRE_PROVIDER: authorization identity mismatch")


def formal_dependencies_satisfied(
    *, canary_status: str, sequential_status: str,
) -> bool:
    return (
        canary_status == "TECHNICAL_PATH_CONFIRMED"
        and sequential_status == "REQUIRED_GATE_SATISFIED"
    )
