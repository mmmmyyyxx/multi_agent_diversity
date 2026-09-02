"""Fail-closed authorization checks for future API experiment entry points."""

from __future__ import annotations

import hmac
from typing import Any, Mapping

from .manifest import MANIFEST_STATUSES, preregistration_hash


class AuthorizationError(RuntimeError):
    """Raised before any provider call when authorization facts do not match."""


def require_api_authorization(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    role: str,
    explicit_user_authorized: bool = False,
) -> None:
    """Require user, manifest, role, phase, budget, and preregistration agreement."""
    authorization = manifest.get("api_authorization", {})
    if not explicit_user_authorized:
        raise AuthorizationError("explicit user API authorization is required")
    status = manifest.get("status")
    if status not in MANIFEST_STATUSES or MANIFEST_STATUSES.index(status) < MANIFEST_STATUSES.index("RUNNING"):
        raise AuthorizationError("manifest lifecycle has not entered RUNNING")
    if authorization.get("authorized") is not True:
        raise AuthorizationError("manifest does not authorize API access")
    if phase not in authorization.get("allowed_phases", []):
        raise AuthorizationError(f"phase is not API-authorized: {phase}")
    if role not in authorization.get("allowed_roles", []):
        raise AuthorizationError(f"role is not API-authorized: {role}")
    if manifest.get("budget", {}).get("frozen_before_run") is not True:
        raise AuthorizationError("API budget must be frozen before the call")
    recorded = manifest.get("artifacts", {}).get("preregistration", {}).get("sha256")
    actual = preregistration_hash(manifest)
    if not isinstance(recorded, str) or not hmac.compare_digest(recorded, actual):
        raise AuthorizationError("preregistration hash mismatch")
