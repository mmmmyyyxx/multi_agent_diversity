"""Experiment-governance primitives with no model or dataset side effects."""

from .artifacts import (
    build_sha256_manifest,
    compare_tree_snapshot,
    scan_sanitized_artifacts,
    snapshot_tree_hashes,
    validate_fact_assertions,
    validate_report_contract,
)
from .authorization import AuthorizationError, require_api_authorization
from .manifest import (
    MANIFEST_STATUSES,
    load_manifest,
    preregistration_hash,
    validate_manifest,
)
from .provenance import build_provenance, source_tree_hash
from .telemetry import (
    FUNNEL_STAGES,
    summarize_api_ledger,
    summarize_funnel,
    validate_evaluation_access,
)

__all__ = [
    "AuthorizationError",
    "FUNNEL_STAGES",
    "MANIFEST_STATUSES",
    "build_provenance",
    "build_sha256_manifest",
    "compare_tree_snapshot",
    "load_manifest",
    "preregistration_hash",
    "require_api_authorization",
    "scan_sanitized_artifacts",
    "source_tree_hash",
    "snapshot_tree_hashes",
    "summarize_api_ledger",
    "summarize_funnel",
    "validate_evaluation_access",
    "validate_fact_assertions",
    "validate_manifest",
    "validate_report_contract",
]
