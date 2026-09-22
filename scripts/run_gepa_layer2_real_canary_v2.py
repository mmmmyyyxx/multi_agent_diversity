"""Fresh-execution GEPA Layer-2 real canary (pre-authorized freeze family).

The scientific path is inherited from the audited Level-B canary.  This module
changes only provider binding, execution identity, and fresh-freeze paths.
"""

from __future__ import annotations

from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_level_b_gepa_real_canary as base  # noqa: E402
from scripts.anti_overfitting_shadow_support import source_items  # noqa: E402
from multi_dataset_diverse_rl.governance.execution_harness_v2 import (  # noqa: E402
    INITIALIZATION_POLICY,
    PROVIDER_PROFILE,
    ROLE_MODEL,
    SOLVER_MODEL,
    endpoint_fingerprint_from_environment,
    preflight_provider_binding,
)
from multi_dataset_diverse_rl.governance.startup_identity import (  # noqa: E402
    StartupIdentityError,
    build_startup_bundle,
    read_bundle,
    validate_startup_bundle,
    write_bundle,
)


EXPERIMENT_ID = "gepa_layer2_real_canary_v2"
ATTEMPT_ID = "gepa_layer2_real_canary_v2_authorized3"
SEED = 80
MANIFEST = ROOT / "experiments/manifests/gepa_layer2_real_canary_v2.yaml"
PROTOCOL = ROOT / "experiments/gepa_layer2_real_canary_v2/PROTOCOL.md"
DEFAULT_PREP = ROOT / "runs/gepa_layer2_real_canary_v2_prep_authorized3"
DEFAULT_RUN = ROOT / "runs/gepa_layer2_real_canary_v2_authorized3"
DEFAULT_REPORT = ROOT / "reports/gepa_layer2_real_canary_v2_authorized3"
AUTH_ENV = "GEPA_LAYER2_REAL_CANARY_V2_AUTHORIZED3"

base.EXPERIMENT_ID = EXPERIMENT_ID
base.ATTEMPT_ID = ATTEMPT_ID
base.SEED = SEED
base.MANIFEST = MANIFEST
base.PROTOCOL = PROTOCOL
base.DEFAULT_PREP = DEFAULT_PREP
base.DEFAULT_RUN = DEFAULT_RUN
base.DEFAULT_REPORT = DEFAULT_REPORT
base.AUTH_ENV = AUTH_ENV

_base_config = base.config
_base_protocol_document = base.protocol_document
_base_source_paths = base.source_paths
_base_preflight = base.preflight


def config(out: Path, *, optimize_path: Path, validation_path: Path):
    cfg = _base_config(out, optimize_path=optimize_path, validation_path=validation_path)
    return replace(
        cfg,
        models=replace(
            cfg.models,
            provider_profile=PROVIDER_PROFILE,
            agent_model=SOLVER_MODEL,
            optimizer_model=ROLE_MODEL,
            evaluator_model=ROLE_MODEL,
        ),
        training=replace(cfg.training, seed=SEED),
    )


def protocol_document() -> dict[str, Any]:
    protocol = _base_protocol_document()
    protocol.update(
        {
            "schema_version": "gepa_layer2_real_canary_v2_protocol_v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": ATTEMPT_ID,
            "seed": SEED,
            "provider_profile": PROVIDER_PROFILE,
            "initialization_policy": INITIALIZATION_POLICY,
            "source_parent": "fresh canonical Optimize100 initialization",
            "private_parent_dependency": False,
            "initialization_cost_accounting": "SEPARATE_INITIALIZATION_SOLVER_CALLS",
            "root_focus": [],
            "root_anchor": [],
        }
    )
    return protocol


def source_paths() -> list[Path]:
    return sorted(
        set(_base_source_paths())
        | {
            Path("multi_dataset_diverse_rl/governance/execution_harness_v2.py"),
            Path("multi_dataset_diverse_rl/governance/startup_identity.py"),
            Path("scripts/run_gepa_layer2_real_canary_v2.py"),
        },
        key=lambda path: path.as_posix(),
    )


def _fresh_rows() -> dict[str, list[dict[str, str]]]:
    _items, raw = source_items()
    folds = json.loads(
        (ROOT / "experiments/anti_overfitting_split_v1/fold_assignment.json").read_text(
            encoding="utf-8"
        )
    )["folds"]
    split = json.loads(
        (ROOT / "experiments/anti_overfitting_split_v1/split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "fold_a": [raw[digest] for digest in folds["fold_a"]],
        "fold_b": [raw[digest] for digest in folds["fold_b"]],
        "fold_c": [raw[digest] for digest in folds["fold_c"]],
        "validation": [raw[digest] for digest in split["question_hashes"]["validation"]],
    }


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(
            {"question": row["question"], "answer": row["answer"]} for row in rows
        )


def _startup_bundle(prep: Path, *, execution_source_sha: str | None = None) -> dict[str, Any]:
    """Build the one canonical expected identity for prepare and runtime."""

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    protocol = protocol_document()
    source_files = [
        {"path": path.as_posix(), "sha256": base.source_freeze_sha256(ROOT / path)}
        for path in source_paths()
    ]
    data_hashes = {
        path.name: base.source_freeze_sha256(path)
        for path in sorted((prep / "splits_private").glob("*.csv"))
    }
    return build_startup_bundle(
        manifest=manifest,
        protocol=protocol,
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        scientific_method_anchor_sha=str(
            manifest["execution_freeze"]["scientific_method_anchor_sha"]
        ),
        execution_source_sha=execution_source_sha or base.git("rev-parse", "HEAD"),
        provider_profile=PROVIDER_PROFILE,
        endpoint_fingerprint=str(
            manifest["execution_freeze"]["provider"]["endpoint_fingerprint"]
        ),
        models=manifest["execution_freeze"]["provider"]["models"],
        data_hashes=data_hashes,
        initialization={"policy": INITIALIZATION_POLICY},
        seeds=[SEED],
        local_patience=3,
        team_patience=2,
        saturation_mode="single_opportunity_engineering_canary",
        source_files=source_files,
    )


def prepare(prep: Path) -> dict[str, Any]:
    if prep.exists():
        raise FileExistsError("fresh canary prep root required")
    if base.git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before canary freeze")
    if preflight()["gate"] != "PASS":
        raise RuntimeError("canary preflight must pass before freeze")
    rows = _fresh_rows()
    prep.mkdir(parents=True)
    for name, values in rows.items():
        _write_rows(prep / "splits_private" / f"{name}.csv", values)
    _write_rows(prep / "splits_private/optimize100.csv", rows["fold_a"] + rows["fold_b"])
    protocol = protocol_document()
    base.write_json(prep / "protocol_freeze.json", protocol)
    base.write_json(
        prep / "test_access_registry.json",
        {"events": [], "validation50_calls": 0, "test50_calls": 0},
    )
    startup = _startup_bundle(prep)
    source_files = startup["scientific_identity"]["payload"]["source_files"]
    data_hashes = startup["scientific_identity"]["payload"]["data_hashes"]
    write_bundle(prep / "startup_identity", startup)
    freeze = {
        "execution_commit": base.git("rev-parse", "HEAD"),
        "hash_semantics": base.SOURCE_FREEZE_HASH_SEMANTICS,
        "protocol_sha256": base.sha256_json(protocol),
        "preregistration_sha256": startup["scientific_identity"]["preregistration_sha256"],
        "run_identity_sha256": startup["run_identity"]["run_identity_sha256"],
        "attempt_id": ATTEMPT_ID,
        "files": source_files,
        "private_split_sha256": data_hashes,
        "private_parent_dependencies": [],
        "initialization_policy": INITIALIZATION_POLICY,
    }
    base.write_json(prep / "source_freeze.json", freeze)
    result = {
        "gate": "PASS",
        "ready_to_run": True,
        "authorization_state": (
            startup["authorization"]["authorization_state"]
        ),
        "execution_commit": freeze["execution_commit"],
        "protocol_sha256": freeze["protocol_sha256"],
        "provider_attempts": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    base.write_json(prep / "phase_a_gate.json", result)
    return result


def verify_startup_identity(
    prep: Path, *, require_authorized: bool
) -> dict[str, Any]:
    freeze = base.read_json(prep / "source_freeze.json")
    if base.git("rev-parse", "HEAD") != freeze["execution_commit"]:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: execution source mismatch")
    if base.sha256_json(base.read_json(prep / "protocol_freeze.json")) != freeze[
        "protocol_sha256"
    ]:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: protocol freeze mismatch")
    for row in freeze["files"]:
        if base.source_freeze_sha256(ROOT / row["path"]) != row["sha256"]:
            raise StartupIdentityError(
                f"ABORT_PRE_PROVIDER: source freeze mismatch: {row['path']}"
            )
    for name, digest in freeze["private_split_sha256"].items():
        if base.source_freeze_sha256(prep / "splits_private" / name) != digest:
            raise StartupIdentityError(
                f"ABORT_PRE_PROVIDER: split freeze mismatch: {name}"
            )
    expected = _startup_bundle(prep, execution_source_sha=freeze["execution_commit"])
    stored = read_bundle(prep / "startup_identity")
    result = validate_startup_bundle(
        stored=stored,
        expected=expected,
        require_authorized=require_authorized,
        phase="canary",
        roles=("solver", "reflection"),
    )
    if result["preregistration_sha256"] != freeze["preregistration_sha256"]:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: source/preregistration mismatch")
    if result["run_identity_sha256"] != freeze["run_identity_sha256"]:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: source/run identity mismatch")
    return result


def verify_freeze(prep: Path) -> None:
    verify_startup_identity(prep, require_authorized=False)


def startup_dry_run(prep: Path) -> dict[str, Any]:
    """Production-path authorization replay that deliberately stops pre-provider."""

    return verify_startup_identity(prep, require_authorized=True)


def start_run_attempt(prep: Path, run_root: Path) -> dict[str, Any]:
    # Scientific RUNNING is emitted only after static identity and authorization pass.
    identity = verify_startup_identity(prep, require_authorized=True)
    if run_root.exists():
        raise FileExistsError("fresh canary run root required; retry/resume forbidden")
    staging = run_root.with_name(f".{run_root.name}.{ATTEMPT_ID}.starting")
    if staging.exists():
        raise FileExistsError("fresh canary launch staging root required")
    freeze = base.read_json(prep / "source_freeze.json")
    lifecycle = {
        "schema_version": base.LAUNCH_TRANSACTION_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "status": "RUNNING",
        "source_commit": freeze["execution_commit"],
        "protocol_sha256": freeze["protocol_sha256"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "run_identity_sha256": identity["run_identity_sha256"],
        "provider_call_boundary_reached": False,
        "provider_calls_observed": 0,
        "events": [{"status": "RUNNING", "timestamp": base._utc_now()}],
    }
    run_root.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    base._atomic_write_json(staging / base.RUN_LIFECYCLE_FILE, lifecycle)
    os.replace(staging, run_root)
    return lifecycle


def authorize(run_root: Path) -> None:
    lifecycle = base.read_json(run_root / base.RUN_LIFECYCLE_FILE)
    if lifecycle.get("status") != "RUNNING":
        raise StartupIdentityError("ABORT_PRE_PROVIDER: lifecycle is not RUNNING")
    if lifecycle.get("attempt_id") != ATTEMPT_ID:
        raise StartupIdentityError("ABORT_PRE_PROVIDER: lifecycle attempt mismatch")


def _provider_freeze(manifest: dict[str, Any]) -> dict[str, Any]:
    frozen = manifest.get("execution_freeze", {}).get("provider", {})
    if not isinstance(frozen, dict):
        raise RuntimeError("ABORT_PRE_PROVIDER: provider freeze missing")
    return frozen


def preflight() -> dict[str, Any]:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    cfg = config(
        ROOT / "runs/_preflight_only",
        optimize_path=ROOT / "PREPARED_OPTIMIZE100.csv",
        validation_path=ROOT / "PREPARED_SHADOW50.csv",
    )
    provider = preflight_provider_binding(
        cfg, _provider_freeze(manifest), construct_client=False
    )
    protocol = protocol_document()
    checks = {
        "provider_profile_explicit": cfg.models.provider_profile == PROVIDER_PROFILE,
        "solver_model_exact": cfg.models.agent_model == SOLVER_MODEL,
        "role_model_exact": cfg.models.optimizer_model == ROLE_MODEL,
        "endpoint_fingerprint": bool(provider.endpoint_fingerprint),
        "fresh_initialization": protocol["initialization_policy"] == INITIALIZATION_POLICY,
        "private_parent_independent": protocol["private_parent_dependency"] is False,
        "one_opportunity": protocol["opportunities"] == 1,
        "validation_zero": protocol["validation50_calls"] == 0,
        "test_zero": protocol["test50_calls"] == 0,
        "authorization_required": manifest.get("status") == "PREFLIGHT_PASS"
        and manifest.get("api_authorization", {}).get("authorized") is False,
    }
    return {
        "gate": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "ready_to_run": all(checks.values()),
        "authorization_state": (
            "AUTHORIZATION_REQUIRED"
        ),
        "endpoint_fingerprint": provider.endpoint_fingerprint,
        "provider_attempts": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }


base.config = config
base.protocol_document = protocol_document
base.source_paths = source_paths
base.prepare = prepare
base.preflight = preflight
base.verify_freeze = verify_freeze
base.start_run_attempt = start_run_attempt
base.authorize = authorize


if __name__ == "__main__":
    if "--startup-dry-run" in sys.argv:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--startup-dry-run", action="store_true")
        parser.add_argument("--prep", type=Path, required=True)
        args = parser.parse_args()
        print(json.dumps(startup_dry_run(args.prep), indent=2, sort_keys=True))
    else:
        base.main()
