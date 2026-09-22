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


EXPERIMENT_ID = "gepa_layer2_real_canary_v2"
ATTEMPT_ID = "gepa_layer2_real_canary_v2_authorized1"
SEED = 80
MANIFEST = ROOT / "experiments/manifests/gepa_layer2_real_canary_v2.yaml"
PROTOCOL = ROOT / "experiments/gepa_layer2_real_canary_v2/PROTOCOL.md"
DEFAULT_PREP = ROOT / "runs/gepa_layer2_real_canary_v2_prep"
DEFAULT_RUN = ROOT / "runs/gepa_layer2_real_canary_v2_attempt1"
DEFAULT_REPORT = ROOT / "reports/gepa_layer2_real_canary_v2_attempt1"
AUTH_ENV = "GEPA_LAYER2_REAL_CANARY_V2_AUTHORIZED"

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
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    freeze = {
        "execution_commit": base.git("rev-parse", "HEAD"),
        "hash_semantics": base.SOURCE_FREEZE_HASH_SEMANTICS,
        "protocol_sha256": base.sha256_json(protocol),
        "preregistration_sha256": base.preregistration_hash(manifest),
        "attempt_id": ATTEMPT_ID,
        "files": [
            {"path": path.as_posix(), "sha256": base.source_freeze_sha256(ROOT / path)}
            for path in source_paths()
        ],
        "private_split_sha256": {
            path.name: base.source_freeze_sha256(path)
            for path in sorted((prep / "splits_private").glob("*.csv"))
        },
        "private_parent_dependencies": [],
        "initialization_policy": INITIALIZATION_POLICY,
    }
    base.write_json(prep / "source_freeze.json", freeze)
    result = {
        "gate": "PASS",
        "ready_to_run": True,
        "authorization_state": "AUTHORIZATION_REQUIRED",
        "execution_commit": freeze["execution_commit"],
        "protocol_sha256": freeze["protocol_sha256"],
        "provider_attempts": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    base.write_json(prep / "phase_a_gate.json", result)
    return result


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
    provider = preflight_provider_binding(cfg, _provider_freeze(manifest))
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
        "authorization_state_valid": manifest.get("status") == "PREFLIGHT_PASS"
        and isinstance(manifest.get("api_authorization", {}).get("authorized"), bool),
    }
    return {
        "gate": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "ready_to_run": all(checks.values()),
        "authorization_state": (
            "AUTHORIZED" if manifest.get("api_authorization", {}).get("authorized") is True
            else "AUTHORIZATION_REQUIRED"
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


if __name__ == "__main__":
    base.main()
