"""Read-only Formal V3 governance, source-poison and isolated-prep replay audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance import production_execution as admission  # noqa: E402
from multi_dataset_diverse_rl.formal_validation_policy import formal_validation50_policy  # noqa: E402
from multi_dataset_diverse_rl.governance.freeze_hash import (  # noqa: E402
    normalized_lf_bytes, source_freeze_sha256,
)
from multi_dataset_diverse_rl.governance.startup_identity import StartupIdentityError  # noqa: E402
from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory  # noqa: E402


REQUIRED_SOURCE_PATHS = frozenset({
    "multi_dataset_diverse_rl/experiment.py",
    "multi_dataset_diverse_rl/saturation.py",
    "multi_dataset_diverse_rl/versions.py",
    "multi_dataset_diverse_rl/provider_factory.py",
    "multi_dataset_diverse_rl/provider_credentials.py",
    "multi_dataset_diverse_rl/llm_client.py",
    "multi_dataset_diverse_rl/native_feed.py",
    "multi_dataset_diverse_rl/local_optimizers/gepa_native.py",
    "multi_dataset_diverse_rl/local_optimizers/gepa_optimizer.py",
    "multi_dataset_diverse_rl/team_search/controller.py",
    "multi_dataset_diverse_rl/team_search/execution_runtime.py",
    "multi_dataset_diverse_rl/team_search/system_runtime.py",
    "multi_dataset_diverse_rl/team_search/task_builder.py",
    "multi_dataset_diverse_rl/team_search/primary_responsibility_scheduler.py",
    "multi_dataset_diverse_rl/team_search/feasibility.py",
    "multi_dataset_diverse_rl/production_formal_saturation.py",
    "multi_dataset_diverse_rl/governance/production_execution.py",
    "scripts/run_experiment.py",
    "scripts/prepare_gepa_saturation_comparison_v3.py",
    "scripts/audit_formal_v3_offline_freeze.py",
    "experiments/gepa_saturation_comparison_v3/PROTOCOL.md",
})


def _files(prep: Path) -> dict[str, str]:
    return {
        path.relative_to(prep).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(prep.rglob("*")) if path.is_file()
    }


def audit(prep_a: Path, prep_b: Path, *, attempt_number: int = 1) -> dict[str, object]:
    if attempt_number not in {1, 2}:
        raise ValueError("unsupported Formal attempt number")
    a, b = _files(prep_a), _files(prep_b)
    if a != b:
        raise AssertionError("two isolated Formal V3 preparations differ in bytes")
    expected_cells = {
        f"gepa_saturation_comparison_v3_seed{seed}_{scope}_attempt{attempt_number}"
        for seed in (80, 81, 82) for scope in ("native", "layer2")
    }
    if {path.name for path in prep_a.iterdir() if path.is_dir()} != expected_cells:
        raise AssertionError("Formal V3 cell inventory mismatch")
    old_create, old_from_environment = (
        ProviderClientFactory.create, ProviderClientFactory.from_environment,
    )
    provider_constructions = 0

    def forbidden(*_args, **_kwargs):
        nonlocal provider_constructions
        provider_constructions += 1
        raise AssertionError("provider construction during offline freeze audit")

    ProviderClientFactory.create = forbidden
    ProviderClientFactory.from_environment = forbidden
    original_hash = admission.source_freeze_sha256
    poison_cases = 0
    try:
        reference_sources = None
        for cell in sorted(expected_cells):
            prep = prep_a / cell
            manifest = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
            protocol = json.loads((prep / "protocol.json").read_text(encoding="utf-8"))
            source_paths = manifest["execution"]["source_paths"]
            if reference_sources is None:
                reference_sources = source_paths
            elif source_paths != reference_sources:
                raise AssertionError("formal cells do not share one source closure")
            if not REQUIRED_SOURCE_PATHS <= set(source_paths):
                raise AssertionError("formal source closure misses a required active dependency")
            if attempt_number == 2:
                if (manifest.get("post_freeze_validation50") != formal_validation50_policy(ROOT)
                        or protocol.get("post_freeze_validation50") != manifest["post_freeze_validation50"]
                        or "experiments/anti_overfitting_split_v1/split_manifest.json"
                        not in source_paths):
                    raise AssertionError("Formal Validation50 endpoint identity mismatch")
                if not {
                    "multi_dataset_diverse_rl/formal_final_team.py",
                    "multi_dataset_diverse_rl/formal_trajectory.py",
                    "scripts/freeze_formal_v3_execution.py",
                    "scripts/derive_formal_trajectory_trace.py",
                } <= set(source_paths):
                    raise AssertionError("formal final-team or trajectory source missing")
                active_python = {
                    path.relative_to(ROOT).as_posix()
                    for directory in ("multi_dataset_diverse_rl", "infrastructure/common_solver_contract_v1")
                    for path in (ROOT / directory).rglob("*.py")
                }
                if not active_python <= set(source_paths):
                    raise AssertionError("formal production Python source escapes frozen closure")
            if not any(path.startswith("infrastructure/common_solver_contract_v1/")
                       for path in source_paths):
                raise AssertionError("common Solver contract source omitted")
            if attempt_number == 1:
                if manifest["execution_gate"]["real_v4_diagnostic"] != "PENDING_SCIENTIFIC_VALIDITY":
                    raise AssertionError("historical Formal prerequisite changed")
            elif (manifest.get("diagnostic_prerequisite") != admission.formal_attempt2_prerequisite(ROOT)
                  or protocol.get("diagnostic_prerequisite") != manifest["diagnostic_prerequisite"]
                  or manifest["execution_gate"]["real_v4_diagnostic"] != "SCIENTIFICALLY_VALID"):
                raise AssertionError("Formal attempt2 pilot evidence mismatch")
            if manifest["api_authorization"]["authorized"]:
                raise AssertionError("offline formal freeze must not authorize API execution")
            if manifest["scientific"]["emergency_max_provider_calls"] != 100_000:
                raise AssertionError("emergency ceiling drift")
            if manifest["formal_v3_contract"] != admission.formal_v3_contract():
                raise AssertionError("active formal V4 binding drift")
            if protocol["validation50_calls"] or protocol["test50_calls"]:
                raise AssertionError("external split access drift")
            admission.validate_execution(root=ROOT, prep=prep, require_authorized=False)
            if cell != sorted(expected_cells)[0]:
                continue
            for relative in source_paths:
                target = ROOT / relative
                raw = target.read_bytes()
                poison = normalized_lf_bytes(raw) + b"\n# offline-source-poison\n"
                poisoned_sha = hashlib.sha256(poison).hexdigest()
                target_resolved = target.resolve()

                def poisoned_hash(path, *, _target=target_resolved, _sha=poisoned_sha):
                    if Path(path).resolve() == _target:
                        return _sha
                    return original_hash(path)

                admission.source_freeze_sha256 = poisoned_hash
                try:
                    try:
                        admission.validate_execution(
                            root=ROOT, prep=prep, require_authorized=False,
                        )
                    except StartupIdentityError:
                        poison_cases += 1
                    else:
                        raise AssertionError("source byte poison passed pre-provider admission")
                finally:
                    admission.source_freeze_sha256 = original_hash
    finally:
        admission.source_freeze_sha256 = original_hash
        ProviderClientFactory.create = old_create
        ProviderClientFactory.from_environment = old_from_environment
    if provider_constructions:
        raise AssertionError("provider construction occurred during offline audit")
    return {
        "status": "PASS", "cells": len(expected_cells),
        "deterministic_artifacts": len(a), "prep_replay_identical": True,
        "source_poison_cases": poison_cases,
        "provider_constructions": 0,
        "validation50_calls": 0, "test50_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-a", type=Path, required=True)
    parser.add_argument("--prep-b", type=Path, required=True)
    parser.add_argument("--attempt-number", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    print(json.dumps(audit(args.prep_a, args.prep_b, attempt_number=args.attempt_number), sort_keys=True, indent=2))
