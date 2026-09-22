"""Execution gate and explicit configurations for the GEPA saturation study.

The formal experiment deliberately remains blocked until both predecessor gates
are satisfied and a later one-time authorization freeze is created.  This file
contains no alternative scientific implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config  # noqa: E402
from multi_dataset_diverse_rl.governance.execution_harness_v2 import (  # noqa: E402
    FORMAL_SEEDS,
    INITIALIZATION_POLICY,
    LOCAL_NO_UPDATE_PATIENCE,
    PROVIDER_PROFILE,
    ROLE_MODEL,
    SOLVER_MODEL,
    TEAM_NO_UPDATE_PATIENCE,
    formal_dependencies_satisfied,
    preflight_provider_binding,
)
from multi_dataset_diverse_rl.governance.freeze_hash import source_freeze_sha256  # noqa: E402
from multi_dataset_diverse_rl.governance.startup_identity import (  # noqa: E402
    build_startup_bundle,
    validate_startup_bundle,
)


EXPERIMENT_ID = "gepa_saturation_comparison_v2"
MANIFEST = ROOT / "experiments/manifests/gepa_saturation_comparison_v2.yaml"


def startup_bundle(manifest: dict[str, Any]) -> dict[str, Any]:
    execution_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "arms": ["GEPA_NATIVE_SATURATION", "GEPA_LAYER2_SATURATION"],
        "seeds": list(FORMAL_SEEDS),
        "initialization_policy": INITIALIZATION_POLICY,
        "local_no_update_patience": LOCAL_NO_UPDATE_PATIENCE,
        "team_no_update_patience": TEAM_NO_UPDATE_PATIENCE,
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    provider = manifest["execution_freeze"]["provider"]
    source_paths = (
        Path("multi_dataset_diverse_rl/governance/startup_identity.py"),
        Path("multi_dataset_diverse_rl/governance/execution_harness_v2.py"),
        Path("scripts/run_gepa_saturation_comparison_v2.py"),
        MANIFEST.relative_to(ROOT),
    )
    return build_startup_bundle(
        manifest=manifest,
        protocol=protocol,
        experiment_id=EXPERIMENT_ID,
        attempt_id="gepa_saturation_comparison_v2_execution_gated",
        scientific_method_anchor_sha=str(
            manifest["execution_freeze"]["scientific_method_anchor_sha"]
        ),
        execution_source_sha=execution_sha,
        provider_profile=PROVIDER_PROFILE,
        endpoint_fingerprint=str(provider["endpoint_fingerprint"]),
        models=provider["models"],
        data_hashes={
            str(key): str(value)
            for key, value in manifest.get("data", {}).get("split_hashes", {}).items()
        },
        initialization={"policy": INITIALIZATION_POLICY},
        seeds=FORMAL_SEEDS,
        local_patience=LOCAL_NO_UPDATE_PATIENCE,
        team_patience=TEAM_NO_UPDATE_PATIENCE,
        saturation_mode="formal_saturation",
        source_files=[
            {"path": path.as_posix(), "sha256": source_freeze_sha256(ROOT / path)}
            for path in source_paths
        ],
    )


def common_config(*, seed: int, out: Path, optimize_path: Path, shadow_path: Path) -> Config:
    if seed not in FORMAL_SEEDS:
        raise ValueError("formal seed must be one of 80, 81, 82")
    return Config.from_flat(
        task_type="bbh",
        dataset_format="mars",
        comparison_task_id="disambiguation_qa",
        benchmark="BBH",
        answer_format="option_letter",
        train_path=str(optimize_path),
        val_path=str(shadow_path),
        test_path="VALIDATION50_AND_TEST50_BLOCKED",
        train_size=100,
        val_size=50,
        test_size=0,
        provider_profile=PROVIDER_PROFILE,
        agent_model=SOLVER_MODEL,
        optimizer_model=ROLE_MODEL,
        evaluator_model=ROLE_MODEL,
        temperature=0.0,
        solver_max_tokens=1800,
        seed=seed,
        agents=5,
        out_dir=str(out),
        final_test_enabled=False,
        preserve_final_checkpoint=True,
    )


def native_config(**kwargs: Any) -> Config:
    cfg = common_config(**kwargs)
    return replace(
        cfg,
        training=replace(cfg.training, initialization_mode="shared_identical"),
    )


def layer2_config(**kwargs: Any) -> Config:
    cfg = common_config(**kwargs)
    return replace(
        cfg,
        training=replace(
            cfg.training,
            initialization_mode="shared_identical",
            target_scheduler="primary_responsibility_persistent_realizability",
        ),
    )


def preflight() -> dict[str, Any]:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    provider_freeze = manifest["execution_freeze"]["provider"]
    native = native_config(
        seed=FORMAL_SEEDS[0],
        out=ROOT / "runs/_preflight_native",
        optimize_path=ROOT / "PREPARED_OPTIMIZE100.csv",
        shadow_path=ROOT / "PREPARED_SHADOW50.csv",
    )
    layer2 = layer2_config(
        seed=FORMAL_SEEDS[0],
        out=ROOT / "runs/_preflight_layer2",
        optimize_path=ROOT / "PREPARED_OPTIMIZE100.csv",
        shadow_path=ROOT / "PREPARED_SHADOW50.csv",
    )
    native_provider = preflight_provider_binding(
        native, provider_freeze, construct_client=False
    )
    layer2_provider = preflight_provider_binding(
        layer2, provider_freeze, construct_client=False
    )
    identity = startup_bundle(manifest)
    validate_startup_bundle(
        stored=identity, expected=identity, require_authorized=False
    )
    dependencies = manifest.get("execution_gate", {}).get("observed_dependencies", {})
    dependency_gate = formal_dependencies_satisfied(
        canary_status=str(dependencies.get("canary", "PENDING")),
        sequential_status=str(dependencies.get("sequential", "PENDING")),
    )
    checks = {
        "provider_profile_explicit_native": native.models.provider_profile == PROVIDER_PROFILE,
        "provider_profile_explicit_layer2": layer2.models.provider_profile == PROVIDER_PROFILE,
        "same_provider": native_provider.endpoint_fingerprint == layer2_provider.endpoint_fingerprint,
        "formal_seeds": list(FORMAL_SEEDS) == [80, 81, 82],
        "local_patience": LOCAL_NO_UPDATE_PATIENCE == 3,
        "team_patience": TEAM_NO_UPDATE_PATIENCE == 2,
        "initialization_policy": manifest["execution_freeze"]["initialization_policy"]
        == INITIALIZATION_POLICY,
        "validation_zero": manifest["access"]["validation50_calls"] == 0,
        "test_zero": manifest["access"]["test50_calls"] == 0,
        "canonical_startup_identity": bool(
            identity["scientific_identity"]["preregistration_sha256"]
        ),
    }
    return {
        "gate": "PASS_EXECUTION_GATED" if all(checks.values()) else "HOLD",
        "checks": checks,
        "dependencies_satisfied": dependency_gate,
        "executable": False,
        "status": "PREREGISTERED_EXECUTION_GATED",
        "provider_attempts": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
        "preregistration_sha256": identity["scientific_identity"][
            "preregistration_sha256"
        ],
    }


def execute() -> None:
    result = preflight()
    if not result["dependencies_satisfied"]:
        raise RuntimeError("formal saturation is execution-gated on canary and sequential evidence")
    raise RuntimeError("formal saturation requires a new post-gate one-time authorization freeze")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print(json.dumps(preflight(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
