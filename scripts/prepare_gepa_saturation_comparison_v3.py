# HISTORICAL_REPLAY_ONLY: preserved frozen reproduction utility; new experiments use scripts/run_experiment.py.
"""Create six fresh, execution-gated Formal V3 preparations without API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.experiment import experiment_spec_from_mapping  # noqa: E402
from multi_dataset_diverse_rl.formal_validation_policy import formal_validation50_policy  # noqa: E402
from multi_dataset_diverse_rl.governance.execution_harness_v2 import (  # noqa: E402
    FORMAL_SEEDS, INITIALIZATION_POLICY, PROVIDER_PROFILE, ROLE_MODEL,
    SOLVER_MODEL, endpoint_fingerprint_from_environment,
)
from multi_dataset_diverse_rl.governance.production_execution import (  # noqa: E402
    FORMAL_ATTEMPT3_LAST_MILE, FORMAL_ATTEMPT4_EXECUTION,
    _expected_bundle, formal_attempt2_prerequisite, formal_attempt3_incident,
    formal_v3_contract, validate_execution,
)
from multi_dataset_diverse_rl.governance.startup_identity import (  # noqa: E402
    canonical_json_bytes, write_bundle,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import verify_frozen_gepa  # noqa: E402
from multi_dataset_diverse_rl.versions import LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION  # noqa: E402
from scripts.prepare_post_refactor_gepa_canary import _private_splits, source_paths  # noqa: E402


SUCCESSOR_ID = "gepa_saturation_comparison_v3"
DEFAULT_PREP = ROOT / "runs" / "gepa_saturation_comparison_v3_offline_prep1"
EMERGENCY = {
    "emergency_max_provider_calls": 100_000,
    "emergency_max_optimizer_steps": 100_000,
    "emergency_max_team_epochs": 10_000,
    "emergency_max_wall_seconds": 86_400,
}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8",
    ).strip()


def frozen_payload(*, execution_source_sha: str, seed: int, scope: str,
                   attempt_number: int = 1) -> tuple[dict, dict]:
    if seed not in FORMAL_SEEDS or scope not in {"native", "layer2"}:
        raise ValueError("Formal V3 supports only seeds 80/81/82 and native/layer2")
    if attempt_number not in {1, 2, 3, 4}:
        raise ValueError("Formal V3 supports only attempts 1, 2, 3 and 4")
    attempt = f"{SUCCESSOR_ID}_seed{seed}_{scope}_attempt{attempt_number}"
    scientific = {
        "backend": "gepa", "optimization_scope": scope,
        "stopping_regime": "saturation", "task_identity": "BBH_disambiguation_qa",
        "data_identity": "anti_overfitting_split_v1_fold_a+b_to_c",
        "fixed_budget_units": 1,
        "local_no_update_patience": 3, "team_no_update_patience": 2,
        "layer2_protocol_version": (
            LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION if scope == "layer2" else None
        ),
        **EMERGENCY,
    }
    spec = experiment_spec_from_mapping(scientific)
    contract = formal_v3_contract()
    sources = sorted(set(source_paths()) | {
        "scripts/prepare_gepa_saturation_comparison_v3.py",
        "scripts/audit_formal_v3_offline_freeze.py",
        "scripts/build_formal_v3_closure_report.py",
        "experiments/gepa_saturation_comparison_v3/PROTOCOL.md",
    })
    prerequisite = None
    validation_policy = None
    if attempt_number >= 2:
        prerequisite = formal_attempt2_prerequisite(ROOT)
        validation_policy = formal_validation50_policy(ROOT)
        sources = sorted(set(sources) | {
            "experiments/anti_overfitting_split_v1/split_manifest.json",
            "experiments/gepa_saturation_comparison_v3/PILOT_CLOSURE.json",
            "experiments/gepa_saturation_comparison_v3/POST_FREEZE_VALIDATION50_EVALUATION.md",
            "multi_dataset_diverse_rl/formal_trajectory.py",
            "multi_dataset_diverse_rl/formal_final_team.py",
            "scripts/derive_formal_trajectory_trace.py",
            "scripts/freeze_formal_v3_execution.py",
        })
    if attempt_number >= 3:
        sources = sorted(set(sources) | {
            "scripts/audit_formal_v3_json_roundtrip.py",
        })
    if attempt_number == 4:
        sources = sorted(set(sources) | {
            "scripts/audit_formal_v3_concurrent_equivalence.py",
            "scripts/benchmark_formal_v3_local_batch.py",
            "scripts/compare_formal_v3_concurrent_equivalence.py",
        })
    manifest = {
        "schema_version": "formal_gepa_saturation_freeze_v3",
        "experiment_id": attempt, "attempt_id": attempt,
        "successor_id": SUCCESSOR_ID,
        "status": "PREREGISTERED_EXECUTION_GATED",
        "supersedes_unexecuted": "gepa_saturation_comparison_v2",
        "method_identity": spec.method_identity,
        "spec_identity": spec.identity(), "scientific": scientific,
        "formal_v3_contract": contract,
        "runtime": {
            "seed": seed, "provider_profile": PROVIDER_PROFILE,
            "endpoint_fingerprint": endpoint_fingerprint_from_environment(),
            "solver_model": SOLVER_MODEL, "optimizer_model": ROLE_MODEL,
            "evaluator_model": ROLE_MODEL,
            **({"eval_solver_call_concurrency": 16} if attempt_number == 4 else {}),
        },
        "models": {
            "solver": {"model": SOLVER_MODEL, "thinking": False},
            "reflection": {"model": ROLE_MODEL},
        },
        "dependency": {"gepa": verify_frozen_gepa()},
        "execution": {
            "scientific_method_anchor_sha": (
                "9737626373790aeb55a8ab6b99937b6d3085eace"
                if attempt_number >= 3 else execution_source_sha
            ),
            "execution_source_sha": execution_source_sha,
            "initialization_policy": INITIALIZATION_POLICY,
            "source_paths": sources,
        },
        "execution_gate": {"real_v4_diagnostic": (
            "SCIENTIFICALLY_VALID" if attempt_number >= 2 else "PENDING_SCIENTIFIC_VALIDITY"
        )},
        "api_authorization": {
            "authorized": False, "authorization_state": "AUTHORIZATION_REQUIRED",
            "allowed_roles": ["solver", "reflection"], "allowed_phases": ["formal"],
        },
        "access": {"validation50_calls": 0, "test50_calls": 0},
    }
    if prerequisite is not None:
        manifest["diagnostic_prerequisite"] = prerequisite
        manifest["post_freeze_validation50"] = validation_policy
        manifest["preexecution_refreeze"] = {
            "superseded_execution_source_sha": "504dadb4024a0c69cc2f8aac6e6d400f77f3a680",
            "reason": "held_out_validation50_was_mislabeled_shadow50_fold_c",
            "formal_real_calls_observed": 0,
            "scientific_search_changed": False,
        }
    if attempt_number >= 3:
        manifest["execution_repair"] = formal_attempt3_incident(ROOT)
        manifest["preexecution_hardening"] = dict(FORMAL_ATTEMPT3_LAST_MILE)
    if attempt_number == 4:
        manifest["execution_scheduling"] = dict(FORMAL_ATTEMPT4_EXECUTION)
    protocol = {
        "schema_version": "formal_gepa_saturation_protocol_v3",
        "experiment_id": attempt, "successor_id": SUCCESSOR_ID,
        "seed": seed, "arm": "GEPA_NATIVE" if scope == "native" else "GEPA_LAYER2_V4",
        "formal_v3_contract": contract,
        "initialization_policy": INITIALIZATION_POLICY,
        "optimize_rows": 100, "shadow_rows": 50,
        "data_identity": scientific["data_identity"],
        "scientific_budget": "none_saturation_only",
        "local_no_update_patience": 3, "team_no_update_patience": 2,
        "emergency_ceiling": EMERGENCY,
        "validation50_calls": 0, "test50_calls": 0,
        "provider_profile": PROVIDER_PROFILE,
        "solver_model": SOLVER_MODEL, "solver_thinking": False,
        "reflection_model": ROLE_MODEL,
        "execution_prerequisite": "real_seed81_v4_diagnostic_scientifically_valid_then_separate_authorization",
    }
    if prerequisite is not None:
        protocol["diagnostic_prerequisite"] = prerequisite
        protocol["post_freeze_validation50"] = validation_policy
        protocol["preexecution_refreeze"] = manifest["preexecution_refreeze"]
        protocol["post_search_final_team"] = {
            "native": "replicate_first_returned_native_candidate_to_all_five_else_initial_team",
            "layer2": "preserve_final_committed_five_member_team",
            "validation50": "POST_FREEZE_VALIDATION50_EVALUATION_separate_authorization",
            "test50": "SEALED_ZERO_CALLS_STAGE0",
        }
    if attempt_number >= 3:
        protocol["execution_repair"] = manifest["execution_repair"]
        protocol["preexecution_hardening"] = manifest["preexecution_hardening"]
    if attempt_number == 4:
        protocol["execution_scheduling"] = manifest["execution_scheduling"]
    return manifest, protocol


def prepare(prep_root: Path, *, attempt_number: int = 1) -> dict[str, object]:
    if prep_root.exists():
        raise FileExistsError("fresh Formal V3 prep root required")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before freeze")
    source = _git("rev-parse", "HEAD")
    prep_root.mkdir(parents=True)
    cells = []
    for seed in FORMAL_SEEDS:
        for scope in ("native", "layer2"):
            manifest, protocol = frozen_payload(
                execution_source_sha=source, seed=seed, scope=scope,
                attempt_number=attempt_number,
            )
            prep = prep_root / manifest["attempt_id"]
            prep.mkdir()
            _private_splits(prep)
            for name, payload in (("manifest.json", manifest), ("protocol.json", protocol)):
                with (prep / name).open("xb") as handle:
                    handle.write(canonical_json_bytes(payload) + b"\n")
            write_bundle(prep / "startup_identity", _expected_bundle(
                root=ROOT, manifest=manifest, protocol=protocol,
                execution_source_sha=source, prep=prep,
            ))
            permit = validate_execution(root=ROOT, prep=prep, require_authorized=False)
            cells.append({
                "attempt_id": permit.attempt_id,
                "preregistration_sha256": permit.preregistration_sha256,
                "run_identity_sha256": permit.run_identity_sha256,
            })
    return {
        "status": "PREREGISTERED_EXECUTION_GATED",
        "successor_id": SUCCESSOR_ID, "execution_source_sha": source,
        "cells": cells, "provider_calls": 0,
        "validation50_calls": 0, "test50_calls": 0,
        "authorization_consumed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-root", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--attempt-number", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prep_root, attempt_number=args.attempt_number), sort_keys=True, indent=2))
