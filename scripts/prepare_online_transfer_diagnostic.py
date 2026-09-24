"""Zero-API freeze for the Seed81 online local-to-team transfer diagnostic."""

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
from multi_dataset_diverse_rl.governance.execution_harness_v2 import (  # noqa: E402
    INITIALIZATION_POLICY, endpoint_fingerprint_from_environment,
)
from multi_dataset_diverse_rl.governance.production_execution import (  # noqa: E402
    _expected_bundle, validate_execution,
)
from multi_dataset_diverse_rl.governance.startup_identity import (  # noqa: E402
    canonical_json_bytes, write_bundle,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import (  # noqa: E402
    local_gepa_budget_capacity,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import verify_frozen_gepa  # noqa: E402
from scripts.prepare_post_refactor_gepa_canary import _private_splits, source_paths  # noqa: E402

EXPERIMENT_ID = "gepa_layer2_local_to_team_transfer_diagnostic_v1"
SEED = 81
METHOD_ANCHOR_SHA = "3b61caafae810dfed072721b8bf949a97b4c3aeb"
DEFAULT_PREP = ROOT / "runs" / EXPERIMENT_ID / "prep"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8",
    ).strip()


def frozen_payload(*, execution_source_sha: str) -> tuple[dict, dict]:
    capacity = local_gepa_budget_capacity(
        metric_budget=36, validation_size=12, reflection_minibatch_size=3,
    )
    if capacity.max_accepted_children != 1 or capacity.max_rejected_proposals != 4:
        raise ValueError("GEPA local capacity no longer matches diagnostic stop guard")
    scientific = {
        "backend": "gepa", "optimization_scope": "layer2",
        "stopping_regime": "fixed_budget", "task_identity": "BBH_disambiguation_qa",
        "data_identity": "anti_overfitting_split_v1_fold_a+b_to_c",
        "fixed_budget_units": 10,
        "local_no_update_patience": 3, "team_no_update_patience": 2,
    }
    spec = experiment_spec_from_mapping(scientific)
    diagnostic_contract = {
        "accepted_mutation_target": 5,
        "max_opportunities": 10,
        "reflection_proposal_ceiling": 20,
        "successful_provider_ceiling": 1200,
        "transport_attempt_ceiling": 4800,
        "mandatory_full": True,
        "diagnostic_full_is_admission_inert": True,
    }
    sources = [
        *source_paths(), "scripts/prepare_online_transfer_diagnostic.py",
        "scripts/audit_online_transfer_diagnostic.py",
        "experiments/gepa_layer2_local_to_team_transfer_diagnostic_v1/PROTOCOL.md",
    ]
    manifest = {
        "schema_version": "online_transfer_diagnostic_freeze_v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": EXPERIMENT_ID,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "method_identity": spec.method_identity,
        "spec_identity": spec.identity(),
        "scientific": scientific,
        "diagnostic_contract": diagnostic_contract,
        "runtime": {
            "seed": SEED, "provider_profile": "lwj",
            "endpoint_fingerprint": endpoint_fingerprint_from_environment(),
            "solver_model": "qwen3-8b", "optimizer_model": "qwen3.7-flash",
            "evaluator_model": "qwen3.7-flash",
        },
        "models": {
            "solver": {"model": "qwen3-8b", "thinking": False},
            "reflection": {"model": "qwen3.7-flash"},
        },
        "dependency": {"gepa": verify_frozen_gepa()},
        "execution": {
            "scientific_method_anchor_sha": METHOD_ANCHOR_SHA,
            "execution_source_sha": execution_source_sha,
            "initialization_policy": INITIALIZATION_POLICY,
            "source_paths": sorted(set(sources)),
        },
        "api_authorization": {
            "authorized": False,
            "authorization_state": "AUTHORIZATION_REQUIRED",
            "allowed_roles": ["solver", "reflection"],
            "allowed_phases": ["diagnostic"],
        },
        "access": {"validation50_calls": 0, "test50_calls": 0},
    }
    protocol = {
        "schema_version": "online_local_to_team_transfer_diagnostic_v1",
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "initialization_policy": INITIALIZATION_POLICY,
        "optimize_rows": 100, "shadow_rows": 50,
        "parent_policy": "actual_current_committed_state_before_each_opportunity",
        "target_policy": "unchanged_primary_responsibility_persistent_realizability_top1",
        "local_metric_call_budget_per_opportunity": 36,
        "team_minibatch_rows": 12,
        "full_rows_per_accepted_mutation": 100,
        "diagnostic_full_policy": "mandatory_all_local_accepts_admission_inert",
        "ordinary_shadow_policy": "winner_only_after_common_safe",
        "proposal_ceiling_guard": "stop_before_next_complete_opportunity_if_4_proposals_could_overshoot",
        **diagnostic_contract,
        "validation50_calls": 0, "test50_calls": 0,
        "provider_profile": "lwj", "solver_model": "qwen3-8b",
        "solver_thinking": False, "reflection_model": "qwen3.7-flash",
        "scientific_method_change": False,
    }
    return manifest, protocol


def prepare(prep: Path) -> dict[str, str | bool]:
    if prep.exists():
        raise FileExistsError("fresh prep root required")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before freeze")
    source = _git("rev-parse", "HEAD")
    manifest, protocol = frozen_payload(execution_source_sha=source)
    prep.mkdir(parents=True)
    _private_splits(prep)
    for name, payload in (("manifest.json", manifest), ("protocol.json", protocol)):
        with (prep / name).open("xb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
    bundle = _expected_bundle(
        root=ROOT, manifest=manifest, protocol=protocol,
        execution_source_sha=source, prep=prep,
    )
    write_bundle(prep / "startup_identity", bundle)
    permit = validate_execution(root=ROOT, prep=prep, require_authorized=False)
    return {
        "status": "READY_FOR_AUTHORIZATION",
        "attempt_id": permit.attempt_id,
        "execution_source_sha": source,
        "preregistration_sha256": permit.preregistration_sha256,
        "run_identity_sha256": permit.run_identity_sha256,
        "provider_boundary_reached": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.prep), sort_keys=True, indent=2))
