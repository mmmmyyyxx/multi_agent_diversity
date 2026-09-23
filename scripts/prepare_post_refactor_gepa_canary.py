"""Zero-API preparation of one post-refactor GEPA/Layer2 canary.

Run from the exact clean execution-source commit. The fresh ignored prep root
contains private Optimize/Shadow rows; tracked reports contain hashes only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import (  # noqa: E402
    verify_frozen_gepa,
)

EXPERIMENT_ID = "gepa_layer2_real_canary_post_refactor_v2"
SEED = 80
SCIENTIFIC_METHOD_ANCHOR_SHA = "f762545e3e4c49a7ba9f8cac53b380f14c606332"
DEFAULT_PREP = ROOT / "runs" / EXPERIMENT_ID / "prep"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8",
    ).strip()


def _write_new(path: Path, value: dict) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def _private_splits(prep: Path) -> None:
    fold = json.loads((ROOT / "experiments/anti_overfitting_split_v1/fold_assignment.json").read_text(encoding="utf-8"))["folds"]
    wanted = {digest for name in ("fold_a", "fold_b", "fold_c") for digest in fold[name]}
    source = ROOT / "strict_splits_bbh_seed42/disambiguation_qa"
    by_hash: dict[str, dict[str, str]] = {}
    for filename in ("opt.csv", "val.csv", "test.csv"):
        with (source / filename).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                digest = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()
                if digest in wanted:
                    by_hash[digest] = {"question": row["question"], "answer": row["answer"]}
    if len(by_hash) != 150 or any(len(fold[name]) != 50 for name in ("fold_a", "fold_b", "fold_c")):
        raise ValueError("frozen 50/50/50 fold materialization is incomplete")
    splits = {
        "optimize100.csv": [*fold["fold_a"], *fold["fold_b"]],
        "shadow50.csv": list(fold["fold_c"]),
    }
    private = prep / "splits_private"
    private.mkdir()
    for filename, ids in splits.items():
        with (private / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(by_hash[digest] for digest in ids)


def source_paths() -> list[str]:
    paths = list((ROOT / "multi_dataset_diverse_rl").rglob("*.py"))
    paths += list((ROOT / "infrastructure/common_solver_contract_v1").rglob("*.py"))
    paths += [
        ROOT / "scripts/run_experiment.py",
        ROOT / "scripts/prepare_post_refactor_gepa_canary.py",
        ROOT / "experiments/anti_overfitting_split_v1/fold_assignment.json",
    ]
    return sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in paths)


def frozen_payload(*, execution_source_sha: str) -> tuple[dict, dict]:
    scientific = {
        "backend": "gepa", "optimization_scope": "layer2",
        "stopping_regime": "fixed_budget",
        "task_identity": "BBH_disambiguation_qa",
        "data_identity": "anti_overfitting_split_v1_fold_a+b_to_c",
        "fixed_budget_units": 1,
        "local_no_update_patience": 3,
        "team_no_update_patience": 2,
    }
    spec = experiment_spec_from_mapping(scientific)
    manifest = {
        "schema_version": "unified_production_canary_freeze_v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": EXPERIMENT_ID,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "method_identity": spec.method_identity,
        "spec_identity": spec.identity(),
        "scientific": scientific,
        "runtime": {
            "seed": SEED, "provider_profile": "lwj",
            "endpoint_fingerprint": endpoint_fingerprint_from_environment(),
            "solver_model": "qwen3-8b",
            "optimizer_model": "qwen3.7-flash",
            "evaluator_model": "qwen3.7-flash",
        },
        "models": {
            "solver": {"model": "qwen3-8b", "thinking": False},
            "reflection": {"model": "qwen3.7-flash"},
        },
        "dependency": {"gepa": verify_frozen_gepa()},
        "execution": {
            "scientific_method_anchor_sha": SCIENTIFIC_METHOD_ANCHOR_SHA,
            "execution_source_sha": execution_source_sha,
            "initialization_policy": INITIALIZATION_POLICY,
            "source_paths": source_paths(),
        },
        "api_authorization": {
            "authorized": False,
            "authorization_state": "AUTHORIZATION_REQUIRED",
            "allowed_roles": ["solver", "reflection"],
            "allowed_phases": ["canary"],
        },
        "access": {"validation50_calls": 0, "test50_calls": 0},
    }
    protocol = {
        "schema_version": "post_refactor_gepa_layer2_canary_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "purpose": "technical_local_empirical_path_only",
        "initialization_policy": INITIALIZATION_POLICY,
        "optimize_rows": 100, "shadow_rows": 50,
        "target_selection": "current_primary_responsibility_deterministic_top1",
        "local_metric_call_budget": 36,
        "team_opportunities": 1,
        "validation50_calls": 0, "test50_calls": 0,
        "provider_profile": "lwj",
        "solver_model": "qwen3-8b", "solver_thinking": False,
        "reflection_model": "qwen3.7-flash",
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
    _write_new(prep / "manifest.json", manifest)
    _write_new(prep / "protocol.json", protocol)
    bundle = _expected_bundle(
        root=ROOT, manifest=manifest, protocol=protocol,
        execution_source_sha=source, prep=prep,
    )
    write_bundle(prep / "startup_identity", bundle)
    permit = validate_execution(root=ROOT, prep=prep, require_authorized=False)
    return {
        "attempt_id": permit.attempt_id,
        "execution_source_sha": source,
        "preregistration_sha256": permit.preregistration_sha256,
        "run_identity_sha256": permit.run_identity_sha256,
        "ready_for_authorization": True,
        "provider_boundary_reached": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prep), sort_keys=True, indent=2))
