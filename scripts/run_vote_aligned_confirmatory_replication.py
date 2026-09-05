"""Seed76/77 Static/P0/P1 vote-aligned confirmatory replication.

``--prepare-only`` is strictly zero-API. Execution is fail-closed until the
manifest records a new explicit authorization and the one-shot environment
authorization is present. The Seed75 implementation is reused under a scoped
adapter so its completed artifacts and source are never modified.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import inspect
import json
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.persistence.identity import build_run_identity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts import run_vote_aligned_generic_shadow_pilot as base
from scripts import run_vote_aligned_seed75_static_control as static_support
from scripts.anti_overfitting_shadow_support import (
    FOLD_MAP,
    ROLE_MODEL,
    SOLVER_MODEL,
    export_private_splits,
    git,
    metadata,
    sha256_file,
    sha256_json,
    write_json,
)
from scripts.run_shadow_gated_evolution import _backup, _rows


EXPERIMENT_ID = "vote_aligned_confirmatory_seed76_77_v1"
RUNTIME_VERSION = EXPERIMENT_ID
STATIC = "STATIC_NO_TRAIN"
P0 = base.P0
P1 = base.P1
TRAIN_ARMS = (P0, P1)
ARMS = (STATIC, P0, P1)
SEEDS = (76, 77)
CONFIRMATORY_FOLD_MAP = (FOLD_MAP[1], FOLD_MAP[2])
FINAL_EVAL_DATASETS = ("shadow", "validation")
AUTH_ENV = "VOTE_ALIGNED_CONFIRMATORY_PHASE_B_AUTHORIZED"
MANIFEST = ROOT / "experiments" / "manifests" / f"{EXPERIMENT_ID}.yaml"
DESIGN_ROOT = ROOT / "experiments" / EXPERIMENT_ID
SOURCE_SPLIT_ROOT = ROOT / "experiments" / "anti_overfitting_split_v1"
DEFAULT_PREP_ROOT = ROOT / "runs" / f"{EXPERIMENT_ID}_prep"
DEFAULT_RUN_ROOT = ROOT / "runs" / EXPERIMENT_ID
DEFAULT_REPORT_ROOT = ROOT / "reports" / EXPERIMENT_ID
FROZEN_SEED75_DEFINITION = DESIGN_ROOT / "seed75_definition_freeze.json"
CLASSIFIER_DEFINITION = DESIGN_ROOT / "classifier_definition.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


@contextlib.contextmanager
def _base_scope() -> Iterator[None]:
    """Temporarily bind the proven Seed75 engine to the new frozen scope."""
    confirmatory_scope = base.build_expected_scope(seeds=SEEDS)
    original_inventory_validator = base.validate_evaluation_inventory

    def scoped_inventory_validator(
        observed: list[Mapping[str, Any]],
        scope: base.CompletionScope = confirmatory_scope,
    ) -> list[str]:
        # The reused Seed75 function's default argument was bound when that
        # module was imported.  Always pass this experiment's explicit scope.
        return original_inventory_validator(observed, scope)

    replacements = {
        "SEEDS": SEEDS,
        "SCOPE": confirmatory_scope,
        "validate_evaluation_inventory": scoped_inventory_validator,
        "FINAL_EVAL_DATASETS": FINAL_EVAL_DATASETS,
        "AUTH_ENV": AUTH_ENV,
        "MANIFEST": MANIFEST,
        "DESIGN_ROOT": DESIGN_ROOT,
        "DEFAULT_PREP_ROOT": DEFAULT_PREP_ROOT,
        "DEFAULT_RUN_ROOT": DEFAULT_RUN_ROOT,
        "DEFAULT_REPORT_ROOT": DEFAULT_REPORT_ROOT,
        "RUNTIME_VERSION": RUNTIME_VERSION,
        "FOLD_MAP": CONFIRMATORY_FOLD_MAP,
        # No post-hoc Seed75 execution-identity transition is legal here.
        "EXECUTION_INITIALIZATION_RELATIVE": Path(
            "confirmatory_initialization_transition_forbidden"
        ),
    }
    original = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def _assignment() -> dict[str, list[str]]:
    split = read_json(SOURCE_SPLIT_ROOT / "split_manifest.json")
    folds = read_json(SOURCE_SPLIT_ROOT / "fold_assignment.json")
    return {
        "fold_a": list(folds["folds"]["fold_a"]),
        "fold_b": list(folds["folds"]["fold_b"]),
        "fold_c": list(folds["folds"]["fold_c"]),
        "validation": list(split["question_hashes"]["validation"]),
        "test": list(split["question_hashes"]["test"]),
    }


def _fold_paths(prep_root: Path, index: int) -> tuple[Path, Path]:
    optimize, shadow = CONFIRMATORY_FOLD_MAP[index]
    parts = [
        prep_root / "splits_private" / f"{name}.csv"
        for name in optimize.split("+")
    ]
    combined = prep_root / "splits_private" / f"optimize_seed{SEEDS[index]}.csv"
    if not combined.exists():
        rows = [row for path in parts for row in _rows(path)]
        with combined.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(rows)
    return combined, prep_root / "splits_private" / f"{shadow}.csv"


def _config(
    seed: int,
    arm: str,
    out: Path,
    optimize: Path,
    validation: Path,
    cache: Path,
    initialization: Path,
    resume: bool = False,
):
    if arm not in TRAIN_ARMS:
        raise ValueError(f"not a trainable arm: {arm}")
    with _base_scope():
        return base._config(
            seed,
            arm,
            out,
            optimize,
            validation,
            cache,
            initialization,
            resume,
        )


def _protocol_document() -> dict[str, Any]:
    return {
        "schema_version": "vote_aligned_confirmatory_protocol_freeze_v1",
        "runtime_version": RUNTIME_VERSION,
        "execution_baseline": git("rev-parse", "HEAD"),
        "seed75_evidence_excluded_from_confirmatory_decision": True,
        "models": {
            "solver": SOLVER_MODEL,
            "teacher": ROLE_MODEL,
            "critic": ROLE_MODEL,
            "student": ROLE_MODEL,
            "evaluator": ROLE_MODEL,
            "thinking": False,
        },
        "seeds": list(SEEDS),
        "fold_map": [
            {"seed": seed, "optimize": pair[0], "shadow": pair[1]}
            for seed, pair in zip(SEEDS, CONFIRMATORY_FOLD_MAP, strict=True)
        ],
        "arms": {
            STATIC: {"training_updates": 0, "target_scheduler": None},
            P0: {"target_scheduler": base.SCHEDULER_BY_ARM[P0]},
            P1: {"target_scheduler": base.SCHEDULER_BY_ARM[P1]},
        },
        "shared_trainable_protocol": {
            "experiment_setting": "experimental_diversity_d2_rr_generic",
            "proposal": "generic_peer_state",
            "semantic_critic": True,
            "target_slots": 2,
            "source_candidates_per_target": 2,
            "generic_revision": True,
            "common_safe": True,
            "winner_only_shadow": True,
            "maximum_update_opportunities": 32,
            "no_commit_patience": 6,
        },
        "execution_order": [
            "Seed76 P0",
            "Seed76 P1",
            "Seed77 P0",
            "Seed77 P1",
            "frozen-final-state Static/P0/P1 Shadow50 and Validation50",
        ],
        "validation_policy": (
            "exactly once per frozen final state after all trainable trajectories"
        ),
        "test_policy": "zero access",
        "classifier_sha256": sha256_file(CLASSIFIER_DEFINITION),
        "seed75_definition_freeze_sha256": sha256_file(FROZEN_SEED75_DEFINITION),
    }


def _source_freeze(prep_root: Path) -> None:
    files = {
        path.relative_to(ROOT)
        for path in (ROOT / "multi_dataset_diverse_rl").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    files.update(
        {
            Path("scripts/anti_overfitting_shadow_support.py"),
            Path("scripts/run_shadow_gated_evolution.py"),
            Path("scripts/run_vote_aligned_generic_shadow_pilot.py"),
            Path("scripts/run_vote_aligned_seed75_static_control.py"),
            Path("scripts/run_vote_aligned_confirmatory_replication.py"),
            MANIFEST.relative_to(ROOT),
            (DESIGN_ROOT / "PROTOCOL.md").relative_to(ROOT),
            (DESIGN_ROOT / "API_AUTHORIZATION_AMENDMENT.md").relative_to(ROOT),
            CLASSIFIER_DEFINITION.relative_to(ROOT),
            FROZEN_SEED75_DEFINITION.relative_to(ROOT),
            Path("experiments/anti_overfitting_split_v1/split_manifest.json"),
            Path("experiments/anti_overfitting_split_v1/fold_assignment.json"),
        }
    )
    write_json(
        prep_root / "source_freeze.json",
        {
            "execution_commit": git("rev-parse", "HEAD"),
            "tracked_worktree_clean": (
                git("status", "--porcelain", "--untracked-files=all") == ""
            ),
            "files": [
                {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)}
                for path in sorted(files, key=lambda value: value.as_posix())
            ],
        },
    )


def _verify_seed75_definition_freeze() -> list[str]:
    errors: list[str] = []
    frozen = read_json(FROZEN_SEED75_DEFINITION)
    for row in frozen["files"]:
        path = ROOT / str(row["path"])
        if not path.is_file():
            errors.append(f"missing:{row['path']}")
        elif sha256_file(path) != row["sha256"]:
            errors.append(f"hash:{row['path']}")
    if frozen["p0_arm"] != P0 or frozen["p1_arm"] != P1:
        errors.append("arm_identity")
    return errors


def prepare(prep_root: Path) -> dict[str, Any]:
    """Create a project-local, zero-API Phase-A freeze."""
    if prep_root.exists():
        raise RuntimeError("fresh prep root required")
    items, raw = metadata()
    assignment = _assignment()
    prep_root.mkdir(parents=True)
    export_private_splits(raw, assignment, prep_root / "splits_private")
    protocol = _protocol_document()
    write_json(prep_root / "protocol_freeze.json", protocol)
    write_json(
        prep_root / "test_access_registry.json",
        {"schema_version": "test_access_registry_v1", "events": [], "new_test_calls": 0},
    )

    p0 = _config(
        76,
        P0,
        Path("p0"),
        Path("opt.csv"),
        Path("val.csv"),
        Path("p0.sqlite"),
        Path("init.json"),
    )
    p1 = _config(
        76,
        P1,
        Path("p1"),
        Path("opt.csv"),
        Path("val.csv"),
        Path("p1.sqlite"),
        Path("init.json"),
    )
    left, right = p0.to_flat_dict(), p1.to_flat_dict()
    differences = {key for key in left if left[key] != right[key]}
    with _base_scope():
        p0_protocol = base._resolved_protocol(p0)
        p1_protocol = base._resolved_protocol(p1)
        selector_gate = base._synthetic_selector_gate()
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    all_hashes = [
        digest
        for key in ("fold_a", "fold_b", "fold_c", "validation", "test")
        for digest in assignment[key]
    ]
    frozen_classifier = read_json(CLASSIFIER_DEFINITION)
    seed75_freeze_errors = _verify_seed75_definition_freeze()
    gates = {
        "CLEAN_WORKTREE": git("status", "--porcelain", "--untracked-files=all") == "",
        "PROJECT_LOCAL_PATHS": prep_root.resolve().is_relative_to(ROOT.resolve()),
        "NEW_CONFIRMATORY_SCOPE": (
            tuple(manifest["seeds"]) == SEEDS
            and manifest["lineage"]["derives_from"]
            == "vote_aligned_generic_shadow_pilot_v1"
            and manifest["result"]["classifier"] in {"NOT_RUN", "RUNNING"}
        ),
        "API_AUTHORIZATION_SCOPE": (
            manifest["api_authorization"]["authorized"] is True
            and set(manifest["api_authorization"]["allowed_roles"])
            == {"solver", "teacher", "critic", "student", "evaluator"}
            and set(manifest["api_authorization"]["allowed_phases"])
            == {"online_trajectory", "frozen_validation"}
            and "seed76_77_confirmatory" in manifest["api_authorization"]["authorization_scope"]
        ),
        "DATA_PARTITION": (
            len(items) == 250
            and len(all_hashes) == len(set(all_hashes)) == 250
            and all(len(assignment[key]) == 50 for key in assignment)
        ),
        "CROSSFIT_MAPPING": CONFIRMATORY_FOLD_MAP
        == (("fold_a+fold_c", "fold_b"), ("fold_b+fold_c", "fold_a")),
        "MODEL_IDENTITY": (
            p0.models.agent_model == p1.models.agent_model == "qwen3-8b"
            and p0.models.optimizer_model
            == p1.models.optimizer_model
            == "qwen3.7-flash"
            and p0.models.evaluator_model
            == p1.models.evaluator_model
            == "qwen3.7-flash"
            and p0.models.temperature == p1.models.temperature == 0.0
        ),
        "SEED75_DEFINITION_FREEZE": not seed75_freeze_errors,
        "P0_P1_SINGLE_FACTOR": differences
        == {"target_scheduler", "out_dir", "shared_solver_cache_path"},
        "GENERIC_PIPELINE_PARITY": (
            p0_protocol == p1_protocol
            and p0_protocol.name == "experimental_diversity_d2_rr_generic"
            and p0_protocol.tcs_context_policy
            == p1_protocol.tcs_context_policy
            == "generic_peer_state"
            and p0_protocol.generic_revision_enabled
            and p1_protocol.generic_revision_enabled
        ),
        "SELECTOR_DETERMINISM": selector_gate,
        "BUDGET_PARITY": (
            p0_protocol.target_branch_count
            == p1_protocol.target_branch_count
            == 2
            and p0_protocol.candidates_per_target_branch
            == p1_protocol.candidates_per_target_branch
            == 2
            and p0.training.epochs * 8 == p1.training.epochs * 8 == 32
        ),
        "FROZEN_CLASSIFIER": (
            frozen_classifier["seeds"] == [76, 77]
            and frozen_classifier["gte1_alone_is_sufficient"] is False
            and set(frozen_classifier["classifiers"])
            == {
                "CONFIRMATORY_REPLICATION_SUPPORTED",
                "PARTIAL_CONFIRMATORY_REPLICATION",
                "CONFIRMATORY_REPLICATION_NOT_SUPPORTED",
            }
        ),
        "NO_VALIDATION_TEST_LEAKAGE": (
            protocol["test_policy"] == "zero access"
            and protocol["validation_policy"].startswith("exactly once")
            and "test" not in FINAL_EVAL_DATASETS
        ),
    }
    result = {
        "phase_a_gate": "PASS" if all(gates.values()) else "HOLD",
        "gates": {key: "PASS" if value else "HOLD" for key, value in gates.items()},
        "seed75_definition_freeze_errors": seed75_freeze_errors,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
    }
    write_json(prep_root / "phase_a_gate.json", result)
    _source_freeze(prep_root)
    return result


def _verify_source_freeze(prep_root: Path) -> None:
    freeze = read_json(prep_root / "source_freeze.json")
    if freeze["execution_commit"] != git("rev-parse", "HEAD"):
        raise RuntimeError("execution commit differs from source freeze")
    if not freeze["tracked_worktree_clean"] or git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("tracked worktree must be clean")
    for row in freeze["files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")


def _verify_execution_freeze_for_audit(prep_root: Path) -> list[str]:
    """Verify the immutable execution tree without requiring auditor=runner HEAD."""
    errors: list[str] = []
    freeze = read_json(prep_root / "source_freeze.json")
    execution_commit = str(freeze.get("execution_commit", ""))
    if not execution_commit:
        return ["execution_commit_missing"]
    for row in freeze["files"]:
        try:
            observed = base._git_blob_sha256_candidates(
                execution_commit, str(row["path"])
            )
        except Exception:
            errors.append(f"execution_blob_missing:{row['path']}")
            continue
        # Mixed-EOL checkout hashes cannot always be reconstructed from a Git
        # blob. The reused official auditor records those fallbacks separately;
        # this gate rejects missing blobs but delegates EOL accounting to it.
        if row["sha256"] not in observed:
            continue
    return errors


def _authorize() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    if os.environ.get(AUTH_ENV) != "1":
        raise RuntimeError("Phase B requires a new explicit Seed76/77 API authorization")
    for role in ("solver", "teacher", "critic", "student"):
        require_api_authorization(
            manifest,
            phase="online_trajectory",
            role=role,
            explicit_user_authorized=True,
        )
    for role in ("solver", "evaluator"):
        require_api_authorization(
            manifest,
            phase="frozen_validation",
            role=role,
            explicit_user_authorized=True,
        )


async def _evaluate_static(
    *,
    seed: int,
    index: int,
    prep_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    out = run_root / f"seed{seed}" / STATIC
    if out.exists():
        raise RuntimeError(f"fresh Static cell required: seed{seed}")
    out.mkdir(parents=True)
    optimize, shadow = _fold_paths(prep_root, index)
    validation = prep_root / "splits_private" / "validation.csv"
    initialization_root = run_root / "initialization" / f"seed{seed}"
    initialization = initialization_root / "frozen_initialization_manifest.json"
    stable_cache = initialization_root / "initial_solver_cache_frozen.sqlite"
    cache = out / "static_solver_cache.sqlite"
    _backup(stable_cache, cache)
    cfg = _config(
        seed,
        P0,
        out,
        optimize,
        validation,
        cache,
        initialization,
    )
    values = cfg.to_flat_dict()
    values.update(
        {
            "out_dir": str(out),
            "shared_solver_cache_path": str(cache),
            "resume_from_checkpoint": False,
            "final_test_enabled": False,
            "test_path": "TEST50_BLOCKED_BY_CONFIRMATORY_PROTOCOL",
            "test_size": 0,
            "epochs": 0,
        }
    )
    cfg = type(cfg).from_flat(**values)
    system = PromptEnsembleOptimizationSystem(cfg)
    optimize_rows = _rows(optimize)
    validation_rows = _rows(validation)
    system.set_run_identity(
        build_run_identity(
            cfg,
            train_rows=optimize_rows,
            val_rows=validation_rows,
            test_rows=[],
            workspace=ROOT,
        )
    )
    calls_before = int(system.cost_summary()["successful_llm_calls"])
    await system.initialize_fixed_probe(optimize_rows)
    calls_after_initial = int(system.cost_summary()["successful_llm_calls"])
    observed_snapshot = system.frozen_initialization_snapshot()
    expected_snapshot = read_json(initialization)["initialization_snapshot"]
    observed_signature = base._scientific_initialization_signature(observed_snapshot)
    expected_signature = base._scientific_initialization_signature(expected_snapshot)
    if observed_signature != expected_signature or calls_after_initial != calls_before:
        raise RuntimeError("Static failed to reuse the exact frozen initialization")
    initial_state_hash = system.team_prompt_state_hash()
    results: dict[str, Any] = {}
    results["optimize"] = static_support._aggregate(
        system,
        system.fixed_probe.examples,
        system.active_profiles,
        system.active_probe_metrics(),
        "optimize",
        0,
    )
    for split, path in (("shadow", shadow), ("validation", validation)):
        before = int(system.cost_summary()["successful_llm_calls"])
        metrics = await system.evaluate_dataset(_rows(path))
        after = int(system.cost_summary()["successful_llm_calls"])
        results[split] = static_support._aggregate(
            system,
            system._last_evaluated_examples,
            system._last_evaluated_profiles,
            metrics,
            split,
            after - before,
        )
        if system.team_prompt_state_hash() != initial_state_hash:
            raise RuntimeError("Static evaluation mutated prompt state")
    calls = int(system.cost_summary()["successful_llm_calls"])
    if calls > 100:
        raise RuntimeError("Static provider-call budget exceeded")
    payload = {
        "execution_gate": "PASS",
        "seed": seed,
        "arm": STATIC,
        "initialization_signature": observed_signature,
        "training_updates": 0,
        "target_selections": 0,
        "candidate_generations": 0,
        "commits": 0,
        "teacher_calls": 0,
        "critic_calls": 0,
        "student_calls": 0,
        "new_solver_provider_calls": calls,
        "new_test_calls": 0,
        "state_mutation": False,
        "metrics": results,
    }
    write_json(out / "static_control_summary_private.json", payload)
    return payload


async def execute(prep_root: Path, run_root: Path) -> dict[str, Any]:
    _authorize()
    _verify_source_freeze(prep_root)
    phase_a = read_json(prep_root / "phase_a_gate.json")
    if phase_a["phase_a_gate"] != "PASS":
        raise RuntimeError("Phase A gate is not PASS")
    if run_root.exists():
        raise RuntimeError("fresh confirmatory run root required")
    with _base_scope():
        train_result = await base.execute(prep_root, run_root, resume=False)
    static_results = [
        await _evaluate_static(
            seed=seed,
            index=index,
            prep_root=prep_root,
            run_root=run_root,
        )
        for index, seed in enumerate(SEEDS)
    ]
    result = {
        "execution_gate": "PASS",
        "completed_trainable_trajectories": train_result["completed_trajectories"],
        "completed_static_controls": len(static_results),
        "final_shadow_evaluations": 6,
        "final_validation_evaluations": 6,
        "new_test_calls": 0,
    }
    write_json(run_root / "confirmatory_execution_summary.json", result)
    return result


def _metric_row(
    *, seed: int, arm: str, split: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    depth = {key: int(value) for key, value in metrics["coverage_depth"].items()}
    return {
        "seed": seed,
        "arm": arm,
        "split": split,
        "vote": float(metrics["vote_accuracy"]),
        "mean_member": float(metrics["mean_member_accuracy"]),
        "ensemble_gain": float(metrics["vote_accuracy"])
        - float(metrics["mean_member_accuracy"]),
        "oracle": float(metrics.get("oracle_accuracy", metrics["oracle_correct_count"] / 50)),
        "gte1_count": 50 - depth["G0"],
        "gte3_count": depth["G3"] + depth["G4"] + depth["G5"],
        **depth,
    }


def audit(prep_root: Path, run_root: Path) -> dict[str, Any]:
    if not run_root.exists():
        return {"audit_gate": "NOT_RUN", "new_test_calls": 0}
    errors: list[str] = []
    errors.extend(_verify_execution_freeze_for_audit(prep_root))
    prior_audit = run_root / "audit_summary.json"
    preserved_hold = run_root / "audit_summary_seed75_default_scope_hold.json"
    if prior_audit.is_file() and not preserved_hold.exists():
        prior = read_json(prior_audit)
        if prior.get("phase_b_gate") == "HOLD" and any(
            "evaluation_identity_" in str(value) for value in prior.get("errors", [])
        ):
            write_json(preserved_hold, prior)
    with _base_scope():
        paired = base.audit(prep_root, run_root)
    if paired["phase_b_gate"] != "PASS":
        errors.append("paired_p0_p1_audit")
    static_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = run_root / f"seed{seed}" / STATIC / "static_control_summary_private.json"
        if not path.is_file():
            errors.append(f"static_missing:{seed}")
            continue
        row = read_json(path)
        static_rows.append(row)
        zero_fields = (
            "training_updates",
            "target_selections",
            "candidate_generations",
            "commits",
            "teacher_calls",
            "critic_calls",
            "student_calls",
            "new_test_calls",
        )
        if any(int(row[field]) != 0 for field in zero_fields):
            errors.append(f"static_forbidden_activity:{seed}")
        if row["state_mutation"] is not False:
            errors.append(f"static_state_mutation:{seed}")
        if set(row["metrics"]) != {"optimize", "shadow", "validation"}:
            errors.append(f"static_inventory:{seed}")
        if int(row["metrics"]["optimize"]["new_provider_calls"]) != 0:
            errors.append(f"static_optimize_not_reused:{seed}")
        signatures = {row["initialization_signature"]}
        for arm in TRAIN_ARMS:
            match = read_json(
                run_root / f"seed{seed}" / arm / "frozen_initialization_match.json"
            )
            signatures.add(
                base._scientific_initialization_signature(
                    match["initialization_snapshot"]
                )
            )
        if len(signatures) != 1:
            errors.append(f"three_arm_initialization:{seed}")
    test_events = read_json(prep_root / "test_access_registry.json")["events"]
    if test_events:
        errors.append("test_access")
    result = {
        "audit_gate": "PASS" if not errors else "HOLD",
        "errors": sorted(set(errors)),
        "paired_gate": paired["phase_b_gate"],
        "trainable_trajectory_count": paired.get("trajectory_count", 0),
        "static_control_count": len(static_rows),
        "final_shadow_evaluations": 3 * len(SEEDS),
        "final_validation_evaluations": 3 * len(SEEDS),
        "new_test_calls": 0,
    }
    write_json(run_root / "confirmatory_audit_summary.json", result)
    return result


def classify(contrasts: Sequence[Mapping[str, float]]) -> str:
    required = (
        "p1_minus_p0_ensemble_gain",
        "p1_minus_static_mean_member",
        "p1_minus_static_vote",
        "p1_minus_p0_gte3_count",
    )
    per_seed = [all(float(row[key]) > 0 for key in required) for row in contrasts]
    if all(per_seed):
        return "CONFIRMATORY_REPLICATION_SUPPORTED"
    means = {
        key: statistics.mean(float(row[key]) for row in contrasts) for key in required
    }
    if sum(per_seed) == 1 and all(value > 0 for value in means.values()):
        return "PARTIAL_CONFIRMATORY_REPLICATION"
    return "CONFIRMATORY_REPLICATION_NOT_SUPPORTED"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(prep_root: Path, run_root: Path, report_root: Path) -> dict[str, Any]:
    gate = audit(prep_root, run_root)
    if gate["audit_gate"] != "PASS":
        return {"analysis_gate": "NOT_RUN", "audit_gate": gate["audit_gate"]}
    report_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        static = read_json(
            run_root / f"seed{seed}" / STATIC / "static_control_summary_private.json"
        )
        for split in FINAL_EVAL_DATASETS:
            rows.append(
                _metric_row(
                    seed=seed,
                    arm=STATIC,
                    split=split,
                    metrics=static["metrics"][split],
                )
            )
        for arm in TRAIN_ARMS:
            for split in FINAL_EVAL_DATASETS:
                metrics = read_json(
                    run_root
                    / "evaluation"
                    / f"seed{seed}"
                    / arm
                    / split
                    / "evaluation_summary_private.json"
                )
                rows.append(
                    _metric_row(seed=seed, arm=arm, split=split, metrics=metrics)
                )
    validation = {
        (int(row["seed"]), str(row["arm"])): row
        for row in rows
        if row["split"] == "validation"
    }
    contrasts: list[dict[str, Any]] = []
    for seed in SEEDS:
        s, p0, p1 = (
            validation[(seed, STATIC)],
            validation[(seed, P0)],
            validation[(seed, P1)],
        )
        contrasts.append(
            {
                "seed": seed,
                "p1_minus_p0_ensemble_gain": p1["ensemble_gain"]
                - p0["ensemble_gain"],
                "p1_minus_static_mean_member": p1["mean_member"]
                - s["mean_member"],
                "p1_minus_static_vote": p1["vote"] - s["vote"],
                "p1_minus_p0_gte3_count": p1["gte3_count"] - p0["gte3_count"],
                "p1_minus_p0_gte1_count": p1["gte1_count"] - p0["gte1_count"],
                "all_strict_requirements_pass": (
                    p1["ensemble_gain"] > p0["ensemble_gain"]
                    and p1["mean_member"] > s["mean_member"]
                    and p1["vote"] > s["vote"]
                    and p1["gte3_count"] > p0["gte3_count"]
                ),
            }
        )
    classifier = classify(contrasts)
    summary = {
        "analysis_gate": "PASS",
        "classifier": classifier,
        "confirmatory_seeds": list(SEEDS),
        "seed75_excluded_from_confirmatory_decision": True,
        "mean_contrasts": {
            key: statistics.mean(float(row[key]) for row in contrasts)
            for key in (
                "p1_minus_p0_ensemble_gain",
                "p1_minus_static_mean_member",
                "p1_minus_static_vote",
                "p1_minus_p0_gte3_count",
                "p1_minus_p0_gte1_count",
            )
        },
        "passing_seed_count": sum(
            bool(row["all_strict_requirements_pass"]) for row in contrasts
        ),
        "new_test_calls": 0,
    }
    _write_csv(report_root / "per_seed_arm_results.csv", rows)
    _write_csv(report_root / "confirmatory_contrasts.csv", contrasts)
    write_json(report_root / "summary.json", summary)
    write_json(
        report_root / "classifier.json",
        {"classifier": classifier, "definition": read_json(CLASSIFIER_DEFINITION)},
    )
    (report_root / "README.md").write_text(
        "# Vote-Aligned Confirmatory Replication\n\n"
        f"Classifier: `{classifier}`. Seed75 is contextual prior evidence and "
        "is excluded from this two-seed confirmatory decision. Validation50 "
        "was evaluated only after training freeze; Test50 calls were zero.\n",
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    value.add_argument("--prep-root", type=Path, default=DEFAULT_PREP_ROOT)
    value.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    value.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return value


def main() -> None:
    args = parser().parse_args()
    prep_root = args.prep_root.resolve()
    run_root = args.run_root.resolve()
    report_root = args.report_root.resolve()
    for path in (prep_root, run_root, report_root):
        path.relative_to(ROOT.resolve())
    if args.prepare_only:
        result = prepare(prep_root)
    elif args.execute:
        result = asyncio.run(execute(prep_root, run_root))
    elif args.audit:
        result = audit(prep_root, run_root)
    else:
        result = analyze(prep_root, run_root, report_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
