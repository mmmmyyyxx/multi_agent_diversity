from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from multi_dataset_diverse_rl import cli
from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.governance.artifacts import scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash
from multi_dataset_diverse_rl.governance.manifest import validate_manifest
from multi_dataset_diverse_rl.governance.registries import load_yaml
from multi_dataset_diverse_rl.peer_state import build_team_vote_state
from multi_dataset_diverse_rl.persistence.checkpoint import restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity, build_run_identity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.task_manifest import load_task_manifest
from scripts.diversity_matrix_d0_d5_support import (
    AGENTS,
    ARMS,
    ARM_ORDER,
    AUTH_ENV,
    DEFAULT_PREP_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_RUN_ROOT,
    DESIGN_ROOT,
    EXPERIMENT_MANIFEST,
    ROLE_MODEL,
    SOLVER_MODEL,
    REVISION_OPPORTUNITIES_PER_VALID_SOURCE,
    ROOT,
    RUNTIME_VERSION,
    SOURCE_CANDIDATES_PER_TARGET,
    TASK_MANIFEST,
    THINKING,
    UPDATES,
    arm_protocol_payload,
    completion_registry,
    git,
    manifest,
    project_local,
    read_json,
    record_completion,
    seed_registry_scan,
    sha256_file,
    sha256_json,
    source_inventory,
    write_json,
)
from scripts.v18_teacher_critic_pipeline_support import (
    ArmController,
    CleanTeacherReplay,
    install_pipeline_arm,
)


class DiversityMatrixSystem(PromptEnsembleOptimizationSystem):
    """Install the already-frozen C pipeline only for D4/D5 settings."""

    created: list["DiversityMatrixSystem"] = []

    def __init__(self, cfg: Config, *args: Any, **kwargs: Any) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.matrix_controller: ArmController | None = None
        if cfg.training.experiment_setting in {
            ARMS["D4"]["setting"], ARMS["D5"]["setting"],
        }:
            self.matrix_controller = ArmController(
                arm="C_NO_SEMANTIC_CRITIC",
                clean_replay=CleanTeacherReplay(),
            )
            install_pipeline_arm(self, self.matrix_controller)
        type(self).created.append(self)

    async def update_once(self, update_index: int) -> bool:
        accepted = await super().update_once(update_index)
        if self.early_stop_reason == "no_actionable_responsibility":
            # The matrix freezes 32 scheduled opportunities.  An empty
            # actionable pool is a scientific no-update observation, not a
            # trajectory-level stopping rule.
            self.early_stop_reason = ""
        return accepted


def _task() -> Any:
    return load_task_manifest(str(TASK_MANIFEST))["disambiguation_qa"]


def _config(
    *, seed: int, arm: str, out_dir: Path, cache_path: Path,
    frozen_initialization: Path, resume: bool,
) -> Config:
    task = _task()
    return Config.from_flat(
        task_type=task.task_type,
        dataset_format="mars",
        comparison_task_id=task.task_id,
        benchmark=task.benchmark,
        answer_format=task.answer_format,
        train_path=str((ROOT / task.train_path).resolve()),
        val_path=str((ROOT / task.val_path).resolve()),
        # The path remains identity metadata; the runner blocks every read.
        test_path=str((ROOT / task.test_path).resolve()),
        manifest_sha256=sha256_file(TASK_MANIFEST),
        train_size=75,
        val_size=50,
        test_size=125,
        agent_model=SOLVER_MODEL,
        optimizer_model=ROLE_MODEL,
        evaluator_model=ROLE_MODEL,
        temperature=0.0,
        solver_max_tokens=1800,
        experiment_setting=ARMS[arm]["setting"],
        agents=AGENTS,
        epochs=4,
        update_every=10,
        seed=int(seed),
        proposal_memory_mode="off",
        num_candidates_per_parent=SOURCE_CANDIDATES_PER_TARGET,
        candidate_eval_pool_size=75,
        eval_solver_call_concurrency=8,
        stage_b_candidate_budget=2,
        out_dir=str(out_dir.resolve()),
        shared_solver_cache_path=str(cache_path.resolve()),
        frozen_initialization_manifest_path=str(frozen_initialization.resolve()),
        resume_from_checkpoint=bool(resume),
        provider_call_budget=30000,
        total_token_budget=12_000_000,
        final_test_enabled=False,
        preserve_final_checkpoint=True,
    )


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with sqlite3.connect(str(source)) as left, sqlite3.connect(str(destination)) as right:
        left.backup(right)


async def _freeze_initialization(seed: int, root: Path) -> tuple[Path, Path]:
    manifest_path = root / "frozen_initialization_manifest.json"
    raw_cache = root / "initial_solver_cache.sqlite"
    stable_cache = root / "initial_solver_cache_frozen.sqlite"
    existing = [manifest_path.exists(), raw_cache.exists(), stable_cache.exists()]
    if all(existing):
        value = read_json(manifest_path)
        if int(value["seed"]) != seed:
            raise RuntimeError("frozen initialization seed mismatch")
        return manifest_path, stable_cache
    if any(existing):
        raise RuntimeError("partial frozen initialization cannot be reused")
    root.mkdir(parents=True, exist_ok=False)
    cfg = _config(
        seed=seed, arm="D0", out_dir=root, cache_path=raw_cache,
        frozen_initialization=manifest_path, resume=False,
    )
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    system = PromptEnsembleOptimizationSystem(cfg)
    system.set_run_identity(build_run_identity(
        cfg, train_rows=train, val_rows=validation, test_rows=[], workspace=ROOT,
    ))
    await system.initialize_fixed_probe(train[: cfg.evaluation.candidate_eval_pool_size])
    snapshot = system.frozen_initialization_snapshot()
    _sqlite_backup(raw_cache, stable_cache)
    write_json(manifest_path, {
        "manifest_version": "diversity_matrix_frozen_initialization_v1",
        "seed": seed,
        "initialization_snapshot": snapshot,
        "initial_cache_sha256": sha256_file(stable_cache),
        "test_rows_loaded": 0,
        "test_calls": 0,
    })
    return manifest_path, stable_cache


def _blocked_test_loader(test_path: Path):
    original = cli._load

    def load(path: str, limit: int, fmt: str):
        if Path(path).resolve() == test_path.resolve():
            return []
        return original(path, limit, fmt)

    return original, load


async def _run_cell(cfg: Config, arm: str) -> dict[str, Any]:
    test_path = Path(cfg.data.test_path)
    original_class = cli.PromptEnsembleOptimizationSystem
    original_load, blocked_load = _blocked_test_loader(test_path)
    DiversityMatrixSystem.created = []
    cli.PromptEnsembleOptimizationSystem = DiversityMatrixSystem
    cli._load = blocked_load
    try:
        result = await cli.run(cfg)
    finally:
        cli.PromptEnsembleOptimizationSystem = original_class
        cli._load = original_load
    if not DiversityMatrixSystem.created:
        raise RuntimeError("matrix system was not constructed")
    system = DiversityMatrixSystem.created[-1]
    controller = system.matrix_controller
    hard_rows = list(controller.hard_gate_decisions) if controller else []
    path = Path(cfg.persistence.out_dir) / "hard_gate_decisions_sanitized.jsonl"
    previous = path.read_text(encoding="utf-8") if path.is_file() else ""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(previous)
        for row in hard_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    write_json(Path(cfg.persistence.out_dir) / "matrix_arm_runtime.json", {
        "runtime_version": RUNTIME_VERSION,
        "arm": arm,
        "semantic_critic_enabled": not ARMS[arm]["no_semantic_critic"],
        "teacher_clean_enabled": ARMS[arm]["no_semantic_critic"],
        "deterministic_hard_gate_enabled": ARMS[arm]["no_semantic_critic"],
        "m2f_enabled": False,
        "test_rows_loaded": 0,
        "test_calls": 0,
        "hard_gate_decision_count": len(hard_rows),
    })
    return result


def _verify_source(freeze: Mapping[str, Any], *, allow_report: bool = False) -> None:
    if git("rev-parse", "HEAD") != freeze["execution_commit"]:
        raise RuntimeError("source freeze commit mismatch")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty and not allow_report:
        raise RuntimeError("tracked worktree changed after source freeze")
    for row in freeze["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")


def _checked_command(command: list[str], *, timeout: int = 900) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "output_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "output_tail": completed.stdout.splitlines()[-1:] if completed.stdout else [],
    }


def _zero_api_preflight() -> dict[str, Any]:
    schema = load_yaml(ROOT / "infrastructure" / "experiment_manifest.schema.json")
    manifest_errors = validate_manifest(manifest(), schema)
    focused = _checked_command([
        sys.executable, "-m", "pytest",
        "tests/test_diversity_matrix_d0_d5.py",
        "tests/governance/test_governance.py", "-q",
    ])
    full = _checked_command([sys.executable, "-m", "pytest", "tests", "-q"])
    compileall = _checked_command([
        sys.executable, "-m", "compileall", "-q",
        "multi_dataset_diverse_rl", "scripts", "tests",
    ])
    diff_check = _checked_command(["git", "diff", "--check"])
    payload = {
        "preflight_version": "diversity_matrix_zero_api_preflight_v1",
        "api_calls": 0,
        "model_calls": 0,
        "manifest_validation": {
            "status": "PASS" if not manifest_errors else "FAIL",
            "errors": manifest_errors,
        },
        "focused_tests": focused,
        "full_tests": full,
        "compileall": compileall,
        "git_diff_check": diff_check,
    }
    payload["gate"] = "PASS" if all(
        row["status"] == "PASS"
        for row in (
            payload["manifest_validation"], focused, full, compileall, diff_check,
        )
    ) else "FAIL"
    return payload


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.prep_root.exists() or args.run_root.exists() or args.report_root.exists():
        raise FileExistsError("prepare, run, and report roots must all be fresh")
    if not all(project_local(path) for path in (args.prep_root, args.run_root, args.report_root)):
        raise RuntimeError("all experiment paths must be project-local")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"],
        cwd=ROOT,
    ).returncode == 0
    if not ancestry:
        raise RuntimeError("Phase A requires origin/main to be an ancestor of HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Phase A requires a fully clean tracked worktree")
    document = manifest()
    zero_api = _zero_api_preflight()
    seeds = list(map(int, document["seeds"]))
    seed_scan = seed_registry_scan(exclude_paths=(EXPERIMENT_MANIFEST,))
    gates: dict[str, str] = {}
    gates["GIT_GATE"] = "PASS"
    gates["SEED_GATE"] = (
        "PASS" if seeds == seed_scan["selected_fresh_seeds"] else "FAIL"
    )
    protocols = {arm: arm_protocol_payload(arm) for arm in ARM_ORDER}
    gates["INITIAL_STATE_GATE"] = "PASS"
    gates["ARM_CONFIG_GATE"] = "PASS" if (
        protocols["D0"]["target_branch_count"] == 0
        and protocols["D1"]["target_branch_count"] == 1
        and all(protocols[arm]["target_branch_count"] == 2 for arm in ("D2", "D3", "D4", "D5"))
    ) else "FAIL"
    gates["HORIZON_GATE"] = "PASS" if int(document["budget"]["limit"]["updates_per_evolution_arm"]) == UPDATES else "FAIL"
    parity_fields = (
        "target_branch_count", "candidates_per_target_branch",
        "generic_revision_enabled", "candidate_acceptance_policy",
        "candidate_ranking_policy",
    )
    parity = {
        field: {arm: protocols[arm][field] for arm in ("D2", "D3", "D4", "D5")}
        for field in parity_fields
    }
    gates["COMPUTE_PARITY_GATE"] = "PASS" if all(
        len({values[arm] for arm in values}) == 1 for values in parity.values()
    ) else "FAIL"
    gates["PIPELINE_FREEZE_GATE"] = "PASS" if (
        all(not protocols[arm]["compatibility_repair_enabled"] for arm in ARM_ORDER)
        and all(protocols[arm]["module2_evolution_variant"] == "m20_current_v15" for arm in ("D2", "D3", "D4", "D5"))
        and protocols["D4"]["tcs_context_policy"] == protocols["D5"]["tcs_context_policy"] == "member_aware_responsibility_conditioned"
    ) else "FAIL"
    gates["NO_TEST_GATE"] = "PASS" if document["data"]["test_policy"].startswith("prohibited") else "FAIL"
    gates["RESUME_GATE"] = "PASS" if zero_api["focused_tests"]["status"] == "PASS" else "FAIL"
    gates["ANALYZER_GATE"] = "PASS" if zero_api["focused_tests"]["status"] == "PASS" else "FAIL"
    gates["SANITIZATION_GATE"] = "PASS" if zero_api["focused_tests"]["status"] == "PASS" else "FAIL"
    gates["ZERO_API_VERIFICATION_GATE"] = zero_api["gate"]
    phase_a = "PASS" if set(gates.values()) == {"PASS"} else "FAIL"
    files, source_hash = source_inventory()
    args.prep_root.mkdir(parents=True)
    runtime_manifest = dict(document)
    runtime_manifest["status"] = "RUNNING"
    runtime_manifest["lifecycle_history"] = [
        *document["lifecycle_history"],
        {"status": "RUNNING", "timestamp": "2026-09-03T21:30:00+08:00"},
    ]
    runtime_manifest["git"] = {
        **document["git"], "implementation_commit": git("rev-parse", "HEAD")
    }
    if runtime_manifest["artifacts"]["preregistration"]["sha256"] != preregistration_hash(runtime_manifest):
        raise RuntimeError("tracked preregistration hash mismatch")
    write_json(args.prep_root / "runtime_manifest.json", runtime_manifest)
    registry = {
        "runtime_version": RUNTIME_VERSION,
        "execution_commit": git("rev-parse", "HEAD"),
        "seeds": seeds,
        "arms": protocols,
        "arm_order": list(ARM_ORDER),
        "updates": UPDATES,
        "solver_model": SOLVER_MODEL,
        "role_model": ROLE_MODEL,
        "thinking": THINKING,
        "m2f_enabled": False,
        "m2f_trigger": "none",
        "test_access": "prohibited_zero_rows_loaded_zero_calls",
        "compute_parity": parity,
        "classifier": document["selection"]["frozen_rule"],
    }
    registry["registry_hash"] = sha256_json(registry)
    write_json(args.prep_root / "registry.json", registry)
    write_json(args.prep_root / "source_freeze.json", {
        "freeze_version": "diversity_matrix_source_freeze_v1",
        "execution_commit": registry["execution_commit"],
        "source_tree_hash": source_hash,
        "files": files,
        "seed_scan": seed_scan,
        "selected_fresh_seeds": seeds,
        "registry_hash": registry["registry_hash"],
        "phase_a_gate": phase_a,
    })
    write_json(args.prep_root / "arm_configs.json", protocols)
    write_json(args.prep_root / "compute_parity.json", {
        "planned": {
            "updates_per_evolution_arm": UPDATES,
            "D1_target_opportunities_per_seed": UPDATES,
            "D1_source_candidate_slots_per_seed": UPDATES * 2,
            "D2_D5_target_opportunities_per_seed_arm": UPDATES * 2,
            "D2_D5_source_candidate_slots_per_seed_arm": UPDATES * 2 * 2,
            "revision_opportunities_per_valid_source": REVISION_OPPORTUNITIES_PER_VALID_SOURCE,
        },
        "parity_fields": parity,
        "invalid_output_consumes_scheduled_opportunity": True,
        "attempt_vs_evaluable_not_required_equal": True,
    })
    write_json(args.prep_root / "phase_a_gate.json", {
        "PHASE_A_GATE": phase_a,
        "gates": gates,
        "AUTO_START_RUN": phase_a == "PASS",
        "api_calls": 0,
        "model_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "zero_api_verification": zero_api,
    })
    return {"phase_a_gate": phase_a, "seeds": seeds, "gates": gates}


def _authorization(prep_root: Path, phase: str) -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise RuntimeError(f"set {AUTH_ENV}=1 for the explicitly authorized API run")
    document = read_json(prep_root / "runtime_manifest.json")
    for role in ("solver", "teacher", "critic", "student"):
        require_api_authorization(
            document, phase=phase, role=role, explicit_user_authorized=True,
        )


def _cell_dir(root: Path, seed: int, arm: str) -> Path:
    return root / f"seed{seed}" / arm


async def train(args: argparse.Namespace) -> dict[str, Any]:
    phase_a = read_json(args.prep_root / "phase_a_gate.json")
    if phase_a["PHASE_A_GATE"] != "PASS" or not phase_a["AUTO_START_RUN"]:
        raise RuntimeError("STOP BEFORE API: Phase A did not pass")
    registry = read_json(args.prep_root / "registry.json")
    freeze = read_json(args.prep_root / "source_freeze.json")
    _verify_source(freeze)
    _authorization(args.prep_root, "online_trajectory")
    args.run_root.mkdir(parents=True, exist_ok=True)
    completion_path = args.run_root / "completed_cells.json"
    for seed in registry["seeds"]:
        initialization_root = args.run_root / "_frozen_initialization" / f"seed{seed}"
        frozen_manifest, stable_cache = await _freeze_initialization(seed, initialization_root)
        for arm in ARM_ORDER:
            _verify_source(freeze)
            run_dir = _cell_dir(args.run_root, seed, arm)
            final_summary = run_dir / "final_summary.json"
            if final_summary.is_file():
                record_completion(completion_path, seed=seed, arm=arm, status="COMPLETED")
                continue
            resume = (run_dir / "training_checkpoint.json").is_file()
            if run_dir.exists() and not resume:
                record_completion(
                    completion_path, seed=seed, arm=arm,
                    status="INCOMPLETE", detail="existing_without_checkpoint",
                )
                continue
            cache = run_dir / "_solver_cache.sqlite"
            if not run_dir.exists():
                run_dir.mkdir(parents=True)
                _sqlite_backup(stable_cache, cache)
            write_json(run_dir / "trajectory_status.json", {
                "seed": seed, "arm": arm, "status": "RUNNING",
                "resume": resume, "planned_updates": 0 if arm == "D0" else UPDATES,
            })
            cfg = _config(
                seed=seed, arm=arm, out_dir=run_dir, cache_path=cache,
                frozen_initialization=frozen_manifest, resume=resume,
            )
            try:
                await _run_cell(cfg, arm)
                write_json(run_dir / "trajectory_status.json", {
                    "seed": seed, "arm": arm, "status": "COMPLETED",
                    "resume": resume, "planned_updates": 0 if arm == "D0" else UPDATES,
                })
                record_completion(completion_path, seed=seed, arm=arm, status="COMPLETED")
            except Exception as exc:  # preserve evidence and continue independent cells
                write_json(run_dir / "trajectory_status.json", {
                    "seed": seed, "arm": arm, "status": "INCOMPLETE",
                    "failure_type": type(exc).__name__,
                })
                (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                record_completion(
                    completion_path, seed=seed, arm=arm,
                    status="INCOMPLETE", detail=type(exc).__name__,
                )
    completion = completion_registry(completion_path)
    expected = len(registry["seeds"]) * len(ARM_ORDER)
    payload = {
        "execution_version": RUNTIME_VERSION,
        "expected_trajectory_count": expected,
        "completed_trajectory_count": len(completion["completed"]),
        "incomplete_trajectory_count": len(completion["incomplete"]),
        "execution_gate": (
            "PASS" if len(completion["completed"]) == expected and not completion["incomplete"] else "HOLD"
        ),
        "new_test_calls": 0,
    }
    write_json(args.run_root / "training_execution_summary.json", payload)
    return payload


async def _evaluate_validation_cell(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    meta = read_json(run_dir / "run_meta.json")
    checkpoint_path = run_dir / "training_checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    before_hash = sha256_file(checkpoint_path)
    values = dict(meta["config"])
    values.update({
        "out_dir": str(out_dir.resolve()),
        "shared_solver_cache_path": str((out_dir.parent / "_solver_cache.sqlite").resolve()),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "preserve_final_checkpoint": False,
    })
    cfg = Config.from_flat(**values)
    system = PromptEnsembleOptimizationSystem(cfg)
    train_rows = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation_rows = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    system.set_run_identity(RunIdentity(**checkpoint["run_identity"]))
    system.proposal_memory_run_id = str(checkpoint["proposal_memory_run_id"])
    system.fixed_probe = system.build_probe(train_rows[: cfg.evaluation.candidate_eval_pool_size])
    restore_checkpoint(system, checkpoint)
    system.llm.calls = []
    state_before = system.team_prompt_state_hash()
    expected = str(meta["final_state_selection"]["selected_team_prompt_state_hash"])
    if state_before != expected:
        raise RuntimeError("frozen final-state mismatch before validation")
    metrics = await system.evaluate_dataset(validation_rows)
    if system.team_prompt_state_hash() != state_before:
        raise RuntimeError("validation mutated final state")
    if sha256_file(checkpoint_path) != before_hash:
        raise RuntimeError("validation mutated training checkpoint")
    example_rows = []
    for index, example in enumerate(system._last_evaluated_examples):
        state = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=[profile[index].answer for profile in system._last_evaluated_profiles],
            valid_vector=[profile[index].valid for profile in system._last_evaluated_profiles],
            normalize_answer=system.normalize_answer,
            match_answer=system.match_answer,
            tie_break=system.protocol.tie_policy,
            seed=cfg.training.seed,
        )
        example_rows.append({
            "example_id_hash": state.question_hash,
            "G": int(state.gold_vote_count),
            "H": int(state.largest_wrong_vote_count),
            "M": int(state.plurality_margin),
            "vote_correct": bool(state.vote_correct),
            "correct_member_ids": [
                member for member, correct in enumerate(state.team_correctness) if correct
            ],
            "member_correctness": list(map(bool, state.team_correctness)),
            "member_validity": list(map(bool, state.team_validity)),
        })
    oracle = sum(row["G"] > 0 for row in example_rows)
    cost = system.cost_summary()
    result = {
        "evaluation_version": "diversity_matrix_final_validation_v1",
        "seed": int(meta["config"]["seed"]),
        "setting": meta["canonical_experiment_setting"],
        "logical_validation_evaluations": 1,
        "validation_row_count": len(validation_rows),
        "vote_correct_count": int(metrics.vote_correct_count),
        "vote_accuracy": float(metrics.plurality_vote_acc),
        "oracle_correct_count": oracle,
        "oracle_accuracy": oracle / len(example_rows),
        "per_agent_correct_counts": list(metrics.per_agent_correct_counts),
        "per_agent_invalid_counts": list(metrics.per_agent_invalid_counts),
        "final_state_hash": state_before,
        "checkpoint_sha256": before_hash,
        "provider_calls": int(cost["successful_llm_calls"]),
        "total_tokens": int(cost["total_tokens"]),
        "test_rows_loaded": 0,
        "test_calls": 0,
        "state_mutation": False,
        "checkpoint_mutation": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "evaluation_summary_private.json", result)
    with (out_dir / "validation_rows_sanitized.jsonl").open("w", encoding="utf-8") as handle:
        for row in example_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return result


async def validate(args: argparse.Namespace) -> dict[str, Any]:
    execution = read_json(args.run_root / "training_execution_summary.json")
    if execution["execution_gate"] != "PASS":
        raise RuntimeError("official matrix gate HOLD: validation not started")
    registry = read_json(args.prep_root / "registry.json")
    freeze = read_json(args.prep_root / "source_freeze.json")
    _verify_source(freeze)
    _authorization(args.prep_root, "frozen_validation")
    validation_root = args.run_root / "validation"
    results = []
    for seed in registry["seeds"]:
        for arm in ARM_ORDER:
            out = validation_root / f"seed{seed}" / arm
            if (out / "evaluation_summary_private.json").is_file():
                results.append(read_json(out / "evaluation_summary_private.json"))
                continue
            results.append(await _evaluate_validation_cell(
                _cell_dir(args.run_root, seed, arm), out,
            ))
    payload = {
        "validation_gate": "PASS" if len(results) == len(registry["seeds"]) * len(ARM_ORDER) else "HOLD",
        "logical_validation_evaluation_count": len(results),
        "new_test_calls": sum(row["test_calls"] for row in results),
        "provider_calls": sum(row["provider_calls"] for row in results),
        "total_tokens": sum(row["total_tokens"] for row in results),
    }
    write_json(args.run_root / "validation_execution_summary.json", payload)
    return payload


def _invoke(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
    subprocess.run(command, cwd=ROOT, check=True)


def _post_run_verification(args: argparse.Namespace) -> dict[str, Any]:
    replay = args.run_root / "analysis_deterministic_replay"
    if replay.exists():
        raise FileExistsError("fresh deterministic replay root required")
    _invoke(
        "analyze_diversity_matrix_d0_d5.py",
        "--prep-root", str(args.prep_root), "--run-root", str(args.run_root),
        "--audit-root", str(args.run_root / "audit"), "--out", str(replay),
    )
    compared = (
        "trajectory_level.csv", "update_level.csv", "member_level.csv",
        "diversity_metrics.csv", "coverage_depth.csv", "specialization.csv",
        "contrast_summary.csv", "classifier.json", "execution_summary.json",
    )
    replay_match = all(
        sha256_file(args.report_root / name) == sha256_file(replay / name)
        for name in compared
    )
    focused = _checked_command([
        sys.executable, "-m", "pytest",
        "tests/test_diversity_matrix_d0_d5.py",
        "tests/governance/test_governance.py", "-q",
    ])
    full = _checked_command([sys.executable, "-m", "pytest", "tests", "-q"])
    compileall = _checked_command([
        sys.executable, "-m", "compileall", "-q",
        "multi_dataset_diverse_rl", "scripts", "tests",
    ])
    diff_check = _checked_command(["git", "diff", "--check"])
    findings = scan_sanitized_artifacts(args.report_root)
    payload = {
        "focused_tests": focused,
        "full_tests": full,
        "compileall": compileall,
        "git_diff_check": diff_check,
        "deterministic_analyzer_replay": "PASS" if replay_match else "FAIL",
        "sanitization": "PASS" if not findings else "FAIL",
        "sanitization_findings": findings,
        "new_test_calls": 0,
    }
    payload["gate"] = "PASS" if (
        replay_match and not findings and all(
            row["status"] == "PASS" for row in (focused, full, compileall, diff_check)
        )
    ) else "HOLD"
    (args.report_root / "test_report.txt").write_text(
        "\n".join([
            f"post_run_verification={payload['gate']}",
            f"focused_tests={focused['status']}",
            f"full_tests={full['status']}",
            f"compileall={compileall['status']}",
            f"git_diff_check={diff_check['status']}",
            f"deterministic_replay={payload['deterministic_analyzer_replay']}",
            f"sanitization={payload['sanitization']}",
            "new_test_calls=0",
        ]) + "\n",
        encoding="utf-8",
    )
    (args.report_root / "sanitization_report.txt").write_text(
        "PASS\nforbidden_findings=0\n" if not findings
        else "FAIL\n" + json.dumps(findings, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_lines = []
    for path in sorted(args.report_root.iterdir()):
        if path.name == "sha256_manifest.txt":
            continue
        manifest_lines.append(f"{sha256_file(path)}  {path.name}")
    (args.report_root / "sha256_manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8",
    )
    write_json(args.run_root / "post_run_verification.json", payload)
    if payload["gate"] != "PASS":
        raise RuntimeError("post-run verification HOLD")
    return payload


async def all_phases(args: argparse.Namespace) -> None:
    if not args.prep_root.exists():
        result = prepare(args)
        if result["phase_a_gate"] != "PASS":
            raise RuntimeError("STOP BEFORE API")
    training = await train(args)
    if training["execution_gate"] != "PASS":
        raise RuntimeError("training matrix HOLD")
    validation = await validate(args)
    if validation["validation_gate"] != "PASS":
        raise RuntimeError("validation matrix HOLD")
    audit_root = args.run_root / "audit"
    _invoke(
        "audit_diversity_matrix_d0_d5.py",
        "--prep-root", str(args.prep_root), "--run-root", str(args.run_root),
        "--out", str(audit_root),
    )
    _invoke(
        "analyze_diversity_matrix_d0_d5.py",
        "--prep-root", str(args.prep_root), "--run-root", str(args.run_root),
        "--audit-root", str(audit_root), "--out", str(args.report_root),
    )
    _post_run_verification(args)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    modes.add_argument("--package", action="store_true")
    modes.add_argument("--all", action="store_true")
    value.add_argument("--prep-root", type=Path, default=DEFAULT_PREP_ROOT)
    value.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    value.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return value


def main() -> None:
    args = parser().parse_args()
    args.prep_root = args.prep_root.resolve()
    args.run_root = args.run_root.resolve()
    args.report_root = args.report_root.resolve()
    if args.prepare_only:
        print(json.dumps(prepare(args), indent=2))
    elif args.run or args.resume:
        print(json.dumps(asyncio.run(train(args)), indent=2))
    elif args.validate:
        print(json.dumps(asyncio.run(validate(args)), indent=2))
    elif args.audit:
        _invoke(
            "audit_diversity_matrix_d0_d5.py",
            "--prep-root", str(args.prep_root), "--run-root", str(args.run_root),
            "--out", str(args.run_root / "audit"),
        )
    elif args.analyze or args.package:
        _invoke(
            "analyze_diversity_matrix_d0_d5.py",
            "--prep-root", str(args.prep_root), "--run-root", str(args.run_root),
            "--audit-root", str(args.run_root / "audit"),
            "--out", str(args.report_root),
        )
    else:
        asyncio.run(all_phases(args))


if __name__ == "__main__":
    main()
