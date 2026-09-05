"""Vote-aligned target scheduling on the frozen Shadow-gated D2 pipeline.

``--prepare-only`` is zero-API. Run and resume stay fail-closed until the task
manifest and the one-shot environment authorization both permit Phase B.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import inspect
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from multi_dataset_diverse_rl import cli
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.governance.artifacts import scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.persistence.checkpoint import restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity, build_run_identity
from multi_dataset_diverse_rl.protocol import (
    candidate_budget_contract,
    experiment_protocol,
)
from multi_dataset_diverse_rl.responsibility import MemberAwareRepairOpportunity
from multi_dataset_diverse_rl.shadow_gate import (
    MAX_NO_SHADOW_APPROVED_COMMIT_STREAK,
    SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT,
    assert_winner_only_event,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.vote_aligned_scheduler import (
    DIRECT_FLIP,
    FALLBACK_RR,
    NEAR_MARGIN,
    PURE_COVERAGE,
    RR_GENERIC_SCHEDULER,
    VOTE_ALIGNED_RR_SCHEDULER,
    VOTE_ALIGNED_SCHEDULER_VERSION,
    select_vote_aligned_targets,
)
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
from scripts.run_shadow_gated_evolution import (
    ShadowGatedSystem,
    _backup,
    _evaluate_final_dataset,
    _rows,
)


P0 = "P0_SHADOW_D2_GENERIC"
P1 = "P1_SHADOW_VOTE_ALIGNED_GENERIC"
ARMS = (P0, P1)
SCHEDULER_BY_ARM = {P0: RR_GENERIC_SCHEDULER, P1: VOTE_ALIGNED_RR_SCHEDULER}
SEEDS = (75,)
AUTH_ENV = "VOTE_ALIGNED_SHADOW_PHASE_B_AUTHORIZED"
MANIFEST = ROOT / "experiments" / "manifests" / "vote_aligned_generic_shadow_pilot_v1.yaml"
DESIGN_ROOT = ROOT / "experiments" / "vote_aligned_generic_shadow_pilot_v1"
SOURCE_SPLIT_ROOT = ROOT / "experiments" / "anti_overfitting_split_v1"
DEFAULT_PREP_ROOT = ROOT / "runs" / "vote_aligned_generic_shadow_pilot_v1_prep"
DEFAULT_RUN_ROOT = ROOT / "runs" / "vote_aligned_generic_shadow_pilot_v1"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "vote_aligned_generic_shadow_pilot_v1"
RUNTIME_VERSION = "vote_aligned_generic_shadow_pilot_v1"


class VoteAlignedShadowSystem(ShadowGatedSystem):
    """Shared Shadow write-back with scheduler selected only by Config."""


def _config(
    seed: int,
    arm: str,
    out: Path,
    optimize_path: Path,
    validation_path: Path,
    cache: Path,
    initialization: Path,
    resume: bool,
) -> Config:
    return Config.from_flat(
        task_type="bbh",
        dataset_format="mars",
        comparison_task_id="disambiguation_qa",
        benchmark="BBH",
        answer_format="option_letter",
        train_path=str(optimize_path),
        val_path=str(validation_path),
        test_path="TEST50_BLOCKED_BY_GOVERNANCE",
        manifest_sha256=sha256_file(MANIFEST),
        train_size=100,
        val_size=50,
        test_size=0,
        agent_model=SOLVER_MODEL,
        optimizer_model=ROLE_MODEL,
        evaluator_model=ROLE_MODEL,
        temperature=0.0,
        solver_max_tokens=1800,
        experiment_setting="experimental_diversity_d2_rr_generic",
        target_scheduler=SCHEDULER_BY_ARM[arm],
        agents=5,
        epochs=4,
        update_every=13,
        seed=seed,
        proposal_memory_mode="off",
        num_candidates_per_parent=2,
        candidate_eval_pool_size=100,
        eval_solver_call_concurrency=8,
        stage_b_candidate_budget=2,
        out_dir=str(out),
        shared_solver_cache_path=str(cache),
        frozen_initialization_manifest_path=str(initialization),
        resume_from_checkpoint=resume,
        provider_call_budget=50000,
        total_token_budget=15_000_000,
        final_test_enabled=False,
        preserve_final_checkpoint=True,
    )


def _resolved_protocol(cfg: Config):
    budget = candidate_budget_contract(
        cfg.training.experiment_setting,
        candidates_per_target_branch=cfg.tcs.num_candidates_per_parent,
        stage_b_budget_per_branch=cfg.evaluation.stage_b_candidate_budget,
        stage_a_channel_top_k=cfg.evaluation.stage_a_channel_top_k,
        representative_size=cfg.evaluation.stage_a_representative_size,
        coverage_size=cfg.evaluation.stage_a_coverage_size,
        conversion_size=cfg.evaluation.stage_a_conversion_size,
        preservation_size=cfg.evaluation.stage_a_preservation_size,
    )
    return experiment_protocol(
        cfg.training.experiment_setting,
        initialization_mode=cfg.training.initialization_mode,
        tie_policy=cfg.peer_state.vote_tie_break,
        candidate_budget_contract=budget,
        allow_legacy_setting=cfg.training.allow_legacy_setting,
        allow_auxiliary_setting=cfg.training.allow_auxiliary_setting,
    )


def _assignment() -> dict[str, list[str]]:
    split = json.loads((SOURCE_SPLIT_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((SOURCE_SPLIT_ROOT / "fold_assignment.json").read_text(encoding="utf-8"))
    return {
        "fold_a": list(folds["folds"]["fold_a"]),
        "fold_b": list(folds["folds"]["fold_b"]),
        "fold_c": list(folds["folds"]["fold_c"]),
        "validation": list(split["question_hashes"]["validation"]),
        "test": list(split["question_hashes"]["test"]),
    }


def _fold_paths(prep_root: Path, index: int) -> tuple[Path, Path]:
    optimize, shadow = FOLD_MAP[index]
    parts = [prep_root / "splits_private" / f"{name}.csv" for name in optimize.split("+")]
    combined = prep_root / "splits_private" / f"optimize_group_{index + 1}.csv"
    if not combined.exists():
        rows = [row for path in parts for row in _rows(path)]
        with combined.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(rows)
    return combined, prep_root / "splits_private" / f"{shadow}.csv"


def _protocol_document() -> dict[str, Any]:
    return {
        "runtime_version": RUNTIME_VERSION,
        "execution_baseline": git("rev-parse", "HEAD"),
        "historical_requested_baseline": "dc4971832754a9dbf5d64e08e571a65d75edd473",
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
            {"seed": SEEDS[index], "optimize": pair[0], "shadow": pair[1]}
            for index, pair in enumerate(FOLD_MAP[:len(SEEDS)])
        ],
        "arms": {arm: {"target_scheduler": SCHEDULER_BY_ARM[arm]} for arm in ARMS},
        "shared_protocol": {
            "experiment_setting": "experimental_diversity_d2_rr_generic",
            "proposal": "generic_peer_state",
            "target_slots": 2,
            "source_candidates_per_target": 2,
            "generic_revision": True,
            "common_safe": True,
            "winner_only_shadow": True,
            "shadow_vote_delta_minimum": 0,
            "shadow_target_loss_maximum": SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT,
            "maximum_update_opportunities": 32,
            "no_commit_patience": MAX_NO_SHADOW_APPROVED_COMMIT_STREAK,
        },
        "near_margin": (
            "vote_flip_gain=0 and margin_gain>0 and "
            "current_M+margin_gain=0"
        ),
        "validation_policy": "one final frozen-state evaluation only",
        "test_policy": "zero access",
    }


def _source_freeze(prep_root: Path) -> None:
    files = {
        path.relative_to(ROOT)
        for path in (ROOT / "multi_dataset_diverse_rl").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    files.update({
        Path("scripts/anti_overfitting_shadow_support.py"),
        Path("scripts/run_shadow_gated_evolution.py"),
        Path("scripts/run_vote_aligned_generic_shadow_pilot.py"),
        Path("experiments/manifests/vote_aligned_generic_shadow_pilot_v1.yaml"),
        Path("experiments/vote_aligned_generic_shadow_pilot_v1/PROTOCOL.md"),
        Path("experiments/vote_aligned_generic_shadow_pilot_v1/selector_definition.json"),
        Path("experiments/anti_overfitting_split_v1/split_manifest.json"),
        Path("experiments/anti_overfitting_split_v1/fold_assignment.json"),
    })
    ordered = sorted(files, key=lambda path: path.as_posix())
    write_json(prep_root / "source_freeze.json", {
        "execution_commit": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": git("status", "--porcelain", "--untracked-files=all") == "",
        "files": [
            {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)}
            for path in ordered
        ],
    })


def _synthetic_selector_gate() -> bool:
    def row(agent: int, key: str, *, flip: int = 0, gain: int = 1, coverage: bool = False):
        return MemberAwareRepairOpportunity(
            agent_id=agent,
            question_hash=key,
            vote_flip_gain=flip,
            margin_gain=gain,
            member_error=True,
            coverage_opportunity=coverage,
            conversion_opportunity=not coverage,
            dominant_wrong_member=False,
            unique_correct=False,
            pivotal_correct=False,
            oracle_soft_utility_gain=0.0,
        )
    assigned = {
        0: [row(0, "coverage", coverage=True)],
        1: [row(1, "near")],
        2: [row(2, "flip", flip=1)],
    }
    margins = {"coverage": -2, "near": -1, "flip": 0}
    first = select_vote_aligned_targets(
        assigned=assigned,
        current_margin_by_question=margins,
        seed=75,
        update_index=0,
    )
    replay = select_vote_aligned_targets(
        assigned=assigned,
        current_margin_by_question=margins,
        seed=75,
        update_index=0,
    )
    return (
        first == replay
        and first.targets == (2, 1)
        and len(set(first.targets)) == len(first.targets)
    )


def prepare(prep_root: Path) -> dict[str, Any]:
    items, raw = metadata()
    assignment = _assignment()
    prep_root.mkdir(parents=True, exist_ok=True)
    export_private_splits(raw, assignment, prep_root / "splits_private")
    protocol = _protocol_document()
    write_json(prep_root / "protocol_freeze.json", protocol)
    write_json(prep_root / "test_access_registry.json", {
        "schema_version": "test_access_registry_v1",
        "events": [],
        "new_test_calls": 0,
    })
    p0 = _config(75, P0, Path("p0"), Path("opt.csv"), Path("val.csv"), Path("p0.sqlite"), Path("init.json"), False)
    p1 = _config(75, P1, Path("p1"), Path("opt.csv"), Path("val.csv"), Path("p1.sqlite"), Path("init.json"), False)
    left = p0.to_flat_dict()
    right = p1.to_flat_dict()
    allowed_differences = {
        "target_scheduler", "out_dir", "shared_solver_cache_path"
    }
    actual_differences = {key for key in left if left[key] != right[key]}
    p0_protocol = _resolved_protocol(p0)
    p1_protocol = _resolved_protocol(p1)
    protocol_equal = p0_protocol == p1_protocol
    all_hashes = [digest for key in ("fold_a", "fold_b", "fold_c", "validation", "test") for digest in assignment[key]]
    old_split = json.loads((SOURCE_SPLIT_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    selector_source = inspect.getsource(select_vote_aligned_targets)
    shadow_source = inspect.getsource(ShadowGatedSystem.approve_writeback_candidate)
    gates = {
        "GIT_GATE": git("status", "--porcelain", "--untracked-files=all") == "",
        "SPLIT_GATE": (
            len(items) == 250
            and len(all_hashes) == len(set(all_hashes)) == 250
            and sha256_json(sorted(assignment["validation"]))
            == sha256_json(sorted(old_split["question_hashes"]["validation"]))
            and sha256_json(sorted(assignment["test"]))
            == sha256_json(sorted(old_split["question_hashes"]["test"]))
        ),
        "CROSSFIT_GATE": (
            tuple(SEEDS) == (75,)
            and tuple(FOLD_MAP[0]) == ("fold_a+fold_b", "fold_c")
        ),
        "MODEL_GATE": (
            p0.models.agent_model == p1.models.agent_model == "qwen3-8b"
            and p0.models.optimizer_model == p1.models.optimizer_model == "qwen3.7-flash"
            and p0.models.evaluator_model == p1.models.evaluator_model == "qwen3.7-flash"
            and p0.models.temperature == p1.models.temperature == 0.0
        ),
        "P0_IDENTITY_GATE": (
            p0.training.target_scheduler == RR_GENERIC_SCHEDULER
            and p0_protocol.name == "experimental_diversity_d2_rr_generic"
            and p0_protocol.target_selection_policy == "responsibility_round_robin_dual"
        ),
        "P1_SELECTOR_GATE": (
            p1.training.target_scheduler == VOTE_ALIGNED_RR_SCHEDULER
            and _synthetic_selector_gate()
            and "validation" not in selector_source.lower()
            and "test" not in selector_source.lower()
        ),
        "GENERIC_PIPELINE_PARITY_GATE": (
            protocol_equal
            and actual_differences == allowed_differences
            and p0_protocol.tcs_context_policy == p1_protocol.tcs_context_policy == "generic_peer_state"
            and p0_protocol.generic_revision_enabled
            and p1_protocol.generic_revision_enabled
        ),
        "SHADOW_GATE_PARITY_GATE": (
            SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT == 2
            and MAX_NO_SHADOW_APPROVED_COMMIT_STREAK == 6
            and "select_targets" not in shadow_source
            and "_run_teacher" not in shadow_source
        ),
        "COMPUTE_GATE": (
            p0_protocol.target_branch_count == p1_protocol.target_branch_count == 2
            and p0_protocol.candidates_per_target_branch == p1_protocol.candidates_per_target_branch == 2
            and p0.training.epochs * 8 == p1.training.epochs * 8 == 32
            and p0.evaluation.stage_b_candidate_budget == p1.evaluation.stage_b_candidate_budget == 2
        ),
        "NO_VALIDATION_TEST_LEAKAGE_GATE": (
            protocol["validation_policy"].startswith("one final")
            and protocol["test_policy"] == "zero access"
            and not json.loads((prep_root / "test_access_registry.json").read_text(encoding="utf-8"))["events"]
        ),
    }
    result = {
        "phase_a_gate": "PASS" if all(gates.values()) else "HOLD",
        "gates": {key: "PASS" if value else "HOLD" for key, value in gates.items()},
        "api_calls": 0,
        "validation_calls": 0,
        "new_test_calls": 0,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "scheduler_version": VOTE_ALIGNED_SCHEDULER_VERSION,
        "prior_partial_seed75_artifact_reused": False,
    }
    write_json(prep_root / "phase_a_gate.json", result)
    _source_freeze(prep_root)
    return result


def _authorized() -> None:
    document = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    if os.environ.get(AUTH_ENV) != "1":
        raise RuntimeError("Phase B requires a new explicit API authorization")
    for role in ("solver", "teacher", "critic", "student"):
        require_api_authorization(
            document,
            phase="online_trajectory",
            role=role,
            explicit_user_authorized=True,
        )
    for role in ("solver", "evaluator"):
        require_api_authorization(
            document,
            phase="frozen_validation",
            role=role,
            explicit_user_authorized=True,
        )


def _verify_source_freeze(prep_root: Path) -> None:
    freeze = json.loads((prep_root / "source_freeze.json").read_text(encoding="utf-8"))
    if not freeze["tracked_worktree_clean"] or git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Phase B requires a clean tracked worktree")
    if freeze["execution_commit"] != git("rev-parse", "HEAD"):
        raise RuntimeError("execution commit differs from Phase-A source freeze")
    for row in freeze["files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")


async def _freeze_initialization(
    seed: int,
    optimize: Path,
    validation: Path,
    root: Path,
) -> tuple[Path, Path]:
    manifest = root / "frozen_initialization_manifest.json"
    stable = root / "initial_solver_cache_frozen.sqlite"
    if manifest.is_file() and stable.is_file():
        return manifest, stable
    root.mkdir(parents=True, exist_ok=False)
    raw_cache = root / "initial_solver_cache.sqlite"
    cfg = _config(seed, P0, root, optimize, validation, raw_cache, manifest, False)
    system = PromptEnsembleOptimizationSystem(cfg)
    train = _rows(optimize)
    system.set_run_identity(build_run_identity(
        cfg,
        train_rows=train,
        val_rows=_rows(validation),
        test_rows=[],
        workspace=ROOT,
    ))
    await system.initialize_fixed_probe(train)
    _backup(raw_cache, stable)
    write_json(manifest, {
        "manifest_version": "vote_aligned_matched_initialization_v1",
        "seed": seed,
        "initialization_snapshot": system.frozen_initialization_snapshot(),
        "new_test_calls": 0,
    })
    return manifest, stable


async def execute(prep_root: Path, run_root: Path, *, resume: bool) -> dict[str, Any]:
    _authorized()
    _verify_source_freeze(prep_root)
    phase_a = json.loads((prep_root / "phase_a_gate.json").read_text(encoding="utf-8"))
    if phase_a["phase_a_gate"] != "PASS":
        raise RuntimeError("Phase A gate is not PASS")
    run_root.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(SEEDS):
        optimize, shadow = _fold_paths(prep_root, index)
        validation = prep_root / "splits_private" / "validation.csv"
        initialization, stable = await _freeze_initialization(
            seed, optimize, validation, run_root / "initialization" / f"seed{seed}"
        )
        for arm in ARMS:
            out = run_root / f"seed{seed}" / arm
            if (out / "final_summary.json").is_file():
                continue
            if out.exists() and not (resume and (out / "training_checkpoint.json").is_file()):
                raise RuntimeError("existing incomplete cell requires --resume")
            out.mkdir(parents=True, exist_ok=True)
            cache = out / "solver_cache.sqlite"
            if not cache.exists():
                _backup(stable, cache)
            cfg = _config(seed, arm, out, optimize, validation, cache, initialization, resume)
            VoteAlignedShadowSystem.shadow_rows = _rows(shadow)
            VoteAlignedShadowSystem.shadow_enabled = True
            VoteAlignedShadowSystem.created = []
            original_system = cli.PromptEnsembleOptimizationSystem
            original_load = cli._load

            def blocked_test(path: str, limit: int, fmt: str) -> list[dict[str, Any]]:
                if path == "TEST50_BLOCKED_BY_GOVERNANCE":
                    return []
                return original_load(path, limit, fmt)

            cli.PromptEnsembleOptimizationSystem = VoteAlignedShadowSystem
            cli._load = blocked_test
            try:
                await cli.run(cfg)
            finally:
                cli.PromptEnsembleOptimizationSystem = original_system
                cli._load = original_load
            registry = out / "shadow_evaluation_registry.json"
            events = json.loads(registry.read_text(encoding="utf-8"))["events"] if registry.is_file() else {}
            with (out / "shadow_gate_events_sanitized.jsonl").open("w", encoding="utf-8") as handle:
                for _, event in sorted(events.items(), key=lambda item: int(item[0])):
                    handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

    evaluations: list[dict[str, Any]] = []
    for index, seed in enumerate(SEEDS):
        _, shadow = _fold_paths(prep_root, index)
        validation = prep_root / "splits_private" / "validation.csv"
        for arm in ARMS:
            cell = run_root / f"seed{seed}" / arm
            evaluations.append(await _evaluate_final_dataset(
                cell, shadow, "shadow", run_root / "evaluation" / f"seed{seed}" / arm / "shadow"
            ))
            evaluations.append(await _evaluate_final_dataset(
                cell, validation, "validation", run_root / "evaluation" / f"seed{seed}" / arm / "validation"
            ))
    result = {
        "execution_gate": "PASS",
        "completed_trajectories": 6,
        "final_shadow_evaluations": 6,
        "final_validation_evaluations": 6,
        "provider_calls_in_final_evaluation": sum(int(row["provider_calls"]) for row in evaluations),
        "new_test_calls": 0,
    }
    write_json(run_root / "execution_summary.json", result)
    return result


def audit(prep_root: Path, run_root: Path) -> dict[str, Any]:
    if not run_root.exists():
        return {"phase_b_gate": "NOT_RUN", "new_test_calls": 0}
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    freeze = json.loads((prep_root / "source_freeze.json").read_text(encoding="utf-8"))
    for frozen in freeze["files"]:
        if sha256_file(ROOT / frozen["path"]) != frozen["sha256"]:
            errors.append(f"source_freeze:{frozen['path']}")
    for seed in SEEDS:
        init_hashes: set[str] = set()
        for arm in ARMS:
            cell = run_root / f"seed{seed}" / arm
            required = ["final_summary.json", "training_checkpoint.json", "run_meta.json", "completed_update_registry.json"]
            for name in required:
                if not (cell / name).is_file():
                    errors.append(f"missing:{seed}:{arm}:{name}")
            if any(not (cell / name).is_file() for name in required):
                continue
            checkpoint = json.loads((cell / "training_checkpoint.json").read_text(encoding="utf-8"))
            meta = json.loads((cell / "run_meta.json").read_text(encoding="utf-8"))
            scheduler = str(meta["config"]["target_scheduler"])
            if scheduler != SCHEDULER_BY_ARM[arm]:
                errors.append(f"scheduler:{seed}:{arm}")
            if meta["config"]["agent_model"] != SOLVER_MODEL or meta["config"]["optimizer_model"] != ROLE_MODEL:
                errors.append(f"model:{seed}:{arm}")
            if meta["config"]["evaluator_model"] != ROLE_MODEL:
                errors.append(f"evaluator_model:{seed}:{arm}")
            if meta["config"]["experiment_setting"] != "experimental_diversity_d2_rr_generic":
                errors.append(f"protocol:{seed}:{arm}")
            if int(meta["config"]["num_candidates_per_parent"]) != 2:
                errors.append(f"source_candidate_budget:{seed}:{arm}")
            if int(meta["config"]["stage_b_candidate_budget"]) != 2:
                errors.append(f"stage_b_budget:{seed}:{arm}")
            if str(meta["config"]["proposal_memory_mode"]) != "off":
                errors.append(f"proposal_memory:{seed}:{arm}")
            if int(checkpoint["planned_update_count"]) != 32 or int(checkpoint["completed_update_count"]) > 32:
                errors.append(f"budget:{seed}:{arm}")
            if not checkpoint["training_completed"]:
                errors.append(f"training_incomplete:{seed}:{arm}")
            if any(bool(checkpoint[key]) for key in ("test_evaluation_count", "test_used_for_selection", "test_used_for_training", "test_called_before_training_complete")):
                errors.append(f"test_isolation:{seed}:{arm}")
            match = json.loads((cell / "frozen_initialization_match.json").read_text(encoding="utf-8"))
            if not match["matched"]:
                errors.append(f"initialization:{seed}:{arm}")
            init_hashes.add(str(match["initialization_snapshot"]["initial_train_state_hash"]))
            target_rows = checkpoint["target_priority_audit"]
            expected_stage = "responsibility_round_robin_dual" if arm == P0 else "vote_aligned_lane_prioritized_rr"
            if any(row.get("selection_pool_stage") != expected_stage for row in target_rows):
                errors.append(f"target_audit:{seed}:{arm}")
            if any(len(row.get("selected_target_ids", [])) != len(set(row.get("selected_target_ids", []))) for row in target_rows):
                errors.append(f"duplicate_target:{seed}:{arm}")
            if any(len(row.get("selected_target_ids", [])) > 2 for row in target_rows):
                errors.append(f"target_slot_budget:{seed}:{arm}")
            if arm == P1 and any(
                row.get("scheduler_version") != VOTE_ALIGNED_SCHEDULER_VERSION
                or "rr_cursor_after" not in row
                or "slot_decisions" not in row
                for row in target_rows
            ):
                errors.append(f"selector_replay_evidence:{seed}:{arm}")
            for decision in checkpoint["candidate_decisions"]:
                if len(decision.get("branches", [])) > 2:
                    errors.append(f"branch_budget:{seed}:{arm}")
                if any(
                    int(branch.get("funnel", {}).get("requested_candidate_count", 0)) > 2
                    for branch in decision.get("branches", [])
                ):
                    errors.append(f"candidate_budget:{seed}:{arm}")
            event_path = cell / "shadow_gate_events_sanitized.jsonl"
            events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()] if event_path.is_file() else []
            for event in events:
                try:
                    assert_winner_only_event(event)
                except (AssertionError, KeyError, ValueError) as exc:
                    errors.append(f"shadow_event:{seed}:{arm}:{exc}")
            rows.append({
                "seed": seed,
                "arm": arm,
                "completed_updates": int(checkpoint["completed_update_count"]),
                "early_stop_reason": str(checkpoint["early_stop_reason"]),
                "shadow_evaluations": len(events),
                "accepted_commits": int(checkpoint["accepted_state_count"]),
            })
            completed = int(checkpoint["completed_update_count"])
            reason = str(checkpoint["early_stop_reason"])
            if completed < 32 and reason != (
                f"no_shadow_approved_commit_streak_{MAX_NO_SHADOW_APPROVED_COMMIT_STREAK}"
            ):
                errors.append(f"early_stop:{seed}:{arm}")
        if len(init_hashes) != 1:
            errors.append(f"paired_initialization:{seed}")
    evaluations = list((run_root / "evaluation").rglob("evaluation_summary_private.json"))
    if len(evaluations) != 12:
        errors.append("final_evaluation_inventory")
    result = {
        "phase_b_gate": "PASS" if not errors else "HOLD",
        "errors": sorted(set(errors)),
        "trajectory_count": len(rows),
        "final_evaluation_count": len(evaluations),
        "trajectories": rows,
        "new_test_calls": 0,
    }
    write_json(run_root / "audit_summary.json", result)
    return result


def analyze(prep_root: Path, run_root: Path, report_root: Path) -> dict[str, Any]:
    gate = audit(prep_root, run_root)
    if gate["phase_b_gate"] != "PASS":
        return {"analysis_gate": "NOT_RUN", "phase_b_gate": gate["phase_b_gate"]}
    report_root.mkdir(parents=True, exist_ok=True)
    trajectories: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    lanes: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            cell = run_root / f"seed{seed}" / arm
            checkpoint = json.loads((cell / "training_checkpoint.json").read_text(encoding="utf-8"))
            evaluations = {
                split: json.loads((run_root / "evaluation" / f"seed{seed}" / arm / split / "evaluation_summary_private.json").read_text(encoding="utf-8"))
                for split in ("shadow", "validation")
            }
            decisions = checkpoint["candidate_decisions"]
            funnels = [row.get("funnel", {}) for row in decisions]
            target_audits = checkpoint["target_priority_audit"]
            shadow_events_path = cell / "shadow_gate_events_sanitized.jsonl"
            shadow_events = {
                int(row["update_index"]): row
                for row in (
                    json.loads(line)
                    for line in shadow_events_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            lane_counts = {
                DIRECT_FLIP: 0,
                NEAR_MARGIN: 0,
                PURE_COVERAGE: 0,
                FALLBACK_RR: 0,
            }
            lane_commit_counts = dict.fromkeys(lane_counts, 0)
            for target_row in target_audits:
                for slot in target_row.get("slot_decisions", []):
                    lane = str(slot["lane_selected"])
                    lane_counts[lane] += 1
                    update_index = int(target_row["update_index"])
                    event = shadow_events.get(update_index)
                    selected_target = int(slot["selected_target_id"])
                    is_optimize_winner = bool(
                        event and int(event["target_agent_id"]) == selected_target
                    )
                    is_commit = bool(is_optimize_winner and event["passed"])
                    if is_commit:
                        lane_commit_counts[lane] += 1
                    lanes.append({
                        "seed": seed,
                        "arm": arm,
                        "update_index": update_index,
                        **slot,
                        "optimize_winner": is_optimize_winner,
                        "shadow_approved_commit": is_commit,
                    })
            validation = evaluations["validation"]
            shadow = evaluations["shadow"]
            final_optimize = checkpoint["training_dynamics"][-1]
            optimize_vote = float(final_optimize["team_vote_accuracy"])
            optimize_member = float(final_optimize["mean_member_accuracy"])
            optimize_oracle = float(final_optimize["oracle_accuracy"])
            actual_opportunities = int(checkpoint["completed_update_count"])
            early_stop_index = (
                actual_opportunities - 1
                if actual_opportunities < int(checkpoint["planned_update_count"])
                else ""
            )
            trajectories.append({
                "seed": seed,
                "arm": arm,
                "scheduled_update_opportunities": int(checkpoint["planned_update_count"]),
                "actual_update_opportunities": actual_opportunities,
                "early_stop_reason": checkpoint["early_stop_reason"],
                "early_stop_update_index": early_stop_index,
                "target_slots": sum(len(row.get("selected_target_ids", [])) for row in target_audits),
                "generation_attempts": sum(int(row.get("teacher_calls", 0)) for row in funnels),
                "valid_candidates": sum(int(row.get("valid_candidate_count", 0)) for row in funnels),
                "feasible_candidates": sum(int(row.get("constraint_feasible", 0)) for row in funnels),
                "optimize_winners": sum(bool(row.get("optimize_winner_hash")) for row in decisions),
                "shadow_evaluations": len(shadow_events),
                "shadow_rejections": sum(not bool(row["passed"]) for row in shadow_events.values()),
                "shadow_approved_commits": int(checkpoint["accepted_state_count"]),
                "optimize_vote": optimize_vote,
                "optimize_mean_member": optimize_member,
                "optimize_oracle": optimize_oracle,
                "shadow_vote": shadow["vote_accuracy"],
                "shadow_mean_member": shadow["mean_member_accuracy"],
                "shadow_oracle": shadow["oracle_correct_count"] / 50,
                "validation_vote": validation["vote_accuracy"],
                "validation_mean_member": validation["mean_member_accuracy"],
                "validation_oracle": validation["oracle_correct_count"] / 50,
                "validation_ensemble_gain": validation["vote_accuracy"] - validation["mean_member_accuracy"],
                "optimize_to_shadow_vote_gap": optimize_vote - shadow["vote_accuracy"],
                "optimize_to_validation_vote_gap": optimize_vote - validation["vote_accuracy"],
                "shadow_to_validation_vote_gap": shadow["vote_accuracy"] - validation["vote_accuracy"],
                "optimize_to_shadow_member_gap": optimize_member - shadow["mean_member_accuracy"],
                "optimize_to_validation_member_gap": optimize_member - validation["mean_member_accuracy"],
                "shadow_to_validation_member_gap": shadow["mean_member_accuracy"] - validation["mean_member_accuracy"],
                "optimize_to_shadow_oracle_gap": optimize_oracle - shadow["oracle_correct_count"] / 50,
                "optimize_to_validation_oracle_gap": optimize_oracle - validation["oracle_correct_count"] / 50,
                "shadow_to_validation_oracle_gap": shadow["oracle_correct_count"] / 50 - validation["oracle_correct_count"] / 50,
                **{f"target_opportunities_{lane}": count for lane, count in lane_counts.items()},
                **{f"commits_{lane}": count for lane, count in lane_commit_counts.items()},
            })
            for member, value in enumerate(validation["per_agent_accuracies"]):
                members.append({"seed": seed, "arm": arm, "member": member, "validation_accuracy": value})
            coverage.append({"seed": seed, "arm": arm, **validation["coverage_depth"]})

    def write_csv(name: str, values: list[dict[str, Any]]) -> None:
        if not values:
            (report_root / name).write_text("", encoding="utf-8")
            return
        with (report_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_csv("trajectory_summary.csv", trajectories)
    write_csv("member_results.csv", members)
    write_csv("lane_targeting.csv", lanes)
    write_csv("coverage_depth.csv", coverage)
    by_key = {(row["seed"], row["arm"]): row for row in trajectories}
    contrasts = []
    for seed in SEEDS:
        p0, p1 = by_key[(seed, P0)], by_key[(seed, P1)]
        contrasts.append({
            "seed": seed,
            "validation_vote_delta": p1["validation_vote"] - p0["validation_vote"],
            "validation_mean_member_delta": p1["validation_mean_member"] - p0["validation_mean_member"],
            "validation_ensemble_gain_delta": p1["validation_ensemble_gain"] - p0["validation_ensemble_gain"],
            "validation_oracle_delta": p1["validation_oracle"] - p0["validation_oracle"],
        })
    write_csv("contrast_summary.csv", contrasts)
    mean_vote = sum(row["validation_vote_delta"] for row in contrasts) / len(contrasts)
    mean_member = sum(row["validation_mean_member_delta"] for row in contrasts) / len(contrasts)
    mean_ensemble = sum(row["validation_ensemble_gain_delta"] for row in contrasts) / len(contrasts)
    wins = sum(row["validation_vote_delta"] > 0 for row in contrasts)
    losses = sum(row["validation_vote_delta"] < 0 for row in contrasts)
    if mean_vote > 0 and wins > losses and mean_member >= -0.01 and mean_ensemble > 0:
        classifier = "VOTE_ALIGNED_SPECIALIZATION_SUPPORTED"
    elif mean_ensemble > 0 and mean_vote <= 0:
        classifier = "VOTE_STRUCTURE_IMPROVED_WITHOUT_FINAL_GAIN"
    elif mean_vote < 0 and mean_member < -0.01:
        classifier = "SPECIALIZATION_HARMFUL"
    else:
        classifier = "NO_CLEAR_VOTE_ALIGNED_BENEFIT"
    summary = {
        "analysis_gate": "PASS",
        "classifier": classifier,
        "mean_paired_validation_vote_delta": mean_vote,
        "mean_paired_validation_member_delta": mean_member,
        "mean_paired_ensemble_gain_delta": mean_ensemble,
        "vote_wins_ties_losses": [wins, len(contrasts) - wins - losses, losses],
        "new_test_calls": 0,
    }
    write_json(report_root / "summary.json", summary)
    write_json(report_root / "classifier.json", {"classifier": classifier})
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    value.add_argument("--prep-root", type=Path, default=DEFAULT_PREP_ROOT)
    value.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    value.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return value


def main() -> None:
    args = parser().parse_args()
    prep = args.prep_root.resolve()
    run = args.run_root.resolve()
    report = args.report_root.resolve()
    if args.prepare_only:
        result = prepare(prep)
    elif args.run:
        result = asyncio.run(execute(prep, run, resume=False))
    elif args.resume:
        result = asyncio.run(execute(prep, run, resume=True))
    elif args.audit:
        result = audit(prep, run)
    else:
        result = analyze(prep, run, report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
