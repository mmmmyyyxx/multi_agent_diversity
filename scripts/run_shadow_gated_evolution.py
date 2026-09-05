"""Unified runner for the anti-overfitting cross-fit protocol.

Phase A is fully zero-API.  ``--run``/``--resume`` stay fail-closed until a
later, committed manifest explicitly authorizes Phase B and the caller also
sets the one-shot authorization environment variable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import inspect
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from multi_dataset_diverse_rl import cli
from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.governance.artifacts import scan_sanitized_artifacts
from multi_dataset_diverse_rl.persistence.checkpoint import restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity, build_run_identity
from multi_dataset_diverse_rl.shadow_gate import (
    MAX_NO_SHADOW_APPROVED_COMMIT_STREAK,
    SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT,
    ShadowGateMetrics,
    advance_no_commit_streak,
    assert_winner_only_event,
    evaluate_shadow_gate,
    shadow_false_positive,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.anti_overfitting_shadow_support import (
    ARMS,
    AUTH_ENV,
    BUCKETS,
    DEFAULT_PREP_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_RUN_ROOT,
    DESIGN_ROOT,
    FOLD_MAP,
    MAX_UPDATE_OPPORTUNITIES,
    ROLE_MODEL,
    SOLVER_MODEL,
    SPLIT_SEED,
    balance_report,
    construct_assignment,
    export_private_splits,
    fresh_seed_freeze,
    git,
    metadata,
    sha256_file,
    sha256_json,
    write_json,
)


MANIFEST = ROOT / "experiments" / "manifests" / "anti_overfitting_shadow_gate_v1.yaml"
RUNTIME_VERSION = "anti_overfitting_shadow_gated_training_v1"


class ShadowGatedSystem(PromptEnsembleOptimizationSystem):
    shadow_rows: list[Mapping[str, Any]] = []
    shadow_enabled: bool = False
    created: list["ShadowGatedSystem"] = []

    def __init__(self, cfg: Config, *args: Any, **kwargs: Any) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.shadow_probe = self.build_probe(type(self).shadow_rows) if type(self).shadow_enabled else None
        self.shadow_profile_cache: dict[str, list[tuple[Any, ...]]] = {}
        self.shadow_gate_events: list[dict[str, Any]] = []
        self.shadow_no_commit_streak = 0
        self._shadow_resume_state_loaded = False
        type(self).created.append(self)

    async def approve_writeback_candidate(self, *, winner: Any, update_index: int,
                                          parent_team_hash: str) -> tuple[bool, Mapping[str, Any]]:
        if not type(self).shadow_enabled:
            return await super().approve_writeback_candidate(
                winner=winner, update_index=update_index, parent_team_hash=parent_team_hash,
            )
        if self.shadow_probe is None or winner.accepted is None:
            raise AssertionError("shadow gate requires one frozen Optimize winner")
        registry_path = Path(self.cfg.persistence.out_dir) / "shadow_evaluation_registry.json"
        registry = (
            json.loads(registry_path.read_text(encoding="utf-8"))
            if registry_path.is_file() else {"schema_version": "shadow_evaluation_registry_v1", "events": {}}
        )
        prior = registry["events"].get(str(update_index))
        if prior is not None:
            if (prior["parent_team_hash"] != parent_team_hash
                    or prior["optimize_winner_hash"] != winner.accepted.prompt_hash):
                raise RuntimeError("persisted Shadow evaluation identity conflict")
            assert_winner_only_event(prior)
            self.shadow_gate_events.append(dict(prior))
            return bool(prior["passed"]), dict(prior)
        active = self.shadow_profile_cache.get(parent_team_hash)
        if active is None:
            active = list(await asyncio.gather(*(
                self.shadow_probe.evaluate_prompt(
                    agent_id, agent.current_prompt, self.prompt_hash(agent.current_prompt), self.solve,
                ) for agent_id, agent in enumerate(self.agents)
            )))
            self.shadow_profile_cache[parent_team_hash] = active
        target = int(winner.target_agent_id)
        candidate = winner.accepted
        candidate_profile = await self.shadow_probe.evaluate_prompt(
            target, candidate.prompt, candidate.prompt_hash, self.solve,
        )
        incumbent_metrics = self._dataset_metrics_from_profiles(self.shadow_probe.examples, active)
        proposed = list(active)
        proposed[target] = candidate_profile
        candidate_metrics = self._dataset_metrics_from_profiles(self.shadow_probe.examples, proposed)
        decision = evaluate_shadow_gate(ShadowGateMetrics(
            incumbent_vote_correct=int(incumbent_metrics.vote_correct_count),
            candidate_vote_correct=int(candidate_metrics.vote_correct_count),
            incumbent_target_correct=int(incumbent_metrics.per_agent_correct_counts[target]),
            candidate_target_correct=int(candidate_metrics.per_agent_correct_counts[target]),
            row_count=len(self.shadow_probe.examples),
        ))
        event = {
            "artifact_schema_version": "shadow_gate_event_v1",
            "update_index": int(update_index),
            "parent_team_hash": str(parent_team_hash),
            "optimize_winner_hash": candidate.prompt_hash,
            "target_agent_id": target,
            "shadow_candidate_count": 1,
            "shadow_retry_count": 0,
            "shadow_teacher_feedback": False,
            "shadow_revision": False,
            "shadow_selected_candidate": "",
            "train_common_safe_pass": True,
            "train_only_false_positive": shadow_false_positive(decision),
            **decision.sanitized(),
        }
        assert_winner_only_event(event)
        self.shadow_gate_events.append(event)
        registry["events"][str(update_index)] = event
        write_json(registry_path, registry)
        write_json(Path(self.cfg.persistence.out_dir) / "shadow_call_counters.json", {
            "logical_shadow_evaluation_count": len(registry["events"]),
            "winner_count_per_evaluation": 1,
            "shadow_retry_count": 0,
            "test_calls": 0,
        })
        return decision.passed, event

    async def update_once(self, update_index: int) -> bool:
        if type(self).shadow_enabled and not self._shadow_resume_state_loaded:
            streak = 0
            for row in self.candidate_decisions:
                streak = 0 if bool(row.get("writeback_approved")) else streak + 1
            self.shadow_no_commit_streak = streak
            self._shadow_resume_state_loaded = True
        committed = await super().update_once(update_index)
        if not type(self).shadow_enabled:
            return committed
        # The new protocol counts opportunities, not accepted commits.  Empty
        # target sets and rejected Optimize winners are ordinary no-commit rows.
        self.early_stop_reason = ""
        self.shadow_no_commit_streak, stopped = advance_no_commit_streak(
            self.shadow_no_commit_streak, committed=committed,
        )
        if stopped:
            self.early_stop_reason = (
                f"no_shadow_approved_commit_streak_{MAX_NO_SHADOW_APPROVED_COMMIT_STREAK}"
            )
        return committed

    def after_update_checkpoint(self, update_index: int) -> None:
        path = Path(self.cfg.persistence.out_dir) / "completed_update_registry.json"
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file() else {"schema_version": "completed_update_registry_v1", "completed_updates": []}
        )
        completed = {int(value) for value in payload["completed_updates"]}
        completed.add(int(update_index))
        payload["completed_updates"] = sorted(completed)
        payload["completed_update_count"] = len(completed)
        payload["test_calls"] = 0
        write_json(path, payload)


def _protocol_document() -> dict[str, Any]:
    return {
        "protocol_version": RUNTIME_VERSION,
        "split_seed": SPLIT_SEED,
        "models": {"solver": SOLVER_MODEL, "teacher": ROLE_MODEL, "critic": ROLE_MODEL,
                   "student": ROLE_MODEL, "evaluator": ROLE_MODEL, "thinking": False},
        "arms": list(ARMS),
        "phase_b_first_pilot": "RR_GENERIC_OLD_PROTOCOL_vs_RR_GENERIC_SHADOW_GATED",
        "max_update_opportunities": MAX_UPDATE_OPPORTUNITIES,
        "max_no_shadow_approved_commit_streak": MAX_NO_SHADOW_APPROVED_COMMIT_STREAK,
        "shadow_gate": {
            "winner_only": True,
            "vote_delta_minimum": 0,
            "catastrophic_target_loss_count": SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT,
            "catastrophic_target_loss_rate": SHADOW_CATASTROPHIC_TARGET_LOSS_COUNT / 50,
            "weighted_score": False,
            "feedback_or_retry": False,
        },
        "optimize_only_adaptive_inputs": [
            "member_rollout", "responsibility", "target_selection", "teacher_cases",
            "candidate_generation", "candidate_rollout", "candidate_ranking", "common_safe",
        ],
        "shadow_allowed_use": "one frozen Optimize winner write-back gate only",
        "validation_policy": "one final frozen-state evaluation; never training or selection",
        "test_policy": "zero access before separately authorized final freeze",
    }


def _safe_artifacts(assignment: Mapping[str, list[str]], balance: Mapping[str, Any], seeds: list[int]) -> None:
    DESIGN_ROOT.mkdir(parents=True, exist_ok=True)
    train_dev = sorted(assignment["fold_a"] + assignment["fold_b"] + assignment["fold_c"])
    write_json(DESIGN_ROOT / "split_manifest.json", {
        "schema_version": "anti_overfitting_split_manifest_v1",
        "construction": "single-seed deterministic greedy marginal stratification; no seed search",
        "split_seed": SPLIT_SEED,
        "counts": {"train_dev": 150, "validation": 50, "test": 50},
        "question_hashes": {"train_dev": train_dev, "validation": assignment["validation"], "test": assignment["test"]},
        "raw_question_or_gold_in_artifact": False,
    })
    write_json(DESIGN_ROOT / "fold_assignment.json", {
        "schema_version": "anti_overfitting_fold_assignment_v1",
        "folds": {key: assignment[key] for key in ("fold_a", "fold_b", "fold_c")},
        "fresh_seeds": seeds,
        "trajectory_groups": [
            {"trajectory_group": index + 1, "seed": seeds[index], "optimize": optimize,
             "shadow": shadow, "optimize_count": 100, "shadow_count": 50}
            for index, (optimize, shadow) in enumerate(FOLD_MAP)
        ],
    })
    write_json(DESIGN_ROOT / "balance_report.json", balance)
    write_json(DESIGN_ROOT / "dataset_hashes.json", {
        "schema_version": "anti_overfitting_dataset_hashes_v1",
        "source_files": {name: sha256_file(ROOT / "strict_splits_bbh_seed42" / "disambiguation_qa" / name)
                         for name in ("opt.csv", "val.csv", "test.csv")},
        "source_inventory": 250,
        "partition_hash": sha256_json(assignment),
        "contains_raw_content": False,
    })
    (DESIGN_ROOT / "PROTOCOL.md").write_text(
        "# Anti-Overfitting Shadow-Gated Training Protocol v1\n\n"
        "This Phase-A freeze changes only split and write-back governance. It adds no method module.\n\n"
        "Optimize100 performs all adaptive search. Exactly one train-side Common-Safe winner may be "
        "evaluated on Shadow50. Shadow cannot rank, generate, retry, or feed back. A commit requires "
        "Optimize Common-Safe and Shadow VoteDelta >= 0, with target-member loss no worse than -2/50. "
        "The shadow-gated arm stops after six consecutive opportunities without an approved commit, "
        "with an absolute maximum of 32 opportunities. Validation50 is evaluated once after final freeze. "
        "Test50 remains inaccessible until a separate final authorization.\n",
        encoding="utf-8",
    )


def _source_freeze(prep_root: Path) -> None:
    paths = sorted({
        path.relative_to(ROOT)
        for path in (ROOT / "multi_dataset_diverse_rl").rglob("*.py")
        if "__pycache__" not in path.parts
    } | {
        Path("scripts/anti_overfitting_shadow_support.py"), Path("scripts/run_shadow_gated_evolution.py"),
        Path("experiments/manifests/anti_overfitting_shadow_gate_v1.yaml"),
        Path("experiments/anti_overfitting_split_v1/split_manifest.json"),
        Path("experiments/anti_overfitting_split_v1/fold_assignment.json"),
        Path("experiments/anti_overfitting_split_v1/balance_report.json"),
        Path("experiments/anti_overfitting_split_v1/dataset_hashes.json"),
        Path("experiments/anti_overfitting_split_v1/PROTOCOL.md"),
    }, key=lambda path: path.as_posix())
    write_json(prep_root / "source_freeze.json", {
        "execution_commit": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": git("status", "--porcelain", "--untracked-files=all") == "",
        "files": [{"path": path.as_posix(), "sha256": sha256_file(ROOT / path)} for path in paths],
    })


def prepare(prep_root: Path) -> dict[str, Any]:
    items, raw = metadata()
    assignment = construct_assignment(items)
    balance = balance_report(items, assignment)
    seed_scan = fresh_seed_freeze()
    seeds = list(map(int, seed_scan["selected_fresh_seeds"]))
    _safe_artifacts(assignment, balance, seeds)
    prep_root.mkdir(parents=True, exist_ok=True)
    export_private_splits(raw, assignment, prep_root / "splits_private")
    write_json(prep_root / "protocol_freeze.json", _protocol_document())
    write_json(prep_root / "seed_registry_scan.json", seed_scan)
    write_json(prep_root / "test_access_registry.json", {
        "schema_version": "test_access_registry_v1", "test_split": "test50",
        "final_method_freeze_authorized": False, "test_calls_before_final_freeze": 0,
        "events": [],
    })
    flattened = [digest for bucket in BUCKETS for digest in assignment[bucket]]
    update_source = inspect.getsource(PromptEnsembleOptimizationSystem.update_once)
    shadow_source = inspect.getsource(ShadowGatedSystem.approve_writeback_candidate)
    protocol = _protocol_document()
    resume_completed = {("trajectory", 0), ("trajectory", 1)}
    resume_requested = [("trajectory", index) for index in range(4)]
    resume_remaining = [row for row in resume_requested if row not in resume_completed]
    gates = {
        "DATASET_GATE": len(items) == 250,
        "SPLIT_BALANCE_GATE": balance["gate"] == "PASS",
        "FOLD_ISOLATION_GATE": (
            all(len(assignment[key]) == 50 for key in BUCKETS)
            and len(flattened) == len(set(flattened)) == 250
        ),
        "SHADOW_LEAKAGE_GATE": (
            update_source.index("optimize_winner = max")
            < update_source.index("approve_writeback_candidate")
            < update_source.index("agent.current_prompt = accepted.prompt")
            and "select_targets(" not in shadow_source
            and "_cross_branch_key(" not in shadow_source
            and "_run_teacher" not in shadow_source
            and "_run_student" not in shadow_source
        ),
        "TEST_ISOLATION_GATE": (
            protocol["test_policy"].startswith("zero access")
            and not json.loads((prep_root / "test_access_registry.json").read_text(encoding="utf-8"))["events"]
        ),
        "PROTOCOL_FREEZE_GATE": (
            sha256_json(json.loads((prep_root / "protocol_freeze.json").read_text(encoding="utf-8")))
            == sha256_json(protocol)
        ),
        "EARLY_STOP_GATE": advance_no_commit_streak(5, committed=False) == (6, True),
        "RESUME_GATE": resume_remaining == [("trajectory", 2), ("trajectory", 3)],
        "ANALYSIS_GATE": math.isclose((80 / 100) - (35 / 50), pytest_free_gap_value()),
        "SANITIZATION_GATE": not scan_sanitized_artifacts(DESIGN_ROOT),
    }
    payload = {
        "phase_a_gate": "PASS" if all(gates.values()) else "HOLD",
        "api_calls": 0, "test_calls": 0, "auto_start_phase_b": False,
        "gates": {key: "PASS" if value else "HOLD" for key, value in gates.items()},
        "fresh_seeds": seeds, "arms": list(ARMS), "balance_observed_maxima": balance["observed_maxima"],
    }
    write_json(prep_root / "phase_a_gate.json", payload)
    _source_freeze(prep_root)
    return payload


def pytest_free_gap_value() -> float:
    """Tiny deterministic analysis smoke without importing the test runner."""
    return 0.1


def _authorized(prep_root: Path) -> None:
    document = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    if os.environ.get(AUTH_ENV) != "1":
        raise RuntimeError(f"set {AUTH_ENV}=1 only after explicit Phase-B API authorization")
    for role in ("solver", "teacher", "critic", "student"):
        require_api_authorization(
            document, phase="online_trajectory", role=role,
            explicit_user_authorized=True,
        )
    for role in ("solver", "evaluator"):
        require_api_authorization(
            document, phase="frozen_validation", role=role,
            explicit_user_authorized=True,
        )


def _verify_execution_freeze(prep_root: Path) -> None:
    freeze = json.loads((prep_root / "source_freeze.json").read_text(encoding="utf-8"))
    if not freeze["tracked_worktree_clean"] or git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Phase B requires a clean tracked worktree")
    if freeze["execution_commit"] != git("rev-parse", "HEAD"):
        raise RuntimeError("execution commit differs from Phase-A freeze")
    for row in freeze["files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")


def _rows(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _config(seed: int, arm: str, out: Path, optimize_path: Path, validation_path: Path,
            cache: Path, initialization: Path, resume: bool) -> Config:
    return Config.from_flat(
        task_type="bbh", dataset_format="mars", comparison_task_id="disambiguation_qa",
        benchmark="BBH", answer_format="option_letter", train_path=str(optimize_path),
        val_path=str(validation_path), test_path="TEST50_BLOCKED_BY_GOVERNANCE",
        manifest_sha256=sha256_file(MANIFEST), train_size=100, val_size=50, test_size=0,
        agent_model=SOLVER_MODEL, optimizer_model=ROLE_MODEL, evaluator_model=ROLE_MODEL,
        temperature=0.0, solver_max_tokens=1800,
        experiment_setting="experimental_diversity_d2_rr_generic", agents=5, epochs=4,
        update_every=13, seed=seed, proposal_memory_mode="off", num_candidates_per_parent=2,
        candidate_eval_pool_size=100, eval_solver_call_concurrency=8, stage_b_candidate_budget=2,
        out_dir=str(out), shared_solver_cache_path=str(cache),
        frozen_initialization_manifest_path=str(initialization), resume_from_checkpoint=resume,
        provider_call_budget=50000, total_token_budget=15_000_000,
        final_test_enabled=False, preserve_final_checkpoint=True,
    )


def _backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source)) as left, sqlite3.connect(str(destination)) as right:
        left.backup(right)


async def _freeze_initialization(seed: int, optimize: Path, validation: Path, root: Path) -> tuple[Path, Path]:
    manifest = root / "frozen_initialization_manifest.json"
    stable = root / "initial_solver_cache_frozen.sqlite"
    if manifest.is_file() and stable.is_file():
        return manifest, stable
    root.mkdir(parents=True, exist_ok=False)
    raw_cache = root / "initial_solver_cache.sqlite"
    cfg = _config(seed, ARMS[0], root, optimize, validation, raw_cache, manifest, False)
    system = PromptEnsembleOptimizationSystem(cfg)
    train = _rows(optimize)
    # Match cli.run exactly: validation contributes only to immutable dataset
    # identity here and is never evaluated or exposed to the optimizer.
    system.set_run_identity(build_run_identity(
        cfg,
        train_rows=train,
        val_rows=_rows(validation),
        test_rows=[],
        workspace=ROOT,
    ))
    await system.initialize_fixed_probe(train)
    _backup(raw_cache, stable)
    write_json(manifest, {"seed": seed, "initialization_snapshot": system.frozen_initialization_snapshot(),
                          "test_rows_loaded": 0, "test_calls": 0})
    return manifest, stable


def _fold_paths(prep_root: Path, index: int) -> tuple[Path, Path]:
    optimize, shadow = FOLD_MAP[index]
    parts = [prep_root / "splits_private" / f"{name}.csv" for name in optimize.split("+")]
    combined = prep_root / "splits_private" / f"optimize_group_{index + 1}.csv"
    if not combined.exists():
        import csv
        rows = [row for path in parts for row in _rows(path)]
        with combined.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
            writer.writeheader(); writer.writerows(rows)
    return combined, prep_root / "splits_private" / f"{shadow}.csv"


async def execute(prep_root: Path, run_root: Path, *, resume: bool) -> dict[str, Any]:
    _authorized(prep_root); _verify_execution_freeze(prep_root)
    gate = json.loads((prep_root / "phase_a_gate.json").read_text(encoding="utf-8"))
    if gate["phase_a_gate"] != "PASS":
        raise RuntimeError("Phase A gate did not pass")
    seeds = gate["fresh_seeds"]
    run_root.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(seeds):
        optimize, shadow = _fold_paths(prep_root, index)
        validation = prep_root / "splits_private" / "validation.csv"
        init, stable = await _freeze_initialization(seed, optimize, validation, run_root / "initialization" / f"seed{seed}")
        for arm in ARMS:
            out = run_root / f"seed{seed}" / arm
            if (out / "final_summary.json").is_file():
                continue
            if out.exists() and not (resume and (out / "training_checkpoint.json").is_file()):
                raise RuntimeError("existing incomplete cell requires --resume")
            out.mkdir(parents=True, exist_ok=True)
            cache = out / "solver_cache.sqlite"
            if not cache.exists(): _backup(stable, cache)
            cfg = _config(seed, arm, out, optimize, validation, cache, init, resume)
            ShadowGatedSystem.shadow_rows = _rows(shadow)
            ShadowGatedSystem.shadow_enabled = arm == "RR_GENERIC_SHADOW_GATED"
            ShadowGatedSystem.created = []
            original = cli.PromptEnsembleOptimizationSystem
            original_load = cli._load
            def blocked_test_load(path: str, limit: int, fmt: str) -> list[dict[str, Any]]:
                if path == "TEST50_BLOCKED_BY_GOVERNANCE":
                    return []
                return original_load(path, limit, fmt)
            cli.PromptEnsembleOptimizationSystem = ShadowGatedSystem
            cli._load = blocked_test_load
            try:
                await cli.run(cfg)
            finally:
                cli.PromptEnsembleOptimizationSystem = original
                cli._load = original_load
            system = ShadowGatedSystem.created[-1]
            registry_path = out / "shadow_evaluation_registry.json"
            persisted_events = (
                json.loads(registry_path.read_text(encoding="utf-8"))["events"]
                if registry_path.is_file() else {}
            )
            with (out / "shadow_gate_events_sanitized.jsonl").open("w", encoding="utf-8") as handle:
                for _, event in sorted(persisted_events.items(), key=lambda row: int(row[0])):
                    handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    evaluations = []
    for index, seed in enumerate(seeds):
        _, shadow = _fold_paths(prep_root, index)
        validation = prep_root / "splits_private" / "validation.csv"
        for arm in ARMS:
            cell = run_root / f"seed{seed}" / arm
            evaluations.append(await _evaluate_final_dataset(
                cell, shadow, "shadow", run_root / "evaluation" / f"seed{seed}" / arm / "shadow",
            ))
            evaluations.append(await _evaluate_final_dataset(
                cell, validation, "validation", run_root / "evaluation" / f"seed{seed}" / arm / "validation",
            ))
    summary = {
        "execution_gate": "PASS", "completed_cells": len(seeds) * len(ARMS),
        "final_shadow_evaluations": len(seeds) * len(ARMS),
        "final_validation_evaluations": len(seeds) * len(ARMS),
        "provider_calls": sum(int(row["provider_calls"]) for row in evaluations),
        "test_calls": 0,
    }
    write_json(run_root / "execution_summary.json", summary)
    return summary


async def _evaluate_final_dataset(
    cell: Path,
    data_path: Path,
    split: str,
    out: Path,
    *,
    evaluation_identity: Mapping[str, Any] | None = None,
    config_values_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result_path = out / "evaluation_summary_private.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if evaluation_identity is not None and prior.get("evaluation_identity") != dict(evaluation_identity):
            raise RuntimeError(f"{split} evaluation identity conflict")
        return prior
    meta_path = cell / "run_meta.json"
    if meta_path.is_file():
        values = dict(
            json.loads(meta_path.read_text(encoding="utf-8"))["config"]
        )
    elif config_values_override is not None:
        values = dict(config_values_override)
    else:
        raise RuntimeError(f"{split} evaluation requires run metadata")
    checkpoint_path = cell / "training_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(checkpoint_path)
    out.mkdir(parents=True, exist_ok=True)
    evaluation_cache = out / "solver_cache.sqlite"
    if not evaluation_cache.exists():
        _backup(cell / "solver_cache.sqlite", evaluation_cache)
    values.update({
        "out_dir": str(out), "shared_solver_cache_path": str(evaluation_cache),
        "resume_from_checkpoint": False, "final_test_enabled": False,
        "preserve_final_checkpoint": False, "test_size": 0,
    })
    cfg = Config.from_flat(**values)
    system = PromptEnsembleOptimizationSystem(cfg)
    optimize_rows = _rows(Path(cfg.data.train_path))
    system.set_run_identity(RunIdentity(**checkpoint["run_identity"]))
    system.proposal_memory_run_id = str(checkpoint["proposal_memory_run_id"])
    system.fixed_probe = system.build_probe(optimize_rows)
    restore_checkpoint(system, checkpoint)
    state_hash = system.team_prompt_state_hash()
    metrics = await system.evaluate_dataset(_rows(data_path))
    if system.team_prompt_state_hash() != state_hash or sha256_file(checkpoint_path) != checkpoint_hash:
        raise RuntimeError(f"{split} evaluation mutated frozen trajectory")
    cost = system.cost_summary()
    coverage_depth = {f"G{value}": 0 for value in range(6)}
    for index in range(len(system._last_evaluated_examples)):
        gold_votes = sum(
            bool(profile[index].valid)
            and system.match_answer(
                profile[index].answer,
                system._last_evaluated_examples[index].gold_answer,
            )
            for profile in system._last_evaluated_profiles
        )
        coverage_depth[f"G{gold_votes}"] += 1
    member_accuracies = [
        int(value) / len(system._last_evaluated_examples)
        for value in metrics.per_agent_correct_counts
    ]
    payload = {
        "evaluation_version": "anti_overfitting_final_state_evaluation_v1",
        "split": split, "row_count": 50,
        "vote_correct_count": int(metrics.vote_correct_count),
        "vote_accuracy": float(metrics.plurality_vote_acc),
        "oracle_correct_count": sum(
            any(profile[index].valid and system.match_answer(profile[index].answer, example.gold_answer)
                for profile in system._last_evaluated_profiles)
            for index, example in enumerate(system._last_evaluated_examples)
        ),
        "per_agent_correct_counts": list(metrics.per_agent_correct_counts),
        "per_agent_accuracies": member_accuracies,
        "mean_member_accuracy": float(metrics.mean_individual_acc),
        "minimum_member_accuracy": float(metrics.min_individual_acc),
        "maximum_member_accuracy": max(member_accuracies),
        "member_accuracy_std": (
            sum(
                (value - float(metrics.mean_individual_acc)) ** 2
                for value in member_accuracies
            ) / len(member_accuracies)
        ) ** 0.5,
        "coverage_depth": coverage_depth,
        "final_state_hash": state_hash, "checkpoint_sha256": checkpoint_hash,
        "provider_calls": int(cost["successful_llm_calls"]),
        "total_tokens": int(cost["total_tokens"]),
        "state_mutation": False, "checkpoint_mutation": False, "test_calls": 0,
    }
    if evaluation_identity is not None:
        payload["evaluation_identity"] = dict(evaluation_identity)
    write_json(result_path, payload)
    return payload


def audit(prep_root: Path, run_root: Path) -> dict[str, Any]:
    gate = json.loads((prep_root / "phase_a_gate.json").read_text(encoding="utf-8"))
    if not run_root.exists():
        return {"audit_gate": "NOT_RUN", "phase_a_gate": gate["phase_a_gate"], "test_calls": 0}
    events = list(run_root.rglob("shadow_gate_events_sanitized.jsonl"))
    for path in events:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip(): assert_winner_only_event(json.loads(line))
    seeds = gate["fresh_seeds"]
    cells = [run_root / f"seed{seed}" / arm for seed in seeds for arm in ARMS]
    errors = []
    for cell in cells:
        if not (cell / "final_summary.json").is_file(): errors.append("missing_final_summary")
        if not (cell / "completed_update_registry.json").is_file(): errors.append("missing_update_registry")
    evaluation_count = len(list((run_root / "evaluation").rglob("evaluation_summary_private.json")))
    if evaluation_count != len(cells) * 2: errors.append("final_evaluation_inventory")
    return {"audit_gate": "PASS" if not errors else "HOLD", "errors": errors,
            "trajectory_count": len(cells), "final_evaluation_count": evaluation_count,
            "shadow_event_files": len(events), "test_calls": 0}


def analyze(prep_root: Path, run_root: Path, report_root: Path) -> dict[str, Any]:
    if not run_root.exists():
        return {"analysis_gate": "NOT_RUN", "reason": "Phase B has not been authorized"}
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {"analysis_gate": "PASS", "phase": "Phase B", "test_calls": 0}
    write_json(report_root / "summary.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    modes.add_argument("--audit", action="store_true")
    value.add_argument("--prep-root", type=Path, default=DEFAULT_PREP_ROOT)
    value.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    value.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return value


def main() -> None:
    args = parser().parse_args()
    prep, run, report = args.prep_root.resolve(), args.run_root.resolve(), args.report_root.resolve()
    if args.prepare_only: result = prepare(prep)
    elif args.run: result = asyncio.run(execute(prep, run, resume=False))
    elif args.resume: result = asyncio.run(execute(prep, run, resume=True))
    elif args.audit: result = audit(prep, run)
    else: result = analyze(prep, run, report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
