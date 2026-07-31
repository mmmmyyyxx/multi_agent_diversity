"""Legacy v6/v7 frozen-initialization A/B/C mechanism runner.

This runner deliberately launches exactly one treatment at a time.  A caller
must inspect the prior treatment's mechanism gate before launching the next
one, so a failed short pilot cannot silently progress to the 32-update phase.
Run it only from its original pinned v7 commit. The active v8 runtime
intentionally rejects its legacy responsibility modes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.cli import _load, run
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.identity import build_run_identity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.task_manifest import load_task_manifest


MANIFEST_VERSION = "frontier_matched_frozen_initialization_v1"
PILOT_VERSION = "frontier_matched_pilot_v1"
TREATMENTS = {
    "A": {
        "label": "v6-owner",
        "responsibility_mode": "unique_owner_v6",
        "member_catchup_mode": "off",
    },
    "B": {
        "label": "v7-frontier-core",
        "responsibility_mode": "frontier_joint_v7",
        "member_catchup_mode": "off",
    },
    "C": {
        "label": "v7-full",
        "responsibility_mode": "frontier_joint_v7",
        "member_catchup_mode": "fallback_v1",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or run one matched frontier-responsibility treatment."
    )
    parser.add_argument("--action", choices=("freeze", "run"), required=True)
    parser.add_argument("--treatment", choices=tuple(TREATMENTS), default="A")
    parser.add_argument("--phase", choices=("short", "full"), default="short")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        default="configs/task_level_comparison_strict_bbh_seed42.yaml",
    )
    parser.add_argument("--task", default="disambiguation_qa")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--run_suffix", default="")
    parser.add_argument("--refresh_frozen_manifest", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dry_run", type=int, choices=(0, 1), default=0)
    return parser


def _root(workspace: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _phase_values(phase: str) -> dict[str, Any]:
    # Four fixed updates per epoch: steps 24/48/72/75.  The short phase is
    # exactly two such epochs (eight updates); the full phase is eight (32).
    return {
        "epochs": 2 if phase == "short" else 8,
        "update_every": 24,
        "final_test_enabled": phase == "full",
    }


def _base_values(
    *, workspace: Path, manifest_path: Path, task_id: str, seed: int,
    out_dir: Path, cache_path: Path, treatment: str,
    frozen_manifest: Path | None = None, phase: str = "short",
) -> dict[str, Any]:
    tasks = load_task_manifest(str(manifest_path))
    if task_id not in tasks:
        raise ValueError(f"unknown task: {task_id}")
    task = tasks[task_id]
    treatment_values = TREATMENTS[treatment]
    return {
        "method_version": "member_aware_peer_state_v7",
        "experiment_setting": "shared_member_aware_full",
        "task_type": task.task_type,
        "dataset_format": "mars",
        "comparison_task_id": task.task_id,
        "benchmark": task.benchmark,
        "answer_format": task.answer_format,
        "train_path": str((workspace / task.train_path).resolve()),
        "val_path": str((workspace / task.val_path).resolve()),
        "test_path": str((workspace / task.test_path).resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_size": 75,
        "val_size": 50,
        "test_size": 125,
        "seed": seed,
        "agents": 5,
        "initialization_mode": "shared_identical",
        "proposal_memory_mode": "off",
        "responsibility_mode": treatment_values["responsibility_mode"],
        "member_catchup_mode": treatment_values["member_catchup_mode"],
        "responsibility_max_wait_updates": 8,
        "member_uplift_tolerance": 5,
        "candidate_eval_pool_size": 75,
        "num_candidates_per_parent": 2,
        "stage_b_candidate_budget": 2,
        "eval_solver_call_concurrency": 20,
        "out_dir": str(out_dir),
        "shared_solver_cache_path": str(cache_path),
        "resume_from_checkpoint": False,
        "frozen_initialization_manifest_path": (
            str(frozen_manifest.resolve()) if frozen_manifest is not None else ""
        ),
        **_phase_values(phase),
    }


def _config(**kwargs: Any) -> Config:
    return Config.from_flat(**kwargs)


async def _freeze(cfg: Config, workspace: Path) -> dict[str, Any]:
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    test = _load(cfg.data.test_path, cfg.data.test_size, cfg.data.dataset_format)
    system = PromptEnsembleOptimizationSystem(cfg)
    system.set_run_identity(build_run_identity(
        cfg, train_rows=train, val_rows=validation, test_rows=test, workspace=workspace,
    ))
    await system.initialize_fixed_probe(train)
    return system.frozen_initialization_snapshot()


def _sqlite_backup(source: Path, destination: Path) -> None:
    """Create a stable, standalone SQLite snapshot rather than copying WAL files."""
    with sqlite3.connect(str(source)) as read_connection:
        with sqlite3.connect(str(destination)) as write_connection:
            read_connection.backup(write_connection)


def _freeze_action(args: argparse.Namespace, workspace: Path, root: Path, manifest_path: Path) -> None:
    frozen_root = root / f"frozen_seed{args.seed}"
    cache = frozen_root / "initial_solver_cache.sqlite"
    stable_cache = frozen_root / "initial_solver_cache_frozen.sqlite"
    manifest_path_out = frozen_root / "frozen_initialization_manifest.json"
    recoverable_partial = (
        frozen_root.is_dir() and cache.is_file() and stable_cache.is_file()
        and not manifest_path_out.exists()
    )
    refreshable = (
        bool(args.refresh_frozen_manifest) and frozen_root.is_dir()
        and cache.is_file() and stable_cache.is_file() and manifest_path_out.is_file()
    )
    if frozen_root.exists() and not recoverable_partial and not refreshable:
        raise FileExistsError(f"frozen initialization root must be new: {frozen_root}")
    cfg = _config(**_base_values(
        workspace=workspace, manifest_path=manifest_path, task_id=args.task,
        seed=args.seed, out_dir=frozen_root, cache_path=cache, treatment="A",
    ))
    if args.dry_run:
        print(json.dumps({"action": "freeze", "config": cfg.to_flat_dict()}, indent=2))
        return
    if not frozen_root.exists():
        frozen_root.mkdir(parents=True)
    snapshot = asyncio.run(_freeze(cfg, workspace))
    if not stable_cache.exists():
        _sqlite_backup(cache, stable_cache)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "pilot_version": PILOT_VERSION,
        "seed": args.seed,
        "task_id": args.task,
        "initialization_snapshot": snapshot,
        "initial_cache_sha256": hashlib.sha256(stable_cache.read_bytes()).hexdigest(),
        "initial_cache_ready": True,
    }
    manifest_path_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "action": "freeze", "frozen_root": str(frozen_root),
        "initial_train_state_hash": snapshot["initial_train_state_hash"],
        "initial_cache_sha256": manifest["initial_cache_sha256"],
    }, indent=2), flush=True)


def _run_action(args: argparse.Namespace, workspace: Path, root: Path, manifest_path: Path) -> None:
    frozen_root = root / f"frozen_seed{args.seed}"
    frozen_manifest = frozen_root / "frozen_initialization_manifest.json"
    frozen_cache = frozen_root / "initial_solver_cache_frozen.sqlite"
    if not frozen_manifest.is_file() or not frozen_cache.is_file():
        raise FileNotFoundError("freeze must complete before a treatment can start")
    label = TREATMENTS[args.treatment]["label"]
    suffix = str(args.run_suffix).strip()
    if suffix and not suffix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run_suffix may contain only letters, digits, underscores, and hyphens")
    run_name = f"{label}_seed{args.seed}_{args.phase}{'_' + suffix if suffix else ''}"
    run_root = root / run_name
    if run_root.exists():
        raise FileExistsError(f"treatment output root must be new: {run_root}")
    mutable_cache = run_root / "_shared_solver_cache.sqlite"
    cfg = _config(**_base_values(
        workspace=workspace, manifest_path=manifest_path, task_id=args.task,
        seed=args.seed, out_dir=run_root / args.task / f"shared_member_aware_full_seed{args.seed}",
        cache_path=mutable_cache, treatment=args.treatment,
        frozen_manifest=frozen_manifest, phase=args.phase,
    ))
    run_manifest = {
        "pilot_version": PILOT_VERSION,
        "treatment": args.treatment,
        "treatment_label": label,
        "phase": args.phase,
        "run_suffix": suffix,
        "seed": args.seed,
        "task_id": args.task,
        "frozen_initialization_manifest_sha256": hashlib.sha256(
            frozen_manifest.read_bytes()
        ).hexdigest(),
        "expected_updates": cfg.training.epochs * 4,
        "final_test_enabled": cfg.persistence.final_test_enabled,
        "proposal_memory_mode": cfg.tcs.proposal_memory_mode,
        "responsibility_mode": cfg.responsibility.responsibility_mode,
        "member_catchup_mode": cfg.responsibility.member_catchup_mode,
    }
    if args.dry_run:
        print(json.dumps({"action": "run", "run_manifest": run_manifest, "config": cfg.to_flat_dict()}, indent=2))
        return
    run_root.mkdir(parents=True)
    shutil.copy2(frozen_cache, mutable_cache)
    (run_root / "matched_pilot_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    result = asyncio.run(run(cfg))
    print(json.dumps({
        "action": "run", "run_root": str(run_root),
        "selected_test_present": result["selected_test"] is not None,
        "selection_summary": result["selection_summary"],
    }, indent=2), flush=True)


def main() -> None:
    args = _parser().parse_args()
    workspace = args.workspace.resolve()
    manifest_path = Path(args.manifest)
    manifest_path = (
        manifest_path.resolve()
        if manifest_path.is_absolute() else (workspace / manifest_path).resolve()
    )
    root = _root(workspace, args.out_root)
    if args.action == "freeze":
        _freeze_action(args, workspace, root, manifest_path)
    else:
        _run_action(args, workspace, root, manifest_path)


if __name__ == "__main__":
    main()
