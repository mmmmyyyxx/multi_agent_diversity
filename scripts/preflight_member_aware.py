from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.cli import build_dataset
from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import PersistentSolverCache
from multi_dataset_diverse_rl.persistence.identity import build_run_identity, validate_run_identity
from multi_dataset_diverse_rl.protocol import CandidateBudgetContract, experiment_protocol
from multi_dataset_diverse_rl.task_manifest import load_task_manifest, resolve_task_ids
from multi_dataset_diverse_rl.utils import load_jsonl
from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTING_NAMES, select_settings
from scripts.run_task_level_accuracy import RUNNER_FIELDS, _task_split_integrity


EXPECTED_SETTINGS = [
    "shared_baseline",
    "shared_independent_accuracy",
    "shared_peer_state_vote_first",
    "shared_peer_state_member_pareto",
    "shared_member_aware_responsibility",
    "shared_member_aware_full",
]


def preflight(workspace: Path, allow_dirty: bool = False) -> dict:
    errors = []
    configs = [Config.from_flat(**setting.resolved_overrides()) for setting in select_settings("all")]
    if DEFAULT_EXPERIMENT_SETTING_NAMES != EXPECTED_SETTINGS:
        errors.append("experiment settings do not match the frozen six-setting protocol")
    for cfg in configs:
        if cfg.training.method_version != "member_aware_peer_state_v1":
            errors.append(f"unexpected method version: {cfg.training.method_version}")
        if cfg.training.agents != 5 or cfg.peer_state.aggregation_mode != "plurality":
            errors.append("all settings must use five equal-weight plurality voters")
        if cfg.peer_state.vote_tie_break != "abstain":
            errors.append("all canonical settings must use tie-as-abstain")
    budget = CandidateBudgetContract(2, 2, 2, 12, 6, 6, 4)
    protocols = {
        name: experiment_protocol(
            name,
            initialization_mode="shared_identical",
            tie_policy="abstain",
            candidate_budget_contract=budget,
        )
        for name in EXPECTED_SETTINGS
    }
    responsibility = protocols["shared_member_aware_responsibility"]
    full = protocols["shared_member_aware_full"]
    responsibility_payload = responsibility.__dict__ | {
        "name": full.name,
        "tcs_context_policy": full.tcs_context_policy,
    }
    if responsibility_payload != full.__dict__:
        errors.append(
            "member-aware responsibility and full settings must differ only in TCS context"
        )
    help_result = subprocess.run(
        [sys.executable, "scripts/run_task_level_accuracy.py", "--help"],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if help_result.returncode != 0:
        errors.append("task runner parser failed to build")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=workspace, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"ok": False, "errors": [f"git inspection failed: {exc}"]}
    if dirty and not allow_dirty:
        errors.append("git working tree is not clean")
    return {
        "ok": not errors, "git_commit": head, "git_dirty": dirty,
        "method_version": "member_aware_peer_state_v1", "settings": EXPECTED_SETTINGS,
        "legacy_compatibility_enabled": False, "errors": errors,
    }


def _role_environment(cfg: Config, role: str) -> dict[str, Any]:
    key_env = getattr(cfg.models, f"{role}_api_key_env")
    base_env = getattr(cfg.models, f"{role}_base_url_env")
    return {
        "key_env": key_env or "OPENAI_API_KEY",
        "base_url_env": base_env or "OPENAI_BASE_URL/OPENAI_API_BASE",
        "key_present": bool(os.getenv(key_env) if key_env else os.getenv("OPENAI_API_KEY")),
        "base_url": (
            os.getenv(base_env, "")
            if base_env
            else os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", ""))
        ),
    }


def run_specific_preflight(args: argparse.Namespace, workspace: Path) -> dict:
    errors: list[str] = []
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = workspace / manifest_path
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"manifest does not exist: {manifest_path}"], "runs": []}
    tasks = load_task_manifest(str(manifest_path))
    task_ids = resolve_task_ids(args.tasks, tasks, args.benchmarks)
    settings = select_settings(args.settings)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        errors.append("at least one seed is required")
    root = Path(args.out_root)
    if not root.is_absolute():
        root = workspace / root
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    defaults = Config().to_flat_dict()
    run_reports = []
    for task_id in task_ids:
        task = tasks[task_id]
        try:
            integrity = _task_split_integrity(task, args.dataset_format, str(workspace))
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{task_id}: split integrity failed: {exc}")
            continue
        for setting in settings:
            for seed in seeds:
                run_dir = root / task_id / f"{setting.name}_seed{seed}"
                values = {
                    **setting.resolved_overrides(),
                    "task_type": task.task_type,
                    "dataset_format": args.dataset_format,
                    "comparison_task_id": task.task_id,
                    "benchmark": task.benchmark,
                    "answer_format": task.answer_format,
                    "train_path": str((workspace / task.train_path).resolve()),
                    "val_path": str((workspace / task.val_path).resolve()),
                    "test_path": str((workspace / task.test_path).resolve()),
                    "manifest_sha256": manifest_sha,
                    "out_dir": str(run_dir),
                    "shared_solver_cache_path": str(root / "_shared_solver_cache.sqlite"),
                    "seed": seed,
                }
                for name in RUNNER_FIELDS:
                    value = getattr(args, name)
                    if value is not None:
                        values[name] = bool(value) if isinstance(defaults[name], bool) else value
                try:
                    cfg = Config.from_flat(**values)
                    if any(not model.strip() for model in (
                        cfg.models.agent_model, cfg.models.optimizer_model, cfg.models.evaluator_model,
                    )):
                        raise ValueError("solver, optimizer, and evaluator model names must be non-empty")
                    role_environment = {
                        role: _role_environment(cfg, role) for role in ("solver", "optimizer", "evaluator")
                    }
                    for role, environment in role_environment.items():
                        if not environment["key_present"]:
                            raise ValueError(f"{role} API key is unavailable via {environment['key_env']}")
                        if not environment["base_url"]:
                            raise ValueError(f"{role} base URL is unavailable via {environment['base_url_env']}")
                    for split in ("train", "val", "test"):
                        requested = getattr(cfg.data, f"{split}_size")
                        available = int(integrity[f"{'opt' if split == 'train' else split}_count"])
                        if requested <= 0 or requested > available:
                            raise ValueError(
                                f"{split}_size={requested} must be within available count {available}"
                            )
                    if cfg.tcs.num_candidates_per_parent <= 0:
                        raise ValueError("num_candidates_per_parent must be positive")
                    if not 0 < cfg.evaluation.stage_b_candidate_budget <= cfg.tcs.num_candidates_per_parent:
                        raise ValueError("stage_b_candidate_budget must be within generated candidate count")
                    if min(
                        cfg.tcs.tcs_assigned_coverage_limit,
                        cfg.tcs.tcs_assigned_conversion_limit,
                        cfg.tcs.tcs_preservation_limit,
                        cfg.tcs.tcs_representative_limit,
                        cfg.tcs.tcs_member_error_limit,
                        cfg.tcs.tcs_context_max_chars,
                    ) <= 0:
                        raise ValueError("all TCS context limits must be positive")
                    if cfg.persistence.max_total_llm_calls <= 0 or cfg.persistence.max_total_tokens <= 0:
                        raise ValueError("run-specific preflight requires positive LLM call and token budgets")
                    cache_path = Path(cfg.persistence.shared_solver_cache_path)
                    if not cache_path.is_absolute():
                        raise ValueError("shared_solver_cache_path must resolve to an absolute path")
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    PersistentSolverCache(cache_path).ready_entry_count()
                    split_rows = {
                        "train": build_dataset(load_jsonl(cfg.data.train_path, cfg.data.train_size), cfg.data.dataset_format),
                        "val": build_dataset(load_jsonl(cfg.data.val_path, cfg.data.val_size), cfg.data.dataset_format),
                        "test": build_dataset(load_jsonl(cfg.data.test_path, cfg.data.test_size), cfg.data.dataset_format),
                    }
                    identity = build_run_identity(
                        cfg,
                        train_rows=split_rows["train"],
                        val_rows=split_rows["val"],
                        test_rows=split_rows["test"],
                        workspace=workspace,
                    )
                    for artifact_name in ("run_meta.json", "training_checkpoint.json"):
                        artifact = run_dir / artifact_name
                        if artifact.exists():
                            payload = json.loads(artifact.read_text(encoding="utf-8"))
                            actual = payload["run_identity"]
                            validate_run_identity(identity, actual)
                    if run_dir.exists() and any(run_dir.iterdir()) and not any(
                        (run_dir / name).exists() for name in ("run_meta.json", "training_checkpoint.json")
                    ):
                        raise ValueError("non-empty output directory has no run identity artifact")
                    run_reports.append({
                        "task": task_id,
                        "setting": setting.name,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "run_identity": identity.to_dict(),
                        "shared_solver_cache_path": str(cache_path),
                        "split_integrity": integrity,
                        "role_environment": role_environment,
                    })
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{task_id}/{setting.name}/seed{seed}: {exc}")
    return {"ok": not errors, "errors": errors, "runs": run_reports}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--allow_dirty", type=int, choices=[0, 1], default=0)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--settings", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--dataset_format", default="mars")
    parser.add_argument("--out_root", default="")
    defaults = Config().to_flat_dict()
    for name in RUNNER_FIELDS:
        default = defaults[name]
        parser.add_argument(f"--{name}", type=int if isinstance(default, bool) else type(default), default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = args.workspace.resolve()
    report = preflight(workspace, bool(args.allow_dirty))
    if args.manifest:
        if not args.out_root:
            report["errors"].append("--out_root is required with --manifest")
            report["ok"] = False
        else:
            run_report = run_specific_preflight(args, workspace)
            report["run_specific"] = run_report
            report["errors"].extend(run_report["errors"])
            report["ok"] = report["ok"] and run_report["ok"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
