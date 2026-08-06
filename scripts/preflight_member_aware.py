from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    validate_mutable_decision_procedure,
)
from multi_dataset_diverse_rl.persistence.identity import build_run_identity, validate_run_identity
from multi_dataset_diverse_rl.provider_credentials import (
    resolve_api_key,
    resolve_base_url,
)
from multi_dataset_diverse_rl.protocol import CandidateBudgetContract, experiment_protocol
from multi_dataset_diverse_rl.task_manifest import load_task_manifest, resolve_task_ids
from multi_dataset_diverse_rl.tcs import TCS_PROTOCOL_VERSION
from multi_dataset_diverse_rl.utils import load_jsonl
from multi_dataset_diverse_rl.versions import (
    CANDIDATE_ACCEPTANCE_VERSION,
    CANDIDATE_SELECTION_VERSION,
    CANDIDATE_PROTOCOL_FILTER_VERSION,
    CHECKPOINT_SELECTION_VERSION,
    CHECKPOINT_VERSION,
    EVALUATION_PROTOCOL_VERSION,
    METHOD_VERSION,
    MUTABLE_PROMPT_CONTRACT_VERSION,
    PRESERVATION_POLICY_VERSION,
    PROPOSAL_MEMORY_VERSION,
    SERVICE_ROUTING_VERSION,
    STUDENT_INVALID_RECOVERY_VERSION,
    STUDENT_PROMPT_CONTRACT_VERSION,
    TARGET_SELECTION_VERSION,
    TCS_CONTEXT_VERSION,
    TEST_ISOLATION_VERSION,
)
from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTING_NAMES, select_settings
from scripts.run_task_level_accuracy import (
    RUNNER_FIELDS,
    _task_split_integrity,
    effective_proposal_memory_mode,
)


EXPECTED_SETTINGS = [
    "shared_baseline",
    "shared_independent_accuracy",
    "shared_peer_state_vote_first",
    "shared_peer_state_member_first_safe",
    "shared_member_aware_responsibility",
    "shared_member_aware_full",
]


def _validate_configured_initial_prompts(cfg: Config) -> None:
    if cfg.training.initialization_mode == "shared_identical":
        validate_mutable_decision_procedure(cfg.training.shared_prompt)
        return
    if cfg.training.initialization_mode != "provided_prompt_set":
        raise ValueError(
            f"unknown initialization mode: {cfg.training.initialization_mode}"
        )
    try:
        prompts = json.loads(cfg.training.provided_prompts_json)
    except json.JSONDecodeError as exc:
        raise ValueError("provided_prompts_json is not valid JSON") from exc
    if not isinstance(prompts, list) or len(prompts) != cfg.training.agents:
        raise ValueError("provided_prompt_set must contain exactly five prompts")
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("provided_prompt_set prompts must be non-empty strings")
        validate_mutable_decision_procedure(prompt)


def preflight(workspace: Path, allow_dirty: bool = False) -> dict:
    errors = []
    configs = [Config.from_flat(**setting.resolved_overrides()) for setting in select_settings("all")]
    if DEFAULT_EXPERIMENT_SETTING_NAMES != EXPECTED_SETTINGS:
        errors.append("experiment settings do not match the frozen six-setting protocol")
    for cfg in configs:
        try:
            _validate_configured_initial_prompts(cfg)
        except ValueError as exc:
            errors.append(str(exc))
        if cfg.training.method_version != METHOD_VERSION:
            errors.append(f"unexpected method version: {cfg.training.method_version}")
        if cfg.tcs.student_invalid_max_retries < 0:
            errors.append("student_invalid_max_retries cannot be negative")
        if cfg.tcs.student_upstream_regeneration_max_count not in {0, 1}:
            errors.append(
                "student_upstream_regeneration_max_count must be zero or one"
            )
        if cfg.training.agents != 5 or cfg.peer_state.aggregation_mode != "plurality":
            errors.append("all settings must use five equal-weight plurality voters")
        if cfg.peer_state.vote_tie_break != "abstain":
            errors.append("all canonical settings must use tie-as-abstain")
        if cfg.responsibility.member_uplift_tolerance < 0:
            errors.append("member_uplift_tolerance cannot be negative")
        if cfg.responsibility.responsibility_mode != "single_service_member_aware_v10":
            errors.append(
                "responsibility_mode must be 'single_service_member_aware_v10'"
            )
        if cfg.evaluation.candidate_eval_pool_size <= 0:
            errors.append("fixed probe must contain at least one example")
        if not (
            0
            < cfg.evaluation.stage_b_candidate_budget
            <= cfg.tcs.num_candidates_per_parent
        ):
            errors.append(
                "stage_b_candidate_budget must be within generated candidate count"
            )
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
        "requested_name": full.requested_name,
        "tcs_context_policy": full.tcs_context_policy,
    }
    if responsibility_payload != full.__dict__:
        errors.append(
            "member-aware responsibility and full settings must differ only in TCS context"
        )
    if not responsibility.service_routing_enabled or not full.service_routing_enabled:
        errors.append("S4 and S5 must enable unique service routing")
    if any(
        protocols[name].service_routing_enabled
        for name in EXPECTED_SETTINGS[:4]
    ):
        errors.append("S0-S3 must not enable service routing or anchors")
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
        "method_version": METHOD_VERSION, "target_selection_version": TARGET_SELECTION_VERSION,
        "candidate_acceptance_version": CANDIDATE_ACCEPTANCE_VERSION,
        "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
        "preservation_policy_version": PRESERVATION_POLICY_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_selection_version": CHECKPOINT_SELECTION_VERSION,
        "test_isolation_version": TEST_ISOLATION_VERSION,
        "student_invalid_recovery_version": STUDENT_INVALID_RECOVERY_VERSION,
        "mutable_prompt_contract_version": MUTABLE_PROMPT_CONTRACT_VERSION,
        "student_prompt_contract_version": STUDENT_PROMPT_CONTRACT_VERSION,
        "candidate_protocol_filter_version": CANDIDATE_PROTOCOL_FILTER_VERSION,
        "tcs_protocol_version": TCS_PROTOCOL_VERSION,
        "tcs_context_version": TCS_CONTEXT_VERSION,
        "proposal_memory_version": PROPOSAL_MEMORY_VERSION,
        "service_routing_version": SERVICE_ROUTING_VERSION,
        "proposal_memory_mode": Config().tcs.proposal_memory_mode,
        "checkpoint_version": CHECKPOINT_VERSION, "settings": EXPECTED_SETTINGS,
        "legacy_compatibility_enabled": False, "errors": errors,
    }


def _role_environment(cfg: Config, role: str) -> dict[str, Any]:
    key_env = getattr(cfg.models, f"{role}_api_key_env")
    base_env = getattr(cfg.models, f"{role}_base_url_env")
    resolved_key_env, key = resolve_api_key(key_env)
    resolved_base_env, base_url = resolve_base_url(base_env)
    return {
        "key_env": resolved_key_env,
        "base_url_env": resolved_base_env,
        "key_present": bool(key),
        "base_url": base_url,
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
                values["proposal_memory_mode"] = effective_proposal_memory_mode(
                    setting.name,
                    str(values.get("proposal_memory_mode", defaults["proposal_memory_mode"])),
                )
                try:
                    cfg = Config.from_flat(**values)
                    _validate_configured_initial_prompts(cfg)
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
                    if cfg.responsibility.member_uplift_tolerance < 0:
                        raise ValueError("member_uplift_tolerance cannot be negative")
                    if (
                        cfg.responsibility.responsibility_mode
                        != "single_service_member_aware_v10"
                    ):
                        raise ValueError(
                            "responsibility_mode must be "
                            "'single_service_member_aware_v10'"
                        )
                    if cfg.tcs.student_invalid_max_retries < 0:
                        raise ValueError(
                            "student_invalid_max_retries cannot be negative"
                        )
                    if cfg.tcs.student_upstream_regeneration_max_count not in {0, 1}:
                        raise ValueError(
                            "student_upstream_regeneration_max_count must be zero or one"
                        )
                    if cfg.evaluation.candidate_eval_pool_size <= 0:
                        raise ValueError("fixed probe must contain at least one example")
                    planned_update_count = cfg.training.epochs * max(
                        1,
                        math.ceil(
                            cfg.data.train_size
                            / max(1, cfg.training.update_every)
                        ),
                    )
                    if (
                        setting.name == "shared_member_aware_full"
                        and cfg.training.epochs == 8
                        and cfg.training.update_every == 25
                        and cfg.data.train_size == 75
                        and planned_update_count != 24
                    ):
                        raise ValueError(
                            "high-frequency Full planned_update_count must equal 24"
                        )
                    if min(
                        cfg.tcs.tcs_max_pattern_summaries,
                        cfg.tcs.tcs_max_evidence_cases,
                        cfg.tcs.tcs_context_max_chars,
                        cfg.tcs.teacher_total_max_chars,
                        cfg.tcs.teacher_field_max_chars,
                        cfg.tcs.critic_feedback_max_chars,
                        cfg.tcs.candidate_prompt_max_chars,
                        cfg.tcs.total_candidate_prompt_max_chars,
                    ) <= 0:
                        raise ValueError("all TCS context limits must be positive")
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
                        "tcs_context_version": TCS_CONTEXT_VERSION,
                        "proposal_memory_version": PROPOSAL_MEMORY_VERSION,
                        "proposal_memory_mode": cfg.tcs.proposal_memory_mode,
                        "mutable_prompt_contract_version": (
                            MUTABLE_PROMPT_CONTRACT_VERSION
                        ),
                        "student_prompt_contract_version": (
                            STUDENT_PROMPT_CONTRACT_VERSION
                        ),
                        "candidate_protocol_filter_version": (
                            CANDIDATE_PROTOCOL_FILTER_VERSION
                        ),
                        "split_integrity": integrity,
                        "role_environment": role_environment,
                        "planned_update_count": planned_update_count,
                        "validation_solver_call_count": 0,
                        "estimated_solver_calls": {
                            "lower": 4786,
                            "upper": 7180,
                        },
                        "estimated_role_calls": {
                            "lower": 60,
                            "upper": 288,
                        },
                        "estimated_total_tokens": {
                            "lower": 1703422,
                            "upper": 2555133,
                        },
                        "estimated_maximum_student_recovery_calls": (
                            planned_update_count
                            * (
                                cfg.tcs.student_invalid_max_retries + 1
                            )
                            * (
                                1
                                + cfg.tcs.student_upstream_regeneration_max_count
                            )
                        ),
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
