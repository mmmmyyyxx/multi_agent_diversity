from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash
from multi_dataset_diverse_rl.persistence.identity import build_run_identity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.task_manifest import load_task_manifest
from scripts.run_v18_hybrid_online_accumulation import (
    MANIFEST as TASK_MANIFEST,
    _candidate_rows,
    _config,
    _profile_snapshot,
    write_json,
    write_jsonl,
)
from scripts.v18_hybrid_online_accumulation_support import sha256_json
from scripts.v18_teacher_critic_pipeline_support import (
    ArmController,
    CleanTeacherReplay,
    install_pipeline_arm,
)


ARMS = ("A_CANONICAL", "C_NO_SEMANTIC_CRITIC")
SEEDS = (68,)
UPDATES = 8
AUTH_ENV = "V18_NO_SEMANTIC_CRITIC_ONLINE_AUTHORIZED"
RUNTIME_VERSION = "v18_no_semantic_critic_online_trajectory_v1"
NEUTRAL_TOTAL_CORRECT_TOLERANCE = 1
MANIFEST_PATH = ROOT / "experiments/manifests/v18_no_semantic_critic_online.yaml"


def _registry_model(registry: Mapping[str, Any]) -> dict[str, Any]:
    return dict(registry.get("model") or {
        "solver": "qwen3-14b",
        "teacher": "qwen3-14b",
        "critic": "qwen3-14b",
        "student": "qwen3-14b",
        "thinking": False,
    })


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under_root(path: Path) -> bool:
    return path.resolve() == ROOT.resolve() or ROOT.resolve() in path.resolve().parents


def classify(*, commits_a: int, commits_c: int, vote_correct_a: int, vote_correct_c: int,
             wins: int, losses: int) -> str:
    throughput = commits_c > commits_a
    delta = vote_correct_c - vote_correct_a
    if throughput and delta > 0 and wins > losses:
        return "ONLINE_THROUGHPUT_AND_VOTE_SUPPORTED"
    if throughput and abs(delta) <= NEUTRAL_TOTAL_CORRECT_TOLERANCE:
        return "ONLINE_THROUGHPUT_ONLY"
    if throughput and delta < -NEUTRAL_TOTAL_CORRECT_TOLERANCE:
        return "ONLINE_THROUGHPUT_WITH_TRANSFER_REGRESSION"
    return "NO_CLEAR_ONLINE_ADVANTAGE"


def _funnel_counts(decision: Mapping[str, Any]) -> dict[str, int]:
    branches = list(decision.get("branches", []))
    funnels = [row.get("funnel", {}) for row in branches]
    return {
        "target_branches": len(branches),
        "teacher_plans": sum(int(row.get("teacher_calls", 0)) for row in funnels),
        "critic_calls": sum(int(row.get("critic_calls", 0)) for row in funnels),
        "critic_approvals": sum(int(row.get("critic_approved", 0)) for row in funnels),
        "critic_rejections": sum(int(row.get("critic_semantic_rejections", 0)) for row in funnels),
        "student_reaches": sum(int(row.get("student_calls", 0) > 0) for row in funnels),
        "student_calls": sum(int(row.get("student_calls", 0)) for row in funnels),
        "strict_valid_candidates": sum(int(row.get("valid_candidate_count", 0)) for row in funnels),
        "feasible_candidates": sum(int(row.get("constraint_feasible", 0)) for row in funnels),
        "infrastructure_failures": sum(int(row.get("infrastructure_failed_updates", 0)) for row in funnels),
    }


async def _evaluate_prompt_states(
    system: PromptEnsembleOptimizationSystem,
    validation: Sequence[Mapping[str, Any]],
    prompt_states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if system.validation_evaluation_count != 0:
        raise RuntimeError("validation was accessed before training freeze")
    probe = system.build_validation_probe(validation)
    rows: list[dict[str, Any]] = []
    for state in prompt_states:
        prompts = list(state["prompts"])
        profiles = list(await asyncio.gather(*(
            probe.evaluate_prompt(
                agent_id, prompt, system.prompt_hash(prompt), system.solve
            )
            for agent_id, prompt in enumerate(prompts)
        )))
        system.validation_evaluation_count += 1
        snapshot = _profile_snapshot(system, probe.examples, profiles)
        snapshot.update({
            "state_index": int(state["state_index"]),
            "after_update_index": int(state["after_update_index"]),
            "team_state_hash": str(state["team_state_hash"]),
        })
        rows.append(snapshot)
    return rows


async def _run_trajectory(
    *, registry: Mapping[str, Any], freeze: Mapping[str, Any], task: Any,
    seed: int, arm: str, run_dir: Path, cache_path: Path,
    expected_initial_hash: str | None,
) -> tuple[dict[str, Any], str]:
    run_dir.mkdir(parents=True, exist_ok=False)
    model = _registry_model(registry)
    cfg = _config(
        task=task,
        seed=seed,
        run_dir=run_dir,
        cache_path=cache_path,
        agent_model=str(model["solver"]),
        optimizer_model=str(model["teacher"]),
        evaluator_model=str(model["critic"]),
    )
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    system = PromptEnsembleOptimizationSystem(cfg)
    system.set_run_identity(build_run_identity(
        cfg, train_rows=train, val_rows=validation, test_rows=[], workspace=ROOT,
    ))
    system.planned_update_count = UPDATES
    controller = ArmController(arm=arm, clean_replay=CleanTeacherReplay())
    if arm == "C_NO_SEMANTIC_CRITIC":
        install_pipeline_arm(system, controller)
    await system.initialize_fixed_probe(train[: cfg.evaluation.candidate_eval_pool_size])
    initial = system.frozen_initialization_snapshot()
    initial_hash = sha256_json(initial)
    if expected_initial_hash is not None and initial_hash != expected_initial_hash:
        raise RuntimeError("matched initialization differs within seed")

    train_states = [{
        "state_index": 0, "after_update_index": -1,
        "team_state_hash": system.team_prompt_state_hash(),
        **_profile_snapshot(system, system.fixed_probe.examples, system.active_profiles),
    }]
    prompt_states: list[dict[str, Any]] = [{
        "state_index": 0, "after_update_index": -1,
        "team_state_hash": system.team_prompt_state_hash(),
        "prompts": [agent.current_prompt for agent in system.agents],
    }]
    updates: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for update_index in range(UPDATES):
        _verify_freeze(freeze)
        before = train_states[-1]
        gate_before = len(controller.hard_gate_decisions)
        accepted = await system.update_once(update_index)
        system.completed_update_count = update_index + 1
        after = {
            "state_index": len(train_states), "after_update_index": update_index,
            "team_state_hash": system.team_prompt_state_hash(),
            **_profile_snapshot(system, system.fixed_probe.examples, system.active_profiles),
        }
        train_states.append(after)
        decision = system.candidate_decisions[-1]
        candidate_rows = _candidate_rows(seed=seed, arm=arm, update_index=update_index, decision=decision)
        candidates.extend(candidate_rows)
        funnel = _funnel_counts(decision)
        hard_rows = controller.hard_gate_decisions[gate_before:]
        committed_target = decision.get("target_agent_id") if accepted else None
        member_before = before["metrics"]["per_agent_correct_counts"]
        member_after = after["metrics"]["per_agent_correct_counts"]
        row = {
            "seed": seed, "arm": arm, "update_index": update_index,
            "parent_team_hash": decision.get("parent_team_hash", before["team_state_hash"]),
            "successor_team_hash": after["team_state_hash"],
            "selected_target_ids": list(decision.get("selected_target_ids", [])),
            **funnel,
            "hard_gate_passes": sum(bool(item.get("pass")) for item in hard_rows),
            "hard_gate_rejections": sum(not bool(item.get("pass")) for item in hard_rows),
            "committed": bool(accepted),
            "committed_target": int(committed_target) if committed_target is not None else None,
            "train_vote_gain": max(0, int(after["metrics"]["vote_correct_count"]) - int(before["metrics"]["vote_correct_count"])),
            "train_vote_loss": max(0, int(before["metrics"]["vote_correct_count"]) - int(after["metrics"]["vote_correct_count"])),
            "train_vote_delta": int(after["metrics"]["vote_correct_count"]) - int(before["metrics"]["vote_correct_count"]),
            "train_oracle_delta": int(after["metrics"]["oracle_correct_count"]) - int(before["metrics"]["oracle_correct_count"]),
            "train_target_delta": (
                int(member_after[int(committed_target)]) - int(member_before[int(committed_target)])
                if committed_target is not None else None
            ),
        }
        updates.append(row)
        if accepted:
            prompt_states.append({
                "state_index": len(prompt_states), "after_update_index": update_index,
                "team_state_hash": after["team_state_hash"],
                "prompts": [agent.current_prompt for agent in system.agents],
            })
        print(json.dumps({"seed": seed, "arm": arm, "updates": update_index + 1,
                          "accepted": sum(item["committed"] for item in updates)}), flush=True)
        if system.early_stop_reason:
            break

    system.mark_training_complete(UPDATES)
    validation_states = await _evaluate_prompt_states(system, validation, prompt_states)
    if system.test_evaluation_count:
        raise RuntimeError("test isolation failure")
    validation_by_update = {int(row["after_update_index"]): row for row in validation_states}
    previous = validation_states[0]
    for row in updates:
        if row["committed"]:
            current = validation_by_update[int(row["update_index"])]
            row["validation_vote_delta"] = int(current["metrics"]["vote_correct_count"]) - int(previous["metrics"]["vote_correct_count"])
            row["validation_oracle_delta"] = int(current["metrics"]["oracle_correct_count"]) - int(previous["metrics"]["oracle_correct_count"])
            target = int(row["committed_target"])
            row["validation_target_delta"] = int(current["metrics"]["per_agent_correct_counts"][target]) - int(previous["metrics"]["per_agent_correct_counts"][target])
            previous = current
        else:
            row.update(validation_vote_delta=None, validation_oracle_delta=None, validation_target_delta=None)

    system.final_state_selection = {
        "selected_checkpoint_source": "final_active_state",
        "selected_checkpoint_update_index": system.completed_update_count,
        "validation_used_for_selection": False,
        "test_evaluation_count": 0,
    }
    system.flush_artifacts()
    write_json(run_dir / "initialization_snapshot.json", initial)
    write_json(run_dir / "private_prompt_states.json", prompt_states)
    write_jsonl(run_dir / "train_states.jsonl", train_states)
    write_jsonl(run_dir / "validation_states.jsonl", validation_states)
    write_jsonl(run_dir / "update_lineage.jsonl", updates)
    write_jsonl(run_dir / "candidate_level_sanitized.jsonl", candidates)
    write_jsonl(run_dir / "hard_gate_decisions_sanitized.jsonl", controller.hard_gate_decisions)
    cost = system.cost_summary()
    final_train = train_states[-1]["metrics"]
    final_val = validation_states[-1]["metrics"]
    summary = {
        "runtime_version": registry["runtime_version"],
        "execution_commit": registry["execution_commit"],
        "model": dict(model),
        "seed": seed, "arm": arm,
        "underlying_setting": cfg.training.experiment_setting,
        "planned_update_count": UPDATES,
        "completed_update_count": system.completed_update_count,
        "early_stop_reason": system.early_stop_reason,
        "accepted_commit_count": sum(row["committed"] for row in updates),
        "distinct_members_updated": sorted({row["committed_target"] for row in updates if row["committed"]}),
        "validation_evaluation_count": len(validation_states),
        "validation_access_started_after_training_complete": True,
        "validation_used_for_selection": False,
        "test_evaluation_count": 0,
        "initialization_snapshot_hash": initial_hash,
        "initial_team_hash": train_states[0]["team_state_hash"],
        "final_team_hash": train_states[-1]["team_state_hash"],
        "funnel": {key: sum(int(row[key]) for row in updates) for key in (
            "target_branches", "teacher_plans", "critic_calls", "critic_approvals", "critic_rejections",
            "hard_gate_passes", "hard_gate_rejections", "student_reaches", "student_calls",
            "strict_valid_candidates", "feasible_candidates", "infrastructure_failures",
        )},
        "final_train_metrics": final_train,
        "final_validation_metrics": final_val,
        "cost": cost,
        "logical_solver_requests": int(system.prompt_question_evaluator.cache_hits + system.prompt_question_evaluator.cache_misses),
        "solver_cache_hits": int(system.prompt_question_evaluator.cache_hits),
        "solver_cache_misses": int(system.prompt_question_evaluator.cache_misses),
    }
    write_json(run_dir / "online_run_summary.json", summary)
    _verify_freeze(freeze)
    return summary, initial_hash


def _verify_freeze(freeze: Mapping[str, Any]) -> None:
    if _git("rev-parse", "HEAD") != freeze["execution_commit"]:
        raise RuntimeError("execution commit differs from freeze")
    if _git("status", "--porcelain"):
        raise RuntimeError("tracked worktree is dirty")
    for row in freeze["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or _file_sha(path) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")


def prepare(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit("fresh prep root required")
    if _git("status", "--porcelain"):
        raise SystemExit("tracked worktree must be clean")
    if args.run_root.exists():
        raise SystemExit("fresh run root required")
    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    if not seeds:
        raise SystemExit("manifest must freeze at least one seed")
    expected_hash = preregistration_hash(manifest)
    if manifest["artifacts"]["preregistration"]["sha256"] != expected_hash:
        raise SystemExit("tracked preregistration hash mismatch")
    execution_commit = _git("rev-parse", "HEAD")
    model_document = manifest["model"]
    optimizer_roles = model_document["optimizer_roles"]
    if optimizer_roles["teacher"] != optimizer_roles["student"]:
        raise SystemExit("Teacher and Student must share optimizer_model in this runner")
    model = {
        "solver": str(model_document["solver"]),
        "teacher": str(optimizer_roles["teacher"]),
        "critic": str(optimizer_roles["critic"]),
        "student": str(optimizer_roles["student"]),
        "thinking": bool(model_document["thinking"]),
    }
    if model["thinking"] is not False:
        raise SystemExit("this runner requires thinking=false")
    limit = manifest["budget"]["limit"]
    if int(limit["updates_per_seed_arm"]) != UPDATES:
        raise SystemExit("manifest update budget differs from frozen runner")
    if int(limit["trajectories"]) != len(seeds) * len(ARMS):
        raise SystemExit("manifest trajectory budget does not match seeds and arms")
    runtime_manifest = dict(manifest)
    runtime_manifest["status"] = "RUNNING"
    runtime_manifest["lifecycle_history"] = [*manifest["lifecycle_history"], {"status": "RUNNING", "timestamp": "2026-09-03T00:00:00+08:00"}]
    runtime_manifest["git"] = {**manifest["git"], "implementation_commit": execution_commit}
    args.out.mkdir(parents=True)
    write_json(args.out / "runtime_manifest.json", runtime_manifest)
    registry = {
        "runtime_version": str(manifest.get("runtime_version", RUNTIME_VERSION)),
        "execution_commit": execution_commit,
        "seeds": list(seeds), "arms": list(ARMS), "updates": UPDATES,
        "model": model,
        "execution_order": {str(seed): list(ARMS) for seed in seeds},
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "task": "disambiguation_qa",
        "validation_policy": "post_training_frozen_state_replay_only",
        "test_policy": "prohibited_zero_rows_loaded_zero_calls",
        "classifier": {
            "neutral_total_correct_tolerance": NEUTRAL_TOTAL_CORRECT_TOLERANCE,
            "labels_in_precedence_order": [
                "ONLINE_THROUGHPUT_AND_VOTE_SUPPORTED", "ONLINE_THROUGHPUT_ONLY",
                "ONLINE_THROUGHPUT_WITH_TRANSFER_REGRESSION", "NO_CLEAR_ONLINE_ADVANTAGE",
            ],
        },
    }
    registry["registry_hash"] = sha256_json(registry)
    write_json(args.out / "registry.json", registry)
    freeze_files = [
        "scripts/v18_no_semantic_critic_online.py",
        "scripts/v18_teacher_critic_pipeline_support.py",
        "scripts/run_v18_hybrid_online_accumulation.py",
        "multi_dataset_diverse_rl/system.py", "multi_dataset_diverse_rl/tcs.py",
        "multi_dataset_diverse_rl/protocol.py", "multi_dataset_diverse_rl/versions.py",
        manifest_path.relative_to(ROOT).as_posix(),
    ]
    freeze = {
        "execution_commit": execution_commit,
        "registry_hash": registry["registry_hash"],
        "files": [{"path": path, "sha256": _file_sha(ROOT / path)} for path in freeze_files],
    }
    write_json(args.out / "source_freeze.json", freeze)
    write_json(args.out / "preflight.json", {
        "gate": "PASS", "api_calls": 0, "test_calls": 0,
        "api_authorized_for_run": bool(manifest["api_authorization"]["authorized"]),
        "execution_ready": bool(manifest["api_authorization"]["authorized"]),
        "canonical_a_passthrough": True,
        "deterministic_hard_gate_frozen": True,
        "seeds": list(seeds), "arms": list(ARMS), "updates": UPDATES,
        "project_local_paths": _under_root(args.out) and _under_root(args.run_root),
    })


async def run(args: argparse.Namespace) -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise SystemExit(f"set {AUTH_ENV}=1 for this authorized run")
    if args.out.exists():
        raise SystemExit("fresh run root required")
    registry = json.loads((args.prep / "registry.json").read_text(encoding="utf-8"))
    freeze = json.loads((args.prep / "source_freeze.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.prep / "runtime_manifest.json").read_text(encoding="utf-8"))
    if registry["registry_hash"] != freeze["registry_hash"]:
        raise SystemExit("registry/freeze mismatch")
    for phase, roles in (("online_trajectory", ("solver", "teacher", "critic", "student")),
                         ("frozen_validation", ("solver",))):
        for role in roles:
            require_api_authorization(manifest, phase=phase, role=role, explicit_user_authorized=True)
    _verify_freeze(freeze)
    task = load_task_manifest(str(TASK_MANIFEST))[registry["task"]]
    args.out.mkdir(parents=True)
    completed = []
    for seed in registry["seeds"]:
        seed_root = args.out / f"seed{seed}"
        seed_root.mkdir()
        cache = seed_root / "_shared_solver_cache.sqlite"
        initial_hash = None
        for arm in registry["execution_order"][str(seed)]:
            summary, arm_hash = await _run_trajectory(
                registry=registry, freeze=freeze, task=task, seed=int(seed), arm=arm,
                run_dir=seed_root / arm, cache_path=cache,
                expected_initial_hash=initial_hash,
            )
            initial_hash = arm_hash
            completed.append(summary)
    write_json(args.out / "execution_summary.json", {
        "runtime_version": RUNTIME_VERSION, "execution_commit": registry["execution_commit"],
        "trajectory_count": len(completed), "seeds": registry["seeds"], "arms": registry["arms"],
        "test_evaluation_count": sum(row["test_evaluation_count"] for row in completed),
        "infrastructure_failure_count": sum(row["funnel"]["infrastructure_failures"] for row in completed),
        "initialization_matched_within_seed": all(len({row["initialization_snapshot_hash"] for row in completed if row["seed"] == seed}) == 1 for seed in registry["seeds"]),
    })


def audit(args: argparse.Namespace) -> None:
    registry = json.loads((args.prep / "registry.json").read_text(encoding="utf-8"))
    freeze = json.loads((args.prep / "source_freeze.json").read_text(encoding="utf-8"))
    blockers: list[str] = []
    summaries = []
    for seed in registry["seeds"]:
        hashes = set()
        for arm in ARMS:
            path = args.run_root / f"seed{seed}" / arm
            if not (path / "online_run_summary.json").is_file():
                blockers.append(f"missing_run:{seed}:{arm}")
                continue
            row = json.loads((path / "online_run_summary.json").read_text(encoding="utf-8"))
            summaries.append(row); hashes.add(row["initialization_snapshot_hash"])
            if "model" in registry and row.get("model") != _registry_model(registry):
                blockers.append(f"model_identity:{seed}:{arm}")
            run_meta = json.loads((path / "run_meta.json").read_text(encoding="utf-8"))
            run_config = run_meta.get("config", {})
            expected_model = _registry_model(registry)
            if (
                run_config.get("agent_model") != expected_model["solver"]
                or run_config.get("optimizer_model") != expected_model["teacher"]
                or run_config.get("evaluator_model") != expected_model["critic"]
            ):
                blockers.append(f"persisted_model_identity:{seed}:{arm}")
            if row["completed_update_count"] != UPDATES and row["early_stop_reason"] != "no_actionable_responsibility":
                blockers.append(f"update_budget:{seed}:{arm}")
            if row["test_evaluation_count"] != 0 or row["validation_used_for_selection"]:
                blockers.append(f"evaluation_isolation:{seed}:{arm}")
            if not row["validation_access_started_after_training_complete"]:
                blockers.append(f"validation_timing:{seed}:{arm}")
            if row["funnel"]["infrastructure_failures"]:
                blockers.append(f"infrastructure:{seed}:{arm}")
            if arm == "A_CANONICAL" and (path / "hard_gate_decisions_sanitized.jsonl").read_text(encoding="utf-8").strip():
                blockers.append(f"canonical_hard_gate_contamination:{seed}")
            if arm == "C_NO_SEMANTIC_CRITIC":
                hard_total = row["funnel"]["hard_gate_passes"] + row["funnel"]["hard_gate_rejections"]
                if hard_total != row["funnel"]["critic_calls"]:
                    blockers.append(f"hard_gate_accounting:{seed}")
                if int(row["cost"]["tokens_by_role"]["critic"]) != 0:
                    blockers.append(f"semantic_critic_api_in_c:{seed}")
        if len(hashes) != 1:
            blockers.append(f"initialization_mismatch:{seed}")
    expected_trajectories = len(registry["seeds"]) * len(ARMS)
    if len(summaries) != expected_trajectories:
        blockers.append("trajectory_count")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "audit.json", {
        "gate": "PASS" if not blockers else "HOLD", "blockers": blockers,
        "trajectory_count": len(summaries), "expected_trajectory_count": expected_trajectories,
        "test_evaluation_count": sum(row.get("test_evaluation_count", 0) for row in summaries),
        "execution_commit": registry["execution_commit"], "source_freeze_checked": freeze["execution_commit"] == registry["execution_commit"],
    })
    if blockers:
        raise SystemExit("audit HOLD: " + ",".join(blockers))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields)); writer.writeheader()
        for row in rows: writer.writerow({field: row.get(field) for field in fields})


def _transition_mechanisms(states: list[dict[str, Any]]) -> dict[str, int]:
    recovery = deepening = conversion = accumulation = 0
    gained_by_question: dict[str, set[int]] = defaultdict(set)
    for before, after in zip(states, states[1:]):
        old = {row["example_id_hash"]: row for row in before["examples"]}
        for row in after["examples"]:
            prior = old[row["example_id_hash"]]
            g0, g1 = int(prior["correct_member_count"]), int(row["correct_member_count"])
            if g0 == 0 and g1 >= 1: recovery += 1
            if g0 == 1 and g1 >= 2: deepening += 1
            if not prior["vote_correct"] and row["vote_correct"]: conversion += 1
            new_members = set(row["correct_member_ids"]) - set(prior["correct_member_ids"])
            gained_by_question[row["example_id_hash"]].update(new_members)
    accumulation = sum(len(members) >= 2 for members in gained_by_question.values())
    final = states[-1]
    persistent_singletons = sum(int(row["correct_member_count"]) == 1 for row in final["examples"])
    return {"coverage_recovery_0_to_1": recovery, "support_deepening_1_to_2plus": deepening,
            "coverage_to_vote_conversion": conversion, "cross_member_accumulation": accumulation,
            "persistent_singleton_coverage": persistent_singletons}


def analyze(args: argparse.Namespace) -> None:
    audit_payload = json.loads((args.gate / "audit.json").read_text(encoding="utf-8"))
    if audit_payload["gate"] != "PASS": raise SystemExit("official audit is not PASS")
    registry = json.loads((args.prep / "registry.json").read_text(encoding="utf-8"))
    report_manifest = ROOT / registry.get("manifest_path", MANIFEST_PATH.relative_to(ROOT).as_posix())
    manifest_document = yaml.safe_load(report_manifest.read_text(encoding="utf-8"))
    experiment_id = str(manifest_document["experiment_id"])
    post_result_extension = experiment_id.endswith("_seed69_70_extension")
    summaries = []; updates = []; validation_rows = []; compute_rows = []
    for seed in registry["seeds"]:
        for arm in ARMS:
            base = args.run_root / f"seed{seed}" / arm
            summary = json.loads((base / "online_run_summary.json").read_text(encoding="utf-8"))
            lineage = _read_jsonl(base / "update_lineage.jsonl")
            candidate_rows = _read_jsonl(base / "candidate_level_sanitized.jsonl")
            feasible_by_update = Counter(
                int(item["update_index"]) for item in candidate_rows
                if bool(item.get("feasible"))
            )
            for item in lineage:
                item["feasible_candidates"] = feasible_by_update[int(item["update_index"])]
            states = _read_jsonl(base / "validation_states.jsonl")
            mechanism = _transition_mechanisms(states)
            final = summary["final_validation_metrics"]
            final_train = summary["final_train_metrics"]
            funnel = dict(summary["funnel"])
            funnel["feasible_candidates"] = sum(feasible_by_update.values())
            commits = int(summary["accepted_commit_count"])
            row = {"seed": seed, "arm": arm, "updates": summary["completed_update_count"],
                   "commits": commits, "distinct_members_updated": len(summary["distinct_members_updated"]),
                   "student_reaches": funnel["student_reaches"], "feasible_candidates": funnel["feasible_candidates"],
                   "validation_vote_correct": final["vote_correct_count"], "validation_vote_acc": final["vote_acc"],
                   "validation_oracle_correct": final["oracle_correct_count"], "validation_oracle_acc": final["oracle_acc"],
                   "train_vote_correct": final_train["vote_correct_count"], "train_vote_acc": final_train["vote_acc"],
                   "train_oracle_correct": final_train["oracle_correct_count"], "train_oracle_acc": final_train["oracle_acc"],
                   "member_correct_counts": json.dumps(final["per_agent_correct_counts"], separators=(",", ":")),
                   **mechanism}
            summaries.append(row)
            for item in lineage:
                updates.append({key: item.get(key) for key in (
                    "seed", "arm", "update_index", "target_branches", "teacher_plans", "critic_approvals",
                    "critic_rejections", "hard_gate_passes", "hard_gate_rejections", "student_reaches",
                    "strict_valid_candidates", "feasible_candidates", "committed", "committed_target",
                    "train_vote_gain", "train_vote_loss", "train_vote_delta", "train_target_delta",
                    "validation_vote_delta", "validation_oracle_delta", "validation_target_delta")})
            validation_rows.append({"seed": seed, "arm": arm, "vote_correct": final["vote_correct_count"],
                                    "vote_acc": final["vote_acc"], "oracle_correct": final["oracle_correct_count"],
                                    "oracle_acc": final["oracle_acc"], "member_accuracies": json.dumps(final["per_agent_acc"], separators=(",", ":"))})
            total_tokens = int(summary["cost"]["total_tokens"])
            initial_vote = int(states[0]["metrics"]["vote_correct_count"])
            vote_gain = int(final["vote_correct_count"]) - initial_vote
            compute_rows.append({"seed": seed, "arm": arm, "api_calls": summary["cost"]["successful_llm_calls"],
                                 "logical_solver_requests": summary["logical_solver_requests"],
                                 "teacher_tokens": summary["cost"]["tokens_by_role"]["teacher"],
                                 "critic_tokens": summary["cost"]["tokens_by_role"]["critic"],
                                 "student_tokens": summary["cost"]["tokens_by_role"]["student"],
                                 "solver_tokens": summary["cost"]["tokens_by_role"]["solver"],
                                 "total_tokens": total_tokens,
                                 "vote_gain_per_100k_tokens": vote_gain * 100000 / total_tokens if total_tokens else None,
                                 "feasible_per_100k_tokens": funnel["feasible_candidates"] * 100000 / total_tokens if total_tokens else None,
                                 "commits_per_100k_tokens": commits * 100000 / total_tokens if total_tokens else None})
    contrasts=[]; wins=ties=losses=0
    for seed in registry["seeds"]:
        by_arm={row["arm"]:row for row in summaries if row["seed"]==seed}
        delta=int(by_arm["C_NO_SEMANTIC_CRITIC"]["validation_vote_correct"])-int(by_arm["A_CANONICAL"]["validation_vote_correct"])
        wins += delta>0; ties += delta==0; losses += delta<0
        contrasts.append({"seed":seed,"c_minus_a_vote_correct":delta,
                          "c_minus_a_vote_acc":delta/50,
                          "c_minus_a_commits":by_arm["C_NO_SEMANTIC_CRITIC"]["commits"]-by_arm["A_CANONICAL"]["commits"]})
    commits_a=sum(row["commits"] for row in summaries if row["arm"]=="A_CANONICAL")
    commits_c=sum(row["commits"] for row in summaries if row["arm"]=="C_NO_SEMANTIC_CRITIC")
    vote_a=sum(row["validation_vote_correct"] for row in summaries if row["arm"]=="A_CANONICAL")
    vote_c=sum(row["validation_vote_correct"] for row in summaries if row["arm"]=="C_NO_SEMANTIC_CRITIC")
    label=classify(commits_a=commits_a,commits_c=commits_c,vote_correct_a=vote_a,vote_correct_c=vote_c,wins=wins,losses=losses)
    args.out.mkdir(parents=True, exist_ok=False)
    _write_csv(args.out/"trajectory_summary.csv",summaries,list(summaries[0]))
    _write_csv(args.out/"update_funnel.csv",updates,list(updates[0]))
    accepted=[row for row in updates if row["committed"]]
    _write_csv(args.out/"accepted_commits.csv",accepted,list(updates[0]))
    _write_csv(args.out/"validation_results.csv",validation_rows,list(validation_rows[0]))
    _write_csv(args.out/"compute_metrics.csv",compute_rows,list(compute_rows[0]))
    _write_csv(args.out/"paired_contrasts.csv",contrasts,list(contrasts[0]))
    write_json(args.out/"classifier.json",{"label":label,"commits_a":commits_a,"commits_c":commits_c,
               "validation_vote_correct_a":vote_a,"validation_vote_correct_c":vote_c,"wins":wins,"ties":ties,"losses":losses,
               "neutral_total_correct_tolerance":NEUTRAL_TOTAL_CORRECT_TOLERANCE})
    write_json(args.out/"summary.json",{
        "experiment_id":experiment_id,
        "seed_count":len(registry["seeds"]),
        "trajectory_count":len(summaries),
        "post_result_extension":post_result_extension,
        "classifier":label,
        "commits":{"A_CANONICAL":commits_a,"C_NO_SEMANTIC_CRITIC":commits_c},
        "final_validation_vote_correct":{"A_CANONICAL":vote_a,"C_NO_SEMANTIC_CRITIC":vote_c},
        "paired_wins_ties_losses":{"wins":wins,"ties":ties,"losses":losses},
        "new_test_calls":0,
    })
    write_json(args.out/"preregistration.json",registry)
    write_json(args.out/"provenance.json",{"execution_commit":registry["execution_commit"],"audit_gate":"PASS",
               "raw_artifacts_modified":False,"test_accessed":False,"validation_used_for_trajectory":False,
               "post_result_extension":post_result_extension})
    total_validation_rows = 50 * len(registry["seeds"])
    write_json(args.out/"fact_assertions.json",{"pass":True,"trajectory_count":len(summaries),"test_calls":0,
               "seeds":registry["seeds"],"arms":registry["arms"],"historical_four_commit_reference_is_diagnostic_only":True})
    write_json(args.out/"api_ledger_summary.json",{"scope":"aggregate_only","total_successful_calls":sum(int(row["api_calls"]) for row in compute_rows),
               "tokens_by_role":{role:sum(int(row[f"{role}_tokens"]) for row in compute_rows) for role in ("teacher","critic","student","solver")}})
    write_json(args.out/"evaluation_access_summary.json",{"train_accessed":True,"validation_accessed_after_training_freeze":True,
               "validation_used_for_selection":False,"test_accessed":False,"test_calls":0})
    write_json(args.out/"funnel_summary.json",{"A":{k:sum(int(s[k]) for s in summaries if s["arm"]=="A_CANONICAL") for k in ("student_reaches","feasible_candidates","commits")},
               "C":{k:sum(int(s[k]) for s in summaries if s["arm"]=="C_NO_SEMANTIC_CRITIC") for k in ("student_reaches","feasible_candidates","commits")}})
    (args.out/"manifest_snapshot.yaml").write_text(
        yaml.safe_dump(manifest_document, sort_keys=False),
        encoding="utf-8",
    )
    def arm_total(arm: str, key: str) -> int:
        return sum(int(row[key]) for row in summaries if row["arm"] == arm)
    train_rows = 75 * len(registry["seeds"])
    readme=(f"# Canonical vs No-Semantic-Critic Online Trajectory\n\n"
            f"Official audit: **PASS**. Frozen classifier: **{label}**. Evidence scope is {len(registry['seeds'])} frozen seed pair(s).\n\n"
            + ("**Provenance:** Seeds69/70 are a post-Seed68-result extension, not an untouched three-seed preregistration. Any combined Seed68-70 result is descriptive only.\n\n" if post_result_extension else "")
            + "| Metric | A Canonical | C No Semantic Critic |\n|---|---:|---:|\n"
            f"| Student reaches | {arm_total('A_CANONICAL','student_reaches')} | {arm_total('C_NO_SEMANTIC_CRITIC','student_reaches')} |\n"
            f"| Feasible candidates | {arm_total('A_CANONICAL','feasible_candidates')} | {arm_total('C_NO_SEMANTIC_CRITIC','feasible_candidates')} |\n"
            f"| Accepted commits | {commits_a} | {commits_c} |\n"
            f"| Distinct member-seed updates | {arm_total('A_CANONICAL','distinct_members_updated')} | {arm_total('C_NO_SEMANTIC_CRITIC','distinct_members_updated')} |\n"
            f"| Final train Vote total | {arm_total('A_CANONICAL','train_vote_correct')}/{train_rows} | {arm_total('C_NO_SEMANTIC_CRITIC','train_vote_correct')}/{train_rows} |\n"
            f"| Final train Oracle total | {arm_total('A_CANONICAL','train_oracle_correct')}/{train_rows} | {arm_total('C_NO_SEMANTIC_CRITIC','train_oracle_correct')}/{train_rows} |\n"
            f"| Final validation Vote total | {vote_a}/{total_validation_rows} | {vote_c}/{total_validation_rows} |\n"
            f"| Final validation Oracle total | {arm_total('A_CANONICAL','validation_oracle_correct')}/{total_validation_rows} | {arm_total('C_NO_SEMANTIC_CRITIC','validation_oracle_correct')}/{total_validation_rows} |\n"
            f"| Coverage to vote conversions | {arm_total('A_CANONICAL','coverage_to_vote_conversion')} | {arm_total('C_NO_SEMANTIC_CRITIC','coverage_to_vote_conversion')} |\n"
            f"| Persistent singleton coverage | {arm_total('A_CANONICAL','persistent_singleton_coverage')} | {arm_total('C_NO_SEMANTIC_CRITIC','persistent_singleton_coverage')} |\n\n"
            f"C-A final validation Vote total = {vote_c-vote_a:+d}/{total_validation_rows} and W/T/L = {wins}/{ties}/{losses}. The historical four-commit reference remains diagnostic only.\n\n"
            "Validation was evaluated only after each online trajectory was frozen and never affected target selection, candidate acceptance, ranking, or commits. Intermediate frozen states were replayed post hoc only to attribute accepted-transition validation gains and losses. Test125 was not loaded for evaluation and received zero calls.\n")
    (args.out/"README.md").write_text(readme,encoding="utf-8")
    if post_result_extension:
        historical_path = ROOT / "reports/v18_no_semantic_critic_online_trajectory_20260903/trajectory_summary.csv"
        with historical_path.open(encoding="utf-8", newline="") as handle:
            historical_rows = list(csv.DictReader(handle))
        combined_rows = sorted(
            [*historical_rows, *summaries],
            key=lambda item: (int(item["seed"]), str(item["arm"])),
        )
        _write_csv(
            args.out/"combined_seed68_70_trajectory_summary.csv",
            combined_rows,
            list(summaries[0]),
        )

        def combined_total(arm: str, key: str) -> int:
            return sum(int(float(row[key])) for row in combined_rows if row["arm"] == arm)

        combined_deltas = []
        for seed in (68, 69, 70):
            by_arm = {row["arm"]: row for row in combined_rows if int(row["seed"]) == seed}
            combined_deltas.append(
                int(float(by_arm["C_NO_SEMANTIC_CRITIC"]["validation_vote_correct"]))
                - int(float(by_arm["A_CANONICAL"]["validation_vote_correct"]))
            )
        combined_wins = sum(delta > 0 for delta in combined_deltas)
        combined_ties = sum(delta == 0 for delta in combined_deltas)
        combined_losses = sum(delta < 0 for delta in combined_deltas)
        combined_commits_a = combined_total("A_CANONICAL", "commits")
        combined_commits_c = combined_total("C_NO_SEMANTIC_CRITIC", "commits")
        combined_vote_a = combined_total("A_CANONICAL", "validation_vote_correct")
        combined_vote_c = combined_total("C_NO_SEMANTIC_CRITIC", "validation_vote_correct")
        combined_label = classify(
            commits_a=combined_commits_a,
            commits_c=combined_commits_c,
            vote_correct_a=combined_vote_a,
            vote_correct_c=combined_vote_c,
            wins=combined_wins,
            losses=combined_losses,
        )
        write_json(args.out/"combined_seed68_70_summary.json", {
            "evidence_scope": "descriptive_post_result_extension",
            "untouched_three_seed_preregistration": False,
            "seeds": [68, 69, 70],
            "classifier_under_unchanged_rule": combined_label,
            "student_reaches": {
                "A_CANONICAL": combined_total("A_CANONICAL", "student_reaches"),
                "C_NO_SEMANTIC_CRITIC": combined_total("C_NO_SEMANTIC_CRITIC", "student_reaches"),
            },
            "feasible_candidates": {
                "A_CANONICAL": combined_total("A_CANONICAL", "feasible_candidates"),
                "C_NO_SEMANTIC_CRITIC": combined_total("C_NO_SEMANTIC_CRITIC", "feasible_candidates"),
            },
            "commits": {"A_CANONICAL": combined_commits_a, "C_NO_SEMANTIC_CRITIC": combined_commits_c},
            "final_validation_vote_correct": {"A_CANONICAL": combined_vote_a, "C_NO_SEMANTIC_CRITIC": combined_vote_c},
            "final_validation_oracle_correct": {
                "A_CANONICAL": combined_total("A_CANONICAL", "validation_oracle_correct"),
                "C_NO_SEMANTIC_CRITIC": combined_total("C_NO_SEMANTIC_CRITIC", "validation_oracle_correct"),
            },
            "paired_vote_deltas": combined_deltas,
            "paired_wins_ties_losses": {"wins": combined_wins, "ties": combined_ties, "losses": combined_losses},
            "test_calls": 0,
        })
        combined_note = (
            "# Descriptive Seed68-70 summary\n\n"
            "This aggregate combines the original Seed68 result with the post-result Seed69/70 extension. "
            "It is descriptive and is not an untouched three-seed preregistration.\n\n"
            f"Under the unchanged frozen classifier, the descriptive label is **{combined_label}**. "
            f"C-A validation Vote deltas are {combined_deltas}; W/T/L is "
            f"{combined_wins}/{combined_ties}/{combined_losses}. Aggregate validation Vote is "
            f"{combined_vote_a}/150 for A and {combined_vote_c}/150 for C. "
            f"Accepted commits are {combined_commits_a} for A and {combined_commits_c} for C.\n\n"
            "No test data was evaluated.\n"
        )
        (args.out/"COMBINED_SEED68_70.md").write_text(combined_note, encoding="utf-8")
    write_json(args.out/"sanitization_manifest.json",{"status":"PASS","raw_text_published":False,
               "forbidden_content":["prompts","questions","gold answers","model answers","raw responses","endpoints","credentials","SQLite","checkpoints","absolute paths"]})
    files=[path for path in args.out.iterdir() if path.is_file() and path.name!="sha256_manifest.json"]
    write_json(args.out/"sha256_manifest.json",{"files":[{"path":path.name,"sha256":_file_sha(path)} for path in sorted(files)]})


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare"); p.add_argument("--out",type=Path,required=True); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--manifest",type=Path,default=MANIFEST_PATH)
    p=sub.add_parser("run"); p.add_argument("--prep",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("audit"); p.add_argument("--prep",type=Path,required=True); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("analyze"); p.add_argument("--prep",type=Path,required=True); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--gate",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    for name in ("out","run_root","prep","gate","manifest"):
        path=getattr(args,name,None)
        if path is not None and not _under_root(path): raise SystemExit(f"{name} must be project-local")
    if args.command=="prepare": prepare(args)
    elif args.command=="run": asyncio.run(run(args))
    elif args.command=="audit": audit(args)
    else: analyze(args)


if __name__ == "__main__": main()
