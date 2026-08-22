from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.persistence.identity import build_run_identity
from multi_dataset_diverse_rl.responsibility import (
    compute_member_aware_repair_opportunity,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.task_manifest import load_task_manifest
from scripts.v18_hybrid_online_accumulation_support import (
    ARMS,
    AUTHORIZATION_ENV,
    HYBRID,
    UPDATES,
    W1,
    canonical_json,
    generation_key,
    hybrid_targets,
    sha256_json,
)


MANIFEST = ROOT / "configs/task_level_comparison_strict_bbh_seed42.yaml"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def verify_source_freeze(freeze: Mapping[str, Any]) -> None:
    if git("rev-parse", "HEAD") != freeze["execution_commit"]:
        raise RuntimeError("execution commit differs from source freeze")
    if git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must remain clean")
    for row in freeze["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            raise RuntimeError(f"frozen source missing: {row['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"frozen source changed: {row['path']}")


class HybridOnlineSystem(PromptEnsembleOptimizationSystem):
    def select_targets(self, assigned, update_index):
        original_targets, payload = super().select_targets(assigned, update_index)
        if len(original_targets) < 2:
            return original_targets, payload
        ordered_payload = sorted(
            payload, key=lambda row: (int(row["selection_rank"]), int(row["agent_id"]))
        )
        w1_order = [int(row["agent_id"]) for row in ordered_payload]
        targets = hybrid_targets(
            seed=self.cfg.training.seed,
            update_index=update_index,
            w1_order=w1_order,
            responsibility_eligible=w1_order,
        )
        selected = set(targets)
        for row in payload:
            row["selected"] = int(row["agent_id"]) in selected
        for row in self.repairability_adjusted_target_scores[-len(payload):]:
            row["selected"] = int(row["agent_id"]) in selected
        audit = self.target_priority_audit[-1]
        audit["selection_pool_stage"] = "w1_rank1_plus_responsibility_constrained_rr"
        audit["selected_target_ids"] = list(targets)
        audit["selector_override"] = "frozen_hybrid_base_v1"
        audit["w1_top2_before_hybrid"] = list(original_targets)
        self.selected_target_ids = list(targets)
        return targets, payload


def _config(*, task: Any, seed: int, run_dir: Path, cache_path: Path) -> Config:
    return Config.from_flat(
        task_type=task.task_type,
        dataset_format="mars",
        comparison_task_id=task.task_id,
        benchmark=task.benchmark,
        answer_format=task.answer_format,
        train_path=str((ROOT / task.train_path).resolve()),
        val_path=str((ROOT / task.val_path).resolve()),
        test_path=str((ROOT / task.test_path).resolve()),
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        train_size=75,
        val_size=50,
        test_size=125,
        agent_model="qwen3-14b",
        optimizer_model="qwen3-14b",
        evaluator_model="qwen3-14b",
        temperature=0.0,
        solver_max_tokens=1800,
        experiment_setting="experimental_v16_efficacy_g_matched",
        agents=5,
        epochs=1,
        update_every=10,
        seed=int(seed),
        proposal_memory_mode="off",
        num_candidates_per_parent=2,
        candidate_eval_pool_size=75,
        eval_solver_call_concurrency=8,
        stage_b_candidate_budget=2,
        out_dir=str(run_dir.resolve()),
        shared_solver_cache_path=str(cache_path.resolve()),
        resume_from_checkpoint=False,
        provider_call_budget=8000,
        total_token_budget=3_000_000,
        final_test_enabled=False,
        preserve_final_checkpoint=False,
    )


def _profile_snapshot(system: PromptEnsembleOptimizationSystem, examples, profiles) -> dict[str, Any]:
    metrics = system._dataset_metrics_from_profiles(examples, profiles)
    example_rows = []
    for index, example in enumerate(examples):
        state = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=[profile[index].answer for profile in profiles],
            valid_vector=[profile[index].valid for profile in profiles],
            normalize_answer=system.normalize_answer,
            match_answer=system.match_answer,
            tie_break=system.protocol.tie_policy,
            seed=system.cfg.training.seed,
        )
        opportunities = [
            compute_member_aware_repair_opportunity(
                team_state=state,
                peer_context=build_peer_vote_context(state, agent_id),
                tau=system.cfg.peer_state.soft_vote_tau,
            )
            for agent_id in range(5)
        ]
        wrong = [row for row in opportunities if row.member_error]
        if state.vote_correct or not wrong:
            eligible = []
        else:
            best = max((int(row.vote_flip_gain), int(row.margin_gain)) for row in wrong)
            eligible = sorted(
                row.agent_id
                for row in wrong
                if (int(row.vote_flip_gain), int(row.margin_gain)) == best
            )
        example_rows.append({
            "example_id_hash": state.question_hash,
            "G": int(state.gold_vote_count),
            "H": int(state.largest_wrong_vote_count),
            "M": int(state.plurality_margin),
            "vote_correct": bool(state.vote_correct),
            "oracle_covered": bool(state.gold_vote_count > 0),
            "correct_member_ids": [
                index for index, correct in enumerate(state.team_correctness) if correct
            ],
            "correct_member_count": int(state.gold_vote_count),
            "responsibility_eligible_member_ids": eligible,
        })
    oracle = sum(row["oracle_covered"] for row in example_rows)
    return {
        "metrics": {
            **metrics.to_dict(),
            "oracle_correct_count": oracle,
            "oracle_acc": oracle / max(1, len(example_rows)),
        },
        "examples": example_rows,
    }


async def _validation_snapshot(
    system: PromptEnsembleOptimizationSystem,
    validation: Sequence[Mapping[str, Any]],
    *,
    state_index: int,
    after_update_index: int,
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if system.validation_probe is None:
        system.validation_probe = system.build_validation_probe(validation)
    probe = system.validation_probe
    profiles = list(await asyncio.gather(*(
        probe.evaluate_prompt(
            agent_id,
            agent.current_prompt,
            system.prompt_hash(agent.current_prompt),
            system.solve,
        )
        for agent_id, agent in enumerate(system.agents)
    )))
    system.validation_evaluation_count += 1
    result = _profile_snapshot(system, probe.examples, profiles)
    result.update({
        "state_index": int(state_index),
        "after_update_index": int(after_update_index),
        "team_state_hash": system.team_prompt_state_hash(),
    })
    previous_by_hash = (
        {row["example_id_hash"]: row for row in previous["examples"]}
        if previous is not None else {}
    )
    for row in result["examples"]:
        before = previous_by_hash.get(row["example_id_hash"])
        prior_members = set(before["correct_member_ids"]) if before else set()
        row["newly_correct_member_count"] = len(
            set(row["correct_member_ids"]) - prior_members
        )
    return result


def _candidate_rows(
    *, seed: int, arm: str, update_index: int, decision: Mapping[str, Any]
) -> list[dict[str, Any]]:
    accepted_hash = str(decision.get("accepted_prompt_hash", ""))
    branch_winners = {
        int(row["target_agent_id"]): str(row.get("branch_winner_hash", ""))
        for row in decision.get("branches", [])
    }
    grouped: dict[int, list[dict[str, Any]]] = {}
    rows = []
    slot_by_target: dict[tuple[int, str], int] = {}
    for candidate in decision.get("candidates", []):
        target = int(candidate["target_agent_id"])
        stage = str(candidate.get("candidate_stage", "source"))
        stage_name = "revision" if stage == "loss_blind_generic_revision" else "source"
        slot_key = (target, stage_name)
        slot_by_target[slot_key] = slot_by_target.get(slot_key, 0) + 1
        constraint = candidate.get("constraint") or {}
        evaluation = candidate.get("evaluation") or {}
        row = {
            "seed": int(seed),
            "arm": arm,
            "update_index": int(update_index),
            "parent_team_hash": str(decision["parent_team_hash"]),
            "target_member": target,
            "target_selection_rank": int(candidate["target_selection_rank"]),
            "candidate_id": str(candidate["prompt_hash"]),
            "candidate_stage": stage_name,
            "source_slot": slot_by_target[slot_key],
            "generation_key": generation_key(
                experiment_seed=seed,
                update_index=update_index,
                target_member=target,
                source_slot=slot_by_target[slot_key],
                candidate_stage=stage_name,
                parent_team_hash=str(decision["parent_team_hash"]),
            ),
            "valid": bool(evaluation),
            "feasible": bool(constraint.get("passed", False)),
            "common_safe_outcome": (
                "passed" if constraint.get("passed", False) else "rejected"
            ),
            "sanitized_rejection_reasons": sorted(
                map(str, constraint.get("rejection_reasons", ()))
            ),
            "target_gain": constraint.get("target_gain"),
            "vote_gain_count": constraint.get("vote_gain_count"),
            "vote_loss_count": constraint.get("vote_loss_count"),
            "vote_net_gain": constraint.get("vote_net_gain"),
            "selected_as_branch_winner": (
                branch_winners.get(target, "") == str(candidate["prompt_hash"])
            ),
            "winner": accepted_hash == str(candidate["prompt_hash"]),
            "branch_rank": None,
            "cell_rank": None,
        }
        rows.append(row)
        grouped.setdefault(target, []).append(row)
    for target_rows in grouped.values():
        feasible = [row for row in target_rows if row["feasible"]]
        feasible.sort(
            key=lambda row: (
                int(row["vote_net_gain"] or 0),
                int(row["target_gain"] or 0),
                -int(row["vote_loss_count"] or 0),
                str(row["candidate_id"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(feasible, start=1):
            row["branch_rank"] = rank
    branch_winner_rows = [row for row in rows if row["selected_as_branch_winner"]]
    branch_winner_rows.sort(key=lambda row: (not row["winner"], row["target_selection_rank"]))
    for rank, row in enumerate(branch_winner_rows, start=1):
        row["cell_rank"] = rank
    return rows


async def run_trajectory(
    *,
    registry: Mapping[str, Any],
    freeze: Mapping[str, Any],
    task: Any,
    seed: int,
    arm: str,
    run_dir: Path,
    cache_path: Path,
    expected_initial_hash: str | None,
) -> tuple[dict[str, Any], str]:
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = _config(task=task, seed=seed, run_dir=run_dir, cache_path=cache_path)
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    test_identity = _load(cfg.data.test_path, cfg.data.test_size, cfg.data.dataset_format)
    system_type = HybridOnlineSystem if arm == HYBRID else PromptEnsembleOptimizationSystem
    system = system_type(cfg)
    system.set_run_identity(build_run_identity(
        cfg,
        train_rows=train,
        val_rows=validation,
        test_rows=test_identity,
        workspace=ROOT,
    ))
    system.planned_update_count = UPDATES
    probe = list(train[: min(len(train), cfg.evaluation.candidate_eval_pool_size)])
    await system.initialize_fixed_probe(probe)
    initial_snapshot = system.frozen_initialization_snapshot()
    initial_hash = sha256_json(initial_snapshot)
    if expected_initial_hash is not None and initial_hash != expected_initial_hash:
        raise RuntimeError("matched arm initial state differs within seed")
    system.record_training_dynamics(update_index=-1)
    validation_states = [await _validation_snapshot(
        system,
        validation,
        state_index=0,
        after_update_index=-1,
        previous=None,
    )]
    train_states = [{
        "state_index": 0,
        "after_update_index": -1,
        "team_state_hash": system.team_prompt_state_hash(),
        **_profile_snapshot(system, system.fixed_probe.examples, system.active_profiles),
    }]
    update_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for update_index in range(UPDATES):
        verify_source_freeze(freeze)
        before_train = train_states[-1]
        before_validation = validation_states[-1]
        refresh_before = int(system.responsibility_refresh_count)
        incumbent_profiles = [tuple(profile) for profile in system.active_profiles]
        accepted = await system.update_once(update_index)
        system.completed_update_count = update_index + 1
        system.record_training_dynamics(
            update_index=update_index, incumbent_profiles=incumbent_profiles
        )
        after_train = {
            "state_index": len(train_states),
            "after_update_index": update_index,
            "team_state_hash": system.team_prompt_state_hash(),
            **_profile_snapshot(system, system.fixed_probe.examples, system.active_profiles),
        }
        train_states.append(after_train)
        decision = system.candidate_decisions[-1]
        candidates = _candidate_rows(
            seed=seed, arm=arm, update_index=update_index, decision=decision
        )
        candidate_rows.extend(candidates)
        if accepted:
            if after_train["team_state_hash"] == before_train["team_state_hash"]:
                raise RuntimeError("accepted update did not change team state")
            if int(system.responsibility_refresh_count) <= refresh_before:
                raise RuntimeError("responsibility was not recomputed after commit")
            current_validation = await _validation_snapshot(
                system,
                validation,
                state_index=len(validation_states),
                after_update_index=update_index,
                previous=before_validation,
            )
            validation_states.append(current_validation)
        else:
            current_validation = before_validation
            if after_train["team_state_hash"] != before_train["team_state_hash"]:
                raise RuntimeError("no-commit update changed team state")
        committed_target = decision.get("target_agent_id") if accepted else None
        train_before_metrics = before_train["metrics"]
        train_after_metrics = after_train["metrics"]
        val_before_metrics = before_validation["metrics"]
        val_after_metrics = current_validation["metrics"]
        committed_target_int = int(committed_target) if committed_target is not None else None
        update_rows.append({
            "seed": seed,
            "arm": arm,
            "update_index": update_index,
            "parent_team_hash": decision["parent_team_hash"],
            "successor_team_hash": after_train["team_state_hash"],
            "target1": decision["selected_target_ids"][0] if decision["selected_target_ids"] else None,
            "target2": decision["selected_target_ids"][1] if len(decision["selected_target_ids"]) > 1 else None,
            "target1_selector_type": "W1_RANK1",
            "target2_selector_type": "W1_RANK2" if arm == W1 else "RESPONSIBILITY_RR",
            "feasible_branch_count": sum(
                bool(row.get("branch_winner_hash")) for row in decision.get("branches", [])
            ),
            "feasible_candidate_count": sum(row["feasible"] for row in candidates),
            "committed": bool(accepted),
            "committed_target": committed_target_int,
            "train_target_delta": (
                int(train_after_metrics["per_agent_correct_counts"][committed_target_int])
                - int(train_before_metrics["per_agent_correct_counts"][committed_target_int])
                if committed_target_int is not None else None
            ),
            "train_vote_delta": (
                int(train_after_metrics["vote_correct_count"])
                - int(train_before_metrics["vote_correct_count"])
            ) if accepted else None,
            "train_oracle_delta": (
                int(train_after_metrics["oracle_correct_count"])
                - int(train_before_metrics["oracle_correct_count"])
            ) if accepted else None,
            "validation_target_delta": (
                int(val_after_metrics["per_agent_correct_counts"][committed_target_int])
                - int(val_before_metrics["per_agent_correct_counts"][committed_target_int])
                if committed_target_int is not None else None
            ),
            "validation_vote_delta": (
                int(val_after_metrics["vote_correct_count"])
                - int(val_before_metrics["vote_correct_count"])
            ) if accepted else None,
            "validation_oracle_delta": (
                int(val_after_metrics["oracle_correct_count"])
                - int(val_before_metrics["oracle_correct_count"])
            ) if accepted else None,
            "responsibility_refresh_count_before": refresh_before,
            "responsibility_refresh_count_after": int(system.responsibility_refresh_count),
            "validation_state_index_before": int(before_validation["state_index"]),
            "validation_state_index_after": int(current_validation["state_index"]),
            "validation_evaluated": bool(accepted),
        })
        print(json.dumps({
            "seed": seed,
            "arm": arm,
            "completed_updates": update_index + 1,
            "planned_updates": UPDATES,
            "accepted_commits": sum(row["committed"] for row in update_rows),
        }), flush=True)
        if system.early_stop_reason:
            break
    system.mark_training_complete(UPDATES)
    system.final_state_selection = {
        "selected_checkpoint_source": "final_active_state",
        "selected_checkpoint_update_index": system.completed_update_count,
        "validation_used_for_selection": False,
        "test_evaluation_count": 0,
    }
    if system.test_evaluation_count != 0:
        raise RuntimeError("new test call detected")
    system.flush_artifacts()
    write_json(run_dir / "initialization_snapshot.json", initial_snapshot)
    write_json(run_dir / "final_prompts_private.json", [agent.current_prompt for agent in system.agents])
    write_jsonl(run_dir / "validation_states.jsonl", validation_states)
    write_jsonl(run_dir / "train_states.jsonl", train_states)
    write_jsonl(run_dir / "update_lineage.jsonl", update_rows)
    write_jsonl(run_dir / "candidate_level_sanitized.jsonl", candidate_rows)
    cost = system.cost_summary()
    summary = {
        "run_version": "v18_hybrid_online_accumulation_trajectory_v1",
        "execution_commit": registry["execution_commit"],
        "seed": seed,
        "arm": arm,
        "underlying_setting": cfg.training.experiment_setting,
        "selector_override": "none" if arm == W1 else "frozen_hybrid_base_v1",
        "planned_update_count": UPDATES,
        "completed_update_count": system.completed_update_count,
        "early_stop_reason": system.early_stop_reason,
        "accepted_commit_count": sum(row["committed"] for row in update_rows),
        "experimental_prompt_commits": sum(row["committed"] for row in update_rows),
        "experimental_trajectory_transitions": len(validation_states) - 1,
        "validation_state_count": len(validation_states),
        "validation_evaluation_count": system.validation_evaluation_count,
        "validation_used_for_selection": False,
        "new_test_calls": system.test_evaluation_count,
        "initialization_snapshot_hash": initial_hash,
        "initial_team_hash": train_states[0]["team_state_hash"],
        "final_team_hash": train_states[-1]["team_state_hash"],
        "target_members": sorted({
            int(target)
            for row in update_rows
            for target in (row["target1"], row["target2"])
            if target is not None
        }),
        "cost": cost,
        "prompt_question_cache_hits": system.prompt_question_evaluator.cache_hits,
        "prompt_question_cache_misses": system.prompt_question_evaluator.cache_misses,
        "infrastructure_failure_count": sum(
            int((decision.get("funnel") or {}).get("infrastructure_failed_updates", 0))
            for decision in system.candidate_decisions
        ),
    }
    write_json(run_dir / "online_run_summary.json", summary)
    verify_source_freeze(freeze)
    return summary, initial_hash


async def main_async(args: argparse.Namespace) -> None:
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        raise SystemExit(f"set {AUTHORIZATION_ENV}=1 only for authorized Phase B")
    if args.out_root.exists():
        raise SystemExit("fresh output root required")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    freeze = json.loads(args.source_freeze.read_text(encoding="utf-8"))
    if registry["registry_content_hash"] != freeze["registry_content_hash"]:
        raise SystemExit("registry/freeze identity mismatch")
    payload = {key: value for key, value in registry.items() if key != "registry_content_hash"}
    if sha256_json(payload) != registry["registry_content_hash"]:
        raise SystemExit("registry content hash mismatch")
    verify_source_freeze(freeze)
    tasks = load_task_manifest(str(MANIFEST))
    task = tasks[registry["task"]]
    args.out_root.mkdir(parents=True)
    completed = []
    for seed in registry["seeds"]:
        seed_root = args.out_root / f"seed{seed}"
        seed_root.mkdir()
        cache_path = seed_root / "_shared_solver_cache.sqlite"
        initial_hash = None
        for arm in registry["execution_order"][str(seed)]:
            summary, arm_initial_hash = await run_trajectory(
                registry=registry,
                freeze=freeze,
                task=task,
                seed=int(seed),
                arm=arm,
                run_dir=seed_root / arm,
                cache_path=cache_path,
                expected_initial_hash=initial_hash,
            )
            initial_hash = arm_initial_hash
            completed.append(summary)
            print(json.dumps({
                "completed_trajectories": len(completed),
                "planned_trajectories": 6,
                "last_seed": seed,
                "last_arm": arm,
            }), flush=True)
    if len(completed) != 6:
        raise RuntimeError("not every frozen trajectory completed")
    write_json(args.out_root / "execution_summary.json", {
        "execution_version": "v18_hybrid_online_accumulation_execution_v1",
        "execution_commit": registry["execution_commit"],
        "trajectory_count": len(completed),
        "seeds": registry["seeds"],
        "arms": registry["arms"],
        "new_test_calls": sum(row["new_test_calls"] for row in completed),
        "infrastructure_failure_count": sum(
            row["infrastructure_failure_count"] for row in completed
        ),
        "initialization_matched_within_seed": all(
            len({
                row["initialization_snapshot_hash"] for row in completed
                if row["seed"] == seed
            }) == 1
            for seed in registry["seeds"]
        ),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    if ROOT.resolve() not in args.out_root.parents:
        raise SystemExit("output root must be project-local")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
