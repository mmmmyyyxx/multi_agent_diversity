from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    common_monotone_safe_key,
    evaluate_constraints,
)
from multi_dataset_diverse_rl.compatibility_repair import (
    REPAIR_MAX_TOKENS,
    REPAIR_SYSTEM_PROMPT,
    build_repair_request,
    parse_repair_output,
)
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.evaluation.fixed_probe import evaluate_candidate_profile
from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    validate_mutable_decision_procedure,
)
from scripts.generic_m20_probe_support import evaluation_system, state_hash
from scripts.prepare_v18_m2f_trigger_extension_pilot import canonical_hash


AUTH_ENV = "V18_M2F_TRIGGER_EXTENSION_AUTHORIZED"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def freeze_errors(
    registry: Mapping[str, Any], freeze: Mapping[str, Any],
    phase_a: Mapping[str, Any],
) -> list[str]:
    errors = []
    head = git("rev-parse", "HEAD")
    if head != str(registry.get("execution_commit")):
        errors.append("registry_execution_commit")
    if head != str(freeze.get("execution_commit")):
        errors.append("freeze_execution_commit")
    if git("status", "--porcelain"):
        errors.append("tracked_worktree_dirty")
    content = dict(registry)
    recorded_hash = str(content.pop("registry_content_hash", ""))
    if recorded_hash != canonical_hash(content):
        errors.append("registry_content_hash")
    if recorded_hash != str(freeze.get("registry_content_hash")):
        errors.append("freeze_registry_hash")
    if phase_a.get("phase_a_gate") != "PASS":
        errors.append("phase_a_gate")
    if int(registry.get("eligible_count", 0)) != 7:
        errors.append("eligible_inventory")
    for row in freeze.get("source_files", []):
        path = ROOT / str(row["path"])
        if not path.is_file():
            errors.append(f"missing:{row['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != str(row["sha256"]):
            errors.append(f"hash:{row['path']}")
    return errors


def prompt_answer_profile(rows: Sequence[Mapping[str, Any]]) -> tuple[PromptAnswer, ...]:
    return tuple(PromptAnswer(**dict(row)) for row in rows)


def evaluate_profile(
    *, system: Any, prompt: str, prompt_hash: str,
    profile: Sequence[PromptAnswer], target: int,
    assigned_hashes: set[str],
) -> tuple[CandidateEvaluation, Any, CandidateEvaluation]:
    evaluation = evaluate_candidate_profile(
        prompt=prompt,
        prompt_hash=prompt_hash,
        examples=system.fixed_probe.examples,
        active_profiles=system.active_profiles,
        initial_profiles=system.initial_profiles,
        candidate_profile=profile,
        target_agent_id=target,
        assigned_question_hashes=assigned_hashes,
        normalize_answer=system.normalize_answer,
        match_answer=system.match_answer,
        tie_break=system.protocol.tie_policy,
        seed=system.cfg.training.seed,
        tau=system.cfg.peer_state.soft_vote_tau,
    )
    incumbent = system.active_evaluation(target)
    return evaluation, evaluate_constraints(evaluation, incumbent), incumbent


def validation_oracle_delta(
    candidate: CandidateEvaluation, incumbent: CandidateEvaluation,
) -> int:
    return sum(value > 0 for value in candidate.team_outcome.gold_vote_counts) - sum(
        value > 0 for value in incumbent.team_outcome.gold_vote_counts
    )


def compact_metrics(evaluation: CandidateEvaluation, constraint: Any) -> dict[str, Any]:
    return {
        "target_correct_count": int(evaluation.competence.correct_count),
        "target_gain": int(constraint.target_gain),
        "vote_gain_count": int(constraint.vote_gain_count),
        "vote_loss_count": int(constraint.vote_loss_count),
        "vote_net_gain": int(constraint.vote_net_gain),
        "terminal_invalid_count": int(evaluation.competence.terminal_invalid_count),
        "common_safe_feasible": bool(constraint.passed),
        "rejection_reasons": list(constraint.rejection_reasons),
    }


async def train_repair(
    *, case: dict[str, Any], cell: Path, cache: Path,
) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
    system = evaluation_system(case, out_dir=cell / "train", cache_path=cache)
    target = int(case["target_agent_id"])
    assigned = set(map(str, case["assigned_question_hashes"]))
    before = state_hash(system)
    event: dict[str, Any] = {
        "case_id": case["case_id"],
        "seed": int(case["source_seed"]),
        "update_index": int(case["source_update_index"]),
        "target_agent_id": target,
        "parent_team_hash": case["parent_team_hash"],
        "source_candidate_hash": case["source_candidate_hash"],
        "source_metrics": case["source_metrics"],
        "extended_m2f_eligible": bool(case["extended_m2f_eligible"]),
        "repair_attempted": True,
        "repair_output_valid": False,
        "repair_evaluable": False,
        "repair_feasible": False,
        "repair_candidate_hash": "",
        "targeting_retained": False,
        "responsibility_targeting_retention": None,
        "target_gain_retention_ratio": None,
        "team_prompt_commits": 0,
        "trajectory_mutations": 0,
        "validation_calls_before_train_freeze": 0,
        "test_calls": 0,
    }
    request = build_repair_request(
        parent_prompt=case["parent_prompt"],
        source_candidate_prompt=case["source_candidate_prompt"],
        repair_evidence=case["repair_evidence"],
        loss_evidence=case["loss_evidence"],
        numeric_summary=case["numeric_summary"],
    )
    if hashlib.sha256(request.encode("utf-8")).hexdigest() != case["repair_input_hash"]:
        raise RuntimeError("frozen repair request mismatch")
    result = await system.llm.chat_result(
        str(case["base_config"]["optimizer_model"]),
        REPAIR_SYSTEM_PROMPT,
        request,
        system.cfg.tcs.student_temperature,
        REPAIR_MAX_TOKENS,
        "optimizer",
        "extended_m2f_compatibility_repair",
    )
    try:
        repaired_prompt = parse_repair_output(
            result.text,
            source_candidate_prompt=case["source_candidate_prompt"],
            supplied_evidence=[*case["repair_evidence"], *case["loss_evidence"]],
        )
        validate_mutable_decision_procedure(repaired_prompt)
    except ValueError as exc:
        event["terminal_failure_class"] = type(exc).__name__
        if state_hash(system) != before:
            raise RuntimeError("invalid repair mutated parent state")
        atomic_write(cell / "train_result.json", event)
        return event, None, list(system.llm.calls)
    event["repair_output_valid"] = True
    repaired_hash = system.prompt_hash(repaired_prompt)
    repaired_profile = await system.fixed_probe.evaluate_prompt(
        target, repaired_prompt, repaired_hash, system.solve
    )
    evaluation, constraint, _ = evaluate_profile(
        system=system,
        prompt=repaired_prompt,
        prompt_hash=repaired_hash,
        profile=repaired_profile,
        target=target,
        assigned_hashes=assigned,
    )
    event["repair_evaluable"] = True
    event["repair_feasible"] = bool(constraint.passed)
    event["repair_candidate_hash"] = repaired_hash
    event["repair_metrics"] = compact_metrics(evaluation, constraint)
    event["repair_metrics"]["ranking_key"] = list(
        common_monotone_safe_key(evaluation, int(case["source_generation"]))
    )
    source_repairs = {
        str(row["question_hash"]) for row in case["repair_evidence"]
    }
    repaired_repairs = {
        example.question_hash
        for example, before_answer, after_answer in zip(
            system.fixed_probe.examples,
            system.active_profiles[target],
            repaired_profile,
            strict=True,
        )
        if example.question_hash in assigned
        and not (
            before_answer.valid
            and system.match_answer(before_answer.answer, example.gold_answer)
        )
        and after_answer.valid
        and system.match_answer(after_answer.answer, example.gold_answer)
    }
    retained = len(source_repairs & repaired_repairs)
    retention = retained / max(1, len(source_repairs))
    event["retained_source_responsibility_repairs"] = retained
    event["repair_responsibility_gain_count"] = len(repaired_repairs)
    event["responsibility_targeting_retention"] = retention
    event["targeting_retained"] = retention >= 0.8
    source_target_gain = int(case["source_metrics"]["target_gain"])
    event["target_gain_retention_ratio"] = (
        int(constraint.target_gain) / source_target_gain
        if source_target_gain > 0 else None
    )
    if state_hash(system) != before:
        raise RuntimeError("repair evaluation mutated parent state")
    atomic_write(cell / "repair_private.json", {
        "repair_candidate_hash": repaired_hash,
        "repair_candidate_prompt": repaired_prompt,
    })
    atomic_write(cell / "train_result.json", event)
    return event, repaired_prompt, list(system.llm.calls)


async def validation_pair(
    *, case: dict[str, Any], repaired_prompt: str | None,
    cell: Path, cache: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation_case = {
        **case,
        "questions": case["validation_questions"],
        "initial_profiles": case["validation_parent_profiles"],
        "active_profiles": case["validation_parent_profiles"],
    }
    system = evaluation_system(
        validation_case, out_dir=cell / "validation", cache_path=cache
    )
    target = int(case["target_agent_id"])
    before = state_hash(system)
    source_cached = case.get("source_validation_profile_cached")
    if source_cached is not None:
        source_profile = prompt_answer_profile(source_cached)
        source_profile_origin = "historical_read_only_cache"
    else:
        source_profile = await system.fixed_probe.evaluate_prompt(
            target,
            case["source_candidate_prompt"],
            case["source_candidate_hash"],
            system.solve,
        )
        source_profile_origin = "prospective_frozen_evaluator"
    source_eval, source_constraint, source_incumbent = evaluate_profile(
        system=system,
        prompt=case["source_candidate_prompt"],
        prompt_hash=case["source_candidate_hash"],
        profile=source_profile,
        target=target,
        assigned_hashes=set(),
    )
    payload: dict[str, Any] = {
        "source_profile_origin": source_profile_origin,
        "source_validation_metrics": {
            **compact_metrics(source_eval, source_constraint),
            "oracle_delta": validation_oracle_delta(source_eval, source_incumbent),
        },
        "repair_validation_evaluated": False,
        "repair_validation_metrics": None,
    }
    if repaired_prompt is not None:
        repaired_hash = system.prompt_hash(repaired_prompt)
        repaired_profile = await system.fixed_probe.evaluate_prompt(
            target, repaired_prompt, repaired_hash, system.solve
        )
        repaired_eval, repaired_constraint, repaired_incumbent = evaluate_profile(
            system=system,
            prompt=repaired_prompt,
            prompt_hash=repaired_hash,
            profile=repaired_profile,
            target=target,
            assigned_hashes=set(),
        )
        payload["repair_validation_evaluated"] = True
        payload["repair_validation_metrics"] = {
            **compact_metrics(repaired_eval, repaired_constraint),
            "oracle_delta": validation_oracle_delta(
                repaired_eval, repaired_incumbent
            ),
        }
    if state_hash(system) != before:
        raise RuntimeError("validation evaluation mutated parent state")
    atomic_write(cell / "validation_result.json", payload)
    return payload, list(system.llm.calls)


def call_counts(calls: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    solver = sum(
        str(row.get("role", row.get("call_role", ""))).lower() == "solver"
        or "solver" in str(row.get("purpose", "")).lower()
        for row in calls
    )
    optimizer = sum(
        str(row.get("role", row.get("call_role", ""))).lower() == "optimizer"
        or "repair" in str(row.get("purpose", "")).lower()
        for row in calls
    )
    return {
        "model_calls": len(calls),
        "solver_calls": int(solver),
        "optimizer_calls": int(optimizer),
    }


async def main_async(args: argparse.Namespace) -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise SystemExit(f"API execution blocked: {AUTH_ENV}=1 required")
    registry = read_json(args.registry.resolve())
    freeze = read_json(args.source_freeze.resolve())
    phase_a = read_json(args.phase_a_gate.resolve())
    errors = freeze_errors(registry, freeze, phase_a)
    if errors:
        raise SystemExit("source freeze gate failed: " + ",".join(errors))
    root = args.out_root.resolve()
    if root.exists() or ROOT.resolve() not in root.parents:
        raise SystemExit("out_root must be fresh and project-local")
    root.mkdir(parents=True)
    cache = root / "shared_solver_cache.sqlite"
    train_rows = []
    repaired_prompts: dict[str, str | None] = {}
    all_calls: list[dict[str, Any]] = []
    for index, case in enumerate(registry["cases"], start=1):
        if freeze_errors(registry, freeze, phase_a):
            raise RuntimeError("source freeze changed during train phase")
        cell = root / f"cell_{index}_{case['source_candidate_hash'][:12]}"
        cell.mkdir(parents=True)
        row, prompt, calls = await train_repair(
            case=case, cell=cell, cache=cache
        )
        train_rows.append(row)
        repaired_prompts[str(case["case_id"])] = prompt
        all_calls.extend(calls)
        print(json.dumps({
            "phase": "train_repair",
            "completed": index,
            "total": len(registry["cases"]),
            "valid": sum(bool(item["repair_output_valid"]) for item in train_rows),
            "feasible": sum(bool(item["repair_feasible"]) for item in train_rows),
        }), flush=True)
    train_freeze = {
        "train_decisions_frozen": True,
        "case_count": len(train_rows),
        "repair_attempt_count": sum(bool(row["repair_attempted"]) for row in train_rows),
        "valid_repair_count": sum(bool(row["repair_output_valid"]) for row in train_rows),
        "feasible_repair_count": sum(bool(row["repair_feasible"]) for row in train_rows),
        "train_result_hashes": {
            row["case_id"]: canonical_hash(row) for row in train_rows
        },
        "validation_calls_before_freeze": 0,
        "test_calls": 0,
    }
    train_freeze["freeze_hash"] = canonical_hash(train_freeze)
    atomic_write(root / "train_decisions_frozen.json", train_freeze)

    validation_rows = []
    for index, case in enumerate(registry["cases"], start=1):
        if freeze_errors(registry, freeze, phase_a):
            raise RuntimeError("source freeze changed during validation phase")
        cell = root / f"cell_{index}_{case['source_candidate_hash'][:12]}"
        row, calls = await validation_pair(
            case=case,
            repaired_prompt=repaired_prompts[str(case["case_id"])],
            cell=cell,
            cache=cache,
        )
        validation_rows.append({"case_id": case["case_id"], **row})
        all_calls.extend(calls)
        print(json.dumps({
            "phase": "validation",
            "completed": index,
            "total": len(registry["cases"]),
        }), flush=True)

    logical_train_evaluations = sum(
        bool(row["repair_evaluable"]) for row in train_rows
    )
    logical_validation_evaluations = len(validation_rows) + sum(
        bool(row["repair_validation_evaluated"]) for row in validation_rows
    )
    summary = {
        "pilot_version": "v18_m2f_trigger_extension_pilot_v1",
        "execution_commit": git("rev-parse", "HEAD"),
        "registry_content_hash": registry["registry_content_hash"],
        "phase_a_gate": "PASS",
        "phase_b_gate": "PASS",
        "eligible_source_count": 7,
        "repair_attempt_count": sum(bool(row["repair_attempted"]) for row in train_rows),
        "valid_repair_count": sum(bool(row["repair_output_valid"]) for row in train_rows),
        "evaluable_repair_count": sum(bool(row["repair_evaluable"]) for row in train_rows),
        "feasible_repair_count": sum(bool(row["repair_feasible"]) for row in train_rows),
        "logical_train_evaluator_calls": logical_train_evaluations,
        "logical_validation_evaluator_calls": logical_validation_evaluations,
        "new_test_calls": 0,
        "call_counts": call_counts(all_calls),
        "train_decisions_freeze_hash": train_freeze["freeze_hash"],
        "team_prompt_commits": 0,
        "trajectory_mutations": 0,
        "cells": [
            {**train, **validation}
            for train, validation in zip(train_rows, validation_rows, strict=True)
        ],
    }
    atomic_write(root / "pilot_summary.json", summary)
    atomic_write(root / "call_metadata.json", all_calls)
    print(json.dumps({
        key: summary[key]
        for key in (
            "phase_b_gate", "repair_attempt_count", "valid_repair_count",
            "feasible_repair_count", "logical_validation_evaluator_calls",
            "new_test_calls", "call_counts",
        )
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--phase_a_gate", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
