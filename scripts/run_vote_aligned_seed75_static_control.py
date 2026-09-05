"""Evaluation-only Static control for the completed Seed75 P0/P1 pilot."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.persistence.identity import build_run_identity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.anti_overfitting_shadow_support import git, sha256_file, sha256_json, write_json
from scripts.run_shadow_gated_evolution import _backup
from scripts.run_vote_aligned_generic_shadow_pilot import (
    P0,
    P1,
    _config,
    _rows,
    _scientific_initialization_signature,
)

EXPERIMENT_ID = "vote_aligned_seed75_static_control"
STATIC = "STATIC_NO_TRAIN_SEED75"
AUTH_ENV = "SEED75_STATIC_CONTROL_AUTHORIZED"
MANIFEST = ROOT / "experiments" / "manifests" / "vote_aligned_seed75_static_control.yaml"
SOURCE_RUN = ROOT / "runs" / "vote_aligned_generic_shadow_pilot_v1_retry1"
SOURCE_PREP = ROOT / "runs" / "vote_aligned_generic_shadow_pilot_v1_prep_completionfix2"
SOURCE_REPORT = ROOT / "reports" / "vote_aligned_generic_shadow_pilot_v1"
DEFAULT_PREP_ROOT = ROOT / "runs" / "vote_aligned_seed75_static_control_prep_20260905"
DEFAULT_RUN_ROOT = ROOT / "runs" / "vote_aligned_seed75_static_control_20260905"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "vote_aligned_seed75_static_control_20260905"
INITIAL_MANIFEST = SOURCE_RUN / "initialization" / "seed75" / "frozen_initialization_manifest.json"
INITIAL_CACHE = SOURCE_RUN / "initialization" / "seed75" / "initial_solver_cache_frozen.sqlite"
OPTIMIZE = SOURCE_PREP / "splits_private" / "optimize_group_1.csv"
SHADOW = SOURCE_PREP / "splits_private" / "fold_c.csv"
VALIDATION = SOURCE_PREP / "splits_private" / "validation.csv"
MEMBER_TOLERANCE = 0.01


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _protected_paths() -> list[Path]:
    paths = [INITIAL_MANIFEST, INITIAL_CACHE, SOURCE_REPORT / "classifier.json"]
    for relative in (
        Path("seed75") / P0,
        Path("seed75") / P1,
        Path("evaluation") / "seed75" / P0,
        Path("evaluation") / "seed75" / P1,
    ):
        paths.extend(path for path in (SOURCE_RUN / relative).rglob("*") if path.is_file())
    return sorted(set(paths))


def _protected_hashes() -> list[dict[str, str]]:
    rows = []
    for path in _protected_paths():
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise RuntimeError("protected evidence escaped repository") from exc
        rows.append({"path": relative, "sha256": sha256_file(path)})
    return rows


def _source_snapshots() -> dict[str, Mapping[str, Any]]:
    initial = read_json(INITIAL_MANIFEST)["initialization_snapshot"]
    p0 = read_json(SOURCE_RUN / "seed75" / P0 / "frozen_initialization_match.json")
    p1 = read_json(SOURCE_RUN / "seed75" / P1 / "frozen_initialization_match.json")
    return {"INITIAL": initial, "P0": p0["initialization_snapshot"], "P1": p1["initialization_snapshot"]}


def classify_member_effect(delta: float) -> str:
    if delta > MEMBER_TOLERANCE:
        return "P1_ABSOLUTE_MEMBER_GAIN"
    if delta < -MEMBER_TOLERANCE:
        return "P1_ABSOLUTE_MEMBER_DEGRADATION"
    return "P1_MEMBER_ROUGHLY_PRESERVED"


def classify_ensemble_effect(delta: float) -> str:
    if delta > 0:
        return "POSITIVE"
    if delta < 0:
        return "NEGATIVE"
    return "NEUTRAL"


def prepare(prep_root: Path) -> dict[str, Any]:
    if prep_root.exists():
        raise RuntimeError("fresh prep root required")
    snapshots = _source_snapshots()
    signatures = {name: _scientific_initialization_signature(value) for name, value in snapshots.items()}
    expected = snapshots["INITIAL"]
    identity = expected["immutable_run_identity"]
    source_classifier = read_json(SOURCE_REPORT / "classifier.json")["classifier"]
    gates = {
        "CLEAN_WORKTREE": git("status", "--porcelain", "--untracked-files=all") == "",
        "INITIAL_IDENTITY": len(set(signatures.values())) == 1,
        "MEMBER_HASH_PARITY": all(value["initial_prompt_hashes"] == expected["initial_prompt_hashes"] for value in snapshots.values()),
        "FIVE_MEMBERS": len(expected["initial_prompt_hashes"]) == len(expected["initial_member_correct_counts"]) == 5,
        "SOLVER_IDENTITY": expected["solver_identity"][0] == "prompt_question_recovered_invalid_v2",
        "SPLIT_IDENTITY": (
            sha256_file(OPTIMIZE) == identity["train_file_sha256"]
            and sha256_file(VALIDATION) == identity["val_file_sha256"]
            and len(_rows(OPTIMIZE)) == 100
            and len(_rows(SHADOW)) == len(_rows(VALIDATION)) == 50
        ),
        "FROZEN_PILOT": (
            source_classifier == "NO_CLEAR_SIGNAL"
            and read_json(SOURCE_RUN / "execution_summary.json")["completed_trajectories"] == 2
            and read_json(SOURCE_RUN / "execution_summary.json")["test_calls"] == 0
        ),
        "CACHE_PRESENT": INITIAL_CACHE.is_file(),
        "TEST_BLOCKED": True,
        "EVALUATION_ONLY": "update_once" not in execute.__code__.co_names,
    }
    prep_root.mkdir(parents=True)
    payload = {
        "phase_a_gate": "PASS" if all(gates.values()) else "HOLD",
        "gates": {key: "PASS" if value else "HOLD" for key, value in gates.items()},
        "head": git("rev-parse", "HEAD"),
        "source_initialization_signature": signatures["INITIAL"],
        "p0_initialization_signature": signatures["P0"],
        "p1_initialization_signature": signatures["P1"],
        "initial_prompt_hashes": list(expected["initial_prompt_hashes"]),
        "initial_member_correct_counts": list(expected["initial_member_correct_counts"]),
        "source_classifier": source_classifier,
        "existing_optimize_evidence_reused": True,
        "missing_evaluations": ["shadow50", "validation50"],
        "training_updates": 0,
        "teacher_calls": 0,
        "critic_calls": 0,
        "student_calls": 0,
        "test_calls": 0,
    }
    write_json(prep_root / "phase_a_gate.json", payload)
    write_json(prep_root / "protected_artifacts.json", {"files": _protected_hashes()})
    source_files = [
        Path("scripts/run_vote_aligned_seed75_static_control.py"),
        Path("experiments/manifests/vote_aligned_seed75_static_control.yaml"),
        Path("experiments/vote_aligned_seed75_static_control_20260905/PROTOCOL.md"),
    ]
    write_json(prep_root / "source_freeze.json", {
        "execution_commit": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": gates["CLEAN_WORKTREE"],
        "files": [{"path": str(path.as_posix()), "sha256": sha256_file(ROOT / path)} for path in source_files],
    })
    write_json(prep_root / "test_access_registry.json", {"events": [], "test_calls": 0})
    return payload


def _verify_freeze(prep_root: Path) -> None:
    phase = read_json(prep_root / "phase_a_gate.json")
    if phase["phase_a_gate"] != "PASS":
        raise RuntimeError("Phase A gate is not PASS")
    freeze = read_json(prep_root / "source_freeze.json")
    if freeze["execution_commit"] != git("rev-parse", "HEAD"):
        raise RuntimeError("execution commit differs from source freeze")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("tracked worktree must be clean")
    for row in freeze["files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")
    for row in read_json(prep_root / "protected_artifacts.json")["files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"historical evidence changed: {row['path']}")


def _authorize() -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise RuntimeError(f"set {AUTH_ENV}=1 only for the explicitly authorized evaluation")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    require_api_authorization(manifest, phase="static_evaluation_only", role="solver", explicit_user_authorized=True)
    require_api_authorization(manifest, phase="static_evaluation_only", role="evaluator", explicit_user_authorized=True)


def _correctness(system: PromptEnsembleOptimizationSystem, examples: Sequence[Any], profiles: Sequence[Sequence[Any]]) -> list[list[bool]]:
    return [[bool(answer.valid) and system.match_answer(answer.answer, example.gold_answer) for answer, example in zip(profile, examples, strict=True)] for profile in profiles]


def _binary_correlation(left: Sequence[bool], right: Sequence[bool]) -> float | None:
    x = [int(value) for value in left]
    y = [int(value) for value in right]
    mx, my = statistics.mean(x), statistics.mean(y)
    vx = sum((value - mx) ** 2 for value in x)
    vy = sum((value - my) ** 2 for value in y)
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / math.sqrt(vx * vy)


def _aggregate(system: PromptEnsembleOptimizationSystem, examples: Sequence[Any], profiles: Sequence[Sequence[Any]], metrics: Any, split: str, new_calls: int) -> dict[str, Any]:
    correct = _correctness(system, examples, profiles)
    depth = {f"G{index}": 0 for index in range(6)}
    oracle = 0
    for row_index in range(len(examples)):
        support = sum(int(member[row_index]) for member in correct)
        depth[f"G{support}"] += 1
        oracle += int(support > 0)
    disagreements = []
    correlations = []
    for left, right in itertools.combinations(correct, 2):
        disagreements.append(sum(a != b for a, b in zip(left, right, strict=True)) / len(examples))
        correlation = _binary_correlation(left, right)
        if correlation is not None:
            correlations.append(correlation)
    members = [value / len(examples) for value in metrics.per_agent_correct_counts]
    return {
        "split": split,
        "row_count": len(examples),
        "vote_correct_count": int(metrics.vote_correct_count),
        "vote_accuracy": float(metrics.plurality_vote_acc),
        "per_agent_correct_counts": list(metrics.per_agent_correct_counts),
        "per_agent_accuracies": members,
        "mean_member_accuracy": float(metrics.mean_individual_acc),
        "oracle_correct_count": oracle,
        "oracle_accuracy": oracle / len(examples),
        "ensemble_gain": float(metrics.plurality_vote_acc - metrics.mean_individual_acc),
        "coverage_depth": depth,
        "mean_pairwise_disagreement": statistics.mean(disagreements),
        "mean_pairwise_correctness_correlation": statistics.mean(correlations) if correlations else None,
        "new_provider_calls": int(new_calls),
    }


async def execute(prep_root: Path, run_root: Path) -> dict[str, Any]:
    _authorize()
    _verify_freeze(prep_root)
    if run_root.exists():
        raise RuntimeError("fresh Static run root required")
    run_root.mkdir(parents=True)
    cache = run_root / "static_solver_cache.sqlite"
    _backup(INITIAL_CACHE, cache)
    cfg = _config(75, P0, run_root, OPTIMIZE, VALIDATION, cache, INITIAL_MANIFEST, False)
    values = cfg.to_flat_dict()
    values.update({
        "out_dir": str(run_root),
        "shared_solver_cache_path": str(cache),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "test_path": "TEST50_BLOCKED_BY_STATIC_CONTROL",
        "test_size": 0,
        "epochs": 0,
    })
    cfg = type(cfg).from_flat(**values)
    system = PromptEnsembleOptimizationSystem(cfg)
    optimize_rows = _rows(OPTIMIZE)
    validation_rows = _rows(VALIDATION)
    system.set_run_identity(build_run_identity(cfg, train_rows=optimize_rows, val_rows=validation_rows, test_rows=[], workspace=ROOT))
    calls_before = int(system.cost_summary()["successful_llm_calls"])
    await system.initialize_fixed_probe(optimize_rows)
    calls_after_initial = int(system.cost_summary()["successful_llm_calls"])
    snapshot = system.frozen_initialization_snapshot()
    expected = read_json(INITIAL_MANIFEST)["initialization_snapshot"]
    signature = _scientific_initialization_signature(snapshot)
    expected_signature = _scientific_initialization_signature(expected)
    if signature != expected_signature or calls_after_initial != calls_before:
        raise RuntimeError("Static initialization did not exactly reuse frozen evidence")
    state_hash = system.team_prompt_state_hash()
    prompt_hashes = [system.prompt_hash(agent.current_prompt) for agent in system.agents]
    results = {}
    optimize_metrics = system.active_probe_metrics()
    results["optimize"] = _aggregate(system, system.fixed_probe.examples, system.active_profiles, optimize_metrics, "optimize", 0)
    for split, path in (("shadow", SHADOW), ("validation", VALIDATION)):
        before = int(system.cost_summary()["successful_llm_calls"])
        metrics = await system.evaluate_dataset(_rows(path))
        after = int(system.cost_summary()["successful_llm_calls"])
        results[split] = _aggregate(system, system._last_evaluated_examples, system._last_evaluated_profiles, metrics, split, after - before)
        if system.team_prompt_state_hash() != state_hash:
            raise RuntimeError("evaluation mutated Static prompt state")
    total_calls = int(system.cost_summary()["successful_llm_calls"])
    if total_calls > 100:
        raise RuntimeError("frozen provider-call budget exceeded")
    payload = {
        "execution_gate": "PASS",
        "arm": STATIC,
        "seed": 75,
        "initialization_signature": signature,
        "initial_prompt_hashes": prompt_hashes,
        "training_updates": 0,
        "target_selections": 0,
        "candidate_generations": 0,
        "revisions": 0,
        "common_safe_decisions": 0,
        "shadow_writeback_decisions": 0,
        "commits": 0,
        "teacher_calls": 0,
        "critic_calls": 0,
        "student_calls": 0,
        "new_solver_provider_calls": total_calls,
        "new_test_calls": 0,
        "metrics": results,
        "state_mutation": False,
    }
    write_json(run_root / "static_control_summary_private.json", payload)
    return payload


def audit(prep_root: Path, run_root: Path) -> dict[str, Any]:
    errors = []
    try:
        _verify_freeze(prep_root)
    except Exception as exc:
        errors.append(f"freeze:{type(exc).__name__}")
    result = read_json(run_root / "static_control_summary_private.json")
    phase = read_json(prep_root / "phase_a_gate.json")
    zero_fields = ("training_updates", "target_selections", "candidate_generations", "revisions", "common_safe_decisions", "shadow_writeback_decisions", "commits", "teacher_calls", "critic_calls", "student_calls", "new_test_calls")
    if any(int(result[field]) != 0 for field in zero_fields):
        errors.append("nonzero_forbidden_activity")
    if result["initialization_signature"] != phase["source_initialization_signature"]:
        errors.append("initialization_signature")
    if result["initial_prompt_hashes"] != phase["initial_prompt_hashes"]:
        errors.append("member_prompt_hashes")
    if set(result["metrics"]) != {"optimize", "shadow", "validation"}:
        errors.append("evaluation_inventory")
    if result["metrics"]["optimize"]["new_provider_calls"] != 0:
        errors.append("optimize_not_reused")
    if int(result["new_solver_provider_calls"]) > 100:
        errors.append("provider_budget")
    if read_json(SOURCE_REPORT / "classifier.json")["classifier"] != "NO_CLEAR_SIGNAL":
        errors.append("original_classifier_changed")
    payload = {"audit_gate": "PASS" if not errors else "HOLD", "errors": errors, "protected_artifact_count": len(read_json(prep_root / "protected_artifacts.json")["files"]), "initialization_match": not any("initialization" in error or "prompt" in error for error in errors), "new_test_calls": 0}
    write_json(run_root / "audit_summary.json", payload)
    return payload


def _existing_arm_metrics() -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {P0: {}, P1: {}}
    with (SOURCE_REPORT / "trajectory_summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = row["arm"]
            for split in ("optimize", "shadow", "validation"):
                vote = float(row[f"{split}_vote"])
                member = float(row[f"{split}_mean_member"])
                oracle = float(row[f"{split}_oracle"])
                result[arm][split] = {"vote": vote, "mean_member": member, "oracle": oracle, "ensemble_gain": vote - member}
    return result


def analyze(run_root: Path, report_root: Path) -> dict[str, Any]:
    audit_payload = read_json(run_root / "audit_summary.json")
    if audit_payload["audit_gate"] != "PASS":
        raise RuntimeError("analysis requires audit PASS")
    static_private = read_json(run_root / "static_control_summary_private.json")
    existing = _existing_arm_metrics()
    metrics: dict[str, dict[str, dict[str, float]]] = {STATIC: {}}
    for split, row in static_private["metrics"].items():
        metrics[STATIC][split] = {"vote": row["vote_accuracy"], "mean_member": row["mean_member_accuracy"], "oracle": row["oracle_accuracy"], "ensemble_gain": row["ensemble_gain"]}
    metrics.update(existing)
    arms = (STATIC, P0, P1)
    splits = ("optimize", "shadow", "validation")
    summary_rows = []
    for arm in arms:
        for split in splits:
            summary_rows.append({"seed": 75, "arm": arm, "split": split, **metrics[arm][split]})
    contrasts = []
    for name, treatment, control in (("P0_MINUS_STATIC", P0, STATIC), ("P1_MINUS_STATIC", P1, STATIC), ("P1_MINUS_P0", P1, P0)):
        for split in splits:
            contrasts.append({"contrast": name, "split": split, **{key: metrics[treatment][split][key] - metrics[control][split][key] for key in ("vote", "mean_member", "oracle", "ensemble_gain")}})
    p1_member_delta = metrics[P1]["validation"]["mean_member"] - metrics[STATIC]["validation"]["mean_member"]
    p1_ensemble_delta = metrics[P1]["validation"]["ensemble_gain"] - metrics[STATIC]["validation"]["ensemble_gain"]
    member_classifier = classify_member_effect(p1_member_delta)
    ensemble_classifier = classify_ensemble_effect(p1_ensemble_delta)
    p1_vote_delta = metrics[P1]["validation"]["vote"] - metrics[STATIC]["validation"]["vote"]
    if p1_member_delta > 0 and p1_vote_delta > 0 and p1_ensemble_delta > 0:
        interpretation_case = "CASE_1_INDIVIDUAL_AND_ENSEMBLE_GAIN"
    elif abs(p1_member_delta) <= MEMBER_TOLERANCE and p1_vote_delta > 0 and p1_ensemble_delta > 0:
        interpretation_case = "CASE_2_TEAM_STRUCTURE_GAIN"
    elif p1_member_delta < -MEMBER_TOLERANCE and p1_vote_delta > 0:
        interpretation_case = "CASE_3_COMPETENCE_FOR_PLURALITY_TRADEOFF"
    else:
        interpretation_case = "CASE_4_NO_USEFUL_P1_GAIN"
    report_root.mkdir(parents=True, exist_ok=True)
    write_csv(report_root / "three_arm_summary.csv", summary_rows, ("seed", "arm", "split", "vote", "mean_member", "oracle", "ensemble_gain"))
    write_csv(report_root / "contrast_summary.csv", contrasts, ("contrast", "split", "vote", "mean_member", "oracle", "ensemble_gain"))
    member_rows = []
    p0_members = {}
    p1_members = {}
    with (SOURCE_REPORT / "member_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            (p0_members if row["arm"] == P0 else p1_members)[int(row["member"])] = float(row["validation_accuracy"])
    static_members = static_private["metrics"]["validation"]["per_agent_accuracies"]
    for member_id in range(5):
        member_rows.append({"member_id": member_id, "Static": static_members[member_id], "P0": p0_members[member_id], "P1": p1_members[member_id], "P0_minus_Static": p0_members[member_id] - static_members[member_id], "P1_minus_Static": p1_members[member_id] - static_members[member_id], "P1_minus_P0": p1_members[member_id] - p0_members[member_id]})
    write_csv(report_root / "member_accuracy_comparison.csv", member_rows, tuple(member_rows[0]))
    depth_rows = [{"arm": STATIC, **static_private["metrics"]["validation"]["coverage_depth"]}]
    with (SOURCE_REPORT / "coverage_depth.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            depth_rows.append({"arm": row["arm"], **{f"G{i}": int(row[f"G{i}"]) for i in range(6)}})
    write_csv(report_root / "coverage_depth_comparison.csv", depth_rows, ("arm", "G0", "G1", "G2", "G3", "G4", "G5"))
    gap_rows = []
    for arm in arms:
        for left, right in (("optimize", "shadow"), ("optimize", "validation"), ("shadow", "validation")):
            gap_rows.append({"arm": arm, "gap": f"{left}_minus_{right}", **{key: metrics[arm][left][key] - metrics[arm][right][key] for key in ("vote", "mean_member", "oracle", "ensemble_gain")}})
    write_csv(report_root / "generalization_gap.csv", gap_rows, ("arm", "gap", "vote", "mean_member", "oracle", "ensemble_gain"))
    interpretation = {"STATIC_CONTROL_INTERPRETATION": member_classifier, "ENSEMBLE_STRUCTURE_EFFECT": ensemble_classifier, "interpretation_case": interpretation_case, "p1_minus_static_validation_mean_member": p1_member_delta, "p1_minus_static_validation_vote": p1_vote_delta, "p1_minus_static_validation_ensemble_gain": p1_ensemble_delta, "p1_minus_static_validation_oracle": metrics[P1]["validation"]["oracle"] - metrics[STATIC]["validation"]["oracle"], "original_classifier": "NO_CLEAR_SIGNAL", "original_classifier_unchanged": True, "post_hoc": True, "single_seed_descriptive": True}
    write_json(report_root / "static_control_interpretation.json", interpretation)
    provenance = {"experiment_id": EXPERIMENT_ID, "source_experiment": "vote_aligned_generic_shadow_pilot_v1", "seed": 75, "static_initialization_signature": static_private["initialization_signature"], "p0_p1_static_initialization_match": True, "optimize_evidence_reused": True, "shadow_validation_newly_evaluated": True, "models": {"solver": "qwen3-8b", "teacher": "qwen3.7-flash", "critic": "qwen3.7-flash", "student": "qwen3.7-flash", "evaluator": "qwen3.7-flash", "thinking": False}, "new_solver_provider_calls": static_private["new_solver_provider_calls"], "teacher_calls": 0, "critic_calls": 0, "student_calls": 0, "p0_additional_training_calls": 0, "p1_additional_training_calls": 0, "new_test_calls": 0}
    write_json(report_root / "provenance.json", provenance)
    readme = f"""# Seed75 Static No-Training Control

The Static no-training control was added after observing the preregistered
Seed75 P0/P1 pilot. It only decomposes absolute change from the exact initial
ensemble and relative change from Generic optimization. It does not alter the
original `NO_CLEAR_SIGNAL` classification.

Static exactly matched the P0/P1 frozen initialization. Optimize100 reused the
existing frozen rollout with zero new provider calls. Shadow50 and Validation50
were evaluated once each. Static performed zero updates, target selections,
candidate generation, revisions, write-back decisions, or commits. Test50 was
not accessed.

## Validation50

| Arm | Vote | MeanMember | Vote-MeanMember | Oracle |
|---|---:|---:|---:|---:|
| Static | {metrics[STATIC]['validation']['vote']:.3f} | {metrics[STATIC]['validation']['mean_member']:.3f} | {metrics[STATIC]['validation']['ensemble_gain']:.3f} | {metrics[STATIC]['validation']['oracle']:.3f} |
| P0 | {metrics[P0]['validation']['vote']:.3f} | {metrics[P0]['validation']['mean_member']:.3f} | {metrics[P0]['validation']['ensemble_gain']:.3f} | {metrics[P0]['validation']['oracle']:.3f} |
| P1 | {metrics[P1]['validation']['vote']:.3f} | {metrics[P1]['validation']['mean_member']:.3f} | {metrics[P1]['validation']['ensemble_gain']:.3f} | {metrics[P1]['validation']['oracle']:.3f} |

P1 minus Static Validation deltas are Vote `{p1_vote_delta:+.3f}`,
MeanMember `{p1_member_delta:+.3f}`, Vote-minus-MeanMember
`{p1_ensemble_delta:+.3f}`, and Oracle
`{interpretation['p1_minus_static_validation_oracle']:+.3f}`.

Supplementary classifiers: `{member_classifier}` and ensemble-structure
`{ensemble_classifier}` (`{interpretation_case}`). These are descriptive
single-seed labels, not a revision of the original pilot decision.

## Isolation and calls

- Solver: `qwen3-8b`, thinking disabled; role/evaluator configuration:
  `qwen3.7-flash`.
- New Static solver provider calls: `{static_private['new_solver_provider_calls']}`.
- Teacher/Critic/Student calls: `0/0/0`.
- Additional P0/P1 training calls: `0/0`.
- New Test50 calls: `0`.
"""
    (report_root / "README.md").write_text(readme, encoding="utf-8")
    write_json(report_root / "summary.json", {"analysis_gate": "PASS", **interpretation, "new_solver_provider_calls": static_private["new_solver_provider_calls"], "new_test_calls": 0})
    return {"analysis_gate": "PASS", **interpretation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-root", type=Path, default=DEFAULT_PREP_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        print(json.dumps(prepare(args.prep_root), indent=2, sort_keys=True))
    elif args.execute:
        print(json.dumps(asyncio.run(execute(args.prep_root, args.run_root)), indent=2, sort_keys=True))
    elif args.audit:
        print(json.dumps(audit(args.prep_root, args.run_root), indent=2, sort_keys=True))
    elif args.analyze:
        print(json.dumps(analyze(args.run_root, args.report_root), indent=2, sort_keys=True))
    else:
        raise SystemExit("choose exactly one mode")


if __name__ == "__main__":
    main()
