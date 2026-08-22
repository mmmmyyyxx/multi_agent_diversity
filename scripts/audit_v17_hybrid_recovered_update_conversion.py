from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multi_dataset_diverse_rl.peer_state import TeamVoteState, build_team_vote_state
from v17_hybrid_target_allocation_support import HYBRID, RR, probe_system


AUDIT_VERSION = "v17_hybrid_recovered_update_conversion_audit_v1"
EXPECTED_EXECUTION_COMMIT = "43230a5734a971ead70841d71277105512fa0adc"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            key: (
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple)) else value
            )
            for key, value in row.items()
        } for row in rows)


def load_validation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            {"question": str(row["question"]), "answer": str(row["answer"])}
            for row in csv.DictReader(handle)
        ]


def cached_answers(
    cache_path: Path,
    prompt_hash: str,
    expected_questions: set[str],
    system: Any,
) -> dict[str, dict[str, Any]]:
    uri = f"file:{cache_path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT model_request_identity, question_hash, answer_json FROM solver_cache "
            "WHERE state='ready' AND prompt_hash=? "
            "AND parser_version=? "
            "AND temperature=? AND evaluation_replica_seed=? "
            "AND solver_model=? AND max_tokens=? AND output_contract_version=? "
            "ORDER BY cache_key",
            (
                str(prompt_hash),
                system.prompt_question_evaluator.parser_version,
                system.prompt_question_evaluator.temperature,
                system.prompt_question_evaluator.decoding_seed,
                system.cfg.models.agent_model,
                system.cfg.models.solver_max_tokens,
                system.cfg.peer_state.solver_output_contract_version,
            ),
        ).fetchall()
    finally:
        connection.close()
    selected = select_persisted_identity_rows(rows, expected_questions)
    result: dict[str, dict[str, Any]] = {}
    for question_hash, raw in selected:
        payload = json.loads(str(raw))
        if not isinstance(payload, dict):
            raise ValueError("cached validation observation must be an object")
        result[str(question_hash)] = payload
    return result


def select_persisted_identity_rows(
    rows: Iterable[tuple[Any, Any, Any]], expected_questions: set[str]
) -> list[tuple[str, str]]:
    """Select one exact persisted request identity without using current env identity."""
    grouped: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for raw_identity, raw_question, raw_answer in rows:
        identity = str(raw_identity)
        question = str(raw_question)
        if question not in expected_questions:
            continue
        bucket = grouped.setdefault(identity, {})
        if question in bucket:
            duplicates.add(identity)
        bucket[question] = str(raw_answer)
    exact = [
        identity for identity, values in grouped.items()
        if set(values) == expected_questions and identity not in duplicates
    ]
    if len(exact) != 1:
        raise ValueError("cache must contain exactly one complete persisted request identity")
    return sorted(grouped[exact[0]].items())


def team_state(system: Any, example: Any, observations: Iterable[dict[str, Any]]) -> TeamVoteState:
    rows = list(observations)
    return build_team_vote_state(
        question_hash=example.question_hash,
        gold_answer=example.gold_answer,
        answers=[str(row.get("answer", "")) for row in rows],
        valid_vector=[bool(row.get("valid")) for row in rows],
        normalize_answer=system.normalize_answer,
        match_answer=system.match_answer,
        tie_break=system.protocol.tie_policy,
        seed=system.cfg.training.seed,
    )


def coverage_role(state: TeamVoteState) -> str:
    if state.gold_vote_count == 0:
        return "uncovered"
    if state.vote_correct:
        return "plurality_correct"
    if state.gold_vote_count == 1:
        return "singleton_correct"
    return "nonwinning_correct_coalition"


def conversion_distances(state: TeamVoteState) -> tuple[int, int]:
    if state.vote_correct:
        return 0, 0
    margin_points_needed = max(0, 1 - int(state.plurality_margin))
    additional_gold_without_wrong_reduction = margin_points_needed
    dominant_wrong_to_gold_flips = math.ceil(margin_points_needed / 2)
    return additional_gold_without_wrong_reduction, dominant_wrong_to_gold_flips


def transition_row(
    *,
    transition_id: str,
    parent_id: str,
    target_member: int,
    candidate_id: str,
    conceptual_arms: str,
    question_hash: str,
    before: TeamVoteState,
    after: TeamVoteState,
) -> dict[str, Any]:
    before_target = bool(before.team_correctness[target_member])
    after_target = bool(after.team_correctness[target_member])
    oracle_before = before.gold_vote_count > 0
    oracle_after = after.gold_vote_count > 0
    no_reduction, dominant_flips = conversion_distances(after)
    if after.largest_wrong_vote_count < before.largest_wrong_vote_count:
        wrong_direction = "reduced"
    elif after.largest_wrong_vote_count > before.largest_wrong_vote_count:
        wrong_direction = "increased"
    else:
        wrong_direction = "unchanged"
    return {
        "transition_id": transition_id,
        "parent_id": parent_id,
        "target_member": int(target_member),
        "candidate_id": candidate_id,
        "conceptual_arms": conceptual_arms,
        "question_hash": question_hash,
        "target_correct_before": before_target,
        "target_correct_after": after_target,
        "target_gain": int(not before_target and after_target),
        "target_loss": int(before_target and not after_target),
        "oracle_before": oracle_before,
        "oracle_after": oracle_after,
        "oracle_gain": int(not oracle_before and oracle_after),
        "oracle_loss": int(oracle_before and not oracle_after),
        "vote_correct_before": bool(before.vote_correct),
        "vote_correct_after": bool(after.vote_correct),
        "vote_gain": int(not before.vote_correct and after.vote_correct),
        "vote_loss": int(before.vote_correct and not after.vote_correct),
        "G_before": int(before.gold_vote_count),
        "G_after": int(after.gold_vote_count),
        "H_before": int(before.largest_wrong_vote_count),
        "H_after": int(after.largest_wrong_vote_count),
        "M_before": int(before.plurality_margin),
        "M_after": int(after.plurality_margin),
        "delta_G": int(after.gold_vote_count - before.gold_vote_count),
        "delta_H": int(after.largest_wrong_vote_count - before.largest_wrong_vote_count),
        "delta_M": int(after.plurality_margin - before.plurality_margin),
        "wrong_coalition_direction": wrong_direction,
        "coverage_role_after": coverage_role(after),
        "additional_gold_votes_needed_without_wrong_reduction": no_reduction,
        "minimum_dominant_wrong_to_gold_flips_needed": dominant_flips,
        "g_transition": f"G{before.gold_vote_count}_to_G{after.gold_vote_count}",
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_gain_rows = [row for row in rows if row["oracle_gain"]]
    return {
        "changed_row_count": len(rows),
        "target_gain_count": sum(row["target_gain"] for row in rows),
        "target_loss_count": sum(row["target_loss"] for row in rows),
        "oracle_gain_count": sum(row["oracle_gain"] for row in rows),
        "oracle_loss_count": sum(row["oracle_loss"] for row in rows),
        "vote_gain_count": sum(row["vote_gain"] for row in rows),
        "vote_loss_count": sum(row["vote_loss"] for row in rows),
        "wrong_coalition_reduced_count": sum(
            row["wrong_coalition_direction"] == "reduced" for row in rows
        ),
        "wrong_coalition_increased_count": sum(
            row["wrong_coalition_direction"] == "increased" for row in rows
        ),
        "oracle_gain_g_transition_counts": dict(sorted(Counter(
            row["g_transition"] for row in oracle_gain_rows
        ).items())),
        "oracle_gain_coverage_role_after_counts": dict(sorted(Counter(
            row["coverage_role_after"] for row in oracle_gain_rows
        ).items())),
        "oracle_gain_wrong_coalition_direction_counts": dict(sorted(Counter(
            row["wrong_coalition_direction"] for row in oracle_gain_rows
        ).items())),
        "oracle_gain_additional_gold_votes_needed_counts": dict(sorted(Counter(
            str(row["additional_gold_votes_needed_without_wrong_reduction"])
            for row in oracle_gain_rows
        ).items())),
        "oracle_gain_minimum_dominant_flips_needed_counts": dict(sorted(Counter(
            str(row["minimum_dominant_wrong_to_gold_flips_needed"])
            for row in oracle_gain_rows
        ).items())),
    }


def winner_cells(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("prospective_*/cells/*/cell_result.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        if not cell.get("would_commit"):
            continue
        winning = [row for row in cell["branches"] if row["produced_cell_best"]]
        if len(winning) != 1 or not winning[0].get("branch_winner_id"):
            raise ValueError("WOULD_COMMIT cell lacks exactly one winner")
        branch = winning[0]
        branch_path = (
            run_root / str(cell["case_id"]) / "branches"
            / str(branch["canonical_branch_key"]) / "branch_result.json"
        )
        branch_payload = json.loads(branch_path.read_text(encoding="utf-8"))
        candidates = [
            row for row in branch_payload["candidates"]
            if row["candidate_id"] == branch["branch_winner_id"]
        ]
        if len(candidates) != 1 or not candidates[0]["feasible"]:
            raise ValueError("winner candidate evidence mismatch")
        rows.append({
            "parent_id": str(cell["case_id"]),
            "arm": str(cell["arm"]),
            "target_member": int(branch["target_member"]),
            "branch_type": str(branch["branch_type"]),
            "candidate_id": str(branch["branch_winner_id"]),
            "candidate_stage": str(candidates[0]["candidate_stage"]),
            "train": dict(candidates[0]["train"]),
            "validation_expected": {
                "target_delta": int(cell["realized_validation_target_delta"]),
                "vote_delta": int(cell["realized_validation_vote_delta"]),
                "oracle_delta": int(cell["realized_validation_oracle_delta"]),
                "vote_gain_count": int(cell["validation_vote_gain_count"]),
                "vote_loss_count": int(cell["validation_vote_loss_count"]),
                "oracle_gain_count": int(cell["validation_oracle_gain_count"]),
                "oracle_loss_count": int(cell["validation_oracle_loss_count"]),
            },
            "parent_validation": dict(cell["parent_validation"]),
            "decision_frozen_before_validation": bool(cell["decision_frozen_before_validation"]),
        })
    if len(rows) != 5 or Counter(row["arm"] for row in rows) != Counter({HYBRID: 2, RR: 3}):
        raise ValueError("expected exactly two Hybrid and three RR WOULD_COMMIT cells")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    registry_path = args.registry.resolve()
    scratch = args.scratch.resolve()
    out = args.out.resolve()
    project = ROOT.resolve()
    if any(project not in path.parents for path in (run_root, registry_path, scratch, out)):
        raise SystemExit("all inputs and outputs must be project-local")
    if scratch.exists():
        raise SystemExit("scratch directory must be fresh")
    if out.exists():
        raise SystemExit("output directory must be fresh")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must be clean")
    auditor_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_summary = json.loads((run_root / "probe_summary.json").read_text(encoding="utf-8"))
    if run_summary.get("execution_commit") != EXPECTED_EXECUTION_COMMIT:
        raise SystemExit("unexpected source-run execution commit")
    if run_summary.get("test_calls") != 0 or run_summary.get("team_prompt_commit_count") != 0:
        raise SystemExit("source run isolation mismatch")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cases = {str(row["case_id"]): row for row in registry["cases"]}
    cells = winner_cells(run_root)
    cache_path = run_root / "_shared_solver_cache.sqlite"
    evidence_before = {
        "cache_sha256": sha256(cache_path),
        "probe_summary_sha256": sha256(run_root / "probe_summary.json"),
        "audit_sha256": sha256(run_root / "audit.json"),
        "registry_sha256": sha256(registry_path),
    }
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        key = (cell["parent_id"], cell["candidate_id"])
        if key not in unique:
            unique[key] = dict(cell, conceptual_arms=[])
        unique[key]["conceptual_arms"].append(cell["arm"])
        if unique[key]["target_member"] != cell["target_member"]:
            raise ValueError("shared candidate target mismatch")
    if len(unique) != 3:
        raise ValueError("five conceptual cells must deduplicate to three transitions")

    transition_rows: list[dict[str, Any]] = []
    unique_rows: list[dict[str, Any]] = []
    api_calls = 0
    for (parent_id, candidate_id), item in sorted(unique.items()):
        case = cases[parent_id]
        target = int(item["target_member"])
        system = probe_system(case, target=target, out_dir=scratch / parent_id, cache_path="")
        if system.llm.calls:
            raise AssertionError("audit system unexpectedly contains API calls")
        validation_path = Path(system.cfg.data.val_path)
        if not validation_path.is_absolute():
            validation_path = ROOT / validation_path
        probe = system.build_probe(load_validation_rows(validation_path))
        expected_questions = {row.question_hash for row in probe.examples}
        parent_maps = [
            cached_answers(cache_path, system.prompt_hash(prompt), expected_questions, system)
            for prompt in case["parent_prompts"]
        ]
        candidate_map = cached_answers(cache_path, candidate_id, expected_questions, system)
        transition_id = canonical_hash({"parent_id": parent_id, "candidate_id": candidate_id})
        rows: list[dict[str, Any]] = []
        parent_vote = parent_oracle = candidate_vote = candidate_oracle = 0
        parent_member = candidate_member = 0
        for example in probe.examples:
            question_hash = example.question_hash
            parent_observations = [mapping[question_hash] for mapping in parent_maps]
            candidate_observations = list(parent_observations)
            candidate_observations[target] = candidate_map[question_hash]
            before = team_state(system, example, parent_observations)
            after = team_state(system, example, candidate_observations)
            parent_vote += int(before.vote_correct)
            candidate_vote += int(after.vote_correct)
            parent_oracle += int(before.gold_vote_count > 0)
            candidate_oracle += int(after.gold_vote_count > 0)
            parent_member += int(before.team_correctness[target])
            candidate_member += int(after.team_correctness[target])
            row = transition_row(
                transition_id=transition_id,
                parent_id=parent_id,
                target_member=target,
                candidate_id=candidate_id,
                conceptual_arms="|".join(sorted(item["conceptual_arms"])),
                question_hash=question_hash,
                before=before,
                after=after,
            )
            if any((
                row["target_gain"], row["target_loss"], row["oracle_gain"], row["oracle_loss"],
                row["vote_gain"], row["vote_loss"], row["delta_G"], row["delta_H"], row["delta_M"],
            )):
                rows.append(row)
        metrics = summarize_rows(rows)
        expected = item["validation_expected"]
        reconstructed = {
            "target_delta": candidate_member - parent_member,
            "vote_delta": candidate_vote - parent_vote,
            "oracle_delta": candidate_oracle - parent_oracle,
            "vote_gain_count": metrics["vote_gain_count"],
            "vote_loss_count": metrics["vote_loss_count"],
            "oracle_gain_count": metrics["oracle_gain_count"],
            "oracle_loss_count": metrics["oracle_loss_count"],
        }
        if reconstructed != expected:
            raise AssertionError(f"validation reconstruction mismatch: {parent_id}")
        if (
            parent_vote != int(item["parent_validation"]["vote"])
            or parent_oracle != int(item["parent_validation"]["oracle"])
        ):
            raise AssertionError("parent validation summary mismatch")
        if int(item["train"]["target_delta"]) <= 0 or int(item["train"]["vote_delta"]) != 0:
            raise AssertionError("unexpected train-side Common-Safe acceptance path")
        unique_rows.append({
            "transition_id": transition_id,
            "parent_id": parent_id,
            "target_member": target,
            "candidate_id": candidate_id,
            "conceptual_arms": "|".join(sorted(item["conceptual_arms"])),
            "conceptual_cell_count": len(item["conceptual_arms"]),
            "candidate_stage": item["candidate_stage"],
            "train_common_safe_outcome": "passed_target_strict_vote_nonregression",
            "train_target_delta": int(item["train"]["target_delta"]),
            "train_vote_delta": int(item["train"]["vote_delta"]),
            "train_vote_gain_count": int(item["train"]["vote_gain_count"]),
            "train_vote_loss_count": int(item["train"]["vote_loss_count"]),
            "train_oracle_delta": int(item["train"]["oracle_delta"]),
            "validation_target_delta": reconstructed["target_delta"],
            "validation_vote_delta": reconstructed["vote_delta"],
            "validation_oracle_delta": reconstructed["oracle_delta"],
            **metrics,
        })
        transition_rows.extend(rows)
        api_calls += len(system.llm.calls)

    conceptual_rows: list[dict[str, Any]] = []
    unique_by_key = {(row["parent_id"], row["candidate_id"]): row for row in unique_rows}
    for cell in sorted(cells, key=lambda row: (row["parent_id"], row["arm"])):
        source = unique_by_key[(cell["parent_id"], cell["candidate_id"])]
        conceptual_rows.append({
            "parent_id": cell["parent_id"],
            "arm": cell["arm"],
            "branch_type": cell["branch_type"],
            "target_member": cell["target_member"],
            "candidate_id": cell["candidate_id"],
            "transition_id": source["transition_id"],
            "train_common_safe_outcome": source["train_common_safe_outcome"],
            "train_target_delta": source["train_target_delta"],
            "train_vote_delta": source["train_vote_delta"],
            "validation_target_delta": source["validation_target_delta"],
            "validation_vote_delta": source["validation_vote_delta"],
            "validation_oracle_delta": source["validation_oracle_delta"],
            "validation_oracle_gain_count": source["oracle_gain_count"],
            "validation_oracle_loss_count": source["oracle_loss_count"],
            "validation_vote_gain_count": source["vote_gain_count"],
            "validation_vote_loss_count": source["vote_loss_count"],
        })

    evidence_after = {
        "cache_sha256": sha256(cache_path),
        "probe_summary_sha256": sha256(run_root / "probe_summary.json"),
        "audit_sha256": sha256(run_root / "audit.json"),
        "registry_sha256": sha256(registry_path),
    }
    if evidence_before != evidence_after:
        raise AssertionError("source evidence changed during read-only audit")
    aggregate = summarize_rows(transition_rows)
    oracle_gain_rows = [row for row in transition_rows if row["oracle_gain"]]
    all_g0_to_g1 = bool(oracle_gain_rows) and all(
        row["G_before"] == 0 and row["G_after"] == 1 for row in oracle_gain_rows
    )
    all_vote_neutral = all(not row["vote_gain"] and not row["vote_loss"] for row in transition_rows)
    all_oracle_gains_nonwinning = bool(oracle_gain_rows) and all(
        not row["vote_correct_after"] for row in oracle_gain_rows
    )
    if all_g0_to_g1 and all_vote_neutral and all_oracle_gains_nonwinning:
        diagnosis = "SINGLETON_COVERAGE_RECOVERY_WITHOUT_VOTE_CONVERSION"
    elif aggregate["vote_gain_count"] and aggregate["vote_loss_count"]:
        diagnosis = "VOTE_GAIN_LOSS_CANCELLATION"
    elif aggregate["oracle_gain_count"]:
        diagnosis = "MIXED_COVERAGE_CONVERSION_STRUCTURE"
    else:
        diagnosis = "NO_VALIDATION_COVERAGE_RECOVERY"
    summary = {
        "audit_version": AUDIT_VERSION,
        "status": "PASS",
        "source_run_execution_commit": EXPECTED_EXECUTION_COMMIT,
        "auditor_commit": auditor_commit,
        "audit_mode": "zero_api_immutable_artifact_reconstruction",
        "api_calls": api_calls,
        "new_test_calls": 0,
        "actual_prompt_commits": 0,
        "trajectory_mutations": 0,
        "conceptual_would_commit_count": len(conceptual_rows),
        "hybrid_conceptual_would_commit_count": sum(row["arm"] == HYBRID for row in conceptual_rows),
        "rr_conceptual_would_commit_count": sum(row["arm"] == RR for row in conceptual_rows),
        "deduplicated_transition_count": len(unique_rows),
        "validation_row_count_per_transition": 50,
        "aggregate_unique_transition_metrics": aggregate,
        "all_oracle_gains_are_G0_to_G1": all_g0_to_g1,
        "all_oracle_gains_remain_nonwinning": all_oracle_gains_nonwinning,
        "simultaneous_vote_gain_and_loss": bool(
            aggregate["vote_gain_count"] and aggregate["vote_loss_count"]
        ),
        "diagnosis": diagnosis,
        "common_safe_explanation": {
            "all_unique_updates_train_target_strictly_improved": all(
                row["train_target_delta"] > 0 for row in unique_rows
            ),
            "all_unique_updates_train_vote_nonregressed": all(
                row["train_vote_delta"] >= 0 for row in unique_rows
            ),
            "train_target_deltas": [row["train_target_delta"] for row in unique_rows],
            "train_vote_deltas": [row["train_vote_delta"] for row in unique_rows],
            "validation_target_deltas": [row["validation_target_delta"] for row in unique_rows],
            "validation_vote_deltas": [row["validation_vote_delta"] for row in unique_rows],
            "validation_oracle_deltas": [row["validation_oracle_delta"] for row in unique_rows],
        },
        "source_evidence_unchanged": True,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", summary)
    write_csv(out / "conceptual_update_level.csv", conceptual_rows)
    write_csv(out / "unique_update_level.csv", unique_rows)
    write_csv(out / "transition_level.csv", transition_rows)
    write_json(out / "source_evidence.json", {
        "source_run_execution_commit": EXPECTED_EXECUTION_COMMIT,
        "auditor_commit": auditor_commit,
        "source_artifact_sha256": evidence_after,
        "source_artifacts_modified": False,
        "cache_access_mode": "sqlite_immutable_read_only",
    })
    files = {
        path.name: sha256(path)
        for path in sorted(out.iterdir()) if path.name != "sha256_manifest.json"
    }
    write_json(out / "sha256_manifest.json", {
        "manifest_version": "v17_hybrid_recovered_conversion_manifest_v1",
        "files": files,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
