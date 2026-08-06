from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.cli import _load, _member_gain_summary, build_dataset
from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import SCHEMA_VERSION
from multi_dataset_diverse_rl.evaluation.prompt_question import (
    PromptAnswer,
    PromptQuestionEvaluator,
)
from multi_dataset_diverse_rl.evaluation.validation import dataset_metrics_from_dict
from multi_dataset_diverse_rl.evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION
from multi_dataset_diverse_rl.peer_state import build_team_vote_state
from multi_dataset_diverse_rl.persistence.identity import (
    PROMPT_QUESTION_EVALUATOR_VERSION,
    build_run_identity,
    solver_request_components,
    solver_request_identity,
    validate_run_identity,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.task_manifest import load_task_manifest, resolve_task_ids
from multi_dataset_diverse_rl.tasks import get_task_spec
from multi_dataset_diverse_rl.utils import load_jsonl
from multi_dataset_diverse_rl.versions import METHOD_VERSION
from scripts.experiment_config import select_settings


FROZEN_INITIALIZATION_MANIFEST_VERSION = "final_method_frozen_initialization_v1"
COMPARISON_CACHE_MANIFEST_VERSION = "matched_task_seed_observation_cache_v2"

CACHE_COLUMNS = (
    "cache_key", "schema_version", "state", "owner_id", "updated_at",
    "created_at", "model_request_identity", "solver_model",
    "endpoint_identity", "output_contract_version", "parser_version",
    "temperature", "max_tokens", "evaluation_replica_seed", "prompt_hash",
    "question_hash", "answer_json",
)
CACHE_CONTENT_COLUMNS = (
    "cache_key", "schema_version", "model_request_identity", "solver_model",
    "endpoint_identity", "output_contract_version", "parser_version",
    "temperature", "max_tokens", "evaluation_replica_seed", "prompt_hash",
    "question_hash", "answer_json",
)


RUNNER_OWNED_FIELDS = {
    "task_type", "dataset_format", "comparison_task_id", "benchmark", "answer_format",
    "train_path", "val_path", "test_path", "manifest_sha256", "out_dir", "seed",
    "method_version", "experiment_setting",
}
RUNNER_FIELDS = tuple(
    name for name in Config().to_flat_dict() if name not in RUNNER_OWNED_FIELDS
)


def effective_proposal_memory_mode(
    setting_name: str,
    requested_mode: str,
) -> str:
    """Keep the reporting-only baseline outside the proposal mechanism.

    A matched memory run still needs its own baseline in the same output root.
    The baseline has no responsibility-conditioned TCS and therefore cannot
    instantiate state-local proposal memory.  Only the optimized Full member
    run receives the requested treatment; this is intentional, visible in its
    own run metadata, and does not alter the baseline prompt team.
    """
    if setting_name == "shared_static_reference":
        return "off"
    return str(requested_mode)


def _task_split_protocol(task) -> dict[str, Any]:
    paths = {str(task.train_path), str(task.val_path), str(task.test_path)}
    if len(paths) < 3:
        return {"split_protocol": "reused_file", "leakage_warning": True}
    return {"split_protocol": "task_manifest_split", "leakage_warning": False}


def _task_split_integrity(task, dataset_format: str, workspace: str) -> dict[str, Any]:
    def resolve(path: str) -> Path:
        value = Path(path)
        return value if value.is_absolute() else Path(workspace) / value

    paths = {"opt": resolve(task.train_path), "val": resolve(task.val_path), "test": resolve(task.test_path)}
    rows = {
        name: __import__("multi_dataset_diverse_rl.cli", fromlist=["build_dataset"]).build_dataset(
            __import__("multi_dataset_diverse_rl.utils", fromlist=["load_jsonl"]).load_jsonl(str(path), -1),
            dataset_format,
        )
        for name, path in paths.items()
    }

    def question_hash(value: Any) -> str:
        return hashlib.sha256(" ".join(str(value or "").split()).lower().encode("utf-8")).hexdigest()

    hashes = {name: {question_hash(row["question"]) for row in values} for name, values in rows.items()}
    overlaps = {
        "opt_val_question_overlap": len(hashes["opt"] & hashes["val"]),
        "opt_test_question_overlap": len(hashes["opt"] & hashes["test"]),
        "val_test_question_overlap": len(hashes["val"] & hashes["test"]),
    }
    protocol = _task_split_protocol(task)
    if protocol["split_protocol"] == "task_manifest_split" and any(overlaps.values()):
        raise ValueError(
            f"Strict split overlap for task={task.task_id}: "
            f"opt_val={overlaps['opt_val_question_overlap']} "
            f"opt_test={overlaps['opt_test_question_overlap']} "
            f"val_test={overlaps['val_test_question_overlap']}"
        )
    answer_space = {
        "option_letter_invalid_gold_count": 0,
        "option_letter_ambiguous_gold_count": 0,
        "option_letter_empty_label_set_count": 0,
    }
    if task.answer_format == "option_letter":
        matcher = get_task_spec(task.task_type).match_answer
        for values in rows.values():
            for row in values:
                labels = {
                    match.group(1).upper()
                    for match in re.finditer(
                        r"(?:^|\n)\s*\(([A-Z])\)\s+",
                        row["question"],
                        flags=re.MULTILINE,
                    )
                }
                if not labels:
                    labels = {
                        match.group(1).upper()
                        for match in re.finditer(
                            r"(?:^|\n)\s*([A-Z])[\).]\s+",
                            row["question"],
                            flags=re.MULTILINE,
                        )
                    }
                matches = [
                    label for label in labels
                    if matcher(label, row["answer"])
                ]
                answer_space["option_letter_empty_label_set_count"] += int(not labels)
                answer_space["option_letter_invalid_gold_count"] += int(
                    bool(labels) and not matches
                )
                answer_space["option_letter_ambiguous_gold_count"] += int(
                    bool(labels) and len(matches) > 1
                )
        if (
            answer_space["option_letter_invalid_gold_count"]
            or answer_space["option_letter_ambiguous_gold_count"]
        ):
            raise ValueError(
                f"Option-letter answer-space integrity failed for task={task.task_id}: "
                + json.dumps(answer_space, sort_keys=True)
            )
    return {
        **protocol, **overlaps,
        **answer_space,
        **{f"{name}_count": len(rows[name]) for name in paths},
        **{f"{name}_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run task-level Member-Aware Prompt-Team experiments."
    )
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--settings", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--dataset_format", default="mars")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--resume_completed", type=int, choices=[0, 1], default=0)
    parser.add_argument("--optimized_only", type=int, choices=[0, 1], default=0)
    defaults = Config().to_flat_dict()
    for name in RUNNER_FIELDS:
        default = defaults[name]
        arg_type = int if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{name}", type=arg_type, default=None)
    return parser


def _validate_setting_sequence(
    setting_names: Sequence[str],
    *,
    optimized_only: bool,
) -> None:
    """Validate whether a run has a matched baseline test reference.

    ``optimized_only`` is an explicit pilot/reporting mode. It still creates
    the frozen optimization-probe initialization, but it does not run or
    synthesize a reporting-only baseline test evaluation.
    """
    if optimized_only:
        if len(setting_names) != 1 or setting_names[0] == "shared_static_reference":
            raise ValueError(
                "optimized_only requires exactly one non-baseline setting"
            )
        return
    if any(name != "shared_static_reference" for name in setting_names):
        if not setting_names or setting_names[0] != "shared_static_reference":
            raise ValueError(
                "optimized comparisons must run shared_static_reference first "
                "to provide the single initial-test reference"
            )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source)) as read_connection:
        with sqlite3.connect(str(destination)) as write_connection:
            read_connection.backup(write_connection)


def _sqlite_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with sqlite3.connect(str(path)) as connection:
        for statement in connection.iterdump():
            digest.update(statement.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _canonical_json(value: str) -> tuple[str, dict[str, Any]]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("cached solver answer must be a JSON object")
    PromptAnswer(**payload)
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        payload,
    )


def _cache_key_evaluator(cfg: Config) -> PromptQuestionEvaluator:
    return PromptQuestionEvaluator(
        model_request_identity=solver_request_identity(cfg),
        parser_version=cfg.peer_state.parser_version,
        temperature=cfg.models.temperature,
        decoding_seed=cfg.training.seed,
        cache_metadata=solver_request_components(cfg),
        version=PROMPT_QUESTION_EVALUATOR_VERSION,
    )


def _solver_cache_snapshot(
    path: Path,
    *,
    expected_evaluator: PromptQuestionEvaluator | None = None,
) -> dict[str, Any]:
    """Return a content-addressed, secret-free view of a solver cache."""
    with sqlite3.connect(str(path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT {', '.join(CACHE_COLUMNS)} FROM solver_cache ORDER BY cache_key"
        ).fetchall()
    entries: dict[str, dict[str, Any]] = {}
    invalid_entries: list[dict[str, str]] = []
    non_ready_count = 0
    for raw in rows:
        values = {name: raw[name] for name in CACHE_COLUMNS}
        cache_key = str(values["cache_key"])
        if str(values["state"]) != "ready":
            non_ready_count += 1
            continue
        reasons: list[str] = []
        if str(values["schema_version"]) != SCHEMA_VERSION:
            reasons.append("schema_version")
        answer_text = values.get("answer_json")
        payload: dict[str, Any] = {}
        canonical_answer = ""
        if not isinstance(answer_text, str) or not answer_text:
            reasons.append("missing_answer_json")
        else:
            try:
                canonical_answer, payload = _canonical_json(answer_text)
            except (TypeError, ValueError, json.JSONDecodeError):
                reasons.append("invalid_answer_json")
        values["answer_json"] = canonical_answer
        if expected_evaluator is not None:
            expected_key = expected_evaluator.key(
                str(values["prompt_hash"]), str(values["question_hash"])
            )
            if cache_key != expected_key:
                reasons.append("cache_key")
            expected_metadata = {
                **expected_evaluator.cache_metadata,
                "model_request_identity": expected_evaluator.model_request_identity,
                "parser_version": expected_evaluator.parser_version,
                "temperature": expected_evaluator.temperature,
                "evaluation_replica_seed": expected_evaluator.decoding_seed,
            }
            for name in (
                "model_request_identity", "solver_model", "endpoint_identity",
                "output_contract_version", "parser_version", "temperature",
                "max_tokens", "evaluation_replica_seed",
            ):
                expected = expected_metadata[name]
                actual = values[name]
                if name == "temperature":
                    matched = float(actual) == float(expected)
                elif name in {"max_tokens", "evaluation_replica_seed"}:
                    matched = int(actual) == int(expected)
                else:
                    matched = str(actual) == str(expected)
                if not matched:
                    reasons.append(name)
        if reasons:
            invalid_entries.append({
                "cache_key": cache_key,
                "reason": ",".join(sorted(set(reasons))),
            })
            continue
        observation_hash = hashlib.sha256(canonical_answer.encode("utf-8")).hexdigest()
        content_payload = [values[name] for name in CACHE_CONTENT_COLUMNS]
        content_hash = hashlib.sha256(
            json.dumps(
                content_payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        entries[cache_key] = {
            "values": values,
            "content_hash": content_hash,
            "observation_hash": observation_hash,
            "parsed_answer_hash": hashlib.sha256(
                str(payload.get("answer", "")).encode("utf-8")
            ).hexdigest(),
            "response_hash": str(payload.get("response_hash", "")),
            "valid": bool(payload.get("valid", False)),
            "terminal_invalid": bool(payload.get("terminal_invalid", False)),
            "answer_payload": payload,
        }
    content_hash = hashlib.sha256(
        json.dumps(
            [(key, entries[key]["content_hash"]) for key in sorted(entries)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "entries": entries,
        "entry_count": len(entries),
        "content_hash": content_hash,
        "non_ready_entry_count": non_ready_count,
        "invalid_entries": invalid_entries,
    }


def _comparison_cache_source(
    *,
    comparison_reference_cache_path: Path,
) -> tuple[Path, str]:
    if not comparison_reference_cache_path.is_file():
        raise FileNotFoundError(
            "The cumulative task-seed comparison cache must exist before a setting "
            f"runs: {comparison_reference_cache_path}"
        )
    return comparison_reference_cache_path, "cumulative_task_seed_observation_reference"


def _merge_ready_solver_cache(
    source: Path,
    destination: Path,
    *,
    expected_evaluator: PromptQuestionEvaluator | None = None,
) -> dict[str, Any]:
    """Merge completed exact observations without ever overwriting the first value."""
    source_snapshot = _solver_cache_snapshot(
        source, expected_evaluator=expected_evaluator,
    )
    destination_before = _solver_cache_snapshot(
        destination, expected_evaluator=expected_evaluator,
    )
    source_entries = source_snapshot["entries"]
    destination_entries = destination_before["entries"]
    missing_reference_keys = sorted(set(destination_entries) - set(source_entries))
    duplicate_keys = sorted(set(source_entries) & set(destination_entries))
    conflicts = [
        {
            "cache_key": key,
            "reference_observation_hash": destination_entries[key]["observation_hash"],
            "local_observation_hash": source_entries[key]["observation_hash"],
        }
        for key in duplicate_keys
        if source_entries[key]["content_hash"] != destination_entries[key]["content_hash"]
    ]
    new_keys = sorted(set(source_entries) - set(destination_entries))
    precondition_failed = bool(
        source_snapshot["non_ready_entry_count"]
        or destination_before["non_ready_entry_count"]
        or source_snapshot["invalid_entries"]
        or destination_before["invalid_entries"]
        or missing_reference_keys
        or conflicts
    )
    if not precondition_failed and new_keys:
        placeholders = ", ".join("?" for _ in CACHE_COLUMNS)
        rows = [
            tuple(source_entries[key]["values"][name] for name in CACHE_COLUMNS)
            for key in new_keys
        ]
        with sqlite3.connect(str(destination)) as destination_connection:
            destination_connection.executemany(
                f"INSERT INTO solver_cache ({', '.join(CACHE_COLUMNS)}) "
                f"VALUES ({placeholders})",
                rows,
            )
    destination_after = _solver_cache_snapshot(
        destination, expected_evaluator=expected_evaluator,
    )
    passed = (
        not precondition_failed
        and destination_after["entry_count"]
        == destination_before["entry_count"] + len(new_keys)
    )
    return {
        "gate": "PASS" if passed else "FAIL",
        "source_ready_entry_count": source_snapshot["entry_count"],
        "reference_entry_count_before": destination_before["entry_count"],
        "reference_entry_count_after": destination_after["entry_count"],
        "new_entries_merged": len(new_keys) if passed else 0,
        "duplicate_same_observation_count": len(duplicate_keys) - len(conflicts),
        "exact_request_conflict_count": len(conflicts),
        "conflicts": conflicts,
        "missing_reference_count": len(missing_reference_keys),
        "missing_reference_keys": missing_reference_keys,
        "source_non_ready_entry_count": source_snapshot["non_ready_entry_count"],
        "reference_non_ready_entry_count": destination_before["non_ready_entry_count"],
        "source_invalid_entry_count": len(source_snapshot["invalid_entries"]),
        "reference_invalid_entry_count": len(destination_before["invalid_entries"]),
        "source_invalid_entries": source_snapshot["invalid_entries"],
        "reference_invalid_entries": destination_before["invalid_entries"],
        "parent_reference_hash": destination_before["content_hash"],
        "local_cache_hash_after_run": source_snapshot["content_hash"],
        "result_reference_hash": destination_after["content_hash"],
    }


def _test_observation_audit(
    *,
    snapshot: Mapping[str, Any],
    prompt_hashes: Sequence[str],
    test_rows: Sequence[Mapping[str, Any]],
    task_type: str,
    tie_break: str,
    seed: int,
) -> dict[str, Any]:
    """Build a sanitized per-question observation and team-vote fingerprint."""
    entries = snapshot["entries"]
    by_prompt_question = {
        (str(row["values"]["prompt_hash"]), str(row["values"]["question_hash"])): row
        for row in entries.values()
    }
    task_spec = get_task_spec(task_type)
    per_member: list[dict[str, dict[str, Any]]] = [dict() for _ in prompt_hashes]
    team_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    correct_counts = [0 for _ in prompt_hashes]
    normalize = lambda value: task_spec.extract_pred(f"FINAL_ANSWER: {value}", None)
    for test_row in test_rows:
        question = str(test_row["question"])
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        gold = task_spec.parse_gold(test_row["answer"], question)
        answers: list[str] = []
        valid_vector: list[bool] = []
        observation_hashes: list[str] = []
        terminal_vector: list[bool] = []
        question_missing = False
        for agent_id, prompt_hash in enumerate(prompt_hashes):
            entry = by_prompt_question.get((str(prompt_hash), question_hash))
            if entry is None:
                missing.append({"agent_id": agent_id, "question_hash": question_hash})
                question_missing = True
                continue
            payload = entry["answer_payload"]
            answer = str(payload.get("answer", ""))
            valid = bool(payload.get("valid", False))
            terminal_invalid = bool(payload.get("terminal_invalid", False))
            correct = bool(valid and task_spec.match_answer(answer, gold))
            correct_counts[agent_id] += int(correct)
            record = {
                "cache_key": str(entry["values"]["cache_key"]),
                "observation_hash": str(entry["observation_hash"]),
                "parsed_answer_hash": str(entry["parsed_answer_hash"]),
                "response_hash": str(entry["response_hash"]),
                "valid": valid,
                "terminal_invalid": terminal_invalid,
                "correct": correct,
            }
            per_member[agent_id][question_hash] = record
            answers.append(answer)
            valid_vector.append(valid)
            observation_hashes.append(str(entry["observation_hash"]))
            terminal_vector.append(terminal_invalid)
        if question_missing:
            continue
        state = build_team_vote_state(
            question_hash=question_hash,
            gold_answer=gold,
            answers=answers,
            valid_vector=valid_vector,
            normalize_answer=normalize,
            match_answer=task_spec.match_answer,
            tie_break=tie_break,
            seed=seed,
        )
        team_rows.append({
            "question_hash": question_hash,
            "member_observation_hashes": observation_hashes,
            "member_terminal_invalid": terminal_vector,
            "vote_answer_hash": hashlib.sha256(
                state.vote_answer.encode("utf-8")
            ).hexdigest(),
            "vote_correct": state.vote_correct,
            "top_tie": state.top_tie,
        })
    team_rows.sort(key=lambda row: row["question_hash"])
    vector_text = json.dumps(
        team_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "prompt_hashes": list(prompt_hashes),
        "test_question_count": len(test_rows),
        "test_question_set_hash": hashlib.sha256(
            json.dumps(
                sorted(
                    hashlib.sha256(str(row["question"]).encode("utf-8")).hexdigest()
                    for row in test_rows
                ),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "per_member": per_member,
        "per_agent_correct_counts": correct_counts,
        "team_vote_vector": team_rows,
        "team_vote_vector_hash": hashlib.sha256(vector_text.encode("utf-8")).hexdigest(),
        "team_vote_correct_count": sum(bool(row["vote_correct"]) for row in team_rows),
        "missing_entry_count": len(missing),
        "missing_entries": missing,
    }


def _compare_unchanged_test_observations(
    prior: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    prior_setting: str,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    aggregate_mismatches: list[int] = []
    unchanged_ids: list[int] = []
    for agent_id, (prior_hash, current_hash) in enumerate(zip(
        prior["prompt_hashes"], current["prompt_hashes"], strict=True,
    )):
        if prior_hash != current_hash:
            continue
        unchanged_ids.append(agent_id)
        left = prior["per_member"][agent_id]
        right = current["per_member"][agent_id]
        question_hashes = sorted(set(left) | set(right))
        drift_questions = [
            question_hash
            for question_hash in question_hashes
            if left.get(question_hash) != right.get(question_hash)
        ]
        if drift_questions:
            mismatches.append({
                "agent_id": agent_id,
                "drift_count": len(drift_questions),
                "question_hashes": drift_questions,
            })
        if (
            prior["per_agent_correct_counts"][agent_id]
            != current["per_agent_correct_counts"][agent_id]
        ):
            aggregate_mismatches.append(agent_id)
    exact_team = len(unchanged_ids) == len(current["prompt_hashes"])
    team_mismatch = bool(
        exact_team
        and (
            prior["team_vote_vector_hash"] != current["team_vote_vector_hash"]
            or prior["team_vote_correct_count"] != current["team_vote_correct_count"]
        )
    )
    return {
        "prior_setting": prior_setting,
        "unchanged_member_ids": unchanged_ids,
        "per_question_drift_count": sum(row["drift_count"] for row in mismatches),
        "per_question_drifts": mismatches,
        "aggregate_correct_count_drift_member_ids": aggregate_mismatches,
        "exact_unchanged_team": exact_team,
        "team_vote_vector_drift": team_mismatch,
        "passed": not mismatches and not aggregate_mismatches and not team_mismatch,
    }


def _not_applicable_test_audit(
    prompt_hashes: Sequence[str],
    *,
    test_evaluation_count: int,
) -> dict[str, Any]:
    """Describe a no-test run without fabricating zero-valued observations."""
    if test_evaluation_count != 0:
        raise ValueError(
            "A no-test comparison manifest requires test_evaluation_count=0"
        )
    return {
        "prompt_hashes": list(prompt_hashes),
        "final_test_enabled": False,
        "final_test_evaluated": False,
        "final_test_evaluation_count": 0,
        "test_observation_status": "not_applicable",
        "test_member_count_status": "not_applicable",
        "test_drift_status": "not_applicable",
    }


def _test_observation_comparisons(
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    final_test_enabled: bool,
) -> list[dict[str, Any]]:
    """Compare test observations only when the run actually evaluated test."""
    if not final_test_enabled:
        return []
    return [
        _compare_unchanged_test_observations(
            prior["test_audit"],
            current,
            prior_setting=prior["setting"],
        )
        for prior in history
    ]


async def _freeze_initialization(
    cfg: Config,
    *,
    workspace: Path,
) -> dict[str, Any]:
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    test = _load(cfg.data.test_path, cfg.data.test_size, cfg.data.dataset_format)
    system = PromptEnsembleOptimizationSystem(cfg)
    system.set_run_identity(build_run_identity(
        cfg,
        train_rows=train,
        val_rows=validation,
        test_rows=test,
        workspace=workspace,
    ))
    probe = train[: min(len(train), cfg.evaluation.candidate_eval_pool_size)]
    await system.initialize_fixed_probe(probe)
    return system.frozen_initialization_snapshot()


def _ensure_frozen_initialization(
    cfg: Config,
    *,
    workspace: Path,
    frozen_root: Path,
) -> tuple[Path, Path]:
    raw_cache = frozen_root / "initial_solver_cache.sqlite"
    stable_cache = frozen_root / "initial_solver_cache_frozen.sqlite"
    manifest_path = frozen_root / "frozen_initialization_manifest.json"
    existing = (raw_cache.exists(), stable_cache.exists(), manifest_path.exists())
    if all(existing):
        manifest = _read_json(manifest_path)
        if manifest.get("manifest_version") != FROZEN_INITIALIZATION_MANIFEST_VERSION:
            raise ValueError(f"Frozen initialization manifest version mismatch: {manifest_path}")
        if manifest.get("task_id") != cfg.data.comparison_task_id:
            raise ValueError(f"Frozen initialization task mismatch: {manifest_path}")
        if int(manifest.get("seed", -1)) != cfg.training.seed:
            raise ValueError(f"Frozen initialization seed mismatch: {manifest_path}")
        return manifest_path, stable_cache
    if any(existing):
        raise RuntimeError(f"Partial frozen initialization must not be reused: {frozen_root}")

    frozen_root.mkdir(parents=True, exist_ok=False)
    freeze_values = cfg.to_flat_dict()
    freeze_values.update({
        "experiment_setting": "shared_static_reference",
        "out_dir": str(frozen_root),
        "shared_solver_cache_path": str(raw_cache),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "frozen_initialization_manifest_path": "",
    })
    freeze_cfg = Config.from_flat(**freeze_values)
    snapshot = asyncio.run(_freeze_initialization(freeze_cfg, workspace=workspace))
    _sqlite_backup(raw_cache, stable_cache)
    manifest = {
        "manifest_version": FROZEN_INITIALIZATION_MANIFEST_VERSION,
        "method_version": METHOD_VERSION,
        "task_id": cfg.data.comparison_task_id,
        "seed": cfg.training.seed,
        "initialization_snapshot": snapshot,
        "initial_cache_sha256": hashlib.sha256(stable_cache.read_bytes()).hexdigest(),
        "initial_cache_ready": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path, stable_cache


def _completed_run(run_dir: Path, expected_identity) -> bool:
    required = [
        "final_summary.json",
        "history.json",
        "best_prompts.json",
        "run_meta.json",
        "tcs_rounds.jsonl",
        "candidate_funnel.json",
        "solver_invalid_outputs.jsonl",
        "student_recovery_observations.jsonl",
        "cost_summary.json",
        "frozen_initialization_match.json",
        "comparison_cache_match.json",
        "repairability_adjusted_target_scores.jsonl",
        "dual_target_branch_decisions.jsonl",
        "dual_target_commit_decisions.jsonl",
        "repairability_failure_events.jsonl",
        "repairability_reset_events.jsonl",
    ]
    try:
        metadata = _read_json(run_dir / "run_meta.json")
        summary = _read_json(run_dir / "final_summary.json")
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("config", {}).get("proposal_memory_mode", "off") == "state_local_v1":
        required.extend((
            "proposal_memory_events_sanitized.jsonl",
            "proposal_memory_summary.json",
            "proposal_memory_key_isolation_audit.json",
            "proposal_rotation_trajectory.jsonl",
        ))
    if not all((run_dir / filename).exists() for filename in required):
        return False
    if metadata["method_version"] != METHOD_VERSION:
        raise ValueError(f"Completed run has an incompatible method version: {run_dir}")
    if metadata["legacy_compatibility_enabled"] is not False:
        raise ValueError(f"Completed run enabled legacy compatibility: {run_dir}")
    if metadata.get("solver_output_contract_version") != SOLVER_OUTPUT_CONTRACT_VERSION:
        raise ValueError(f"Completed run has an incompatible solver output contract: {run_dir}")
    if not metadata.get("shared_solver_cache_path"):
        raise ValueError(f"Completed run has no persistent shared solver cache: {run_dir}")
    validate_run_identity(expected_identity, metadata["run_identity"])
    if not {"initial_test", "selected_test", "member_gain", "selection_summary"} <= set(summary):
        raise ValueError(f"Completed run has an incompatible final summary: {run_dir}")
    return True


def main() -> None:
    args = _parser().parse_args()
    workspace = args.workspace.resolve()
    tasks = load_task_manifest(str((workspace / args.manifest).resolve()))
    manifest_sha256 = hashlib.sha256((workspace / args.manifest).resolve().read_bytes()).hexdigest()
    task_ids = resolve_task_ids(args.tasks, tasks, args.benchmarks)
    settings = select_settings(
        args.settings,
        allow_legacy_setting=bool(args.allow_legacy_setting),
        allow_auxiliary_setting=bool(args.allow_auxiliary_setting),
    )
    setting_names = [setting.name for setting in settings]
    _validate_setting_sequence(
        setting_names,
        optimized_only=bool(args.optimized_only),
    )
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    root = (workspace / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_test_by_task_seed: dict[tuple[str, int], dict[str, Any]] = {}
    mutable_cache_paths: set[str] = set()
    frozen_manifests: dict[tuple[str, int], dict[str, Any]] = {}
    comparison_history: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def resolved_values(
        *,
        task,
        setting,
        seed: int,
        run_dir: Path,
        cache_path: Path,
        frozen_manifest_path: Path | None,
    ) -> dict[str, Any]:
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
            "manifest_sha256": manifest_sha256,
            "out_dir": str(run_dir),
            "shared_solver_cache_path": str(cache_path),
            "frozen_initialization_manifest_path": (
                str(frozen_manifest_path) if frozen_manifest_path is not None else ""
            ),
            "seed": seed,
        }
        defaults = Config().to_flat_dict()
        for name in RUNNER_FIELDS:
            value = getattr(args, name)
            if value is not None:
                values[name] = bool(value) if isinstance(defaults[name], bool) else value
        values["proposal_memory_mode"] = effective_proposal_memory_mode(
            setting.name,
            str(values.get("proposal_memory_mode", defaults["proposal_memory_mode"])),
        )
        return values

    for task_id in task_ids:
        task = tasks[task_id]
        split_integrity = _task_split_integrity(task, args.dataset_format, str(workspace))
        for seed in seeds:
            baseline_setting = settings[0]
            freeze_root = root / "_frozen_initialization" / task_id / f"seed{seed}"
            freeze_values = resolved_values(
                task=task,
                setting=baseline_setting,
                seed=seed,
                run_dir=freeze_root,
                cache_path=freeze_root / "initial_solver_cache.sqlite",
                frozen_manifest_path=None,
            )
            freeze_cfg = Config.from_flat(**freeze_values)
            frozen_manifest_path, frozen_cache_path = _ensure_frozen_initialization(
                freeze_cfg,
                workspace=workspace,
                frozen_root=freeze_root,
            )
            frozen_manifests[(task_id, seed)] = {
                "task_id": task_id,
                "seed": seed,
                "manifest_sha256": hashlib.sha256(
                    frozen_manifest_path.read_bytes()
                ).hexdigest(),
                "cache_sha256": hashlib.sha256(
                    frozen_cache_path.read_bytes()
                ).hexdigest(),
                "initial_comparison_reference_cache_sha256": None,
            }
            comparison_reference_cache_path = (
                freeze_root / "comparison_reference_solver_cache.sqlite"
            )
            if not comparison_reference_cache_path.exists():
                _sqlite_backup(frozen_cache_path, comparison_reference_cache_path)
            frozen_manifests[(task_id, seed)][
                "initial_comparison_reference_cache_sha256"
            ] = _sqlite_content_sha256(comparison_reference_cache_path)

            for setting in settings:
                run_dir = root / task_id / f"{setting.name}_seed{seed}"
                final_path = run_dir / "final_summary.json"
                mutable_cache_path = run_dir / "_solver_cache.sqlite"
                cache_key = str(mutable_cache_path.resolve()).lower()
                if cache_key in mutable_cache_paths:
                    raise RuntimeError(f"Mutable solver cache path was reused: {mutable_cache_path}")
                mutable_cache_paths.add(cache_key)
                values = resolved_values(
                    task=task,
                    setting=setting,
                    seed=seed,
                    run_dir=run_dir,
                    cache_path=mutable_cache_path,
                    frozen_manifest_path=frozen_manifest_path,
                )
                cfg = Config.from_flat(**values)
                split_rows = {
                    "train": build_dataset(load_jsonl(cfg.data.train_path, cfg.data.train_size), cfg.data.dataset_format),
                    "val": build_dataset(load_jsonl(cfg.data.val_path, cfg.data.val_size), cfg.data.dataset_format),
                    "test": build_dataset(load_jsonl(cfg.data.test_path, cfg.data.test_size), cfg.data.dataset_format),
                }
                expected_identity = build_run_identity(
                    cfg,
                    train_rows=split_rows["train"],
                    val_rows=split_rows["val"],
                    test_rows=split_rows["test"],
                    workspace=workspace,
                )
                cache_source_path, cache_source_role = _comparison_cache_source(
                    comparison_reference_cache_path=comparison_reference_cache_path,
                )
                cache_evaluator = _cache_key_evaluator(cfg)
                if (
                    args.resume_completed
                    and run_dir.is_dir()
                    and _completed_run(run_dir, expected_identity)
                ):
                    metrics = _read_json(final_path)
                else:
                    if run_dir.exists():
                        raise FileExistsError(
                            f"Run output must be new when not reusing an exact completed run: {run_dir}"
                        )
                    run_dir.mkdir(parents=True)
                    reference_before = _solver_cache_snapshot(
                        cache_source_path, expected_evaluator=cache_evaluator,
                    )
                    _sqlite_backup(cache_source_path, mutable_cache_path)
                    starting_snapshot = _solver_cache_snapshot(
                        mutable_cache_path, expected_evaluator=cache_evaluator,
                    )
                    starting_cache_sha256 = starting_snapshot["content_hash"]
                    reference_cache_sha256 = reference_before["content_hash"]
                    initial_match = (
                        starting_cache_sha256 == reference_cache_sha256
                        and starting_snapshot["entry_count"] == reference_before["entry_count"]
                        and not starting_snapshot["invalid_entries"]
                        and not reference_before["invalid_entries"]
                        and not starting_snapshot["non_ready_entry_count"]
                        and not reference_before["non_ready_entry_count"]
                    )
                    (run_dir / "comparison_cache_match.json").write_text(
                        json.dumps({
                            "manifest_version": COMPARISON_CACHE_MANIFEST_VERSION,
                            "task": task_id,
                            "seed": seed,
                            "setting": setting.name,
                            "source_role": cache_source_role,
                            "starting_cache_sha256": starting_cache_sha256,
                            "reference_cache_sha256": reference_cache_sha256,
                            "parent_reference_hash": reference_cache_sha256,
                            "reference_entry_count_before": reference_before["entry_count"],
                            "local_clone_entry_count": starting_snapshot["entry_count"],
                            "reference_cache_path_hash": hashlib.sha256(
                                str(cache_source_path.resolve()).lower().encode("utf-8")
                            ).hexdigest(),
                            "local_cache_path_hash": hashlib.sha256(
                                str(mutable_cache_path.resolve()).lower().encode("utf-8")
                            ).hexdigest(),
                            "cache_chain_continuity": initial_match,
                            "matched": initial_match,
                        }, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    if not initial_match:
                        raise RuntimeError(
                            f"Comparison reference clone mismatch before run: {run_dir}"
                        )
                    cmd = [sys.executable, "-m", "multi_dataset_diverse_rl.cli"]
                    for name, value in cfg.to_flat_dict().items():
                        cmd.extend([f"--{name}", str(int(value) if isinstance(value, bool) else value)])
                    subprocess.run(cmd, cwd=workspace, check=True)
                    metrics = _read_json(final_path)
                run_meta = _read_json(run_dir / "run_meta.json")
                merge_audit = _merge_ready_solver_cache(
                    mutable_cache_path,
                    comparison_reference_cache_path,
                    expected_evaluator=cache_evaluator,
                )
                cache_match_path = run_dir / "comparison_cache_match.json"
                cache_match = _read_json(cache_match_path)
                provider_misses = int(run_meta.get("shared_solver_cache_misses", 0))
                new_local_entries = max(
                    0,
                    int(merge_audit["source_ready_entry_count"])
                    - int(merge_audit["reference_entry_count_before"]),
                )
                unexpected_provider_recall_count = max(
                    0, provider_misses - new_local_entries,
                )
                unaccounted_new_entry_count = max(
                    0, new_local_entries - provider_misses,
                )
                cache_match.update(merge_audit)
                cache_match.update({
                    "local_entry_count_after_run": merge_audit["source_ready_entry_count"],
                    "post_run_reference_cache_sha256": merge_audit["result_reference_hash"],
                    "shared_solver_cache_hit_count": int(
                        run_meta.get("shared_solver_cache_hits", 0)
                    ),
                    "provider_cache_miss_count": provider_misses,
                    "unexpected_provider_recall_count": unexpected_provider_recall_count,
                    "unaccounted_new_entry_count": unaccounted_new_entry_count,
                })
                cache_match["matched"] = bool(
                    cache_match.get("cache_chain_continuity")
                    and merge_audit["gate"] == "PASS"
                    and unexpected_provider_recall_count == 0
                    and unaccounted_new_entry_count == 0
                )
                cache_match["gate"] = "PASS" if cache_match["matched"] else "FAIL"
                cache_match_path.write_text(
                    json.dumps(cache_match, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if cache_match["gate"] != "PASS":
                    raise RuntimeError(f"Comparison cache gate failed: {run_dir}")
                frozen_match = _read_json(run_dir / "frozen_initialization_match.json")
                if frozen_match.get("matched") is not True:
                    raise RuntimeError(f"Frozen initialization mismatch: {run_dir}")

                selected_test = metrics["selected_test"]
                initial_test = metrics.get("initial_test")
                member_gain = metrics.get("member_gain")
                baseline_key = (task_id, seed)
                selection_summary = metrics["selection_summary"]
                if selected_test is None:
                    if cfg.persistence.final_test_enabled:
                        raise RuntimeError(f"Final test was enabled but no test metrics exist: {run_dir}")
                    if int(selection_summary.get("test_evaluation_count", -1)) != 0:
                        raise RuntimeError(f"Test was evaluated in a no-test run: {run_dir}")
                else:
                    if setting.name == "shared_static_reference":
                        baseline_test_by_task_seed[baseline_key] = selected_test
                        initial_test = selected_test
                        member_gain = metrics["member_gain"]
                    elif (
                        (initial_test is None or member_gain is None)
                        and not bool(args.optimized_only)
                    ):
                        if baseline_key not in baseline_test_by_task_seed:
                            raise ValueError(
                                "shared_static_reference must run before optimized settings "
                                "so test is evaluated once per optimized run"
                            )
                        initial_test = baseline_test_by_task_seed[baseline_key]
                        member_gain = _member_gain_summary(
                            dataset_metrics_from_dict(initial_test),
                            dataset_metrics_from_dict(selected_test),
                        )
                local_after = _solver_cache_snapshot(
                    mutable_cache_path, expected_evaluator=cache_evaluator,
                )
                prompts = _read_json(run_dir / "best_prompts.json")
                prompt_hashes = [
                    PromptEnsembleOptimizationSystem.prompt_hash(str(prompt))
                    for prompt in prompts
                ]
                if cfg.persistence.final_test_enabled:
                    test_audit = _test_observation_audit(
                        snapshot=local_after,
                        prompt_hashes=prompt_hashes,
                        test_rows=split_rows["test"],
                        task_type=cfg.data.task_type,
                        tie_break=cfg.peer_state.vote_tie_break,
                        seed=seed,
                    )
                    test_audit.update({
                        "final_test_enabled": True,
                        "final_test_evaluated": True,
                        "final_test_evaluation_count": int(
                            selection_summary["test_evaluation_count"]
                        ),
                        "test_observation_status": "evaluated",
                        "test_member_count_status": "available",
                        "test_drift_status": "checked",
                    })
                else:
                    test_audit = _not_applicable_test_audit(
                        prompt_hashes,
                        test_evaluation_count=int(
                            selection_summary["test_evaluation_count"]
                        ),
                    )
                history_key = (task_id, seed)
                comparisons = _test_observation_comparisons(
                    comparison_history.get(history_key, []),
                    test_audit,
                    final_test_enabled=cfg.persistence.final_test_enabled,
                )
                cache_match = _read_json(cache_match_path)
                cache_match.update({
                    "final_test_enabled": test_audit["final_test_enabled"],
                    "final_test_evaluated": test_audit["final_test_evaluated"],
                    "final_test_evaluation_count": test_audit[
                        "final_test_evaluation_count"
                    ],
                    "test_observation_status": test_audit[
                        "test_observation_status"
                    ],
                    "test_member_count_status": test_audit[
                        "test_member_count_status"
                    ],
                    "test_drift_status": test_audit["test_drift_status"],
                    "final_prompt_hashes": prompt_hashes,
                    "unchanged_prompt_comparisons": comparisons,
                })
                if cfg.persistence.final_test_enabled:
                    cache_match.update({
                        "test_question_set_hash": test_audit[
                            "test_question_set_hash"
                        ],
                        "test_observation_missing_count": test_audit[
                            "missing_entry_count"
                        ],
                        "test_team_vote_vector_hash": test_audit[
                            "team_vote_vector_hash"
                        ],
                        "test_team_vote_correct_count": test_audit[
                            "team_vote_correct_count"
                        ],
                        "test_per_agent_correct_counts": test_audit[
                            "per_agent_correct_counts"
                        ],
                        "unchanged_prompt_drift_count": sum(
                            int(row["per_question_drift_count"])
                            for row in comparisons
                        ),
                        "unchanged_prompt_aggregate_drift_count": sum(
                            len(row["aggregate_correct_count_drift_member_ids"])
                            for row in comparisons
                        ),
                        "unchanged_team_vote_drift_count": sum(
                            int(row["team_vote_vector_drift"])
                            for row in comparisons
                        ),
                    })
                cache_match["matched"] = bool(
                    cache_match.get("matched")
                    and (
                        not cfg.persistence.final_test_enabled
                        or (
                            test_audit["missing_entry_count"] == 0
                            and all(row["passed"] for row in comparisons)
                        )
                    )
                )
                cache_match["gate"] = "PASS" if cache_match["matched"] else "FAIL"
                cache_match_path.write_text(
                    json.dumps(cache_match, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if cache_match["gate"] != "PASS":
                    raise RuntimeError(f"Comparison observation gate failed: {run_dir}")
                if cfg.persistence.final_test_enabled:
                    comparison_history.setdefault(history_key, []).append({
                        "setting": setting.name,
                        "test_audit": test_audit,
                    })
                rows.append({
                    "task_id": task_id, "benchmark": task.benchmark, "setting": setting.name, "seed": seed,
                    "vote_acc_initial": (
                        initial_test["plurality_vote_acc"] if initial_test is not None else None
                    ),
                    "vote_acc_selected": (
                        selected_test["plurality_vote_acc"] if selected_test is not None else None
                    ),
                    "vote_gain": (
                        selected_test["plurality_vote_acc"]
                        - initial_test["plurality_vote_acc"]
                        if selected_test is not None and initial_test is not None else None
                    ),
                    "minimum_member_correct_count_gain": (
                        member_gain["minimum_member_correct_count_gain"]
                        if member_gain is not None else None
                    ),
                    "mean_member_correct_count_gain": (
                        member_gain["mean_member_correct_count_gain"]
                        if member_gain is not None else None
                    ),
                    "minimum_member_accuracy_gain": (
                        member_gain["minimum_member_accuracy_gain"]
                        if member_gain is not None else None
                    ),
                    "mean_member_accuracy_gain": (
                        member_gain["mean_member_accuracy_gain"]
                        if member_gain is not None else None
                    ),
                    "improved_agent_count": (
                        member_gain["improved_agent_count"] if member_gain is not None else None
                    ),
                    "regressed_agent_count": (
                        member_gain["regressed_agent_count"] if member_gain is not None else None
                    ),
                    "all_members_improved": (
                        member_gain["all_members_improved"] if member_gain is not None else None
                    ),
                    "optimized_only": bool(args.optimized_only),
                    "reporting_initial_test_reference_available": initial_test is not None,
                    "selected_mean_individual_acc": (
                        selected_test["mean_individual_acc"] if selected_test is not None else None
                    ),
                    "selected_min_individual_acc": (
                        selected_test["min_individual_acc"] if selected_test is not None else None
                    ),
                    "selected_mean_soft_vote_utility": (
                        selected_test["mean_soft_vote_utility"] if selected_test is not None else None
                    ),
                    "selected_mean_invalid_rate": (
                        selected_test["mean_invalid_rate"] if selected_test is not None else None
                    ),
                    "selected_tie_rate": (
                        selected_test["tie_rate"] if selected_test is not None else None
                    ),
                    "final_test_enabled": cfg.persistence.final_test_enabled,
                    "test_evaluation_count": selection_summary["test_evaluation_count"],
                    "planned_update_count": run_meta["planned_update_count"],
                    "completed_update_count": run_meta["completed_update_count"],
                    "early_stop_reason": run_meta.get("early_stop_reason", ""),
                    "frozen_initialization_matched": True,
                    "frozen_initialization_manifest_sha256": frozen_manifests[
                        (task_id, seed)
                    ]["manifest_sha256"],
                    "starting_comparison_reference_cache_sha256": cache_match[
                        "reference_cache_sha256"
                    ],
                    "mutable_cache_identity": hashlib.sha256(
                        cache_key.encode("utf-8")
                    ).hexdigest(),
                    "run_identity": expected_identity.to_dict(),
                    **split_integrity,
                })
                (root / "accuracy_results.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
                )
                (root / "experiment_runs.jsonl").write_text(
                    "".join(json.dumps({
                        "task_id": row["task_id"],
                        "setting": row["setting"],
                        "seed": row["seed"],
                        "run_identity": row["run_identity"],
                    }, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
    if len(mutable_cache_paths) != len(rows):
        raise RuntimeError("Every run must have a distinct mutable solver cache")
    (root / "matched_initialization_manifest.json").write_text(
        json.dumps({
            "manifest_version": FROZEN_INITIALIZATION_MANIFEST_VERSION,
            "frozen_initializations": list(frozen_manifests.values()),
            "run_count": len(rows),
            "distinct_mutable_cache_count": len(mutable_cache_paths),
            "optimized_only": bool(args.optimized_only),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    columns = list(rows[0]) if rows else [
        "task_id", "benchmark", "setting", "seed", "vote_acc_selected"
    ]
    with (root / "accuracy_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
