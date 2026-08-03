import json

import pytest

from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.task_manifest import ComparisonTask
from scripts.run_task_level_accuracy import (
    RUNNER_FIELDS,
    _comparison_cache_source,
    _completed_run,
    _parser,
    _task_split_integrity,
    effective_proposal_memory_mode,
)


def identity():
    return RunIdentity(
        method_version="member_aware_peer_state_v9",
        experiment_setting="shared_member_aware_full",
        git_commit="commit",
        git_dirty=False,
        config_fingerprint="config",
        manifest_sha256="manifest",
        train_file_sha256="train",
        val_file_sha256="val",
        test_file_sha256="test",
        train_question_set_hash="train-q",
        val_question_set_hash="val-q",
        test_question_set_hash="test-q",
    )


def test_task_runner_parser_builds_and_resume_completed_is_registered_once():
    parser = _parser()
    assert parser is not None
    assert "resume_completed" not in RUNNER_FIELDS
    actions = [action.dest for action in parser._actions]
    assert actions.count("resume_completed") == 1


def test_memory_treatment_applies_only_to_responsibility_conditioned_full_run():
    assert effective_proposal_memory_mode(
        "shared_baseline", "state_local_v1"
    ) == "off"
    assert effective_proposal_memory_mode(
        "shared_member_aware_full", "state_local_v1"
    ) == "state_local_v1"


def test_completed_run_requires_exact_identity(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for filename, payload in {
        "final_summary.json": {
            "initial_test": {},
            "selected_test": {},
            "member_gain": {},
            "selection_summary": {},
        },
        "history.json": [],
        "best_prompts.json": ["p"] * 5,
        "run_meta.json": {
            "method_version": "member_aware_peer_state_v9",
            "legacy_compatibility_enabled": False,
            "solver_output_contract_version": "task_output_contract_v1",
            "shared_solver_cache_path": "shared.sqlite",
            "run_identity": identity().to_dict(),
        },
        "cost_summary.json": {"total_llm_calls": 1},
        "candidate_funnel.json": {"update_count": 1},
        "frozen_initialization_match.json": {"matched": True},
        "comparison_cache_match.json": {"matched": True},
    }.items():
        (run / filename).write_text(json.dumps(payload), encoding="utf-8")
    (run / "tcs_rounds.jsonl").write_text("", encoding="utf-8")
    (run / "solver_invalid_outputs.jsonl").write_text("", encoding="utf-8")
    (run / "student_recovery_observations.jsonl").write_text(
        "", encoding="utf-8"
    )
    assert _completed_run(run, identity()) is True
    metadata = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    metadata["run_identity"]["config_fingerprint"] = "different"
    (run / "run_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Run identity mismatch"):
        _completed_run(run, identity())


def test_incomplete_run_is_not_reused(tmp_path):
    assert _completed_run(tmp_path, identity()) is False


def test_memory_completed_run_requires_memory_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    payloads = {
        "final_summary.json": {
            "initial_test": {}, "selected_test": {}, "member_gain": {}, "selection_summary": {},
        },
        "history.json": [], "best_prompts.json": ["p"] * 5,
        "run_meta.json": {
            "method_version": "member_aware_peer_state_v9", "legacy_compatibility_enabled": False,
            "solver_output_contract_version": "task_output_contract_v1",
            "shared_solver_cache_path": "shared.sqlite", "run_identity": identity().to_dict(),
            "config": {"proposal_memory_mode": "state_local_v1"},
        },
        "cost_summary.json": {}, "candidate_funnel.json": {},
        "frozen_initialization_match.json": {"matched": True},
        "comparison_cache_match.json": {"matched": True},
    }
    for name, payload in payloads.items():
        (run / name).write_text(json.dumps(payload), encoding="utf-8")
    for name in ("tcs_rounds.jsonl", "solver_invalid_outputs.jsonl", "student_recovery_observations.jsonl"):
        (run / name).write_text("", encoding="utf-8")
    assert _completed_run(run, identity()) is False
    for name in (
        "proposal_memory_events_sanitized.jsonl", "proposal_memory_summary.json",
        "proposal_memory_key_isolation_audit.json", "proposal_rotation_trajectory.jsonl",
    ):
        (run / name).write_text("{}" if name.endswith(".json") else "", encoding="utf-8")
    assert _completed_run(run, identity()) is True


def test_task_split_integrity_rejects_option_letter_gold_outside_question(tmp_path):
    paths = []
    for index, name in enumerate(("train", "val", "test")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({
            "question": f"q{index}\n(A) first\n(B) second",
            "answer": "literal answer",
        }) + "\n", encoding="utf-8")
        paths.append(str(path))
    task = ComparisonTask(
        "toy", "BBH", "bbh", "option_letter", *paths,
    )
    with pytest.raises(ValueError, match="answer-space integrity"):
        _task_split_integrity(task, "legacy", str(tmp_path))


def test_task_split_integrity_accepts_unique_option_letter_gold(tmp_path):
    paths = []
    for index, name in enumerate(("train", "val", "test")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({
            "question": f"q{index}\n(A) first\n(B) second",
            "answer": "(B)",
        }) + "\n", encoding="utf-8")
        paths.append(str(path))
    task = ComparisonTask(
        "toy", "BBH", "bbh", "option_letter", *paths,
    )
    audit = _task_split_integrity(task, "legacy", str(tmp_path))
    assert audit["option_letter_invalid_gold_count"] == 0
    assert audit["option_letter_ambiguous_gold_count"] == 0


def test_optimized_setting_requires_baseline_comparison_cache(tmp_path):
    reference = tmp_path / "comparison.sqlite"
    with pytest.raises(FileNotFoundError, match="cumulative task-seed comparison cache"):
        _comparison_cache_source(
            comparison_reference_cache_path=reference,
        )
    reference.write_bytes(b"baseline plus matched test")
    source, role = _comparison_cache_source(
        comparison_reference_cache_path=reference,
    )
    assert (source, role) == (
        reference,
        "cumulative_task_seed_observation_reference",
    )
