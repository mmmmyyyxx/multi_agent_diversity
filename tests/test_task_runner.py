import asyncio
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import (
    PersistentSolverCache,
)
from multi_dataset_diverse_rl.evaluation.prompt_question import (
    PromptAnswer,
    PromptQuestionEvaluator,
)
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.task_manifest import ComparisonTask
from scripts.run_task_level_accuracy import (
    RUNNER_FIELDS,
    _assert_runner_owned_child_command,
    _assert_runner_owned_child_paths,
    _build_child_command,
    _comparison_cache_source,
    _completed_run,
    _not_applicable_test_audit,
    _parser,
    _prepare_setting_local_cache,
    _sqlite_backup,
    _task_split_integrity,
    _test_observation_comparisons,
    _validate_setting_sequence,
    _with_runner_owned_paths,
    effective_proposal_memory_mode,
)


def identity():
    return RunIdentity(
        method_version="member_aware_peer_state_v14",
        experiment_setting="shared_full_dual_target_rcru",
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
    assert "optimized_only" not in RUNNER_FIELDS
    assert "shared_solver_cache_path" not in RUNNER_FIELDS
    assert "frozen_initialization_manifest_path" not in RUNNER_FIELDS
    actions = [action.dest for action in parser._actions]
    assert actions.count("resume_completed") == 1
    assert actions.count("optimized_only") == 1
    assert "shared_solver_cache_path" not in actions
    assert "frozen_initialization_manifest_path" not in actions


def test_task_runner_parser_rejects_runner_owned_path_overrides():
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--manifest", "manifest.yaml",
            "--out_root", "runs",
            "--shared_solver_cache_path", "root.sqlite",
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--manifest", "manifest.yaml",
            "--out_root", "runs",
            "--frozen_initialization_manifest_path", "wrong.json",
        ])


def test_runner_owned_paths_override_generic_and_setting_values(tmp_path):
    run_dir = tmp_path / "runs" / "setting"
    local = run_dir / "_solver_cache.sqlite"
    manifest = tmp_path / "frozen" / "frozen_initialization_manifest.json"
    values = _with_runner_owned_paths(
        {
            "shared_solver_cache_path": str(tmp_path / "hostile.sqlite"),
            "frozen_initialization_manifest_path": str(tmp_path / "wrong.json"),
        },
        run_dir=run_dir,
        solver_cache_path=local,
        frozen_manifest_path=manifest,
    )
    assert Path(values["shared_solver_cache_path"]) == local.resolve()
    assert Path(values["frozen_initialization_manifest_path"]) == manifest.resolve()


def test_prelaunch_path_assertions_reject_reference_cache_and_wrong_manifest(tmp_path):
    run_dir = tmp_path / "run"
    local = run_dir / "_solver_cache.sqlite"
    reference = tmp_path / "reference.sqlite"
    manifest = tmp_path / "frozen.json"
    child = {
        "shared_solver_cache_path": str(local),
        "frozen_initialization_manifest_path": str(manifest),
    }
    assert _assert_runner_owned_child_paths(
        child,
        run_dir=run_dir,
        frozen_manifest_path=manifest,
        comparison_reference_cache_path=reference,
    )["setting_local_cache_isolated"] is True

    tampered_cache = dict(child, shared_solver_cache_path=str(reference))
    with pytest.raises(RuntimeError, match="runner_owned_solver_cache_path_mismatch"):
        _assert_runner_owned_child_paths(
            tampered_cache,
            run_dir=run_dir,
            frozen_manifest_path=manifest,
            comparison_reference_cache_path=reference,
        )
    tampered_manifest = dict(
        child,
        frozen_initialization_manifest_path=str(tmp_path / "wrong.json"),
    )
    with pytest.raises(RuntimeError, match="runner_owned_frozen_manifest_path_mismatch"):
        _assert_runner_owned_child_paths(
            tampered_manifest,
            run_dir=run_dir,
            frozen_manifest_path=manifest,
            comparison_reference_cache_path=reference,
        )


def test_final_child_command_tampering_fails_before_launch(tmp_path):
    run_dir = tmp_path / "run"
    local = run_dir / "_solver_cache.sqlite"
    reference = tmp_path / "reference.sqlite"
    manifest = tmp_path / "frozen.json"
    child = _with_runner_owned_paths(
        {},
        run_dir=run_dir,
        solver_cache_path=local,
        frozen_manifest_path=manifest,
    )
    command = _build_child_command(child, python_executable="python")
    cache_value_index = command.index("--shared_solver_cache_path") + 1
    command[cache_value_index] = str(reference)
    with pytest.raises(RuntimeError, match="runner_owned_solver_cache_path_mismatch"):
        _assert_runner_owned_child_command(
            command,
            run_dir=run_dir,
            frozen_manifest_path=manifest,
            comparison_reference_cache_path=reference,
        )


def _cache_evaluator(path=None):
    return PromptQuestionEvaluator(
        model_request_identity="request-v1",
        parser_version="parser-v1",
        temperature=0.0,
        decoding_seed=46,
        cache_metadata={
            "solver_model": "solver",
            "endpoint_identity": "endpoint",
            "output_contract_version": "contract",
            "max_tokens": 1800,
        },
        shared_cache=(PersistentSolverCache(path) if path is not None else None),
    )


def _seed_observation_cache(path, *, correct_count, entry_count=75):
    evaluator = _cache_evaluator()
    cache = PersistentSolverCache(path)
    for index in range(entry_count):
        prompt_hash = "prompt"
        question_hash = f"question-{index}"
        key = evaluator.key(prompt_hash, question_hash)
        metadata = {
            **evaluator.cache_metadata,
            "model_request_identity": evaluator.model_request_identity,
            "parser_version": evaluator.parser_version,
            "temperature": evaluator.temperature,
            "evaluation_replica_seed": evaluator.decoding_seed,
            "prompt_hash": prompt_hash,
            "question_hash": question_hash,
        }
        assert cache._claim_or_read(key, metadata)[0] == "owner"
        answer = "A" if index < correct_count else "B"
        cache._store(key, PromptAnswer(answer, f"trace-{answer}-{index}", True))


def test_exact_frozen_observations_are_reused_from_setting_local_clone(tmp_path):
    frozen = tmp_path / "frozen.sqlite"
    reference = tmp_path / "reference.sqlite"
    hostile = tmp_path / "hostile-root.sqlite"
    local = tmp_path / "run" / "_solver_cache.sqlite"
    _seed_observation_cache(frozen, correct_count=62)
    _seed_observation_cache(hostile, correct_count=60)
    _sqlite_backup(frozen, reference)

    audit = _prepare_setting_local_cache(
        comparison_reference_cache_path=reference,
        frozen_cache_path=frozen,
        setting_local_cache_path=local,
        expected_evaluator=_cache_evaluator(),
    )
    assert audit["frozen_ready_entry_count"] == 75
    assert audit["local_frozen_entry_count"] == 75
    assert audit["frozen_observation_set_matched"] is True
    assert audit["setting_local_cache_isolated"] is True
    assert local.resolve() not in {reference.resolve(), hostile.resolve()}

    provider_calls = 0

    async def evaluate_all():
        nonlocal provider_calls
        evaluator = _cache_evaluator(local)
        answers = []
        for index in range(75):
            async def forbidden(*_args):
                nonlocal provider_calls
                provider_calls += 1
                raise AssertionError("provider must not be called for frozen rows")

            answers.append(await evaluator.evaluate(
                question=f"question-{index}",
                question_hash=f"question-{index}",
                prompt="prompt",
                prompt_hash="prompt",
                agent_id=0,
                solve=forbidden,
            ))
        return answers

    answers = asyncio.run(evaluate_all())
    assert sum(answer.answer == "A" for answer in answers) == 62
    assert provider_calls == 0


def test_partial_frozen_ready_set_is_cloned_exactly_without_fixed_count_assumption(
    tmp_path,
):
    frozen = tmp_path / "partial-frozen.sqlite"
    reference = tmp_path / "partial-reference.sqlite"
    local = tmp_path / "run" / "_solver_cache.sqlite"
    _seed_observation_cache(frozen, correct_count=62, entry_count=62)
    _sqlite_backup(frozen, reference)
    audit = _prepare_setting_local_cache(
        comparison_reference_cache_path=reference,
        frozen_cache_path=frozen,
        setting_local_cache_path=local,
        expected_evaluator=_cache_evaluator(),
    )
    assert audit["frozen_ready_entry_count"] == 62
    assert audit["local_frozen_entry_count"] == 62
    assert audit["missing_frozen_observation_count"] == 0
    assert audit["conflicting_frozen_observation_count"] == 0
    assert audit["frozen_observation_set_matched"] is True


def test_optimized_only_allows_one_non_baseline_setting_without_synthetic_reference():
    _validate_setting_sequence(
        ["shared_full_dual_target_rcru"],
        optimized_only=True,
    )
    with pytest.raises(ValueError, match="exactly one non-baseline"):
        _validate_setting_sequence(
            ["shared_static_reference"], optimized_only=True
        )
    with pytest.raises(ValueError, match="exactly one non-baseline"):
        _validate_setting_sequence(
            ["shared_full_dual_target_rcru", "shared_member_aware_dual_target"],
            optimized_only=True,
        )


def test_standard_comparison_still_requires_baseline_first():
    _validate_setting_sequence(
        ["shared_static_reference", "shared_full_dual_target_rcru"],
        optimized_only=False,
    )
    with pytest.raises(ValueError, match="shared_static_reference first"):
        _validate_setting_sequence(
            ["shared_full_dual_target_rcru"],
            optimized_only=False,
        )


def test_no_test_runner_manifest_is_not_applicable_and_has_no_fake_arrays():
    audit = _not_applicable_test_audit(
        ["a", "b", "c", "d", "e"],
        test_evaluation_count=0,
    )
    assert audit == {
        "prompt_hashes": ["a", "b", "c", "d", "e"],
        "final_test_enabled": False,
        "final_test_evaluated": False,
        "final_test_evaluation_count": 0,
        "test_observation_status": "not_applicable",
        "test_member_count_status": "not_applicable",
        "test_drift_status": "not_applicable",
    }
    assert "per_agent_correct_counts" not in audit
    assert "team_vote_vector" not in audit
    assert _test_observation_comparisons(
        [{"setting": "prior", "test_audit": {"invalid": "would crash"}}],
        audit,
        final_test_enabled=False,
    ) == []


def test_no_test_runner_rejects_nonzero_test_evaluation_count():
    with pytest.raises(ValueError, match="test_evaluation_count=0"):
        _not_applicable_test_audit(
            ["a", "b", "c", "d", "e"],
            test_evaluation_count=1,
        )


def test_memory_treatment_applies_only_to_responsibility_conditioned_full_run():
    assert effective_proposal_memory_mode(
        "shared_static_reference", "state_local_v1"
    ) == "off"
    assert effective_proposal_memory_mode(
        "shared_full_dual_target_rcru", "state_local_v1"
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
            "method_version": "member_aware_peer_state_v14",
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
    for name in (
        "repairability_adjusted_target_scores.jsonl",
        "dual_target_branch_decisions.jsonl",
        "dual_target_commit_decisions.jsonl",
        "repairability_failure_events.jsonl",
        "repairability_reset_events.jsonl",
    ):
        (run / name).write_text("", encoding="utf-8")
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
            "method_version": "member_aware_peer_state_v14", "legacy_compatibility_enabled": False,
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
    for name in (
        "tcs_rounds.jsonl",
        "solver_invalid_outputs.jsonl",
        "student_recovery_observations.jsonl",
        "repairability_adjusted_target_scores.jsonl",
        "dual_target_branch_decisions.jsonl",
        "dual_target_commit_decisions.jsonl",
        "repairability_failure_events.jsonl",
        "repairability_reset_events.jsonl",
    ):
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
