import json

import pytest

from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from scripts.run_task_level_accuracy import RUNNER_FIELDS, _completed_run, _parser


def identity():
    return RunIdentity(
        method_version="peer_state_counterfactual_v1",
        experiment_setting="shared_peer_state_full",
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


def test_completed_run_requires_exact_identity(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for filename, payload in {
        "final_summary.json": {"plurality_vote_acc": 0.5},
        "history.json": [],
        "best_prompts.json": ["p"] * 5,
        "run_meta.json": {
            "method_version": "peer_state_counterfactual_v1",
            "legacy_compatibility_enabled": False,
            "solver_output_contract_version": "task_output_contract_v1",
            "shared_solver_cache_path": "shared.sqlite",
            "run_identity": identity().to_dict(),
        },
        "cost_summary.json": {"total_llm_calls": 1},
    }.items():
        (run / filename).write_text(json.dumps(payload), encoding="utf-8")
    (run / "tcs_rounds.jsonl").write_text("", encoding="utf-8")
    (run / "solver_invalid_outputs.jsonl").write_text("", encoding="utf-8")
    assert _completed_run(run, identity()) is True
    metadata = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    metadata["run_identity"]["config_fingerprint"] = "different"
    (run / "run_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Run identity mismatch"):
        _completed_run(run, identity())


def test_incomplete_run_is_not_reused(tmp_path):
    assert _completed_run(tmp_path, identity()) is False
