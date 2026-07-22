import json

from scripts.run_task_level_accuracy import _completed_run


def test_completed_run_requires_current_metadata_and_core_outputs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for filename, payload in {
        "final_summary.json": {"plurality_vote_acc": 0.5},
        "history.json": [],
        "best_prompts.json": ["p"] * 5,
        "run_meta.json": {
            "method_version": "peer_state_counterfactual_v1",
            "legacy_compatibility_enabled": False,
        },
    }.items():
        (run / filename).write_text(json.dumps(payload), encoding="utf-8")
    assert _completed_run(run) is True
    (run / "run_meta.json").write_text(json.dumps({"method_version": "old"}), encoding="utf-8")
    assert _completed_run(run) is False
