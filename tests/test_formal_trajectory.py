"""The post-execution trajectory projection never enters optimization."""

import json

import pytest

from scripts.derive_formal_trajectory_trace import derive


def test_completed_native_trajectory_is_exclusive_and_deterministic(tmp_path):
    summary = {
        "experiment_id": "gepa_saturation_comparison_v3_seed80_native_attempt2",
        "mode_id": "GEPA_NATIVE", "seed": 80,
        "validation50_calls": 0, "test50_calls": 0,
        "final_native_candidate_hash": "a" * 64,
        "events": [{
            "index": 1, "kind": "NATIVE_OPTIMIZATION_UNIT",
            "candidate_ids": ["candidate-1"], "state_hash": "a" * 64,
            "stop_reason": "SATURATION_REACHED",
            "telemetry": {"backend_termination_reason": "SATURATION_REACHED",
                          "backend_saturation": {"local_no_update_counter": 3}},
        }],
    }
    (tmp_path / "execution_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "run_lifecycle.json").write_text(
        json.dumps({"status": "RUNNING"}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed frozen execution"):
        derive(tmp_path)
    (tmp_path / "run_lifecycle.json").write_text(
        json.dumps({"status": "EXECUTION_COMPLETE"}), encoding="utf-8",
    )
    assert derive(tmp_path)["record_count"] == 1
    output = (tmp_path / "formal_trajectory_trace.jsonl").read_text(encoding="utf-8")
    assert json.loads(output)["final_native_candidate_hash"] == "a" * 64
    with pytest.raises(FileExistsError):
        derive(tmp_path)
    assert (tmp_path / "formal_trajectory_trace.jsonl").read_text(encoding="utf-8") == output
