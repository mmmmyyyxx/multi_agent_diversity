"""The post-execution trajectory projection never enters optimization."""

import json

import pytest

from scripts.derive_formal_trajectory_trace import derive
from scripts.freeze_formal_v3_execution import freeze
from multi_dataset_diverse_rl.formal_final_team import persist_final_team, team_hash
from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary


def test_completed_native_trajectory_is_exclusive_and_deterministic(tmp_path):
    (tmp_path / "ledger.jsonl").write_text("", encoding="utf-8")
    final = persist_final_team(
        tmp_path / "final_team_materialization.json", mode="GEPA_NATIVE",
        initial_prompts=["P0"] * 5, final_prompts=["P1"] * 5,
        candidate_id="a" * 64, initial_team_hash=team_hash(["P0"] * 5),
        search_final_identity="a" * 64,
    )
    summary = {
        "experiment_id": "gepa_saturation_comparison_v3_seed80_native_attempt2",
        "mode_id": "GEPA_NATIVE", "seed": 80,
        "validation50_calls": 0, "test50_calls": 0,
        "final_native_candidate_hash": "a" * 64,
        "initial_team_hash": team_hash(["P0"] * 5),
        "final_team_materialization": final,
        "ledger": ledger_summary(tmp_path / "ledger.jsonl"),
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
        json.dumps({"status": "RUNNING", "attempt_id": summary["experiment_id"]}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed frozen execution"):
        derive(tmp_path)
    (tmp_path / "run_lifecycle.json").write_text(
        json.dumps({"status": "EXECUTION_COMPLETE", "attempt_id": summary["experiment_id"],
                    "provider_attempts": 0, "provider_successes": 0, "provider_failures": 0}), encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        derive(tmp_path)
    freeze(tmp_path)
    assert derive(tmp_path)["record_count"] == 1
    output = (tmp_path / "formal_trajectory_trace.jsonl").read_text(encoding="utf-8")
    assert json.loads(output)["final_native_candidate_hash"] == "a" * 64
    with pytest.raises(FileExistsError):
        derive(tmp_path)
    assert (tmp_path / "formal_trajectory_trace.jsonl").read_text(encoding="utf-8") == output
