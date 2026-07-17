import csv
import hashlib
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from scripts.analyze_prompt_evolution import analyze, normalize_prompt, prompt_change_ratio


INITIAL = "Use a short explicit method and verify the final answer."
CANDIDATE = (
    "First classify the relation and bind every entity. Check qualifiers, eliminate "
    "contradictions, compare the remaining options, then verify the final answer."
)


def _hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_run(root: Path, *, partial: bool = False, legacy: bool = False) -> Path:
    run = root / "task" / "shared_vote_error_pareto_tcs_residual_cycle_guard_seed42"
    run.mkdir(parents=True)
    config = {
        "agents": 1,
        "prompt_max_change_ratio": 0.20,
        "prompt_large_shift_warmup_accepts": 2,
        "prompt_large_shift_min_vote_delta": 0.02,
        "baseline_allowed_vote_loss": 0.0,
        "mechanism_trust_region_enabled": True,
    }
    _write_json(run / "run_meta.json", {
        "comparison_task_id": "task",
        "agents": 1,
        "initial_agent_prompts": [INITIAL],
        "config": config,
    })
    _write_json(run / "prompt_history.json", {
        "agent_0": {"initial_prompt": INITIAL, "current_prompt": CANDIDATE}
    })
    common = {
        "event": "candidate_evaluated", "epoch": 1, "step": 10, "agent_id": 0,
        "update_attempt_id": "attempt-1", "parent_id": "g0_parent",
        "parent_prompt_hash": _hash(INITIAL), "accuracy_delta": 0.0,
        "vote_delta": 0.03, "vote_loss_rate": 0.0, "pivotal_loss_rate": 0.0,
        "shared_error_rescue_score": 0.1, "shared_error_creation_score": 0.0,
        "accuracy_guard_passed": True, "invalid_guard_passed": True,
        "error_dependence_guard_passed": True, "behavior_cycle_guard_passed": True,
        "prompt_trust_region_passed": True, "pareto_feasible": True,
        "pareto_selected": True, "in_top_beam": True,
    }
    optimizer = {
        **common, "candidate_id": "g1_optimizer", "prompt_hash": _hash(CANDIDATE),
        "prompt_preview": CANDIDATE, "prompt_change_ratio": prompt_change_ratio(INITIAL, CANDIDATE),
        "candidate_source": "student", "candidate_pool_source": "optimizer",
        "is_top1": True, "active_prompt_changed": True,
    }
    existing = {
        **common, "candidate_id": "g0_parent", "prompt_hash": _hash(INITIAL),
        "prompt_preview": INITIAL, "prompt_change_ratio": 0.0,
        "candidate_source": "existing_beam", "candidate_pool_source": "existing_beam",
        "is_top1": False, "active_prompt_changed": False,
    }
    if legacy:
        for key in (
            "pivotal_loss_rate", "shared_error_rescue_score", "shared_error_creation_score",
            "behavior_cycle_guard_passed", "prompt_trust_region_passed",
        ):
            optimizer.pop(key, None)
    rows = [optimizer, existing]
    if not partial:
        rows.append({
            "event": "beam_update_summary", "epoch": 1, "step": 10, "agent_id": 0,
            "update_attempt_id": "attempt-1", "student_candidate_count_raw": 1,
            "num_optimizer_candidates": 1,
        })
    _write_jsonl(run / "update_logs.jsonl", rows)
    if partial:
        _write_json(run / "training_checkpoint.json", {"version": 3, "cursor": 9})
    return run


def _read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_prompt_distance_matches_runtime_definition():
    system = TraceBeamSearchSystem.__new__(TraceBeamSearchSystem)
    pairs = [
        ("  SAME\nPrompt ", "same prompt"),
        (INITIAL, CANDIDATE),
        ("", "new prompt"),
    ]
    for parent, candidate in pairs:
        assert prompt_change_ratio(parent, candidate) == pytest.approx(
            system.prompt_change_ratio(parent, candidate)
        )


def test_analysis_records_all_evaluated_candidates_and_funnel(tmp_path):
    root = tmp_path / "runs"
    _make_run(root)
    before = {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "audit"
    summary = analyze([root], output)

    candidates = _read_csv(output / "prompt_candidate_trajectory.csv")
    attempts = _read_csv(output / "prompt_update_attempt_summary.csv")
    assert len(candidates) == 2
    assert len(attempts) == 1
    assert sum(row["became_active_top1"].lower() == "true" for row in candidates) == 1
    optimizer = next(row for row in candidates if row["candidate_pool_source"] == "optimizer")
    existing = next(row for row in candidates if row["candidate_pool_source"] == "existing_beam")
    assert optimizer["large_shift"].lower() == "true"
    assert optimizer["large_shift_warmup_exempt"].lower() == "true"
    assert optimizer["large_shift_supported"].lower() == "true"
    assert all(optimizer[name].lower() == "true" for name in (
        "large_shift_vote_support_passed", "large_shift_accuracy_support_passed",
        "large_shift_vote_loss_support_passed", "large_shift_pivotal_loss_support_passed",
        "large_shift_shared_error_support_passed",
    ))
    assert existing["large_shift"].lower() == "false"
    assert summary["api_calls"] == 0
    assert summary["model_calls"] == 0
    assert before == {path: path.read_bytes() for path in before}


def test_legacy_partial_run_produces_partial_report(tmp_path):
    root = tmp_path / "legacy_runs"
    _make_run(root, partial=True, legacy=True)
    output = tmp_path / "audit"
    summary = analyze([root], output)

    assert summary["coverage"]["run_count"] == 1
    assert summary["coverage"]["candidate_count"] == 2
    assert summary["coverage"]["runs_with_checkpoint"] == 1
    assert summary["coverage"]["complete_attempt_count"] == 0
    assert (output / "PROMPT_EVOLUTION_AUDIT.md").exists()
    assert len(_read_csv(output / "prompt_agent_path_summary.csv")) == 1
