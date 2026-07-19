import json
from pathlib import Path

import pytest

from scripts.audit_runs import choose_retention, inspect_root
from scripts.prune_runs import apply_plan, collect_targets, validate_target


def write_complete_run(root: Path, *, setting: str = "unknown_setting", method_version: str = "") -> Path:
    run_dir = root / "task" / f"{setting}_seed42"
    run_dir.mkdir(parents=True)
    config = {"setting": setting, "seed": 42, "comparison_task_id": "task", "baseline_only": False}
    if method_version:
        config["method_version"] = method_version
    (run_dir / "run_meta.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps([{"epoch": "final", "test": {"vote_acc": 0.5}}]), encoding="utf-8")
    (run_dir / "cost_summary.json").write_text("{}", encoding="utf-8")
    return run_dir


def planned_row(root: Path, classification: str = "FAILED") -> dict:
    return {
        "path": root.name,
        "resolved_path": str(root.resolve()),
        "classification": classification,
        "keep": False,
        "reason": "test cleanup",
        "size_bytes": 2,
        "file_count": 1,
        "versions": [],
        "settings": [],
        "tasks": [],
        "seeds": [],
    }


def test_unknown_completed_run_is_ambiguous_and_retained(tmp_path):
    root = tmp_path / "runs_unknown"
    write_complete_run(root)
    row = inspect_root(root)
    choose_retention([row], {})
    assert row["classification"] == "AMBIGUOUS"
    assert row["keep"] is True


@pytest.mark.parametrize("name", ["../runs_escape", "runs_nested/path", "config", "."])
def test_validate_target_refuses_unsafe_paths(tmp_path, name):
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_target(tmp_path.resolve(), name, [])


def test_validate_target_refuses_lock_and_active_process(tmp_path):
    root = tmp_path / "runs_locked"
    root.mkdir()
    (root / "RUNNING.lock").touch()
    with pytest.raises(RuntimeError, match="active lock"):
        validate_target(tmp_path.resolve(), root.name, [])
    (root / "RUNNING.lock").unlink()
    with pytest.raises(RuntimeError, match="active process"):
        validate_target(tmp_path.resolve(), root.name, [f"runner --out_root {root.resolve()}"])


def test_validate_target_refuses_symlink(tmp_path):
    source = tmp_path / "outside"
    source.mkdir()
    link = tmp_path / "runs_link"
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable for this Windows account")
    with pytest.raises(ValueError, match="symlink"):
        validate_target(tmp_path.resolve(), link.name, [])


def test_collect_targets_keeps_ambiguous_and_only_selects_direct_runs(tmp_path):
    delete = tmp_path / "runs_delete"
    ambiguous = tmp_path / "runs_ambiguous"
    delete.mkdir()
    ambiguous.mkdir()
    plan = {
        "schema_version": 2,
        "workspace": str(tmp_path.resolve()),
        "roots": [planned_row(delete), planned_row(ambiguous, "AMBIGUOUS")],
    }
    targets = collect_targets(tmp_path.resolve(), plan, [])
    assert [path.name for path, _ in targets] == ["runs_delete"]
    assert ambiguous.exists()


def test_apply_writes_compact_record_before_deleting(tmp_path):
    root = tmp_path / "runs_failed"
    root.mkdir()
    (root / "error.log").write_text("ERROR test failure", encoding="utf-8")
    targets = [(root.resolve(), planned_row(root))]
    records = apply_plan(tmp_path.resolve(), targets, command_line_provider=lambda: [])
    record_path = tmp_path / "run_records" / "runs_failed.json"
    index_path = tmp_path / "run_records" / "index.jsonl"
    assert not root.exists()
    assert record_path.exists() and index_path.exists()
    assert records[0]["failure_summary"] == "ERROR test failure"


def test_collect_targets_is_a_non_destructive_dry_run(tmp_path):
    root = tmp_path / "runs_dry"
    root.mkdir()
    plan = {"schema_version": 2, "workspace": str(tmp_path.resolve()), "roots": [planned_row(root)]}
    assert collect_targets(tmp_path.resolve(), plan, [])
    assert root.exists()
