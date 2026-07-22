import json

from multi_dataset_diverse_rl.cli import configure_runtime_probe_version
from multi_dataset_diverse_rl.config import Config
from scripts.experiment_config import select_settings
from scripts.preflight_v9_pilot import MATCHED_V9_SETTINGS, run_preflight, static_preflight_errors


def test_v9_matched_static_preflight_passes():
    assert static_preflight_errors() == []


def test_v9_runtime_probe_version_matches_each_preset_and_v8_is_unchanged():
    names = [
        *MATCHED_V9_SETTINGS,
        "shared_accuracy_rollout_embedding_tcs",
        "shared_vote_ready_rollout_diversity_tcs",
    ]
    for setting in select_settings(",".join(names)):
        cfg = Config(**setting.resolved_overrides())
        holder = type("Holder", (), {})()
        assert configure_runtime_probe_version(holder, cfg) == cfg.probe_stability_version
        assert holder.prompt_probe_version == cfg.probe_stability_version


def test_preflight_checks_run_meta_commit_and_probe_version(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_dir = run_root / "task" / "setting_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps({
        "git_commit": "head",
        "probe_stability_version": "probe-v2",
        "prompt_probe_version": "probe-v2",
    }), encoding="utf-8")
    monkeypatch.setattr("scripts.preflight_v9_pilot._git", lambda _workspace, *args: "" if args[0] == "status" else "head")
    report = run_preflight(tmp_path, run_root=run_root)
    assert report["ok"] is True
    assert report["checked_run_meta"] == 1


def test_preflight_rejects_dirty_tree_and_stale_run_meta(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    (run_root / "run_meta.json").write_text(json.dumps({
        "git_commit": "old",
        "probe_stability_version": "probe-v2",
        "prompt_probe_version": "probe-v1",
    }), encoding="utf-8")
    monkeypatch.setattr("scripts.preflight_v9_pilot._git", lambda _workspace, *args: " M file" if args[0] == "status" else "head")
    report = run_preflight(tmp_path, run_root=run_root)
    assert report["ok"] is False
    assert any("not clean" in error for error in report["errors"])
    assert any("git_commit" in error for error in report["errors"])
    assert any("probe version mismatch" in error for error in report["errors"])
