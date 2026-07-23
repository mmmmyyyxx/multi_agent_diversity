import json
from pathlib import Path

import yaml

from scripts.preflight_member_aware import build_parser, run_specific_preflight


def test_run_specific_preflight_builds_identity_and_checks_inputs(tmp_path, monkeypatch):
    split_paths = {}
    for name, question in (("train", "train q"), ("val", "val q"), ("test", "test q")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"question": question, "answer": "A"}) + "\n", encoding="utf-8")
        split_paths[name] = str(path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "tasks": {
            "task": {
                "benchmark": "BBH",
                "task_type": "bbh",
                "answer_format": "option_letter",
                "train_path": split_paths["train"],
                "val_path": split_paths["val"],
                "test_path": split_paths["test"],
            }
        }
    }), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    args = build_parser().parse_args([
        "--manifest", str(manifest),
        "--tasks", "task",
        "--settings", "shared_member_aware_full",
        "--seeds", "42",
        "--out_root", str(tmp_path / "runs"),
        "--train_size", "1",
        "--val_size", "1",
        "--test_size", "1",
        "--num_candidates_per_parent", "1",
        "--stage_b_candidate_budget", "1",
        "--max_total_llm_calls", "100",
        "--max_total_tokens", "10000",
    ])
    workspace = Path(__file__).resolve().parents[1]
    report = run_specific_preflight(args, workspace)
    assert report["ok"] is True
    assert len(report["runs"]) == 1
    run = report["runs"][0]
    assert run["split_integrity"]["opt_val_question_overlap"] == 0
    assert run["run_identity"]["experiment_setting"] == "shared_member_aware_full"
    assert Path(run["shared_solver_cache_path"]).name == "_shared_solver_cache.sqlite"
    assert Path(run["shared_solver_cache_path"]).is_file()

    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True)
    stale = dict(run["run_identity"])
    stale["config_fingerprint"] = "stale"
    (run_dir / "run_meta.json").write_text(
        json.dumps({"run_identity": stale}), encoding="utf-8",
    )
    rejected = run_specific_preflight(args, workspace)
    assert rejected["ok"] is False
    assert "Run identity mismatch" in rejected["errors"][0]
