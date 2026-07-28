"""Build a payload-free comparison report for one v4 and v5 matched pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


FORBIDDEN_TOKENS = ("http://", "https://", "final_answer:", "openai_api_key")
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?:[a-z]:[\\/]|file://|\\\\[^\\/\s]+[\\/][^\\/\s]+|(?:^|[\s\"'=])/(?:[^\s\"']*))"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_facts(run: Path) -> dict[str, Any]:
    meta = read_json(run / "run_meta.json")
    summary = read_json(run / "final_summary.json")
    behavior = read_json(run / "final_test_differentiation.json")
    final_state = meta["final_state_selection"]
    return {
        "method_version": meta["method_version"],
        "git_commit": meta["run_identity"]["git_commit"],
        "git_dirty": meta["run_identity"]["git_dirty"],
        "seed": meta["config"]["seed"],
        "models": {
            "solver": meta["config"]["agent_model"],
            "optimizer": meta["config"]["optimizer_model"],
            "evaluator": meta["config"]["evaluator_model"],
        },
        "split_hashes": {
            key: meta["run_identity"][key]
            for key in ("train_file_sha256", "val_file_sha256", "test_file_sha256")
        },
        "initial_prompt_hashes": meta["initial_prompt_hashes"],
        "planned_update_count": meta["planned_update_count"],
        "completed_update_count": meta["completed_update_count"],
        "training_completed": meta["training_completed"],
        "validation_used": meta["validation_used"],
        "selected_source": final_state["selected_checkpoint_source"],
        "selected_update_index": final_state["selected_checkpoint_update_index"],
        "test_evaluation_count": meta["test_evaluation_count"],
        "test_before_training_complete": meta["test_called_before_training_complete"],
        "test_used_for_training": meta["test_used_for_training"],
        "test_used_for_selection": meta["test_used_for_selection"],
        "final_test": {
            key: behavior[key]
            for key in (
                "team_vote_correct_count", "team_vote_accuracy", "mean_H", "mean_G", "mean_M",
                "oracle_covered_but_vote_wrong_rate", "terminal_invalid_count", "n_eff",
            )
        },
        "reported_selected_vote_correct_count": summary["selected_test"]["vote_correct_count"],
    }


def scan(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            text = file.read_text(encoding="utf-8").lower()
            if any(token in text for token in FORBIDDEN_TOKENS) or ABSOLUTE_PATH_PATTERN.search(text):
                raise ValueError(f"sanitized scan failed: {file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4_baseline", type=Path, required=True)
    parser.add_argument("--v4_full", type=Path, required=True)
    parser.add_argument("--v5_baseline", type=Path, required=True)
    parser.add_argument("--v5_full", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"out_dir must be fresh: {args.out_dir}")

    v4_baseline, v4_full = run_facts(args.v4_baseline), run_facts(args.v4_full)
    v5_baseline, v5_full = run_facts(args.v5_baseline), run_facts(args.v5_full)
    alignment = read_jsonl(args.v5_full / "target_owner_context_alignment.jsonl")
    alignment_pass_count = sum(bool(row["assertion_passed"]) for row in alignment)
    shared_cache_pairing = (
        read_json(args.v4_full / "run_meta.json")["shared_solver_cache_path"]
        == read_json(args.v5_full / "run_meta.json")["shared_solver_cache_path"]
    )
    matched = {
        "seed": v4_full["seed"] == v5_full["seed"] == 45,
        "models": v4_full["models"] == v5_full["models"],
        "split_hashes": v4_full["split_hashes"] == v5_full["split_hashes"],
        "initial_prompt_hashes": v4_full["initial_prompt_hashes"] == v5_full["initial_prompt_hashes"],
        "shared_cache_path": shared_cache_pairing,
        "v4_lifecycle": (
            v4_full["planned_update_count"] == v4_full["completed_update_count"] == 32
            and v4_full["training_completed"] and v4_full["selected_source"] == "final_active_state"
            and v4_full["test_evaluation_count"] == 1 and not v4_full["test_before_training_complete"]
        ),
        "v5_lifecycle": (
            v5_full["planned_update_count"] == v5_full["completed_update_count"] == 32
            and v5_full["training_completed"] and v5_full["selected_source"] == "final_active_state"
            and v5_full["test_evaluation_count"] == 1 and not v5_full["test_before_training_complete"]
        ),
    }
    if not all(matched.values()):
        raise ValueError(f"matched-pair assertion failed: {matched}")

    deltas = {
        "team_vote_correct_count": v5_full["final_test"]["team_vote_correct_count"] - v4_full["final_test"]["team_vote_correct_count"],
        "mean_H": v5_full["final_test"]["mean_H"] - v4_full["final_test"]["mean_H"],
        "oracle_covered_but_vote_wrong_rate": (
            v5_full["final_test"]["oracle_covered_but_vote_wrong_rate"]
            - v4_full["final_test"]["oracle_covered_but_vote_wrong_rate"]
        ),
    }
    gate = {
        "v5_owner_context_alignment": alignment_pass_count == len(alignment) == 32,
        "v5_lower_mean_H": deltas["mean_H"] < 0,
        "v5_fewer_oracle_covered_vote_wrong": deltas["oracle_covered_but_vote_wrong_rate"] < 0,
        "v5_test_vote_not_lower": deltas["team_vote_correct_count"] >= 0,
    }
    result = {
        "artifact_schema_version": "v4_v5_matched_pair_seed45_v1",
        "v4_baseline": v4_baseline,
        "v4_full": v4_full,
        "v5_baseline": v5_baseline,
        "v5_full": v5_full,
        "matched_assertions": matched,
        "v5_owner_context_alignment": {
            "update_count": len(alignment), "passed_count": alignment_pass_count,
        },
        "deltas_v5_minus_v4": deltas,
        "predeclared_mechanism_gate": gate,
        "stable_method_evidence_gate_passed": all(gate.values()),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "pair_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        "# v4-v5 Seed-45 Matched Pair\n\n"
        "This is one clean-tree, fresh-cache, matched seed comparison. It is development evidence, "
        "not a generalization claim.\n\n"
        "## Protocol\n\n"
        "- Both Full runs used seed 45, identical split hashes, initial prompt hashes, GPT-4o-mini roles, "
        "a common persistent solver cache, 32 planned updates, final-active-state selection, and one post-training test.\n"
        f"- v4 source: `{v4_full['git_commit']}`; v5 source: `{v5_full['git_commit']}`.\n\n"
        "## Predeclared mechanism checks\n\n"
        f"- v5 owner/context alignment: `{gate['v5_owner_context_alignment']}` ({alignment_pass_count}/{len(alignment)}).\n"
        f"- v5 lower mean H: `{gate['v5_lower_mean_H']}`; delta `{deltas['mean_H']}`.\n"
        f"- v5 fewer oracle-covered vote-wrong cases: `{gate['v5_fewer_oracle_covered_vote_wrong']}`; "
        f"delta `{deltas['oracle_covered_but_vote_wrong_rate']}`.\n"
        f"- v5 test vote not lower: `{gate['v5_test_vote_not_lower']}`; "
        f"delta `{deltas['team_vote_correct_count']}`.\n\n"
        f"All four conditions passed: `{result['stable_method_evidence_gate_passed']}`. "
        "The failed oracle-covered condition means this seed does not yet support a stable "
        "team-conversion improvement claim.\n\n"
        "Only aggregate counts, hashes, versions, and numeric behavior metrics are published. "
        "Prompts, examples, answers, responses, cache locations, checkpoints, and absolute paths are excluded.\n",
        encoding="utf-8",
    )
    manifest = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in args.out_dir.iterdir() if file.is_file()
    }
    (args.out_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scan(args.out_dir)
    print(json.dumps({"ok": True, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
