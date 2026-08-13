from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from scripts.v17_formal_support import (
    ARMS, CALL_CEILING, EXECUTION_ORDER, SEEDS, TOKEN_CEILING,
    git, read_json, sha256_file, split_freeze, write_json,
)


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def train_audit(root: Path, freeze: dict) -> dict:
    blockers: list[str] = []
    cells = []
    initial: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    for seed in SEEDS:
        expected_order = list(EXECUTION_ORDER[seed])
        observed = []
        for arm in expected_order:
            setting = ARMS[arm]
            run = root / f"seed{seed}" / "disambiguation_qa" / f"{setting}_seed{seed}"
            observed.append(arm)
            required = (
                "run_meta.json", "final_summary.json", "cost_summary.json",
                "training_checkpoint.json", "best_prompts.json",
                "frozen_initialization_match.json", "comparison_cache_match.json",
            )
            if any(not (run / name).is_file() for name in required):
                blockers.append(f"missing_train_cell:{seed}:{arm}")
                continue
            meta = read_json(run / "run_meta.json")
            final = read_json(run / "final_summary.json")
            cost = read_json(run / "cost_summary.json")
            checkpoint = read_json(run / "training_checkpoint.json")
            init = read_json(run / "frozen_initialization_match.json")
            cache = read_json(run / "comparison_cache_match.json")
            selection = final.get("selection_summary", {})
            if meta.get("run_identity", {}).get("git_commit") != freeze.get("git_head"):
                blockers.append(f"source_identity:{seed}:{arm}")
            if meta.get("canonical_experiment_setting") != setting:
                blockers.append(f"setting:{seed}:{arm}")
            if not init.get("matched"):
                blockers.append(f"initialization:{seed}:{arm}")
            initial[seed].add(json.dumps(
                init.get("initialization_snapshot", {}).get(
                    "initial_prompt_hashes", []
                ), separators=(",", ":")
            ))
            expected_updates = 0 if arm == "S0" else 8
            if int(meta.get("completed_update_count", -1)) != expected_updates:
                blockers.append(f"updates:{seed}:{arm}")
            if (
                bool(selection.get("validation_used"))
                or int(selection.get("validation_evaluation_count", -1)) != 0
                or int(selection.get("test_evaluation_count", -1)) != 0
                or bool(meta.get("final_test_enabled"))
            ):
                blockers.append(f"train_isolation:{seed}:{arm}")
            if (
                int(cost.get("provider_call_budget", -1)) != CALL_CEILING
                or int(cost.get("total_token_budget", -1)) != TOKEN_CEILING
                or cost.get("provider_call_budget_exhausted")
                or cost.get("total_token_budget_exhausted")
            ):
                blockers.append(f"budget:{seed}:{arm}")
            if cache.get("immutable_comparison_cache") is not True or int(cache.get("new_entries_merged", -1)) != 0:
                blockers.append(f"cache_isolation:{seed}:{arm}")
            generic = jsonl(run / "loss_blind_generic_revision_events.jsonl")
            repair = jsonl(run / "online_compatibility_repair_events.jsonl")
            dynamics = jsonl(run / "training_dynamics.jsonl")
            if arm in {"S1", "S2"}:
                if repair or any(
                    row.get("responsibility_evidence_exposed")
                    or row.get("candidate_specific_loss_evidence_exposed")
                    for row in generic
                ):
                    blockers.append(f"generic_leakage:{seed}:{arm}")
            elif generic:
                blockers.append(f"unexpected_generic:{seed}:{arm}")
            if arm == "S3" and repair:
                blockers.append(f"unexpected_repair:{seed}")
            if not dynamics:
                blockers.append(f"training_dynamics:{seed}:{arm}")
                continue
            start_counts = [int(value) for value in dynamics[0]["per_agent_correct_counts"]]
            end_counts = [int(value) for value in dynamics[-1]["per_agent_correct_counts"]]
            gains = [right - left for left, right in zip(start_counts, end_counts, strict=True)]
            cells.append({
                "seed": seed, "arm": arm, "setting": setting,
                "final_state_hash": selection.get("selected_team_prompt_state_hash"),
                "checkpoint_sha256": sha256_file(run / "training_checkpoint.json"),
                "completed_updates": int(meta.get("completed_update_count", -1)),
                "final_train_vote_accuracy": float(dynamics[-1]["team_vote_accuracy"]),
                "final_train_vote_correct_count": int(dynamics[-1]["team_vote_correct_count"]),
                "g_min": min(gains), "g_sum": sum(gains),
                "accepted_updates": int(dynamics[-1]["accepted_update_count_so_far"]),
                "provider_calls": int(cost.get("successful_llm_calls", 0)),
                "prompt_tokens": int(cost.get("prompt_tokens", 0)),
                "completion_tokens": int(cost.get("completion_tokens", 0)),
                "total_tokens": int(cost.get("total_tokens", 0)),
                "generic_revision_attempted": sum(bool(row.get("revision_attempted")) for row in generic),
                "generic_revision_feasible": sum(bool(row.get("revision_feasible")) for row in generic),
                "generic_revision_committed": sum(bool(row.get("revision_committed")) for row in generic),
                "repair_eligible": sum(bool(row.get("repair_eligible")) for row in repair),
                "repair_attempted": sum(bool(row.get("repair_attempted")) for row in repair),
                "repair_valid": sum(bool(row.get("repair_valid")) for row in repair),
                "repair_feasible": sum(bool(row.get("repair_feasible")) for row in repair),
                "repair_committed": sum(bool(row.get("repair_committed")) for row in repair),
                "training_trajectory": [{
                    "update_index": int(row.get("update_index", -1)),
                    "team_state_hash": row.get("team_prompt_state_hash", ""),
                    "selected_target_ids": row.get("selected_target_ids", []),
                    "train_vote_accuracy": float(row["team_vote_accuracy"]),
                    "g_min": min(
                        int(value) - start_counts[index]
                        for index, value in enumerate(row["per_agent_correct_counts"])
                    ),
                    "g_sum": sum(
                        int(value) - start_counts[index]
                        for index, value in enumerate(row["per_agent_correct_counts"])
                    ),
                    "accepted_update_count": int(row["accepted_update_count_so_far"]),
                } for row in dynamics],
            })
        if observed != expected_order:
            blockers.append(f"order:{seed}")
        if len(initial[seed]) != 1:
            blockers.append(f"initial_hash_mismatch:{seed}")
    if len(cells) != 15:
        blockers.append("train_cell_inventory")
    return {
        "gate": "PASS" if not blockers else "FAIL",
        "phase": "train", "blockers": sorted(set(blockers)),
        "execution_commit": freeze.get("git_head"), "cells": cells,
        "validation_evaluations": 0, "test_evaluations": 0,
    }


def evaluation_audit(root: Path, freeze: dict, phase: str) -> dict:
    blockers: list[str] = []
    rows = []
    if read_json(root / "train_protocol_gate.json").get("gate") != "PASS":
        blockers.append("train_gate")
    for seed in SEEDS:
        for arm in EXECUTION_ORDER[seed]:
            path = root / phase / f"seed{seed}" / arm / "evaluation_summary_private.json"
            if not path.is_file():
                blockers.append(f"missing_{phase}:{seed}:{arm}")
                continue
            row = read_json(path)
            expected_count = 50 if phase == "validation" else 125
            if (
                row.get("phase") != phase
                or int(row.get("logical_evaluation_count", -1)) != 1
                or int(row.get("row_count", -1)) != expected_count
                or row.get("state_mutation")
                or row.get("selection_change")
                or row.get("checkpoint_mutation")
            ):
                blockers.append(f"{phase}_contract:{seed}:{arm}")
            rows.append(row)
    if len(rows) != 15:
        blockers.append(f"{phase}_inventory")
    if phase == "validation":
        test_calls = 0
    else:
        seal = read_json(root / "pre_test_seal.json")
        if seal.get("gate") != "PASS":
            blockers.append("pre_test_seal")
        test_calls = len(rows)
    return {
        "gate": "PASS" if not blockers else "FAIL",
        "phase": phase, "blockers": sorted(set(blockers)),
        "execution_commit": freeze.get("git_head"),
        "logical_evaluation_count": len(rows),
        "test_evaluations": test_calls,
        "state_mutations": 0 if not blockers else None,
        "selection_changes": 0 if not blockers else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = read_json(args.freeze)
    blockers = []
    if freeze.get("source_freeze_status") != "PASS": blockers.append("source_freeze")
    if git("rev-parse", "HEAD") != freeze.get("git_head"): blockers.append("git_head")
    if split_freeze()["gate"] != "PASS": blockers.append("dataset_freeze")
    result = train_audit(args.run_root, freeze) if args.phase == "train" else evaluation_audit(args.run_root, freeze, args.phase)
    if blockers:
        result["blockers"] = sorted(set(result["blockers"] + blockers))
        result["gate"] = "FAIL"
    if args.out.exists():
        raise FileExistsError("fresh protocol gate output required")
    write_json(args.out, result)
    print(json.dumps({"gate": result["gate"], "phase": args.phase, "blockers": result["blockers"]}, indent=2))
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
