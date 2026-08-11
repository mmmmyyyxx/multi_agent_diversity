from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample
from multi_dataset_diverse_rl.module2_context import build_module2_context_sets
from multi_dataset_diverse_rl.peer_state import build_team_vote_state


DESIGN_ROOT = (
    ROOT
    / "experiments"
    / "v16_module2_candidate_design_fixed_trajectory_isolation_20260811"
)
REPLAY_SOURCE = (
    ROOT
    / "experiments"
    / "v15_bottleneck_isolation_offline_audit_seed48_50_20260811"
    / "scripts"
    / "audit_bottleneck_isolation.py"
)


def _load_replay() -> Any:
    spec = importlib.util.spec_from_file_location("v16_historical_replay", REPLAY_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen historical replay utility")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _csv_groups(name: str) -> dict[tuple[int, int, int], list[dict[str, str]]]:
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    with (DESIGN_ROOT / "tables" / name).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(int(row["seed"]), int(row["update_index"]), int(row["target_agent_id"]))].append(row)
    return groups


def _team(record: dict[str, Any], seed: int):
    return build_team_vote_state(
        question_hash=record["question_hash"],
        gold_answer=record["gold_answer"],
        answers=record["answers"],
        valid_vector=record["validity"],
        normalize_answer=lambda value: str(value or "").strip(),
        match_answer=lambda prediction, gold: prediction == gold,
        tie_break="abstain",
        seed=seed,
    )


def replay() -> dict[str, Any]:
    module = _load_replay()
    expected_repair = _csv_groups("repair_set_by_state.csv")
    expected_preservation = _csv_groups("preservation_set_by_state.csv")
    consumed: set[Path] = set()
    counters = {
        "historical_branch_replay_count": 0,
        "repair_membership_mismatch": 0,
        "preservation_membership_mismatch": 0,
        "tier_mismatch": 0,
        "repair_distance_mismatch": 0,
        "c2_c3_membership_mismatch": 0,
        "vote_correct_repair_item_count": 0,
    }
    for seed in (48, 49, 50):
        data = module.load_run(seed, consumed)
        for decision in data["decisions"]:
            update = int(decision["update_index"])
            rank = module.active_state_rank(data["accepted"], update)
            current_records = data["states"][rank]
            states = tuple(_team(current_records[qid], seed) for qid in data["qids"])
            examples = tuple(
                ProbeExample(
                    question=qid,
                    question_hash=qid,
                    gold_answer=current_records[qid]["gold_answer"],
                )
                for qid in data["qids"]
            )
            for branch in decision.get("branches", []):
                target = int(branch["target_agent_id"])
                stable = {
                    qid
                    for qid in data["qids"]
                    if all(
                        data["states"][prior][qid]["correctness"][target]
                        for prior in range(rank + 1)
                    )
                }
                sets = build_module2_context_sets(
                    examples=examples,
                    states=states,
                    target_agent_id=target,
                    assigned_question_hashes=set(map(str, branch.get("assigned_question_hashes", []))),
                    stable_correct_question_hashes=stable,
                    accepted_state_count=rank + 1,
                    normalize_answer=lambda value: str(value or "").strip(),
                    match_answer=lambda prediction, gold: prediction == gold,
                    tie_break="abstain",
                    seed=seed,
                )
                key = (seed, update, target)
                expected_r = expected_repair.get(key, [])
                expected_p = expected_preservation.get(key, [])
                actual_r_hashes = [row.question_hash for row in sets.repair]
                actual_p_hashes = [row.question_hash for row in sets.preservation]
                if actual_r_hashes != [row["question_hash"] for row in expected_r]:
                    counters["repair_membership_mismatch"] += 1
                if actual_p_hashes != [row["question_hash"] for row in expected_p]:
                    counters["preservation_membership_mismatch"] += 1
                actual_tiers = [row.tier for row in sets.repair] + [row.tier for row in sets.preservation]
                expected_tiers = [row["tier"] for row in expected_r] + [row["tier"] for row in expected_p]
                if actual_tiers != expected_tiers:
                    counters["tier_mismatch"] += 1
                if [row.repair_distance for row in sets.repair] != [int(row["repair_distance"]) for row in expected_r]:
                    counters["repair_distance_mismatch"] += 1
                counters["vote_correct_repair_item_count"] += sum(
                    states[data["qids"].index(row.question_hash)].vote_correct
                    for row in sets.repair
                )
                counters["historical_branch_replay_count"] += 1
    passed = (
        counters["historical_branch_replay_count"] == 192
        and all(
            counters[name] == 0
            for name in counters
            if name != "historical_branch_replay_count"
        )
    )
    return {
        "replay_version": "experimental_v16_module2_historical_parity_v1",
        **counters,
        "c0_context_compatibility": "PASS",
        "status": "PASS" if passed else "FAIL",
        "api_calls": 0,
        "test_calls": 0,
        "validation_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = replay()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
