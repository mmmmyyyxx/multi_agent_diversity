"""Read-only reconstruction of a frozen V4 GEPA boundary; no provider imports."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(attempt_root: Path, *, update: int = 3, member: int = 3) -> dict:
    frozen = json.loads((attempt_root / "raw_evidence_freeze.json").read_text(encoding="utf-8"))
    if frozen["execution_source_sha"] != "85812a7d891e6a2c3bfdca00a1cb4d14074735a4":
        raise ValueError("unexpected frozen execution source")
    if any(_sha(attempt_root / row["path"]) != row["sha256"] for row in frozen["files"]):
        raise ValueError("frozen attempt2 artifact changed")
    task_name = f"seed81_update{update}_member{member}_layer2_evidence"
    task_root = attempt_root / "run" / "local_gepa" / task_name
    with (task_root / "gepa_state.bin").open("rb") as handle:
        state = pickle.load(handle)  # Locally produced, SHA-pinned private artifact.
    materialized = state["program_candidates"]
    if materialized != json.loads((task_root / "candidates.json").read_text(encoding="utf-8")):
        raise ValueError("GEPA state/candidate artifact mismatch")
    lineage_path = task_root.parent / f"{task_name}.lineage.jsonl"
    events = [json.loads(line) for line in lineage_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    accepted = [row for row in events if row["event_type"] == "candidate_accepted"]
    proposals = [row for row in events if row["event_type"] == "proposal_end"]
    proposal_by_hash = {row["proposal_hash"]: row for row in proposals}
    log = json.loads((task_root / "run_log.json").read_text(encoding="utf-8"))
    log_by_index = {row["new_program_idx"]: row for row in log}
    hashes = [hashlib.sha256(row["decision_procedure"].encode("utf-8")).hexdigest()
              for row in materialized]
    frontier = sorted({int(index) for members in state["program_at_pareto_front_valset"].values()
                       for index in members})
    changed = [index for index in frontier if index != 0 and hashes[index] != hashes[0]]
    # Each accepted prompt was checked against the same frozen packet by the
    # GEPA callback and adapter. Match its exact hash before using that fact.
    valid_unique: list[int] = []
    seen: set[str] = set()
    for index in changed:
        proposal = proposal_by_hash.get(hashes[index])
        if proposal is None or proposal["contract_invalid"]:
            continue
        if hashes[index] in seen:
            continue
        seen.add(hashes[index])
        valid_unique.append(index)
    scores = [sum(row.values()) / len(row) for row in state["prog_candidate_val_subscores"]]
    returned = sorted(valid_unique, key=lambda index: (-scores[index], index))[:4]
    accepted_by_index = {row["candidate_index"]: row["iteration"] for row in accepted}
    returned_rows = []
    for index in returned:
        log_row = log_by_index.get(index)
        if log_row is None or index not in accepted_by_index:
            raise ValueError("returned candidate lacks accepted-event score evidence")
        returned_rows.append({
            "candidate_index": index,
            "candidate_id": f"gepa:{index}:{hashes[index][:12]}",
            "candidate_hash": hashes[index],
            "local_score": scores[index],
            "local_acceptance_delta": (
                sum(log_row["new_subsample_scores"]) - sum(log_row["subsample_scores"])
            ),
        })
    if len(accepted) != len({row["candidate_index"] for row in accepted}):
        raise ValueError("accepted GEPA indices are not unique")
    return {
        "schema_version": "v4_seed81_attempt2_opportunity_forensics_v1",
        "execution_source_sha": frozen["execution_source_sha"],
        "raw_evidence_freeze_sha256": _sha(attempt_root / "raw_evidence_freeze.json"),
        "update_index": update,
        "member_id": member,
        "gepa_proposal_attempts": len(proposals),
        "gepa_accepted_mutation_events": len(accepted),
        "accepted_candidate_indices": [row["candidate_index"] for row in accepted],
        "accepted_iterations": [row["iteration"] for row in accepted],
        "all_materialized_candidate_indices": list(range(len(materialized))),
        "pareto_frontier_candidate_indices": frontier,
        "changed_frontier_indices": changed,
        "valid_unique_frontier_indices": valid_unique,
        "returned_candidate_indices": returned,
        "local_optimization_result_candidate_count": len(returned),
        "returned_candidates": returned_rows,
        "accepted_mutations_equal_returned_count_in_this_run": len(accepted) == len(returned),
        "proposal_event_counts": dict(sorted(Counter(row["event_type"] for row in events).items())),
        "materialized_candidate_hashes": hashes,
        "materialized_candidate_local_scores": scores,
        "frontier_return_cap": 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = reconstruct(args.attempt_root)
    data = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
    print(data, end="")


if __name__ == "__main__":
    main()
