from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generic_m20_probe_support import evaluation_system
from m2f_probe_support import SOURCE_EXECUTION_COMMIT, SOURCE_HASHES, read_cached_answers
from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.responsibility import compute_member_aware_repair_opportunity

AUDIT_VERSION = "v16_m2f_critical_competence_exchange_audit_v1"
EXPECTED_M2F_EXECUTION = "f8ba9098bc781e9ee9b0171504a9b02f7e80f61e"


def candidate_states(system: Any, target: int, answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    states, _, _ = system.current_states_and_opportunities()
    result = {}
    for index, parent in enumerate(states):
        observation = answers[parent.question_hash]
        team_answers = list(parent.team_answers)
        team_validity = list(parent.team_validity)
        team_answers[target] = str(observation.get("answer", ""))
        team_validity[target] = bool(observation.get("valid"))
        result[parent.question_hash] = build_team_vote_state(
            question_hash=parent.question_hash,
            gold_answer=parent.gold_answer,
            answers=team_answers,
            valid_vector=team_validity,
            normalize_answer=system.normalize_answer,
            match_answer=system.match_answer,
            tie_break=system.protocol.tie_policy,
            seed=system.cfg.training.seed,
        )
    return result


def exclusive_critical_role(state: Any, target: int) -> str:
    if not state.team_correctness[target]:
        return "none"
    opportunity = compute_member_aware_repair_opportunity(
        team_state=state,
        peer_context=build_peer_vote_context(state, target),
    )
    # Unique coverage is the stronger, mutually exclusive role. This matches
    # the frozen collateral audit's unique-before-pivotal loss hierarchy.
    if opportunity.unique_correct:
        return "unique"
    if opportunity.pivotal_correct:
        return "pivotal"
    return "noncritical_correct"


def exchange_for_candidate(
    *, parent_states: dict[str, Any], candidate_states_by_hash: dict[str, Any],
    target: int, responsibility: set[str],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    rows = []
    for question_hash in sorted(parent_states):
        parent = parent_states[question_hash]
        candidate = candidate_states_by_hash[question_hash]
        parent_role = exclusive_critical_role(parent, target)
        candidate_role = exclusive_critical_role(candidate, target)
        parent_correct = bool(parent.team_correctness[target])
        candidate_correct = bool(candidate.team_correctness[target])
        gained = not parent_correct and candidate_correct
        lost = parent_correct and not candidate_correct
        old_critical_lost = lost and parent_role in {"unique", "pivotal"}
        new_unique = gained and candidate_role == "unique"
        new_pivotal = gained and candidate_role == "pivotal"
        parent_oracle = any(parent.team_correctness)
        candidate_oracle = any(candidate.team_correctness)
        vote_gain = not parent.vote_correct and candidate.vote_correct
        vote_loss = parent.vote_correct and not candidate.vote_correct
        responsibility_repair = question_hash in responsibility and gained

        counts[f"old_{parent_role}_lost"] += int(old_critical_lost)
        counts["new_unique_gained"] += int(new_unique)
        counts["new_pivotal_gained"] += int(new_pivotal)
        counts["critical_gain"] += int(new_unique or new_pivotal)
        counts["critical_loss"] += int(old_critical_lost)
        counts["oracle_gain"] += int(not parent_oracle and candidate_oracle)
        counts["oracle_loss"] += int(parent_oracle and not candidate_oracle)
        counts["vote_gain"] += int(vote_gain)
        counts["vote_loss"] += int(vote_loss)
        counts["responsibility_repair"] += int(responsibility_repair)
        if responsibility_repair:
            counts[f"responsibility_to_{candidate_role}"] += 1
            counts["responsibility_to_vote_conversion"] += int(vote_gain)
            counts["responsibility_to_oracle_gain"] += int(not parent_oracle and candidate_oracle)
        if any((old_critical_lost, new_unique, new_pivotal, vote_gain, vote_loss, responsibility_repair)):
            rows.append({
                "question_hash": question_hash,
                "parent_role": parent_role,
                "candidate_role": candidate_role,
                "old_critical_lost": old_critical_lost,
                "new_unique_gained": new_unique,
                "new_pivotal_gained": new_pivotal,
                "oracle_gain": not parent_oracle and candidate_oracle,
                "oracle_loss": parent_oracle and not candidate_oracle,
                "vote_gain": vote_gain,
                "vote_loss": vote_loss,
                "responsibility_repair": responsibility_repair,
            })
    values = dict(counts)
    values["critical_net"] = values.get("critical_gain", 0) - values.get("critical_loss", 0)
    values["oracle_delta"] = values.get("oracle_gain", 0) - values.get("oracle_loss", 0)
    values["vote_net"] = values.get("vote_gain", 0) - values.get("vote_loss", 0)
    return values, rows


def audit(registry: dict[str, Any], run: dict[str, Any], source_cache: Path, repair_cache: Path):
    if run.get("execution_commit") != EXPECTED_M2F_EXECUTION or run.get("cell_count") != 7:
        raise ValueError("M2F execution provenance mismatch")
    if run.get("validation_calls") or run.get("test_calls"):
        raise ValueError("M2F run is not train-only")
    cells = run["cells"]
    if tuple(row["source_candidate_hash"] for row in cells) != SOURCE_HASHES:
        raise ValueError("source candidate identity mismatch")
    cases = {row["source_candidate_hash"]: row for row in registry["cases"]}
    pair_rows = []
    event_rows = []
    aggregate = Counter()
    for cell in cells:
        source_hash = cell["source_candidate_hash"]
        case = cases[source_hash]
        target = int(case["target_agent_id"])
        system = evaluation_system(case, out_dir=ROOT / "runs/m2f_critical_exchange_read_only", cache_path=repair_cache)
        parent_states_list, _, _ = system.current_states_and_opportunities()
        parent_states = {row.question_hash: row for row in parent_states_list}
        source_answers = read_cached_answers(source_cache, source_hash, system)
        repair_answers = read_cached_answers(repair_cache, cell["repair_candidate_hash"], system)
        if set(source_answers) != set(parent_states) or set(repair_answers) != set(parent_states):
            raise ValueError("candidate rollout does not cover exact train probe")
        responsibility = set(case["assigned_question_hashes"])
        source_metrics, source_events = exchange_for_candidate(
            parent_states=parent_states,
            candidate_states_by_hash=candidate_states(system, target, source_answers),
            target=target, responsibility=responsibility,
        )
        repair_metrics, repair_events = exchange_for_candidate(
            parent_states=parent_states,
            candidate_states_by_hash=candidate_states(system, target, repair_answers),
            target=target, responsibility=responsibility,
        )
        if source_metrics.get("old_pivotal_lost", 0) != len(cell["source_loss_hashes"]["pivotal"]):
            raise AssertionError("source pivotal loss mismatch")
        if repair_metrics.get("old_pivotal_lost", 0) != cell["repair_metrics"]["pivotal_loss_count"]:
            raise AssertionError("repair pivotal loss mismatch")
        row = {
            "case_id": cell["case_id"], "seed": cell["seed"],
            "target_agent_id": target, "source_candidate_hash": source_hash,
            "repair_candidate_hash": cell["repair_candidate_hash"],
        }
        for prefix, metrics in (("source", source_metrics), ("repair", repair_metrics)):
            for key in (
                "old_pivotal_lost", "old_unique_lost", "new_pivotal_gained",
                "new_unique_gained", "critical_gain", "critical_loss", "critical_net",
                "oracle_gain", "oracle_loss", "oracle_delta", "vote_gain", "vote_loss",
                "vote_net", "responsibility_repair", "responsibility_to_pivotal",
                "responsibility_to_unique", "responsibility_to_noncritical_correct",
                "responsibility_to_vote_conversion", "responsibility_to_oracle_gain",
            ):
                row[f"{prefix}_{key}"] = int(metrics.get(key, 0))
                aggregate[f"{prefix}_{key}"] += int(metrics.get(key, 0))
        pair_rows.append(row)
        for variant, events in (("SOURCE_M20", source_events), ("REPAIRED_M2F", repair_events)):
            event_rows.extend({"case_id": cell["case_id"], "candidate_variant": variant, **event} for event in events)
    summary = {
        "audit_version": AUDIT_VERSION, "status": "PASS", "pair_count": 7,
        "source_execution_commit": SOURCE_EXECUTION_COMMIT,
        "m2f_execution_commit": EXPECTED_M2F_EXECUTION,
        **dict(sorted(aggregate.items())),
        "source_critical_compensation_ratio": aggregate["source_critical_gain"] / max(1, aggregate["source_critical_loss"]),
        "repair_critical_compensation_ratio": aggregate["repair_critical_gain"] / max(1, aggregate["repair_critical_loss"]),
        "api_calls": 0, "model_calls": 0, "validation_calls": 0, "test_calls": 0,
        "role_policy": "exclusive_unique_then_pivotal_v1",
    }
    return pair_rows, event_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("fresh repo-local output required")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    run = json.loads((args.run_root / "probe_summary.json").read_text(encoding="utf-8"))
    pair_rows, event_rows, summary = audit(
        registry, run, args.source_cache, args.run_root / "shared_solver_cache.sqlite"
    )
    args.out.mkdir(parents=True)
    write_csv(args.out / "critical_exchange_pairs.csv", pair_rows)
    write_csv(args.out / "critical_exchange_events.csv", event_rows)
    (args.out / "critical_exchange_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
