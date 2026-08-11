from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.peer_state import build_team_vote_state

OUT = Path(__file__).resolve().parents[1]
TABLES = OUT / "tables"
INPUT_REPORT = ROOT / "reports" / "v15_three_seed_development_formal_20260811"
SEEDS = (48, 49, 50)
SETTINGS = (
    "shared_generic_evolution",
    "shared_member_aware_dual_target",
    "shared_responsibility_conditioned_dual_target",
)
LABEL = {
    "shared_generic_evolution": "S0",
    "shared_member_aware_dual_target": "S1",
    "shared_responsibility_conditioned_dual_target": "S2",
}
N = 75


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        if names:
            writer.writeheader()
            writer.writerows(rows)


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.mean(rows) if rows else None


def median(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.median(rows) if rows else None


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def exact_state(row: dict[str, Any], seed: int) -> dict[str, Any]:
    state = build_team_vote_state(
        question_hash=str(row["question_hash"]),
        gold_answer=str(row["gold_answer"]),
        answers=[str(value) for value in row["team_answers"]],
        valid_vector=[bool(value) for value in row["team_validity"]],
        normalize_answer=lambda value: str(value or "").strip(),
        match_answer=lambda prediction, gold: prediction == gold,
        tie_break="abstain",
        seed=seed,
    )
    if (
        tuple(state.team_correctness) != tuple(bool(x) for x in row["team_correctness"])
        or state.vote_correct != bool(row["vote_correct"])
        or state.vote_answer != str(row["vote_answer"])
        or state.gold_vote_count != int(row["gold_vote_count"])
    ):
        raise AssertionError("repository plurality replay differs from recorded train state")
    wrong_counts = sorted((count for _, count in state.wrong_vote_histogram), reverse=True)
    return {
        "question_hash": state.question_hash,
        "gold_answer": state.gold_answer,
        "answers": tuple(state.team_answers),
        "validity": tuple(state.team_validity),
        "correctness": tuple(state.team_correctness),
        "correct_ids": tuple(i for i, value in enumerate(state.team_correctness) if value),
        "G": state.gold_vote_count,
        "vote_correct": state.vote_correct,
        "vote_answer": state.vote_answer,
        "top_tie": state.top_tie,
        "wrong_coalition_sizes": wrong_counts,
        "repair_distance": repair_distance(state, seed),
    }


def repair_distance(state: Any, seed: int) -> int:
    if state.vote_correct:
        return 0
    wrong = [i for i, correct in enumerate(state.team_correctness) if not correct]
    for count in range(1, len(wrong) + 1):
        for subset in itertools.combinations(wrong, count):
            answers = list(state.team_answers)
            validity = list(state.team_validity)
            for agent_id in subset:
                answers[agent_id] = state.gold_answer
                validity[agent_id] = True
            counterfactual = build_team_vote_state(
                question_hash=state.question_hash,
                gold_answer=state.gold_answer,
                answers=answers,
                valid_vector=validity,
                normalize_answer=lambda value: str(value or "").strip(),
                match_answer=lambda prediction, gold: prediction == gold,
                tie_break="abstain",
                seed=seed,
            )
            if counterfactual.vote_correct:
                return count
    raise AssertionError("repairing every wrong member must make the vote correct")


def flatten_context_hashes(row: dict[str, Any]) -> set[str]:
    value = row.get("selected_context_pattern_question_hashes", {})
    hashes: set[str] = set()
    if isinstance(value, dict):
        for items in value.values():
            if isinstance(items, list):
                hashes.update(str(item) for item in items)
    return hashes


def candidate_primary_failure(funnel: dict[str, Any]) -> str:
    if int(funnel.get("infrastructure_failed_updates", 0)):
        return "INFRASTRUCTURE"
    generation_failures = sum(int(funnel.get(key, 0)) for key in (
        "teacher_invalid_responses", "teacher_truncated_responses",
        "critic_invalid_responses", "critic_truncated_responses",
        "critic_semantic_rejections", "student_invalid_responses",
        "student_truncated_responses", "student_cycle_exhausted",
    ))
    if int(funnel.get("valid_candidate_count", 0)) == 0:
        return "GENERATION_OR_CRITIC" if generation_failures else "NO_VALID_CANDIDATE"
    if int(funnel.get("constraint_feasible", 0)) == 0:
        target_reg = int(funnel.get("rejected_target_regression", 0))
        vote_reg = int(funnel.get("rejected_team_vote_regression", 0))
        no_progress = int(funnel.get("rejected_no_target_or_vote_progress", 0))
        if target_reg or vote_reg:
            return "COMMON_SAFE_CONFLICT"
        if no_progress:
            return "NO_TARGET_OR_VOTE_PROGRESS"
        return "CANDIDATE_QUALITY_OR_FEASIBILITY"
    if int(funnel.get("acceptable_candidates", 0)) and not bool(funnel.get("accepted_candidate", False)):
        return "CROSS_BRANCH_COMPETITION"
    return "OTHER"


def update_label(row: dict[str, Any]) -> str:
    discovery = int(row["new_0_to_1_count"]) + int(row["new_0_to_2plus_count"])
    reinforce = int(row["reinforce_1_to_2_count"]) + int(row["reinforce_1_to_3plus_count"]) + int(row["reinforce_2_to_3plus_count"])
    conversion = int(row["vote_wrong_to_correct_count"])
    if discovery == reinforce == conversion == 0:
        return "NO_MEANINGFUL_COVERAGE_CHANGE"
    if conversion > 0 and discovery == 0 and reinforce == 0:
        return "CONSENSUS_CONVERSION"
    if discovery > reinforce + conversion and int(row["vote_net_gain"]) <= 0:
        return "DISCOVERY_DOMINANT"
    if reinforce > discovery and conversion == 0:
        return "REINFORCEMENT_DOMINANT"
    if conversion > 0 and conversion >= max(discovery, reinforce):
        return "CONSENSUS_CONVERSION"
    return "MIXED"


def load_run(seed: int, setting: str, consumed: set[Path]) -> dict[str, Any]:
    run = ROOT / "runs" / f"v15f{seed}" / "disambiguation_qa" / f"{setting}_seed{seed}"
    paths = {
        "peer": run / "peer_state_history.jsonl",
        "dynamics": run / "training_dynamics.jsonl",
        "decisions": run / "candidate_decisions.jsonl",
        "commits": run / "dual_target_commit_decisions.jsonl",
        "responsibility": run / "responsibility_assignments.jsonl",
        "scores": run / "repairability_adjusted_target_scores.jsonl",
        "contexts": run / "tcs_context_history.jsonl",
    }
    for path in paths.values():
        consumed.add(path)
    peer = read_jsonl(paths["peer"])
    dynamics = read_jsonl(paths["dynamics"])
    decisions = read_jsonl(paths["decisions"])
    commits = read_jsonl(paths["commits"])
    responsibility = read_jsonl(paths["responsibility"])
    scores = read_jsonl(paths["scores"])
    contexts = read_jsonl(paths["contexts"])
    accepted_decisions = [row for row in decisions if row.get("accepted_prompt_hash")]
    if len(dynamics) != 33 or len(decisions) != 32 or len(commits) != 32:
        raise AssertionError("expected fixed 32-update training artifacts")
    if sum(bool(row["accepted"]) for row in dynamics[1:]) != len(accepted_decisions):
        raise AssertionError("accepted update count differs between dynamics and decisions")
    raw_groups = [peer[index:index + N] for index in range(0, len(peer), N)]
    if setting == "shared_generic_evolution":
        # S0 persists one parent-state block per raw update.  The child of an
        # accepted update u is therefore the parent block at u + 1.  All three
        # audited runs have their last acceptance before update 31.
        if len(raw_groups) != 32 or any(int(row["update_index"]) >= 31 for row in accepted_decisions if row.get("accepted_prompt_hash")):
            raise AssertionError("S0 parent-state history cannot recover every accepted child")
        groups = [raw_groups[0], *[raw_groups[int(row["update_index"]) + 1] for row in accepted_decisions if row.get("accepted_prompt_hash")]]
    else:
        if len(raw_groups) != len(accepted_decisions) + 1:
            raise AssertionError("member-aware peer-state history is not initial + accepted states")
        groups = raw_groups
    question_order = [row["question_hash"] for row in groups[0]]
    states: list[dict[str, dict[str, Any]]] = []
    for group in groups:
        if [row["question_hash"] for row in group] != question_order:
            raise AssertionError("question order changed across accepted states")
        states.append({row["question_hash"]: exact_state(row, seed) for row in group})
    accepted_indices = [int(row["update_index"]) for row in accepted_decisions]
    accepted_targets = [int(row["target_agent_id"]) for row in accepted_decisions]
    state_raw_indices = [-1, *accepted_indices]
    if responsibility and len(responsibility) != len(states):
        raise AssertionError("responsibility snapshots do not align with accepted states")
    return {
        "run": run, "states": states, "question_order": question_order,
        "accepted_indices": accepted_indices, "accepted_targets": accepted_targets,
        "state_raw_indices": state_raw_indices, "dynamics": dynamics,
        "decisions": decisions, "responsibility": responsibility,
        "scores": scores, "contexts": contexts,
    }


def main() -> None:
    if OUT.exists() and any(path for path in OUT.iterdir() if path.name != "scripts"):
        raise FileExistsError("audit output must be fresh apart from its frozen script")
    TABLES.mkdir(parents=True, exist_ok=True)
    consumed: set[Path] = {
        INPUT_REPORT / "train_results_by_seed.csv",
        ROOT / "multi_dataset_diverse_rl" / "peer_state.py",
        ROOT / "multi_dataset_diverse_rl" / "responsibility.py",
        ROOT / "multi_dataset_diverse_rl" / "system.py",
    }
    train_reference: dict[tuple[int, str], dict[str, str]] = {}
    with (INPUT_REPORT / "train_results_by_seed.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] in SETTINGS:
                train_reference[(int(row["seed"]), row["setting"])] = row

    state_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    orphan_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    vote_conversion_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    orphan_survival_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    run_cache: dict[tuple[int, str], dict[str, Any]] = {}

    for seed in SEEDS:
        for setting in SETTINGS:
            label = LABEL[setting]
            data = load_run(seed, setting, consumed)
            run_cache[(seed, setting)] = data
            states = data["states"]
            qids = data["question_order"]
            accepted_indices = data["accepted_indices"]
            targets = data["accepted_targets"]
            dynamics = data["dynamics"]

            for state_rank, state in enumerate(states):
                histogram = Counter(value["G"] for value in state.values())
                oracle = sum(value["G"] > 0 for value in state.values())
                votes = sum(value["vote_correct"] for value in state.values())
                recorded = dynamics[0] if state_rank == 0 else dynamics[accepted_indices[state_rank - 1] + 1]
                checks = {
                    "g_histogram_sum": sum(histogram.values()) == N,
                    "oracle_match": oracle == int(recorded["oracle_correct_count"]),
                    "vote_match": votes == int(recorded["team_vote_correct_count"]),
                    "g_histogram_match": all(int(recorded["G_histogram"].get(str(g), recorded["G_histogram"].get(g, 0))) == histogram[g] for g in range(6)),
                }
                if not all(checks.values()):
                    raise AssertionError(f"state replay check failed: {seed} {label} {state_rank} {checks}")
                state_rows.append({
                    "seed": seed, "setting": label, "accepted_state_rank": state_rank,
                    "raw_update_index": data["state_raw_indices"][state_rank],
                    **{f"G{g}": histogram[g] for g in range(6)},
                    "oracle_correct_count": oracle, "vote_correct_count": votes,
                    "oracle_vote_gap": oracle - votes,
                    "fragmented_coverage_count": sum(value["G"] > 0 and not value["vote_correct"] for value in state.values()),
                    "singleton_wrong_count": sum(value["G"] == 1 and not value["vote_correct"] for value in state.values()),
                    "near_boundary_count": sum(value["repair_distance"] == 1 and not value["vote_correct"] for value in state.values()),
                    "repair_distance_2plus_count": sum(value["repair_distance"] >= 2 and not value["vote_correct"] for value in state.values()),
                })
                replay_checks.append({"seed": seed, "setting": label, "state_rank": state_rank, **checks})

            event_records: list[dict[str, Any]] = []
            for rank in range(1, len(states)):
                parent, child = states[rank - 1], states[rank]
                raw_update = accepted_indices[rank - 1]
                target = targets[rank - 1]
                for q in qids:
                    if abs(child[q]["G"] - parent[q]["G"]) > 1:
                        raise AssertionError("one-member commit changed G by more than one")
                    for agent_id in range(5):
                        if agent_id != target and (
                            child[q]["answers"][agent_id] != parent[q]["answers"][agent_id]
                            or child[q]["validity"][agent_id] != parent[q]["validity"][agent_id]
                        ):
                            raise AssertionError("accepted transition changed a non-target member")
                matrix = Counter((parent[q]["G"], child[q]["G"]) for q in qids)
                if sum(matrix[(0, g)] for g in range(2, 6)) or sum(matrix[(1, g)] for g in range(3, 6)):
                    raise AssertionError("one-member commit produced an impossible multi-step G transition")
                for gp in range(6):
                    for gc in range(6):
                        transition_rows.append({"seed": seed, "setting": label, "update_index": raw_update, "accepted_update_rank": rank, "target_agent_id": target, "g_parent": gp, "g_child": gc, "transition_count": matrix[(gp, gc)]})
                wrong_to_correct = [q for q in qids if not parent[q]["vote_correct"] and child[q]["vote_correct"]]
                correct_to_wrong = [q for q in qids if parent[q]["vote_correct"] and not child[q]["vote_correct"]]
                newly_correct = {q: sorted(set(child[q]["correct_ids"]) - set(parent[q]["correct_ids"])) for q in qids}
                for q in wrong_to_correct:
                    vote_conversion_rows.append({
                        "seed": seed, "setting": label, "update_index": raw_update,
                        "accepted_update_rank": rank, "example_id": q,
                        "parent_G": parent[q]["G"], "child_G": child[q]["G"],
                        "correct_coalition_size": child[q]["G"],
                        "parent_wrong_coalition_sizes": compact_json(parent[q]["wrong_coalition_sizes"]),
                        "child_wrong_coalition_sizes": compact_json(child[q]["wrong_coalition_sizes"]),
                        "target_agent_id": target,
                        "target_agent_newly_correct": target in newly_correct[q],
                        "repair_distance_parent": parent[q]["repair_distance"],
                        "repair_distance_child": child[q]["repair_distance"],
                    })
                new_orphans = [q for q in qids if parent[q]["G"] == 0 and child[q]["G"] == 1 and not child[q]["vote_correct"]]
                for q in new_orphans:
                    event_records.append({"question_hash": q, "creation_rank": rank, "creation_update": raw_update, "creator": target})
                row = {
                    "seed": seed, "setting": label, "update_index": raw_update,
                    "accepted_update_rank": rank, "target_agent_id": target,
                    "target_gain": sum(child[q]["correctness"][target] for q in qids) - sum(parent[q]["correctness"][target] for q in qids),
                    "team_vote_gain_count": len(wrong_to_correct), "team_vote_loss_count": len(correct_to_wrong),
                    "vote_net_gain": len(wrong_to_correct) - len(correct_to_wrong),
                    "oracle_gain": sum(child[q]["G"] > 0 for q in qids) - sum(parent[q]["G"] > 0 for q in qids),
                    "mean_member_gain": sum(child[q]["G"] - parent[q]["G"] for q in qids) / 5.0,
                    "new_0_to_1_count": matrix[(0, 1)],
                    "new_0_to_2plus_count": sum(matrix[(0, g)] for g in range(2, 6)),
                    "reinforce_1_to_2_count": matrix[(1, 2)],
                    "reinforce_1_to_3plus_count": sum(matrix[(1, g)] for g in range(3, 6)),
                    "reinforce_2_to_3plus_count": sum(matrix[(2, g)] for g in range(3, 6)),
                    "vote_wrong_to_correct_count": len(wrong_to_correct),
                    "vote_correct_to_wrong_count": len(correct_to_wrong),
                    "new_orphan_count": len(new_orphans),
                    "orphan_rescued_count": sum(parent[q]["G"] == 1 and not parent[q]["vote_correct"] and child[q]["vote_correct"] for q in qids),
                    "persistent_orphan_candidates_created": 0,
                }
                decomposition_rows.append(row)

            # Orphan event fate and exact revisit.
            contexts_by_update_target = defaultdict(set)
            for context in data["contexts"]:
                contexts_by_update_target[(int(context["update_index"]), int(context["target_agent_id"]))].update(flatten_context_hashes(context))
            for event_index, event in enumerate(event_records):
                q = event["question_hash"]; creation_rank = event["creation_rank"]; creator = event["creator"]
                later_all = range(creation_rank + 1, len(states))
                lost_rank = next((rank for rank in later_all if states[rank][q]["G"] == 0), None)
                episode_stop = len(states) if lost_rank is None else lost_rank
                later = range(creation_rank + 1, episode_stop)
                first_reinforcement = next((rank for rank in later if states[rank][q]["G"] >= 2), None)
                first_conversion = next((rank for rank in later if states[rank][q]["vote_correct"]), None)
                reinforcing_ids = [] if first_reinforcement is None else sorted(set(states[first_reinforcement][q]["correct_ids"]) - set(states[first_reinforcement - 1][q]["correct_ids"]))
                final = states[-1][q]
                if first_conversion is not None:
                    fate = "rescued_to_vote"
                elif lost_rank is not None:
                    fate = "coverage_lost_again"
                elif final["G"] == 1 and not final["vote_correct"]:
                    fate = "singleton_until_end"
                elif final["G"] >= 2 and not final["vote_correct"]:
                    fate = "reinforced_but_still_vote_wrong"
                else:
                    fate = "other"
                event_id = f"{seed}-{label}-{event_index}"
                orphan = {
                    "event_id": event_id, "seed": seed, "setting": label, "example_id": q,
                    "creation_update": event["creation_update"], "creation_accepted_rank": creation_rank,
                    "creating_target_agent": creator, "creating_agent_id": creator,
                    "G_before": states[creation_rank - 1][q]["G"], "G_after": states[creation_rank][q]["G"],
                    "vote_before": states[creation_rank - 1][q]["vote_correct"], "vote_after": states[creation_rank][q]["vote_correct"],
                    "repair_distance_after": states[creation_rank][q]["repair_distance"],
                    "first_reinforcement_update": "" if first_reinforcement is None else data["state_raw_indices"][first_reinforcement],
                    "latency_to_reinforcement": "" if first_reinforcement is None else first_reinforcement - creation_rank,
                    "reinforcing_agent_ids": compact_json(reinforcing_ids),
                    "first_vote_conversion_update": "" if first_conversion is None else data["state_raw_indices"][first_conversion],
                    "latency_to_vote_conversion": "" if first_conversion is None else first_conversion - creation_rank,
                    "final_G": final["G"], "final_vote_correct": final["vote_correct"],
                    "final_repair_distance": final["repair_distance"], "fate": fate,
                    "coverage_lost_after_creation": lost_rank is not None,
                }
                orphan_rows.append(orphan)
                if first_reinforcement is not None or first_conversion is not None:
                    rescue_rows.append({
                        "event_id": event_id, "seed": seed, "setting": label, "example_id": q,
                        "first_reinforcement_update": orphan["first_reinforcement_update"],
                        "latency_to_reinforcement": orphan["latency_to_reinforcement"],
                        "reinforcing_agent_ids": orphan["reinforcing_agent_ids"],
                        "first_vote_conversion_update": orphan["first_vote_conversion_update"],
                        "latency_to_vote_conversion": orphan["latency_to_vote_conversion"],
                        "final_vote_correct": final["vote_correct"],
                    })
                # Exact branch assignment revisit after creation.
                revisit_candidates = []
                terminal_raw_update = min(
                    [data["state_raw_indices"][rank] for rank in (first_conversion, lost_rank) if rank is not None],
                    default=None,
                )
                for decision in data["decisions"]:
                    update = int(decision["update_index"])
                    if update <= int(event["creation_update"]):
                        continue
                    if terminal_raw_update is not None and update > terminal_raw_update:
                        continue
                    for branch in decision.get("branches", []):
                        target = int(branch["target_agent_id"])
                        if target != creator and q in set(branch.get("assigned_question_hashes", [])):
                            revisit_candidates.append((update, target, q in contexts_by_update_target[(update, target)]))
                first_revisit = min(revisit_candidates, default=None)
                routing_rows.append({
                    "event_id": event_id, "seed": seed, "setting": label, "example_id": q,
                    "creating_agent_id": creator,
                    "ever_revisited_by_other_member": first_revisit is not None,
                    "first_revisit_update": "" if first_revisit is None else first_revisit[0],
                    "first_revisit_agent_id": "" if first_revisit is None else first_revisit[1],
                    "revisit_latency_raw_updates": "" if first_revisit is None else first_revisit[0] - int(event["creation_update"]),
                    "explicit_conditioned_context_revisit": (
                        any(item[2] for item in revisit_candidates) if label == "S2" else "not_applicable"
                    ),
                    "ever_vote_converted": first_conversion is not None,
                    "converted_after_revisit": bool(first_revisit is not None and first_conversion is not None and data["state_raw_indices"][first_conversion] >= first_revisit[0]),
                    "analysis_support": "EXACT_BRANCH_ASSIGNMENT_HASHES",
                })
                for lag in range(0, len(states) - creation_rank):
                    rank = creation_rank + lag
                    alive = states[rank][q]["G"] > 0 and not states[rank][q]["vote_correct"]
                    # Once conversion/loss occurred, the original orphan event is no longer alive.
                    if any(states[r][q]["vote_correct"] or states[r][q]["G"] == 0 for r in range(creation_rank, rank + 1)):
                        alive = False
                    orphan_survival_rows.append({"event_id": event_id, "seed": seed, "setting": label, "lag_accepted_updates": lag, "alive": alive})

            persistent_event_ids = {row["event_id"] for row in orphan_rows if row["seed"] == seed and row["setting"] == label and row["fate"] == "singleton_until_end"}
            for row in decomposition_rows:
                if row["seed"] == seed and row["setting"] == label:
                    row["persistent_orphan_candidates_created"] = sum(
                        orphan["event_id"] in persistent_event_ids and orphan["creation_update"] == row["update_index"]
                        for orphan in orphan_rows
                    )
                    row["primary_label"] = update_label(row)

            # Per-example summary.
            for q in qids:
                seq = [state[q] for state in states]
                first_discovery = next((rank for rank in range(1, len(seq)) if seq[rank - 1]["G"] == 0 and seq[rank]["G"] > 0), None)
                first_singleton = next((rank for rank in range(1, len(seq)) if seq[rank - 1]["G"] == 0 and seq[rank]["G"] == 1 and not seq[rank]["vote_correct"]), None)
                first_reinforcement = None if first_singleton is None else next((rank for rank in range(first_singleton + 1, len(seq)) if seq[rank]["G"] >= 2), None)
                first_conversion = next((rank for rank in range(1, len(seq)) if not seq[rank - 1]["vote_correct"] and seq[rank]["vote_correct"]), None)
                new_ids = [] if first_discovery is None else sorted(set(seq[first_discovery]["correct_ids"]) - set(seq[first_discovery - 1]["correct_ids"]))
                reinforce_ids = [] if first_reinforcement is None else sorted(set(seq[first_reinforcement]["correct_ids"]) - set(seq[first_reinforcement - 1]["correct_ids"]))
                ever_orphan = first_singleton is not None
                per_example_rows.append({
                    "seed": seed, "setting": label, "example_id": q,
                    "initial_G": seq[0]["G"], "final_G": seq[-1]["G"],
                    "initial_vote_correct": seq[0]["vote_correct"], "final_vote_correct": seq[-1]["vote_correct"],
                    "first_discovery_update": "" if first_discovery is None else data["state_raw_indices"][first_discovery],
                    "first_discovering_agent": compact_json(new_ids),
                    "first_singleton_update": "" if first_singleton is None else data["state_raw_indices"][first_singleton],
                    "first_reinforcement_update": "" if first_reinforcement is None else data["state_raw_indices"][first_reinforcement],
                    "first_reinforcing_agent": compact_json(reinforce_ids),
                    "first_vote_conversion_update": "" if first_conversion is None else data["state_raw_indices"][first_conversion],
                    "initial_repair_distance": seq[0]["repair_distance"],
                    "min_repair_distance": min(value["repair_distance"] for value in seq),
                    "final_repair_distance": seq[-1]["repair_distance"],
                    "ever_orphan": ever_orphan, "ever_reinforced": first_reinforcement is not None,
                    "ever_vote_converted": first_conversion is not None,
                    "persistent_orphan": ever_orphan and seq[-1]["G"] == 1 and not seq[-1]["vote_correct"],
                    "coverage_lost_after_discovery": bool(first_discovery is not None and any(value["G"] == 0 for value in seq[first_discovery + 1:])),
                })

            # Funnel uses unique initially-uncovered examples.
            initial_uncovered = {q for q in qids if states[0][q]["G"] == 0}
            discovered = {q for q in initial_uncovered if any(state[q]["G"] > 0 for state in states[1:])}
            orphan_discovered = {q for q in discovered if any(state[q]["G"] == 1 and not state[q]["vote_correct"] for state in states[1:])}
            reinforced = {q for q in orphan_discovered if any(state[q]["G"] >= 2 for state in states[1:])}
            reached_r1 = {q for q in discovered if any(state[q]["repair_distance"] == 1 and not state[q]["vote_correct"] for state in states[1:])}
            converted = {q for q in discovered if any(state[q]["vote_correct"] for state in states[1:])}
            still_converted = {q for q in converted if states[-1][q]["vote_correct"]}
            funnel_rows.append({"seed": seed, "setting": label, "initially_uncovered": len(initial_uncovered), "ever_discovered": len(discovered), "singleton_orphan_discoveries": len(orphan_discovered), "ever_reinforced": len(reinforced), "reached_repair_distance_1": len(reached_r1), "ever_vote_converted": len(converted), "still_vote_converted_final": len(still_converted)})

            # Late plateau attempts for S1/S2 only.
            if label in {"S1", "S2"}:
                last_accept = max(accepted_indices) if accepted_indices else -1
                final_persistent_q = {row["example_id"] for row in orphan_rows if row["seed"] == seed and row["setting"] == label and row["fate"] == "singleton_until_end"}
                final_assignment = data["responsibility"][-1].get("service_assignment_by_question", {}) if data["responsibility"] else {}
                persistent_service_agents = {int(final_assignment[q]["service_agent_id"]) for q in final_persistent_q if q in final_assignment}
                for decision in data["decisions"]:
                    if int(decision["update_index"]) <= last_accept:
                        continue
                    selected = {int(value) for value in decision.get("selected_target_ids", [])}
                    for branch in decision.get("branches", []):
                        target = int(branch["target_agent_id"])
                        priority = next((row for row in decision.get("agent_target_priorities", []) if int(row["agent_id"]) == target), {})
                        assigned = set(branch.get("assigned_question_hashes", []))
                        funnel = branch.get("funnel", {})
                        plateau_rows.append({
                            "seed": seed, "setting": label, "last_accepted_update_index": last_accept,
                            "attempt_update_index": int(decision["update_index"]),
                            "attempts_after_last_accept": 31 - last_accept,
                            "target_agent_id": target,
                            "expected_update_value": priority.get("expected_update_value", ""),
                            "branch_failure_count": priority.get("branch_failure_count", ""),
                            "repairability_discount": priority.get("repairability_discount", ""),
                            "updates_since_selected": priority.get("updates_since_selected", ""),
                            "selection_rank": priority.get("selection_rank", ""),
                            "selected": target in selected,
                            "assigned_persistent_orphan_count": len(assigned & final_persistent_q),
                            "target_is_persistent_orphan_service_agent": target in persistent_service_agents,
                            "valid_candidate_count": int(funnel.get("valid_candidate_count", 0)),
                            "critic_semantic_rejections": int(funnel.get("critic_semantic_rejections", 0)),
                            "stage_b_evaluated": int(funnel.get("stage_b_evaluated", 0)),
                            "constraint_feasible": int(funnel.get("constraint_feasible", 0)),
                            "rejected_target_regression": int(funnel.get("rejected_target_regression", 0)),
                            "rejected_team_vote_regression": int(funnel.get("rejected_team_vote_regression", 0)),
                            "rejected_no_target_or_vote_progress": int(funnel.get("rejected_no_target_or_vote_progress", 0)),
                            "primary_failure_layer": candidate_primary_failure(funnel),
                            "candidate_level_orphan_repair_distance_change": "UNSUPPORTED_BY_EXISTING_ARTIFACTS",
                        })

            # Mechanism metrics per run.
            own_orphans = [row for row in orphan_rows if row["seed"] == seed and row["setting"] == label]
            own_routing = [row for row in routing_rows if row["seed"] == seed and row["setting"] == label]
            own_updates = [row for row in decomposition_rows if row["seed"] == seed and row["setting"] == label]
            created = len(own_orphans)
            positive_member = sum(max(0.0, float(row["mean_member_gain"]) * 5.0) for row in own_updates)
            positive_vote = sum(max(0, int(row["vote_net_gain"])) for row in own_updates)
            final_dyn = dynamics[-1]
            mechanism_rows.append({
                "seed": seed, "setting": label, "accepted_updates": len(accepted_indices),
                "discoveries_0_to_positive": sum(int(row["new_0_to_1_count"]) + int(row["new_0_to_2plus_count"]) for row in own_updates),
                "singleton_discoveries_0_to_1": sum(int(row["new_0_to_1_count"]) for row in own_updates),
                "orphan_created_count": created,
                "orphan_reinforced_count": sum(bool(row["first_reinforcement_update"] != "") for row in own_orphans),
                "orphan_reinforcement_rate": None if not created else sum(bool(row["first_reinforcement_update"] != "") for row in own_orphans) / created,
                "orphan_vote_converted_count": sum(bool(row["first_vote_conversion_update"] != "") for row in own_orphans),
                "orphan_vote_conversion_rate": None if not created else sum(bool(row["first_vote_conversion_update"] != "") for row in own_orphans) / created,
                "persistent_singleton_count": sum(row["fate"] == "singleton_until_end" for row in own_orphans),
                "persistent_singleton_rate": None if not created else sum(row["fate"] == "singleton_until_end" for row in own_orphans) / created,
                "mean_updates_to_reinforcement": mean(float(row["latency_to_reinforcement"]) for row in own_orphans if row["latency_to_reinforcement"] != ""),
                "median_updates_to_reinforcement": median(float(row["latency_to_reinforcement"]) for row in own_orphans if row["latency_to_reinforcement"] != ""),
                "mean_updates_to_vote_conversion": mean(float(row["latency_to_vote_conversion"]) for row in own_orphans if row["latency_to_vote_conversion"] != ""),
                "median_updates_to_vote_conversion": median(float(row["latency_to_vote_conversion"]) for row in own_orphans if row["latency_to_vote_conversion"] != ""),
                "coverage_discovery_per_accepted_update": sum(int(row["new_0_to_1_count"]) + int(row["new_0_to_2plus_count"]) for row in own_updates) / max(1, len(accepted_indices)),
                "orphan_fraction_of_discovery": None if not own_updates or sum(int(row["new_0_to_1_count"]) + int(row["new_0_to_2plus_count"]) for row in own_updates) == 0 else created / sum(int(row["new_0_to_1_count"]) + int(row["new_0_to_2plus_count"]) for row in own_updates),
                "orphans_ever_revisited_by_other_member": sum(bool(row["ever_revisited_by_other_member"]) for row in own_routing),
                "cross_member_revisit_rate": None if not created else sum(bool(row["ever_revisited_by_other_member"]) for row in own_routing) / created,
                "revisited_orphans_converted": sum(bool(row["ever_revisited_by_other_member"] and row["ever_vote_converted"]) for row in own_routing),
                "nonrevisited_orphans_converted": sum(bool(not row["ever_revisited_by_other_member"] and row["ever_vote_converted"]) for row in own_routing),
                "mean_revisit_latency_raw_updates": mean(float(row["revisit_latency_raw_updates"]) for row in own_routing if row["revisit_latency_raw_updates"] != ""),
                "positive_member_gain_sum": positive_member,
                "positive_vote_net_gain_sum": positive_vote,
                "accepted_updates_with_positive_vote_net": sum(int(row["vote_net_gain"]) > 0 for row in own_updates),
                "accepted_updates_with_member_gain_but_zero_vote_net": sum(float(row["mean_member_gain"]) > 0 and int(row["vote_net_gain"]) == 0 for row in own_updates),
                "vote_yield_per_member_gain": None if positive_member == 0 else positive_vote / positive_member,
                "final_oracle_correct_count": int(final_dyn["oracle_correct_count"]),
                "final_vote_correct_count": int(final_dyn["team_vote_correct_count"]),
                "final_oracle_vote_gap": int(final_dyn["oracle_correct_count"]) - int(final_dyn["team_vote_correct_count"]),
                "final_pairwise_correctness_correlation": float(final_dyn["mean_pairwise_correctness_correlation"]),
                "final_effective_ensemble_size": float(final_dyn["n_eff"]),
            })

            # Final aggregate consistency.
            reference = train_reference[(seed, setting)]
            if int(reference["final_train_vote"]) != int(dynamics[-1]["team_vote_correct_count"]):
                raise AssertionError("final vote differs from published train reference")
            expected_members = [int(reference[f"member_{i}_correct"]) for i in range(5)]
            if expected_members != [int(value) for value in dynamics[-1]["per_agent_correct_counts"]]:
                raise AssertionError("final member counts differ from published train reference")
            if int(reference["accepted_updates"]) != len(accepted_indices):
                raise AssertionError("accepted update count differs from published train reference")

    # Aggregate orphan survival by setting and lag.
    survival_summary = []
    groups = defaultdict(list)
    for row in orphan_survival_rows:
        groups[(row["setting"], row["lag_accepted_updates"])].append(bool(row["alive"]))
    for (setting, lag), values in sorted(groups.items()):
        survival_summary.append({"setting": setting, "lag_accepted_updates": lag, "events_with_followup": len(values), "alive_count": sum(values), "alive_fraction": sum(values) / len(values)})

    write_csv(TABLES / "trajectory_state_summary.csv", state_rows)
    write_csv(TABLES / "g_transition_matrix.csv", transition_rows)
    write_csv(TABLES / "accepted_update_decomposition.csv", decomposition_rows)
    write_csv(TABLES / "orphan_creation_events.csv", orphan_rows)
    write_csv(TABLES / "orphan_rescue_events.csv", rescue_rows)
    write_csv(TABLES / "per_example_trajectory.csv", per_example_rows)
    write_csv(TABLES / "vote_conversion_events.csv", vote_conversion_rows)
    write_csv(TABLES / "routing_revisit_analysis.csv", routing_rows)
    write_csv(TABLES / "plateau_failure_analysis.csv", plateau_rows)
    write_csv(TABLES / "orphan_survival.csv", survival_summary)
    write_csv(TABLES / "consensus_conversion_funnel.csv", funnel_rows)
    write_csv(TABLES / "mechanism_metrics_by_seed_setting.csv", mechanism_rows)
    write_csv(TABLES / "replay_sanity_checks.csv", replay_checks)

    manifest_rows = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(consumed)
    ]
    if any("test" in row["path"].lower() for row in manifest_rows):
        raise AssertionError("test artifact path entered the consumed manifest")
    manifest = {
        "audit_version": "v15_coverage_fragmentation_consensus_audit_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "current_git_head": git("rev-parse", "HEAD"),
        "input_report_root": str(INPUT_REPORT.relative_to(ROOT)),
        "seeds": list(SEEDS), "settings": list(SETTINGS),
        "train_trajectory_only": True, "test_artifacts_consumed": False,
        "api_calls": 0, "model_calls": 0, "training_runs": 0, "test_evaluation": 0,
        "method_source_modification": 0,
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": sha256(Path(__file__)),
        "consumed_artifacts": manifest_rows,
        "figures_generated": False,
    }
    write_json(OUT / "audit_manifest.json", manifest)
    computed = {
        "audit_status": "PASS",
        "trajectory_replay": "PASS",
        "state_replay_check_count": len(replay_checks),
        "state_replay_failure_count": 0,
        "mechanism_metrics": mechanism_rows,
        "consensus_conversion_funnel": funnel_rows,
        "plateau_failure_layer_counts": {
            f"{seed}-{setting}": dict(Counter(row["primary_failure_layer"] for row in plateau_rows if row["seed"] == seed and row["setting"] == setting))
            for seed in SEEDS for setting in ("S1", "S2")
        },
        "cross_member_revisit_analysis": "EXACT_BRANCH_ASSIGNMENT_HASHES; explicit context hashes available where emitted",
        "unsupported": [
            "Rejected candidate per-example profiles are not persisted, so candidate-level orphan repair-distance deltas cannot be reconstructed.",
            "Responsibility snapshots occur at accepted team states; within-state routing is inferred from exact branch assigned_question_hashes.",
        ],
        "test_artifacts_consumed": False,
        "api_calls": 0,
    }
    write_json(OUT / "computed_facts.json", computed)
    print(json.dumps({"audit_status": "PASS", "states": len(state_rows), "accepted_transitions": len(decomposition_rows), "orphan_events": len(orphan_rows), "test_artifacts_consumed": False, "api_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
