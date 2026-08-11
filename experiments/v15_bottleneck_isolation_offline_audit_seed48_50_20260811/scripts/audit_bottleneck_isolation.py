from __future__ import annotations

import bisect
import csv
import hashlib
import itertools
import json
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.peer_state import build_team_vote_state


OUT = Path(__file__).resolve().parents[1]
TABLES = OUT / "tables"
SEEDS = (48, 49, 50)
SETTING = "shared_responsibility_conditioned_dual_target"
N = 75


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        if names:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rebuild(row: dict[str, Any], seed: int) -> dict[str, Any]:
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
        or state.gold_vote_count != int(row["gold_vote_count"])
    ):
        raise AssertionError("recorded train state differs from repository plurality replay")
    return state_record(state, seed)


def state_record(state: Any, seed: int) -> dict[str, Any]:
    return {
        "question_hash": state.question_hash,
        "gold_answer": state.gold_answer,
        "answers": tuple(state.team_answers),
        "validity": tuple(state.team_validity),
        "correctness": tuple(state.team_correctness),
        "G": int(state.gold_vote_count),
        "vote_correct": bool(state.vote_correct),
        "repair_distance": repair_distance(state, seed),
    }


def repair_distance(state: Any, seed: int) -> int:
    if state.vote_correct:
        return 0
    wrong = [agent for agent, correct in enumerate(state.team_correctness) if not correct]
    for count in range(1, len(wrong) + 1):
        for subset in itertools.combinations(wrong, count):
            answers = list(state.team_answers)
            validity = list(state.team_validity)
            for agent in subset:
                answers[agent] = state.gold_answer
                validity[agent] = True
            candidate = build_team_vote_state(
                question_hash=state.question_hash,
                gold_answer=state.gold_answer,
                answers=answers,
                valid_vector=validity,
                normalize_answer=lambda value: str(value or "").strip(),
                match_answer=lambda prediction, gold: prediction == gold,
                tie_break="abstain",
                seed=seed,
            )
            if candidate.vote_correct:
                return count
    raise AssertionError("full repair did not produce a correct vote")


def candidate_state(parent: dict[str, Any], target: int, answer: str, valid: bool, seed: int) -> dict[str, Any]:
    answers = list(parent["answers"])
    validity = list(parent["validity"])
    answers[target] = answer
    validity[target] = valid
    state = build_team_vote_state(
        question_hash=parent["question_hash"],
        gold_answer=parent["gold_answer"],
        answers=answers,
        valid_vector=validity,
        normalize_answer=lambda value: str(value or "").strip(),
        match_answer=lambda prediction, gold: prediction == gold,
        tie_break="abstain",
        seed=seed,
    )
    return state_record(state, seed)


def context_hashes(row: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    value = row.get("selected_context_pattern_question_hashes", {})
    if isinstance(value, dict):
        for items in value.values():
            if isinstance(items, list):
                result.update(str(item) for item in items)
    return result


def sign(value: int) -> str:
    return "+" if value > 0 else "-" if value < 0 else "0"


def geometry_type(target_gain: int, vote_net: int) -> str:
    if target_gain > 0 and vote_net > 0:
        return "A_target_plus_vote_plus"
    if target_gain > 0 and vote_net == 0:
        return "B_target_plus_vote_zero"
    if target_gain == 0 and vote_net > 0:
        return "C_target_zero_vote_plus"
    if target_gain > 0 and vote_net < 0:
        return "D_target_plus_vote_minus"
    if target_gain < 0 and vote_net > 0:
        return "E_target_minus_vote_plus"
    return "F_no_joint_value"


def load_cache(cache_path: Path, prompt_hashes: set[str], qids: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    uri = f"file:{cache_path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        marks = ",".join("?" for _ in prompt_hashes)
        rows = connection.execute(
            f"SELECT prompt_hash, question_hash, state, answer_json FROM solver_cache WHERE prompt_hash IN ({marks})",
            tuple(sorted(prompt_hashes)),
        ).fetchall()
    finally:
        connection.close()
    allowed = set(qids)
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for prompt_hash, question_hash, state, payload in rows:
        if question_hash not in allowed:
            continue
        if state != "ready" or not payload:
            raise AssertionError("candidate observation is not ready")
        parsed = json.loads(payload)
        observations[(str(prompt_hash), str(question_hash))] = {
            "answer": str(parsed.get("answer", "")),
            "valid": bool(parsed.get("valid", False)),
            "terminal_invalid": bool(parsed.get("terminal_invalid", False)),
        }
    expected = {(prompt_hash, qid) for prompt_hash in prompt_hashes for qid in qids}
    if set(observations) != expected:
        raise AssertionError(f"candidate cache coverage mismatch: missing={len(expected - set(observations))}")
    return observations


def load_run(seed: int, consumed: set[Path]) -> dict[str, Any]:
    run = ROOT / "runs" / f"v15f{seed}" / "disambiguation_qa" / f"{SETTING}_seed{seed}"
    paths = {
        "peer": run / "peer_state_history.jsonl",
        "dynamics": run / "training_dynamics.jsonl",
        "decisions": run / "candidate_decisions.jsonl",
        "commits": run / "dual_target_commit_decisions.jsonl",
        "responsibility": run / "responsibility_assignments.jsonl",
        "contexts": run / "tcs_context_history.jsonl",
        "scores": run / "repairability_adjusted_target_scores.jsonl",
        "cache": run / "_solver_cache.sqlite",
    }
    consumed.update(paths.values())
    peer = read_jsonl(paths["peer"])
    decisions = read_jsonl(paths["decisions"])
    dynamics = read_jsonl(paths["dynamics"])
    commits = read_jsonl(paths["commits"])
    responsibility = read_jsonl(paths["responsibility"])
    contexts = read_jsonl(paths["contexts"])
    scores = read_jsonl(paths["scores"])
    accepted = [int(row["update_index"]) for row in decisions if row.get("accepted_prompt_hash")]
    if len(decisions) != 32 or len(dynamics) != 33 or len(commits) != 32:
        raise AssertionError("unexpected formal update inventory")
    groups = [peer[index:index + N] for index in range(0, len(peer), N)]
    if len(groups) != len(accepted) + 1 or len(responsibility) != len(groups):
        raise AssertionError("accepted state / responsibility inventory mismatch")
    qids = [str(row["question_hash"]) for row in groups[0]]
    states = []
    for group in groups:
        if [str(row["question_hash"]) for row in group] != qids:
            raise AssertionError("question order changed")
        states.append({str(row["question_hash"]): rebuild(row, seed) for row in group})
    context_map: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in contexts:
        context_map[(int(row["update_index"]), int(row["target_agent_id"]))].update(context_hashes(row))
    prompt_hashes = {
        str(candidate["prompt_hash"])
        for decision in decisions
        for candidate in decision.get("candidates", [])
        if candidate.get("evaluation")
    }
    observations = load_cache(paths["cache"], prompt_hashes, qids)
    return {
        "run": run,
        "paths": paths,
        "qids": qids,
        "states": states,
        "accepted": accepted,
        "decisions": decisions,
        "dynamics": dynamics,
        "responsibility": responsibility,
        "context_map": context_map,
        "observations": observations,
        "prompt_hashes": prompt_hashes,
        "scores": scores,
    }


def build_orphans(data: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    states = data["states"]
    accepted = data["accepted"]
    events = []
    for rank in range(1, len(states)):
        parent, child = states[rank - 1], states[rank]
        update = accepted[rank - 1]
        target = next(
            int(row["target_agent_id"])
            for row in data["decisions"]
            if int(row["update_index"]) == update and row.get("accepted_prompt_hash")
        )
        for qid in data["qids"]:
            if parent[qid]["G"] == 0 and child[qid]["G"] == 1 and not child[qid]["vote_correct"]:
                events.append({
                    "event_id": f"{seed}-S2-{len(events)}",
                    "question_hash": qid,
                    "creation_rank": rank,
                    "creation_update": update,
                    "creator": target,
                })
    return events


def active_state_rank(accepted: list[int], update: int) -> int:
    return bisect.bisect_left(accepted, update)


def main() -> None:
    preserved_report_files = {"scripts", "README.md", "AUDIT_REPORT.md", "audit_summary.json"}
    if OUT.exists() and any(path for path in OUT.iterdir() if path.name not in preserved_report_files):
        raise FileExistsError("output must be fresh apart from scripts")
    TABLES.mkdir(parents=True, exist_ok=True)
    consumed: set[Path] = {
        ROOT / "multi_dataset_diverse_rl" / "peer_state.py",
        ROOT / "multi_dataset_diverse_rl" / "responsibility.py",
        ROOT / "multi_dataset_diverse_rl" / "system.py",
    }
    opportunity_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    residual_effect_rows: list[dict[str, Any]] = []
    cache_checks: list[dict[str, Any]] = []

    for seed in SEEDS:
        data = load_run(seed, consumed)
        states = data["states"]
        accepted = data["accepted"]
        decisions_by_update = {int(row["update_index"]): row for row in data["decisions"]}
        orphans = build_orphans(data, seed)

        for event in orphans:
            qid = event["question_hash"]
            creator = int(event["creator"])
            eligible_opportunity_count = 0
            eligible_update_count = 0
            selected_count = assigned_count = context_count = valid_count = feasible_count = accepted_count = 0
            first: dict[str, int] = {}
            terminal = "run_end"
            for update in range(int(event["creation_update"]) + 1, 32):
                rank = active_state_rank(accepted, update)
                parent = states[rank][qid]
                if parent["vote_correct"]:
                    terminal = "vote_converted"
                    break
                if parent["G"] == 0:
                    terminal = "coverage_lost"
                    break
                responsibility = data["responsibility"][rank]
                eligible = {
                    int(agent) for agent in responsibility.get("eligible_agents_by_question", {}).get(qid, [])
                    if int(agent) != creator and not parent["correctness"][int(agent)]
                }
                if not eligible:
                    continue
                eligible_update_count += 1
                eligible_opportunity_count += len(eligible)
                first.setdefault("eligible", update)
                decision = decisions_by_update[update]
                selected = set(int(agent) for agent in decision.get("selected_target_ids", [])) & eligible
                branches = {int(row["target_agent_id"]): row for row in decision.get("branches", [])}
                candidates_by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for candidate in decision.get("candidates", []):
                    if candidate.get("evaluation"):
                        candidates_by_target[int(candidate["target_agent_id"])].append(candidate)
                for agent in sorted(eligible):
                    is_selected = agent in selected
                    branch = branches.get(agent, {})
                    is_assigned = is_selected and qid in set(branch.get("assigned_question_hashes", []))
                    is_context = is_assigned and qid in data["context_map"].get((update, agent), set())
                    valid_candidates = candidates_by_target.get(agent, []) if is_context else []
                    is_valid = bool(valid_candidates)
                    is_feasible = any(bool(candidate.get("constraint", {}).get("passed")) for candidate in valid_candidates)
                    accepted_hash = str(decision.get("accepted_prompt_hash", ""))
                    is_accepted = any(str(candidate["prompt_hash"]) == accepted_hash for candidate in valid_candidates)
                    if is_selected:
                        selected_count += 1; first.setdefault("selected", update)
                    if is_assigned:
                        assigned_count += 1; first.setdefault("assigned", update)
                    if is_context:
                        context_count += 1; first.setdefault("context", update)
                    if is_valid:
                        valid_count += 1; first.setdefault("valid", update)
                    if is_feasible:
                        feasible_count += 1; first.setdefault("feasible", update)
                    if is_accepted:
                        accepted_count += 1; first.setdefault("accepted", update)
                    opportunity_rows.append({
                        "event_id": event["event_id"], "seed": seed, "question_hash": qid,
                        "update_index": update, "state_rank": rank, "creator_agent_id": creator,
                        "eligible_agent_id": agent, "selected": is_selected, "assigned_to_branch": is_assigned,
                        "explicit_context_hit": is_context, "valid_candidate_available": is_valid,
                        "common_safe_feasible_available": is_feasible, "accepted_candidate": is_accepted,
                        "parent_repair_distance": parent["repair_distance"],
                    })
            # Determine eventual episode fate directly from accepted states.
            converted = False
            lost = False
            for state in states[int(event["creation_rank"]) + 1:]:
                if state[qid]["vote_correct"]:
                    converted = True
                    break
                if state[qid]["G"] == 0:
                    lost = True
                    break
            event_rows.append({
                "event_id": event["event_id"], "seed": seed, "question_hash": qid,
                "creation_update": event["creation_update"], "creator_agent_id": creator,
                "eligible_update_count": eligible_update_count,
                "eligible_agent_update_opportunity_count": eligible_opportunity_count,
                "selected_opportunity_count": selected_count,
                "assigned_opportunity_count": assigned_count,
                "explicit_context_hit_count": context_count,
                "valid_candidate_opportunity_count": valid_count,
                "common_safe_feasible_opportunity_count": feasible_count,
                "accepted_opportunity_count": accepted_count,
                "ever_eligible": eligible_opportunity_count > 0,
                "ever_selected": selected_count > 0, "ever_assigned": assigned_count > 0,
                "ever_context_hit": context_count > 0, "ever_valid_candidate": valid_count > 0,
                "ever_common_safe_feasible": feasible_count > 0, "ever_accepted": accepted_count > 0,
                "eventually_vote_converted": converted, "coverage_lost": lost,
                "first_eligible_update": first.get("eligible", ""),
                "first_selected_update": first.get("selected", ""),
                "first_assigned_update": first.get("assigned", ""),
                "first_context_update": first.get("context", ""),
                "first_valid_candidate_update": first.get("valid", ""),
                "first_feasible_update": first.get("feasible", ""),
                "first_accepted_update": first.get("accepted", ""),
                "terminal_status_at_raw_attempt_scan": terminal,
            })

        # Reconstruct every evaluated candidate from the read-only cache.
        for decision in data["decisions"]:
            update = int(decision["update_index"])
            rank = active_state_rank(accepted, update)
            parent = states[rank]
            branches = {int(row["target_agent_id"]): row for row in decision.get("branches", [])}
            for candidate_index, candidate in enumerate(decision.get("candidates", [])):
                if not candidate.get("evaluation"):
                    continue
                target = int(candidate["target_agent_id"])
                prompt_hash = str(candidate["prompt_hash"])
                constraint = candidate["constraint"]
                assigned = set(branches.get(target, {}).get("assigned_question_hashes", []))
                context = data["context_map"].get((update, target), set())
                parent_target_count = sum(int(parent[qid]["correctness"][target]) for qid in data["qids"])
                candidate_target_count = 0
                parent_vote_count = sum(int(parent[qid]["vote_correct"]) for qid in data["qids"])
                candidate_vote_count = 0
                vote_gains = vote_losses = target_gains = target_losses = 0
                assigned_repairs = context_repairs = context_vote_conversions = 0
                all_repair_distance_improvements = all_repair_distance_regressions = 0
                context_orphan_count = 0
                vector_vote: list[bool] = []
                vector_g: list[int] = []
                for qid in data["qids"]:
                    observation = data["observations"][(prompt_hash, qid)]
                    child = candidate_state(parent[qid], target, observation["answer"], observation["valid"], seed)
                    candidate_target_count += int(child["correctness"][target])
                    candidate_vote_count += int(child["vote_correct"])
                    dg = child["G"] - parent[qid]["G"]
                    dr = child["repair_distance"] - parent[qid]["repair_distance"]
                    dv = int(child["vote_correct"]) - int(parent[qid]["vote_correct"])
                    target_delta = int(child["correctness"][target]) - int(parent[qid]["correctness"][target])
                    vote_gains += dv > 0; vote_losses += dv < 0
                    target_gains += target_delta > 0; target_losses += target_delta < 0
                    all_repair_distance_improvements += dr < 0
                    all_repair_distance_regressions += dr > 0
                    repaired = target_delta > 0 or dr < 0 or dv > 0
                    assigned_repairs += qid in assigned and repaired
                    context_repairs += qid in context and repaired
                    context_vote_conversions += qid in context and dv > 0
                    is_context_orphan = qid in context and parent[qid]["G"] == 1 and not parent[qid]["vote_correct"]
                    context_orphan_count += is_context_orphan
                    if qid in assigned or qid in context:
                        residual_effect_rows.append({
                            "seed": seed, "update_index": update, "candidate_hash": prompt_hash,
                            "target_agent_id": target, "question_hash": qid,
                            "assigned_to_branch": qid in assigned, "explicit_context_hit": qid in context,
                            "parent_G": parent[qid]["G"], "delta_G": dg,
                            "parent_repair_distance": parent[qid]["repair_distance"], "delta_repair_distance": dr,
                            "parent_vote_correct": parent[qid]["vote_correct"], "delta_vote_correct": dv,
                            "target_correct_delta": target_delta,
                            "candidate_passed": bool(constraint["passed"]),
                        })
                    vector_vote.append(bool(child["vote_correct"]))
                    vector_g.append(int(child["G"]))
                target_gain = candidate_target_count - parent_target_count
                vote_net = candidate_vote_count - parent_vote_count
                if target_gain != int(constraint["target_gain"]) or vote_gains != int(constraint["vote_gain_count"]) or vote_losses != int(constraint["vote_loss_count"]):
                    raise AssertionError("cache replay differs from candidate constraint counts")
                outcome = candidate["evaluation"]["team_outcome"]
                if vector_vote != [bool(value) for value in outcome["vote_correct_vector"]] or vector_g != [int(value) for value in outcome["gold_vote_counts"]]:
                    raise AssertionError("cache replay differs from stored candidate vectors")
                reasons = [str(value) for value in constraint.get("rejection_reasons", [])]
                accepted_hash = str(decision.get("accepted_prompt_hash", ""))
                candidate_rows.append({
                    "seed": seed, "update_index": update, "candidate_index": candidate_index,
                    "candidate_hash": prompt_hash, "target_agent_id": target,
                    "target_gain": target_gain, "target_sign": sign(target_gain),
                    "vote_gain_count": vote_gains, "vote_loss_count": vote_losses,
                    "vote_net_gain": vote_net, "vote_sign": sign(vote_net),
                    "geometry_type": geometry_type(target_gain, vote_net),
                    "constraint_passed": bool(constraint["passed"]),
                    "globally_accepted": prompt_hash == accepted_hash,
                    "rejection_reasons": compact(reasons),
                    "target_nonregression_passed": bool(constraint["target_nonregression_passed"]),
                    "team_vote_nonregression_passed": bool(constraint["team_vote_nonregression_passed"]),
                    "target_or_vote_progress_passed": bool(constraint["target_or_vote_progress_passed"]),
                    "terminal_invalid_nonregression_passed": bool(constraint["terminal_invalid_nonregression_passed"]),
                    "assigned_residual_count": len(assigned), "explicit_context_residual_count": len(context),
                    "context_orphan_count": context_orphan_count,
                    "assigned_residuals_repaired": assigned_repairs,
                    "context_residuals_repaired": context_repairs,
                    "context_vote_conversions": context_vote_conversions,
                    "all_repair_distance_improvements": all_repair_distance_improvements,
                    "all_repair_distance_regressions": all_repair_distance_regressions,
                    "target_example_gains": target_gains, "target_example_losses": target_losses,
                    "replay_support": "EXACT_READ_ONLY_SOLVER_CACHE_REPLAY",
                })
        cache_checks.append({
            "seed": seed, "evaluated_candidate_count": sum(1 for row in candidate_rows if row["seed"] == seed),
            "unique_candidate_prompt_count": len(data["prompt_hashes"]),
            "question_count_per_prompt": N, "missing_candidate_question_observation_count": 0,
            "constraint_count_mismatch": 0, "stored_vector_mismatch": 0,
        })

    # Event-level and opportunity-level funnels.
    funnel_rows: list[dict[str, Any]] = []
    stages = [
        ("created", lambda row: True), ("eligible", lambda row: bool(row["ever_eligible"])),
        ("selected", lambda row: bool(row["ever_selected"])), ("assigned", lambda row: bool(row["ever_assigned"])),
        ("explicit_context", lambda row: bool(row["ever_context_hit"])),
        ("valid_candidate", lambda row: bool(row["ever_valid_candidate"])),
        ("common_safe_feasible", lambda row: bool(row["ever_common_safe_feasible"])),
        ("accepted", lambda row: bool(row["ever_accepted"])),
        ("vote_converted", lambda row: bool(row["eventually_vote_converted"])),
    ]
    for seed_scope in [*SEEDS, "all"]:
        rows = event_rows if seed_scope == "all" else [row for row in event_rows if row["seed"] == seed_scope]
        prefix_rows = list(rows)
        previous_prefix = None
        for stage, predicate in stages:
            independent_count = sum(predicate(row) for row in rows)
            prefix_rows = [row for row in prefix_rows if predicate(row)]
            prefix_count = len(prefix_rows)
            funnel_rows.append({
                "seed_scope": seed_scope, "stage": stage,
                "independent_ever_hit_event_count": independent_count,
                "strict_prefix_event_count": prefix_count,
                "independent_fraction_of_created": independent_count / len(rows) if rows else None,
                "strict_prefix_fraction_of_created": prefix_count / len(rows) if rows else None,
                "strict_prefix_retention_from_previous_stage": (
                    None if previous_prefix in (None, 0) else prefix_count / previous_prefix
                ),
            })
            previous_prefix = prefix_count

    opportunity_summary: list[dict[str, Any]] = []
    for seed_scope in [*SEEDS, "all"]:
        rows = opportunity_rows if seed_scope == "all" else [row for row in opportunity_rows if row["seed"] == seed_scope]
        opportunity_summary.append({
            "seed_scope": seed_scope,
            "eligible_agent_update_opportunities": len(rows),
            "selected": sum(bool(row["selected"]) for row in rows),
            "assigned_to_branch": sum(bool(row["assigned_to_branch"]) for row in rows),
            "explicit_context_hits": sum(bool(row["explicit_context_hit"]) for row in rows),
            "valid_candidate_opportunities": sum(bool(row["valid_candidate_available"]) for row in rows),
            "common_safe_feasible_opportunities": sum(bool(row["common_safe_feasible_available"]) for row in rows),
            "accepted_opportunities": sum(bool(row["accepted_candidate"]) for row in rows),
            "p_context_given_eligible_opportunity": sum(bool(row["explicit_context_hit"]) for row in rows) / len(rows) if rows else None,
            "p_context_given_assigned_opportunity": (
                sum(bool(row["explicit_context_hit"]) for row in rows) / sum(bool(row["assigned_to_branch"]) for row in rows)
                if sum(bool(row["assigned_to_branch"]) for row in rows) else None
            ),
        })

    geometry_summary: list[dict[str, Any]] = []
    for seed_scope in [*SEEDS, "all"]:
        scoped = candidate_rows if seed_scope == "all" else [row for row in candidate_rows if row["seed"] == seed_scope]
        for rejected_scope in ("all_evaluated", "rejected_only"):
            rows = scoped if rejected_scope == "all_evaluated" else [row for row in scoped if not row["constraint_passed"]]
            for category in (
                "A_target_plus_vote_plus", "B_target_plus_vote_zero", "C_target_zero_vote_plus",
                "D_target_plus_vote_minus", "E_target_minus_vote_plus", "F_no_joint_value",
            ):
                selected = [row for row in rows if row["geometry_type"] == category]
                geometry_summary.append({
                    "seed_scope": seed_scope, "candidate_scope": rejected_scope, "geometry_type": category,
                    "candidate_count": len(selected), "fraction": len(selected) / len(rows) if rows else None,
                    "context_orphan_exposures": sum(int(row["context_orphan_count"]) for row in selected),
                    "context_residuals_repaired": sum(int(row["context_residuals_repaired"]) for row in selected),
                    "context_vote_conversions": sum(int(row["context_vote_conversions"]) for row in selected),
                    "all_repair_distance_improvements": sum(int(row["all_repair_distance_improvements"]) for row in selected),
                    "all_repair_distance_regressions": sum(int(row["all_repair_distance_regressions"]) for row in selected),
                })

    rejection_summary = []
    sign_geometry_summary = []
    for seed_scope in [*SEEDS, "all"]:
        scoped = candidate_rows if seed_scope == "all" else [row for row in candidate_rows if row["seed"] == seed_scope]
        counter: Counter[str] = Counter()
        for row in scoped:
            if not row["constraint_passed"]:
                counter.update(json.loads(row["rejection_reasons"]))
        for reason, count in sorted(counter.items()):
            rejection_summary.append({"seed_scope": seed_scope, "rejection_reason": reason, "candidate_count": count})
        rejected = [row for row in scoped if not row["constraint_passed"]]
        signs = Counter((row["target_sign"], row["vote_sign"]) for row in rejected)
        for target_sign in ("+", "0", "-"):
            for vote_sign in ("+", "0", "-"):
                count = signs[(target_sign, vote_sign)]
                sign_geometry_summary.append({
                    "seed_scope": seed_scope, "target_sign": target_sign, "vote_sign": vote_sign,
                    "rejected_candidate_count": count,
                    "fraction_of_rejected": count / len(rejected) if rejected else None,
                })

    write_csv(TABLES / "propagation_opportunity_events.csv", opportunity_rows)
    write_csv(TABLES / "propagation_event_funnel.csv", event_rows)
    write_csv(TABLES / "propagation_funnel_summary.csv", funnel_rows)
    write_csv(TABLES / "opportunity_normalized_summary.csv", opportunity_summary)
    write_csv(TABLES / "candidate_conflict_geometry.csv", candidate_rows)
    write_csv(TABLES / "candidate_conflict_geometry_summary.csv", geometry_summary)
    write_csv(TABLES / "candidate_residual_effects.csv", residual_effect_rows)
    write_csv(TABLES / "candidate_rejection_reason_summary.csv", rejection_summary)
    write_csv(TABLES / "rejected_candidate_sign_matrix.csv", sign_geometry_summary)
    write_csv(TABLES / "cache_replay_sanity.csv", cache_checks)

    manifest_rows = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(consumed)
    ]
    if any("test" in row["path"].lower() for row in manifest_rows):
        raise AssertionError("test artifact entered manifest")
    manifest = {
        "audit_version": "v15_bottleneck_isolation_offline_audit_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "current_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seeds": list(SEEDS), "setting": "S2", "train_trajectory_only": True,
        "api_calls": 0, "model_calls": 0, "training_runs": 0, "test_evaluation": 0,
        "test_artifacts_consumed": False, "method_source_modification": 0, "figures_generated": False,
        "cache_access_mode": "sqlite_read_only_uri_mode_ro",
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": sha256(Path(__file__)),
        "consumed_artifacts": manifest_rows,
    }
    write_json(OUT / "audit_manifest.json", manifest)
    write_json(OUT / "computed_facts.json", {
        "audit_status": "PASS", "orphan_event_count": len(event_rows),
        "eligible_opportunity_count": len(opportunity_rows), "evaluated_candidate_count": len(candidate_rows),
        "candidate_residual_effect_row_count": len(residual_effect_rows),
        "candidate_cache_coverage": "COMPLETE", "cache_replay_checks": cache_checks,
        "test_artifacts_consumed": False, "api_calls": 0,
    })
    print(compact({
        "audit_status": "PASS", "orphan_events": len(event_rows),
        "eligible_opportunities": len(opportunity_rows), "evaluated_candidates": len(candidate_rows),
        "cache_coverage": "COMPLETE", "api_calls": 0, "test_artifacts_consumed": False,
    }))


if __name__ == "__main__":
    main()
