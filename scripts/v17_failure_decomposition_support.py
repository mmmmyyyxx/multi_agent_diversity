"""Pure, zero-provider reconstruction helpers for the frozen V17 audit.

This module deliberately opens only historical JSON/JSONL/CSV files and
SQLite caches in read-only mode.  It never constructs an evaluator, solver,
optimizer, or client.  Raw questions, answers, prompts, traces, and cache
payloads remain in the private inputs/process and are never returned by the
public aggregation helpers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample, PromptAnswer
from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.tasks import get_task_spec
from multi_dataset_diverse_rl.team_differentiation import team_behavior_metrics
from multi_dataset_diverse_rl.utils import normalize_prompt_text


SOURCE_COMMIT_PREFIX = "ef9124e"
SOURCE_COMMIT = "ef9124eb2ddbfbe8d04c3c849c08a9e6875c7e61"
RUN_ID = "v17_failure_decomposition_20260820"
SEEDS = (56, 57, 58)
ARMS = ("S0", "S1", "S2", "S3", "S4")
SPLITS = ("train", "validation", "test")
SPLIT_FILES = {"train": "opt.csv", "validation": "val.csv", "test": "test.csv"}
SPLIT_SIZES = {"train": 75, "validation": 50, "test": 125}
SETTING_DIRS = {
    "S0": "shared_static_reference",
    "S1": "experimental_v17_formal_generic_2x2_matched",
    "S2": "experimental_v16_efficacy_g_matched",
    "S3": "experimental_v16_efficacy_r_m20",
    "S4": "experimental_v16_efficacy_r_m2f",
}

ZERO_API_COUNTERS = {
    "api_calls": 0,
    "model_calls": 0,
    "solver_calls": 0,
    "optimizer_calls": 0,
    "evaluator_calls": 0,
}


class EvidenceError(RuntimeError):
    """A frozen source artifact did not satisfy the audit contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float | int | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def arm_run_dir(repo: Path, seed: int, arm: str) -> Path:
    return repo / "runs" / "v17_formal_5arm_3seed_20260813" / f"seed{seed}" / "disambiguation_qa" / f"{SETTING_DIRS[arm]}_seed{seed}"


def cache_path(repo: Path, split: str, seed: int, arm: str) -> Path:
    if split not in {"validation", "test"}:
        raise ValueError(f"cache split is not held-out: {split}")
    return repo / "runs" / "v17_formal_5arm_3seed_20260813" / split / f"seed{seed}" / arm / "_solver_cache.sqlite"


def strict_split_path(repo: Path, split: str) -> Path:
    return repo / "strict_splits_bbh_seed42" / "disambiguation_qa" / SPLIT_FILES[split]


def load_examples(repo: Path, split: str) -> tuple[list[ProbeExample], list[dict[str, str]]]:
    path = strict_split_path(repo, split)
    if not path.is_file():
        raise EvidenceError(f"missing frozen split: {path.name}")
    spec = get_task_spec("bbh")
    raw: list[dict[str, str]] = []
    examples: list[ProbeExample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question = str(row["question"])
            answer = str(row["answer"])
            question_hash = sha256_text(question)
            raw.append({"question": question, "answer": answer, "question_hash": question_hash})
            examples.append(ProbeExample(question=question, question_hash=question_hash, gold_answer=spec.parse_gold(answer, question)))
    if len(examples) != SPLIT_SIZES[split]:
        raise EvidenceError(f"unexpected {split} size: {len(examples)}")
    if len({row.question_hash for row in examples}) != len(examples):
        raise EvidenceError(f"duplicate question hash in {split}")
    return examples, raw


def _answer_from_mapping(value: Mapping[str, Any]) -> PromptAnswer:
    """Build a minimal private answer object; deliberately discard trace text."""
    return PromptAnswer(
        answer=str(value.get("answer", "")),
        trace="",
        valid=bool(value.get("valid", False)),
        validity_status=str(value.get("validity_status", "")),
        terminal_invalid=bool(value.get("terminal_invalid", False)),
        response_hash=str(value.get("response_hash", "")),
        request_identity=str(value.get("request_identity", "")),
        solver_attempt_count=int(value.get("solver_attempt_count", 1) or 1),
        raw_invalid_attempt_count=int(value.get("raw_invalid_attempt_count", 0) or 0),
        recovered_from_invalid=bool(value.get("recovered_from_invalid", False)),
    )


def load_train_profiles(repo: Path, seed: int, arm: str, expected: int) -> tuple[list[list[PromptAnswer]], list[str]]:
    checkpoint = read_json(arm_run_dir(repo, seed, arm) / "training_checkpoint.json")
    raw_profiles = checkpoint.get("active_profiles")
    prompts = checkpoint.get("prompts")
    if not isinstance(raw_profiles, list) or len(raw_profiles) != 5:
        raise EvidenceError(f"missing five active profiles for {arm}/seed{seed}")
    if not isinstance(prompts, list) or len(prompts) != 5:
        raise EvidenceError(f"missing five prompts for {arm}/seed{seed}")
    profiles = [[_answer_from_mapping(row) for row in profile] for profile in raw_profiles]
    if any(len(profile) != expected for profile in profiles):
        raise EvidenceError(f"train profile length mismatch for {arm}/seed{seed}")
    return profiles, [sha256_text(normalize_prompt_text(str(prompt))) for prompt in prompts]


def load_cached_profiles(
    repo: Path,
    split: str,
    seed: int,
    arm: str,
    prompt_hashes: Sequence[str],
    examples: Sequence[ProbeExample],
) -> list[list[PromptAnswer]]:
    path = cache_path(repo, split, seed, arm)
    if not path.is_file():
        raise EvidenceError(f"missing held-out cache: {split}/seed{seed}/{arm}")
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT prompt_hash, question_hash, answer_json FROM solver_cache WHERE state = 'ready'"
        ).fetchall()
    finally:
        conn.close()
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for prompt_hash, question_hash, answer_json in rows:
        key = (str(prompt_hash), str(question_hash))
        if key in indexed:
            raise EvidenceError(f"duplicate ready cache key for {split}/seed{seed}/{arm}")
        indexed[key] = json.loads(str(answer_json))
    profiles: list[list[PromptAnswer]] = []
    for prompt_hash in prompt_hashes:
        profile: list[PromptAnswer] = []
        for example in examples:
            item = indexed.get((prompt_hash, example.question_hash))
            if item is None:
                raise EvidenceError(f"cache miss for frozen profile {split}/seed{seed}/{arm}")
            profile.append(_answer_from_mapping(item))
        profiles.append(profile)
    return profiles


def normalize_answer(spec: Any, answer: str) -> str:
    return str(spec.extract_pred(f"FINAL_ANSWER: {answer}", None))


def reconstruct_rows(
    *, examples: Sequence[ProbeExample], profiles: Sequence[Sequence[PromptAnswer]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = get_task_spec("bbh")
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        answers = [profile[index].answer for profile in profiles]
        validity = [profile[index].valid for profile in profiles]
        state = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=answers,
            valid_vector=validity,
            normalize_answer=lambda value: normalize_answer(spec, str(value)),
            match_answer=spec.match_answer,
            tie_break="abstain",
            seed=seed,
        )
        pivotal = []
        for agent_id in range(5):
            peer = build_peer_vote_context(state, agent_id)
            if state.team_correctness[agent_id] and state.vote_correct and peer.peer_margin <= 0:
                pivotal.append(agent_id)
        unique = [agent_id for agent_id, value in enumerate(state.team_correctness) if value and state.gold_vote_count == 1]
        rows.append({
            "question_hash": example.question_hash,
            "team_correctness": [int(value) for value in state.team_correctness],
            "validity": [int(value) for value in state.team_validity],
            "vote_correct": int(state.vote_correct),
            "oracle_correct": int(state.gold_vote_count > 0),
            "G": int(state.gold_vote_count),
            "H": int(state.largest_wrong_vote_count),
            "M": int(state.plurality_margin),
            "top_tie": int(state.top_tie),
            "unique_agents": unique,
            "pivotal_agents": pivotal,
            "correct_agent_count": int(sum(state.team_correctness)),
        })
    metrics = team_behavior_metrics(
        examples=examples,
        profiles=profiles,
        normalize_answer=lambda value: normalize_answer(spec, str(value)),
        match_answer=spec.match_answer,
        tie_break="abstain",
        seed=seed,
    )
    return rows, metrics


def summarise_rows(rows: Sequence[Mapping[str, Any]], behavior: Mapping[str, Any]) -> dict[str, Any]:
    n = len(rows)
    per_agent = [sum(int(row["team_correctness"][agent]) for row in rows) for agent in range(5)]
    unique = [sum(agent in row["unique_agents"] for row in rows) for agent in range(5)]
    pivotal = [sum(agent in row["pivotal_agents"] for row in rows) for agent in range(5)]
    return {
        "question_count": n,
        "vote_correct_count": sum(int(row["vote_correct"]) for row in rows),
        "vote_accuracy": pct(sum(int(row["vote_correct"]) for row in rows), n),
        "oracle_correct_count": sum(int(row["oracle_correct"]) for row in rows),
        "oracle_accuracy": pct(sum(int(row["oracle_correct"]) for row in rows), n),
        "mean_member_accuracy": mean([value / n for value in per_agent]),
        "min_member_accuracy": min(per_agent) / n,
        "max_member_accuracy": max(per_agent) / n,
        "per_agent_correct_count": per_agent,
        "per_agent_accuracy": [value / n for value in per_agent],
        "unique_correct_count": unique,
        "pivotal_correct_count": pivotal,
        "mean_G": mean(row["G"] for row in rows),
        "mean_H": mean(row["H"] for row in rows),
        "mean_M": mean(row["M"] for row in rows),
        "G0_count": sum(int(row["G"]) == 0 for row in rows),
        "G1_count": sum(int(row["G"]) == 1 for row in rows),
        "G2_count": sum(int(row["G"]) == 2 for row in rows),
        "G3_count": sum(int(row["G"]) == 3 for row in rows),
        "G4_count": sum(int(row["G"]) == 4 for row in rows),
        "G5_count": sum(int(row["G"]) == 5 for row in rows),
        "n_eff": behavior["n_eff"],
        "mean_pairwise_correctness_correlation": behavior["mean_pairwise_correctness_correlation"],
        "mean_off_diagonal_same_wrong_excess": behavior["mean_off_diagonal_same_wrong_excess"],
        "oracle_covered_but_vote_wrong_rate": behavior["oracle_covered_but_vote_wrong_rate"],
        "terminal_invalid_count": behavior["terminal_invalid_count"],
    }


def transition_category(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if int(left["vote_correct"]) and not int(right["vote_correct"]):
        return "vote_loss"
    if not int(left["vote_correct"]) and int(right["vote_correct"]):
        return "vote_gain"
    if int(left["oracle_correct"]) and not int(right["oracle_correct"]):
        return "oracle_loss_no_vote_change"
    if not int(left["oracle_correct"]) and int(right["oracle_correct"]):
        return "oracle_gain_no_vote_change"
    if int(left["vote_correct"]):
        return "stable_vote_correct"
    return "stable_vote_wrong"


def make_transition_rows(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *, seed: int, split: str, contrast: str
) -> list[dict[str, Any]]:
    right_by_hash = {str(row["question_hash"]): row for row in right}
    if len(right_by_hash) != len(right):
        raise EvidenceError("duplicate right-side question hash")
    result: list[dict[str, Any]] = []
    for old in left:
        new = right_by_hash.get(str(old["question_hash"]))
        if new is None:
            raise EvidenceError("transition question universe mismatch")
        result.append({
            "seed": seed,
            "split": split,
            "contrast": contrast,
            "question_hash": str(old["question_hash"]),
            "category": transition_category(old, new),
            "vote_before": int(old["vote_correct"]),
            "vote_after": int(new["vote_correct"]),
            "oracle_before": int(old["oracle_correct"]),
            "oracle_after": int(new["oracle_correct"]),
            "G_before": int(old["G"]),
            "G_after": int(new["G"]),
            "H_before": int(old["H"]),
            "H_after": int(new["H"]),
            "M_before": int(old["M"]),
            "M_after": int(new["M"]),
            "correct_agent_count_before": int(old["correct_agent_count"]),
            "correct_agent_count_after": int(new["correct_agent_count"]),
            "correct_member_ids_before": [
                agent for agent, value in enumerate(old["team_correctness"]) if value
            ],
            "correct_member_ids_after": [
                agent for agent, value in enumerate(new["team_correctness"]) if value
            ],
            "unique_agents_before": list(old["unique_agents"]),
            "unique_agents_after": list(new["unique_agents"]),
            "pivotal_agents_before": list(old["pivotal_agents"]),
            "pivotal_agents_after": list(new["pivotal_agents"]),
        })
    return result


def transition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    by_category = Counter(str(row["category"]) for row in rows)
    vote_gains = by_category["vote_gain"]
    vote_losses = by_category["vote_loss"]
    # Vote gains/losses do not imply oracle gains/losses.  Count oracle
    # transitions directly so a conversion of already-present coverage is not
    # misreported as newly discovered coverage.
    oracle_gains = sum(
        not int(row["oracle_before"]) and int(row["oracle_after"])
        for row in rows
    )
    oracle_losses = sum(
        int(row["oracle_before"]) and not int(row["oracle_after"])
        for row in rows
    )
    return {
        "question_count": count,
        "vote_gain_count": vote_gains,
        "vote_loss_count": vote_losses,
        "vote_net_count": vote_gains - vote_losses,
        "oracle_gain_count": oracle_gains,
        "oracle_loss_count": oracle_losses,
        "oracle_net_count": oracle_gains - oracle_losses,
        "stable_vote_correct_count": by_category["stable_vote_correct"],
        "stable_vote_wrong_count": by_category["stable_vote_wrong"],
        "vote_loss_rate": pct(vote_losses, count),
        "vote_gain_rate": pct(vote_gains, count),
        "mean_G_delta": mean(int(row["G_after"]) - int(row["G_before"]) for row in rows),
        "mean_H_delta": mean(int(row["H_after"]) - int(row["H_before"]) for row in rows),
        "mean_M_delta": mean(int(row["M_after"]) - int(row["M_before"]) for row in rows),
    }


def normalized_entropy(counts: Sequence[int]) -> float:
    total = sum(int(value) for value in counts)
    if total <= 0:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in counts if value)
    return entropy / math.log(len(counts)) if len(counts) > 1 else 0.0


def gini(counts: Sequence[int]) -> float:
    values = [float(value) for value in counts]
    total = sum(values)
    if total <= 0:
        return 0.0
    return sum(abs(a - b) for a in values for b in values) / (2 * len(values) * total)


def rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted((value, index) for index, value in enumerate(values))
        result = [0.0] * len(values)
        cursor = 0
        while cursor < len(order):
            end = cursor
            while end + 1 < len(order) and order[end + 1][0] == order[cursor][0]:
                end += 1
            rank = (cursor + end) / 2 + 1
            for _, index in order[cursor:end + 1]:
                result[index] = rank
            cursor = end + 1
        return result
    a, b = ranks(left), ranks(right)
    mean_a, mean_b = mean(a), mean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    variance_a = sum((x - mean_a) ** 2 for x in a)
    variance_b = sum((y - mean_b) ** 2 for y in b)
    return numerator / math.sqrt(variance_a * variance_b) if variance_a and variance_b else None


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def training_log_summary(repo: Path, seed: int, arm: str) -> dict[str, Any]:
    run_dir = arm_run_dir(repo, seed, arm)
    dynamics = load_jsonl_if_exists(run_dir / "training_dynamics.jsonl")
    branches = load_jsonl_if_exists(run_dir / "dual_target_branch_decisions.jsonl")
    commits = load_jsonl_if_exists(run_dir / "dual_target_commit_decisions.jsonl")
    funnels = read_json(run_dir / "candidate_funnel.json") if (run_dir / "candidate_funnel.json").is_file() else {}
    repair_events = load_jsonl_if_exists(run_dir / "online_compatibility_repair_events.jsonl")
    target_audit = load_jsonl_if_exists(run_dir / "target_priority_audit.jsonl")
    final = next((row for row in reversed(dynamics) if int(row.get("update_index", -1)) >= -1), {})
    commits_by_agent = [0] * 5
    selected_counts = [0] * 5
    target_by_update: dict[int, list[int]] = defaultdict(list)
    for row in branches:
        target = row.get("target_agent_id")
        if target is not None:
            selected_counts[int(target)] += 1
            target_by_update[int(row.get("update_index", -1))].append(int(target))
    for row in commits:
        target = row.get("committed_target_id")
        if target is not None:
            commits_by_agent[int(target)] += 1
    generated = sum(int(row.get("candidate_count", 0) or 0) for row in branches)
    valid = sum(
        int(row.get("valid_candidate_count", 0) or 0)
        for row in funnels.get("updates", [])
        if isinstance(row, Mapping)
    )
    feasible = sum(int(row.get("passed_candidate_count", 0) or 0) for row in branches)
    accepted = sum(1 for row in commits if row.get("committed_target_id") is not None)
    return {
        "dynamics": dynamics,
        "branches": branches,
        "commits": commits,
        "candidate_funnel": funnels,
        "repair_events": repair_events,
        "target_audit": target_audit,
        "final_dynamics": final,
        "selected_counts": selected_counts,
        "commits_by_agent": commits_by_agent,
        "targets_by_update": {str(key): sorted(value) for key, value in target_by_update.items()},
        "generated_candidate_count": generated,
        "valid_candidate_count": valid,
        "passed_candidate_count": feasible,
        "accepted_update_count": accepted,
        "planned_update_count": max((int(row.get("update_index", -1)) for row in dynamics), default=-1) + 1,
    }


def target_metrics(log: Mapping[str, Any]) -> dict[str, Any]:
    counts = list(log["selected_counts"])
    commit_counts = list(log["commits_by_agent"])
    updates = int(log["planned_update_count"])
    target_audit = list(log["target_audit"])
    no_actionable = sum(1 for row in target_audit if str(row.get("no_actionable_reason", "")))
    return {
        "target_counts": counts,
        "target_shares": [pct(value, sum(counts)) for value in counts],
        "target_entropy": normalized_entropy(counts),
        "target_gini": gini(counts),
        "target_max_share": pct(max(counts, default=0), sum(counts)),
        "commit_counts": commit_counts,
        "commit_shares": [pct(value, sum(commit_counts)) for value in commit_counts],
        "commit_entropy": normalized_entropy(commit_counts),
        "commit_gini": gini(commit_counts),
        "commit_max_share": pct(max(commit_counts, default=0), sum(commit_counts)),
        "planned_updates": updates,
        "target_slots": sum(counts),
        "no_actionable_updates": no_actionable,
        "target_switch_count": sum(
            1
            for previous, current in zip(
                [row.get("selected_target_ids", []) for row in target_audit],
                [row.get("selected_target_ids", []) for row in target_audit][1:],
                strict=False,
            )
            if previous != current
        ),
    }


def classify_hypotheses(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Frozen deterministic decision rules; no effect sizes are tuned post hoc."""
    by_seed = {int(row["seed"]): row for row in seed_rows}
    h1 = [row for row in by_seed.values() if row["s2_s1_train_delta"] > row["s2_s1_test_delta"] and row["s2_s1_gap"] > 0]
    h1_aggregate = mean(row["s2_s1_gap"] for row in by_seed.values())
    h1_status = "SUPPORTED" if len(h1) >= 2 and (h1_aggregate or 0.0) > 0 else "NOT_SUPPORTED"
    h2_structural = [row for row in by_seed.values() if row["s2_oracle_minus_s1"] >= 0 and row["s2_vote_minus_s1"] < 0 and row["s2_oracle_vote_gap_minus_s1"] > 0]
    h2_any_structure = [row for row in by_seed.values() if row["s2_oracle_minus_s1"] >= 0 or row["s2_coverage_delta"] > 0]
    h2_status = "SUPPORTED" if len(h2_structural) >= 2 else ("MIXED" if h2_any_structure else "NOT_SUPPORTED")
    h3_concentrated = [row for row in by_seed.values() if row["s2_entropy_minus_s1"] < 0]
    h3_associated = [row for row in h3_concentrated if row["high_target_agent_test_delta"] < row["low_target_agent_test_delta"]]
    h3_status = "SUPPORTED" if len(h3_concentrated) >= 2 and len(h3_associated) >= 2 else ("MIXED" if len(h3_concentrated) >= 2 else "NOT_SUPPORTED")
    h4a = [row for row in by_seed.values() if row["s2_accepted_updates"] < row["s1_accepted_updates"]]
    h4a_status = "SUPPORTED" if len(h4a) >= 2 and mean(row["s2_accepted_updates"] - row["s1_accepted_updates"] for row in by_seed.values()) < 0 else "NOT_SUPPORTED"
    h4b = [row for row in by_seed.values() if row["s2_test_gain_per_commit"] < row["s1_test_gain_per_commit"] and row["s2_s1_gap"] > 0]
    h4b_status = "SUPPORTED" if len(h4b) >= 2 else "NOT_SUPPORTED"
    h5 = [row for row in by_seed.values() if row["s2_specialization_train_minus_s1"] > 0 and row["s2_specialization_test_minus_s1"] <= 0 and row["specialization_measure_count"] >= 2]
    h5_status = "SUPPORTED" if len(h5) >= 2 else ("MIXED" if h5 else "NOT_SUPPORTED")
    return [
        {"hypothesis_id": "H1", "status": h1_status, "supporting_seed_count": len(h1), "rule": "train-test S2-S1 degradation"},
        {"hypothesis_id": "H2", "status": h2_status, "supporting_seed_count": len(h2_structural), "rule": "oracle/coverage retained while vote worsens"},
        {"hypothesis_id": "H3", "status": h3_status, "supporting_seed_count": len(h3_associated), "rule": "concentration plus descriptive transfer association", "detail": "CONCENTRATED_BUT_NO_TRANSFER_ASSOCIATION" if len(h3_concentrated) >= 2 and len(h3_associated) < 2 else ""},
        {"hypothesis_id": "H4A", "status": h4a_status, "supporting_seed_count": len(h4a), "rule": "accepted update throughput"},
        {"hypothesis_id": "H4B", "status": h4b_status, "supporting_seed_count": len(h4b), "rule": "test gain per commit"},
        {"hypothesis_id": "H5", "status": h5_status, "supporting_seed_count": len(h5), "rule": "train specialization fades on test across two measures"},
    ]


FORBIDDEN_TEXT = ("\\\\", "/home/", "d:\\", "c:\\", "question", "answer", "prompt", "trace", "sqlite", "raw_response", "endpoint", "api_key")


def assert_public_sanitized(report_dir: Path) -> None:
    for path in report_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf"}:
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in lowered:
                raise EvidenceError(f"sanitization violation in public artifact: {path.name}")
