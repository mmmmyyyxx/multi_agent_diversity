"""Offline prompt-evolution distance, funnel, and path analysis.

This script reads existing run artifacts only. It never constructs a model client,
performs a rollout, or mutates source run files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DISTANCE_BINS: Sequence[Tuple[float, float, str]] = (
    (0.00, 0.20, "[0.00, 0.20)"),
    (0.20, 0.35, "[0.20, 0.35)"),
    (0.35, 0.45, "[0.35, 0.45)"),
    (0.45, 0.60, "[0.45, 0.60)"),
    (0.60, 1.00, "[0.60, 1.00]"),
)

CANDIDATE_FIELDS = [
    "run_root", "run_dir", "task", "setting", "seed", "epoch", "step",
    "agent_id", "update_attempt_id", "candidate_id", "parent_id",
    "candidate_generation", "parent_generation", "candidate_pool_source",
    "candidate_source", "generation_batch_type", "parent_is_current_active",
    "parent_is_existing_beam", "parent_is_nonactive_beam",
    "candidate_prompt_hash", "parent_prompt_hash", "active_prompt_before_hash",
    "initial_prompt_hash", "candidate_char_length", "parent_char_length",
    "candidate_word_count", "parent_word_count", "prompt_change_ratio_parent",
    "prompt_change_ratio_active_before", "prompt_change_ratio_initial",
    "parent_to_initial_ratio", "active_before_to_initial_ratio",
    "distance_reconstruction_available", "distance_reconstruction_reason",
    "agent_accept_count_before", "prompt_max_change_ratio",
    "prompt_large_shift_warmup_accepts", "prompt_large_shift_min_vote_delta",
    "large_shift", "large_shift_warmup_exempt", "large_shift_supported",
    "large_shift_vote_support_passed", "large_shift_accuracy_support_passed",
    "large_shift_vote_loss_support_passed",
    "large_shift_pivotal_loss_support_passed",
    "large_shift_shared_error_support_passed", "mechanism_contract_passed",
    "behavior_cycle_guard_passed", "prompt_trust_region_passed",
    "error_dependence_guard_passed", "accuracy_guard_passed",
    "invalid_guard_passed", "exact_prompt_cycle", "matched_behavior_archive",
    "max_behavior_cycle_similarity", "behavior_cycle_overlap",
    "rejection_reason", "vote_delta", "accuracy_delta", "vote_gain_rate",
    "vote_loss_rate", "vote_margin_delta", "pivotal_rescue_rate",
    "pivotal_loss_rate", "shared_error_rescue_score",
    "shared_error_creation_score", "boundary_shared_error_net_gain",
    "optimizer_schema_valid", "optimizer_redundant_filtered",
    "original_accuracy_invalid_dependence_feasible", "trajectory_feasible",
    "pareto_feasible", "pareto_rank", "pareto_selected", "retained_in_beam",
    "became_active_top1", "active_prompt_changed", "candidate_outcome",
    "preserved_mechanism_count", "modified_mechanism_present",
    "target_residual_family", "proposal_metadata_available",
    "behavior_transition_l1",
]

ATTEMPT_FIELDS = [
    "run_root", "run_dir", "task", "setting", "seed", "epoch", "step",
    "agent_id", "update_attempt_id", "raw_student_candidate_count",
    "final_optimizer_candidate_count", "evaluated_candidate_count",
    "existing_beam_candidate_count", "mean_prompt_change_ratio",
    "median_prompt_change_ratio", "p75_prompt_change_ratio",
    "p90_prompt_change_ratio", "max_prompt_change_ratio",
    "large_shift_candidate_count", "large_shift_candidate_rate",
    "trust_region_reject_count", "trust_region_reject_rate",
    "cycle_reject_count", "dependence_reject_count", "pareto_feasible_count",
    "pareto_retained_count", "pareto_not_retained_count",
    "existing_beam_won", "active_prompt_changed", "active_candidate_id",
    "active_candidate_prompt_hash", "active_candidate_change_ratio", "active_candidate_source",
    "active_prompt_distance_from_initial_before",
    "active_prompt_distance_from_initial_after",
    "active_prompt_cumulative_path_length", "direct_distance_from_initial",
    "active_top1_count", "attempt_complete",
]


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt or "").strip()).lower()


def prompt_change_ratio(parent_prompt: str, candidate_prompt: str) -> float:
    """Match TraceBeamSearchSystem.prompt_change_ratio exactly."""
    parent = normalize_prompt(parent_prompt)
    candidate = normalize_prompt(candidate_prompt)
    value = 1.0 - SequenceMatcher(None, parent, candidate).ratio()
    if math.isnan(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def normalized_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def short_prompt_hash(prompt: str) -> str:
    return hashlib.sha1(str(prompt).encode("utf-8")).hexdigest()[:12]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row if key not in fields})
    all_fields = list(fields) + extras
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in all_fields})


def parse_generation(identifier: Any) -> Optional[int]:
    match = re.match(r"^g(\d+)(?:_|$)", str(identifier or ""))
    return int(match.group(1)) if match else None


def quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def safe_mean(values: Iterable[Any]) -> Optional[float]:
    numbers = [value for item in values if (value := optional_float(item)) is not None]
    return statistics.mean(numbers) if numbers else None


def distance_bin(value: Optional[float]) -> Optional[str]:
    if value is None or value < 0 or value > 1:
        return None
    for low, high, label in DISTANCE_BINS:
        if low <= value < high or (high == 1.0 and low <= value <= high):
            return label
    return None


def discover_runs(roots: Sequence[Path]) -> List[Tuple[Path, Path]]:
    found: List[Tuple[Path, Path]] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        candidates = [resolved] if (resolved / "run_meta.json").exists() else [p.parent for p in resolved.rglob("run_meta.json")]
        for run_dir in sorted(candidates):
            key = str(run_dir.resolve()).lower()
            if key not in seen:
                seen.add(key)
                found.append((resolved, run_dir.resolve()))
    return found


def collect_prompt_catalog(run_dir: Path, initial_prompts: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    prompts: Dict[str, str] = {}
    mechanisms: Dict[str, Dict[str, Any]] = {}

    def add_prompt(value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        prompts[normalized_prompt_hash(value)] = value
        prompts[short_prompt_hash(value)] = value

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("prompt", "current_prompt", "initial_prompt"):
                add_prompt(value.get(key))
            prompt_hash = str(value.get("prompt_hash", "") or "")
            if prompt_hash and any(key in value for key in ("preserved_mechanisms", "modified_mechanism", "new_or_modified_mechanism", "change_summary", "target_residual_family")):
                mechanisms[prompt_hash] = {
                    "preserved_mechanisms": value.get("preserved_mechanisms"),
                    "modified_mechanism": value.get("modified_mechanism", value.get("new_or_modified_mechanism")),
                    "change_summary": value.get("change_summary"),
                    "target_residual_family": value.get("target_residual_family"),
                }
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for prompt in initial_prompts:
        add_prompt(prompt)
    for name in (
        "prompt_history.json", "training_checkpoint.json", "best_state.json",
        "last_state.json", "selected_state.json", "best_prompts.json",
    ):
        value = read_json(run_dir / name)
        if value is not None:
            walk(value)
    return prompts, mechanisms


def infer_run_identity(root: Path, run_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    name = run_dir.name
    match = re.search(r"_seed(-?\d+)$", name)
    seed = int(match.group(1)) if match else optional_int(meta.get("config", {}).get("seed"))
    setting = name[: match.start()] if match else name
    task = str(meta.get("comparison_task_id") or run_dir.parent.name)
    return {
        "run_root": root.name,
        "run_dir": str(run_dir),
        "task": task,
        "setting": setting,
        "seed": seed,
    }


def candidate_outcome(row: Dict[str, Any]) -> str:
    source = str(row.get("candidate_pool_source") or "")
    reason = str(row.get("rejection_reason") or "")
    if source == "current_active_fallback":
        return "current_fallback"
    if reason == "mechanism_contract_missing":
        return "mechanism_contract_rejected"
    if as_bool(row.get("accuracy_guard_passed")) is False:
        return "accuracy_guard_rejected"
    if as_bool(row.get("invalid_guard_passed")) is False:
        return "invalid_guard_rejected"
    if as_bool(row.get("error_dependence_guard_passed")) is False:
        return "dependence_guard_rejected"
    if reason == "exact_prompt_cycle" or as_bool(row.get("exact_prompt_cycle")) is True:
        return "exact_prompt_cycle_rejected"
    if reason in {"behavior_cycle", "accepted_state_cycle", "rejected_failure_cycle"} or as_bool(row.get("behavior_cycle_guard_passed")) is False:
        return "behavior_cycle_rejected"
    if reason == "unsupported_large_prompt_shift":
        return "unsupported_large_prompt_shift"
    if as_bool(row.get("became_active_top1")):
        return "active_top1"
    if as_bool(row.get("retained_in_beam")):
        return "retained_nonactive_beam"
    if source == "existing_beam":
        return "existing_beam"
    if as_bool(row.get("pareto_selected")) is False:
        return "pareto_not_retained"
    return "unknown"


def build_candidate_row(
    identity: Dict[str, Any],
    raw: Dict[str, Any],
    trajectory: Dict[str, Any],
    config: Dict[str, Any],
    catalog: Dict[str, str],
    mechanisms: Dict[str, Dict[str, Any]],
    active_hash: str,
    active_prompt: Optional[str],
    initial_hash: str,
    initial_prompt: Optional[str],
    accept_count: int,
) -> Dict[str, Any]:
    row = dict(identity)
    row.update({
        "epoch": optional_int(raw.get("epoch")),
        "step": optional_int(raw.get("step")),
        "agent_id": optional_int(raw.get("agent_id")),
        "update_attempt_id": raw.get("update_attempt_id", ""),
        "candidate_id": raw.get("candidate_id", ""),
        "parent_id": raw.get("parent_id", ""),
        "candidate_generation": parse_generation(raw.get("candidate_id")),
        "parent_generation": parse_generation(raw.get("parent_id")),
        "candidate_pool_source": raw.get("candidate_pool_source", ""),
        "candidate_source": raw.get("candidate_source", ""),
        "generation_batch_type": raw.get("generation_batch_type", ""),
    })
    candidate_hash = str(raw.get("prompt_hash") or "")
    parent_hash = str(raw.get("parent_prompt_hash") or "")
    preview = str(raw.get("prompt_preview") or "")
    if candidate_hash and preview and normalized_prompt_hash(preview) == candidate_hash:
        catalog[candidate_hash] = preview
    candidate_prompt = catalog.get(candidate_hash)
    parent_prompt = catalog.get(parent_hash)
    parent_is_active = bool(parent_hash and parent_hash == active_hash)
    parent_is_beam = bool(raw.get("parent_id") or raw.get("candidate_pool_source") == "existing_beam")
    parent_ratio = optional_float(raw.get("prompt_change_ratio"))

    def ratio(left_hash: str, left_prompt: Optional[str], right_hash: str, right_prompt: Optional[str]) -> Optional[float]:
        if left_hash and right_hash and left_hash == right_hash:
            return 0.0
        if left_prompt is not None and right_prompt is not None:
            return prompt_change_ratio(left_prompt, right_prompt)
        return None

    active_ratio = parent_ratio if parent_is_active else ratio(active_hash, active_prompt, candidate_hash, candidate_prompt)
    initial_ratio = ratio(initial_hash, initial_prompt, candidate_hash, candidate_prompt)
    parent_initial = ratio(initial_hash, initial_prompt, parent_hash, parent_prompt)
    active_initial = ratio(initial_hash, initial_prompt, active_hash, active_prompt)
    missing = []
    for key, value in (
        ("candidate_to_active", active_ratio), ("candidate_to_initial", initial_ratio),
        ("parent_to_initial", parent_initial), ("active_to_initial", active_initial),
    ):
        if value is None:
            missing.append(key)

    source = str(raw.get("candidate_pool_source") or "")
    threshold = float(config.get("prompt_max_change_ratio", 0.45) or 0.45)
    warmup = int(config.get("prompt_large_shift_warmup_accepts", 2) or 0)
    minimum_vote = float(config.get("prompt_large_shift_min_vote_delta", 0.02) or 0.0)
    allowed_vote_loss = float(config.get("baseline_allowed_vote_loss", 0.0) or 0.0)
    threshold_exceeded = bool(source == "optimizer" and parent_ratio is not None and parent_ratio > threshold)
    vote_pass = fnum(raw.get("vote_delta")) >= minimum_vote
    accuracy_pass = fnum(raw.get("accuracy_delta")) >= 0.0
    vote_loss_pass = fnum(raw.get("vote_loss_rate")) <= allowed_vote_loss
    pivotal_pass = fnum(raw.get("pivotal_loss_rate")) <= 0.0
    shared_pass = fnum(raw.get("shared_error_creation_score")) <= fnum(raw.get("shared_error_rescue_score"))
    mechanism_enabled = bool(config.get("mechanism_trust_region_enabled", False))
    large_supported = vote_pass and accuracy_pass and vote_loss_pass and (not mechanism_enabled or (pivotal_pass and shared_pass))

    merged = dict(raw)
    for key, value in trajectory.items():
        if key not in merged or merged.get(key) in (None, ""):
            merged[key] = value
    proposal = mechanisms.get(candidate_hash, {})
    preserved = proposal.get("preserved_mechanisms")
    transition = raw.get("candidate_transition_vector")
    transition_l1 = None
    if isinstance(transition, dict):
        values = [abs(value) for item in transition.values() if (value := optional_float(item)) is not None]
        transition_l1 = sum(values) if values else 0.0

    row.update({
        "parent_is_current_active": parent_is_active,
        "parent_is_existing_beam": parent_is_beam,
        "parent_is_nonactive_beam": parent_is_beam and not parent_is_active,
        "candidate_prompt_hash": candidate_hash,
        "parent_prompt_hash": parent_hash,
        "active_prompt_before_hash": active_hash,
        "initial_prompt_hash": initial_hash,
        "candidate_char_length": len(candidate_prompt) if candidate_prompt is not None else None,
        "parent_char_length": len(parent_prompt) if parent_prompt is not None else None,
        "candidate_word_count": len(candidate_prompt.split()) if candidate_prompt is not None else None,
        "parent_word_count": len(parent_prompt.split()) if parent_prompt is not None else None,
        "prompt_change_ratio_parent": parent_ratio,
        "prompt_change_ratio_active_before": active_ratio,
        "prompt_change_ratio_initial": initial_ratio,
        "parent_to_initial_ratio": parent_initial,
        "active_before_to_initial_ratio": active_initial,
        "distance_reconstruction_available": not missing,
        "distance_reconstruction_reason": "" if not missing else "missing_full_prompt:" + ",".join(missing),
        "agent_accept_count_before": accept_count,
        "prompt_max_change_ratio": threshold,
        "prompt_large_shift_warmup_accepts": warmup,
        "prompt_large_shift_min_vote_delta": minimum_vote,
        "large_shift": threshold_exceeded,
        "large_shift_warmup_exempt": threshold_exceeded and accept_count < warmup,
        "large_shift_supported": large_supported,
        "large_shift_vote_support_passed": vote_pass,
        "large_shift_accuracy_support_passed": accuracy_pass,
        "large_shift_vote_loss_support_passed": vote_loss_pass,
        "large_shift_pivotal_loss_support_passed": pivotal_pass,
        "large_shift_shared_error_support_passed": shared_pass,
        "mechanism_contract_passed": merged.get("mechanism_contract_passed"),
        "behavior_cycle_guard_passed": merged.get("behavior_cycle_guard_passed"),
        "prompt_trust_region_passed": merged.get("prompt_trust_region_passed"),
        "error_dependence_guard_passed": merged.get("error_dependence_guard_passed"),
        "accuracy_guard_passed": merged.get("accuracy_guard_passed"),
        "invalid_guard_passed": merged.get("invalid_guard_passed"),
        "exact_prompt_cycle": merged.get("exact_prompt_cycle"),
        "matched_behavior_archive": merged.get("matched_behavior_archive"),
        "max_behavior_cycle_similarity": merged.get("max_behavior_cycle_similarity"),
        "behavior_cycle_overlap": merged.get("behavior_cycle_overlap"),
        "rejection_reason": merged.get("rejection_reason", ""),
        "vote_delta": raw.get("vote_delta"),
        "accuracy_delta": raw.get("accuracy_delta"),
        "vote_gain_rate": raw.get("vote_gain_rate"),
        "vote_loss_rate": raw.get("vote_loss_rate"),
        "vote_margin_delta": raw.get("vote_margin_delta"),
        "pivotal_rescue_rate": raw.get("pivotal_rescue_rate"),
        "pivotal_loss_rate": raw.get("pivotal_loss_rate"),
        "shared_error_rescue_score": raw.get("shared_error_rescue_score"),
        "shared_error_creation_score": raw.get("shared_error_creation_score"),
        "boundary_shared_error_net_gain": raw.get("boundary_shared_error_net_gain"),
        "optimizer_schema_valid": True if source == "optimizer" else None,
        "optimizer_redundant_filtered": False if source == "optimizer" else None,
        "original_accuracy_invalid_dependence_feasible": all(
            as_bool(merged.get(key)) is not False
            for key in ("accuracy_guard_passed", "invalid_guard_passed", "error_dependence_guard_passed")
        ),
        "trajectory_feasible": not bool(merged.get("rejection_reason")),
        "pareto_feasible": merged.get("pareto_feasible"),
        "pareto_rank": merged.get("pareto_rank"),
        "pareto_selected": merged.get("pareto_selected"),
        "retained_in_beam": merged.get("in_top_beam"),
        "became_active_top1": merged.get("is_top1"),
        "active_prompt_changed": merged.get("active_prompt_changed"),
        "preserved_mechanism_count": len(preserved) if isinstance(preserved, list) else None,
        "modified_mechanism_present": (
            bool(str(proposal.get("modified_mechanism") or "").strip())
            if proposal and proposal.get("modified_mechanism") is not None else None
        ),
        "target_residual_family": proposal.get("target_residual_family") if proposal else None,
        "proposal_metadata_available": bool(proposal),
        "behavior_transition_l1": transition_l1,
    })
    row["candidate_outcome"] = candidate_outcome(row)
    return row


def fnum(value: Any, default: float = 0.0) -> float:
    parsed = optional_float(value)
    return parsed if parsed is not None else default


def analyze_run(root: Path, run_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    meta = read_json(run_dir / "run_meta.json", {}) or {}
    config = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    identity = infer_run_identity(root, run_dir, meta)
    updates = read_jsonl(run_dir / "update_logs.jsonl")
    trajectories = read_jsonl(run_dir / "trajectory_events.jsonl")
    candidate_raw = [row for row in updates if row.get("event") == "candidate_evaluated"]
    summaries = [row for row in updates if row.get("event") == "beam_update_summary"]
    initial_prompts = list(meta.get("initial_agent_prompts") or [])
    agent_count = int(meta.get("agents") or config.get("agents") or max(5, len(initial_prompts)))
    if len(initial_prompts) < agent_count:
        shared = str(config.get("shared_prompt") or "")
        initial_prompts.extend([shared] * (agent_count - len(initial_prompts)))
    catalog, mechanisms = collect_prompt_catalog(run_dir, initial_prompts)

    trajectory_index: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in trajectories:
        key = (row.get("epoch"), row.get("step"), row.get("agent_id"), row.get("candidate_id"))
        trajectory_index[key] = row

    groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in candidate_raw:
        epoch = optional_int(row.get("epoch"))
        step = optional_int(row.get("step"))
        agent = optional_int(row.get("agent_id"))
        if epoch is None or step is None or agent is None:
            continue
        groups[(epoch, step, agent)].append(row)
    summary_index = {
        (optional_int(row.get("epoch")), optional_int(row.get("step")), optional_int(row.get("agent_id"))): row
        for row in summaries
    }

    active_hash = {i: normalized_prompt_hash(initial_prompts[i]) for i in range(agent_count)}
    active_prompt: Dict[int, Optional[str]] = {i: initial_prompts[i] for i in range(agent_count)}
    initial_hash = dict(active_hash)
    accept_count = Counter()
    cumulative_path: Dict[int, Optional[float]] = {i: 0.0 for i in range(agent_count)}
    candidates: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []

    for key in sorted(groups):
        epoch, step, agent = key
        raw_rows = groups[key]
        before_hash = active_hash.get(agent, "")
        before_prompt = active_prompt.get(agent)
        initial_prompt = initial_prompts[agent] if agent < len(initial_prompts) else None
        built: List[Dict[str, Any]] = []
        for raw in raw_rows:
            trajectory = trajectory_index.get((epoch, step, agent, raw.get("candidate_id")), {})
            built.append(build_candidate_row(
                identity, raw, trajectory, config, catalog, mechanisms,
                before_hash, before_prompt, initial_hash.get(agent, ""), initial_prompt,
                accept_count[agent],
            ))
        candidates.extend(built)
        summary = summary_index.get(key, {})
        optimizer_rows = [row for row in built if row.get("candidate_pool_source") == "optimizer"]
        ratios = [value for row in optimizer_rows if (value := optional_float(row.get("prompt_change_ratio_parent"))) is not None]
        top = [row for row in built if as_bool(row.get("became_active_top1"))]
        active_row = top[0] if len(top) == 1 else None
        changed = bool(active_row and as_bool(active_row.get("active_prompt_changed")))
        before_initial = optional_float(built[0].get("active_before_to_initial_ratio")) if built else None
        step_ratio = optional_float(active_row.get("prompt_change_ratio_active_before")) if active_row else None
        after_hash = before_hash
        after_prompt = before_prompt
        if changed and active_row:
            after_hash = str(active_row.get("candidate_prompt_hash") or "")
            after_prompt = catalog.get(after_hash)
            if cumulative_path[agent] is not None and step_ratio is not None:
                cumulative_path[agent] = fnum(cumulative_path[agent]) + step_ratio
            else:
                cumulative_path[agent] = None
            accept_count[agent] += 1
        after_initial = None
        if after_hash == initial_hash.get(agent):
            after_initial = 0.0
        elif after_prompt is not None and initial_prompt is not None:
            after_initial = prompt_change_ratio(initial_prompt, after_prompt)
        active_hash[agent] = after_hash
        active_prompt[agent] = after_prompt

        raw_student = optional_int(summary.get("student_candidate_count_raw"))
        if raw_student is None:
            raw_student = max(
                [optional_int(row.get("student_candidate_count_raw")) or 0 for row in raw_rows],
                default=0,
            )
        final_optimizer = optional_int(summary.get("num_optimizer_candidates"))
        if final_optimizer is None:
            final_optimizer = len(optimizer_rows)
        trust_reject = sum(row.get("candidate_outcome") == "unsupported_large_prompt_shift" for row in optimizer_rows)
        cycle_reject = sum(row.get("candidate_outcome") in {"exact_prompt_cycle_rejected", "behavior_cycle_rejected"} for row in optimizer_rows)
        dependence_reject = sum(row.get("candidate_outcome") == "dependence_guard_rejected" for row in optimizer_rows)
        large_count = sum(as_bool(row.get("large_shift")) is True for row in optimizer_rows)
        attempts.append({
            **identity,
            "epoch": epoch, "step": step, "agent_id": agent,
            "update_attempt_id": summary.get("update_attempt_id") or (raw_rows[0].get("update_attempt_id") if raw_rows else ""),
            "raw_student_candidate_count": raw_student,
            "final_optimizer_candidate_count": final_optimizer,
            "evaluated_candidate_count": len(built),
            "existing_beam_candidate_count": sum(row.get("candidate_pool_source") == "existing_beam" for row in built),
            "mean_prompt_change_ratio": safe_mean(ratios),
            "median_prompt_change_ratio": quantile(ratios, 0.50),
            "p75_prompt_change_ratio": quantile(ratios, 0.75),
            "p90_prompt_change_ratio": quantile(ratios, 0.90),
            "max_prompt_change_ratio": max(ratios) if ratios else None,
            "large_shift_candidate_count": large_count,
            "large_shift_candidate_rate": large_count / len(optimizer_rows) if optimizer_rows else None,
            "trust_region_reject_count": trust_reject,
            "trust_region_reject_rate": trust_reject / len(optimizer_rows) if optimizer_rows else None,
            "cycle_reject_count": cycle_reject,
            "dependence_reject_count": dependence_reject,
            "pareto_feasible_count": sum(as_bool(row.get("pareto_feasible")) is True for row in built),
            "pareto_retained_count": sum(as_bool(row.get("retained_in_beam")) is True for row in built),
            "pareto_not_retained_count": sum(row.get("candidate_outcome") == "pareto_not_retained" for row in optimizer_rows),
            "existing_beam_won": bool(active_row and active_row.get("candidate_pool_source") == "existing_beam"),
            "active_prompt_changed": changed,
            "active_candidate_id": active_row.get("candidate_id") if active_row else None,
            "active_candidate_prompt_hash": active_row.get("candidate_prompt_hash") if active_row else None,
            "active_candidate_change_ratio": step_ratio,
            "active_candidate_source": active_row.get("candidate_pool_source") if active_row else None,
            "active_prompt_distance_from_initial_before": before_initial,
            "active_prompt_distance_from_initial_after": after_initial,
            "active_prompt_cumulative_path_length": cumulative_path[agent],
            "direct_distance_from_initial": after_initial,
            "active_top1_count": len(top),
            "attempt_complete": bool(summary) and len(top) == 1,
        })

    agent_rows: List[Dict[str, Any]] = []
    attempts_by_agent: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        attempts_by_agent[int(row["agent_id"])].append(row)
    for agent in range(agent_count):
        rows = attempts_by_agent.get(agent, [])
        changed_rows = [row for row in rows if as_bool(row.get("active_prompt_changed"))]
        ratios = [value for row in changed_rows if (value := optional_float(row.get("active_candidate_change_ratio"))) is not None]
        final_direct = optional_float(rows[-1].get("direct_distance_from_initial")) if rows else 0.0
        final_path = optional_float(rows[-1].get("active_prompt_cumulative_path_length")) if rows else 0.0
        agent_rows.append({
            **identity, "agent_id": agent, "update_attempt_count": len(rows),
            "active_update_count": len(changed_rows),
            "active_update_rate": len(changed_rows) / len(rows) if rows else 0.0,
            "initial_prompt_hash": initial_hash.get(agent),
            "final_active_prompt_hash": active_hash.get(agent),
            "unique_active_prompt_count": len(
                {initial_hash.get(agent)}
                | {row.get("active_candidate_prompt_hash") for row in changed_rows if row.get("active_candidate_prompt_hash")}
            ),
            "mean_active_change_ratio": safe_mean(ratios),
            "median_active_change_ratio": quantile(ratios, 0.50),
            "p90_active_change_ratio": quantile(ratios, 0.90),
            "max_active_change_ratio": max(ratios) if ratios else None,
            "large_active_update_count": sum(value > float(config.get("prompt_max_change_ratio", 0.45) or 0.45) for value in ratios),
            "large_active_update_rate": sum(value > float(config.get("prompt_max_change_ratio", 0.45) or 0.45) for value in ratios) / len(ratios) if ratios else 0.0,
            "final_direct_distance_from_initial": final_direct,
            "active_cumulative_path_length": final_path,
            "path_to_direct_ratio": final_path / final_direct if final_path is not None and final_direct not in (None, 0.0) else None,
            "unknown_active_step_distance_count": len(changed_rows) - len(ratios),
            "mean_active_vote_delta": safe_mean(row.get("active_vote_delta") for row in rows),
        })

    coverage = {
        **identity,
        "candidate_count": len(candidates),
        "optimizer_candidate_count": sum(row.get("candidate_pool_source") == "optimizer" for row in candidates),
        "attempt_count": len(attempts),
        "complete_attempt_count": sum(row.get("attempt_complete") for row in attempts),
        "full_distance_reconstruction_count": sum(row.get("distance_reconstruction_available") for row in candidates),
        "proposal_metadata_count": sum(row.get("proposal_metadata_available") for row in candidates),
        "has_final_predictions": (run_dir / "test_final_predictions.jsonl").exists(),
        "has_checkpoint": (run_dir / "training_checkpoint.json").exists(),
    }
    return candidates, attempts, agent_rows, coverage


def build_bin_summary(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row.get("candidate_pool_source") != "optimizer":
            continue
        label = distance_bin(optional_float(row.get("prompt_change_ratio_parent")))
        if label:
            groups[(row.get("run_root"), row.get("task"), row.get("setting"), row.get("seed"), label)].append(row)
    totals = Counter((row.get("run_root"), row.get("task"), row.get("setting"), row.get("seed")) for row in candidates if row.get("candidate_pool_source") == "optimizer")
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        root, task, setting, seed, label = key
        count = len(rows)
        result.append({
            "run_root": root, "task": task, "setting": setting, "seed": seed,
            "distance_bin": label, "candidate_count": count,
            "fraction_of_all_candidates": count / totals[(root, task, setting, seed)],
            "guard_feasible_rate": safe_mean(
                1.0 if as_bool(row.get("original_accuracy_invalid_dependence_feasible")) and as_bool(row.get("trajectory_feasible")) else 0.0
                for row in rows
            ),
            "pareto_feasible_rate": safe_mean(1.0 if as_bool(row.get("pareto_feasible")) else 0.0 for row in rows),
            "pareto_retained_rate": safe_mean(1.0 if as_bool(row.get("retained_in_beam")) else 0.0 for row in rows),
            "active_top1_rate": safe_mean(1.0 if as_bool(row.get("became_active_top1")) else 0.0 for row in rows),
            "vote_gain_rate_probability": safe_mean(1.0 if fnum(row.get("vote_delta")) > 0 else 0.0 for row in rows),
            "accuracy_regression_probability": safe_mean(1.0 if fnum(row.get("accuracy_delta")) < 0 else 0.0 for row in rows),
            "mean_vote_delta": safe_mean(row.get("vote_delta") for row in rows),
            "mean_accuracy_delta": safe_mean(row.get("accuracy_delta") for row in rows),
            "mean_vote_margin_delta": safe_mean(row.get("vote_margin_delta") for row in rows),
            "mean_pivotal_rescue_rate": safe_mean(row.get("pivotal_rescue_rate") for row in rows),
            "mean_pivotal_loss_rate": safe_mean(row.get("pivotal_loss_rate") for row in rows),
            "mean_shared_error_net_gain": safe_mean(row.get("boundary_shared_error_net_gain") for row in rows),
            "mean_behavior_transition_l1": safe_mean(row.get("behavior_transition_l1") for row in rows),
        })
    return result


def paired_guard_comparison(attempts: Sequence[Dict[str, Any]], candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    residual = "shared_vote_error_pareto_tcs_residual_specialization"
    cycle = "shared_vote_error_pareto_tcs_residual_cycle_guard"
    keys = sorted({(row.get("run_root"), row.get("task"), row.get("seed")) for row in attempts})
    result = []
    for root, task, seed in keys:
        subsets = {}
        for setting in (residual, cycle):
            attempt_rows = [r for r in attempts if (r.get("run_root"), r.get("task"), r.get("seed"), r.get("setting")) == (root, task, seed, setting)]
            candidate_rows = [r for r in candidates if (r.get("run_root"), r.get("task"), r.get("seed"), r.get("setting"), r.get("candidate_pool_source")) == (root, task, seed, setting, "optimizer")]
            if not attempt_rows:
                continue
            ratios = [value for row in candidate_rows if (value := optional_float(row.get("prompt_change_ratio_parent"))) is not None]
            active_ratios = [value for row in attempt_rows if as_bool(row.get("active_prompt_changed")) and (value := optional_float(row.get("active_candidate_change_ratio"))) is not None]
            final_by_agent = []
            for agent_id in sorted({row.get("agent_id") for row in attempt_rows}, key=str):
                agent_attempts = [row for row in attempt_rows if row.get("agent_id") == agent_id]
                final_by_agent.append(agent_attempts[-1])
            subsets[setting] = {
                "optimizer_candidate_count": len(candidate_rows),
                "prompt_distance_mean": safe_mean(ratios),
                "prompt_distance_p50": quantile(ratios, 0.50),
                "prompt_distance_p75": quantile(ratios, 0.75),
                "prompt_distance_p90": quantile(ratios, 0.90),
                "large_shift_candidate_rate": safe_mean(1.0 if as_bool(row.get("large_shift")) else 0.0 for row in candidate_rows),
                "active_update_rate": safe_mean(1.0 if as_bool(row.get("active_prompt_changed")) else 0.0 for row in attempt_rows),
                "active_candidate_change_ratio": safe_mean(active_ratios),
                "active_cumulative_path_length": safe_mean(row.get("active_prompt_cumulative_path_length") for row in final_by_agent),
                "active_cumulative_path_known_agent_count": sum(optional_float(row.get("active_prompt_cumulative_path_length")) is not None for row in final_by_agent),
                "active_distance_from_initial": safe_mean(row.get("direct_distance_from_initial") for row in final_by_agent),
                "active_distance_known_agent_count": sum(optional_float(row.get("direct_distance_from_initial")) is not None for row in final_by_agent),
                "dependence_rejection_rate": safe_mean(1.0 if row.get("candidate_outcome") == "dependence_guard_rejected" else 0.0 for row in candidate_rows),
                "trust_rejection_rate": safe_mean(1.0 if row.get("candidate_outcome") == "unsupported_large_prompt_shift" else 0.0 for row in candidate_rows),
                "cycle_rejection_rate": safe_mean(1.0 if row.get("candidate_outcome") in {"exact_prompt_cycle_rejected", "behavior_cycle_rejected"} else 0.0 for row in candidate_rows),
                "pareto_feasible_rate": safe_mean(1.0 if as_bool(row.get("pareto_feasible")) else 0.0 for row in candidate_rows),
                "optimizer_underfill_rate": safe_mean(1.0 if fnum(row.get("final_optimizer_candidate_count")) < fnum(row.get("raw_student_candidate_count")) else 0.0 for row in attempt_rows),
                "pareto_not_retained_rate": safe_mean(1.0 if row.get("candidate_outcome") == "pareto_not_retained" else 0.0 for row in candidate_rows),
                "existing_beam_win_rate": safe_mean(1.0 if as_bool(row.get("existing_beam_won")) else 0.0 for row in attempt_rows),
            }
        if len(subsets) != 2:
            continue
        row = {"run_root": root, "task": task, "seed": seed}
        for metric, residual_value in subsets[residual].items():
            cycle_value = subsets[cycle].get(metric)
            row[f"residual_{metric}"] = residual_value
            row[f"cycle_trust_{metric}"] = cycle_value
            row[f"delta_{metric}"] = (
                cycle_value - residual_value
                if isinstance(cycle_value, (int, float)) and isinstance(residual_value, (int, float))
                else None
            )
        result.append(row)
    return result


def probability_summary(candidates: Sequence[Dict[str, Any]], attempts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    optimizer = [row for row in candidates if row.get("candidate_pool_source") == "optimizer" and optional_float(row.get("prompt_change_ratio_parent")) is not None]
    large = [row for row in optimizer if fnum(row.get("prompt_change_ratio_parent")) > 0.45]
    active = [row for row in optimizer if as_bool(row.get("became_active_top1"))]
    active_distances = [
        value for row in active
        if (value := optional_float(row.get("prompt_change_ratio_active_before"))) is not None
    ]
    ratios = [fnum(row.get("prompt_change_ratio_parent")) for row in optimizer]
    trust_rejections = [row for row in optimizer if row.get("candidate_outcome") == "unsupported_large_prompt_shift"]
    by_bin = {}
    for _, _, label in DISTANCE_BINS:
        rows = [row for row in optimizer if distance_bin(optional_float(row.get("prompt_change_ratio_parent"))) == label]
        by_bin[label] = {
            "candidate_count": len(rows),
            "p_pareto_feasible": safe_mean(1.0 if as_bool(row.get("pareto_feasible")) else 0.0 for row in rows),
            "p_active_top1": safe_mean(1.0 if as_bool(row.get("became_active_top1")) else 0.0 for row in rows),
            "p_vote_gain": safe_mean(1.0 if fnum(row.get("vote_delta")) > 0 else 0.0 for row in rows),
            "p_accuracy_regression": safe_mean(1.0 if fnum(row.get("accuracy_delta")) < 0 else 0.0 for row in rows),
        }
    agent_attempts: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        agent_attempts[(row.get("run_root"), row.get("task"), row.get("setting"), row.get("seed"), row.get("agent_id"))].append(row)
    majority_agents = 0
    eligible_agents = 0
    for rows in agent_attempts.values():
        rates = [fnum(row.get("large_shift_candidate_rate")) for row in rows if optional_float(row.get("large_shift_candidate_rate")) is not None]
        if rates:
            eligible_agents += 1
            majority_agents += safe_mean(1.0 if value > 0 else 0.0 for value in rates) > 0.5
    return {
        "optimizer_candidate_count": len(optimizer),
        "known_parent_distance_count": len(ratios),
        "p_change_ratio_gt_0_45_given_optimizer_candidate": len(large) / len(optimizer) if optimizer else None,
        "parent_distance_mean": safe_mean(ratios),
        "parent_distance_p50": quantile(ratios, 0.50),
        "parent_distance_p75": quantile(ratios, 0.75),
        "parent_distance_p90": quantile(ratios, 0.90),
        "p_trust_rejection_given_change_ratio_gt_0_45": sum(row.get("candidate_outcome") == "unsupported_large_prompt_shift" for row in large) / len(large) if large else None,
        "active_top1_optimizer_count": len(active),
        "active_before_distance_known_count": len(active_distances),
        "active_before_distance_coverage": len(active_distances) / len(active) if active else None,
        "active_before_distance_mean": safe_mean(active_distances),
        "active_before_distance_p50": quantile(active_distances, 0.50),
        "active_before_distance_p75": quantile(active_distances, 0.75),
        "active_before_distance_p90": quantile(active_distances, 0.90),
        "p_change_ratio_gt_0_45_given_active_top1": sum(value > 0.45 for value in active_distances) / len(active_distances) if active_distances else None,
        "trust_rejection_count": len(trust_rejections),
        "trust_rejection_rate_all_optimizer_candidates": len(trust_rejections) / len(optimizer) if optimizer else None,
        "active_update_rate": safe_mean(1.0 if as_bool(row.get("active_prompt_changed")) else 0.0 for row in attempts),
        "agents_with_large_shift_in_majority_of_attempts": majority_agents,
        "agents_with_candidate_attempts": eligible_agents,
        "distance_bins": by_bin,
        "optimizer_candidate_outcomes": dict(sorted(Counter(
            str(row.get("candidate_outcome") or "unknown") for row in optimizer
        ).items())),
        "all_candidate_outcomes": dict(sorted(Counter(
            str(row.get("candidate_outcome") or "unknown") for row in candidates
        ).items())),
    }


def mechanism_summary(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    optimizer = [row for row in candidates if row.get("candidate_pool_source") == "optimizer"]
    known = [row for row in optimizer if as_bool(row.get("proposal_metadata_available"))]
    large = [row for row in optimizer if fnum(row.get("prompt_change_ratio_parent"), -1.0) > 0.45]
    large_known = [row for row in large if as_bool(row.get("proposal_metadata_available"))]
    large_modified_known = [row for row in large_known if as_bool(row.get("modified_mechanism_present")) is not None]
    large_preserved_known = [row for row in large_known if optional_float(row.get("preserved_mechanism_count")) is not None]
    small_behavior_large = [
        row for row in optimizer
        if 0.0 <= fnum(row.get("prompt_change_ratio_parent"), -1.0) < 0.20
        and fnum(row.get("behavior_transition_l1")) > 0.25
    ]
    family_groups: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    for row in known:
        family = str(row.get("target_residual_family") or "").strip()
        if family:
            family_groups[(row.get("run_root"), row.get("task"), row.get("setting"), row.get("seed"), row.get("agent_id"))].append(family)
    repeated = sum(len(values) - len(set(values)) for values in family_groups.values())
    family_mentions = sum(len(values) for values in family_groups.values())
    return {
        "optimizer_candidate_count": len(optimizer),
        "proposal_metadata_known_count": len(known),
        "proposal_metadata_coverage": len(known) / len(optimizer) if optimizer else None,
        "large_shift_count": len(large),
        "large_shift_metadata_known_count": len(large_known),
        "large_shift_metadata_coverage": len(large_known) / len(large) if large else None,
        "large_shift_modified_mechanism_known_count": len(large_modified_known),
        "large_shift_with_modified_mechanism_rate_known": safe_mean(
            1.0 if as_bool(row.get("modified_mechanism_present")) else 0.0 for row in large_modified_known
        ),
        "large_shift_preserved_mechanism_known_count": len(large_preserved_known),
        "large_shift_missing_preserved_mechanism_rate_known": safe_mean(
            1.0 if fnum(row.get("preserved_mechanism_count")) < 1.0 else 0.0 for row in large_preserved_known
        ),
        "small_text_large_behavior_threshold": 0.25,
        "small_text_large_behavior_count": len(small_behavior_large),
        "target_residual_family_mention_count": family_mentions,
        "repeated_target_residual_family_count": repeated,
        "repeated_target_residual_family_rate": repeated / family_mentions if family_mentions else None,
    }


def render_report(summary: Dict[str, Any], paired: Sequence[Dict[str, Any]]) -> str:
    probabilities = summary["probabilities"]
    pilot_probabilities = summary.get("probabilities_by_run_root", {}).get(
        "runs_vote_v7_strict_pilot_0d6093e", probabilities
    )
    coverage = summary["coverage"]
    mechanism = summary["mechanism_analysis"]
    rewrite_problem = bool(
        fnum(pilot_probabilities.get("p_change_ratio_gt_0_45_given_optimizer_candidate")) > 0.5
        and fnum(pilot_probabilities.get("parent_distance_p75")) > 0.45
        and int(pilot_probabilities.get("agents_with_large_shift_in_majority_of_attempts") or 0)
        > int(pilot_probabilities.get("agents_with_candidate_attempts") or 0) / 2
    )
    trust_too_strict = fnum(pilot_probabilities.get("trust_rejection_rate_all_optimizer_candidates")) > 0.20
    report = [
        "# Prompt Evolution Audit", "",
        "## 1. Data coverage and missing fields", "",
        f"- Runs discovered: {coverage['run_count']}",
        f"- Evaluated candidates: {coverage['candidate_count']}",
        f"- Optimizer candidates: {probabilities['optimizer_candidate_count']}",
        f"- Full five-distance reconstruction coverage: {coverage['full_distance_reconstruction_rate']:.3f}",
        f"- Proposal-mechanism metadata coverage: {coverage['proposal_metadata_rate']:.3f}",
        f"- Completed runs / checkpoint-bearing runs: {coverage['runs_with_final_predictions']} / {coverage['runs_with_checkpoint']}",
        "- Parent distance, guard status, Pareto status, outcome, accept-count sequence, and warmup/support conditions are reconstructable.",
        "- Full candidate text and proposal modified-mechanism fields are absent for many rejected/non-retained candidates; their derived lengths and non-parent distances are left blank rather than guessed.", "",
        "## 2. Candidate prompt-distance distribution", "",
        "Pilot-only statistics (the partial formal run is excluded):",
        f"- Mean / p50 / p75 / p90 parent distance: {fmt(pilot_probabilities.get('parent_distance_mean'))} / {fmt(pilot_probabilities.get('parent_distance_p50'))} / {fmt(pilot_probabilities.get('parent_distance_p75'))} / {fmt(pilot_probabilities.get('parent_distance_p90'))}",
        f"- P(change_ratio > 0.45 | optimizer candidate): {fmt(pilot_probabilities.get('p_change_ratio_gt_0_45_given_optimizer_candidate'))}",
        f"- Evidence supports excessive full-rewrite jumps: **{rewrite_problem}**", "",
        "## 3. Active prompt-distance distribution", "",
        f"- P(change_ratio > 0.45 | active top-1 optimizer candidate): {fmt(pilot_probabilities.get('p_change_ratio_gt_0_45_given_active_top1'))}",
        f"- Active-before distance coverage: {pilot_probabilities.get('active_before_distance_known_count')} / {pilot_probabilities.get('active_top1_optimizer_count')} ({fmt(pilot_probabilities.get('active_before_distance_coverage'))})",
        f"- Known active-before mean / p50 / p75 / p90: {fmt(pilot_probabilities.get('active_before_distance_mean'))} / {fmt(pilot_probabilities.get('active_before_distance_p50'))} / {fmt(pilot_probabilities.get('active_before_distance_p75'))} / {fmt(pilot_probabilities.get('active_before_distance_p90'))}",
        f"- Active prompt update rate: {fmt(pilot_probabilities.get('active_update_rate'))}",
        "- Agent-level cumulative path and direct distance are reported separately in `prompt_agent_path_summary.csv`; blank path segments indicate unavailable full prompt text, not zero movement.", "",
        "## 4. Large-shift candidate analysis", "",
        f"- Large optimizer candidates: {int(round(fnum(pilot_probabilities.get('p_change_ratio_gt_0_45_given_optimizer_candidate')) * int(pilot_probabilities.get('optimizer_candidate_count') or 0)))}",
        f"- P(trust rejection | change_ratio > 0.45): {fmt(pilot_probabilities.get('p_trust_rejection_given_change_ratio_gt_0_45'))}",
        "- Warmup exemptions and five support predicates are reconstructed from chronological active updates, run config, and logged candidate metrics.", "",
        "## 5. Trust-region rejection decomposition", "",
        f"- Unsupported-large-shift rejections: {pilot_probabilities.get('trust_rejection_count')}",
        f"- Trust rejection / optimizer candidate: {fmt(pilot_probabilities.get('trust_rejection_rate_all_optimizer_candidates'))}",
        "- Individual rejected candidates and failed support predicates are available in `prompt_candidate_trajectory.csv`.", "",
        "## 6. Full candidate-selection funnel", "",
        "Candidate outcomes are mutually exclusive and separate original guards, cycle/trust guards, Pareto retention, existing-beam wins, and active top-1 selection. Evaluated candidates cannot represent pre-evaluation schema/redundancy-filtered proposals; attempt-level raw/final counts preserve that funnel stage.", "",
        f"Pilot optimizer outcomes: `{json.dumps(pilot_probabilities.get('optimizer_candidate_outcomes', {}), sort_keys=True)}`", "",
        "## 7. Residual vs Cycle/Trust paired comparison", "",
        f"- Matched task x seed pairs: {len(paired)}",
        "- See `prompt_guard_paired_comparison.csv`; active-update differences are decomposed into underfill, dependence, cycle, trust, Pareto non-retention, and existing-beam wins.", "",
        "## 8. Agent cumulative evolution paths", "",
        "`prompt_agent_path_summary.csv` distinguishes cumulative adjacent movement from direct distance to the initial prompt, exposing both gradual drift and return-toward-origin paths.", "",
        "## 9. Mechanism-contract analysis", "",
        f"- Proposal metadata coverage is {coverage['proposal_metadata_rate']:.3f}; this is insufficient for an all-candidate semantic mechanism analysis.",
        f"- Large-shift proposal metadata coverage: {mechanism.get('large_shift_metadata_known_count')} / {mechanism.get('large_shift_count')} ({fmt(mechanism.get('large_shift_metadata_coverage'))}).",
        f"- Explicit modified-mechanism coverage among large shifts: {mechanism.get('large_shift_modified_mechanism_known_count')} / {mechanism.get('large_shift_count')}; present rate within that known subset: {fmt(mechanism.get('large_shift_with_modified_mechanism_rate_known'))}.",
        f"- Explicit preserved-mechanism coverage among large shifts: {mechanism.get('large_shift_preserved_mechanism_known_count')} / {mechanism.get('large_shift_count')}; missing-list rate within that known subset: {fmt(mechanism.get('large_shift_missing_preserved_mechanism_rate_known'))}.",
        f"- Small-text/large-behavior cases (distance <0.20 and transition L1 >{mechanism.get('small_text_large_behavior_threshold')}): {mechanism.get('small_text_large_behavior_count')}.",
        f"- Repeated target-family mentions: {mechanism.get('repeated_target_residual_family_count')} / {mechanism.get('target_residual_family_mention_count')} ({fmt(mechanism.get('repeated_target_residual_family_rate'))}).",
        "- No new LLM labeling was used. Preserved-mechanism counts are reported only where archived state metadata can be matched by prompt hash.", "",
        "## 10. Interpretation of evolution speed", "",
        "Evolution-speed conclusions use active update rate, active step distance, cumulative path, direct initial distance, and logged behavior deltas. Missing full prompt bodies are explicitly separated from genuine zero-distance updates.", "",
        "## 11. Whether current trust region is too strict", "",
        f"- Trust-region-too-strict criterion (>20% optimizer rejection): **{trust_too_strict}**.",
        f"- Observed pilot rate: {fmt(pilot_probabilities.get('trust_rejection_rate_all_optimizer_candidates'))}.", "",
        "## 12. Whether patch-based generation needs a separate v8 experiment", "",
        (
            "A separate v8 patch-generation experiment is justified by the offline thresholds; do not alter or mix the current v7 formal run."
            if rewrite_problem or trust_too_strict
            else "The offline thresholds do not justify changing v7 mid-experiment. Patch-based generation may still be tested later as a separate v8 ablation, using a new commit and output root."
        ), "",
    ]
    return "\n".join(report)


def fmt(value: Any) -> str:
    parsed = optional_float(value)
    return "NA" if parsed is None else f"{parsed:.4f}"


def analyze(run_roots: Sequence[Path], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_candidates: List[Dict[str, Any]] = []
    all_attempts: List[Dict[str, Any]] = []
    all_agents: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for root, run_dir in discover_runs(run_roots):
        candidates, attempts, agents, coverage = analyze_run(root, run_dir)
        all_candidates.extend(candidates)
        all_attempts.extend(attempts)
        all_agents.extend(agents)
        coverage_rows.append(coverage)

    bin_rows = build_bin_summary(all_candidates)
    paired_rows = paired_guard_comparison(all_attempts, all_candidates)
    probabilities = probability_summary(all_candidates, all_attempts)
    mechanisms = mechanism_summary(all_candidates)
    root_names = sorted({row.get("run_root") for row in all_candidates} | {row.get("run_root") for row in all_attempts})
    probabilities_by_run_root = {
        str(root_name): probability_summary(
            [row for row in all_candidates if row.get("run_root") == root_name],
            [row for row in all_attempts if row.get("run_root") == root_name],
        )
        for root_name in root_names
    }
    full_count = sum(int(row.get("full_distance_reconstruction_count") or 0) for row in coverage_rows)
    candidate_count = sum(int(row.get("candidate_count") or 0) for row in coverage_rows)
    proposal_count = sum(int(row.get("proposal_metadata_count") or 0) for row in coverage_rows)
    summary = {
        "analysis_mode": "offline_only",
        "api_calls": 0,
        "model_calls": 0,
        "run_roots": [str(path.resolve()) for path in run_roots],
        "coverage": {
            "run_count": len(coverage_rows),
            "candidate_count": candidate_count,
            "attempt_count": len(all_attempts),
            "complete_attempt_count": sum(int(row.get("complete_attempt_count") or 0) for row in coverage_rows),
            "full_distance_reconstruction_count": full_count,
            "full_distance_reconstruction_rate": full_count / candidate_count if candidate_count else 0.0,
            "proposal_metadata_count": proposal_count,
            "proposal_metadata_rate": proposal_count / candidate_count if candidate_count else 0.0,
            "runs_with_final_predictions": sum(bool(row.get("has_final_predictions")) for row in coverage_rows),
            "runs_with_checkpoint": sum(bool(row.get("has_checkpoint")) for row in coverage_rows),
            "per_run": coverage_rows,
        },
        "probabilities": probabilities,
        "probabilities_by_run_root": probabilities_by_run_root,
        "mechanism_analysis": mechanisms,
        "trust_rejections": [
            {key: row.get(key) for key in (
                "run_root", "task", "setting", "seed", "epoch", "step", "agent_id",
                "candidate_id", "prompt_change_ratio_parent", "agent_accept_count_before",
                "large_shift_warmup_exempt", "large_shift_vote_support_passed",
                "large_shift_accuracy_support_passed", "large_shift_vote_loss_support_passed",
                "large_shift_pivotal_loss_support_passed",
                "large_shift_shared_error_support_passed", "vote_delta", "accuracy_delta",
                "vote_loss_rate", "pivotal_loss_rate", "shared_error_rescue_score",
                "shared_error_creation_score",
            )}
            for row in all_candidates if row.get("candidate_outcome") == "unsupported_large_prompt_shift"
        ],
        "runtime_logging_change_required": False,
        "runtime_logging_decision": "Existing logs answer the core distance/trust/path questions. Missing all-candidate prompt text and proposal mechanism metadata are reported as coverage limitations; system.py was not modified.",
    }

    write_csv(output_dir / "prompt_candidate_trajectory.csv", all_candidates, CANDIDATE_FIELDS)
    write_csv(output_dir / "prompt_update_attempt_summary.csv", all_attempts, ATTEMPT_FIELDS)
    write_csv(output_dir / "prompt_change_bin_summary.csv", bin_rows, [
        "run_root", "task", "setting", "seed", "distance_bin", "candidate_count",
        "fraction_of_all_candidates", "guard_feasible_rate", "pareto_feasible_rate",
        "pareto_retained_rate", "active_top1_rate", "vote_gain_rate_probability",
        "accuracy_regression_probability", "mean_vote_delta", "mean_accuracy_delta",
        "mean_vote_margin_delta", "mean_pivotal_rescue_rate", "mean_pivotal_loss_rate",
        "mean_shared_error_net_gain", "mean_behavior_transition_l1",
    ])
    write_csv(output_dir / "prompt_guard_paired_comparison.csv", paired_rows, ["run_root", "task", "seed"])
    write_csv(output_dir / "prompt_agent_path_summary.csv", all_agents, [
        "run_root", "run_dir", "task", "setting", "seed", "agent_id",
        "update_attempt_count", "active_update_count", "active_update_rate",
        "initial_prompt_hash", "final_active_prompt_hash", "unique_active_prompt_count",
        "mean_active_change_ratio", "median_active_change_ratio", "p90_active_change_ratio",
        "max_active_change_ratio", "large_active_update_count", "large_active_update_rate",
        "final_direct_distance_from_initial", "active_cumulative_path_length",
        "path_to_direct_ratio", "unknown_active_step_distance_count",
    ])
    (output_dir / "prompt_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "PROMPT_EVOLUTION_AUDIT.md").write_text(
        render_report(summary, paired_rows), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", action="append", required=True, help="Run root or individual run directory; repeatable.")
    parser.add_argument("--output_dir", required=True, help="Directory for offline audit artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze([Path(value) for value in args.run_root], Path(args.output_dir))
    print(json.dumps({
        "run_count": summary["coverage"]["run_count"],
        "candidate_count": summary["coverage"]["candidate_count"],
        "api_calls": summary["api_calls"],
        "output_dir": str(Path(args.output_dir).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
