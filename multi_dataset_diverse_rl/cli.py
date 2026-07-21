import asyncio
import hashlib
import json
import os
import random
import sys
import time
import uuid

import numpy as np

from .config import Config, build_parser
from .state_conditioned import paired_c0_metrics, question_state, state_conditioned_validation_key
from .tasks import infer_option_count
from .utils import canonical_aggregation_mode, ensure_dir, load_jsonl, set_seed


LEGACY_QUESTION_KEYS = ["question", "input", "query", "problem"]
LEGACY_ANSWER_KEYS = ["answer", "output", "target", "label", "response"]
MARS_QUESTION_KEYS = ["question", "input", "query", "problem", "prompt"]
MARS_ANSWER_KEYS = ["answer", "target", "gold", "gold_answer", "label", "output"]
TASK_KEYS = ["task", "task_name", "category", "subject", "bbh_task"]


def _first_present(record, keys):
    for key in keys:
        if key in record and record.get(key) is not None:
            value = record.get(key)
            if isinstance(value, str):
                if value.strip():
                    return value
            else:
                return value
    return None


def build_dataset(raw_records, dataset_format="legacy"):
    fmt = str(dataset_format or "legacy").strip().lower()
    if fmt == "mars":
        q_keys = MARS_QUESTION_KEYS
        a_keys = MARS_ANSWER_KEYS
    else:
        q_keys = LEGACY_QUESTION_KEYS
        a_keys = LEGACY_ANSWER_KEYS

    rows = []
    for idx, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"Cannot parse dataset record at index {idx}: expected object, got {type(record).__name__}")
        question = _first_present(record, q_keys)
        answer = _first_present(record, a_keys)
        if question is None or answer is None:
            available = ", ".join(sorted(str(k) for k in record.keys()))
            raise ValueError(
                f"Cannot find question/answer fields in record index {idx} "
                f"for dataset_format={fmt}. available_keys=[{available}]"
            )
        row = {"question": str(question), "answer": answer}
        task = _first_present(record, TASK_KEYS)
        if task is not None:
            row["task"] = str(task)
        for meta_key in ["subject", "category", "task_name", "bbh_task"]:
            if meta_key in record and record.get(meta_key) is not None:
                row[meta_key] = str(record.get(meta_key))
        rows.append(row)
    return rows


def split_train_validation(raw_train, cfg):
    records = list(raw_train)
    if not records:
        return [], []
    ratio = max(0.0, min(0.8, float(cfg.val_split_ratio)))
    requested_val = int(cfg.val_size or 0)
    if requested_val <= 0:
        requested_val = int(round(len(records) * ratio))
    requested_val = max(1, min(requested_val, max(1, len(records) - 1)))
    rng = random.Random(int(cfg.seed))
    indices = list(range(len(records)))
    rng.shuffle(indices)
    train_records = [records[i] for i in indices[requested_val:]]
    val_records = [records[i] for i in indices[:requested_val]]
    return train_records, val_records


def build_candidate_eval_pool(train_data, val_data, cfg):
    # Keep validation independent: prompt candidates are optimized only on
    # training examples. ``val_data`` remains in the signature for callers
    # from older scripts, but is deliberately not a candidate-pool source.
    source = list(train_data or [])
    if not source:
        return []
    pool_size = min(max(1, int(cfg.candidate_eval_pool_size or 1)), len(source))
    rng = random.Random(int(cfg.seed) + int(cfg.candidate_eval_seed_offset))
    indices = list(range(len(source)))
    rng.shuffle(indices)
    return [source[i] for i in indices[:pool_size]]


def select_competence_probe_indices(train_data, cfg, saved_indices=None):
    """Select one deterministic optimization-split probe for the entire run."""
    if saved_indices is not None:
        indices = [int(value) for value in saved_indices]
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(train_data) for index in indices):
            raise ValueError("competence probe checkpoint indices are invalid for the optimization split")
        return indices
    size = int(getattr(cfg, "competence_probe_size", 0) or 0)
    size = len(train_data) if size <= 0 else min(size, len(train_data))
    indices = list(range(len(train_data)))
    random.Random(int(cfg.seed) + int(getattr(cfg, "competence_probe_seed_offset", 7000))).shuffle(indices)
    return indices[:size]


def _stratified_sample(records, size, rng):
    if size <= 0 or not records:
        return []
    buckets = {}
    for record in records:
        key = (
            record.get("subject")
            or record.get("task")
            or record.get("task_name")
            or record.get("category")
            or record.get("bbh_task")
            or ""
        )
        buckets.setdefault(str(key), []).append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    out = []
    while len(out) < size and keys:
        progressed = False
        for key in list(keys):
            bucket = buckets[key]
            if not bucket:
                keys.remove(key)
                continue
            out.append(bucket.pop())
            progressed = True
            if len(out) >= size:
                break
        if not progressed:
            break
    return out


def select_candidate_eval_batch(train_data, candidate_eval_pool, cfg, epoch, step, anchor_idx=None):
    batch_size = max(0, int(cfg.candidate_eval_batch_size or 0))
    repeats = max(1, int(cfg.candidate_eval_repeats or 1))
    if batch_size <= 0:
        return []
    strategy = str(cfg.candidate_eval_strategy or "random").lower()
    batches = []
    base_seed = int(cfg.seed) + int(cfg.candidate_eval_seed_offset) + int(epoch) * 100000 + int(step) * 97
    source = list(train_data if strategy == "random" else (candidate_eval_pool or train_data))
    if not source:
        return []

    # A single deterministic permutation lets repeats cover distinct examples
    # until the source is exhausted, instead of independently re-shuffling into
    # heavily overlapping batches.
    rng = random.Random(base_seed)
    permutation = list(range(len(source)))
    rng.shuffle(permutation)
    remaining = list(source)
    for repeat in range(repeats):
        if strategy == "stratified":
            sample_source = remaining or list(source)
            batch = _stratified_sample(sample_source, min(batch_size, len(sample_source)), random.Random(base_seed + repeat))
            selected_ids = {id(row) for row in batch}
            remaining = [row for row in remaining if id(row) not in selected_ids]
        else:
            target_size = min(batch_size, len(source))
            indices = []
            if strategy == "random" and anchor_idx is not None and 0 <= int(anchor_idx) < len(train_data):
                anchor = int(anchor_idx)
                indices.append(anchor)
            start = repeat * target_size
            for offset in range(len(source)):
                index = permutation[(start + offset) % len(source)]
                if index not in indices:
                    indices.append(index)
                if len(indices) >= target_size:
                    break
            batch = [source[i] for i in indices[:target_size]]
        batches.extend(batch)
    return batches


def select_state_conditioned_candidate_eval_batch(
    train_data, candidate_eval_pool, cfg, epoch, step, state_records=None
):
    """Build disjoint representative/coverage/conversion pools for V9.

    State labels come only from already-recorded train rollouts. Unknown pool
    items remain eligible for the natural representative sample and are never
    assigned a fabricated state.
    """
    source = list(candidate_eval_pool or train_data or [])
    batch_size = max(0, int(cfg.candidate_eval_batch_size or 0))
    if not source or batch_size <= 0:
        return []
    by_hash = {
        str(row.get("question_hash", "")): row
        for row in (state_records or [])
        if isinstance(row, dict) and str(row.get("question_hash", ""))
    }
    rng = random.Random(int(cfg.seed) + int(cfg.candidate_eval_seed_offset) + int(epoch) * 100000 + int(step) * 97)
    shuffled = list(source)
    rng.shuffle(shuffled)
    used = set()
    representative_budget = min(
        batch_size, max(0, int(getattr(cfg, "candidate_batch_representative_size", 12)))
    )
    targeted_budget = max(0, batch_size - representative_budget)
    coverage_budget = min(
        targeted_budget, max(0, int(getattr(cfg, "candidate_batch_coverage_size", 6)))
    )
    conversion_budget = min(
        targeted_budget - coverage_budget,
        max(0, int(getattr(cfg, "candidate_batch_conversion_size", 6))),
    )

    def record_hash(record):
        return hashlib.sha1(str(record.get("question", "")).encode("utf-8")).hexdigest()[:12]

    def take(pool_name, count, predicate):
        selected = []
        for record in shuffled:
            key = record_hash(record)
            if key in used or not predicate(by_hash.get(key)):
                continue
            state_row = by_hash.get(key) or {}
            state = question_state(state_row.get("metrics", {}).get("gold_vote_count", state_row.get("gold_vote_count", 0)))
            selected.append({
                **record,
                "_candidate_pool": pool_name,
                "_candidate_state": state,
                "_option_count": infer_option_count(record.get("question", "")),
            })
            used.add(key)
            if len(selected) >= count:
                break
        return selected

    coverage = take(
        "coverage", coverage_budget,
        lambda row: row is not None and question_state(row.get("metrics", {}).get("gold_vote_count", row.get("gold_vote_count", 0))) in {"C0", "C1"},
    )
    conversion_size = conversion_budget
    conversion_candidates = []
    for record in shuffled:
        key = record_hash(record)
        row = by_hash.get(key)
        if key in used or row is None:
            continue
        state = question_state(row.get("metrics", {}).get("gold_vote_count", row.get("gold_vote_count", 0)))
        if state != "C2":
            continue
        conversion_candidates.append({
            **record,
            "_candidate_pool": "conversion",
            "_candidate_state": "C2",
            "_option_count": infer_option_count(record.get("question", "")),
        })
    conversion = []
    option_groups = {}
    for record in conversion_candidates:
        option_groups.setdefault(int(record.get("_option_count", 0) or 0), []).append(record)
    while len(conversion) < conversion_size and any(option_groups.values()):
        for option_count in sorted(option_groups):
            if option_groups[option_count] and len(conversion) < conversion_size:
                record = option_groups[option_count].pop(0)
                conversion.append(record)
                used.add(record_hash(record))
    remaining = max(0, batch_size - len(coverage) - len(conversion))
    representative = take("representative", max(representative_budget, remaining), lambda row: True)
    result = coverage + conversion + representative
    if len(result) < batch_size:
        for record in shuffled:
            key = record_hash(record)
            if key in used:
                continue
            state_row = by_hash.get(key) or {}
            result.append({
                **record,
                "_candidate_pool": "representative",
                "_candidate_state": question_state(
                    state_row.get("metrics", {}).get("gold_vote_count", state_row.get("gold_vote_count", 0))
                ) if state_row else "UNKNOWN",
                "_option_count": infer_option_count(record.get("question", "")),
            })
            used.add(key)
            if len(result) >= batch_size:
                break
    return result[:batch_size]


def snapshot_agent_prompts(system):
    return [str(agent.current_prompt) for agent in system.agents]


def restore_agent_prompts(system, prompts, selected_epoch=None):
    if not prompts or len(prompts) != len(system.agents):
        raise ValueError(f"Cannot restore prompts: got {len(prompts) if prompts else 0}, expected {len(system.agents)}")
    for agent_id, (agent, prompt) in enumerate(zip(system.agents, prompts)):
        agent.current_prompt = str(prompt)
        prompt_hash = system._hash(prompt)
        existing = None
        for item in getattr(agent, "prompt_beam", []) or []:
            if system._hash(str(item.get("prompt", ""))) == prompt_hash:
                existing = dict(item)
                break
        if existing is None:
            existing = system._make_beam_item(str(prompt), None, {}, None, 0, candidate_id=f"selected_a{agent_id}_{prompt_hash}")
        existing["prompt"] = str(prompt)
        existing["id"] = str(existing.get("id", "")) or f"selected_a{agent_id}_{prompt_hash}"
        rest = [
            dict(item)
            for item in (getattr(agent, "prompt_beam", []) or [])
            if system._hash(str(item.get("prompt", ""))) != prompt_hash
        ]
        agent.prompt_beam = [existing] + rest[: max(0, int(getattr(system.cfg, "beam_size", 1) or 1) - 1)]
    if hasattr(system, "sync_prompt_history_current_state"):
        system.sync_prompt_history_current_state(
            event="restore_best_prompts",
            epoch="final",
            step=0,
            selected_epoch=selected_epoch,
        )


def vote_first_validation_key(epoch_record):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    epoch = int(epoch_record.get("epoch", 0) or 0)
    return (
        -float(val.get("vote_acc", 0.0) or 0.0),
        -float(val.get("mean_individual_acc", 0.0) or 0.0),
        -float(val.get("mean_vote_margin", -1.0) if val.get("mean_vote_margin") is not None else -1.0),
        float(val.get("mean_invalid_rate", 0.0) or 0.0),
        epoch,
    )


def vote_first_validation_key_fields(epoch_record):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    return [
        float(val.get("vote_acc", 0.0) or 0.0),
        float(val.get("mean_individual_acc", 0.0) or 0.0),
        float(val.get("mean_vote_margin", -1.0) if val.get("mean_vote_margin") is not None else -1.0),
        float(val.get("mean_invalid_rate", 0.0) or 0.0),
        int(epoch_record.get("epoch", 0) or 0),
    ]


def vote_first_tiebreak_key(epoch_record):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    return (
        -float(val.get("mean_individual_acc", 0.0) or 0.0),
        -float(val.get("mean_vote_margin", -1.0) if val.get("mean_vote_margin") is not None else -1.0),
        float(val.get("mean_invalid_rate", 0.0) or 0.0),
        int(epoch_record.get("epoch", 0) or 0),
    )


def vote_competence_first_validation_key(epoch_record):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    return (
        -float(val.get("plurality_vote_acc", val.get("vote_acc", 0.0)) or 0.0),
        -float(val.get("bottom2_mean_acc", 0.0) or 0.0),
        -float(val.get("coverage_depth_c2", 0.0) or 0.0),
        float(val.get("best_minus_bottom2_gap", 0.0) or 0.0),
        -float(val.get("mean_normalized_plurality_margin", val.get("mean_vote_margin", -1.0)) if val.get("mean_normalized_plurality_margin", val.get("mean_vote_margin")) is not None else -1.0),
        -float(val.get("mean_individual_acc", 0.0) or 0.0),
        float(val.get("mean_invalid_rate", 0.0) or 0.0),
        int(epoch_record.get("epoch", 0) or 0),
    )


def vote_generalization_first_validation_key(epoch_record):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    stable_tie_break = (
        str(epoch_record.get("method_version", "")) == "v8_stable_qd_lineage"
        and bool(epoch_record.get("validation_stable_specialization_tie_break_enabled", True))
    )
    return (
        -float(val.get("plurality_vote_acc", val.get("vote_acc", 0.0)) or 0.0),
        -float(val.get("mean_individual_acc", 0.0) or 0.0),
        -float(val.get("bottom2_mean_acc", 0.0) or 0.0),
        -float(val.get("coverage_depth_c1", 0.0) or 0.0),
        -float(val.get("coverage_depth_c2", 0.0) or 0.0),
        -float(
            val.get("mean_normalized_plurality_margin", val.get("mean_vote_margin", -1.0))
            if val.get("mean_normalized_plurality_margin", val.get("mean_vote_margin")) is not None
            else -1.0
        ),
        float(val.get("mean_invalid_rate", 0.0) or 0.0),
        -float(val.get("stable_specialization_score", 0.0) or 0.0) if stable_tie_break else 0.0,
        int(epoch_record.get("epoch", 0) or 0),
    )


def rollout_vote_first_validation_key(epoch_record):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    return (
        -float(val.get("plurality_vote_acc", val.get("vote_acc", 0.0)) or 0.0),
        -float(val.get("coverage_depth_c3", 0.0) or 0.0),
        -float(val.get("mean_individual_acc", 0.0) or 0.0),
        -float(val.get("bottom2_mean_acc", 0.0) or 0.0),
        -float(val.get("oracle_to_vote_conversion_rate", 0.0) or 0.0),
        -float(val.get("mean_gold_plurality_margin", val.get("mean_normalized_plurality_margin", -1.0)) or 0.0),
        float(val.get("dominant_wrong_concentration", 1.0) or 0.0),
        float(val.get("mean_invalid_rate", 0.0) or 0.0),
        -float(val.get("rollout_embedding_diversity", val.get("mean_embedding_diversity", 0.0)) or 0.0),
        int(epoch_record.get("epoch", 0) or 0),
    )


def state_conditioned_vote_first_validation_key(epoch_record):
    return state_conditioned_validation_key(epoch_record)


def rollout_method_metadata(cfg, system=None):
    if str(getattr(cfg, "method_version", "legacy")) not in {
        "v8_accuracy_rollout_embedding", "v8_rollout_qd_vote_ready", "v9_state_conditioned_error",
    }:
        return {}
    metadata = {
        "method_version": str(cfg.method_version),
        "beam_policy_version": str(cfg.beam_policy_version),
        "active_team_selector_version": str(cfg.active_team_selector_version),
        "candidate_generation_policy_version": str(cfg.candidate_generation_policy_version),
        "tcs_candidate_policy_version": str(cfg.tcs_candidate_policy_version),
        "archive_policy_version": str(cfg.archive_policy_version),
        "joint_quality_filter_version": str(cfg.joint_quality_filter_version),
        "probe_stability_version": str(cfg.probe_stability_version),
        "parent_selection_version": str(cfg.parent_selection_version),
        "mechanism_diversity_enabled": False,
        "mechanism_metadata_required": False,
        "mechanism_distance_used_for_selection": False,
        "mechanism_based_decision_count": int(getattr(system, "mechanism_based_decision_count", 0)) if system is not None else 0,
        "capability_labeling_enabled": False,
        "prompt_text_diversity_used": False,
    }
    if str(getattr(cfg, "method_version", "legacy")) == "v9_state_conditioned_error":
        metadata.update({
            "state_conditioned_enabled": True,
            "state_accuracy_tie_epsilon": float(cfg.state_accuracy_tie_epsilon),
            "state_coverage_enabled": bool(cfg.state_coverage_enabled),
            "state_c2_wrong_split_enabled": bool(cfg.state_c2_wrong_split_enabled),
            "state_trace_tiebreak_enabled": bool(cfg.state_trace_tiebreak_enabled),
            "state_joint_total_correct_slack_rate": float(cfg.state_joint_total_correct_slack_rate),
            "state_representative_capacity": int(cfg.state_representative_capacity),
            "candidate_batch_representative_size": int(cfg.candidate_batch_representative_size),
            "candidate_batch_coverage_size": int(cfg.candidate_batch_coverage_size),
            "candidate_batch_conversion_size": int(cfg.candidate_batch_conversion_size),
            "composite_rollout_distance_used_for_selection": False,
            "trace_diversity_role": "diagnostic_or_last_tiebreak_only",
        })
    return metadata


def is_better_validation_state(epoch_record, best_epoch_record, best_score, reward_mode, selection_mode, min_delta=0.0):
    if str(selection_mode or "existing").lower() == "state_conditioned_vote_first":
        if best_epoch_record is None:
            return True
        current_val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
        baseline_val = best_epoch_record.get("initial_validation", best_epoch_record.get("val", {}))
        baseline_mean = float(baseline_val.get("mean_individual_acc", 0.0) or 0.0)
        current_mean = float(current_val.get("mean_individual_acc", 0.0) or 0.0)
        epsilon = float(current_val.get("state_validation_accuracy_guard_epsilon", 0.02) or 0.02)
        if current_mean < baseline_mean - epsilon:
            return False
        return state_conditioned_vote_first_validation_key(epoch_record) < state_conditioned_vote_first_validation_key(best_epoch_record)
    if str(selection_mode or "existing").lower() == "rollout_vote_first":
        return best_epoch_record is None or rollout_vote_first_validation_key(epoch_record) < rollout_vote_first_validation_key(best_epoch_record)
    if str(selection_mode or "existing").lower() == "vote_generalization_first":
        return best_epoch_record is None or vote_generalization_first_validation_key(epoch_record) < vote_generalization_first_validation_key(best_epoch_record)
    if str(selection_mode or "existing").lower() == "vote_competence_first":
        return best_epoch_record is None or vote_competence_first_validation_key(epoch_record) < vote_competence_first_validation_key(best_epoch_record)
    if str(selection_mode or "existing").lower() == "vote_first":
        if best_epoch_record is None:
            return True
        current_val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
        best_val = best_epoch_record.get("val", {}) if isinstance(best_epoch_record.get("val", {}), dict) else {}
        vote_improvement = float(current_val.get("vote_acc", 0.0) or 0.0) - float(best_val.get("vote_acc", 0.0) or 0.0)
        threshold = max(0.0, float(min_delta or 0.0))
        if vote_improvement > threshold:
            return True
        if vote_improvement < -threshold:
            return False
        return vote_first_tiebreak_key(epoch_record) < vote_first_tiebreak_key(best_epoch_record)
    return validation_score(epoch_record, reward_mode) > float(best_score) + float(min_delta)


def write_selected_prompts(path, system, epoch, metric_name, validation_score, best_state_selection_mode="existing", epoch_record=None):
    prompts = snapshot_agent_prompts(system)
    val = epoch_record.get("val", {}) if isinstance(epoch_record, dict) and isinstance(epoch_record.get("val", {}), dict) else {}
    payload = {
        "selected_epoch": epoch,
        "early_stopping_metric": metric_name,
        "best_validation_score": validation_score,
        "best_state_selection_mode": str(best_state_selection_mode or "existing"),
        "best_state_selection_key": (
            list(rollout_vote_first_validation_key(epoch_record))
            if str(best_state_selection_mode or "existing").lower() == "rollout_vote_first" and isinstance(epoch_record, dict)
            else list(state_conditioned_vote_first_validation_key(epoch_record))
            if str(best_state_selection_mode or "existing").lower() == "state_conditioned_vote_first" and isinstance(epoch_record, dict)
            else list(vote_generalization_first_validation_key(epoch_record))
            if str(best_state_selection_mode or "existing").lower() == "vote_generalization_first" and isinstance(epoch_record, dict)
            else list(vote_competence_first_validation_key(epoch_record))
            if str(best_state_selection_mode or "existing").lower() == "vote_competence_first" and isinstance(epoch_record, dict)
            else vote_first_validation_key_fields(epoch_record)
            if str(best_state_selection_mode or "existing").lower() == "vote_first" and isinstance(epoch_record, dict)
            else None
        ),
        "selected_vote_acc": float(val.get("vote_acc", 0.0) or 0.0),
        "selected_oracle_acc": float(val.get("oracle_acc", 0.0) or 0.0),
        "selected_mean_individual_acc": float(val.get("mean_individual_acc", 0.0) or 0.0),
        "selected_mean_vote_margin": float(val.get("mean_vote_margin", -1.0) if val.get("mean_vote_margin") is not None else -1.0),
        "selected_mean_boundary_useful_diversity": float(val.get("mean_boundary_useful_diversity", 0.0) or 0.0),
        "selected_mean_invalid_rate": float(val.get("mean_invalid_rate", 0.0) or 0.0),
        "agents": [
            {
                "agent_id": i,
                "prompt_hash": system._hash(prompt),
                "prompt": prompt,
                "lineage_state": dict(getattr(system.agents[i], "lineage_state", {})),
            }
            for i, prompt in enumerate(prompts)
        ],
        "joint_team_metrics": dict(getattr(system, "latest_joint_team_metrics", {}) or {}),
    }
    if str(getattr(system.cfg, "method_version", "legacy")) == "v9_state_conditioned_error":
        payload.update({
            "coverage_case_assignment_per_agent": dict(
                getattr(system, "coverage_case_assignment_per_agent", {})
            ),
            "c0_rescue_count_per_agent": dict(
                getattr(system, "c0_rescue_count_per_agent", {})
            ),
            "c1_deepening_count_per_agent": dict(
                getattr(system, "c1_deepening_count_per_agent", {})
            ),
        })
    payload.update(rollout_method_metadata(system.cfg, system))
    if str(getattr(system.cfg, "method_version", "legacy")) in {"v8_2_hybrid_progressive", "v8_stable_qd_lineage"}:
        payload.update({
            "method_version": str(system.cfg.method_version),
            "competence_schedule_version": str(system.cfg.competence_schedule_version),
            "target_selector_version": str(system.cfg.target_selector_version),
            "beam_policy_version": str(system.cfg.beam_policy_version),
            "tcs_candidate_policy_version": str(system.cfg.tcs_candidate_policy_version),
            "mechanism_signature_version": str(system.cfg.mechanism_signature_version),
        })
        if str(getattr(system.cfg, "method_version", "legacy")) == "v8_stable_qd_lineage":
            payload.update({
                "active_team_selector_version": str(system.cfg.active_team_selector_version),
                "lineage_policy_version": str(system.cfg.lineage_policy_version),
                "mechanism_distance_version": str(system.cfg.mechanism_distance_version),
                "candidate_refill_version": str(system.cfg.candidate_refill_version),
                "archive_policy_version": str(system.cfg.archive_policy_version),
                "joint_quality_filter_version": str(system.cfg.joint_quality_filter_version),
                "probe_stability_version": str(system.cfg.probe_stability_version),
                "parent_selection_version": str(system.cfg.parent_selection_version),
            })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_selected_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    prompts = [str(agent.get("prompt", "")) for agent in payload.get("agents", []) if isinstance(agent, dict)]
    if not prompts or any(not p for p in prompts):
        raise ValueError(f"No valid prompts found in {path}")
    return payload, prompts


from .persistence.checkpoint import (
    BEHAVIOR_CONFIG_FIELDS, CHECKPOINT_VERSION, abort_incompatible_checkpoint,
    build_training_checkpoint, checkpoint_behavior_config, checkpoint_behavior_config_fingerprint,
    checkpoint_compatible, checkpoint_config_signature, checkpoint_incompatibility_reasons,
    checkpoint_path, clear_training_checkpoint, read_json_file, restore_cost_summary,
    migrate_checkpoint, restore_prompt_history, restore_system_state,
    write_json_atomic, write_training_checkpoint,
)





def validation_score(epoch_record, reward_mode="guarded_diversity"):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    mode = str(reward_mode).lower()
    if mode == "accuracy_only":
        return float(val.get("vote_acc", 0.0) or 0.0)
    if mode == "coverage_useful_diversity":
        return (
            0.4 * float(val.get("vote_acc", 0.0) or 0.0)
            + 0.3 * float(val.get("oracle_acc", 0.0) or 0.0)
            + 0.2 * float(val.get("mean_useful_diversity", 0.0) or 0.0)
            - 0.2 * float(val.get("mean_invalid_rate", 0.0) or 0.0)
        )
    if mode in {"vote_useful_diversity", "competence_depth_schedule"}:
        return (
            0.5 * float(val.get("vote_acc", 0.0) or 0.0)
            + 0.2 * float(val.get("mean_vote_margin", -1.0) if val.get("mean_vote_margin") is not None else -1.0)
            + 0.15 * float(val.get("mean_boundary_useful_diversity", 0.0) or 0.0)
            - 0.15 * float(val.get("mean_invalid_rate", 0.0) or 0.0)
        )
    return (
        float(val.get("vote_acc", 0.0) or 0.0)
        + 0.2 * float(val.get("mean_embedding_diversity", 0.0) or 0.0)
        - 0.1 * float(val.get("mean_invalid_rate", 0.0) or 0.0)
    )


def validation_metric_name(reward_mode, best_state_selection_mode="existing"):
    if str(best_state_selection_mode or "existing").lower() == "state_conditioned_vote_first":
        return "state_conditioned_vote_first(vote,mean,-C0,C2_vote,C2_margin,-invalid,earlier)"
    if str(best_state_selection_mode or "existing").lower() == "rollout_vote_first":
        return "rollout_vote_first(vote,C3,mean,bottom2,conversion,margin,-wrong,-invalid,rollout_div,earlier)"
    if str(best_state_selection_mode or "existing").lower() == "vote_competence_first":
        return "vote_competence_first(vote,bottom2,C2,-gap,margin,mean,-invalid,earlier)"
    if str(best_state_selection_mode or "existing").lower() == "vote_first":
        return "vote_first(vote,mean_individual,margin,-invalid,earlier_epoch)"
    mode = str(reward_mode).lower()
    if mode == "accuracy_only":
        return "vote_acc"
    if mode == "coverage_useful_diversity":
        return "vote+oracle+useful_div-invalid"
    if mode in {"vote_useful_diversity", "competence_depth_schedule"}:
        return "vote+margin+boundary_div-invalid"
    return "vote_acc+embedding_div-invalid"


def uses_vote_useful_metrics(reward_mode):
    return str(reward_mode).lower() == "vote_useful_diversity"


def auto_train_rollout_concurrency(cfg):
    configured = int(getattr(cfg, "train_rollout_concurrency", 0) or 0)
    if configured > 0:
        return configured
    candidate_pool = max(1, int(cfg.beam_size)) * max(1, int(cfg.num_candidates_per_parent)) + max(1, int(cfg.beam_size))
    candidate_eval = candidate_pool if int(getattr(cfg, "candidate_eval_concurrency", 0) or 0) <= 0 else int(cfg.candidate_eval_concurrency)
    peak_solver_requests = max(1, candidate_eval) * max(1, int(cfg.candidate_eval_batch_size or 1)) * max(1, int(cfg.agents or 1))
    return max(1, peak_solver_requests // max(1, int(cfg.agents or 1)))


async def main_async():
    parser = build_parser()
    args = parser.parse_args()
    cfg = Config(**vars(args))
    cfg.baseline_only = bool(int(cfg.baseline_only))
    cfg.eval_test_each_epoch = bool(int(cfg.eval_test_each_epoch))
    cfg.beam_refresh_each_epoch = bool(int(cfg.beam_refresh_each_epoch))
    cfg.legacy_beam_rescore_each_epoch = bool(int(cfg.legacy_beam_rescore_each_epoch))
    old_refresh_explicit = "--beam_refresh_each_epoch" in sys.argv
    if old_refresh_explicit:
        cfg.legacy_beam_rescore_each_epoch = cfg.beam_refresh_each_epoch
    if str(cfg.method_version) == "v8_stable_qd_lineage" and old_refresh_explicit and cfg.beam_refresh_each_epoch:
        print(
            "[DEPRECATION] --beam_refresh_each_epoch=1 is ignored by V8; "
            "legacy beam rescore remains disabled.",
            flush=True,
        )
    if str(cfg.method_version) in {"v8_stable_qd_lineage", "v8_accuracy_rollout_embedding", "v8_rollout_qd_vote_ready"}:
        cfg.legacy_beam_rescore_each_epoch = False
    cfg.use_joint_trace_diversity_evaluator = bool(int(cfg.use_joint_trace_diversity_evaluator))
    cfg.invalid_binary = bool(int(cfg.invalid_binary))
    cfg.use_baseline_relative_reward = bool(int(cfg.use_baseline_relative_reward))
    cfg.candidate_reuse_recorded_rollouts = bool(int(cfg.candidate_reuse_recorded_rollouts))
    cfg.solver_rollout_singleflight = bool(int(cfg.solver_rollout_singleflight))
    cfg.candidate_eval_prompt_dedup = bool(int(cfg.candidate_eval_prompt_dedup))
    cfg.candidate_eval_cache_logging = bool(int(cfg.candidate_eval_cache_logging))
    cfg.transient_retry_forever = bool(int(cfg.transient_retry_forever))
    cfg.llm_call_logging = bool(int(cfg.llm_call_logging))
    cfg.no_effective_evolution_stop_enabled = bool(int(cfg.no_effective_evolution_stop_enabled))
    cfg.teacher_critic_use_voting_failure = bool(int(cfg.teacher_critic_use_voting_failure))
    cfg.boundary_selector_enabled = bool(int(cfg.boundary_selector_enabled))
    cfg.shared_error_metrics_enabled = bool(int(cfg.shared_error_metrics_enabled))
    cfg.residual_specialization_enabled = bool(int(cfg.residual_specialization_enabled))
    cfg.error_dependence_guard_enabled = bool(int(cfg.error_dependence_guard_enabled))
    cfg.residual_cycle_guard_enabled = bool(int(cfg.residual_cycle_guard_enabled))
    cfg.mechanism_trust_region_enabled = bool(int(cfg.mechanism_trust_region_enabled))
    cfg.behavior_cycle_guard_enabled = bool(int(cfg.behavior_cycle_guard_enabled))
    cfg.prompt_trust_region_enabled = bool(int(cfg.prompt_trust_region_enabled))
    cfg.competence_schedule_monotonic = bool(int(cfg.competence_schedule_monotonic))
    cfg.competence_depth1_candidate_guard_enabled = bool(int(cfg.competence_depth1_candidate_guard_enabled))
    cfg.validation_stable_specialization_tie_break_enabled = bool(int(cfg.validation_stable_specialization_tie_break_enabled))
    for field in (
        "state_conditioned_enabled", "state_coverage_enabled",
        "state_c2_wrong_split_enabled", "state_trace_tiebreak_enabled",
    ):
        setattr(cfg, field, bool(int(getattr(cfg, field))))
    for field in (
        "candidate_refill_enabled", "candidate_refill_require_task_repair",
        "candidate_refill_require_distinct_mechanism", "candidate_refill_feed_rejection_reasons",
        "candidate_refill_stop_when_requirements_met", "probation_archive_enabled",
        "probation_require_mechanism_novelty", "probation_parent_enabled",
        "target_selector_fairness_enabled", "joint_refresh_on_safe_archive_change",
        "joint_refresh_on_probation_promotion", "joint_refresh_on_representative_change",
        "joint_refresh_force_final_epoch", "joint_refresh_skip_when_no_dirty_prompt",
    ):
        setattr(cfg, field, bool(int(getattr(cfg, field))))
    cfg.resume_from_checkpoint = bool(int(getattr(cfg, "resume_from_checkpoint", False)))

    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)

    raw_test = load_jsonl(cfg.test_path, cfg.test_size)
    test_data = build_dataset(raw_test, cfg.dataset_format)

    if cfg.baseline_only:
        train_data = []
        val_data = []
        candidate_eval_pool = []
        cfg.candidate_eval_pool_actual_size = 0
        print(f"Loaded baseline test={len(test_data)}")
    else:
        raw_train = load_jsonl(cfg.train_path, cfg.train_size)
        if cfg.val_path:
            train_data = build_dataset(raw_train, cfg.dataset_format)
            val_data = build_dataset(load_jsonl(cfg.val_path, cfg.val_size), cfg.dataset_format)
            val_source = cfg.val_path
        else:
            split_train, split_val = split_train_validation(raw_train, cfg)
            train_data = build_dataset(split_train, cfg.dataset_format)
            val_data = build_dataset(split_val, cfg.dataset_format)
            val_source = f"{cfg.train_path}:split"
        candidate_eval_pool = build_candidate_eval_pool(train_data, val_data, cfg)
        cfg.candidate_eval_pool_actual_size = len(candidate_eval_pool)
        print(f"Loaded train={len(train_data)} val={len(val_data)} test={len(test_data)} val_source={val_source}")
        print(
            f"Candidate eval: strategy={cfg.candidate_eval_strategy} "
            f"pool={len(candidate_eval_pool)} batch_size={cfg.candidate_eval_batch_size} "
            f"repeats={cfg.candidate_eval_repeats}"
        )

    preflight_resume_candidate = None
    if cfg.resume_from_checkpoint and not cfg.baseline_only:
        preflight_resume_candidate = read_json_file(checkpoint_path(cfg))
        if preflight_resume_candidate is not None:
            preflight_reasons = checkpoint_incompatibility_reasons(preflight_resume_candidate, cfg, train_data)
            if preflight_reasons:
                abort_incompatible_checkpoint(cfg, preflight_reasons)

    from .system import TraceBeamSearchSystem

    system = TraceBeamSearchSystem(cfg)

    if cfg.baseline_only:
        test_metrics = await system.evaluate_dataset(test_data, split_name="test_epoch1")
        epoch_record = {"epoch": 1, "train": {}, "test": test_metrics}
        system.history.append(epoch_record)
        if uses_vote_useful_metrics(cfg.reward_mode):
            print(
                "Baseline: "
                f"test_vote_acc={test_metrics['vote_acc']:.4f}, "
                f"test_oracle_acc={test_metrics.get('oracle_acc', 0.0):.4f}, "
                f"test_aggregation_gap={test_metrics.get('aggregation_gap', 0.0):.4f}, "
                f"test_useful_div={test_metrics.get('mean_useful_diversity', 0.0):.4f}, "
                f"test_invalid={test_metrics['mean_invalid_rate']:.4f}"
            )
        else:
            print(
                "Baseline: "
                f"test_embedding_div={test_metrics['mean_embedding_diversity']:.4f}, "
                f"test_embedding_overlap={test_metrics['mean_embedding_overlap']:.4f}, "
                f"test_invalid={test_metrics['mean_invalid_rate']:.4f}, "
                f"test_vote_acc={test_metrics['vote_acc']:.4f}"
            )
        system.save_state("last_state", extra=epoch_record)
        system.save_state("best_state", extra=epoch_record)
        with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(system.history, f, ensure_ascii=False, indent=2)
        system.flush_update_logs()
        system.flush_train_step_logs()
        system.flush_train_trace_history_logs()
        system.flush_test_trace_history_logs()
        system.flush_prompt_history()
        system.flush_llm_call_logs()
        system.write_cost_summary()
        return

    best_score = -1e30
    best_epoch = 0
    best_epoch_record = None
    epochs_without_improvement = 0
    stopped_early = False
    no_effective_evolution_counter = 0
    no_effective_evolution_stopped = False
    no_effective_evolution_reason = ""
    best_prompts_path = os.path.join(cfg.out_dir, "best_prompts.json")
    resume_payload = None
    resume_epoch_index = 0
    resume_cursor = 0
    resume_accumulators = {}
    resume_stage = "training"
    resume_epoch_record = None
    if cfg.resume_from_checkpoint:
        candidate = preflight_resume_candidate if preflight_resume_candidate is not None else read_json_file(checkpoint_path(cfg))
        incompatibility_reasons = checkpoint_incompatibility_reasons(candidate, cfg, train_data)
        if not incompatibility_reasons:
            resume_payload = candidate
            resume_stage = str(candidate.get("stage", "training") or "training")
            candidate = migrate_checkpoint(candidate, cfg)
            restore_system_state(system, candidate.get("state", {}))
            restore_prompt_history(system)
            restore_cost_summary(system)
            history = read_json_file(os.path.join(cfg.out_dir, "history.json"))
            if isinstance(history, list):
                system.history = history
            best_score = float(candidate.get("best_score", best_score))
            best_epoch = int(candidate.get("best_epoch", best_epoch) or 0)
            best_epoch_record = next(
                (record for record in system.history if isinstance(record, dict) and int(record.get("epoch", 0) or 0) == best_epoch),
                None,
            )
            epochs_without_improvement = int(candidate.get("epochs_without_improvement", epochs_without_improvement) or 0)
            stopped_early = bool(candidate.get("stopped_early", False))
            no_effective_evolution_counter = int(candidate.get("no_effective_evolution_counter", 0) or 0)
            no_effective_evolution_stopped = bool(candidate.get("no_effective_evolution_stopped", False))
            no_effective_evolution_reason = str(candidate.get("no_effective_evolution_reason", ""))
            resume_epoch_index = int(candidate.get("epoch_index", 0) or 0)
            if resume_stage == "training":
                resume_cursor = int(candidate.get("cursor", 0) or 0)
                resume_accumulators = candidate.get("train_accumulators", {}) if isinstance(candidate.get("train_accumulators", {}), dict) else {}
                completed_epoch_num = resume_epoch_index + 1
                epoch_already_recorded = any(
                    isinstance(record, dict) and record.get("epoch") == completed_epoch_num
                    for record in system.history
                )
                if resume_cursor >= len(train_data) and epoch_already_recorded:
                    resume_stage = "between_epochs"
                    resume_epoch_index = completed_epoch_num
                    resume_payload = None
                    resume_cursor = 0
                    resume_accumulators = {}
                elif no_effective_evolution_stopped:
                    resume_cursor = len(train_data)
            else:
                if resume_stage == "epoch_evaluated" and isinstance(candidate.get("epoch_record"), dict):
                    resume_epoch_record = candidate.get("epoch_record")
                resume_payload = None
                resume_cursor = 0
                resume_accumulators = {}
                if resume_stage == "between_epochs" and (stopped_early or no_effective_evolution_stopped):
                    resume_epoch_index = int(cfg.epochs)
            print(
                f"[RESUME] Loaded training checkpoint stage={resume_stage} "
                f"epoch={min(resume_epoch_index + 1, cfg.epochs)} "
                f"cursor={resume_cursor}/{len(train_data)} best_epoch={best_epoch}",
                flush=True,
            )
            system.update_logs.append(
                {
                    **system._base_log_fields(),
                    "event": "run_resumed",
                    "execution_session_id": str(getattr(system, "execution_session_id", "") or ""),
                    "previous_execution_session_id": str(candidate.get("execution_session_id", "") or getattr(system, "previous_execution_session_id", "")),
                    "resumed_from_epoch": int(resume_epoch_index + 1),
                    "resumed_from_cursor": int(resume_cursor),
                    "checkpoint_path": checkpoint_path(cfg),
                }
            )
            system.flush_update_logs()
        elif candidate is not None:
            abort_incompatible_checkpoint(cfg, incompatibility_reasons)

    rollout_qd_method = str(cfg.method_version) in {
        "v8_accuracy_rollout_embedding", "v8_rollout_qd_vote_ready", "v9_state_conditioned_error",
    }
    adaptive_competence_schedule = bool(cfg.competence_depth_enabled) and str(cfg.competence_schedule_mode) == "baseline_relative_opt_snapshot"
    fixed_competence_probe = []
    if rollout_qd_method:
        fixed_competence_probe = list(candidate_eval_pool or train_data)
        system.current_fixed_probe_hash = system._fixed_probe_hash(fixed_competence_probe)
        system.prompt_probe_version = (
            "state_conditioned_fixed_probe_v1"
            if system._is_state_conditioned_method()
            else "rollout_fixed_probe_v1"
        )
        system.competence_probe_indices = [train_data.index(item) for item in fixed_competence_probe]
        system.competence_probe_question_hashes = [system._hash(item["question"]) for item in fixed_competence_probe]
        system.write_run_meta()
    elif adaptive_competence_schedule:
        has_saved_baseline = bool(system.initial_competence_probe_metrics)
        if has_saved_baseline:
            probe_indices = select_competence_probe_indices(train_data, cfg, system.competence_probe_indices)
        else:
            if cfg.resume_from_checkpoint and preflight_resume_candidate is not None:
                raise RuntimeError("compatible competence checkpoint is missing its initial optimization probe baseline")
            probe_indices = select_competence_probe_indices(train_data, cfg)
        fixed_competence_probe = [train_data[index] for index in probe_indices]
        probe_hashes = [system._hash(example["question"]) for example in fixed_competence_probe]
        if has_saved_baseline:
            if probe_hashes != list(system.competence_probe_question_hashes):
                raise RuntimeError("competence probe question hashes changed while restoring checkpoint")
        else:
            system.competence_probe_indices = list(probe_indices)
            system.competence_probe_question_hashes = list(probe_hashes)
            system.initial_active_prompt_hashes = [system._hash(prompt) for prompt in system._active_prompt_list()]
            initial_probe = await system.evaluate_competence_probe(
                fixed_competence_probe, probe_name="initial_opt_competence", epoch=0
            )
            if list(initial_probe.get("question_hashes", [])) != probe_hashes:
                raise RuntimeError("initial competence probe hashes do not match the fixed optimization probe")
            system.initial_competence_probe_metrics = dict(initial_probe)
            system.latest_competence_probe_metrics = dict(initial_probe)
            write_training_checkpoint(
                cfg,
                system,
                epoch_index=0,
                cursor=0,
                order=[],
                train_accumulators={},
                best_score=best_score,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                stopped_early=stopped_early,
                no_effective_evolution_counter=no_effective_evolution_counter,
                no_effective_evolution_stopped=no_effective_evolution_stopped,
                no_effective_evolution_reason=no_effective_evolution_reason,
                stage="between_epochs",
            )
        system.write_run_meta()

    if system._is_state_conditioned_method():
        initial_record = next(
            (
                record for record in system.history
                if isinstance(record, dict) and isinstance(record.get("initial_validation"), dict)
            ),
            None,
        )
        if initial_record is None:
            initial_validation = await system.evaluate_dataset(
                val_data, split_name="val_initial_state_conditioned"
            )
            initial_validation["state_validation_accuracy_guard_epsilon"] = float(
                cfg.state_validation_accuracy_guard_epsilon
            )
            initial_record = {
                "epoch": 0,
                "train": {},
                "val": dict(initial_validation),
                "initial_validation": dict(initial_validation),
                **rollout_method_metadata(cfg, system),
            }
            system.history.append(initial_record)
            best_epoch = 0
            best_epoch_record = initial_record
            best_score = validation_score(initial_record, cfg.reward_mode)
            system.save_state("best_state", extra=initial_record)
            write_selected_prompts(
                best_prompts_path,
                system,
                0,
                validation_metric_name(cfg.reward_mode, cfg.best_state_selection_mode),
                best_score,
                cfg.best_state_selection_mode,
                initial_record,
            )
            with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(system.history, f, ensure_ascii=False, indent=2)
            system.write_cost_summary()
        elif best_epoch_record is None:
            best_epoch = int(initial_record.get("epoch", 0) or 0)
            best_epoch_record = initial_record
            best_score = validation_score(initial_record, cfg.reward_mode)

    if resume_epoch_record is not None:
        epoch_num = int(resume_epoch_record.get("epoch", resume_epoch_index + 1) or (resume_epoch_index + 1))
        train_metrics = resume_epoch_record.get("train", {}) if isinstance(resume_epoch_record.get("train", {}), dict) else {}
        val_metrics = resume_epoch_record.get("val", {}) if isinstance(resume_epoch_record.get("val", {}), dict) else {}
        if not any(isinstance(record, dict) and record.get("epoch") == epoch_num for record in system.history):
            system.history.append(resume_epoch_record)
        print(
            f"[RESUME] Finalizing checkpointed epoch {epoch_num}: "
            f"train_vote_acc={float(train_metrics.get('vote_acc', 0.0) or 0.0):.4f}, "
            f"train_oracle_acc={float(train_metrics.get('oracle_acc', 0.0) or 0.0):.4f}, "
            f"train_gap={float(train_metrics.get('aggregation_gap', 0.0) or 0.0):.4f}, "
            f"train_useful_div={float(train_metrics.get('mean_useful_diversity', 0.0) or 0.0):.4f}, "
            f"train_invalid={float(train_metrics.get('mean_invalid_rate', 0.0) or 0.0):.4f}, "
            f"val_vote_acc={float(val_metrics.get('vote_acc', 0.0) or 0.0):.4f}, "
            f"val_oracle_acc={float(val_metrics.get('oracle_acc', 0.0) or 0.0):.4f}, "
            f"val_gap={float(val_metrics.get('aggregation_gap', 0.0) or 0.0):.4f}, "
            f"val_useful_div={float(val_metrics.get('mean_useful_diversity', 0.0) or 0.0):.4f}, "
            f"val_invalid={float(val_metrics.get('mean_invalid_rate', 0.0) or 0.0):.4f}",
            flush=True,
        )
        system.save_state("last_state", extra=resume_epoch_record)
        system.flush_train_step_logs()
        system.flush_train_trace_history_logs()
        system.flush_test_trace_history_logs()
        system.flush_prompt_history()
        system.flush_llm_call_logs()
        system.write_cost_summary()
        with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(system.history, f, ensure_ascii=False, indent=2)

        score = validation_score(resume_epoch_record, cfg.reward_mode)
        min_delta = float(getattr(cfg, "early_stopping_min_delta", 0.0) or 0.0)
        if is_better_validation_state(
            resume_epoch_record,
            best_epoch_record,
            best_score,
            cfg.reward_mode,
            cfg.best_state_selection_mode,
            min_delta,
        ):
            best_score = score
            best_epoch = epoch_num
            best_epoch_record = resume_epoch_record
            epochs_without_improvement = 0
            system.save_state("best_state", extra=resume_epoch_record)
            write_selected_prompts(
                best_prompts_path,
                system,
                best_epoch,
                validation_metric_name(cfg.reward_mode, cfg.best_state_selection_mode),
                best_score,
                cfg.best_state_selection_mode,
                resume_epoch_record,
            )
        else:
            epochs_without_improvement += 1

        raw_patience = getattr(cfg, "early_stopping_patience", -1)
        patience = int(raw_patience) if raw_patience is not None else -1
        should_stop_after_epoch = False
        if patience >= 0 and epochs_without_improvement >= patience:
            stopped_early = True
            should_stop_after_epoch = True
            print(
                f"Early stopping at epoch {epoch_num}: "
                f"best_epoch={best_epoch}, metric={validation_metric_name(cfg.reward_mode, cfg.best_state_selection_mode)}, "
                f"best_validation_score={best_score:.4f}, "
                f"epochs_without_improvement={epochs_without_improvement}",
                flush=True,
            )
        if no_effective_evolution_stopped:
            stopped_early = True
            should_stop_after_epoch = True
        write_training_checkpoint(
            cfg,
            system,
            epoch_index=epoch_num,
            cursor=0,
            order=[],
            train_accumulators={},
            best_score=best_score,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            stopped_early=stopped_early,
            no_effective_evolution_counter=no_effective_evolution_counter,
            no_effective_evolution_stopped=no_effective_evolution_stopped,
            no_effective_evolution_reason=no_effective_evolution_reason,
            stage="between_epochs",
        )
        resume_epoch_index = int(cfg.epochs) if should_stop_after_epoch else epoch_num
        resume_epoch_record = None

    for epoch in range(resume_epoch_index, cfg.epochs):
        strength_used = float(system.specialization_strength)
        if resume_payload is not None and epoch == resume_epoch_index:
            order = [int(x) for x in resume_payload.get("order", [])]
        else:
            order = list(range(len(train_data)))
            random.shuffle(order)
        train_embedding_diversity = []
        train_embedding_overlap = []
        train_invalid_rate = []
        train_vote_correct = []
        train_any_correct = []
        train_useful_diversity = []
        train_individual_correct = []
        if resume_payload is not None and epoch == resume_epoch_index:
            train_embedding_diversity = [float(x) for x in resume_accumulators.get("train_embedding_diversity", [])]
            train_embedding_overlap = [float(x) for x in resume_accumulators.get("train_embedding_overlap", [])]
            train_invalid_rate = [float(x) for x in resume_accumulators.get("train_invalid_rate", [])]
            train_vote_correct = [int(x) for x in resume_accumulators.get("train_vote_correct", [])]
            train_any_correct = [int(x) for x in resume_accumulators.get("train_any_correct", [])]
            train_useful_diversity = [float(x) for x in resume_accumulators.get("train_useful_diversity", [])]
            train_individual_correct = [list(map(int, row)) for row in resume_accumulators.get("train_individual_correct", [])]
        train_rollout_concurrency = max(1, auto_train_rollout_concurrency(cfg))

        cursor = min(max(0, int(resume_cursor if resume_payload is not None and epoch == resume_epoch_index else 0)), len(order))
        while cursor < len(order):
            window_end = min(len(order), ((cursor // max(1, int(cfg.update_every))) + 1) * max(1, int(cfg.update_every)))
            batch_end = min(window_end, cursor + train_rollout_concurrency)
            batch_positions = list(range(cursor, batch_end))

            async def solve_position(pos):
                idx = order[pos]
                ex = train_data[idx]
                q = ex["question"]
                gold = system.task_spec.parse_gold(ex["answer"], q)
                solved = await system.solve_train_example_without_update(q, gold)
                return pos, idx, solved

            solved_rows = await asyncio.gather(*[solve_position(pos) for pos in batch_positions])
            solved_rows.sort(key=lambda x: x[0])

            for pos, idx, solved in solved_rows:
                step = pos
                if system._is_state_conditioned_method():
                    eval_batch = select_state_conditioned_candidate_eval_batch(
                        train_data,
                        candidate_eval_pool,
                        cfg,
                        epoch=epoch + 1,
                        step=step + 1,
                        state_records=[*getattr(system, "recent_window_records", []), solved],
                    )
                else:
                    eval_batch = select_candidate_eval_batch(
                        train_data,
                        candidate_eval_pool,
                        cfg,
                        epoch=epoch + 1,
                        step=step + 1,
                        anchor_idx=idx,
                    )
                do_update = ((step + 1) % cfg.update_every == 0)
                out = await system.record_train_rollout(
                    solved,
                    do_update=do_update,
                    eval_batch=eval_batch,
                    step_id=step + 1,
                    epoch_id=epoch + 1,
                )
                update_summary = out.get("update_summary", {}) if isinstance(out.get("update_summary", {}), dict) else {}
                no_effective_evolution_counter = int(update_summary.get("no_effective_evolution_counter", no_effective_evolution_counter) or 0)
                no_effective_evolution_stopped = bool(update_summary.get("no_effective_evolution_stopped", False))
                no_effective_evolution_reason = str(update_summary.get("no_effective_evolution_reason", ""))
                if no_effective_evolution_stopped:
                    print(
                        f"No-effective-evolution stopping at epoch {epoch + 1} step {step + 1}: "
                        f"counter={no_effective_evolution_counter}, "
                        f"reason={no_effective_evolution_reason}"
                    )
                train_embedding_diversity.append(float(out.get("embedding_diversity", 0.0)))
                train_embedding_overlap.append(float(out.get("mean_embedding_overlap", 0.0)))
                train_invalid_rate.append(float(out.get("invalid_rate", 0.0)))
                train_vote_correct.append(int(out.get("vote_correct", 0)))
                train_any_correct.append(int(out.get("any_correct", 0)))
                train_useful_diversity.append(float(out.get("useful_diversity", 0.0)))
                train_individual_correct.append([int(value) for value in out.get("individual_correct", [])])

                if (step + 1) % 10 == 0 or (step + 1) == len(order):
                    vote_acc = float(np.mean(train_vote_correct)) if train_vote_correct else 0.0
                    oracle_acc = float(np.mean(train_any_correct)) if train_any_correct else 0.0
                    print(
                        f"Epoch {epoch + 1} Step {step + 1}/{len(order)} "
                        f"train_vote_acc={vote_acc:.4f} "
                        f"train_oracle_acc={oracle_acc:.4f} "
                        f"train_gap={oracle_acc - vote_acc:.4f} "
                        f"train_useful_div={float(np.mean(train_useful_diversity)):.4f} "
                        f"train_invalid={float(np.mean(train_invalid_rate)):.4f}"
                    )
                write_training_checkpoint(
                    cfg,
                    system,
                    epoch_index=epoch,
                    cursor=step + 1,
                    order=order,
                    train_accumulators={
                        "train_embedding_diversity": train_embedding_diversity,
                        "train_embedding_overlap": train_embedding_overlap,
                        "train_invalid_rate": train_invalid_rate,
                        "train_vote_correct": train_vote_correct,
                        "train_any_correct": train_any_correct,
                        "train_useful_diversity": train_useful_diversity,
                        "train_individual_correct": train_individual_correct,
                    },
                    best_score=best_score,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                    stopped_early=stopped_early,
                    no_effective_evolution_counter=no_effective_evolution_counter,
                    no_effective_evolution_stopped=no_effective_evolution_stopped,
                    no_effective_evolution_reason=no_effective_evolution_reason,
                )
                system.flush_train_step_logs()
                system.flush_train_trace_history_logs()
                system.flush_update_logs()
                system.flush_prompt_history()
                system.flush_llm_call_logs()
                system.write_cost_summary()
                if no_effective_evolution_stopped:
                    break
            cursor = batch_end
            write_training_checkpoint(
                cfg,
                system,
                epoch_index=epoch,
                cursor=cursor,
                order=order,
                train_accumulators={
                    "train_embedding_diversity": train_embedding_diversity,
                    "train_embedding_overlap": train_embedding_overlap,
                    "train_invalid_rate": train_invalid_rate,
                    "train_vote_correct": train_vote_correct,
                    "train_any_correct": train_any_correct,
                    "train_useful_diversity": train_useful_diversity,
                    "train_individual_correct": train_individual_correct,
                },
                best_score=best_score,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                stopped_early=stopped_early,
                no_effective_evolution_counter=no_effective_evolution_counter,
                no_effective_evolution_stopped=no_effective_evolution_stopped,
                no_effective_evolution_reason=no_effective_evolution_reason,
            )
            system.flush_train_step_logs()
            system.flush_train_trace_history_logs()
            system.flush_update_logs()
            system.flush_prompt_history()
            system.flush_llm_call_logs()
            system.write_cost_summary()
            if no_effective_evolution_stopped:
                break

        train_rows = [{"individual_correct": row} for row in train_individual_correct]
        train_competence = system._summarize_rollout_rows(train_rows)
        train_metrics = {
            "mean_embedding_diversity": float(np.mean(train_embedding_diversity)) if train_embedding_diversity else 0.0,
            "mean_embedding_overlap": float(np.mean(train_embedding_overlap)) if train_embedding_overlap else 0.0,
            "mean_invalid_rate": float(np.mean(train_invalid_rate)) if train_invalid_rate else 0.0,
            "vote_acc": float(np.mean(train_vote_correct)) if train_vote_correct else 0.0,
            "oracle_acc": float(np.mean(train_any_correct)) if train_any_correct else 0.0,
            "aggregation_gap": (float(np.mean(train_any_correct)) - float(np.mean(train_vote_correct))) if train_any_correct and train_vote_correct else 0.0,
            "mean_useful_diversity": float(np.mean(train_useful_diversity)) if train_useful_diversity else 0.0,
            "no_effective_evolution_counter": int(no_effective_evolution_counter),
            "no_effective_evolution_stopped": bool(no_effective_evolution_stopped),
            "no_effective_evolution_reason": no_effective_evolution_reason,
            **{key: train_competence[key] for key in (
                "per_agent_acc", "min_individual_acc", "bottom2_mean_acc", "bottom3_mean_acc",
                "max_individual_acc", "individual_acc_std", "best_minus_worst_gap",
                "best_minus_bottom2_gap", "coverage_depth_c1", "coverage_depth_c2",
                "coverage_depth_c3", "coverage_depth_c4", "coverage_depth_c5",
                "c1_minus_c2", "c2_minus_c3",
            )},
        }
        train_metrics.update({
            "online_train_mean_individual_acc": float(train_metrics.get("mean_individual_acc", 0.0) or 0.0),
            "online_train_bottom2_mean_acc": float(train_metrics.get("bottom2_mean_acc", 0.0) or 0.0),
            "online_train_C1": float(train_metrics.get("coverage_depth_c1", 0.0) or 0.0),
            "online_train_C2": float(train_metrics.get("coverage_depth_c2", 0.0) or 0.0),
        })
        refresh_summary = None
        if (
            str(getattr(cfg, "method_version", "legacy")) not in {"v8_stable_qd_lineage", "v8_accuracy_rollout_embedding", "v8_rollout_qd_vote_ready"}
            and cfg.legacy_beam_rescore_each_epoch
        ):
            refresh_batch_size = max(1, int(cfg.candidate_eval_batch_size or 10))
            refresh_batch = [train_data[i] for i in order[: min(refresh_batch_size, len(order))]]
            refresh_summary = await system.refresh_all_prompt_beams(refresh_batch, epoch_id=epoch + 1)
        joint_team_summary = None
        if str(getattr(cfg, "method_version", "legacy")) in {"v8_stable_qd_lineage", "v8_accuracy_rollout_embedding", "v8_rollout_qd_vote_ready", "v9_state_conditioned_error"}:
            if str(getattr(cfg, "method_version", "legacy")) == "v8_stable_qd_lineage":
                system.expire_probation_branches(epoch + 1)
            joint_team_summary = await system.refresh_joint_active_team_if_needed(
                fixed_competence_probe,
                epoch=epoch + 1,
                final_epoch=(epoch + 1 == int(cfg.epochs)),
            )
        system.specialization_strength_history.append(strength_used)
        if strength_used > 0.0:
            system.effective_specialization_epoch_count += 1
            if system.first_nonzero_specialization_epoch is None:
                system.first_nonzero_specialization_epoch = epoch + 1
        competence_schedule_record = None
        if adaptive_competence_schedule:
            probe_metrics = await system.evaluate_competence_probe(
                fixed_competence_probe,
                probe_name=f"opt_competence_epoch{epoch + 1}",
                epoch=epoch + 1,
            )
            if list(probe_metrics.get("question_hashes", [])) != list(system.competence_probe_question_hashes):
                raise RuntimeError("competence probe drift detected after epoch-end beam refresh")
            competence_schedule_record = system.complete_competence_epoch(
                snapshot_metrics=probe_metrics, epoch=epoch + 1
            )
        else:
            system.complete_competence_epoch(train_metrics.get("per_agent_acc", []), epoch + 1)
        val_metrics = await system.evaluate_dataset(val_data, split_name=f"val_epoch{epoch + 1}")
        if system._is_state_conditioned_method():
            val_metrics["state_validation_accuracy_guard_epsilon"] = float(
                cfg.state_validation_accuracy_guard_epsilon
            )
            initial_record = next(
                (
                    record for record in system.history
                    if isinstance(record, dict) and isinstance(record.get("initial_validation"), dict)
                ),
                {},
            )
            initial_state_map = dict(
                initial_record.get("initial_validation", {}).get("state_by_question_hash", {}) or {}
            )
            val_metrics.update(paired_c0_metrics(
                initial_state_map,
                dict(val_metrics.get("state_by_question_hash", {}) or {}),
            ))
        train_metrics["specialization_strength"] = strength_used
        train_metrics["next_epoch_specialization_strength"] = float(system.specialization_strength)
        epoch_record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        if system._is_state_conditioned_method():
            initial_record = next(
                (record for record in system.history if isinstance(record, dict) and record.get("initial_validation")),
                None,
            )
            epoch_record["initial_validation"] = dict(
                (initial_record or {}).get("initial_validation", val_metrics)
            )
        epoch_record.update(rollout_method_metadata(cfg, system))
        if str(getattr(cfg, "method_version", "legacy")) in {"v8_2_hybrid_progressive", "v8_stable_qd_lineage"}:
            epoch_record.update({
                "method_version": str(cfg.method_version),
                "competence_schedule_version": str(cfg.competence_schedule_version),
                "target_selector_version": str(cfg.target_selector_version),
                "beam_policy_version": str(cfg.beam_policy_version),
                "tcs_candidate_policy_version": str(cfg.tcs_candidate_policy_version),
                "mechanism_signature_version": str(cfg.mechanism_signature_version),
                "beam_slot_state": dict(getattr(system, "beam_slot_state", {})),
                "validation_stable_specialization_tie_break_enabled": bool(
                    getattr(cfg, "validation_stable_specialization_tie_break_enabled", True)
                ),
            })
            if joint_team_summary is not None:
                epoch_record["joint_team_selection"] = joint_team_summary
        if competence_schedule_record is not None:
            epoch_record["competence_schedule"] = competence_schedule_record
        if refresh_summary is not None:
            epoch_record["beam_refresh"] = refresh_summary
        if cfg.eval_test_each_epoch:
            epoch_record["test"] = await system.evaluate_dataset(test_data, split_name=f"test_epoch{epoch + 1}")
        write_training_checkpoint(
            cfg,
            system,
            epoch_index=epoch,
            cursor=len(order),
            order=order,
            train_accumulators={
                "train_embedding_diversity": train_embedding_diversity,
                "train_embedding_overlap": train_embedding_overlap,
                "train_invalid_rate": train_invalid_rate,
                "train_vote_correct": train_vote_correct,
                "train_any_correct": train_any_correct,
                "train_useful_diversity": train_useful_diversity,
                "train_individual_correct": train_individual_correct,
            },
            best_score=best_score,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            stopped_early=stopped_early,
            no_effective_evolution_counter=no_effective_evolution_counter,
            no_effective_evolution_stopped=no_effective_evolution_stopped,
            no_effective_evolution_reason=no_effective_evolution_reason,
            stage="epoch_evaluated",
            epoch_record=epoch_record,
        )
        system.history.append(epoch_record)

        print(
            f"Epoch {epoch + 1}: "
            f"train_vote_acc={train_metrics['vote_acc']:.4f}, "
            f"train_oracle_acc={train_metrics['oracle_acc']:.4f}, "
            f"train_gap={train_metrics['aggregation_gap']:.4f}, "
            f"train_useful_div={train_metrics['mean_useful_diversity']:.4f}, "
            f"train_invalid={train_metrics['mean_invalid_rate']:.4f}, "
            f"online_train_bottom2={train_metrics.get('online_train_bottom2_mean_acc', 0.0):.4f}, "
            f"opt_probe_bottom2={float((competence_schedule_record or {}).get('snapshot_bottom2_mean_acc', 0.0)):.4f}, "
            f"initial_probe_bottom2={float((competence_schedule_record or {}).get('initial_bottom2_mean_acc', 0.0)):.4f}, "
            f"bottom2_gain={float((competence_schedule_record or {}).get('bottom2_gain', 0.0)):.4f}, "
            f"opt_probe_mean={float((competence_schedule_record or {}).get('snapshot_mean_individual_acc', 0.0)):.4f}, "
            f"opt_probe_C1={float((competence_schedule_record or {}).get('snapshot_c1', 0.0)):.4f}, "
            f"opt_probe_C2={float((competence_schedule_record or {}).get('snapshot_c2', 0.0)):.4f}, "
            f"gate={'pass' if not (competence_schedule_record or {}).get('gate_failure_reasons') else '|'.join((competence_schedule_record or {}).get('gate_failure_reasons', []))}, "
            f"s_raw={float((competence_schedule_record or {}).get('raw_specialization_strength', 0.0)):.4f}, "
            f"s_next={float(system.specialization_strength):.4f}, "
            f"val_vote_acc={val_metrics['vote_acc']:.4f}, "
            f"val_oracle_acc={val_metrics.get('oracle_acc', 0.0):.4f}, "
            f"val_gap={val_metrics.get('aggregation_gap', 0.0):.4f}, "
            f"val_useful_div={val_metrics.get('mean_useful_diversity', 0.0):.4f}, "
            f"val_invalid={val_metrics['mean_invalid_rate']:.4f}"
        )

        system.save_state("last_state", extra=epoch_record)
        system.flush_train_step_logs()
        system.flush_train_trace_history_logs()
        system.flush_test_trace_history_logs()
        system.flush_prompt_history()
        system.flush_llm_call_logs()
        system.write_cost_summary()
        with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(system.history, f, ensure_ascii=False, indent=2)

        score = validation_score(epoch_record, cfg.reward_mode)
        min_delta = float(getattr(cfg, "early_stopping_min_delta", 0.0) or 0.0)
        if is_better_validation_state(
            epoch_record,
            best_epoch_record,
            best_score,
            cfg.reward_mode,
            cfg.best_state_selection_mode,
            min_delta,
        ):
            best_score = score
            best_epoch = epoch + 1
            best_epoch_record = epoch_record
            epochs_without_improvement = 0
            system.save_state("best_state", extra=epoch_record)
            write_selected_prompts(
                best_prompts_path,
                system,
                best_epoch,
                validation_metric_name(cfg.reward_mode, cfg.best_state_selection_mode),
                best_score,
                cfg.best_state_selection_mode,
                epoch_record,
            )
        else:
            epochs_without_improvement += 1

        raw_patience = getattr(cfg, "early_stopping_patience", -1)
        patience = int(raw_patience) if raw_patience is not None else -1
        should_stop_after_epoch = False
        if patience >= 0 and epochs_without_improvement >= patience:
            stopped_early = True
            should_stop_after_epoch = True
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"best_epoch={best_epoch}, metric={validation_metric_name(cfg.reward_mode, cfg.best_state_selection_mode)}, "
                f"best_validation_score={best_score:.4f}, "
                f"epochs_without_improvement={epochs_without_improvement}"
            )
        if no_effective_evolution_stopped:
            stopped_early = True
            should_stop_after_epoch = True
        write_training_checkpoint(
            cfg,
            system,
            epoch_index=epoch + 1,
            cursor=0,
            order=[],
            train_accumulators={},
            best_score=best_score,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            stopped_early=stopped_early,
            no_effective_evolution_counter=no_effective_evolution_counter,
            no_effective_evolution_stopped=no_effective_evolution_stopped,
            no_effective_evolution_reason=no_effective_evolution_reason,
            stage="between_epochs",
        )
        if should_stop_after_epoch:
            break
        resume_payload = None
        resume_cursor = 0
        resume_accumulators = {}

    if os.path.exists(best_prompts_path):
        payload, prompts = read_selected_prompts(best_prompts_path)
        best_epoch = int(payload.get("selected_epoch", best_epoch) or best_epoch)
        restore_agent_prompts(system, prompts, selected_epoch=best_epoch)
        if str(getattr(cfg, "method_version", "legacy")) == "v8_stable_qd_lineage":
            for agent, saved in zip(system.agents, payload.get("agents", [])):
                if isinstance(saved, dict) and isinstance(saved.get("lineage_state"), dict):
                    agent.lineage_state = dict(saved["lineage_state"])
        if str(getattr(cfg, "method_version", "legacy")) in {
            "v8_stable_qd_lineage", "v8_accuracy_rollout_embedding", "v8_rollout_qd_vote_ready",
        }:
            system.latest_joint_team_metrics = dict(payload.get("joint_team_metrics", {}) or {})
        selected_probe = next(
            (
                record for record in reversed(getattr(system, "competence_probe_history", []))
                if isinstance(record, dict) and int(record.get("epoch", -1)) == int(best_epoch)
            ),
            None,
        )
        if selected_probe is not None:
            system.latest_competence_probe_metrics = dict(selected_probe)

    final_test_metrics = await system.evaluate_dataset(test_data, split_name="test_final")
    final_record = {
        "epoch": "final",
        "selected_epoch": best_epoch,
        "best_validation_score": best_score,
        "early_stopped": bool(stopped_early),
        "early_stopping_patience": int(getattr(cfg, "early_stopping_patience", -1)),
        "early_stopping_min_delta": float(getattr(cfg, "early_stopping_min_delta", 0.0) or 0.0),
        "no_effective_evolution_counter": int(no_effective_evolution_counter),
        "no_effective_evolution_stopped": bool(no_effective_evolution_stopped),
        "no_effective_evolution_reason": no_effective_evolution_reason,
        "test_evaluated_on": "best_state",
        "test": final_test_metrics,
    }
    final_record.update(rollout_method_metadata(cfg, system))
    if str(getattr(cfg, "method_version", "legacy")) in {"v8_2_hybrid_progressive", "v8_stable_qd_lineage"}:
        final_record.update({
            "method_version": str(cfg.method_version),
            "competence_schedule_version": str(cfg.competence_schedule_version),
            "target_selector_version": str(cfg.target_selector_version),
            "beam_policy_version": str(cfg.beam_policy_version),
            "tcs_candidate_policy_version": str(cfg.tcs_candidate_policy_version),
            "mechanism_signature_version": str(cfg.mechanism_signature_version),
        })
    system.history.append(final_record)
    if hasattr(system, "sync_prompt_history_current_state"):
        system.sync_prompt_history_current_state(
            event="selected_state_evaluated",
            epoch="final",
            step=0,
            selected_epoch=best_epoch,
        )
    system.save_state("selected_state", extra=final_record)
    system.save_state("last_state", extra=final_record)
    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(system.history, f, ensure_ascii=False, indent=2)
    system.flush_update_logs()
    system.flush_train_step_logs()
    system.flush_train_trace_history_logs()
    system.flush_test_trace_history_logs()
    system.flush_prompt_history()
    system.flush_llm_call_logs()
    system.write_cost_summary()
    final_summary = {
        "selected_epoch": best_epoch,
        "best_validation_score": best_score,
        "test": final_test_metrics,
        "candidate_channel_funnel": dict(getattr(system, "candidate_channel_funnel", {})),
        "latest_joint_team_metrics": dict(getattr(system, "latest_joint_team_metrics", {})),
        "cost_summary": dict(getattr(system, "cost_summary", {})),
    }
    final_summary.update(rollout_method_metadata(cfg, system))
    write_json_atomic(
        os.path.join(cfg.out_dir, "final_summary.json"),
        final_summary,
    )
    clear_training_checkpoint(cfg)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
