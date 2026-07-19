import asyncio
import hashlib
import json
import os
import random
import time
import uuid

import numpy as np

from .config import Config, build_parser
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


def is_better_validation_state(epoch_record, best_epoch_record, best_score, reward_mode, selection_mode, min_delta=0.0):
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
            list(vote_generalization_first_validation_key(epoch_record))
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


def checkpoint_path(cfg):
    return os.path.join(cfg.out_dir, "training_checkpoint.json")


def read_json_file(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json_atomic(path, payload):
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    for attempt in range(3):
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            return
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt == 2:
                raise
            time.sleep(0.1 * (attempt + 1))


def restore_system_state(system, state_payload):
    agents = state_payload.get("agents", []) if isinstance(state_payload, dict) else []
    if not isinstance(agents, list) or len(agents) != len(system.agents):
        raise ValueError(f"Cannot restore state: got {len(agents) if isinstance(agents, list) else 0} agents, expected {len(system.agents)}")
    for agent, saved in zip(system.agents, agents):
        if not isinstance(saved, dict):
            continue
        agent.initial_prompt = str(saved.get("initial_prompt", agent.initial_prompt))
        agent.current_prompt = str(saved.get("current_prompt", agent.current_prompt))
        prompt_beam = saved.get("prompt_beam", [])
        if isinstance(prompt_beam, list) and prompt_beam:
            agent.prompt_beam = [dict(item) for item in prompt_beam if isinstance(item, dict)]
        else:
            agent.prompt_beam = [system._make_beam_item(agent.current_prompt, None, {}, None, 0)]
        history = saved.get("history", [])
        agent.history = [str(x) for x in history] if isinstance(history, list) and history else [agent.current_prompt]
        agent.accept_count = int(saved.get("accept_count", 0) or 0)
        agent.reject_count = int(saved.get("reject_count", 0) or 0)
        if hasattr(agent, "restore_trajectory_state"):
            agent.restore_trajectory_state(saved)
    recent_window_records = state_payload.get("recent_window_records", [])
    system.recent_window_records = (
        [dict(record) for record in recent_window_records if isinstance(record, dict)]
        if isinstance(recent_window_records, list)
        else []
    )
    system.specialization_strength = float(state_payload.get("specialization_strength", 0.0) or 0.0)
    system.effective_residual_strength = float(state_payload.get("effective_residual_strength", system.specialization_strength) or system.specialization_strength)
    system.previous_epoch_per_agent_acc = [float(x) for x in state_payload.get("previous_epoch_per_agent_acc", [])]
    system.previous_epoch_bottom2_mean_acc = float(state_payload.get("previous_epoch_bottom2_mean_acc", 0.0) or 0.0)
    system.competence_phase_epoch = int(state_payload.get("competence_phase_epoch", 1) or 1)
    system.competence_schedule_version = str(state_payload.get("competence_schedule_version", "competence_depth_v1"))
    system.specialization_strength_history = [float(x) for x in state_payload.get("specialization_strength_history", [])]
    system.competence_probe_indices = [int(x) for x in state_payload.get("competence_probe_indices", [])]
    system.competence_probe_question_hashes = [str(x) for x in state_payload.get("competence_probe_question_hashes", [])]
    system.initial_competence_probe_metrics = dict(state_payload.get("initial_competence_probe_metrics", {}) or {})
    system.latest_competence_probe_metrics = dict(state_payload.get("latest_competence_probe_metrics", {}) or {})
    system.competence_probe_history = [dict(x) for x in state_payload.get("competence_probe_history", []) if isinstance(x, dict)]
    system.initial_active_prompt_hashes = [str(x) for x in state_payload.get("initial_active_prompt_hashes", [])]
    first_nonzero = state_payload.get("first_nonzero_specialization_epoch")
    system.first_nonzero_specialization_epoch = int(first_nonzero) if first_nonzero is not None else None
    system.effective_specialization_epoch_count = int(state_payload.get("effective_specialization_epoch_count", 0) or 0)
    system.depth1_guard_rejection_count = int(state_payload.get("depth1_guard_rejection_count", 0) or 0)
    for field in (
        "catastrophic_accuracy_guard_rejection_count",
        "soft_error_dependence_penalty_count",
        "soft_cycle_penalty_count",
        "soft_mechanism_shift_penalty_count",
        "exploration_candidate_count",
        "exploration_slot_occupancy_count",
        "exploration_to_active_conversion_count",
    ):
        setattr(system, field, int(state_payload.get(field, 0) or 0))
    system.hybrid_selector_history = [dict(x) for x in state_payload.get("hybrid_selector_history", []) if isinstance(x, dict)]
    system.mechanism_signature_history = [dict(x) for x in state_payload.get("mechanism_signature_history", []) if isinstance(x, dict)]
    system.mechanism_signature_by_prompt_hash = {
        str(key): [str(value) for value in values]
        for key, values in dict(state_payload.get("mechanism_signature_by_prompt_hash", {}) or {}).items()
        if isinstance(values, list)
    }
    system.beam_slot_state = dict(state_payload.get("beam_slot_state", {}) or {})
    system.exploration_slot_candidates = [dict(x) for x in state_payload.get("exploration_slot_candidates", []) if isinstance(x, dict)]
    system.prompt_overlength_rejection_count = int(state_payload.get("prompt_overlength_rejection_count", 0) or 0)
    system.truncated_prompt_count = int(state_payload.get("truncated_prompt_count", 0) or 0)
    system.mechanism_embedding_cache = {str(key): list(value) for key, value in dict(state_payload.get("mechanism_embedding_cache", {}) or {}).items()}
    system.prompt_probe_cache = {str(key): dict(value) for key, value in dict(state_payload.get("prompt_probe_cache", {}) or {}).items()}
    system.mechanism_embedding_cache_hit_count = int(state_payload.get("mechanism_embedding_cache_hit_count", 0) or 0)
    system.mechanism_embedding_cache_miss_count = int(state_payload.get("mechanism_embedding_cache_miss_count", 0) or 0)
    system.full_probe_cache_hit_count = int(state_payload.get("full_probe_cache_hit_count", 0) or 0)
    system.full_probe_missing_pair_evaluation_count = int(state_payload.get("full_probe_missing_pair_evaluation_count", 0) or 0)
    system.behavior_profile_by_prompt_hash = {str(key): dict(value) for key, value in dict(state_payload.get("behavior_profile_by_prompt_hash", {}) or {}).items()}
    system.joint_team_selection_history = [dict(value) for value in state_payload.get("joint_team_selection_history", []) if isinstance(value, dict)]
    system.lineage_history = [dict(value) for value in state_payload.get("lineage_history", []) if isinstance(value, dict)]
    system.quality_diversity_archive_history = [dict(value) for value in state_payload.get("quality_diversity_archive_history", []) if isinstance(value, dict)]
    system.behavior_profile_history = [dict(value) for value in state_payload.get("behavior_profile_history", []) if isinstance(value, dict)]
    system.latest_joint_team_metrics = dict(state_payload.get("latest_joint_team_metrics", {}) or {})
    system.joint_quality_anchor_metrics = dict(state_payload.get("joint_quality_anchor_metrics", {}) or {})
    for field in (
        "qd_no_diversification_epochs", "qd_change_limit_relaxed_epoch",
        "qd_previous_active_niche_count",
        "probation_to_safe_conversion_count", "probation_expired_count",
        "candidate_starvation_count", "mechanism_starvation_count",
        "search_branch_starvation_count", "refill_requirements_unmet_count",
    ):
        setattr(system, field, int(state_payload.get(field, 0) or 0))
    system.per_agent_optimizer_update_count = {
        str(key): int(value) for key, value in dict(state_payload.get("per_agent_optimizer_update_count", {}) or {}).items()
    }
    for field in (
        "total_agent_update_count", "task_repair_niche_occupancy_count",
        "mechanism_niche_occupancy_count", "peer_collapse_soft_count",
        "peer_collapse_hard_rejection_count",
    ):
        setattr(system, field, int(state_payload.get(field, 0) or 0))
    python_state = state_payload.get("python_random_state")
    if isinstance(python_state, list):
        def as_tuple(value):
            return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value
        random.setstate(as_tuple(python_state))
    numpy_state = state_payload.get("numpy_random_state")
    if isinstance(numpy_state, list) and len(numpy_state) == 5:
        np.random.set_state((str(numpy_state[0]), np.array(numpy_state[1], dtype=np.uint32), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))


def restore_prompt_history(system):
    path = os.path.join(system.cfg.out_dir, "prompt_history.json")
    payload = read_json_file(path)
    if isinstance(payload, dict):
        system.prompt_history = payload
    if hasattr(system, "sync_prompt_history_current_state"):
        system.sync_prompt_history_current_state(event="checkpoint_resume", epoch="resume", step=0)


def restore_cost_summary(system):
    payload = read_json_file(os.path.join(system.cfg.out_dir, "cost_summary.json"))
    if isinstance(payload, dict):
        base = system._empty_cost_summary() if hasattr(system, "_empty_cost_summary") else {}
        base.update(payload)
        system.cost_summary = base


CHECKPOINT_VERSION = 5

# Fields that can change the objective, candidate distribution, optimizer
# behavior, validation decision, or final aggregation of an interrupted run.
BEHAVIOR_CONFIG_FIELDS = (
        "task_type",
        "dataset_format",
        "comparison_task_id",
        "benchmark",
        "answer_format",
        "train_path",
        "val_path",
        "test_path",
        "train_size",
        "val_size",
        "val_split_ratio",
        "test_size",
        "agents",
        "init_mode",
        "shared_prompt",
        "reward_mode",
        "accuracy_guard_epsilon",
        "invalid_guard_epsilon",
        "reward_weight_div_delta",
        "reward_weight_invalid_delta",
        "reward_weight_vote_delta",
        "reward_weight_vote_margin",
        "reward_weight_boundary_diversity",
        "reward_weight_coverage",
        "reward_weight_useful_diversity",
        "use_baseline_relative_reward",
        "reward_schedule_mode",
        "reward_diversity_warmup_updates",
        "reward_weight_div_delta_early",
        "reward_weight_div_delta_late",
        "reward_weight_vote_delta_early",
        "reward_weight_vote_delta_late",
        "reward_weight_vote_margin_early",
        "reward_weight_vote_margin_late",
        "reward_weight_boundary_diversity_early",
        "reward_weight_boundary_diversity_late",
        "reward_weight_coverage_early",
        "reward_weight_coverage_late",
        "reward_weight_useful_diversity_early",
        "reward_weight_useful_diversity_late",
        "reward_weight_target_accuracy_early",
        "reward_weight_target_accuracy_late",
        "accuracy_guard_epsilon_early",
        "accuracy_guard_epsilon_late",
        "candidate_selection_mode",
        "best_state_selection_mode",
        "vote_tie_break",
        "aggregation_mode",
        "optimizer_architecture",
        "optimizer_fallback_mode",
        "teacher_critic_max_rounds",
        "teacher_question_pass_threshold",
        "teacher_critic_use_voting_failure",
        "teacher_temperature",
        "critic_temperature",
        "student_temperature",
        "teacher_max_tokens",
        "critic_max_tokens",
        "student_max_tokens",
        "student_json_retry_on_parse_fail",
        "student_json_max_retries",
        "student_json_repair_enabled",
        "student_json_repair_max_tokens",
        "student_json_repair_temperature",
        "student_candidate_schema_mode",
        "student_candidate_max_chars_per_field",
        "student_candidate_prompt_max_chars",
        "student_candidate_prompt_soft_max_chars",
        "student_candidate_prompt_hard_max_chars",
        "student_force_minified_json",
        "beam_size",
        "num_candidates_per_parent",
        "optimizer_parent_concurrency",
        "beam_refresh_each_epoch",
        "update_every",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "candidate_eval_batch_size",
        "candidate_eval_strategy",
        "candidate_eval_pool_size",
        "candidate_eval_repeats",
        "candidate_eval_seed_offset",
        "candidate_eval_data_source",
        "candidate_eval_execution_mode",
        "candidate_reuse_recorded_rollouts",
        "candidate_eval_concurrency",
        "solver_rollout_singleflight",
        "candidate_eval_prompt_dedup",
        "candidate_eval_cache_logging",
        "agent_model",
        "optimizer_model",
        "evaluator_model",
        "max_tokens",
        "temperature",
        "optimizer_max_tokens",
        "optimizer_temperature",
        "evaluator_max_tokens",
        "evaluator_temperature",
        "solver_base_url_env",
        "evaluator_base_url_env",
        "diversity_metric",
        "use_joint_trace_diversity_evaluator",
        "invalid_binary",
        "embedding_model",
        "trace_embedding_chunk_words",
        "trace_embedding_chunk_overlap",
        "eval_test_each_epoch",
        "no_effective_evolution_patience",
        "no_effective_evolution_min_optimizer_candidates",
        "no_effective_evolution_stop_enabled",
        "boundary_selector_enabled",
        "shared_error_metrics_enabled",
        "residual_specialization_enabled",
        "error_dependence_guard_enabled",
        "residual_cycle_guard_enabled",
        "mechanism_trust_region_enabled",
        "specialization_ema",
        "specialization_support_shrinkage",
        "capability_loss_weight",
        "specialization_update_period",
        "capability_affinity_weight",
        "capability_coverage_gap_weight",
        "pivotal_loss_guard_epsilon",
        "shared_error_creation_epsilon",
        "behavior_cycle_guard_enabled",
        "behavior_archive_size",
        "behavior_cycle_similarity_threshold",
        "behavior_cycle_min_overlap",
        "behavior_cycle_improvement_epsilon",
        "behavior_cycle_margin_epsilon",
        "prompt_trust_region_enabled",
        "prompt_max_change_ratio",
        "prompt_large_shift_warmup_accepts",
        "prompt_large_shift_min_vote_delta",
        "baseline_allowed_vote_loss",
        "competence_depth_enabled",
        "competence_depth2_aux_enabled",
        "competence_progressive_residual_enabled",
        "competence_floor_low",
        "competence_floor_high",
        "competence_selector_weight",
        "competence_extra_support_shrinkage",
        "competence_weight_accuracy_gain",
        "competence_weight_accuracy_loss",
        "competence_weight_depth2_gain",
        "competence_weight_depth2_loss",
        "competence_weight_vote_gain_early",
        "competence_weight_vote_loss_early",
        "competence_schedule_version",
        "competence_schedule_mode",
        "competence_probe_size",
        "competence_probe_seed_offset",
        "competence_relative_low_delta",
        "competence_relative_high_delta",
        "competence_schedule_ema",
        "competence_schedule_max_step",
        "competence_schedule_monotonic",
        "competence_mean_guard_epsilon",
        "competence_c1_guard_epsilon",
        "competence_c2_guard_epsilon",
        "competence_depth1_candidate_guard_enabled",
        "competence_depth1_candidate_guard_epsilon",
        "competence_min_effective_specialization_epochs",
        "method_version", "target_selector_mode", "target_selector_version", "beam_policy_version",
        "tcs_candidate_policy_version", "mechanism_signature_version",
        "competence_weight_depth1_gain", "competence_weight_depth1_loss", "competence_residual_floor",
        "catastrophic_target_accuracy_loss_epsilon", "soft_guard_error_dependence_weight",
        "soft_guard_cycle_weight", "soft_guard_mechanism_shift_weight",
        "soft_guard_accuracy_regression_weight", "mechanism_novelty_bonus_weight",
        "active_team_selector_version", "lineage_policy_version", "mechanism_distance_version",
        "mechanism_sequence_distance_weight", "mechanism_embedding_distance_weight",
        "mechanism_near_duplicate_similarity_threshold", "behavior_correct_set_weight",
        "behavior_rescue_weight", "behavior_shared_wrong_weight", "behavior_support_shrinkage",
        "team_diversity_mean_behavior_weight", "team_diversity_min_behavior_weight",
        "team_diversity_mechanism_weight", "team_diversity_rescue_balance_weight",
        "joint_team_vote_epsilon_questions", "joint_team_mean_epsilon_questions",
        "joint_team_bottom2_epsilon_questions", "joint_team_c1_epsilon_questions",
        "joint_team_c2_epsilon_questions", "joint_team_per_agent_accuracy_epsilon",
        "lineage_provisional_epochs", "lineage_commit_epochs", "lineage_switch_confirmation_epochs",
        "lineage_mechanism_drift_weight", "lineage_behavior_drift_weight",
        "lineage_soft_drift_threshold", "lineage_hard_drift_threshold",
        "lineage_switch_min_accuracy_gain", "lineage_switch_min_vote_gain",
        "peer_collapse_soft_similarity", "peer_collapse_hard_similarity",
        "validation_stable_specialization_tie_break_enabled",
        "candidate_refill_version", "archive_policy_version", "joint_quality_filter_version",
        "probe_stability_version", "parent_selection_version", "candidate_refill_enabled",
        "candidate_refill_max_rounds", "candidate_refill_candidates_per_round",
        "candidate_refill_max_unique_candidates_per_parent", "candidate_refill_min_safe_non_incumbent",
        "candidate_refill_require_task_repair", "candidate_refill_require_distinct_mechanism",
        "candidate_refill_feed_rejection_reasons", "candidate_refill_stop_when_requirements_met",
        "candidate_refill_max_solver_calls_per_agent_update", "probation_archive_enabled",
        "probation_archive_size_per_agent", "probation_archive_ttl_updates", "probation_max_accuracy_loss",
        "probation_max_c1_loss_questions", "probation_max_c2_loss_questions",
        "probation_require_mechanism_novelty", "candidate_c1_catastrophic_loss_questions",
        "candidate_c2_catastrophic_loss_questions", "qd_archive_size_per_agent",
        "joint_representative_beam_size", "qd_parent_selection_mode",
        "qd_niche_min_parent_opportunities_per_epoch", "probation_parent_enabled",
        "probe_stability_fold_count", "probe_stability_seed_offset", "joint_vote_band_questions",
        "joint_mean_band_correct_count", "joint_bottom2_band_correct_count", "joint_c1_band_questions",
        "joint_c2_band_questions", "joint_allowed_vote_loss_questions", "joint_allowed_c1_loss_questions",
        "joint_allowed_c2_loss_questions", "joint_allowed_total_agent_correct_loss",
        "joint_allowed_bottom2_correct_loss", "joint_allowed_per_agent_correct_loss",
        "joint_team_max_active_changes_early", "joint_team_max_active_changes_late",
        "joint_team_change_limit_switch_strength", "joint_team_no_diversification_patience",
        "joint_team_change_limit_relaxation", "lineage_commit_required_snapshots",
        "lineage_switch_confirmation_snapshots", "qd_readiness_min_distinct_niches",
        "qd_readiness_min_diversity", "qd_readiness_max_fold_gap", "residual_specialization_qd_floor",
        "behavior_error_overlap_weight", "behavior_wrong_answer_dispersion_weight",
        "behavior_wrong_support_shrinkage", "min_optimizer_updates_per_agent_per_epoch",
        "target_selector_fairness_enabled",
        "split_integrity_json",
)


def _normalize_behavior_config_types(payload):
    defaults = Config()
    normalized = dict(payload)
    for field, value in list(normalized.items()):
        if isinstance(getattr(defaults, field, None), bool):
            value = bool(value)
        normalized[field] = value
    return normalized


def checkpoint_behavior_config(cfg):
    payload = _normalize_behavior_config_types(
        {field: getattr(cfg, field, None) for field in BEHAVIOR_CONFIG_FIELDS}
    )
    if bool(getattr(cfg, "competence_depth_enabled", False)):
        payload["effective_aggregation_mode"] = canonical_aggregation_mode(
            str(getattr(cfg, "aggregation_mode", "majority") or "majority")
        )
        payload["plurality_boundary_version"] = "plurality_boundary_v1"
    if not bool(getattr(cfg, "competence_depth_enabled", False)):
        for field in (
            "student_candidate_prompt_soft_max_chars", "student_candidate_prompt_hard_max_chars",
            "competence_depth_enabled", "competence_depth2_aux_enabled",
            "competence_progressive_residual_enabled", "competence_floor_low", "competence_floor_high",
            "competence_selector_weight", "competence_extra_support_shrinkage",
            "competence_weight_accuracy_gain", "competence_weight_accuracy_loss",
            "competence_weight_depth2_gain", "competence_weight_depth2_loss",
            "competence_weight_vote_gain_early", "competence_weight_vote_loss_early",
            "competence_schedule_version",
            "competence_schedule_mode", "competence_probe_size", "competence_probe_seed_offset",
            "competence_relative_low_delta", "competence_relative_high_delta",
            "competence_schedule_ema", "competence_schedule_max_step", "competence_schedule_monotonic",
            "competence_mean_guard_epsilon", "competence_c1_guard_epsilon", "competence_c2_guard_epsilon",
            "competence_depth1_candidate_guard_enabled", "competence_depth1_candidate_guard_epsilon",
            "competence_min_effective_specialization_epochs",
        ):
            payload.pop(field, None)
    if str(getattr(cfg, "competence_schedule_mode", "absolute_legacy")) != "baseline_relative_opt_snapshot":
        for field in (
            "competence_schedule_mode", "competence_probe_size", "competence_probe_seed_offset",
            "competence_relative_low_delta", "competence_relative_high_delta", "competence_schedule_ema",
            "competence_schedule_max_step", "competence_schedule_monotonic", "competence_mean_guard_epsilon",
            "competence_c1_guard_epsilon", "competence_c2_guard_epsilon",
            "competence_depth1_candidate_guard_enabled", "competence_depth1_candidate_guard_epsilon",
            "competence_min_effective_specialization_epochs",
        ):
            payload.pop(field, None)
    if str(getattr(cfg, "method_version", "legacy")) not in {"v8_2_hybrid_progressive", "v8_stable_qd_lineage"}:
        for field in (
            "method_version", "target_selector_mode", "target_selector_version", "beam_policy_version",
            "tcs_candidate_policy_version", "mechanism_signature_version", "competence_weight_depth1_gain",
            "competence_weight_depth1_loss", "competence_residual_floor",
            "catastrophic_target_accuracy_loss_epsilon", "soft_guard_error_dependence_weight",
            "soft_guard_cycle_weight", "soft_guard_mechanism_shift_weight",
            "soft_guard_accuracy_regression_weight", "mechanism_novelty_bonus_weight",
        ):
            payload.pop(field, None)
    qd_fields = (
        "active_team_selector_version", "lineage_policy_version", "mechanism_distance_version",
        "mechanism_sequence_distance_weight", "mechanism_embedding_distance_weight",
        "mechanism_near_duplicate_similarity_threshold", "behavior_correct_set_weight",
        "behavior_rescue_weight", "behavior_shared_wrong_weight", "behavior_support_shrinkage",
        "team_diversity_mean_behavior_weight", "team_diversity_min_behavior_weight",
        "team_diversity_mechanism_weight", "team_diversity_rescue_balance_weight",
        "joint_team_vote_epsilon_questions", "joint_team_mean_epsilon_questions",
        "joint_team_bottom2_epsilon_questions", "joint_team_c1_epsilon_questions",
        "joint_team_c2_epsilon_questions", "joint_team_per_agent_accuracy_epsilon",
        "lineage_provisional_epochs", "lineage_commit_epochs", "lineage_switch_confirmation_epochs",
        "lineage_mechanism_drift_weight", "lineage_behavior_drift_weight",
        "lineage_soft_drift_threshold", "lineage_hard_drift_threshold",
        "lineage_switch_min_accuracy_gain", "lineage_switch_min_vote_gain",
        "peer_collapse_soft_similarity", "peer_collapse_hard_similarity",
        "validation_stable_specialization_tie_break_enabled",
        "candidate_refill_version", "archive_policy_version", "joint_quality_filter_version",
        "probe_stability_version", "parent_selection_version", "candidate_refill_enabled",
        "candidate_refill_max_rounds", "candidate_refill_candidates_per_round",
        "candidate_refill_max_unique_candidates_per_parent", "candidate_refill_min_safe_non_incumbent",
        "candidate_refill_require_task_repair", "candidate_refill_require_distinct_mechanism",
        "candidate_refill_feed_rejection_reasons", "candidate_refill_stop_when_requirements_met",
        "candidate_refill_max_solver_calls_per_agent_update", "probation_archive_enabled",
        "probation_archive_size_per_agent", "probation_archive_ttl_updates", "probation_max_accuracy_loss",
        "probation_max_c1_loss_questions", "probation_max_c2_loss_questions",
        "probation_require_mechanism_novelty", "candidate_c1_catastrophic_loss_questions",
        "candidate_c2_catastrophic_loss_questions", "qd_archive_size_per_agent",
        "joint_representative_beam_size", "qd_parent_selection_mode",
        "qd_niche_min_parent_opportunities_per_epoch", "probation_parent_enabled",
        "probe_stability_fold_count", "probe_stability_seed_offset", "joint_vote_band_questions",
        "joint_mean_band_correct_count", "joint_bottom2_band_correct_count", "joint_c1_band_questions",
        "joint_c2_band_questions", "joint_allowed_vote_loss_questions", "joint_allowed_c1_loss_questions",
        "joint_allowed_c2_loss_questions", "joint_allowed_total_agent_correct_loss",
        "joint_allowed_bottom2_correct_loss", "joint_allowed_per_agent_correct_loss",
        "joint_team_max_active_changes_early", "joint_team_max_active_changes_late",
        "joint_team_change_limit_switch_strength", "joint_team_no_diversification_patience",
        "joint_team_change_limit_relaxation", "lineage_commit_required_snapshots",
        "lineage_switch_confirmation_snapshots", "qd_readiness_min_distinct_niches",
        "qd_readiness_min_diversity", "qd_readiness_max_fold_gap", "residual_specialization_qd_floor",
        "behavior_error_overlap_weight", "behavior_wrong_answer_dispersion_weight",
        "behavior_wrong_support_shrinkage", "min_optimizer_updates_per_agent_per_epoch",
        "target_selector_fairness_enabled",
    )
    if str(getattr(cfg, "method_version", "legacy")) != "v8_stable_qd_lineage":
        for field in qd_fields:
            payload.pop(field, None)
    if str(getattr(cfg, "reward_mode", "")) != "coverage_useful_diversity":
        for field in (
            "reward_weight_coverage", "reward_weight_useful_diversity", "reward_weight_coverage_early",
            "reward_weight_coverage_late", "reward_weight_useful_diversity_early",
            "reward_weight_useful_diversity_late",
        ):
            payload.pop(field, None)
    return payload


def checkpoint_behavior_config_fingerprint(cfg):
    payload = checkpoint_behavior_config(cfg)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_config_signature(cfg):
    """Compatibility alias retained for older callers and tests."""
    return checkpoint_behavior_config(cfg)


def build_training_checkpoint(
    cfg,
    system,
    *,
    epoch_index,
    cursor,
    order,
    train_accumulators,
    best_score,
    best_epoch,
    epochs_without_improvement,
    stopped_early,
    no_effective_evolution_counter,
    no_effective_evolution_stopped,
    no_effective_evolution_reason,
    stage="training",
    epoch_record=None,
):
    payload = {
        "version": CHECKPOINT_VERSION,
        "stage": str(stage),
        "updated_at": time.time(),
        "seed": int(cfg.seed),
        "execution_session_id": str(getattr(system, "execution_session_id", "") or ""),
        "epochs": int(cfg.epochs),
        "train_size": int(cfg.train_size),
        "behavior_config": checkpoint_behavior_config(cfg),
        "behavior_config_fingerprint": checkpoint_behavior_config_fingerprint(cfg),
        "config_signature": checkpoint_config_signature(cfg),
        "epoch_index": int(epoch_index),
        "cursor": int(cursor),
        "order": [int(x) for x in order],
        "train_accumulators": train_accumulators,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "stopped_early": bool(stopped_early),
        "no_effective_evolution_counter": int(no_effective_evolution_counter),
        "no_effective_evolution_stopped": bool(no_effective_evolution_stopped),
        "no_effective_evolution_reason": str(no_effective_evolution_reason),
        "state": {
            "recent_window_records": list(getattr(system, "recent_window_records", [])),
            "specialization_strength": float(getattr(system, "specialization_strength", 0.0)),
            "effective_residual_strength": float(getattr(system, "effective_residual_strength", 0.0)),
            "previous_epoch_per_agent_acc": list(getattr(system, "previous_epoch_per_agent_acc", [])),
            "previous_epoch_bottom2_mean_acc": float(getattr(system, "previous_epoch_bottom2_mean_acc", 0.0)),
            "competence_phase_epoch": int(getattr(system, "competence_phase_epoch", 1)),
            "competence_schedule_version": str(getattr(system, "competence_schedule_version", "competence_depth_v1")),
            "specialization_strength_history": list(getattr(system, "specialization_strength_history", [0.0])),
            "competence_probe_indices": list(getattr(system, "competence_probe_indices", [])),
            "competence_probe_question_hashes": list(getattr(system, "competence_probe_question_hashes", [])),
            "initial_competence_probe_metrics": dict(getattr(system, "initial_competence_probe_metrics", {})),
            "latest_competence_probe_metrics": dict(getattr(system, "latest_competence_probe_metrics", {})),
            "competence_probe_history": list(getattr(system, "competence_probe_history", [])),
            "initial_active_prompt_hashes": list(getattr(system, "initial_active_prompt_hashes", [])),
            "first_nonzero_specialization_epoch": getattr(system, "first_nonzero_specialization_epoch", None),
            "effective_specialization_epoch_count": int(getattr(system, "effective_specialization_epoch_count", 0)),
            "depth1_guard_rejection_count": int(getattr(system, "depth1_guard_rejection_count", 0)),
            "catastrophic_accuracy_guard_rejection_count": int(getattr(system, "catastrophic_accuracy_guard_rejection_count", 0)),
            "soft_error_dependence_penalty_count": int(getattr(system, "soft_error_dependence_penalty_count", 0)),
            "soft_cycle_penalty_count": int(getattr(system, "soft_cycle_penalty_count", 0)),
            "soft_mechanism_shift_penalty_count": int(getattr(system, "soft_mechanism_shift_penalty_count", 0)),
            "exploration_candidate_count": int(getattr(system, "exploration_candidate_count", 0)),
            "exploration_slot_occupancy_count": int(getattr(system, "exploration_slot_occupancy_count", 0)),
            "exploration_to_active_conversion_count": int(getattr(system, "exploration_to_active_conversion_count", 0)),
            "hybrid_selector_history": list(getattr(system, "hybrid_selector_history", [])),
            "mechanism_signature_history": list(getattr(system, "mechanism_signature_history", [])),
            "mechanism_signature_by_prompt_hash": dict(getattr(system, "mechanism_signature_by_prompt_hash", {})),
            "beam_slot_state": dict(getattr(system, "beam_slot_state", {})),
            "exploration_slot_candidates": list(getattr(system, "exploration_slot_candidates", [])),
            "prompt_overlength_rejection_count": int(getattr(system, "prompt_overlength_rejection_count", 0)),
            "truncated_prompt_count": int(getattr(system, "truncated_prompt_count", 0)),
            "mechanism_embedding_cache": dict(getattr(system, "mechanism_embedding_cache", {})),
            "prompt_probe_cache": dict(getattr(system, "prompt_probe_cache", {})),
            "mechanism_embedding_cache_hit_count": int(getattr(system, "mechanism_embedding_cache_hit_count", 0)),
            "mechanism_embedding_cache_miss_count": int(getattr(system, "mechanism_embedding_cache_miss_count", 0)),
            "full_probe_cache_hit_count": int(getattr(system, "full_probe_cache_hit_count", 0)),
            "full_probe_missing_pair_evaluation_count": int(getattr(system, "full_probe_missing_pair_evaluation_count", 0)),
            "behavior_profile_by_prompt_hash": dict(getattr(system, "behavior_profile_by_prompt_hash", {})),
            "joint_team_selection_history": list(getattr(system, "joint_team_selection_history", [])),
            "lineage_history": list(getattr(system, "lineage_history", [])),
            "quality_diversity_archive_history": list(getattr(system, "quality_diversity_archive_history", [])),
            "behavior_profile_history": list(getattr(system, "behavior_profile_history", [])),
            "latest_joint_team_metrics": dict(getattr(system, "latest_joint_team_metrics", {})),
            "joint_quality_anchor_metrics": dict(getattr(system, "joint_quality_anchor_metrics", {})),
            "total_agent_update_count": int(getattr(system, "total_agent_update_count", 0)),
            "task_repair_niche_occupancy_count": int(getattr(system, "task_repair_niche_occupancy_count", 0)),
            "mechanism_niche_occupancy_count": int(getattr(system, "mechanism_niche_occupancy_count", 0)),
            "peer_collapse_soft_count": int(getattr(system, "peer_collapse_soft_count", 0)),
            "peer_collapse_hard_rejection_count": int(getattr(system, "peer_collapse_hard_rejection_count", 0)),
            "qd_no_diversification_epochs": int(getattr(system, "qd_no_diversification_epochs", 0)),
            "qd_change_limit_relaxed_epoch": int(getattr(system, "qd_change_limit_relaxed_epoch", -1)),
            "qd_previous_active_niche_count": int(getattr(system, "qd_previous_active_niche_count", 0)),
            "probation_to_safe_conversion_count": int(getattr(system, "probation_to_safe_conversion_count", 0)),
            "probation_expired_count": int(getattr(system, "probation_expired_count", 0)),
            "candidate_starvation_count": int(getattr(system, "candidate_starvation_count", 0)),
            "mechanism_starvation_count": int(getattr(system, "mechanism_starvation_count", 0)),
            "search_branch_starvation_count": int(getattr(system, "search_branch_starvation_count", 0)),
            "refill_requirements_unmet_count": int(getattr(system, "refill_requirements_unmet_count", 0)),
            "per_agent_optimizer_update_count": dict(getattr(system, "per_agent_optimizer_update_count", {})),
            "python_random_state": random.getstate(),
            "numpy_random_state": (
                lambda state: [state[0], state[1].tolist(), state[2], state[3], state[4]]
            )(np.random.get_state()),
            "agents": [
                {
                    "agent_id": i,
                    "initial_prompt": a.initial_prompt,
                    "current_prompt": a.current_prompt,
                    "prompt_beam": a.prompt_beam,
                    "history": a.history,
                    "accept_count": a.accept_count,
                    "reject_count": a.reject_count,
                    **a.trajectory_state_dict(),
                }
                for i, a in enumerate(system.agents)
            ],
        },
    }
    if epoch_record is not None:
        payload["epoch_record"] = epoch_record
    return payload


def write_training_checkpoint(cfg, system, **kwargs):
    write_json_atomic(checkpoint_path(cfg), build_training_checkpoint(cfg, system, **kwargs))


def clear_training_checkpoint(cfg):
    path = checkpoint_path(cfg)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def checkpoint_incompatibility_reasons(payload, cfg, train_data):
    reasons = []
    if not isinstance(payload, dict):
        return ["checkpoint payload is missing or is not a JSON object"]
    if int(payload.get("version", 0) or 0) != CHECKPOINT_VERSION:
        reasons.append(f"version: checkpoint={payload.get('version')!r} current={CHECKPOINT_VERSION}")
    if int(payload.get("seed", -1)) != int(cfg.seed):
        reasons.append(f"seed: checkpoint={payload.get('seed')!r} current={cfg.seed!r}")
    if int(payload.get("epochs", -1)) != int(cfg.epochs):
        reasons.append(f"epochs: checkpoint={payload.get('epochs')!r} current={cfg.epochs!r}")
    if int(payload.get("train_size", -1)) != int(cfg.train_size):
        reasons.append(f"train_size: checkpoint={payload.get('train_size')!r} current={cfg.train_size!r}")
    saved_config = payload.get("behavior_config")
    saved_fingerprint = str(payload.get("behavior_config_fingerprint", "") or "")
    current_config = checkpoint_behavior_config(cfg)
    current_fingerprint = checkpoint_behavior_config_fingerprint(cfg)
    if not isinstance(saved_config, dict) or not saved_fingerprint:
        reasons.append("behavior_config: checkpoint behavior configuration or fingerprint is missing")
    elif saved_fingerprint != current_fingerprint and _normalize_behavior_config_types(saved_config) != current_config:
        reasons.append(
            "behavior_config_fingerprint: checkpoint and current optimization behavior differ"
        )
        if (
            str(saved_config.get("method_version", "")) == "v8_2_hybrid_progressive"
            and str(current_config.get("method_version", "")) == "v8_stable_qd_lineage"
        ):
            reasons.append("V8 behavior fingerprint mismatch: joint quality-diversity lineage policy changed")
        if (
            str(saved_config.get("method_version", "")) == "v8_stable_qd_lineage"
            and not str(saved_config.get("candidate_refill_version", "") or "")
        ):
            reasons.append("Stable-QD checkpoint predates the refill/probation search-loop policy")
        for key in sorted(set(BEHAVIOR_CONFIG_FIELDS) | set(saved_config) | set(current_config)):
            saved_value = saved_config.get(key)
            current_value = current_config.get(key)
            if json.dumps(saved_value, sort_keys=True, default=str) != json.dumps(
                current_value, sort_keys=True, default=str
            ):
                label = f"{key} mismatch" if key in {"competence_schedule_version", "competence_schedule_mode"} else key
                reasons.append(f"{label}: checkpoint={saved_value!r} current={current_value!r}")
    epoch_index = int(payload.get("epoch_index", -1))
    if epoch_index < 0 or epoch_index > int(cfg.epochs):
        reasons.append(f"epoch_index: checkpoint={payload.get('epoch_index')!r} current_epochs={cfg.epochs!r}")
    stage = str(payload.get("stage", "training") or "training")
    order = payload.get("order", [])
    if stage in {"between_epochs", "epoch_evaluated"}:
        if not isinstance(order, list):
            reasons.append("order: checkpoint value is not a list")
        return reasons
    if stage != "training":
        reasons.append(f"stage: unsupported checkpoint stage {stage!r}")
        return reasons
    cursor = int(payload.get("cursor", -1))
    if not isinstance(order, list):
        reasons.append("order: checkpoint value is not a list")
    elif len(order) != len(train_data):
        reasons.append(f"order length: checkpoint={len(order)} current_train={len(train_data)}")
    if not (0 <= cursor <= len(order) if isinstance(order, list) else False):
        reasons.append(f"cursor: checkpoint={payload.get('cursor')!r} order_length={len(order) if isinstance(order, list) else 'invalid'}")
    state = payload.get("state", {})
    saved_window = state.get("recent_window_records") if isinstance(state, dict) else None
    expected_window_size = cursor % max(1, int(cfg.update_every))
    if saved_window is None and expected_window_size:
        reasons.append(
            "recent_window_records: checkpoint stopped inside an update window but does not contain window state"
        )
    elif saved_window is not None and not isinstance(saved_window, list):
        reasons.append("recent_window_records: checkpoint value is not a list")
    elif isinstance(saved_window, list) and len(saved_window) != expected_window_size:
        reasons.append(
            f"recent_window_records: checkpoint={len(saved_window)} expected={expected_window_size} for cursor={cursor}"
        )

    # Old resume code could advance a boundary after silently losing its window.
    train_step_path = os.path.join(cfg.out_dir, "train_step_logs.jsonl")
    if cursor > 0 and cursor % max(1, int(cfg.update_every)) == 0 and os.path.exists(train_step_path):
        try:
            with open(train_step_path, "r", encoding="utf-8") as f:
                last_row = next((json.loads(line) for line in reversed(f.readlines()) if line.strip()), {})
            update_summary = last_row.get("update_summary", {}) if isinstance(last_row, dict) else {}
            if (
                int(last_row.get("epoch", 0) or 0) == epoch_index + 1
                and int(last_row.get("step", 0) or 0) == cursor
                and isinstance(update_summary, dict)
                and str(update_summary.get("skipped_reason", "")) == "window_not_ready"
            ):
                reasons.append(
                    "recent_window_records: update boundary was skipped as window_not_ready; this run cannot be resumed faithfully"
                )
        except (OSError, json.JSONDecodeError):
            pass
    return reasons


def checkpoint_compatible(payload, cfg, train_data):
    return not checkpoint_incompatibility_reasons(payload, cfg, train_data)


def abort_incompatible_checkpoint(cfg, reasons):
    print(
        "[RESUME] ERROR: Incompatible training_checkpoint.json; refusing to start from scratch in the same run directory.",
        flush=True,
    )
    print(f"[RESUME] Checkpoint path: {checkpoint_path(cfg)}", flush=True)
    for reason in reasons[:20]:
        print(f"[RESUME] Incompatibility: {reason}", flush=True)
    if len(reasons) > 20:
        print(f"[RESUME] Incompatibility: ... {len(reasons) - 20} more", flush=True)
    raise SystemExit(2)


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
        "candidate_refill_enabled", "candidate_refill_require_task_repair",
        "candidate_refill_require_distinct_mechanism", "candidate_refill_feed_rejection_reasons",
        "candidate_refill_stop_when_requirements_met", "probation_archive_enabled",
        "probation_require_mechanism_novelty", "probation_parent_enabled",
        "target_selector_fairness_enabled",
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

    adaptive_competence_schedule = bool(cfg.competence_depth_enabled) and str(cfg.competence_schedule_mode) == "baseline_relative_opt_snapshot"
    fixed_competence_probe = []
    if adaptive_competence_schedule:
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
        if cfg.beam_refresh_each_epoch:
            refresh_batch_size = max(1, int(cfg.candidate_eval_batch_size or 10))
            refresh_batch = [train_data[i] for i in order[: min(refresh_batch_size, len(order))]]
            refresh_summary = await system.refresh_all_prompt_beams(refresh_batch, epoch_id=epoch + 1)
        joint_team_summary = None
        if str(getattr(cfg, "method_version", "legacy")) == "v8_stable_qd_lineage":
            system.expire_probation_branches(epoch + 1)
            joint_team_summary = await system.select_joint_active_team(
                fixed_competence_probe, epoch=epoch + 1
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
        train_metrics["specialization_strength"] = strength_used
        train_metrics["next_epoch_specialization_strength"] = float(system.specialization_strength)
        epoch_record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
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
    clear_training_checkpoint(cfg)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
