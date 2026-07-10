import asyncio
import json
import os
import random
import time

import numpy as np

from .config import Config, build_parser
from .utils import ensure_dir, load_jsonl, set_seed


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
    source = list(val_data or train_data or [])
    if not source:
        return []
    pool_size = min(max(1, int(cfg.candidate_eval_pool_size or 1)), len(source))
    rng = random.Random(int(cfg.seed) + int(cfg.candidate_eval_seed_offset))
    indices = list(range(len(source)))
    rng.shuffle(indices)
    return [source[i] for i in indices[:pool_size]]


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
    for repeat in range(repeats):
        rng = random.Random(base_seed + repeat)
        if strategy == "random":
            if not train_data:
                batch = []
            else:
                indices = []
                if anchor_idx is not None and 0 <= int(anchor_idx) < len(train_data):
                    indices.append(int(anchor_idx))
                while len(indices) < min(batch_size, len(train_data)):
                    indices.append(rng.randrange(len(train_data)))
                batch = [train_data[i] for i in indices[:batch_size]]
        else:
            source = candidate_eval_pool or train_data
            if strategy == "stratified":
                batch = _stratified_sample(list(source), min(batch_size, len(source)), rng)
            else:
                indices = list(range(len(source)))
                rng.shuffle(indices)
                batch = [source[i] for i in indices[: min(batch_size, len(source))]]
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


def write_selected_prompts(path, system, epoch, metric_name, validation_score):
    prompts = snapshot_agent_prompts(system)
    payload = {
        "selected_epoch": epoch,
        "early_stopping_metric": metric_name,
        "best_validation_score": validation_score,
        "agents": [
            {"agent_id": i, "prompt_hash": system._hash(prompt), "prompt": prompt}
            for i, prompt in enumerate(prompts)
        ],
    }
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
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


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


def checkpoint_config_signature(cfg):
    fields = [
        "task_type",
        "dataset_format",
        "comparison_task_id",
        "benchmark",
        "answer_format",
        "train_path",
        "val_path",
        "test_path",
        "agents",
        "init_mode",
        "reward_mode",
        "optimizer_architecture",
        "beam_size",
        "num_candidates_per_parent",
        "update_every",
        "candidate_eval_batch_size",
        "candidate_eval_strategy",
        "candidate_eval_pool_size",
        "candidate_eval_seed_offset",
        "agent_model",
        "max_tokens",
        "temperature",
    ]
    return {field: str(getattr(cfg, field, "")) for field in fields}


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
        "version": 1,
        "stage": str(stage),
        "updated_at": time.time(),
        "seed": int(cfg.seed),
        "epochs": int(cfg.epochs),
        "train_size": int(cfg.train_size),
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
            "agents": [
                {
                    "agent_id": i,
                    "initial_prompt": a.initial_prompt,
                    "current_prompt": a.current_prompt,
                    "prompt_beam": a.prompt_beam,
                    "history": a.history,
                    "accept_count": a.accept_count,
                    "reject_count": a.reject_count,
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
    if int(payload.get("version", 0) or 0) != 1:
        reasons.append(f"version: checkpoint={payload.get('version')!r} current=1")
    if int(payload.get("seed", -1)) != int(cfg.seed):
        reasons.append(f"seed: checkpoint={payload.get('seed')!r} current={cfg.seed!r}")
    if int(payload.get("epochs", -1)) != int(cfg.epochs):
        reasons.append(f"epochs: checkpoint={payload.get('epochs')!r} current={cfg.epochs!r}")
    if int(payload.get("train_size", -1)) != int(cfg.train_size):
        reasons.append(f"train_size: checkpoint={payload.get('train_size')!r} current={cfg.train_size!r}")
    saved_signature = payload.get("config_signature", {})
    if isinstance(saved_signature, dict):
        current_signature = checkpoint_config_signature(cfg)
        for key, value in saved_signature.items():
            if key in current_signature and str(current_signature[key]) != str(value):
                reasons.append(f"{key}: checkpoint={value!r} current={current_signature[key]!r}")
    else:
        reasons.append("config_signature: checkpoint value is missing or is not an object")
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
    return (
        float(val.get("vote_acc", 0.0) or 0.0)
        + 0.2 * float(val.get("mean_embedding_diversity", 0.0) or 0.0)
        - 0.1 * float(val.get("mean_invalid_rate", 0.0) or 0.0)
    )


def validation_metric_name(reward_mode):
    mode = str(reward_mode).lower()
    if mode == "accuracy_only":
        return "vote_acc"
    if mode == "coverage_useful_diversity":
        return "vote+oracle+useful_div-invalid"
    return "vote_acc+embedding_div-invalid"


def uses_coverage_useful_metrics(reward_mode):
    return str(reward_mode).lower() == "coverage_useful_diversity"


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
    cfg.transient_retry_forever = bool(int(cfg.transient_retry_forever))
    cfg.llm_call_logging = bool(int(cfg.llm_call_logging))
    cfg.no_effective_evolution_stop_enabled = bool(int(cfg.no_effective_evolution_stop_enabled))
    cfg.teacher_critic_use_voting_failure = bool(int(cfg.teacher_critic_use_voting_failure))
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
        if uses_coverage_useful_metrics(cfg.reward_mode):
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
        elif candidate is not None:
            abort_incompatible_checkpoint(cfg, incompatibility_reasons)

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
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch_num
            epochs_without_improvement = 0
            system.save_state("best_state", extra=resume_epoch_record)
            write_selected_prompts(best_prompts_path, system, best_epoch, validation_metric_name(cfg.reward_mode), best_score)
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
                f"best_epoch={best_epoch}, metric={validation_metric_name(cfg.reward_mode)}, "
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
        if resume_payload is not None and epoch == resume_epoch_index:
            train_embedding_diversity = [float(x) for x in resume_accumulators.get("train_embedding_diversity", [])]
            train_embedding_overlap = [float(x) for x in resume_accumulators.get("train_embedding_overlap", [])]
            train_invalid_rate = [float(x) for x in resume_accumulators.get("train_invalid_rate", [])]
            train_vote_correct = [int(x) for x in resume_accumulators.get("train_vote_correct", [])]
            train_any_correct = [int(x) for x in resume_accumulators.get("train_any_correct", [])]
            train_useful_diversity = [float(x) for x in resume_accumulators.get("train_useful_diversity", [])]
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
        }
        refresh_summary = None
        if cfg.beam_refresh_each_epoch:
            refresh_batch_size = max(1, int(cfg.candidate_eval_batch_size or 10))
            refresh_batch = [train_data[i] for i in order[: min(refresh_batch_size, len(order))]]
            refresh_summary = await system.refresh_all_prompt_beams(refresh_batch, epoch_id=epoch + 1)

        val_metrics = await system.evaluate_dataset(val_data, split_name=f"val_epoch{epoch + 1}")
        epoch_record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
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
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            system.save_state("best_state", extra=epoch_record)
            write_selected_prompts(best_prompts_path, system, best_epoch, validation_metric_name(cfg.reward_mode), best_score)
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
                f"best_epoch={best_epoch}, metric={validation_metric_name(cfg.reward_mode)}, "
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
