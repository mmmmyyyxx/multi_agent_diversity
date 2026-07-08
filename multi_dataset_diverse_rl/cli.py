import asyncio
import json
import os
import random

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


def validation_score(epoch_record, reward_mode="guarded_diversity"):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    mode = str(reward_mode).lower()
    if mode == "accuracy_only":
        return float(val.get("vote_acc", 0.0) or 0.0)
    if mode in {"coverage_useful_diversity", "coverage_rescue_diversity"}:
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
    if mode in {"coverage_useful_diversity", "coverage_rescue_diversity"}:
        return "vote+oracle+useful_div-invalid"
    return "vote_acc+embedding_div-invalid"


def uses_coverage_useful_metrics(reward_mode):
    return str(reward_mode).lower() in {"coverage_useful_diversity", "coverage_rescue_diversity"}


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
    for epoch in range(cfg.epochs):
        order = list(range(len(train_data)))
        random.shuffle(order)
        train_embedding_diversity = []
        train_embedding_overlap = []
        train_invalid_rate = []
        train_vote_correct = []
        train_any_correct = []
        train_useful_diversity = []
        train_rollout_concurrency = max(1, auto_train_rollout_concurrency(cfg))

        cursor = 0
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
                    if uses_coverage_useful_metrics(cfg.reward_mode):
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
                    else:
                        print(
                            f"Epoch {epoch + 1} Step {step + 1}/{len(order)} "
                            f"train_embedding_div={float(np.mean(train_embedding_diversity)):.4f} "
                            f"train_embedding_overlap={float(np.mean(train_embedding_overlap)):.4f} "
                            f"train_invalid={float(np.mean(train_invalid_rate)):.4f} "
                            f"train_vote_acc={float(np.mean(train_vote_correct)):.4f}"
                        )
                if no_effective_evolution_stopped:
                    break
            cursor = batch_end
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
        system.history.append(epoch_record)

        if uses_coverage_useful_metrics(cfg.reward_mode):
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
        else:
            print(
                f"Epoch {epoch + 1}: "
                f"train_embedding_div={train_metrics['mean_embedding_diversity']:.4f}, "
                f"train_embedding_overlap={train_metrics['mean_embedding_overlap']:.4f}, "
                f"train_invalid={train_metrics['mean_invalid_rate']:.4f}, "
                f"train_vote_acc={train_metrics['vote_acc']:.4f}, "
                f"val_embedding_div={val_metrics['mean_embedding_diversity']:.4f}, "
                f"val_embedding_overlap={val_metrics['mean_embedding_overlap']:.4f}, "
                f"val_invalid={val_metrics['mean_invalid_rate']:.4f}, "
                f"val_vote_acc={val_metrics['vote_acc']:.4f}"
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
        if patience >= 0 and epochs_without_improvement >= patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"best_epoch={best_epoch}, metric={validation_metric_name(cfg.reward_mode)}, "
                f"best_validation_score={best_score:.4f}, "
                f"epochs_without_improvement={epochs_without_improvement}"
            )
            break
        if no_effective_evolution_stopped:
            stopped_early = True
            break

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


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
