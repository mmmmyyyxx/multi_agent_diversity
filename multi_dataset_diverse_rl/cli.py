import asyncio
import json
import os
import random

import numpy as np

from .config import Config, build_parser
from .utils import ensure_dir, extract_question_answer, load_jsonl, parse_gold, set_seed


def build_dataset(raw_records):
    return [{"question": extract_question_answer(x)[0], "answer": extract_question_answer(x)[1]} for x in raw_records]


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


def snapshot_agent_prompts(system):
    return [str(agent.current_prompt) for agent in system.agents]


def restore_agent_prompts(system, prompts):
    if not prompts or len(prompts) != len(system.agents):
        raise ValueError(f"Cannot restore prompts: got {len(prompts) if prompts else 0}, expected {len(system.agents)}")
    for agent, prompt in zip(system.agents, prompts):
        agent.current_prompt = str(prompt)
        if agent.prompt_beam:
            agent.prompt_beam[0]["prompt"] = str(prompt)
        else:
            agent.prompt_beam = [{"id": "", "prompt": str(prompt), "score": None, "metrics": {}, "parent_id": None, "generation": 0}]


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


def validation_score(epoch_record, reward_mode="embedding_local_acc_invalid"):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    if str(reward_mode).lower() == "accuracy_only":
        return float(val.get("vote_acc", 0.0) or 0.0)
    return (
        float(val.get("vote_acc", 0.0) or 0.0)
        + 0.2 * float(val.get("mean_embedding_diversity", 0.0) or 0.0)
        - 0.1 * float(val.get("mean_invalid_rate", 0.0) or 0.0)
    )


def validation_metric_name(reward_mode):
    if str(reward_mode).lower() == "accuracy_only":
        return "vote_acc"
    return "vote_acc+embedding_div-invalid"


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
    cfg.local_validity_binary = bool(int(cfg.local_validity_binary))
    cfg.invalid_binary = bool(int(cfg.invalid_binary))
    cfg.candidate_reuse_recorded_rollouts = bool(int(cfg.candidate_reuse_recorded_rollouts))
    cfg.transient_retry_forever = bool(int(cfg.transient_retry_forever))
    cfg.llm_call_logging = bool(int(cfg.llm_call_logging))

    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)

    raw_test = load_jsonl(cfg.test_path, cfg.test_size)
    test_data = build_dataset(raw_test)

    if cfg.baseline_only:
        train_data = []
        val_data = []
        print(f"Loaded baseline test={len(test_data)}")
    else:
        raw_train = load_jsonl(cfg.train_path, cfg.train_size)
        if cfg.val_path:
            train_data = build_dataset(raw_train)
            val_data = build_dataset(load_jsonl(cfg.val_path, cfg.val_size))
            val_source = cfg.val_path
        else:
            split_train, split_val = split_train_validation(raw_train, cfg)
            train_data = build_dataset(split_train)
            val_data = build_dataset(split_val)
            val_source = f"{cfg.train_path}:split"
        print(f"Loaded train={len(train_data)} val={len(val_data)} test={len(test_data)} val_source={val_source}")

    from .system import TraceBeamSearchSystem

    system = TraceBeamSearchSystem(cfg)

    if cfg.baseline_only:
        test_metrics = await system.evaluate_dataset(test_data, split_name="test_epoch1")
        epoch_record = {"epoch": 1, "train": {}, "test": test_metrics}
        system.history.append(epoch_record)
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
        return

    best_score = -1e30
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    best_prompts_path = os.path.join(cfg.out_dir, "best_prompts.json")
    for epoch in range(cfg.epochs):
        order = list(range(len(train_data)))
        random.shuffle(order)
        train_embedding_diversity = []
        train_embedding_overlap = []
        train_invalid_rate = []
        train_vote_correct = []
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
                gold = parse_gold(ex["answer"], cfg.task_type, question=q)
                solved = await system.solve_train_example_without_update(q, gold)
                return pos, idx, solved

            solved_rows = await asyncio.gather(*[solve_position(pos) for pos in batch_positions])
            solved_rows.sort(key=lambda x: x[0])

            for pos, idx, solved in solved_rows:
                step = pos
                eval_batch = []
                if cfg.candidate_eval_batch_size > 0:
                    batch_indices = [idx]
                    while len(batch_indices) < min(cfg.candidate_eval_batch_size, len(train_data)):
                        batch_indices.append(random.choice(order))
                    eval_batch = [train_data[b] for b in batch_indices]
                do_update = ((step + 1) % cfg.update_every == 0)
                out = await system.record_train_rollout(
                    solved,
                    do_update=do_update,
                    eval_batch=eval_batch,
                    step_id=step + 1,
                    epoch_id=epoch + 1,
                )
                train_embedding_diversity.append(float(out.get("embedding_diversity", 0.0)))
                train_embedding_overlap.append(float(out.get("mean_embedding_overlap", 0.0)))
                train_invalid_rate.append(float(out.get("invalid_rate", 0.0)))
                train_vote_correct.append(int(out.get("vote_correct", 0)))

                if (step + 1) % 10 == 0 or (step + 1) == len(order):
                    print(
                        f"Epoch {epoch + 1} Step {step + 1}/{len(order)} "
                        f"train_embedding_div={float(np.mean(train_embedding_diversity)):.4f} "
                        f"train_embedding_overlap={float(np.mean(train_embedding_overlap)):.4f} "
                        f"train_invalid={float(np.mean(train_invalid_rate)):.4f} "
                        f"train_vote_acc={float(np.mean(train_vote_correct)):.4f}"
                    )
            cursor = batch_end

        train_metrics = {
            "mean_embedding_diversity": float(np.mean(train_embedding_diversity)) if train_embedding_diversity else 0.0,
            "mean_embedding_overlap": float(np.mean(train_embedding_overlap)) if train_embedding_overlap else 0.0,
            "mean_invalid_rate": float(np.mean(train_invalid_rate)) if train_invalid_rate else 0.0,
            "vote_acc": float(np.mean(train_vote_correct)) if train_vote_correct else 0.0,
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

    if os.path.exists(best_prompts_path):
        payload, prompts = read_selected_prompts(best_prompts_path)
        restore_agent_prompts(system, prompts)
        best_epoch = int(payload.get("selected_epoch", best_epoch) or best_epoch)

    final_test_metrics = await system.evaluate_dataset(test_data, split_name="test_final")
    final_record = {
        "epoch": "final",
        "selected_epoch": best_epoch,
        "best_validation_score": best_score,
        "early_stopped": bool(stopped_early),
        "early_stopping_patience": int(getattr(cfg, "early_stopping_patience", -1)),
        "early_stopping_min_delta": float(getattr(cfg, "early_stopping_min_delta", 0.0) or 0.0),
        "test_evaluated_on": "best_state",
        "test": final_test_metrics,
    }
    system.history.append(final_record)
    system.save_state("selected_state", extra=final_record)
    system.save_state("last_state", extra=final_record)
    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(system.history, f, ensure_ascii=False, indent=2)
    system.flush_update_logs()
    system.flush_train_step_logs()
    system.flush_train_trace_history_logs()
    system.flush_test_trace_history_logs()
    system.flush_prompt_history()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
