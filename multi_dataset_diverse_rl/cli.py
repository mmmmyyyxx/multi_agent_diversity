import asyncio
import json
import os
import random

import numpy as np

from .config import Config, build_parser
from .utils import ensure_dir, extract_question_answer, load_jsonl, parse_gold, set_seed


def build_dataset(raw_records):
    out = []
    for x in raw_records:
        q, a = extract_question_answer(x)
        out.append({"question": q, "answer": a})
    return out


def split_train_validation(raw_train, cfg):
    records = list(raw_train)
    if not records:
        return [], []

    ratio = max(0.0, min(0.8, float(getattr(cfg, "val_split_ratio", 0.2))))
    requested_val = int(getattr(cfg, "val_size", 0) or 0)
    if requested_val <= 0:
        requested_val = int(round(len(records) * ratio))
    requested_val = max(1, min(requested_val, max(1, len(records) - 1)))

    rng = random.Random(int(cfg.seed))
    indices = list(range(len(records)))
    rng.shuffle(indices)
    val_indices = set(indices[:requested_val])
    train_records = [records[i] for i in indices[requested_val:]]
    val_records = [records[i] for i in indices[:requested_val]]
    return train_records, val_records


def early_stopping_value(epoch_record, metric_name):
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), dict) else {}
    if metric_name == "val_mean_family_homogeneity_rate":
        return -float(val.get("mean_family_homogeneity_rate", 0.0) or 0.0)
    return float(val.get("mean_family_diversity", 0.0) or 0.0)


def snapshot_agent_prompts(system):
    return [str(agent.current_prompt) for agent in system.agents]


def restore_agent_prompts(system, prompts):
    if not prompts or len(prompts) != len(system.agents):
        raise ValueError(f"Cannot restore prompts: got {len(prompts) if prompts else 0}, expected {len(system.agents)}")
    for agent, prompt in zip(system.agents, prompts):
        agent.current_prompt = str(prompt)


def write_selected_prompts(path, system, epoch, metric_name, validation_score):
    prompts = snapshot_agent_prompts(system)
    payload = {
        "selected_epoch": epoch,
        "early_stopping_metric": metric_name,
        "best_validation_score": validation_score,
        "agents": [
            {
                "agent_id": i,
                "prompt_hash": system._prompt_hash(prompt),
                "prompt": prompt,
            }
            for i, prompt in enumerate(prompts)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def read_selected_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    agents = payload.get("agents", []) if isinstance(payload, dict) else []
    prompts = [str(agent.get("prompt", "")) for agent in agents if isinstance(agent, dict)]
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError(f"No valid prompts found in {path}")
    return payload, prompts


async def main_async():
    parser = build_parser()
    args = parser.parse_args()
    cfg = Config(**vars(args))
    cfg.transient_retry_forever = bool(int(cfg.transient_retry_forever))
    cfg.baseline_only = bool(int(cfg.baseline_only))
    cfg.family_expansion_enabled = bool(int(cfg.family_expansion_enabled))
    cfg.use_dual_family_labels = bool(int(cfg.use_dual_family_labels))
    cfg.family_rejudge_on_low_confidence = bool(int(cfg.family_rejudge_on_low_confidence))
    cfg.llm_call_logging = bool(int(cfg.llm_call_logging))
    cfg.eval_test_each_epoch = bool(int(cfg.eval_test_each_epoch))

    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)

    raw_test = load_jsonl(cfg.test_path, cfg.test_size)
    test_data = build_dataset(raw_test)

    if cfg.baseline_only:
        print(f"Loaded baseline test={len(test_data)}")
    else:
        raw_train = load_jsonl(cfg.train_path, cfg.train_size)
        if cfg.val_path:
            raw_val = load_jsonl(cfg.val_path, cfg.val_size)
            train_data = build_dataset(raw_train)
            val_data = build_dataset(raw_val)
            val_source = cfg.val_path
        else:
            split_train, split_val = split_train_validation(raw_train, cfg)
            train_data = build_dataset(split_train)
            val_data = build_dataset(split_val)
            val_source = f"{cfg.train_path}:split"
        print(f"Loaded train={len(train_data)} val={len(val_data)} test={len(test_data)} val_source={val_source}")

    from .system import TextualGradientRLSystem

    system = TextualGradientRLSystem(cfg)

    if cfg.baseline_only:
        test_metrics = await system.evaluate_dataset(test_data, split_name="test_epoch1")
        train_metrics = {
            "mean_family_homogeneity_rate": 0.0,
            "mean_family_diversity": 0.0,
            "mean_llm_direct_diversity_score": 0.0,
            "vote_acc": 0.0,
        }
        epoch_record = {
            "epoch": 1,
            "train": train_metrics,
            "test": test_metrics,
        }
        system.history.append(epoch_record)
        print(
            "Baseline: "
            f"test_family_div={test_metrics['mean_family_diversity']:.4f}, "
            f"test_family_homo_rate={test_metrics['mean_family_homogeneity_rate']:.4f}, "
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
        system.flush_reasoning_summary_history_logs()
        system.flush_prompt_history()
        return

    best_validation_score = -1e30
    best_epoch = 0
    best_prompts_path = os.path.join(cfg.out_dir, "best_prompts.json")
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(cfg.epochs):
        order = list(range(len(train_data)))
        random.shuffle(order)

        train_family_homogeneity_rate = []
        train_family_diversity = []
        train_direct_diversity = []
        train_vote_correct = []

        for step, idx in enumerate(order):
            ex = train_data[idx]
            q = ex["question"]
            gold = parse_gold(ex["answer"], cfg.task_type, question=q)

            eval_batch = []
            if cfg.candidate_eval_batch_size > 0:
                batch_indices = [idx]
                while len(batch_indices) < min(cfg.candidate_eval_batch_size, len(train_data)):
                    batch_indices.append(random.choice(order))
                eval_batch = [train_data[b] for b in batch_indices]

            out = await system.rollout_train_example(
                q,
                gold,
                do_update=((step + 1) % cfg.update_every == 0),
                eval_batch=eval_batch,
                step_id=step + 1,
                epoch_id=epoch + 1,
            )
            train_family_homogeneity_rate.append(float(out.get("team_family_homogeneity_rate", 0.0)))
            train_family_diversity.append(float(out.get("team_family_diversity", 0.0)))
            if out.get("llm_direct_diversity_score") is not None:
                train_direct_diversity.append(float(out.get("llm_direct_diversity_score", 0.0)))
            train_vote_correct.append(out["vote_correct"])

            if (step + 1) % 10 == 0 or (step + 1) == len(order):
                print(
                    f"Epoch {epoch + 1} Step {step + 1}/{len(order)} "
                    f"train_family_homo_rate={float(np.mean(train_family_homogeneity_rate)):.4f} "
                    f"train_family_div={float(np.mean(train_family_diversity)):.4f} "
                    f"train_vote_acc={float(np.mean(train_vote_correct)):.4f}"
                )

        train_metrics = {
            "mean_family_homogeneity_rate": float(np.mean(train_family_homogeneity_rate)) if train_family_homogeneity_rate else 0.0,
            "mean_family_diversity": float(np.mean(train_family_diversity)) if train_family_diversity else 0.0,
            "mean_llm_direct_diversity_score": float(np.mean(train_direct_diversity)) if train_direct_diversity else 0.0,
            "vote_acc": float(np.mean(train_vote_correct)) if train_vote_correct else 0.0,
        }
        val_metrics = await system.evaluate_dataset(val_data, split_name=f"val_epoch{epoch + 1}")
        test_metrics = None
        if cfg.eval_test_each_epoch:
            test_metrics = await system.evaluate_dataset(test_data, split_name=f"test_epoch{epoch + 1}")

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        }
        if test_metrics is not None:
            epoch_record["test"] = test_metrics
        system.history.append(epoch_record)

        print(
            f"Epoch {epoch + 1}: "
            f"train_family_homo_rate={train_metrics['mean_family_homogeneity_rate']:.4f}, "
            f"train_family_div={train_metrics['mean_family_diversity']:.4f}, "
            f"train_vote_acc={train_metrics['vote_acc']:.4f}, "
            f"val_family_homo_rate={val_metrics['mean_family_homogeneity_rate']:.4f}, "
            f"val_family_div={val_metrics['mean_family_diversity']:.4f}, "
            f"val_vote_acc={val_metrics['vote_acc']:.4f}"
        )

        system.save_state("last_state", extra=epoch_record)
        system.flush_train_step_logs()
        system.flush_train_trace_history_logs()
        system.flush_test_trace_history_logs()
        system.flush_reasoning_summary_history_logs()
        system.flush_prompt_history()
        with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(system.history, f, ensure_ascii=False, indent=2)

        validation_score = early_stopping_value(epoch_record, cfg.early_stopping_metric)
        if validation_score > best_validation_score + float(cfg.early_stopping_min_delta):
            best_validation_score = validation_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            system.save_state("best_state", extra=epoch_record)
            write_selected_prompts(
                best_prompts_path,
                system,
                epoch=best_epoch,
                metric_name=cfg.early_stopping_metric,
                validation_score=best_validation_score,
            )
        else:
            epochs_without_improvement += 1

        if int(cfg.early_stopping_patience) >= 0 and epochs_without_improvement > int(cfg.early_stopping_patience):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"best_epoch={best_epoch}, metric={cfg.early_stopping_metric}, "
                f"best_validation_score={best_validation_score:.4f}"
            )
            break

    final_training_record = {
        "epoch": "final_training_state",
        "selected_epoch": best_epoch,
        "early_stopped": stopped_early,
        "early_stopping_metric": cfg.early_stopping_metric,
        "best_validation_score": best_validation_score,
        "note": "State before restoring validation-selected best prompts for final test.",
    }
    system.save_state("final_training_state", extra=final_training_record)

    if not os.path.exists(best_prompts_path):
        best_epoch = best_epoch or len([h for h in system.history if isinstance(h.get("epoch"), int)])
        write_selected_prompts(
            best_prompts_path,
            system,
            epoch=best_epoch,
            metric_name=cfg.early_stopping_metric,
            validation_score=best_validation_score,
        )
    selected_prompt_payload, selected_prompts = read_selected_prompts(best_prompts_path)
    restore_agent_prompts(system, selected_prompts)
    selected_prompt_hashes = [system._prompt_hash(prompt) for prompt in selected_prompts]
    best_epoch = int(selected_prompt_payload.get("selected_epoch", best_epoch) or best_epoch)
    print(
        "Restored validation-selected best prompts before final test: "
        f"selected_epoch={best_epoch}, metric={cfg.early_stopping_metric}, "
        f"best_validation_score={best_validation_score:.4f}, "
        f"prompts_file={best_prompts_path}"
    )

    final_test_metrics = await system.evaluate_dataset(test_data, split_name="test_final")
    final_record = {
        "epoch": "final",
        "selected_epoch": best_epoch,
        "early_stopped": stopped_early,
        "early_stopping_metric": cfg.early_stopping_metric,
        "best_validation_score": best_validation_score,
        "test_evaluated_on": "best_state",
        "selected_prompts_path": best_prompts_path,
        "selected_prompt_hashes": selected_prompt_hashes,
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
    system.flush_reasoning_summary_history_logs()
    system.flush_prompt_history()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
