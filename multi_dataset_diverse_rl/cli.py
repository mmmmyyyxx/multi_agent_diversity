import asyncio
import json
import os
import random

import numpy as np

from .config import Config, build_parser
from .system import TextualGradientRLSystem
from .utils import ensure_dir, extract_question_answer, load_jsonl, parse_gold, set_seed


def build_dataset(raw_records):
    out = []
    for x in raw_records:
        q, a = extract_question_answer(x)
        out.append({"question": q, "answer": a})
    return out


async def main_async():
    parser = build_parser()
    args = parser.parse_args()
    cfg = Config(**vars(args))
    cfg.transient_retry_forever = bool(int(cfg.transient_retry_forever))
    cfg.baseline_only = bool(int(cfg.baseline_only))
    cfg.family_expansion_enabled = bool(int(cfg.family_expansion_enabled))
    cfg.use_dual_family_labels = bool(int(cfg.use_dual_family_labels))

    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)

    raw_test = load_jsonl(cfg.test_path, cfg.test_size)
    test_data = build_dataset(raw_test)

    if cfg.baseline_only:
        print(f"Loaded baseline test={len(test_data)}")
    else:
        raw_train = load_jsonl(cfg.train_path, cfg.train_size)
        train_data = build_dataset(raw_train)
        print(f"Loaded train={len(train_data)} test={len(test_data)}")

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

    best_test_diversity = -1.0
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
        test_metrics = await system.evaluate_dataset(test_data, split_name=f"test_epoch{epoch + 1}")

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "test": test_metrics,
        }
        system.history.append(epoch_record)

        print(
            f"Epoch {epoch + 1}: "
            f"train_family_homo_rate={train_metrics['mean_family_homogeneity_rate']:.4f}, "
            f"train_family_div={train_metrics['mean_family_diversity']:.4f}, "
            f"train_vote_acc={train_metrics['vote_acc']:.4f}, "
            f"test_family_homo_rate={test_metrics['mean_family_homogeneity_rate']:.4f}, "
            f"test_family_div={test_metrics['mean_family_diversity']:.4f}, "
            f"test_vote_acc={test_metrics['vote_acc']:.4f}"
        )

        system.save_state("last_state", extra=epoch_record)
        system.flush_train_step_logs()
        system.flush_train_trace_history_logs()
        system.flush_test_trace_history_logs()
        system.flush_reasoning_summary_history_logs()
        system.flush_prompt_history()
        with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(system.history, f, ensure_ascii=False, indent=2)

        if test_metrics["mean_family_diversity"] > best_test_diversity:
            best_test_diversity = test_metrics["mean_family_diversity"]
            system.save_state("best_state", extra=epoch_record)

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
