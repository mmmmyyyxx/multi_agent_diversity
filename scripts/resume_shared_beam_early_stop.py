import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.cli import auto_train_rollout_concurrency, build_dataset, validation_score, write_selected_prompts
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.utils import load_jsonl, parse_gold, set_seed


def _read_json(path: Path, default: Any):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _numeric_epoch_rows(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in history if isinstance(row, dict) and isinstance(row.get("epoch"), int)]


def _restore_from_prompt_history(system: TraceBeamSearchSystem, prompt_history: Dict[str, Any]) -> bool:
    restored = False
    for agent_id, agent in enumerate(system.agents):
        row = prompt_history.get(str(agent_id), {}) if isinstance(prompt_history, dict) else {}
        beam = row.get("prompt_beam", [])
        if isinstance(beam, list) and beam:
            agent.prompt_beam = [dict(x) for x in beam if isinstance(x, dict)]
            if agent.prompt_beam:
                agent.current_prompt = str(agent.prompt_beam[0].get("prompt", agent.current_prompt))
                restored = True
        current = str(row.get("current_prompt", "") or "")
        if current:
            agent.current_prompt = current
            if not agent.prompt_beam:
                agent.prompt_beam = [system._make_beam_item(current, None, {}, None, 0)]
            elif str(agent.prompt_beam[0].get("prompt", "")) != current:
                agent.prompt_beam[0]["prompt"] = current
            restored = True
    return restored


def _restore_from_state(system: TraceBeamSearchSystem, state: Dict[str, Any]) -> bool:
    agents = state.get("agents", []) if isinstance(state, dict) else []
    if not isinstance(agents, list) or not agents:
        return False
    restored = False
    for row in agents:
        if not isinstance(row, dict):
            continue
        agent_id = int(row.get("agent_id", -1))
        if not (0 <= agent_id < len(system.agents)):
            continue
        agent = system.agents[agent_id]
        prompt = str(row.get("current_prompt", "") or "")
        if prompt:
            agent.current_prompt = prompt
            restored = True
        beam = row.get("prompt_beam", [])
        if isinstance(beam, list) and beam:
            agent.prompt_beam = [dict(x) for x in beam if isinstance(x, dict)]
            restored = True
        if not agent.prompt_beam:
            agent.prompt_beam = [system._make_beam_item(agent.current_prompt, None, {}, None, 0)]
        agent.history = [str(x) for x in row.get("history", []) if str(x)] or [agent.initial_prompt]
        agent.accept_count = int(row.get("accept_count", 0) or 0)
        agent.reject_count = int(row.get("reject_count", 0) or 0)
    return restored


def _best_validation(history: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    best_score = -1e30
    best_epoch = 0
    epochs_since_improvement = 0
    for row in _numeric_epoch_rows(history):
        score = validation_score(row)
        epoch = int(row.get("epoch", 0))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
    return best_score, best_epoch, epochs_since_improvement


def _load_best_prompts_score(run_dir: Path, fallback_score: float, fallback_epoch: int) -> Tuple[float, int]:
    path = run_dir / "best_prompts.json"
    if not path.exists():
        return fallback_score, fallback_epoch
    payload = _read_json(path, {})
    return (
        float(payload.get("best_validation_score", fallback_score) or fallback_score),
        int(payload.get("selected_epoch", fallback_epoch) or fallback_epoch),
    )


async def _train_one_epoch(
    system: TraceBeamSearchSystem,
    cfg: Config,
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
    epoch: int,
) -> Dict[str, Any]:
    order = list(range(len(train_data)))
    random.shuffle(order)
    train_embedding_diversity: List[float] = []
    train_embedding_overlap: List[float] = []
    train_invalid_rate: List[float] = []
    train_vote_correct: List[int] = []
    train_rollout_concurrency = max(1, auto_train_rollout_concurrency(cfg))

    cursor = 0
    while cursor < len(order):
        window = max(1, int(cfg.update_every))
        window_end = min(len(order), ((cursor // window) + 1) * window)
        batch_end = min(window_end, cursor + train_rollout_concurrency)
        batch_positions = list(range(cursor, batch_end))

        async def solve_position(pos: int):
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
            eval_batch: List[Dict[str, str]] = []
            if int(cfg.candidate_eval_batch_size) > 0:
                batch_indices = [idx]
                while len(batch_indices) < min(int(cfg.candidate_eval_batch_size), len(train_data)):
                    batch_indices.append(random.choice(order))
                eval_batch = [train_data[b] for b in batch_indices]
            out = await system.record_train_rollout(
                solved,
                do_update=((step + 1) % int(cfg.update_every) == 0),
                eval_batch=eval_batch,
                step_id=step + 1,
                epoch_id=epoch,
            )
            train_embedding_diversity.append(float(out.get("embedding_diversity", 0.0)))
            train_embedding_overlap.append(float(out.get("mean_embedding_overlap", 0.0)))
            train_invalid_rate.append(float(out.get("invalid_rate", 0.0)))
            train_vote_correct.append(int(out.get("vote_correct", 0)))

            if (step + 1) % 10 == 0 or (step + 1) == len(order):
                print(
                    f"Epoch {epoch} Step {step + 1}/{len(order)} "
                    f"train_embedding_div={float(np.mean(train_embedding_diversity)):.4f} "
                    f"train_embedding_overlap={float(np.mean(train_embedding_overlap)):.4f} "
                    f"train_invalid={float(np.mean(train_invalid_rate)):.4f} "
                    f"train_vote_acc={float(np.mean(train_vote_correct)):.4f}",
                    flush=True,
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
        refresh_summary = await system.refresh_all_prompt_beams(refresh_batch, epoch_id=epoch)

    val_metrics = await system.evaluate_dataset(val_data, split_name=f"val_epoch{epoch}")
    epoch_record: Dict[str, Any] = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
    if refresh_summary is not None:
        epoch_record["beam_refresh"] = refresh_summary
    print(
        f"Epoch {epoch}: "
        f"train_embedding_div={train_metrics['mean_embedding_diversity']:.4f}, "
        f"train_embedding_overlap={train_metrics['mean_embedding_overlap']:.4f}, "
        f"train_invalid={train_metrics['mean_invalid_rate']:.4f}, "
        f"train_vote_acc={train_metrics['vote_acc']:.4f}, "
        f"val_embedding_div={val_metrics['mean_embedding_diversity']:.4f}, "
        f"val_embedding_overlap={val_metrics['mean_embedding_overlap']:.4f}, "
        f"val_invalid={val_metrics['mean_invalid_rate']:.4f}, "
        f"val_vote_acc={val_metrics['vote_acc']:.4f}",
        flush=True,
    )
    return epoch_record


async def main_async():
    parser = argparse.ArgumentParser(description="Resume shared_beam training with validation early stopping.")
    parser.add_argument("--run_dir", type=str, default="runs_mmlu_subject_balanced_default_size_4way/shared_beam_seed42")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max_additional_epochs", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=225)
    parser.add_argument("--llm_call_timeout", type=float, default=240.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    meta = _read_json(run_dir / "run_meta.json", {})
    cfg_dict = dict(meta.get("config", {}))
    cfg_dict["out_dir"] = str(run_dir.resolve())
    cfg_dict["eval_solver_call_concurrency"] = int(args.eval_solver_call_concurrency)
    cfg_dict["llm_call_timeout"] = float(args.llm_call_timeout)
    cfg = Config(**cfg_dict)
    cfg.baseline_only = False
    cfg.eval_test_each_epoch = False
    cfg.transient_retry_forever = bool(int(cfg.transient_retry_forever))
    cfg.beam_refresh_each_epoch = bool(int(cfg.beam_refresh_each_epoch))
    cfg.llm_call_logging = bool(int(cfg.llm_call_logging))

    set_seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    train_data = build_dataset(load_jsonl(cfg.train_path, cfg.train_size))
    val_data = build_dataset(load_jsonl(cfg.val_path, cfg.val_size))
    if not train_data or not val_data:
        raise RuntimeError("resume requires non-empty train and validation data")

    history_raw = _read_json(run_dir / "history.json", [])
    history = _numeric_epoch_rows(history_raw if isinstance(history_raw, list) else [])
    prompt_history_payload = _read_json(run_dir / "prompt_history.json", {})
    last_state_payload = _read_json(run_dir / "last_state.json", {})
    last_epoch = max([int(row.get("epoch", 0)) for row in history] + [0])
    best_score, best_epoch, epochs_since_improvement = _best_validation(history)
    best_score, best_epoch = _load_best_prompts_score(run_dir, best_score, best_epoch)
    print(
        f"Resume from epoch={last_epoch}; best_epoch={best_epoch}; "
        f"best_validation_score={best_score:.6f}; stale_epochs={epochs_since_improvement}",
        flush=True,
    )

    system = TraceBeamSearchSystem(cfg)
    restored = _restore_from_prompt_history(system, prompt_history_payload)
    if not restored:
        restored = _restore_from_state(system, last_state_payload)
    if not restored:
        raise RuntimeError("could not restore prompts/beam from prompt_history.json or last_state.json")
    system.history = list(history)

    best_prompts_path = run_dir / "best_prompts.json"
    max_epoch = last_epoch + max(1, int(args.max_additional_epochs))
    current_epoch = last_epoch
    while current_epoch < max_epoch:
        if epochs_since_improvement >= int(args.patience):
            print(f"Early stop before next epoch: stale_epochs={epochs_since_improvement}", flush=True)
            break
        current_epoch += 1
        epoch_record = await _train_one_epoch(system, cfg, train_data, val_data, current_epoch)
        system.history.append(epoch_record)

        score = validation_score(epoch_record)
        improved = score > best_score + float(args.min_delta)
        if improved:
            best_score = score
            best_epoch = current_epoch
            epochs_since_improvement = 0
            system.save_state("best_state", extra=epoch_record)
            write_selected_prompts(str(best_prompts_path), system, best_epoch, "vote_acc+embedding_div-invalid", best_score)
        else:
            epochs_since_improvement += 1

        system.save_state("last_state", extra=epoch_record)
        system.flush_train_step_logs()
        system.flush_train_trace_history_logs()
        system.flush_test_trace_history_logs()
        system.flush_prompt_history()
        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(system.history, f, ensure_ascii=False, indent=2)
        print(
            f"Validation score={score:.6f}; improved={improved}; "
            f"best_epoch={best_epoch}; stale_epochs={epochs_since_improvement}/{args.patience}",
            flush=True,
        )

    final_summary = {
        "event": "resume_early_stop_complete",
        "last_epoch": current_epoch,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "stale_epochs": epochs_since_improvement,
        "patience": int(args.patience),
    }
    with (run_dir / "resume_early_stop_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False), flush=True)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
