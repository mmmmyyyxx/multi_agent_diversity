from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Config, add_config_arguments, config_from_args
from .persistence.checkpoint import build_checkpoint, load_checkpoint, restore_checkpoint
from .system import PromptEnsembleOptimizationSystem
from .utils import load_jsonl


LEGACY_QUESTION_KEYS = ("question", "input", "query", "problem")
LEGACY_ANSWER_KEYS = ("answer", "output", "target", "label", "response")
MARS_QUESTION_KEYS = (*LEGACY_QUESTION_KEYS, "prompt")
MARS_ANSWER_KEYS = ("answer", "target", "gold", "gold_answer", "label", "output")


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    return next((record[key] for key in keys if key in record and record[key] is not None), None)


def build_dataset(raw_records, dataset_format="legacy") -> list[dict[str, Any]]:
    mars = str(dataset_format).lower() == "mars"
    q_keys = MARS_QUESTION_KEYS if mars else LEGACY_QUESTION_KEYS
    a_keys = MARS_ANSWER_KEYS if mars else LEGACY_ANSWER_KEYS
    rows = []
    for index, record in enumerate(raw_records):
        question = _first_present(record, q_keys)
        answer = _first_present(record, a_keys)
        if question is None or answer is None:
            raise ValueError(f"Cannot find question/answer fields in record {index}")
        row = {"question": str(question), "answer": answer}
        task = _first_present(record, ("task", "task_name", "category", "subject", "bbh_task"))
        if task is not None:
            row["task"] = str(task)
        for key in ("task_name", "category", "subject", "bbh_task"):
            if record.get(key) is not None:
                row[key] = str(record[key])
        rows.append(row)
    return rows


def _load(path: str, limit: int, fmt: str) -> list[dict[str, Any]]:
    return build_dataset(load_jsonl(path, limit), fmt)


def _write_checkpoint(system, cfg: Config, epoch_index: int, update_index: int, best_state: Mapping[str, Any]) -> None:
    system.artifacts.write_json(
        "training_checkpoint.json",
        build_checkpoint(system, epoch_index=epoch_index, update_index=update_index, best_state=best_state),
    )


async def run(cfg: Config) -> dict[str, Any]:
    random.seed(cfg.training.seed)
    train = [] if cfg.training.baseline_only else _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = [] if cfg.training.baseline_only else _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    test = _load(cfg.data.test_path, cfg.data.test_size, cfg.data.dataset_format)
    system = PromptEnsembleOptimizationSystem(cfg)

    if cfg.training.baseline_only:
        final_metrics = await system.evaluate_dataset(test)
        system.history = [{"epoch": 0, "test": {key: value for key, value in final_metrics.items() if key != "rows"}}]
        system.flush_artifacts()
        system.artifacts.write_json("final_summary.json", final_metrics)
        return final_metrics

    probe = list(train[: min(len(train), cfg.evaluation.candidate_eval_pool_size)])
    checkpoint_path = Path(cfg.persistence.out_dir) / "training_checkpoint.json"
    payload = load_checkpoint(checkpoint_path) if cfg.persistence.resume_from_checkpoint else None
    if payload is None:
        await system.initialize_fixed_probe(probe)
    else:
        system.fixed_probe = system.build_probe(probe)
    initial_validation = None
    best_state: dict[str, Any] = {}
    start_epoch = start_update = 0
    if payload is not None:
        start_epoch, start_update, best_state = restore_checkpoint(system, payload)
        initial_validation = dict(best_state.get("initial_validation", best_state.get("metrics", {})))
    else:
        initial_validation = await system.evaluate_dataset(validation)
        best_state = {
            "key": system.validation_key(initial_validation, initial_validation, 0),
            "epoch": 0,
            "prompts": [agent.current_prompt for agent in system.agents],
            "metrics": {key: value for key, value in initial_validation.items() if key != "rows"},
            "initial_validation": initial_validation,
        }

    updates_per_epoch = max(1, math.ceil(len(train) / max(1, cfg.training.update_every)))
    for epoch in range(start_epoch, cfg.training.epochs):
        first_update = start_update if epoch == start_epoch else 0
        for update in range(first_update, updates_per_epoch):
            await system.update_once(epoch * updates_per_epoch + update)
            _write_checkpoint(system, cfg, epoch, update + 1, best_state)
        validation_metrics = await system.evaluate_dataset(validation)
        key = system.validation_key(validation_metrics, initial_validation, epoch + 1)
        system.history.append({
            "epoch": epoch + 1,
            "validation": {name: value for name, value in validation_metrics.items() if name != "rows"},
            "validation_feasible": key is not None,
        })
        if key is not None and (best_state.get("key") is None or tuple(key) > tuple(best_state["key"])):
            best_state = {
                "key": key, "epoch": epoch + 1,
                "prompts": [agent.current_prompt for agent in system.agents],
                "metrics": {name: value for name, value in validation_metrics.items() if name != "rows"},
                "initial_validation": initial_validation,
            }
        _write_checkpoint(system, cfg, epoch + 1, 0, best_state)
        start_update = 0

    for agent, prompt in zip(system.agents, best_state["prompts"], strict=True):
        agent.current_prompt = str(prompt)
    final_metrics = await system.evaluate_dataset(test)
    system.artifacts.write_json("best_prompts.json", best_state["prompts"])
    system.artifacts.write_json("final_summary.json", final_metrics)
    system.flush_artifacts()
    checkpoint_path.unlink(missing_ok=True)
    return final_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Peer-State Counterfactual Prompt Optimization")
    return add_config_arguments(parser)


async def main_async() -> None:
    cfg = config_from_args(build_parser().parse_args())
    result = await run(cfg)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
