from __future__ import annotations

import argparse
import asyncio
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Config, add_config_arguments, config_from_args
from .evaluation.validation import DatasetMetrics
from .evaluation.validation import dataset_metrics_from_dict
from .persistence.checkpoint import build_checkpoint, load_checkpoint, restore_checkpoint
from .persistence.identity import build_run_identity
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


def _progress_line(
    *,
    epoch: int | str,
    step: int | str,
    vote_acc: float,
    individual_acc: float,
) -> str:
    return (
        f"epoch={epoch} step={step} "
        f"vote_acc={vote_acc:.4f} individual_acc={individual_acc:.4f}"
    )


def _print_progress(*, epoch: int | str, step: int | str, metrics: DatasetMetrics) -> None:
    print(
        _progress_line(
            epoch=epoch,
            step=step,
            vote_acc=metrics.plurality_vote_acc,
            individual_acc=metrics.mean_individual_acc,
        ),
        flush=True,
    )


async def run(cfg: Config) -> dict[str, Any]:
    random.seed(cfg.training.seed)
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    test = _load(cfg.data.test_path, cfg.data.test_size, cfg.data.dataset_format)
    system = PromptEnsembleOptimizationSystem(cfg)
    system.set_run_identity(build_run_identity(
        cfg,
        train_rows=train,
        val_rows=validation,
        test_rows=test,
        workspace=Path.cwd(),
    ))

    if not system.protocol.optimization_enabled:
        team_state_hash = system.team_prompt_state_hash()
        selection_summary = {
            "validation_used": False,
            "validation_selection_policy": "baseline_no_selection",
            "validation_unique_state_count": 0,
            "validation_evaluation_count": 0,
            "validation_reused_state_count": 0,
            "selected_checkpoint_source": "initial_baseline",
            "selected_by_validation": False,
            "selected_checkpoint_update_index": 0,
            "selected_team_prompt_state_hash": team_state_hash,
            "selected_epoch": 0,
            "selection_changed": False,
            "validation_key": None,
            "test_evaluation_count": 0,
            "test_used_for_selection": False,
            "test_called_before_selection": False,
        }
        system.complete_validation_selection(selection_summary)
        initial_test = await system.evaluate_selected_test(test)
        selection_summary.update({
            "test_evaluation_count": system.test_evaluation_count,
            "test_used_for_selection": system.test_used_for_selection,
            "test_called_before_selection": system.test_called_before_selection,
        })
        final_payload = _final_payload(
            initial_test,
            initial_test,
            selection_summary=selection_summary,
        )
        system.history = [{"epoch": 0, "test": _metrics_summary(initial_test)}]
        system.artifacts.write_json(
            "best_prompts.json", [agent.current_prompt for agent in system.agents]
        )
        system.flush_artifacts()
        system.artifacts.write_json("final_summary.json", final_payload)
        return final_payload

    probe = list(train[: min(len(train), cfg.evaluation.candidate_eval_pool_size)])
    checkpoint_path = Path(cfg.persistence.out_dir) / "training_checkpoint.json"
    payload = load_checkpoint(checkpoint_path) if cfg.persistence.resume_from_checkpoint else None
    system.validation_probe = system.build_validation_probe(validation)
    if payload is None:
        await system.initialize_fixed_probe(probe)
    else:
        system.fixed_probe = system.build_probe(probe)
    initial_validation: DatasetMetrics
    best_state: dict[str, Any] = {}
    start_epoch = start_update = 0
    if payload is not None:
        start_epoch, start_update, best_state = restore_checkpoint(system, payload)
        initial_validation = dataset_metrics_from_dict(
            best_state["initial_validation"]
        )
        current_validation, current_validation_audit = (
            await system.evaluate_validation_state(validation)
        )
    else:
        initial_validation, current_validation_audit = (
            await system.evaluate_validation_state(validation)
        )
        current_validation = initial_validation
        initial_state_hash = system.team_prompt_state_hash()
        best_state = {
            "key": system.validation_key(initial_validation, initial_validation, 0),
            "epoch": 0,
            "update_index": 0,
            "team_prompt_state_hash": initial_state_hash,
            "prompts": [agent.current_prompt for agent in system.agents],
            "metrics": _metrics_summary(initial_validation),
            "initial_validation": initial_validation.to_dict(),
            "validation_audit": dict(current_validation_audit),
        }

    updates_per_epoch = max(1, math.ceil(len(train) / max(1, cfg.training.update_every)))
    key = system.validation_key(
        current_validation,
        initial_validation,
        int(best_state.get("update_index", 0)),
    )
    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_decision_start = len(system.candidate_decisions)
        first_update = start_update if epoch == start_epoch else 0
        for update in range(first_update, updates_per_epoch):
            global_update_index = epoch * updates_per_epoch + update
            accepted = await system.update_once(global_update_index)
            current_validation, current_validation_audit = (
                await system.evaluate_validation_state(validation)
            )
            key = system.validation_key(
                current_validation,
                initial_validation,
                global_update_index + 1,
            )
            if accepted and key is not None and (
                best_state.get("key") is None
                or tuple(key) > tuple(best_state["key"])
            ):
                best_state = {
                    "key": key,
                    "epoch": epoch + 1,
                    "update_index": global_update_index + 1,
                    "team_prompt_state_hash": (
                        current_validation_audit["team_prompt_state_hash"]
                    ),
                    "prompts": [
                        agent.current_prompt for agent in system.agents
                    ],
                    "metrics": _metrics_summary(current_validation),
                    "initial_validation": initial_validation.to_dict(),
                    "validation_audit": dict(current_validation_audit),
                }
            train_step = min((update + 1) * cfg.training.update_every, len(train))
            _print_progress(
                epoch=f"{epoch + 1}/{cfg.training.epochs}",
                step=f"{train_step}/{len(train)}",
                metrics=system.active_probe_metrics(),
            )
            _write_checkpoint(system, cfg, epoch, update + 1, best_state)
        system.history.append({
            "epoch": epoch + 1,
            "team_prompt_state_hash": current_validation_audit[
                "team_prompt_state_hash"
            ],
            "validation_cache_hit": current_validation_audit[
                "validation_cache_hit"
            ],
            "validation_result_source": current_validation_audit[
                "validation_result_source"
            ],
            "validation": _metrics_summary(current_validation),
            "validation_feasible": key is not None,
            "member_objective": _member_gain_summary(
                initial_validation, current_validation
            ),
            "candidate_funnel": system.candidate_funnel_summary(
                system.candidate_decisions[epoch_decision_start:]
            ),
        })
        _write_checkpoint(system, cfg, epoch + 1, 0, best_state)
        start_update = 0

    selected_prompts = [str(prompt) for prompt in best_state["prompts"]]
    for agent, prompt in zip(system.agents, selected_prompts, strict=True):
        agent.current_prompt = str(prompt)
    selection_summary = {
        "validation_used": True,
        "validation_selection_policy": (
            "validation_checkpoint_selection_v2"
        ),
        "validation_unique_state_count": len(
            system.validation_state_cache
        ),
        "validation_evaluation_count": system.validation_evaluation_count,
        "validation_reused_state_count": system.validation_reuse_count,
        "selected_checkpoint_source": "validation",
        "selected_by_validation": True,
        "selected_checkpoint_update_index": int(
            best_state.get("update_index", 0)
        ),
        "selected_team_prompt_state_hash": str(
            best_state["team_prompt_state_hash"]
        ),
        "selected_epoch": int(best_state["epoch"]),
        "selection_changed": selected_prompts != [
            agent.initial_prompt for agent in system.agents
        ],
        "validation_key": best_state["key"],
        "test_evaluation_count": 0,
        "test_used_for_selection": False,
        "test_called_before_selection": False,
    }
    system.complete_validation_selection(selection_summary)
    _write_checkpoint(system, cfg, cfg.training.epochs, 0, best_state)
    selected_test = await system.evaluate_selected_test(test)
    selection_summary.update({
        "test_evaluation_count": system.test_evaluation_count,
        "test_used_for_selection": system.test_used_for_selection,
        "test_called_before_selection": system.test_called_before_selection,
    })
    _write_checkpoint(system, cfg, cfg.training.epochs, 0, best_state)
    final_payload = _final_payload(
        None,
        selected_test,
        selection_summary=selection_summary,
    )
    system.artifacts.write_json("best_prompts.json", selected_prompts)
    system.artifacts.write_json("final_summary.json", final_payload)
    system.flush_artifacts()
    checkpoint_path.unlink(missing_ok=True)
    return final_payload


def _member_gain_summary(
    initial: DatasetMetrics,
    selected: DatasetMetrics,
) -> dict[str, Any]:
    gains = tuple(
        current - baseline
        for current, baseline in zip(
            selected.per_agent_correct_counts,
            initial.per_agent_correct_counts,
            strict=True,
        )
    )
    accuracy_gains = tuple(
        current - baseline
        for current, baseline in zip(
            selected.per_agent_acc,
            initial.per_agent_acc,
            strict=True,
        )
    )
    return {
        "gain_counts": gains,
        "minimum_gain_count": min(gains),
        "total_gain_count": sum(gains),
        "mean_gain": sum(gains) / len(gains),
        "minimum_member_correct_count_gain": min(gains),
        "mean_member_correct_count_gain": sum(gains) / len(gains),
        "minimum_member_accuracy_gain": min(accuracy_gains),
        "mean_member_accuracy_gain": sum(accuracy_gains) / len(accuracy_gains),
        "improved_agent_count": sum(value > 0 for value in gains),
        "regressed_agent_count": sum(value < 0 for value in gains),
        "all_members_non_regressed": all(value >= 0 for value in gains),
        "all_members_improved": all(value > 0 for value in gains),
    }


def _final_payload(
    initial: DatasetMetrics | None,
    selected: DatasetMetrics,
    *,
    selection_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "initial_test": initial.to_dict() if initial is not None else None,
        "selected_test": selected.to_dict(),
        "member_gain": (
            _member_gain_summary(initial, selected)
            if initial is not None else None
        ),
        "selection_summary": dict(selection_summary),
    }


def _metrics_summary(metrics: DatasetMetrics) -> dict[str, Any]:
    payload = metrics.to_dict()
    payload.pop("rows")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Member-Aware Peer-State Prompt-Team Optimization"
    )
    return add_config_arguments(parser)


async def main_async() -> None:
    cfg = config_from_args(build_parser().parse_args())
    result = await run(cfg)
    selected = result["selected_test"]
    print(_progress_line(
        epoch="final",
        step="final",
        vote_acc=float(selected["plurality_vote_acc"]),
        individual_acc=float(selected["mean_individual_acc"]),
    ), flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
