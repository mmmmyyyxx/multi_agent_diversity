from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Config, add_config_arguments, config_from_args
from .evaluation.validation import DatasetMetrics
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


def _write_checkpoint(
    system,
    cfg: Config,
    epoch_index: int,
    update_index: int,
    training_state: Mapping[str, Any],
) -> None:
    system.artifacts.write_json(
        "training_checkpoint.json",
        build_checkpoint(
            system,
            epoch_index=epoch_index,
            update_index=update_index,
            training_state=training_state,
        ),
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
    # Validation remains part of exact dataset identity but is never evaluated.
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
        system.planned_update_count = 0
        system.completed_update_count = 0
        system.mark_training_complete(0)
        selection_summary = {
            "validation_used": False,
            "validation_selection_policy": "none",
            "validation_unique_state_count": 0,
            "validation_evaluation_count": 0,
            "validation_reused_state_count": 0,
            "selected_checkpoint_source": "final_active_state",
            "selected_by_validation": False,
            "selected_checkpoint_update_index": 0,
            "selected_team_prompt_state_hash": team_state_hash,
            "selected_epoch": 0,
            "selection_changed": False,
            "checkpoint_selection": "none",
            "test_evaluation_count": 0,
            "test_used_for_selection": False,
            "test_used_for_training": False,
            "test_called_before_training_complete": False,
        }
        system.final_state_selection = dict(selection_summary)
        initial_test = await system.evaluate_final_test(test)
        selection_summary.update({
            "test_evaluation_count": system.test_evaluation_count,
            "test_used_for_selection": system.test_used_for_selection,
            "test_used_for_training": system.test_used_for_training,
            "test_called_before_training_complete": (
                system.test_called_before_training_complete
            ),
        })
        system.final_state_selection = dict(selection_summary)
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
    updates_per_epoch = max(
        1,
        math.ceil(len(train) / max(1, cfg.training.update_every)),
    )
    planned_update_count = cfg.training.epochs * updates_per_epoch
    system.planned_update_count = planned_update_count
    if payload is None:
        await system.initialize_fixed_probe(probe)
    else:
        system.fixed_probe = system.build_probe(probe)
    training_state: dict[str, Any]
    start_epoch = start_update = 0
    if payload is not None:
        start_epoch, start_update, training_state = restore_checkpoint(system, payload)
        if system.planned_update_count != planned_update_count:
            raise ValueError("checkpoint planned update count mismatch")
    else:
        initial_state_hash = system.team_prompt_state_hash()
        training_state = {
            "planned_update_count": planned_update_count,
            "initial_team_state_hash": initial_state_hash,
        }
        system.record_training_dynamics(update_index=-1)
        _verify_frozen_initialization(system, cfg)
    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_decision_start = len(system.candidate_decisions)
        first_update = start_update if epoch == start_epoch else 0
        for update in range(first_update, updates_per_epoch):
            global_update_index = epoch * updates_per_epoch + update
            incumbent_profiles = [tuple(profile) for profile in system.active_profiles]
            accepted = await system.update_once(global_update_index)
            system.completed_update_count = global_update_index + 1
            system.record_training_dynamics(
                update_index=global_update_index,
                incumbent_profiles=incumbent_profiles,
            )
            train_step = min((update + 1) * cfg.training.update_every, len(train))
            _print_progress(
                epoch=f"{epoch + 1}/{cfg.training.epochs}",
                step=f"{train_step}/{len(train)}",
                metrics=system.active_probe_metrics(),
            )
            _write_checkpoint(system, cfg, epoch, update + 1, training_state)
        system.history.append({
            "epoch": epoch + 1,
            "completed_update_count": system.completed_update_count,
            "team_prompt_state_hash": system.team_prompt_state_hash(),
            "active_probe": _metrics_summary(system.active_probe_metrics()),
            "candidate_funnel": system.candidate_funnel_summary(
                system.candidate_decisions[epoch_decision_start:]
            ),
        })
        _write_checkpoint(system, cfg, epoch + 1, 0, training_state)
        start_update = 0

    if system.completed_update_count != planned_update_count:
        raise RuntimeError("not every planned update completed")
    system.mark_training_complete(planned_update_count)
    selected_prompts = [agent.current_prompt for agent in system.agents]
    selected_hash = system.team_prompt_state_hash()
    selection_summary = {
        "validation_used": False,
        "validation_selection_policy": "none",
        "validation_unique_state_count": 0,
        "validation_evaluation_count": 0,
        "validation_reused_state_count": 0,
        "selected_checkpoint_source": "final_active_state",
        "selected_by_validation": False,
        "selected_checkpoint_update_index": planned_update_count,
        "selected_team_prompt_state_hash": selected_hash,
        "selected_epoch": planned_update_count,
        "selection_changed": selected_hash != training_state[
            "initial_team_state_hash"
        ],
        "checkpoint_selection": "none",
        "test_evaluation_count": 0,
        "test_used_for_selection": False,
        "test_used_for_training": False,
        "test_called_before_training_complete": False,
    }
    system.final_state_selection = dict(selection_summary)
    _write_checkpoint(system, cfg, cfg.training.epochs, 0, training_state)
    selected_test = None
    if cfg.persistence.final_test_enabled:
        selected_test = await system.evaluate_final_test(test)
    selection_summary.update({
        "test_evaluation_count": system.test_evaluation_count,
        "test_used_for_selection": system.test_used_for_selection,
        "test_used_for_training": system.test_used_for_training,
        "test_called_before_training_complete": (
            system.test_called_before_training_complete
        ),
        "final_test_enabled": cfg.persistence.final_test_enabled,
    })
    system.final_state_selection = dict(selection_summary)
    _write_checkpoint(system, cfg, cfg.training.epochs, 0, training_state)
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
    selected: DatasetMetrics | None,
    *,
    selection_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "initial_test": initial.to_dict() if initial is not None else None,
        "selected_test": selected.to_dict() if selected is not None else None,
        "member_gain": (
            _member_gain_summary(initial, selected)
            if initial is not None and selected is not None else None
        ),
        "selection_summary": dict(selection_summary),
    }


def _verify_frozen_initialization(
    system: PromptEnsembleOptimizationSystem,
    cfg: Config,
) -> None:
    """Fail closed before update zero when a matched-pilot state diverges."""
    manifest_path = str(cfg.persistence.frozen_initialization_manifest_path).strip()
    if not manifest_path:
        return
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        expected = manifest["initialization_snapshot"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("frozen initialization manifest is unreadable") from exc
    actual = system.frozen_initialization_snapshot()
    fields = (
        "initial_prompt_hashes", "initial_member_correct_counts", "initial_team_outcome",
        "initial_vote_oracle_ghm_hash", "initial_train_state_hash", "probe_hash",
        "solver_request_identity", "solver_identity", "immutable_run_identity",
    )
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in fields
        if json.loads(json.dumps(expected.get(field), sort_keys=True))
        != json.loads(json.dumps(actual.get(field), sort_keys=True))
    }
    audit = {
        "frozen_initialization_manifest_version": manifest.get("manifest_version", ""),
        "matched": not mismatches,
        "checked_fields": list(fields),
        "mismatches": mismatches,
        "initialization_snapshot": actual,
    }
    system.artifacts.write_json("frozen_initialization_match.json", audit)
    if mismatches:
        raise RuntimeError(
            "frozen initialization mismatch before update zero: "
            + json.dumps(sorted(mismatches), separators=(",", ":"))
        )


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
    if selected is None:
        print("final test skipped by final_test_enabled=0", flush=True)
        return
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
