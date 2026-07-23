from __future__ import annotations

import base64
import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..evaluation.fixed_probe import PromptAnswer
from ..persistence.identity import validate_run_identity
from ..responsibility import MemberAwareRepairOpportunity, ResponsibilityState
from ..system import METHOD_VERSION


CHECKPOINT_VERSION = 5


def _random_state_payload() -> str:
    return base64.b64encode(pickle.dumps(random.getstate())).decode("ascii")


def build_checkpoint(
    system,
    *,
    epoch_index: int,
    update_index: int,
    best_state: Mapping[str, Any],
) -> dict[str, Any]:
    if system.fixed_probe is None:
        raise RuntimeError("cannot checkpoint before fixed probe initialization")
    if system.validation_probe is None:
        raise RuntimeError("cannot checkpoint before validation probe initialization")
    if system.run_identity is None:
        raise RuntimeError("cannot checkpoint without run identity")
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "method_version": METHOD_VERSION,
        "run_identity": system.run_identity.to_dict(),
        "probe_version": system.fixed_probe.version,
        "probe_hash": system.fixed_probe.probe_hash,
        "epoch_index": int(epoch_index),
        "update_index": int(update_index),
        "best_state": dict(best_state),
        "prompts": [agent.current_prompt for agent in system.agents],
        "previous_active_prompts": [agent.previous_active_prompt for agent in system.agents],
        "active_profiles": [[asdict(row) for row in profile] for profile in system.active_profiles],
        "initial_profiles": [[asdict(row) for row in profile] for profile in system.initial_profiles],
        "responsibility_state": asdict(system.responsibility_state),
        "cached_responsibility_owners": dict(system.cached_responsibility_owners),
        "cached_responsibility_assignments": {
            str(agent_id): [asdict(row) for row in rows]
            for agent_id, rows in system.cached_responsibility_assignments.items()
        },
        "cached_member_opportunities": {
            question_hash: [asdict(row) for row in rows]
            for question_hash, rows in system.cached_member_opportunities.items()
        },
        "previous_accuracy_summaries": dict(system.previous_accuracy_summaries),
        "previous_peer_summaries": dict(system.previous_peer_summaries),
        "previous_responsibility_summaries": dict(system.previous_responsibility_summaries),
        "agent_selection_counts": dict(system.agent_selection_counts),
        "target_priority_audit": list(system.target_priority_audit),
        "history": list(system.history),
        "peer_state_history": list(system.peer_state_history),
        "responsibility_assignments": list(system.responsibility_assignments),
        "candidate_decisions": list(system.candidate_decisions),
        "tcs_context_history": list(system.tcs_context_history),
        "tcs_rounds": list(system.tcs_rounds),
        "solver_invalid_outputs": list(system.solver_invalid_outputs),
        "llm_calls": list(system.llm.calls),
        "fixed_probe": system.fixed_probe.to_dict(),
        "validation_probe": system.validation_probe.to_dict(),
        "shared_solver_cache_audit": {
            "path": str(system.cfg.persistence.shared_solver_cache_path or ""),
            "ready_entries": (
                system.shared_solver_cache.ready_entry_count()
                if system.shared_solver_cache is not None
                else len(system.prompt_question_evaluator.cache)
            ),
            "content_hash": (
                system.shared_solver_cache.ready_content_hash()
                if system.shared_solver_cache is not None
                else ""
            ),
        },
        "random_state": _random_state_payload(),
    }


def validate_checkpoint(payload: Mapping[str, Any], system) -> None:
    if "checkpoint_version" not in payload or "method_version" not in payload or "run_identity" not in payload:
        raise ValueError("Legacy checkpoint lacks exact run identity and cannot be resumed")
    if int(payload["checkpoint_version"]) != CHECKPOINT_VERSION or str(payload["method_version"]) != METHOD_VERSION:
        raise ValueError("Checkpoint is incompatible with member_aware_peer_state_v1")
    if system.run_identity is None:
        raise RuntimeError("run identity must be set before checkpoint validation")
    validate_run_identity(system.run_identity, payload["run_identity"])
    if system.fixed_probe is None or system.validation_probe is None:
        raise RuntimeError("fixed and validation probes must exist before checkpoint restore")
    if str(payload["probe_version"]) != system.fixed_probe.version or str(payload["probe_hash"]) != system.fixed_probe.probe_hash:
        raise ValueError("Fixed probe cache version or hash mismatch. Start a new run.")


def restore_checkpoint(system, payload: Mapping[str, Any]) -> tuple[int, int, dict[str, Any]]:
    validate_checkpoint(payload, system)
    prompts = payload["prompts"]
    previous_prompts = payload["previous_active_prompts"]
    if len(prompts) != 5 or len(previous_prompts) != 5:
        raise ValueError("checkpoint must contain exactly five agent prompts")
    for agent, prompt, previous in zip(system.agents, prompts, previous_prompts, strict=True):
        agent.current_prompt = str(prompt)
        agent.previous_active_prompt = None if previous is None else str(previous)
    system.active_profiles = [
        tuple(PromptAnswer(**row) for row in profile) for profile in payload["active_profiles"]
    ]
    system.initial_profiles = [
        tuple(PromptAnswer(**row) for row in profile) for profile in payload["initial_profiles"]
    ]
    raw_state = dict(payload["responsibility_state"])
    for field in (
        "assigned_load_by_agent",
        "updates_since_selected_by_agent",
        "accepted_updates_by_agent",
    ):
        raw_state[field] = {int(key): int(value) for key, value in raw_state[field].items()}
    raw_state["seeded_rank_by_agent"] = {
        int(key): str(value) for key, value in raw_state["seeded_rank_by_agent"].items()
    }
    raw_state["primary_owner_by_question"] = {
        str(key): int(value) for key, value in raw_state["primary_owner_by_question"].items()
    }
    raw_state["owner_age_by_question"] = {
        str(key): int(value) for key, value in raw_state["owner_age_by_question"].items()
    }
    system.responsibility_state = ResponsibilityState(**raw_state)
    system.cached_responsibility_owners = {
        str(key): int(value) for key, value in payload["cached_responsibility_owners"].items()
    }
    system.cached_responsibility_assignments = {
        int(agent_id): [MemberAwareRepairOpportunity(**row) for row in rows]
        for agent_id, rows in payload["cached_responsibility_assignments"].items()
    }
    system.cached_member_opportunities = {
        str(question_hash): tuple(
            MemberAwareRepairOpportunity(**row) for row in rows
        )
        for question_hash, rows in payload["cached_member_opportunities"].items()
    }
    for field in (
        "previous_accuracy_summaries",
        "previous_peer_summaries",
        "previous_responsibility_summaries",
    ):
        setattr(system, field, {int(key): str(value) for key, value in payload[field].items()})
    system.agent_selection_counts = {
        int(key): int(value) for key, value in payload["agent_selection_counts"].items()
    }
    system.target_priority_audit = list(payload["target_priority_audit"])
    for name in (
        "history",
        "peer_state_history",
        "responsibility_assignments",
        "candidate_decisions",
        "tcs_context_history",
        "tcs_rounds",
        "solver_invalid_outputs",
    ):
        setattr(system, name, list(payload[name]))
    system._audited_invalid_keys = {
        (str(row["prompt_hash"]), str(row["question_hash"]))
        for row in system.solver_invalid_outputs
    }
    system.llm.calls = list(payload["llm_calls"])
    system.fixed_probe.restore(payload["fixed_probe"])
    system.validation_probe.restore(payload["validation_probe"])
    random.setstate(pickle.loads(base64.b64decode(str(payload["random_state"]))))
    return int(payload["epoch_index"]), int(payload["update_index"]), dict(payload["best_state"])


def load_checkpoint(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
