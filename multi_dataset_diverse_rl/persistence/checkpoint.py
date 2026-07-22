from __future__ import annotations

import base64
import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..evaluation.fixed_probe import PromptAnswer
from ..responsibility import ResponsibilityState
from ..system import METHOD_VERSION


CHECKPOINT_VERSION = 1


def _random_state_payload() -> str:
    return base64.b64encode(pickle.dumps(random.getstate())).decode("ascii")


def build_checkpoint(system, *, epoch_index: int, update_index: int, best_state: Mapping[str, Any]) -> dict[str, Any]:
    if system.fixed_probe is None:
        raise RuntimeError("cannot checkpoint before fixed probe initialization")
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "method_version": METHOD_VERSION,
        "probe_version": system.fixed_probe.version,
        "probe_hash": system.fixed_probe.probe_hash,
        "epoch_index": int(epoch_index),
        "update_index": int(update_index),
        "best_state": dict(best_state),
        "prompts": [agent.current_prompt for agent in system.agents],
        "prompt_memory": [agent.prompt_memory for agent in system.agents],
        "active_profiles": [[asdict(row) for row in profile] for profile in system.active_profiles],
        "initial_profiles": [[asdict(row) for row in profile] for profile in system.initial_profiles],
        "responsibility_state": asdict(system.responsibility_state),
        "cached_responsibility_owners": dict(system.cached_responsibility_owners),
        "cached_responsibility_assignments": {
            str(agent_id): [asdict(row) for row in rows]
            for agent_id, rows in system.cached_responsibility_assignments.items()
        },
        "history": list(system.history),
        "peer_state_history": list(system.peer_state_history),
        "responsibility_assignments": list(system.responsibility_assignments),
        "candidate_decisions": list(system.candidate_decisions),
        "prompt_memory_history": list(system.prompt_memory_history),
        "fixed_probe": system.fixed_probe.to_dict(),
        "random_state": _random_state_payload(),
    }


def validate_checkpoint(payload: Mapping[str, Any], system) -> None:
    if (
        int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION
        or str(payload.get("method_version", "")) != METHOD_VERSION
    ):
        raise ValueError("Legacy checkpoint is incompatible with peer_state_counterfactual_v1. Start a new run.")
    if system.fixed_probe is None:
        raise RuntimeError("fixed probe must be initialized before checkpoint restore")
    if (
        str(payload.get("probe_version", "")) != system.fixed_probe.version
        or str(payload.get("probe_hash", "")) != system.fixed_probe.probe_hash
    ):
        raise ValueError("Fixed probe cache version or hash mismatch. Start a new run.")


def restore_checkpoint(system, payload: Mapping[str, Any]) -> tuple[int, int, dict[str, Any]]:
    validate_checkpoint(payload, system)
    for agent, prompt, memory in zip(
        system.agents, payload.get("prompts", []), payload.get("prompt_memory", []), strict=True,
    ):
        agent.current_prompt = str(prompt)
        agent.prompt_memory = [dict(row) for row in memory]
    system.active_profiles = [tuple(PromptAnswer(**row) for row in profile) for profile in payload.get("active_profiles", [])]
    system.initial_profiles = [tuple(PromptAnswer(**row) for row in profile) for profile in payload.get("initial_profiles", [])]
    raw_state = dict(payload.get("responsibility_state", {}))
    raw_state["agent_updates_since_last_selected"] = {
        int(key): int(value) for key, value in raw_state.get("agent_updates_since_last_selected", {}).items()
    }
    raw_state["assigned_load_per_agent"] = {
        int(key): int(value) for key, value in raw_state.get("assigned_load_per_agent", {}).items()
    }
    system.responsibility_state = ResponsibilityState(**raw_state)
    from ..responsibility import AgentExampleCredit
    system.cached_responsibility_owners = {
        str(key): int(value) for key, value in dict(payload.get("cached_responsibility_owners", {})).items()
    }
    system.cached_responsibility_assignments = {
        int(agent_id): [AgentExampleCredit(**row) for row in rows]
        for agent_id, rows in dict(payload.get("cached_responsibility_assignments", {})).items()
    }
    for name in (
        "history", "peer_state_history", "responsibility_assignments",
        "candidate_decisions", "prompt_memory_history",
    ):
        setattr(system, name, list(payload.get(name, [])))
    system.fixed_probe.restore(payload.get("fixed_probe", {}))
    random.setstate(pickle.loads(base64.b64decode(str(payload["random_state"]))))
    return int(payload.get("epoch_index", 0)), int(payload.get("update_index", 0)), dict(payload.get("best_state", {}))


def load_checkpoint(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
