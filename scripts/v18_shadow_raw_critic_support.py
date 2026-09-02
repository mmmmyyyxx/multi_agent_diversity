from __future__ import annotations

import hashlib
import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.tcs import STUDENT_SYSTEM_PROMPT, extract_json_obj, teacher_repair_plan_hash
from scripts.v18_safety_only_critic_pilot_support import canonical_hash, _plan_from_critic_prompt


ROOT = Path(__file__).resolve().parents[1]
AUTH_ENV = "V18_SHADOW_RAW_CRITIC_PILOT_AUTHORIZED"
ARMS = ("canonical_control", "shadow_raw")
LABELS = (
    "CRITIC_OVER_FILTERING_CAUSALLY_SUPPORTED",
    "CRITIC_FILTERING_JUSTIFIED",
    "MIXED_SHADOW_RAW_SIGNAL",
    "NO_CLEAR_SIGNAL",
)


@dataclass
class SharedCanonicalReplay:
    results: dict[str, LLMCallResult] = field(default_factory=dict)
    replay_hits: int = 0
    new_calls: int = 0
    shadow_events: list[dict[str, Any]] = field(default_factory=list)
    valid_rejected_plan_hashes: set[str] = field(default_factory=set)

    @staticmethod
    def key(model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int | None, client_role: str, role: str) -> str:
        return canonical_hash([model, system_prompt, user_prompt, temperature, max_tokens, client_role, role])


def _role(system_prompt: str, client_role: str, logical_role: str | None) -> str:
    if logical_role:
        return logical_role
    if client_role == "solver":
        return "solver"
    if client_role == "evaluator":
        return "critic"
    if system_prompt in {STUDENT_SYSTEM_PROMPT, "Return strict JSON only."}:
        return "student"
    return "teacher"


def shadow_transform(result: LLMCallResult, critic_prompt: str, valid_rejected_plan_hashes: set[str]) -> tuple[LLMCallResult, dict[str, Any] | None]:
    payload = extract_json_obj(result.text)
    if not isinstance(payload, dict) or set(payload) != {"failed_checks", "risk_case_ids", "feedback"}:
        return result, None
    failed = payload.get("failed_checks")
    risks = payload.get("risk_case_ids")
    feedback = payload.get("feedback")
    if not isinstance(failed, list) or not isinstance(risks, list) or not isinstance(feedback, str) or not failed:
        return result, None
    plan = _plan_from_critic_prompt(critic_prompt)
    plan_hash = teacher_repair_plan_hash(plan)
    if plan_hash not in valid_rejected_plan_hashes:
        return result, None
    event = {
        "teacher_plan_hash": plan_hash,
        "original_critic_response_hash": hashlib.sha256(result.text.encode()).hexdigest(),
        "original_failed_checks": sorted(map(str, failed)),
        "critic_feedback_hash": hashlib.sha256(feedback.encode()).hexdigest(),
        "original_rejected": True,
        "shadow_effective_approved": True,
    }
    transformed = LLMCallResult(
        text=json.dumps({"failed_checks": [], "risk_case_ids": [], "feedback": ""}, separators=(",", ":")),
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        latency_seconds=0.0,
        finish_reason="stop",
    )
    return transformed, event


def install_mode(system: Any, mode: str, replay: SharedCanonicalReplay) -> None:
    if mode not in ARMS:
        raise ValueError(mode)
    original = system._chat

    async def paired_chat(self: Any, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int | None, client_role: str, logical_role: str | None = None) -> LLMCallResult:
        role = _role(system_prompt, client_role, logical_role)
        key = replay.key(model, system_prompt, user_prompt, temperature, max_tokens, client_role, role)
        if mode == "shadow_raw" and role in {"teacher", "critic", "student"} and key in replay.results:
            replay.replay_hits += 1
            result = replay.results[key]
            if role == "critic":
                transformed, event = shadow_transform(result, system_prompt, replay.valid_rejected_plan_hashes)
                if event is not None:
                    replay.shadow_events.append(event)
                return transformed
            return result
        result = await original(model, system_prompt, user_prompt, temperature, max_tokens, client_role, logical_role)
        if mode == "canonical_control" and role in {"teacher", "critic", "student"}:
            replay.results[key] = result
        replay.new_calls += 1
        return result

    system._chat = types.MethodType(paired_chat, system)


def classify(*, rejected_witnesses: int, feasible_branches: int, would_commit_branches: int, validation_vote_delta_sum: int) -> str:
    if rejected_witnesses < 3:
        return "NO_CLEAR_SIGNAL"
    if feasible_branches == 0:
        return "CRITIC_FILTERING_JUSTIFIED"
    if feasible_branches >= 2 and would_commit_branches >= 1 and validation_vote_delta_sum >= -1:
        return "CRITIC_OVER_FILTERING_CAUSALLY_SUPPORTED"
    return "MIXED_SHADOW_RAW_SIGNAL"
