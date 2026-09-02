from __future__ import annotations

import hashlib
import json
import re
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import mutable_prompt_violation_reasons
from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.tcs import STUDENT_SYSTEM_PROMPT, TeacherRepairPlan


ROOT = Path(__file__).resolve().parents[1]
AUTH_ENV = "V18_SAFETY_ONLY_CRITIC_PILOT_AUTHORIZED"
RUN_ROOT = ROOT / "runs" / "v18_safety_only_critic_pilot_20260902"
PREP_ROOT = ROOT / "runs" / "v18_safety_only_critic_prep_20260902"
REPORT_ROOT = ROOT / "reports" / "v18_safety_only_critic_pilot_20260902"
HISTORICAL_ROOT = ROOT / "runs" / "v18_hybrid_online_accumulation_pilot_20260822"
ARMS = ("canonical_llm", "deterministic_safety_only")
LABELS = (
    "SAFETY_ONLY_CRITIC_SUPPORTED",
    "THROUGHPUT_ONLY",
    "SEMANTIC_CRITIC_HAS_FILTERING_VALUE",
    "NO_CLEAR_SIGNAL",
)

_ANTI = re.compile(
    r"\bgold\s+answer\b|\bcase[_ -]?id\b|\bcopy\b.{0,30}\bpeer\b|"
    r"\bpeer\b.{0,30}\b(?:prompt|procedure)\b|\b[0-9a-f]{24,64}\b|"
    r"\b(?:always|default)\b.{0,20}\b(?:choose|select|return)\b.{0,20}\b(?:option\s*)?[abc]\b",
    re.IGNORECASE,
)
_OUTPUT = re.compile(
    r"\bfinal\s+answer\b|\boutput\s+(?:contract|interface|format)\b|"
    r"\bresponse\s+format\b|\banswer\s+label\b|\b(?:only\s+return|return\s+only|"
    r"provide\s+only|only\s+provide)\b|\bwithout\s+(?:any\s+)?additional\s+"
    r"(?:commentary|explanation|reasoning|text)\b",
    re.IGNORECASE,
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safety_only_decision(plan: TeacherRepairPlan) -> dict[str, Any]:
    payload = asdict(plan)
    fields = [str(value).strip() for value in payload.values() if isinstance(value, str)]
    if not fields or any(not value for value in fields[:3]):
        return {"failed_checks": ["actionable_specificity"], "risk_case_ids": [], "feedback": "deterministic schema safety rejection"}
    text = " ".join(fields)
    if _ANTI.search(text):
        return {"failed_checks": ["shortcut_or_copying"], "risk_case_ids": [], "feedback": "deterministic anti-cheating rejection"}
    if mutable_prompt_violation_reasons(text) or _OUTPUT.search(text):
        return {"failed_checks": ["preservation_or_output_risk"], "risk_case_ids": [], "feedback": "deterministic output-contract rejection"}
    return {"failed_checks": [], "risk_case_ids": [], "feedback": ""}


def _plan_from_critic_prompt(system_prompt: str) -> TeacherRepairPlan:
    marker = "TeacherRepairPlan:\n"
    if marker not in system_prompt:
        raise ValueError("deterministic critic request missing TeacherRepairPlan")
    payload = json.loads(system_prompt.rsplit(marker, 1)[1])
    return TeacherRepairPlan(**payload)


class SharedRoleReplay:
    def __init__(self) -> None:
        self.results: dict[str, LLMCallResult] = {}
        self.replay_hits = 0
        self.new_calls = 0
        self.safety_decisions = 0
        self.safety_api_calls = 0

    @staticmethod
    def key(model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int | None, client_role: str, logical_role: str) -> str:
        return canonical_hash([model, system_prompt, user_prompt, temperature, max_tokens, client_role, logical_role])


def install_critic_mode(system: Any, mode: str, replay: SharedRoleReplay) -> None:
    if mode not in ARMS:
        raise ValueError(mode)
    original = system._chat

    async def paired_chat(self: Any, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int | None, client_role: str, logical_role: str | None = None) -> LLMCallResult:
        role = logical_role
        if role is None:
            role = "solver" if client_role == "solver" else "critic" if client_role == "evaluator" else "student" if system_prompt in {STUDENT_SYSTEM_PROMPT, "Return strict JSON only."} else "teacher"
        if mode == "deterministic_safety_only" and role == "critic":
            decision = safety_only_decision(_plan_from_critic_prompt(system_prompt))
            replay.safety_decisions += 1
            return LLMCallResult(json.dumps(decision, separators=(",", ":")), 0, 0, 0, 0.0, "stop")
        key = replay.key(model, system_prompt, user_prompt, temperature, max_tokens, client_role, role)
        if mode == "deterministic_safety_only" and role in {"teacher", "student"} and key in replay.results:
            replay.replay_hits += 1
            return replay.results[key]
        result = await original(model, system_prompt, user_prompt, temperature, max_tokens, client_role, logical_role)
        if role in {"teacher", "student"}:
            replay.results[key] = result
        replay.new_calls += 1
        return result

    system._chat = types.MethodType(paired_chat, system)


def classify(summary: dict[str, dict[str, float]]) -> str:
    canonical = summary["canonical_llm"]
    safety = summary["deterministic_safety_only"]
    reach_up = safety["student_reach_rate"] - canonical["student_reach_rate"] >= (2 / 6)
    feasible_up = safety["feasible_per_branch"] > canonical["feasible_per_branch"]
    commit_ok = safety["would_commit_per_branch"] >= canonical["would_commit_per_branch"] - 0.05
    validation_bad = safety["validation_vote_delta_sum"] < canonical["validation_vote_delta_sum"] - 2
    quality_bad = safety["feasible_per_student"] < canonical["feasible_per_student"] * 0.5
    if reach_up and feasible_up and commit_ok and not validation_bad:
        return "SAFETY_ONLY_CRITIC_SUPPORTED"
    if reach_up and (quality_bad or validation_bad):
        return "SEMANTIC_CRITIC_HAS_FILTERING_VALUE"
    if reach_up:
        return "THROUGHPUT_ONLY"
    return "NO_CLEAR_SIGNAL"
