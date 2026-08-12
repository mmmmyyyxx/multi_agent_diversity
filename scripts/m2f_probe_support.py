from __future__ import annotations

import ast
import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generic_m20_probe_support import evaluation_system, responsibility_evidence_hash, state_payload, team_prompt_hash
from multi_dataset_diverse_rl.peer_state import build_peer_vote_context
from multi_dataset_diverse_rl.utils import normalize_prompt_text

VERSION = "v16_m2f_candidate_specific_compatibility_repair_v1"
SOURCE_EXECUTION_COMMIT = "b18be6cf893b77b698b9f4ff92af788baff19d0e"
SOURCE_AUDIT_COMMIT = "3a5a16284755251ef6ba341d5496736529e85a8a"
SOURCE_HASHES = (
    "3b494ddc756e70cf877ee46224ad72d4fd81cba71dd77754bf801ef849879671",
    "df37e4360977a65edcb47bc7786fbc48869469065668ed5aabdaa703d98af8d2",
    "2249467342abe57b0f01f7aaa251153ceeb91f0bfddf1b030e1fb4e862b4d20f",
    "c06c7420a7252455b29066ba2990a4a749d7893f86dffd591a89ddce96438481",
    "62c02b9dbd25d08c910dbc1d672926981279763d16e1bbdff1b2efac85fc14d2",
    "d70344b3a14adafd45fb4ff7a265928e27426c9ca11f7480004399b1f4ffa6a3",
    "18289914279d8f2c62ef67d95ef79055e551ca221ea2318ee9b15e0cbcbb131f",
)
MODEL = "qwen3-14b"
THINKING = False
AUTH_ENV = "M2F_COMPATIBILITY_REPAIR_PROBE_AUTHORIZED"
STABLE_CAP = 2
REPAIR_MAX_TOKENS = 3000

REPAIR_SYSTEM_PROMPT = """You repair one already-targeted member prompt. Return strict JSON only with exactly one field: repaired_prompt. Do not quote, identify, or memorize examples. Do not add answer lookup rules, hashes, labels, or the immutable output contract."""
REPAIR_INSTRUCTION = """The SOURCE CANDIDATE successfully corrected the assigned responsibility examples below, but introduced new failures relative to the parent prompt.

Revise the SOURCE CANDIDATE, not the parent from scratch.
Primary requirement: retain the responsibility-specific behavior that produced the successful repairs.
Secondary requirement: remove, narrow, or revert only broader behavioral changes likely to have caused the observed competence losses.
Do not change the assigned responsibility objective. Do not memorize question text, option letters, gold answers, or individual examples. Do not add a universal rule merely to satisfy loss examples. Prefer restoring parent behavior outside the targeted responsibility mechanism.
Return one complete revised member prompt as strict JSON: {"repaired_prompt":"..."}."""


def sha_text(value: str) -> str:
    return hashlib.sha256(normalize_prompt_text(value).encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_cached_answers(cache: Path, prompt_hash: str, system: Any) -> dict[str, dict[str, Any]]:
    con = sqlite3.connect(f"file:{cache.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        con.execute("PRAGMA query_only=ON")
        rows = con.execute(
            "SELECT question_hash,answer_json FROM solver_cache WHERE state='ready' AND prompt_hash=? "
            "AND model_request_identity=? AND parser_version=? AND temperature=? AND evaluation_replica_seed=? "
            "AND solver_model=? AND max_tokens=? AND output_contract_version=? ORDER BY cache_key",
            (prompt_hash, system.prompt_question_evaluator.model_request_identity,
             system.prompt_question_evaluator.parser_version, system.prompt_question_evaluator.temperature,
             system.prompt_question_evaluator.decoding_seed, system.cfg.models.agent_model,
             system.cfg.models.solver_max_tokens, system.cfg.peer_state.solver_output_contract_version),
        ).fetchall()
    finally:
        con.close()
    result: dict[str, dict[str, Any]] = {}
    for question_hash, raw in rows:
        if question_hash in result:
            raise ValueError("duplicate cached observation")
        result[str(question_hash)] = json.loads(raw)
    return result


def competence_role(system: Any, target: int, state: Any, stable: set[str]) -> str:
    peer = build_peer_vote_context(state, target)
    if state.gold_vote_count == 1:
        return "unique"
    if state.vote_correct and peer.peer_margin <= 0:
        return "pivotal"
    return "stable" if state.question_hash in stable else "fragile"


def build_repair_request(case: dict[str, Any]) -> str:
    payload = {
        "parent_member_prompt": case["parent_prompt"],
        "source_m20_candidate_prompt": case["source_candidate_prompt"],
        "successful_assigned_responsibility_repairs": case["repair_evidence"],
        "candidate_specific_competence_losses": case["loss_evidence"],
        "numeric_summary": case["numeric_summary"],
    }
    return REPAIR_INSTRUCTION + "\n\nRepairInput:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_repair_output(text: str, case: dict[str, Any]) -> str:
    from multi_dataset_diverse_rl.utils import extract_json_obj
    payload = extract_json_obj(text)
    if not isinstance(payload, dict) or set(payload) != {"repaired_prompt"}:
        raise ValueError("repair response must contain exactly repaired_prompt")
    prompt = normalize_prompt_text(payload["repaired_prompt"] if isinstance(payload["repaired_prompt"], str) else "")
    if not prompt or prompt == normalize_prompt_text(case["source_candidate_prompt"]):
        raise ValueError("repair prompt is empty or unchanged")
    lowered = prompt.lower()
    forbidden = ("final_answer:", "question_hash", "gold answer", "answer choice a", "answer choice b", "answer choice c", "answer choice d")
    if any(token in lowered for token in forbidden):
        raise ValueError("repair prompt contains forbidden answer or protocol material")
    for row in [*case["repair_evidence"], *case["loss_evidence"]]:
        normalized = " ".join(str(row["question"]).lower().split())
        if len(normalized) >= 32 and normalized in " ".join(lowered.split()):
            raise ValueError("repair prompt memorizes supplied question")
    return prompt


def _function_source(commit: str, path: str, names: set[str]) -> dict[str, str]:
    text = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT, text=True, encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found[node.name] = "\n".join(lines[node.lineno - 1:node.end_lineno])
    return found


def evaluator_compatibility(current_commit: str) -> dict[str, Any]:
    names = {"prompt_hash", "evaluate_candidates", "_module2_candidate_effects"}
    old = _function_source(SOURCE_EXECUTION_COMMIT, "multi_dataset_diverse_rl/system.py", names)
    new = _function_source(current_commit, "multi_dataset_diverse_rl/system.py", names)
    exact_files = [
        "multi_dataset_diverse_rl/candidate_selection.py",
        "multi_dataset_diverse_rl/peer_state.py",
        "multi_dataset_diverse_rl/responsibility.py",
        "multi_dataset_diverse_rl/responsibility_contribution.py",
    ]
    file_checks = {}
    for path in exact_files:
        a = subprocess.check_output(["git", "show", f"{SOURCE_EXECUTION_COMMIT}:{path}"], cwd=ROOT)
        b = subprocess.check_output(["git", "show", f"{current_commit}:{path}"], cwd=ROOT)
        file_checks[path] = hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
    function_checks = {name: old.get(name) == new.get(name) and name in old for name in names}
    status = all(file_checks.values()) and all(function_checks.values())
    return {"status": "PASS" if status else "FAIL", "source_execution_commit": SOURCE_EXECUTION_COMMIT,
            "current_commit": current_commit, "exact_file_checks": file_checks,
            "exact_evaluator_function_checks": function_checks,
            "semantic_dimensions": ["target rollout evaluator", "plurality aggregator", "terminal-invalid handling", "target gain", "vote gain/loss", "common-safe", "responsibility membership"]}


def loss_decomposition(source: set[str], repaired: set[str]) -> dict[str, int]:
    return {"recovered": len(source - repaired), "persistent": len(source & repaired), "new": len(repaired - source)}


def classify(*, retention: float, source_loss: int, repair_loss: int, collateral_wins: int,
             collateral_losses: int, source_pivotal: int, repair_pivotal: int,
             rescues: int, source_infeasible: int) -> tuple[str, dict[str, bool]]:
    rescue_required = 2 if source_infeasible >= 2 else (source_infeasible + 1) // 2
    criteria = {"targeting": retention >= .8,
                "collateral": repair_loss < source_loss and collateral_wins > collateral_losses,
                "pivotal": repair_pivotal <= source_pivotal,
                "rescue": rescues >= rescue_required}
    if not criteria["targeting"]: label = "TARGETING_LOST"
    elif not criteria["collateral"]: label = "COLLATERAL_NOT_REDUCED"
    elif not criteria["pivotal"]: label = "PIVOTAL_COMPATIBILITY_FAILED"
    elif not criteria["rescue"]: label = "NO_FEASIBILITY_RESCUE"
    elif all(criteria.values()): label = "REPAIR_WORKS"
    else: label = "MIXED"
    return label, {**criteria, "rescue_required": rescue_required}
