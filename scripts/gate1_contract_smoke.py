from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.output_contract import solver_system_prompt
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.utils import normalize_spaces


COMMIT = "336ab63f41fe37cb6e24a4fb7f721b8fdcbbcb90"
OLD_CACHE = REPO_ROOT / "runs_member_aware_per_update_validation_d1f5fa5" / "_shared_solver_cache.sqlite"
DATA_ROOT = REPO_ROOT / "strict_splits_bbh_seed42" / "disambiguation_qa"
ACCEPTED_PROMPT_HASH = "4dd6d3e6bec84ebd8dae379f696a517595b229c416a5b29633c68816c746cc4d"
LENGTH_QUESTION_HASH = "867fb721775a6e51929d6610e6474633adec971668761f614abba153cc40da0a"
MARKDOWN_PROMPT_HASH = "fa33523b297f7db4f8653c00da9a2cc3a6d637ef9f38d85cdb044a51d1bb0380"
MARKDOWN_QUESTION_HASH = "8568aaba955ed9da9e8f9da9ca37b20e7f24d3a87e0c02bbc1dc0a01f17eb4cf"


def question_hash(question: str) -> str:
    return hashlib.sha256(normalize_spaces(question).encode("utf-8")).hexdigest()


def prompt_hash(prompt: str) -> str:
    lines = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.rstrip() for line in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def find_prompt(target_hash: str) -> str:
    roots = (
        REPO_ROOT / "runs_member_aware_per_update_validation_d1f5fa5",
        REPO_ROOT / "runs_member_aware_one_update_29f2ea9",
    )
    for root in roots:
        for path in root.rglob("candidate_decisions.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                stack: list[Any] = [json.loads(line)]
                while stack:
                    value = stack.pop()
                    if isinstance(value, dict):
                        if value.get("prompt_hash") == target_hash and isinstance(value.get("prompt"), str):
                            prompt = value["prompt"]
                            if prompt_hash(prompt) != target_hash:
                                raise RuntimeError(f"historical prompt hash mismatch: {target_hash}")
                            return prompt
                        stack.extend(value.values())
                    elif isinstance(value, list):
                        stack.extend(value)
    raise RuntimeError(f"historical prompt not found: {target_hash}")


def load_questions() -> dict[str, str]:
    questions: dict[str, str] = {}
    for split in ("opt", "val", "test"):
        path = DATA_ROOT / f"{split}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                questions[question_hash(row["question"])] = row["question"]
    return questions


def normal_question_hashes() -> list[str]:
    uri = f"file:{OLD_CACHE.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT question_hash, answer_json FROM solver_cache "
            "WHERE state = 'ready' AND prompt_hash = ? ORDER BY question_hash",
            (ACCEPTED_PROMPT_HASH,),
        ).fetchall()
    finally:
        connection.close()
    excluded = {LENGTH_QUESTION_HASH, MARKDOWN_QUESTION_HASH}
    result = []
    for value, answer_json in rows:
        answer = json.loads(answer_json)
        if answer.get("validity_status") == "valid" and value not in excluded:
            result.append(str(value))
    return result


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the targeted real-API Gate 1 Solver contract smoke.")
    value.add_argument("--out-dir", type=Path, required=True)
    value.add_argument("--repeats", type=int, default=3)
    value.add_argument("--normal-count", type=int, default=8)
    value.add_argument("--seed", type=int, default=42)
    return value


async def run(args: argparse.Namespace) -> int:
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if not 5 <= args.normal_count <= 10:
        raise ValueError("--normal-count must be between 5 and 10")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if current_commit != COMMIT:
        raise RuntimeError(f"Gate 1 requires commit {COMMIT}, got {current_commit}")
    if not OLD_CACHE.is_file():
        raise FileNotFoundError(f"historical diagnostic cache not found: {OLD_CACHE}")
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    prompts = {
        ACCEPTED_PROMPT_HASH: find_prompt(ACCEPTED_PROMPT_HASH),
        MARKDOWN_PROMPT_HASH: find_prompt(MARKDOWN_PROMPT_HASH),
    }
    normal_hashes = normal_question_hashes()[: args.normal_count]
    if len(normal_hashes) < args.normal_count:
        raise RuntimeError("historical cache contains too few normal control questions")
    cases = [
        ("known_length_failure", ACCEPTED_PROMPT_HASH, LENGTH_QUESTION_HASH),
        ("known_markdown_final_answer_b", MARKDOWN_PROMPT_HASH, MARKDOWN_QUESTION_HASH),
    ] + [
        (f"normal_control_{index:02d}", ACCEPTED_PROMPT_HASH, value)
        for index, value in enumerate(normal_hashes, 1)
    ]

    definitions = [
        {
            "case": name,
            "prompt_hash": p_hash,
            "question_hash": q_hash,
            "prompt": prompts[p_hash],
            "question": questions[q_hash],
        }
        for name, p_hash, q_hash in cases
    ]
    (output / "case_definitions.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    records_path = output / "gate1_records.jsonl"
    raw_path = output / "gate1_raw_responses.jsonl"
    prompts_path = output / "gate1_system_prompts.jsonl"

    for case_index, definition in enumerate(definitions):
        for repetition in range(1, args.repeats + 1):
            cache_path = output / f"cache_case{case_index:02d}_rep{repetition:02d}.sqlite"
            run_dir = output / f"case{case_index:02d}_rep{repetition:02d}"
            run_dir.mkdir()
            cfg = Config.from_flat(
                method_version="member_aware_peer_state_v2",
                experiment_setting="shared_member_aware_full",
                agents=5,
                initialization_mode="shared_identical",
                task_type="bbh",
                dataset_format="legacy",
                answer_format="option_letter",
                temperature=0.0,
                solver_max_tokens=1800,
                seed=args.seed,
                out_dir=str(run_dir),
                shared_solver_cache_path=str(cache_path),
                resume_from_checkpoint=False,
            )
            system = PromptEnsembleOptimizationSystem(cfg)
            p_hash = definition["prompt_hash"]
            q_hash = definition["question_hash"]
            try:
                answer = await system.prompt_question_evaluator.evaluate(
                    question=definition["question"],
                    question_hash=q_hash,
                    prompt=definition["prompt"],
                    prompt_hash=p_hash,
                    agent_id=0,
                    solve=system.solve,
                )
                call = system.llm.calls[-1]
                error = ""
            except Exception as exc:
                answer = None
                call = system.llm.calls[-1] if system.llm.calls else {}
                error = f"{type(exc).__name__}: {exc}"

            formal_prompt = solver_system_prompt(definition["prompt"], "option_letter")
            record = {
                "git_commit": COMMIT,
                "case": definition["case"],
                "repetition": repetition,
                "prompt_hash": p_hash,
                "question_hash": q_hash,
                "request_identity": (
                    answer.request_identity if answer is not None
                    else system.prompt_question_evaluator.model_request_identity
                ),
                "finish_reason": call.get("finish_reason", ""),
                "prompt_tokens": int(call.get("prompt_tokens", 0)),
                "completion_tokens": int(call.get("completion_tokens", 0)),
                "validity_status": answer.validity_status if answer is not None else "api_error",
                "final_answer_line_count": answer.final_answer_line_count if answer is not None else 0,
                "raw_final_answer_payload": answer.raw_final_answer_payload if answer is not None else "",
                "response_hash": answer.response_hash if answer is not None else "",
                "solver_max_tokens": cfg.models.solver_max_tokens,
                "temperature": cfg.models.temperature,
                "cache_path": str(cache_path),
                "cache_ready_entries": system.shared_solver_cache.ready_entry_count(),
                "system_prompt_hash": hashlib.sha256(formal_prompt.encode("utf-8")).hexdigest(),
                "error": error,
            }
            append_jsonl(records_path, record)
            append_jsonl(raw_path, {**record, "raw_response": answer.trace if answer is not None else ""})
            append_jsonl(
                prompts_path,
                {
                    "case": definition["case"],
                    "repetition": repetition,
                    "prompt_hash": p_hash,
                    "question_hash": q_hash,
                    "candidate_prompt": definition["prompt"],
                    "system_prompt": formal_prompt,
                },
            )
            print(json.dumps(record, ensure_ascii=False), flush=True)

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    status_counts = Counter(row["validity_status"] for row in records)
    finish_counts = Counter(row["finish_reason"] for row in records)
    line_counts = Counter(row["final_answer_line_count"] for row in records)
    gate1_pass = len(records) == len(definitions) * args.repeats and all(
        row["validity_status"] == "valid"
        and row["final_answer_line_count"] == 1
        and row["finish_reason"] != "length"
        for row in records
    )
    summary = {
        "git_commit": COMMIT,
        "records": len(records),
        "repeats": args.repeats,
        "normal_controls": args.normal_count,
        "solver_max_tokens": 1800,
        "output_dir": str(output),
        "validity_status_counts": dict(status_counts),
        "finish_reason_counts": dict(finish_counts),
        "final_answer_line_count_counts": {str(key): value for key, value in line_counts.items()},
        "gate1_pass": gate1_pass,
    }
    (output / "gate1_run_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if gate1_pass else 2


def main() -> int:
    return asyncio.run(run(parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
