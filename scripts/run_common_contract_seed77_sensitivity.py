"""Seed77 2x2 solver-contract sensitivity diagnostic.

The study is deliberately evaluation-only.  It compares LF/CRLF question
serialization and omitted/sent provider seed for the common P0 and frozen
Diversity Seed77 P1 prompts on the first ten ExternalValidation cases.  It
cannot load Test50 or run optimization.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI

from infrastructure.common_solver_contract_v1.contract import (
    CONTRACT_SPEC,
    canonical_json_bytes,
    canonical_question_payload,
    contract_identity,
    parse_solver_output,
    solver_system_prompt,
)
from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url


STUDY_ID = "common_contract_seed77_sensitivity_v1"
AUTH_ENV = "COMMON_CONTRACT_SEED77_SENSITIVITY_AUTHORIZED"
DEFAULT_SOURCE = PROJECT_ROOT / "runs/common_solver_contract_v1_seed77_prep_20260907/private_replay_registry.json"
DEFAULT_PREP = PROJECT_ROOT / "runs/common_contract_seed77_sensitivity_prep_20260907"
DEFAULT_RUN = PROJECT_ROOT / "runs/common_contract_seed77_sensitivity_20260907"
DEFAULT_REPORT = PROJECT_ROOT / "reports/common_contract_seed77_sensitivity_20260907"
CASE_COUNT = 10
CONDITIONS = (
    ("LF_SEED_OMITTED", "LF", False),
    ("LF_SEED_SENT", "LF", True),
    ("CRLF_SEED_OMITTED", "CRLF", False),
    ("CRLF_SEED_SENT", "CRLF", True),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    ).strip()


def _plurality(labels: Sequence[str | None]) -> str | None:
    valid = [label for label in labels if label]
    if not valid:
        return None
    counts = Counter(valid)
    maximum = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == maximum)
    return winners[0] if len(winners) == 1 else None


def _variant_request(
    *, prompt: str, question: str, newline: str, send_seed: bool
) -> dict[str, Any]:
    canonical = canonical_question_payload(question)
    rendered_question = canonical if newline == "LF" else canonical.replace("\n", "\r\n")
    request: dict[str, Any] = {
        "model": CONTRACT_SPEC.model,
        "messages": [
            {"role": "system", "content": solver_system_prompt(prompt)},
            {"role": "user", "content": rendered_question},
        ],
        "temperature": CONTRACT_SPEC.temperature,
        "max_tokens": CONTRACT_SPEC.max_tokens,
        "extra_body": {"enable_thinking": CONTRACT_SPEC.enable_thinking},
    }
    if send_seed:
        request["seed"] = 77
    return request


def prepare(source: Path, prep: Path, report: Path) -> dict[str, Any]:
    for path in (prep, report):
        if path.exists():
            raise FileExistsError(f"fresh path required: {path}")
        path.resolve().relative_to(PROJECT_ROOT.resolve())
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("tracked worktree must be clean before freeze")
    source_payload = read_json(source)
    states = {
        str(row["state_id"]): row
        for row in source_payload["states"]
        if row["state_id"] in {"P0_COMMON", "DIVERSITY_SEED77_P1_FINAL"}
    }
    if set(states) != {"P0_COMMON", "DIVERSITY_SEED77_P1_FINAL"}:
        raise AssertionError("Seed77 P0/Diversity state inventory mismatch")
    cases = sorted(source_payload["cases"], key=lambda row: int(row["position"]))[:CASE_COUNT]
    if [int(row["position"]) for row in cases] != list(range(CASE_COUNT)):
        raise AssertionError("first-ten deterministic case rule failed")
    private = {
        "study_id": STUDY_ID,
        "source_contract_identity": contract_identity(),
        "selection_rule": "FIRST_10_EXTERNAL_VALIDATION_CASES_BY_FROZEN_POSITION",
        "states": [states["P0_COMMON"], states["DIVERSITY_SEED77_P1_FINAL"]],
        "cases": cases,
        "conditions": [
            {"condition": name, "newline": newline, "provider_seed_sent": sent}
            for name, newline, sent in CONDITIONS
        ],
        "test50_accessed": False,
        "optimization": False,
    }
    prep.mkdir(parents=True)
    write_json(prep / "private_registry.json", private)
    source_files = (
        Path("infrastructure/common_solver_contract_v1/contract.py"),
        Path("scripts/run_common_contract_seed77_sensitivity.py"),
    )
    freeze = {
        "study_id": STUDY_ID,
        "execution_commit": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": True,
        "source_registry_sha256": sha256_file(source),
        "files": [
            {"path": path.as_posix(), "sha256": sha256_file(PROJECT_ROOT / path)}
            for path in source_files
        ],
    }
    write_json(prep / "source_freeze.json", freeze)
    report.mkdir(parents=True)
    protocol = {
        "study_id": STUDY_ID,
        "phase": "evaluation_only_contract_sensitivity",
        "seed": 77,
        "case_selection": private["selection_rule"],
        "case_count": CASE_COUNT,
        "states": list(states),
        "factorial": {
            "question_newline": ["LF", "CRLF"],
            "provider_seed": ["OMITTED", "SENT_77"],
        },
        "unchanged": [
            "endpoint",
            "qwen3-8b",
            "thinking=false",
            "temperature=0",
            "max_tokens=1800",
            "system message",
            "strict parser",
            "operational retry cap",
        ],
        "interpretation": "descriptive small-sample sensitivity; not method selection",
        "classifier": {
            "DIVERSITY_MORE_SENSITIVE": "Diversity Vote range exceeds P0 by >=0.20 or MeanMember range exceeds P0 by >=0.10",
            "SHARED_CONTRACT_SENSITIVITY": "both P0 and Diversity Vote ranges are >=0.20",
            "NO_MATERIAL_SENSITIVITY": "both Vote ranges and Diversity MeanMember range are <=0.10",
            "INCONCLUSIVE_MIXED_SENSITIVITY": "otherwise",
        },
        "test50_accessed": False,
        "optimization": False,
    }
    write_json(report / "protocol_freeze.json", protocol)
    facts = {
        "phase_a_gate": "PASS",
        "api_calls": 0,
        "state_count": 2,
        "condition_count": 4,
        "case_count": CASE_COUNT,
        "test50_accessed": False,
        "optimization": False,
    }
    write_json(report / "phase_a_gate.json", facts)
    write_json(
        report / "provenance.json",
        {
            "study_id": STUDY_ID,
            "execution_commit": freeze["execution_commit"],
            "source_contract_identity": contract_identity(),
            "raw_evidence_tracked": False,
        },
    )
    write_json(
        report / "sha256_manifest.json",
        {
            path.name: sha256_file(path)
            for path in sorted(report.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        },
    )
    return facts


def _verify_freeze(prep: Path, *, require_current_checkout: bool = True) -> dict[str, Any]:
    freeze = read_json(prep / "source_freeze.json")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze["execution_commit"], "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("source-freeze commit is not an ancestor of execution HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        if require_current_checkout:
            raise RuntimeError("tracked worktree must be clean")
    for row in freeze["files"]:
        if require_current_checkout:
            observed = sha256_file(PROJECT_ROOT / row["path"])
        else:
            content = subprocess.check_output(
                ["git", "show", f"{freeze['execution_commit']}:{row['path']}"],
                cwd=PROJECT_ROOT,
            )
            observed = hashlib.sha256(content).hexdigest()
        if observed != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")
    return freeze


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    if status is not None:
        return int(status) in CONTRACT_SPEC.retry_status_codes
    return isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError))


async def execute(prep: Path, output: Path) -> dict[str, Any]:
    if os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1 is required")
    _verify_freeze(prep)
    if output.exists():
        raise FileExistsError("fresh output root required")
    output.mkdir(parents=True)
    registry = read_json(prep / "private_registry.json")
    _, api_key = resolve_api_key("DASHSCOPE_API_KEY")
    _, base_url = resolve_base_url("DASHSCOPE_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("provider credentials unavailable")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    provider_attempts = provider_success = provider_failure = 0
    prompt_tokens = completion_tokens = 0

    async def call(request: dict[str, Any]) -> tuple[str, int, int, str, bool]:
        nonlocal provider_attempts, provider_success, provider_failure, prompt_tokens, completion_tokens
        identity = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        if identity in cache:
            return cache[identity], 0, 0, identity, True
        for attempt in range(1, CONTRACT_SPEC.transport_attempt_cap + 1):
            provider_attempts += 1
            try:
                response = await client.chat.completions.create(
                    **request, timeout=CONTRACT_SPEC.timeout_seconds
                )
                raw = response.choices[0].message.content or ""
                usage = response.usage
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                provider_success += 1
                prompt_tokens += pt
                completion_tokens += ct
                cache[identity] = raw
                return raw, pt, ct, identity, False
            except Exception as exc:
                provider_failure += 1
                if not _retryable(exc) or attempt == CONTRACT_SPEC.transport_attempt_cap:
                    raise
                await asyncio.sleep(CONTRACT_SPEC.retry_backoff_seconds[attempt - 1])
        raise AssertionError("unreachable")

    completed_cells: list[str] = []
    for name, newline, sent in CONDITIONS:
        for state in registry["states"]:
            state_id = str(state["state_id"])
            for case in registry["cases"]:
                labels: list[str | None] = []
                valids: list[bool] = []
                request_ids: list[str] = []
                for prompt in state["ordered_prompts"]:
                    request = _variant_request(
                        prompt=str(prompt),
                        question=str(case["question"]),
                        newline=newline,
                        send_seed=sent,
                    )
                    raw, _, _, request_id, _ = await call(request)
                    parsed = parse_solver_output(raw, question=str(case["question"]))
                    labels.append(parsed.answer if parsed.valid else None)
                    valids.append(parsed.valid)
                    request_ids.append(request_id)
                vote = _plurality(labels)
                rows.append(
                    {
                        "condition": name,
                        "state_id": state_id,
                        "case_position": int(case["position"]),
                        "member_labels": labels,
                        "member_valid": valids,
                        "request_identities": request_ids,
                        "vote_correct": vote == case["gold"],
                        "oracle_correct": any(label == case["gold"] for label in labels),
                    }
                )
            completed_cells.append(f"{name}:{state_id}")
            write_json(output / "progress_private.json", {"completed_cells": completed_cells})
            write_json(output / "predictions_private.json", rows)
            write_json(output / "raw_cache_private.json", cache)
    summary = {
        "status": "PASS",
        "study_id": STUDY_ID,
        "completed_cells": len(completed_cells),
        "row_count": len(rows),
        "logical_calls": sum(len(row["member_labels"]) for row in rows),
        "provider_attempts": provider_attempts,
        "provider_success": provider_success,
        "provider_failure": provider_failure,
        "cache_hits": sum(len(row["member_labels"]) for row in rows) - provider_success,
        "cache_entries": len(cache),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "test50_accessed": False,
        "optimization": False,
    }
    write_json(output / "summary_private.json", summary)
    write_json(output / "execution_manifest.json", {**summary, "status": "COMPLETE"})
    return summary


def analyze(prep: Path, run_root: Path, report: Path) -> dict[str, Any]:
    if (report / "summary.json").exists():
        raise FileExistsError("analysis report already exists")
    _verify_freeze(prep, require_current_checkout=False)
    execution = read_json(run_root / "execution_manifest.json")
    registry = read_json(prep / "private_registry.json")
    rows = read_json(run_root / "predictions_private.json")
    # The first completed execution persisted summary.status=PASS over the
    # intended COMPLETE marker.  Inventory and isolation fields are the
    # authoritative completion evidence for that preserved run.
    if execution["status"] not in {"PASS", "COMPLETE"} or execution["provider_failure"] != 0:
        raise RuntimeError("execution gate failed")
    if len(rows) != 80 or execution["test50_accessed"] or execution["optimization"]:
        raise RuntimeError("inventory/isolation gate failed")
    gold = {int(row["position"]): str(row["gold"]) for row in registry["cases"]}
    aggregates: list[dict[str, Any]] = []
    for condition, _, _ in CONDITIONS:
        for state_id in ("P0_COMMON", "DIVERSITY_SEED77_P1_FINAL"):
            selected = [row for row in rows if row["condition"] == condition and row["state_id"] == state_id]
            member_count = len(selected[0]["member_labels"])
            member_correct = [0] * member_count
            invalid = 0
            for row in selected:
                answer = gold[int(row["case_position"])]
                for index, (label, valid) in enumerate(zip(row["member_labels"], row["member_valid"], strict=True)):
                    member_correct[index] += int(bool(valid) and label == answer)
                    invalid += int(not bool(valid))
            aggregates.append(
                {
                    "condition": condition,
                    "state_id": state_id,
                    "vote_accuracy": sum(bool(row["vote_correct"]) for row in selected) / CASE_COUNT,
                    "oracle_accuracy": sum(bool(row["oracle_correct"]) for row in selected) / CASE_COUNT,
                    "mean_member_accuracy": sum(member_correct) / (CASE_COUNT * member_count),
                    "invalid_rate": invalid / (CASE_COUNT * member_count),
                }
            )
    report.mkdir(parents=True, exist_ok=True)
    with (report / "per_cell_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    by_state = {
        state_id: [row for row in aggregates if row["state_id"] == state_id]
        for state_id in ("P0_COMMON", "DIVERSITY_SEED77_P1_FINAL")
    }
    ranges = {
        state_id: {
            key: max(row[key] for row in state_rows) - min(row[key] for row in state_rows)
            for key in ("vote_accuracy", "oracle_accuracy", "mean_member_accuracy", "invalid_rate")
        }
        for state_id, state_rows in by_state.items()
    }
    p0_vote = ranges["P0_COMMON"]["vote_accuracy"]
    diversity_vote = ranges["DIVERSITY_SEED77_P1_FINAL"]["vote_accuracy"]
    diversity_mean = ranges["DIVERSITY_SEED77_P1_FINAL"]["mean_member_accuracy"]
    p0_mean = ranges["P0_COMMON"]["mean_member_accuracy"]
    if diversity_vote >= p0_vote + 0.20 or diversity_mean >= p0_mean + 0.10:
        classifier = "DIVERSITY_MORE_SENSITIVE"
    elif p0_vote >= 0.20 and diversity_vote >= 0.20:
        classifier = "SHARED_CONTRACT_SENSITIVITY"
    elif max(p0_vote, diversity_vote, diversity_mean) <= 0.10:
        classifier = "NO_MATERIAL_SENSITIVITY"
    else:
        classifier = "INCONCLUSIVE_MIXED_SENSITIVITY"
    summary = {
        "analysis_gate": "PASS",
        "classifier": classifier,
        "ranges": ranges,
        "case_count": CASE_COUNT,
        "condition_count": 4,
        "test50_accessed": False,
        "optimization": False,
        "caveat": "single realization per cell; omitted-seed provider nondeterminism remains possible",
    }
    write_json(report / "summary.json", summary)
    write_json(
        report / "audit.json",
        {
            "gate": "PASS",
            "execution_status_observed": execution["status"],
            "completion_reconciled_from_full_inventory": execution["status"] == "PASS",
            "completed_cells": execution["completed_cells"],
            "row_count": execution["row_count"],
            "provider_success": execution["provider_success"],
            "provider_failure": execution["provider_failure"],
            "test50_accessed": False,
            "optimization": False,
            "source_freeze": "PASS",
        },
    )
    (report / "README.md").write_text(
        "# Seed77 common-contract sensitivity\n\n"
        f"Classifier: `{classifier}`. This is a 10-case descriptive 2x2 diagnostic, "
        "not a method-selection experiment. Test50 was not accessed and optimization was not run.\n",
        encoding="utf-8",
        newline="\n",
    )
    forbidden = re.compile(r"(?:[A-Za-z]:\\|DASHSCOPE|api[_-]?key|bearer\s+|raw_response|question_text|gold_answer|\.sqlite|checkpoint)", re.I)
    findings = []
    for path in report.iterdir():
        if path.is_file() and path.name not in {"sanitization_manifest.json", "sha256_manifest.json"}:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if forbidden.search(line):
                    findings.append({"file": path.name, "line": line_number})
    write_json(report / "sanitization_manifest.json", {"status": "PASS" if not findings else "FAIL", "findings": findings})
    if findings:
        raise RuntimeError("sanitization failed")
    write_json(
        report / "sha256_manifest.json",
        {
            path.name: sha256_file(path)
            for path in sorted(report.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    for path in (args.prep, args.run_root, args.report):
        path.resolve().relative_to(PROJECT_ROOT.resolve())
    if args.prepare:
        result = prepare(args.source.resolve(), args.prep.resolve(), args.report.resolve())
    elif args.execute:
        result = asyncio.run(execute(args.prep.resolve(), args.run_root.resolve()))
    else:
        result = analyze(args.prep.resolve(), args.run_root.resolve(), args.report.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
