"""Run the non-committing contract-adapted RG-GEPA fixed-parent pilot.

Only closed-vocabulary reflection hypotheses cross the optimizer boundary.
Candidate prompts are rendered deterministically and raw reflection text is
kept only in the ignored runtime root.  The canonical method is untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.common_solver_contract_v1.contract import contract_identity  # noqa: E402
from multi_dataset_diverse_rl.candidate_selection import (  # noqa: E402
    common_monotone_safe_key,
    evaluate_constraints,
)
from multi_dataset_diverse_rl.experimental_contract_adapted_rg_gepa import (  # noqa: E402
    AVOIDANCE_PRIORITIES,
    BEHAVIORAL_CHANGES,
    ContractAdaptedProtocol,
    FAILURE_PATTERNS,
    PRESERVATION_PRIORITIES,
    parse_edit_hypothesis,
    render_contract_adapted_prompt,
    renderer_vocabulary_identity,
)
from multi_dataset_diverse_rl.experimental_rg_gepa import (  # noqa: E402
    CandidateScore,
    TeamVector,
    progressive_promotions,
    team_pareto_winner,
    validate_ledger_record,
)
from scripts import run_responsibility_guided_gepa_fixed_parent_pilot as base  # noqa: E402


AUTH_ENV = "CONTRACT_ADAPTED_RG_GEPA_PILOT_AUTHORIZED"
DEFAULT_PREP = ROOT / "runs" / "contract_adapted_rg_gepa_fixed_parent_pilot_v1_prep_20260909"
DEFAULT_RUN = ROOT / "runs" / "contract_adapted_rg_gepa_fixed_parent_pilot_v1_20260909"
DEFAULT_REPORT = ROOT / "reports" / "contract_adapted_rg_gepa_fixed_parent_pilot_v1_execution"
AUDIT_ONLY_CASES = frozenset({"seed76_u0_coverage_target1", "seed77_u0_coverage_target2"})


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _serialized_protocol() -> dict[str, Any]:
    """Return the protocol exactly as it appears after a JSON round trip."""

    return json.loads(json.dumps(asdict(ContractAdaptedProtocol()), sort_keys=True))


def _hypothesis_request(case: dict[str, Any], candidate_index: int) -> tuple[str, str]:
    evidence = case["minibatch_private"][:4]
    cases = "\n\n".join(
        f"Example {offset + 1}:\nQuestion: {row['question']}\nGold: {row['answer']}"
        for offset, row in enumerate(evidence)
    )
    choices = {
        "failure_pattern": list(FAILURE_PATTERNS),
        "behavioral_change": list(BEHAVIORAL_CHANGES),
        "preserve": list(PRESERVATION_PRIORITIES),
        "avoid": list(AVOIDANCE_PRIORITIES),
    }
    system = (
        "Select one bounded edit hypothesis for a reasoning procedure. Return exactly one JSON "
        "object with the four requested keys and one listed symbolic value per key. Do not write "
        "a replacement prompt, prose, markdown, examples, or any interface instruction."
    )
    user = (
        f"Responsibility lane: {case['responsibility_type']}.\n"
        f"Allowed values: {json.dumps(choices, sort_keys=True)}\n\n"
        "Choose a general diagnosis and change that may repair the responsibility evidence while "
        "preserving broad behavior. The program, not you, will render the candidate.\n\n"
        f"Optimization evidence (do not memorize):\n{cases}\n\nMutation index: {candidate_index}"
    )
    return system, user


def _candidate_meta(case: dict[str, Any], candidate_id: str, stage: str) -> dict[str, Any]:
    return {
        "seed": case["source_seed"],
        "parent_id": case["case_id"],
        "update_index": case["source_update_index"],
        "candidate_id": candidate_id,
        # The durable v2 ledger names the optimizer transport family rather
        # than the candidate representation variant.
        "proposal_engine": "gepa_reflection",
        "evaluation_stage": stage,
    }


def _winner(rows: list[dict[str, Any]], *, pareto: bool) -> tuple[str | None, list[str]]:
    scores = [
        CandidateScore(row["candidate_id"], TeamVector(**row["full"]), 0, True)
        for row in rows
    ]
    if pareto:
        got, frontier = team_pareto_winner(scores)
        return (got.candidate_id if got else None, [item.candidate_id for item in frontier])
    got = max(rows, key=lambda row: tuple(row["current_selector_key"]), default=None)
    return (got["candidate_id"] if got else None, [])


async def _run_case(
    case: dict[str, Any],
    root: Path,
    ledger_writer: base.RGGEPAExecutionLedger,
) -> dict[str, Any]:
    case_root = root / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=False)
    protocol = ContractAdaptedProtocol()
    system = base._CommonSystem(base._cfg(case, case_root), ledger_writer=ledger_writer)
    system.set_solver_stage(_candidate_meta(case, "parent", "full_team"))
    try:
        await system.initialize_fixed_probe(case["questions"])
    finally:
        system.set_solver_stage(None)
    assert system.fixed_probe is not None

    target = int(case["target_member"])
    assigned = set(case["assigned_example_ids"])
    parent_profile = system.active_profiles[target]
    parent = base._evaluation(
        system, target, case["parent_prompts"][target], parent_profile, None, assigned
    )
    mini_ids = [item["example_id"] for item in case["minibatch_private"]]
    by_hash = {row.question_hash: index for index, row in enumerate(system.fixed_probe.examples)}
    mini_indices = [by_hash[item] for item in mini_ids]
    parent_mini = base._evaluation(
        system,
        target,
        case["parent_prompts"][target],
        tuple(parent_profile[index] for index in mini_indices),
        mini_indices,
        assigned,
    )
    base._record_minibatch_parent_reuse(case, ledger_writer)

    candidates: list[dict[str, Any]] = []
    for index in range(protocol.candidates_per_case):
        candidate_id = f"B_PRIME_{index}"
        begin = len(system.llm.calls)
        system_prompt, user_prompt = _hypothesis_request(case, index)
        try:
            response = await system.llm.chat_result(
                "qwen3.7-flash",
                system_prompt,
                user_prompt,
                0.3,
                300,
                "optimizer",
                "reflection",
            )
        finally:
            base._persist_new_optimizer_calls(
                system,
                begin,
                _candidate_meta(case, candidate_id, "reflection"),
                ledger_writer,
            )
        try:
            hypothesis = parse_edit_hypothesis(response.text)
            prompt = render_contract_adapted_prompt(
                case["parent_prompts"][target], hypothesis
            )
        except Exception as exc:
            _write(
                case_root / f"{candidate_id}_reflection_failure_private.json",
                {
                    "candidate_id": candidate_id,
                    "raw_response": response.text,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise RuntimeError(
                f"contract-adapted hypothesis failed before candidate generation: {candidate_id}"
            ) from exc

        hard, reason = base._hard_gate(prompt, case["questions"])
        if not hard:
            raise RuntimeError("deterministic renderer produced a hard-gate-invalid candidate")
        hypothesis_payload = asdict(hypothesis)
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_index": index,
            "proposal_engine": "contract_adapted_reflection",
            "prompt": prompt,
            "prompt_hash": _sha(prompt.encode("utf-8")),
            "reflection_response_hash": _sha(response.text.encode("utf-8")),
            "hypothesis": hypothesis_payload,
            "hypothesis_hash": hypothesis.identity(),
            "schema_valid": True,
            "renderer_version": protocol.renderer_version,
            "renderer_vocabulary_hash": renderer_vocabulary_identity(),
            "raw_reflection_embedded": False,
            "hard_gate_passed": True,
            "hard_gate_reason": "",
            "generated": True,
            "audit_only": case["case_id"] in AUDIT_ONLY_CASES,
        }
        mini_profile = await base._profile(
            system,
            target,
            prompt,
            mini_indices,
            _candidate_meta(case, candidate_id, "minibatch_candidate"),
        )
        mini_eval = base._evaluation(
            system, target, prompt, mini_profile, mini_indices, assigned
        )
        row["mini"] = asdict(base._vector(mini_eval, parent_mini))
        candidates.append(row)

    promotions = progressive_promotions(
        parent=base._vector(parent_mini),
        candidates=[
            CandidateScore(row["candidate_id"], TeamVector(**row["mini"]), 0, True)
            for row in candidates
        ],
    )
    promoted_ids = {row.candidate_id for row in promotions}
    for row in candidates:
        row["promoted"] = row["candidate_id"] in promoted_ids
        row["progressive_promoted"] = row["promoted"]
        if not row["promoted"] and not row["audit_only"]:
            continue
        stage = "full_member" if row["promoted"] else "audit_only"
        full_profile = await base._profile(
            system,
            target,
            row["prompt"],
            None,
            _candidate_meta(case, row["candidate_id"], stage),
        )
        full = base._evaluation(system, target, row["prompt"], full_profile, None, assigned)
        constraint = evaluate_constraints(full, parent)
        row["full"] = asdict(base._vector(full, parent))
        row["current_selector_key"] = list(
            common_monotone_safe_key(full, int(row["candidate_index"]))
        )
        row["feasible"] = bool(constraint.passed)
        row["constraint_reasons"] = list(constraint.rejection_reasons)

    eligible = [
        row
        for row in candidates
        if row.get("promoted") and row.get("feasible") and "full" in row
    ]
    b0, _ = _winner(eligible, pareto=False)
    b1, frontier = _winner(eligible, pareto=True)
    runtime_accounting = base._runtime_accounting(system)
    private = {
        "case": case,
        "parent": asdict(base._vector(parent)),
        "parent_mini": asdict(base._vector(parent_mini)),
        "candidates": candidates,
        "winners": {"B0_PRIME": b0, "B1_PRIME": b1},
        "b1_prime_frontier": frontier,
        "actual_commits": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "protocol": asdict(protocol),
        "runtime_accounting": runtime_accounting,
    }
    _write(case_root / "private_result.json", private)
    return {
        "case_id": case["case_id"],
        "seed": case["source_seed"],
        "responsibility_type": case["responsibility_type"],
        "parent": private["parent"],
        "candidates": [base._public_candidate(row) for row in candidates],
        "winners": private["winners"],
        "b1_prime_frontier": frontier,
        "actual_commits": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "runtime_accounting": runtime_accounting,
    }


def _summary(results: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    pool = [candidate for result in results for candidate in result["candidates"]]
    valid = [candidate for candidate in pool if candidate["hard_gate_passed"]]
    promoted = [candidate for candidate in valid if candidate.get("promoted")]
    full = [candidate for candidate in promoted if "full" in candidate]
    feasible = [candidate for candidate in full if candidate.get("feasible")]
    arms = [
        {
            "arm": arm,
            "proposal_engine": "contract_adapted_reflection",
            "evaluation_mode": "progressive",
            "selection_mode": selection,
            "generated": len(pool),
            "schema_valid": sum(bool(row.get("schema_valid")) for row in pool),
            "contract_valid": len(valid),
            "promoted": len(promoted),
            "full_evaluated": len(full),
            "feasible": len(feasible),
            "would_commit": sum(result["winners"][arm] is not None for result in results),
        }
        for arm, selection in (
            ("B0_PRIME", "current"),
            ("B1_PRIME", "team_pareto"),
        )
    ]
    calls: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "logical_calls": 0,
            "provider_calls": 0,
            "successful_provider_calls": 0,
            "failed_provider_attempts": 0,
            "cache_hits": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    logical_ids: dict[str, set[str]] = defaultdict(set)
    for row in ledger:
        stage = str(row["evaluation_stage"])
        logical_ids[stage].add(str(row["logical_call_id"]))
        calls[stage]["provider_calls"] += int(row["provider_attempts"])
        calls[stage]["successful_provider_calls"] += int(row["successful_provider_calls"])
        calls[stage]["failed_provider_attempts"] += int(row["provider_attempts"]) - int(
            row["successful_provider_calls"]
        )
        calls[stage]["cache_hits"] += int(bool(row["cache_hit"]))
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            calls[stage][key] += int(row[key])
    for stage, ids in logical_ids.items():
        calls[stage]["logical_calls"] = len(ids)
    return {
        "arms": arms,
        "cost_by_stage": [{"stage": stage, **values} for stage, values in sorted(calls.items())],
        "api_calls": sum(int(row["successful_provider_calls"]) for row in ledger),
        "validation_calls": 0,
        "test_calls": 0,
    }


async def execute(args: argparse.Namespace) -> None:
    if not args.authorize_api or os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"--authorize-api and {AUTH_ENV}=1 are required")
    if args.run.exists() or args.report.exists():
        raise FileExistsError("contract-adapted execution roots must be fresh")
    registry = _read(args.prep / "private_registry.json")
    if (
        registry["protocol"] != _serialized_protocol()
        or registry["protocol_hash"] != ContractAdaptedProtocol().identity()
    ):
        raise RuntimeError("frozen contract-adapted protocol mismatch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", registry["execution_commit"], "HEAD"],
        cwd=ROOT,
    ).returncode:
        raise RuntimeError("frozen execution source is not an ancestor of HEAD")
    args.run.mkdir(parents=True)
    ledger_writer = base.RGGEPAExecutionLedger(args.run / "api_ledger_private.jsonl")
    results: list[dict[str, Any]] = []
    for case in registry["cases"]:
        case["minibatch_private"] = [
            next(row for row in case["questions"] if row["example_id"] == item["example_id"])
            for item in registry["minibatches"][case["case_id"]]
        ]
        results.append(await _run_case(case, args.run, ledger_writer))
        _write(
            args.run / "progress.json",
            {"completed_cases": len(results), "total_cases": 6, "validation_calls": 0, "test_calls": 0},
        )
    summary = _summary(results, ledger_writer.rows)
    _write(args.run / "result_sanitized.json", {"results": results, "summary": summary})
    print(json.dumps({"status": "PASS", "completed_cases": 6, "test_calls": 0}, sort_keys=True))


def audit(run: Path, prep: Path) -> dict[str, Any]:
    payload = _read(run / "result_sanitized.json")
    results = payload["results"]
    if len(results) != 6:
        raise RuntimeError("contract-adapted audit failed: expected six cases")
    if any(
        int(row["actual_commits"]) != 0
        or int(row["validation_calls"]) != 0
        or int(row["test_calls"]) != 0
        for row in results
    ):
        raise RuntimeError("contract-adapted audit failed: isolation violation")
    registry = _read(prep / "private_registry.json")
    if (
        registry["protocol"] != _serialized_protocol()
        or registry["protocol_hash"] != ContractAdaptedProtocol().identity()
    ):
        raise RuntimeError("contract-adapted audit failed: protocol drift")
    expected_cases = {str(case["case_id"]) for case in registry["cases"]}
    if {str(row["case_id"]) for row in results} != expected_cases:
        raise RuntimeError("contract-adapted audit failed: case identity drift")

    candidate_count = 0
    for result in results:
        private = _read(run / result["case_id"] / "private_result.json")
        if len(result["candidates"]) != 2 or len(private["candidates"]) != 2:
            raise RuntimeError("contract-adapted audit failed: candidate budget drift")
        parent_prompt = private["case"]["parent_prompts"][int(private["case"]["target_member"])]
        for candidate in private["candidates"]:
            candidate_count += 1
            hypothesis = parse_edit_hypothesis(json.dumps(candidate["hypothesis"]))
            expected_prompt = render_contract_adapted_prompt(parent_prompt, hypothesis)
            if candidate["prompt"] != expected_prompt:
                raise RuntimeError("contract-adapted audit failed: renderer replay mismatch")
            hard, reason = base._hard_gate(candidate["prompt"], private["case"]["questions"])
            if not hard or reason or not candidate["schema_valid"]:
                raise RuntimeError("contract-adapted audit failed: candidate contract invalid")
            if candidate["raw_reflection_embedded"]:
                raise RuntimeError("contract-adapted audit failed: raw reflection entered candidate")
            if candidate["renderer_vocabulary_hash"] != renderer_vocabulary_identity():
                raise RuntimeError("contract-adapted audit failed: renderer vocabulary drift")
    if candidate_count != 12:
        raise RuntimeError("contract-adapted audit failed: expected 12 candidates")

    ledger = [
        json.loads(line)
        for line in (run / "api_ledger_private.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in ledger:
        validate_ledger_record(row)
    if len({str(row["provider_attempt_id"]) for row in ledger}) != len(ledger):
        raise RuntimeError("contract-adapted audit failed: duplicate ledger identity")
    full_team_cases = {
        str(row["parent_id"]) for row in ledger if row["evaluation_stage"] == "full_team"
    }
    if full_team_cases != expected_cases:
        raise RuntimeError("contract-adapted audit failed: parent coverage incomplete")
    parent_reuse = [row for row in ledger if row["evaluation_stage"] == "minibatch_parent"]
    if len(parent_reuse) != 72 or any(not row["cache_hit"] for row in parent_reuse):
        raise RuntimeError("contract-adapted audit failed: minibatch parent reuse incomplete")

    reflection = [row for row in ledger if row["evaluation_stage"] == "reflection"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reflection:
        if row["client_role"] != "optimizer" or row["logical_role"] != "reflection":
            raise RuntimeError("contract-adapted audit failed: reflection role drift")
        grouped[(str(row["parent_id"]), str(row["candidate_id"]))].append(row)
    expected_reflections = {
        (case_id, candidate_id)
        for case_id in expected_cases
        for candidate_id in ("B_PRIME_0", "B_PRIME_1")
    }
    if set(grouped) != expected_reflections:
        raise RuntimeError("contract-adapted audit failed: reflection coverage incomplete")
    for key, attempts in grouped.items():
        ordered = sorted(attempts, key=lambda row: int(row["attempt_index"]))
        if sum(bool(row["success"]) for row in ordered) != 1 or not ordered[-1]["success"]:
            raise RuntimeError(f"contract-adapted audit failed: reflection completion: {key}")

    runtime: dict[str, int] = defaultdict(int)
    for result in results:
        for key, value in result["runtime_accounting"].items():
            runtime[key] += int(value)
    if dict(runtime) != base._durable_accounting(ledger):
        raise RuntimeError("contract-adapted audit failed: runtime/durable accounting mismatch")
    return {
        "status": "PASS",
        "case_count": 6,
        "candidate_count": 12,
        "schema_valid_candidates": 12,
        "contract_valid_candidates": 12,
        "actual_commits": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "reflection_logical_calls": len(grouped),
        "reflection_provider_attempts": len(reflection),
        "reflection_successful_calls": sum(bool(row["success"]) for row in reflection),
        "ledger_records": len(ledger),
        "runtime_durable_reconciliation": "PASS",
        "renderer_replay": "PASS",
    }


def report(run: Path, report_root: Path, prep: Path) -> None:
    if report_root.exists():
        raise FileExistsError("fresh sanitized report root required")
    payload = _read(run / "result_sanitized.json")
    results = payload["results"]
    ledger = [
        json.loads(line)
        for line in (run / "api_ledger_private.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = _summary(results, ledger)
    gate = audit(run, prep)
    report_root.mkdir(parents=True)
    _write(report_root / "protocol.json", {
        **_read(prep / "PRE_API_FREEZE.json"),
        "solver_contract_identity": contract_identity(),
        "actual_commits": 0,
        "validation_calls": 0,
        "test_calls": 0,
    })
    _jsonl(
        report_root / "candidate_manifest.jsonl",
        [candidate for result in results for candidate in result["candidates"]],
    )
    with (report_root / "candidate_eval_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "case_id", "candidate_id", "hypothesis_hash", "schema_valid",
            "hard_gate_passed", "promoted", "feasible", "mini", "full",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for candidate in result["candidates"]:
                writer.writerow({
                    "case_id": result["case_id"],
                    **{
                        key: json.dumps(candidate[key], sort_keys=True)
                        if isinstance(candidate.get(key), dict)
                        else candidate.get(key, "")
                        for key in fields[1:]
                    },
                })
    _write(report_root / "proposal_quality.json", summary["arms"])
    _write(report_root / "cost_by_stage.json", summary["cost_by_stage"])
    _write(report_root / "selector_disagreements.json", [
        {
            "case_id": result["case_id"],
            "B0_PRIME": result["winners"]["B0_PRIME"],
            "B1_PRIME": result["winners"]["B1_PRIME"],
            "frontier": result["b1_prime_frontier"],
        }
        for result in results
    ])
    _write(report_root / "fact_assertions.json", gate)
    _write(report_root / "api_ledger_summary.json", {
        "api_calls": summary["api_calls"],
        "validation_calls": 0,
        "test_calls": 0,
        "ledger_records": len(ledger),
    })
    _write(report_root / "provenance.json", {
        "source": "six frozen retry4 parents and minibatches",
        "raw_prompts_questions_responses": "excluded",
        "historical_artifacts_modified": 0,
    })
    arms = summary["arms"]
    (report_root / "README.md").write_text(
        "# Contract-adapted RG-GEPA fixed-parent pilot v1\n\n"
        "The reflection model selected closed-vocabulary edit hypotheses. A deterministic renderer "
        "created every mutable candidate; raw reflection text never entered a Solver prompt. The six "
        "parents, responsibility evidence, two-candidate budget, progressive evaluation, and shared-pool "
        "current/Pareto comparison were frozen before execution. No prompt was committed and Validation "
        "and Test were not accessed.\n\n"
        "| Arm | Generated | Schema valid | Contract valid | Promoted | Full eval | Feasible | Would commit |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['arm']} | {row['generated']} | {row['schema_valid']} | {row['contract_valid']} | "
            f"{row['promoted']} | {row['full_evaluated']} | {row['feasible']} | {row['would_commit']} |"
            for row in arms
        )
        + "\n",
        encoding="utf-8",
    )
    forbidden = ("FINAL_ANSWER:", "api_key", "dashscope", "https://", "D:\\\\")
    hashes = []
    for path in sorted(report_root.iterdir()):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore").lower()
        if any(item.lower() in text for item in forbidden):
            raise RuntimeError(f"sanitization failure: {path.name}")
        hashes.append({"path": path.name, "sha256": _sha(data), "bytes": len(data)})
    _write(report_root / "sanitization_manifest.json", {"status": "PASS", "files": len(hashes)})
    _write(report_root / "sha256_manifest.json", {"files": hashes})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--authorize-api", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.prep = args.prep.resolve()
    args.run = args.run.resolve()
    args.report = args.report.resolve()
    if args.audit:
        print(json.dumps(audit(args.run, args.prep), sort_keys=True))
    elif args.report_only:
        report(args.run, args.report, args.prep)
        print(json.dumps({"status": "PASS", "report": str(args.report)}, sort_keys=True))
    else:
        asyncio.run(execute(args))


if __name__ == "__main__":
    main()
