"""Run the frozen six-parent contract-adapted RG-GEPA V2 pilot.

V1 remains untouched for historical replay. V2 shares its wire builder with the
qualified interface, renders only program-owned prompt text, evaluates all 12
candidates for proposal-quality audit, and keeps selection restricted to the
pre-full-evaluation progressive promotion set.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.candidate_selection import (  # noqa: E402
    common_monotone_safe_key,
    evaluate_constraints,
)
from multi_dataset_diverse_rl.experimental_contract_adapted_rg_gepa import (  # noqa: E402
    ContractAdaptedProtocolV2,
    HypothesisSelectionV2,
    build_hypothesis_request_v2,
    hypothesis_interface_v2_identity,
    parse_hypothesis_selection_v2,
    render_contract_adapted_prompt_v2,
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


AUTH_ENV = "CONTRACT_ADAPTED_RG_GEPA_V2_PILOT_AUTHORIZED"
DEFAULT_PREP = ROOT / "runs" / "contract_adapted_rg_gepa_fixed_parent_pilot_v2_prep_20260910"
DEFAULT_RUN = ROOT / "runs" / "contract_adapted_rg_gepa_fixed_parent_pilot_v2_20260910"
DEFAULT_REPORT = ROOT / "reports" / "contract_adapted_rg_gepa_fixed_parent_pilot_v2_execution"
PROPOSAL_OPERATOR_VERSION = "contract_adapted_rg_gepa_v2"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _protocol_payload() -> dict[str, Any]:
    protocol = ContractAdaptedProtocolV2()
    return {
        **json.loads(json.dumps(asdict(protocol), sort_keys=True)),
        "hypothesis_interface_hash": protocol.hypothesis_interface_hash,
        "renderer_vocabulary_hash": protocol.renderer_vocabulary_hash,
    }


def _verify_source_freeze(prep: Path) -> None:
    freeze = _read(prep / "PRE_API_FREEZE.json")
    for row in freeze["source_files"]:
        path = ROOT / str(row["path"])
        if not path.is_file() or _sha(path.read_bytes()) != str(row["sha256"]):
            raise RuntimeError(f"V2 source freeze mismatch: {row['path']}")


class V2ExecutionLedger(base.RGGEPAExecutionLedger):
    def append(self, record: dict[str, Any]) -> None:
        enriched = {**record, "proposal_operator_version": PROPOSAL_OPERATOR_VERSION}
        super().append(enriched)


def _meta(case: dict[str, Any], candidate_id: str, stage: str) -> dict[str, Any]:
    return {
        "seed": case["source_seed"],
        "parent_id": case["case_id"],
        "update_index": case["source_update_index"],
        "candidate_id": candidate_id,
        "proposal_engine": "gepa_reflection",
        "evaluation_stage": stage,
    }


def _scientific_context(case: dict[str, Any]) -> tuple[str, ...]:
    target = int(case["target_member"])
    profile = {str(row["question_hash"]): row for row in case["parent_profile"]}
    lines = [f"Parent procedure:\n{case['parent_prompts'][target]}", "Observed responsibility cases:"]
    for offset, evidence in enumerate(case["minibatch_private"][:4], start=1):
        state = profile[str(evidence["example_id"])]
        lines.append(
            f"Case {offset}: question={evidence['question']} | gold={evidence['answer']} | "
            f"target_prediction={state['team_answers'][target]} | "
            f"target_correct={bool(state['team_correctness'][target])} | "
            f"source={evidence['source_type']}"
        )
    return tuple(lines)


def _request(case: dict[str, Any], candidate_index: int) -> tuple[str, str]:
    return build_hypothesis_request_v2(
        responsibility_lane=str(case["responsibility_type"]),
        context_lines=_scientific_context(case),
        mutation_index=candidate_index,
    )


def _winner(rows: list[dict[str, Any]], *, pareto: bool) -> tuple[str | None, list[str]]:
    if pareto:
        got, frontier = team_pareto_winner([
            CandidateScore(row["candidate_id"], TeamVector(**row["full"]), 0, True)
            for row in rows
        ])
        return (got.candidate_id if got else None, [row.candidate_id for row in frontier])
    got = max(rows, key=lambda row: tuple(row["current_selector_key"]), default=None)
    return (got["candidate_id"] if got else None, [])


async def _run_case(
    case: dict[str, Any], root: Path, ledger: V2ExecutionLedger
) -> dict[str, Any]:
    case_root = root / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=False)
    protocol = ContractAdaptedProtocolV2()
    system = base._CommonSystem(base._cfg(case, case_root), ledger_writer=ledger)
    system.set_solver_stage(_meta(case, "parent", "full_team"))
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
    mini_ids = [row["example_id"] for row in case["minibatch_private"]]
    by_hash = {row.question_hash: index for index, row in enumerate(system.fixed_probe.examples)}
    mini_indices = [by_hash[example_id] for example_id in mini_ids]
    parent_mini = base._evaluation(
        system,
        target,
        case["parent_prompts"][target],
        tuple(parent_profile[index] for index in mini_indices),
        mini_indices,
        assigned,
    )
    base._record_minibatch_parent_reuse(case, ledger)

    candidates: list[dict[str, Any]] = []
    for index in range(2):
        candidate_id = f"B_V2_{index}"
        system_prompt, user_prompt = _request(case, index)
        begin = len(system.llm.calls)
        try:
            response = await system.llm.chat_result(
                protocol.optimizer_model,
                system_prompt,
                user_prompt,
                protocol.reflection_temperature,
                protocol.reflection_max_tokens,
                "optimizer",
                protocol.reflection_role,
            )
        finally:
            base._persist_new_optimizer_calls(system, begin, _meta(case, candidate_id, "reflection"), ledger)
        try:
            selection = parse_hypothesis_selection_v2(response.text)
            prompt = render_contract_adapted_prompt_v2(case["parent_prompts"][target], selection)
        except Exception as exc:
            _write(case_root / f"{candidate_id}_failure_private.json", {
                "candidate_id": candidate_id,
                "raw_response": response.text,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise RuntimeError(f"V2 reflection failed before candidate generation: {candidate_id}") from exc
        hard, reason = base._hard_gate(prompt, case["questions"])
        if not hard:
            raise RuntimeError("V2 deterministic renderer produced a hard-gate-invalid candidate")
        mini_profile = await base._profile(
            system, target, prompt, mini_indices, _meta(case, candidate_id, "minibatch_candidate")
        )
        mini_eval = base._evaluation(system, target, prompt, mini_profile, mini_indices, assigned)
        candidates.append({
            "candidate_id": candidate_id,
            "candidate_index": index,
            "proposal_engine": "contract_adapted_reflection_v2",
            "proposal_operator_version": PROPOSAL_OPERATOR_VERSION,
            "prompt": prompt,
            "prompt_hash": _sha(prompt.encode("utf-8")),
            "reflection_response_hash": _sha(response.text.encode("utf-8")),
            "selection": [selection.failure_id, selection.edit_id, selection.preserve_id, selection.avoid_id],
            "selection_hash": selection.identity(),
            "schema_valid": True,
            "hard_gate_passed": True,
            "hard_gate_reason": "",
            "raw_reflection_embedded": False,
            "mini": asdict(base._vector(mini_eval, parent_mini)),
        })

    # Promotion is frozen before any full result exists.
    promoted = {
        row.candidate_id
        for row in progressive_promotions(
            parent=base._vector(parent_mini),
            candidates=[
                CandidateScore(row["candidate_id"], TeamVector(**row["mini"]), 0, True)
                for row in candidates
            ],
        )
    }
    for row in candidates:
        row["promoted"] = row["candidate_id"] in promoted
        row["progressive_promoted"] = row["promoted"]

    # Every candidate is evaluated exactly once on the full fixed probe. Results
    # from non-promoted candidates are audit-only and cannot affect the winners.
    for row in candidates:
        stage = "full_member" if row["promoted"] else "audit_only"
        full_profile = await base._profile(
            system, target, row["prompt"], None, _meta(case, row["candidate_id"], stage)
        )
        full = base._evaluation(system, target, row["prompt"], full_profile, None, assigned)
        constraint = evaluate_constraints(full, parent)
        row["full"] = asdict(base._vector(full, parent))
        row["feasible"] = bool(constraint.passed)
        row["constraint_reasons"] = list(constraint.rejection_reasons)
        row["current_selector_key"] = list(
            common_monotone_safe_key(full, int(row["candidate_index"]))
        )
        row["full_evaluation_role"] = "operational" if row["promoted"] else "audit_only"

    operational = [row for row in candidates if row["promoted"] and row["feasible"]]
    all_feasible = [row for row in candidates if row["feasible"]]
    b0, _ = _winner(operational, pareto=False)
    b1, frontier = _winner(operational, pareto=True)
    audit_best, _ = _winner(all_feasible, pareto=False)
    runtime = base._runtime_accounting(system)
    private = {
        "case": case,
        "parent": asdict(base._vector(parent)),
        "parent_mini": asdict(base._vector(parent_mini)),
        "candidates": candidates,
        "promotion_ids_frozen_before_full": sorted(promoted),
        "winners": {"B0_PRIME": b0, "B1_PRIME": b1},
        "b1_prime_frontier": frontier,
        "all_candidate_audit_best": audit_best,
        "actual_commits": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "protocol": _protocol_payload(),
        "runtime_accounting": runtime,
    }
    _write(case_root / "private_result.json", private)
    return {
        "case_id": case["case_id"],
        "seed": case["source_seed"],
        "responsibility_type": case["responsibility_type"],
        "parent": private["parent"],
        "candidates": [base._public_candidate(row) for row in candidates],
        "promotion_ids_frozen_before_full": sorted(promoted),
        "winners": private["winners"],
        "b1_prime_frontier": frontier,
        "all_candidate_audit_best": audit_best,
        "actual_commits": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "runtime_accounting": runtime,
    }


def summarize(results: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for result in results for row in result["candidates"]]
    promoted = [row for row in candidates if row["promoted"]]
    feasible = [row for row in candidates if row["feasible"]]
    promoted_feasible = [row for row in promoted if row["feasible"]]
    selector_differences = sum(
        result["winners"]["B0_PRIME"] != result["winners"]["B1_PRIME"]
        for result in results
    )
    if len(feasible) <= 1:
        proposal_label = "ADAPTED_PROPOSAL_QUALITY_POOR"
    elif selector_differences:
        proposal_label = "PARETO_SELECTION_DIFFERENTIATES"
    else:
        proposal_label = "PROPOSAL_SIGNAL_PARETO_NOT_NEEDED"
    return {
        "status": "PASS",
        "generated": len(candidates),
        "schema_valid": sum(bool(row["schema_valid"]) for row in candidates),
        "contract_valid": sum(bool(row["hard_gate_passed"]) for row in candidates),
        "promoted": len(promoted),
        "full_evaluated": sum("full" in row for row in candidates),
        "true_feasible_yield": len(feasible),
        "promoted_feasible": len(promoted_feasible),
        "missed_feasible_by_progressive": len([row for row in feasible if not row["promoted"]]),
        "progressive_feasible_recall": (
            len(promoted_feasible) / len(feasible) if feasible else None
        ),
        "B0_prime_would_commit": sum(result["winners"]["B0_PRIME"] is not None for result in results),
        "B1_prime_would_commit": sum(result["winners"]["B1_PRIME"] is not None for result in results),
        "selector_difference_cases": selector_differences,
        "frozen_local_classifier": proposal_label,
        "validation_superiority_claimed": False,
        "api_calls": sum(int(row["successful_provider_calls"]) for row in ledger),
        "provider_attempts": sum(int(row["provider_attempts"]) for row in ledger),
        "input_tokens": sum(int(row["input_tokens"]) for row in ledger),
        "output_tokens": sum(int(row["output_tokens"]) for row in ledger),
        "total_tokens": sum(int(row["total_tokens"]) for row in ledger),
        "actual_commits": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not args.authorize_api or os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"--authorize-api and {AUTH_ENV}=1 are required")
    if args.run.exists() or args.report.exists():
        raise FileExistsError("V2 execution roots must be fresh")
    _verify_source_freeze(args.prep)
    registry = _read(args.prep / "private_registry.json")
    protocol = ContractAdaptedProtocolV2()
    if (
        registry["protocol"] != _protocol_payload()
        or registry["protocol_hash"] != protocol.identity()
        or registry["hypothesis_interface_hash"] != hypothesis_interface_v2_identity()
        or registry["renderer_vocabulary_hash"] != renderer_vocabulary_identity()
    ):
        raise RuntimeError("V2 frozen protocol/interface/renderer mismatch")
    # Fresh CLI execution gets a fresh process-wide cache. Explicit reset also
    # makes repeated imported execution fail-safe rather than inherit a prior run.
    base._CommonSystem.raw_cache = {}
    args.run.mkdir(parents=True)
    ledger = V2ExecutionLedger(args.run / "api_ledger_private.jsonl")
    results: list[dict[str, Any]] = []
    for case in registry["cases"]:
        case["minibatch_private"] = [
            next(row for row in case["questions"] if row["example_id"] == item["example_id"])
            | {"source_type": item["source_type"]}
            for item in registry["minibatches"][case["case_id"]]
        ]
        results.append(await _run_case(case, args.run, ledger))
        _write(args.run / "progress.json", {
            "completed_cases": len(results),
            "total_cases": 6,
            "validation_calls": 0,
            "test_calls": 0,
        })
    summary = summarize(results, ledger.rows)
    _write(args.run / "result_sanitized.json", {"results": results, "summary": summary})
    return summary


def audit(run: Path, prep: Path) -> dict[str, Any]:
    _verify_source_freeze(prep)
    registry = _read(prep / "private_registry.json")
    payload = _read(run / "result_sanitized.json")
    results = payload["results"]
    if len(results) != 6 or {row["case_id"] for row in results} != {
        row["case_id"] for row in registry["cases"]
    }:
        raise RuntimeError("V2 audit case identity mismatch")
    candidate_count = 0
    for result in results:
        private = _read(run / result["case_id"] / "private_result.json")
        if len(private["candidates"]) != 2:
            raise RuntimeError("V2 audit candidate budget mismatch")
        parent_prompt = private["case"]["parent_prompts"][int(private["case"]["target_member"])]
        for row in private["candidates"]:
            candidate_count += 1
            selection = HypothesisSelectionV2(*row["selection"])
            expected = render_contract_adapted_prompt_v2(parent_prompt, selection)
            if row["prompt"] != expected or row["raw_reflection_embedded"]:
                raise RuntimeError("V2 audit deterministic rendering mismatch")
            hard, reason = base._hard_gate(row["prompt"], private["case"]["questions"])
            if not hard or reason or not row["schema_valid"]:
                raise RuntimeError("V2 audit contract validity mismatch")
            if "full" not in row:
                raise RuntimeError("V2 audit full proposal evidence incomplete")
            if row["promoted"] != (row["candidate_id"] in private["promotion_ids_frozen_before_full"]):
                raise RuntimeError("V2 audit promotion identity mismatch")
    if candidate_count != 12:
        raise RuntimeError("V2 audit expected 12 candidates")
    if any(
        row["actual_commits"] or row["validation_calls"] or row["test_calls"]
        for row in results
    ):
        raise RuntimeError("V2 audit isolation mismatch")

    ledger = [
        json.loads(line)
        for line in (run / "api_ledger_private.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in ledger:
        validate_ledger_record(row)
        if row.get("proposal_operator_version") != PROPOSAL_OPERATOR_VERSION:
            raise RuntimeError("V2 audit proposal operator provenance mismatch")
    if len({row["provider_attempt_id"] for row in ledger}) != len(ledger):
        raise RuntimeError("V2 audit duplicate ledger identity")
    reflections = [row for row in ledger if row["evaluation_stage"] == "reflection"]
    if len({row["logical_call_id"] for row in reflections}) != 12:
        raise RuntimeError("V2 audit reflection logical coverage mismatch")
    if sum(bool(row["success"]) for row in reflections) != 12:
        raise RuntimeError("V2 audit reflection success mismatch")
    runtime: dict[str, int] = {}
    for result in results:
        for key, value in result["runtime_accounting"].items():
            runtime[key] = runtime.get(key, 0) + int(value)
    if runtime != base._durable_accounting(ledger):
        raise RuntimeError("V2 audit runtime/durable accounting mismatch")
    summary = summarize(results, ledger)
    if summary["generated"] != 12 or summary["contract_valid"] != 12 or summary["full_evaluated"] != 12:
        raise RuntimeError("V2 audit proposal evidence gate mismatch")
    return {**summary, "source_freeze": "PASS", "renderer_replay": "PASS", "ledger_records": len(ledger)}


def report(run: Path, report_root: Path, prep: Path) -> None:
    if report_root.exists():
        raise FileExistsError("V2 report root must be fresh")
    payload = _read(run / "result_sanitized.json")
    gate = audit(run, prep)
    report_root.mkdir(parents=True)
    _write(report_root / "fact_assertions.json", gate)
    _write(report_root / "scientific_summary.json", {
        key: gate[key]
        for key in (
            "generated", "schema_valid", "contract_valid", "promoted", "full_evaluated",
            "true_feasible_yield", "promoted_feasible", "missed_feasible_by_progressive",
            "progressive_feasible_recall", "B0_prime_would_commit", "B1_prime_would_commit",
            "selector_difference_cases", "frozen_local_classifier",
        )
    })
    _write(report_root / "candidate_manifest.json", [
        candidate for result in payload["results"] for candidate in result["candidates"]
    ])
    _write(report_root / "selector_comparison.json", [
        {
            "case_id": result["case_id"],
            "B0_PRIME": result["winners"]["B0_PRIME"],
            "B1_PRIME": result["winners"]["B1_PRIME"],
            "frontier": result["b1_prime_frontier"],
        }
        for result in payload["results"]
    ])
    _write(report_root / "protocol.json", _read(prep / "PRE_API_FREEZE.json"))
    _write(report_root / "provenance.json", {
        "raw_prompts_questions_answers_responses": "excluded",
        "proposal_operator_version": PROPOSAL_OPERATOR_VERSION,
        "historical_artifacts_modified": 0,
    })
    (report_root / "README.md").write_text(
        "# Contract-adapted RG-GEPA fixed-parent pilot V2\n\n"
        "Six frozen parents received two V2 closed-action reflection candidates each. All 12 candidates "
        "were fully evaluated exactly once for proposal-quality audit. Progressive promotion was frozen "
        "before those full results; only promoted candidates were eligible for B0-prime/B1-prime selection. "
        "Non-promoted results are audit-only and cannot alter promotion or winners. No commit, Validation, "
        "or Test call occurred.\n\n"
        f"Local classifier: `{gate['frozen_local_classifier']}`. This fixed-parent result does not establish "
        "validation superiority or a coalition-aware Pareto claim.\n",
        encoding="utf-8",
    )
    forbidden = ("FINAL_ANSWER:", "api_key", "dashscope", "https://", "D:\\\\")
    hashes = []
    for path in sorted(report_root.iterdir()):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore").lower()
        if any(item.lower() in text for item in forbidden):
            raise RuntimeError(f"V2 report sanitization failure: {path.name}")
        hashes.append({"path": path.name, "sha256": _sha(data), "bytes": len(data)})
    _write(report_root / "sanitization_manifest.json", {"status": "PASS", "files_checked": len(hashes)})
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
        print(json.dumps(asyncio.run(execute(args)), sort_keys=True))


if __name__ == "__main__":
    main()
