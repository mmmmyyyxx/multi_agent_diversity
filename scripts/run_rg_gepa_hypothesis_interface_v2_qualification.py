"""Isolated 12-call schema qualification for RG-GEPA Hypothesis Interface V2.

This is engineering evidence only.  It never calls a Solver, evaluates a
dataset, renders a scientific candidate, or accesses Validation/Test data.
"""
from __future__ import annotations

import argparse
import asyncio
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

from multi_dataset_diverse_rl.config import Config  # noqa: E402
from multi_dataset_diverse_rl.experimental_contract_adapted_rg_gepa import (  # noqa: E402
    SchemaQualificationV2Protocol,
    build_hypothesis_request_v2,
    hypothesis_interface_v2_identity,
    parse_hypothesis_selection_v2,
)
from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient  # noqa: E402


AUTH_ENV = "RG_GEPA_HYPOTHESIS_V2_QUALIFICATION_AUTHORIZED"
DEFAULT_PREP = ROOT / "runs" / "rg_gepa_hypothesis_interface_v2_qualification_prep_20260910"
DEFAULT_RUN = ROOT / "runs" / "rg_gepa_hypothesis_interface_v2_qualification_20260910"
DEFAULT_REPORT = ROOT / "reports" / "rg_gepa_hypothesis_interface_v2_qualification_execution"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def qualification_cases() -> list[dict[str, Any]]:
    lanes = (
        ("coverage", "one uncovered responsibility and one broad-behavior preservation concern"),
        ("margin_support", "one near-margin responsibility and one consistency preservation concern"),
        ("direct_flip", "one direct-flip responsibility and one unsupported-assumption concern"),
    )
    return [
        {
            "qualification_id": f"Q{index + 1:02d}",
            "lane": lane,
            "symbolic_evidence": evidence,
            "mutation_index": index % 2,
        }
        for index in range(12)
        for lane, evidence in (lanes[index % len(lanes)],)
    ]


def build_request(case: dict[str, Any]) -> tuple[str, str]:
    return build_hypothesis_request_v2(
        responsibility_lane=str(case["lane"]),
        context_lines=(f"Symbolic evidence: {case['symbolic_evidence']}",),
        mutation_index=int(case["mutation_index"]),
    )


class QualificationLedger:
    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError("qualification ledger must be fresh")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.attempt_ids: set[str] = set()

    def append(self, row: dict[str, Any]) -> None:
        required = {
            "qualification_id", "logical_call_id", "provider_attempt_id", "attempt_index",
            "model", "client_role", "logical_role", "success", "input_tokens",
            "output_tokens", "total_tokens",
        }
        if set(row) != required:
            raise ValueError("qualification ledger schema mismatch")
        if row["client_role"] != "optimizer" or row["logical_role"] != "schema_qualification":
            raise ValueError("qualification ledger role mismatch")
        if int(row["input_tokens"]) + int(row["output_tokens"]) != int(row["total_tokens"]):
            raise ValueError("qualification ledger token mismatch")
        identity = str(row["provider_attempt_id"])
        if not identity or identity in self.attempt_ids:
            raise ValueError("qualification ledger duplicate attempt identity")
        self.attempt_ids.add(identity)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _persist_calls(
    client: RoleAwareLLMClient,
    begin: int,
    case: dict[str, Any],
    ledger: QualificationLedger,
) -> None:
    logical_id = str(case["qualification_id"])
    for offset, call in enumerate(client.calls[begin:], start=1):
        if call.get("client_role") != "optimizer":
            raise RuntimeError("non-optimizer call occurred during schema qualification")
        attempt = int(call.get("attempt", offset))
        row = {
            "qualification_id": logical_id,
            "logical_call_id": logical_id,
            "provider_attempt_id": f"{logical_id}:attempt:{attempt}",
            "attempt_index": attempt,
            "model": str(call.get("model", "")),
            "client_role": "optimizer",
            "logical_role": "schema_qualification",
            "success": bool(call.get("success", False)),
            "input_tokens": int(call.get("prompt_tokens", 0)),
            "output_tokens": int(call.get("completion_tokens", 0)),
            "total_tokens": int(call.get("total_tokens", 0)),
        }
        ledger.append(row)


def _cfg(run: Path) -> Config:
    return Config.from_flat(
        optimizer_model="qwen3.7-flash",
        evaluator_model="qwen3.7-flash",
        agent_model="qwen3-8b",
        out_dir=str(run),
        provider_call_budget=100,
        total_token_budget=1_000_000,
        final_test_enabled=False,
    )


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol = SchemaQualificationV2Protocol()
    if not args.authorize_api or os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"--authorize-api and {AUTH_ENV}=1 are required")
    if args.run.exists() or args.report.exists():
        raise FileExistsError("qualification run/report roots must be fresh")
    registry = _read(args.prep / "private_registry.json")
    if (
        registry["protocol_hash"] != protocol.identity()
        or registry["interface_hash"] != hypothesis_interface_v2_identity()
        or registry["cases"] != qualification_cases()
    ):
        raise RuntimeError("qualification freeze mismatch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", registry["execution_commit"], "HEAD"], cwd=ROOT
    ).returncode:
        raise RuntimeError("qualification execution source is not an ancestor of HEAD")

    args.run.mkdir(parents=True)
    client = RoleAwareLLMClient(_cfg(args.run))
    ledger = QualificationLedger(args.run / "api_ledger_private.jsonl")
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for case in registry["cases"]:
        system_prompt, user_prompt = build_request(case)
        begin = len(client.calls)
        result = None
        try:
            result = await client.chat_result(
                protocol.optimizer_model,
                system_prompt,
                user_prompt,
                protocol.temperature,
                protocol.max_tokens,
                "optimizer",
                "schema_qualification",
            )
        finally:
            _persist_calls(client, begin, case, ledger)
        assert result is not None
        selection = None
        error = ""
        try:
            selection = parse_hypothesis_selection_v2(result.text)
        except ValueError as exc:
            error = str(exc)
        public_rows.append({
            "qualification_id": case["qualification_id"],
            "response_hash": _sha(result.text.encode("utf-8")),
            "schema_valid": selection is not None,
            "selection_hash": selection.identity() if selection is not None else None,
            "parse_error_category": error,
            "finish_reason": result.finish_reason,
        })
        private_rows.append({
            "qualification_id": case["qualification_id"],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": result.text,
        })
        _write(args.run / "progress.json", {
            "completed": len(public_rows),
            "total": protocol.request_count,
            "schema_valid": sum(row["schema_valid"] for row in public_rows),
            "solver_calls": 0,
            "validation_calls": 0,
            "test_calls": 0,
        })
    _write(args.run / "responses_private.json", private_rows)
    passed = sum(row["schema_valid"] for row in public_rows) == protocol.request_count
    summary = {
        "status": "PASS" if passed else "HOLD",
        "qualification_calls": protocol.request_count,
        "schema_valid": sum(row["schema_valid"] for row in public_rows),
        "schema_invalid": sum(not row["schema_valid"] for row in public_rows),
        "solver_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "scientific_evidence_eligible": False,
    }
    _write(args.run / "result_sanitized.json", {"summary": summary, "rows": public_rows})
    return summary


def audit(run: Path, prep: Path) -> dict[str, Any]:
    protocol = SchemaQualificationV2Protocol()
    payload = _read(run / "result_sanitized.json")
    rows = payload["rows"]
    ledger_rows = [
        json.loads(line)
        for line in (run / "api_ledger_private.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    registry = _read(prep / "private_registry.json")
    if registry["protocol_hash"] != protocol.identity() or registry["cases"] != qualification_cases():
        raise RuntimeError("qualification audit freeze mismatch")
    if len(rows) != 12 or {row["qualification_id"] for row in rows} != {
        case["qualification_id"] for case in qualification_cases()
    }:
        raise RuntimeError("qualification audit logical coverage mismatch")
    if any(row["selection_hash"] is not None for row in rows if not row["schema_valid"]):
        raise RuntimeError("invalid qualification row carries a selection")
    grouped: dict[str, list[dict[str, Any]]] = {}
    attempt_ids: set[str] = set()
    for row in ledger_rows:
        identity = str(row["provider_attempt_id"])
        if identity in attempt_ids:
            raise RuntimeError("qualification audit duplicate attempt identity")
        attempt_ids.add(identity)
        if row["client_role"] != "optimizer" or row["logical_role"] != "schema_qualification":
            raise RuntimeError("qualification audit role mismatch")
        if int(row["input_tokens"]) + int(row["output_tokens"]) != int(row["total_tokens"]):
            raise RuntimeError("qualification audit token mismatch")
        grouped.setdefault(str(row["logical_call_id"]), []).append(row)
    if set(grouped) != {case["qualification_id"] for case in qualification_cases()}:
        raise RuntimeError("qualification audit ledger coverage mismatch")
    for logical_id, attempts in grouped.items():
        ordered = sorted(attempts, key=lambda row: int(row["attempt_index"]))
        if sum(bool(row["success"]) for row in ordered) != 1 or not ordered[-1]["success"]:
            raise RuntimeError(f"qualification audit completion mismatch: {logical_id}")
    valid = sum(bool(row["schema_valid"]) for row in rows)
    return {
        "status": "PASS" if valid == 12 else "HOLD",
        "qualification_calls": 12,
        "schema_valid": valid,
        "schema_invalid": 12 - valid,
        "provider_attempts": len(ledger_rows),
        "successful_provider_calls": sum(bool(row["success"]) for row in ledger_rows),
        "failed_provider_attempts": sum(not bool(row["success"]) for row in ledger_rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in ledger_rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in ledger_rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in ledger_rows),
        "solver_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "scientific_evidence_eligible": False,
    }


def report(run: Path, report_root: Path, prep: Path) -> None:
    if report_root.exists():
        raise FileExistsError("qualification report root must be fresh")
    payload = _read(run / "result_sanitized.json")
    gate = audit(run, prep)
    report_root.mkdir(parents=True)
    _write(report_root / "qualification_results.json", payload["rows"])
    _write(report_root / "fact_assertions.json", gate)
    _write(report_root / "classifier.json", {
        "status": gate["status"],
        "pass_rule": "12_OF_12_STRICT_SCHEMA_VALID",
        "fixed_parent_pilot_authorized": False,
        "scientific_claim": "NONE_ENGINEERING_QUALIFICATION_ONLY",
    })
    _write(report_root / "provenance.json", {
        "protocol_hash": _read(prep / "private_registry.json")["protocol_hash"],
        "interface_hash": hypothesis_interface_v2_identity(),
        "raw_responses": "excluded",
        "dataset_evidence_used": False,
    })
    (report_root / "README.md").write_text(
        "# RG-GEPA Hypothesis Interface V2 schema qualification\n\n"
        "This is an isolated engineering qualification, not scientific evidence. Twelve qwen3.7-flash "
        "requests used synthetic symbolic contexts and an exact four-position enum-ID array. No cleanup, "
        "alias mapping, schema retry, Solver, candidate rollout, Validation, Test, or experiment parent was "
        "used. PASS requires 12/12 strict parser acceptance. A PASS does not itself authorize the fixed-parent pilot.\n",
        encoding="utf-8",
    )
    forbidden = ("FINAL_ANSWER:", "api_key", "dashscope", "https://", "D:\\\\")
    hashes = []
    for path in sorted(report_root.iterdir()):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore").lower()
        if any(item.lower() in text for item in forbidden):
            raise RuntimeError(f"qualification sanitization failure: {path.name}")
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
        result = asyncio.run(execute(args))
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
