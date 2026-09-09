"""Zero-API durability and production-audit tests for the RG-GEPA ledger."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_responsibility_guided_gepa_fixed_parent_pilot.py"


def _load():
    spec = importlib.util.spec_from_file_location("rg_gepa_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base(module, *, parent_id: str, candidate_id: str, stage: str) -> dict:
    return {
        "ledger_version": module.RG_GEPA_LEDGER_VERSION,
        "seed": 76,
        "parent_id": parent_id,
        "update_index": 0,
        "candidate_id": candidate_id,
        "proposal_engine": "gepa_reflection" if stage == "reflection" else "current",
        "evaluation_stage": stage,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_hit": False,
        "logical_role": "solver",
        "client_role": "solver",
        "success": True,
    }


def _solver_row(module, *, parent_id: str, ordinal: int) -> dict:
    row = _base(module, parent_id=parent_id, candidate_id="parent", stage="full_team")
    row.update({
        "provider_attempt_id": f"{parent_id}:full:{ordinal}",
        "logical_call_id": f"{parent_id}:full:{ordinal}",
        "attempt_index": 0,
        "record_kind": "solver_logical_invocation",
        "provider_attempts": 1,
        "successful_provider_calls": 1,
    })
    module.validate_ledger_record(row)
    return row


def _cache_row(module, *, parent_id: str, ordinal: int) -> dict:
    row = _base(module, parent_id=parent_id, candidate_id="parent", stage="minibatch_parent")
    row.update({
        "provider_attempt_id": f"{parent_id}:cache:{ordinal}",
        "logical_call_id": f"{parent_id}:cache:{ordinal}",
        "attempt_index": 0,
        "record_kind": "cache_reuse",
        "provider_attempts": 0,
        "successful_provider_calls": 0,
        "cache_hit": True,
    })
    module.validate_ledger_record(row)
    return row


def _reflection_row(
    module,
    *,
    parent_id: str,
    candidate_id: str,
    attempt: int = 1,
    success: bool = True,
    input_tokens: int = 2,
    output_tokens: int = 3,
) -> dict:
    row = _base(module, parent_id=parent_id, candidate_id=candidate_id, stage="reflection")
    row.update({
        "provider_attempt_id": f"{parent_id}:{candidate_id}:reflection:attempt:{attempt}",
        "logical_call_id": f"{parent_id}:{candidate_id}:reflection",
        "attempt_index": attempt,
        "record_kind": "optimizer_provider_attempt",
        "provider_attempts": 1,
        "successful_provider_calls": int(success),
        "logical_role": "reflection",
        "client_role": "optimizer",
        "success": success,
        "input_tokens": input_tokens if success else 0,
        "output_tokens": output_tokens if success else 0,
        "total_tokens": (input_tokens + output_tokens) if success else 0,
    })
    module.validate_ledger_record(row)
    return row


def _runtime(rows: list[dict]) -> dict[str, int]:
    solver = [row for row in rows if row["record_kind"] == "solver_logical_invocation"]
    optimizer = [row for row in rows if row["record_kind"] == "optimizer_provider_attempt"]
    return {
        "solver_logical_calls": len(solver),
        "solver_provider_attempts": sum(row["provider_attempts"] for row in solver),
        "solver_cache_hits": sum(bool(row["cache_hit"]) for row in solver),
        "solver_input_tokens": sum(row["input_tokens"] for row in solver),
        "solver_output_tokens": sum(row["output_tokens"] for row in solver),
        "optimizer_provider_attempts": len(optimizer),
        "optimizer_successful_attempts": sum(bool(row["success"]) for row in optimizer),
        "optimizer_failed_attempts": sum(not bool(row["success"]) for row in optimizer),
        "optimizer_input_tokens": sum(row["input_tokens"] for row in optimizer),
        "optimizer_output_tokens": sum(row["output_tokens"] for row in optimizer),
    }


def _write_fixture(
    tmp_path: Path,
    module,
    *,
    include_full_team: bool = True,
    include_reflections: bool = True,
    retry: bool = False,
    missing_reflection: tuple[str, str] | None = None,
) -> Path:
    cases = [f"case-{index}" for index in range(6)]
    rows: list[dict] = []
    for index, case in enumerate(cases):
        if include_full_team:
            rows.append(_solver_row(module, parent_id=case, ordinal=index))
        rows.extend(_cache_row(module, parent_id=case, ordinal=offset) for offset in range(12))
        if include_reflections:
            for candidate in ("B_0", "B_1"):
                if missing_reflection == (case, candidate):
                    continue
                if retry and case == "case-0" and candidate == "B_0":
                    rows.append(_reflection_row(module, parent_id=case, candidate_id=candidate, attempt=1, success=False))
                    rows.append(_reflection_row(module, parent_id=case, candidate_id=candidate, attempt=2))
                else:
                    rows.append(_reflection_row(module, parent_id=case, candidate_id=candidate))
    results = [
        {
            "case_id": case,
            "actual_commits": 0,
            "test_calls": 0,
            "candidates": [{}, {}, {}, {}],
            "runtime_accounting": _runtime([row for row in rows if row["parent_id"] == case]),
        }
        for case in cases
    ]
    (tmp_path / "result_sanitized.json").write_text(json.dumps({"results": results}), encoding="utf-8")
    with (tmp_path / "api_ledger_private.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return tmp_path


def test_writer_persists_optimizer_attempt_immediately(tmp_path: Path) -> None:
    module = _load()
    writer = module.RGGEPAExecutionLedger(tmp_path / "ledger.jsonl")
    row = _reflection_row(module, parent_id="case", candidate_id="B_0")
    writer.append(row)
    disk = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
    assert writer.rows == disk == [row]


def test_interrupted_case_keeps_already_persisted_reflection(tmp_path: Path) -> None:
    module = _load()
    writer = module.RGGEPAExecutionLedger(tmp_path / "ledger.jsonl")
    writer.append(_reflection_row(module, parent_id="case", candidate_id="B_0"))
    with pytest.raises(RuntimeError, match="interrupted"):
        raise RuntimeError("interrupted before case completion")
    assert json.loads(writer.path.read_text(encoding="utf-8"))["candidate_id"] == "B_0"


def test_optimizer_helper_persists_retry_attempts_from_fake_provider(tmp_path: Path) -> None:
    module = _load()
    writer = module.RGGEPAExecutionLedger(tmp_path / "ledger.jsonl")
    fake = SimpleNamespace(llm=SimpleNamespace(calls=[
        {"client_role": "optimizer", "role": "reflection", "attempt": 1, "success": False,
         "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"client_role": "optimizer", "role": "reflection", "attempt": 2, "success": True,
         "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    ]))
    module._persist_new_optimizer_calls(fake, 0, {
        "seed": 76, "parent_id": "case", "update_index": 0, "candidate_id": "B_0",
        "proposal_engine": "gepa_reflection", "evaluation_stage": "reflection",
    }, writer)
    disk = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
    assert [row["success"] for row in disk] == [False, True]
    assert [row["attempt_index"] for row in disk] == [1, 2]


def test_profile_clears_solver_stage_after_exception() -> None:
    module = _load()

    class Probe:
        async def evaluate_prompt(self, *_args):
            raise RuntimeError("solver failed")

    class FakeSystem:
        fixed_probe = Probe()
        solve = object()

        def __init__(self):
            self.stages = []

        def set_solver_stage(self, value):
            self.stages.append(value)

        @staticmethod
        def prompt_hash(value):
            return value

    system = FakeSystem()
    with pytest.raises(RuntimeError, match="solver failed"):
        asyncio.run(module._profile(system, 0, "prompt", None, {"evaluation_stage": "full_member"}))
    assert system.stages == [{"evaluation_stage": "full_member"}, None]


def test_audit_requires_parent_full_team_ledger_coverage(tmp_path: Path) -> None:
    module = _load()
    with pytest.raises(RuntimeError, match="parent full-team ledger coverage"):
        module.audit(_write_fixture(tmp_path, module, include_full_team=False))


def test_audit_rejects_zero_reflection_rows(tmp_path: Path) -> None:
    module = _load()
    with pytest.raises(RuntimeError, match="reflection logical-call coverage"):
        module.audit(_write_fixture(tmp_path, module, include_reflections=False))


def test_audit_accepts_twelve_logical_reflections(tmp_path: Path) -> None:
    module = _load()
    result = module.audit(_write_fixture(tmp_path, module))
    assert result["status"] == "PASS"
    assert result["reflection_logical_calls"] == 12
    assert result["reflection_successful_calls"] == 12
    assert result["reflection_provider_attempts"] == 12
    assert result["reflection_failed_attempts"] == 0
    assert result["reflection_total_tokens"] == 60


def test_audit_accepts_retry_and_counts_physical_attempts(tmp_path: Path) -> None:
    module = _load()
    result = module.audit(_write_fixture(tmp_path, module, retry=True))
    assert result["reflection_logical_calls"] == 12
    assert result["reflection_successful_calls"] == 12
    assert result["reflection_provider_attempts"] == 13
    assert result["reflection_failed_attempts"] == 1


def test_audit_rejects_missing_b_candidate_reflection(tmp_path: Path) -> None:
    module = _load()
    with pytest.raises(RuntimeError, match="reflection logical-call coverage"):
        module.audit(_write_fixture(tmp_path, module, missing_reflection=("case-2", "B_1")))


def test_audit_rejects_duplicate_successful_reflection(tmp_path: Path) -> None:
    module = _load()
    run = _write_fixture(tmp_path, module)
    duplicate = _reflection_row(module, parent_id="case-0", candidate_id="B_0", attempt=2)
    with (run / "api_ledger_private.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")
    with pytest.raises(RuntimeError, match="one successful attempt"):
        module.audit(run)


def test_audit_rejects_reflection_whose_final_attempt_failed(tmp_path: Path) -> None:
    module = _load()
    run = _write_fixture(tmp_path, module)
    failed = _reflection_row(
        module, parent_id="case-0", candidate_id="B_0", attempt=2, success=False
    )
    with (run / "api_ledger_private.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failed) + "\n")
    with pytest.raises(RuntimeError, match="final attempt must succeed"):
        module.audit(run)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("client_role", "solver", "reflection role attribution"),
        ("logical_role", "proposal", "reflection role attribution"),
        ("proposal_engine", "current", "reflection engine/record-kind"),
        ("record_kind", "solver_logical_invocation", "solver logical-invocation accounting"),
        ("evaluation_stage", "full_member", "reflection logical-call coverage"),
    ],
)
def test_audit_rejects_reflection_attribution_drift(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    module = _load()
    run = _write_fixture(tmp_path, module)
    ledger_path = run / "api_ledger_private.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    next(row for row in rows if row["evaluation_stage"] == "reflection")[field] = value
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match=message):
        module.audit(run)


def test_token_arithmetic_and_attempt_identity_are_fail_closed(tmp_path: Path) -> None:
    module = _load()
    row = _reflection_row(module, parent_id="case", candidate_id="B_0")
    row["total_tokens"] = 99
    with pytest.raises(ValueError, match="token accounting"):
        module.validate_ledger_record(row)
    writer = module.RGGEPAExecutionLedger(tmp_path / "ledger.jsonl")
    good = _reflection_row(module, parent_id="case", candidate_id="B_0")
    writer.append(good)
    with pytest.raises(ValueError, match="duplicate"):
        writer.append(good)
