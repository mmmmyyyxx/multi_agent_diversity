"""Zero-provider fault injection at the current production boundaries."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.governance.production_execution import (
    ValidatedExecutionContext, admit_execution, validate_local_readiness,
)
from multi_dataset_diverse_rl.persistence.durable_io import read_json
from multi_dataset_diverse_rl.team_search.execution_runtime import CappedDurableLedger
from scripts.audit_online_transfer_diagnostic import audit


def _permit(prep: Path) -> ValidatedExecutionContext:
    return ValidatedExecutionContext(
        experiment_id="gepa_layer2_local_to_team_transfer_diagnostic_v2",
        attempt_id="fake-attempt", execution_source_sha="a" * 40,
        preregistration_sha256="b" * 64, run_identity_sha256="c" * 64,
        provider_profile="lwj", endpoint_fingerprint="fake",
        allowed_phase="diagnostic", allowed_roles=("solver", "reflection"),
        prep_root=prep,
    )


@pytest.mark.parametrize("stale", ["run", "staging"])
def test_stale_local_state_rejected_before_authorization(tmp_path: Path, stale: str) -> None:
    prep = tmp_path / "prep"
    prep.mkdir()
    root = tmp_path / "run"
    (root if stale == "run" else tmp_path / ".run.starting").mkdir()
    with pytest.raises(Exception, match="stale run or staging root"):
        admit_execution(_permit(prep), root)
    assert not (prep / "authorization_consumed.json").exists()


def test_readiness_rehearses_and_cleans_without_consuming(tmp_path: Path) -> None:
    root = tmp_path / "run"
    validate_local_readiness(root)
    assert not root.exists()
    assert not list(tmp_path.glob(".run.readiness-*"))


@pytest.mark.parametrize("fault", [
    "summary_write", "summary_replace", "summary_readback",
    "lifecycle_write", "lifecycle_replace",
])
def test_finalization_fault_never_publishes_complete_without_summary(
    tmp_path: Path, monkeypatch, fault: str,
) -> None:
    import scripts.run_experiment as entry
    import multi_dataset_diverse_rl.governance.production_execution as governance
    import multi_dataset_diverse_rl.persistence.durable_io as filesystem

    prep = tmp_path / "prep"
    prep.mkdir()
    run_root = tmp_path / "run"
    permit = _permit(prep)
    monkeypatch.setattr(entry, "validate_execution", lambda **_: permit)
    async def completed(_permit, *, root):
        return {"ledger": {
            "provider_attempts": 0, "successful_provider_calls": 0,
            "failed_provider_attempts": 0,
        }}
    monkeypatch.setattr(entry, "execute_online_transfer_diagnostic", completed)
    real_write = entry.atomic_write_json
    real_read = entry.read_json
    real_replace = filesystem.os.replace
    fail_count = 0

    def injected_write(path, value):
        nonlocal fail_count
        if fault == "summary_write" and Path(path).parent == run_root and Path(path).name == "execution_summary.json":
            raise OSError("injected summary temp write failure")
        if fault == "lifecycle_write" and Path(path).parent == run_root and Path(path).name == "run_lifecycle.json" and fail_count == 0:
            fail_count += 1
            raise OSError("injected lifecycle temp write failure")
        return real_write(path, value)

    def injected_replace(source, destination):
        nonlocal fail_count
        target = str(destination)
        if fault == "summary_replace" and str(run_root) in target and target.endswith("execution_summary.json"):
            raise OSError("injected summary replace failure")
        if fault == "lifecycle_replace" and str(run_root) in target and target.endswith("run_lifecycle.json") and fail_count == 0:
            fail_count += 1
            raise OSError("injected lifecycle replace failure")
        return real_replace(source, destination)

    def injected_read(path):
        if fault == "summary_readback" and Path(path).parent == run_root and Path(path).name == "execution_summary.json":
            return {"corrupt": True}
        return real_read(path)

    monkeypatch.setattr(entry, "atomic_write_json", injected_write)
    monkeypatch.setattr(entry, "read_json", injected_read)
    monkeypatch.setattr(governance, "atomic_write_json", injected_write)
    monkeypatch.setattr(filesystem.os, "replace", injected_replace)
    with pytest.raises((OSError, RuntimeError)) as failure:
        asyncio.run(entry.execute_frozen(prep, run_root))
    assert run_root.exists(), str(failure.value)
    lifecycle = read_json(run_root / "run_lifecycle.json")
    assert lifecycle["status"] != "EXECUTION_COMPLETE"
    if fault in {"summary_write", "summary_replace"}:
        assert not (run_root / "execution_summary.json").exists()
    else:
        assert read_json(run_root / "execution_summary.json") == {
            "ledger": {"provider_attempts": 0, "successful_provider_calls": 0,
                       "failed_provider_attempts": 0},
        }
    assert lifecycle["status"] == "FAILED_START"


def _row(identity: str, *, success: int, cache: bool = False) -> dict:
    return {
        "record_id": identity, "phase": "initialization", "logical_role": "solver",
        "client_role": "solver", "provider_attempts": 0 if cache else 1,
        "successful_provider_calls": success, "cache_hit": cache,
        "input_tokens": 0 if cache else 2, "output_tokens": 0 if cache else 1,
        "total_tokens": 0 if cache else 3, "seed": 81, "arm": "fake",
        "update_index": -1, "target_member": -1,
    }


def test_ledger_failure_poisons_and_releases_reservation(tmp_path: Path, monkeypatch) -> None:
    import multi_dataset_diverse_rl.team_search.execution_runtime as runtime

    ledger = CappedDurableLedger(tmp_path / "ledger.jsonl", successful_ceiling=2, attempt_ceiling=3)
    ledger.reserve_provider_attempt("solver")
    monkeypatch.setattr(runtime, "append_jsonl", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        ledger.append(_row("one", success=1))
    assert ledger.poisoned and ledger._inflight == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        ledger.reserve_provider_attempt("reflection")


@pytest.mark.parametrize("corruption", ["missing", "invalid_json", "empty_object"])
def test_incomplete_offline_evidence_is_hold(tmp_path: Path, corruption: str) -> None:
    if corruption == "invalid_json":
        (tmp_path / "execution_summary.json").write_text("{", encoding="utf-8")
    elif corruption == "empty_object":
        (tmp_path / "execution_summary.json").write_text("{}", encoding="utf-8")
    assert audit(tmp_path)["gate"] == "HOLD"


@pytest.mark.parametrize("legacy", [
    "V17_FORMAL_SOURCE_FREEZE", "V16_M2F_ONLINE_SOURCE_FREEZE",
])
@pytest.mark.parametrize("cwd_kind", ["repository", "scripts", "unrelated"])
def test_legacy_environment_poison_rejected_by_real_cli_fresh_process(
    tmp_path: Path, legacy: str, cwd_kind: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    cwd = {"repository": root, "scripts": root / "scripts", "unrelated": tmp_path}[cwd_kind]
    env = os.environ.copy()
    env[legacy] = "poisoned-frozen-source"
    process = subprocess.run(
        [sys.executable, str(root / "scripts/run_experiment.py"),
         "--prep", str(tmp_path / "absent-prep"), "--preflight"],
        cwd=cwd, env=env, text=True, capture_output=True, timeout=30,
    )
    assert process.returncode != 0
    assert "legacy source-freeze environment" in process.stderr


def test_reflection_post_provider_callback_failure_is_one_success_no_retry(
    tmp_path: Path,
) -> None:
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient
    from multi_dataset_diverse_rl.team_search.execution_runtime import (
        LOCAL_OPTIMIZER_INVOCATION, ReflectionLM, ledger_summary,
    )

    physical = 0
    async def fake_provider(*_):
        nonlocal physical
        physical += 1
        return "Use semantic context."

    llm = RoleAwareLLMClient(Config.from_flat(seed=81), override=fake_provider)
    ledger = CappedDurableLedger(tmp_path / "ledger.jsonl", successful_ceiling=2, attempt_ceiling=2)
    llm.provider_attempt_guard = lambda: ledger.reserve_provider_attempt("reflection")
    def callback(stage: str) -> None:
        if stage == "after_successful_provider_call":
            raise RuntimeError("post-success local callback failure")
    llm.source_freeze_guard = callback
    bridge = ReflectionLM(SimpleNamespace(llm=llm, ledger=ledger, arm="fake"))

    async def invoke() -> None:
        token = LOCAL_OPTIMIZER_INVOCATION.set({
            "loop": asyncio.get_running_loop(), "optimizer_model": "qwen3.7-flash",
            "run_seed": 81, "update_index": 0, "target_member": 2,
        })
        try:
            await asyncio.to_thread(bridge, "one reflection")
        finally:
            LOCAL_OPTIMIZER_INVOCATION.reset(token)

    with pytest.raises(RuntimeError, match="post-success"):
        asyncio.run(invoke())
    summary = ledger_summary(tmp_path / "ledger.jsonl")
    assert physical == 1
    assert len(llm.calls) == 1 and llm.calls[0]["success"] is True
    assert summary["provider_attempts"] == summary["successful_provider_calls"] == 1
    assert summary["failed_provider_attempts"] == ledger._inflight == 0


def test_solver_post_provider_parser_failure_does_not_retry_or_lose_success(
    tmp_path: Path, monkeypatch,
) -> None:
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.team_search.execution_runtime import (
        CommonContractExecutionSystem, ledger_summary,
    )
    import infrastructure.common_solver_contract_v1.evaluator as common_evaluator

    physical = 0
    class Completion:
        async def create(self, **_):
            nonlocal physical
            physical += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="FINAL_ANSWER: A"),
                                         finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda **_: fake)
    ledger = CappedDurableLedger(tmp_path / "ledger.jsonl", successful_ceiling=2, attempt_ceiling=2)
    system = CommonContractExecutionSystem(
        Config.from_flat(seed=81), arm="fake", ledger=ledger, raw_cache={},
    )
    system.set_stage({
        "phase": "initialization", "update_index": -1,
        "target_member": -1, "candidate_id": "P0",
    })
    monkeypatch.setattr(
        common_evaluator, "parse_solver_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("parser fault")),
    )
    with pytest.raises(RuntimeError, match="parser fault"):
        asyncio.run(system.common.evaluate(
            decision_procedure="Use context.",
            question="Choose one.\nOptions:\n(A) One\n(B) Two",
        ))
    summary = ledger_summary(tmp_path / "ledger.jsonl")
    assert physical == 1
    assert summary["provider_attempts"] == summary["successful_provider_calls"] == 1
    assert summary["failed_provider_attempts"] == ledger._inflight == 0


@pytest.mark.parametrize("failure_mode", ["two_then_success", "nonretryable", "timeout"])
def test_real_solver_transport_retry_rows_reconcile(
    tmp_path: Path, monkeypatch, failure_mode: str,
) -> None:
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.team_search.execution_runtime import (
        CommonContractExecutionSystem, ledger_summary,
    )
    import infrastructure.common_solver_contract_v1.evaluator as common_evaluator

    physical = 0
    class Forbidden(Exception):
        status_code = 403
    class Completion:
        async def create(self, **_):
            nonlocal physical
            physical += 1
            if failure_mode == "two_then_success" and physical <= 2:
                raise ConnectionError("transient")
            if failure_mode == "nonretryable":
                raise Forbidden("denied")
            if failure_mode == "timeout":
                raise TimeoutError("timed out")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="FINAL_ANSWER: A"),
                                         finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda **_: fake)
    async def no_backoff(_):
        return None
    monkeypatch.setattr(common_evaluator.asyncio, "sleep", no_backoff)
    ledger = CappedDurableLedger(tmp_path / "ledger.jsonl", successful_ceiling=8, attempt_ceiling=8)
    system = CommonContractExecutionSystem(
        Config.from_flat(seed=81), arm="fake", ledger=ledger, raw_cache={},
    )
    system.set_stage({
        "phase": "initialization", "update_index": -1,
        "target_member": -1, "candidate_id": "P0",
    })
    call = lambda: system.common.evaluate(
        decision_procedure="Use context.",
        question="Choose one.\nOptions:\n(A) One\n(B) Two",
    )
    if failure_mode == "two_then_success":
        result = asyncio.run(call())
        assert result.transport_attempts == 3
    else:
        with pytest.raises((Forbidden, TimeoutError)):
            asyncio.run(call())
    summary = ledger_summary(tmp_path / "ledger.jsonl")
    expected_attempts = {"two_then_success": 3, "nonretryable": 1, "timeout": 4}[failure_mode]
    assert physical == summary["provider_attempts"] == expected_attempts
    assert summary["successful_provider_calls"] == int(failure_mode == "two_then_success")
    assert summary["failed_provider_attempts"] == physical - summary["successful_provider_calls"]
    assert ledger._inflight == 0


def test_eight_concurrent_solver_requests_never_overshoot_shared_success_ceiling(
    tmp_path: Path, monkeypatch,
) -> None:
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.team_search.execution_runtime import (
        CommonContractExecutionSystem, ledger_summary,
    )

    physical = 0
    class Completion:
        async def create(self, **_):
            nonlocal physical
            physical += 1
            await asyncio.sleep(0)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="FINAL_ANSWER: A"),
                                         finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda **_: fake)
    ledger = CappedDurableLedger(tmp_path / "ledger.jsonl", successful_ceiling=3, attempt_ceiling=8)
    system = CommonContractExecutionSystem(
        Config.from_flat(seed=81), arm="fake", ledger=ledger, raw_cache={},
    )
    system.set_stage({
        "phase": "initialization", "update_index": -1,
        "target_member": -1, "candidate_id": "P0",
    })
    async def eight():
        return await asyncio.gather(*(
            system.common.evaluate(
                decision_procedure="Use context.",
                question=f"Choose {index}.\nOptions:\n(A) One\n(B) Two",
            ) for index in range(8)
        ), return_exceptions=True)
    outcomes = asyncio.run(eight())
    summary = ledger_summary(tmp_path / "ledger.jsonl")
    assert physical == summary["provider_attempts"] == summary["successful_provider_calls"] == 3
    assert sum(not isinstance(row, Exception) for row in outcomes) == 3
    assert sum(isinstance(row, RuntimeError) for row in outcomes) == 5
    assert ledger._inflight == 0


def test_malformed_success_response_is_durable_without_retry(
    tmp_path: Path, monkeypatch,
) -> None:
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.team_search.execution_runtime import (
        CommonContractExecutionSystem, ledger_summary,
    )
    physical = 0
    class MalformedUsage:
        @property
        def prompt_tokens(self):
            raise RuntimeError("local response extraction fault")
    class Completion:
        async def create(self, **_):
            nonlocal physical
            physical += 1
            return SimpleNamespace(choices=[], usage=MalformedUsage())
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda **_: fake)
    ledger = CappedDurableLedger(tmp_path / "ledger.jsonl", successful_ceiling=2, attempt_ceiling=2)
    system = CommonContractExecutionSystem(
        Config.from_flat(seed=81), arm="fake", ledger=ledger, raw_cache={},
    )
    system.set_stage({
        "phase": "initialization", "update_index": -1,
        "target_member": -1, "candidate_id": "P0",
    })
    with pytest.raises(RuntimeError, match="local response extraction fault"):
        asyncio.run(system.common.evaluate(
            decision_procedure="Use context.",
            question="Choose one.\nOptions:\n(A) One\n(B) Two",
        ))
    summary = ledger_summary(tmp_path / "ledger.jsonl")
    assert physical == summary["provider_attempts"] == summary["successful_provider_calls"] == 1
    assert summary["failed_provider_attempts"] == ledger._inflight == 0
    row = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert row["postprocess_failed"] is True


def test_reflection_malformed_provider_response_is_success_not_failed_retry(
    tmp_path: Path,
) -> None:
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient
    from multi_dataset_diverse_rl.team_search.execution_runtime import (
        LOCAL_OPTIMIZER_INVOCATION, ReflectionLM, ledger_summary,
    )
    physical = 0
    class Completion:
        async def create(self, **_):
            nonlocal physical
            physical += 1
            return SimpleNamespace(choices=[])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
    llm = RoleAwareLLMClient(Config.from_flat(seed=81))
    llm.clients["optimizer"] = fake
    ledger = CappedDurableLedger(tmp_path / "ledger.jsonl", successful_ceiling=2, attempt_ceiling=2)
    llm.provider_attempt_guard = lambda: ledger.reserve_provider_attempt("reflection")
    bridge = ReflectionLM(SimpleNamespace(llm=llm, ledger=ledger, arm="fake"))
    async def invoke() -> None:
        token = LOCAL_OPTIMIZER_INVOCATION.set({
            "loop": asyncio.get_running_loop(), "optimizer_model": "qwen3.7-flash",
            "run_seed": 81, "update_index": 0, "target_member": 2,
        })
        try:
            await asyncio.to_thread(bridge, "one reflection")
        finally:
            LOCAL_OPTIMIZER_INVOCATION.reset(token)
    with pytest.raises(IndexError):
        asyncio.run(invoke())
    summary = ledger_summary(tmp_path / "ledger.jsonl")
    assert physical == summary["provider_attempts"] == summary["successful_provider_calls"] == 1
    assert summary["failed_provider_attempts"] == ledger._inflight == 0
    assert len(llm.calls) == 1 and llm.calls[0]["error_type"] == "postprocess_failed"
