"""Strict post-search JSON boundary and the aborted Native shape, with no API."""

from __future__ import annotations

from dataclasses import replace
import copy
import json
import os
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.formal_final_team import persist_final_team, team_hash
from multi_dataset_diverse_rl.governance.production_execution import ValidatedExecutionContext
from multi_dataset_diverse_rl.persistence.durable_io import (
    atomic_write_json, canonical_json_payload, read_json,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary
from scripts import run_experiment as runner
from scripts.derive_formal_trajectory_trace import derive
from scripts.freeze_formal_v3_execution import freeze
from scripts.freeze_formal_v3_execution import verify_freeze
from scripts import freeze_formal_v3_execution as freeze_module
from scripts import derive_formal_trajectory_trace as derive_module
from multi_dataset_diverse_rl.persistence import durable_io
from multi_dataset_diverse_rl.governance import production_execution


def test_strict_json_boundary_normalizes_all_nested_tuples():
    payload = {
        "events": [{"candidate_ids": ("a", "b"),
                    "telemetry": {"nested": (True, {"score": (1, 0.5)})}}],
        "candidate_diagnostics": [{"ids": ("a", ("b",))}],
        "transition_trace": [{"changed": ("a", "b")}],
        "evidence_view_trace": [{"ids": ("x", "y")}],
        "stop_reason": "SATURATION_REACHED",
        "count": 2,
    }
    normalized = canonical_json_payload(payload)
    assert payload["events"][0]["candidate_ids"] == ("a", "b")
    assert normalized["events"][0]["candidate_ids"] == ["a", "b"]
    assert normalized["events"][0]["telemetry"]["nested"] == [True, {"score": [1, 0.5]}]
    assert normalized["candidate_diagnostics"][0]["ids"] == ["a", ["b"]]
    assert normalized["transition_trace"][0]["changed"] == ["a", "b"]
    assert normalized["evidence_view_trace"][0]["ids"] == ["x", "y"]
    assert normalized["stop_reason"] == payload["stop_reason"]
    assert normalized["count"] == 2 and type(normalized["count"]) is int
    assert type(normalized["events"][0]["telemetry"]["nested"][0]) is bool


@pytest.mark.parametrize("invalid", [object(), {"set"}, float("nan"), float("inf"), {1: "bad key"}])
def test_strict_json_boundary_rejects_non_json_values(invalid):
    with pytest.raises((TypeError, ValueError)):
        canonical_json_payload({"invalid": invalid})


def _offline_native(tmp_path: Path, monkeypatch, *, tamper=None, unsupported=False):
    attempt = "gepa_saturation_comparison_v3_seed80_native_attempt3"
    prep = tmp_path / "prep"
    prep.mkdir()
    run = tmp_path / "run"
    permit = ValidatedExecutionContext(
        experiment_id=attempt, attempt_id=attempt,
        execution_source_sha="a" * 40, preregistration_sha256="b" * 64,
        run_identity_sha256="c" * 64, provider_profile="lwj",
        endpoint_fingerprint="offline", allowed_phase="formal",
        allowed_roles=("solver", "reflection"), prep_root=prep,
    )

    def fake_admit(value, run_root):
        assert value == permit
        run_root.mkdir()
        (run_root / "ledger.jsonl").write_text("", encoding="utf-8")
        atomic_write_json(run_root / "run_lifecycle.json", {
            "attempt_id": attempt, "run_identity_sha256": permit.run_identity_sha256,
            "status": "RUNNING", "provider_boundary_reached": False,
            "provider_client_constructed": False, "provider_attempts": 0,
            "provider_successes": 0, "provider_failures": 0,
            "events": [{"status": "RUNNING", "timestamp": "offline"}],
        })
        return replace(value, run_root=run_root)

    async def fake_search(admitted, *, root):
        del root
        assert admitted.run_root == run
        initial = ["P0"] * 5
        candidate = ["P1"] * 5
        initial_hash = team_hash(initial)
        final_team = persist_final_team(
            run / "final_team_materialization.json", mode="GEPA_NATIVE",
            initial_prompts=initial, final_prompts=candidate,
            candidate_id="candidate-a", initial_team_hash=initial_hash,
            search_final_identity="candidate-a",
        )
        result = {
            "experiment_id": attempt, "mode_id": "GEPA_NATIVE", "seed": 80,
            "initial_team_hash": initial_hash,
            "final_native_candidate_hash": "candidate-a",
            "final_team_materialization": final_team,
            "stop_reason": "SATURATION_REACHED",
            "events": [{
                "index": 1, "kind": "LOCAL_OPTIMIZATION",
                "request_identity": "offline", "candidate_ids": ("a", "b"),
                "committed_candidate_id": None, "state_hash": "candidate-a",
                "stop_reason": "SATURATION_REACHED",
                "telemetry": {"backend_termination_reason": "SATURATION_REACHED",
                              "backend_saturation": {"nested": (1, 2)}},
            }],
            "candidate_diagnostics": [{"score": 1.5, "ids": ("a", "b")}],
            "transition_trace": [{"hash": "hash-a", "ids": ("a",)}],
            "evidence_view_trace": [{"ids": ("example-a",)}],
            "ledger": ledger_summary(run / "ledger.jsonl"),
            "validation50_calls": 0, "test50_calls": 0,
        }
        if unsupported:
            result["unsupported"] = object()
        return result

    monkeypatch.setattr(runner, "validate_execution", lambda **_kwargs: permit)
    monkeypatch.setattr(runner, "admit_execution", fake_admit)
    monkeypatch.setattr(runner, "execute_formal_gepa_saturation", fake_search)
    if tamper is not None:
        original_read = runner.read_json

        def corrupt_before_readback(path):
            if Path(path).name == "execution_summary.json":
                value = original_read(path)
                tamper(value)
                atomic_write_json(path, value)
            return original_read(path)

        monkeypatch.setattr(runner, "read_json", corrupt_before_readback)
    return prep, run


def test_seed80_native_abort_shape_completes_freezes_and_derives(tmp_path, monkeypatch):
    prep, run = _offline_native(tmp_path, monkeypatch)
    del prep
    result = __import__("asyncio").run(runner.execute_frozen(tmp_path / "prep", run))
    assert result["stop_reason"] == "SATURATION_REACHED"
    assert result["events"][0]["candidate_ids"] == ["a", "b"]
    assert read_json(run / "execution_summary.json") == result
    assert read_json(run / "run_lifecycle.json")["status"] == "EXECUTION_COMPLETE"
    assert freeze(run)["artifact_count"] >= 4
    assert derive(run)["record_count"] == 1
    assert json.loads((run / "formal_trajectory_trace.jsonl").read_text(encoding="utf-8"))[
        "record_type"
    ] == "NATIVE_OPTIMIZATION_UNIT"


@pytest.mark.parametrize("field", ["candidate_id", "score", "hash", "stop_reason", "ledger_count"])
def test_execution_summary_readback_corruption_still_aborts(tmp_path, monkeypatch, field):
    def tamper(value):
        if field == "candidate_id":
            value["events"][0]["candidate_ids"][0] = "other"
        elif field == "score":
            value["candidate_diagnostics"][0]["score"] = 9.0
        elif field == "hash":
            value["transition_trace"][0]["hash"] = "other"
        elif field == "stop_reason":
            value["stop_reason"] = "OTHER"
        else:
            value["ledger"]["provider_attempts"] = 1

    _, run = _offline_native(tmp_path, monkeypatch, tamper=tamper)
    with pytest.raises(RuntimeError, match="execution summary read-back mismatch"):
        __import__("asyncio").run(runner.execute_frozen(tmp_path / "prep", run))
    assert read_json(run / "run_lifecycle.json")["status"] == "FAILED_START"


def test_execution_rejects_unsupported_value_before_summary_write(tmp_path, monkeypatch):
    _, run = _offline_native(tmp_path, monkeypatch, unsupported=True)
    with pytest.raises(TypeError):
        __import__("asyncio").run(runner.execute_frozen(tmp_path / "prep", run))
    assert not (run / "execution_summary.json").exists()
    assert read_json(run / "run_lifecycle.json")["status"] == "FAILED_START"


@pytest.mark.parametrize("boundary", [
    "summary_temp_write", "summary_replace", "summary_readback",
    "lifecycle_write", "lifecycle_replace",
])
def test_execution_finalization_fault_never_marks_corrupt_summary_complete(
    tmp_path, monkeypatch, boundary,
):
    _, run = _offline_native(tmp_path, monkeypatch)
    if boundary == "summary_temp_write":
        original = durable_io.json.dump

        def fail_summary_dump(payload, handle, **kwargs):
            if isinstance(payload, dict) and payload.get("experiment_id"):
                raise OSError("injected summary temporary write")
            return original(payload, handle, **kwargs)

        monkeypatch.setattr(durable_io.json, "dump", fail_summary_dump)
    elif boundary in {"summary_replace", "lifecycle_replace"}:
        original = durable_io.atomic_replace
        name = "execution_summary.json" if boundary == "summary_replace" else "run_lifecycle.json"
        failed = False

        def fail_replace(source, destination):
            nonlocal failed
            if Path(destination).name == name and (name != "run_lifecycle.json" or Path(destination).exists()) and not failed:
                failed = True
                raise OSError(f"injected {boundary}")
            return original(source, destination)

        monkeypatch.setattr(durable_io, "atomic_replace", fail_replace)
        if boundary == "lifecycle_replace":
            monkeypatch.setattr(production_execution, "atomic_write_json", durable_io.atomic_write_json)
    elif boundary == "summary_readback":
        original = runner.read_json
        failed = False

        def fail_readback(path):
            nonlocal failed
            if Path(path).name == "execution_summary.json" and not failed:
                failed = True
                raise OSError("injected summary read-back")
            return original(path)

        monkeypatch.setattr(runner, "read_json", fail_readback)
    else:
        original = production_execution.atomic_write_json
        failed = False

        def fail_lifecycle_write(path, payload):
            nonlocal failed
            if Path(path).name == "run_lifecycle.json" and not failed:
                failed = True
                raise OSError("injected lifecycle temporary write")
            return original(path, payload)

        monkeypatch.setattr(production_execution, "atomic_write_json", fail_lifecycle_write)
    with pytest.raises(OSError, match="injected"):
        __import__("asyncio").run(runner.execute_frozen(tmp_path / "prep", run))
    assert read_json(run / "run_lifecycle.json")["status"] != "EXECUTION_COMPLETE"
    assert not list(run.glob(".*.tmp"))
    if boundary in {"summary_temp_write", "summary_replace"}:
        assert not (run / "execution_summary.json").exists()


def _completed_native(tmp_path, monkeypatch):
    _, run = _offline_native(tmp_path, monkeypatch)
    __import__("asyncio").run(runner.execute_frozen(tmp_path / "prep", run))
    assert read_json(run / "run_lifecycle.json")["status"] == "EXECUTION_COMPLETE"
    return run


@pytest.mark.parametrize("boundary", ["temp_write", "replace", "readback"])
def test_raw_freeze_fault_is_recoverable_without_provider_rerun(tmp_path, monkeypatch, boundary):
    run = _completed_native(tmp_path, monkeypatch)
    original_summary = (run / "execution_summary.json").read_bytes()
    with monkeypatch.context() as patch:
        if boundary == "temp_write":
            original = freeze_module.json.dump

            def fail_dump(payload, handle, **kwargs):
                if isinstance(payload, dict) and payload.get("schema_version") == freeze_module.SCHEMA:
                    raise OSError("injected freeze temporary write")
                return original(payload, handle, **kwargs)

            patch.setattr(freeze_module.json, "dump", fail_dump)
        elif boundary == "replace":
            patch.setattr(freeze_module, "atomic_replace", lambda *_args: (_ for _ in ()).throw(OSError("injected freeze replace")))
        else:
            patch.setattr(freeze_module, "verify_freeze", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected freeze read-back")))
        with pytest.raises(OSError, match="injected"):
            freeze(run)
    assert read_json(run / "run_lifecycle.json")["status"] == "EXECUTION_COMPLETE"
    assert not (run / "execution_evidence_freeze.json").exists()
    assert not list(run.glob(".execution_evidence_freeze.json.*.tmp"))
    assert (run / "execution_summary.json").read_bytes() == original_summary
    frozen = freeze(run)
    assert verify_freeze(run)["artifact_count"] == frozen["artifact_count"]


@pytest.mark.parametrize("boundary", ["temp_write", "serialize_midway", "replace", "readback"])
def test_trajectory_fault_is_recoverable_without_provider_rerun(tmp_path, monkeypatch, boundary):
    run = _completed_native(tmp_path, monkeypatch)
    if boundary == "serialize_midway":
        summary = read_json(run / "execution_summary.json")
        second = copy.deepcopy(summary["events"][0])
        second["index"] = 2
        summary["events"].append(second)
        atomic_write_json(run / "execution_summary.json", summary)
    freeze(run)
    frozen_hash = freeze_module._digest(run / "execution_evidence_freeze.json")[1]
    with monkeypatch.context() as patch:
        if boundary == "temp_write":
            import builtins
            original_open = builtins.open

            def fail_open(path, mode="r", *args, **kwargs):
                if ".formal_trajectory_trace.jsonl." in str(path) and mode == "x":
                    raise OSError("injected trajectory temporary write")
                return original_open(path, mode, *args, **kwargs)

            patch.setattr(builtins, "open", fail_open)
        elif boundary == "serialize_midway":
            original = derive_module.json.dumps
            calls = 0

            def fail_second(value, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected trajectory serialization midway")
                return original(value, **kwargs)

            patch.setattr(derive_module.json, "dumps", fail_second)
        elif boundary == "replace":
            patch.setattr(derive_module, "atomic_replace", lambda *_args: (_ for _ in ()).throw(OSError("injected trajectory replace")))
        else:
            import builtins
            original_open = builtins.open

            def fail_readback(path, mode="r", *args, **kwargs):
                if Path(path).name == "formal_trajectory_trace.jsonl" and mode == "rb":
                    raise OSError("injected trajectory read-back")
                return original_open(path, mode, *args, **kwargs)

            patch.setattr(builtins, "open", fail_readback)
        with pytest.raises(OSError, match="injected"):
            derive(run)
    assert read_json(run / "run_lifecycle.json")["status"] == "EXECUTION_COMPLETE"
    assert not (run / "formal_trajectory_trace.jsonl").exists()
    assert not list(run.glob(".formal_trajectory_trace.jsonl.*.tmp"))
    assert freeze_module._digest(run / "execution_evidence_freeze.json")[1] == frozen_hash
    assert derive(run)["record_count"] >= 1


@pytest.mark.parametrize("corruption", [
    "schema", "attempt", "count", "duplicate", "traversal", "absolute",
    "size", "sha", "missing", "extra", "symlink",
])
def test_full_freeze_inventory_rejects_corruption(tmp_path, monkeypatch, corruption):
    run = _completed_native(tmp_path, monkeypatch)
    freeze(run)
    inventory_path = run / "execution_evidence_freeze.json"
    inventory = read_json(inventory_path)
    if corruption == "schema":
        inventory["schema_version"] = "other"
    elif corruption == "attempt":
        inventory["attempt_id"] = "other"
    elif corruption == "count":
        inventory["artifact_count"] += 1
    elif corruption == "duplicate":
        inventory["files"].append(dict(inventory["files"][0]))
        inventory["artifact_count"] += 1
    elif corruption == "traversal":
        inventory["files"][0]["path"] = "../escape"
    elif corruption == "absolute":
        inventory["files"][0]["path"] = "C:/escape"
    elif corruption == "size":
        inventory["files"][0]["size_bytes"] += 1
    elif corruption == "sha":
        inventory["files"][0]["sha256"] = "0" * 64
    elif corruption == "missing":
        (run / inventory["files"][0]["path"]).unlink()
    elif corruption == "extra":
        (run / "unlisted.json").write_text("{}", encoding="utf-8")
    else:
        try:
            os.symlink(run / "execution_summary.json", run / "linked.json")
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation unavailable")
    if corruption in {"schema", "attempt", "count", "duplicate", "traversal", "absolute", "size", "sha"}:
        atomic_write_json(inventory_path, inventory)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        verify_freeze(run)
