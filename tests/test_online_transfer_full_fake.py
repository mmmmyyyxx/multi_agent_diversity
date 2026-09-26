"""Zero-provider rehearsal of the production online diagnostic composition."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.governance.production_execution import (
    ValidatedExecutionContext, terminal_lifecycle,
)
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json, io_path, read_json
from multi_dataset_diverse_rl.production_transfer_diagnostic import execute_online_transfer_diagnostic
from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
from multi_dataset_diverse_rl.team_search.system_runtime import FrozenResponsibilitySnapshot
from scripts.audit_online_transfer_diagnostic import audit
from scripts.prepare_online_transfer_diagnostic_v2 import frozen_payload
from scripts.prepare_post_refactor_gepa_canary import _private_splits


def rehearse(
    tmp_path: Path, monkeypatch, rehearsal: int, *, through_cli: bool = False,
    scenario: str = "commit",
) -> dict:
    # Exceed the intended formal root and its deepest categorical-profile path
    # on native Windows without relying on the machine-wide long-path switch.
    base = tmp_path / ("深 路径 " + "x" * 36) / ("depth " + "y" * 12)
    prep = base / "prep"
    prep.mkdir(parents=True)
    _private_splits(prep)
    manifest, protocol = frozen_payload(execution_source_sha="a" * 40)
    # This fixture replays the superseded v2 routed-source freeze. Current
    # Layer-2 raw-legal behavior has its own source/poison regression tests.
    def v2_routed_snapshot(system, *, update_index):
        _, assigned = system.assign_responsibilities(update_index=update_index)
        states, _, _ = system.current_states_and_opportunities()
        return FrozenResponsibilitySnapshot(
            assigned={member: tuple(rows) for member, rows in assigned.items()},
            state_by_question={row.question_hash: row for row in states},
            current_margin_by_question={row.question_hash: row.plurality_margin for row in states},
            source_version="historical_service_routed_v1",
        )
    monkeypatch.setattr(
        "multi_dataset_diverse_rl.production_transfer_diagnostic.freeze_current_responsibility",
        v2_routed_snapshot,
    )
    (prep / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (prep / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    labels = {}
    shadow_questions = set()
    for name in ("optimize100.csv", "shadow50.csv"):
        with (prep / "splits_private" / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                question = row["question"].replace("\r\n", "\n").strip()
                labels[question] = row["answer"].strip("()")
                if name == "shadow50.csv":
                    shadow_questions.add(question)
    run_root = base / "run"
    assert len(str(run_root / "system/team_full_categorical_profiles" / ("a" * 64 + ".json"))) >= 260
    if not through_cli:
        run_root.mkdir()
        (run_root / "run_lifecycle.json").write_text(
            json.dumps({"status": "RUNNING", "provider_client_constructed": False,
                        "events": [{"status": "RUNNING"}]}),
            encoding="utf-8",
        )
    permit = ValidatedExecutionContext(
        experiment_id=manifest["experiment_id"], attempt_id=manifest["attempt_id"],
        execution_source_sha="a" * 40, preregistration_sha256="b" * 64,
        run_identity_sha256="c" * 64, provider_profile="lwj",
        endpoint_fingerprint=manifest["runtime"]["endpoint_fingerprint"],
        allowed_phase="diagnostic", allowed_roles=("solver", "reflection"),
        prep_root=prep, run_root=None if through_cli else run_root,
    )
    calls = {"solver": 0, "reflection": 0}

    class FakeCompletion:
        async def create(self, **request):
            if request["model"] == "qwen3.7-flash":
                calls["reflection"] += 1
                content = (
                    "Distinguish the referent by checking pronoun agreement and "
                    "local semantic context before choosing an option."
                )
            else:
                calls["solver"] += 1
                question = request["messages"][1]["content"]
                gold = labels[question]
                changed = "Distinguish the referent by checking" in request["messages"][0]["content"]
                # The parent retains broad competence, while the child's
                # local repair is consistently useful and has no regression.
                score = hashlib.sha256(question.encode("utf-8")).digest()[0]
                ceiling = 148 if scenario == "minibatch_fail" else 180
                correct = score < 128 or (changed and score < ceiling)
                if scenario == "shadow_fail" and changed and question in shadow_questions:
                    correct = False
                answer = gold if correct else ("B" if gold == "A" else "A")
                content = f"FINAL_ANSWER: {answer}"
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
            )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletion()))
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda *a, **kw: fake)
    monkeypatch.setattr(ProviderClientFactory, "create", lambda *a, **kw: fake)
    if through_cli:
        import scripts.run_experiment as entry

        monkeypatch.setattr(entry, "validate_execution", lambda **_: permit)
        monkeypatch.setattr(sys, "argv", [
            "scripts/run_experiment.py", "--prep", str(prep),
            "--run-root", str(run_root), "--execute",
        ])
        entry.main()
        result = read_json(run_root / "execution_summary.json")
    else:
        result = asyncio.run(execute_online_transfer_diagnostic(
            permit, root=Path(__file__).resolve().parents[1],
        ))
    assert calls["solver"] >= 100
    assert calls["reflection"] >= 1
    assert result["accepted_mutations"] >= 1
    if scenario == "commit":
        assert result["opportunities"] >= 2, (result["stop_reason"], result["parent_sequence"])
        assert result["commits"] >= 1, result
        assert result["parent_sequence"][0] != result["parent_sequence"][1]
    elif scenario == "minibatch_fail":
        assert any(not row["team_minibatch"]["passed"] for row in result["candidate_diagnostics"]), result["candidate_diagnostics"]
        assert any(row["full"]["diagnostic_only"] for row in result["candidate_diagnostics"])
    else:
        assert any(row["team_minibatch"]["passed"] for row in result["candidate_diagnostics"])
        assert any(row["ordinary_shadow"] not in {"NOT_REACHED", "PASS"} for row in result["candidate_diagnostics"]), result["candidate_diagnostics"]
        assert result["commits"] == 0
    assert result["validation50_calls"] == result["test50_calls"] == 0
    assert result["ledger"]["successful_provider_calls"] == sum(calls.values())
    assert result["stage_accounting"]["initialization"]["logical_solver_rows"] == 500
    endpoint_root = run_root / "system" / "endpoint_identifiability_states"
    with os.scandir(io_path(endpoint_root)) as entries:
        endpoint_rows = [read_json(entry.path) for entry in entries if entry.name.endswith(".json")]
    assert any(row.get("trigger") == "fixed_probe_initialization" for row in endpoint_rows)
    if not through_cli:
        atomic_write_json(run_root / "execution_summary.json", result)
        usage = result["ledger"]
        terminal_lifecycle(
            permit, status="EXECUTION_COMPLETE",
            provider_attempts=usage["provider_attempts"],
            provider_successes=usage["successful_provider_calls"],
            provider_failures=usage["failed_provider_attempts"],
        )
    verified = audit(run_root)
    assert verified["gate"] == "PASS", (rehearsal, verified)
    assert not (base / ".run.starting").exists()
    assert not any(
        name.endswith(".tmp")
        for _, _, names in os.walk(io_path(run_root)) for name in names
    )
    return {
        key: result[key] for key in (
            "initial_team_hash", "final_team_hash", "opportunities",
            "accepted_mutations", "reflection_proposals", "commits", "stop_reason",
        )
    }


@pytest.mark.parametrize("rehearsal", range(3))
def test_real_topology_fake_provider_reaches_local_and_team(
    tmp_path: Path, monkeypatch, rehearsal: int,
) -> None:
    rehearse(tmp_path, monkeypatch, rehearsal)


def test_fresh_process_real_cli_fake_provider(tmp_path: Path) -> None:
    source = Path(__file__).resolve()
    code = """
import importlib.util, pathlib, pytest, sys
spec = importlib.util.spec_from_file_location('full_fake_rehearsal', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
monkeypatch = pytest.MonkeyPatch()
try:
    module.rehearse(pathlib.Path(sys.argv[2]), monkeypatch, 3, through_cli=True)
finally:
    monkeypatch.undo()
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(source), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], text=True,
        capture_output=True, timeout=180,
    )
    assert process.returncode == 0, process.stderr[-4000:]


@pytest.mark.skip(reason="superseded V1 minibatch fixture; diagnostic Full isolation is tested in test_online_transfer_diagnostic.py")
def test_mandatory_diagnostic_full_after_real_minibatch_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    rehearse(tmp_path, monkeypatch, 4, scenario="minibatch_fail")


def test_ordinary_promoted_full_then_shadow_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    rehearse(tmp_path, monkeypatch, 5, scenario="shadow_fail")


def test_two_consecutive_rehearsals_have_no_mutable_state_leak(tmp_path: Path, monkeypatch) -> None:
    with monkeypatch.context() as first:
        left = rehearse(tmp_path / "first", first, 6)
    with monkeypatch.context() as second:
        right = rehearse(tmp_path / "second", second, 7)
    assert left == right
