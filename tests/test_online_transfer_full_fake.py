"""Zero-provider rehearsal of the production online diagnostic composition."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
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
from multi_dataset_diverse_rl.team_search.system_runtime import SystemResponsibilityAssignmentFactory
from multi_dataset_diverse_rl.team_search.feasibility import (
    Layer2EvidenceInfeasible, Layer2FeasibilityReason,
)
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder
from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary
from multi_dataset_diverse_rl.team_search.evidence_audit import sanitize_v4_evidence_trace
from scripts.audit_online_transfer_diagnostic import audit
from scripts.prepare_online_transfer_diagnostic_v2 import frozen_payload
from scripts.prepare_online_transfer_diagnostic_v3 import frozen_payload as v3_frozen_payload
from scripts.prepare_post_refactor_gepa_canary import _private_splits


def rehearse(
    tmp_path: Path, monkeypatch, rehearsal: int, *, through_cli: bool = False,
    scenario: str = "commit", v3: bool = False,
    v4: bool = False,
) -> dict:
    # Exceed the intended formal root and its deepest categorical-profile path
    # on native Windows without relying on the machine-wide long-path switch.
    base = tmp_path / ("深 路径 " + "x" * 36) / ("depth " + "y" * 12)
    prep = base / "prep"
    prep.mkdir(parents=True)
    _private_splits(prep)
    manifest, protocol = (
        v3_frozen_payload if (v3 or v4) else frozen_payload
    )(execution_source_sha="a" * 40)
    if v4:
        manifest["experiment_id"] = manifest["attempt_id"] = (
            "gepa_layer2_local_to_team_transfer_diagnostic_v4"
        )
        protocol["experiment_id"] = manifest["experiment_id"]
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
    if not (v3 or v4):
        monkeypatch.setattr(
            "multi_dataset_diverse_rl.production_transfer_diagnostic.freeze_current_responsibility",
            v2_routed_snapshot,
        )
    (prep / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (prep / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    labels = {}
    shadow_questions = set()
    minibatch_hashes: set[str] = set()
    minibatch_questions: set[str] = set()
    if scenario == "common_safe_fail":
        original_assignment = SystemResponsibilityAssignmentFactory.build_from_member

        def capture_assignment(self, **kwargs):
            selected = original_assignment(self, **kwargs)
            minibatch_hashes.update(selected.local_validation_example_ids)
            minibatch_questions.update(
                row.question.replace("\r\n", "\n").strip()
                for row in self.system.fixed_probe.examples
                if row.question_hash in selected.local_validation_example_ids
            )
            return selected

        monkeypatch.setattr(
            SystemResponsibilityAssignmentFactory, "build_from_member", capture_assignment,
        )
    if scenario == "feasible_rerank":
        original_assignment = SystemResponsibilityAssignmentFactory.build_from_member

        def exclude_first_member(self, **kwargs):
            if kwargs["member_id"] == 0:
                raise Layer2EvidenceInfeasible(
                    Layer2FeasibilityReason.INSUFFICIENT_PRIMARY_REPAIR_QUOTA
                )
            return original_assignment(self, **kwargs)

        monkeypatch.setattr(
            SystemResponsibilityAssignmentFactory, "build_from_member", exclude_first_member,
        )
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
                if scenario in {"two_commits", "transition_chain"}:
                    content += f" Apply refinement step {calls['reflection']}."
            else:
                calls["solver"] += 1
                if scenario == "transport_failure" and calls["solver"] >= 101:
                    raise ConnectionError("synthetic V4 transport failure")
                if scenario == "postprocess_failure" and calls["solver"] >= 101:
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="FINAL_ANSWER: A"),
                                                 finish_reason="stop")],
                        usage=SimpleNamespace(prompt_tokens="invalid-token-count", completion_tokens=5),
                    )
                question = request["messages"][1]["content"]
                gold = labels[question]
                changed = "Distinguish the referent by checking" in request["messages"][0]["content"]
                # The parent retains broad competence, while the child's
                # local repair is consistently useful and has no regression.
                score = hashlib.sha256(question.encode("utf-8")).digest()[0]
                ceiling = 148 if scenario == "minibatch_fail" else 180
                if scenario in {"two_commits", "transition_chain"} and changed:
                    match = re.search(r"refinement step (\d+)", request["messages"][0]["content"])
                    if match:
                        ceiling = min(256, 150 + 40 * int(match.group(1)))
                        if scenario == "transition_chain" and int(match.group(1)) == 3 and question in shadow_questions:
                            ceiling = 0
                correct = score < 128 or (changed and score < ceiling)
                if scenario == "common_safe_fail" and changed:
                    correct = question in minibatch_questions
                if scenario == "no_feasible":
                    correct = False
                if scenario == "proposal_ceiling":
                    correct = score < 128
                if scenario in {"shadow_fail", "budget_stress"} and changed and question in shadow_questions:
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
    # The zero-network harness removes all real credentials. Keep the normal
    # role-client/factory path under test using inert, in-memory placeholders.
    monkeypatch.setattr(
        "multi_dataset_diverse_rl.llm_client.resolve_api_key",
        lambda _configured, _profile: ("OFFLINE_FAKE_KEY_NAME", "OFFLINE_FAKE_VALUE"),
    )
    monkeypatch.setattr(
        "multi_dataset_diverse_rl.llm_client.resolve_base_url",
        lambda _configured, _profile: ("OFFLINE_FAKE_BASE_NAME", "https://invalid.example"),
    )
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda *a, **kw: fake)
    monkeypatch.setattr(ProviderClientFactory, "create", lambda *a, **kw: fake)
    if through_cli:
        import scripts.run_experiment as entry

        monkeypatch.setattr(entry, "validate_execution", lambda **_: permit)
        monkeypatch.setattr(sys, "argv", [
            "scripts/run_experiment.py", "--prep", str(prep),
            "--run-root", str(run_root), "--execute",
        ])
        if scenario in {"transport_failure", "postprocess_failure"}:
            expected_error = ConnectionError if scenario == "transport_failure" else ValueError
            with pytest.raises(expected_error):
                entry.main()
            usage = ledger_summary(run_root / "ledger.jsonl")
            lifecycle = read_json(run_root / "run_lifecycle.json")
            assert lifecycle["status"] == "ABORTED"
            assert lifecycle["provider_attempts"] == usage["provider_attempts"]
            assert lifecycle["provider_successes"] == usage["successful_provider_calls"]
            assert not (run_root / "execution_summary.json").exists()
            assert usage["provider_attempts"] > 0
            if scenario == "transport_failure":
                assert usage["failed_provider_attempts"] >= 1
            else:
                assert usage["successful_provider_calls"] >= 1
                with open(io_path(run_root / "ledger.jsonl"), encoding="utf-8") as handle:
                    assert any(json.loads(line).get("postprocess_failed") for line in handle)
            return {"failure": scenario, "ledger": usage, "lifecycle": lifecycle["status"]}
        entry.main()
        result = read_json(run_root / "execution_summary.json")
    else:
        result = asyncio.run(execute_online_transfer_diagnostic(
            permit, root=Path(__file__).resolve().parents[1],
        ))
    assert calls["solver"] >= 100
    if scenario == "no_feasible":
        assert calls["reflection"] == 0
        assert result["stop_reason"] == "NO_FEASIBLE_LAYER2_OPPORTUNITY"
        assert result["opportunities"] == result["accepted_mutations"] == 0
        assert result["feasibility_trace"][0]["selected_member"] is None
        assert not result["evidence_view_trace"]
        assert all(row["failure_count"] == 0 for row in result["feasibility_trace"][0]["members"])
        assert result["stage_accounting"]["opportunities"] == []
        for field in ("provider_attempts", "provider_successes", "failed_provider_attempts",
                      "logical_solver_rows", "reflection_provider_records", "cache_hits"):
            assert result["stage_accounting"]["global"][field] == result["stage_accounting"]["initialization"][field]
    elif scenario == "proposal_ceiling":
        assert result["accepted_mutations"] == 0
        assert result["reflection_proposals"] == 20
        assert result["stop_reason"] == "REFLECTION_PROPOSAL_CEILING_REACHED"
        assert len(result["feasibility_trace"]) == result["opportunities"]
        assert len(result["evidence_view_trace"]) == result["opportunities"]
        assert calls["reflection"] == 20
    else:
        assert calls["reflection"] >= 1
        assert result["accepted_mutations"] >= 1
    if v4 and scenario != "no_feasible":
        root = result["evidence_view_trace"][0]
        if scenario == "feasible_rerank":
            first = result["feasibility_trace"][0]
            assert first["selected_member"] == 1
            skipped = next(row for row in first["members"] if row["member_id"] == 0)
            assert skipped["raw_rank"] == 1
            assert skipped["feasible_rank"] is None
            assert skipped["failure_count"] == 0
            assert skipped["feasibility_reason"] == "INSUFFICIENT_PRIMARY_REPAIR_QUOTA"
        else:
            assert root["raw_V"] == root["assignment_V"] == root["packet_V"] == 50
        assert root["responsibility_universe"]["responsibility_universe_count"] == "50"
        assert root["responsibility_scheduled"]["count"] <= 36
        assert root["focus_ids"] == root["anchor_ids"] == []
        assert root["team_minibatch_ids"] == root["local_eval_ids"]
        assert len(root["team_minibatch_ids"]) == 12
        assert len(root["evidence_delivered"]["batch_ids"]) <= 12
        assert len(root["evidence_delivered"]["role_item_ids"]) <= 36
    if scenario in {"no_feasible", "proposal_ceiling"}:
        pass
    elif scenario in {"commit", "two_commits", "transition_chain", "feasible_rerank"}:
        assert result["opportunities"] >= 2, (result["stop_reason"], result["parent_sequence"])
        assert result["commits"] >= (2 if scenario in {"two_commits", "transition_chain"} else 1), result
        assert result["parent_sequence"][0] != result["parent_sequence"][1]
        if scenario == "transition_chain":
            assert len(result["parent_sequence"]) >= 3
            assert result["parent_sequence"][1] == result["parent_sequence"][2]
            assert result["final_team_hash"] != result["parent_sequence"][2]
            first, rejected, resumed = result["evidence_view_trace"][:3]
            assert first["committed_candidate_id"] is not None
            assert first["successor_team_hash"] == rejected["parent_team_hash"]
            assert rejected["committed_candidate_id"] is None
            assert rejected["successor_team_hash"] == rejected["parent_team_hash"]
            assert resumed["parent_team_hash"] == rejected["parent_team_hash"]
            assert (
                result["feasibility_trace"][2]["latest_transition_effect_hash_by_member"]
                == result["feasibility_trace"][1]["latest_transition_effect_hash_by_member"]
            )
            before = {row["member_id"]: row for row in result["feasibility_trace"][1]["members"]}
            after = {row["member_id"]: row for row in result["feasibility_trace"][2]["members"]}
            rejected_target = rejected["target_member"]
            assert after[rejected_target]["failure_count"] == before[rejected_target]["failure_count"] + 1
            assert all(
                after[member]["failure_count"] == before[member]["failure_count"]
                for member in before if member != rejected_target
            )
    elif scenario == "minibatch_fail":
        assert any(not row["team_minibatch"]["passed"] for row in result["candidate_diagnostics"]), result["candidate_diagnostics"]
        assert any(row["full"]["diagnostic_only"] for row in result["candidate_diagnostics"])
        if v4:
            assert all(row["successor_team_hash"] == row["parent_team_hash"]
                       for row in result["evidence_view_trace"])
    elif scenario == "common_safe_fail":
        assert any(row["team_minibatch"]["passed"] for row in result["candidate_diagnostics"])
        assert any(row["ordinary_common_safe"] not in {"NOT_REACHED", "PASS"}
                   for row in result["candidate_diagnostics"])
        assert result["commits"] == 0
        if v4:
            assert all(row["successor_team_hash"] == row["parent_team_hash"]
                       for row in result["evidence_view_trace"])
    elif scenario == "budget_stress":
        assert result["accepted_mutations"] == 5
        assert result["commits"] == 0
        assert result["ledger"]["successful_provider_calls"] <= 1200
        assert result["ledger"]["provider_attempts"] <= 4800
        assert result["ledger"]["cache_hits"] > 0
        phase_rows = [row["by_phase"] for row in result["stage_accounting"]["opportunities"]]
        for phase in ("team_minibatch_eval", "team_full_eval", "team_shadow_eval"):
            assert sum(row[phase]["cache_hits"] for row in phase_rows) > 0
    else:
        assert any(row["team_minibatch"]["passed"] for row in result["candidate_diagnostics"])
        assert any(row["ordinary_shadow"] not in {"NOT_REACHED", "PASS"} for row in result["candidate_diagnostics"]), result["candidate_diagnostics"]
        assert result["commits"] == 0
        if v4:
            assert all(row["successor_team_hash"] == row["parent_team_hash"]
                       for row in result["evidence_view_trace"])
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
    } | ({"evidence_view_trace": result["evidence_view_trace"]} if v4 else {})


def test_v4_raw_legal_bounded_full_stack_commit(tmp_path: Path, monkeypatch) -> None:
    result = rehearse(tmp_path, monkeypatch, 90, v4=True)
    trace = sanitize_v4_evidence_trace(result["evidence_view_trace"][0])
    assert trace["responsibility_universe_count"] >= trace["responsibility_scheduled_count"]
    assert trace["nominal_role_item_slots"] <= 36
    assert trace["packet_hash"]
    assert len(trace["responsibility_scheduled_ids_sha256"]) == 64
    assert len(trace["focus_ids_sha256"]) == 64
    assert len(trace["anchor_ids_sha256"]) == 64
    assert len(trace["delivered_batch_ids_sha256"]) == 64
    assert "ids" not in trace


def test_v4_no_feasible_is_scientific_stop_without_opportunity_provider_calls(
    tmp_path: Path, monkeypatch,
) -> None:
    rehearse(tmp_path, monkeypatch, 92, v4=True, scenario="no_feasible")


def test_v4_feasible_reranking_skips_unselectable_highest_raw_rank(
    tmp_path: Path, monkeypatch,
) -> None:
    rehearse(tmp_path, monkeypatch, 98, v4=True, scenario="feasible_rerank")


def test_v4_multi_commit_successor_state(tmp_path: Path, monkeypatch) -> None:
    rehearse(tmp_path, monkeypatch, 93, v4=True, scenario="two_commits")


def test_v4_commit_no_commit_commit_successor_chain(tmp_path: Path, monkeypatch) -> None:
    rehearse(tmp_path, monkeypatch, 95, v4=True, scenario="transition_chain")


def test_v4_fake_provider_budget_stress_to_five_local_accepts(
    tmp_path: Path, monkeypatch,
) -> None:
    rehearse(tmp_path, monkeypatch, 94, v4=True, scenario="budget_stress")


def test_v4_proposal_ceiling_has_no_post_guard_assignment_or_provider(
    tmp_path: Path, monkeypatch,
) -> None:
    rehearse(tmp_path, monkeypatch, 96, v4=True, scenario="proposal_ceiling")


@pytest.mark.parametrize("scenario", ("transport_failure", "postprocess_failure"))
def test_v4_failed_physical_or_postprocess_attempt_is_durable_and_aborted(
    tmp_path: Path, monkeypatch, scenario: str,
) -> None:
    rehearse(tmp_path, monkeypatch, 97, v4=True, scenario=scenario, through_cli=True)


@pytest.mark.parametrize("scenario", ("minibatch_fail", "common_safe_fail", "shadow_fail"))
def test_v4_raw_legal_downstream_boundaries(tmp_path: Path, monkeypatch, scenario: str) -> None:
    rehearse(tmp_path, monkeypatch, 91, v4=True, scenario=scenario)


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
    assert process.returncode == 0, (process.stdout[-1500:], process.stderr[-4000:])


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


@pytest.mark.parametrize("rehearsal", range(3))
def test_v3_raw_legal_full_stack_exposes_packet_schedule_capacity(
    tmp_path: Path, monkeypatch, rehearsal: int,
) -> None:
    original = Layer2EvidenceRequestBuilder.build
    observed = []

    def capture(self, request, assignment):
        repair = tuple(row for row in assignment.evidence
                       if row.evidence_group == "repair" and
                       self._matches_primary_lane(row, assignment.primary_responsibility_lane))
        observed.append({
            "repair_count": len(repair),
            "local_eval_count": len(assignment.local_validation_example_ids),
            "metric_budget": request.local_metric_budget,
        })
        return original(self, request, assignment)

    monkeypatch.setattr(Layer2EvidenceRequestBuilder, "build", capture)
    with pytest.raises(ValueError, match="packet schedule must cover every selected search example"):
        rehearse(tmp_path, monkeypatch, rehearsal, v3=True)
    assert observed
    assert observed[0]["repair_count"] > observed[0]["metric_budget"]
    assert observed[0]["local_eval_count"] == 12


def test_v3_fresh_cli_also_exposes_packet_schedule_capacity(tmp_path: Path) -> None:
    source = Path(__file__).resolve()
    code = """
import importlib.util, pathlib, pytest, sys
spec = importlib.util.spec_from_file_location('full_fake_rehearsal', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
monkeypatch = pytest.MonkeyPatch()
try:
    with pytest.raises(ValueError, match='packet schedule must cover every selected search example'):
        module.rehearse(pathlib.Path(sys.argv[2]), monkeypatch, 80, through_cli=True, v3=True)
finally:
    monkeypatch.undo()
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(source), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], text=True,
        capture_output=True, timeout=180,
    )
    assert process.returncode == 0, (process.stdout[-1500:], process.stderr[-4000:])
