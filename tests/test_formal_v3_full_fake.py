"""Zero-API rehearsal of the formal V3 production composition."""

from __future__ import annotations

import asyncio
import copy
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.governance.production_execution import ValidatedExecutionContext
from multi_dataset_diverse_rl.production_formal_saturation import execute_formal_gepa_saturation
from multi_dataset_diverse_rl.formal_trajectory import derive_formal_trajectory_records
from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalOptimizationResult, LocalPromptCandidate, OpaqueOptimizerState,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary
from multi_dataset_diverse_rl.team_search.evidence_audit import sanitize_v4_evidence_trace
from multi_dataset_diverse_rl.team_search.system_runtime import SystemTeamCandidateEvaluator
from multi_dataset_diverse_rl.shadow_gate import ShadowGateMetrics, evaluate_shadow_gate
from multi_dataset_diverse_rl.team_search.candidate_evaluator import EvaluationCost
from multi_dataset_diverse_rl.versions import LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION
from multi_dataset_diverse_rl.governance.production_execution import formal_v3_contract
from scripts.prepare_gepa_saturation_comparison_v3 import frozen_payload
from scripts.prepare_post_refactor_gepa_canary import _private_splits


@pytest.fixture(autouse=True)
def _no_real_provider(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("real provider construction is forbidden in fake formal test")
    monkeypatch.setattr(ProviderClientFactory, "from_environment", forbidden)
    monkeypatch.setattr(ProviderClientFactory, "create", forbidden)


def _formal_fixture(
    tmp_path: Path, monkeypatch, *, scope: str, ceiling: int = 6000,
    optimizer_ceiling: int = 1000, solver_failure: str | None = None,
    candidate_improves: bool = False,
):
    # Exercise RoleAwareLLMClient's normal credential-resolution and factory
    # path without placing a usable secret in the subprocess environment.
    monkeypatch.setattr(
        "multi_dataset_diverse_rl.llm_client.resolve_api_key",
        lambda _configured, _profile: ("OFFLINE_FAKE_KEY_NAME", "OFFLINE_FAKE_VALUE"),
    )
    monkeypatch.setattr(
        "multi_dataset_diverse_rl.llm_client.resolve_base_url",
        lambda _configured, _profile: ("OFFLINE_FAKE_BASE_NAME", "https://invalid.example"),
    )
    prep, run_root = tmp_path / "prep", tmp_path / "run"
    prep.mkdir()
    run_root.mkdir()
    _private_splits(prep)
    (run_root / "run_lifecycle.json").write_text(json.dumps({
        "status": "RUNNING", "provider_client_constructed": False,
        "events": [{"status": "RUNNING"}],
    }), encoding="utf-8")
    manifest = {
        "experiment_id": f"gepa_saturation_comparison_v3_seed80_{scope}",
        "scientific": {
            "backend": "gepa", "optimization_scope": scope,
            "stopping_regime": "saturation", "task_identity": "bbh_disambiguation_qa",
            "data_identity": "fake_optimize100_shadow50", "fixed_budget_units": 1,
            "local_no_update_patience": 3, "team_no_update_patience": 2,
            "layer2_protocol_version": (
                LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION if scope == "layer2" else None
            ),
            "emergency_max_provider_calls": ceiling,
            "emergency_max_optimizer_steps": optimizer_ceiling,
            "emergency_max_team_epochs": 100,
            "emergency_max_wall_seconds": 3600,
        },
        "runtime": {
            "seed": 80, "solver_model": "qwen3-8b",
            "optimizer_model": "qwen3.7-flash", "evaluator_model": "qwen3.7-flash",
        },
    }
    (prep / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    labels = {}
    for name in ("optimize100.csv", "shadow50.csv"):
        with (prep / "splits_private" / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                labels[row["question"].replace("\r\n", "\n").strip()] = row["answer"].strip("()")
    calls = {"solver": 0, "reflection": 0, "create": 0, "from_environment": 0}

    class FakeCompletion:
        async def create(self, **request):
            if request["model"] == "qwen3.7-flash":
                calls["reflection"] += 1
                content = (
                    "Identify each candidate referent and compare pronoun agreement, "
                    "then use contextual evidence before selecting the answer."
                )
            else:
                calls["solver"] += 1
                if solver_failure == "transport":
                    raise ConnectionError("fake transport failure")
                if solver_failure == "postprocess" and calls["solver"] == 1:
                    class BrokenUsage:
                        @property
                        def prompt_tokens(self):
                            raise ValueError("fake postprocess failure")
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="offline"), finish_reason="stop")],
                        usage=BrokenUsage(),
                    )
                question = request["messages"][1]["content"]
                gold = labels[question]
                score = hashlib.sha256(question.encode("utf-8")).digest()[0]
                procedure = request["messages"][0]["content"]
                answer = (
                    gold if candidate_improves and "Check pronoun agreement against each referent." in procedure
                    else gold if score < 150 else ("B" if gold == "A" else "A")
                )
                content = f"FINAL_ANSWER: {answer}"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
            )

    fake = SimpleNamespace(_formal_offline_sentinel=True, chat=SimpleNamespace(completions=FakeCompletion()))
    def fake_from_environment(*_args, **_kwargs):
        calls["from_environment"] += 1
        return fake
    def fake_create(*_args, **_kwargs):
        calls["create"] += 1
        return fake
    monkeypatch.setattr(ProviderClientFactory, "from_environment", fake_from_environment)
    monkeypatch.setattr(ProviderClientFactory, "create", fake_create)
    permit = ValidatedExecutionContext(
        experiment_id=manifest["experiment_id"], attempt_id=manifest["experiment_id"],
        execution_source_sha="a" * 40, preregistration_sha256="b" * 64,
        run_identity_sha256="c" * 64, provider_profile="lwj",
        endpoint_fingerprint="fake-endpoint", allowed_phase="formal",
        allowed_roles=("solver", "reflection"), prep_root=prep, run_root=run_root,
    )
    return permit, calls


def _assert_success_ledger(permit, calls):
    rows = [json.loads(line) for line in (
        permit.run_root / "ledger.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert all(row["phase"] and row["logical_role"] for row in rows)
    assert all(row["input_tokens"] + row["output_tokens"] == row["total_tokens"] for row in rows)
    summary = ledger_summary(permit.run_root / "ledger.jsonl")
    assert summary["provider_attempts"] == calls["solver"] + calls["reflection"]
    assert summary["successful_provider_calls"] == summary["provider_attempts"]
    assert summary["failed_provider_attempts"] == 0
    assert summary["input_tokens"] + summary["output_tokens"] == summary["total_tokens"]
    assert summary["cache_hits"] == sum(bool(row["cache_hit"]) for row in rows)
    assert sum(row["provider_attempts"] for row in rows) == summary["provider_attempts"]
    assert sum(row["successful_provider_calls"] for row in rows) == summary["successful_provider_calls"]
    assert calls["from_environment"] == 1
    assert calls["create"] == int(calls["reflection"] > 0)
    return rows, summary


def _capture_case(case: str, permit, calls, result):
    """Optional private sanitized evidence for the offline closure report."""

    destination = os.environ.get("FORMAL_V3_EVIDENCE_CAPTURE_DIR")
    if not destination:
        return
    rows = [json.loads(line) for line in (
        permit.run_root / "ledger.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    stages = {}
    roles = {}
    for row in rows:
        for bucket, key in ((stages, row["phase"]), (roles, row["logical_role"])):
            totals = bucket.setdefault(key, {
                "provider_attempts": 0, "successful_provider_calls": 0,
                "cache_hits": 0, "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0,
            })
            for field in totals:
                totals[field] += int(bool(row["cache_hit"])) if field == "cache_hits" else int(row[field])
    payload = {
        "case": case, "stop_reason": result["stop_reason"],
        "fake_physical_calls": {"solver": calls["solver"], "reflection": calls["reflection"]},
        "provider_constructor_calls": {
            "from_environment": calls["from_environment"], "create": calls["create"],
        },
        "ledger": ledger_summary(permit.run_root / "ledger.jsonl"),
        "stages": stages, "roles": roles,
        "events": [{
            "index": event["index"], "kind": event["kind"],
            "stop_reason": event["stop_reason"],
            "committed": event["committed_candidate_id"] is not None,
            "team_no_update_counter": (event["telemetry"].get("saturation") or {}).get(
                "team_no_update_counter"
            ),
            "epoch_end_reason": event["telemetry"].get("epoch_end_reason"),
            "backend_termination_reason": event["telemetry"].get("backend_termination_reason"),
            "local_no_update_counter": event["telemetry"].get("backend_saturation", {}).get(
                "local_no_update_counter"
            ) if isinstance(event["telemetry"].get("backend_saturation"), dict) else None,
        } for event in result["events"]],
        "evidence": [
            sanitize_v4_evidence_trace(row)
            if "evidence_delivered" in row else {
                "update_index": row["update_index"],
                "delivery_not_reached_due_to_emergency": True,
                "responsibility_scheduled_count": len(row["responsibility_scheduled_ids"]),
            }
            for row in result.get("evidence_view_trace", [])
        ],
        "validation50_calls": result["validation50_calls"],
        "test50_calls": result["test50_calls"],
    }
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{case}.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )


def _inject_one_local_candidate(monkeypatch):
    async def local_candidate(_self, task, **_kwargs):
        child = task.parent_prompt + "\nCheck pronoun agreement against each referent."
        return LocalOptimizationResult(
            candidates=(LocalPromptCandidate(
                candidate_id="fake-local-child", prompt=child, local_score=0.9,
                per_example_scores={}, parent_ids=(), generation=1,
                backend_metadata={"local_acceptance_delta": 0.1,
                                  "local_parent_score": 0.8,
                                  "local_full_validation_delta": 0.1},
            ),),
            backend_name="gepa", backend_version="fake",
            optimizer_state=OpaqueOptimizerState("gepa", "fake", {
                "telemetry": {"accepted_mutations": 1},
                "saturation": {"optimizer_steps": 1},
            }),
            solver_calls=0, optimizer_calls=0, input_tokens=0, output_tokens=0,
            total_tokens=0, termination_reason="SATURATION_REACHED",
        )
    monkeypatch.setattr(GEPALocalPromptOptimizer, "optimize_saturation", local_candidate)


def test_formal_fake_fixture_blocks_both_unpatched_provider_factories():
    with pytest.raises(AssertionError, match="real provider construction is forbidden"):
        ProviderClientFactory.create(api_key="fake", base_url="https://invalid.example")
    with pytest.raises(AssertionError, match="real provider construction is forbidden"):
        ProviderClientFactory.from_environment(provider_profile="lwj")


@pytest.mark.parametrize("seed", [80, 81, 82])
@pytest.mark.parametrize("scope", ["native", "layer2"])
def test_formal_v3_freeze_binds_active_method_and_keeps_execution_gated(
    monkeypatch, seed, scope,
):
    monkeypatch.setattr(
        "scripts.prepare_gepa_saturation_comparison_v3.endpoint_fingerprint_from_environment",
        lambda: "f" * 64,
    )
    manifest, protocol = frozen_payload(execution_source_sha="a" * 40, seed=seed, scope=scope)
    assert manifest["formal_v3_contract"] == protocol["formal_v3_contract"] == formal_v3_contract()
    assert manifest["experiment_id"] == f"gepa_saturation_comparison_v3_seed{seed}_{scope}_attempt1"
    assert manifest["api_authorization"]["authorized"] is False
    assert manifest["execution_gate"]["real_v4_diagnostic"] != "SCIENTIFICALLY_VALID"
    assert manifest["scientific"]["stopping_regime"] == "saturation"
    assert manifest["scientific"]["layer2_protocol_version"] == (
        LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION if scope == "layer2" else None
    )
    assert protocol["validation50_calls"] == protocol["test50_calls"] == 0


def test_formal_fake_fixture_replaces_both_provider_factories(tmp_path, monkeypatch):
    _permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="native")
    direct = ProviderClientFactory.create(api_key="fake", base_url="https://invalid.example")
    solver = ProviderClientFactory.from_environment(provider_profile="lwj")
    assert direct is solver
    assert direct._formal_offline_sentinel is True
    assert calls["create"] == calls["from_environment"] == 1


def test_formal_v3_layer2_no_feasible_is_scientific_zero_opportunity_delta(tmp_path, monkeypatch):
    from multi_dataset_diverse_rl.team_search.feasibility import (
        Layer2EvidenceInfeasible, Layer2FeasibilityReason,
    )
    from multi_dataset_diverse_rl.team_search.system_runtime import SystemResponsibilityAssignmentFactory

    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="layer2")
    def impossible(self, **kwargs):
        del self, kwargs
        raise Layer2EvidenceInfeasible(Layer2FeasibilityReason.INSUFFICIENT_PRIMARY_REPAIR_QUOTA)
    monkeypatch.setattr(SystemResponsibilityAssignmentFactory, "build_from_member", impossible)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert result["stop_reason"] == "NO_FEASIBLE_LAYER2_OPPORTUNITY"
    assert result["events"] == []
    assert result["commits"] == 0
    # Homogeneous P0 reuses the same provider realization across five members.
    assert calls["solver"] == 100
    assert calls["reflection"] == 0
    assert result["ledger"]["provider_attempts"] == 100
    assert result["ledger"]["cache_hits"] == 400
    assert derive_formal_trajectory_records(result)[0]["record_type"] == "LAYER2_NO_FEASIBLE_STOP"
    _assert_success_ledger(permit, calls)
    _capture_case("layer2_no_feasible", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_native_durable_ceiling_preempts_search_after_initialization(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="native", ceiling=100)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert result["stop_reason"] == "EMERGENCY_PROVIDER_CALL_CEILING"
    assert result["events"][0]["telemetry"]["backend_termination_reason"] == "pre_native_durable_ceiling"
    assert calls["solver"] == 100 and calls["reflection"] == 0
    assert calls["from_environment"] == 1 and calls["create"] == 0
    assert result["ledger"]["cache_hits"] == 400
    _assert_success_ledger(permit, calls)
    _capture_case("native_provider_emergency", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_native_optimizer_emergency_is_not_saturation(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(
        tmp_path, monkeypatch, scope="native", optimizer_ceiling=1,
    )
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert result["stop_reason"] == "EMERGENCY_OPTIMIZER_STEP_CEILING"
    assert result["events"][0]["telemetry"]["backend_termination_reason"] == result["stop_reason"]
    _assert_success_ledger(permit, calls)
    _capture_case("native_optimizer_emergency", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_native_unexpected_backend_return_is_not_saturation(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="native")

    async def unexpected(_self, _task, **_kwargs):
        return LocalOptimizationResult(
            candidates=(), backend_name="gepa", backend_version="fake",
            optimizer_state=OpaqueOptimizerState("gepa", "fake", {
                "telemetry": {"accepted_mutations": 0},
                "saturation": {"optimizer_steps": 1},
            }),
            solver_calls=0, optimizer_calls=0, input_tokens=0, output_tokens=0,
            total_tokens=0, termination_reason="UNEXPECTED_BACKEND_RETURN",
        )

    monkeypatch.setattr(GEPALocalPromptOptimizer, "optimize_saturation", unexpected)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert result["stop_reason"] == "OPERATIONAL_ABORT"
    assert result["events"][0]["telemetry"]["backend_termination_reason"] == "UNEXPECTED_BACKEND_RETURN"
    assert calls["solver"] == 100 and calls["reflection"] == 0
    _assert_success_ledger(permit, calls)
    _capture_case("native_unexpected_return", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


@pytest.mark.parametrize("failure,expected", [
    ("transport", ConnectionError), ("postprocess", ValueError),
])
def test_formal_v3_initialization_failure_is_durable_and_not_saturation(
    tmp_path, monkeypatch, failure, expected,
):
    permit, calls = _formal_fixture(
        tmp_path, monkeypatch, scope="native", solver_failure=failure,
    )
    with pytest.raises(expected, match=f"fake {failure} failure"):
        asyncio.run(execute_formal_gepa_saturation(
            permit, root=Path(__file__).resolve().parents[1],
        ))
    rows = [json.loads(line) for line in (
        permit.run_root / "ledger.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = ledger_summary(permit.run_root / "ledger.jsonl")
    assert summary["provider_attempts"] == calls["solver"]
    assert summary["cache_hits"] == sum(bool(row["cache_hit"]) for row in rows)
    assert len({row["record_id"] for row in rows}) == len(rows)
    if failure == "transport":
        assert summary["cache_hits"] == 0
        assert summary["successful_provider_calls"] == 0
        assert summary["failed_provider_attempts"] == summary["provider_attempts"]
    else:
        assert summary["provider_attempts"] == summary["successful_provider_calls"] == calls["solver"]
        assert summary["failed_provider_attempts"] == 0
        assert sum(row.get("postprocess_failed") is True for row in rows) == 1


def test_formal_v3_native_reaches_factual_local_saturation(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="native", ceiling=6000)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert result["stop_reason"] == "SATURATION_REACHED"
    event = result["events"][0]
    assert event["telemetry"]["backend_termination_reason"] == "SATURATION_REACHED"
    assert event["telemetry"]["backend_saturation"]["local_no_update_counter"] == 3
    assert calls["reflection"] > 0
    assert calls["from_environment"] == 1 and calls["create"] == 1
    assert result["ledger"]["provider_attempts"] == calls["solver"] + calls["reflection"]
    assert derive_formal_trajectory_records(result)[0]["record_type"] == "NATIVE_OPTIMIZATION_UNIT"
    _assert_success_ledger(permit, calls)
    _capture_case("native_saturation", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_layer2_reaches_factual_team_saturation(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="layer2", ceiling=6000)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert result["stop_reason"] == "SATURATION_REACHED"
    assert result["events"]
    assert result["events"][-1]["telemetry"]["saturation"]["team_no_update_counter"] == 2
    evidence = [sanitize_v4_evidence_trace(row) for row in result["evidence_view_trace"]]
    assert evidence
    assert all(row["responsibility_universe_count"] >= row["responsibility_scheduled_count"]
               and row["nominal_role_item_slots"] <= 36 for row in evidence)
    assert all(row["scheduled_but_not_delivered_count"] >= 0 for row in evidence)
    trace = derive_formal_trajectory_records(result)
    assert len([row for row in trace if row["record_type"] == "LAYER2_OPPORTUNITY"]) == len(result["events"])
    corrupted = copy.deepcopy(result)
    corrupted["evidence_view_trace"][0]["packet_V"] += 1
    with pytest.raises(ValueError, match="packet identity"):
        derive_formal_trajectory_records(corrupted)
    assert calls["reflection"] > 0
    assert calls["from_environment"] == 1 and calls["create"] == 1
    assert result["ledger"]["provider_attempts"] == calls["solver"] + calls["reflection"]
    _assert_success_ledger(permit, calls)
    _capture_case("layer2_saturation", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_layer2_local_emergency_preempts_team_admission(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="layer2")

    async def emergency(_self, _task, **_kwargs):
        return LocalOptimizationResult(
            candidates=(), backend_name="gepa", backend_version="fake",
            optimizer_state=OpaqueOptimizerState("gepa", "fake", {
                "telemetry": {"accepted_mutations": 0},
                "saturation": {"stop_reason": "EMERGENCY_OPTIMIZER_STEP_CEILING"},
            }),
            solver_calls=0, optimizer_calls=0, input_tokens=0, output_tokens=0,
            total_tokens=0, termination_reason="EMERGENCY_OPTIMIZER_STEP_CEILING",
        )
    monkeypatch.setattr(GEPALocalPromptOptimizer, "optimize_saturation", emergency)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert result["stop_reason"] == "EMERGENCY_OPTIMIZER_STEP_CEILING"
    assert result["events"] == [] and result["commits"] == 0
    assert calls["solver"] == 100 and calls["reflection"] == 0
    assert result["ledger"]["cache_hits"] == 400
    _assert_success_ledger(permit, calls)
    _capture_case("layer2_local_emergency", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_layer2_local_positive_minibatch_rejects_with_read_only_full(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="layer2")
    _inject_one_local_candidate(monkeypatch)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert result["stop_reason"] == "SATURATION_REACHED"
    assert result["events"] and result["commits"] == 0
    rows, summary = _assert_success_ledger(permit, calls)
    phases = {row["phase"] for row in rows}
    assert "team_minibatch_eval" in phases
    assert "diagnostic_full_eval" in phases
    assert "team_full_eval" not in phases and "team_shadow_eval" not in phases
    assert result["ledger"] == summary
    assert result["ledger"]["provider_attempts"] == calls["solver"] + calls["reflection"]
    trace = derive_formal_trajectory_records(result)
    rejected = [row for row in trace if row["record_type"] == "LOCAL_ACCEPTED_CANDIDATE"]
    assert rejected and all(row["full"]["diagnostic_only"] for row in rejected)
    assert all(row["ordinary_common_safe"] == "NOT_REACHED" for row in rejected)
    _capture_case("layer2_minibatch_rejection", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_layer2_common_safe_rejection_skips_shadow(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="layer2")
    _inject_one_local_candidate(monkeypatch)
    original = SystemTeamCandidateEvaluator.evaluate_minibatch

    def positive_minibatch(self, assignment, candidate, minibatch):
        metrics, cost = original(self, assignment, candidate, minibatch)
        return replace(metrics, target_delta=1, responsibility_delta=1), cost

    monkeypatch.setattr(SystemTeamCandidateEvaluator, "evaluate_minibatch", positive_minibatch)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    rows, _summary = _assert_success_ledger(permit, calls)
    phases = {row["phase"] for row in rows}
    assert result["commits"] == 0
    assert "team_minibatch_eval" in phases and "team_full_eval" in phases
    assert "team_shadow_eval" not in phases
    _capture_case("layer2_common_safe_rejection", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_layer2_shadow_rejection_does_not_commit(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(
        tmp_path, monkeypatch, scope="layer2", candidate_improves=True,
    )
    _inject_one_local_candidate(monkeypatch)
    shadow_checks = {"count": 0}

    def reject_shadow(self, assignment, candidate):
        del self, assignment, candidate
        shadow_checks["count"] += 1
        return evaluate_shadow_gate(ShadowGateMetrics(40, 39, 30, 29, 50)), EvaluationCost(0, 0, 0)

    monkeypatch.setattr(SystemTeamCandidateEvaluator, "evaluate_shadow", reject_shadow)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    rows, _summary = _assert_success_ledger(permit, calls)
    phases = {row["phase"] for row in rows}
    assert shadow_checks["count"] > 0
    assert result["commits"] == 0
    assert "team_minibatch_eval" in phases and "team_full_eval" in phases
    _capture_case("layer2_shadow_rejection", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_layer2_successful_commit_restarts_from_successor(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(
        tmp_path, monkeypatch, scope="layer2", candidate_improves=True,
    )
    _inject_one_local_candidate(monkeypatch)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    _assert_success_ledger(permit, calls)
    assert result["commits"] >= 1
    first_commit = next(index for index, row in enumerate(result["events"])
                        if row["committed_candidate_id"] is not None)
    first = result["events"][first_commit]
    assert first["telemetry"]["epoch_end_reason"] == "TEAM_COMMIT"
    assert first["telemetry"]["saturation"]["team_no_update_counter"] == 0
    assert first["state_hash"] != result["initial_team_hash"]
    trace = derive_formal_trajectory_records(result)
    committed = [row for row in trace if row["record_type"] == "LAYER2_OPPORTUNITY"
                 and row["transition"]["committed_candidate_id"] is not None]
    assert committed and committed[0]["transition"]["candidate_transition"] is not None
    if first_commit + 1 < len(result["events"]):
        assert result["events"][first_commit + 1]["request_identity"] != first["request_identity"]
    _capture_case("layer2_commit_restart", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0


def test_formal_v3_layer2_partial_sampler_delivery_is_observed_not_inferred(tmp_path, monkeypatch):
    permit, calls = _formal_fixture(tmp_path, monkeypatch, scope="layer2")

    async def partial_delivery(_self, task, *, batch_sampler, **_kwargs):
        loader = SimpleNamespace(all_ids=lambda: tuple(range(len(task.search_examples))))
        batch_sampler.next_minibatch_ids(loader, None)
        return LocalOptimizationResult(
            candidates=(), backend_name="gepa", backend_version="fake",
            optimizer_state=OpaqueOptimizerState("gepa", "fake", {
                "telemetry": {"accepted_mutations": 0},
                "saturation": {"optimizer_steps": 1},
            }),
            solver_calls=0, optimizer_calls=0, input_tokens=0, output_tokens=0,
            total_tokens=0, termination_reason="SATURATION_REACHED",
        )

    monkeypatch.setattr(GEPALocalPromptOptimizer, "optimize_saturation", partial_delivery)
    result = asyncio.run(execute_formal_gepa_saturation(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    evidence = [sanitize_v4_evidence_trace(row) for row in result["evidence_view_trace"]]
    assert result["stop_reason"] == "SATURATION_REACHED"
    assert evidence and all(row["delivered_batch_count"] == 1 for row in evidence)
    assert any(row["scheduled_but_not_delivered_count"] > 0 for row in evidence)
    assert all(row["responsibility_universe_count"] >= row["responsibility_scheduled_count"]
               and row["nominal_role_item_slots"] <= 36 for row in evidence)
    _assert_success_ledger(permit, calls)
    _capture_case("layer2_partial_delivery", permit, calls, result)
    assert result["validation50_calls"] == result["test50_calls"] == 0
