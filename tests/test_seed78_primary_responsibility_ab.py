from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import yaml
from openai import APIConnectionError

from infrastructure.common_solver_contract_v1.contract import CONTRACT_SPEC
from multi_dataset_diverse_rl.governance.manifest import (
    preregistration_hash,
    validate_manifest,
)
from multi_dataset_diverse_rl.responsibility import MemberAwareRepairOpportunity
from multi_dataset_diverse_rl.team_search.system_runtime import FrozenResponsibilitySnapshot
from scripts.run_seed78_primary_responsibility_ab import (
    ARM_A,
    ARM_B,
    AUTH_ENV,
    DurableLedger,
    Seed78System,
    _authorize,
    _arm_a_selection,
    _classify,
    _config,
    _bind_paired_validation_cache,
    _mechanism_metrics,
    _ledger_summary,
    _retryable,
    preflight,
    protocol_document,
)


ROOT = Path(__file__).parents[1]


def test_openai_sdk_connection_error_uses_frozen_transport_retry() -> None:
    error = APIConnectionError(
        request=httpx.Request("POST", "https://transport-audit.invalid")
    )
    assert _retryable(error) is True


def test_failed_attempt_rows_are_durable_without_inflating_logical_calls(
    tmp_path: Path,
) -> None:
    ledger = DurableLedger(tmp_path / "ledger.jsonl")
    common = {
        "phase": "initialization",
        "logical_role": "solver",
        "client_role": "solver",
        "cache_hit": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "seed": 78,
        "arm": "LEVEL_B_REAL_CANARY",
        "update_index": -1,
        "target_member": -1,
    }
    ledger.append({
        **common,
        "record_id": "failed-1",
        "record_kind": "solver_provider_attempt_failure",
        "provider_attempts": 1,
        "successful_provider_calls": 0,
    })
    ledger.append({
        **common,
        "record_id": "success-1",
        "record_kind": "solver_logical_completion",
        "provider_attempts": 1,
        "successful_provider_calls": 1,
    })
    assert _ledger_summary(ledger.path) == {
        "logical_calls": 1,
        "provider_attempts": 2,
        "successful_provider_calls": 1,
        "failed_provider_attempts": 1,
        "cache_hits": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_sdk_connection_failure_retries_four_times_and_persists_each_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenCompletions:
        async def create(self, **_kwargs):
            raise APIConnectionError(
                request=httpx.Request("POST", "https://transport-audit.invalid")
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": BrokenCompletions()})()

    monkeypatch.setattr(
        "scripts.run_seed78_primary_responsibility_ab.AsyncOpenAI", FakeClient
    )
    cfg = _config(
        tmp_path / "system",
        optimize_path=tmp_path / "optimize.csv",
        validation_path=tmp_path / "validation.csv",
    )
    monkeypatch.setenv(cfg.models.solver_api_key_env, "fake-key")
    monkeypatch.setenv(cfg.models.solver_base_url_env, "https://transport-audit.invalid")
    ledger = DurableLedger(tmp_path / "failed-ledger.jsonl")
    system = Seed78System(
        cfg,
        arm="LEVEL_B_REAL_CANARY",
        ledger=ledger,
        raw_cache={},
    )
    system._solver_stage = {"phase": "initialization"}

    async def scenario() -> None:
        with pytest.raises(APIConnectionError):
            await system.solve(
                "Who left?\nOptions:\n(A) Alex\n(B) Blair",
                0,
                "Use grammar and context to select the best option.",
            )

    asyncio.run(scenario())
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert len(rows) == CONTRACT_SPEC.transport_attempt_cap == 4
    assert [row["attempt_index"] for row in rows] == [1, 2, 3, 4]
    assert all(row["record_kind"] == "solver_provider_attempt_failure" for row in rows)
    assert all(row["error_type"] == "APIConnectionError" for row in rows)
    assert _ledger_summary(ledger.path)["provider_attempts"] == 4
    assert _ledger_summary(ledger.path)["failed_provider_attempts"] == 4


def _opportunity(member: int) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(
        agent_id=member,
        question_hash=f"q-{member}",
        vote_flip_gain=1,
        margin_gain=0,
        member_error=True,
        coverage_opportunity=False,
        conversion_opportunity=False,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def test_protocol_freezes_seed75_split_and_delayed_validation() -> None:
    protocol = protocol_document()
    assert protocol["seed"] == 78
    assert protocol["split_identity"] == {
        "optimize100": "anti_overfitting_split_v1/fold_a+fold_b",
        "shadow50": "anti_overfitting_split_v1/fold_c",
        "validation50": "anti_overfitting_split_v1/validation",
        "test50": "blocked_zero_calls",
    }
    assert protocol["only_difference"] == "target_scheduler"
    assert protocol["shared"]["always_two_targets"] is True
    assert "after_both_training_trajectories_freeze" in protocol["data_roles"]["validation50"]
    assert preflight()["gate"] == "PASS"
    assert preflight()["api_calls"] == 0


def test_execution_entry_point_fails_closed_without_runtime_authorization(monkeypatch) -> None:
    monkeypatch.delenv(AUTH_ENV, raising=False)
    with pytest.raises(PermissionError, match=AUTH_ENV):
        _authorize()


def test_retry1_manifest_authorization_passes_after_explicit_reauthorization(monkeypatch) -> None:
    monkeypatch.setenv(AUTH_ENV, "1")
    _authorize()


def test_run_identity_config_uses_real_frozen_split_paths(tmp_path: Path) -> None:
    optimize = tmp_path / "optimize100.csv"
    validation = tmp_path / "validation50.csv"
    optimize.write_text("question,answer\nq,A\n", encoding="utf-8")
    validation.write_text("question,answer\nv,B\n", encoding="utf-8")
    cfg = _config(tmp_path / "out", optimize_path=optimize, validation_path=validation)
    assert Path(cfg.data.train_path).is_file()
    assert Path(cfg.data.val_path).is_file()
    assert Path(cfg.data.train_path).resolve() == optimize.resolve()
    assert Path(cfg.data.val_path).resolve() == validation.resolve()


def test_arm_a_zero_score_fill_is_two_distinct_legal_fallback_targets() -> None:
    selected, lanes, _, rows = _arm_a_selection(
        FrozenResponsibilitySnapshot({}, {}, {}),
        update_index=4,
        cursor={},
    )
    assert len(selected) == len(set(selected)) == 2
    assert set(lanes.values()) == {"fallback"}
    assert all(row["fallback_reason"] == "always_two_targets_zero_score_fill" for row in rows)


def test_arm_a_preserves_scored_target_and_fills_only_second_slot() -> None:
    opportunity = _opportunity(3)
    selected, lanes, _, _ = _arm_a_selection(
        FrozenResponsibilitySnapshot(
            {3: (opportunity,)},
            {},
            {opportunity.question_hash: -1},
        ),
        update_index=0,
        cursor={},
    )
    assert selected[0] == 3
    assert len(selected) == len(set(selected)) == 2
    assert lanes[3] == "direct_flip"
    assert lanes[selected[1]] == "fallback"


def test_mechanism_accounting_and_frozen_classifier() -> None:
    def row(targets, committed):
        return {"selected_target_ids": targets, "committed_member_id": committed}

    a_traj = {"events": [row([0, 1], 0), row([0, 1], 0)]}
    b_traj = {"events": [row([0, 1], 0), row([0, 1], 1)]}
    a = _mechanism_metrics(a_traj, {"total_tokens": 200})
    b = _mechanism_metrics(b_traj, {"total_tokens": 100})
    assert a["target_commit_distribution_mismatch"] == pytest.approx(0.5)
    assert b["target_commit_distribution_mismatch"] == pytest.approx(0.0)
    assert a["max_unresolved_target_streak_by_member"]["1"] == 2
    assert b["max_unresolved_target_streak_by_member"]["1"] == 1
    a["validation50"] = {"vote_accuracy": 0.60}
    b["validation50"] = {"vote_accuracy": 0.60}
    assert _classify({ARM_A: a, ARM_B: b}) == "PRIMARY_RESPONSIBILITY_SCHEDULER_SUPPORTED"


def test_manifest_schema_and_preregistration_hash() -> None:
    manifest_path = ROOT / "experiments/manifests/seed78_primary_responsibility_ab_v1.yaml"
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    manifest = yaml.safe_load(manifest_path.read_text())
    recorded = manifest["artifacts"]["preregistration"]["sha256"]
    assert recorded == preregistration_hash(manifest)
    assert validate_manifest(manifest, schema) == []


def test_paired_validation_binds_one_exact_request_cache() -> None:
    class Common:
        def __init__(self) -> None:
            self.cache: dict[str, str] = {"old": "value"}

    class System:
        def __init__(self) -> None:
            self.common = Common()

    systems = {ARM_A: System(), ARM_B: System()}
    shared = _bind_paired_validation_cache(systems)  # type: ignore[arg-type]
    assert shared == {}
    assert systems[ARM_A].common.cache is shared
    assert systems[ARM_B].common.cache is shared
