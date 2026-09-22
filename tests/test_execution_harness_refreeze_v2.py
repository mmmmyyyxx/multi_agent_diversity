from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.governance.execution_harness_v2 import (
    ExecutionBindingError,
    FORMAL_SEEDS,
    INITIALIZATION_POLICY,
    LOCAL_NO_UPDATE_PATIENCE,
    PROVIDER_PROFILE,
    ROLE_MODEL,
    SOLVER_MODEL,
    TEAM_NO_UPDATE_PATIENCE,
    formal_dependencies_satisfied,
    fresh_initialization_identity,
    preflight_provider_binding,
)


ROOT = Path(__file__).resolve().parents[1]


def frozen(endpoint: str = "a" * 64):
    return {
        "provider_profile": "lwj",
        "endpoint_fingerprint": endpoint,
        "models": {
            "solver": {"model": SOLVER_MODEL, "status": "USED"},
            "teacher": {"model": None, "status": "UNUSED"},
            "critic": {"model": None, "status": "UNUSED"},
            "student": {"model": None, "status": "UNUSED"},
            "evaluator": {"model": ROLE_MODEL, "status": "USED"},
        },
    }


def lwj_config() -> Config:
    return Config.from_flat(
        provider_profile="lwj",
        agent_model=SOLVER_MODEL,
        optimizer_model=ROLE_MODEL,
        evaluator_model=ROLE_MODEL,
    )


def test_new_harness_fails_closed_on_missing_or_myx_profile():
    cfg = lwj_config()
    missing = dict(frozen())
    missing.pop("provider_profile")
    with pytest.raises(ExecutionBindingError, match="manifest provider_profile"):
        preflight_provider_binding(cfg, missing, environment={}, construct_client=False)

    myx = replace(cfg, models=replace(cfg.models, provider_profile="myx"))
    with pytest.raises(ExecutionBindingError, match="provider_profile must be lwj"):
        preflight_provider_binding(myx, frozen(), environment={}, construct_client=False)


def test_provider_preflight_constructs_no_request_and_persists_only_hash(monkeypatch):
    endpoint = "https://private.example.invalid/compatible-mode/v1"
    import hashlib

    digest = hashlib.sha256(endpoint.rstrip("/").encode()).hexdigest()
    constructed = []

    class FakeClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(
        "multi_dataset_diverse_rl.governance.execution_harness_v2.AsyncOpenAI",
        FakeClient,
    )
    result = preflight_provider_binding(
        lwj_config(),
        frozen(digest),
        environment={
            "LWJ_DASHSCOPE_API_KEY": "secret-never-persist",
            "LWJ_DASHSCOPE_BASE_URL": endpoint,
        },
    )
    assert len(constructed) == 1
    assert result.endpoint_fingerprint == digest
    assert result.raw_endpoint_persisted is False
    assert result.api_key_persisted is False
    assert "secret" not in str(result.to_dict())
    assert "private.example" not in str(result.to_dict())


def test_fresh_initialization_structural_replay_is_deterministic():
    args = dict(
        seed=80,
        canonical_prompt="canonical",
        question_hashes=["q2", "q1"],
        provider_profile=PROVIDER_PROFILE,
        endpoint_fingerprint="e" * 64,
        solver_model=SOLVER_MODEL,
        member_profile_hashes=[f"m{i}" for i in range(5)],
        team_state_hash="team",
    )
    first = fresh_initialization_identity(**args)
    second = fresh_initialization_identity(**args)
    assert first == second
    assert first["policy"] == INITIALIZATION_POLICY
    assert len(set(first["member_prompt_hashes"])) == 1
    assert first["focus"] == first["anchor"] == []


def test_new_runner_sources_do_not_reference_private_parent_freeze():
    for name in (
        "run_gepa_layer2_real_canary_v2.py",
        "run_sequential_symmetry_breaking_online_pilot_v2.py",
        "run_gepa_saturation_comparison_v2.py",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "selected_parent_tasks_private" not in source
        assert "provider_profile=PROVIDER_PROFILE" in source or name.startswith(
            "run_sequential"
        )


def test_formal_gate_and_frozen_constants():
    assert tuple(FORMAL_SEEDS) == (80, 81, 82)
    assert LOCAL_NO_UPDATE_PATIENCE == 3
    assert TEAM_NO_UPDATE_PATIENCE == 2
    assert not formal_dependencies_satisfied(
        canary_status="PENDING", sequential_status="PENDING"
    )
    assert formal_dependencies_satisfied(
        canary_status="TECHNICAL_PATH_CONFIRMED",
        sequential_status="REQUIRED_GATE_SATISFIED",
    )
