from __future__ import annotations

import asyncio
import json

import pytest

from multi_dataset_diverse_rl.compatibility_repair import (
    build_loss_blind_generic_revision_request,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


ARMS = {
    "experimental_v16_efficacy_g_matched": ("generic_peer_state", False, True),
    "experimental_v16_efficacy_r_m20": ("member_aware_responsibility_conditioned", False, False),
    "experimental_v16_efficacy_r_m2f": ("member_aware_responsibility_conditioned", True, False),
}


def test_formal_arm_protocols_are_exact_and_compute_matched():
    for setting, expected in ARMS.items():
        system = PromptEnsembleOptimizationSystem(Config.from_flat(
            out_dir="runs/test", experiment_setting=setting,
        ))
        protocol = system.protocol
        assert (protocol.tcs_context_policy, protocol.compatibility_repair_enabled, protocol.generic_revision_enabled) == expected
        assert protocol.target_branch_count == 2
        assert protocol.candidates_per_target_branch == 2
        assert protocol.candidate_acceptance_policy == "fixed_peer_monotone_target_or_vote"


def test_loss_blind_request_contains_no_outcome_or_responsibility_evidence():
    request = build_loss_blind_generic_revision_request(
        parent_prompt="parent-rule", source_candidate_prompt="source-rule",
    )
    payload = json.loads(request.split("RevisionInput:\n", 1)[1])
    assert set(payload) == {"parent_member_prompt", "source_candidate_prompt"}
    lowered = request.lower()
    for forbidden in ("question_hash", "gold_answer", "loss_evidence", "responsibility_evidence"):
        assert forbidden not in lowered


def test_provider_budget_stops_before_extra_call():
    calls = 0

    async def override(*_args):
        nonlocal calls
        calls += 1
        return "ok"

    cfg = Config.from_flat(provider_call_budget=1, total_token_budget=100)
    client = RoleAwareLLMClient(cfg, override=override)
    asyncio.run(client.chat("m", "s", "u", 0, 1, "optimizer"))
    with pytest.raises(RuntimeError, match="provider_call_budget_exhausted"):
        asyncio.run(client.chat("m", "s", "u", 0, 1, "optimizer"))
    assert calls == 1


def test_preregistration_freezes_new_seeds_and_train_only():
    spec = json.loads(open(
        "experiments/v16_module2_compute_matched_efficacy_20260813/preregistration.json",
        encoding="utf-8",
    ).read())
    assert spec["development_seed_excluded"] == 52
    assert spec["formal_seeds"] == [53, 54, 55]
    assert spec["validation_enabled"] is False
    assert spec["final_test_enabled"] is False
    assert spec["max_provider_calls_per_run"] == 8000
    assert spec["max_total_tokens_per_run"] == 3000000
