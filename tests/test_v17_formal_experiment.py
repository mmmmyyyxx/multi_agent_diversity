from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl import cli
from multi_dataset_diverse_rl.compatibility_repair import (
    build_loss_blind_generic_revision_request,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import load_checkpoint
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.run_task_level_accuracy import _parser, _validate_setting_sequence
from scripts import evaluate_v17_formal_final_states as final_eval
from scripts.v17_formal_support import (
    ARMS, EXPECTED_SPLITS, SEEDS, classify_three_seed,
    formal_target_schedule, recursive_sanitize, split_freeze,
)


def system(setting: str, *, seed: int = 56, out_dir: str = "runs/v17_test"):
    return PromptEnsembleOptimizationSystem(Config.from_flat(
        experiment_setting=setting, seed=seed, out_dir=out_dir,
    ))


def test_formal_seeds_dataset_hashes_sizes_and_disjointness():
    assert SEEDS == (56, 57, 58)
    frozen = split_freeze()
    assert frozen["gate"] == "PASS"
    assert frozen["sample_id_overlaps"] == {"opt_val": 0, "opt_test": 0, "val_test": 0}
    for name, (count, digest) in EXPECTED_SPLITS.items():
        assert frozen["splits"][name]["row_count"] == count
        assert frozen["splits"][name]["frozen_split_hash"] == digest


def test_s0_and_s1_protocols_are_exact():
    s0 = system(ARMS["S0"]).protocol
    s1 = system(ARMS["S1"]).protocol
    assert not s0.optimization_enabled and s0.target_branch_count == 0
    assert s1.target_branch_count == 2
    assert s1.candidates_per_target_branch == 2
    assert s1.candidate_budget_contract.total_generated_candidates_per_update == 4
    assert not s1.modules.member_aware_dual_target_search
    assert s1.target_selection_policy == "round_robin_dual_formal"
    assert s1.sample_pool_policy == "individual_errors"
    assert s1.tcs_context_policy == "generic_accuracy"
    assert s1.responsibility_refresh_policy == "off"
    assert not s1.service_routing_enabled
    assert s1.generic_revision_enabled
    assert not s1.compatibility_repair_enabled


@pytest.mark.parametrize("seed", SEEDS)
def test_s1_dual_round_robin_schedule(seed):
    instance = system(ARMS["S1"], seed=seed)
    observed = [list(instance.select_targets({}, update)[0]) for update in range(8)]
    assert observed == formal_target_schedule(seed)
    assert all(left != right for left, right in observed)
    assert len(observed) == 8


def test_s1_and_s2_share_generic_revision_without_loss_evidence():
    s1, s2 = system(ARMS["S1"]).protocol, system(ARMS["S2"]).protocol
    assert s1.generic_revision_enabled and s2.generic_revision_enabled
    assert s1.candidate_acceptance_policy == s2.candidate_acceptance_policy
    assert s1.candidate_ranking_policy == s2.candidate_ranking_policy
    assert s1.target_branch_count == s2.target_branch_count == 2
    assert s1.candidates_per_target_branch == s2.candidates_per_target_branch == 2
    request = build_loss_blind_generic_revision_request(
        parent_prompt="parent", source_candidate_prompt="candidate",
    ).lower()
    for forbidden in ("responsibility_evidence", "loss_evidence", "gold_answer", "question_hash"):
        assert forbidden not in request


def test_frozen_s2_s3_s4_protocol_identities():
    s2, s3, s4 = (system(ARMS[name]).protocol for name in ("S2", "S3", "S4"))
    assert s2.name == "experimental_v16_efficacy_g_matched"
    assert s3.name == "experimental_v16_efficacy_r_m20"
    assert s4.name == "experimental_v16_efficacy_r_m2f"
    assert s3.module2_evolution_variant == s4.module2_evolution_variant == "m20_current_v15"
    assert not s3.compatibility_repair_enabled and s4.compatibility_repair_enabled
    identity_fields = set(s3.__dict__) - {
        "name", "requested_name", "display_name", "compatibility_repair_enabled",
    }
    assert all(getattr(s3, field) == getattr(s4, field) for field in identity_fields)


def test_v17_setting_sequence_allowed_in_any_preregistered_order():
    for seed in SEEDS:
        settings = [ARMS[name] for name in __import__(
            "scripts.v17_formal_support", fromlist=["EXECUTION_ORDER"]
        ).EXECUTION_ORDER[seed]]
        _validate_setting_sequence(settings, optimized_only=False)


def test_runner_cache_mode_is_explicit():
    args = _parser().parse_args([
        "--manifest", "m", "--out_root", "o", "--immutable_comparison_cache", "1",
    ])
    assert args.immutable_comparison_cache == 1


def test_classifier_all_boundaries():
    assert classify_three_seed((1, 1, 1))["label"] == "CONSISTENT_POSITIVE"
    assert classify_three_seed((1, 1, 0))["label"] == "MAJORITY_POSITIVE"
    assert classify_three_seed((4, -1, -1))["label"] == "POSITIVE_MEAN_HETEROGENEOUS"
    assert classify_three_seed((1, 1, -3))["label"] == "MIXED_NONPOSITIVE"
    assert classify_three_seed((1, -1, 0))["label"] == "NOT_SUPPORTED"


def _split(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.jsonl"
    path.write_text(json.dumps({"question": f"{name}-q", "answer": "A"}) + "\n")
    return str(path)


def test_preserved_final_checkpoint_for_static_state(tmp_path, monkeypatch):
    async def solver(*_args):
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    class Offline(PromptEnsembleOptimizationSystem):
        def __init__(self, cfg): super().__init__(cfg, solver=solver)

    monkeypatch.setattr(cli, "PromptEnsembleOptimizationSystem", Offline)
    cfg = Config.from_flat(
        experiment_setting=ARMS["S0"], train_path=_split(tmp_path, "train"),
        val_path=_split(tmp_path, "val"), test_path=_split(tmp_path, "test"),
        train_size=1, val_size=1, test_size=1, answer_format="option_letter",
        final_test_enabled=False, preserve_final_checkpoint=True,
        out_dir=str(tmp_path / "run"),
    )
    asyncio.run(cli.run(cfg))
    checkpoint = load_checkpoint(tmp_path / "run" / "training_checkpoint.json")
    assert checkpoint is not None
    assert checkpoint["completed_update_count"] == 0


def test_final_state_evaluator_restores_without_mutation(tmp_path, monkeypatch):
    async def solver(*_args):
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    class Offline(PromptEnsembleOptimizationSystem):
        def __init__(self, cfg): super().__init__(cfg, solver=solver)

    monkeypatch.setattr(cli, "PromptEnsembleOptimizationSystem", Offline)
    monkeypatch.setattr(final_eval, "PromptEnsembleOptimizationSystem", Offline)
    run = tmp_path / "run"
    cfg = Config.from_flat(
        experiment_setting=ARMS["S0"], train_path=_split(tmp_path, "train2"),
        val_path=_split(tmp_path, "val2"), test_path=_split(tmp_path, "test2"),
        train_size=1, val_size=1, test_size=1, answer_format="option_letter",
        final_test_enabled=False, preserve_final_checkpoint=True, out_dir=str(run),
    )
    asyncio.run(cli.run(cfg))
    before = (run / "training_checkpoint.json").read_bytes()
    result = asyncio.run(final_eval.evaluate_cell(run, "validation", tmp_path / "eval"))
    assert result["logical_evaluation_count"] == 1
    assert result["provider_calls"] == 1
    assert result["state_mutation"] is False
    assert result["checkpoint_mutation"] is False
    assert (run / "training_checkpoint.json").read_bytes() == before


def test_preregistration_exposure_and_phase_authorizations():
    spec = json.loads(Path(
        "experiments/v17_formal_5arm_3seed_20260813/preregistration.json"
    ).read_text(encoding="utf-8"))
    assert spec["paper_untouched_test"] is False
    assert spec["historical_test_exposure"] is True
    assert spec["validation_selection_enabled"] is False
    assert spec["phase_authorizations"] == {
        "train": "V17_FORMAL_TRAIN_AUTHORIZED",
        "validation": "V17_FORMAL_VALIDATION_AUTHORIZED",
        "test": "V17_FORMAL_TEST_AUTHORIZED",
    }
    assert spec["max_provider_calls_per_run"] == 8000
    assert spec["max_total_tokens_per_run"] == 3_000_000


@pytest.mark.parametrize("payload", [
    {"prompt": "secret"},
    {"value": "D:\\private\\file.json"},
    {"value": "FINAL_ANSWER: A"},
])
def test_report_sanitizer_rejects_sensitive_content(payload):
    assert recursive_sanitize(payload)
