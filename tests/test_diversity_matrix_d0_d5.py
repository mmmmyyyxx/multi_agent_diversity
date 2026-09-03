from __future__ import annotations

from pathlib import Path

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.governance.manifest import validate_manifest
from multi_dataset_diverse_rl.governance.registries import load_yaml
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.analyze_diversity_matrix_d0_d5 import (
    _support_accumulation,
    _validation_diversity,
)
from scripts.diversity_matrix_d0_d5_support import (
    ARMS,
    ARM_ORDER,
    ROLE_MODEL,
    SOLVER_MODEL,
    classifier,
    manifest,
    recursive_sanitize,
)
from scripts.run_diversity_matrix_d0_d5 import _blocked_test_loader, _config


ROOT = Path(__file__).resolve().parents[1]


def _system(arm: str, seed: int = 72) -> PromptEnsembleOptimizationSystem:
    return PromptEnsembleOptimizationSystem(Config.from_flat(
        experiment_setting=ARMS[arm]["setting"], seed=seed,
        out_dir="runs/diversity_matrix_unit_test",
    ))


def test_matrix_manifest_and_frozen_models_are_valid() -> None:
    schema = load_yaml(ROOT / "infrastructure/experiment_manifest.schema.json")
    assert validate_manifest(manifest(), schema) == []
    assert ARM_ORDER == ("D0", "D1", "D2", "D3", "D4", "D5")
    assert SOLVER_MODEL == "qwen3-14b"
    assert ROLE_MODEL == "qwen3.7-flash"
    assert manifest()["seeds"] == [72, 73, 74]


def test_factorial_protocol_is_incremental_and_compute_matched() -> None:
    protocols = {arm: _system(arm).protocol for arm in ARM_ORDER}
    assert not protocols["D0"].optimization_enabled
    assert protocols["D1"].target_branch_count == 1
    for arm in ("D2", "D3", "D4", "D5"):
        protocol = protocols[arm]
        assert protocol.target_branch_count == 2
        assert protocol.candidates_per_target_branch == 2
        assert protocol.generic_revision_enabled
        assert not protocol.compatibility_repair_enabled
        assert protocol.candidate_acceptance_policy == "fixed_peer_monotone_target_or_vote"
        assert protocol.candidate_ranking_policy == "common_monotone_safe"
    assert protocols["D2"].target_selection_policy == protocols["D4"].target_selection_policy == "responsibility_round_robin_dual"
    assert protocols["D3"].target_selection_policy == protocols["D5"].target_selection_policy == "repairability_adjusted_responsibility"
    assert protocols["D2"].tcs_context_policy == protocols["D3"].tcs_context_policy == "generic_peer_state"
    assert protocols["D4"].tcs_context_policy == protocols["D5"].tcs_context_policy == "member_aware_responsibility_conditioned"


def test_eligible_round_robin_is_deterministic_and_never_borrows() -> None:
    instance = _system("D2", seed=72)
    assignments = {0: [object()], 2: [object()], 4: [object()]}
    schedules = [instance.select_targets(assignments, update)[0] for update in range(5)]
    assert schedules == [(2, 4), (4, 0), (2, 4), (4, 0), (0, 2)]
    assert all(set(targets).issubset(assignments) for targets in schedules)


def test_matrix_config_freezes_32_updates_and_no_test(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    config = _config(
        seed=72, arm="D5", out_dir=tmp_path / "out",
        cache_path=tmp_path / "cache.sqlite", frozen_initialization=frozen,
        resume=False,
    )
    assert config.training.epochs == 4
    assert config.training.update_every == 10
    assert config.models.agent_model == SOLVER_MODEL
    assert config.models.optimizer_model == config.models.evaluator_model == ROLE_MODEL
    assert config.persistence.final_test_enabled is False
    original, blocked = _blocked_test_loader(Path(config.data.test_path))
    assert blocked(config.data.test_path, 125, config.data.dataset_format) == []
    assert original is not blocked


def test_classifier_is_frozen_and_sanitizer_rejects_secrets() -> None:
    assert classifier([1, 1, 1])["label"] == "CONSISTENT_POSITIVE"
    assert classifier([1, 1, -1])["label"] == "MAJORITY_POSITIVE"
    assert classifier([0, 0, 0])["label"] == "NEUTRAL"
    assert classifier([-1, -1, 1])["label"] == "NEGATIVE"
    assert classifier([2, -1, -1])["label"] == "MIXED"
    assert recursive_sanitize({"endpoint": "secret"})
    assert recursive_sanitize({"safe_hash": "abc"}) == []


def test_analysis_metrics_have_fixed_semantics() -> None:
    validation = [
        {"G": 1, "member_correctness": [True, False, False, False, False]},
        {"G": 2, "member_correctness": [True, True, False, False, False]},
        {"G": 0, "member_correctness": [False, False, False, False, False]},
    ]
    metrics = _validation_diversity(validation)
    assert metrics["coverage_depth"] == [1, 1, 1, 0, 0, 0]
    assert metrics["unique_coverage_count"] == 1
    transitions = _support_accumulation([
        {"update_index": 0, "question_hash": "a", "target_agent_id": 0, "G_before": 0, "G_after": 1, "target_correct_before": False, "target_correct_after": True, "vote_correct_before": False, "vote_correct_after": False},
        {"update_index": 1, "question_hash": "a", "target_agent_id": 2, "G_before": 1, "G_after": 2, "target_correct_before": False, "target_correct_after": True, "vote_correct_before": False, "vote_correct_after": True},
    ])
    assert transitions["zero_to_one_recoveries"] == 1
    assert transitions["zero_to_one_to_two_plus_deepenings"] == 1
    assert transitions["cross_member_accumulation"] == 1
    assert transitions["coverage_to_vote_conversions"] == 1
