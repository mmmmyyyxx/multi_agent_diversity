from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from scripts import run_vote_aligned_confirmatory_replication as confirmatory
from scripts import run_vote_aligned_generic_shadow_pilot as seed75


def test_scope_is_new_confirmatory_replication() -> None:
    assert confirmatory.SEEDS == (76, 77)
    assert confirmatory.ARMS == (
        confirmatory.STATIC,
        confirmatory.P0,
        confirmatory.P1,
    )
    assert confirmatory.CONFIRMATORY_FOLD_MAP == (
        ("fold_a+fold_c", "fold_b"),
        ("fold_b+fold_c", "fold_a"),
    )


def test_manifest_records_scoped_seed76_77_authorization() -> None:
    document = yaml.safe_load(confirmatory.MANIFEST.read_text(encoding="utf-8"))
    assert document["status"] == "COMPLETED"
    assert document["seeds"] == [76, 77]
    assert document["api_authorization"]["authorized"] is True
    assert set(document["api_authorization"]["allowed_roles"]) == {
        "solver", "teacher", "critic", "student", "evaluator"
    }
    assert set(document["api_authorization"]["allowed_phases"]) == {
        "online_trajectory", "frozen_validation"
    }
    assert "seed76_77_confirmatory" in document["api_authorization"]["authorization_scope"]
    assert document["data"]["test_policy"] == "prohibited; zero calls"
    assert document["result"]["classifier"] == (
        "CONFIRMATORY_REPLICATION_NOT_SUPPORTED"
    )


def test_execution_still_requires_one_shot_environment_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(confirmatory.AUTH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="explicit Seed76/77 API authorization"):
        confirmatory._authorize()


def test_seed75_engine_scope_adapter_restores_every_global() -> None:
    names = (
        "SEEDS",
        "SCOPE",
        "validate_evaluation_inventory",
        "FINAL_EVAL_DATASETS",
        "AUTH_ENV",
        "MANIFEST",
        "DESIGN_ROOT",
        "DEFAULT_PREP_ROOT",
        "DEFAULT_RUN_ROOT",
        "DEFAULT_REPORT_ROOT",
        "RUNTIME_VERSION",
        "FOLD_MAP",
        "EXECUTION_INITIALIZATION_RELATIVE",
    )
    before = {name: getattr(seed75, name) for name in names}
    with confirmatory._base_scope():
        assert seed75.SEEDS == (76, 77)
        assert seed75.SCOPE.expected_trajectories == 4
        assert seed75.FOLD_MAP == confirmatory.CONFIRMATORY_FOLD_MAP
        assert seed75.MANIFEST == confirmatory.MANIFEST
    assert {name: getattr(seed75, name) for name in names} == before


def test_scope_adapter_rebinds_seed75_validator_default() -> None:
    observed = [
        {"seed": seed, "arm": arm, "dataset_role": role}
        for seed in confirmatory.SEEDS
        for arm in confirmatory.TRAIN_ARMS
        for role in confirmatory.FINAL_EVAL_DATASETS
    ]
    with confirmatory._base_scope():
        assert seed75.validate_evaluation_inventory(observed) == []


def test_p0_p1_are_exact_single_factor_configs(tmp_path: Path) -> None:
    common = (
        tmp_path / "optimize.csv",
        tmp_path / "validation.csv",
        tmp_path / "initialization.json",
    )
    p0 = confirmatory._config(
        76,
        confirmatory.P0,
        tmp_path / "p0",
        common[0],
        common[1],
        tmp_path / "p0.sqlite",
        common[2],
    ).to_flat_dict()
    p1 = confirmatory._config(
        76,
        confirmatory.P1,
        tmp_path / "p1",
        common[0],
        common[1],
        tmp_path / "p1.sqlite",
        common[2],
    ).to_flat_dict()
    differences = {key for key in p0 if p0[key] != p1[key]}
    assert differences == {"target_scheduler", "out_dir", "shared_solver_cache_path"}
    assert p0["agent_model"] == p1["agent_model"] == "qwen3-8b"
    assert p0["optimizer_model"] == p1["optimizer_model"] == "qwen3.7-flash"
    assert p0["evaluator_model"] == p1["evaluator_model"] == "qwen3.7-flash"
    assert p0["test_size"] == p1["test_size"] == 0
    assert p0["final_test_enabled"] is p1["final_test_enabled"] is False


def test_protocol_freezes_static_p0_p1_and_depth_hypothesis() -> None:
    protocol = confirmatory._protocol_document()
    assert protocol["seed75_evidence_excluded_from_confirmatory_decision"] is True
    assert protocol["seeds"] == [76, 77]
    assert tuple(protocol["arms"]) == confirmatory.ARMS
    assert protocol["models"] == {
        "solver": "qwen3-8b",
        "teacher": "qwen3.7-flash",
        "critic": "qwen3.7-flash",
        "student": "qwen3.7-flash",
        "evaluator": "qwen3.7-flash",
        "thinking": False,
    }
    assert protocol["shared_trainable_protocol"]["semantic_critic"] is True
    assert protocol["test_policy"] == "zero access"


def test_classifier_requires_strict_per_seed_replication() -> None:
    passing = {
        "p1_minus_p0_ensemble_gain": 0.01,
        "p1_minus_static_mean_member": 0.01,
        "p1_minus_static_vote": 0.02,
        "p1_minus_p0_gte3_count": 1,
    }
    failing = {**passing, "p1_minus_p0_gte3_count": 0}
    assert confirmatory.classify([passing, passing]) == (
        "CONFIRMATORY_REPLICATION_SUPPORTED"
    )
    assert confirmatory.classify([passing, failing]) == (
        "PARTIAL_CONFIRMATORY_REPLICATION"
    )
    negative = {**failing, "p1_minus_p0_ensemble_gain": -0.01}
    assert confirmatory.classify([passing, negative]) == (
        "CONFIRMATORY_REPLICATION_NOT_SUPPORTED"
    )


def test_static_execution_has_no_training_or_test_path() -> None:
    source = inspect.getsource(confirmatory._evaluate_static)
    assert "update_once" not in source
    assert "evaluate_final_test" not in source
    assert '"epochs": 0' in source
    assert '"test_size": 0' in source
    assert "TEST50_BLOCKED_BY_CONFIRMATORY_PROTOCOL" in source


def test_phase_b_execution_orders_training_before_static_evaluation() -> None:
    source = inspect.getsource(confirmatory.execute)
    assert source.index("base.execute") < source.index("_evaluate_static")


def test_audit_before_execution_is_not_run(tmp_path: Path) -> None:
    assert confirmatory.audit(tmp_path / "prep", tmp_path / "run") == {
        "audit_gate": "NOT_RUN",
        "new_test_calls": 0,
    }


def test_all_default_paths_are_project_local() -> None:
    for path in (
        confirmatory.DEFAULT_PREP_ROOT,
        confirmatory.DEFAULT_RUN_ROOT,
        confirmatory.DEFAULT_REPORT_ROOT,
    ):
        path.resolve().relative_to(confirmatory.ROOT.resolve())
