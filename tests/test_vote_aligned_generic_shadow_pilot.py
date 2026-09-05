from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_vote_aligned_generic_shadow_pilot as pilot


def _config(arm: str, root: Path):
    return pilot._config(
        75,
        arm,
        root / arm,
        root / "optimize.csv",
        root / "validation.csv",
        root / f"{arm}.sqlite",
        root / "initialization.json",
        False,
    )


def test_paired_configs_change_only_scheduler_and_private_paths(tmp_path: Path) -> None:
    p0 = _config(pilot.P0, tmp_path).to_flat_dict()
    p1 = _config(pilot.P1, tmp_path).to_flat_dict()
    differences = {key for key in p0 if p0[key] != p1[key]}

    assert differences == {
        "out_dir",
        "shared_solver_cache_path",
        "target_scheduler",
    }
    assert p0["agent_model"] == p1["agent_model"] == "qwen3-8b"
    assert p0["optimizer_model"] == p1["optimizer_model"] == "qwen3.7-flash"
    assert p0["evaluator_model"] == p1["evaluator_model"] == "qwen3.7-flash"
    assert p0["test_size"] == p1["test_size"] == 0
    assert p0["final_test_enabled"] is p1["final_test_enabled"] is False


def test_scheduler_does_not_change_generic_proposal_protocol(tmp_path: Path) -> None:
    p0 = pilot._resolved_protocol(_config(pilot.P0, tmp_path))
    p1 = pilot._resolved_protocol(_config(pilot.P1, tmp_path))

    assert p0 == p1
    assert p0.name == "experimental_diversity_d2_rr_generic"
    assert p0.tcs_context_policy == "generic_peer_state"
    assert p0.generic_revision_enabled is True
    assert p0.target_branch_count == p1.target_branch_count == 2
    assert p0.candidates_per_target_branch == p1.candidates_per_target_branch == 2


def test_phase_b_authorization_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pilot.AUTH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="new explicit API authorization"):
        pilot._authorized()


def test_manifest_authorizes_only_fresh_seed75_paired_execution() -> None:
    import yaml

    document = yaml.safe_load(pilot.MANIFEST.read_text(encoding="utf-8"))
    assert document["api_authorization"]["authorized"] is True
    assert document["api_authorization"]["allowed_roles"] == [
        "solver", "teacher", "critic", "student", "evaluator"
    ]
    assert document["api_authorization"]["allowed_phases"] == [
        "online_trajectory", "frozen_validation"
    ]
    assert "fresh_root_only_no_resume" in document["api_authorization"]["authorization_scope"]
    assert document["termination"]["resume_authorized"] is False
    assert document["reauthorization"]["prior_attempt_resumed"] is False
    assert document["reauthorization"]["fresh_root_required"] is True
    assert document["data"]["test_policy"] == "prohibited; zero new calls"


def test_protocol_freezes_required_split_models_and_zero_test() -> None:
    protocol = pilot._protocol_document()

    assert protocol["seeds"] == [75]
    assert protocol["fold_map"] == [
        {"seed": 75, "optimize": "fold_a+fold_b", "shadow": "fold_c"}
    ]
    assert protocol["models"] == {
        "solver": "qwen3-8b",
        "teacher": "qwen3.7-flash",
        "critic": "qwen3.7-flash",
        "student": "qwen3.7-flash",
        "evaluator": "qwen3.7-flash",
        "thinking": False,
    }
    assert protocol["test_policy"] == "zero access"
    assert protocol["shared_protocol"]["maximum_update_opportunities"] == 32
    assert protocol["shared_protocol"]["target_slots"] == 2
    assert protocol["shared_protocol"]["source_candidates_per_target"] == 2
    assert protocol["completion_scope"] == {
        "expected_trajectories": 2,
        "expected_shadow_evaluations": 2,
        "expected_validation_evaluations": 2,
        "expected_final_evaluation_artifacts": 4,
    }


def test_completion_scope_is_derived_from_production_dimensions() -> None:
    scope = pilot.build_expected_scope()
    assert scope.seeds == (75,)
    assert scope.arms == (pilot.P0, pilot.P1)
    assert scope.expected_trajectories == 2
    assert scope.expected_final_evaluations == 4
    assert scope.final_evaluation_identities == (
        (75, pilot.P0, "shadow"),
        (75, pilot.P0, "validation"),
        (75, pilot.P1, "shadow"),
        (75, pilot.P1, "validation"),
    )


def test_evaluation_identity_inventory_accepts_exact_scope() -> None:
    observed = [
        {"seed": seed, "arm": arm, "dataset_role": role}
        for seed, arm, role in pilot.SCOPE.final_evaluation_identities
    ]
    assert pilot.validate_evaluation_inventory(observed) == []


@pytest.mark.parametrize(
    "observed,error_fragment",
    [
        (
            [
                {"seed": seed, "arm": arm, "dataset_role": role}
                for seed, arm, role in pilot.SCOPE.final_evaluation_identities[:-1]
            ],
            "evaluation_identity_missing",
        ),
        (
            [
                *(
                    {"seed": seed, "arm": arm, "dataset_role": role}
                    for seed, arm, role in pilot.SCOPE.final_evaluation_identities
                ),
                {"seed": 75, "arm": pilot.P0, "dataset_role": "shadow"},
            ],
            "evaluation_identity_duplicate",
        ),
        (
            [
                *(
                    {"seed": seed, "arm": arm, "dataset_role": role}
                    for seed, arm, role in pilot.SCOPE.final_evaluation_identities
                ),
                {"seed": 75, "arm": pilot.P0, "dataset_role": "test"},
            ],
            "test_evaluation_present",
        ),
        (
            [
                *(
                    {"seed": seed, "arm": arm, "dataset_role": role}
                    for seed, arm, role in pilot.SCOPE.final_evaluation_identities
                ),
                {"seed": 76, "arm": pilot.P0, "dataset_role": "shadow"},
            ],
            "evaluation_identity_unexpected",
        ),
        (
            [
                *(
                    {"seed": seed, "arm": arm, "dataset_role": role}
                    for seed, arm, role in pilot.SCOPE.final_evaluation_identities
                ),
                {"seed": 75, "arm": "WRONG_ARM", "dataset_role": "shadow"},
            ],
            "evaluation_identity_unexpected",
        ),
    ],
)
def test_evaluation_identity_inventory_rejects_bad_scope(
    observed: list[dict[str, object]], error_fragment: str
) -> None:
    errors = pilot.validate_evaluation_inventory(observed)
    assert any(error_fragment in error for error in errors)


def test_audit_before_phase_b_is_not_run(tmp_path: Path) -> None:
    result = pilot.audit(tmp_path / "prep", tmp_path / "missing-run")
    assert result == {"phase_b_gate": "NOT_RUN", "new_test_calls": 0}


def test_partial_user_stopped_execution_remains_hold(tmp_path: Path) -> None:
    prep = tmp_path / "prep"
    run = tmp_path / "run"
    prep.mkdir()
    (prep / "source_freeze.json").write_text(
        json.dumps({"files": []}), encoding="utf-8"
    )
    (run / "seed75" / pilot.P0).mkdir(parents=True)
    (run / "USER_STOPPED.json").write_text("{}", encoding="utf-8")

    result = pilot.audit(prep, run)

    assert result["phase_b_gate"] == "HOLD"
    assert result["completion_status"] == "USER_ABORTED_INCOMPLETE"
    assert result["trajectory_count"] == 0
    assert result["expected_trajectory_count"] == 2
    assert result["final_evaluation_count"] == 0
    assert result["expected_final_evaluation_count"] == 4
    assert any(error.startswith("missing:75:") for error in result["errors"])
    assert any(
        error.startswith("evaluation_identity_missing:")
        for error in result["errors"]
    )


def test_prepare_source_contains_test_sentinel_and_no_test_evaluation() -> None:
    source = Path(pilot.__file__).read_text(encoding="utf-8")
    assert 'test_path="TEST50_BLOCKED_BY_GOVERNANCE"' in source
    assert "test_size=0" in source
    assert "final_test_enabled=False" in source
    assert 'FINAL_EVAL_DATASETS = ("shadow", "validation")' in source
    assert '"test" in scope.final_eval_datasets' in source


def test_selector_synthetic_gate_is_deterministic() -> None:
    assert pilot._synthetic_selector_gate() is True


def test_scientific_initialization_signature_excludes_execution_provenance() -> None:
    shared = {
        "initial_prompt_hashes": ["prompt"] * 5,
        "initial_member_correct_counts": [59] * 5,
        "initial_team_outcome": {"team_vote_correct_count": 59},
        "initial_vote_oracle_ghm_hash": "ghm",
        "probe_hash": "probe",
        "solver_request_identity": "request",
        "solver_identity": ["solver"],
    }
    p0 = {
        **shared,
        "initial_train_state_hash": "provenance-coupled-p0",
        "immutable_run_identity": {"git_commit": "old"},
    }
    p1 = {
        **shared,
        "initial_train_state_hash": "provenance-coupled-p1",
        "immutable_run_identity": {"git_commit": "new"},
    }

    assert pilot._scientific_initialization_signature(p0) == (
        pilot._scientific_initialization_signature(p1)
    )


def test_scientific_initialization_signature_detects_behavior_change() -> None:
    first = {
        "initial_prompt_hashes": ["prompt"] * 5,
        "initial_member_correct_counts": [59] * 5,
        "initial_team_outcome": {"team_vote_correct_count": 59},
        "initial_vote_oracle_ghm_hash": "ghm",
        "probe_hash": "probe",
        "solver_request_identity": "request",
        "solver_identity": ["solver"],
    }
    second = {**first, "initial_member_correct_counts": [58, 59, 59, 59, 59]}

    assert pilot._scientific_initialization_signature(first) != (
        pilot._scientific_initialization_signature(second)
    )
