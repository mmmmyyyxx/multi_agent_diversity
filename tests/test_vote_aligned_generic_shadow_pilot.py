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


def test_manifest_keeps_phase_b_unauthorized() -> None:
    import yaml

    document = yaml.safe_load(pilot.MANIFEST.read_text(encoding="utf-8"))
    assert document["api_authorization"] == {
        "authorized": False,
        "allowed_roles": [],
        "allowed_phases": [],
    }
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


def test_audit_before_phase_b_is_not_run(tmp_path: Path) -> None:
    result = pilot.audit(tmp_path / "prep", tmp_path / "missing-run")
    assert result == {"phase_b_gate": "NOT_RUN", "new_test_calls": 0}


def test_prepare_source_contains_test_sentinel_and_no_test_evaluation() -> None:
    source = Path(pilot.__file__).read_text(encoding="utf-8")
    assert 'test_path="TEST50_BLOCKED_BY_GOVERNANCE"' in source
    assert "test_size=0" in source
    assert "final_test_enabled=False" in source
    assert 'for split in ("shadow", "validation")' in source
    assert '"test"' not in source.split('for split in ("shadow", "validation")', 1)[1].split("write_csv", 1)[0]


def test_selector_synthetic_gate_is_deterministic() -> None:
    assert pilot._synthetic_selector_gate() is True
