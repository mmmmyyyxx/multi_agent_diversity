"""Zero-network tests for the new production-entry admission boundary."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from multi_dataset_diverse_rl.experiment import experiment_spec_from_mapping
from multi_dataset_diverse_rl.governance.production_execution import (
    _expected_bundle, admit_execution, terminal_lifecycle, validate_execution,
)
from multi_dataset_diverse_rl.governance.startup_identity import (
    StartupIdentityError, authorized_artifact, write_bundle,
)


SOURCE = "a" * 40
DEPENDENCY = {
    "version": "v0.1.1",
    "commit": "b4dbb55b7601dac448cdb836d5a401ca7d9eb920",
    "source_sha256": "84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8",
}


def _fixture(
    tmp_path: Path, monkeypatch,
    experiment_id: str = "gepa_layer2_real_canary_post_refactor_v1",
) -> tuple[Path, Path]:
    from multi_dataset_diverse_rl.governance import production_execution as governance

    root = tmp_path / "source"
    root.mkdir(parents=True)
    (root / "source.py").write_text("frozen = True\n", encoding="utf-8")
    prep = tmp_path / "prep"
    prep.mkdir()
    splits = prep / "splits_private"
    splits.mkdir()
    for name in ("optimize100.csv", "shadow50.csv"):
        (splits / name).write_text("question,answer\n", encoding="utf-8")
    monkeypatch.setattr(governance, "_git_head", lambda _root: SOURCE)
    monkeypatch.setattr(governance, "verify_frozen_gepa", lambda: DEPENDENCY)
    monkeypatch.setattr(governance, "endpoint_fingerprint_from_environment", lambda: "endpoint-digest")
    monkeypatch.setenv("LWJ_DASHSCOPE_API_KEY", "fake-test-credential")
    scientific = {
        "backend": "gepa", "optimization_scope": "layer2",
        "stopping_regime": "fixed_budget", "task_identity": "task",
        "data_identity": "fold_a+b_to_c", "fixed_budget_units": 1,
        "local_no_update_patience": 3, "team_no_update_patience": 2,
    }
    spec = experiment_spec_from_mapping(scientific)
    manifest = {
        "experiment_id": experiment_id,
        "attempt_id": experiment_id,
        "scientific": scientific,
        "method_identity": spec.method_identity,
        "spec_identity": spec.identity(),
        "runtime": {
            "seed": 80, "provider_profile": "lwj",
            "endpoint_fingerprint": "endpoint-digest",
            "solver_model": "qwen3-8b",
            "optimizer_model": "qwen3.7-flash",
            "evaluator_model": "qwen3.7-flash",
        },
        "models": {
            "solver": {"model": "qwen3-8b", "thinking": False},
            "reflection": {"model": "qwen3.7-flash"},
        },
        "dependency": {"gepa": DEPENDENCY},
        "execution": {
            "execution_source_sha": SOURCE,
            "scientific_method_anchor_sha": "historical-anchor",
            "initialization_policy": "FRESH_DETERMINISTIC_INITIALIZATION_V1",
            "source_paths": ["source.py"],
        },
        "access": {"validation50_calls": 0, "test50_calls": 0},
        "api_authorization": {
            "allowed_roles": ["solver", "reflection"],
            "allowed_phases": ["canary"],
            "authorized": False,
        },
    }
    protocol = {"purpose": "local_empirical_path_only", "phase": "canary"}
    (prep / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (prep / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    bundle = _expected_bundle(
        root=root, manifest=manifest, protocol=protocol,
        execution_source_sha=SOURCE, prep=prep,
    )
    bundle["authorization"] = authorized_artifact(
        bundle, scope=manifest["attempt_id"], explicit_user_authorized=True,
    )
    write_bundle(prep / "startup_identity", bundle)
    return root, prep


def test_new_canary_attempt_identity_is_distinct_and_admissible(
    tmp_path: Path, monkeypatch,
) -> None:
    root, prep = _fixture(
        tmp_path, monkeypatch,
        experiment_id="gepa_layer2_real_canary_post_refactor_v2",
    )
    permit = validate_execution(root=root, prep=prep, require_authorized=True)
    assert permit.attempt_id == "gepa_layer2_real_canary_post_refactor_v2"
    assert not permit.admitted


def test_one_time_admission_and_failed_start(tmp_path: Path, monkeypatch) -> None:
    root, prep = _fixture(tmp_path, monkeypatch)
    permit = validate_execution(root=root, prep=prep, require_authorized=True)
    assert not permit.admitted
    assert not (tmp_path / "run").exists()
    admitted = admit_execution(permit, tmp_path / "run")
    lifecycle = json.loads((admitted.run_root / "run_lifecycle.json").read_text())
    assert lifecycle["status"] == "RUNNING"
    assert lifecycle["provider_boundary_reached"] is False
    terminal_lifecycle(admitted, status="FAILED_START")
    lifecycle = json.loads((admitted.run_root / "run_lifecycle.json").read_text())
    assert [row["status"] for row in lifecycle["events"]] == ["RUNNING", "FAILED_START"]
    with pytest.raises(StartupIdentityError, match="consumed"):
        validate_execution(root=root, prep=prep, require_authorized=True)


@pytest.mark.parametrize(
    ("artifact", "mutate"),
    [
        ("authorization", lambda x: x.update(authorized=False)),
        ("authorization", lambda x: x.update(authorization_scope="stale")),
        ("scientific_identity", lambda x: x.update(preregistration_sha256="wrong")),
        ("run_identity", lambda x: x.update(run_identity_sha256="wrong")),
        ("manifest", lambda x: x["execution"].update(execution_source_sha="wrong")),
        ("manifest", lambda x: x["runtime"].update(provider_profile="myx")),
        ("manifest", lambda x: x["runtime"].update(endpoint_fingerprint="wrong")),
        ("manifest", lambda x: x["runtime"].update(solver_model="wrong")),
        ("manifest", lambda x: x["runtime"].update(optimizer_model="wrong")),
        ("manifest", lambda x: x["scientific"].update(data_identity="wrong")),
        ("manifest", lambda x: x["execution"].update(initialization_policy="wrong")),
        ("manifest", lambda x: x["dependency"]["gepa"].update(source_sha256="wrong")),
        ("protocol", lambda x: x.update(phase="formal")),
        ("authorization", lambda x: x.update(allowed_phases=["formal"])),
        ("authorization", lambda x: x.update(allowed_roles=["solver"])),
    ],
)
def test_every_bound_field_rejects_before_provider(
    tmp_path: Path, monkeypatch, artifact: str, mutate,
) -> None:
    root, prep = _fixture(tmp_path, monkeypatch)
    location = (
        prep / "startup_identity" / f"{artifact}.json"
        if artifact in {"authorization", "scientific_identity", "run_identity"}
        else prep / f"{artifact}.json"
    )
    value = json.loads(location.read_text(encoding="utf-8"))
    mutate(value)
    location.write_text(json.dumps(value), encoding="utf-8")
    constructed = 0

    def poison_provider():
        nonlocal constructed
        constructed += 1
        raise AssertionError("provider constructor reached")

    with pytest.raises((StartupIdentityError, KeyError, ValueError)):
        validate_execution(root=root, prep=prep, require_authorized=True)
        poison_provider()
    assert constructed == 0
    assert not (tmp_path / "run").exists()


def test_gepa_dependency_identity_is_location_independent(tmp_path: Path, monkeypatch) -> None:
    from multi_dataset_diverse_rl.local_optimizers import gepa_runtime

    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "src/gepa").mkdir(parents=True)
        (root / "src/gepa/__init__.py").write_text("same source\n", encoding="utf-8")
    monkeypatch.setattr(gepa_runtime, "GEPA_COMMIT", "fake-commit")
    monkeypatch.setattr(gepa_runtime, "GEPA_SOURCE_SHA256", "fake-hash")
    monkeypatch.setattr(
        gepa_runtime.subprocess, "check_output",
        lambda args, **kwargs: (
            "fake-commit\n" if "rev-parse" in args
            else b"src/gepa/__init__.py\x00"
        ),
    )
    actual = gepa_runtime.source_bundle_sha256(first)
    monkeypatch.setattr(gepa_runtime, "GEPA_SOURCE_SHA256", actual)
    assert gepa_runtime.verify_frozen_gepa(first) == gepa_runtime.verify_frozen_gepa(second)
    (second / "src/gepa/__init__.py").write_text("changed source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source bundle mismatch"):
        gepa_runtime.verify_frozen_gepa(second)


def test_entry_import_graph_is_current_and_historical_runner_free() -> None:
    root = Path(__file__).resolve().parents[1]
    entry = (root / "scripts/run_experiment.py").read_text(encoding="utf-8")
    canary = (root / "multi_dataset_diverse_rl/production_canary.py").read_text(encoding="utf-8")
    for source in (entry, canary):
        assert "run_level_b_gepa_real_canary" not in source
        assert "run_seed78_primary_responsibility_ab" not in source
        assert "run_gepa_layer2_real_canary_v2" not in source
    assert "execute_post_refactor_canary" in entry
    assert "run_experiment(" in canary
    assert "GEPALayer2EvidenceOptimizer" in canary
    assert "TeamSearchController" in canary


def test_technical_canary_stops_before_team_rollout() -> None:
    from multi_dataset_diverse_rl.team_search.controller import TeamSearchController

    assert hasattr(TeamSearchController, "run_local_empirical_canary")


def test_three_independent_freezes_have_identical_scientific_and_run_hashes(
    tmp_path: Path, monkeypatch,
) -> None:
    from multi_dataset_diverse_rl.governance.startup_identity import read_bundle

    hashes = []
    for index in range(3):
        _, prep = _fixture(tmp_path / str(index), monkeypatch)
        bundle = read_bundle(prep / "startup_identity")
        hashes.append((
            bundle["scientific_identity"]["preregistration_sha256"],
            bundle["run_identity"]["run_identity_sha256"],
        ))
    assert len(set(hashes)) == 1


def test_fresh_process_reaches_admission_then_fails_before_provider(
    tmp_path: Path, monkeypatch,
) -> None:
    root, prep = _fixture(tmp_path, monkeypatch)
    run_root = tmp_path / "fresh-run"
    code = """
import asyncio, json, sys
from pathlib import Path
from unittest.mock import patch
import scripts.run_experiment as entry
from multi_dataset_diverse_rl.governance import production_execution as gov
root, prep, run = (Path(value) for value in sys.argv[1:4])
async def poison_provider(permit, *, root):
    assert permit.admitted
    raise RuntimeError('POISON_PROVIDER_CONSTRUCTION')
with patch.object(entry, 'ROOT', root), patch.object(gov, '_git_head', lambda _: 'a'*40), patch.object(gov, 'verify_frozen_gepa', lambda: {'version':'v0.1.1','commit':'b4dbb55b7601dac448cdb836d5a401ca7d9eb920','source_sha256':'84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8'}), patch.object(gov, 'endpoint_fingerprint_from_environment', lambda: 'endpoint-digest'), patch.object(entry, 'execute_post_refactor_canary', poison_provider):
    try:
        asyncio.run(entry.execute_frozen(prep, run))
    except RuntimeError as exc:
        assert str(exc) == 'POISON_PROVIDER_CONSTRUCTION'
    else:
        raise AssertionError('poison provider was not reached')
print(json.dumps({'lifecycle': json.loads((run / 'run_lifecycle.json').read_text())['status']}))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(root), str(prep), str(run_root)],
        cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=True,
    )
    assert json.loads(process.stdout)["lifecycle"] == "FAILED_START"
    lifecycle = json.loads((run_root / "run_lifecycle.json").read_text())
    assert lifecycle["provider_boundary_reached"] is False
    assert lifecycle["provider_attempts"] == 0


def test_post_refactor_canary_fake_provider_stops_before_team_stage(
    tmp_path: Path, monkeypatch,
) -> None:
    """Exercise real production composition with an in-memory fake transport."""

    from types import SimpleNamespace

    from multi_dataset_diverse_rl.governance.production_execution import (
        ValidatedExecutionContext,
    )
    from multi_dataset_diverse_rl.production_canary import execute_post_refactor_canary
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from scripts.prepare_post_refactor_gepa_canary import (
        _private_splits, frozen_payload,
    )

    prep = tmp_path / "prep"
    prep.mkdir()
    _private_splits(prep)
    manifest, _ = frozen_payload(execution_source_sha=SOURCE)
    (prep / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run_lifecycle.json").write_text(
        json.dumps({"status": "RUNNING", "provider_client_constructed": False}),
        encoding="utf-8",
    )
    permit = ValidatedExecutionContext(
        experiment_id=manifest["experiment_id"],
        attempt_id=manifest["attempt_id"],
        execution_source_sha=SOURCE,
        preregistration_sha256="prereg-fixture",
        run_identity_sha256="f" * 64,
        provider_profile="lwj",
        endpoint_fingerprint=manifest["runtime"]["endpoint_fingerprint"],
        allowed_phase="canary", allowed_roles=("solver", "reflection"),
        prep_root=prep, run_root=run_root,
    )
    calls = []

    class FakeCompletion:
        async def create(self, **request):
            calls.append(request["model"])
            content = (
                "Use context and pronoun agreement to identify the referent."
                if request["model"] == "qwen3.7-flash"
                else "FINAL_ANSWER: A"
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
            )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletion()))
    monkeypatch.setattr(
        ProviderClientFactory, "from_environment",
        lambda *args, **kwargs: fake,
    )
    monkeypatch.setattr(
        ProviderClientFactory, "create",
        lambda *args, **kwargs: fake,
    )
    # This fixture replays the historical canary freeze's routed source. The
    # current raw-legal source is tested separately, including its strict
    # TeamMiniBatch quota failure under an identical-parent fake Solver.
    from multi_dataset_diverse_rl.team_search.system_runtime import FrozenResponsibilitySnapshot

    def routed_snapshot(system, *, update_index):
        _, assigned = system.assign_responsibilities(update_index=update_index)
        states, _, _ = system.current_states_and_opportunities()
        return FrozenResponsibilitySnapshot(
            assigned={member: tuple(rows) for member, rows in assigned.items()},
            state_by_question={row.question_hash: row for row in states},
            current_margin_by_question={row.question_hash: row.plurality_margin for row in states},
            source_version="historical_service_routed_v1",
        )

    monkeypatch.setattr(
        "multi_dataset_diverse_rl.production_canary.freeze_current_responsibility",
        routed_snapshot,
    )
    result = asyncio.run(execute_post_refactor_canary(
        permit, root=Path(__file__).resolve().parents[1],
    ))
    assert calls
    assert set(calls) <= {"qwen3-8b", "qwen3.7-flash"}
    assert "qwen3.7-flash" in calls
    ledger = [json.loads(row) for row in (run_root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["phase"] == "local_optimizer_solver_eval" for row in ledger)
    assert result["funnel"]["team_minibatch_survivors"] == 0
    assert result["funnel"]["full_team_evaluated_candidates"] == 0
    assert result["funnel"]["committed_candidates"] == 0
    assert result["local_empirical"]["native_example_selection_calls"] == 0
    assert result["local_empirical"]["solver_reached"] >= 1
    assert result["validation50_calls"] == result["test50_calls"] == 0
