from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from multi_dataset_diverse_rl.governance.startup_identity import (
    StartupIdentityError,
    authorized_artifact,
    build_startup_bundle,
    canonical_json_bytes,
    read_bundle,
    scientific_manifest_payload,
    validate_startup_bundle,
    write_bundle,
)


def manifest() -> dict:
    return {
        "experiment_id": "example",
        "status": "PREFLIGHT_PASS",
        "lifecycle_history": [{"status": "PREFLIGHT_PASS", "timestamp": "a"}],
        "design": {"method": "fixed"},
        "data": {"split": "s"},
        "model": {"solver": "qwen3-8b"},
        "seeds": [80],
        "budget": {"frozen_before_run": True},
        "api_authorization": {
            "authorized": False,
            "authorization_scope": "pending",
            "allowed_roles": ["reflection", "solver"],
            "allowed_phases": ["canary"],
        },
        "git": {"scientific_method_anchor": "a" * 40, "result_commit": None},
        "execution_freeze": {
            "initialization_policy": "FRESH_DETERMINISTIC_INITIALIZATION_V1",
            "execution_source_sha": "historical-not-authoritative",
        },
        "artifacts": {"preregistration": {"sha256": "self-reference-forbidden"}},
        "result": None,
    }


def bundle(*, attempt: str = "attempt-a", profile: str = "lwj", seed: int = 80):
    return build_startup_bundle(
        manifest=manifest(),
        protocol={"name": "p", "seed": seed},
        experiment_id="example",
        attempt_id=attempt,
        scientific_method_anchor_sha="a" * 40,
        execution_source_sha="b" * 40,
        provider_profile=profile,
        endpoint_fingerprint="c" * 64,
        models={"solver": {"model": "qwen3-8b"}, "reflection": {"model": "qwen3.7-flash"}},
        data_hashes={"optimize": "d" * 64},
        initialization={"policy": "FRESH_DETERMINISTIC_INITIALIZATION_V1"},
        seeds=[seed],
        local_patience=3,
        team_patience=2,
        saturation_mode="canary",
        source_files=[{"path": "x\\y.py", "sha256": "e" * 64}],
    )


def grant(value: dict) -> dict:
    result = deepcopy(value)
    result["authorization"] = authorized_artifact(
        result, scope="test-only-zero-api", explicit_user_authorized=True
    )
    return result


def test_operational_manifest_fields_do_not_change_scientific_payload():
    first = manifest()
    second = deepcopy(first)
    second["status"] = "RUNNING"
    second["lifecycle_history"] = [{"status": "RUNNING", "timestamp": "different"}]
    second["api_authorization"]["authorized"] = True
    second["api_authorization"]["authorization_scope"] = "one-time"
    second["git"]["result_commit"] = "f" * 40
    second["artifacts"]["preregistration"]["sha256"] = "changed"
    assert scientific_manifest_payload(first) == scientific_manifest_payload(second)


def test_canonical_generation_is_byte_stable_and_paths_are_posix():
    values = [bundle() for _ in range(3)]
    assert len({canonical_json_bytes(value) for value in values}) == 1
    source = values[0]["scientific_identity"]["payload"]["source_files"][0]
    assert source["path"] == "x/y.py"


def test_disk_round_trip_and_authorization_are_exact(tmp_path: Path):
    expected = grant(bundle())
    write_bundle(tmp_path, expected)
    stored = read_bundle(tmp_path)
    result = validate_startup_bundle(
        stored=stored,
        expected=bundle(),
        require_authorized=True,
        phase="canary",
        roles=("solver", "reflection"),
    )
    assert result["status"] == "PRE_PROVIDER_AUTHORIZATION_VALID"
    assert result["provider_boundary_reached"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scientific_identity", "payload", "protocol_sha256"), "0" * 64),
        (("scientific_identity", "payload", "manifest_sha256"), "1" * 64),
        (("scientific_identity", "payload", "execution_source_sha"), "2" * 40),
        (("scientific_identity", "payload", "provider", "profile"), "myx"),
        (("scientific_identity", "payload", "provider", "endpoint_fingerprint"), "3" * 64),
        (("scientific_identity", "payload", "models", "solver", "model"), "other"),
        (("scientific_identity", "payload", "seeds", 0), 81),
        (("scientific_identity", "payload", "stopping", "local_no_update_patience"), 4),
        (("scientific_identity", "payload", "stopping", "team_no_update_patience"), 4),
        (("scientific_identity", "payload", "data_hashes", "optimize"), "4" * 64),
        (("scientific_identity", "payload", "initialization", "policy"), "other"),
    ],
)
def test_every_bound_field_poison_fails_pre_provider(path, value):
    expected = bundle()
    stored = grant(deepcopy(expected))
    target = stored
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(StartupIdentityError, match="ABORT_PRE_PROVIDER"):
        validate_startup_bundle(
            stored=stored,
            expected=expected,
            require_authorized=True,
            phase="canary",
            roles=("solver", "reflection"),
        )


def test_stale_authorization_cannot_authorize_another_attempt():
    first = grant(bundle(attempt="a"))
    second = bundle(attempt="b")
    stale = deepcopy(second)
    stale["authorization"] = first["authorization"]
    with pytest.raises(StartupIdentityError, match="authorization attempt_id mismatch"):
        validate_startup_bundle(
            stored=stale,
            expected=second,
            require_authorized=True,
            phase="canary",
            roles=("solver",),
        )


def test_attempt_identity_changes_run_identity_not_scientific_identity():
    first = bundle(attempt="a")
    second = bundle(attempt="b")
    assert first["scientific_identity"] == second["scientific_identity"]
    assert first["run_identity"] != second["run_identity"]


def test_non_bound_metadata_does_not_affect_validation():
    expected = bundle()
    stored = grant(deepcopy(expected))
    stored["authorization"]["display_name"] = "irrelevant"
    stored["authorization"]["issued_at"] = "irrelevant"
    validate_startup_bundle(
        stored=stored,
        expected=expected,
        require_authorized=True,
        phase="canary",
        roles=("solver",),
    )


def test_profile_change_is_bound_and_myx_is_rejected():
    with pytest.raises(StartupIdentityError, match="provider_profile must be lwj"):
        bundle(profile="myx")


def test_serialized_files_contain_no_self_hash_input(tmp_path: Path):
    value = bundle()
    write_bundle(tmp_path, value)
    scientific = json.loads((tmp_path / "scientific_identity.json").read_text())
    manifest_payload = scientific["payload"]
    assert "artifacts" not in manifest_payload
    assert "preregistration_sha256" not in manifest_payload


def test_lifecycle_running_is_not_written_before_authorization(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    prep = tmp_path / "prep"
    prep.mkdir()
    run = tmp_path / "run"
    code = f"""
import importlib.util
from pathlib import Path
from multi_dataset_diverse_rl.governance.startup_identity import StartupIdentityError
path = Path(r'{(root / 'scripts/run_gepa_layer2_real_canary_v2.py').as_posix()}')
spec = importlib.util.spec_from_file_location('isolated_canary', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def reject(*args, **kwargs):
    raise StartupIdentityError('ABORT_PRE_PROVIDER: authorization required')
module.verify_startup_identity = reject
try:
    module.start_run_attempt(Path(r'{prep.as_posix()}'), Path(r'{run.as_posix()}'))
except StartupIdentityError:
    raise SystemExit(0)
raise SystemExit(3)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, check=False)
    assert result.returncode == 0
    assert not run.exists()


def test_provider_is_not_constructed_by_identity_validation(tmp_path: Path, monkeypatch):
    expected = bundle()
    stored = grant(deepcopy(expected))
    write_bundle(tmp_path, stored)
    constructed = []
    monkeypatch.setattr(
        "multi_dataset_diverse_rl.governance.execution_harness_v2.AsyncOpenAI",
        lambda **kwargs: constructed.append(kwargs),
    )
    validate_startup_bundle(
        stored=read_bundle(tmp_path),
        expected=expected,
        require_authorized=True,
        phase="canary",
        roles=("solver",),
    )
    assert constructed == []
