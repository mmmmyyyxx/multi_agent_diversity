"""Zero-API independent-prep and byte-poison audit of a frozen attempt3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sitecustomize  # pre-import network guard supplied by formal_zero_api_runner

from multi_dataset_diverse_rl.governance import production_execution as admission
from multi_dataset_diverse_rl.governance.freeze_hash import normalized_lf_bytes
from multi_dataset_diverse_rl.governance.startup_identity import StartupIdentityError
from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
from scripts.prepare_online_transfer_diagnostic_v4_attempt3 import ATTEMPT3_ID, ROOT


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def audit(prep_a: Path, prep_b: Path) -> dict[str, object]:
    if sitecustomize.network_attempt_count():
        raise AssertionError("network attempted before audit")
    a, b = _hashes(prep_a), _hashes(prep_b)
    if a != b:
        raise AssertionError("independent attempt3 preps differ")
    manifest = json.loads((prep_a / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((prep_a / "protocol.json").read_text(encoding="utf-8"))
    if (manifest["attempt_id"] != ATTEMPT3_ID
            or manifest["api_authorization"]["authorized"]
            or manifest["access"] != {"validation50_calls": 0, "test50_calls": 0}
            or (prep_a.parent / "run").exists()
            or (prep_a / "authorization_consumed.json").exists()
            or (prep_b / "authorization_consumed.json").exists()):
        raise AssertionError("attempt3 authorization or split governance mismatch")
    create = ProviderClientFactory.create
    from_environment = ProviderClientFactory.from_environment
    original_hash = admission.source_freeze_sha256
    provider_constructors = 0
    poisoned = 0

    def forbidden(*args, **kwargs):
        nonlocal provider_constructors
        provider_constructors += 1
        raise AssertionError("provider constructor reached")

    ProviderClientFactory.create = forbidden
    ProviderClientFactory.from_environment = forbidden
    try:
        for prep in (prep_a, prep_b):
            permit = admission.validate_execution(root=ROOT, prep=prep, require_authorized=False)
            if permit.attempt_id != ATTEMPT3_ID or permit.admitted:
                raise AssertionError("attempt3 preflight permit mismatch")
        for relative in manifest["execution"]["source_paths"]:
            target = (ROOT / relative).resolve()
            fake_sha = hashlib.sha256(normalized_lf_bytes(target.read_bytes()) + b"\n# poison\n").hexdigest()

            def hash_with_poison(path, *, _target=target, _sha=fake_sha):
                return _sha if Path(path).resolve() == _target else original_hash(path)

            admission.source_freeze_sha256 = hash_with_poison
            try:
                admission.validate_execution(root=ROOT, prep=prep_a, require_authorized=False)
            except StartupIdentityError:
                poisoned += 1
            else:
                raise AssertionError(f"source poison escaped: {relative}")
            finally:
                admission.source_freeze_sha256 = original_hash
    finally:
        admission.source_freeze_sha256 = original_hash
        ProviderClientFactory.create = create
        ProviderClientFactory.from_environment = from_environment
    if provider_constructors or poisoned != len(manifest["execution"]["source_paths"]):
        raise AssertionError("source poison or provider isolation incomplete")
    if sitecustomize.network_attempt_count():
        raise AssertionError("network attempted during audit")
    identity = json.loads((prep_a / "startup_identity/scientific_identity.json").read_text(encoding="utf-8"))
    run = json.loads((prep_a / "startup_identity/run_identity.json").read_text(encoding="utf-8"))
    return {
        "gate": "PASS",
        "attempt_id": ATTEMPT3_ID,
        "execution_source_sha": manifest["execution"]["execution_source_sha"],
        "protocol_sha256": identity["payload"]["protocol_sha256"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "run_identity_sha256": run["run_identity_sha256"],
        "sample_unit": protocol["sample_unit"],
        "source_paths": len(manifest["execution"]["source_paths"]),
        "source_poison_cases": poisoned,
        "prep_files_byte_identical": len(a),
        "provider_constructor_calls": provider_constructors,
        "network_attempts": sitecustomize.network_attempt_count(),
        "real_api_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
        "authorization_consumed": False,
        "real_execution_started": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-a", type=Path, required=True)
    parser.add_argument("--prep-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.prep_a, args.prep_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2))
