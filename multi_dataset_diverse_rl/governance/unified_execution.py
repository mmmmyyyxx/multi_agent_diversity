"""Current V2.2 admission fails closed until a fresh execution binding is frozen."""
import hashlib
import subprocess
from .source_identity import hash_scope
from .repository import validate_manifest_v2
from .startup_identity import canonical_sha256
from ..persistence.durable_io import read_json
from ..benchmarks.math_domain_binding import execution_binding, BINDING_BLOCKER
from ..search.schemas import SearchContractError

PREP_SCHEMA = "unified_canary_prep_v1"
CURRENT_OFFLINE_PROFILE = 'experiments/execution_bindings/math_v2_2_offline_profile_v1.json'


def bound_preflight(root, manifest):
    errors = validate_manifest_v2(root, manifest)
    errors.append(BINDING_BLOCKER)
    return dict(gate='HOLD', blockers=errors, ready_for_authorization=False,
        provider_attempts=0, validation_calls=0, test_calls=0, real_api_authorized=False)


def preexecution_manifest(root, *, source_sha, frozen=True, binding_path=None,
        experiment_id='UNBOUND_V2_2_DRAFT'):
    # A new default cannot resolve a historical V2.1 execution graph.
    contract = read_json(root / (binding_path or CURRENT_OFFLINE_PROFILE))
    binding = execution_binding(root, contract)
    raise SearchContractError('HOLD_PRE_PROVIDER: ' + ','.join(binding.blockers()))


def execution_identity(root, contract):
    binding = execution_binding(root, contract)
    raise SearchContractError('HOLD_PRE_PROVIDER: ' + ','.join(binding.blockers()))


def execution_scope(manifest, contract):
    raise SearchContractError(BINDING_BLOCKER)


def prepare_canary(root, manifest, *, destination, arm='A4', seed=81):
    raise SearchContractError('HOLD_PRE_PROVIDER: ' + ','.join(bound_preflight(root, manifest)['blockers']))


def validate_prep(root, prep, *, require_authorized=False):
    # No credentials, ledger, data reader, transport or old authority is constructed.
    raise SearchContractError(BINDING_BLOCKER)


async def execute_canary(root, prep, run_root):
    validate_prep(root, prep, require_authorized=True)


def consumption_path(root, scope):
    return root / "runs/unified_authorization_consumption" / (canonical_sha256(scope) + ".json")


def validate_frozen_source(root, receipt):
    entries = receipt["execution_closure"]["files"]
    if not entries or len({r["path"] for r in entries}) != len(entries):
        raise SearchContractError("SOURCE_CLOSURE_INVALID")
    for r in entries:
        p = (root / r["path"]).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != r["raw_sha256"]:
            raise SearchContractError("SOURCE_IDENTITY_MISMATCH: " + r["path"])
    if hash_scope(root, [root / r["path"] for r in entries]) != receipt["execution_closure"]:
        raise SearchContractError("SOURCE_CLOSURE_RECEIPT_MISMATCH")


def verify_source_commit(root, source, identity):
    for r in identity["execution_closure"]["files"]:
        blob = subprocess.check_output(["git", "show", source + ":" + r["path"]], cwd=root, stderr=subprocess.DEVNULL)
        if hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() != r["sha256"]:
            raise SearchContractError("EXECUTION_SOURCE_COMMIT_BYTE_MISMATCH: " + r["path"])


def inventory(run_root):
    return {"files": [{"path": p.relative_to(run_root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                      for p in sorted(run_root.rglob("*")) if p.is_file() and p.name != "raw_evidence_inventory.json"]}
