"""Zero-API semantic-closure freeze for the fresh Seed81 transfer diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.versions import (
    LAYER2_TEAM_SEARCH_PROTOCOL_VERSION,
    PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
    LAYER2_RESPONSIBILITY_SOURCE_VERSION,
    TEAM_MINIBATCH_CONTRACT_VERSION,
)
from scripts.prepare_online_transfer_diagnostic_v2 import (
    _git, _private_splits, _expected_bundle, validate_execution,
    canonical_json_bytes, write_bundle, frozen_payload as v2_frozen_payload,
)

EXPERIMENT_ID = "gepa_layer2_local_to_team_transfer_diagnostic_v3"
DEFAULT_PREP = ROOT / "runs" / EXPERIMENT_ID / "prep_final1"


def frozen_payload(*, execution_source_sha: str) -> tuple[dict, dict]:
    manifest, protocol = v2_frozen_payload(execution_source_sha=execution_source_sha)
    manifest["schema_version"] = "online_transfer_diagnostic_freeze_v3"
    manifest["experiment_id"] = manifest["attempt_id"] = EXPERIMENT_ID
    manifest["execution"]["scientific_method_anchor_sha"] = execution_source_sha
    manifest["execution"]["source_paths"] = sorted(set(manifest["execution"]["source_paths"]) | {
        "scripts/prepare_online_transfer_diagnostic_v3.py",
        "experiments/gepa_layer2_local_to_team_transfer_diagnostic_v3/PROTOCOL.md",
    })
    manifest["integrity_amendment"] = {
        "supersedes_unexecuted_attempt": "gepa_layer2_local_to_team_transfer_diagnostic_v2",
        "reason": "layer2_semantic_closure_before_execution",
        "scientific_method_changed": True,
    }
    manifest["semantic_contract"] = {
        "layer2_protocol": LAYER2_TEAM_SEARCH_PROTOCOL_VERSION,
        "team_minibatch": TEAM_MINIBATCH_CONTRACT_VERSION,
        "scheduler": PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
        "responsibility_source": LAYER2_RESPONSIBILITY_SOURCE_VERSION,
        "local_eval_identity": "same_frozen_team_minibatch12_ids",
        "local_acceptance": "target_member_only_no_team_feedback",
    }
    protocol["schema_version"] = "online_local_to_team_transfer_diagnostic_v3"
    protocol["experiment_id"] = EXPERIMENT_ID
    protocol["semantic_contract"] = manifest["semantic_contract"]
    protocol["scientific_method_change"] = True
    return manifest, protocol


def prepare(prep: Path) -> dict[str, str | bool]:
    if prep.exists():
        raise FileExistsError("fresh prep root required")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before freeze")
    source = _git("rev-parse", "HEAD")
    manifest, protocol = frozen_payload(execution_source_sha=source)
    prep.mkdir(parents=True)
    _private_splits(prep)
    for name, payload in (("manifest.json", manifest), ("protocol.json", protocol)):
        with (prep / name).open("xb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
    write_bundle(prep / "startup_identity", _expected_bundle(
        root=ROOT, manifest=manifest, protocol=protocol,
        execution_source_sha=source, prep=prep,
    ))
    permit = validate_execution(root=ROOT, prep=prep, require_authorized=False)
    return {
        "status": "READY_FOR_AUTHORIZATION",
        "attempt_id": permit.attempt_id,
        "execution_source_sha": source,
        "preregistration_sha256": permit.preregistration_sha256,
        "run_identity_sha256": permit.run_identity_sha256,
        "provider_boundary_reached": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    print(json.dumps(prepare(parser.parse_args().prep), sort_keys=True, indent=2))
