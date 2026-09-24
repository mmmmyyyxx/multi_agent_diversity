"""Fresh, zero-provider freeze of the v2 sampling-integrity closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.production_execution import (  # noqa: E402
    _expected_bundle, validate_execution,
)
from multi_dataset_diverse_rl.governance.startup_identity import (  # noqa: E402
    canonical_json_bytes, write_bundle,
)
from scripts.prepare_online_transfer_diagnostic import (  # noqa: E402
    _git, frozen_payload as v1_frozen_payload,
)
from scripts.prepare_post_refactor_gepa_canary import _private_splits  # noqa: E402

EXPERIMENT_ID = "gepa_layer2_local_to_team_transfer_diagnostic_v2"
DEFAULT_PREP = ROOT / "runs" / EXPERIMENT_ID / "prep"


def frozen_payload(*, execution_source_sha: str) -> tuple[dict, dict]:
    manifest, protocol = v1_frozen_payload(execution_source_sha=execution_source_sha)
    manifest["schema_version"] = "online_transfer_diagnostic_freeze_v2"
    manifest["experiment_id"] = manifest["attempt_id"] = EXPERIMENT_ID
    paths = set(manifest["execution"]["source_paths"])
    paths |= {
        "scripts/prepare_online_transfer_diagnostic_v2.py",
        "experiments/gepa_layer2_local_to_team_transfer_diagnostic_v2/PROTOCOL.md",
    }
    manifest["execution"]["source_paths"] = sorted(paths)
    manifest["integrity_amendment"] = {
        "supersedes_unexecuted_attempt": "gepa_layer2_local_to_team_transfer_diagnostic_v1",
        "reason": "sampling_integrity_and_candidate_stage_telemetry_only",
        "scientific_method_changed": False,
    }
    protocol["schema_version"] = "online_local_to_team_transfer_diagnostic_v2"
    protocol["experiment_id"] = EXPERIMENT_ID
    protocol["accepted_frontier_enforcement"] = "immediate_pre_team_typed_abort"
    protocol["candidate_stage_costs"] = "durable_ledger_partitioned_shared_and_attributable"
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
    args = parser.parse_args()
    print(json.dumps(prepare(args.prep), sort_keys=True, indent=2))
