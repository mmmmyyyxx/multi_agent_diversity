# HISTORICAL_REPLAY_ONLY: preserved frozen reproduction utility; new experiments use scripts/run_experiment.py.
"""Freeze Seed81 V4 attempt3 prospectively without executing provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.production_execution import (
    _expected_bundle, validate_execution,
)
from multi_dataset_diverse_rl.governance.startup_identity import (
    canonical_json_bytes, write_bundle,
)
from multi_dataset_diverse_rl.governance.v4_attempt3_contract import (
    diagnostic_contract, resource_upper_bounds,
)
from multi_dataset_diverse_rl.governance.v4_source_closure import active_v4_source_paths
from scripts.prepare_online_transfer_diagnostic_v4 import (
    EXPERIMENT_ID, _git, _private_splits, frozen_payload as v4_frozen_payload,
)


ATTEMPT3_ID = "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt3"
DEFAULT_PREP = ROOT / "runs" / ATTEMPT3_ID / "prep_a"


def frozen_payload(*, execution_source_sha: str) -> tuple[dict, dict]:
    manifest, protocol = v4_frozen_payload(
        execution_source_sha=execution_source_sha,
        attempt_id="gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2",
    )
    manifest["attempt_id"] = ATTEMPT3_ID
    manifest["execution"]["scientific_method_anchor_sha"] = (
        "85812a7d891e6a2c3bfdca00a1cb4d14074735a4"
    )
    manifest["diagnostic_contract"] = diagnostic_contract()
    manifest["execution"]["source_paths"] = list(active_v4_source_paths(ROOT, attempt3=True))
    manifest["execution_refresh"] = {
        "scientific_method_changed": False,
        "execution_implementation_refreshed": True,
        "supersedes_for_execution": "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2",
        "reason": "diagnostic_returned_candidate_sampling_integrity_repair",
    }
    manifest["diagnostic_protocol_repair"] = {
        "prior_attempt": "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2",
        "prior_validity": "INVALID_NOT_EVALUABLE_DIAGNOSTIC_SAMPLING_INTEGRITY",
        "scientific_method_changed": False,
        "sampled_boundary": "Layer1_to_Layer2_returned_candidates",
        "prior_provider_calls_excluded": True,
    }
    protocol.pop("accepted_mutation_target")
    protocol.update(diagnostic_contract())
    protocol["diagnostic_full_policy"] = "mandatory_all_returned_strict_positive_admission_inert"
    protocol["full_rows_per_returned_candidate"] = protocol.pop("full_rows_per_accepted_mutation")
    protocol["proposal_ceiling_guard"] = "stop_before_next_opportunity_if_12_proposals_could_overshoot"
    protocol["resource_upper_bounds"] = resource_upper_bounds()
    protocol["within_opportunity_target_crossing"] = "retain_all_returned_candidates_then_stop"
    protocol["diagnostic_protocol_repair_only"] = True
    protocol["scientific_method_change"] = False
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
        "experiment_id": EXPERIMENT_ID,
        "execution_source_sha": source,
        "preregistration_sha256": permit.preregistration_sha256,
        "run_identity_sha256": permit.run_identity_sha256,
        "provider_boundary_reached": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    print(json.dumps(prepare(parser.parse_args().prep), sort_keys=True, indent=2))
