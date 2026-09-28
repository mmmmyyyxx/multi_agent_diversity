"""Zero-API fresh V4 freeze; no authorization or formal run creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.versions import (  # noqa: E402
    LAYER2_EVIDENCE_PACKET_V4_VERSION,
    LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
    LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
    LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
    PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
    LAYER2_RESPONSIBILITY_SOURCE_VERSION,
    TEAM_MINIBATCH_CONTRACT_VERSION,
)
from scripts.prepare_online_transfer_diagnostic_v3 import (  # noqa: E402
    _git, _private_splits, _expected_bundle, validate_execution,
    canonical_json_bytes, write_bundle, frozen_payload as v3_frozen_payload,
)
from multi_dataset_diverse_rl.governance.v4_source_closure import (  # noqa: E402
    active_v4_source_paths,
)


EXPERIMENT_ID = "gepa_layer2_local_to_team_transfer_diagnostic_v4"
ATTEMPT2_ID = "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt2"
DEFAULT_PREP = ROOT / "runs" / ATTEMPT2_ID / "prep"


def frozen_payload(*, execution_source_sha: str,
                   attempt_id: str = EXPERIMENT_ID) -> tuple[dict, dict]:
    if attempt_id not in {EXPERIMENT_ID, ATTEMPT2_ID}:
        raise ValueError("unsupported V4 diagnostic attempt")
    manifest, protocol = v3_frozen_payload(execution_source_sha=execution_source_sha)
    manifest["schema_version"] = "online_transfer_diagnostic_freeze_v4"
    manifest["experiment_id"] = EXPERIMENT_ID
    manifest["attempt_id"] = attempt_id
    manifest["execution"]["scientific_method_anchor_sha"] = execution_source_sha
    if attempt_id == ATTEMPT2_ID:
        manifest["execution"]["source_paths"] = list(active_v4_source_paths(ROOT))
        manifest["execution_refresh"] = {
            "scientific_method_changed": False,
            "execution_implementation_refreshed": True,
            "supersedes_for_execution": EXPERIMENT_ID,
            "reason": "current_shared_v4_selector_and_complete_source_identity_closure",
        }
    else:
        # Preserve the historical attempt's payload constructor for replay.
        manifest["execution"]["source_paths"] = sorted(
            set(manifest["execution"]["source_paths"]) | {
                "scripts/prepare_online_transfer_diagnostic_v4.py",
                "experiments/gepa_layer2_local_to_team_transfer_diagnostic_v4/PROTOCOL.md",
            }
        )
    manifest["integrity_amendment"] = {
        "supersedes_unexecuted_attempt": "gepa_layer2_local_to_team_transfer_diagnostic_v3",
        "reason": "approved_bounded_evidence_and_feasibility_admission",
        "scientific_method_changed": True,
    }
    manifest["semantic_contract"] = {
        "layer2_protocol": LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
        "responsibility_source": LAYER2_RESPONSIBILITY_SOURCE_VERSION,
        "scheduler": PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
        "feasibility_policy": LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
        "packet_version": LAYER2_EVIDENCE_PACKET_V4_VERSION,
        "evidence_selection_policy": LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
        "team_minibatch": TEAM_MINIBATCH_CONTRACT_VERSION,
        "local_eval_identity": "same_frozen_team_minibatch12_ids",
        "local_acceptance": "target_member_only_no_team_feedback",
        "responsibility_value": "raw_V_not_discounted_target_score",
        "no_feasible_stop": "NO_FEASIBLE_LAYER2_OPPORTUNITY",
    }
    protocol["schema_version"] = "online_local_to_team_transfer_diagnostic_v4"
    protocol["experiment_id"] = EXPERIMENT_ID
    protocol["semantic_contract"] = manifest["semantic_contract"]
    protocol["target_policy"] = "preselect_evidence_feasible_then_original_V_over_one_plus_f"
    protocol["packet_capacity"] = {
        "local_metric_budget": 36,
        "batch_size": 3,
        "nominal_role_item_slots": 36,
        "focus_anchor": "exact_latest_transition_never_truncated",
        "repair": "first_min_Rfull_capacity_existing_deterministic_order",
        "delivery": "separately_observed_not_implied_by_schedule",
    }
    protocol["scientific_method_change"] = True
    return manifest, protocol


def prepare(prep: Path) -> dict[str, str | bool]:
    if prep.exists():
        raise FileExistsError("fresh prep root required")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before freeze")
    source = _git("rev-parse", "HEAD")
    manifest, protocol = frozen_payload(
        execution_source_sha=source, attempt_id=ATTEMPT2_ID,
    )
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
