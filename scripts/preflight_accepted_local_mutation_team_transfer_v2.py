"""Zero-API integrity preflight for mandatory-Full team-transfer v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest


REPORT = ROOT / "reports/accepted_local_mutation_team_transfer_v2_prep_20260919"
MANIFEST = ROOT / "experiments/manifests/accepted_local_mutation_team_transfer_v2.yaml"
V1_REPORT = ROOT / "reports/accepted_local_mutation_team_transfer_v1_prep_20260919"
V1_MANIFEST = ROOT / "experiments/manifests/accepted_local_mutation_team_transfer_v1.yaml"
V1_PROTOCOL = ROOT / "experiments/accepted_local_mutation_team_transfer_v1/PROTOCOL.md"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(private_bundle: Path, v1_private_bundle: Path) -> dict:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = read(ROOT / "infrastructure/experiment_manifest.schema.json")
    freeze = read(REPORT / "accepted_mutation_freeze.json")
    private = read(private_bundle)
    v1_private = read(v1_private_bundle)
    supersession = read(REPORT / "v1_supersession.json")
    checks = {
        "manifest_schema": validate_manifest(manifest, schema) == [],
        "preregistration_hash": manifest["artifacts"]["preregistration"]["sha256"] == preregistration_hash(manifest),
        "authorization_pending": manifest["api_authorization"]["authorized"] is False,
        "private_bundle_hash": sha(private_bundle) == freeze["private_bundle_sha256"],
        "same_baseline": private["baseline_team_hash"] == v1_private["baseline_team_hash"],
        "same_five_mutations": private["accepted_mutations"] == v1_private["accepted_mutations"],
        "candidate_prompt_hashes": all(
            hashlib.sha256(row["accepted_candidate_prompt"].encode()).hexdigest()
            == row["accepted_candidate_hash"] for row in private["accepted_mutations"]
        ),
        "mandatory_full_five": manifest["design"]["mandatory_full_candidate_count"] == 5,
        "minibatch_diagnostic": manifest["design"]["team_minibatch_role"] == "TEAM_MINIBATCH_DIAGNOSTIC_GATE_V1",
        "v1_protocol_unchanged": sha(V1_PROTOCOL) == supersession["v1_protocol_sha256"],
        "v1_manifest_unchanged": sha(V1_MANIFEST) == supersession["v1_manifest_sha256"],
        "v1_private_unchanged": sha(v1_private_bundle) == supersession["v1_private_bundle_sha256"],
        "v1_report_unchanged": build_sha256_manifest(V1_REPORT) == supersession["v1_report_tree"],
        "sanitization": scan_sanitized_artifacts(REPORT) == [],
        "report_hash_replay": read(REPORT / "sha256_manifest.json") == build_sha256_manifest(REPORT),
        "formal_run_absent": not (ROOT / manifest["design"]["formal_run_root"]).exists(),
        "handoff_not_ready": read(REPORT / "EXPERIMENT_HANDOFF.json")["READY_TO_RUN"] is False,
    }
    return {
        "status": "AUTHORIZATION_REQUIRED" if all(checks.values()) else "HOLD",
        "checks": checks, "api_calls": 0,
        "mutation_count": len(private["accepted_mutations"]),
        "mandatory_full_candidates": manifest["design"]["mandatory_full_candidate_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-bundle", type=Path, required=True)
    parser.add_argument("--v1-private-bundle", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.private_bundle.resolve(), args.v1_private_bundle.resolve()), indent=2))


if __name__ == "__main__":
    main()
