"""Zero-API integrity preflight for accepted-mutation team transfer."""

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


IDENTITY = "accepted_local_mutation_team_transfer_v1"
REPORT = ROOT / "reports/accepted_local_mutation_team_transfer_v1_prep_20260919"
MANIFEST = ROOT / "experiments/manifests/accepted_local_mutation_team_transfer_v1.yaml"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(private_bundle: Path) -> dict:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = read(ROOT / "infrastructure/experiment_manifest.schema.json")
    freeze = read(REPORT / "accepted_mutation_freeze.json")
    private = read(private_bundle)
    mutations = private["accepted_mutations"]
    checks = {
        "manifest_schema": validate_manifest(manifest, schema) == [],
        "preregistration_hash": manifest["artifacts"]["preregistration"]["sha256"] == preregistration_hash(manifest),
        "api_authorization_pending": manifest["api_authorization"]["authorized"] is False,
        "private_bundle_hash": sha(private_bundle) == freeze["private_bundle_sha256"],
        "baseline_team_hash": private["baseline_team_hash"] == freeze["baseline_team_hash"],
        "exact_five_mutations": len(mutations) == freeze["mutation_count"] == 5,
        "candidate_prompt_hashes": all(
            hashlib.sha256(row["accepted_candidate_prompt"].encode()).hexdigest()
            == row["accepted_candidate_hash"] for row in mutations
        ),
        "deterministic_order": [(row["target_member"], row["source_proposal_index"]) for row in mutations]
            == [(1, 2), (2, 8), (3, 5), (4, 2), (4, 4)],
        "sanitization": scan_sanitized_artifacts(REPORT) == [],
        "report_hash_replay": read(REPORT / "sha256_manifest.json") == build_sha256_manifest(REPORT),
        "formal_run_absent": not (ROOT / manifest["design"]["formal_run_root"]).exists(),
        "handoff_not_ready": read(REPORT / "EXPERIMENT_HANDOFF.json")["READY_TO_RUN"] is False,
    }
    return {
        "status": "AUTHORIZATION_REQUIRED" if all(checks.values()) else "HOLD",
        "checks": checks, "api_calls": 0, "mutation_count": len(mutations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-bundle", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.private_bundle.resolve()), indent=2))


if __name__ == "__main__":
    main()
