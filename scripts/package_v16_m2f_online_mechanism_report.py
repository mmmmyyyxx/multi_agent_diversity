from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ABSOLUTE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|file://|https?://|sqlite://)")
FORBIDDEN_KEYS = (
    "prompt", "question", "gold", "answer", "response", "endpoint",
    "credential", "api_key", "cache", "checkpoint", "path",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def scan(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if not lowered.endswith("_hash") and any(token in lowered for token in FORBIDDEN_KEYS):
                raise ValueError(f"sensitive key at {location}/{key}")
            scan(child, f"{location}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan(child, f"{location}/{index}")
    elif isinstance(value, str) and ABSOLUTE_PATH.search(value):
        raise ValueError(f"sensitive value at {location}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(run: Path, audit_path: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("fresh report destination required")
    audit = read_json(audit_path)
    if audit.get("gate") != "PASS":
        raise ValueError("protocol gate must pass before packaging")
    events = read_jsonl(run / "online_compatibility_repair_events.jsonl")
    destination.mkdir(parents=True)
    summary = {
        "report_version": "v16_m2f_online_mechanism_seed52_v1",
        **audit,
        "online_contribution_status": (
            "OBSERVED" if audit["repair_attributable_accepted_updates"] > 0
            else "NOT_OBSERVED_IN_SEED52"
        ),
        "formal_compute_matched_claim": "NOT_TESTED",
    }
    scan(summary)
    (destination / "mechanism_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "update_index", "target_agent_id", "parent_team_hash",
        "responsibility_evidence_hash", "source_candidate_hash",
        "source_responsibility_gain", "source_target_gain", "source_vote_gain",
        "source_vote_loss", "source_common_safe", "repair_eligible",
        "repair_attempted", "repair_output_valid", "repaired_candidate_hash",
        "retained_source_responsibility_repairs", "repaired_responsibility_gain",
        "repaired_target_gain", "repaired_vote_gain", "repaired_vote_loss",
        "repaired_common_safe", "repair_feasible", "repair_committed",
    )
    with (destination / "repair_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow({key: event.get(key) for key in fields})
    readme = (
        "# M2F Train-Only Online Mechanism Pilot — Seed 52\n\n"
        "This directory contains sanitized mechanism evidence only. The raw "
        "training run remains local and ignored. No prompt, question, gold/model "
        "answer, raw response, cache, checkpoint, credential, endpoint, or local "
        "path is included.\n\n"
        f"Protocol gate: `{audit['gate']}`. Repair-attributable accepted updates: "
        f"`{audit['repair_attributable_accepted_updates']}`. This single-seed "
        "pilot does not establish the later compute-matched formal comparison.\n"
    )
    if ABSOLUTE_PATH.search(readme):
        raise ValueError("README sanitization failed")
    (destination / "README.md").write_text(readme, encoding="utf-8")
    manifest_rows = []
    for path in sorted(destination.iterdir()):
        if path.name == "sanitized_manifest.json":
            continue
        manifest_rows.append({
            "file": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
    manifest = {"sanitization": "PASS", "files": manifest_rows}
    scan(manifest)
    (destination / "sanitized_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    package(args.run, args.audit, args.destination)


if __name__ == "__main__":
    main()
