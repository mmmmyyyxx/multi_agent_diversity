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


def scan(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if not lowered.endswith("_hash") and any(x in lowered for x in FORBIDDEN_KEYS):
                raise ValueError(f"sensitive key at {location}/{key}")
            scan(child, f"{location}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan(child, f"{location}/{index}")
    elif isinstance(value, str) and ABSOLUTE_PATH.search(value):
        raise ValueError(f"sensitive value at {location}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(audit_path: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("fresh report destination required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("gate") != "PASS":
        raise ValueError("protocol gate must pass before packaging")
    destination.mkdir(parents=True)
    summary = {key: audit[key] for key in (
        "gate", "blockers", "execution_commit", "experiment_version",
        "validation_evaluations", "test_evaluations", "formal_efficacy_classifier",
    )}
    rows = audit["rows"]
    arms = sorted({row["arm"] for row in rows})
    summary["arm_aggregates"] = {
        arm: {
            "seed_count": len(selected := [row for row in rows if row["arm"] == arm]),
            "mean_final_train_vote_accuracy": sum(row["final_train_vote_accuracy"] for row in selected) / len(selected),
            "mean_g_min": sum(row["g_min"] for row in selected) / len(selected),
            "mean_g_sum": sum(row["g_sum"] for row in selected) / len(selected),
            "total_accepted_updates": sum(row["accepted_updates"] for row in selected),
            "total_provider_calls": sum(row["provider_calls"] for row in selected),
            "total_tokens": sum(row["total_tokens"] for row in selected),
            "total_generic_revision_committed": sum(row["generic_revision_committed"] for row in selected),
            "total_repair_committed": sum(row["repair_committed"] for row in selected),
        }
        for arm in arms
    }
    scan(summary)
    (destination / "efficacy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, table in (("seed_metrics.csv", rows), ("paired_contrasts.csv", audit["paired_contrasts"])):
        with (destination / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    readme = (
        "# Module2 compute-matched efficacy — Seeds 53-55\n\n"
        "Sanitized train-only evidence for G-Matched, R-M20, and R-M2F. "
        "Validation and test evaluations are zero. Raw prompts, examples, answers, "
        "responses, caches, checkpoints, credentials, endpoints, and local paths "
        "remain local and are not included.\n\n"
        "The primary outcomes are final train VoteAcc, g_min, and g_sum. Mechanism "
        "and cost fields are secondary; third-layer specialization fields are "
        "analysis-only.\n"
    )
    (destination / "README.md").write_text(readme, encoding="utf-8")
    for path in destination.iterdir():
        if path.suffix == ".json":
            scan(json.loads(path.read_text(encoding="utf-8")))
        elif path.suffix in {".csv", ".md"} and ABSOLUTE_PATH.search(path.read_text(encoding="utf-8")):
            raise ValueError(f"sensitive path in {path.name}")
    files = [{"file": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size}
             for path in sorted(destination.iterdir())]
    (destination / "sanitized_manifest.json").write_text(
        json.dumps({"sanitization": "PASS", "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    package(args.audit, args.destination)


if __name__ == "__main__":
    main()
