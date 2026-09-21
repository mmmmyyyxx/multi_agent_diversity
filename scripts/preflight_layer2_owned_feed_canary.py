"""Fail-closed zero-API preflight for Layer-2-owned evidence canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {
    "README.md",
    "ownership_contract.json",
    "responsibility_packet_schema.json",
    "example_selection_audit.json",
    "backend_no_selection_audit.json",
    "direct_layer2_influence_audit.json",
    "heldout_isolation.json",
    "backend_fidelity_manifest.json",
    "pilot_protocol.json",
    "sha256_manifest.json",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gepa", "mars"), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = args.report_dir.resolve()
    manifest = read(args.manifest.resolve())
    names = {path.name for path in report.iterdir()} if report.is_dir() else set()
    checks = {
        "report_complete": REQUIRED.issubset(names),
        "authorization_false": manifest.get("authorized") is False,
        "ready_to_run_false": manifest.get("READY_TO_RUN") is False,
        "fresh_run_root_not_consumed": manifest.get("formal_run_root_created") is False,
        "real_provider_calls": 0,
        "Validation50_calls": 0,
        "Test50_calls": 0,
    }
    if checks["report_complete"]:
        heldout = read(report / "heldout_isolation.json")
        selection = read(report / "backend_no_selection_audit.json")
        influence = read(report / "direct_layer2_influence_audit.json")
        checks.update(
            {
                "heldout_isolated": heldout.get("status") == "PASS",
                "backend_selection_zero": selection.get(
                    "backend_example_selection_calls"
                ) == 0,
                "direct_layer2_influence": influence.get("status") == "PASS",
            }
        )
    status = "READY_FOR_AUTHORIZATION" if all(
        value is True or value == 0 for value in checks.values()
    ) else "HOLD"
    payload = {
        "backend": args.backend,
        "status": status,
        "READY_TO_RUN": False,
        "reason": "real-provider authorization is intentionally absent",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
