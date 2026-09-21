"""Zero-API preflight for prepared native-feed canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gepa", "mars"), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = args.report_dir.resolve()
    required = {
        "README.md",
        "upstream_source_manifest.json",
        "native_feed_manifest.json",
        "layer2_contract_hash.json",
        "fake_provider_native_flow.json",
        "heldout_isolation.json",
        "sha256_manifest.json",
    }
    checks = {
        "report_complete": report.is_dir()
        and required.issubset(path.name for path in report.iterdir()),
        "fake_provider_pass": False,
        "heldout_isolated": False,
        "api_authorization_absent": True,
        "real_api_calls": 0,
    }
    if checks["report_complete"]:
        fake = json.loads((report / "fake_provider_native_flow.json").read_text())
        heldout = json.loads((report / "heldout_isolation.json").read_text())
        checks["fake_provider_pass"] = fake.get("status") == "PASS"
        checks["heldout_isolated"] = (
            heldout.get("Validation50_calls") == 0
            and heldout.get("Test50_calls") == 0
        )
    payload = {
        "backend": args.backend,
        "status": "READY_FOR_AUTHORIZATION" if (
            checks["report_complete"] is True
            and checks["fake_provider_pass"] is True
            and checks["heldout_isolated"] is True
            and checks["api_authorization_absent"] is True
            and checks["real_api_calls"] == 0
        ) else "HOLD",
        "READY_TO_RUN": False,
        "reason": "manifest authorization remains false until a later explicit task",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
