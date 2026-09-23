"""Single current production entrypoint.

Manifests select backend, scope and stopping orthogonally. A centralized
execution composition function supplies providers, datasets and persistence;
the scientific engine remains independent of those details.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.experiment import (  # noqa: E402
    ExperimentContractError,
    experiment_spec_from_mapping,
    runtime_context_from_mapping,
)


def _load(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("production manifest must be an object")
    return value


def preflight(manifest: Mapping[str, Any]) -> dict[str, Any]:
    spec = experiment_spec_from_mapping(manifest.get("scientific", {}))
    runtime = runtime_context_from_mapping(manifest.get("runtime", {}))
    factory = str(manifest.get("execution_factory", ""))
    return {
        # A factory reference is composition metadata, not an authorization or
        # a verified source/preregistration binding. Keep real execution closed
        # until a frozen execution adapter is connected to this CLI.
        "gate": "HOLD",
        "blockers": ["FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND"],
        "mode": spec.mode_id,
        "method_identity": spec.method_identity,
        "stopping_regime": spec.stopping_regime.value,
        "spec_identity": spec.identity(),
        "seed": runtime.seed,
        "execution_factory_declared": bool(factory),
        "provider_attempts": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }


async def _execute(manifest: Mapping[str, Any]):
    # The typed engine can run deterministic offline fixtures through its
    # public API. This CLI is reserved for a future frozen execution adapter;
    # no manifest-provided function may construct a provider before governance.
    raise ExperimentContractError("ABORT_PRE_PROVIDER: FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = _load(args.manifest)
    if args.preflight:
        print(json.dumps(preflight(manifest), indent=2, sort_keys=True))
        return
    asyncio.run(_execute(manifest))


if __name__ == "__main__":
    main()
