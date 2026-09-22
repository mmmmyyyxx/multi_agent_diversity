"""Single current production entrypoint.

Manifests select backend, scope and stopping orthogonally. A centralized
execution composition function supplies providers, datasets and persistence;
the scientific engine remains independent of those details.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.experiment import (  # noqa: E402
    ExperimentInputs,
    ExperimentServices,
    experiment_spec_from_mapping,
    run_experiment,
    runtime_context_from_mapping,
)


def _load(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("production manifest must be an object")
    return value


def _composition_factory(reference: str):
    if reference.count(":") != 1:
        raise ValueError("execution_factory must be module:function")
    module_name, function_name = reference.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("execution_factory is not callable")
    return factory


def preflight(manifest: Mapping[str, Any]) -> dict[str, Any]:
    spec = experiment_spec_from_mapping(manifest.get("scientific", {}))
    runtime = runtime_context_from_mapping(manifest.get("runtime", {}))
    factory = str(manifest.get("execution_factory", ""))
    return {
        "gate": "PASS" if factory else "HOLD",
        "mode": spec.mode_id,
        "stopping_regime": spec.stopping_regime.value,
        "spec_identity": spec.identity(),
        "seed": runtime.seed,
        "execution_factory_bound": bool(factory),
        "provider_attempts": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }


async def _execute(manifest: Mapping[str, Any]):
    spec = experiment_spec_from_mapping(manifest.get("scientific", {}))
    runtime = runtime_context_from_mapping(manifest.get("runtime", {}))
    factory = _composition_factory(str(manifest.get("execution_factory", "")))
    inputs, services = factory(spec=spec, runtime=runtime, manifest=manifest)
    if not isinstance(inputs, ExperimentInputs) or not isinstance(services, ExperimentServices):
        raise TypeError("execution factory must return (ExperimentInputs, ExperimentServices)")
    return await run_experiment(spec, runtime, inputs, services)


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
    result = asyncio.run(_execute(manifest))
    print(json.dumps({
        "mode": result.mode_id,
        "stopping_regime": result.stopping_regime,
        "stop_reason": result.stop_reason,
        "event_count": len(result.events),
        "final_state_hash": result.final_state_hash,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
