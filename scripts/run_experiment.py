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
from multi_dataset_diverse_rl.governance.production_execution import (  # noqa: E402
    admit_execution, terminal_lifecycle, validate_execution,
)
from multi_dataset_diverse_rl.production_canary import (  # noqa: E402
    execute_post_refactor_canary,
)
from multi_dataset_diverse_rl.production_transfer_diagnostic import (  # noqa: E402
    execute_online_transfer_diagnostic,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import (  # noqa: E402
    ledger_summary,
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


def governed_preflight(prep: Path) -> dict[str, Any]:
    permit = validate_execution(root=ROOT, prep=prep, require_authorized=False)
    return {
        "gate": "PREREGISTERED_NOT_EXECUTED",
        "ready_for_authorization": True,
        "attempt_id": permit.attempt_id,
        "preregistration_sha256": permit.preregistration_sha256,
        "run_identity_sha256": permit.run_identity_sha256,
        "provider_boundary_reached": False,
        "provider_attempts": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }


async def _execute(manifest: Mapping[str, Any]):
    # Legacy in-memory manifests never grant real provider access. Explicit
    # offline fixtures use the engine API with injected fake services.
    del manifest
    raise ExperimentContractError("ABORT_PRE_PROVIDER: FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND")


async def execute_frozen(prep: Path, run_root: Path) -> dict[str, Any]:
    permit = validate_execution(root=ROOT, prep=prep, require_authorized=True)
    admitted = admit_execution(permit, run_root)
    try:
        if admitted.experiment_id == "gepa_layer2_local_to_team_transfer_diagnostic_v1":
            result = await execute_online_transfer_diagnostic(admitted, root=ROOT)
        else:
            result = await execute_post_refactor_canary(admitted, root=ROOT)
    except BaseException:
        ledger_path = run_root / "ledger.jsonl"
        usage = ledger_summary(ledger_path) if ledger_path.exists() else {}
        attempts = int(usage.get("provider_attempts", 0))
        terminal_lifecycle(
            admitted, status="ABORTED" if attempts else "FAILED_START",
            provider_attempts=attempts,
            provider_successes=int(usage.get("successful_provider_calls", 0)),
            provider_failures=int(usage.get("failed_provider_attempts", 0)),
        )
        raise
    usage = result["ledger"]
    terminal_lifecycle(
        admitted, status="EXECUTION_COMPLETE",
        provider_attempts=int(usage["provider_attempts"]),
        provider_successes=int(usage["successful_provider_calls"]),
        provider_failures=int(usage["failed_provider_attempts"]),
    )
    path = run_root / "execution_summary.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--prep", type=Path)
    parser.add_argument("--run-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = governed_preflight(args.prep) if args.prep else preflight(_load(args.manifest))
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.prep is None or args.run_root is None:
        raise ExperimentContractError("ABORT_PRE_PROVIDER: frozen prep and fresh run root are required")
    print(json.dumps(asyncio.run(execute_frozen(args.prep, args.run_root)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
