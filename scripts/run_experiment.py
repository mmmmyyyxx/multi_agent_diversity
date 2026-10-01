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
from multi_dataset_diverse_rl.search.schemas import SearchMethodConfig  # noqa: E402
from multi_dataset_diverse_rl.governance.production_execution import (  # noqa: E402
    admit_execution, terminal_lifecycle, validate_execution,
)
from multi_dataset_diverse_rl.production_canary import (  # noqa: E402
    execute_post_refactor_canary,
)
from multi_dataset_diverse_rl.production_transfer_diagnostic import (  # noqa: E402
    execute_online_transfer_diagnostic,
)
from multi_dataset_diverse_rl.production_formal_saturation import (  # noqa: E402
    execute_formal_gepa_saturation,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import (  # noqa: E402
    ledger_summary,
)
from multi_dataset_diverse_rl.persistence.durable_io import (  # noqa: E402
    atomic_write_json, canonical_json_payload, read_json,
)


def _load(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("production manifest must be an object")
    return value


def preflight(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("execution_binding") is not None:
        from multi_dataset_diverse_rl.governance.unified_execution import bound_preflight
        return bound_preflight(ROOT, manifest)
    scientific = manifest.get("scientific", {})
    if isinstance(scientific, dict) and scientific.get("method") in {"unified_team_prompt_search_v1", "unified_team_prompt_search_v2"}:
        method = SearchMethodConfig.from_mapping({key: value for key, value in scientific.items()
                                                 if key != "benchmark_id"})
        runtime = runtime_context_from_mapping(manifest.get("runtime", {}))
        from multi_dataset_diverse_rl.benchmarks.registry import benchmark_preflight
        benchmark_id = str(manifest.get("benchmark_id", scientific.get("benchmark_id", "bbh")))
        capability = benchmark_preflight(benchmark_id) if benchmark_id != "bbh" else None
        return {
            "gate": "HOLD",
            "blockers": (["FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND"]
                         + (list(capability["blockers"]) if capability else [])),
            "benchmark_id": benchmark_id,
            "method": method.method,
            "method_identity": method.identity(),
            "seed": runtime.seed,
            "aggregation_model": runtime.optimizer_model,
            "provider_attempts": 0,
            "validation_calls": 0,
            "test_calls": 0,
        }
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
    if (prep / "prep.json").is_file():
        from multi_dataset_diverse_rl.governance.unified_execution import validate_prep
        value = validate_prep(ROOT, prep)
        return {"gate": "CANARY_READY_NOT_AUTHORIZED", "ready_for_authorization": True,
                "attempt_id": value["scope"]["attempt_id"], "startup_identity_sha256": value["startup_identity_sha256"],
                "provider_attempts": 0, "validation_calls": 0, "test_calls": 0}
    permit = validate_execution(root=ROOT, prep=prep, require_authorized=False)
    formal_pending = permit.experiment_id.startswith("gepa_saturation_comparison_v3_")
    gate = "PREREGISTERED_NOT_EXECUTED"
    if formal_pending:
        manifest = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
        if permit.attempt_id.endswith("_attempt1"):
            gate = "SUPERSEDED_BEFORE_EXECUTION"
        elif permit.attempt_id.endswith("_attempt2"):
            gate = (
                "INVALID_EXECUTION_CONFORMANCE"
                if permit.attempt_id == "gepa_saturation_comparison_v3_seed80_native_attempt2"
                else "SUPERSEDED_BEFORE_EXECUTION"
            )
        else:
            formal_pending = manifest.get("execution_gate", {}).get(
                "real_v4_diagnostic"
            ) != "SCIENTIFICALLY_VALID"
            gate = "PREREGISTERED_EXECUTION_GATED" if formal_pending else gate
    return {
        "gate": gate,
        "ready_for_authorization": not formal_pending and gate == "PREREGISTERED_NOT_EXECUTED",
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
    if (prep / "prep.json").is_file():
        from multi_dataset_diverse_rl.governance.unified_execution import execute_canary
        return await execute_canary(ROOT, prep, run_root)
    permit = validate_execution(root=ROOT, prep=prep, require_authorized=True)
    admitted = admit_execution(permit, run_root)
    try:
        if admitted.experiment_id in {
            "gepa_layer2_local_to_team_transfer_diagnostic_v2",
            "gepa_layer2_local_to_team_transfer_diagnostic_v3",
            "gepa_layer2_local_to_team_transfer_diagnostic_v4",
        }:
            result = await execute_online_transfer_diagnostic(admitted, root=ROOT)
        elif admitted.experiment_id.startswith("gepa_saturation_comparison_v3_"):
            result = await execute_formal_gepa_saturation(admitted, root=ROOT)
        else:
            result = await execute_post_refactor_canary(admitted, root=ROOT)
        summary_path = run_root / "execution_summary.json"
        persisted_result = canonical_json_payload(result)
        atomic_write_json(summary_path, persisted_result)
        if read_json(summary_path) != persisted_result:
            raise RuntimeError("execution summary read-back mismatch")
        usage = persisted_result["ledger"]
        terminal_lifecycle(
            admitted, status="EXECUTION_COMPLETE",
            provider_attempts=int(usage["provider_attempts"]),
            provider_successes=int(usage["successful_provider_calls"]),
            provider_failures=int(usage["failed_provider_attempts"]),
        )
        return persisted_result
    except BaseException:
        ledger_path = run_root / "ledger.jsonl"
        usage = ledger_summary(ledger_path) if ledger_path.exists() else {}
        attempts = int(usage.get("provider_attempts", 0))
        lifecycle_path = run_root / "run_lifecycle.json"
        try:
            lifecycle_status = read_json(lifecycle_path).get("status")
        except FileNotFoundError:
            # Only test doubles can admit without the required run-local fact;
            # the terminal writer remains the authoritative fail-closed check.
            lifecycle_status = "RUNNING"
        if lifecycle_status == "RUNNING":
            terminal_lifecycle(
                admitted, status="ABORTED" if attempts else "FAILED_START",
                provider_attempts=attempts,
                provider_successes=int(usage.get("successful_provider_calls", 0)),
                provider_failures=int(usage.get("failed_provider_attempts", 0)),
            )
        raise


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
        prep = args.prep if args.prep is None or args.prep.is_absolute() else ROOT / args.prep
        manifest = args.manifest if args.manifest is None or args.manifest.is_absolute() else ROOT / args.manifest
        result = governed_preflight(prep) if prep else preflight(_load(manifest))
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.prep is None or args.run_root is None:
        raise ExperimentContractError("ABORT_PRE_PROVIDER: frozen prep and fresh run root are required")
    prep = args.prep if args.prep.is_absolute() else ROOT / args.prep
    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    print(json.dumps(asyncio.run(execute_frozen(prep, run_root)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
