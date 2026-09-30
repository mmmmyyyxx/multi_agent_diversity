"""Offline 50 ms fake-Solver calibration for one GEPA evaluation batch."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import time


def _fixture():
    path = Path(__file__).resolve().parents[1] / "tests/test_formal_v3_concurrent_local_batch.py"
    spec = importlib.util.spec_from_file_location("concurrent_batch_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing local batch fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _measure(module, concurrency: int, *, serial: bool) -> dict:
    system = module.FakeSystem(32, concurrency, delay=0.05, vary_delay=False)
    adapter, stages = module._adapter(system, asyncio.get_running_loop())
    if serial:
        evaluator = adapter.evaluator

        class SerialView:
            solver_contract_id = evaluator.solver_contract_id
            output_contract_id = evaluator.output_contract_id

            def evaluate(self, decision_procedure, example):
                return evaluator.evaluate(decision_procedure, example)

        adapter.evaluator = SerialView()
    batch = module._batch(system, list(range(32)))
    start = time.perf_counter()
    result = await asyncio.to_thread(
        adapter.evaluate, batch, {"decision_procedure": "Use context clues."}, True,
    )
    elapsed = time.perf_counter() - start
    assert stages[-1] is None
    assert system.physical_calls == len(result.scores) == 32
    assert result.scores == [float(i % 2 == 0) for i in range(32)]
    return {"batch_size": 32, "configured_concurrency": concurrency,
            "elapsed_seconds": round(elapsed, 4), "max_inflight": system.max_inflight,
            "physical_calls": system.physical_calls, "logical_evaluations": len(result.scores)}


def main(out: Path) -> None:
    if os.environ.get("FORMAL_ZERO_API_GUARD_REQUIRED") != "1":
        raise RuntimeError("pre-import zero-network guard is required")
    if out.exists():
        raise FileExistsError("benchmark output must be new")
    module = _fixture()

    async def measure():
        reference = await _measure(module, 1, serial=True)
        variants = [await _measure(module, value, serial=False)
                    for value in (1, 4, 8, 16, 32)]
        return reference, variants

    reference, variants = asyncio.run(measure())
    payload = {
        "schema_version": "formal_v3_local_batch_fake_benchmark_v1",
        "fake_delay_ms_per_request": 50,
        "serial_reference": reference,
        "concurrent_variants": [
            {**row, "speedup_vs_serial": round(
                reference["elapsed_seconds"] / row["elapsed_seconds"], 3
            )}
            for row in variants
        ],
        "all_scientific_evaluations_preserved": all(
            row["logical_evaluations"] == reference["logical_evaluations"] == 32
            and row["physical_calls"] == reference["physical_calls"] == 32
            for row in variants
        ),
        "real_provider_calls": 0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
