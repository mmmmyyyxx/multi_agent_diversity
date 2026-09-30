"""Capture private deterministic Formal V3 fake executions for old/new comparison.

Run only through tests/formal_zero_api_runner.py --offline-command. The output
contains private synthetic prompts and stays under ignored runs/ storage.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _helper():
    source = ROOT / "tests/test_formal_v3_full_fake.py"
    spec = importlib.util.spec_from_file_location("formal_v3_fake_reference", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing frozen formal fake fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _split_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [hashlib.sha256(row["question"].encode("utf-8")).hexdigest()
                for row in csv.DictReader(handle)]


def main(out: Path, *, solver_concurrency: int | None = None) -> None:
    if out.exists():
        raise FileExistsError("equivalence capture must use a fresh ignored directory")
    out.mkdir(parents=True)
    helper = _helper()
    cases = (
        ("native", "native", None, False, False),
        ("layer2_no_commit", "layer2", None, False, False),
        ("layer2_shadow_reject", "layer2",
         "```Check pronoun agreement against each referent.```", True, True),
        ("layer2_commit", "layer2",
         "```Check pronoun agreement against each referent.```", True, False),
    )
    capture: dict[str, object] = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
        "cases": {},
    }
    for name, scope, reflection, improves, shadow_hurts in cases:
        patch = pytest.MonkeyPatch()
        try:
            (out / name).mkdir()
            permit, calls = helper._formal_fixture(
                out / name, patch, scope=scope, reflection_text=reflection,
                candidate_improves=improves, shadow_candidate_hurts=shadow_hurts,
            )
            if solver_concurrency is not None:
                manifest_path = permit.prep_root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["runtime"]["eval_solver_call_concurrency"] = solver_concurrency
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            reflection_requests: list[str] = []
            reflection_responses: list[str] = []

            def instrument(factory):
                def construct(*args, **kwargs):
                    client = factory(*args, **kwargs)
                    completions = client.chat.completions
                    if not getattr(completions, "_equivalence_observed", False):
                        original = completions.create

                        async def observed_create(**request):
                            reflection = request.get("model") == "qwen3.7-flash"
                            if reflection:
                                reflection_requests.append(hashlib.sha256(
                                    json.dumps(request, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")
                                ).hexdigest())
                            response = await original(**request)
                            if reflection:
                                reflection_responses.append(hashlib.sha256(
                                    str(response.choices[0].message.content).encode("utf-8")
                                ).hexdigest())
                            return response

                        completions.create = observed_create
                        completions._equivalence_observed = True
                    return client
                return construct

            patch.setattr(helper.ProviderClientFactory, "from_environment",
                          instrument(helper.ProviderClientFactory.from_environment))
            patch.setattr(helper.ProviderClientFactory, "create",
                          instrument(helper.ProviderClientFactory.create))
            patch.setattr(helper.formal_runner, "validate_execution", lambda **_kwargs: permit)
            patch.setattr(helper.formal_runner, "admit_execution", lambda _permit, _run: permit)
            result = asyncio.run(helper.formal_runner.execute_frozen(
                permit.prep_root, permit.run_root,
            ))
            ledger = _json_rows(permit.run_root / "ledger.jsonl")
            normalized_ledger = [
                {key: value for key, value in row.items() if key != "record_id"}
                for row in ledger
            ]
            lineage = {
                str(path.relative_to(permit.run_root)).replace("\\", "/"): _json_rows(path)
                for path in sorted(permit.run_root.rglob("*.lineage.jsonl"))
            }
            capture["cases"][name] = {
                "summary": result,
                "ledger_semantic_multiset": sorted(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")), count)
                    for row, count in Counter(
                        json.dumps(row, sort_keys=True, separators=(",", ":"))
                        for row in normalized_ledger
                    ).items()
                ),
                "lineage": lineage,
                "fake_physical_calls": calls,
                "optimize_question_ids": _split_ids(permit.prep_root / "splits_private/optimize100.csv"),
                "shadow_question_ids": _split_ids(permit.prep_root / "splits_private/shadow50.csv"),
                "reflection_request_sha256": reflection_requests,
                "reflection_response_sha256": reflection_responses,
            }
            print(name, result["stop_reason"], len(ledger), flush=True)
        finally:
            patch.undo()
    (out / "private_semantic_capture.json").write_text(
        json.dumps(capture, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solver-concurrency", type=int)
    args = parser.parse_args()
    main(args.out, solver_concurrency=args.solver_concurrency)
