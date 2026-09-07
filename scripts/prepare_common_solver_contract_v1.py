"""Freeze COMMON_SOLVER_CONTRACT_V1 and a zero-API cross-repo replay plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.common_solver_contract_v1.contract import (
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
    canonical_json_bytes,
    canonical_question_payload,
    contract_identity,
    public_contract_manifest,
)
from infrastructure.common_solver_contract_v1.entrypoints import (
    from_diversity,
    from_gepa,
    from_mars,
)


REPORT_NAMES = (
    "README.md",
    "contract_manifest.json",
    "serialization_parity_smoke.json",
    "replay_registry_public.json",
    "fact_assertions.json",
    "provenance.json",
    "sanitization_manifest.json",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def render_mars_question(question: str) -> str:
    value = canonical_question_payload(question)
    stem, options = value.split("\nOptions:\n", 1)
    parsed = re.findall(r"^\(([A-Z])\)\s*(.*?)\s*$", options, re.MULTILINE)
    return "\n".join([stem, "Options:", *(f"({label}) {text}" for label, text in parsed)])


def _state(state_id: str, repo: str, seed: int | None, prompts: Sequence[str]) -> dict[str, Any]:
    exact = [str(value) for value in prompts]
    return {
        "state_id": state_id,
        "source_repo": repo,
        "source_seed": seed,
        "aggregation": "single" if len(exact) == 1 else "equal_weight_plurality_tie_incorrect",
        "ordered_prompts": exact,
        "ordered_prompt_sha256": [sha256_bytes(value.encode("utf-8")) for value in exact],
        "unique_prompt_count": len(set(exact)),
    }


def load_inputs(
    root: Path, seed: int = 76
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    sibling = root.parent
    gepa = sibling / "independent_gepa_repro"
    mars = sibling / "MARS"
    files = {
        "external_validation_diversity": root
        / "runs/anti_overfitting_shadow_gate_v1_prep_20260904/splits_private/validation.csv",
        "external_validation_gepa": gepa
        / "data/private_bundles/gepa_capacity_probe_disambiguation_qa_20260905/splits/external_validation.jsonl",
        "p0": gepa
        / "data/private_bundles/gepa_capacity_probe_disambiguation_qa_20260905/initialization/p0.txt",
        f"mars_seed{seed}": mars
        / f"experiments/mars_single_prompt_capacity_20260905_r1/private/run_seed{seed}/trajectory_private.json",
        f"gepa_seed{seed}": gepa
        / f"runs/gepa_single_prompt_capacity_20260905/seed_{seed}/candidate_state.json",
        f"diversity_seed{seed}": root
        / f"runs/vote_aligned_confirmatory_seed76_77_v1/seed{seed}/P1_SHADOW_VOTE_ALIGNED_GENERIC/best_prompts.json",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen inputs missing: {missing}")

    raw_rows = read_csv(files["external_validation_diversity"])
    gepa_rows = [
        json.loads(line)
        for line in files["external_validation_gepa"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_rows) != 50 or len(gepa_rows) != 50:
        raise AssertionError("ExternalValidation50 inventory changed")
    cases: list[dict[str, Any]] = []
    for position, (diversity_row, gepa_row) in enumerate(zip(raw_rows, gepa_rows, strict=True)):
        raw_question = str(diversity_row["question"])
        canonical = canonical_question_payload(raw_question)
        gepa_rendered = "\n".join(
            [
                str(gepa_row["question"]).strip(),
                "Options:",
                *(
                    f"({label}) {choice}"
                    for label, choice in zip(
                        gepa_row["option_labels"], gepa_row["choices"], strict=True
                    )
                ),
            ]
        )
        if canonical != canonical_question_payload(gepa_rendered):
            raise AssertionError(f"logical case mismatch at position {position}")
        cases.append(
            {
                "position": position,
                "case_id": sha256_bytes(canonical.encode("utf-8")),
                "question": canonical,
                "gold": str(diversity_row["answer"]).strip().strip("()").upper(),
                "gepa_example": gepa_row,
                "diversity_raw_question": raw_question,
                "mars_rendered_question": render_mars_question(raw_question),
            }
        )

    p0 = files["p0"].read_text(encoding="utf-8").strip()
    mars_payload = read_json(files[f"mars_seed{seed}"])
    mars_final = str(mars_payload["best_prompt"])
    if sha256_bytes(mars_final.encode("utf-8")) != str(mars_payload["best_prompt_hash"]):
        raise AssertionError("MARS frozen prompt hash mismatch")
    gepa_payload = read_json(files[f"gepa_seed{seed}"])["result"]
    best_idx = int(gepa_payload["best_idx"])
    gepa_final = str(gepa_payload["candidates"][best_idx]["system_prompt"])
    diversity_final = [str(value) for value in read_json(files[f"diversity_seed{seed}"])]
    if len(diversity_final) != 5:
        raise AssertionError("Diversity frozen final must contain five prompts")
    states = [
        _state("P0_COMMON", "COMMON", None, [p0, p0, p0, p0, p0]),
        _state(f"MARS_SEED{seed}_FINAL", "MARS", seed, [mars_final]),
        _state(f"GEPA_SEED{seed}_FINAL", "GEPA", seed, [gepa_final]),
        _state(f"DIVERSITY_SEED{seed}_P1_FINAL", "Diversity", seed, diversity_final),
    ]
    return cases, states, files


def serialization_smoke(cases: Sequence[Mapping[str, Any]], p0: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases[:20]:
        requests = {
            "Diversity": from_diversity(
                prompt=p0, raw_question=str(case["diversity_raw_question"])
            ),
            "MARS": from_mars(
                prompt=p0, rendered_question=str(case["mars_rendered_question"])
            ),
            "GEPA": from_gepa(prompt=p0, example=case["gepa_example"]),
        }
        payloads = {name: canonical_json_bytes(value) for name, value in requests.items()}
        if len(set(payloads.values())) != 1:
            raise AssertionError(f"serialization parity failed at position {case['position']}")
        rows.append(
            {
                "case_position": int(case["position"]),
                "case_id": str(case["case_id"]),
                "request_sha256": sha256_bytes(next(iter(payloads.values()))),
                "request_utf8_bytes": len(next(iter(payloads.values()))),
                "entrypoint_count": 3,
                "exact_bytes_equal": True,
            }
        )
    return {
        "gate": "PASS",
        "contract_id": COMMON_SOLVER_CONTRACT_ID,
        "case_selection": "FIRST_20_EXTERNALVALIDATION50_BY_FROZEN_ORDER",
        "case_count": len(rows),
        "entrypoints": ["Diversity", "GEPA", "MARS"],
        "all_serialized_request_bytes_equal": all(row["exact_bytes_equal"] for row in rows),
        "rows": rows,
    }


def sanitize(report: Path) -> dict[str, Any]:
    forbidden = re.compile(
        r"(?:[A-Za-z]:\\|DASHSCOPE|api[_-]?key|authorization\s*:|bearer\s+|"
        r"raw_response|question_text|gold_answer|model_answer|\.sqlite|checkpoint)",
        re.IGNORECASE,
    )
    findings: list[dict[str, Any]] = []
    for path in sorted(report.iterdir()):
        if not path.is_file() or path.name == "sanitization_manifest.json":
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                findings.append({"file": path.name, "line": line_no})
    return {
        "status": "PASS" if not findings else "FAIL",
        "files_scanned": len([path for path in report.iterdir() if path.is_file()]),
        "finding_count": len(findings),
        "findings": findings,
    }


def run(root: Path, report: Path, prep: Path, seed: int = 76) -> dict[str, Any]:
    root = root.resolve()
    report = report.resolve()
    prep = prep.resolve()
    if root not in report.parents or root not in prep.parents:
        raise ValueError("all generated artifacts must stay inside the Diversity repository")
    if report.exists() and any(report.iterdir()):
        raise FileExistsError(f"report directory is not fresh: {report}")
    if prep.exists() and any(prep.iterdir()):
        raise FileExistsError(f"prep directory is not fresh: {prep}")
    report.mkdir(parents=True, exist_ok=True)
    prep.mkdir(parents=True, exist_ok=True)

    cases, states, source_files = load_inputs(root, seed=seed)
    before = {name: sha256_file(path) for name, path in source_files.items()}
    p0 = states[0]["ordered_prompts"][0]
    smoke = serialization_smoke(cases, p0)
    private_registry = {
        "contract_id": COMMON_SOLVER_CONTRACT_ID,
        "contract_identity": contract_identity(),
        "split": "ExternalValidation50",
        "test50_accessed": False,
        "selection_or_optimization": False,
        "cases": [
            {
                "position": row["position"],
                "case_id": row["case_id"],
                "question": row["question"],
                "gold": row["gold"],
            }
            for row in cases
        ],
        "states": states,
    }
    write_json(prep / "private_replay_registry.json", private_registry)

    contract_module = root / "infrastructure/common_solver_contract_v1/contract.py"
    evaluator_module = root / "infrastructure/common_solver_contract_v1/evaluator.py"
    entrypoints_module = root / "infrastructure/common_solver_contract_v1/entrypoints.py"
    package_module = root / "infrastructure/common_solver_contract_v1/__init__.py"
    runner_module = root / "scripts/run_common_solver_contract_replay.py"
    manifest = dict(public_contract_manifest())
    manifest["source_sha256"] = {
        "contract.py": sha256_file(contract_module),
        "entrypoints.py": sha256_file(entrypoints_module),
        "evaluator.py": sha256_file(evaluator_module),
        "__init__.py": sha256_file(package_module),
        "run_common_solver_contract_replay.py": sha256_file(runner_module),
    }
    manifest["freeze_status"] = "SOURCE_FROZEN_ZERO_API"
    write_json(report / "contract_manifest.json", manifest)
    write_json(report / "serialization_parity_smoke.json", smoke)

    public_states = [
        {
            key: value
            for key, value in state.items()
            if key != "ordered_prompts"
        }
        for state in states
    ]
    public_registry = {
        "status": "EXECUTION_READY_PENDING_EXPLICIT_API_AUTHORIZATION",
        "selection_rule": f"SEED{seed}_REQUESTED_FOR_FROZEN_ARTIFACT_COMMON_CONTRACT_REPLAY",
        "split": "ExternalValidation50",
        "case_count": 50,
        "case_order_sha256": canonical_hash([row["case_id"] for row in cases]),
        "states": public_states,
        "planned_unique_prompt_case_requests": sum(
            state["unique_prompt_count"] for state in states
        )
        * 50,
        "optimization_rerun": False,
        "test50_accessed": False,
        "dropped_or_missing_state_note": None,
    }
    write_json(report / "replay_registry_public.json", public_registry)

    after = {name: sha256_file(path) for name, path in source_files.items()}
    facts = {
        "gate": "PASS",
        "api_calls": 0,
        "test50_accessed": False,
        "optimization_rerun": False,
        "historical_source_count": len(source_files),
        "historical_sources_unchanged": before == after,
        "serialization_entrypoint_count": 3,
        "serialization_smoke_case_count": 20,
        "serialization_exact_bytes_equal": smoke["all_serialized_request_bytes_equal"],
        "replay_state_count": len(states),
        "replay_case_count": len(cases),
        "contract_identity": contract_identity(),
        "execution_status": "NOT_RUN_REQUIRES_API_AUTHORIZATION",
    }
    if not all(
        [
            facts["historical_sources_unchanged"],
            facts["serialization_exact_bytes_equal"],
            facts["api_calls"] == 0,
            not facts["test50_accessed"],
        ]
    ):
        facts["gate"] = "FAIL"
    write_json(report / "fact_assertions.json", facts)
    provenance = {
        "report_version": f"common_solver_contract_v1_seed{seed}_prep_v1",
        "repository_commits": {
            "Diversity": git(root, "rev-parse", "HEAD"),
            "GEPA": git(root.parent / "independent_gepa_repro", "rev-parse", "HEAD"),
            "MARS": git(root.parent / "MARS", "rev-parse", "HEAD"),
        },
        "historical_input_sha256": before,
        "generated_artifact_scope": [
            report.relative_to(root).as_posix(),
            prep.relative_to(root).as_posix(),
        ],
        "private_registry_tracked": False,
    }
    write_json(report / "provenance.json", provenance)
    readme = """# COMMON_SOLVER_CONTRACT_V1 preparation

This zero-API preparation freezes one shared serialization, retry, parsing,
cache-identity, and accounting contract for cross-repository frozen-artifact
evaluation. Diversity, GEPA, and MARS boundary adapters produced identical
canonical request bytes for the first 20 frozen ExternalValidation50 cases.

No optimization was rerun. No model API was called. Test50 remained locked.

This replay includes P0 plus the requested seed's frozen MARS, GEPA, and
five-member Diversity final states. Eligibility is based only on exact frozen
artifact completeness, before common-contract results exist.

The actual ExternalValidation50 replay is not executed by this preparation. It
requires separate explicit API authorization and a fresh project-local output
root. Raw accuracy from the earlier repository-specific evaluators remains
descriptive only until this common replay is complete.
"""
    (report / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    sanitization = sanitize(report)
    write_json(report / "sanitization_manifest.json", sanitization)
    if facts["gate"] != "PASS" or sanitization["status"] != "PASS":
        raise AssertionError("preparation gate failed")
    sha_manifest = {
        path.name: sha256_file(path)
        for path in sorted(report.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    write_json(report / "sha256_manifest.json", sha_manifest)
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--seed", type=int, default=76)
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports/common_solver_contract_v1_prep_20260906",
    )
    parser.add_argument(
        "--prep",
        type=Path,
        default=root / "runs/common_solver_contract_v1_prep_20260906",
    )
    args = parser.parse_args()
    result = run(args.root, args.report, args.prep, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
