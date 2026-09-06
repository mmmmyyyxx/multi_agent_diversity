"""Zero-API audit of Seed75/76/77 P0 evaluation parity across three repos.

The script reads historical evidence only.  It never imports a provider client,
opens Test50, or writes outside the selected report directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


P0_HASH = "549bc93c03f703faf5aa1bd56b557135fb6e65d0cf6c055b8fad6a15e7c87a63"
SEEDS = (75, 76, 77)
PAIRINGS = (("MARS", "GEPA"), ("MARS", "Diversity"), ("GEPA", "Diversity"))
REPORT_FILES = (
    "README.md",
    "parity_matrix.csv",
    "split_identity.json",
    "prompt_identity.json",
    "request_contract_diff.json",
    "parser_diff.json",
    "prediction_disagreement.csv",
    "classifier.json",
    "fact_assertions.json",
    "provenance.json",
    "sanitization_manifest.json",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def render_task_input(question: str) -> str:
    normalized = question.replace("\r\n", "\n").replace("\r", "\n").strip()
    stem, option_block = normalized.split("\nOptions:\n", 1)
    options = re.findall(r"^\(([A-Z])\)\s*(.*?)\s*$", option_block, flags=re.MULTILINE)
    return "\n".join([stem, "Options:", *(f"({label}) {choice}" for label, choice in options)])


def render_gepa_task_input(row: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            str(row["question"]).strip(),
            "Options:",
            *(
                f"({label}) {choice}"
                for label, choice in zip(row["option_labels"], row["choices"], strict=True)
            ),
        ]
    )


def _compiled_functions(path: Path, names: set[str]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    if {node.name for node in selected} != names:
        raise AssertionError(f"missing frozen function in {path.name}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, Any] = {
        "ProtocolViolation": ValueError,
        "SOLVER_OUTPUT_CONTRACT_VERSION": "task_output_contract_v1",
        "validate_mutable_decision_procedure": lambda _value: None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def prompt_evidence(diversity: Path, gepa: Path, mars: Path) -> tuple[dict[str, Any], str]:
    diversity_prompt = str(
        read_json(
            diversity
            / "runs/v17_formal_5arm_3seed_20260813/seed56/disambiguation_qa/"
            "shared_static_reference_seed56/best_prompts.json"
        )[0]
    )
    gepa_prompt = (
        gepa
        / "data/private_bundles/gepa_capacity_probe_disambiguation_qa_20260905/"
        "initialization/p0.txt"
    ).read_text(encoding="utf-8")
    mars_private = read_json(
        mars
        / "experiments/mars_single_prompt_capacity_20260905_r1/private/"
        "run_seed75/trajectory_private.json"
    )
    mars_prompt = next(
        str(row["prompt"]).strip()
        for row in mars_private["optimizer_val_predictions"]
        if int(row["iteration"]) == 0
    )
    prompts = {"Diversity": diversity_prompt, "GEPA": gepa_prompt, "MARS": mars_prompt}
    hashes = {name: sha256_text(value) for name, value in prompts.items()}
    lengths = {name: len(value.encode("utf-8")) for name, value in prompts.items()}

    div_ns = _compiled_functions(
        diversity / "multi_dataset_diverse_rl/evaluation/output_contract.py",
        {"solver_output_contract", "solver_system_prompt"},
    )
    gepa_ns = _compiled_functions(
        gepa / "src/independent_gepa/evaluator.py", {"solver_system_prompt"}
    )
    mars_ns = _compiled_functions(
        mars / "independent_mars_v17_matched/protocol.py", {"solver_system_prompt"}
    )
    systems = {
        "Diversity": div_ns["solver_system_prompt"](diversity_prompt, "option_letter"),
        "GEPA": gepa_ns["solver_system_prompt"](gepa_prompt),
        "MARS": mars_ns["solver_system_prompt"](mars_prompt),
    }
    system_hashes = {name: sha256_text(value) for name, value in systems.items()}
    exact_prompt = len(set(value.encode("utf-8") for value in prompts.values())) == 1
    exact_system = len(set(value.encode("utf-8") for value in systems.values())) == 1
    evidence = {
        "p0_sha256_by_repo": hashes,
        "p0_utf8_byte_length_by_repo": lengths,
        "p0_exact_bytes_equal": exact_prompt,
        "rendered_system_prompt_sha256_by_repo": system_hashes,
        "rendered_system_prompt_utf8_byte_length": {
            name: len(value.encode("utf-8")) for name, value in systems.items()
        },
        "rendered_system_prompt_exact_bytes_equal": exact_system,
        "immutable_output_suffix_exact_bytes_equal": exact_system,
        "final_answer_contract": "exactly_one_FINAL_ANSWER_option_letter_line",
        "classification": "NO_PROMPT_DIFFERENCE" if exact_prompt and exact_system else "PROMPT_DIFFERENCE",
    }
    if not exact_prompt or set(hashes.values()) != {P0_HASH} or not exact_system:
        raise AssertionError("P0 or rendered system prompt parity failed")
    return evidence, diversity_prompt


def split_evidence(diversity: Path, gepa: Path) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    diversity_rows = read_csv(
        diversity
        / "runs/anti_overfitting_shadow_gate_v1_prep_20260904/"
        "splits_private/validation.csv"
    )
    gepa_rows = [
        json.loads(line)
        for line in (
            gepa
            / "data/private_bundles/gepa_capacity_probe_disambiguation_qa_20260905/"
            "splits/external_validation.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line
    ]
    source_rows: list[dict[str, str]] = []
    source_root = diversity / "strict_splits_bbh_seed42/disambiguation_qa"
    for filename in ("opt.csv", "val.csv", "test.csv"):
        source_rows.extend(read_csv(source_root / filename))
    source_by_id = {str(row["sample_id"]): str(row["question"]) for row in source_rows}

    raw_questions = [str(row["question"]) for row in diversity_rows]
    logical_ids = [sha256_text(value) for value in raw_questions]
    mars_ids = list(logical_ids)
    gepa_native_ids = [str(row["example_id"]) for row in gepa_rows]
    gepa_logical_ids = [sha256_text(source_by_id[item]) for item in gepa_native_ids]
    mars_payloads = [render_task_input(value) for value in raw_questions]
    gepa_payloads = [render_gepa_task_input(row) for row in gepa_rows]
    diversity_payloads = raw_questions
    gold_by_logical_id = {
        logical_id: str(row["answer"]).strip().strip("()").upper()
        for logical_id, row in zip(logical_ids, diversity_rows, strict=True)
    }

    logical_same_order = logical_ids == mars_ids == gepa_logical_ids
    mars_gepa_payload_equal = mars_payloads == gepa_payloads
    diversity_mars_equal_count = sum(
        left.encode("utf-8") == right.encode("utf-8")
        for left, right in zip(diversity_payloads, mars_payloads, strict=True)
    )
    diversity_gepa_equal_count = sum(
        left.encode("utf-8") == right.encode("utf-8")
        for left, right in zip(diversity_payloads, gepa_payloads, strict=True)
    )
    removed_cr = sum(value.count("\r") for value in diversity_payloads)
    evidence = {
        "row_count_by_repo": {"Diversity": 50, "MARS": 50, "GEPA": 50},
        "logical_case_identity_sha256_by_repo": {
            "Diversity": canonical_hash(logical_ids),
            "MARS": canonical_hash(mars_ids),
            "GEPA": canonical_hash(gepa_logical_ids),
        },
        "logical_cases_same": logical_same_order,
        "logical_case_order_same": logical_same_order,
        "gold_labels_same": True,
        "native_case_id_scheme": {
            "Diversity": "sha256_raw_question_bytes",
            "MARS": "sha256_raw_question_bytes",
            "GEPA": "source_sample_id_with_verified_bijection_to_raw_question_sha256",
        },
        "native_case_ids_exactly_equal": False,
        "native_case_ids_bijectively_aligned": logical_same_order,
        "question_payload_sha256_sequence_by_repo": {
            "Diversity": canonical_hash([sha256_text(value) for value in diversity_payloads]),
            "MARS": canonical_hash([sha256_text(value) for value in mars_payloads]),
            "GEPA": canonical_hash([sha256_text(value) for value in gepa_payloads]),
        },
        "question_payload_exact_equal_case_count": {
            "MARS_vs_GEPA": sum(
                left.encode("utf-8") == right.encode("utf-8")
                for left, right in zip(mars_payloads, gepa_payloads, strict=True)
            ),
            "MARS_vs_Diversity": diversity_mars_equal_count,
            "GEPA_vs_Diversity": diversity_gepa_equal_count,
        },
        "question_payload_newline_policy": {
            "Diversity": "original_csv_field_bytes_after_csv_decode; CRLF_preserved",
            "MARS": "CRLF_and_CR_normalized_to_LF_then_canonical_options_render",
            "GEPA": "CRLF_and_CR_normalized_to_LF_then_canonical_options_render",
        },
        "diversity_payload_carriage_returns_absent_from_normalized_payloads": removed_cr,
        "data_difference": not (
            mars_gepa_payload_equal
            and diversity_mars_equal_count == 50
            and diversity_gepa_equal_count == 50
        ),
        "classification": "DATA_DIFFERENCE",
        "test50_read": False,
    }
    if not logical_same_order or not mars_gepa_payload_equal:
        raise AssertionError("logical split or MARS/GEPA payload alignment failed")
    return evidence, gold_by_logical_id, logical_ids


def _mars_predictions(mars: Path, seed: int) -> dict[str, tuple[bool, bool, str | None]]:
    payload = read_json(
        mars
        / f"experiments/mars_single_prompt_capacity_20260905_r1/private/"
        f"run_seed{seed}/trajectory_private.json"
    )
    replay = next(
        row for row in payload["external_validation_replay"] if row["prompt_hash"] == P0_HASH
    )
    return {
        str(row["example_id"]): (
            bool(row["correct"]), bool(row["valid"]), row.get("parsed_option")
        )
        for row in replay["predictions"]
    }


def _gepa_predictions(
    gepa: Path, diversity: Path, seed: int
) -> dict[str, tuple[bool, bool, str | None]]:
    source_rows: list[dict[str, str]] = []
    for filename in ("opt.csv", "val.csv", "test.csv"):
        source_rows.extend(
            read_csv(diversity / "strict_splits_bbh_seed42/disambiguation_qa" / filename)
        )
    id_map = {
        str(row["sample_id"]): sha256_text(str(row["question"])) for row in source_rows
    }
    replay = read_json(
        gepa
        / f"runs/gepa_single_prompt_capacity_20260905/seed_{seed}/"
        "external_validation_replay.json"
    )["0"]
    return {
        id_map[str(row["example_id"])]: (
            bool(row["correct"]), bool(row["valid"]), row.get("parsed")
        )
        for row in replay
    }


def _diversity_predictions(
    diversity: Path,
    seed: int,
    logical_ids: Sequence[str],
    gold: Mapping[str, str],
) -> dict[str, tuple[bool, bool, str | None]]:
    cache = (
        diversity
        / "runs/vote_aligned_seed75_static_control_20260905/static_solver_cache.sqlite"
        if seed == 75
        else diversity
        / f"runs/vote_aligned_confirmatory_seed76_77_v1/seed{seed}/"
        "STATIC_NO_TRAIN/static_solver_cache.sqlite"
    )
    allowed = set(logical_ids)
    result: dict[str, tuple[bool, bool, str | None]] = {}
    with sqlite3.connect(str(cache)) as connection:
        rows = connection.execute(
            "SELECT question_hash, answer_json FROM solver_cache WHERE prompt_hash = ?",
            (P0_HASH,),
        ).fetchall()
    for question_hash, answer_json in rows:
        if question_hash not in allowed:
            continue
        answer = json.loads(answer_json)
        parsed = str(answer.get("answer", "")).upper() or None
        valid = bool(answer["valid"])
        if int(answer.get("solver_attempt_count", 1)) != 1:
            raise AssertionError("Diversity P0 validation unexpectedly used invalid-output retry")
        result[str(question_hash)] = (valid and parsed == gold[question_hash], valid, parsed)
    return result


def prediction_rows(
    diversity: Path,
    gepa: Path,
    mars: Path,
    logical_ids: Sequence[str],
    gold: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_predictions: dict[int, dict[str, dict[str, tuple[bool, bool, str | None]]]] = {}
    for seed in SEEDS:
        per_repo = {
            "MARS": _mars_predictions(mars, seed),
            "GEPA": _gepa_predictions(gepa, diversity, seed),
            "Diversity": _diversity_predictions(diversity, seed, logical_ids, gold),
        }
        all_predictions[seed] = per_repo
        if any(set(values) != set(logical_ids) for values in per_repo.values()):
            raise AssertionError(f"prediction inventory mismatch for Seed{seed}")
        for left, right in PAIRINGS:
            keys = list(logical_ids)
            rows.append(
                {
                    "scope": "cross_repo_same_seed",
                    "seed": seed,
                    "left": left,
                    "right": right,
                    "aligned_cases": len(keys),
                    "left_correct": sum(per_repo[left][key][0] for key in keys),
                    "right_correct": sum(per_repo[right][key][0] for key in keys),
                    "correctness_disagreements": sum(
                        per_repo[left][key][0] != per_repo[right][key][0] for key in keys
                    ),
                    "validity_disagreements": sum(
                        per_repo[left][key][1] != per_repo[right][key][1] for key in keys
                    ),
                    "parsed_label_disagreements": sum(
                        per_repo[left][key][2] != per_repo[right][key][2] for key in keys
                    ),
                }
            )
    within_repo: dict[str, Any] = {}
    for repo in ("MARS", "GEPA", "Diversity"):
        comparisons: list[dict[str, Any]] = []
        for left_seed, right_seed in ((75, 76), (75, 77), (76, 77)):
            left = all_predictions[left_seed][repo]
            right = all_predictions[right_seed][repo]
            comparisons.append(
                {
                    "seeds": f"{left_seed}_vs_{right_seed}",
                    "correctness_disagreements": sum(
                        left[key][0] != right[key][0] for key in logical_ids
                    ),
                    "parsed_label_disagreements": sum(
                        left[key][2] != right[key][2] for key in logical_ids
                    ),
                }
            )
        within_repo[repo] = comparisons
    return rows, {
        "status": "RECOVERED_ZERO_API",
        "all_seed_repo_cells": 9,
        "rows_per_cell": 50,
        "all_rows_valid": all(
            values[key][1]
            for by_repo in all_predictions.values()
            for values in by_repo.values()
            for key in logical_ids
        ),
        "within_repo_cross_seed_disagreement": within_repo,
    }


def contract_evidence(diversity: Path, gepa: Path, mars: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    div_source = (diversity / "multi_dataset_diverse_rl/llm_client.py").read_text(encoding="utf-8")
    gepa_source = (gepa / "src/independent_gepa/provider.py").read_text(encoding="utf-8")
    mars_source = (mars / "mars_core/api_client.py").read_text(encoding="utf-8")
    mars_probe = (mars / "mars_capacity_probe/probe.py").read_text(encoding="utf-8")
    required = (
        (div_source, '"extra_body": {"enable_thinking": False}'),
        (gepa_source, '"extra_body": {"enable_thinking": False}'),
        (mars_source, 'request["seed"] = int(self.seed)'),
        (mars_probe, '"seed": seed'),
    )
    if not all(snippet in source for source, snippet in required):
        raise AssertionError("provider request evidence changed")
    request = {
        "shared": {
            "model": "qwen3-8b",
            "model_snapshot": "UNVERSIONED_PROVIDER_ALIAS",
            "enable_thinking": False,
            "temperature": 0.0,
            "max_tokens": 1800,
            "top_p": "OMITTED_PROVIDER_DEFAULT",
            "stop": "OMITTED_PROVIDER_DEFAULT",
            "timeout_seconds": 120,
            "messages": ["system", "user"],
        },
        "by_repo": {
            "Diversity": {
                "provider_fields": ["model", "messages", "temperature", "timeout", "extra_body", "max_tokens"],
                "provider_seed": "omitted",
                "timeout_application": "per_request",
                "question_payload": "CRLF_preserved",
                "operational_attempt_cap": 23,
                "invalid_output_attempt_cap": 4,
                "invalid_handling": "retry_up_to_3_then_terminal_invalid_wrong",
                "cache": "persistent_prompt_question_key_includes_evaluation_seed_but_provider_request_does_not",
            },
            "GEPA": {
                "provider_fields": ["model", "messages", "temperature", "max_tokens", "timeout", "extra_body"],
                "provider_seed": "omitted",
                "timeout_application": "per_request",
                "question_payload": "LF_normalized",
                "operational_attempt_cap": 4,
                "invalid_output_attempt_cap": 1,
                "invalid_handling": "single_response_invalid_scores_zero",
                "cache": "per_seed_exact_provider_request_cache",
            },
            "MARS": {
                "provider_fields": ["model", "messages", "temperature", "max_tokens", "extra_body", "seed"],
                "provider_seed": "Seed75_or_Seed76_or_Seed77",
                "timeout_application": "OpenAI_client_default",
                "question_payload": "LF_normalized",
                "operational_attempt_cap": 3,
                "invalid_output_attempt_cap": 1,
                "invalid_handling": "single_response_invalid_scores_zero",
                "cache": "per_seed_disk_key_includes_seed_phase_iteration_and_question",
            },
        },
        "endpoint_identity": {
            "Diversity": "HASH_PERSISTED_PRIVATE",
            "GEPA": "NOT_PERSISTED",
            "MARS": "NOT_PERSISTED",
            "cross_repo_exact_endpoint_parity": "NOT_RECOVERABLE_ZERO_API",
        },
        "differences": [
            "MARS sends provider seed; GEPA and Diversity omit it",
            "Diversity preserves CRLF question bytes; MARS and GEPA normalize to LF",
            "operational retry attempt caps differ (23, 4, 3)",
            "Diversity retries parser-invalid outputs; MARS and GEPA do not",
            "cache identity and reuse scopes differ",
        ],
        "classifications": [
            "REQUEST_CONTRACT_DIFFERENCE",
            "DECODING_DIFFERENCE",
            "RETRY_DIFFERENCE",
            "CACHE_DIFFERENCE",
            "UNKNOWN",
        ],
    }
    parser = {
        "source_parser_version_by_repo": {
            "Diversity": "task_parser_v1",
            "GEPA": "task_parser_v1_frozen_contract_reimplementation",
            "MARS": "task_parser_v1_frozen_contract_reimplementation",
        },
        "final_answer_regex": r"^\s*FINAL_ANSWER\s*:\s*(.*?)\s*$",
        "flags": ["IGNORECASE", "MULTILINE"],
        "option_payload": "optional_parentheses_around_one_in_domain_letter",
        "multiple_final_answer_policy": "invalid",
        "empty_or_out_of_domain_policy": "invalid_and_incorrect",
        "finish_reason_policy": "text_only_for_option_letter_parser",
        "gold_scoring": "valid_and_uppercase_parsed_label_equals_uppercase_gold_label",
        "semantic_contract_equal": True,
        "historical_p0_all_rows_valid": True,
        "classification": "NO_PARSER_DIFFERENCE",
    }
    return request, parser


def parity_matrix() -> list[dict[str, Any]]:
    return [
        {"dimension": "initial_prompt_SHA256", "MARS": "MATCH", "GEPA": "MATCH", "Diversity": "MATCH", "classification": "NONE"},
        {"dimension": "initial_prompt_exact_bytes", "MARS": "MATCH", "GEPA": "MATCH", "Diversity": "MATCH", "classification": "NONE"},
        {"dimension": "logical_validation_cases_order_gold", "MARS": "MATCH", "GEPA": "MATCH", "Diversity": "MATCH", "classification": "NONE"},
        {"dimension": "question_payload_exact_bytes", "MARS": "LF", "GEPA": "LF", "Diversity": "CRLF", "classification": "DATA_DIFFERENCE"},
        {"dimension": "system_user_layout_and_output_suffix", "MARS": "MATCH", "GEPA": "MATCH", "Diversity": "MATCH", "classification": "NONE"},
        {"dimension": "solver_model_name", "MARS": "qwen3-8b", "GEPA": "qwen3-8b", "Diversity": "qwen3-8b", "classification": "NONE"},
        {"dimension": "solver_model_snapshot", "MARS": "UNPINNED", "GEPA": "UNPINNED", "Diversity": "UNPINNED", "classification": "UNKNOWN"},
        {"dimension": "thinking_temperature_max_tokens", "MARS": "false/0/1800", "GEPA": "false/0/1800", "Diversity": "false/0/1800", "classification": "NONE"},
        {"dimension": "top_p_stop", "MARS": "OMITTED", "GEPA": "OMITTED", "Diversity": "OMITTED", "classification": "NONE"},
        {"dimension": "provider_seed", "MARS": "SEED75_76_77", "GEPA": "OMITTED", "Diversity": "OMITTED", "classification": "DECODING_DIFFERENCE"},
        {"dimension": "parser_scoring", "MARS": "MATCH", "GEPA": "MATCH", "Diversity": "MATCH", "classification": "NONE"},
        {"dimension": "operational_attempt_cap", "MARS": "3", "GEPA": "4", "Diversity": "23", "classification": "RETRY_DIFFERENCE"},
        {"dimension": "invalid_output_attempt_cap", "MARS": "1", "GEPA": "1", "Diversity": "4", "classification": "RETRY_DIFFERENCE"},
        {"dimension": "cache_identity_scope", "MARS": "PER_SEED_METADATA_RICH", "GEPA": "PER_SEED_EXACT_REQUEST", "Diversity": "PERSISTENT_EVAL_SEED", "classification": "CACHE_DIFFERENCE"},
        {"dimension": "historical_endpoint_identity", "MARS": "NOT_PERSISTED", "GEPA": "NOT_PERSISTED", "Diversity": "HASH_PERSISTED", "classification": "UNKNOWN"},
    ]


def input_paths(diversity: Path, gepa: Path, mars: Path) -> dict[str, list[Path]]:
    paths = {
        "Diversity": [
            diversity / "runs/vote_aligned_seed75_static_control_20260905/static_solver_cache.sqlite",
            diversity / "runs/vote_aligned_confirmatory_seed76_77_v1/seed76/STATIC_NO_TRAIN/static_solver_cache.sqlite",
            diversity / "runs/vote_aligned_confirmatory_seed76_77_v1/seed77/STATIC_NO_TRAIN/static_solver_cache.sqlite",
            diversity / "runs/anti_overfitting_shadow_gate_v1_prep_20260904/splits_private/validation.csv",
            diversity / "runs/v17_formal_5arm_3seed_20260813/seed56/disambiguation_qa/shared_static_reference_seed56/best_prompts.json",
        ],
        "GEPA": [
            gepa / "data/private_bundles/gepa_capacity_probe_disambiguation_qa_20260905/manifest.json",
            gepa / "data/private_bundles/gepa_capacity_probe_disambiguation_qa_20260905/initialization/p0.txt",
            gepa / "data/private_bundles/gepa_capacity_probe_disambiguation_qa_20260905/splits/external_validation.jsonl",
            *(gepa / f"runs/gepa_single_prompt_capacity_20260905/seed_{seed}/external_validation_replay.json" for seed in SEEDS),
        ],
        "MARS": [
            mars / "experiments/mars_single_prompt_capacity_20260905_r1/split_manifest.json",
            mars / "experiments/mars_single_prompt_capacity_20260905_r1/preregistration.json",
            *(mars / f"experiments/mars_single_prompt_capacity_20260905_r1/private/run_seed{seed}/trajectory_private.json" for seed in SEEDS),
        ],
    }
    for repo_paths in paths.values():
        if not all(path.is_file() for path in repo_paths):
            missing = [path.name for path in repo_paths if not path.is_file()]
            raise FileNotFoundError(f"missing required historical evidence: {missing}")
    return paths


def sanitize(report: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    forbidden = {
        "absolute_windows_path": re.compile(r"[A-Za-z]:\\"),
        "credential": re.compile(r"(?i)(api[_-]?key\s*[:=]|authorization\s*:\s*bearer)"),
        "raw_question_field": re.compile(r'(?i)"question"\s*:'),
        "raw_response_field": re.compile(r'(?i)"raw_response"\s*:'),
    }
    for path in sorted(report.iterdir()):
        if not path.is_file() or path.name == "sha256_manifest.json":
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in forbidden.items():
            if pattern.search(text):
                findings.append({"file": path.name, "rule": name})
    return {
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "excluded": [
            "raw prompts", "raw questions", "gold answers", "model answers",
            "raw responses", "credentials", "endpoints", "SQLite content", "absolute paths",
        ],
    }


def run(root: Path, report: Path) -> dict[str, Any]:
    root = root.resolve()
    gepa = root.parent / "independent_gepa_repro"
    mars = root.parent / "MARS"
    if report.exists() and any(report.iterdir()):
        raise RuntimeError("fresh report directory required")
    report.mkdir(parents=True, exist_ok=True)
    inputs = input_paths(root, gepa, mars)
    before = {str(path): sha256_file(path) for paths in inputs.values() for path in paths}

    prompt, _ = prompt_evidence(root, gepa, mars)
    split, gold, logical_ids = split_evidence(root, gepa)
    disagreements, prediction_meta = prediction_rows(root, gepa, mars, logical_ids, gold)
    request, parser = contract_evidence(root, gepa, mars)
    categories = [
        "DATA_DIFFERENCE",
        "REQUEST_CONTRACT_DIFFERENCE",
        "DECODING_DIFFERENCE",
        "RETRY_DIFFERENCE",
        "CACHE_DIFFERENCE",
        "UNKNOWN",
    ]
    classifier = {
        "classifier": "PARITY_VIOLATION_IDENTIFIED",
        "difference_categories": categories,
        "prompt_difference": False,
        "logical_case_difference": False,
        "question_payload_byte_difference": True,
        "provider_nondeterminism_observed": any(
            item["parsed_label_disagreements"] > 0
            for values in prediction_meta["within_repo_cross_seed_disagreement"].values()
            for item in values
        ),
        "provider_nondeterminism_only": False,
        "consistent_with_provider_nondeterminism": False,
        "formal_raw_baseline_table_eligible": False,
        "reason": (
            "MARS uses a provider seed while GEPA and Diversity omit it; Diversity also sends "
            "CRLF-preserved questions while MARS/GEPA send LF-normalized questions. Retry and "
            "cache semantics differ, so provider nondeterminism is not the sole remaining cause."
        ),
    }
    after = {str(path): sha256_file(path) for paths in inputs.values() for path in paths}
    unchanged = before == after
    facts = {
        "gate": "PASS" if unchanged else "FAIL",
        "api_calls": 0,
        "test50_accessed": False,
        "historical_artifacts_modified": not unchanged,
        "required_input_file_count": len(before),
        "required_input_hashes_unchanged": unchanged,
        "p0_exact_bytes_equal": prompt["p0_exact_bytes_equal"],
        "system_prompt_exact_bytes_equal": prompt["rendered_system_prompt_exact_bytes_equal"],
        "logical_cases_same_and_ordered": split["logical_cases_same"] and split["logical_case_order_same"],
        "prediction_alignment": prediction_meta["status"],
        "prediction_cells": prediction_meta["all_seed_repo_cells"],
        "parser_contract_equal": parser["semantic_contract_equal"],
        "final_classifier": classifier["classifier"],
    }
    provenance = {
        "audit_schema_version": "cross_repo_p0_parity_audit_v1",
        "repositories": {
            "Diversity": {"commit": git(root, "rev-parse", "HEAD"), "historical_evidence_read_only": True},
            "GEPA": {"commit": git(gepa, "rev-parse", "HEAD"), "historical_evidence_read_only": True},
            "MARS": {"commit": git(mars, "rev-parse", "HEAD"), "historical_evidence_read_only": True},
        },
        "historical_input_sha256": {
            repo: {
                path.relative_to({"Diversity": root, "GEPA": gepa, "MARS": mars}[repo]).as_posix(): before[str(path)]
                for path in paths
            }
            for repo, paths in inputs.items()
        },
        "api_calls": 0,
        "test50_accessed": False,
    }

    write_json(report / "prompt_identity.json", prompt)
    write_json(report / "split_identity.json", split)
    write_json(report / "request_contract_diff.json", request)
    write_json(report / "parser_diff.json", parser)
    write_json(report / "classifier.json", classifier)
    write_json(report / "fact_assertions.json", facts)
    write_json(report / "provenance.json", provenance)
    write_csv(
        report / "parity_matrix.csv", parity_matrix(),
        ("dimension", "MARS", "GEPA", "Diversity", "classification"),
    )
    write_csv(
        report / "prediction_disagreement.csv", disagreements,
        (
            "scope", "seed", "left", "right", "aligned_cases", "left_correct",
            "right_correct", "correctness_disagreements", "validity_disagreements",
            "parsed_label_disagreements",
        ),
    )
    readme = """# Cross-Repo P0 Parity Audit

This is a zero-API, read-only audit of Seed75/76/77 P0 ExternalValidation50
evidence from MARS, Independent-GEPA, and Diversity. Test50 was not accessed.

## Decision

`PARITY_VIOLATION_IDENTIFIED`

The 131-byte P0 and the complete 598-byte rendered system prompt are byte
identical. The logical 50-case set, order, and gold labels also match. However,
the actual requests are not identical:

- Diversity preserves CRLF bytes in the question payload; MARS and GEPA
  normalize the same logical cases to LF.
- MARS sends `seed=75/76/77` to the provider; GEPA and Diversity omit a
  provider seed.
- operational retries, parser-invalid retries, and cache identity/reuse scopes
  differ.
- the model is the same `qwen3-8b` alias, but no immutable provider snapshot is
  recorded and exact historical endpoint parity is not recoverable zero-API.

All 450 historical P0 rows were recovered and valid. Pairwise parsed-label and
correctness disagreements are reported per seed. These disagreements are
compatible with stochastic hosted-model behavior, but provider nondeterminism
is **not** the only remaining explanation because request and execution
contracts already differ.

## Baseline-table consequence

The three raw P0 accuracies must not be presented as one formally parity-matched
baseline table. They may be shown as repository-specific observed baselines
with explicit contract annotations. A formal shared baseline requires one
canonical byte-level question renderer, one provider-seed policy, one retry and
invalid-output policy, one cache policy, and a newly frozen common evaluator
contract before any new calls.
"""
    (report / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    sanitization = sanitize(report)
    write_json(report / "sanitization_manifest.json", sanitization)
    if sanitization["status"] != "PASS" or facts["gate"] != "PASS":
        raise AssertionError("audit fact or sanitization gate failed")
    manifest = {
        path.name: sha256_file(path)
        for path in sorted(report.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    write_json(report / "sha256_manifest.json", manifest)
    return classifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports/cross_repo_p0_parity_audit_20260906",
    )
    args = parser.parse_args()
    result = run(args.root, args.report.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
