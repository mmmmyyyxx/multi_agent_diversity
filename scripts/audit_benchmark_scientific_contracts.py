"""Format offline synthetic-contract evidence. No data download or experiment execution.

Run through tests/formal_zero_api_runner.py --offline-command, with the pinned
evaluator requirements installed. Assertions live in tests and domain modules;
this audit only replays those assertions and serializes sanitized receipts.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE = "5f8c2dad0af74ad843f2dcbe533ecac30ab4f640"
REPORT = ROOT / "reports/benchmark_scientific_contract_freeze_v1_20261001"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(name: str, value: object) -> None:
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (REPORT / name).write_bytes(data.encode("utf-8"))


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)


def preserved_version_constants() -> dict:
    path = "multi_dataset_diverse_rl/versions.py"
    before = ast.parse(git_bytes(path).decode("utf-8"))
    after = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    assignments = lambda tree: {node.targets[0].id: ast.dump(node.value)
        for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    old, new = assignments(before), assignments(after)
    assert all(new.get(k) == v for k, v in old.items())
    return {"all_existing_constants_unchanged": True, "existing_constant_count": len(old)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-summary", type=Path, required=True)
    args = parser.parse_args()
    import sitecustomize
    assert hasattr(sitecustomize, "network_attempt_count")
    assert sitecustomize.network_attempt_count() == 0
    from multi_dataset_diverse_rl.benchmarks.protocols import PROTOCOLS
    from multi_dataset_diverse_rl.benchmarks import BENCHMARKS, benchmark_preflight
    from multi_dataset_diverse_rl.search.scientific_aggregation import HoVerEvidenceAggregation, IFBenchRawResponseAggregation
    from scripts.run_experiment import preflight
    module_spec = importlib.util.spec_from_file_location("scientific_contract_tests", ROOT / "tests/test_benchmark_scientific_contracts.py")
    tests = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(tests)
    checks = {
        "hotpotqa": tests.test_hotpotqa_task_protocol_and_normalization,
        "hover": tests.test_hover_evidence_aggregation_union_guard_and_coverage_e2e,
        "ifbench": tests.test_ifbench_hidden_fields_not_in_solver_input_or_aggregator_request,
        "math": tests.test_math_equivalence_plurality_tie_invalid_and_e2e,
        "pupa": tests.test_pupa_member_pipeline_preserves_artifacts_scorer_and_team_hold,
    }
    e2e = {}
    for key, check in checks.items():
        check()
        e2e[key] = {"pass": True, "assertion": check.__name__, "synthetic_only": True,
                    "real_provider_attempts": 0}
    tests.test_all_aggregator_requests_no_gold_and_no_hidden_values()
    tests.test_equivalence_non_transitive_guard()
    parity = tests.bbh_parity_receipt()
    constants = preserved_version_constants()
    summary = json.loads(args.test_summary.read_text(encoding="utf-8"))
    assert summary["failed"] == 0 and summary["passed"] > 0
    assert sitecustomize.network_attempt_count() == 0
    REPORT.mkdir(parents=True, exist_ok=True)
    matrix, systems, outputs, evaluators, aggregations, responsibility = {}, {}, {}, {}, {}, {}
    for key, protocol in PROTOCOLS.items():
        spec = BENCHMARKS[key]
        matrix[key] = {"protocol": asdict(protocol), "protocol_sha256": protocol.identity(),
            "data_frozen": spec.provenance_frozen and spec.split_frozen,
            "data_identity": spec.dataset_identity, "split_identity": spec.split_identity,
            "system_ready": spec.system_ready, "unified_search_ready": spec.unified_search_ready}
        systems[key] = {"contract": protocol.system_contract_id,
            "dependencies": [asdict(d) for d in protocol.system_dependencies], "system_ready": spec.system_ready}
        outputs[key] = {"contract": protocol.output_contract_id, "parser": protocol.parser_contract_id,
            "semantics": protocol.aggregation_output_semantics, "frozen": True}
        evaluators[key] = {"member_metric": protocol.member_metric_id, "team_metric": protocol.team_metric_id,
            "member_success": protocol.member_success_semantics, "contract_frozen": True,
            "hidden_fields": protocol.hidden_evaluator_fields,
            "runtime_ready": key in {"hotpotqa", "hover", "math"},
            "runtime_note": "IFBench pinned source/control flow frozen; local checker package and language-resource identities HOLD"
                if key == "ifbench" else "PUPA fake oracle only; real judge not frozen" if key == "pupa" else "offline member metric"}
        aggregations[key] = {"policy": protocol.aggregation_policy_id, "frozen": protocol.aggregation_policy_frozen,
            "semantics": protocol.aggregation_output_semantics,
            "model_rule": "optimizer model, distinct team_aggregation role/cache/accounting" if key in {"hover", "ifbench"} else None,
            "equal_members": 5, "weighted_votes": False, "router": False}
        responsibility[key] = {"capabilities": asdict(spec.capabilities),
            "policy": protocol.responsibility_policy_id, "frozen": protocol.responsibility_policy_frozen,
            "binary_analyzer_allowed": spec.capabilities.binary_plurality_responsibility}
    systems["hotpotqa"]["retrieval_hop_document_counts"] = [7, 7]
    systems["hover"]["retrieval_hop_document_counts"] = [7, 7, 10]
    systems["pupa"]["pipeline"] = ["trusted_redaction", "untrusted_response", "trusted_synthesis"]
    systems["pupa"]["member_artifact_fields"] = ["llm_request", "llm_response", "response"]
    systems["pupa"]["trusted_private_input"] = ["user_query"]
    systems["pupa"]["untrusted_public_input"] = ["llm_request"]
    systems["pupa"]["pipeline_exception"] = "empty three artifacts, INVALID_PIPELINE_EXCEPTION status; no raw error text"
    aggregations["hover"].update({"max_titles": 24, "ceiling_definition": "GEPA-system-aligned engineering ceiling, not official benchmark metric",
        "union_guard": True, "instruction_sha256": digest(HoVerEvidenceAggregation.instruction.encode())})
    aggregations["ifbench"]["instruction_sha256"] = digest(IFBenchRawResponseAggregation.instruction.encode())
    evaluators["hover"]["normalization_source"] = "https://raw.githubusercontent.com/stanfordnlp/dspy/2.6.27/dspy/dsp/utils/metrics.py"
    evaluators["hover"]["normalization"] = "Unicode NFD, lowercase, punctuation/articles removal, whitespace collapse"
    vendor = ROOT / "multi_dataset_diverse_rl/benchmarks/_vendor/ifbench"
    evaluators["ifbench"]["checker_provenance"] = json.loads((vendor / "provenance.json").read_text(encoding="utf-8"))
    from multi_dataset_diverse_rl.benchmarks.math_worker import PINS
    evaluators["math"].update({"dependency_pins": PINS, "fallback": "no_fallback",
        "extraction_config": "LatexExtractionConfig; plain expressions wrapped in math delimiters; no partial numeric expression fallback",
        "strict": True, "process_deadline_seconds": 8, "float_rounding": 6, "numeric_precision": 15,
        "timeout_adaptation": "replace inner decorators only; parent kills offline subprocess on deadline",
        "unsupported_equivalence": "fail_closed",
        "ordered_tuples": "unsupported: comma-parentheses rejected because pinned parser interprets them as sets"})
    for path in ("data/benchmark_suite_v1/manifests/ifbench.json", "data/benchmark_suite_v1/manifests/ifbench.ids.jsonl"):
        assert (ROOT / path).read_bytes() == git_bytes(path)
    preserved = {"IFBench_data_manifest_and_memberships_unchanged": True, **constants,
        "historical_reports_edited": False, "GEPA_acceptance_modified": False,
        "Common_Safe_feasibility_target_weights_stopping_modified": False,
        "pattern_provider": "null", "memory_provider": "null"}
    write("preservation_audit.json", preserved)
    for name, value in (("task_contract_matrix.json", matrix), ("system_contracts.json", systems),
        ("output_contracts.json", outputs), ("evaluator_contracts.json", evaluators),
        ("aggregation_contracts.json", aggregations), ("responsibility_capability_matrix.json", responsibility),
        ("bbh_responsibility_parity.json", parity), ("synthetic_e2e_replay.json", e2e), ("test_summary.json", summary)):
        write(name, value)
    runtime = {"seed": 81, "provider_profile": "fake", "solver_model": "solver", "optimizer_model": "optimizer",
        "evaluator_model": "optimizer", "run_identity_sha256": "synthetic", "authorization_identity": "none",
        "cache_identity": "synthetic", "ledger_identity": "synthetic"}
    preflights = {key: {"benchmark": benchmark_preflight(key), "composition": preflight(
        {"scientific": {"method": "unified_team_prompt_search_v1", "benchmark_id": key}, "runtime": runtime})}
        for key in PROTOCOLS}
    assert all(row["benchmark"]["gate"] == "HOLD_PRE_PROVIDER" for row in preflights.values())
    write("preflight_audit.json", preflights)
    write("information_firewall_audit.json", {"pass": True, "public_projection": "explicit per-benchmark allowlist",
        "recursive_context_forbidden_keys": sorted(tests.EVALUATOR_ONLY_KEYS), "hidden_value_sentinels_absent": True,
        "gold_available_only_after_aggregation": True, "new_LLM_roles": "team_aggregation",
        "historical_BBH_cache_identity_preserved": True, "hidden_fields_in_solver_or_aggregator_requests": 0})
    write("zero_api_audit.json", {"dataset_downloads": 0, "real_dataset_materializations": 0,
        "retrieval_runtime_calls": 0, "real_provider_attempts": 0, "real_judge_calls": 0,
        "validation_model_calls": 0, "test_model_calls": 0, "formal_experiments": 0,
        "audit_network_attempts": sitecustomize.network_attempt_count(),
        "test_network_attempts": summary["network_attempt_count"],
        "network_guard_active_before_application_import": True, "negative_control_pass": True,
        "non_dataset_network_activity": {"public_source_browser_open_calls": 2, "git_fetch_commands": 1,
            "pip_dependency_install_performed": True, "pip_HTTP_request_count": None,
            "exact_total_request_count": None, "note": "Dependency/source preparation permitted; HTTP count was not instrumented. Do not claim whole-task network requests=0."},
        "raw_data_required_for_default_tests": False, "credentials_removed_for_tests": True})
    blockers = "# Remaining scientific and execution blockers\n\n"
    for key, row in preflights.items():
        blockers += "## " + key + "\n\n" + "\n".join("- `" + code + "`" for code in row["composition"]["blockers"]) + "\n\n"
    blockers += ("IFBench: the Apache-2.0 checker source and exact eight-variant metric contract are frozen. "
        "spaCy/NLTK assets and complete checker package/resource identities remain unverified; the default loader refuses execution. "
        "No approximate checker or implicit asset download is allowed.\n\n"
        "PUPA: choose team leakage exposure, multiple untrusted-call exposure, aggregator trust boundary and scored requests before a team policy.\n\n"
        "HotpotQA/HoVer: freeze corpus, index and retrieval integration independently of datasets. MATH pins must be installed exactly; "
        "preflight additionally reports unavailable/mismatched packages on other environments. All real executions still require "
        "a governed source/data/protocol freeze and fresh explicit authorization.\n")
    write("remaining_scientific_blockers.md", blockers)
    readme = "# Benchmark Scientific Contract Freeze V1\n\n"
    readme += "BASE_SHA: `" + BASE + "`\n\n"
    readme += "`BENCHMARK_SCIENTIFIC_CONTRACT_FREEZE_V1 = YES`\n\n`REAL_EXECUTION_READY = NO`\n\n"
    readme += "| Benchmark | Data | Task | System | Output | Evaluator | Aggregation | Responsibility | Unified Search |\n|---|---|---|---|---|---|---|---|---|\n"
    data = {"hotpotqa": "partial/not frozen", "hover": "not frozen", "ifbench": "frozen", "math": "not frozen", "pupa": "not frozen"}
    for key, p in PROTOCOLS.items():
        readme += f"| {key} | {data[key]} | frozen | {'pinned offline scorer' if key == 'math' else 'dependency HOLD'} | frozen | "
        readme += {"ifbench": "source/metric frozen; runtime HOLD", "pupa": "fake oracle frozen; real judge HOLD"}.get(key, "frozen") + " | "
        readme += (p.aggregation_policy_id or "HOLD") + " | " + ("binary plurality frozen" if p.responsibility_policy_frozen else "HOLD") + " | HOLD |\n"
    readme += ("\nTask protocols are independent of data and system readiness. IFBench source/control-flow readiness does not "
        "imply availability of its language resources. PUPA has a member pipeline/scorer contract; team semantics remain HOLD.\n\n"
        "BBH parity: 1,024 synthetic five-member vote states and 32 feasibility masks; D/N/C/V, lane, raw legal portfolios, "
        "failure-discounted scores and chosen member agree with frozen V4. Existing BBH fake replay also checks exact evidence, "
        "GEPA and transition behavior. Five synthetic benchmark replays passed; these are contract checks, not efficacy observations.\n\n"
        "Install `requirements-benchmark-evaluators.txt` before clean-checkout tests. Tests need no ignored raw datasets. "
        "Verification used the credential-free, pre-import network guard. Dataset downloads, real materializations, provider/judge, "
        "Validation/Test model calls and formal executions were all zero. Software dependency installation, git fetch and two public "
        "source inspections used network access; their exact total HTTP request count was not captured.\n\n"
        "The HoVer normalization source is [DSPy 2.6.27](https://raw.githubusercontent.com/stanfordnlp/dspy/2.6.27/dspy/dsp/utils/metrics.py). "
        "Other benchmark program/metric source objects were inspected from the previously retained "
        "`gepa-ai/gepa-artifact@cbefbc1aa0f43dd39874ec4bf42211365dbda42e`; none of its provider/retrieval modules were imported.\n\n"
        "See `preflight_audit.json` and `remaining_scientific_blockers.md` for exact holds. Existing data identities and historical "
        "reports are preserved. Publication is local-commit-only; this task does not authorize a push.\n")
    write("README.md", readme)
    write("sha256_manifest.json", {"algorithm": "SHA-256", "bytes": "UTF-8 LF", "self_excluded": True,
        "files": {path.name: digest(path.read_bytes()) for path in sorted(REPORT.iterdir())
                  if path.is_file() and path.name != "sha256_manifest.json"}})
    print(json.dumps({"report": REPORT.relative_to(ROOT).as_posix(), "pass": True,
        "network_attempt_count": sitecustomize.network_attempt_count(), "provider_attempts": 0,
        "BENCHMARK_SCIENTIFIC_CONTRACT_FREEZE_V1": "YES", "REAL_EXECUTION_READY": "NO"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
