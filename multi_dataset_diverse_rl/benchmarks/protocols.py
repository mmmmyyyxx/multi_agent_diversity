"""Scientific contracts, independent of data provenance and split readiness."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Mapping

from .. import versions
from ..search.benchmark import BenchmarkInput
from ..search.schemas import SearchContractError


@dataclass(frozen=True)
class SystemDependency:
    identity: str
    required_components: tuple[str, ...]
    integration_frozen: bool
    blocker: str | None


@dataclass(frozen=True)
class BenchmarkProtocolSpec:
    benchmark_id: str
    task_contract_id: str
    system_contract_id: str
    input_contract_id: str
    output_contract_id: str
    parser_contract_id: str
    member_metric_id: str
    team_metric_id: str
    aggregation_policy_id: str | None
    responsibility_policy_id: str | None
    member_success_semantics: str
    aggregation_output_semantics: str
    hidden_evaluator_fields: tuple[str, ...]
    public_solver_fields: tuple[str, ...]
    system_dependencies: tuple[SystemDependency, ...]
    protocol_frozen: bool
    aggregation_policy_frozen: bool
    responsibility_policy_frozen: bool
    unresolved_scientific_blockers: tuple[str, ...]
    trusted_pipeline_input_fields: tuple[str, ...] = ()

    @property
    def system_ready(self) -> bool:
        return all(dep.integration_frozen for dep in self.system_dependencies)

    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()


_BINARY = versions.BINARY_PLURALITY_RESPONSIBILITY_VERSION
_RETRIEVAL_HOLD = "SYSTEM_DEPENDENCY_RETRIEVAL_NOT_FROZEN"
PROTOCOLS = {
    "hotpotqa": BenchmarkProtocolSpec(
        "hotpotqa", versions.HOTPOT_TASK_CONTRACT, "hotpot_gepa_two_hop_7_7_v1",
        "question_only_v1", "short_final_answer_v1", "hotpot_normalized_answer_v1",
        "normalized_answer_em_v1", "normalized_answer_em_v1", "normalized_equal_plurality_v1",
        _BINARY, "Boolean normalized answer EM; F1 reporting only", "normalized answer class",
        ("answer", "supporting_facts", "gold", "evaluation_feedback"), ("question",),
        (SystemDependency("hotpot_wikipedia_colbert_v1", ("wikipedia_corpus", "retrieval_index",
            "retriever", "two_hop_summary_query_answer_integration"), False, _RETRIEVAL_HOLD),),
        True, True, True, ()),
    "hover": BenchmarkProtocolSpec(
        "hover", versions.HOVER_TASK_CONTRACT, "hover_gepa_three_hop_7_7_10_v1",
        "claim_only_v1", "retrieved_evidence_titles_v1", "hover_normalized_titles_v1",
        "gold_title_subset_coverage_v1", "gold_title_subset_coverage_v1", "llm_evidence_union_24_v1",
        None, "Boolean gold-title subset coverage, not a claim verdict", "retrieved evidence titles",
        ("supporting_facts", "supporting_documents", "gold_titles", "label", "coverage_success"),
        ("claim",), (SystemDependency("hover_wiki_abstracts_2017_bm25_v1",
            ("wiki.abstracts.2017", "BM25_k1_0.9_b_0.4", "english_stemming",
             "three_hop_summary_query_retrieval_integration"), False, _RETRIEVAL_HOLD),),
        True, True, False, ("RESPONSIBILITY_POLICY_NOT_FROZEN",)),
    "ifbench": BenchmarkProtocolSpec(
        "ifbench", versions.IFBENCH_TASK_CONTRACT, "ifbench_gepa_constraint_checkers_v1",
        "prompt_only_v1", "raw_free_form_response_v1", "nonempty_raw_response_v1",
        "gepa_fraction_constraints_eight_variants_v1", "gepa_fraction_constraints_eight_variants_v1",
        "llm_equal_raw_responses_v1", None, "fraction satisfied; per-instruction any of eight variants",
        "raw final response, byte-preserving; no answer marker",
        ("instruction_id_list", "kwargs", "success_vector", "score", "evaluator_feedback"),
        ("prompt",), (SystemDependency("ifbench_pinned_apache_checkers_v1",
            ("pinned_checker_code", "spacy_en_core_web_sm", "nltk_resources", "checker_python_packages"),
            False, "IFBENCH_LOCAL_EVALUATOR_DEPENDENCIES_NOT_FROZEN"),),
        True, True, False, ("RESPONSIBILITY_POLICY_NOT_FROZEN",)),
    "math": BenchmarkProtocolSpec(
        "math", versions.MATH_TASK_CONTRACT, "math_verify_0.6.0_isolated_timeout_v1",
        "problem_only_v1", "math_final_expression_v1", "math_verify_no_fallback_v1",
        "math_verify_0.6.0_equivalence_v1", "math_verify_0.6.0_equivalence_v1",
        versions.EQUIVALENCE_PLURALITY_VERSION, _BINARY,
        "Boolean parsed equivalence; parse/error/timeout fail closed", "actual member answer in largest equivalence class",
        ("solution", "answer", "gold", "evaluation_feedback"), ("problem",),
        (SystemDependency("math_verify_0.6.0_dependency_lock_v1",
            ("math-verify==0.6.0", "latex2sympy2_extended==1.0.9", "sympy==1.14.0",
             "antlr4-python3-runtime==4.13.2", "mpmath==1.3.0"), True, None),),
        True, True, True, ()),
    "pupa": BenchmarkProtocolSpec(
        "pupa", versions.PUPA_TASK_CONTRACT, "papillon_trusted_untrusted_trusted_v1",
        "private_query_trusted_pipeline_v1", "papillon_three_artifacts_v1", "papillon_artifacts_v1",
        "papillon_quality_privacy_fake_oracle_v1", "papillon_team_metric_unfrozen",
        None, None, "(forward OR forward==reverse + 1 - leakage)/2; trace >=1",
        "HOLD: team trust boundary and leakage exposure unselected",
        ("pii_str", "reference_response", "judge_outputs", "score"), ("llm_request",),
        (SystemDependency("papillon_model_roles_v1", ("trusted_model", "untrusted_model",
            "GPT-4.1-mini_judge"), False, "PUPA_REAL_JUDGE_AND_MODEL_PIPELINE_NOT_FROZEN"),),
        True, False, False, ("PUPA_TEAM_AGGREGATION_POLICY_NOT_FROZEN", "PUPA_RESPONSIBILITY_POLICY_NOT_FROZEN"),
        trusted_pipeline_input_fields=("user_query",)),
}


def protocol_input(benchmark_id: str, input_id: str, row: Mapping[str, object],
                   output_contract: str) -> BenchmarkInput:
    """Allowlist projection. Evaluator fields never reach solver/aggregator inputs."""
    protocol = PROTOCOLS[benchmark_id]
    field, = protocol.public_solver_fields
    problem = row.get(field)
    if not isinstance(problem, str) or not problem.strip():
        raise SearchContractError("BENCHMARK_PUBLIC_INPUT_INVALID")
    return BenchmarkInput(input_id, problem, output_contract, benchmark_id=benchmark_id,
        benchmark_version=protocol.identity(), parser_contract=protocol.parser_contract_id)
