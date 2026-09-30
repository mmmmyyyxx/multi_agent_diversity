"""Machine-readable benchmark capabilities; unresolved science stays closed."""

from __future__ import annotations

from dataclasses import dataclass

from ..search.schemas import BenchmarkCapabilities, SearchContractError


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    upstream: str
    benchmark_version: str | None
    task_type: str
    output_type: str
    member_metric: str | None
    default_aggregation: str | None
    supported_aggregations: tuple[str, ...]
    capabilities: BenchmarkCapabilities
    dataset_adapter_ready: bool
    evaluator_ready: bool
    provenance_frozen: bool
    split_frozen: bool
    aggregation_policy_frozen: bool
    responsibility_policy_frozen: bool
    output_contract_frozen: bool
    reason: str
    dataset_identity: str | None = None
    split_identity: str | None = None
    requires_llm_aggregation: bool | None = None
    requires_new_responsibility_policy: bool = True

    @property
    def supports_current_plurality_responsibility(self) -> bool:
        return self.capabilities.supports_current_responsibility

    @property
    def supports_boolean_member_success(self) -> bool:
        return self.capabilities.supports_boolean_member_success

    @property
    def aggregation_ready(self) -> bool:
        return bool(self.supported_aggregations and self.aggregation_policy_frozen)

    @property
    def responsibility_ready(self) -> bool:
        return self.capabilities.supports_current_responsibility and self.responsibility_policy_frozen

    @property
    def unified_search_ready(self) -> bool:
        return not self.blockers()

    def blockers(self) -> tuple[str, ...]:
        checks = (
            (self.dataset_adapter_ready, "BENCHMARK_ADAPTER_NOT_READY"),
            (self.provenance_frozen, "BENCHMARK_PROVENANCE_NOT_FROZEN"),
            (self.split_frozen, "BENCHMARK_SPLIT_NOT_FROZEN"),
            (self.evaluator_ready, "BENCHMARK_EVALUATOR_NOT_FROZEN"),
            (self.aggregation_policy_frozen, "AGGREGATION_POLICY_NOT_FROZEN"),
            (self.responsibility_policy_frozen, "RESPONSIBILITY_POLICY_NOT_FROZEN"),
            (self.output_contract_frozen, "OUTPUT_CONTRACT_NOT_FROZEN"),
        )
        return tuple(code for ready, code in checks if not ready)


_NO_VOTE = BenchmarkCapabilities(False, False, False)
_QA_VOTE = BenchmarkCapabilities(True, False, True)

BENCHMARKS: dict[str, BenchmarkSpec] = {
    "hotpotqa": BenchmarkSpec(
        "hotpotqa", "hotpotqa/hotpot", None, "answer-only multi-hop QA (candidate)",
        "normalized answer text", "answer EM (candidate)", "plurality", ("plurality", "llm"),
        _QA_VOTE, True, True, False, False, False, False, True,
        "Project variant, answer-only versus joint metric, corpus and split are unselected; "
        "current BBH responsibility has no HotpotQA implementation.",
    ),
    "hover": BenchmarkSpec(
        "hover", "hover-nlp/hover", None, "claim verification and evidence (candidate)",
        "structured verdict and evidence", None, None, (), _NO_VOTE,
        False, False, False, False, False, False, False,
        "Exact project task, evidence unit, metric, aggregate and split are unselected.",
    ),
    "ifbench": BenchmarkSpec(
        "ifbench", "gepa-ai/gepa-artifact (vendored IFBench)", "cbefbc1aa0f43dd39874ec4bf42211365dbda42e", "instruction following (candidate)",
        "free-form response", None, None, (), _NO_VOTE,
        False, False, True, True, False, False, False,
        "Vendored source and memberships are frozen. GEPA fraction-of-constraints "
        "scorer provenance is audited; evaluator integration, aggregate and responsibility remain unselected.",
        dataset_identity="f552cacf1f51be8bc7b5867609eae5690fd4e250ab254310d86bf35119095a62",
        split_identity="7076ebb0dac504fd279093043dd3851be0d2d577af8d02a530b08a7840181e2d",
    ),
    "pupa": BenchmarkSpec(
        "pupa", "PAPILLON/PUPA (candidate)", None, "privacy-preserving response (candidate)",
        "unselected", None, None, (), _NO_VOTE,
        False, False, False, False, False, False, False,
        "No project PUPA source, privacy task contract or scorer is retained.",
    ),
    "math": BenchmarkSpec(
        "math", "hendrycks/math (candidate)", None, "mathematical answer (candidate)",
        "boxed mathematical expression", None, None, (), _NO_VOTE,
        False, False, False, False, False, False, False,
        "Exact dataset variant and trusted equivalence evaluator are unbound; "
        "equivalence-aware voting is unproved.",
    ),
}

def benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[benchmark_id.casefold()]
    except KeyError as exc:
        raise SearchContractError("BENCHMARK_UNKNOWN") from exc


def benchmark_preflight(benchmark_id: str) -> dict[str, object]:
    spec = benchmark_spec(benchmark_id)
    blockers = spec.blockers()
    return {"benchmark_id": spec.benchmark_id, "gate": "HOLD_PRE_PROVIDER" if blockers else "CAPABILITY_READY",
            "blockers": list(blockers), "reason": spec.reason, "provider_attempts": 0}
