"""Machine-readable benchmark capabilities; unresolved science stays closed."""

from __future__ import annotations

from dataclasses import dataclass

from ..search.schemas import BenchmarkCapabilities, SearchContractError
from .protocols import PROTOCOLS, BenchmarkProtocolSpec


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
    def protocol(self) -> BenchmarkProtocolSpec:
        return PROTOCOLS[self.benchmark_id]

    @property
    def system_ready(self) -> bool:
        return self.protocol.system_ready

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
        blockers = [code for ready, code in checks if not ready]
        blockers.extend(self.protocol.unresolved_scientific_blockers)
        blockers.extend(dep.blocker for dep in self.protocol.system_dependencies
                        if not dep.integration_frozen and dep.blocker)
        if self.benchmark_id == "math":
            import importlib.metadata
            from .math_worker import PINS
            try:
                if any(importlib.metadata.version(k) != v for k, v in PINS.items()):
                    blockers.append("MATH_EVALUATOR_DEPENDENCY_IDENTITY_MISMATCH")
            except importlib.metadata.PackageNotFoundError:
                blockers.append("MATH_EVALUATOR_DEPENDENCIES_NOT_AVAILABLE")
        return tuple(dict.fromkeys(blockers))


def _spec(key: str, *, data_frozen: bool = False) -> BenchmarkSpec:
    protocol = PROTOCOLS[key]
    binary = protocol.responsibility_policy_frozen
    caps = BenchmarkCapabilities(binary, binary, key in {"hotpotqa", "hover", "math"}, binary, binary)
    aggregations = (protocol.aggregation_policy_id,) if protocol.aggregation_policy_id else ()
    return BenchmarkSpec(
        benchmark_id=key, upstream={"hotpotqa": "hotpotqa/hotpot", "hover": "hover-nlp/hover",
            "ifbench": "gepa-ai/gepa-artifact (vendored IFBench)", "math": "hendrycks/math",
            "pupa": "PAPILLON/PUPA"}[key],
        benchmark_version="cbefbc1aa0f43dd39874ec4bf42211365dbda42e" if key == "ifbench" else protocol.task_contract_id,
        task_type=protocol.task_contract_id, output_type=protocol.output_contract_id,
        member_metric=protocol.member_metric_id, default_aggregation=protocol.aggregation_policy_id,
        supported_aggregations=aggregations, capabilities=caps, dataset_adapter_ready=True,
        evaluator_ready=protocol.protocol_frozen, provenance_frozen=data_frozen, split_frozen=data_frozen,
        aggregation_policy_frozen=protocol.aggregation_policy_frozen,
        responsibility_policy_frozen=protocol.responsibility_policy_frozen,
        output_contract_frozen=protocol.protocol_frozen,
        reason="Scientific task/output/evaluator contracts frozen independently of data, system dependencies and responsibility readiness.",
        dataset_identity="f552cacf1f51be8bc7b5867609eae5690fd4e250ab254310d86bf35119095a62" if data_frozen else None,
        split_identity="7076ebb0dac504fd279093043dd3851be0d2d577af8d02a530b08a7840181e2d" if data_frozen else None,
        requires_llm_aggregation=None if key == "pupa" else key in {"hover", "ifbench"},
        requires_new_responsibility_policy=not binary)


BENCHMARKS = {key: _spec(key, data_frozen=key == "ifbench") for key in PROTOCOLS}

def benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[benchmark_id.casefold()]
    except KeyError as exc:
        raise SearchContractError("BENCHMARK_UNKNOWN") from exc


def benchmark_preflight(benchmark_id: str) -> dict[str, object]:
    spec = benchmark_spec(benchmark_id)
    blockers = spec.blockers()
    return {"benchmark_id": spec.benchmark_id, "gate": "HOLD_PRE_PROVIDER" if blockers else "CAPABILITY_READY",
            "blockers": list(blockers), "reason": spec.reason, "provider_attempts": 0,
            "protocol_identity": spec.protocol.identity(),
            "data_frozen": spec.provenance_frozen and spec.split_frozen,
            "task_contract_frozen": spec.protocol.protocol_frozen,
            "system_ready": spec.system_ready,
            "evaluator_contract_frozen": spec.evaluator_ready,
            "output_contract_frozen": spec.output_contract_frozen,
            "aggregation_policy_frozen": spec.aggregation_policy_frozen,
            "responsibility_policy_frozen": spec.responsibility_policy_frozen,
            "validation_calls": 0, "test_calls": 0}
