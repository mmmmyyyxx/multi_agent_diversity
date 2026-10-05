"""Versioned MATH readiness and adapter/split binding for the shared graph."""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path

from ... import versions
from ..access import DataPurpose
from ..data_freeze import file_hash, digest
from ..experiment_splits import ExperimentSplitReader, COUNTS, ROLES
from ..math import MATHBenchmarkAdapter
from ..math_interface import solver_interface_contract
from ..math_worker import PINS
from ..protocols import PROTOCOLS, protocol_input
from ...search.schemas import GlobalStopConfig, SearchContractError, SearchMethodConfig


def provider_bounds(*, optimize_count, shadow_count, members=5):
    # Raw V <= 4*Optimize; every feasible positive V is an integer >= 1.
    # With V/(1+f), an already visited member receives at most Vmax+1
    # selections before an unvisited positive member outranks it, including ties.
    opportunities = 1 + (members - 1) * (4 * optimize_count + 1)
    proposals = (36 - 1) // (2 * 3)
    solver_per_opportunity = 36 + proposals * 12 + 2 * optimize_count + shadow_count
    solver = members * (optimize_count + shadow_count) + opportunities * solver_per_opportunity
    reflection = opportunities * proposals
    pattern = opportunities
    return dict(max_opportunities=opportunities, max_proposals_per_opportunity=proposals,
                solver_per_opportunity=solver_per_opportunity, solver_calls=solver,
                reflection_calls=reflection, pattern_calls=pattern,
                successful_provider_calls=solver + reflection + pattern,
                transport_attempts=(solver + reflection + pattern) * 21)


class MATHExecutionBinding:
    def __init__(self, root: Path, contract: dict):
        self.root = root.resolve()
        self.contract = contract

    def path(self, relative):
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise SearchContractError("EXECUTION_BINDING_PATH_ESCAPE")
        return p

    def reader(self):
        c = self.contract
        return ExperimentSplitReader(self.path(c["canonical_root"]), self.path(c["split_directory"]), "math",
            expected_manifest_sha256=c["split_manifest_sha256"], expected_protocol=versions.MATH_EXPERIMENT_SPLIT_VERSION)

    def blockers(self):
        c = self.contract
        try:
            if c["identity"] != versions.MATH_EXECUTION_BINDING_VERSION or c["benchmark_protocol_sha256"] != PROTOCOLS["math"].identity():
                return ("MATH_BENCHMARK_CONTRACT_MISMATCH",)
            if c.get("solver_output_interface") != solver_interface_contract():
                return ("MATH_SOLVER_INTERFACE_BINDING_MISMATCH",)
            if c.get("canary_attempt_id") != "math_unified_v2_A1_seed81_canary_attempt2" or c.get("cache_namespace") != "math_v2_solver_interface_v1_2_canary_attempt2":
                return ("MATH_FRESH_ATTEMPT_CACHE_BINDING_MISMATCH",)
            if c["split_version"] != versions.MATH_EXPERIMENT_SPLIT_VERSION or c["evaluator"] != PROTOCOLS["math"].member_metric_id:
                return ("MATH_SPLIT_EVALUATOR_CONTRACT_MISMATCH",)
            if c["models"] != {"solver": "qwen3-8b", "optimizer_reflection": "qwen3.7-flash", "pattern": "qwen3.7-flash", "solver_thinking": False}:
                return ("MATH_MODEL_BINDING_MISMATCH",)
            if c["models"]["solver_thinking"] is not False:
                return ("MATH_MODEL_BINDING_MISMATCH",)
            if c["provider"] != "lwj" or c["concurrency"] != {"solver": 1, "optimizer_reflection": 1, "pattern": 1}:
                return ("MATH_PROVIDER_BINDING_MISMATCH",)
            if c["decoding"] != {"temperature": 0.0, "max_output_tokens": 1800, "invalid_response_retries": 0,
                                  "sdk_retries": 0, "transport_retries": 20, "timeout_seconds": 120,
                                  "retry_sleep_seconds": 1.5, "retry_backoff_ceiling_seconds": 60}:
                return ("MATH_DECODING_BINDING_MISMATCH",)
            if c["memory_limits"] != dict(top_k_private=3, top_k_shared=3, max_context_chars=1200, private_storage_limit=24, shared_storage_limit=48):
                return ("MATH_MEMORY_LIMIT_MISMATCH",)
            if c["arms"] != {"A1": [False, False], "A2": [True, False], "A3": [False, True], "A4": [True, True]} or c["seeds"] != [81, 82, 83]:
                return ("MATH_FACTORIAL_BINDING_MISMATCH",)
            if c["aggregation"] != versions.EQUIVALENCE_PLURALITY_VERSION or c["responsibility"] != versions.BINARY_PLURALITY_RESPONSIBILITY_VERSION:
                return ("MATH_AGGREGATION_RESPONSIBILITY_MISMATCH",)
            if c["budget"] != dict(metric_calls=36, reflection_minibatch_size=3, promotion=2, local_patience=3, team_patience=2):
                return ("MATH_SEARCH_BUDGET_MISMATCH",)
            if c["access"] != dict(optimize="adaptive", shadow="private_adaptive_gate", validation="post_freeze_read_only_not_authorized", test="sealed"):
                return ("MATH_ACCESS_POLICY_MISMATCH",)
            if c["canary_phase"] != "first_parent_team_epoch_or_first_commit" or c["cache_policy"] != "exact_request_role_split_cache_v1":
                return ("MATH_PHASE_CACHE_BINDING_MISMATCH",)
            if c["shadow_count"] != COUNTS["math"]["shadow"] or file_hash(self.path(c["pattern_prompt_path"])) != c["pattern_prompt_sha256"]:
                return ("MATH_SHADOW_PATTERN_BINDING_MISMATCH",)
            if c["reference_extractor"] != versions.MATH_REFERENCE_EXTRACTOR_VERSION or c["reference_validity"] != versions.MATH_REFERENCE_VALIDITY_VERSION:
                return ("MATH_REFERENCE_POLICY_MISMATCH",)
            if file_hash(self.path(c["canonical_root"]) / "manifests/math.json") != c["canonical_manifest_sha256"]:
                return ("CANONICAL_MANIFEST_IDENTITY_MISMATCH",)
            reader = self.reader()
            if reader.manifest["counts"] != COUNTS["math"] or reader.manifest["hashes"] != c["membership_hashes"]:
                return ("MATH_SPLIT_MEMBERSHIP_MISMATCH",)
            if reader.manifest["canonical_dataset_manifest_sha256"] != c["canonical_manifest_sha256"]:
                return ("MATH_CANONICAL_SPLIT_BINDING_MISMATCH",)
            members = reader.members
            if len(members) != sum(COUNTS["math"].values()) or any(r["project_split"] not in ROLES or
                    r["source_split"] != ("test" if r["project_split"] == "test" else "train") for r in members):
                return ("MATH_SOURCE_ROLE_MISMATCH",)
            for role in ROLES:
                selected = [r for r in members if r["project_split"] == role]
                if len(selected) != COUNTS["math"][role] or digest([r["stable_example_id"] for r in selected]) != c["membership_hashes"][role]:
                    return ("MATH_SPLIT_MEMBERSHIP_MISMATCH",)
            for i, role in enumerate(ROLES):
                for other in ROLES[i + 1:]:
                    for field in ("stable_example_id", "content_sha256", "input_sha256"):
                        if {r[field] for r in members if r["project_split"] == role} & {r[field] for r in members if r["project_split"] == other}:
                            return ("MATH_SPLIT_OVERLAP",)
            if c["evaluator_pins"] != PINS or any(importlib.metadata.version(k) != v for k, v in PINS.items()):
                return ("MATH_EVALUATOR_DEPENDENCY_IDENTITY_MISMATCH",)
            from ...local_optimizers.gepa_optimizer import verify_frozen_gepa_engine_contract
            verify_frozen_gepa_engine_contract()
            team_path = self.path(c["initial_team_path"])
            if file_hash(team_path) != c["initial_team_artifact_sha256"]:
                return ("MATH_INITIAL_TEAM_ARTIFACT_MISMATCH",)
            team = json.loads(team_path.read_bytes())
            if team["team_version"] != c["initial_team_version"]:
                return ("MATH_INITIAL_TEAM_VERSION_MISMATCH",)
            prompts = tuple(m["prompt"] for m in team["members"])
            from ...evaluation.mutable_prompt_contract import validate_mutable_decision_procedure
            if len(set(prompts)) != 5 or team["initial_team_data_dependency"] != "NONE" or [m["member_id"] for m in team["members"]] != list(range(5)):
                return ("MATH_INITIAL_TEAM_CONTRACT_MISMATCH",)
            import hashlib
            for m in team["members"]:
                validate_mutable_decision_procedure(m["prompt"])
                if hashlib.sha256(m["prompt"].encode()).hexdigest() != m["prompt_sha256"]:
                    return ("MATH_INITIAL_PROMPT_HASH_MISMATCH",)
            if digest([m["prompt_sha256"] for m in team["members"]]) != c["initial_team_sha256"] or team["ordered_team_sha256"] != c["initial_team_sha256"]:
                return ("MATH_INITIAL_TEAM_HASH_MISMATCH",)
            if c["provider_bounds"] != provider_bounds(optimize_count=COUNTS["math"]["optimize"], shadow_count=COUNTS["math"]["shadow"]):
                return ("MATH_PROVIDER_UPPER_BOUND_MISMATCH",)
            return ()
        except (OSError, KeyError, ValueError, importlib.metadata.PackageNotFoundError):
            return ("MATH_RUNTIME_BINDING_INVALID",)

    def method(self, arm):
        if arm not in self.contract["arms"]:
            raise SearchContractError("MATH_ARM_NOT_FROZEN")
        pattern, memory = self.contract["arms"][arm]
        mechanisms = {}
        if pattern:
            mechanisms["pattern_provider_binding"] = digest({"provider": self.contract["provider"], "model": self.contract["models"]["pattern"],
                                                             "prompt": self.contract["pattern_prompt_sha256"], "decoding": self.contract["decoding"]})
        if memory:
            mechanisms["memory"] = self.contract["memory_limits"]
        return SearchMethodConfig.v2(diagnosis_policy=self.contract["responsibility"], aggregation_policy=self.contract["aggregation"],
            pattern_policy=versions.UNIFIED_PATTERN_DIAGNOSTIC_VERSION if pattern else versions.UNIFIED_NULL_PATTERN_VERSION,
            memory_policy=versions.UNIFIED_STRUCTURED_MEMORY_VERSION if memory else versions.UNIFIED_NULL_MEMORY_VERSION,
            mechanism_config=mechanisms, global_stop=GlobalStopConfig(emergency_max_provider_calls=self.contract["provider_bounds"]["successful_provider_calls"]))

    def examples(self, role):
        from ...search.binary_runtime import CorrectnessExample
        purpose = DataPurpose.ADAPTIVE_GATE if role == "shadow" else DataPurpose.EVIDENCE
        rows = self.reader().rows(role, purpose)
        adapter = self.benchmark()
        if hasattr(adapter, "require_scorable"):
            for row in rows:
                adapter.require_scorable(row["reference_final_answer"])
        return tuple(CorrectnessExample(protocol_input("math", r["stable_example_id"], r["content"], adapter.output_contract,
                                        protocol=getattr(adapter,'protocol',None)),
                                        r["reference_final_answer"]) for r in rows)

    def benchmark(self):
        return MATHBenchmarkAdapter()

    def compose(self, *, arm, seed, solver, reflection, pattern_provider, run_root, optimize_fn=None):
        blockers = self.blockers()
        if blockers or seed not in self.contract["seeds"]:
            raise SearchContractError("HOLD_PRE_PROVIDER: " + ",".join(blockers or ("SEED_NOT_FROZEN",)))
        if (solver.output_contract_id != PROTOCOLS["math"].output_contract_id or solver.solver_contract_id != "COMMON_SOLVER_CONTRACT_V1"
                or solver.broker.contract != self.contract or reflection.broker is not solver.broker
                or solver.broker.arm != arm or solver.broker.seed != seed):
            raise SearchContractError("MATH_PROVIDER_PORT_BINDING_MISMATCH")
        from ...search.binary_composition import build_binary_orchestrator
        from ...search.scientific_aggregation import EquivalencePluralityAggregation
        from ...search.gepa_v2 import GEPATeamExposureOptimizer
        team = json.loads(self.path(self.contract["initial_team_path"]).read_bytes())
        if self.contract['identity'] in {versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION}:
            if optimize_fn is not None:raise SearchContractError('LAYER1_OFFICIAL_GEPA_OVERRIDE_FORBIDDEN')
            from ...search.layer1_responsibility import ResponsibilityConditionedOptimizer
            from ...search.layer1_memory import MemoryConditionedOptimizer
            from ...search.pattern_layer1 import PatternMemoryOptimizer
            factory=(PatternMemoryOptimizer if self.contract['identity']==versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION else MemoryConditionedOptimizer if self.contract['identity']==versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION else ResponsibilityConditionedOptimizer)
            if self.contract.get('pattern_policy',{}).get('discovery')==versions.GRADIENT_PATTERN_DISCOVERY_VERSION:
                from ...search.pattern_layer1 import GradientPatternMemoryOptimizer
                factory=GradientPatternMemoryOptimizer
            optimizer=factory(evaluator=solver,reflection_lm=reflection,
                accounting_reader=reflection.accounting,run_root=run_root)
        else:
            optimizer = GEPATeamExposureOptimizer(evaluator=solver, reflection_lm=reflection,
                accounting_reader=reflection.accounting, run_root=run_root, optimize_fn=optimize_fn)
        return build_binary_orchestrator(benchmark=self.benchmark(), aggregation=EquivalencePluralityAggregation(),
            examples=self.examples("optimize"), prompts=tuple(m["prompt"] for m in team["members"]), solver=solver, optimizer=optimizer,
            method=self.method(arm), seed=seed, shadow_loader=lambda: self.examples("shadow"), shadow_count=self.contract["shadow_count"],
            runtime_readiness=self.blockers, pattern_provider=pattern_provider,
            first_parent_epoch=self.contract.get("execution_phase", "canary") == "canary",
            one_production_opportunity=(self.contract.get('identity')in versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS
                and self.contract.get('execution_phase')=='canary'),
            provider_call_reader=lambda: solver.broker.successes)
