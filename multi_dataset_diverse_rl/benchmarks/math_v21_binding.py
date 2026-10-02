"""Fresh V2.1 MATH authority; historical execution bindings grant no access."""
import importlib.metadata
import json

from .math_execution import MATHExecutionBinding, provider_bounds
from .math_domain_v2 import MATHBenchmarkAdapterV2, SETTINGS
from .math_worker import PINS
from .math_v21_interface import MATHV21BenchmarkAdapter, interface_for_contract, v5_interface_contract
from .experiment_splits import ExperimentSplitReader, COUNTS, ROLES
from .data_freeze import digest, file_hash, SOURCE_PINS
from .protocols import MATH_PROTOCOL_V2
from ..search.schemas import SearchMethodConfig, GlobalStopConfig
from ..governance.token_accounting import POLICY
from .. import versions


def competence_binding(contract):
    return dict(identity="MATH_INITIAL_COMPETENCE_OPTIMIZE_V2_1",
        role="optimize", count=150, metric="binary_correct_count",
        evaluator="MATH_EQUIVALENCE_V2", aggregation_independent=True,
        split_identity=contract["split_manifest_sha256"],
        support_identity=contract["membership_hashes"]["optimize"],
        candidate_support="same_complete_optimize", immutable=True)


class MATHV21Binding(MATHExecutionBinding):
    def benchmark(self):
        return MATHV21BenchmarkAdapter(self.contract)

    def reader(self):
        c = self.contract
        return ExperimentSplitReader(self.path(c["canonical_root"]), self.path(c["split_directory"]), "math",
            expected_manifest_sha256=c["split_manifest_sha256"], expected_protocol=versions.MATH_SCORABLE_SPLIT_VERSION)

    def method(self, arm):
        pattern, memory = self.contract["arms"][arm]
        mechanism = {}
        if pattern:
            mechanism["pattern_provider_binding"] = digest(dict(provider=self.contract["provider"],
                model=self.contract["models"]["pattern"], prompt=self.contract["pattern_prompt_sha256"],
                decoding=self.contract["decoding"]))
        if memory:
            mechanism["memory"] = self.contract["memory_limits"]
        return SearchMethodConfig.v2_1(diagnosis_policy=self.contract["responsibility"],
            aggregation_policy=self.contract["aggregation"],
            pattern_policy=versions.UNIFIED_FOCUSED_PATTERN_VERSION if pattern else versions.UNIFIED_NULL_PATTERN_VERSION,
            memory_policy=versions.UNIFIED_EXPERIENCE_MEMORY_VERSION if memory else versions.UNIFIED_NULL_MEMORY_VERSION,
            mechanism_config=mechanism,
            global_stop=GlobalStopConfig(emergency_max_provider_calls=self.contract["provider_bounds"]["successful_provider_calls"]))

    def blockers(self):
        c = self.contract
        try:
            from ..governance.math_paired_validation import POLICY as VALIDATION_POLICY
            decoding=dict(temperature=0.0, max_output_tokens=1800, invalid_response_retries=0, sdk_retries=0,
                transport_retries=20, timeout_seconds=120, retry_sleep_seconds=1.5, retry_backoff_ceiling_seconds=60)
            if c['solver_output_interface']==v5_interface_contract():
                decoding['solver_max_output_tokens']=3600
            fixed = dict(identity=versions.MATH_V2_1_EXECUTION_BINDING_VERSION, benchmark_id="math",
                method_identity=versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION,
                method_implementation_sha="01025f7097ca3a3e248e370347e256ae0aea9046",
                benchmark_protocol_sha256=MATH_PROTOCOL_V2.identity(),
                answer_domain=versions.MATH_ANSWER_DOMAIN_VERSION, reference_validity=versions.MATH_SCORABLE_REFERENCE_VERSION,
                evaluator="MATH_EQUIVALENCE_V2", payload_parser_identity="MATH_PAYLOAD_PARSER_V2",
                split_version=versions.MATH_SCORABLE_SPLIT_VERSION,
                aggregation=versions.EQUIVALENCE_PLURALITY_VERSION, responsibility=versions.BINARY_PLURALITY_RESPONSIBILITY_VERSION,
                allocation_policy=versions.UNIFIED_TARGET_POLICY_VERSION,
                transition_policy=versions.UNIFIED_COMPETENCE_TRANSITION_VERSION,
                stop_policy=versions.UNIFIED_GLOBAL_STOP_VERSION,
                models=dict(solver="qwen3-8b", optimizer_reflection="qwen3.7-flash", pattern="qwen3.7-flash", solver_thinking=False),
                provider="lwj", concurrency=dict(solver=1, optimizer_reflection=1, pattern=1),
                decoding=decoding,
                budget=dict(metric_calls=36, reflection_minibatch_size=3, promotion=2, local_patience=3, team_patience=2),
                memory_limits=dict(top_k_private=3, top_k_shared=3, max_context_chars=1200, private_storage_limit=24, shared_storage_limit=48),
                arms=dict(A1=[False,False], A2=[True,False], A3=[False,True], A4=[True,True]), seeds=[81],
                access=dict(optimize="adaptive", shadow="private_adaptive_gate", validation="post_freeze_read_only_not_authorized", test="sealed"),
                canary_phase="first_parent_team_epoch_or_first_commit", cache_policy="exact_request_role_split_cache_v1",
                shadow_count=300, reference_extractor=versions.MATH_REFERENCE_EXTRACTOR_VERSION,
                solver_output_interface=interface_for_contract(c)[1], evaluator_pins=PINS,
                verify_settings_sha256=digest(SETTINGS), initial_competence_binding=competence_binding(c),
                post_search_validation_policy=VALIDATION_POLICY)
            if any(c.get(k) != v for k,v in fixed.items()):
                return ("MATH_V2_1_SCIENTIFIC_BINDING_MISMATCH",)
            if any(k in c for k in ("parent_binding_path", "amendment_parent_binding_path")):
                return ("HISTORICAL_EXECUTION_AUTHORITY_FORBIDDEN",)
            if c["execution_phase"] not in {"canary", "pilot"}:
                return ("MATH_V2_1_PHASE_MISMATCH",)
            attempt = c["execution_attempt_id"]
            if not attempt.startswith("math_unified_v2_1_A1_seed81_" + c["execution_phase"] + "_attempt") or c["cache_namespace"] != attempt:
                return ("MATH_V2_1_FRESH_ATTEMPT_MISMATCH",)
            if c["canary_attempt_id"] != attempt:
                return ("MATH_V2_1_ATTEMPT_MISMATCH",)
            bounds = provider_bounds(optimize_count=150, shadow_count=300)
            if c["execution_phase"] == "pilot":
                bounds.update(max_opportunities=30_000_000, solver_calls=30_000_000, reflection_calls=30_000_000,
                    pattern_calls=0, successful_provider_calls=30_000_000, transport_attempts=30_000_000)
            if c["provider_bounds"] != bounds:
                return ("MATH_V2_1_RESOURCE_CEILING_MISMATCH",)
            for name, version in PINS.items():
                if importlib.metadata.version(name) != version:
                    return ("MATH_EVALUATOR_DEPENDENCY_IDENTITY_MISMATCH",)
            if c["provider_sdk"] != dict(name="openai", version=importlib.metadata.version("openai")):
                return ("PROVIDER_SDK_IDENTITY_MISMATCH",)
            for path_key, hash_key in [("accounting_policy_path","accounting_policy_sha256"),
                    ("validation_accounting_metadata_path","validation_accounting_metadata_sha256"),
                    ("initial_team_path","initial_team_artifact_sha256"), ("pattern_prompt_path","pattern_prompt_sha256")]:
                if file_hash(self.path(c[path_key])) != c[hash_key]:
                    return ("MATH_V2_1_ARTIFACT_HASH_MISMATCH",)
            if json.loads(self.path(c["accounting_policy_path"]).read_bytes()) != POLICY or json.loads(self.path(c["verify_settings_path"]).read_bytes()) != SETTINGS:
                return ("MATH_V2_1_EVALUATOR_ACCOUNTING_MISMATCH",)
            canonical_path = self.path(c["canonical_root"]) / "manifests/math.json"
            if file_hash(canonical_path) != c["canonical_manifest_sha256"]:
                return ("CANONICAL_MANIFEST_IDENTITY_MISMATCH",)
            canonical = json.loads(canonical_path.read_bytes())
            if any(canonical["source"].get(k) != v for k,v in SOURCE_PINS["math"].items()):
                return ("CANONICAL_SOURCE_IDENTITY_MISMATCH",)
            # Byte hashing is integrity metadata, never held-out projection.
            for source in canonical["source"]["canonical_sources"]:
                if file_hash(self.path(c["canonical_root"]) / "raw/math" / (source["name"] + ".jsonl")) != source["canonical_sha256"]:
                    return ("CANONICAL_SOURCE_HASH_MISMATCH",)
            reader = self.reader()
            if (reader.manifest["counts"] != COUNTS["math"] or reader.manifest["hashes"] != c["membership_hashes"]
                    or reader.manifest["canonical_dataset_manifest_sha256"] != c["canonical_manifest_sha256"]):
                return ("MATH_V2_1_SPLIT_MISMATCH",)
            for role in ROLES:
                rows = [r for r in reader.members if r["project_split"] == role]
                if len(rows) != COUNTS["math"][role] or digest([r["stable_example_id"] for r in rows]) != c["membership_hashes"][role] or any(r["source_split"] != ("test" if role=="test" else "train") for r in rows):
                    return ("MATH_V2_1_MEMBERSHIP_MISMATCH",)
                for other in ROLES:
                    if other == role: continue
                    peers = [r for r in reader.members if r["project_split"] == other]
                    if any({r[k] for r in rows} & {r[k] for r in peers} for k in ("stable_example_id","content_sha256","input_sha256")):
                        return ("MATH_SPLIT_OVERLAP",)
            team = json.loads(self.path(c["initial_team_path"]).read_bytes())
            from ..evaluation.mutable_prompt_contract import validate_mutable_decision_procedure
            import hashlib
            if team["team_version"] != c["initial_team_version"] or team["initial_team_data_dependency"] != "NONE" or [m["member_id"] for m in team["members"]] != list(range(5)):
                return ("MATH_INITIAL_TEAM_CONTRACT_MISMATCH",)
            for member in team["members"]:
                validate_mutable_decision_procedure(member["prompt"])
                if hashlib.sha256(member["prompt"].encode()).hexdigest() != member["prompt_sha256"]:
                    return ("MATH_INITIAL_PROMPT_HASH_MISMATCH",)
            if (len({m["prompt"] for m in team["members"]}) != 5 or digest([m["prompt_sha256"] for m in team["members"]]) != c["initial_team_sha256"]
                    or c["initial_team_sha256"] != "4ca685adff5ca8a97e53b0aa0f5715783c7c6937433bd5bbc1308cbad9e97200"):
                return ("MATH_INITIAL_TEAM_HASH_MISMATCH",)
            metadata = json.loads(self.path(c["validation_accounting_metadata_path"]).read_bytes())
            rows = [r for r in reader.members if r["project_split"] == "validation"]
            if (metadata["split_manifest_sha256"] != c["split_manifest_sha256"] or metadata["solver_output_interface"] != c["solver_output_interface"]
                    or metadata["decoding"] != c["decoding"] or metadata["context"] != "ACCOUNTING_DATA_PREP_CONTEXT"
                    or [(r["example_id"],r["input_sha256"]) for r in metadata["examples"]] != [(r["stable_example_id"],r["input_sha256"]) for r in rows]
                    or any(type(r["blank_prompt_serialized_request_bytes"]) is not int or r["blank_prompt_serialized_request_bytes"] <= 0 for r in metadata["examples"])):
                return ("VALIDATION_ACCOUNTING_METADATA_MISMATCH",)
            if not self.path(c["token_ledger_directory"]).is_relative_to(self.root / "runs") or any(len(c[k]) != 64 for k in ("task_authorization_sha256","continuation_authorization_sha256")):
                return ("MATH_V2_1_AUTHORIZATION_LEDGER_MISMATCH",)
            from ..local_optimizers.gepa_optimizer import verify_frozen_gepa_engine_contract
            verify_frozen_gepa_engine_contract()
            return ()
        except (KeyError,OSError,ValueError,TypeError,importlib.metadata.PackageNotFoundError):
            return ("MATH_V2_1_BINDING_INVALID",)
