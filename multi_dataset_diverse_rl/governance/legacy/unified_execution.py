"""Frozen Unified canary admission; readiness never grants API authorization."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ..source_identity import build_unified_source_identity, hash_scope
from ..repository import validate_manifest_v2
from ..startup_identity import canonical_sha256
from ...persistence.durable_io import atomic_write_json, append_jsonl, read_json
from ...search.schemas import SearchContractError
from ... import versions


PREP_SCHEMA = "unified_canary_prep_v1"


def consumption_path(root, scope):
    return root / "runs/unified_authorization_consumption" / (canonical_sha256(scope) + ".json")


def bound_preflight(root, manifest):
    from ...benchmarks.math_execution import MATHExecutionBinding
    ref = manifest.get("execution_binding", {})
    if ref.get("identity") not in {versions.MATH_EXECUTION_BINDING_VERSION, versions.MATH_AUTONOMOUS_EXECUTION_BINDING_VERSION, versions.MATH_DOMAIN_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION, *versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS}:
        raise SearchContractError("UNSUPPORTED_EXECUTION_BINDING")
    errors = validate_manifest_v2(root, manifest)
    if manifest.get("lifecycle", {}).get("status") != "PREEXECUTION_FROZEN":
        errors.append("PREEXECUTION_NOT_FROZEN")
    path = (root / ref["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        errors.append("EXECUTION_BINDING_HASH_MISMATCH")
        return {"gate": "HOLD", "blockers": errors, "provider_attempts": 0}
    if ref["identity"] in {versions.MATH_AUTONOMOUS_EXECUTION_BINDING_VERSION, versions.MATH_DOMAIN_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION, *versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS}:
        from ...benchmarks.legacy.math_domain_binding import execution_binding
        binding = execution_binding(root, read_json(path))
    else:
        binding = MATHExecutionBinding(root, read_json(path))
    errors.extend(binding.blockers())
    c = binding.contract
    current = c["identity"] in {versions.MATH_V2_1_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION, *versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS}
    expected = {"benchmark_id": "math", "benchmark_protocol_id": c["benchmark_protocol_sha256"],
        "method_family": versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION if current else "unified_team_prompt_search_v2",
        "method_identity": versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION if current else "unified_team_prompt_search_v2",
        "search_engine_identity": binding.method(c.get('execution_arm','A1')).search_engine,
        "search_acceptance_identity": binding.method(c.get('execution_arm','A1')).search_acceptance_policy,
        "evidence_identity": versions.UNIFIED_FOCUSED_EVIDENCE_VERSION if current else "variable_pattern_capable_evidence_v1", "feasibility_identity": "variable_evidence_feasibility_v1",
        "transition_identity": versions.UNIFIED_COMPETENCE_TRANSITION_VERSION if current else "common_safe_v1", "adaptive_gate_identity": "winner_only_shadow_v1",
        "pattern_identity": "null_pattern_v1", "memory_identity": "null_memory_v1", "mechanism_config": {},
        "global_stop_identity": "team_epoch_no_commit_v1", "seed": 81,
        "dataset_manifest_identity": c["canonical_manifest_sha256"], "split_identity": c["split_manifest_sha256"],
        "aggregation_identity": c["aggregation"], "responsibility_identity": c["responsibility"],
        "models": {"solver": "qwen3-8b", "optimizer": "qwen3.7-flash", "solver_thinking": False},
        "concurrency": {"solver": 1, "optimizer": 1},
        "provider_policy": {"identity": "lwj", "frozen": True},
        "cache_policy": {"identity": c["cache_policy"], "frozen": True},
        "solver_output_interface_identity": c["solver_output_interface"]["identity"],
        "parser_identity": c.get("payload_parser_identity",c["solver_output_interface"]["parser_identity"]),
        "budget": dict(regime="saturation", saturation_identity="team_epoch_no_commit_v1",
            successful_provider_call_ceiling=c["provider_bounds"]["successful_provider_calls"],
            transport_attempt_ceiling=c["provider_bounds"]["transport_attempts"]),
        "access": dict(search_access="frozen_search", shadow_access="frozen_adaptive_gate", validation_access="not_authorized", test_access="sealed")}
    if current:
        expected["initial_competence_binding"] = c["initial_competence_binding"]
    if c['identity'] in {versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION}:
        method=binding.method(c['execution_arm'])
        expected.update(pattern_identity=method.pattern_policy,memory_identity=method.memory_policy,evidence_identity=method.evidence_policy,
            mechanism_config=method.mechanism_config)
    if "solver_decoding_policy" in c:
        expected["solver_decoding_policy"] = c["solver_decoding_policy"]
    if "prediction_validity_policy" in c:
        expected["prediction_validity_policy"] = c["prediction_validity_policy"]
    for k in ("invalid_recovery_policy", "low_cost_protocol", "low_cost_subsets_sha256", "optimizer_generation_policy", "optimizer_amendment_authorization_sha256", "optimizer_nonthinking_evidence_policy", "layer1_search_policy", "candidate_contract_identity", "post_search_validation_policy", "pattern_support_id_transport", "pattern_abstraction_guard"):
        if k in c and (k not in {'layer1_search_policy','candidate_contract_identity','post_search_validation_policy'} or c['identity'] in {versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION}): expected[k] = c[k]
    if any(manifest.get(k) != v for k, v in expected.items()):
        errors.append("MANIFEST_EXECUTION_BINDING_MISMATCH")
    if manifest.get("authorization", {}).get("real_api_authorized") is not False:
        errors.append("PREEXECUTION_MANIFEST_CANNOT_AUTHORIZE")
    payload = {k: v for k, v in manifest.items() if k not in {"preregistration_identity", "lifecycle", "authorization"}}
    if manifest.get("preregistration_identity") != canonical_sha256(payload):
        errors.append("PREREGISTRATION_IDENTITY_MISMATCH")
    if not errors:
        identity = execution_identity(root, c)
        if manifest["hash_closure"] != {k: identity[k] for k in manifest["hash_closure"]}:
            errors.append("MANIFEST_SOURCE_CLOSURE_MISMATCH")
        else:
            try:
                verify_source_commit(root, manifest["source_sha"], identity)
            except (SearchContractError, subprocess.CalledProcessError):
                errors.append("EXECUTION_SOURCE_COMMIT_BYTE_MISMATCH")
    return dict(gate="HOLD" if errors else "CANARY_READY_NOT_AUTHORIZED", blockers=errors,
        benchmark_id="math", readiness_identity="MATH_RUNTIME_READINESS_V1", provider_attempts=0,
        validation_calls=0, test_calls=0, real_api_authorized=False,
        binding_sha256=ref["sha256"], ready_for_authorization=not errors)


def execution_identity(root, contract):
    # Input metadata and all import/bootstrap dependencies are explicit. The
    # canonical raw source remains private and has its separate manifest hash.
    configs = [contract["split_directory"] + "/math.json", contract["initial_team_path"],
               contract["pattern_prompt_path"], contract.get("binding_path", versions.MATH_EXECUTION_BINDING_PATH)]
    configs.extend(contract[k] for k in ("parent_binding_path", "accounting_policy_path", "validation_accounting_metadata_path", "verify_settings_path", "amendment_parent_binding_path", "low_cost_subsets_path", "optimizer_amendment_authorization_path", "layer1_parent_binding_path", "layer1_amendment_authorization_path", "memory_parent_binding_path", "memory_amendment_authorization_path", "pattern_parent_binding_path", "pattern_amendment_authorization_path", "rolling_memory_freeze_path") if k in contract)
    configs.extend(contract[k] for k in ('gradient_prompt_path','gradient_parent_binding_path') if k in contract)
    if 'gradient_parent_binding_path' in contract:
        parent=read_json(root/contract['gradient_parent_binding_path'])
        configs.extend(parent[k] for k in ('pattern_amendment_authorization_path','pattern_prompt_path'))
        approval=read_json(root/parent['pattern_amendment_authorization_path'])
        configs.append(approval['guard_amendment']['parent_authority_path'])
    if contract.get('pattern_abstraction_guard') == versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION and 'gradient_parent_binding_path' not in contract:
        approval = read_json(root / contract['pattern_amendment_authorization_path'])
        configs.append(approval['guard_amendment']['parent_authority_path'])
    identity = build_unified_source_identity(root, root / contract["canonical_root"] / "manifests/math.json", [root / p for p in configs])
    files = [root / r["path"] for s in identity["scopes"].values() for r in s["files"]]
    files.append(root / contract["split_directory"] / "math.ids.jsonl")
    # CLI imports are operational dependencies; dynamic composition is pinned
    # by the benchmark and search scopes as well.
    from ..source_identity import local_imports
    pending = [root / "scripts/run_experiment.py"]
    seen = set()
    while pending:
        p = pending.pop()
        if p in seen:
            continue
        seen.add(p)
        pending.extend(local_imports(root, p) - seen)
    closure = hash_scope(root, files + list(seen))
    return {**identity, "execution_closure": closure}


def validate_frozen_source(root, receipt):
    entries = receipt["execution_closure"]["files"]
    if not entries or len({r["path"] for r in entries}) != len(entries):
        raise SearchContractError("SOURCE_CLOSURE_INVALID")
    for r in entries:
        p = (root / r["path"]).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != r["raw_sha256"]:
            raise SearchContractError("SOURCE_IDENTITY_MISMATCH: " + r["path"])
    if hash_scope(root, [root / r["path"] for r in entries]) != receipt["execution_closure"]:
        raise SearchContractError("SOURCE_CLOSURE_RECEIPT_MISMATCH")


def verify_source_commit(root, source, identity):
    for r in identity["execution_closure"]["files"]:
        blob = subprocess.check_output(["git", "show", source + ":" + r["path"]], cwd=root, stderr=subprocess.DEVNULL)
        if hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() != r["sha256"]:
            raise SearchContractError("EXECUTION_SOURCE_COMMIT_BYTE_MISMATCH: " + r["path"])


def prepare_canary(root, manifest, *, destination, arm="A1", seed=81):
    result = bound_preflight(root, manifest)
    if result["blockers"]:
        raise SearchContractError("HOLD_PRE_PROVIDER: " + ",".join(result["blockers"]))
    contract = read_json(root / manifest["execution_binding"]["path"])
    if arm != contract.get('execution_arm','A1') or seed != 81:
        raise SearchContractError("ONLY_BOUND_ARM_SEED81_CANARY_PHASE_IS_PREPARED")
    if destination.exists():
        raise SearchContractError("FRESH_PREP_DESTINATION_REQUIRED")
    contract = read_json(root / manifest["execution_binding"]["path"])
    identity = execution_identity(root, contract)
    if manifest["hash_closure"] != {k: identity[k] for k in manifest["hash_closure"]}:
        raise SearchContractError("MANIFEST_SOURCE_CLOSURE_MISMATCH")
    source = manifest["source_sha"]
    # The freeze commit may add sanitized evidence after the execution-source
    # commit. Every executable/config byte still has to match that source.
    verify_source_commit(root, source, identity)
    destination.mkdir(parents=True)
    scope = execution_scope(manifest, contract)
    payload = dict(schema_version=PREP_SCHEMA, manifest=manifest, source_identity=identity, scope=scope)
    payload["startup_identity_sha256"] = canonical_sha256(payload)
    atomic_write_json(destination / "prep.json", payload)
    atomic_write_json(destination / "authorization.json", dict(explicit_user_authorized=False, single_use=True,
        consumed=False, scope=scope, startup_identity_sha256=payload["startup_identity_sha256"]))
    return payload


def execution_scope(manifest, contract):
    scope = dict(attempt_id=contract.get("execution_attempt_id", contract["canary_attempt_id"]), arm=contract.get('execution_arm','A1'), seed=81,
        phase="team_epoch_no_commit_v1" if contract.get("execution_phase") == "pilot" else "first_parent_team_epoch_or_first_commit",
        source_sha=manifest["source_sha"], preregistration_identity=manifest["preregistration_identity"], binding_sha256=manifest["execution_binding"]["sha256"],
        models=contract["models"], provider=contract["provider"], roles=["solver", "reflection"],
        successful_provider_call_ceiling=contract["provider_bounds"]["successful_provider_calls"],
        transport_attempt_ceiling=contract["provider_bounds"]["transport_attempts"], validation_calls=0, test_calls=0)
    if "solver_decoding_policy" in contract:
        scope["solver_decoding_policy"] = contract["solver_decoding_policy"]
    if "prediction_validity_policy" in contract:
        scope["prediction_validity_policy"] = contract["prediction_validity_policy"]
    if "invalid_recovery_policy" in contract:
        scope["invalid_recovery_policy"] = contract["invalid_recovery_policy"]
        scope["low_cost_protocol"] = contract["low_cost_protocol"]
        scope["low_cost_subsets_sha256"] = contract["low_cost_subsets_sha256"]
        scope["phase"] = contract["canary_phase"] if contract["execution_phase"] == "canary" else contract["stop_policy"]
    if "optimizer_generation_policy" in contract:
        scope["optimizer_generation_policy"] = contract["optimizer_generation_policy"]
        scope["optimizer_amendment_authorization_sha256"] = contract["optimizer_amendment_authorization_sha256"]
        if contract.get('optimizer_nonthinking_evidence_policy') is not None:
            scope['optimizer_nonthinking_evidence_policy']=contract['optimizer_nonthinking_evidence_policy']
    if contract.get('identity') in {versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION}:
        scope.update(layer1_search_policy=contract['layer1_search_policy'],candidate_contract_identity=contract['candidate_contract_identity'],
            cache_policy=contract['cache_policy'],post_search_validation_policy=contract['post_search_validation_policy'],
            layer1_amendment_authorization_sha256=contract['layer1_amendment_authorization_sha256'])
    if contract.get('identity') in {versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION}:
        scope.update(memory_policy_identity=contract['memory_policy_identity'],memory_limits=contract['memory_limits'],
            optimizer_input_schema=contract['optimizer_input_schema'],panel_evidence_policy=contract['panel_evidence_policy'],
            memory_amendment_authorization_sha256=contract['memory_amendment_authorization_sha256'],initial_memory_entries=0)
    if contract.get('identity')==versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION:
        scope.update(roles=['solver','reflection','pattern'],pattern_policy=contract['pattern_policy'],shared_risk_policy=contract['shared_risk_policy'],pattern_amendment_authorization_sha256=contract['pattern_amendment_authorization_sha256'])
        if 'pattern_support_id_transport' in contract:
            scope['pattern_support_id_transport']=contract['pattern_support_id_transport']
        if 'pattern_abstraction_guard' in contract:
            scope['pattern_abstraction_guard']=contract['pattern_abstraction_guard']
        if contract.get('pattern_policy',{}).get('discovery')==versions.GRADIENT_PATTERN_DISCOVERY_VERSION:
            scope.update(roles=['solver','reflection','pattern_gradient','pattern_cluster'],
                gradient_prompt_sha256=contract['gradient_prompt_sha256'],cluster_prompt_sha256=contract['pattern_prompt_sha256'],
                pattern_gradient_call_ceiling=contract['provider_bounds']['pattern_gradient_calls'],
                pattern_cluster_call_ceiling=contract['provider_bounds']['pattern_cluster_calls'])
    if "accounting_policy_path" in contract:
        scope["accounting"] = dict(policy_sha256=contract["accounting_policy_sha256"],
            total_authorization=40_000_000 if contract["identity"] in versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS else 30_000_000, task_sha256=contract["task_authorization_sha256"],
            ledger_directory=contract["token_ledger_directory"],
            validation_metadata_sha256=contract["validation_accounting_metadata_sha256"])
        if "continuation_authorization_sha256" in contract:
            scope["accounting"]["continuation_authorization_sha256"] = contract["continuation_authorization_sha256"]
    return scope


def validate_prep(root, prep, *, require_authorized=False):
    payload = read_json(prep / "prep.json")
    checksum = payload["startup_identity_sha256"]
    if payload.get("schema_version") != PREP_SCHEMA or canonical_sha256({k: v for k, v in payload.items() if k != "startup_identity_sha256"}) != checksum:
        raise SearchContractError("STARTUP_IDENTITY_MISMATCH")
    validate_frozen_source(root, payload["source_identity"])
    manifest = payload["manifest"]
    status = bound_preflight(root, manifest)
    if status["blockers"]:
        raise SearchContractError("HOLD_PRE_PROVIDER: " + ",".join(status["blockers"]))
    contract = read_json(root / manifest["execution_binding"]["path"])
    if execution_identity(root, contract) != payload["source_identity"]:
        raise SearchContractError("STARTUP_CLOSURE_MEMBERSHIP_MISMATCH")
    scope = payload["scope"]
    expected_scope = execution_scope(manifest, contract)
    if scope != expected_scope:
        raise SearchContractError("CANARY_SCOPE_MISMATCH")
    if require_authorized:
        auth = read_json(prep / "authorization.json")
        if (auth.get("explicit_user_authorized") is not True or auth.get("single_use") is not True or auth.get("consumed") is not False
                or auth.get("scope") != scope or auth.get("startup_identity_sha256") != checksum
                or (prep / "authorization_consumed.json").exists() or consumption_path(root, scope).exists()):
            raise SearchContractError("EXACT_SINGLE_USE_AUTHORIZATION_REQUIRED")
    return payload


async def execute_canary(root, prep, run_root):
    payload = validate_prep(root, prep, require_authorized=True)
    if "accounting" in payload["scope"]:
        from .autonomous_math import execute_search
        return await execute_search(root, prep, run_root, payload)
    if run_root.exists():
        raise SearchContractError("FRESH_RUN_ROOT_REQUIRED")
    scope = payload["scope"]
    # Exclusive consumption precedes credentials/client creation; crashes cannot
    # reopen this attempt. Fresh authorization is needed after any failure.
    global_marker = consumption_path(root, scope)
    global_marker.parent.mkdir(parents=True, exist_ok=True)
    with global_marker.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(scope=scope, startup_identity_sha256=payload["startup_identity_sha256"]), stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_write_json(prep / "authorization_consumed.json", read_json(global_marker))
    run_root.mkdir(parents=True)
    atomic_write_json(run_root / "consumed_authorization.json", {**read_json(prep / "authorization.json"), "consumed": True})
    atomic_write_json(run_root / "lifecycle.json", {"status": "RUNNING", "attempt_id": scope["attempt_id"]})
    broker = None
    try:
        from ...benchmarks.math_execution import MATHExecutionBinding
        from ...benchmarks.math import MATHBenchmarkAdapter
        from ...search.provider_runtime import RequestBroker, BenchmarkSolver, ReflectionProvider
        from ...provider_factory import ProviderClientFactory
        from openai import OpenAI
        contract = read_json(root / payload["manifest"]["execution_binding"]["path"])
        client = ProviderClientFactory.from_environment(provider_profile=contract["provider"], client_type=OpenAI,
            max_retries=contract["decoding"]["sdk_retries"], timeout=contract["decoding"]["timeout_seconds"])
        def transport(request):
            response = client.chat.completions.create(**request)
            return dict(text=response.choices[0].message.content, input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)
        broker = RequestBroker(contract=contract, transport=transport, arm=scope["arm"], seed=scope["seed"],
            ledger_writer=lambda row: append_jsonl(run_root / "ledger.jsonl", row),
            raw_writer=lambda row: append_jsonl(run_root / "provider_trace_private.jsonl", row))
        solver = BenchmarkSolver(MATHBenchmarkAdapter(), broker)
        composed = MATHExecutionBinding(root, contract).compose(arm=scope["arm"], seed=scope["seed"], solver=solver,
            reflection=ReflectionProvider(broker), pattern_provider=None, run_root=run_root)
        def observation(stage, payload):
            from dataclasses import is_dataclass, fields
            def plain(value):
                if is_dataclass(value):
                    return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
                if isinstance(value, dict):
                    return {str(k): plain(v) for k, v in value.items()}
                if isinstance(value, (set, frozenset)):
                    return [plain(v) for v in sorted(value)]
                if isinstance(value, (tuple, list)):
                    return [plain(v) for v in value]
                return value
            append_jsonl(run_root / "trajectory_private.jsonl", {"stage": stage, **plain(payload)})
        composed.execution_observer = observation
        composed.state.initialize()
        result = await composed.run(max_opportunities=contract["provider_bounds"]["max_opportunities"])
        final = composed.state.snapshot()
        # Freeze raw execution first in ignored run storage; sanitized reports
        # are a separate read-only audit, never a provider continuation.
        atomic_write_json(run_root / "final_team_private.json", {"prompts": final.member_prompts, "state_id": final.team_state_id})
        summary = dict(result=asdict(result), ledger=broker.usage, validation_calls=0, test_calls=0)
        atomic_write_json(run_root / "execution_summary.json", summary)
        if read_json(run_root / "execution_summary.json") != json.loads(json.dumps(summary)):
            raise SearchContractError("EXECUTION_PERSISTENCE_MISMATCH")
        atomic_write_json(run_root / "lifecycle.json", {"status": "EXECUTION_COMPLETE", "attempt_id": scope["attempt_id"]})
        atomic_write_json(run_root / "raw_evidence_inventory.json", inventory(run_root))
        return summary
    except BaseException:
        atomic_write_json(run_root / "lifecycle.json", {"status": "EXECUTION_ABORTED", "attempt_id": scope["attempt_id"],
            "provider_usage": broker.usage if broker else {"attempts": 0}})
        atomic_write_json(run_root / "raw_evidence_inventory.json", inventory(run_root))
        raise


def inventory(run_root):
    return {"files": [{"path": p.relative_to(run_root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                      for p in sorted(run_root.rglob("*")) if p.is_file() and p.name != "raw_evidence_inventory.json"]}


def preexecution_manifest(root, *, source_sha, frozen=True, binding_path=None, experiment_id="math_v2_pattern_memory_v1"):
    from ...benchmarks.math_execution import MATHExecutionBinding
    binding_path = binding_path or versions.MATH_EXECUTION_BINDING_PATH
    contract = read_json(root / binding_path)
    if contract["identity"] in {versions.MATH_AUTONOMOUS_EXECUTION_BINDING_VERSION, versions.MATH_DOMAIN_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION, *versions.MATH_LOW_COST_EXECUTION_BINDING_VERSIONS}:
        from ...benchmarks.legacy.math_domain_binding import execution_binding
        b = execution_binding(root, contract)
    else:
        b = MATHExecutionBinding(root, contract)
    if b.blockers():
        raise SearchContractError("MATH_BINDING_NOT_READY")
    method = b.method(contract.get('execution_arm','A1'))
    identity = execution_identity(root, contract)
    manifest = dict(schema_version="experiment_manifest_v2", experiment_id=experiment_id,
        method_family=method.method, method_identity=method.method, source_sha=source_sha,
        benchmark_id="math", benchmark_protocol_id=contract["benchmark_protocol_sha256"],
        dataset_manifest_identity=contract["canonical_manifest_sha256"], split_identity=contract["split_manifest_sha256"],
        seed=81, models=dict(solver="qwen3-8b", optimizer="qwen3.7-flash", solver_thinking=False),
        concurrency=dict(solver=1, optimizer=1), provider_policy=dict(identity="lwj", frozen=True),
        cache_policy=dict(identity=contract["cache_policy"], frozen=True),
        solver_output_interface_identity=contract["solver_output_interface"]["identity"],
        parser_identity=contract.get("payload_parser_identity",contract["solver_output_interface"]["parser_identity"]),
        access=dict(search_access="frozen_search", shadow_access="frozen_adaptive_gate", validation_access="not_authorized", test_access="sealed"),
        budget=dict(regime="saturation", saturation_identity=method.global_stop.identity,
            successful_provider_call_ceiling=contract["provider_bounds"]["successful_provider_calls"], transport_attempt_ceiling=contract["provider_bounds"]["transport_attempts"]),
        authorization=dict(real_api_authorized=False, attempt_id=None, authorization_identity=None, single_use=True),
        lifecycle=dict(status="PREEXECUTION_FROZEN" if frozen else "DRAFT", history=[dict(status="DRAFT", event="ZERO_API_BLOCKER_CLOSURE"),
            dict(status="PREEXECUTION_FROZEN" if frozen else "DRAFT", event="FROZEN_SOURCE_RECEIPT" if frozen else "AWAITING_SOURCE_COMMIT")]),
        hash_closure={k: identity[k] for k in ("scientific_source_hash", "governance_hash", "benchmark_contract_hash", "dataset_manifest_hash")},
        execution_binding=dict(identity=contract["identity"], path=binding_path, sha256=hashlib.sha256((root / binding_path).read_bytes()).hexdigest()),
        mechanism_config=method.mechanism_config)
    mapping = dict(search_engine_identity="search_engine", aggregation_identity="aggregation_policy", responsibility_identity="diagnosis_policy",
        transition_identity="transition_policy", adaptive_gate_identity="adaptive_gate_policy", pattern_identity="pattern_policy", memory_identity="memory_policy",
        search_acceptance_identity="search_acceptance_policy", evidence_identity="evidence_policy", feasibility_identity="feasibility_policy")
    manifest.update({k: getattr(method, field) for k, field in mapping.items()})
    if "initial_competence_binding" in contract:
        manifest["initial_competence_binding"] = contract["initial_competence_binding"]
    if "solver_decoding_policy" in contract:
        manifest["solver_decoding_policy"] = contract["solver_decoding_policy"]
    if "prediction_validity_policy" in contract:
        manifest["prediction_validity_policy"] = contract["prediction_validity_policy"]
    manifest["global_stop_identity"] = method.global_stop.identity
    for k in ("invalid_recovery_policy", "low_cost_protocol", "low_cost_subsets_sha256", "optimizer_generation_policy", "optimizer_amendment_authorization_sha256", "optimizer_nonthinking_evidence_policy", "layer1_search_policy", "candidate_contract_identity", "post_search_validation_policy", "pattern_support_id_transport", "pattern_abstraction_guard"):
        if k in contract and (k not in {'layer1_search_policy','candidate_contract_identity','post_search_validation_policy'} or contract['identity'] in {versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION}): manifest[k] = contract[k]
    manifest["preregistration_identity"] = canonical_sha256({k: v for k, v in manifest.items() if k not in {"lifecycle", "authorization"}})
    return manifest
