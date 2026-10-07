"""Frozen Unified canary admission; readiness never grants API authorization."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .source_identity import build_unified_source_identity, hash_scope
from .repository import validate_manifest_v2
from .startup_identity import canonical_sha256
from ..persistence.durable_io import atomic_write_json, append_jsonl, read_json
from ..search.schemas import SearchContractError
from ..search.current_policy import CURRENT_POLICY_BUNDLE


PREP_SCHEMA = "unified_canary_prep_v1"


def consumption_path(root, scope):
    return root / "runs/unified_authorization_consumption" / (canonical_sha256(scope) + ".json")


def bound_preflight(root, manifest):
    from ..benchmarks.math_domain_binding import execution_binding
    ref=manifest.get('execution_binding',{})
    errors=validate_manifest_v2(root,manifest)
    if manifest.get('lifecycle',{}).get('status')!='PREEXECUTION_FROZEN':errors.append('PREEXECUTION_NOT_FROZEN')
    path=(root/ref.get('path','')).resolve()
    if (not path.is_relative_to(root.resolve()) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()!=ref.get('sha256')):
        return dict(gate='HOLD',blockers=errors+['EXECUTION_BINDING_HASH_MISMATCH'],provider_attempts=0)
    try:
        binding=execution_binding(root,read_json(path));errors.extend(binding.blockers())
        if ('numeric_calibration_amendment_path' in binding.contract
                and binding.contract['execution_phase']=='pilot'
                and read_json(root/binding.contract['numeric_calibration_amendment_path']).get('fresh_pilot_condition_met') is not True):
            errors.append('ATTEMPT4_CONFIRMED_STRONG_NUMERIC_LEAKAGE')
        method=binding.method('A4')
        expected=preexecution_manifest(root,source_sha=manifest.get('source_sha'),frozen=False,
            binding_path=ref['path'],experiment_id=manifest.get('experiment_id'))
        ignored={'preregistration_identity','lifecycle','authorization','hash_closure'}
        if any(manifest.get(k)!=v for k,v in expected.items() if k not in ignored):errors.append('MANIFEST_EXECUTION_BINDING_MISMATCH')
        CURRENT_POLICY_BUNDLE.validate_method(method)
    except SearchContractError as error:
        errors.append(str(error));binding=None
    if manifest.get('authorization',{}).get('real_api_authorized') is not False:errors.append('PREEXECUTION_MANIFEST_CANNOT_AUTHORIZE')
    payload={k:v for k,v in manifest.items() if k not in {'preregistration_identity','lifecycle','authorization'}}
    if manifest.get('preregistration_identity')!=canonical_sha256(payload):errors.append('PREREGISTRATION_IDENTITY_MISMATCH')
    if not errors:
        identity=execution_identity(root,binding.contract)
        if manifest['hash_closure']!={k:identity[k] for k in manifest['hash_closure']}:errors.append('MANIFEST_SOURCE_CLOSURE_MISMATCH')
        else:
            try:verify_source_commit(root,manifest['source_sha'],identity)
            except (SearchContractError,subprocess.CalledProcessError):errors.append('EXECUTION_SOURCE_COMMIT_BYTE_MISMATCH')
    phase=binding.contract['execution_phase'] if binding is not None else 'canary'
    return dict(gate='HOLD' if errors else ('PILOT_READY_NOT_AUTHORIZED' if phase=='pilot' else 'CANARY_READY_NOT_AUTHORIZED'),blockers=errors,
        benchmark_id='math',readiness_identity='MATH_RUNTIME_READINESS_V1',provider_attempts=0,
        validation_calls=0,test_calls=0,real_api_authorized=False,binding_sha256=ref.get('sha256'),ready_for_authorization=not errors)



def execution_identity(root, contract):
    CURRENT_POLICY_BUNDLE.validate_contract(contract)
    # Input metadata and all import/bootstrap dependencies are explicit. The
    # canonical raw source remains private and has its separate manifest hash.
    configs = [contract["split_directory"] + "/math.json", contract["initial_team_path"],
               contract["pattern_prompt_path"], contract["binding_path"]]
    configs.extend(contract[k] for k in ("parent_binding_path", "accounting_policy_path", "validation_accounting_metadata_path", "verify_settings_path", "amendment_parent_binding_path", "low_cost_subsets_path", "optimizer_amendment_authorization_path", "layer1_parent_binding_path", "layer1_amendment_authorization_path", "memory_parent_binding_path", "memory_amendment_authorization_path", "pattern_parent_binding_path", "pattern_amendment_authorization_path", "rolling_memory_freeze_path") if k in contract)
    configs.extend(contract[k] for k in ('gradient_prompt_path','gradient_parent_binding_path') if k in contract)
    configs.extend(contract[k] for k in ('pilot_parent_binding_path','pilot_execution_authorization_path') if k in contract)
    provenance_contract = contract
    if 'numeric_calibration_amendment_path' in contract:
        from ..benchmarks.numeric_calibration_contract import validate_numeric_calibration
        from ..benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
        provenance_contract=validate_numeric_calibration(MATHGradientPatternBinding(root,contract))
        configs.extend(contract[k] for k in ('numeric_calibration_parent_binding_path',
            'numeric_calibration_amendment_path'))
    if 'partition_completion_amendment_path' in contract:
        from ..benchmarks.partition_completion_contract import validate_partition_completion
        from ..benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
        provenance_contract=validate_partition_completion(MATHGradientPatternBinding(root,provenance_contract))
        configs.extend(contract[k] for k in ('partition_completion_parent_binding_path',
            'partition_completion_user_scope_path','partition_completion_amendment_path'))
    if 'operational_user_scope_path' in provenance_contract:
        from ..benchmarks.operational_pilot_contract import validate_operational_pilot
        from ..benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
        provenance_contract=validate_operational_pilot(MATHGradientPatternBinding(root,provenance_contract))
        configs.extend(contract[k] for k in ('operational_parent_binding_path','operational_user_scope_path'))
    if 'numeric_admissibility_amendment_path' in provenance_contract:
        from ..benchmarks.numeric_admissibility_contract import validate_numeric_admissibility
        from ..benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
        from .provenance_receipts import verify_receipt_dependencies
        historical=validate_numeric_admissibility(MATHGradientPatternBinding(root,provenance_contract))
        configs.extend(contract[k] for k in ('numeric_admissibility_amendment_path','numeric_parent_binding_path','numeric_user_scope_path') if k in contract)
        configs.extend(verify_receipt_dependencies(root,contract,historical_initial_team=historical))
    elif 'initial_condition_amendment_path' in contract:
        from ..benchmarks.initial_condition_contract import initial_condition_provenance
        from ..benchmarks.math_domain_binding import execution_binding
        from .provenance_receipts import verify_receipt_dependencies
        _, historical=initial_condition_provenance(execution_binding(root,contract))
        amendment=read_json(root/contract['initial_condition_amendment_path'])
        configs.extend(contract[k] for k in ('initial_condition_amendment_path','initial_condition_parent_binding_path'))
        configs.extend(verify_receipt_dependencies(root,amendment,historical_initial_team=historical))
    parent=read_json(root/contract['gradient_parent_binding_path'])
    configs.extend(parent[k] for k in ('pattern_amendment_authorization_path','pattern_prompt_path'))
    approval=read_json(root/parent['pattern_amendment_authorization_path'])
    configs.append(approval['guard_amendment']['parent_authority_path'])
    identity = build_unified_source_identity(root, root / contract["canonical_root"] / "manifests/math.json", [root / p for p in configs])
    files = [root / r["path"] for s in identity["scopes"].values() for r in s["files"]]
    files.append(root / contract["split_directory"] / "math.ids.jsonl")
    # CLI imports are operational dependencies; dynamic composition is pinned
    # by the benchmark and search scopes as well.
    from .source_identity import local_imports
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


def prepare_canary(root, manifest, *, destination, arm="A4", seed=81):
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
    CURRENT_POLICY_BUNDLE.validate_contract(contract)
    scope=dict(attempt_id=contract['execution_attempt_id'],arm=contract['execution_arm'],seed=81,
        phase=contract['canary_phase'] if contract['execution_phase']=='canary' else contract['stop_policy'],
        source_sha=manifest['source_sha'],preregistration_identity=manifest['preregistration_identity'],
        binding_sha256=manifest['execution_binding']['sha256'],models=contract['models'],provider=contract['provider'],
        roles=['solver','reflection','pattern_gradient','pattern_cluster'],
        successful_provider_call_ceiling=contract['provider_bounds']['successful_provider_calls'],
        transport_attempt_ceiling=contract['provider_bounds']['transport_attempts'],validation_calls=0,test_calls=0,
        initial_memory_entries=0,gradient_prompt_sha256=contract['gradient_prompt_sha256'],
        cluster_prompt_sha256=contract['pattern_prompt_sha256'],
        pattern_gradient_call_ceiling=contract['provider_bounds']['pattern_gradient_calls'],
        pattern_cluster_call_ceiling=contract['provider_bounds']['pattern_cluster_calls'])
    for key in ('solver_decoding_policy','prediction_validity_policy','invalid_recovery_policy','low_cost_protocol',
        'low_cost_subsets_sha256','optimizer_generation_policy','optimizer_amendment_authorization_sha256',
        'optimizer_nonthinking_evidence_policy','layer1_search_policy','candidate_contract_identity','cache_policy',
        'post_search_validation_policy','layer1_amendment_authorization_sha256','memory_policy_identity','memory_limits',
        'optimizer_input_schema','panel_evidence_policy','memory_amendment_authorization_sha256','pattern_policy',
        'shared_risk_policy','pattern_amendment_authorization_sha256','pattern_support_id_transport','pattern_abstraction_guard'):
        scope[key]=contract[key]
    scope['accounting']=dict(policy_sha256=contract['accounting_policy_sha256'],total_authorization=40_000_000,
        task_sha256=contract['task_authorization_sha256'],ledger_directory=contract['token_ledger_directory'],
        validation_metadata_sha256=contract['validation_accounting_metadata_sha256'],
        continuation_authorization_sha256=contract['continuation_authorization_sha256'])
    if contract['execution_phase']=='pilot':
        scope.update(execution_phase='pilot',user_scope_sha256=contract.get('numeric_user_scope_sha256',contract.get('initial_condition_amendment_sha256',contract.get('pilot_execution_authorization_sha256'))),
            search_only_scope=contract['search_only_scope'],pilot_observation_policy=contract['pilot_observation_policy'],
            max_opportunities=contract['provider_bounds']['max_opportunities'],
            solver_call_ceiling=contract['provider_bounds']['solver_calls'],
            reflection_call_ceiling=contract['provider_bounds']['reflection_calls'])
    if 'initial_condition_amendment_sha256' in contract:
        scope['initial_condition']={k:contract[k] for k in ('initial_team_version',
            'initial_team_artifact_sha256','initial_team_sha256','initial_condition_amendment_sha256')}
    if 'numeric_admissibility_amendment_sha256' in contract:
        scope['numeric_admissibility']={k:contract[k] for k in ('numeric_parent_binding_sha256',
            'numeric_admissibility_amendment_sha256','gradient_prompt_sha256','pattern_abstraction_guard')}
    if 'numeric_calibration_amendment_sha256' in contract:
        scope['numeric_calibration']={k:contract[k] for k in ('numeric_calibration_parent_binding_sha256',
            'numeric_calibration_amendment_sha256','pattern_abstraction_guard')}
    if 'operational_user_scope_path' in contract:
        scope['runtime_persistence_policy']=contract['runtime_persistence_policy']
        scope['operational_user_scope_sha256']=contract['operational_user_scope_sha256']
        scope['operational_parent_binding_sha256']=contract['operational_parent_binding_sha256']
    if 'partition_completion_policy' in contract:
        scope['partition_completion_policy']=contract['partition_completion_policy']
        scope['partition_completion_amendment_sha256']=contract['partition_completion_amendment_sha256']
        scope['partition_completion_parent_binding_sha256']=contract['partition_completion_parent_binding_sha256']
        scope['user_scope_sha256']=contract['partition_completion_user_scope_sha256']
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
    payload=validate_prep(root,prep,require_authorized=True)
    from ..benchmarks.math_domain_binding import execution_binding
    contract=read_json(root/payload['manifest']['execution_binding']['path'])
    execution_binding(root,contract) # current eligibility precedes credentials and ledger mutation
    from .autonomous_math import execute_search
    return await execute_search(root,prep,run_root,payload)



def inventory(run_root):
    return {"files": [{"path": p.relative_to(run_root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                      for p in sorted(run_root.rglob("*")) if p.is_file() and p.name != "raw_evidence_inventory.json"]}


def preexecution_manifest(root, *, source_sha, frozen=True, binding_path=None, experiment_id="math_v2_pattern_memory_v1"):
    from ..benchmarks.math_domain_binding import execution_binding
    binding_path=binding_path or 'experiments/execution_bindings/math_v2_1_gradient_pattern_offline_profile_v4.json'
    contract=read_json(root/binding_path)
    b=execution_binding(root,contract)
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
    if 'runtime_persistence_policy' in contract:
        manifest['runtime_persistence_policy']=contract['runtime_persistence_policy']
    if 'partition_completion_policy' in contract:
        manifest['partition_completion_policy']=contract['partition_completion_policy']
    for k in ("invalid_recovery_policy", "low_cost_protocol", "low_cost_subsets_sha256", "optimizer_generation_policy", "optimizer_amendment_authorization_sha256", "optimizer_nonthinking_evidence_policy", "layer1_search_policy", "candidate_contract_identity", "post_search_validation_policy", "pattern_support_id_transport", "pattern_abstraction_guard"):
        if k in contract:manifest[k]=contract[k]
    manifest["preregistration_identity"] = canonical_sha256({k: v for k, v in manifest.items() if k not in {"lifecycle", "authorization"}})
    return manifest
