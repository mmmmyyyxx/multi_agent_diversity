"""Data-only amendment over frozen provenance; no inherited execution behavior."""
from copy import deepcopy
from dataclasses import asdict
from .. import versions
from ..search.current_layer1 import GradientPatternLayer1Config
from ..search.textual_gradients import POLICY
from ..search.schemas import SearchContractError

def derive_gradient_contract(parent, *, attempt, binding_path, parent_path, parent_sha256,
        approval_path, approval_sha256, gradient_prompt_path, gradient_prompt_sha256,
        cluster_prompt_path, cluster_prompt_sha256):
    if (parent.get('identity')!=versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION
            or parent.get('pattern_policy',{}).get('discovery')!=versions.PATTERN_AWARE_DISCOVERY_VERSION
            or parent.get('pattern_abstraction_guard')!=versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION):
        raise SearchContractError('GRADIENT_PATTERN_PARENT_INVALID')
    c=deepcopy(parent)
    c.update(binding_path=binding_path,execution_attempt_id=attempt,canary_attempt_id=attempt,cache_namespace=attempt,
        gradient_parent_binding_path=parent_path,gradient_parent_binding_sha256=parent_sha256,
        pattern_amendment_authorization_path=approval_path,pattern_amendment_authorization_sha256=approval_sha256,
        continuation_authorization_sha256=approval_sha256,
        gradient_prompt_path=gradient_prompt_path,gradient_prompt_sha256=gradient_prompt_sha256,
        pattern_prompt_path=cluster_prompt_path,pattern_prompt_sha256=cluster_prompt_sha256,
        pattern_policy=deepcopy(POLICY),layer1_search_policy=asdict(GradientPatternLayer1Config()),
        optimizer_input_schema=versions.GRADIENT_OPTIMIZER_INPUT_VERSION,
        panel_evidence_policy=versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION)
    # Every frozen Optimize row may be wrong; a fresh cluster follows each opportunity.
    # Existing Canary parent stops after exactly one complete opportunity.
    if parent['execution_phase']!='canary':raise SearchContractError('GRADIENT_PATTERN_CEILING_PHASE_UNBOUND')
    bounds=c['provider_bounds']
    bounds.update(pattern_gradient_calls=c['initial_competence_binding']['count'],pattern_cluster_calls=1)
    bounds['pattern_calls']=bounds['pattern_gradient_calls']+bounds['pattern_cluster_calls']
    bounds['successful_provider_calls']=bounds['solver_calls']+bounds['reflection_calls']+bounds['pattern_calls']
    bounds['transport_attempts']=(c['decoding']['transport_retries']+1)*bounds['successful_provider_calls']
    return c
