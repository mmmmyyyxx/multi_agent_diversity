"""Versioned gradient discovery amendment; its code receipt grants no API scope."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

from .. import versions
from ..search.pattern_layer1 import GradientPatternLayer1Config
from ..search.textual_gradients import POLICY, GRADIENT_PROMPT, CLUSTER_PROMPT
from ..search.schemas import SearchContractError
from .data_freeze import file_hash
from .math_pattern_binding import MATHPatternBinding


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


class MATHGradientPatternBinding(MATHPatternBinding):
    def method(self,arm):
        base=super().method(arm)
        c=self.contract
        identity=dict(provider=c['provider'],model=c['models']['pattern'],
            gradient_prompt=c['gradient_prompt_sha256'],cluster_prompt=c['pattern_prompt_sha256'],
            generation_policy=c['optimizer_generation_policy'],discovery=c['pattern_policy'],
            support_id_transport=c['pattern_support_id_transport'],abstraction_guard=c['pattern_abstraction_guard'])
        binding=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        config={**base.mechanism_config,'pattern_policy':deepcopy(POLICY),
            'pattern_provider_binding':binding,'optimizer_input_schema':versions.GRADIENT_OPTIMIZER_INPUT_VERSION,
            'panel_policy':versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION}
        return replace(base,pattern_policy=versions.GRADIENT_PATTERN_DISCOVERY_VERSION,
            evidence_policy=versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION,mechanism_config=config,
            global_stop=replace(base.global_stop,emergency_max_provider_calls=c['provider_bounds']['successful_provider_calls']))

    def blockers(self):
        c=self.contract
        try:
            for path,sha in [('gradient_parent_binding_path','gradient_parent_binding_sha256'),
                ('pattern_amendment_authorization_path','pattern_amendment_authorization_sha256'),
                ('gradient_prompt_path','gradient_prompt_sha256'),('pattern_prompt_path','pattern_prompt_sha256')]:
                if file_hash(self.path(c[path]))!=c[sha]:return ('GRADIENT_PATTERN_DEPENDENCY_HASH_MISMATCH',)
            parent=json.loads(self.path(c['gradient_parent_binding_path']).read_bytes())
            if MATHPatternBinding(self.root,parent).blockers():return ('GRADIENT_PATTERN_PARENT_NOT_CONFORMANT',)
            approval=json.loads(self.path(c['pattern_amendment_authorization_path']).read_bytes())
            if (approval.get('schema_version')!='gradient_pattern_code_amendment_v1'
                    or approval.get('code_change_authorized') is not True
                    or any(approval.get(k) is not False for k in ('real_api_authorized','canary_authorized',
                        'pilot_authorized','validation_authorized','test_authorized','push_authorized'))
                    or approval.get('provider_call_budget')!=0 or approval.get('api_token_budget')!=0
                    or approval.get('parent_binding_sha256')!=c['gradient_parent_binding_sha256']
                    or approval.get('pattern_policy')!=POLICY
                    or approval.get('layer1_search_policy')!=asdict(GradientPatternLayer1Config())):
                return ('GRADIENT_PATTERN_CODE_AUTHORITY_MISMATCH',)
            request_hash=approval.get('user_request_sha256')
            if not isinstance(request_hash,str) or len(request_hash)!=64 or any(x not in '0123456789abcdef' for x in request_hash):
                return ('GRADIENT_PATTERN_CODE_AUTHORITY_MISMATCH',)
            if (json.loads(self.path(c['gradient_prompt_path']).read_bytes())!={'identity':versions.PER_EXAMPLE_GRADIENT_VERSION,'prompt':GRADIENT_PROMPT}
                    or json.loads(self.path(c['pattern_prompt_path']).read_bytes())!={'identity':versions.GRADIENT_PATTERN_DISCOVERY_VERSION,'prompt':CLUSTER_PROMPT}):
                return ('GRADIENT_PATTERN_PROMPT_CONTRACT_MISMATCH',)
            expected=derive_gradient_contract(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
                parent_path=c['gradient_parent_binding_path'],parent_sha256=c['gradient_parent_binding_sha256'],
                approval_path=c['pattern_amendment_authorization_path'],approval_sha256=c['pattern_amendment_authorization_sha256'],
                gradient_prompt_path=c['gradient_prompt_path'],gradient_prompt_sha256=c['gradient_prompt_sha256'],
                cluster_prompt_path=c['pattern_prompt_path'],cluster_prompt_sha256=c['pattern_prompt_sha256'])
            if (c!=expected or not c['execution_attempt_id'].startswith('math_v2_1_gradient_pattern_A4_seed81_')
                    or c['execution_attempt_id']==parent['execution_attempt_id'] or c['cache_namespace']==parent['cache_namespace']):
                return ('GRADIENT_PATTERN_FROZEN_CONTRACT_MISMATCH',)
            return ()
        except (KeyError,TypeError,ValueError,OSError,SearchContractError):return ('GRADIENT_PATTERN_BINDING_INVALID',)
