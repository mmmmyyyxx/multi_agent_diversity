"""Fresh Canary-only Pattern-aware + rolling Memory execution contract."""
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import json

from ... import versions
from ...search.pattern_layer1 import PatternLayer1Config
from ...search.pattern_responsibility import POLICY, ABSTRACTION_GUARD_VERSIONS
from ...search.rolling_risk_memory import POLICY as MEMORY_POLICY
from ...search.action_memory import LIMITS
from ...search.schemas import SearchContractError
from ..data_freeze import file_hash
from ..math_memory_binding import MATHMemoryBinding


def derive_pattern_contract(parent,*,attempt,binding_path,parent_path,parent_sha256,
        approval_path,approval_sha256,prompt_path,prompt_sha256,memory_freeze_path,memory_freeze_sha256,support_id_transport=None,abstraction_guard=None):
    if parent['identity']!=versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION:
        raise SearchContractError('PATTERN_PARENT_BINDING_INVALID')
    c=deepcopy(parent)
    c.update(identity=versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION,execution_arm='A4',
        execution_attempt_id=attempt,canary_attempt_id=attempt,cache_namespace=attempt,binding_path=binding_path,
        pattern_parent_binding_path=parent_path,pattern_parent_binding_sha256=parent_sha256,
        pattern_amendment_authorization_path=approval_path,pattern_amendment_authorization_sha256=approval_sha256,
        continuation_authorization_sha256=approval_sha256,pattern_prompt_path=prompt_path,pattern_prompt_sha256=prompt_sha256,
        rolling_memory_freeze_path=memory_freeze_path,rolling_memory_freeze_sha256=memory_freeze_sha256,
        memory_policy_identity=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION,
        shared_risk_policy=deepcopy(MEMORY_POLICY),pattern_policy=deepcopy(POLICY),
        layer1_search_policy=asdict(PatternLayer1Config()),optimizer_input_schema=versions.PATTERN_OPTIMIZER_INPUT_VERSION,
        panel_evidence_policy=versions.PATTERN_CONDITIONED_EVIDENCE_VERSION)
    c['provider_bounds']['pattern_calls']=1
    c['provider_bounds']['successful_provider_calls']=c['provider_bounds']['solver_calls']+c['provider_bounds']['reflection_calls']+1
    c['provider_bounds']['transport_attempts']=21*c['provider_bounds']['successful_provider_calls']
    if support_id_transport is not None:
        if support_id_transport!=versions.PATTERN_SUPPORT_ID_ALIAS_VERSION:raise SearchContractError('PATTERN_ID_TRANSPORT_INVALID')
        c['pattern_support_id_transport']=support_id_transport
    if abstraction_guard is not None:
        if abstraction_guard not in ABSTRACTION_GUARD_VERSIONS:raise SearchContractError('PATTERN_ABSTRACTION_GUARD_NOT_BOUND')
        c['pattern_abstraction_guard']=abstraction_guard
    return c


def derive_specific_content_approval(parent, *, parent_path, parent_sha256, request_sha256):
    """Version the code amendment; this receipt grants no real API scope."""
    if parent.get('pattern_abstraction_guard') != versions.PATTERN_ABSTRACTION_GUARD_VERSION:
        raise SearchContractError('PATTERN_GUARD_PARENT_AUTHORITY_INVALID')
    approval = deepcopy(parent)
    approval['pattern_abstraction_guard'] = versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
    approval['push_authorized'] = False
    approval['real_api_authorized'] = False
    approval['operational_repair_authority'] = 'CURRENT_USER_SPECIFIC_CONTENT_GUARD_REQUEST'
    approval['guard_amendment'] = dict(schema_version='pattern_specific_content_amendment_v1',
        parent_authority_path=parent_path, parent_authority_sha256=parent_sha256,
        user_request_sha256=request_sha256, code_change_authorized=True,
        real_api_authorized=False, validation_authorized=False, test_authorized=False)
    return approval


class MATHPatternBinding(MATHMemoryBinding):
    def method(self,arm):
        if arm!='A4':raise SearchContractError('PATTERN_MEMORY_CANARY_REQUIRES_A4')
        # The immutable Memory parent selects A3; toggle only the fresh treatment.
        parent=json.loads(self.path(self.contract['pattern_parent_binding_path']).read_bytes())
        base=MATHMemoryBinding(self.root,parent).method('A3')
        c=self.contract
        provider_identity=dict(provider=c['provider'],model=c['models']['pattern'],
            prompt=c['pattern_prompt_sha256'],policy=c['optimizer_generation_policy'],discovery=c['pattern_policy'])
        transport={}
        if 'pattern_support_id_transport' in c:
            transport={'pattern_support_id_transport':c['pattern_support_id_transport']}
            provider_identity.update(transport)
        if 'pattern_abstraction_guard' in c:
            transport['pattern_abstraction_guard']=c['pattern_abstraction_guard']
            provider_identity['pattern_abstraction_guard']=c['pattern_abstraction_guard']
        provider_binding=hashlib.sha256(json.dumps(provider_identity,
            sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return replace(base,pattern_policy=versions.PATTERN_AWARE_DISCOVERY_VERSION,
            memory_policy=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION,
            evidence_policy=versions.PATTERN_CONDITIONED_EVIDENCE_VERSION,
            mechanism_config=dict(memory=deepcopy(LIMITS),shared_risk_policy=deepcopy(MEMORY_POLICY),
                optimizer_input_schema=versions.PATTERN_OPTIMIZER_INPUT_VERSION,
                panel_policy=versions.PATTERN_CONDITIONED_EVIDENCE_VERSION,
                pattern_policy=deepcopy(POLICY),pattern_provider_binding=provider_binding,**transport))

    def blockers(self):
        c=self.contract
        try:
            for path,sha in [('pattern_parent_binding_path','pattern_parent_binding_sha256'),
                ('pattern_amendment_authorization_path','pattern_amendment_authorization_sha256'),
                ('pattern_prompt_path','pattern_prompt_sha256'),('rolling_memory_freeze_path','rolling_memory_freeze_sha256')]:
                if file_hash(self.path(c[path]))!=c[sha]:return ('PATTERN_DEPENDENCY_HASH_MISMATCH',)
            parent=json.loads(self.path(c['pattern_parent_binding_path']).read_bytes())
            if MATHMemoryBinding(self.root,parent).blockers():return ('PATTERN_PARENT_NOT_CONFORMANT',)
            approval=json.loads(self.path(c['pattern_amendment_authorization_path']).read_bytes())
            if c.get('pattern_abstraction_guard') == versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION:
                amendment = approval.get('guard_amendment', {})
                authority_path = self.path(amendment['parent_authority_path'])
                if file_hash(authority_path) != amendment['parent_authority_sha256']:
                    return ('PATTERN_GUARD_PARENT_AUTHORITY_HASH_MISMATCH',)
                request_hash = amendment['user_request_sha256']
                if (not isinstance(request_hash, str) or len(request_hash) != 64
                        or any(x not in '0123456789abcdef' for x in request_hash)):
                    return ('PATTERN_GUARD_AMENDMENT_AUTHORITY_MISMATCH',)
                expected_approval = derive_specific_content_approval(json.loads(authority_path.read_bytes()),
                    parent_path=amendment['parent_authority_path'],
                    parent_sha256=amendment['parent_authority_sha256'], request_sha256=request_hash)
                if approval != expected_approval:
                    return ('PATTERN_GUARD_AMENDMENT_AUTHORITY_MISMATCH',)
            elif 'guard_amendment' in approval:
                return ('PATTERN_GUARD_AMENDMENT_AUTHORITY_MISMATCH',)
            if approval.get('pattern_support_id_transport')!=c.get('pattern_support_id_transport'):
                return ('PATTERN_ID_TRANSPORT_AUTHORITY_MISMATCH',)
            if approval.get('pattern_abstraction_guard')!=c.get('pattern_abstraction_guard'):
                return ('PATTERN_ABSTRACTION_GUARD_AUTHORITY_MISMATCH',)
            freeze=json.loads(self.path(c['rolling_memory_freeze_path']).read_bytes())
            if (approval.get('explicit_human_approval') is not True or approval.get('canary_only') is not True
                    or approval.get('total_accounting_authorization')!=40000000
                    or approval.get('pattern_policy')!=POLICY or approval.get('shared_risk_policy')!=MEMORY_POLICY
                    or approval.get('memory_limits')!=LIMITS or approval.get('layer1_search_policy')!=asdict(PatternLayer1Config())
                    or approval.get('solver_decoding_policy')!=parent['solver_decoding_policy']
                    or approval.get('optimizer_generation_policy')!=parent['optimizer_generation_policy']
                    or freeze.get('memory_identity')!=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION
                    or approval.get('pilot_authorized') is not False or approval.get('validation_authorized') is not False
                    or approval.get('test_model_calls')!=0
                    or approval.get('attachment_sha256')!='978153c3868b1f302eb1a0b4cc246fdd20f050ef593dd563d995cd2dc75a32b0'):return ('PATTERN_AMENDMENT_AUTHORITY_MISMATCH',)
            expected=derive_pattern_contract(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
                parent_path=c['pattern_parent_binding_path'],parent_sha256=c['pattern_parent_binding_sha256'],
                approval_path=c['pattern_amendment_authorization_path'],approval_sha256=c['pattern_amendment_authorization_sha256'],
                prompt_path=c['pattern_prompt_path'],prompt_sha256=c['pattern_prompt_sha256'],
                memory_freeze_path=c['rolling_memory_freeze_path'],memory_freeze_sha256=c['rolling_memory_freeze_sha256'],support_id_transport=c.get('pattern_support_id_transport'),abstraction_guard=c.get('pattern_abstraction_guard'))
            if c!=expected or not c['execution_attempt_id'].startswith('math_v2_1_pattern_A4_seed81_canary_attempt'):
                return ('PATTERN_FROZEN_CONTRACT_MISMATCH',)
            return ()
        except (KeyError,TypeError,ValueError,OSError,SearchContractError):return ('PATTERN_BINDING_INVALID',)
