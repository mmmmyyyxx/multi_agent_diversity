"""Fresh Canary-only Memory-ON amendment over the immutable Layer1 V3 binding."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

from .. import versions
from ..search.action_memory import LIMITS
from ..search.layer1_memory import MemoryLayer1Config
from ..search.schemas import SearchContractError
from .data_freeze import file_hash
from .math_layer1_binding import MATHLayer1Binding


AUTHORITY_SHA256='e07b8174c9a7529c82a83c2004891debb9d1ba9d6616b77b6def090d2bf8dee9'


def derive_memory_contract(parent, *, attempt, binding_path, parent_path, parent_sha256,
                           approval_path, approval_sha256):
    if parent['identity']!=versions.MATH_LAYER1_EXECUTION_BINDING_VERSION:
        raise SearchContractError('MEMORY_PARENT_BINDING_INVALID')
    c=deepcopy(parent)
    c.update(identity=versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,
        binding_path=binding_path,execution_phase='canary',execution_arm='A3',
        execution_attempt_id=attempt,canary_attempt_id=attempt,cache_namespace=attempt,
        memory_parent_binding_path=parent_path,memory_parent_binding_sha256=parent_sha256,
        memory_amendment_authorization_path=approval_path,memory_amendment_authorization_sha256=approval_sha256,
        continuation_authorization_sha256=approval_sha256,
        layer1_search_policy=asdict(MemoryLayer1Config()),memory_policy_identity=versions.STRUCTURED_ACTION_MEMORY_VERSION,
        memory_limits=deepcopy(LIMITS),optimizer_input_schema=versions.LAYER1_INPUT_SCHEMA_VERSION,
        panel_evidence_policy=versions.LAYER1_ANCHOR_EVIDENCE_VERSION)
    c['provider_bounds']['pattern_calls']=0
    c['provider_bounds']['successful_provider_calls']=c['provider_bounds']['solver_calls']+c['provider_bounds']['reflection_calls']
    c['provider_bounds']['transport_attempts']=21*c['provider_bounds']['successful_provider_calls']
    return c


class MATHMemoryBinding(MATHLayer1Binding):
    def method(self,arm):
        if arm!='A3':raise SearchContractError('MEMORY_ON_CANARY_REQUIRES_A3')
        method=super().method(arm)
        return replace(method,search_engine=versions.LAYER1_FEEDBACK_SEARCH_VERSION,
            memory_policy=versions.STRUCTURED_ACTION_MEMORY_VERSION,
            evidence_policy=versions.LAYER1_ANCHOR_EVIDENCE_VERSION,
            mechanism_config={'memory':deepcopy(LIMITS),
                'optimizer_input_schema':versions.LAYER1_INPUT_SCHEMA_VERSION,
                'panel_policy':versions.LAYER1_ANCHOR_EVIDENCE_VERSION})

    def blockers(self):
        c=self.contract
        try:
            parent_path=self.path(c['memory_parent_binding_path'])
            if file_hash(parent_path)!=c['memory_parent_binding_sha256']:return ('MEMORY_PARENT_HASH_MISMATCH',)
            parent=json.loads(parent_path.read_bytes())
            if MATHLayer1Binding(self.root,parent).blockers():return ('MEMORY_PARENT_NOT_CONFORMANT',)
            ap=self.path(c['memory_amendment_authorization_path'])
            if file_hash(ap)!=c['memory_amendment_authorization_sha256']:return ('MEMORY_AUTHORITY_HASH_MISMATCH',)
            approval=json.loads(ap.read_bytes())
            if (approval.get('attachment_sha256')!=AUTHORITY_SHA256
                    or approval.get('explicit_human_approval') is not True
                    or approval.get('canary_only') is not True
                    or approval.get('pilot_authorized') is not False
                    or approval.get('validation_authorized') is not False
                    or approval.get('total_accounting_authorization')!=40000000
                    or approval.get('layer1_search_policy')!=asdict(MemoryLayer1Config())
                    or approval.get('memory_limits')!=LIMITS
                    or approval.get('memory_policy_identity')!=versions.STRUCTURED_ACTION_MEMORY_VERSION
                    or approval.get('solver_decoding_policy')!=parent['solver_decoding_policy']
                    or approval.get('optimizer_generation_policy')!=parent['optimizer_generation_policy']):
                return ('MEMORY_AMENDMENT_AUTHORITY_MISMATCH',)
            expected=derive_memory_contract(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
                parent_path=c['memory_parent_binding_path'],parent_sha256=c['memory_parent_binding_sha256'],
                approval_path=c['memory_amendment_authorization_path'],approval_sha256=c['memory_amendment_authorization_sha256'])
            if (c!=expected or not c['execution_attempt_id'].startswith('math_v2_1_memory_A3_seed81_canary_attempt')
                    or parent['execution_phase']!='canary'):
                return ('MEMORY_FROZEN_CONTRACT_MISMATCH',)
            return ()
        except (KeyError,ValueError,TypeError,OSError,SearchContractError):return ('MEMORY_BINDING_INVALID',)
