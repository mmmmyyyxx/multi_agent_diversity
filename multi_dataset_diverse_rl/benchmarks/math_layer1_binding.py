"""Fresh Layer1-only amendment over an immutable low-cost V2 parent binding."""
from copy import deepcopy
from dataclasses import asdict,replace
import json

from .. import versions
from ..search.schemas import SearchContractError
from ..search.layer1_responsibility import Layer1Config
from .data_freeze import file_hash
from .math_low_cost_binding import MATHLowCostBinding
from .math_solver_decoding import solver_decoding_contract
from .math_optimizer_generation import optimizer_generation_contract


def derive_layer1_contract(parent, *, phase,attempt,binding_path,parent_path,parent_sha256,
        approval_path,approval_sha256,metadata_path,metadata_sha256):
    if parent['identity']!=versions.MATH_LOW_COST_OPTIMIZER_EXECUTION_BINDING_VERSION or phase not in ('canary','pilot'):
        raise SearchContractError('LAYER1_PARENT_OR_PHASE_INVALID')
    c=deepcopy(parent)
    count=12 if phase=='canary' else 60
    c.update(identity=versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,binding_path=binding_path,
        execution_phase=phase,execution_attempt_id=attempt,canary_attempt_id=attempt,cache_namespace=attempt,
        layer1_parent_binding_path=parent_path,layer1_parent_binding_sha256=parent_sha256,
        layer1_amendment_authorization_path=approval_path,layer1_amendment_authorization_sha256=approval_sha256,
        continuation_authorization_sha256=approval_sha256,
        optimizer_amendment_authorization_path=approval_path,optimizer_amendment_authorization_sha256=approval_sha256,
        validation_accounting_metadata_path=metadata_path,validation_accounting_metadata_sha256=metadata_sha256,
        layer1_search_policy=asdict(Layer1Config()),candidate_contract_identity=versions.SEMANTIC_MUTABLE_CONTRACT_VERSION,
        cache_policy=versions.SOLVER_MEMBER_LANE_CACHE_VERSION,
        solver_decoding_policy=solver_decoding_contract(versions.MATH_SOLVER_DECODING_POLICY_V2_VERSION),
        optimizer_generation_policy=optimizer_generation_contract(versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION))
    c['initial_competence_binding'].update(count=count,
        support_identity=c['low_cost_protocol']['membership_hashes']['canary_optimize' if phase=='canary' else 'pilot_optimize'])
    from .math_execution import provider_bounds
    bounds=provider_bounds(optimize_count=count,shadow_count=40)
    maximum=1 if phase=='canary' else bounds['max_opportunities']*2*(count+1)
    per_op=36+4*12+2*count+40
    bounds.update(max_opportunities=maximum,max_proposals_per_opportunity=6,solver_per_opportunity=per_op,
        solver_calls=4*(5*(count+40)+maximum*per_op),reflection_calls=6*maximum,pattern_calls=maximum)
    bounds['successful_provider_calls']=bounds['solver_calls']+bounds['reflection_calls']+bounds['pattern_calls']
    bounds['transport_attempts']=21*bounds['successful_provider_calls']
    c['provider_bounds']=bounds
    from ..governance.math_paired_validation import validation_policy
    c['post_search_validation_policy']=validation_policy(c)
    return c


class MATHLayer1Binding(MATHLowCostBinding):
    def method(self,arm):
        return replace(super().method(arm),search_engine=versions.LAYER1_RESPONSIBILITY_SEARCH_VERSION,
            search_acceptance_policy='layer1_local_guidance_team_admission_v1')

    def blockers(self):
        c=self.contract
        try:
            parent_path=self.path(c['layer1_parent_binding_path'])
            if file_hash(parent_path)!=c['layer1_parent_binding_sha256']:return ('LAYER1_PARENT_HASH_MISMATCH',)
            parent=json.loads(parent_path.read_bytes())
            if MATHLowCostBinding(self.root,parent).blockers():return ('LAYER1_PARENT_NOT_CONFORMANT',)
            approval_path=self.path(c['layer1_amendment_authorization_path'])
            if file_hash(approval_path)!=c['layer1_amendment_authorization_sha256']:return ('LAYER1_APPROVAL_HASH_MISMATCH',)
            approval=json.loads(approval_path.read_bytes())
            if (approval.get('explicit_human_approval') is not True or approval.get('total_accounting_authorization')!=40000000
                    or approval.get('layer1_search_policy')!=asdict(Layer1Config())
                    or approval.get('solver_decoding_policy')!=c['solver_decoding_policy']
                    or approval.get('policy')!=c['optimizer_generation_policy']
                    or approval.get('pattern_real_calls_A1')!=0 or approval.get('test_model_calls')!=0
                    or approval.get('attachment_sha256')!='76630550500117515c4eb999362e0c7ce0b70a6ca6d1d01ea7dfb7ce48a6dc68'):
                return ('LAYER1_AMENDMENT_AUTHORITY_MISMATCH',)
            expected=derive_layer1_contract(parent,phase=c['execution_phase'],attempt=c['execution_attempt_id'],
                binding_path=c['binding_path'],parent_path=c['layer1_parent_binding_path'],parent_sha256=c['layer1_parent_binding_sha256'],
                approval_path=c['layer1_amendment_authorization_path'],approval_sha256=c['layer1_amendment_authorization_sha256'],
                metadata_path=c['validation_accounting_metadata_path'],metadata_sha256=c['validation_accounting_metadata_sha256'])
            if c!=expected or not c['execution_attempt_id'].startswith('math_v2_1_layer1_A1_seed81_'+c['execution_phase']+'_attempt'):
                return ('LAYER1_FROZEN_CONTRACT_MISMATCH',)
            metadata_path=self.path(c['validation_accounting_metadata_path'])
            if file_hash(metadata_path)!=c['validation_accounting_metadata_sha256']:return ('LAYER1_METADATA_HASH_MISMATCH',)
            metadata=json.loads(metadata_path.read_bytes())
            old=json.loads(self.path(parent['validation_accounting_metadata_path']).read_bytes())
            from .math_accounting_prep import solver_request
            from ..governance.token_accounting import serialized_request
            # All changed wire fields have content-independent length. This
            # exact affine projection needs no new held-out raw-data access.
            delta=len(serialized_request(solver_request(c,'','')))-len(serialized_request(solver_request(parent,'','')))
            old['solver_decoding_policy']=c['solver_decoding_policy']
            for r in old['examples']:r['blank_prompt_serialized_request_bytes']+=delta
            if metadata!=old:return ('LAYER1_VALIDATION_RESERVE_METADATA_MISMATCH',)
            return ()
        except (KeyError,ValueError,TypeError,OSError,SearchContractError):
            return ('LAYER1_BINDING_INVALID',)
