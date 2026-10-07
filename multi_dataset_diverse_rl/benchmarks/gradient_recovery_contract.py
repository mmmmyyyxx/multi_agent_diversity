"""Exact opt-in recovery derivation; prior authorities remain immutable receipts."""
from copy import deepcopy
import json

from ..search.gradient_recovery import POLICY
from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.schemas import SearchContractError
from .data_freeze import file_hash


def derive_gradient_recovery(parent, *, attempt, binding_path, parent_path,
        parent_sha256, amendment_path, amendment_sha256, user_scope_path, user_scope_sha256):
    CURRENT_POLICY_BUNDLE.validate_contract(parent)
    if (not attempt.startswith('math_v2_1_gradient_pattern_A4_seed81_')
            or attempt in (parent['execution_attempt_id'],parent['cache_namespace'])
            or binding_path==parent['binding_path']):
        raise SearchContractError('CURRENT_GRADIENT_RECOVERY_FRESH_IDENTITY_REQUIRED')
    c=deepcopy(parent)
    c.update(execution_attempt_id=attempt,cache_namespace=attempt,binding_path=binding_path,
        canary_attempt_id=attempt if parent['execution_phase']=='canary' else None,
        gradient_recovery_parent_binding_path=parent_path,gradient_recovery_parent_binding_sha256=parent_sha256,
        gradient_recovery_amendment_path=amendment_path,gradient_recovery_amendment_sha256=amendment_sha256,
        gradient_recovery_user_scope_path=user_scope_path,gradient_recovery_user_scope_sha256=user_scope_sha256,
        continuation_authorization_sha256=user_scope_sha256,gradient_recovery_policy=deepcopy(POLICY))
    bounds=c['provider_bounds'];extra=2*bounds['pattern_gradient_calls']
    bounds['pattern_gradient_calls']*=3
    bounds['pattern_calls']+=extra;bounds['successful_provider_calls']+=extra
    bounds['transport_attempts']=bounds['successful_provider_calls']*(c['decoding']['transport_retries']+1)
    if 'gross_reservation_upper_bound' in bounds:
        bounds['gross_reservation_upper_bound']=bounds['transport_attempts']*40_000_000
    return c


def validate_gradient_recovery(binding):
    c=binding.contract
    def checked(path,hash_key):
        p=binding.path(c[path])
        if file_hash(p)!=c[hash_key]:raise SearchContractError('CURRENT_GRADIENT_RECOVERY_HASH_MISMATCH')
        return json.loads(p.read_bytes())
    parent=checked('gradient_recovery_parent_binding_path','gradient_recovery_parent_binding_sha256')
    amendment=checked('gradient_recovery_amendment_path','gradient_recovery_amendment_sha256')
    scope=checked('gradient_recovery_user_scope_path','gradient_recovery_user_scope_sha256')
    task=amendment.get('user_task_sha256')
    if (amendment.get('schema_version')!='gradient_contract_recovery_amendment_v1'
            or amendment.get('policy')!=POLICY or amendment.get('code_change_authorized') is not True
            or amendment.get('accepted_output_contract_changed') is not False
            or amendment.get('prompt_changed') is not False or amendment.get('guard_changed') is not False
            or amendment.get('real_api_authorized') is not False
            or not isinstance(task,str) or len(task)!=64 or any(x not in '0123456789abcdef' for x in task)
            or dict(binding_path=c['gradient_recovery_parent_binding_path'],
                binding_sha256=c['gradient_recovery_parent_binding_sha256']) not in amendment.get('parents',[])):
        raise SearchContractError('CURRENT_GRADIENT_RECOVERY_AUTHORITY_MISMATCH')
    required=dict(schema_version='gradient_contract_recovery_user_scope_v1',user_task_sha256=task,
        attempt_id=c['execution_attempt_id'],arm='A4',seed=81,phase=c['execution_phase'],
        parent_binding_sha256=c['gradient_recovery_parent_binding_sha256'],
        recovery_policy=POLICY,validation_authorized=False,test_authorized=False,
        raw_diagnostic_authorized=False,llm_judge_authorized=False,
        operational_fresh_retry_limit='UNBOUNDED_STRICT_OPERATIONAL_INVALID_ONLY',
        other_scientific_changes_authorized=False)
    required.update(user_authorized=True,exact_frozen_api_authorization_required=True,canary_authorized=False)
    if any(scope.get(k)!=v for k,v in required.items()):
        raise SearchContractError('CURRENT_GRADIENT_RECOVERY_USER_SCOPE_MISMATCH')
    if scope.get('real_attempt_authorized') is not (c['execution_phase']=='pilot' and 'offline' not in c['execution_attempt_id']):
        raise SearchContractError('CURRENT_GRADIENT_RECOVERY_USER_SCOPE_MISMATCH')
    from .math_gradient_pattern_binding import MATHGradientPatternBinding
    blockers=MATHGradientPatternBinding(binding.root,parent).blockers()
    if blockers:
        if blockers[0].startswith('CURRENT_DATA_'):raise SearchContractError(blockers[0])
        raise SearchContractError('CURRENT_GRADIENT_RECOVERY_PARENT_INVALID')
    expected=derive_gradient_recovery(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
        parent_path=c['gradient_recovery_parent_binding_path'],parent_sha256=c['gradient_recovery_parent_binding_sha256'],
        amendment_path=c['gradient_recovery_amendment_path'],amendment_sha256=c['gradient_recovery_amendment_sha256'],
        user_scope_path=c['gradient_recovery_user_scope_path'],user_scope_sha256=c['gradient_recovery_user_scope_sha256'])
    if c!=expected:raise SearchContractError('CURRENT_GRADIENT_RECOVERY_SCIENTIFIC_SETTING_CHANGED')
    CURRENT_POLICY_BUNDLE.validate_contract(c)
    return parent
