"""Exact fresh Pilot amendment: deterministic known omissions become unassigned."""
from copy import deepcopy
import json
from .data_freeze import file_hash
from ..search.partition_completion import IDENTITY, POLICY
from ..search.schemas import SearchContractError


def derive_partition_completion(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authority_path, authority_sha256, amendment_path, amendment_sha256):
    if (parent['execution_phase'] != 'pilot' or parent['execution_arm'] != 'A4'
            or parent['seeds'] != [81] or attempt in (parent['execution_attempt_id'],parent['cache_namespace'])
            or binding_path == parent['binding_path']
            or not attempt.startswith('math_v2_1_gradient_pattern_A4_seed81_pilot_attempt')):
        raise SearchContractError('PARTITION_COMPLETION_FRESH_IDENTITY_REQUIRED')
    c=deepcopy(parent)
    c.update(execution_attempt_id=attempt,cache_namespace=attempt,binding_path=binding_path,
        partition_completion_policy=IDENTITY,partition_completion_parent_binding_path=parent_path,
        partition_completion_parent_binding_sha256=parent_sha256,
        partition_completion_user_scope_path=authority_path,partition_completion_user_scope_sha256=authority_sha256,
        partition_completion_amendment_path=amendment_path,partition_completion_amendment_sha256=amendment_sha256,
        continuation_authorization_sha256=authority_sha256)
    return c


def validate_partition_completion(binding):
    c=binding.contract
    def checked(path_key,hash_key):
        p=binding.path(c[path_key])
        if file_hash(p)!=c[hash_key]:raise SearchContractError('PARTITION_COMPLETION_DEPENDENCY_HASH_MISMATCH')
        return json.loads(p.read_bytes())
    parent=checked('partition_completion_parent_binding_path','partition_completion_parent_binding_sha256')
    from .math_gradient_pattern_binding import MATHGradientPatternBinding
    if MATHGradientPatternBinding(binding.root,parent).blockers():
        raise SearchContractError('PARTITION_COMPLETION_PARENT_INVALID')
    expected=derive_partition_completion(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
        parent_path=c['partition_completion_parent_binding_path'],parent_sha256=c['partition_completion_parent_binding_sha256'],
        authority_path=c['partition_completion_user_scope_path'],authority_sha256=c['partition_completion_user_scope_sha256'],
        amendment_path=c['partition_completion_amendment_path'],amendment_sha256=c['partition_completion_amendment_sha256'])
    if c!=expected:raise SearchContractError('PARTITION_COMPLETION_SCIENTIFIC_SETTING_CHANGED')
    amendment=checked('partition_completion_amendment_path','partition_completion_amendment_sha256')
    if (amendment.get('schema_version')!='gradient_partition_completion_amendment_v1'
            or amendment.get('policy')!=POLICY or amendment.get('code_change_authorized') is not True
            or amendment.get('scientific_method_changed') is not False
            or amendment.get('compliance_behavior_changed') is not True
            or amendment.get('parent_binding_sha256')!=c['partition_completion_parent_binding_sha256']
            or amendment.get('provider_call_budget')!=0 or amendment.get('real_api_authorized') is not False):
        raise SearchContractError('PARTITION_COMPLETION_AMENDMENT_INVALID')
    scope=checked('partition_completion_user_scope_path','partition_completion_user_scope_sha256')
    required=dict(schema_version='current_gradient_partition_completion_pilot_user_scope_v1',
        arm='A4',seed=81,phase='pilot_search_only',attempt_id=c['execution_attempt_id'],user_authorized=True,
        scientific_method_changed=False,compliance_behavior_changed=True,partition_completion_policy=IDENTITY,
        parent_binding_sha256=c['partition_completion_parent_binding_sha256'],
        initial_team_version=c['initial_team_version'],initial_memory_entries=0,
        validation_authorized=False,test_authorized=False,raw_diagnostic_authorized=False,
        llm_judge_authorized=False,canary_authorized=False,push_authorized=True,
        operational_retry_policy='FRESH_AFTER_PROVEN_OPERATIONAL_INVALIDITY_ONLY',
        exact_frozen_api_authorization_required=True,cumulative_token_ceiling=40_000_000)
    task=scope.get('user_task_sha256')
    if (any(scope.get(k)!=v for k,v in required.items()) or not isinstance(task,str)
            or len(task)!=64 or any(x not in '0123456789abcdef' for x in task)
            or amendment.get('user_task_sha256')!=task):
        raise SearchContractError('PARTITION_COMPLETION_USER_SCOPE_MISMATCH')
    return parent
