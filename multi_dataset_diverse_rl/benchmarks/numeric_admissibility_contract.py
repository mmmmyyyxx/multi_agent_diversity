"""Strict admissibility amendment; frozen parents supply data-only provenance.

The only scientific deltas are the Gradient prompt and numeric abstraction
guard. Fresh attempt/cache/user-scope identities do not inherit execution
authorization. Parent receipts cannot compose the current treatment.
"""
from copy import deepcopy
import json

from .. import legacy_current_contract_v21 as versions
from .. import versions as historical
from ..search.textual_gradients import POLICY, GRADIENT_PROMPT
from ..search.legacy.current_policy_v21 import CURRENT_POLICY_BUNDLE
from ..search.schemas import SearchContractError
from ..governance.provenance_receipts import verify_receipt_dependencies
from .data_freeze import file_hash
from .initial_condition_contract import initial_condition_provenance


def v5_policy():
    policy=deepcopy(POLICY)
    policy['gradient_policy']['abstraction_guard']=historical.PATTERN_NUMERIC_PROVENANCE_GUARD_VERSION
    return policy


def previous_policy():
    policy=deepcopy(POLICY)
    policy['gradient_policy'].update(prompt_identity=historical.GRADIENT_PROMPT_VERSION,
        abstraction_guard=historical.PATTERN_CONSTRAINT_ENTITY_GUARD_VERSION)
    return policy


def derive_numeric_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, amendment_path, amendment_sha256, gradient_prompt_path,
        gradient_prompt_sha256, user_scope_path=None, user_scope_sha256=None):
    if (not attempt.startswith('math_v2_1_gradient_pattern_A4_seed81_')
            or attempt in {parent['execution_attempt_id'],parent['cache_namespace']}
            or binding_path==parent['binding_path']):
        raise SearchContractError('CURRENT_NUMERIC_ADMISSIBILITY_IDENTITY_REUSED')
    c=deepcopy(parent)
    c.update(binding_path=binding_path,execution_attempt_id=attempt,cache_namespace=attempt,
        canary_attempt_id=attempt if c['execution_phase']=='canary' else None,
        numeric_parent_binding_path=parent_path,numeric_parent_binding_sha256=parent_sha256,
        numeric_admissibility_amendment_path=amendment_path,
        numeric_admissibility_amendment_sha256=amendment_sha256,
        gradient_prompt_path=gradient_prompt_path,gradient_prompt_sha256=gradient_prompt_sha256,
        pattern_abstraction_guard=historical.PATTERN_NUMERIC_PROVENANCE_GUARD_VERSION,
        pattern_policy=v5_policy(),continuation_authorization_sha256=amendment_sha256)
    if user_scope_path is not None:
        c.update(numeric_user_scope_path=user_scope_path,numeric_user_scope_sha256=user_scope_sha256,
            continuation_authorization_sha256=user_scope_sha256)
    return c


def validate_numeric_admissibility(binding):
    c=binding.contract
    def read_checked(path_key,hash_key):
        p=binding.path(c[path_key])
        if file_hash(p)!=c[hash_key]:raise SearchContractError('CURRENT_NUMERIC_ADMISSIBILITY_HASH_MISMATCH')
        return json.loads(p.read_bytes())
    a=read_checked('numeric_admissibility_amendment_path','numeric_admissibility_amendment_sha256')
    required=dict(schema_version='math_numeric_admissibility_amendment_v1',
        code_change_authorized=True,provider_output_admissibility_changed=True,
        core_method_changed=False,provider_call_budget=0,api_token_budget=0,
        pattern_policy=v5_policy(),guard_identity=historical.PATTERN_NUMERIC_PROVENANCE_GUARD_VERSION,
        gradient_prompt_identity=versions.GRADIENT_PROMPT_VERSION)
    if (any(a.get(k)!=v for k,v in required.items())
            or any(a.get(k) is not False for k in ('real_api_authorized','canary_authorized',
                'pilot_authorized','validation_authorized','test_authorized','push_authorized'))
            or not isinstance(a.get('user_request_sha256'),str) or len(a['user_request_sha256'])!=64
            or any(ch not in '0123456789abcdef' for ch in a['user_request_sha256'])):
        raise SearchContractError('CURRENT_NUMERIC_ADMISSIBILITY_AUTHORITY_MISMATCH')
    parent=read_checked('numeric_parent_binding_path','numeric_parent_binding_sha256')
    if (dict(binding_path=c['numeric_parent_binding_path'],binding_sha256=c['numeric_parent_binding_sha256']) not in a['parents']
            or parent['binding_path']!=c['numeric_parent_binding_path']
            or parent['pattern_policy']!=previous_policy()
            or parent['pattern_abstraction_guard']!=historical.PATTERN_CONSTRAINT_ENTITY_GUARD_VERSION
            or parent['initial_team_version']!=versions.MATH_INITIAL_TEAM_VERSION
            or parent['execution_arm']!='A4' or parent['seeds']!=[81]):
        raise SearchContractError('CURRENT_NUMERIC_ADMISSIBILITY_PARENT_MISMATCH')
    # Explicitly validate the initialization-only parent. Historical providers
    # and legacy runtime classes are never imported or composed here.
    from .legacy.math_gradient_pattern_binding_v21 import MATHGradientPatternBinding
    _,archived=initial_condition_provenance(MATHGradientPatternBinding(binding.root,parent))
    verify_receipt_dependencies(binding.root,parent,historical_initial_team=archived)
    verify_receipt_dependencies(binding.root,a,historical_initial_team=archived)
    prompt=read_checked('gradient_prompt_path','gradient_prompt_sha256')
    if (prompt!={'identity':versions.GRADIENT_PROMPT_VERSION,'prompt':GRADIENT_PROMPT}
            or c['gradient_prompt_path']!=a['gradient_prompt_path']
            or c['gradient_prompt_sha256']!=a['gradient_prompt_sha256']):
        raise SearchContractError('CURRENT_NUMERIC_ADMISSIBILITY_PROMPT_MISMATCH')
    if 'numeric_user_scope_path' in c:
        scope=read_checked('numeric_user_scope_path','numeric_user_scope_sha256')
        required_scope=dict(schema_version='current_gradient_pilot_user_scope_v2',
            arm='A4',seed=81,phase='pilot_search_only',attempt_id=c['execution_attempt_id'],
            user_authorized=True,provider_output_admissibility_changed=True,core_method_changed=False,
            parent_binding_sha256=c['numeric_parent_binding_sha256'],
            amendment_sha256=c['numeric_admissibility_amendment_sha256'],
            initial_team_version=versions.MATH_INITIAL_TEAM_VERSION,initial_memory_entries=0,
            validation_authorized=False,test_authorized=False,raw_diagnostic_authorized=False,
            llm_judge_authorized=False,push_authorized=False,canary_authorized=False,
            operational_fresh_retry_limit=0,exact_frozen_api_authorization_required=True,
            user_task_sha256=a['user_request_sha256'])
        if (any(scope.get(k)!=v for k,v in required_scope.items()) or c['execution_phase']!='pilot'):
            raise SearchContractError('CURRENT_NUMERIC_ADMISSIBILITY_USER_SCOPE_MISMATCH')
    expected=derive_numeric_contract(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
        parent_path=c['numeric_parent_binding_path'],parent_sha256=c['numeric_parent_binding_sha256'],
        amendment_path=c['numeric_admissibility_amendment_path'],amendment_sha256=c['numeric_admissibility_amendment_sha256'],
        gradient_prompt_path=c['gradient_prompt_path'],gradient_prompt_sha256=c['gradient_prompt_sha256'],
        user_scope_path=c.get('numeric_user_scope_path'),user_scope_sha256=c.get('numeric_user_scope_sha256'))
    if c!=expected:raise SearchContractError('CURRENT_NUMERIC_ADMISSIBILITY_FROZEN_CONTRACT_MISMATCH')
    # This is a data-only V5 receipt. Current execution eligibility is checked
    # by execution_binding; a historical receipt never selects a runtime guard.
    return archived
