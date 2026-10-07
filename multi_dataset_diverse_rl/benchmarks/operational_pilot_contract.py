"""Fresh operational Pilot bindings preserve every parent scientific setting."""
from copy import deepcopy
import json

from ..persistence.provider_receipts import POLICY
from ..search.schemas import SearchContractError
from .data_freeze import file_hash


def derive_operational_pilot(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authority_path, authority_sha256):
    if (parent['execution_phase'] != 'pilot' or parent['execution_arm'] != 'A4'
            or parent['seeds'] != [81] or attempt == parent['execution_attempt_id']
            or attempt == parent['cache_namespace'] or binding_path == parent['binding_path']
            or not attempt.startswith('math_v2_1_gradient_pattern_A4_seed81_pilot_attempt')):
        raise SearchContractError('OPERATIONAL_PILOT_FRESH_IDENTITY_REQUIRED')
    c = deepcopy(parent)
    c.update(execution_attempt_id=attempt, cache_namespace=attempt,
        binding_path=binding_path, runtime_persistence_policy=deepcopy(POLICY),
        operational_parent_binding_path=parent_path,
        operational_parent_binding_sha256=parent_sha256,
        operational_user_scope_path=authority_path,
        operational_user_scope_sha256=authority_sha256,
        continuation_authorization_sha256=authority_sha256)
    return c


def validate_operational_pilot(binding):
    c = binding.contract
    def checked(path_key, hash_key):
        p = binding.path(c[path_key])
        if file_hash(p) != c[hash_key]:
            raise SearchContractError('OPERATIONAL_PILOT_DEPENDENCY_HASH_MISMATCH')
        return json.loads(p.read_bytes())
    parent = checked('operational_parent_binding_path','operational_parent_binding_sha256')
    # Recursive validation is provenance only; no historical runtime is composed.
    from .legacy.math_gradient_pattern_binding_v21 import MATHGradientPatternBinding
    parent_binding = MATHGradientPatternBinding(binding.root,parent)
    if parent_binding.blockers():
        raise SearchContractError('OPERATIONAL_PILOT_PARENT_INVALID')
    expected = derive_operational_pilot(parent,attempt=c['execution_attempt_id'],
        binding_path=c['binding_path'],parent_path=c['operational_parent_binding_path'],
        parent_sha256=c['operational_parent_binding_sha256'],
        authority_path=c['operational_user_scope_path'],authority_sha256=c['operational_user_scope_sha256'])
    if c != expected:
        raise SearchContractError('OPERATIONAL_PILOT_SCIENTIFIC_SETTING_CHANGED')
    scope = checked('operational_user_scope_path','operational_user_scope_sha256')
    expected_scope = dict(schema_version='current_gradient_operational_pilot_user_scope_v1',
        arm='A4',seed=81,phase='pilot_search_only',attempt_id=c['execution_attempt_id'],
        user_authorized=True,scientific_method_changed=False,
        parent_binding_sha256=c['operational_parent_binding_sha256'],
        initial_team_version=c['initial_team_version'],initial_memory_entries=0,
        validation_authorized=False,test_authorized=False,raw_diagnostic_authorized=False,
        llm_judge_authorized=False,canary_authorized=False,push_authorized=True,
        operational_retry_policy='FRESH_AFTER_PROVEN_OPERATIONAL_INVALIDITY_ONLY',
        exact_frozen_api_authorization_required=True,cumulative_token_ceiling=40_000_000,
        runtime_persistence_policy=POLICY)
    task_hash = scope.get('user_task_sha256')
    if (any(scope.get(k) != v for k,v in expected_scope.items())
            or not isinstance(task_hash,str) or len(task_hash) != 64
            or any(ch not in '0123456789abcdef' for ch in task_hash)):
        raise SearchContractError('OPERATIONAL_PILOT_USER_SCOPE_MISMATCH')
    return parent
