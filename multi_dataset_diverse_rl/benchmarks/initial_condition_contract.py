"""Initial-condition amendment only; historical authority never authorizes a run."""
from copy import deepcopy
import json

from ..current_contract import MATH_INITIAL_TEAM_VERSION
from ..governance.provenance_receipts import verify_receipt_dependencies
from ..search.schemas import SearchContractError
from .data_freeze import file_hash


def derive_identical_initial_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, amendment_path, amendment_sha256, amendment):
    if (not attempt.startswith('math_v2_1_gradient_pattern_A4_seed81_')
            or attempt == parent['execution_attempt_id'] or attempt == parent['cache_namespace']
            or binding_path == parent['binding_path']):
        raise SearchContractError('CURRENT_INITIAL_CONDITION_IDENTITY_REUSED')
    c = deepcopy(parent)
    c.update(binding_path=binding_path, execution_attempt_id=attempt, cache_namespace=attempt,
        canary_attempt_id=attempt if parent['execution_phase']=='canary' else None,
        initial_team_version=MATH_INITIAL_TEAM_VERSION,
        initial_team_sha256=amendment['ordered_team_sha256'],
        initial_team_artifact_sha256=amendment['initial_team_artifact_sha256'],
        initial_condition_parent_binding_path=parent_path,
        initial_condition_parent_binding_sha256=parent_sha256,
        initial_condition_amendment_path=amendment_path,
        initial_condition_amendment_sha256=amendment_sha256,
        continuation_authorization_sha256=amendment_sha256)
    c['post_search_validation_policy']['initial_team']=MATH_INITIAL_TEAM_VERSION
    # The old Pilot authority remains inside the immutable parent receipt only.
    c.pop('pilot_execution_authorization_path', None)
    c.pop('pilot_execution_authorization_sha256', None)
    return c


def initial_condition_provenance(binding):
    """Return checked old bytes for provenance and a strictly derived new binding."""
    c=binding.contract
    if 'initial_condition_amendment_path' not in c:
        return c, None
    path=binding.path(c['initial_condition_amendment_path'])
    if file_hash(path)!=c['initial_condition_amendment_sha256']:
        raise SearchContractError('CURRENT_INITIAL_CONDITION_AMENDMENT_HASH_MISMATCH')
    a=json.loads(path.read_bytes())
    required=dict(schema_version='math_identical_initial_condition_amendment_v1',
        code_change_authorized=True, scientific_method_changed=False, initial_condition_changed=True,
        initial_team_version=MATH_INITIAL_TEAM_VERSION, provider_call_budget=0, api_token_budget=0)
    if (any(a.get(k)!=v for k,v in required.items())
            or any(a.get(k) is not False for k in ('real_api_authorized','canary_authorized',
                'pilot_authorized','validation_authorized','test_authorized','push_authorized'))
            or not isinstance(a.get('user_request_sha256'),str)
            or len(a['user_request_sha256'])!=64
            or any(ch not in '0123456789abcdef' for ch in a['user_request_sha256'])
            or a.get('initial_team_path')!=c['initial_team_path']):
        raise SearchContractError('CURRENT_INITIAL_CONDITION_AUTHORITY_MISMATCH')
    parent_path=binding.path(c['initial_condition_parent_binding_path'])
    parent=json.loads(parent_path.read_bytes())
    if (file_hash(parent_path)!=c['initial_condition_parent_binding_sha256']
            or dict(binding_path=c['initial_condition_parent_binding_path'],
                binding_sha256=c['initial_condition_parent_binding_sha256']) not in a['parents']
            or parent['initial_team_version']!='MATH_GENERIC_TEAM_SEED_V1_1'
            or parent['initial_team_path']!=a['initial_team_path']
            or parent['initial_team_artifact_sha256']!=a['historical_team_artifact_sha256']):
        raise SearchContractError('CURRENT_INITIAL_CONDITION_PARENT_MISMATCH')
    historical=(a['initial_team_path'], a['historical_team_artifact_sha256'], a['archived_team_path'])
    verify_receipt_dependencies(binding.root,a,historical_initial_team=historical)
    verify_receipt_dependencies(binding.root,parent,historical_initial_team=historical)
    archived=json.loads(binding.path(a['archived_team_path']).read_bytes())
    if (archived['team_version']!=parent['initial_team_version']
            or archived['ordered_team_sha256']!=parent['initial_team_sha256']):
        raise SearchContractError('CURRENT_INITIAL_CONDITION_ARCHIVE_MISMATCH')
    expected=derive_identical_initial_contract(parent,attempt=c['execution_attempt_id'],
        binding_path=c['binding_path'],parent_path=c['initial_condition_parent_binding_path'],
        parent_sha256=c['initial_condition_parent_binding_sha256'],
        amendment_path=c['initial_condition_amendment_path'],
        amendment_sha256=c['initial_condition_amendment_sha256'],amendment=a)
    if c!=expected:
        raise SearchContractError('CURRENT_INITIAL_CONDITION_FROZEN_CONTRACT_MISMATCH')
    return parent,historical
