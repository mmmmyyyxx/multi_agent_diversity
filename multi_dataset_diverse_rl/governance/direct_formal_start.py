"""Explicit V2.7 direct-start prerequisite; no substitute Canary evidence."""
import hashlib
from ..search.schemas import SearchContractError
from ..persistence.durable_io import read_json

IDENTITY = 'USER_SELECTED_V27_DIRECT_FORMAL_OFFLINE_PREREQUISITE_V1'
from ..current_contract import CURRENT_METHOD_VERSION as METHOD

def require_policy(value):
    keys={'identity','offline_verification_path','offline_verification_sha256','user_instruction_sha256'}
    if (not isinstance(value,dict) or set(value)!=keys or value['identity']!=IDENTITY
            or not isinstance(value['offline_verification_path'],str)
            or not value['offline_verification_path'].startswith('experiments/protocols/')
            or any(not isinstance(value[k],str) or len(value[k])!=64
                or any(c not in '0123456789abcdef' for c in value[k])
                for k in ('offline_verification_sha256','user_instruction_sha256'))):
        raise SearchContractError('DIRECT_FORMAL_PREREQUISITE_NOT_FROZEN')

def verify(root,contract,payload):
    from .postfreeze_validation import local_path
    from .unified_execution import validate_frozen_source
    value=contract['canary_prerequisite'];require_policy(value)
    if (contract['method_identity']!=METHOD or contract['execution_phase']!='formal'
            or payload['scope']['phase']!='formal_search_only'
            or payload['scope']['method']!=METHOD
            or payload['scope']['attempt_id']!=contract['execution_attempt_id']):
        raise SearchContractError('DIRECT_FORMAL_CURRENT_SCOPE_MISMATCH')
    path=local_path(root,value['offline_verification_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=value['offline_verification_sha256']:
        raise SearchContractError('DIRECT_FORMAL_VERIFICATION_HASH_MISMATCH')
    record=read_json(path)
    if (record.get('identity')!=IDENTITY or record.get('method')!=METHOD
            or record.get('attempt_id')!=contract['execution_attempt_id']
            or record.get('user_instruction_sha256')!=value['user_instruction_sha256']
            or record.get('user_selected_direct_formal') is not True
            or record.get('canary_execution')!='NOT_RUN_CURRENT_VERSION'
            or record.get('new_real_api_calls')!=0
            or record.get('validation_access')!='NOT_AUTHORIZED'
            or record.get('test_access')!='SEALED'
            or record.get('current_suite',{}).get('failed')!=0
            or record.get('current_suite',{}).get('errors')!=0
            or record.get('current_suite',{}).get('passed',0)<743
            or record.get('network_attempts')!=0
            or record.get('compileall')!='PASS'
            or record.get('governance')!='PASS'
            or record.get('deterministic_output_feedback_smokes')!='PASS'):
        raise SearchContractError('DIRECT_FORMAL_OFFLINE_VERIFICATION_HOLD')
    validate_frozen_source(root,payload['source_identity'])
    return dict(identity=IDENTITY,current_canary='NOT_RUN',offline_verification_sha256=value['offline_verification_sha256'],
        user_instruction_sha256=value['user_instruction_sha256'],efficacy_gate=False)
