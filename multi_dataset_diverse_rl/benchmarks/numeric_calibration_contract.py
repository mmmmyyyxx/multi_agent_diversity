"""V6 calibration receipt: exact parent settings, unchanged provider prompts."""
from copy import deepcopy
import json
from .. import legacy_current_contract_v21 as identities
from ..search.schemas import SearchContractError
from ..search.legacy.current_policy_v21 import CURRENT_POLICY_BUNDLE
from ..search.numeric_provenance import STRONG_REASONS, WARNING_REASONS
from .data_freeze import file_hash

POLICY=dict(identity=identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION,
    strong_reasons=list(STRONG_REASONS),warning_reasons=list(WARNING_REASONS),
    bare_numeric_coincidence_hard_reject=False,gradient_prompt_changed=False,
    regeneration=False,core_method_changed=False,bounded_heuristic=True)


def derive_numeric_calibration(parent, *, attempt, binding_path, parent_path,
        parent_sha256, amendment_path, amendment_sha256):
    if (not attempt.startswith('math_v2_1_gradient_pattern_A4_seed81_')
            or attempt in (parent['execution_attempt_id'],parent['cache_namespace'])
            or binding_path==parent['binding_path']):
        raise SearchContractError('CURRENT_NUMERIC_CALIBRATION_FRESH_IDENTITY_REQUIRED')
    c=deepcopy(parent)
    c.update(execution_attempt_id=attempt,cache_namespace=attempt,binding_path=binding_path,
        canary_attempt_id=attempt if parent['execution_phase']=='canary' else None,
        numeric_calibration_parent_binding_path=parent_path,
        numeric_calibration_parent_binding_sha256=parent_sha256,
        numeric_calibration_amendment_path=amendment_path,
        numeric_calibration_amendment_sha256=amendment_sha256,
        continuation_authorization_sha256=amendment_sha256,
        pattern_abstraction_guard=identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)
    c['pattern_policy']['gradient_policy']['abstraction_guard']=identities.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
    return c


def validate_numeric_calibration(binding):
    c=binding.contract
    def checked(path_key,hash_key):
        p=binding.path(c[path_key])
        if file_hash(p)!=c[hash_key]:raise SearchContractError('CURRENT_NUMERIC_CALIBRATION_HASH_MISMATCH')
        return json.loads(p.read_bytes())
    parent=checked('numeric_calibration_parent_binding_path','numeric_calibration_parent_binding_sha256')
    amendment=checked('numeric_calibration_amendment_path','numeric_calibration_amendment_sha256')
    task=amendment.get('user_task_sha256')
    if (amendment.get('schema_version')!='math_numeric_provenance_calibration_amendment_v1'
            or amendment.get('policy')!=POLICY or amendment.get('code_change_authorized') is not True
            or amendment.get('provider_output_admissibility_changed') is not True
            or amendment.get('core_method_changed') is not False
            or any(amendment.get(k) is not False for k in ('real_api_authorized',
                'validation_authorized','test_authorized','llm_judge_authorized'))
            or amendment.get('provider_call_budget')!=0
            or not isinstance(task,str) or len(task)!=64 or any(x not in '0123456789abcdef' for x in task)
            or dict(binding_path=c['numeric_calibration_parent_binding_path'],
                binding_sha256=c['numeric_calibration_parent_binding_sha256']) not in amendment.get('parents',[])):
        raise SearchContractError('CURRENT_NUMERIC_CALIBRATION_AUTHORITY_MISMATCH')
    # Explicit receipt validation only. No historical runtime or old API scope
    # is imported into current execution, and the original V5 guard is preserved.
    from .legacy.math_gradient_pattern_binding_v21 import MATHGradientPatternBinding
    blockers=MATHGradientPatternBinding(binding.root,parent).blockers()
    if blockers:
        if blockers[0].startswith('CURRENT_DATA_'):raise SearchContractError(blockers[0])
        raise SearchContractError('CURRENT_NUMERIC_CALIBRATION_PARENT_INVALID')
    expected=derive_numeric_calibration(parent,attempt=c['execution_attempt_id'],binding_path=c['binding_path'],
        parent_path=c['numeric_calibration_parent_binding_path'],parent_sha256=c['numeric_calibration_parent_binding_sha256'],
        amendment_path=c['numeric_calibration_amendment_path'],amendment_sha256=c['numeric_calibration_amendment_sha256'])
    if c!=expected:raise SearchContractError('CURRENT_NUMERIC_CALIBRATION_SCIENTIFIC_SETTING_CHANGED')
    CURRENT_POLICY_BUNDLE.validate_contract(c)
    return parent
