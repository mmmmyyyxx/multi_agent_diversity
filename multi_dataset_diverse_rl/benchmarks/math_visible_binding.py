"""Fresh V6 feedback binding on the current graph; never mutate a frozen parent.

This construction supplies no authorization. A future governed freeze must bind
new scope, source, prompt hashes and accounting metadata before real dispatch.
"""
from copy import deepcopy
from dataclasses import asdict
import json

from .. import versions
from ..search.current_layer1 import CurrentLayer1Config
from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.schemas import SearchContractError
from ..search.textual_gradients import pattern_policy_for_trajectory, VISIBLE_GRADIENT_PROMPT
from .data_freeze import file_hash
from .math_gradient_pattern_binding import MATHGradientPatternBinding
from .math_v21_interface import v6_interface_contract
from .math_visible_trajectory import trajectory_policy


def visible_gradient_prompt_artifact():
    return dict(identity=versions.GRADIENT_VISIBLE_PROMPT_VERSION, prompt=VISIBLE_GRADIENT_PROMPT)


def derive_visible_accounting_metadata(parent_metadata, parent, visible):
    """Rebind existing request lengths without reading any held-out content."""
    from .math_v21_interface import interface_for_contract, solver_user_content
    from .math_solver_decoding import generation_request_fields
    from ..governance.token_accounting import serialized_request
    from .math_visible_trajectory import frozen_trajectory_policy
    frozen_trajectory_policy(visible)
    unchanged = ('decoding', 'solver_decoding_policy', 'prediction_validity_policy',
        'invalid_recovery_policy', 'low_cost_protocol')
    if (parent_metadata.get('solver_output_interface') != parent['solver_output_interface']
            or any(parent.get(k) != visible.get(k) or parent_metadata.get(k) != parent.get(k)
                for k in unchanged)):
        raise SearchContractError('MATH_VISIBLE_ACCOUNTING_PARENT_MISMATCH')
    def empty_request(contract):
        return dict(model=contract['models']['solver'], **generation_request_fields(contract, 'solver'),
            messages=[dict(role='system', content=interface_for_contract(contract)[0]),
                dict(role='user', content=solver_user_content(contract, '', ''))])
    delta = len(serialized_request(empty_request(visible))) - len(serialized_request(empty_request(parent)))
    metadata = deepcopy(parent_metadata)
    metadata['solver_output_interface'] = visible['solver_output_interface']
    for row in metadata['examples']:
        before = row['blank_prompt_serialized_request_bytes']
        if type(before) is not int or before <= 0 or before + delta <= 0:
            raise SearchContractError('MATH_VISIBLE_ACCOUNTING_LENGTH_INVALID')
        row['blank_prompt_serialized_request_bytes'] += delta
    metadata['visible_interface_length_rebinding'] = dict(
        policy='EXACT_SERIALIZED_EMPTY_REQUEST_FORMAT_DELTA_V1', request_byte_delta=delta,
        new_heldout_content_reads=0, model_calls=0, correctness_evaluations=0)
    return metadata


def derive_visible_trajectory_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authorization_path, authorization_sha256, gradient_prompt_path,
        gradient_prompt_sha256, validation_metadata_path, validation_metadata_sha256):
    if (parent.get('identity') != versions.MATH_V2_2_EXECUTION_BINDING_VERSION
            or not isinstance(attempt, str) or not attempt
            or attempt in {parent['execution_attempt_id'], parent['cache_namespace']}
            or binding_path == parent['binding_path'] or authorization_path == parent['current_user_scope_path']
            or gradient_prompt_path == parent['gradient_prompt_path']
            or validation_metadata_path == parent['validation_accounting_metadata_path']
            or not isinstance(gradient_prompt_sha256, str) or len(gradient_prompt_sha256) != 64):
        raise SearchContractError('MATH_VISIBLE_TRAJECTORY_FRESH_FREEZE_REQUIRED')
    CURRENT_POLICY_BUNDLE.validate_contract(parent)
    policy = trajectory_policy()
    c = deepcopy(parent)
    c.update(identity=versions.MATH_VISIBLE_TRAJECTORY_BINDING_VERSION,
        execution_attempt_id=attempt, cache_namespace=attempt, binding_path=binding_path,
        solver_output_interface=v6_interface_contract(), solver_trajectory_policy=policy,
        optimizer_input_schema=versions.GRADIENT_VISIBLE_OPTIMIZER_INPUT_VERSION,
        layer1_search_policy=asdict(CurrentLayer1Config(
            optimizer_input_schema=versions.GRADIENT_VISIBLE_OPTIMIZER_INPUT_VERSION)),
        pattern_policy=pattern_policy_for_trajectory(policy),
        gradient_prompt_path=gradient_prompt_path, gradient_prompt_sha256=gradient_prompt_sha256,
        trajectory_parent_binding_path=parent_path, trajectory_parent_binding_sha256=parent_sha256,
        current_user_scope_path=authorization_path, current_user_scope_sha256=authorization_sha256,
        continuation_authorization_sha256=authorization_sha256,
        validation_accounting_metadata_path=validation_metadata_path,
        validation_accounting_metadata_sha256=validation_metadata_sha256)
    # All models, decoding, membership, ceilings, algorithms, Memory and scoring
    # are inherited byte-for-byte. These changed feedback/interface identities
    # still require a wholly fresh source/attempt/authorization/cache freeze.
    return c


class MATHVisibleTrajectoryBinding(MATHGradientPatternBinding):
    def __init__(self, root, contract):
        if contract.get('identity') != versions.MATH_VISIBLE_TRAJECTORY_BINDING_VERSION:
            raise SearchContractError('MATH_VISIBLE_TRAJECTORY_FRESH_FREEZE_REQUIRED')
        self.root = root.resolve()
        self.contract = contract

    def blockers(self):
        c = self.contract
        try:
            parent_path = c['trajectory_parent_binding_path']
            if file_hash(self.path(parent_path)) != c['trajectory_parent_binding_sha256']:
                raise SearchContractError('MATH_VISIBLE_PARENT_HASH_MISMATCH')
            parent = json.loads(self.path(parent_path).read_bytes())
            parent_blockers = MATHGradientPatternBinding(self.root, parent).blockers()
            if parent_blockers:
                raise SearchContractError(parent_blockers[0])
            expected = derive_visible_trajectory_contract(parent,
                attempt=c['execution_attempt_id'], binding_path=c['binding_path'],
                parent_path=parent_path, parent_sha256=c['trajectory_parent_binding_sha256'],
                authorization_path=c['current_user_scope_path'], authorization_sha256=c['current_user_scope_sha256'],
                gradient_prompt_path=c['gradient_prompt_path'], gradient_prompt_sha256=c['gradient_prompt_sha256'],
                validation_metadata_path=c['validation_accounting_metadata_path'],
                validation_metadata_sha256=c['validation_accounting_metadata_sha256'])
            if c != expected:
                raise SearchContractError('MATH_VISIBLE_FROZEN_SETTINGS_CHANGED')
            for path_key, hash_key in [('current_user_scope_path', 'current_user_scope_sha256'),
                    ('gradient_prompt_path', 'gradient_prompt_sha256'),
                    ('validation_accounting_metadata_path', 'validation_accounting_metadata_sha256')]:
                if file_hash(self.path(c[path_key])) != c[hash_key]:
                    raise SearchContractError('MATH_VISIBLE_FRESH_DEPENDENCY_HASH_MISMATCH')
            if json.loads(self.path(c['gradient_prompt_path']).read_bytes()) != visible_gradient_prompt_artifact():
                raise SearchContractError('MATH_VISIBLE_GRADIENT_PROMPT_MISMATCH')
            scope = json.loads(self.path(c['current_user_scope_path']).read_bytes())
            required = dict(schema_version='math_visible_solution_user_scope_v1',
                attempt_id=c['execution_attempt_id'], one_attempt_only=True,
                user_authorized=True, real_api_authorized=False,
                scope_kind='VISIBLE_SOLUTION_INTERFACE_CORRECTION_ONLY',
                trajectory_policy=trajectory_policy(), parent_binding_sha256=c['trajectory_parent_binding_sha256'],
                validation_authorized=False, test_authorized=False)
            if (any(scope.get(k) != v for k, v in required.items())
                    or not isinstance(scope.get('user_task_sha256'), str)
                    or len(scope['user_task_sha256']) != 64):
                raise SearchContractError('MATH_VISIBLE_FRESH_USER_SCOPE_REQUIRED')
            CURRENT_POLICY_BUNDLE.validate_contract(c)
            from .current_math_dependencies import validate_effective_math_dependencies
            validate_effective_math_dependencies(self)
            return ()
        except (SearchContractError, KeyError, TypeError, ValueError, OSError) as exc:
            return (str(exc) if isinstance(exc, SearchContractError)
                else 'MATH_VISIBLE_TRAJECTORY_FRESH_FREEZE_REQUIRED',)
