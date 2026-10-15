"""Explicit preparation defaults, repeated-seed identities and module bindings."""
from copy import deepcopy
import re

from ...benchmarks.data_freeze import digest
from ...search.schemas import SearchContractError
from ...benchmarks.math_optimizer_generation import optimizer_generation_contract, pattern_cluster_generation_contract
from ...benchmarks.math_solver_decoding import solver_decoding_contract
from ..single_prompt_policy import independent_trajectory_policy as trajectory_policy
from ...provider_routing import routing_contract

SCHEMA = 'single_prompt_module_ablation_draft_v2'
IDENTITY = 'SINGLE_PROMPT_MODULE_ABLATION_KERNEL_V2'
PARENT = 'experiments/manifests/single_prompt_largest_pattern_v1.json'


def runtime_defaults():
    # Native provider/scoring contracts, not a historical execution policy.
    return dict(identity=IDENTITY, seeds=[42], rounds=20, error_batch_size=3,
             optimizer_structural_draws=4, optimizer_structural_retries=3,
             acceptance='validation_correct_count_strictly_greater_than_retained_profile',
             optimize_gate=False, final_selection='last_validation_retained_prompt',
             validation_count=100, validation_membership_seed=88,
             solver_concurrency=32, gradient_concurrency=8,
             test_solver_concurrency=16, repeats_per_arm=3,
             provider_routing_policy=routing_contract(), solver_decoding_policy=solver_decoding_contract(),
             optimizer_generation_policy=optimizer_generation_contract(),
             cluster_generation_policy=pattern_cluster_generation_contract(),
             trajectory_policy=trajectory_policy(), provider_timeout_seconds=120,
             solver_semantic_draws=4, first_solver_capacity=3600, length_recovery_capacity=6144,
             optimizer_input_limit=991808, optimizer_context_limit=1000000,
             runtime_environment='runs/math_v2_1/runtime', provider_sdk_version='2.37.0',
             paired_cache='solver_only_same_repeat_seed_split_complete_request_across_arms_and_rounds',
             non_solver_cache='forbidden_including_paid_response_replay',
             optimizer_wire='SINGLE_PROMPT_MODULE_OPTIMIZER_WIRE_V1',
             memory='module_explicit_not_implicit',
             provider_failure='accounted_typed_network_retry_other_failure_abort',
             timeout_recovery=dict(identity='ABLATION_ACCOUNTED_NETWORK_RECOVERY_V1',
                 retry_errors=['APITimeoutError', 'APIConnectionError'],
                 backoff_seconds=[2, 4, 8, 16, 30, 60], physical_call_ceiling=None),
             test='separate_frozen_authorized_final_test300_after_all_cells_sealed')


def batch_config():
    return dict(schema_version=SCHEMA, experiment_id='single_prompt_pattern_repeat3_solver_cache_v3',
        parent_manifest=PARENT, seed=42, repeats_per_arm=3, rounds=20,
        arms=[dict(arm_id='A', error_source='all_wrong', module='none'),
              dict(arm_id='B', error_source='random_wrong_batch', module='none'),
              dict(arm_id='C', error_source='random_wrong_batch', module='largest_pattern')],
        runtime=runtime_defaults(), preparation_only=True, real_api_authorized=False)


ALIGNED_SCHEMA = 'single_prompt_module_ablation_draft_v3'
ALIGNED_IDENTITY = 'SINGLE_PROMPT_FULL_DIAGNOSIS_KERNEL_V3'
ALIGNED_PARENT = 'experiments/manifests/single_prompt_pattern_truncation_continuation_v4.json'


def aligned_runtime():
    from .continuation import recovery_policy
    value = runtime_defaults()
    value.update(identity=ALIGNED_IDENTITY, diagnosis_universe='all_current_optimize_errors',
        pattern_representative_count=3, pattern_support_sampling='formal_seeded_task_type_round_robin',
        diagnosis_recovery=recovery_policy())
    return value


def default_config():
    value = batch_config()
    value.update(schema_version=ALIGNED_SCHEMA, experiment_id='single_prompt_full_diagnosis_pattern_v5',
        parent_manifest=ALIGNED_PARENT, runtime=aligned_runtime())
    value['arms'][2].update(error_source='all_wrong', module='full_gradient_pattern')
    return value


def validate_config(value):
    aligned = isinstance(value, dict) and value.get('schema_version') == ALIGNED_SCHEMA
    expected = default_config() if aligned else batch_config()
    if not isinstance(value, dict) or set(value) != set(expected):
        raise SearchContractError('ABLATION_CONFIG_SCHEMA')
    for key in ('seed', 'repeats_per_arm', 'rounds'):
        if type(value[key]) is not int or value[key] < (0 if key == 'seed' else 1):
            raise SearchContractError('ABLATION_CONFIG_INTEGER')
    if (value['schema_version'] != expected['schema_version'] or value['parent_manifest'] != expected['parent_manifest']
            or value['preparation_only'] is not True or value['real_api_authorized'] is not False
            or not isinstance(value['experiment_id'], str)
            or not re.fullmatch('[a-z][a-z0-9_]{0,79}', value['experiment_id'])):
        raise SearchContractError('ABLATION_DRAFT_NOT_EXECUTION_AUTHORIZATION')
    runtime = aligned_runtime() if aligned else runtime_defaults()
    runtime.update(seeds=[value['seed']], rounds=value['rounds'], repeats_per_arm=value['repeats_per_arm'])
    if digest(value['runtime']) != digest(runtime):
        raise SearchContractError('ABLATION_RUNTIME_POLICY_MISMATCH')
    arms = value['arms']
    if not isinstance(arms, list) or not arms:
        raise SearchContractError('ABLATION_ARMS_REQUIRED')
    ids = []
    for arm in arms:
        if (not isinstance(arm, dict) or set(arm) != {'arm_id', 'error_source', 'module'}
                or not isinstance(arm['arm_id'], str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,39}', arm['arm_id'])
                or arm['error_source'] not in {'all_wrong', 'random_wrong_batch'}
                or not isinstance(arm['module'], str)):
            raise SearchContractError('ABLATION_ARM_INVALID')
        ids.append(arm['arm_id'])
    if len(set(ids)) != len(ids):
        raise SearchContractError('ABLATION_DUPLICATE_ARM')
    return deepcopy(value)


def build_plan(config, registry=None):
    from .modules import default_registry
    config = validate_config(config)
    registry = registry or default_registry()
    bindings = {a['arm_id']: registry.binding(a['module']) for a in config['arms']}
    for arm in config['arms']:
        required = bindings[arm['arm_id']].get('required_error_source')
        if required is not None and arm['error_source'] != required:
            raise SearchContractError('ABLATION_MODULE_ERROR_SOURCE_MISMATCH')
    cells = [dict(cell_id=f"{a['arm_id']}_seed{config['seed']}_repeat{repeat:02d}",
                  arm_id=a['arm_id'], seed=config['seed'], repeat_id=repeat,
                  error_source=a['error_source'], module=a['module'])
             for a in config['arms'] for repeat in range(1, config['repeats_per_arm'] + 1)]
    p = config['runtime']
    # Transport failures are accounted but excluded from finite semantic ceilings.
    training = len(cells) * ((config['rounds'] + 1) * (150 + 100) * p['solver_semantic_draws']
                            + config['rounds'] * p['optimizer_structural_draws'])
    for arm in config['arms']:
        bound = bindings[arm['arm_id']]['semantic_calls_per_opportunity']
        training += config['repeats_per_arm'] * config['rounds'] * bound
    body = dict(identity=p['identity'], config=config, module_bindings=bindings, cells=cells,
        cell_order='arm_then_repeat_sequential',
        cache_policy='solver_only_paired_within_repeat_across_arms_and_rounds_no_legacy_cache',
        training_semantic_call_ceiling=training,
        final_test_semantic_call_ceiling=len(cells) * 300 * p['solver_semantic_draws'],
        cumulative_token_ceiling=None, ready_to_run=False,
        execution_blocker='FRESH_SOURCE_DATA_PHASE_FREEZE_AND_AUTHORIZATION_REQUIRED')
    return dict(body, plan_identity=digest(body))


def validate_plan(plan, registry=None):
    if digest(build_plan(plan['config'], registry)) != digest(plan):
        raise SearchContractError('ABLATION_PLAN_IDENTITY_MISMATCH')


def request_namespace(plan_identity, seed, repeat_id):
    if type(repeat_id) is not int or repeat_id < 1:
        raise SearchContractError('ABLATION_REPEAT_ID_REQUIRED')
    return digest(['single_prompt_ablation_repeat_v2', plan_identity, seed, repeat_id])
