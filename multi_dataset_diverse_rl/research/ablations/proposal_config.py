"""Separate single-Parent Pattern-breadth study; no population defaults change."""
from copy import deepcopy

from ...benchmarks.data_freeze import digest
from ...search.schemas import SearchContractError

SCHEMA = 'single_prompt_pattern_scheduling_draft_v1'
IDENTITY = 'SINGLE_PARENT_PATTERN_DIRECTION_SCHEDULING_KERNEL_V1'
MANIFEST = 'experiments/manifests/single_prompt_pattern_scheduling_v1.json'
PROTOCOL = 'docs/design/SINGLE_PROMPT_PATTERN_SCHEDULING_V1.md'
MODES = {'C0': 'single', 'C1': 'sequential', 'C2': 'parallel'}
POLICY = dict(identity='PATTERN_TOP2_SEQUENTIAL_VS_PARALLEL_V1',
    modes=MODES, parent='one_frozen_current_prompt_no_population',
    diagnosis='once_per_opportunity_all_current_optimize_errors',
    ranking='native_largest_unique_support_seed_round_repeated_without_replacement',
    representatives='formal_sample_pattern_support_same_seed_root_round_for_each_pattern',
    one_pattern='one_proposal_no_duplicate_direction',
    no_pattern='one_raw_B_batch_proposal_structured_null',
    dispatch='parallel_all_generation_before_evaluation_sequential_gate_before_next_generation',
    selection='parallel_maximum_validation_then_frozen_pattern_order_strict_gain_sequential_current_parent_strict_gain',
    sequential_feedback='root_diagnosis_preserved_active_support_formal_resample_actual_current_parent_observations',
    sequential_empty_target='skip_without_generation_no_replacement_pattern',
    no_op='skip_slot_without_regeneration',
    resume='audited_complete_boundaries_only_reject_partial_round_no_paid_response_replay')


def runtime():
    from .config import aligned_runtime
    value = aligned_runtime()
    value.update(identity=IDENTITY, validation_membership_seed=42,
        parallel_proposals=deepcopy(POLICY))
    return value


def default_config():
    return dict(schema_version=SCHEMA, experiment_id='single_prompt_pattern_scheduling_v1',
        parent_manifest='experiments/manifests/single_prompt_pattern_c_only_v7.json',
        seed=42, repeats_per_arm=3, rounds=20,
        arms=[dict(arm_id=arm, error_source='all_wrong', module='full_gradient_pattern',
                   proposal_mode=mode) for arm,mode in MODES.items()],
        runtime=runtime(), preparation_only=True, real_api_authorized=False)


def validate_config(value):
    expected=default_config()
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise SearchContractError('PROPOSAL_CONFIG_SCHEMA')
    for key in ('rounds','repeats_per_arm'):
        if type(value[key]) is not int or value[key]<1:
            raise SearchContractError('PROPOSAL_CONFIG_INTEGER')
    expected['rounds']=expected['runtime']['rounds']=value['rounds']
    expected['repeats_per_arm']=expected['runtime']['repeats_per_arm']=value['repeats_per_arm']
    if value!=expected:
        raise SearchContractError('PROPOSAL_SHARED_CONTRACT_MISMATCH')
    return deepcopy(value)


def slots(mode):
    if mode not in MODES.values():raise SearchContractError('PROPOSAL_MODE')
    return 1 if mode=='single' else 2


def build_plan(config,registry=None):
    from .modules import default_registry
    from .semantic_alignment import module_review
    config=validate_config(config);registry=registry or default_registry()
    bindings={a['arm_id']:registry.binding(a['module']) for a in config['arms']}
    if any(b['identity']!='ABLATION_FULL_WRONG_GRADIENT_PATTERN_REPRESENTATIVES_V1' for b in bindings.values()):
        raise SearchContractError('PROPOSAL_FORMAL_MODULE_REQUIRED')
    cells=[dict(cell_id=f"{a['arm_id']}_seed42_repeat{r:02d}",arm_id=a['arm_id'],seed=42,
        repeat_id=r,error_source=a['error_source'],module=a['module'],proposal_mode=a['proposal_mode'])
        for a in config['arms'] for r in range(1,config['repeats_per_arm']+1)]
    p=config['runtime'];per_arm={}
    for arm in config['arms']:
        count=slots(arm['proposal_mode'])
        per_arm[arm['arm_id']]=config['repeats_per_arm']*(250*p['solver_semantic_draws']+
            config['rounds']*(count*(250*p['solver_semantic_draws']+p['optimizer_structural_draws'])+
                              bindings[arm['arm_id']]['semantic_calls_per_opportunity']))
    body=dict(identity=IDENTITY,config=config,module_bindings=bindings,cells=cells,
        module_alignment={a['arm_id']:module_review(a['module'],bindings[a['arm_id']]) for a in config['arms']},
        cell_order='arm_then_repeat_sequential',
        cache_policy='solver_only_paired_within_repeat_across_arms_and_rounds_no_legacy_cache',
        training_semantic_call_ceiling=sum(per_arm.values()),training_semantic_calls_by_arm=per_arm,
        final_test_semantic_call_ceiling=len(cells)*300*p['solver_semantic_draws'],
        proposal_opportunities=sum(slots(c['proposal_mode'])*config['rounds'] for c in cells),
        cumulative_token_ceiling=None,ready_to_run=False,
        execution_blocker='FRESH_SOURCE_DATA_PHASE_FREEZE_AND_AUTHORIZATION_REQUIRED')
    return dict(body,plan_identity=digest(body))
