"""Fresh all-error, same-Parent two-proposal control for the scheduling study.

Reuses the reviewed no-module adapter, candidate generator, evaluation, strict
gate and sealed Test runner. No historical realization or authorization imports.
"""
from copy import deepcopy
from dataclasses import replace

from ...benchmarks.data_freeze import digest, file_hash
from ...persistence.durable_io import read_json, atomic_write_json
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt

SCHEMA = 'single_prompt_full_error_control_draft_v2'
PREP_SCHEMA = 'single_prompt_full_error_control_prep_v2'
IDENTITY = 'SAME_PARENT_TWO_ALL_ERRORS_PROPOSALS_CONTROL_KERNEL_V2'
MANIFEST = 'experiments/manifests/single_prompt_full_error_control_v2.json'
PROTOCOL = 'docs/design/SINGLE_PROMPT_FULL_ERROR_CONTROL_V2.md'
PARENT = 'experiments/manifests/single_prompt_pattern_scheduling_continuation_v2.json'
CONCURRENCY = dict(solver_concurrency=96, gradient_concurrency=16,
                   test_solver_concurrency=96, local_grading_concurrency=16)


def require(value, reason):
    if not value:
        raise SearchContractError(reason)


def default_config():
    from .config import default_config as aligned_config
    value = aligned_config()
    value.update(schema_version=SCHEMA, experiment_id='single_prompt_full_error_control_v2',
                 parent_manifest=PARENT,
                 arms=[dict(arm_id='A2', error_source='all_wrong', module='none', proposal_mode='parallel')])
    value['runtime'].update(identity=IDENTITY, validation_membership_seed=42,
        proposals_per_round=2, proposal_evidence='all_current_parent_errors_each_slot',
        proposal_dispatch='all_generation_before_any_evaluation',
        proposal_selection='maximum_validation_then_slot_strict_gain', **CONCURRENCY)
    return value


def validate_config(value):
    require(isinstance(value, dict) and digest(value) == digest(default_config()), 'FULL_ERROR_CONTROL_CONFIG')
    return deepcopy(value)


def build_plan(config, registry=None):
    from .config import default_config as aligned_config, build_plan as aligned_plan
    config = validate_config(config)
    # Reuse the single-Parent kernel's actual plan construction and module review.
    base = aligned_config()
    base['arms'] = [{k:v for k,v in a.items() if k != 'proposal_mode'} for a in config['arms']]
    plan = aligned_plan(base, registry)
    plan.pop('plan_identity')
    plan.update(config=config, identity=IDENTITY,
        cells=[dict(c,proposal_mode='parallel') for c in plan['cells']],
        training_semantic_call_ceiling=3*(250*4+20*2*(250*4+4)),
        proposal_opportunities=120)
    return dict(plan, plan_identity=digest(plan))


def execution_concurrency(scope):
    require(scope.get('plan', {}).get('config', {}).get('schema_version') == SCHEMA
            and digest(scope.get('execution_concurrency')) == digest(CONCURRENCY)
            and scope.get('phase') in ('train', 'test') and 'continuation' not in scope,
            'FULL_ERROR_CONTROL_EXECUTION_POLICY')
    validate_config(scope['plan']['config'])
    return deepcopy(CONCURRENCY)


def inputs(root):
    from .proposal_governance import inputs as scheduling_inputs
    return scheduling_inputs(root)


def scope_template(root, destination):
    from . import governance as gov
    config = read_json(root / MANIFEST)
    plan = build_plan(config)
    parent, reader = inputs(root)
    prep = destination.relative_to(root).as_posix()
    run = (destination.parent / 'train_run').relative_to(root).as_posix()
    initial = SystemPrompt.from_dict(read_json(root / parent['initial_prompt_path'])['members'][0]['prompt'])
    hashes = gov.source_snapshot(root)
    for path in (MANIFEST, PROTOCOL, PARENT):
        hashes[path] = file_hash(root / path)
    value = dict(study_id=config['experiment_id'], phase='train', source_sha=gov._git(root, 'rev-parse', 'HEAD'),
        source_hashes=hashes, runtime_identity=gov.runtime_identity(), manifest_sha256=file_hash(root / MANIFEST),
        plan=plan, gate_membership=gov.gate_membership(reader, 42), test_plan=gov.test_plan(root, reader, plan),
        optimize_sample_ids=[m['stable_example_id'] for m in reader.members if m['project_split'] == 'optimize'],
        initial_prompt_sha256=initial.prompt_hash, call_ceiling=plan['training_semantic_call_ceiling'], token_ceiling=None,
        execution_concurrency=deepcopy(CONCURRENCY),
        network_recovery=deepcopy(config['runtime']['timeout_recovery']),
        generation_recovery=deepcopy(config['runtime']['diagnosis_recovery']),
        execution=dict(prep_path=prep, run_root=run,
            command=[gov.INTERPRETER, 'scripts/run_experiment.py', '--prep', prep, '--run-root', run, '--execute']))
    return dict(value, scope_identity=digest(value))


def handoff(root, destination, scope):
    from . import governance as gov
    value = gov.handoff_template(root, destination, scope)
    value['protocol'].update(name='All current Optimize errors, same-Parent two-proposal control', version='V2',
        protocol_sha256=file_hash(root / PROTOCOL), manifest_path=MANIFEST,
        preregistration_path=PROTOCOL, preregistration_sha256=file_hash(root / PROTOCOL))
    value['execution'].update(train_solver_concurrency=96, gradient_concurrency=0,
        test_solver_concurrency=96, local_grading_concurrency=16, proposal_slots_by_arm={'A2': 2},
        adaptive_validation_seed=42, allowed_roles={'train': ['solver', 'reflection'], 'test': ['solver']},
        final_admission='last_strict_validation_retained_single_prompt')
    return value


def prepare(root, destination):
    from . import governance as gov
    destination = gov.private_path(root, destination)
    gov.require_selected_runtime(root)
    require(not destination.exists() and not gov._git(root, 'status', '--porcelain', '--untracked-files=no'),
            'FULL_ERROR_CONTROL_FRESH_CLEAN_SOURCE')
    scope = scope_template(root, destination)
    require(not set(scope['source_hashes']) - set(gov._git(root, 'ls-files').splitlines()),
            'FULL_ERROR_CONTROL_UNTRACKED_FREEZE')
    destination.mkdir(parents=True)
    for name, value in [('prep', dict(schema_version=PREP_SCHEMA, scope=scope)),
                        ('authorization', gov.authorization_template(scope)),
                        ('execution_plan', gov.execution_plan_template(scope))]:
        atomic_write_json(destination / (name + '.json'), value)
    atomic_write_json(destination / 'handoff.json', handoff(root, destination, scope))
    return dict(status='FROZEN_NOT_AUTHORIZED', source_sha=scope['source_sha'], scope_identity=scope['scope_identity'],
                cells=3, real_api_calls=0, ready_to_run=False)


def validate_prep(root, directory, *, authorized=False):
    from . import governance as gov
    directory = gov.private_path(root, directory)
    gov.require_selected_runtime(root)
    value = read_json(directory / 'prep.json')
    require(set(value) == {'schema_version', 'scope'} and value['schema_version'] == PREP_SCHEMA
            and not gov._git(root, 'status', '--porcelain', '--untracked-files=no'), 'FULL_ERROR_CONTROL_SOURCE')
    scope = value['scope']
    require(scope == scope_template(root, directory), 'FULL_ERROR_CONTROL_FROZEN_IDENTITY')
    expected, actual = handoff(root, directory, scope), read_json(directory / 'handoff.json')
    require(set(expected) == set(actual) and all(actual[k] == v for k, v in expected.items()
            if k not in ('READY_TO_RUN', 'status')), 'FULL_ERROR_CONTROL_HANDOFF')
    require(read_json(directory / 'execution_plan.json') == gov.execution_plan_template(scope),
            'FULL_ERROR_CONTROL_EXECUTION_PLAN')
    auth = read_json(directory / 'authorization.json')
    gov.validate_authorization(scope, auth, required=authorized)
    if authorized:
        require(actual['READY_TO_RUN'] is True and auth['user_authorization_evidence_sha256'] ==
                file_hash(directory / 'user_authorization_evidence.json'), 'FULL_ERROR_CONTROL_EXACT_HUMAN_GRANT')
    execution_concurrency(scope)
    return scope


def audit_feedback(opportunity, previous, optimize_ids):
    """Prove the delivered set is all current errors, rather than a small Batch."""
    expected = [i for i in optimize_ids if not previous['optimize'][i]['correct']]
    errors = opportunity['errors']
    require([e['example_id'] for e in errors] == expected
            and opportunity['structured_feedback'] is None and opportunity['module_context'] is None,
            'FULL_ERROR_CONTROL_ALL_ERRORS_REQUIRED')
    # Bind each model-side feedback value to the actual retained Parent realization.
    for error in errors:
        observation = previous['optimize'][error['example_id']]
        require(error['solver_response'] == observation['prediction']['text']
                and error['prediction'] == observation['answer'] and error['valid'] == observation['valid']
                and error['correct'] is False and error['invalid_reason'] == observation['prediction']['invalid_reason']
                and error['semantic_attempt_count'] == observation['prediction']['semantic_attempt_count'],
                'FULL_ERROR_CONTROL_PARENT_RESPONSE')


def bind_feedback(prepared, parent, cell, previous, optimize_ids):
    """Keep raw feedback explicitly separate from Pattern diagnosis artifacts."""
    feedback=dict(errors=deepcopy(prepared.errors),structured_feedback=prepared.structured_feedback,
                  module_context=prepared.module_context)
    audit_feedback(feedback,previous,optimize_ids)
    require(prepared.state is None and prepared.audit==dict(mode='raw_errors'),
            'FULL_ERROR_CONTROL_NO_DIAGNOSIS')
    return replace(prepared,audit=dict(raw_feedback=feedback,parent_prompt_sha256=parent.prompt_hash,cell_id=cell['cell_id']))


def project_feedback(audit):
    require(set(audit)-{'diagnostic_draw_counts'}=={'raw_feedback','parent_prompt_sha256','cell_id'},
            'FULL_ERROR_CONTROL_RAW_AUDIT')
    feedback=audit['raw_feedback']
    require(feedback['structured_feedback'] is None and feedback['module_context'] is None,
            'FULL_ERROR_CONTROL_NO_DIAGNOSIS')
    focus=dict(selected_pattern_id=None,selected_support_ids=[],representative_example_ids=[],
               raw_all_error_control=True)
    views=[dict(feedback=deepcopy(feedback),focus=deepcopy(focus)) for _ in range(2)]
    return views,dict(pattern_count=0,selected_pattern_ids=[None,None],pattern_support_counts=[0,0],
        selected_distinct_pattern_count=0,degeneration='RAW_ALL_ERRORS_TWO_SAME_PARENT_PROPOSALS')


def audit_raw_module(audit, previous, cell, optimize_ids, diagnostic_cost):
    audit_feedback(audit['raw_feedback'],previous,optimize_ids)
    require(audit['parent_prompt_sha256']==SystemPrompt.from_dict(previous['prompt']).prompt_hash
            and audit['cell_id']==cell['cell_id'] and not audit['diagnostic_draw_counts']
            and all(v==0 for v in diagnostic_cost.values()),'FULL_ERROR_CONTROL_DIAGNOSTIC_CALL_FORBIDDEN')
