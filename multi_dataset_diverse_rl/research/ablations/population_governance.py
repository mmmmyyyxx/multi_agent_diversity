"""Fresh offline freeze for the independent V12 attempt; never authorizes APIs."""
from copy import deepcopy

from ...benchmarks.data_freeze import digest,file_hash
from ...persistence.durable_io import read_json,atomic_write_json
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt
from .. import single_prompt_governance as shared
from . import governance as gov
from .config import current_config,build_plan

MANIFEST='experiments/manifests/single_prompt_multi_parent_v12.json'
PROTOCOL='docs/design/SINGLE_PROMPT_MULTI_PARENT_V12.md'
PREP_SCHEMA='single_prompt_population_prep_v12'


def inputs(root):
    # This reader verifies published memberships only, not old execution eligibility.
    parent=read_json(root/shared.MANIFEST)
    return parent,shared.validate_manifest(root,parent)


def scope_template(root,destination):
    config=read_json(root/MANIFEST)
    if config!=current_config():raise SearchContractError('MULTI_PARENT_DRAFT_CHANGED')
    plan=build_plan(config);parent,reader=inputs(root)
    prep=destination.relative_to(root).as_posix();run=(destination.parent/'train_run').relative_to(root).as_posix()
    initial=SystemPrompt.from_dict(read_json(root/parent['initial_prompt_path'])['members'][0]['prompt'])
    hashes=gov.source_snapshot(root);hashes[MANIFEST]=file_hash(root/MANIFEST)
    value=dict(study_id=config['experiment_id'],phase='train',source_sha=gov._git(root,'rev-parse','HEAD'),
        source_hashes=hashes,runtime_identity=gov.runtime_identity(),manifest_sha256=file_hash(root/MANIFEST),
        plan=plan,gate_membership=gov.gate_membership(reader,42),test_plan=gov.test_plan(root,reader,plan),
        optimize_sample_ids=[m['stable_example_id'] for m in reader.members if m['project_split']=='optimize'],
        initial_prompt_sha256=initial.prompt_hash,call_ceiling=plan['training_semantic_call_ceiling'],token_ceiling=None,
        network_recovery=deepcopy(config['runtime']['timeout_recovery']),generation_recovery=deepcopy(config['runtime']['diagnosis_recovery']),
        execution=dict(prep_path=prep,run_root=run,
            command=[gov.INTERPRETER,'scripts/run_experiment.py','--prep',prep,'--run-root',run,'--execute']))
    return dict(value,scope_identity=digest(value))


def handoff(root,destination,scope):
    value=gov.handoff_template(root,destination,scope)
    value['protocol'].update(name='Single-Prompt native three-Parent population',version='V12',
        protocol_sha256=file_hash(root/PROTOCOL),manifest_path=MANIFEST,
        preregistration_path=PROTOCOL,preregistration_sha256=file_hash(root/PROTOCOL))
    value['execution'].update(offspring_slots=3,parent_capacity=3,adaptive_validation_seed=42,
        final_admission='pool_argmax_validation_nonregression_no_fallback')
    return value


def prepare(root,destination):
    destination=gov.private_path(root,destination)
    if destination.exists():raise SearchContractError('MULTI_PARENT_FRESH_PREPARATION_REQUIRED')
    gov.require_selected_runtime(root)
    if gov._git(root,'status','--porcelain','--untracked-files=no'):
        raise SearchContractError('MULTI_PARENT_CLEAN_COMMITTED_SOURCE_REQUIRED')
    scope=scope_template(root,destination)
    if set(scope['source_hashes'])-set(gov._git(root,'ls-files').splitlines()):
        raise SearchContractError('MULTI_PARENT_UNCOMMITTED_SOURCE')
    destination.mkdir(parents=True)
    atomic_write_json(destination/'prep.json',dict(schema_version=PREP_SCHEMA,scope=scope))
    atomic_write_json(destination/'authorization.json',gov.authorization_template(scope))
    atomic_write_json(destination/'execution_plan.json',gov.execution_plan_template(scope))
    atomic_write_json(destination/'handoff.json',handoff(root,destination,scope))
    return dict(status='FROZEN_NOT_AUTHORIZED',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],
        cells=len(scope['plan']['cells']),real_api_calls=0)


def validate_prep(root,directory,*,authorized=False):
    directory=gov.private_path(root,directory);value=read_json(directory/'prep.json')
    if value.get('schema_version')!=PREP_SCHEMA or set(value)!={'schema_version','scope'}:
        raise SearchContractError('MULTI_PARENT_PREP_SCHEMA')
    gov.require_selected_runtime(root)
    if gov._git(root,'status','--porcelain','--untracked-files=no'):
        raise SearchContractError('MULTI_PARENT_SOURCE_CHANGED')
    scope=value['scope']
    if scope!=scope_template(root,directory):raise SearchContractError('MULTI_PARENT_FREEZE_CHANGED')
    actual=read_json(directory/'handoff.json');expected=handoff(root,directory,scope)
    if set(actual)!=set(expected) or any(actual[k]!=v for k,v in expected.items() if k not in ('READY_TO_RUN','status')):
        raise SearchContractError('MULTI_PARENT_HANDOFF_CHANGED')
    if read_json(directory/'execution_plan.json')!=gov.execution_plan_template(scope):
        raise SearchContractError('MULTI_PARENT_PLAN_CHANGED')
    gov.validate_authorization(scope,read_json(directory/'authorization.json'),required=authorized)
    if authorized:
        evidence=directory/'user_authorization_evidence.json'
        if not evidence.is_file() or read_json(directory/'authorization.json')['user_authorization_evidence_sha256']!=file_hash(evidence):
            raise SearchContractError('MULTI_PARENT_EXACT_HUMAN_GRANT_REQUIRED')
        if actual['READY_TO_RUN'] is not True:raise SearchContractError('MULTI_PARENT_READY_HANDOFF_REQUIRED')
    return scope
