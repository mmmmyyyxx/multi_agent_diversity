"""Fresh single-use scope for the independent Pattern-breadth experiment."""
from copy import deepcopy

from ...benchmarks.data_freeze import digest,file_hash
from ...persistence.durable_io import read_json,atomic_write_json
from ...search.system_prompt import SystemPrompt
from .. import single_prompt_governance as shared
from . import governance as gov
from .proposal_config import MANIFEST,PROTOCOL,default_config,build_plan
from .pattern_proposals import require

PREP_SCHEMA='single_prompt_pattern_scheduling_prep_v1'


def inputs(root):
    parent=read_json(root/shared.MANIFEST)
    return parent,shared.validate_manifest(root,parent)


def scope_template(root,destination):
    config=read_json(root/MANIFEST);require(config==default_config(),'MANIFEST_CHANGED')
    require(read_json(root/'experiments/ablations/pattern_directions/default.json')==config,'DEFAULT_CONFIG_CHANGED')
    plan=build_plan(config);parent,reader=inputs(root)
    prep=destination.relative_to(root).as_posix();run=(destination.parent/'train_run').relative_to(root).as_posix()
    initial=SystemPrompt.from_dict(read_json(root/parent['initial_prompt_path'])['members'][0]['prompt'])
    hashes=gov.source_snapshot(root)
    for path in (MANIFEST,PROTOCOL,'experiments/ablations/pattern_directions/default.json'):
        hashes[path]=file_hash(root/path)
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
    value['protocol'].update(name='Single-Parent Pattern sequential cumulative versus parallel competition',version='V1',
        protocol_sha256=file_hash(root/PROTOCOL),manifest_path=MANIFEST,
        preregistration_path=PROTOCOL,preregistration_sha256=file_hash(root/PROTOCOL))
    value['execution'].update(proposal_slots_by_arm={'C0':1,'C1':2,'C2':2},adaptive_validation_seed=42,
        allowed_retries=dict(network=scope['network_recovery'],generation=scope['generation_recovery']),
        dispatch=dict(C0='single',C1='gate_before_next_generation',C2='all_generation_before_any_evaluation_serial_transport'),
        final_admission='last_strict_validation_retained_single_prompt')
    return value


def prepare(root,destination):
    destination=gov.private_path(root,destination);require(not destination.exists(),'FRESH_PREPARATION_REQUIRED')
    gov.require_selected_runtime(root)
    require(not gov._git(root,'status','--porcelain','--untracked-files=no'),'CLEAN_COMMITTED_SOURCE_REQUIRED')
    scope=scope_template(root,destination)
    require(not set(scope['source_hashes'])-set(gov._git(root,'ls-files').splitlines()),'UNCOMMITTED_FREEZE_FILE')
    destination.mkdir(parents=True)
    for name,value in [('prep',dict(schema_version=PREP_SCHEMA,scope=scope)),
        ('authorization',gov.authorization_template(scope)),('execution_plan',gov.execution_plan_template(scope)),
        ('handoff',handoff(root,destination,scope))]:atomic_write_json(destination/(name+'.json'),value)
    return dict(status='FROZEN_NOT_AUTHORIZED',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],
        cells=len(scope['plan']['cells']),real_api_calls=0,ready_to_run=False)


def validate_prep(root,directory,*,authorized=False):
    directory=gov.private_path(root,directory);value=read_json(directory/'prep.json')
    require(set(value)=={'schema_version','scope'} and value['schema_version']==PREP_SCHEMA,'PREP_SCHEMA')
    gov.require_selected_runtime(root)
    require(not gov._git(root,'status','--porcelain','--untracked-files=no'),'SOURCE_CHANGED')
    scope=value['scope'];require(scope==scope_template(root,directory),'FROZEN_IDENTITY_CHANGED')
    actual=read_json(directory/'handoff.json');expected=handoff(root,directory,scope)
    require(set(actual)==set(expected) and all(actual[k]==v for k,v in expected.items()
        if k not in ('READY_TO_RUN','status')),'HANDOFF_CHANGED')
    require(read_json(directory/'execution_plan.json')==gov.execution_plan_template(scope),'EXECUTION_PLAN_CHANGED')
    auth=read_json(directory/'authorization.json');gov.validate_authorization(scope,auth,required=authorized)
    if authorized:
        evidence=directory/'user_authorization_evidence.json'
        require(evidence.is_file() and auth['user_authorization_evidence_sha256']==file_hash(evidence)
            and actual['READY_TO_RUN'] is True,'EXACT_HUMAN_GRANT_AND_READY_REQUIRED')
    return scope
