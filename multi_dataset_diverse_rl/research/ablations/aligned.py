"""Fresh full-diagnosis protocol; historical Batch attempts remain immutable."""
from pathlib import Path

from ...benchmarks.data_freeze import digest,file_hash
from ...persistence.durable_io import read_json,atomic_write_json
from ...search.schemas import SearchContractError
from ..single_prompt_governance import private_path,require_selected_runtime
from . import governance as gov
from .config import default_config,build_plan

HISTORICAL_MANIFEST='experiments/manifests/single_prompt_full_diagnosis_pattern_v5.json'
HISTORICAL_NORMALIZED_SHA256='ebb37d318a06b972723f4cf3267da90afa475c4fde21e5d19163d0777cf5d8aa'
MANIFEST='experiments/manifests/single_prompt_module_alignment_v6.json'
PROTOCOL='docs/design/SINGLE_PROMPT_MODULE_ALIGNMENT_V6.md'
MANIFEST_SCHEMA='single_prompt_module_alignment_manifest_v6'
PREP_SCHEMA='single_prompt_module_alignment_prep_v6'
C_MANIFEST='experiments/manifests/single_prompt_pattern_c_only_v7.json'
C_PROTOCOL='docs/design/SINGLE_PROMPT_PATTERN_C_ONLY_V7.md'
C_MANIFEST_SCHEMA='single_prompt_pattern_c_only_manifest_v7'
C_PREP_SCHEMA='single_prompt_pattern_c_only_prep_v7'


def c_manifest_template(root):
    value=manifest_template(root)
    config=default_config()
    config.update(experiment_id='single_prompt_pattern_c_only_v7',parent_manifest=MANIFEST)
    config['arms']=[arm for arm in config['arms'] if arm['arm_id']=='C']
    value.update(schema_version=C_MANIFEST_SCHEMA,experiment_id=config['experiment_id'],
        parents=[MANIFEST],protocol_path=C_PROTOCOL,config=config,plan=build_plan(config),
        isolation=dict(fresh_initial_prompt=True,import_old_checkpoints=False,
            import_old_caches=False,import_old_teacher_responses=False,import_old_cost=False,
            run_arms=['C'],historical_AB='reference_only_not_new_matched_controls'))
    return value


def _protocol(c_only):
    return (C_MANIFEST,C_PROTOCOL,C_PREP_SCHEMA,'V7') if c_only else (MANIFEST,PROTOCOL,PREP_SCHEMA,'V6')


def validate_historical_manifest(root,value):
    import hashlib
    if (hashlib.sha256((root/HISTORICAL_MANIFEST).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=HISTORICAL_NORMALIZED_SHA256
            or value!=read_json(root/HISTORICAL_MANIFEST)):
        raise SearchContractError('ABLATION_FROZEN_V5_MANIFEST_CHANGED')


def manifest_template(root):
    value=gov.manifest_template(root)
    value.update(schema_version=MANIFEST_SCHEMA,experiment_id=default_config()['experiment_id'],
        parents=[HISTORICAL_MANIFEST],
        protocol_path=PROTOCOL,config=default_config(),plan=build_plan(default_config()))
    return value


def validate_manifest(root,value,*,c_only=False):
    if value!=(c_manifest_template(root) if c_only else manifest_template(root)):
        raise SearchContractError('ABLATION_FULL_DIAGNOSIS_MANIFEST_MISMATCH')
    if c_only:
        validate_manifest(root,read_json(root/MANIFEST))
    return gov.validate_manifest(root,read_json(root/gov.MANIFEST))


def _scope(root,destination,*,c_only=False):
    manifest_path,_,_,_=_protocol(c_only)
    manifest=read_json(root/manifest_path)
    reader=validate_manifest(root,manifest,c_only=c_only)
    value=gov._scope(root,destination)
    value.pop('scope_identity')
    plan=manifest['plan']
    value.update(study_id=manifest['experiment_id'],manifest_sha256=file_hash(root/manifest_path),plan=plan,
        call_ceiling=plan['training_semantic_call_ceiling'],test_plan=gov.test_plan(root,reader,plan),
        generation_recovery=plan['config']['runtime']['diagnosis_recovery'])
    return dict(value,scope_identity=digest(value))


def handoff_template(root,destination,scope,*,c_only=False):
    manifest_path,protocol_path,_,version=_protocol(c_only)
    value=gov.handoff_template(root,destination,scope)
    value['protocol'].update(name='Single Prompt full wrong-set diagnosis and formal representatives',version=version,
        protocol_sha256=file_hash(root/protocol_path),manifest_path=manifest_path,preregistration_path=protocol_path,
        preregistration_sha256=file_hash(root/protocol_path))
    value['execution']['allowed_retries']=dict(network=scope['network_recovery'],generation=scope['generation_recovery'])
    return value


def prepare(root,destination,*,c_only=False):
    destination=private_path(root,destination)
    require_selected_runtime(root)
    if destination.exists() or gov._git(root,'status','--porcelain','--untracked-files=no'):
        raise SearchContractError('ABLATION_FRESH_PREPARATION_CLEAN_SOURCE_REQUIRED')
    scope=_scope(root,destination,c_only=c_only)
    if set(scope['source_hashes'])-set(gov._git(root,'ls-files').splitlines()):
        raise SearchContractError('ABLATION_UNCOMMITTED_FREEZE_FILE')
    destination.mkdir(parents=True)
    atomic_write_json(destination/'prep.json',dict(schema_version=_protocol(c_only)[2],scope=scope))
    atomic_write_json(destination/'authorization.json',gov.authorization_template(scope))
    atomic_write_json(destination/'execution_plan.json',gov.execution_plan_template(scope))
    atomic_write_json(destination/'handoff.json',handoff_template(root,destination,scope,c_only=c_only))
    return dict(status='FROZEN_NOT_AUTHORIZED',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],real_api_calls=0)


def validate_prep(root,directory,*,authorized=False):
    directory=private_path(root,directory)
    require_selected_runtime(root)
    value=read_json(directory/'prep.json')
    c_only=value.get('schema_version')==C_PREP_SCHEMA
    if (set(value)!={'schema_version','scope'} or value['schema_version']!=_protocol(c_only)[2]
            or gov._git(root,'status','--porcelain','--untracked-files=no') or value['scope']!=_scope(root,directory,c_only=c_only)):
        raise SearchContractError('ABLATION_FULL_DIAGNOSIS_FROZEN_IDENTITY_CHANGED')
    scope=value['scope']
    auth=read_json(directory/'authorization.json')
    gov.validate_authorization(scope,auth,required=authorized)
    expected=handoff_template(root,directory,scope,c_only=c_only)
    handoff=read_json(directory/'handoff.json')
    if (read_json(directory/'execution_plan.json')!=gov.execution_plan_template(scope)
            or set(handoff)!=set(expected)
            or any(handoff[k]!=v for k,v in expected.items() if k not in ('READY_TO_RUN','status'))
            or (authorized and handoff['READY_TO_RUN'] is not True)):
        raise SearchContractError('ABLATION_FULL_DIAGNOSIS_HANDOFF_CHANGED')
    if authorized and file_hash(directory/'user_authorization_evidence.json')!=auth['user_authorization_evidence_sha256']:
        raise SearchContractError('ABLATION_FULL_DIAGNOSIS_HUMAN_EVIDENCE_CHANGED')
    return scope
