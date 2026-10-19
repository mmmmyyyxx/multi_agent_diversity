"""Independent, bounded real-provider canary using the existing scheduling kernel."""
from copy import deepcopy

from ...benchmarks.data_freeze import digest,file_hash
from ...persistence.durable_io import read_json,atomic_write_json
from .proposal_config import default_config,build_plan
from .pattern_proposals import require

SCHEMA='single_prompt_pattern_canary_manifest_v1'
PREP_SCHEMA='single_prompt_pattern_canary_prep_v1'
STUDY='single_prompt_pattern_canary_v1'
MANIFEST='experiments/manifests/single_prompt_pattern_canary_v1.json'
PROTOCOL='docs/design/SINGLE_PROMPT_PATTERN_CANARY_V1.md'
TOKEN_CEILING=3_000_000


def manifest_template():
    config=default_config()
    config['rounds']=config['runtime']['rounds']=1
    config['repeats_per_arm']=config['runtime']['repeats_per_arm']=1
    plan=build_plan(config)
    return dict(schema_version=SCHEMA,experiment_id=STUDY,
        parent_manifest='experiments/manifests/single_prompt_pattern_scheduling_v1.json',
        config=config,purpose='operational_canary_not_efficacy_not_formal_prefix',
        optimize_count=150,adaptive_validation_count=100,seed=42,
        successful_semantic_call_ceiling=plan['training_semantic_call_ceiling'],
        cumulative_accounted_token_ceiling=TOKEN_CEILING,test_access=False,test_call_ceiling=0,
        real_api_authorized=False,reuse='fresh_attempt_no_old_cache_checkpoint_or_authorization',
        missing_direction='report_partial_coverage_no_duplicate_or_efficacy_based_rerun')


def validate_manifest(value):
    require(value==manifest_template(),'CANARY_MANIFEST_CHANGED')
    return build_plan(value['config'])


def scope_template(root,destination):
    from . import governance as gov,proposal_governance as parent
    manifest=read_json(root/MANIFEST);plan=validate_manifest(manifest)
    # Reuse metadata-only membership, initial Prompt, model and source bindings.
    scope=parent.scope_template(root,destination)
    scope.pop('scope_identity')
    scope.update(study_id=STUDY,manifest_sha256=file_hash(root/MANIFEST),plan=plan,
        call_ceiling=plan['training_semantic_call_ceiling'],token_ceiling=TOKEN_CEILING,
        canary_policy=deepcopy(manifest),
        test_plan=dict(identity='CANARY_TEST_PROHIBITED_V1',sample_ids=[],successful_semantic_call_ceiling=0))
    scope['source_hashes'].update({path:file_hash(root/path) for path in (MANIFEST,PROTOCOL)})
    return dict(scope,scope_identity=digest(scope))


def handoff(root,destination,scope):
    from . import proposal_governance as parent
    value=parent.handoff(root,destination,scope)
    value['protocol'].update(name='Independent one-round Pattern scheduling API canary',
        manifest_path=MANIFEST,protocol_sha256=file_hash(root/PROTOCOL),
        preregistration_path=PROTOCOL,preregistration_sha256=file_hash(root/PROTOCOL))
    value['execution'].update(early_stop_rule='one_outer_round_per_cell_no_efficacy_adaptation',
        cumulative_token_ceiling=TOKEN_CEILING,expected_checkpoint_paths=[
            scope['execution']['run_root']+'/'+c['cell_id']+f'/checkpoint_{n}.json'
            for c in scope['plan']['cells'] for n in (0,1)])
    value['authorization']['test_access_policy']='prohibited_for_entire_canary'
    return value


def prepare(root,destination):
    from . import governance as gov
    destination=gov.private_path(root,destination)
    require(not destination.exists(),'CANARY_FRESH_PREPARATION_REQUIRED')
    gov.require_selected_runtime(root)
    require(not gov._git(root,'status','--porcelain','--untracked-files=no'),'CANARY_CLEAN_SOURCE_REQUIRED')
    scope=scope_template(root,destination)
    require(not set(scope['source_hashes'])-set(gov._git(root,'ls-files').splitlines()),'CANARY_UNTRACKED_FREEZE_FILE')
    destination.mkdir(parents=True)
    for name,value in [('prep',dict(schema_version=PREP_SCHEMA,scope=scope)),
        ('authorization',gov.authorization_template(scope)),('execution_plan',gov.execution_plan_template(scope))]:
        atomic_write_json(destination/(name+'.json'),value)
    atomic_write_json(destination/'handoff.json',handoff(root,destination,scope))
    return dict(status='CANARY_FROZEN_NOT_AUTHORIZED',source_sha=scope['source_sha'],
        scope_identity=scope['scope_identity'],cells=3,semantic_call_ceiling=scope['call_ceiling'],
        token_ceiling=TOKEN_CEILING,test_access=False,real_api_calls=0,ready_to_run=False)


def validate_prep(root,directory,*,authorized=False):
    from . import governance as gov
    directory=gov.private_path(root,directory);value=read_json(directory/'prep.json')
    require(set(value)=={'schema_version','scope'} and value['schema_version']==PREP_SCHEMA,'CANARY_PREP_SCHEMA')
    gov.require_selected_runtime(root)
    require(not gov._git(root,'status','--porcelain','--untracked-files=no'),'CANARY_SOURCE_CHANGED')
    scope=value['scope'];require(scope==scope_template(root,directory),'CANARY_FROZEN_IDENTITY_CHANGED')
    require(scope['phase']=='train','CANARY_TRAIN_ONLY')
    actual=read_json(directory/'handoff.json');expected=handoff(root,directory,scope)
    require(set(actual)==set(expected) and all(actual[k]==v for k,v in expected.items()
        if k not in ('READY_TO_RUN','status')),'CANARY_HANDOFF_CHANGED')
    require(read_json(directory/'execution_plan.json')==gov.execution_plan_template(scope),'CANARY_EXECUTION_PLAN_CHANGED')
    auth=read_json(directory/'authorization.json');gov.validate_authorization(scope,auth,required=authorized)
    require(auth['test_access'] is False,'CANARY_TEST_AUTHORIZATION_FORBIDDEN')
    if authorized:
        evidence=directory/'user_authorization_evidence.json'
        require(evidence.is_file() and auth['user_authorization_evidence_sha256']==file_hash(evidence)
            and actual['READY_TO_RUN'] is True,'CANARY_EXACT_HUMAN_GRANT_AND_READY_REQUIRED')
    return scope


def coverage(results):
    require(len(results)==3 and [r['cell']['arm_id'] for r in results]==['C0','C1','C2']
        and all(r['rounds_completed']==1 for r in results),'CANARY_COMPLETE_CELLS')
    rows=[]
    for result in results:
        metric=result['metrics'][0]
        rows.append(dict(arm_id=result['cell']['arm_id'],selected_patterns=metric['selected_distinct_pattern_count'],
            attempted_proposals=sum(p['generation_draws']>0 for p in metric['proposals']),
            measured_proposals=sum(p['generation_status']=='MEASURED' for p in metric['proposals']),
            accepted_edits=metric['accepted_update_count'],
            second_current_parent_exercised=any(p['slot']==1 and p['parent_prompt_sha256']!=p['diagnosis_source_prompt_sha256']
                for p in metric['proposals'])))
    return dict(rows=rows,two_direction_generation_exercised=all(r['selected_patterns']==2 and r['attempted_proposals']==2 for r in rows[1:]),
        interpretation='operational_coverage_only_no_efficacy_or_diagnosis_accuracy_claim',
        incomplete_coverage='record_only_no_fabricated_patterns_no_automatic_rerun')


def execute(root,prep,run):
    from . import governance as gov
    scope=validate_prep(root,prep,authorized=True)
    outcome=gov.execute(root,prep,run)
    results,seal=gov.verify_training(run,scope)
    summary=dict(status='CANARY_EXECUTION_COMPLETE',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],
        terminal_sha256=seal,coverage=coverage(results),test_access=False,formal_data_import_allowed=False,
        accounting=gov.audit_accounting(run,scope,'train'))
    atomic_write_json(prep.parent/'canary_summary.json',summary)
    return dict(outcome,canary_status=summary['status'],coverage=summary['coverage'])
