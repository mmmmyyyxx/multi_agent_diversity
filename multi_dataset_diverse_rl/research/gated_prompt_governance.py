"""Fresh V2 freezes and explicit adaptive Validation100 access, never V1 resume."""
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from ..benchmarks.data_freeze import digest,file_hash
from ..benchmarks.access import DataPurpose
from ..persistence.durable_io import read_json,atomic_write_json
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..governance.token_accounting import TokenLedger,POLICY_V27_FORMAL_UNCAPPED,OperationalAbort
from ..search.schemas import SearchContractError
from . import single_prompt_governance as shared
from .gated_prompt_policy import policy,gate_membership,call_ceiling,STUDY,MANIFEST,SOURCE_BASE,IDENTITY
from .gated_prompt import GatedPromptBroker,run_cell
from . import gated_prompt_network as network
from . import gated_prompt_execution as execution_amendment

PREP_SCHEMA='single_prompt_validation_gated_prep_v2'
MANIFEST_SCHEMA='single_prompt_validation_gated_manifest_v2'
CELL_SCHEMA='single_prompt_validation_gated_cell_v2'
PROTOCOL='docs/design/SINGLE_PROMPT_VALIDATION_GATED_V2.md'


def manifest_template(root):
    parent=read_json(root/shared.MANIFEST)
    return dict(schema_version=MANIFEST_SCHEMA,experiment_id=STUDY,identity=IDENTITY,source_base_sha=SOURCE_BASE,
        parents=[shared.MANIFEST,'experiments/manifests/single_prompt_timeout_continuation_v1.json'],
        policy=policy(),data=parent['data'],initial_prompt_path=parent['initial_prompt_path'],
        initial_prompt_sha256=parent['initial_prompt_sha256'],protocol_path=PROTOCOL,
        cells=[dict(cell_id=f'{STUDY}_{arm}_seed{seed}',arm=arm,seed=seed,
            manifest_path=f'experiments/manifests/{STUDY}/{STUDY}_{arm}_seed{seed}.json')
            for arm in policy()['arms'] for seed in policy()['seeds']],
        authorization=dict(real_api_authorized=False,adaptive_validation_access=False,test_access=False),
        status='OFFLINE_PREPARATION_ONLY')


def validate_manifest(root,value):
    if value!=manifest_template(root):raise SearchContractError('GATED_MANIFEST_IDENTITY_MISMATCH')
    reader=shared.validate_manifest(root,read_json(root/shared.MANIFEST))
    for cell in value['cells']:
        expected=dict(schema_version=CELL_SCHEMA,study_id=STUDY,**cell,policy_sha256=digest(policy()),source_base_sha=SOURCE_BASE)
        if read_json(root/cell['manifest_path'])!=expected:raise SearchContractError('GATED_CELL_MISMATCH')
    return reader


@dataclass(frozen=True)
class AdaptiveGateAccess:
    scope_identity:str
    membership:dict
    require_authorization:object


def read_gate_rows(reader,access):
    if (type(access) is not AdaptiveGateAccess or re.fullmatch('[0-9a-f]{64}',access.scope_identity) is None
            or access.membership!=gate_membership(reader)):
        raise SearchContractError('GATED_VALIDATION_ACCESS_REQUIRED')
    access.require_authorization()
    return reader._read_rows('validation',frozen_subset_ids=access.membership['sample_ids'])


def source_snapshot(root):
    result=shared.source_snapshot(root)
    manifest=read_json(root/MANIFEST)
    paths=[MANIFEST,'experiments/manifests/math_v27_direct_seed86_formal_v1.yaml','scripts/prepare_gated_prompt_study.py','scripts/dry_run_gated_prompt_study.py',
        'tests/current/test_gated_prompt_study.py',network.MANIFEST,
        'tests/current/test_gated_prompt_network.py',execution_amendment.MANIFEST,
        'tests/current/test_gated_prompt_execution.py']+[c['manifest_path'] for c in manifest['cells']]
    result.update({p:file_hash(root/p) for p in paths})
    return result


def phase_cells(manifest,destination,amended=False):
    suffix=digest(destination)[:16]
    return [dict(c,attempt_id=c['cell_id']+'_train_'+suffix) for c in execution_amendment.selected_cells(manifest,amended)]


def prepare(root,destination,*,continuation_root=None,parent_inventory_pin=None,amended=False):
    destination=shared.private_path(root,destination)
    if destination.exists():raise SearchContractError('GATED_PREP_ALREADY_EXISTS')
    manifest=read_json(root/MANIFEST);reader=validate_manifest(root,manifest)
    if amended:
        execution_amendment.validate_manifest(read_json(root/execution_amendment.MANIFEST))
        if continuation_root is None:raise SearchContractError('GATED_AMENDMENT_REQUIRES_STOPPED_PARENT')
    shared.require_selected_runtime(root)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():
        raise SearchContractError('GATED_SOURCE_NOT_CLEAN')
    sources=source_snapshot(root)
    if set(sources)-set(subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()):
        raise SearchContractError('GATED_SOURCE_NOT_COMMITTED')
    prep=destination.relative_to(root).as_posix(); run=(destination.parent/'train_run').relative_to(root).as_posix()
    scope=dict(study_id=STUDY,phase='train',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_hashes=sources,runtime_identity=shared.runtime_identity(),policy=policy(),
        manifest_sha256=file_hash(root/MANIFEST),gate_membership=gate_membership(reader),
        cells=phase_cells(manifest,prep,amended),call_ceiling=call_ceiling(),token_ceiling=None,
        test_plan=execution_amendment.test_plan(final_test_plan(root,reader),amended),
        execution=dict(prep_path=prep,run_root=run,command=[policy()['runtime_interpreter'],'scripts/run_experiment.py','--prep',prep,'--run-root',run,'--execute']))
    if amended:scope['execution_amendment']=execution_amendment.contract()
    if continuation_root is not None:
        network.validate_operation_manifest(read_json(root/network.MANIFEST))
        pin=read_json(shared.private_path(root,parent_inventory_pin)) if parent_inventory_pin is not None else None
        parent=network.frozen_parent(root,continuation_root,expected_inventory=None if pin is None else pin['parent_inventory'],allow_user_stop=amended)
        if pin is not None and any(pin[k]!=parent[k] for k in ('parent_scope_identity','parent_source_sha','parent_run_root')):
            raise SearchContractError('GATED_NETWORK_INITIAL_PARENT_PIN_MISMATCH')
        parent_scope=read_json(root/parent['parent_run_root']/'scope.json')
        if (parent_scope['runtime_identity']!=scope['runtime_identity'] or parent_scope['gate_membership']!=scope['gate_membership']
                or execution_amendment.test_plan(parent_scope['test_plan'],amended)!=scope['test_plan']):raise SearchContractError('GATED_NETWORK_RUNTIME_OR_DATA_CHANGED')
        scope['continuation']=parent
    scope['scope_identity']=digest(scope)
    destination.mkdir(parents=True)
    atomic_write_json(destination/'prep.json',dict(schema_version=execution_amendment.PREP_SCHEMA if amended else network.PREP_SCHEMA if continuation_root is not None else PREP_SCHEMA,scope=scope))
    atomic_write_json(destination/'authorization.json',dict(scope_identity=scope['scope_identity'],approved=False,
        adaptive_validation_access=False,test_access=False,user_authorization_evidence_sha256=None,
        authorized_attempt_ids=[c['attempt_id'] for c in scope['cells']]))
    return dict(status='FROZEN_NOT_AUTHORIZED',scope_identity=scope['scope_identity'],source_sha=scope['source_sha'],
        call_ceiling=scope['call_ceiling'],validation_count=100,real_api_calls=0)


def validate_prep(root,directory,*,authorized=False):
    directory=shared.private_path(root,directory); prep=read_json(directory/'prep.json')
    amended=prep.get('schema_version')==execution_amendment.PREP_SCHEMA
    continuation=amended or prep.get('schema_version')==network.PREP_SCHEMA
    if set(prep)!={'schema_version','scope'} or prep['schema_version'] not in (PREP_SCHEMA,network.PREP_SCHEMA,execution_amendment.PREP_SCHEMA):raise SearchContractError('GATED_PREP_SCHEMA_MISMATCH')
    scope=prep['scope']; sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if (set(scope)!=({'study_id','phase','source_sha','source_hashes','runtime_identity','policy','manifest_sha256','gate_membership','cells','call_ceiling','token_ceiling','test_plan','execution','scope_identity'}|({'continuation'} if continuation else set())|({'execution_amendment'} if amended else set()))
            or scope['scope_identity']!=digest({k:v for k,v in scope.items() if k!='scope_identity'})
            or scope['study_id']!=STUDY or scope['phase']!='train' or scope['policy']!=policy()
            or scope['call_ceiling']!=call_ceiling() or scope['token_ceiling'] is not None):
        raise SearchContractError('GATED_SCOPE_IDENTITY_MISMATCH')
    shared.require_selected_runtime(root)
    if (scope['source_sha']!=sha or scope['source_hashes']!=source_snapshot(root)
            or scope['runtime_identity']!=shared.runtime_identity()
            or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip()):
        raise SearchContractError('GATED_SOURCE_OR_RUNTIME_CHANGED')
    manifest=read_json(root/MANIFEST);reader=validate_manifest(root,manifest)
    if amended:
        execution_amendment.validate_manifest(read_json(root/execution_amendment.MANIFEST))
        execution_amendment.amendment(scope)
    if continuation:
        network.validate_operation_manifest(read_json(root/network.MANIFEST))
        parent=scope['continuation']
        if network.frozen_parent(root,root/parent['parent_run_root'],expected_inventory=parent['parent_inventory'],allow_user_stop=amended)!=parent:
            raise SearchContractError('GATED_NETWORK_FROZEN_PARENT_CHANGED')
    path=directory.relative_to(root).as_posix(); run=(directory.parent/'train_run').relative_to(root).as_posix()
    if (scope['manifest_sha256']!=file_hash(root/MANIFEST) or scope['gate_membership']!=gate_membership(reader)
            or scope['test_plan']!=execution_amendment.test_plan(final_test_plan(root,reader),amended)
            or scope['cells']!=phase_cells(manifest,path,amended)
            or scope['execution']!=dict(prep_path=path,run_root=run,command=[policy()['runtime_interpreter'],'scripts/run_experiment.py','--prep',path,'--run-root',run,'--execute'])):
        raise SearchContractError('GATED_FROZEN_INPUT_OR_COMMAND_MISMATCH')
    auth=read_json(directory/'authorization.json')
    if (set(auth)!={'scope_identity','approved','adaptive_validation_access','test_access','user_authorization_evidence_sha256','authorized_attempt_ids'}
            or auth['scope_identity']!=scope['scope_identity'] or type(auth['test_access']) is not bool
            or auth['authorized_attempt_ids']!=[c['attempt_id'] for c in scope['cells']]):
        raise SearchContractError('GATED_AUTHORIZATION_SCOPE_MISMATCH')
    if authorized and (auth['approved'] is not True or auth['adaptive_validation_access'] is not True
            or re.fullmatch('[0-9a-f]{64}',str(auth['user_authorization_evidence_sha256'])) is None):
        raise SearchContractError('GATED_EXACT_TRAIN_AND_VALIDATION_AUTHORIZATION_REQUIRED')
    return scope


def execute(root,prep,run):
    prep,run=shared.private_path(root,prep),shared.private_path(root,run)
    if run.exists():raise SearchContractError('GATED_EXISTING_RUN_FORBIDDEN')
    scope=validate_prep(root,prep,authorized=True)
    if run!=root/scope['execution']['run_root']:raise SearchContractError('GATED_RUN_ROOT_MISMATCH')
    consumed=root/'runs/gated_prompt_consumed'/(scope['scope_identity']+'.json');consumed.parent.mkdir(parents=True,exist_ok=True)
    with consumed.open('x',encoding='utf-8') as stream:
        import json,os
        json.dump(dict(scope_identity=scope['scope_identity'],run_root=scope['execution']['run_root']),stream);stream.flush();os.fsync(stream.fileno())
    run.mkdir();atomic_write_json(run/'scope.json',scope);atomic_write_json(run/'authorization.json',read_json(prep/'authorization.json'))
    atomic_write_json(run/'lifecycle.json',dict(status='RUNNING',phase='train'))
    ledger=None;clients=None;current_cell=None;completed=[]
    signatures={p:(root/p).stat() for p in scope['source_hashes']}; auth_sha=file_hash(prep/'authorization.json');prep_sha=file_hash(prep/'prep.json')
    def guard(cell):
        if cell!=current_cell or file_hash(prep/'authorization.json')!=auth_sha or file_hash(prep/'prep.json')!=prep_sha:
            raise SearchContractError('GATED_ACTIVE_SCOPE_CHANGED')
        for p,s in signatures.items():
            now=(root/p).stat()
            if (now.st_size,now.st_mtime_ns)!=(s.st_size,s.st_mtime_ns):raise SearchContractError('GATED_SOURCE_MUTATED')
    try:
        from ..governance.autonomous_math import create_transport
        from ..provider_routing import routing_contract,IDENTITY as ROUTING
        from ..benchmarks.math_optimizer_generation import optimizer_generation_contract
        transport,clients=create_transport(dict(provider=ROUTING,provider_routing_policy=routing_contract(),
            models=dict(solver='gpt-4o-mini',optimizer_reflection='qwen3.7-flash',pattern='qwen3.7-flash',solver_thinking=False),
            decoding=dict(timeout_seconds=120),runtime_persistence_policy=True,optimizer_generation_policy=optimizer_generation_contract()))
        ledger=TokenLedger(run/'ledger',task_sha256=scope['scope_identity'],policy=execution_amendment.ledger_policy(scope))
        broker_class=network.NetworkBroker if 'continuation' in scope else GatedPromptBroker
        broker=broker_class(phase='train',scope=scope,transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(run/'provider_receipts',attempt_id=scope['scope_identity'],startup_identity_sha256=scope['scope_identity']),dispatch_guard=guard)
        progress=network.initialize_resume(root,run,scope,broker) if 'continuation' in scope else {}
        reader=validate_manifest(root,read_json(root/MANIFEST))
        optimize=reader.rows('optimize',DataPurpose.SEARCH_EVALUATION)
        validation=read_gate_rows(reader,AdaptiveGateAccess(scope['scope_identity'],scope['gate_membership'],lambda:validate_prep(root,prep,authorized=True)))
        atomic_write_json(run/'access.json',dict(optimize_ids=[r['stable_example_id'] for r in optimize],validation_gate_ids=scope['gate_membership']['sample_ids'],test_access=False))
        for cell in scope['cells']:
            validate_prep(root,prep,authorized=True);current_cell=cell['cell_id']
            broker.cell_id,broker.attempt_id,broker.seed=cell['cell_id'],cell['attempt_id'],cell['seed']
            directory=network.import_cell(root,run,scope,cell) if 'continuation' in scope else run/current_cell
            if 'continuation' not in scope:directory.mkdir()
            if (directory/'result.json').exists():
                completed.append(read_json(directory/'result.json'))
                print(__import__('json').dumps(dict(cell=current_cell,status='PRESERVED_COMPLETE_CELL',provider_calls=broker.usage_view()['provider_calls'])),flush=True)
                continue
            def persist(name,data):
                path=directory/(name+'.json')
                if path.exists():
                    if 'continuation' in scope and read_json(path)==data:return
                    raise SearchContractError('GATED_ARTIFACT_ALREADY_EXISTS')
                atomic_write_json(path,data)
            initial=read_json(root/read_json(root/MANIFEST)['initial_prompt_path'])['members'][0]['prompt']
            completed.append(run_cell(dict(cell,initial_prompt=initial),optimize,validation,broker,persist,resume=progress.get(current_cell)))
            print(__import__('json').dumps(dict(cell=current_cell,status='CELL_COMPLETE',provider_calls=broker.usage_view()['provider_calls'])),flush=True)
        validate_prep(root,prep,authorized=True)
        if broker.replay is not None:broker.replay.require_consumed()
        atomic_write_json(run/'results.json',completed)
        atomic_write_json(run/'execution_summary.json',dict(usage=broker.usage_view(),token_ledger=ledger.view(),cells_completed=len(completed)))
        if ledger.inflight:raise SearchContractError('GATED_UNRESOLVED_RESERVATION')
        ledger.close();ledger=None;clients.close();clients=None
        atomic_write_json(run/'lifecycle.json',dict(status='EXECUTION_COMPLETE',phase='train'))
        atomic_write_json(run/'terminal.json',dict(status='EXECUTION_COMPLETE',phase='train',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],artifacts=shared.inventory(run)))
        if read_json(prep/'authorization.json')['test_access']:
            from .gated_prompt_test import execute_final_test
            return execute_final_test(root,prep,run)
        return dict(status='EXECUTION_COMPLETE',cells_completed=len(scope['cells']))
    except BaseException as exc:
        # Once training is sealed, a later Test failure belongs only to Test.
        if (run/'terminal.json').exists():raise
        if ledger is not None:
            for key in tuple(ledger.inflight):ledger.reconcile(key,{},outcome='ABORT_FULL_RESERVATION_FALLBACK')
        status='PAUSED_RESOURCE_LIMIT' if str(exc)=='STUDY_CALL_CEILING' else 'EXECUTION_ABORTED'
        atomic_write_json(run/'lifecycle.json',dict(status=status,phase='train',cell_id=current_cell,cells_completed=len(completed)))
        atomic_write_json(run/'abort.json',dict(reason=str(exc) if isinstance(exc,(SearchContractError,OperationalAbort)) else type(exc).__name__,
            cell_id=current_cell,round_complete=False,token_ledger=ledger.view() if ledger else None))
        raise
    finally:
        if ledger is not None:ledger.close()
        if clients is not None:clients.close()


def final_test_plan(root,reader):
    from .gated_prompt_test import formal_test_plan
    return formal_test_plan(root,reader)
