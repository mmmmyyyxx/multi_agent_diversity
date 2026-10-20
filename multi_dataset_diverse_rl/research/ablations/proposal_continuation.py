"""Explicit 96/16 transport amendment over an immutable scheduling prefix.

The scientific plan and request namespaces stay frozen. Execution concurrency is
a separate hashed field. Only Solver paid draws are recoverable; completed
diagnosis and edits are restored as stage artifacts, never Teacher responses.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re
import shutil

from ...benchmarks.data_freeze import digest, file_hash
from ...governance.token_accounting import _digest, serialized_request, reservation
from ...persistence.durable_io import read_json, atomic_write_json, append_jsonl
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt, solver_messages
from .runtime import AblationBroker
from .pattern_proposals import require

PREP_SCHEMA = 'single_prompt_pattern_scheduling_continuation_prep_v2'
MANIFEST = 'experiments/manifests/single_prompt_pattern_scheduling_continuation_v2.json'
PROTOCOL = 'docs/design/SINGLE_PROMPT_PATTERN_SCHEDULING_CONTINUATION_V2.md'
PARENT = 'runs/ablations/pattern_scheduling_seed42_repeat3_formal_20261019/train_run'
PARENT_SCOPE = '797b1a8145131b94874ef9109f89528fa0d5ee3a8d6112ca7cf2a6c70d6068fb'
PARENT_SOURCE = 'b8bb757e601f8679d035cc1c55c4b714467b3e88'
CONCURRENCY = dict(solver_concurrency=96, gradient_concurrency=16, test_solver_concurrency=96,
                   local_grading_concurrency=16)


def manifest_template():
    return dict(schema_version='single_prompt_pattern_scheduling_continuation_manifest_v2',
        experiment_id='single_prompt_pattern_scheduling_continuation_v2',
        parents=['experiments/manifests/single_prompt_pattern_scheduling_v1.json'],protocol_path=PROTOCOL,
        scientific_plan='unchanged_original_nine_cells_seed42_twenty_rounds',
        execution_concurrency=deepcopy(CONCURRENCY),
        continuation='immutable_completed_prefix_saved_partial_diagnosis_edit_exact_paid_solver_draws',
        parent_journal='read_only_full_bound_uncertain_stopped_reservations_in_separate_audit',
        training_semantic_call_ceiling=391740,test_semantic_call_ceiling=10800,
        cumulative_token_ceiling=None,real_api_authorized=False,status='PREPARED_NOT_EXECUTED')


def validate_manifest(value):
    require(value==manifest_template(),'CONTINUATION_MANIFEST')
    return value


def execution_concurrency(scope):
    require(scope.get('execution_concurrency') == CONCURRENCY, 'CONTINUATION_CONCURRENCY')
    require(scope.get('continuation', {}).get('identity') == PREP_SCHEMA or
            scope.get('phase') == 'test' and scope.get('parent_scope_identity'), 'CONTINUATION_SCOPE')
    return deepcopy(CONCURRENCY)


def _zero():
    # Keep the established usage schema: recovered Solver draws are disclosed in
    # a separate journal, not represented as newly paid calls or Teacher replay.
    roles = ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster')
    return dict(provider_calls=0, input_tokens=0, output_tokens=0, cache_hits=0,
                supplier_reported_tokens=0, fallback_usage_calls=0,
                network_failed_calls=0, timeout_retries=0, replayed_calls=0,
                **{r:0 for r in roles}, **{r+'_'+k:0 for r in roles for k in ('input_tokens','output_tokens')})


def _add(a, b):
    require(set(a) == set(b), 'CONTINUATION_USAGE_SCHEMA')
    return {k:a[k]+b[k] for k in a}


def _difference(a, b):
    value = {k:a[k]-b[k] for k in a}
    require(all(type(v) is int and v >= 0 for v in value.values()), 'CONTINUATION_NEGATIVE_USAGE')
    return value


def checkpoint_usage(checkpoints):
    value = deepcopy(checkpoints[0]['bootstrap_usage'])
    for cp in checkpoints[1:]:value = _add(value, cp['metric']['cost'])
    return value


def _receipt(parent, scope, event):
    store = ProviderResponseReceipts(parent/'provider_receipts', attempt_id=event['attempt_id'],
                                    startup_identity_sha256=scope['scope_identity'])
    return store.read(event['reservation_id'])


def audit_parent(root):
    """No parent journal constructor: an interrupted journal stays byte-identical."""
    from . import governance as gov
    from .proposal_config import default_config, build_plan
    from .proposal_audit import audit_cells
    parent = root/PARENT; scope = read_json(parent/'scope.json')
    require(scope['scope_identity'] == PARENT_SCOPE and scope['source_sha'] == PARENT_SOURCE and
            digest({k:v for k,v in scope.items() if k!='scope_identity'}) == PARENT_SCOPE,
            'CONTINUATION_PARENT_IDENTITY')
    require(scope['plan'] == build_plan(default_config()) and scope['phase']=='train', 'CONTINUATION_PARENT_PLAN')
    require(not (parent/'terminal.json').exists() and not (parent.parent/'test_run').exists() and
            read_json(parent/'access.json')['test_access'] is False, 'CONTINUATION_PARENT_ACCESS')
    gov.validate_authorization(scope, read_json(parent/'authorization.json'), required=True, test=True)
    original_prep=parent.parent/'prep';auth=read_json(original_prep/'authorization.json')
    require(read_json(original_prep/'prep.json')['scope']==scope and auth==read_json(parent/'authorization.json')
            and auth['user_authorization_evidence_sha256']==file_hash(original_prep/'user_authorization_evidence.json')
            and read_json(original_prep/'handoff.json')['READY_TO_RUN'] is True,
            'CONTINUATION_PARENT_HUMAN_GRANT')
    require(read_json(root/'runs/ablations/consumed'/f'{PARENT_SCOPE}.json') ==
            dict(scope_identity=PARENT_SCOPE, run_root=PARENT), 'CONTINUATION_PARENT_CONSUMED')
    # A byte-for-byte source snapshot is checked against its actual Git source,
    # rather than pretending the new implementation was the historical version.
    import subprocess
    for path, sha in scope['source_hashes'].items():
        blob = subprocess.check_output(['git','show',PARENT_SOURCE+':'+path],cwd=root)
        variants=(blob,blob.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
        current=(root/path).read_bytes()
        require(any(hashlib.sha256(b).hexdigest()==sha for b in variants) or
                hashlib.sha256(current).hexdigest()==sha and current.replace(b'\r\n',b'\n')==blob.replace(b'\r\n',b'\n'),
                'CONTINUATION_PARENT_SOURCE_HASH')
    data = (parent/'ledger/events.jsonl').read_bytes()
    require(data.endswith(b'\n'), 'CONTINUATION_PARENT_JOURNAL_TRUNCATED')
    reserves={}; charges={}; prev=None; counts={}; paid=[]; old_partial=[]
    completed={}; prefix_usage={}
    for cell_index,cell in enumerate(scope['plan']['cells']):
        folder=parent/cell['cell_id']; cps=[]
        while (folder/f'checkpoint_{len(cps)}.json').exists():
            cps.append(read_json(folder/f'checkpoint_{len(cps)}.json'))
        if not cps:break
        n=len(cps)-1; completed[cell['cell_id']]=n
        usage=checkpoint_usage(cps);prefix_usage[cell['cell_id']]=usage
        last=cps[-1]
        result=dict(cell=cell,scope_identity=cps[0]['scope_identity'],module_binding=cps[0]['scope']['module_binding'],
            module_state=None,rounds_completed=n,prompt_history=[c['prompt'] for c in cps],
            metrics=[c['metric'] for c in cps[1:]],final_prompt=last['prompt'],
            final_optimize_correct=sum(v['correct'] for v in last['optimize'].values()),optimize_count=150,
            final_validation_correct=sum(v['correct'] for v in last['validation'].values()),validation_count=100,
            usage=usage,search_wall_clock_seconds=cps[0]['bootstrap_wall_seconds']+
                sum(c['metric']['wall_clock_seconds'] for c in cps[1:]))
        if (folder/'result.json').exists():require(result==read_json(folder/'result.json'),'CONTINUATION_PARENT_RESULT')
        audit_cells(parent,scope,results=[result],allow_partial=True,cell_offset=cell_index)
        del cps,result,last
    for i,line in enumerate(data.splitlines()):
        event=json.loads(line)
        require(event['sequence']==i and event['previous_sha256']==prev and
                event['event_sha256']==_digest({k:v for k,v in event.items() if k!='event_sha256'}),
                'CONTINUATION_JOURNAL_CHAIN')
        prev=event['event_sha256']
        if event['kind']=='AUTHORIZE':require(i==0 and event['task_sha256']==PARENT_SCOPE,'CONTINUATION_LEDGER_SCOPE')
        elif event['kind']=='RESERVE':
            require(event['reservation_id'] not in reserves,'CONTINUATION_DUPLICATE_RESERVE')
            reserves[event['reservation_id']]=event
        elif event['kind']=='CHARGE':
            require(event['reservation_id'] in reserves and event['reservation_id'] not in charges,
                    'CONTINUATION_ORPHAN_CHARGE')
            charges[event['reservation_id']]=event
        else:raise SearchContractError('CONTINUATION_JOURNAL_KIND')
    pending=[]
    for rid,event in reserves.items():
        cell=event['attempt_id'].removeprefix(PARENT_SCOPE+'_train_')
        require(cell in completed and event['role'] in ('solver','reflection','pattern_gradient','pattern_cluster'),
                'CONTINUATION_RECEIPT_CELL_ROLE')
        usage=counts.setdefault(cell,_zero()); role=event['role'];usage['provider_calls']+=1;usage[role]+=1
        record=None
        if (parent/'provider_receipts'/f'{rid}.json').exists():
            record=_receipt(parent,scope,event)
            require(hashlib.sha256(serialized_request(record['request'])).hexdigest()==event['request_sha256'],
                    'CONTINUATION_RECEIPT_REQUEST')
        charge=charges.get(rid)
        if charge:
            require(record is not None and charge['response_receipt']['integrity_sha256']==
                    read_json(parent/'provider_receipts'/f'{rid}.json')['integrity_sha256'], 'CONTINUATION_RECEIPT_SEAL')
            require(charge['outcome'] in ('PROVIDER_RESPONSE','PROVIDER_TIMEOUT'), 'CONTINUATION_PARENT_FAILURE')
            reliable=charge['reliable_usage']; inp=charge['input_tokens'];out=charge['output_tokens']
            require(charge['charged']==inp+out,'CONTINUATION_CHARGE_SUM')
            if charge['outcome']=='PROVIDER_TIMEOUT':
                require(record.get('error_category') in ('APITimeoutError','APIConnectionError'),'CONTINUATION_NETWORK_TYPE')
                usage['network_failed_calls']+=1;usage['timeout_retries']+=1
        else:
            require(role=='solver','CONTINUATION_UNFINISHED_TEACHER_UNSUPPORTED')
            reliable=False;inp=event['bound']['input_upper_bound'];out=event['bound']['output_hard_cap']
            pending.append(dict(reservation_id=rid,cell_id=cell,input_tokens=inp,output_tokens=out,
                                classification='USER_STOP_UNKNOWN_BILLING_FULL_BOUND_NOT_NETWORK_FAILURE'))
        usage['input_tokens']+=inp;usage['output_tokens']+=out
        usage[role+'_input_tokens']+=inp;usage[role+'_output_tokens']+=out
        usage['supplier_reported_tokens']+=(inp+out) if reliable else 0
        usage['fallback_usage_calls']+=not reliable
        if record and not record.get('error_category'):
            logical=record['logical_identity']; ordinal=logical['ordinal']
            if role=='reflection':
                from .generation import audit_optimizer_request
                audit_optimizer_request(parent,cell,record)
                if ordinal>completed[cell]:
                    from ...search.candidate_contract import parse_candidate_edit
                    from ...search.generation_failures import GeneratedOutputFailure
                    match=re.fullmatch(r'round_(\d+)_proposal_([01])_legal_candidate_draw_([1-4])',logical['stage'])
                    require(match is not None,'CONTINUATION_PARTIAL_REFLECTION_STAGE')
                    number,slot,draw=match.groups();stem=f'round_{number}_proposal_{slot}'
                    op=read_json(parent/cell/(stem+'_opportunity.json'))
                    g=read_json(parent/cell/(stem+f'_generation_{draw}.json'))
                    errors=[];child=None
                    try:child=parse_candidate_edit(record['response']['text'],parent=SystemPrompt.from_dict(op['parent_prompt']),
                        finish_reason=record['response'].get('finish_reason')).prompt
                    except GeneratedOutputFailure as exc:errors=[str(exc)]
                    require(g['errors']==errors and g['candidate']==(child.to_dict() if child is not None else None),
                            'CONTINUATION_PARTIAL_EDIT_RECEIPT')
            if ordinal>completed[cell]:
                old_partial.append((cell,role,logical['stage'],rid))
                if role=='solver':paid.append(dict(cell_id=cell,reservation_id=rid,
                    receipt_sha256=file_hash(parent/'provider_receipts'/f'{rid}.json')))
    # Cache hit counts cannot be inferred as zero just because process RAM is gone.
    # Each scored access is logged; repeated complete identity is exactly one hit.
    seen={}
    for line in (parent/'provider_receipts/solver_cache_access.jsonl').read_text(encoding='utf-8').splitlines():
        access=json.loads(line); key=access['request_sha256'];cell=access['cell_id']
        require(key not in seen or seen[key]==access['observation_sha256'],'CONTINUATION_TERMINAL_OBSERVATION_CHANGED')
        counts[cell]['cache_hits']+=key in seen;seen[key]=access['observation_sha256']
    partials={}
    for cell,n in completed.items():
        partial_usage=_difference(counts[cell],prefix_usage[cell])
        if n==20:require(not any(partial_usage.values()),'CONTINUATION_COMPLETED_CELL_COST')
        else:
            folder=parent/cell; arts={p.stem:read_json(p) for p in folder.glob(f'round_{n+1}_*.json')}
            require(f'round_{n+1}_module' in arts,'CONTINUATION_PARTIAL_MODULE_REQUIRED')
            # This explicitly bounded amendment supports the observed single-focus
            # C2 evaluation interruption. Other cut points fail before transport.
            module=arts[f'round_{n+1}_module']
            require(cell=='C2_seed42_repeat02' and n==12 and
                    len(module['selection']['selected_pattern_ids'])==1,'CONTINUATION_PARTIAL_BOUNDARY')
            gs=[v for k,v in arts.items() if re.fullmatch(f'round_{n+1}_proposal_0_generation_[1-4]',k)]
            require(gs and sorted(g['draw'] for g in gs)==list(range(1,max(g['draw'] for g in gs)+1)) and
                    max(gs,key=lambda g:g['draw'])['disposition']=='GENERATED' and
                    not any(k.endswith('_result') for k in arts),'CONTINUATION_GENERATION_BOUNDARY')
            diagnostic=_zero(); generation=_zero();evaluation=_zero()
            # Partition paid partial costs by role. Only one proposal is present.
            for role,target in [('solver',evaluation),('reflection',generation),('pattern_gradient',diagnostic),('pattern_cluster',diagnostic)]:
                target[role]=partial_usage[role]
                for k in ('input_tokens','output_tokens'):target[role+'_'+k]=partial_usage[role+'_'+k]
            for role,target in [('solver',evaluation),('reflection',generation),('pattern_gradient',diagnostic),('pattern_cluster',diagnostic)]:
                for rid,event in reserves.items():
                    if event['role']!=role or event['attempt_id']!=PARENT_SCOPE+'_train_'+cell:continue
                    record=_receipt(parent,scope,event) if (parent/'provider_receipts'/f'{rid}.json').exists() else None
                    if record and record['logical_identity']['ordinal']<=n:continue
                    charge=charges.get(rid); reliable=bool(charge and charge['reliable_usage'])
                    inp=charge['input_tokens'] if charge else event['bound']['input_upper_bound']
                    out=charge['output_tokens'] if charge else event['bound']['output_hard_cap']
                    target['provider_calls']+=1; target['input_tokens']+=inp;target['output_tokens']+=out
                    target['supplier_reported_tokens']+=(inp+out) if reliable else 0;target['fallback_usage_calls']+=not reliable
                    failed=bool(record and record.get('error_category'))
                    target['network_failed_calls']+=failed;target['timeout_retries']+=failed
            evaluation['cache_hits']=partial_usage['cache_hits']
            require(_add(_add(diagnostic,generation),evaluation)==partial_usage,'CONTINUATION_PARTIAL_COST_PARTITION')
            partials[cell]=dict(ordinal=n+1,usage=partial_usage,diagnostic_usage=diagnostic,
                generation_usage={'0':generation},evaluation_usage={'0':evaluation})
    total=_zero()
    for usage in counts.values():total=_add(total,usage)
    return dict(identity=PREP_SCHEMA,parent_run=PARENT,parent_inventory=gov.inventory(parent),
        request_identity_scope=PARENT_SCOPE,parent_source=PARENT_SOURCE,inherited_usage=total,
        cell_usage=counts,completed_rounds=completed,checkpoint_usage=prefix_usage,
        partials=partials,solver_paid_receipts=paid,terminal_observation_hashes=seen,unresolved_stop_reservations=pending,
        parent_ledger_head=prev,teacher_response_replay=False)


def _scope(root,destination):
    from . import governance as gov, proposal_governance as pg
    validate_manifest(read_json(root/MANIFEST))
    original=read_json(root/PARENT/'scope.json');audit=read_json(destination/'parent_audit.json')
    require(audit['identity']==PREP_SCHEMA and audit['parent_run']==PARENT,'CONTINUATION_AUDIT_IDENTITY')
    value=deepcopy(original)
    for key in ('scope_identity','source_sha','source_hashes','execution','runtime_identity'):value.pop(key,None)
    hashes=gov.source_snapshot(root)
    hashes.update({p:file_hash(root/p) for p in (MANIFEST,PROTOCOL,pg.MANIFEST,pg.PROTOCOL)})
    value.update(study_id='single_prompt_pattern_scheduling_continuation_v2',source_sha=gov._git(root,'rev-parse','HEAD'),
        source_hashes=hashes,runtime_identity=gov.runtime_identity(),manifest_sha256=file_hash(root/MANIFEST),
        execution_concurrency=deepcopy(CONCURRENCY),continuation=audit,
        parent_audit_sha256=file_hash(destination/'parent_audit.json'),
        execution=dict(prep_path=destination.relative_to(root).as_posix(),
            run_root=(destination.parent/'train_run').relative_to(root).as_posix(),
            command=[gov.INTERPRETER,'scripts/run_experiment.py','--prep',destination.relative_to(root).as_posix(),
                '--run-root',(destination.parent/'train_run').relative_to(root).as_posix(),'--execute']))
    value['test_plan']['solver_concurrency']=96
    return dict(value,scope_identity=digest(value))


def handoff(root,destination,scope):
    from . import proposal_governance as pg
    value=pg.handoff(root,destination,scope)
    value['protocol'].update(name='Immutable paid-prefix scheduling concurrency continuation',version='V2',
        manifest_path=MANIFEST,manifest_sha256=file_hash(root/MANIFEST),protocol_sha256=file_hash(root/PROTOCOL),
        preregistration_path=PROTOCOL,preregistration_sha256=file_hash(root/PROTOCOL))
    value['execution'].update(concurrency_amendment=deepcopy(CONCURRENCY),train_solver_concurrency=96,
        gradient_concurrency=16,test_solver_concurrency=96,local_grading_concurrency=16)
    value['resume']='audited_completed_prefix_saved_diagnosis_edit_exact_solver_paid_draws'
    return value


def prepare(root,destination):
    from . import governance as gov
    destination=gov.private_path(root,destination);gov.require_selected_runtime(root)
    require(not destination.exists() and not gov._git(root,'status','--porcelain','--untracked-files=no'),
            'CONTINUATION_FRESH_CLEAN_SOURCE')
    audit=audit_parent(root);destination.mkdir(parents=True)
    atomic_write_json(destination/'parent_audit.json',audit);scope=_scope(root,destination)
    require(not set(scope['source_hashes'])-set(gov._git(root,'ls-files').splitlines()),'CONTINUATION_UNTRACKED_SOURCE')
    for name,value in [('prep',dict(schema_version=PREP_SCHEMA,scope=scope)),
        ('authorization',gov.authorization_template(scope)),('execution_plan',gov.execution_plan_template(scope))]:
        atomic_write_json(destination/(name+'.json'),value)
    atomic_write_json(destination/'handoff.json',handoff(root,destination,scope))
    return dict(status='FROZEN_NOT_AUTHORIZED',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],
        inherited_opportunities=sum(audit['completed_rounds'].values()),real_api_calls=0)


def validate_prep(root,directory,*,authorized=False):
    from . import governance as gov
    directory=gov.private_path(root,directory);gov.require_selected_runtime(root)
    value=read_json(directory/'prep.json');scope=value['scope']
    require(value['schema_version']==PREP_SCHEMA and set(value)=={'schema_version','scope'} and
            not gov._git(root,'status','--porcelain','--untracked-files=no') and scope==_scope(root,directory),
            'CONTINUATION_FROZEN_IDENTITY')
    expected=handoff(root,directory,scope);actual=read_json(directory/'handoff.json')
    require(set(expected)==set(actual) and all(actual[k]==v for k,v in expected.items() if k not in ('READY_TO_RUN','status')),
            'CONTINUATION_HANDOFF')
    require(read_json(directory/'execution_plan.json')==gov.execution_plan_template(scope),'CONTINUATION_EXECUTION_PLAN')
    auth=read_json(directory/'authorization.json');gov.validate_authorization(scope,auth,required=authorized)
    if authorized:require(actual['READY_TO_RUN'] is True and auth['user_authorization_evidence_sha256']==
                          file_hash(directory/'user_authorization_evidence.json'),'CONTINUATION_EXACT_HUMAN_GRANT')
    execution_concurrency(scope)
    return scope


def validate_partial_resume(resume,cell):
    partial=resume['partial'];n=partial['ordinal']
    require(cell['proposal_mode']=='parallel' and n==len(resume['checkpoints']),
            'CONTINUATION_PARTIAL_KERNEL_SCOPE')
    require(resume.get('fresh_continuation_identity')==PREP_SCHEMA,'CONTINUATION_PARTIAL_NOT_AUTHORIZED')
    module=resume['artifacts'][f'round_{n}_module']
    require(len(module['selection']['selected_pattern_ids'])==1,'CONTINUATION_PARTIAL_SINGLE_PROPOSAL')


def restore_prefix(root,scope,cell,folder):
    old=scope['continuation'];count=old['completed_rounds'].get(cell['cell_id'])
    if count is None:return None
    source=root/old['parent_run']/cell['cell_id'];artifacts={}
    for path in source.glob('*.json'):
        match=re.fullmatch(r'(?:checkpoint_(\d+)|round_(\d+)_.+)\.json',path.name)
        if match:
            shutil.copyfile(path,folder/path.name);artifacts[path.stem]=read_json(path)
    return dict(checkpoints=[artifacts[f'checkpoint_{n}'] for n in range(count+1)],
        historical_usage=deepcopy(old['checkpoint_usage'][cell['cell_id']]),artifacts=artifacts,
        partial=deepcopy(old['partials'].get(cell['cell_id'],{})),fresh_continuation_identity=PREP_SCHEMA)


class SchedulingResumeBroker(AblationBroker):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.paid_solver={};self.used_solver=set();self.expected_terminal={}

    def prediction(self,*args,**kwargs):
        value=super().prediction(*args,**kwargs)
        expected=self.expected_terminal.get(value['request_sha256'])
        require(expected is None or digest(value)==expected,'CONTINUATION_PRIOR_SCORE_CHANGED')
        return value

    def complete(self,*,role,split,stage,messages,capacity=None):
        request,key=self._request_identity(role=role,split=split,messages=messages,capacity=capacity)
        slot=(self.cell_id,stage,hashlib.sha256(serialized_request(request)).hexdigest())
        if role=='solver' and slot in self.paid_solver:
            with self.lock:
                require(slot not in self.used_solver,'CONTINUATION_PAID_DRAW_DUPLICATE')
                self.dispatch_guard(self.cell_id);self.used_solver.add(slot)
                entry=self.paid_solver[slot]
                require(entry['record']['request']==request,'CONTINUATION_PAID_REQUEST_CHANGED')
                append_jsonl(self.receipts.directory/'restored_solver_draws.jsonl',dict(cell_id=self.cell_id,
                    stage=stage,request_sha256=key,parent_reservation_id=entry['reservation_id'],
                    parent_receipt_sha256=entry['receipt_sha256'],new_provider_call=False))
            return self._validated_response(deepcopy(entry['record']['response']),request,role,key,reservation(request))
        return super().complete(role=role,split=split,stage=stage,messages=messages,capacity=capacity)


def broker_type(scope):
    return SchedulingResumeBroker if scope.get('execution_concurrency') and scope['phase']=='train' else AblationBroker


def hydrate_solver(root,scope,broker,optimize,validation,directory):
    old=scope['continuation'];rows={split:{r['stable_example_id']:r for r in values}
        for split,values in [('optimize',optimize),('validation',validation)]};imported={}
    broker.expected_terminal=deepcopy(old['terminal_observation_hashes'])
    for cell in scope['plan']['cells']:
        count=old['completed_rounds'].get(cell['cell_id'])
        if count is None:continue
        broker.bind_cell(cell,scope['plan']);folder=root/old['parent_run']/cell['cell_id']
        for n in range(count+1):
            cp=read_json(folder/f'checkpoint_{n}.json')
            profiles=[(cp['prompt'],cp['optimize'],cp['validation'])]
            if n:profiles.extend((p['candidate_prompt'],p['candidate_optimize'],p['candidate_validation'])
                for p in cp['metric']['proposals'] if p['generation_status']=='MEASURED')
            for prompt_value,op,va in profiles:
                prompt=SystemPrompt.from_dict(prompt_value)
                for split,values in [('optimize',op),('validation',va)]:
                    for ident,observation in values.items():
                        row=rows[split][ident]
                        broker.example_context.identity=dict(example_id=ident,prompt_sha256=prompt.prompt_hash,
                            problem_sha256=digest(row['content']['problem']),scoring_reference_sha256=digest(row['reference_final_answer']),
                            scoring_contract='PINNED_MATH_FIRST_VALID_FOUR_DRAWS_V1')
                        _,key=broker._request_identity(role='solver',split=split,messages=solver_messages(prompt,row['content']['problem']))
                        del broker.example_context.identity
                        require(key==observation['request_sha256'] and (key not in broker.cache or broker.cache[key]==observation),
                                'CONTINUATION_SOLVER_CACHE_IDENTITY')
                        require(old['terminal_observation_hashes'].get(key)==digest(observation),
                                'CONTINUATION_TERMINAL_CACHE_EVIDENCE')
                        broker.cache[key]=deepcopy(observation);imported[key]=digest(observation)
    for entry in old['solver_paid_receipts']:
        path=root/old['parent_run']/'provider_receipts'/(entry['reservation_id']+'.json')
        require(file_hash(path)==entry['receipt_sha256'],'CONTINUATION_PAID_RECEIPT_CHANGED')
        record=read_json(path)['record'];request=record['request']
        slot=(entry['cell_id'],record['logical_identity']['stage'],hashlib.sha256(serialized_request(request)).hexdigest())
        require(slot not in broker.paid_solver,'CONTINUATION_DUPLICATE_PAID_DRAW')
        broker.paid_solver[slot]=dict(entry,record=record)
    atomic_write_json(directory/'solver_cache_import.json',dict(identity=PREP_SCHEMA,
        imported_count=len(imported),observations=imported,paid_solver_draws=len(broker.paid_solver),teacher_responses_replayed=0))
