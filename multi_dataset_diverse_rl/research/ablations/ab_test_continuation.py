"""Governed paid A/B Test continuation after exact local scorer repair."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib,json,time
from ...benchmarks.data_freeze import digest,file_hash
from ...benchmarks.math_domain_v2 import require_scorable,worker_deadline
from ...benchmarks.math_power_certificate import POLICY
from ...benchmarks.math_power_domain import power_matrix
from ...governance.token_accounting import TokenLedger,POLICY_V27_FORMAL_UNCAPPED,audit_journal,serialized_request
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...persistence.durable_io import read_json,atomic_write_json,append_jsonl
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt
from ..single_prompt_governance import StudyHeldoutAccess
from . import completed_ab_test as ab,governance as gov
from .test_resume import audit_new_run
from .test_resume_runtime import draw_key,evaluator_timeout
from .test_certified import TestResumeBroker as PreviousBroker
from .analysis import summarize

MANIFEST='experiments/manifests/single_prompt_completed_ab_test_v15.json'
PROTOCOL='docs/design/SINGLE_PROMPT_COMPLETED_AB_TEST_V15.md'
PREP_SCHEMA='single_prompt_completed_ab_test_prep_v15'


def operation_policy():
    p=ab.operation_policy()
    p.update(identity='AB_PAID_TEST_EXACT_POWER_AND_STALL_DIAGNOSIS_V15',
        certified_rejection_policy=deepcopy(POLICY),scoring_worker='math_power_worker',
        local_retry_limit=3,local_stall='save_private_worker_trace_then_raise_MATH_LOCAL_SCORING_STALLED',
        solver_prefix='exact_cell_stage_request_bytes_and_original_request_namespace',
        inherited_completed_cell='retain_every_verified_completed_AB_cell',
        inherited_partial_cell='reconstruct_exact_paid_draws_and_verify_every_prior_terminal_hash',
        inherited_successful_calls='audited_parent_ledger',new_call_ceiling='7200_minus_audited_parent_successful_calls')
    return p


def validate_manifest(root):
    m=read_json(root/MANIFEST)
    ab.require(m['schema_version']=='single_prompt_completed_ab_test_manifest_v15'
        and m['operation_policy']==operation_policy() and m['protocol_path']==PROTOCOL,'AB_CONTINUATION_MANIFEST')
    return m


def audit_test_parent(root):
    m=validate_manifest(root);base=gov.private_path(root,root/m['parent_root']);run=base/'test_run'
    ab.require(file_hash(root/m['parent_pin_path'])==m['parent_pin_sha256'],'AB_CONTINUATION_PIN_CHANGED')
    pin=read_json(root/m['parent_pin_path'])
    ab.require(gov.inventory(base)==pin['parent_inventory'],'AB_CONTINUATION_PARENT_CHANGED')
    s=read_json(base/'prep/prep.json')['scope'];auth=read_json(base/'prep/authorization.json')
    life=read_json(run/'lifecycle.json')
    ab.require(s['scope_identity']==pin['parent_scope']==digest({k:v for k,v in s.items() if k!='scope_identity'})
        and s['source_sha']==pin['parent_source_sha'] and auth['approved'] is True and auth['test_access'] is True
        and auth['scope_identity']==s['scope_identity'] and not auth['train_access'] and not auth['validation_access']
        and auth['roles']==['solver'] and read_json(run/'scope.json')==s and read_json(run/'authorization.json')==auth
        and file_hash(base/'prep/user_authorization_evidence.json')==auth['user_authorization_evidence_sha256']
        and life['status']=='RUNNING' and life['cells_completed']==pin['completed_cells']
        and pin['no_inflight_at_stop'] and pin['max_same_context_retry']>=3 and not (run/'terminal.json').exists(),
        'AB_CONTINUATION_PARENT_IDENTITY')
    view,events=audit_journal(run/'ledger',task_sha256=s['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
    reserves={e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    charges=[e for e in events if e['kind']=='CHARGE']
    ab.require(view==pin['token_ledger'] and len(reserves)==len(charges)==pin['physical_calls'],'AB_CONTINUATION_PARENT_LEDGER')
    usage=dict(provider_calls=len(reserves),input_tokens=view['input_tokens'],output_tokens=view['output_tokens'],
        network_failed_calls=pin['network_failed_calls'])
    audit_new_run(run,s,usage)
    cells={s['scope_identity']+'_test_'+c['cell_id']:c for c in s['plan']['cells']};prefix={}
    for e in charges:
        if e['outcome']=='PROVIDER_TIMEOUT':continue
        ab.require(e['outcome']=='PROVIDER_RESPONSE','AB_CONTINUATION_UNSUPPORTED_PARENT_FAILURE')
        reserve=reserves[e['reservation_id']];cell=cells[reserve['attempt_id']]
        store=ProviderResponseReceipts.__new__(ProviderResponseReceipts)
        store.directory=run/'provider_receipts';store.attempt_id=reserve['attempt_id'];store.startup=s['scope_identity']
        record=store.read(e['reservation_id']);key=draw_key(cell['cell_id'],reserve['stage'],record['request'])
        ab.require(key not in prefix and 'error_category' not in record,'AB_CONTINUATION_DUPLICATE_PAID_DRAW')
        prefix[key]=dict(request=record['request'],response=record['response'],reservation_id=e['reservation_id'],
            receipt_sha256=e['response_receipt']['integrity_sha256'])
    ab.require(len(prefix)==pin['successful_calls'],'AB_CONTINUATION_PREFIX_COUNT')
    observations={};members={c['cell_id']:set() for c in s['plan']['cells']}
    for line in (run/'provider_receipts/solver_cache_access.jsonl').read_bytes().splitlines():
        v=json.loads(line);key=(v['cell_id'],v['request_sha256'])
        ab.require(v['split']=='test' and v['cell_id'] in members and v['example_id'] in s['test_plan']['sample_ids']
            and v['repeat_id']==next(c['repeat_id'] for c in s['plan']['cells'] if c['cell_id']==v['cell_id'])
            and (key not in observations or observations[key]==v['observation_sha256']),'AB_CONTINUATION_TERMINAL_SCOPE')
        observations[key]=v['observation_sha256'];members[v['cell_id']].add(v['example_id'])
    ab.require(len(observations)==pin['terminal_observations'],'AB_CONTINUATION_TERMINAL_COUNT')
    completed={}
    for c in s['plan']['cells'][:pin['completed_cells']]:
        result=read_json(run/c['cell_id']/'result.json')
        ab.require(result['cell']==c and set(result['predictions'])==members[c['cell_id']]==set(s['test_plan']['sample_ids']),
            'AB_CONTINUATION_COMPLETE_RESULT')
        ab.require(all(digest(v)==observations[(c['cell_id'],v['request_sha256'])] for v in result['predictions'].values()),
            'AB_CONTINUATION_COMPLETE_HASH')
        completed[c['cell_id']]=result
    access=read_json(run/'access.json')
    ab.require(access==dict(test_ids=s['test_plan']['sample_ids'],optimize_access=False,validation_access=False),
        'AB_CONTINUATION_PARENT_ACCESS')
    return dict(scope=s,pin=pin,paid_prefix=prefix,terminal_observations=observations,completed=completed)


def scope_template(root,prep,*,snapshots=None):
    m=validate_manifest(root)
    training,parent=(ab.parent_evidence(root),audit_test_parent(root)) if snapshots is None else snapshots
    if snapshots is not None:
        ab.verify_parent_inventory(root,training[2])
        ab.verify_parent_inventory(root,{(Path(m['parent_root'])/p).as_posix():h for p,h in parent['pin']['parent_inventory'].items()})
        ab.require(file_hash(root/m['parent_pin_path'])==m['parent_pin_sha256'],'AB_CONTINUATION_PIN_CHANGED')
    s=ab.scope_template(root,prep,parent_snapshot=training);s.pop('scope_identity')
    ab.require(s['plan']==parent['scope']['plan'] and s['test_plan']==parent['scope']['test_plan']
        and s['final_prompt_sha256']==parent['scope']['final_prompt_sha256'],'AB_CONTINUATION_SOLVER_INPUT_CHANGED')
    s.update(study_id='single_prompt_completed_ab_test_v15',manifest_sha256=file_hash(root/MANIFEST),
        protocol_sha256=file_hash(root/PROTOCOL),call_ceiling=7200-parent['pin']['successful_calls'],
        local_evaluator_recovery=operation_policy(),parent_test_pin_sha256=m['parent_pin_sha256'],
        parent_test_root=m['parent_root'],parent_test_scope=parent['scope']['scope_identity'],
        inherited_test_ledger=parent['pin']['token_ledger'],inherited_physical_calls=parent['pin']['physical_calls'],
        inherited_successful_calls=parent['pin']['successful_calls'],
        continuation=dict(request_identity_scope=parent['scope']['scope_identity']))
    s['source_hashes'][MANIFEST]=file_hash(root/MANIFEST)
    return dict(s,scope_identity=digest(s))


def handoff(root,s):
    value=ab.handoff(root,s)
    value['protocol'].update(name='A/B paid Test continuation, exact power witness and stall diagnosis',version='V15',
        manifest_path=MANIFEST,preregistration_path=PROTOCOL)
    value['execution'].update(semantic_call_ceiling=s['call_ceiling'],total_successful_semantic_ceiling=7200,
        inherited_successful_calls=s['inherited_successful_calls'],
        allowed_retries=dict(local=deepcopy(s['local_evaluator_recovery']),network=deepcopy(s['network_recovery'])))
    return value


def prepare(root,prep):
    raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')


def validate_prep(root,prep,*,authorized=False,snapshots=None):
    prep=gov.private_path(root,prep);gov.require_selected_runtime(root)
    ab.require(not gov._git(root,'status','--porcelain','--untracked-files=no'),'AB_CONTINUATION_SOURCE_DIRTY')
    v=read_json(prep/'prep.json');s=v['scope']
    ab.require(v['schema_version']==PREP_SCHEMA and s==scope_template(root,prep,snapshots=snapshots),'AB_CONTINUATION_SCOPE_CHANGED')
    auth=read_json(prep/'authorization.json');expected=ab.authorization_template(s)
    mutable={'approved','test_access','user_authorization_evidence_sha256'}
    ab.require(set(auth)==set(expected) and all(auth[k]==expected[k] for k in expected if k not in mutable)
        and type(auth['approved']) is bool and type(auth['test_access']) is bool,'AB_CONTINUATION_AUTH_SCOPE')
    actual=read_json(prep/'handoff.json');wanted=handoff(root,s)
    ab.require(set(actual)==set(wanted) and all(actual[k]==v for k,v in wanted.items() if k not in ('READY_TO_RUN','status'))
        and read_json(prep/'execution_plan.json')==s['execution'],'AB_CONTINUATION_HANDOFF')
    if authorized:
        ab.require(auth['approved'] and auth['test_access'] and actual['READY_TO_RUN'] is True
            and file_hash(prep/'user_authorization_evidence.json')==auth['user_authorization_evidence_sha256'],'AB_CONTINUATION_UNAUTHORIZED')
    return s


class Broker(PreviousBroker):
    def retry_local(self,operation,context):
        for attempt in range(self.scope['local_evaluator_recovery']['local_retry_limit']):
            self.dispatch_guard(self.cell_id)
            try:
                with worker_deadline(self.scope['local_evaluator_recovery']['worker_deadline_seconds']):return operation()
            except SearchContractError as exc:
                if not evaluator_timeout(exc):raise
                self.local_retry_count+=1
                append_jsonl(self.receipts.directory/'local_evaluator_recovery.jsonl',dict(context,cell_id=self.cell_id,
                    retry_index=attempt+1,error='MATH_EVALUATOR_TIMEOUT',model_call_added=False,
                    private_worker_trace=getattr(exc,'private_worker_trace',''),
                    disposition='DIAGNOSE_STALL' if attempt==2 else 'RETRY_SAME_PAID_ANSWER'))
                if attempt==2:raise SearchContractError('MATH_LOCAL_SCORING_STALLED') from exc
                time.sleep(self.scope['local_evaluator_recovery']['backoff_seconds'][attempt])

    def score_resolved(self,resolved,row):
        raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')


def audit_continuation(run,s,broker,parent,results):
    view=audit_new_run(run,s,broker.usage_view());usage=broker.usage_view();links=set()
    for role in ('solver','reflection','pattern_gradient','pattern_cluster'):
        charged=view['by_role'].get(role,{})
        ab.require(usage[role]==charged.get('physical_attempts',0)
            and all(usage[role+'_'+k]==charged.get(k,0) for k in ('input_tokens','output_tokens')),'AB_CONTINUATION_ROLE_COST')
    path=run/'provider_receipts/solver_paid_prefix.jsonl'
    for line in path.read_bytes().splitlines() if path.exists() else ():
        v=json.loads(line);key=(v['cell_id'],v['stage'],v['provider_request_sha256'])
        entry=parent['paid_prefix'].get(key)
        ab.require(entry is not None and key not in links and entry['reservation_id']==v['parent_reservation_id']
            and entry['receipt_sha256']==v['parent_receipt_sha256'],'AB_CONTINUATION_REPLAY_LINK')
        links.add(key)
    expected={k for k in parent['paid_prefix'] if k[0] not in parent['completed']}
    ab.require(links==broker.used_prefix==expected and usage['replayed_calls']==len(links),'AB_CONTINUATION_REPLAY_COVERAGE')
    observed={}
    cache=run/'provider_receipts/solver_cache_access.jsonl'
    for line in cache.read_bytes().splitlines() if cache.exists() else ():
        v=json.loads(line);key=(v['cell_id'],v['request_sha256'])
        if key in parent['terminal_observations']:
            ab.require(parent['terminal_observations'][key]==v['observation_sha256'],'AB_CONTINUATION_PRIOR_SCORE_CHANGED')
            observed[key]=v['observation_sha256']
    for result in results:
        cell=result['cell']['cell_id']
        if cell in parent['completed']:
            ab.require(result['predictions']==parent['completed'][cell]['predictions'],'AB_CONTINUATION_COMPLETED_CHANGED')
            observed.update({(cell,v['request_sha256']):digest(v) for v in result['predictions'].values()})
        key=s['scope_identity']+'_test_'+cell;charged=view['by_attempt'].get(key,{})
        ab.require(result['usage']['provider_calls']==charged.get('physical_attempts',0)
            and all(result['usage'][k]==charged.get(k,0) for k in ('input_tokens','output_tokens')),'AB_CONTINUATION_CELL_COST')
    ab.require(observed==parent['terminal_observations'],'AB_CONTINUATION_PRIOR_TERMINAL_COVERAGE')
    ab.require(all(sum(v['usage'][k] for v in results)==usage[k] for k in usage),'AB_CONTINUATION_USAGE_TOTALS')
    ab.require(s['inherited_successful_calls']+usage['provider_calls']-usage['network_failed_calls']<=7200,'AB_CONTINUATION_COMBINED_CEILING')
    return view


def execute(root,prep,run):
    raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')
