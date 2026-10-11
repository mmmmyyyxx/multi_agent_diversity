"""A separately granted, search-closed paired Validation phase; Test stays sealed."""
from copy import deepcopy
from dataclasses import dataclass
import re

from ..search.schemas import SearchContractError
from ..persistence.durable_io import atomic_write_json,append_jsonl,read_json
from .startup_identity import canonical_sha256

PREP_SCHEMA='unified_post_freeze_validation_prep_v1'
ACCOUNTING_SCOPE='FRESH_V25_PAIRED_VALIDATION_6M_V1'
BOUND=dict(identity='PAIRED_POST_FREEZE_VALIDATION_RESOURCE_BOUND_V1',
    max_opportunities=0,token_ceiling=6_000_000,scientific_stopper='NOT_SEARCH',guarantees_saturation=False)
POLICY=dict(identity='CURRENT_PAIRED_VALIDATION_300_V1',count=300,
    teams=['FROZEN_INITIAL_B5','LAST_COMMITTED_FINAL_TEAM'],member_slots=5,
    ordering='EXAMPLE_MEMBER_INITIAL_FINAL',paired_cache='EXACT_REQUEST_SAME_MEMBER_SLOT_SHARED_REALIZATION',
    selection='ALL_FROZEN_VALIDATION_METADATA',search_feedback=False,test_access='sealed',
    resource_truncation='NO_PARTIAL_SCORE_OR_OUTCOME_DEPENDENT_SUBSET',
    analysis=dict(identity='PAIRED_BINARY_VALIDATION_ANALYSIS_V1',
        primary='VoteAcc_final_minus_initial',secondary=['OracleAcc','member_accuracy','validity'],
        exact_test='TWO_SIDED_MCNEMAR_BINOMIAL',bootstrap_resamples=10_000,bootstrap_seed=86,
        confidence_level=0.95,unit='VALIDATION_EXAMPLE',scope='ONE_SEED_ONE_TEAM_NO_COMPONENT_CAUSAL_CLAIM'))


def require_validation_contract(c):
    from .cost_accounting import require_resource_control
    if (c.get('post_freeze_validation_policy')!=POLICY or c.get('operational_pilot')!=BOUND
            or c.get('accounting_scope_policy')!=ACCOUNTING_SCOPE
            or not isinstance(c.get('validation_scope_identity'),str)
            or len(c['validation_scope_identity'])!=64):
        raise SearchContractError('POST_FREEZE_VALIDATION_CAPABILITY_NOT_BOUND')
    require_resource_control(c)


def local_path(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise SearchContractError('VALIDATION_PATH_ESCAPE')
    return path


def scope(root,search_payload,*,attempt,cost_policy,search_run,token_reporting_policy=None):
    from .cost_accounting import frozen_policy,require_resource_control
    require_resource_control(dict(cost_policy=cost_policy,token_reporting_policy=token_reporting_policy))
    c=read_json(root/search_payload['manifest']['execution_binding']['path'])
    if (c['execution_phase']!='formal' or not isinstance(attempt,str)
            or re.fullmatch('[A-Za-z0-9_-]+',attempt) is None
            or attempt in {c['execution_attempt_id'],c['cache_namespace']}):
        raise SearchContractError('FRESH_FORMAL_VALIDATION_SCOPE_REQUIRED')
    if not local_path(root,search_run).is_relative_to(root/'runs'):
        raise SearchContractError('VALIDATION_PARENT_RUN_NOT_LOCAL')
    return dict(attempt_id=attempt,cache_namespace=attempt,phase='post_freeze_paired_validation_only',
        arm='A4',seed=c['execution_seed'],source_sha=search_payload['manifest']['source_sha'],
        parent_search_attempt_id=c['execution_attempt_id'],parent_search_startup_sha256=search_payload['startup_identity_sha256'],
        search_run=search_run,method=c['method_identity'],models=c['models'],
        provider_routing_policy=c['provider_routing_policy'],roles=['solver'],
        cumulative_token_ceiling=BOUND['token_ceiling'],cost_policy=frozen_policy(cost_policy),
        **({'token_reporting_policy':deepcopy(token_reporting_policy)} if token_reporting_policy is not None else {}),
        policy=deepcopy(POLICY),initial_team_artifact_sha256=c['initial_team_artifact_sha256'],
        split_sha256=c['split_manifest_sha256'],validation_membership_sha256=c['membership_hashes']['validation'],
        accounting_directory='runs/'+attempt+'/accounting',
        validation_calls='POST_SEARCH_ONLY',test_calls=0,search_resume_authorized=False)


def prepare(root,search_prep,destination,*,attempt,cost_policy,search_run,token_reporting_policy=None):
    from .unified_execution import validate_prep
    search=validate_prep(root,search_prep)
    if destination.exists():raise SearchContractError('FRESH_VALIDATION_PREP_REQUIRED')
    relative=search_prep.resolve().relative_to(root.resolve()).as_posix()
    frozen_scope=scope(root,search,attempt=attempt,cost_policy=cost_policy,search_run=search_run,
        token_reporting_policy=token_reporting_policy)
    payload=dict(schema_version=PREP_SCHEMA,parent_search_prep=relative,
        source_identity=search['source_identity'],scope=frozen_scope)
    payload['startup_identity_sha256']=canonical_sha256(payload)
    destination.mkdir(parents=True)
    atomic_write_json(destination/'prep.json',payload)
    atomic_write_json(destination/'authorization.json',dict(explicit_user_authorized=False,
        single_use=True,consumed=False,scope=frozen_scope,startup_identity_sha256=payload['startup_identity_sha256']))
    return payload


def validate_prep(root,prep,*,require_authorized=False):
    from .unified_execution import validate_prep as validate_search,consumption_path
    payload=read_json(prep/'prep.json')
    if (payload.get('schema_version')!=PREP_SCHEMA or payload.get('startup_identity_sha256')!=
            canonical_sha256({k:v for k,v in payload.items() if k!='startup_identity_sha256'})):
        raise SearchContractError('VALIDATION_STARTUP_IDENTITY_MISMATCH')
    search=validate_search(root,local_path(root,payload['parent_search_prep']))
    s=payload['scope']
    if (s!=scope(root,search,attempt=s['attempt_id'],cost_policy=s['cost_policy'],search_run=s['search_run'],
            token_reporting_policy=s.get('token_reporting_policy'))
            or payload['source_identity']!=search['source_identity']):
        raise SearchContractError('VALIDATION_FROZEN_SCOPE_MISMATCH')
    if require_authorized:
        auth=read_json(prep/'authorization.json')
        if (auth.get('explicit_user_authorized') is not True or auth.get('single_use') is not True
                or auth.get('consumed') is not False or auth.get('scope')!=s
                or auth.get('startup_identity_sha256')!=payload['startup_identity_sha256']
                or (prep/'authorization_consumed.json').exists() or consumption_path(root,s).exists()):
            raise SearchContractError('EXACT_SINGLE_USE_VALIDATION_AUTHORIZATION_REQUIRED')
    return payload,search


def closed_search(root,search_run,search_payload):
    from .autonomous_math import file_sha,ledger_policy
    from .token_accounting import audit_journal,OperationalAbort
    from ..persistence.provider_receipts import ProviderResponseReceipts
    run=local_path(root,search_run);lifecycle=read_json(run/'lifecycle.json')
    receipts=[p for p in (run/'SEARCH_COMPLETE_RECEIPT.json',run/'SEARCH_CLOSED_RECEIPT.json') if p.is_file()]
    if lifecycle.get('status')!='EXECUTION_COMPLETE' or len(receipts)!=1:
        raise SearchContractError('VALIDATION_REQUIRES_INTEGRITY_COMPLETE_SEARCH_CLOSURE')
    path=receipts[0];receipt=read_json(path)
    if (lifecycle.get('closure_receipt_sha256')!=file_sha(path)
            or receipt.get('search_closed_forever') is not True
            or receipt.get('attempt_id')!=search_payload['scope']['attempt_id']
            or receipt.get('startup_identity_sha256')!=search_payload['startup_identity_sha256']
            or receipt.get('source_sha')!=search_payload['manifest']['source_sha']
            or receipt.get('final_team_sha256')!=file_sha(run/'final_team_private.json')
            or receipt.get('execution_summary_sha256')!=file_sha(run/'execution_summary.json')
            or receipt.get('raw_inventory_sha256')!=file_sha(run/'raw_evidence_inventory.json')):
        raise SearchContractError('VALIDATION_PARENT_CLOSURE_HASH_MISMATCH')
    for record in read_json(run/'raw_evidence_inventory.json')['files']:
        if file_sha(local_path(run,record['path']))!=record['sha256']:
            raise SearchContractError('VALIDATION_PARENT_EVIDENCE_HASH_MISMATCH')
    summary=read_json(run/'execution_summary.json')
    if (summary['accounting']['reserved_inflight']!=0 or summary.get('validation_calls')!=0
            or summary.get('test_calls')!=0):
        raise SearchContractError('VALIDATION_PARENT_LEDGER_OR_ACCESS_MISMATCH')
    c=read_json(root/search_payload['manifest']['execution_binding']['path'])
    try:
        view,events=audit_journal(local_path(root,c['token_ledger_directory']),
            task_sha256=c['task_authorization_sha256'],policy=ledger_policy(c),cost_policy=c.get('cost_policy'))
    except OperationalAbort as exc:raise SearchContractError('PARENT_JOURNAL_AUDIT_FAILED:'+str(exc)) from None
    if view!=summary['accounting']:
        raise SearchContractError('PARENT_JOURNAL_SUMMARY_MISMATCH')
    receipts=ProviderResponseReceipts.__new__(ProviderResponseReceipts)
    receipts.directory=run/'provider_response_receipts_private';receipts.attempt_id=c['execution_attempt_id']
    receipts.startup=search_payload['startup_identity_sha256']
    for charge in (event for event in events if event['kind']=='CHARGE'):
        recorded=charge.get('response_receipt')
        if not isinstance(recorded,dict):raise SearchContractError('PARENT_RESPONSE_RECEIPT_MISSING')
        receipts.read(charge['reservation_id'])
        saved=read_json(receipts.directory/(charge['reservation_id']+'.json'))
        if saved['integrity_sha256']!=recorded['integrity_sha256']:
            raise SearchContractError('PARENT_RESPONSE_RECEIPT_MISMATCH')
    return receipt


@dataclass(frozen=True)
class ValidationAccess:
    scope_identity: str
    split_sha256: str
    membership_sha256: str
    parent_closure_sha256: str

    def require(self,reader):
        from ..benchmarks.data_freeze import digest
        selected=[r for r in reader.members if r['project_split']=='validation']
        if (any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                for v in (self.scope_identity,self.split_sha256,self.membership_sha256,self.parent_closure_sha256))
                or reader.manifest_sha256!=self.split_sha256
                or reader.metadata('validation')['membership_hash']!=self.membership_sha256
                or digest([r['stable_example_id'] for r in selected])!=self.membership_sha256
                or len(selected)!=POLICY['count']):
            raise SearchContractError('POST_FREEZE_VALIDATION_MEMBERSHIP_MISMATCH')


async def execute(root,prep,run_root):
    from .autonomous_math import consume,create_transport,initial_prompts,checked_json,file_sha
    from .token_accounting import TokenLedger,POLICY_V26_VALIDATION_6M
    from ..benchmarks.math_domain_binding import execution_binding
    from ..search.provider_runtime import RequestBroker,BenchmarkSolver
    from ..search.system_prompt import SystemPrompt,team_identity
    from ..search.solver_execution import bounded_ordered_map
    from ..search.scientific_aggregation import EquivalencePluralityAggregation
    from ..benchmarks.protocols import protocol_input
    from ..persistence.provider_receipts import ProviderResponseReceipts
    from ..persistence.exact_output_cache import DurableExactOutputCache
    from .paired_validation_analysis import measure,analyze
    payload,search=validate_prep(root,prep,require_authorized=True)
    s=payload['scope'];receipt=closed_search(root,s['search_run'],search)
    original=read_json(root/search['manifest']['execution_binding']['path'])
    binding=execution_binding(root,original)
    if binding.blockers():raise SearchContractError('VALIDATION_PARENT_BINDING_HOLD')
    parent=local_path(root,s['search_run']);final=read_json(parent/'final_team_private.json')
    teams=(initial_prompts(root,original),tuple(SystemPrompt.from_dict(p) for p in final['prompts']))
    if len(teams[1])!=5 or team_identity(teams[1])!=final['state_id']:
        raise SearchContractError('VALIDATION_FINAL_TEAM_SEAL_MISMATCH')
    c=deepcopy(original);calls=4*POLICY['count']*5*2
    c.update(execution_phase='validation',accounting_scope_policy=ACCOUNTING_SCOPE,
        operational_pilot=deepcopy(BOUND),post_freeze_validation_policy=deepcopy(POLICY),
        validation_scope_identity=canonical_sha256(s),execution_attempt_id=s['attempt_id'],
        cache_namespace=s['cache_namespace'],cost_policy=s['cost_policy'])
    c.pop('token_reporting_policy',None)
    if 'token_reporting_policy' in s:c['token_reporting_policy']=deepcopy(s['token_reporting_policy'])
    c['provider_bounds']=dict(original['provider_bounds'],solver_calls=calls,reflection_calls=0,
        pattern_gradient_calls=0,pattern_cluster_calls=0,pattern_calls=0,successful_provider_calls=calls,
        transport_attempts=calls*(c['decoding']['transport_retries']+1))
    budget=TokenLedger(local_path(root,s['accounting_directory']),task_sha256=canonical_sha256(s),
        policy=POLICY_V26_VALIDATION_6M,cost_policy=s['cost_policy'],best_effort_snapshots=True)
    client=None;broker=None
    consumed=False
    try:
        consume(root,prep,run_root,payload)
        consumed=True
        checked_json(run_root/'SEALED_EVALUATION_TEAMS_PRIVATE.json',dict(
            teams=[[p.to_dict() for p in team] for team in teams],parent_closure=receipt,
            scope_identity=canonical_sha256(s),no_search_resume=True))
        access=ValidationAccess(canonical_sha256(s),s['split_sha256'],s['validation_membership_sha256'],canonical_sha256(receipt))
        reader=binding.reader()
        rows=reader.validation_rows(access)
        benchmark=binding.benchmark();aggregation=EquivalencePluralityAggregation()
        items=tuple(protocol_input('math',r['stable_example_id'],r['content'],benchmark.output_contract,
            protocol=benchmark.protocol) for r in rows)
        transport,client=create_transport(c)
        cache=DurableExactOutputCache(run_root/'resolved_output_cache',dict(
            execution_attempt_id=s['attempt_id'],cache_namespace=s['cache_namespace'],
            startup_identity_sha256=payload['startup_identity_sha256'],source_sha=s['source_sha'],
            authorization_sha256=canonical_sha256(read_json(run_root/'consumed_authorization.json')),
            binding_sha256=canonical_sha256(c),generation_policy_sha256=canonical_sha256(c['solver_decoding_policy']),
            recovery_policy_sha256=canonical_sha256(c['invalid_recovery_policy'])))
        broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=s['seed'],token_ledger=budget,
            validation_only=True,durable_cache=cache,
            response_receipts=ProviderResponseReceipts(run_root/'provider_response_receipts_private',
                attempt_id=s['attempt_id'],startup_identity_sha256=payload['startup_identity_sha256']),
            ledger_writer=lambda value:append_jsonl(run_root/'ledger_private.jsonl',value),
            raw_writer=lambda value:append_jsonl(run_root/'provider_trace_private.jsonl',value))
        jobs=tuple((i,m,a) for i in range(len(rows)) for m in range(5) for a in range(2))
        def solve(job):
            i,m,a=job;solver=BenchmarkSolver(benchmark,broker);solver.observe_member(m)
            value=solver.solve(teams[a][m],items[i],stage='post_freeze_paired_validation',split='validation')
            return i,m,a,value
        results=bounded_ordered_map(solve,jobs,c['solver_execution_policy']['solver_max_concurrency'])
        profiles={(i,m,a):value for i,m,a,value in results}
        measurements=[]
        for i,row in enumerate(rows):
            outputs=tuple(tuple(profiles[i,m,a] for m in range(5)) for a in range(2))
            if any(teams[0][m]==teams[1][m] and outputs[0][m]!=outputs[1][m] for m in range(5)):
                raise SearchContractError('IDENTICAL_PAIRED_MEMBER_EVIDENCE_MISMATCH')
            measurements.append(tuple(measure(benchmark,aggregation,items[i],out,row['reference_final_answer']) for out in outputs))
        checked_json(run_root/'paired_profiles_private.json',dict(results=results,measurements=measurements))
        if budget.reserved or broker.physical_inflight or broker.requests_inflight:
            raise SearchContractError('VALIDATION_UNRESOLVED_ACCOUNTING')
        summary=dict(status='EXECUTION_COMPLETE',policy=POLICY,analysis=analyze(measurements),
            accounting=budget.view(),provider_usage=broker.usage,validation_raw_reads=len(rows),
            validation_search_feedback=0,test_raw_reads=0,test_calls=0,
            parent_search_scientific_complete=receipt['scientific_complete'])
        client.close();client=None;budget.close()
        checked_json(run_root/'execution_summary.json',summary)
        checked_json(run_root/'lifecycle.json',dict(status='EXECUTION_COMPLETE',scope_identity=canonical_sha256(s),
            summary_sha256=file_sha(run_root/'execution_summary.json')))
        return summary
    except BaseException as exc:
        for key in tuple(budget.inflight):budget.reconcile(key,None,outcome='VALIDATION_ABORT_FULL_CHARGE')
        if consumed:
            checked_json(run_root/'execution_summary.json',dict(status='NOT_EVALUABLE',
                error_category=type(exc).__name__,accounting=budget.view(),
                provider_usage=broker.usage if broker is not None else {},partial_scores_published=False,test_calls=0))
            checked_json(run_root/'lifecycle.json',dict(status='EXECUTION_ABORTED',error_category=type(exc).__name__))
        raise
    finally:
        try:
            if client is not None:client.close()
        finally:budget.close()
