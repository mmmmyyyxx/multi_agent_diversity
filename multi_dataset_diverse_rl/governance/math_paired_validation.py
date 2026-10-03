"""Read-only post-search MATH comparison with one shared request realization."""
import hashlib
from pathlib import Path
from .autonomous_math import consume,create_transport,file_sha,initial_prompts
from .unified_execution import validate_prep,validate_frozen_source,inventory,consumption_path
from .startup_identity import canonical_sha256
from .token_accounting import TokenLedger
from ..persistence.durable_io import atomic_write_json,read_json,append_jsonl
from ..benchmarks.math_accounting_prep import validation_rows
from ..benchmarks.math_domain_binding import MATHDomainBinding
from ..benchmarks.protocols import protocol_input
from ..search.provider_runtime import RequestBroker,BenchmarkSolver
from ..search.scientific_aggregation import EquivalencePluralityAggregation
from ..search.schemas import SearchContractError

SCHEMA='math_paired_validation_prep_v1'
POLICY=dict(identity='MATH_PAIRED_VALIDATION_V2',split='validation',count=300,team_size=5,
    initial_team='MATH_GENERIC_TEAM_SEED_V1_1',evaluator='MATH_EQUIVALENCE_V2',
    solver='qwen3-8b',thinking=False,provider='lwj',pattern=False,memory=False,
    logical_evaluations=3000,successful_provider_call_ceiling=3000,transport_attempt_ceiling=63000,
    cache='shared_initial_final_exact_request_identity',search_feedback=False,test_model_calls=0,
    metrics=['VoteAcc','Member0Acc','Member1Acc','Member2Acc','Member3Acc','Member4Acc','MeanMemberAcc','MinMemberAcc','OracleAcc'],
    paired_transitions=['vote_fixed','vote_broken','vote_net','oracle_fixed','oracle_broken','oracle_net'])


def validation_policy(contract):
    from ..benchmarks.math_prediction_validity import frozen_prediction_policy
    policy = frozen_prediction_policy(contract)
    return dict(POLICY, identity='MATH_PAIRED_VALIDATION_V3', prediction_validity_policy=policy,
        invalidity_metrics='total_by_member_and_rate_observation_only') if policy else POLICY


def search_receipt(root,prep,pilot_run):
    payload=validate_prep(root,prep,require_authorized=False)
    c=read_json(root/payload['manifest']['execution_binding']['path'])
    receipt=read_json(pilot_run/'SEARCH_COMPLETE_RECEIPT.json')
    if (c['execution_phase']!='pilot' or receipt['search_closed_forever'] is not True
            or receipt['attempt_id']!=c['execution_attempt_id'] or receipt['source_sha']!=payload['manifest']['source_sha']
            or receipt['startup_identity_sha256']!=payload['startup_identity_sha256']
            or read_json(pilot_run/'lifecycle.json')['status']!='EXECUTION_COMPLETE'
            or not (pilot_run/'consumed_authorization.json').exists()):
        raise SearchContractError('VALID_SEARCH_COMPLETE_RECEIPT_REQUIRED')
    for field,name in [('final_team_sha256','final_team_private.json'),('execution_summary_sha256','execution_summary.json'),
                       ('trajectory_sha256','trajectory_private.jsonl'),('raw_inventory_sha256','raw_evidence_inventory.json')]:
        if receipt[field]!=file_sha(pilot_run/name):
            raise SearchContractError('SEARCH_COMPLETE_HASH_MISMATCH')
    for row in read_json(pilot_run/'raw_evidence_inventory.json')['files']:
        p=(pilot_run/row['path']).resolve()
        if not p.is_relative_to(pilot_run.resolve()) or file_sha(p)!=row['sha256']:
            raise SearchContractError('SEARCH_RAW_EVIDENCE_INVENTORY_MISMATCH')
    if c['post_search_validation_policy']!=validation_policy(c):
        raise SearchContractError('VALIDATION_POLICY_MISMATCH')
    return payload,c,receipt


def prepare_validation(root,search_prep,pilot_run,destination):
    payload,c,receipt=search_receipt(root,search_prep,pilot_run)
    if destination.exists(): raise SearchContractError('FRESH_PREP_DESTINATION_REQUIRED')
    scope=dict(attempt_id=c['execution_attempt_id']+'_validation_attempt1',phase='POST_SEARCH_VALIDATION_ONLY',
        source_sha=receipt['source_sha'],search_receipt_sha256=file_sha(pilot_run/'SEARCH_COMPLETE_RECEIPT.json'),
        final_team_sha256=receipt['final_team_sha256'],split_identity=c['split_manifest_sha256'],
        policy_sha256=canonical_sha256(validation_policy(c)),models=c['models'],roles=['solver'],provider='lwj',
        successful_provider_call_ceiling=3000,transport_attempt_ceiling=63000,validation_calls=3000,test_calls=0)
    if 'solver_decoding_policy' in c:
        scope['solver_decoding_policy']=c['solver_decoding_policy']
    if 'prediction_validity_policy' in c:
        scope['prediction_validity_policy']=c['prediction_validity_policy']
    result=dict(schema_version=SCHEMA,source_identity=payload['source_identity'],
        search_prep=str(search_prep.relative_to(root)),pilot_run=str(pilot_run.relative_to(root)),scope=scope)
    result['startup_identity_sha256']=canonical_sha256(result)
    destination.mkdir(parents=True)
    atomic_write_json(destination/'validation_prep.json',result)
    atomic_write_json(destination/'authorization.json',dict(explicit_user_authorized=False,single_use=True,consumed=False,
        scope=scope,startup_identity_sha256=result['startup_identity_sha256']))
    return result


def validate_validation(root,prep,*,require_authorized):
    payload=read_json(prep/'validation_prep.json')
    if payload['schema_version']!=SCHEMA or payload['startup_identity_sha256']!=canonical_sha256({k:v for k,v in payload.items() if k!='startup_identity_sha256'}):
        raise SearchContractError('VALIDATION_STARTUP_IDENTITY_MISMATCH')
    validate_frozen_source(root,payload['source_identity'])
    search_prep=(root/payload['search_prep']).resolve();pilot_run=(root/payload['pilot_run']).resolve()
    if not search_prep.is_relative_to(root/'runs') or not pilot_run.is_relative_to(root/'runs'):
        raise SearchContractError('LOCAL_SEARCH_RECEIPT_REQUIRED')
    original,c,receipt=search_receipt(root,search_prep,pilot_run)
    expected=dict(attempt_id=c['execution_attempt_id']+'_validation_attempt1',phase='POST_SEARCH_VALIDATION_ONLY',
        source_sha=receipt['source_sha'],search_receipt_sha256=file_sha(pilot_run/'SEARCH_COMPLETE_RECEIPT.json'),
        final_team_sha256=receipt['final_team_sha256'],split_identity=c['split_manifest_sha256'],
        policy_sha256=canonical_sha256(validation_policy(c)),models=c['models'],roles=['solver'],provider='lwj',
        successful_provider_call_ceiling=3000,transport_attempt_ceiling=63000,validation_calls=3000,test_calls=0)
    if 'solver_decoding_policy' in c:
        expected['solver_decoding_policy']=c['solver_decoding_policy']
    if 'prediction_validity_policy' in c:
        expected['prediction_validity_policy']=c['prediction_validity_policy']
    if payload['scope']!=expected or payload['source_identity']!=original['source_identity']:
        raise SearchContractError('VALIDATION_SCOPE_MISMATCH')
    auth=read_json(prep/'authorization.json')
    if require_authorized and (auth.get('explicit_user_authorized') is not True or auth.get('single_use') is not True
            or auth.get('consumed') is not False or auth.get('scope')!=expected
            or auth.get('startup_identity_sha256')!=payload['startup_identity_sha256']
            or consumption_path(root,expected).exists() or (prep/'authorization_consumed.json').exists()):
        raise SearchContractError('VALIDATION_AUTHORIZATION_REQUIRED')
    return payload,c,pilot_run,receipt


def metric_rows(rows):
    n=len(rows)
    members=[sum(r['member_correct'][i] for r in rows)/n for i in range(5)]
    return dict(VoteAcc=sum(r['vote_correct'] for r in rows)/n,OracleAcc=sum(r['oracle_correct'] for r in rows)/n,
        MeanMemberAcc=sum(members)/5,MinMemberAcc=min(members),**{f'Member{i}Acc':v for i,v in enumerate(members)})


def comparison(initial,final):
    if not initial or [r['example_id'] for r in initial]!=[r['example_id'] for r in final]:
        raise SearchContractError('PAIRED_VALIDATION_MEMBERSHIP_MISMATCH')
    a,b=metric_rows(initial),metric_rows(final)
    paired={}
    for name in ['vote','oracle']:
        fixed=sum(not x[name+'_correct'] and y[name+'_correct'] for x,y in zip(initial,final,strict=True))
        broken=sum(x[name+'_correct'] and not y[name+'_correct'] for x,y in zip(initial,final,strict=True))
        paired.update({name+'_fixed':fixed,name+'_broken':broken,name+'_net':fixed-broken})
    delta=b['VoteAcc']-a['VoteAcc']
    return dict(metrics={k:dict(initial=a[k],final=b[k],delta=b[k]-a[k]) for k in POLICY['metrics']},
        paired=paired,signal='POSITIVE' if delta>0 else 'NEGATIVE' if delta<0 else 'NEUTRAL')


def evaluate_team(rows,prompts,solver,benchmark,stage,writer):
    output=[];aggregation=EquivalencePluralityAggregation()
    for row in rows:
        item=protocol_input('math',row['stable_example_id'],{'problem':row['problem']},benchmark.output_contract,protocol=benchmark.protocol)
        benchmark.require_scorable(row['reference'])
        raw=[]
        for member,p in enumerate(prompts):
            if hasattr(solver,'observe_member'): solver.observe_member(member)
            raw.append(solver.solve(p,item,stage=stage,split='validation'))
        raw=tuple(raw)
        parsed=[benchmark.parse_member_output(text,item) for text in raw]
        correctness=[benchmark.score_member_output(p,row['reference'])==1 for p in parsed]
        result=aggregation.aggregate_sync(item=item,member_outputs=raw,benchmark=benchmark)
        evidence=dict(example_id=row['stable_example_id'],request_output_hashes=[hashlib.sha256(v.identity_bytes() if hasattr(v,'identity_bytes') else v.encode()).hexdigest() for v in raw],
            member_correct=correctness,vote_correct=benchmark.score_member_output(result.parsed_output,row['reference'])==1,
            oracle_correct=any(correctness))
        if getattr(benchmark,'invalid_predictions_are_incorrect',False):
            evidence.update(member_valid=[p.valid for p in parsed],invalid_reasons=[p.invalid_reason for p in raw])
        writer(dict(team=stage,**evidence));output.append(evidence)
    return output


async def execute_validation(root,prep,run_root):
    payload,c,pilot_run,receipt=validate_validation(root,prep,require_authorized=True)
    from ..benchmarks.math_domain_binding import execution_binding
    binding=execution_binding(root,c)
    if binding.blockers(): raise SearchContractError('VALIDATION_BINDING_NOT_READY')
    budget=TokenLedger(root/c['token_ledger_directory'],task_sha256=c['task_authorization_sha256'])
    client=None;broker=None
    try:
        consume(root,prep,run_root,payload)
        atomic_write_json(run_root/'accounting_start.json',budget.view())
        rows=validation_rows(root,c,context='POST_SEARCH_VALIDATION_CONTEXT',search_complete_receipt=receipt)
        benchmark=binding.benchmark()
        for row in rows: benchmark.require_scorable(row['reference'])
        final=read_json(pilot_run/'final_team_private.json')['prompts']
        initial=initial_prompts(root,c)
        if len(initial)!=5 or len(final)!=5: raise SearchContractError('VALIDATION_TEAM_SIZE_MISMATCH')
        transport,client=create_transport(c)
        evaluation_contract={**c,'execution_attempt_id':payload['scope']['attempt_id'],
            'provider_bounds':{**c['provider_bounds'],'successful_provider_calls':3000,'transport_attempts':63000}}
        broker=RequestBroker(contract=evaluation_contract,transport=transport,arm='A1',seed=81,token_ledger=budget,validation_only=True,
            ledger_writer=lambda r:append_jsonl(run_root/'ledger.jsonl',r),raw_writer=lambda r:append_jsonl(run_root/'provider_trace_private.jsonl',r))
        solver=BenchmarkSolver(benchmark,broker)
        writer=lambda row:append_jsonl(run_root/'paired_evidence.jsonl',row)
        a=evaluate_team(rows,initial,solver,benchmark,'validation_initial',writer)
        b=evaluate_team(rows,final,solver,benchmark,'validation_final',writer)
        if tuple(initial)==tuple(final) and a!=b: raise SearchContractError('IDENTICAL_TEAM_PAIRED_EVIDENCE_MISMATCH')
        metrics=comparison(a,b)
        summary=dict(policy=validation_policy(c),**metrics,ledger=broker.usage,accounting=budget.view(),
            initial_team_sha256=canonical_sha256(initial),final_team_sha256=canonical_sha256(final),
            search_receipt_sha256=payload['scope']['search_receipt_sha256'],search_closed_forever=True,test_model_calls=0)
        if 'prediction_validity_policy' in c:
            def invalidity(rows):
                by_member=[sum(not r['member_valid'][i] for r in rows) for i in range(5)]
                return dict(denominator=len(rows)*5,total=sum(by_member),by_member=by_member,
                    invalid_rate=sum(by_member)/(len(rows)*5),by_member_rate=[v/len(rows) for v in by_member])
            ai,bi=invalidity(a),invalidity(b)
            summary['prediction_invalidity']=dict(initial=ai,final=bi,delta_invalid_rate=bi['invalid_rate']-ai['invalid_rate'])
        atomic_write_json(run_root/'execution_summary.json',summary)
        if read_json(run_root/'execution_summary.json')!=summary: raise SearchContractError('VALIDATION_PERSISTENCE_MISMATCH')
        if file_sha(pilot_run/'SEARCH_COMPLETE_RECEIPT.json')!=payload['scope']['search_receipt_sha256']:
            raise SearchContractError('SEARCH_RECEIPT_MUTATED')
        atomic_write_json(run_root/'lifecycle.json',dict(status='EXECUTION_COMPLETE',attempt_id=payload['scope']['attempt_id']))
        atomic_write_json(run_root/'accounting_end.json',budget.view())
        atomic_write_json(run_root/'raw_evidence_inventory.json',inventory(run_root))
        return summary
    except BaseException as exc:
        for key in tuple(budget.inflight): budget.reconcile(key,None,outcome='VALIDATION_ABORT_UNKNOWN_FULL_CHARGE')
        if run_root.exists():
            atomic_write_json(run_root/'lifecycle.json',dict(status='EXECUTION_ABORTED',attempt_id=payload['scope']['attempt_id'],error_category=type(exc).__name__))
            atomic_write_json(run_root/'accounting_end.json',budget.view())
            atomic_write_json(run_root/'raw_evidence_inventory.json',inventory(run_root))
        raise
    finally:
        if client: client.close()
        budget.close()
