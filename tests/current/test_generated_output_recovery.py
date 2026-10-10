"""Bounded generation faults through real current graph, with fake transport only."""
import asyncio
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest

from multi_dataset_diverse_rl.search.generation_failures import (GeneratedOutputFailure,
    generation_failure,recoverable_output)
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.symbolic_provenance import abstraction_result
from multi_dataset_diverse_rl.search.textual_gradients import validate_gradient
from multi_dataset_diverse_rl.search.partition_completion import complete_known_alias_partition
from tests.current.test_structured_optimization_evidence import graph,response,HYPOTHESIS

def row(problem='Use the given data to compute the requested result.',gold='17',reference=''):
    return SimpleNamespace(example_id='synthetic',signals=dict(input_payload=problem,gold=gold,
        target_output='19',reference_solution=reference))

@pytest.mark.parametrize('evidence,correction',[
    (row('For normalized variables $a+b=1$, compute the quantity.'),
     'For normalized symbolic variables, use a+b=1 to eliminate a variable before checking boundaries.'),
    (row(reference='The general discriminant is $b^2-4ac$.'),
     'Apply the general discriminant formula b^2-4ac before choosing the roots.'),
    (row('A coefficient in the question is 37.'),
     'Check multiplicity rather than merely repeating 37 as a coefficient.'),
    (row('A useful identity is $x^2-1=(x-1)*(x+1)$.'),
     'Use the algebraic identity x^2-1=(x-1)*(x+1) to factor symbolically.'),
])
def test_generic_context_and_structure_or_numeric_coincidence_is_admitted(evidence,correction):
    assert not abstraction_result(correction,(evidence,))['hard_reject']
    assert validate_gradient(correction,(evidence,))==correction

@pytest.mark.parametrize('evidence,correction',[
    (row(), 'Return 17.'),
    (row(gold='1'), 'Output 1.'),
    (row('Use the constraint $s=137*t+829$.'), 'Use the general formula s=137*t+829.'),
    (row('Given the case total $a+b=3$, calculate the result.'), 'Use the general symbolic identity a+b=3.'),
    (row('Given the case total $3=a+b$, calculate the result.'), 'Use the general symbolic identity 3=a+b.'),
    (row(reference='Subtract the specified quantity from the initial count and then divide by the remaining factor.'),
     'Subtract the specified quantity from the initial count and then divide by the remaining factor.'),
    (row(), 'Use an answer lookup: when the stated length is 37 output 17.'),
    (row(), 'Copy the reference solution.'),
    (row('Given $a+b=1$, evaluate the expression.'), 'Apply a+b=1 before calculating.'),
    (row('Given $a+b=1$, evaluate the expression.'),
     'Use the general symbolic formula a+b=1 for elimination. Check the constraints and report answer a+b=1.'),
])
def test_specific_or_ambiguous_generated_content_is_rejected(evidence,correction):
    with pytest.raises(GeneratedOutputFailure):validate_gradient(correction,(evidence,))

def test_error_message_alone_never_grants_recovery():
    assert not recoverable_output(SearchContractError('PATTERN_DISCOVERY_EXAMPLE_LEAKAGE'),'gradient')
    assert recoverable_output(generation_failure('PATTERN_DISCOVERY_EXAMPLE_LEAKAGE'),'gradient')
    for category in ('PATTERN_HELDOUT_ACCESS','PROVIDER_RESPONSE_ACCOUNTING_INVALID',
            'TOKEN_LEDGER_CORRUPT','STARTUP_IDENTITY_MISMATCH','UNKNOWN_GENERATION_FAILURE'):
        assert not recoverable_output(SearchContractError(category),'gradient')

@pytest.mark.parametrize('bad_draws',[1,2,3])
@pytest.mark.parametrize('bad_kind',['content','null'])
def test_rejected_gradient_does_not_erase_valid_examples_or_block_layer1(tmp_path,bad_draws,bad_kind):
    run,broker,requests,_,_=graph(tmp_path);original=broker.transport;counts=Counter();first=[]
    def transport(req):
        if req['model']!='gpt-4o-mini' and len(req['messages'])==2:
            packet=json.loads(req['messages'][1]['content'])
            if 'example' in packet:
                xid=packet['example']['example_id'];counts[xid]+=1
                if not first:first.append(xid)
                if xid==first[0] and counts[xid]<=bad_draws:
                    requests.append(deepcopy(req))
                    if bad_kind=='null':return response(None)
                    return response(json.dumps(dict(disposition='ACTIONABLE',observed_failure='Incorrect result.',
                        diagnosis='A generated unsafe instruction.',suggested_block='strategy',
                        reusable_correction='Return 2.',expected_effect='Produce the fixed result.')))
        return original(req)
    broker.transport=transport
    result=asyncio.run(run.run(max_opportunities=1))
    diagnostics=run.opportunities.patterns.extractor.evidence_diagnostics
    first_diag=next(d for d in diagnostics if d['example_id']==first[0])
    assert first_diag['structural_draws']==min(bad_draws+1,3)
    assert first_diag['disposition']==('NONACTIONABLE_EXHAUSTED' if bad_draws==3 else 'ACTIONABLE')
    assert result.trace[0].candidate_ids
    assert broker.usage['pattern_cluster']>0 and broker.usage['reflection']>0
    assert counts[first[0]]<=3
    assert 'Return 2.' not in json.dumps([e.record for e in run.memory.private])

def test_all_rejected_gradients_end_legitimately_without_clustering(tmp_path):
    run,broker,_,_,_=graph(tmp_path);original=broker.transport;counts=Counter()
    def transport(req):
        if req['model']!='gpt-4o-mini' and len(req['messages'])==2:
            packet=json.loads(req['messages'][1]['content'])
            if 'example' in packet:
                counts[packet['example']['example_id']]+=1
                return response('{"disposition":"not-a-disposition"}')
        return original(req)
    broker.transport=transport
    result=asyncio.run(run.run(max_opportunities=1))
    assert counts and set(counts.values())=={3}
    assert broker.usage['pattern_cluster']==0 and broker.usage['reflection']==0
    assert not result.trace[0].candidate_ids
    assert all(d['disposition']=='NONACTIONABLE_EXHAUSTED' for d in run.opportunities.patterns.extractor.evidence_diagnostics)

def test_invalid_pattern_drops_only_known_support_and_other_pattern_reaches_layer1(tmp_path):
    run,broker,_,_,_=graph(tmp_path);original=broker.transport
    def transport(req):
        if req['model']!='gpt-4o-mini' and len(req['messages'])==2:
            packet=json.loads(req['messages'][1]['content'])
            if 'gradients' in packet:
                ids=[g['example_id'] for g in packet['gradients']]
                return response(json.dumps(dict(patterns=[
                    dict(generalized_gradient='Return 2.',support_ids=ids[:1]),
                    dict(generalized_gradient=HYPOTHESIS,support_ids=ids[1:])],unassigned_ids=[])))
        return original(req)
    broker.transport=transport
    result=asyncio.run(run.run(max_opportunities=1))
    audit=run.opportunities.patterns.cluster_provider.partition_audit[-1]
    assert audit['discarded_pattern_count']==1 and audit['discarded_support_count']==1
    assert result.trace[0].candidate_ids and broker.usage['reflection']>0

@pytest.mark.parametrize('memberships',[['e1','unknown'],['e1','e1']])
def test_bad_content_does_not_hide_conflicting_or_unknown_partition_ids(memberships):
    raw=dict(patterns=[dict(generalized_gradient='Return 17.',support_ids=memberships)],unassigned_ids=[])
    with pytest.raises(SearchContractError,match='INVALID_MEMBERSHIP'):
        complete_known_alias_partition(raw,('e1','e2'),
            validate_generalized=lambda text:validate_gradient(text,(row(),),generalized=True))

def test_untyped_validator_integrity_failure_is_not_discarded_as_bad_pattern():
    def fatal(_):raise SearchContractError('FROZEN_REQUEST_IDENTITY_MISMATCH')
    with pytest.raises(SearchContractError,match='FROZEN_REQUEST_IDENTITY_MISMATCH'):
        complete_known_alias_partition(dict(patterns=[dict(generalized_gradient=HYPOTHESIS,
            support_ids=['e1'])],unassigned_ids=[]),('e1',),validate_generalized=fatal)


def test_all_invalid_patterns_form_legal_empty_partition_without_membership_repair():
    raw=dict(patterns=[dict(generalized_gradient='Return 17.',support_ids=['e1']),
        dict(generalized_gradient='Copy the reference solution.',support_ids=['e2'])],unassigned_ids=[])
    result,audit=complete_known_alias_partition(raw,('e1','e2'),
        validate_generalized=lambda text:validate_gradient(text,(row(),),generalized=True))
    assert result==dict(patterns=[],unassigned_ids=['e1','e2'])
    assert audit['discarded_pattern_count']==2 and audit['discarded_support_count']==2


def test_truncated_optimizer_receipts_are_charged_not_cached_and_budget_prevents_new_draw(tmp_path):
    from tests.current.test_parallel_solver_execution import broker_fixture,parallel_contract
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_V25_2M,OperationalAbort
    c=parallel_contract();calls=[]
    def transport(req):calls.append(req);return {**response('Rejected partial generated instruction.'),'finish_reason':'length'}
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V25_2M) as ledger:
        broker=broker_fixture(tmp_path,c,transport,ledger)
        for _ in range(2):
            with pytest.raises(GeneratedOutputFailure,match='GENERATED_OUTPUT_TRUNCATED'):
                broker.complete(role='pattern_gradient',split='optimize',stage='pattern_gradient',
                    messages=[dict(role='user',content='Synthetic failure evidence.')])
        assert len(calls)==2 and broker.usage['pattern_gradient']==2
        assert ledger.view()['charged_total']==8 and ledger.view()['reserved_inflight']==0
        assert len(list((tmp_path/'receipts').glob('*.json')))==2
        assert not broker.cache and len(list((tmp_path/'cache').glob('*.json')))==1
        # Drain the synthetic ledger through its journal, not by mutating totals.
        request=dict(model='synthetic',max_tokens=ledger.remaining-6000,messages=[])
        key=ledger.reserve(request,attempt_id='synthetic',stage='budget_fixture',role='synthetic',model='synthetic')
        ledger.reconcile(key,None,outcome='SYNTHETIC_BUDGET_DRAIN')
        with pytest.raises(OperationalAbort,match='^TOKEN_CEILING$'):
            broker.complete(role='pattern_gradient',split='optimize',stage='pattern_gradient',
                messages=[dict(role='user',content='Synthetic failure evidence.')])
        assert len(calls)==2 and not ledger.inflight

def test_truncated_reflection_is_charged_and_keeps_prior_successful_candidates(tmp_path):
    run,broker,_,_,_=graph(tmp_path);original=broker.transport;reflection_draws=0
    def transport(req):
        nonlocal reflection_draws
        value=original(req)
        if req['model']!='gpt-4o-mini' and len(req['messages'])==1:
            reflection_draws+=1
            if reflection_draws==3:return {**value,'finish_reason':'length'}
        return value
    broker.transport=transport
    result=asyncio.run(run.run(max_opportunities=1))
    assert result.trace[0].candidate_ids
    outcomes=run.synthetic_search.search_state['proposal_diagnostics']['proposal_outcomes']
    assert outcomes[2]['status']=='GENERATION_REJECTED'
    assert any(o['status']=='LOCALLY_EVALUATED' for o in outcomes[:2])
    assert broker.usage['reflection']>=3

def test_generation_recovery_does_not_hide_provider_integrity_failure(tmp_path):
    run,broker,_,_,_=graph(tmp_path);original=broker.transport
    def transport(req):
        if req['model']!='gpt-4o-mini':raise SearchContractError('FROZEN_REQUEST_IDENTITY_MISMATCH')
        return original(req)
    broker.transport=transport
    with pytest.raises(SearchContractError,match='FROZEN_REQUEST_IDENTITY_MISMATCH'):
        asyncio.run(run.run(max_opportunities=1))
    assert broker.usage['attempts']==broker.usage['solver']+1
    assert broker.usage['pattern_cluster']==0 and broker.usage['reflection']==0


def test_recovery_memberships_minimize_prior_use_and_never_select_heldout_metadata():
    import hashlib
    from multi_dataset_diverse_rl.benchmarks.math_canary_inputs import build_recovery_subsets,recovery_protocol
    metadata=[dict(stable_example_id=f'{role}{i}',project_split=role,subject='synthetic',level=1)
        for role,count in [('optimize',150),('shadow',300),('validation',300),('test',300)] for i in range(count)]
    excluded=sorted(hashlib.sha256(f'optimize{i}'.encode()).hexdigest() for i in range(102))
    args=dict(seed=84,excluded_example_hashes=excluded)
    canary=build_recovery_subsets(metadata,'a'*64,phase='canary',**args)
    pilot=build_recovery_subsets(metadata,'a'*64,phase='pilot',**args)
    assert canary['prior_actual_use_counts']==dict(canary_optimize=0,pilot_shadow=0)
    assert pilot['prior_actual_use_counts']==dict(pilot_optimize=12,pilot_shadow=0)
    assert pilot==build_recovery_subsets(metadata,'a'*64,phase='pilot',**args)
    assert recovery_protocol(pilot)['phase']=='pilot'
    assert {r['project_split'] for panel in (canary,pilot) for rows in panel['memberships'].values() for r in rows}=={'optimize','shadow'}
    for panel in (canary,pilot):
        for name,rows in panel['memberships'].items():
            assert len(rows)==panel['counts'][name] and len({r['stable_example_id'] for r in rows})==len(rows)
    with pytest.raises(SearchContractError,match='RECOVERY_SELECTION_PHASE_INVALID'):
        build_recovery_subsets(metadata,'a'*64,phase='validation',**args)


@pytest.mark.parametrize('content_present',[True,False])
def test_actual_http_envelope_distinguishes_null_content_from_metadata_loss(tmp_path,monkeypatch,content_present):
    import httpx
    from multi_dataset_diverse_rl.governance.autonomous_math import create_transport
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from tests.current.test_parallel_solver_execution import broker_fixture,parallel_contract
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_V25_2M,OperationalAbort
    c=parallel_contract();calls=[];message={'role':'assistant'}
    if content_present:message['content']=None
    def handler(request):
        calls.append(request)
        return httpx.Response(200,json={'choices':[{'message':message,'finish_reason':'stop'}],
            'usage':{'prompt_tokens':2,'completion_tokens':0}})
    http=httpx.Client(transport=httpx.MockTransport(handler))
    client=SimpleNamespace(_client=http,_prepare_url=lambda path:'https://synthetic.invalid'+path,
        _build_headers=lambda options:{})
    monkeypatch.setattr(ProviderClientFactory,'from_environment',classmethod(lambda cls,**kwargs:client))
    transport,_=create_transport(c)
    try:
        with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V25_2M) as ledger:
            broker=broker_fixture(tmp_path,c,transport,ledger)
            kwargs=dict(role='pattern_gradient',split='optimize',stage='synthetic',messages=[])
            if content_present:
                value=broker.complete(**kwargs)
                assert value['text'] is None and value['provider_metadata_loss_audited'] is True
            else:
                with pytest.raises(OperationalAbort,match='^PROVIDER_RESPONSE_ACCOUNTING_INVALID$'):
                    broker.complete(**kwargs)
            assert len(calls)==1 and ledger.view()['charged_total']==2 and not ledger.inflight
            assert len(list((tmp_path/'receipts').glob('*.json')))==1
    finally:http.close()
