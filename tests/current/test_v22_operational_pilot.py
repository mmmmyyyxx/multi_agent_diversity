"""Fixed-horizon resource admission and real runner with synthetic providers."""
import asyncio,hashlib,json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from multi_dataset_diverse_rl.benchmarks.gradient_pilot_contract import pilot_provider_bounds
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.governance.token_accounting import OperationalAbort,reservation
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
ROOT=Path(__file__).resolve().parents[2]
PROFILE='experiments/execution_bindings/math_v2_2_gradient_pattern_seed81_pilot_v1.json'
def contract():return json.loads((ROOT/PROFILE).read_bytes())

def test_fixed_horizon_ceiling_counts_and_no_saturation_claim():
    c=contract();b=pilot_provider_bounds(c,max_opportunities=64,token_ceiling=24_000_000)
    assert b['solver_calls']==4*(500+64*220)==58320
    assert b['pattern_gradient_calls']==64*60*3==11520
    assert b['pattern_cluster_calls']==64 and b['reflection_calls']==384
    assert b['successful_provider_calls']==70288
    assert b['transport_attempts']==1476048
    assert b['attempt_charged_token_ceiling']==24_000_000
    assert not b['bound_proof']['guarantees_saturation']
    assert b['bound_proof']['scientific_stopper_unchanged']

@pytest.mark.parametrize('k,t',[(-1,24_000_000),(0,24_000_000),(True,1),(1,40_000_000),(1,-1)])
def test_invalid_horizon_and_token_limits_rejected(k,t):
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    with pytest.raises(SearchContractError):pilot_provider_bounds(contract(),max_opportunities=k,token_ceiling=t)

def test_current_binding_has_no_legacy_composition_and_cannot_open_heldout():
    b=execution_binding(ROOT,contract());assert b.blockers()==()
    m=b.method('A4');assert m.method=='unified_team_prompt_search_v2_2'
    assert m.transition_policy=='initial_competence_target_or_team_progress_v3'
    assert b.benchmark().protocol.member_success_semantics.startswith('MATH_SOLVER_INVALID_RECOVERY_V1:')
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    for role in ('validation','test'):
        with pytest.raises(SearchContractError,match='HELDOUT'):b.examples(role)
    c=contract();c['layer1_search_policy']['max_generations']=7
    assert execution_binding(ROOT,c).blockers()
    c=contract();c['provider_bounds']['max_opportunities']=65
    assert execution_binding(ROOT,c).blockers()

def test_reservation_cap_precedes_transport_and_retries_cannot_overspend():
    c=contract();c['operational_pilot']['token_ceiling']=1
    calls=[];b=RequestBroker(contract=c,transport=lambda r:calls.append(r),arm='A4',seed=81)
    with pytest.raises(OperationalAbort,match='TOKEN_CEILING'):
        b.complete(role='reflection',split='optimize',stage='fake',messages=[{'role':'user','content':'Synthetic rule.'}])
    assert not calls and b.usage['attempts']==0

@pytest.mark.parametrize('field,category',[('attempts','TRANSPORT_CEILING'),('successes','PROVIDER_CALL_CEILING')])
def test_resource_stop_categories_precede_transport(field,category):
    c=contract();calls=[];b=RequestBroker(contract=c,transport=lambda r:calls.append(r),arm='A4',seed=81)
    b.usage[field]=c['provider_bounds']['transport_attempts' if field=='attempts' else 'successful_provider_calls']
    with pytest.raises(OperationalAbort,match=category):
        b.complete(role='reflection',split='optimize',stage='fake',messages=[{'role':'user','content':'Synthetic rule.'}])
    assert not calls

def test_provider_visible_fields_unchanged_and_fresh_member_lanes():
    c=contract();old=json.loads((ROOT/c['baseline_scientific_settings_path']).read_bytes())
    before=RequestBroker(contract=old,transport=lambda r:None,arm='A4',seed=81)
    after=RequestBroker(contract=c,transport=lambda r:None,arm='A4',seed=81)
    ids=[]
    for role in ('solver','reflection','pattern_gradient','pattern_cluster'):
        kwargs=dict(role=role,split='optimize',messages=[{'role':'user','content':'Synthetic rule.'}])
        if role=='solver':kwargs['member_slot']=0
        req,key=after._request_identity(**kwargs)
        assert req==before._request_identity(**kwargs)[0]
        if role=='solver':
            for m in range(5):ids.append(after._request_identity(**{**kwargs,'member_slot':m})[1])
    assert len(set(ids))==5

@pytest.mark.parametrize('k,changed,completed',[(1,False,False),(1,True,False),(64,False,True),(0,False,False)])
def test_current_runner_scientific_and_operational_receipts(tmp_path,monkeypatch,k,changed,completed):
    from multi_dataset_diverse_rl.governance import autonomous_math as execution
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY
    c=contract();binding=execution_binding(ROOT,c);adapter=binding.benchmark()
    def examples(role):
        assert role in ('optimize','shadow')
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',
            {'problem':f'Synthetic {role} arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'1')
            for i in range(60 if role=='optimize' else 40))
    monkeypatch.setattr(binding,'examples',examples);monkeypatch.setattr(binding,'blockers',lambda:())
    c['token_ledger_directory']='runs/synthetic_ledger'
    c['provider_bounds']['max_opportunities']=c['operational_pilot']['max_opportunities']=max(k,1)
    if not k:c['operational_pilot']['token_ceiling']=1
    c['initial_competence_binding']['support_identity']=hashlib.sha256(json.dumps(
        [e.item.input_id for e in examples('optimize')],separators=(',',':')).encode()).hexdigest()
    binding.contract=c
    with TokenLedger(tmp_path/c['token_ledger_directory'],task_sha256=c['task_authorization_sha256'],policy=POLICY) as ledger:
        ledger.amend_authorization_40m(authorization_sha256='0'*64,expected_charged_total=0)
    (tmp_path/'binding.json').write_text(json.dumps(c),encoding='utf8')
    prep=tmp_path/'prep';prep.mkdir();run=tmp_path/'runs/synthetic_pilot'
    scope=dict(attempt_id=c['execution_attempt_id'],source_sha='0'*40)
    (prep/'authorization.json').write_text(json.dumps(dict(scope=scope,explicit_user_authorized=True,consumed=False)),encoding='utf8')
    payload=dict(manifest={'execution_binding':{'path':'binding.json'},'source_sha':'0'*40},scope=scope,startup_identity_sha256='0'*64)
    original=execution.read_json
    monkeypatch.setattr(execution,'read_json',lambda p:original(p if p.exists() else ROOT/p.relative_to(tmp_path)))
    monkeypatch.setattr(execution,'execution_binding',lambda r,c:binding)
    requests=[];counter=[];closed=[]
    def transport(req):
        requests.append(req)
        if req['model']=='qwen3-8b':
            prompt,problem=req['messages'][1]['content'].split('\n\n',1)
            i=int(problem.split('arithmetic ')[1].split('.')[0])
            text='FINAL_ANSWER: '+('1' if i>=7 or changed and prompt!='Solve the problem.' else '2')
        elif len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            text=(json.dumps({'gradient':'Check constraints before transforming intermediate expressions.'}) if 'example' in p else
                json.dumps({'patterns':[{'generalized_gradient':'Check constraints before transforming intermediate expressions.',
                    'support_ids':[g['example_id'] for g in p['gradients']]}],'unassigned_ids':[]}))
        else:
            counter.append(1);text=json.dumps({'decision_procedure':f'Inspect constraints and check signs with {len(counter)} independent verifications.',
                'change_summary':'Add sign checks.'})
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',provider_response_accepted=True,
            provider_metadata_loss_audited=True,provider_reasoning_content_present=False,
            provider_reasoning_character_count=None,provider_usage_details={},provider_thinking_indicators=[])
    monkeypatch.setattr(execution,'create_transport',lambda c:(transport,SimpleNamespace(close=lambda:closed.append(True))))
    summary=asyncio.run(execution.execute_search(tmp_path,prep,run,payload))
    assert summary['pilot_completed'] is completed and closed==[True]
    assert summary['validation_calls']==summary['test_calls']==0
    assert (run/'SEARCH_COMPLETE_RECEIPT.json').exists() is completed
    assert (run/'SEARCH_CLOSED_RECEIPT.json').exists() is not completed
    assert summary['accounting']['reserved_inflight']==0 and summary['accounting']['charged_total']==4*len(requests)
    if not k:
        assert summary['stop_reason']=='TOKEN_CEILING' and not requests
        assert not (run/'final_state_private.json').exists()
        return
    assert summary['result']['stop_reason']==('SATURATION_REACHED' if completed else 'OPERATIONAL_OPPORTUNITY_CEILING')
    if changed:
        trace=summary['result']['trace'][0]
        assert trace['allocation_audit']['realized_progress_path']=='TARGET'
        assert trace['allocation_audit']['realized_target_gain']>0
        assert trace['allocation_audit']['realized_team_gain']==0
        assert summary['memory_audit']['success_writes']>=1
