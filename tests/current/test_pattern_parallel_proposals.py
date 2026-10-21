"""Frozen-Parent breadth, native evidence, fresh requests, gates and paid journals."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.research.ablations.proposal_config import default_config,build_plan,SCHEMA
from multi_dataset_diverse_rl.research.ablations.config import current_config,validate_plan
from multi_dataset_diverse_rl.research.ablations.engine import run_cell
from multi_dataset_diverse_rl.research.ablations.pattern_proposals import project_proposals
from multi_dataset_diverse_rl.research.ablations.proposal_audit import audit_transition
from multi_dataset_diverse_rl.research.ablations.offline_runtime import OfflineAblationBroker
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport,synthetic_rows
from multi_dataset_diverse_rl.research.ablations import governance as gov,proposal_governance as pg
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_GATED_32_UNCAPPED
from multi_dataset_diverse_rl.search.system_prompt import SEED,SystemPrompt
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest

ROOT=Path(__file__).resolve().parents[2]


class Transport(SyntheticTransport):
    def __init__(self,patterns=3,failure=None,scores=(2,3)):
        super().__init__();self.patterns=patterns;self.failure=failure;self.scores=scores;self.draws=0
    def for_role(self,role,request):
        value=super().for_role(role,request)
        if role=='pattern_cluster':
            ids=[g['example_id'] for g in json.loads(request['messages'][1]['content'])['gradients']]
            if not self.patterns:value['text']=json.dumps(dict(patterns=[],unassigned_ids=ids))
            else:
                groups=[ids] if self.patterns==1 else [ids[:6],ids[6:]] if self.patterns==2 else [ids[:4],ids[4:8],ids[8:]]
                principles=['Verify algebraic transformations preserve the constraints.',
                    'Enumerate all discrete boundary cases before counting.',
                    'Derive geometric relations from the actual configuration.']
                value['text']=json.dumps(dict(patterns=[dict(generalized_gradient=text,support_ids=group)
                    for text,group in zip(principles,groups)],unassigned_ids=[]))
        if role=='reflection':
            self.draws+=1
            bad=self.failure=='both_bad' or self.failure=='first_bad' and self.draws<=4
            value['text']='{' if bad else json.dumps(dict(block='strategy',content=SEED.strategy if self.failure=='noop' else f'Proposal method {self.draws}'))
        if role=='solver' and 'Proposal method ' in request['messages'][0]['content']:
            text=request['messages'][0]['content'];number=int(request['messages'][1]['content'].split(':')[-1])
            method=next(i for i in range(1,10) if f'Proposal method {i}' in text)
            threshold=self.scores[0 if method==1 else 1]
            value['text']=r'\boxed{'+str(number if number<threshold else number+1)+'}'
        return value


@pytest.fixture
def setup(tmp_path):
    ledgers=[]
    def create(arm='C2',patterns=3,failure=None,scores=(2,3),rounds=1):
        config=default_config();config['rounds']=config['runtime']['rounds']=rounds
        config['repeats_per_arm']=config['runtime']['repeats_per_arm']=1
        plan=build_plan(config);cell=next(c for c in plan['cells'] if c['arm_id']==arm)
        base=tmp_path/str(len(ledgers));base.mkdir();sid=digest(['proposals',len(ledgers)])
        ledger=TokenLedger(base/'ledger',task_sha256=sid,policy=POLICY_GATED_32_UNCAPPED);ledgers.append(ledger)
        transport=Transport(patterns,failure,scores)
        broker=OfflineAblationBroker(plan=plan,phase='train',scope=dict(scope_identity=sid,
            call_ceiling=plan['training_semantic_call_ceiling'],token_ceiling=None,network_recovery=True,
            generation_recovery=config['runtime']['diagnosis_recovery']),transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(base/'receipts',attempt_id='fake',startup_identity_sha256=sid),dispatch_guard=lambda _:None)
        rows=synthetic_rows('optimize',12);val=synthetic_rows('validation',4);artifacts={}
        folder=base/cell['cell_id'];folder.mkdir()
        def persist(name,value):
            assert name not in artifacts
            artifacts[name]=deepcopy(value);atomic_write_json(folder/(name+'.json'),value)
        return plan,cell,broker,transport,rows,val,artifacts,persist,base
    yield create
    for ledger in ledgers:
        assert not ledger.inflight;ledger.close()


def execute(s):
    plan,cell,b,t,rows,val,a,persist,base=s
    result=run_cell(plan,cell,SEED.to_dict(),rows,val,b,persist)
    for n in range(1,plan['config']['rounds']+1):
        audit_transition(a[f'checkpoint_{n-1}'],a[f'checkpoint_{n}'],plan,cell,n,lambda name:a[name])
    return result


def test_configs_budget_scope_and_original_defaults_unchanged():
    p=build_plan(default_config());validate_plan(p)
    assert len(p['cells'])==9 and p['proposal_opportunities']==300
    assert p['training_semantic_calls_by_arm']=={'C0':90420,'C1':150660,'C2':150660}
    assert p['training_semantic_call_ceiling']==391740 and p['final_test_semantic_call_ceiling']==10800
    assert 'population' not in p['config']['runtime'] and current_config()['runtime']['population']['capacity']==3
    assert p['config']['runtime']['validation_membership_seed']==42 and not p['ready_to_run']
    for field in ['solver_decoding_policy','optimizer_generation_policy','candidate_guard','trajectory_policy']:
        assert len({digest(p['config']['runtime'][field]) for c in p['cells']})==1
    bad=default_config();bad['arms'][1]['proposal_mode']='parallel'
    with pytest.raises(SearchContractError):build_plan(bad)


def test_C0_matches_existing_single_parent_kernel_requests_and_state(setup,tmp_path):
    from multi_dataset_diverse_rl.research.ablations.config import default_config as baseline_config,build_plan as base_plan
    current=setup('C0');new=execute(current)
    config=baseline_config();config['rounds']=config['runtime']['rounds']=1
    config['repeats_per_arm']=config['runtime']['repeats_per_arm']=1
    plan=base_plan(config);cell=next(c for c in plan['cells'] if c['arm_id']=='C')
    base=tmp_path/'existing_kernel';sid=digest('existing synthetic kernel');transport=Transport()
    ledger=TokenLedger(base/'ledger',task_sha256=sid,policy=POLICY_GATED_32_UNCAPPED)
    try:
        broker=OfflineAblationBroker(plan=plan,phase='train',scope=dict(scope_identity=sid,
            call_ceiling=plan['training_semantic_call_ceiling'],token_ceiling=None,network_recovery=True,
            generation_recovery=config['runtime']['diagnosis_recovery']),transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(base/'receipts',attempt_id='fake',startup_identity_sha256=sid),dispatch_guard=lambda _:None)
        old=run_cell(plan,cell,SEED.to_dict(),current[4],current[5],broker,lambda *a:None)
        assert old['prompt_history']==new['prompt_history'] and old['usage']==new['usage']
        assert old['final_optimize_correct']==new['final_optimize_correct']
        assert old['final_validation_correct']==new['final_validation_correct']
        def comparable(calls):
            values=[]
            for role,request in calls:
                request=deepcopy(request)
                if role=='pattern_gradient':
                    packet=json.loads(request['messages'][1]['content'])
                    # Separate plans have separate receipt identities. Preserve
                    # all semantic evidence; normalize only its paid request ID.
                    packet['example']['solver_trajectory']['source']['request_sha256']='receipt_identity'
                    request['messages'][1]['content']=json.dumps(packet,sort_keys=True)
                values.append(json.dumps([role,request],sort_keys=True))
            return sorted(values)  # Worker dispatch order is not algorithm order.
        assert comparable(transport.calls)==comparable(current[3].calls)
    finally:ledger.close()


@pytest.mark.parametrize('arm,patterns,expected_distinct,draws',[
    ('C0',3,1,1),('C1',3,2,2),('C2',3,2,2),('C2',2,2,2),('C2',1,1,1),('C2',0,0,1)])
def test_shared_diagnosis_and_native_isolated_evidence(setup,arm,patterns,expected_distinct,draws):
    s=setup(arm,patterns);p,c,b,t,rows,val,a,_,_=s;r=execute(s);m=r['metrics'][0]
    assert r['usage']['pattern_gradient']==12 and r['usage']['pattern_cluster']==1
    assert r['usage']['reflection']==draws and m['distinct_pattern_count']==expected_distinct
    packets=[json.loads(q['messages'][1]['content']) for role,q in t.calls if role=='reflection']
    assert all('parent_context' not in q for q in packets)
    teacher_packets=[json.loads(q['messages'][1]['content']) for role,q in t.calls if role!='solver']
    def no_heldout_feedback(value):
        if isinstance(value,dict):
            assert not {'retained_validation_correct','validation_before','candidate_validation',
                'candidate_validation_score','test_predictions','candidate_proposals'}&set(value)
            for child in value.values():no_heldout_feedback(child)
        elif isinstance(value,list):
            for child in value:no_heldout_feedback(child)
        elif isinstance(value,str):
            assert not value.startswith(('validation_fixture_','test_fixture_'))
    for packet in teacher_packets:no_heldout_feedback(packet)
    if arm!='C1':assert all(q['current_system_prompt']==SEED.to_dict() for q in packets)
    if patterns<=1:
        assert len(packets)==1 or packets[0]==packets[1]
    for q in packets:
        sf=q['structured_feedback']
        if patterns:
            ids={e['example_id'] for e in q['errors']}
            assert len(sf['patterns'])==1 and len(ids)<=3
            assert {g['example_id'] for g in sf['gradients']}==ids=={d['example_id'] for d in sf['diagnostics']}
            assert ids<=set(sf['patterns'][0]['support_ids'])
        else:assert sf is None and len(q['errors'])==3
    if arm=='C2' and patterns>=2:
        assert packets[0]['structured_feedback']['patterns'][0]['pattern_id']!=packets[1]['structured_feedback']['patterns'][0]['pattern_id']
    from multi_dataset_diverse_rl.research.ablations.generation import audit_optimizer_request
    records=[read_json(path)['record'] for path in (s[-1]/'receipts').glob('*.json')]
    reflection=[x for x in records if x['logical_identity']['role']=='reflection']
    assert len({x['logical_identity']['stage'] for x in reflection})==draws
    assert len({v['request_sha256'] for k,v in a.items() if '_generation_' in k})==draws
    for record in reflection:audit_optimizer_request(s[-1],c['cell_id'],record)


def test_ties_are_reproducible_and_first_focus_matches_C0(setup):
    s=setup('C2',3);execute(s);a=s[6];audit=a['round_1_module']['audit']
    from multi_dataset_diverse_rl.research.ablations.interfaces import PreparedFeedback
    prepared=PreparedFeedback([],None,None,audit,None)
    views,selection=project_proposals(prepared,42,1,'parallel')
    single,original=project_proposals(prepared,42,1,'single')
    assert views[0]==single[0] and original['selected_pattern_ids'][0]==selection['selected_pattern_ids'][0]
    assert project_proposals(prepared,42,1,'parallel')==(views,selection)
    assert project_proposals(prepared,42,1,'sequential')==(views,selection)
    assert selection['pattern_support_counts']==[4,4] and len(set(selection['selected_pattern_ids']))==2


@pytest.mark.parametrize('failure,measured,accepted',[('first_bad',1,True),('both_bad',0,False),('noop',0,False),(None,2,True)])
def test_failed_or_noop_slot_does_not_replace_other_slot(setup,failure,measured,accepted):
    r=execute(setup(failure=failure));m=r['metrics'][0]
    assert sum(p['generation_status']=='MEASURED' for p in m['proposals'])==measured
    assert m['accepted']==accepted
    if failure=='first_bad':assert m['proposals'][0]['generation_status']=='EXHAUSTED' and m['winning_proposal_id'].endswith('proposal1')
    if failure=='noop':assert r['usage']['reflection']==2 and m['decision']=='SKIPPED_NO_OP'


@pytest.mark.parametrize('scores,accepted',[( (0,0),False),((2,2),True),((1,3),True)])
def test_complete_evaluation_strict_gate_and_pattern_order_tie(setup,scores,accepted):
    s=setup(scores=scores);r=execute(s);m=r['metrics'][0]
    assert m['accepted']==accepted and all(p['candidate_validation'] is not None for p in m['proposals'])
    if scores==(2,2):
        assert m['winning_proposal_id']==m['proposals'][0]['proposal_id']
    events=[role for role,_ in s[3].calls]
    last_reflection=max(i for i,x in enumerate(events) if x=='reflection')
    assert not any(x=='solver' for x in events[events.index('reflection'):last_reflection])


def test_no_serial_parent_updates_and_accepted_then_rejected(setup):
    s=setup(rounds=2,scores=(2,3));r=execute(s)
    assert [m['accepted'] for m in r['metrics']]==[True,False]
    for n in (1,2):
        for slot in (0,1):assert s[6][f'round_{n}_proposal_{slot}_opportunity']['parent_prompt']==r['prompt_history'][n-1]


def test_resume_boundary_skips_paid_prefix_and_partial_round_fails_closed(setup):
    s=setup(rounds=2);p,c,b,t,rows,val,a,persist,_=s;r=execute(s)
    prefix={k:deepcopy(v) for k,v in a.items() if k.startswith('round_1_') or k in ('checkpoint_0','checkpoint_1')}
    historical={k:a['checkpoint_0']['bootstrap_usage'][k]+a['checkpoint_1']['metric']['cost'][k] for k in b.usage_view()}
    resumed={}
    before=b.usage_view()['reflection']
    run_cell(p,c,SEED.to_dict(),rows,val,b,lambda n,v:resumed.update({n:deepcopy(v)}),
        resume=dict(checkpoints=(deepcopy(a[f'checkpoint_{n}']) for n in (0,1)),artifacts=prefix,historical_usage=historical))
    assert b.usage_view()['reflection']-before==2 and 'checkpoint_1' not in resumed
    prefix['round_2_proposal_0_generation_1']={}
    before=b.usage_view()['provider_calls']
    with pytest.raises(SearchContractError,match='PARTIAL_ROUND'):
        run_cell(p,c,SEED.to_dict(),rows,val,b,lambda *a:None,
            resume=dict(checkpoints=[a['checkpoint_0'],a['checkpoint_1']],artifacts=prefix,historical_usage=historical))
    assert b.usage_view()['provider_calls']==before


@pytest.mark.parametrize('tamper',['parent','evidence','winner','cost','candidate','diagnosis'])
def test_audit_rejects_resealed_semantic_tampering(setup,tamper):
    s=setup();execute(s);p,c,b,t,rows,val,a,_,_=s;bad=deepcopy(a);cp=bad['checkpoint_1']
    if tamper=='parent':cp['parent_prompt']['role']='another Parent'
    elif tamper=='evidence':bad['round_1_proposal_1_opportunity']['structured_feedback']=bad['round_1_proposal_0_opportunity']['structured_feedback']
    elif tamper=='winner':cp['metric']['winning_proposal_id']=cp['metric']['proposals'][0]['proposal_id']
    elif tamper=='cost':cp['metric']['cost']['provider_calls']+=1
    elif tamper=='candidate':cp['metric']['proposals'][1]['candidate_prompt']['role']='extra block edit'
    else:bad['round_1_module']['audit']['complete_diagnosis']['errors'].pop()
    cp['integrity_sha256']=digest({k:v for k,v in cp.items() if k!='integrity_sha256'})
    with pytest.raises(SearchContractError):audit_transition(bad['checkpoint_0'],cp,p,c,1,lambda n:bad[n])


def test_governed_fake_train_and_test_seal_and_charge_once(tmp_path,monkeypatch):
    config=default_config();config['rounds']=config['runtime']['rounds']=1;config['repeats_per_arm']=config['runtime']['repeats_per_arm']=1
    plan=build_plan(config);optimize=synthetic_rows('optimize',150);pool=synthetic_rows('validation',300);test=synthetic_rows('test',300)
    prep=tmp_path/'runs/attempt/prep';train=prep.parent/'train_run'
    class Reader:
        manifest_sha256=digest('fake split')
        members=[dict(stable_example_id=r['stable_example_id'],project_split='validation') for r in pool]
        def rows(self,split,purpose):assert split=='optimize';return optimize
        def _read_rows(self,split,frozen_subset_ids):assert split=='validation';return tuple(r for r in pool if r['stable_example_id'] in frozen_subset_ids)
        def heldout_rows(self,access):assert (train/'terminal.json').exists();access.require_authorization();return test
    reader=Reader();manifest=dict(initial_prompt_path='initial.json')
    atomic_write_json(tmp_path/'initial.json',dict(members=[dict(prompt=SEED.to_dict())]))
    for name in ('prep','handoff','execution_plan'):atomic_write_json(prep/(name+'.json'),dict(fake=True))
    scope=dict(phase='train',source_sha='0'*40,source_hashes={},plan=plan,gate_membership=gov.gate_membership(reader,42),
        initial_prompt_sha256=SEED.prompt_hash,optimize_sample_ids=[r['stable_example_id'] for r in optimize],
        test_plan=dict(sample_ids=[r['stable_example_id'] for r in test],successful_semantic_call_ceiling=3600),
        call_ceiling=plan['training_semantic_call_ceiling'],token_ceiling=None,
        network_recovery=config['runtime']['timeout_recovery'],generation_recovery=config['runtime']['diagnosis_recovery'])
    scope['scope_identity']=digest(scope);auth=gov.authorization_template(scope)
    auth.update(approved=True,adaptive_validation_access=True,test_access=True,user_authorization_evidence_sha256=digest('fake grant'))
    atomic_write_json(prep/'authorization.json',auth)
    monkeypatch.setattr(gov,'validate_prep',lambda *a,**k:scope);monkeypatch.setattr(pg,'inputs',lambda root:(manifest,reader))
    transport=SyntheticTransport()
    class Clients:
        def close(self):pass
    monkeypatch.setattr(gov,'_transport',lambda:(transport,Clients()))
    gov._execute_phase(tmp_path,prep,train,scope,train_scope=scope)
    results,seal=gov.verify_training(train,scope)
    assert len(results)==3 and read_json(train/'execution_summary.json')['usage']['reflection']==3
    assert all(r['rounds_completed']==1 for r in results)
    test_scope=gov.final_test_scope(scope,seal)
    gov._execute_phase(tmp_path,prep,train.parent/'test_run',test_scope,train_scope=scope,parent_results=results,terminal_sha=seal)
    summary=read_json(train.parent/'test_run/execution_summary.json')
    assert summary['usage']['solver']==300 and summary['usage']['cache_hits']==600
    assert gov.verify_training(train,scope)[1]==seal


@pytest.mark.parametrize('scores,accepted',[( (2,3),[True,True]),((2,1),[True,False]),((0,3),[False,True])])
def test_sequential_cumulative_gate_uses_latest_parent(setup,scores,accepted):
    s=setup('C1',2,scores=scores);r=execute(s);m=r['metrics'][0]
    assert [p['accepted'] for p in m['proposals']]==accepted
    assert m['accepted_update_count']==sum(accepted) and m['winning_proposal_id'] is None
    a,b=m['proposals'];expected_parent=a['candidate_prompt'] if accepted[0] else SEED.to_dict()
    assert b['parent_prompt']==expected_parent and b['validation_before']==(scores[0] if accepted[0] else 0)
    assert b['diagnosis_source_prompt_sha256']==SEED.prompt_hash
    assert r['usage']['pattern_gradient']==12 and r['usage']['pattern_cluster']==1
    op=s[6]['round_1_proposal_1_opportunity']
    if accepted[0]:
        assert op['diagnosis_context']['diagnosis_source_prompt_sha256']==SEED.prompt_hash
        assert op['diagnosis_context']['current_parent_prompt_sha256']==SystemPrompt.from_dict(expected_parent).prompt_hash
        assert all(not v['correct'] for v in op['diagnosis_context']['current_parent_observations'])
        assert b['same_block_as_first_accepted_edit'] is True
        assert b['repair_metrics']['preserved_prior_root_repairs']+b['repair_metrics']['broken_prior_root_repairs']==b['repair_metrics']['prior_root_repairs']
    else:assert 'diagnosis_context' not in op
    if scores==(2,1):assert b['candidate_validation_score']>0 and not b['accepted']


def test_sequential_resolved_second_pattern_skips_without_replacement(setup):
    s=setup('C1',2,scores=(12,12));r=execute(s);m=r['metrics'][0]
    assert m['selected_distinct_pattern_count']==2 and m['distinct_pattern_count']==1
    assert m['proposals'][1]['generation_status']=='SKIPPED_NO_CURRENT_ERRORS'
    assert m['proposals'][1]['generation_draws']==0 and r['usage']['reflection']==1


def test_invalid_partitions_exhaust_once_then_one_raw_fallback(setup):
    s=setup('C2',3);t=s[3];original=t.for_role
    def respond(role,request):
        value=original(role,request)
        if role=='pattern_cluster':value['text']='not JSON'
        return value
    t.for_role=respond;r=execute(s)
    assert r['usage']['pattern_cluster']==3 and r['usage']['reflection']==1
    assert r['metrics'][0]['pattern_count']==0 and len(r['metrics'][0]['proposals'])==1


def test_budget_curve_has_no_future_parallel_winner(setup):
    from multi_dataset_diverse_rl.research.ablations.proposal_analysis import budget_comparison
    r=execute(setup('C2',2));curve=budget_comparison([r])['curves'][r['cell']['cell_id']]
    assert [p['candidate_evaluations'] for p in curve]==[0,2]
    r=execute(setup('C1',2));curve=budget_comparison([r])['curves'][r['cell']['cell_id']]
    assert [p['candidate_evaluations'] for p in curve]==[0,1,2]
    assert [p['validation_correct'] for p in curve]==[0,2,3]


def test_summary_separates_rounds_edits_draws_legal_and_measured_costs(setup):
    from multi_dataset_diverse_rl.research.ablations.analysis import summarize
    samples=[setup(arm,2) for arm in ('C0','C1','C2')]
    results=[execute(s) for s in samples];summary=summarize(samples[0][0],results)
    a,b,c=summary['rows']
    assert (b['accepted_rounds'],b['updates'],b['double_update_opportunities'])==(1,2,1)
    assert b['attempted_proposals']==b['structurally_legal_proposals']==b['measured_proposals']==2
    assert c['rejected_measured_proposals']==1 and c['second_pattern_win_denominator']==1
    assert c['second_pattern_win_rate']==1 and b['second_pattern_win_rate'] is None
    for row,result in zip(summary['rows'],results):
        assert sum(v['physical_calls'] for v in row['role_usage'].values())==result['usage']['provider_calls']
        assert sum(v['input_tokens']+v['output_tokens'] for v in row['role_usage'].values())==row['optimization_tokens']


@pytest.mark.parametrize('mutation',['authorization','scope','plan','handoff','evidence'])
def test_frozen_preparation_refuses_mismatch_before_transport(tmp_path,monkeypatch,mutation):
    scope=dict(scope_identity=digest('fake freeze'),plan=dict(cells=[dict(cell_id='C0_seed42_repeat01')]))
    folder=tmp_path/'runs/attempt/prep'
    handoff=dict(READY_TO_RUN=False,status='FROZEN_NOT_AUTHORIZED',protocol=dict(identity='frozen'))
    plan=dict(identity='frozen execution')
    monkeypatch.setattr(gov,'require_selected_runtime',lambda root:None)
    monkeypatch.setattr(gov,'_git',lambda *a:'')
    monkeypatch.setattr(pg,'scope_template',lambda *a:deepcopy(scope))
    monkeypatch.setattr(pg,'handoff',lambda *a:deepcopy(handoff))
    monkeypatch.setattr(gov,'execution_plan_template',lambda *a:deepcopy(plan))
    def forbidden():raise AssertionError('A failed preflight must not construct clients')
    monkeypatch.setattr(gov,'_transport',forbidden)
    auth=gov.authorization_template(scope)
    for name,value in [('prep',dict(schema_version=pg.PREP_SCHEMA,scope=scope)),
        ('authorization',auth),('handoff',handoff),('execution_plan',plan)]:
        atomic_write_json(folder/(name+'.json'),value)
    assert gov.validate_prep(tmp_path,folder)==scope
    if mutation=='scope':
        atomic_write_json(folder/'prep.json',dict(schema_version=pg.PREP_SCHEMA,scope=dict(scope,scope_identity=digest('other'))))
    elif mutation=='plan':atomic_write_json(folder/'execution_plan.json',dict(identity='changed'))
    elif mutation=='handoff':atomic_write_json(folder/'handoff.json',dict(handoff,protocol=dict(identity='changed')))
    elif mutation=='evidence':
        auth.update(approved=True,adaptive_validation_access=True,user_authorization_evidence_sha256=digest('missing proof'))
        atomic_write_json(folder/'authorization.json',auth)
    with pytest.raises(SearchContractError):gov.validate_prep(tmp_path,folder,authorized=True)


def test_prepare_persists_identity_before_building_hashed_handoff(tmp_path,monkeypatch):
    scope=dict(source_sha='0'*40,source_hashes={},scope_identity=digest('fresh'),
        plan=dict(cells=[dict(cell_id='C0_seed42_repeat01')]))
    monkeypatch.setattr(gov,'require_selected_runtime',lambda root:None)
    monkeypatch.setattr(gov,'_git',lambda *a:'')
    monkeypatch.setattr(pg,'scope_template',lambda *a:deepcopy(scope))
    monkeypatch.setattr(gov,'execution_plan_template',lambda *a:dict(identity='fake execution'))
    def hashed_handoff(root,folder,actual):
        assert actual==scope
        assert all((folder/(name+'.json')).is_file() for name in ('prep','authorization','execution_plan'))
        return dict(prep_identity=digest(read_json(folder/'prep.json')),READY_TO_RUN=False)
    monkeypatch.setattr(pg,'handoff',hashed_handoff)
    folder=tmp_path/'runs/fresh/prep';result=pg.prepare(tmp_path,folder)
    assert result['real_api_calls']==0 and result['ready_to_run'] is False
    assert read_json(folder/'handoff.json')['prep_identity']==digest(read_json(folder/'prep.json'))
    assert read_json(folder/'authorization.json')['approved'] is False
