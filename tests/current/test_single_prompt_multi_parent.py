"""Actual native strategies, labeled evidence, independent gates and offline seals."""
from copy import deepcopy
from pathlib import Path
import json
import pytest

from multi_dataset_diverse_rl.research.ablations.config import current_config,build_plan,validate_plan
from multi_dataset_diverse_rl.research.ablations.engine import run_cell
from multi_dataset_diverse_rl.research.ablations.multi_parent import SinglePromptPool,POLICY
from multi_dataset_diverse_rl.research.ablations.population_audit import audit_cells
from multi_dataset_diverse_rl.research.ablations.offline_runtime import OfflineAblationBroker
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport,synthetic_rows
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.population_policies import FitnessWeightedSampling,BestThreeRetention,PerMemberArgmaxAssembly
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_GATED_32_UNCAPPED


def obs(n,correct):return {str(i):dict(correct=i<correct,valid=True) for i in range(n)}


def test_native_classes_and_unprotected_top_three():
    p=SinglePromptPool(('0','1','2'))
    assert type(p.sampling) is FitnessWeightedSampling and type(p.retention) is BestThreeRetention
    assert type(p.assembly) is PerMemberArgmaxAssembly
    p.update(SEED,obs(3,0))
    for i in range(3):p.update(SEED.edit('strategy',f'child {i}'),obs(3,i+1))
    assert len(p.parents)==3 and SEED.prompt_hash not in p.parents
    assert p.best().fitness.correct==3


def test_sampling_is_identical_across_repeats_and_has_positive_mass():
    left,right=SinglePromptPool(('0','1')),SinglePromptPool(('0','1'))
    for pool in (left,right):
        pool.update(SEED,obs(2,0));pool.update(SEED.edit('answer','Different'),obs(2,1))
    assert left.draw(42,4)[1]==right.draw(42,4)[1]
    _,plan=left.draw(42,4)
    assert len(plan['selected_prompt_sha256'])==3 and all(x['probability']>0 for x in plan['distribution'])
    assert left.draw(42,5)[1]['round']==5


def test_default_identity_budget_and_all_seed_read_points():
    c=current_config();p=build_plan(c);validate_plan(p)
    assert len(p['cells'])==9 and c['rounds']==20 and c['seed']==42
    assert c['runtime']['validation_membership_seed']==42
    assert p['training_semantic_call_ceiling']==437940 and p['final_test_semantic_call_ceiling']==10800
    assert c['runtime']['population']==POLICY and not p['ready_to_run']
    c['runtime']['population']['capacity']=4
    with pytest.raises(SearchContractError):build_plan(c)


@pytest.fixture
def study(tmp_path):
    c=current_config();c['rounds']=c['runtime']['rounds']=2
    c['repeats_per_arm']=c['runtime']['repeats_per_arm']=1
    p=build_plan(c);cell=p['cells'][0]
    class Transport(SyntheticTransport):
        draws=0
        def for_role(self,role,request):
            value=super().for_role(role,request)
            if role=='reflection':
                self.draws+=1;value['text']=json.dumps(dict(block='strategy',content=f'Method {self.draws}'))
            return value
    transport=Transport();sid=digest('offline multi parent')
    ledger=TokenLedger(tmp_path/'ledger',task_sha256=sid,policy=POLICY_GATED_32_UNCAPPED)
    broker=OfflineAblationBroker(plan=p,phase='train',scope=dict(scope_identity=sid,
        call_ceiling=p['training_semantic_call_ceiling'],token_ceiling=None,network_recovery=True),
        transport=transport,ledger=ledger,
        receipts=ProviderResponseReceipts(tmp_path/'receipts',attempt_id='fake',startup_identity_sha256=sid),dispatch_guard=lambda _:None)
    rows,validation=synthetic_rows('optimize',3),synthetic_rows('validation',3)
    folder=tmp_path/cell['cell_id'];folder.mkdir()
    def persist(name,value):atomic_write_json(folder/(name+'.json'),value)
    yield p,cell,broker,transport,rows,validation,persist,tmp_path
    assert not ledger.inflight;ledger.close()


def run(study):
    p,c,b,t,rows,val,persist,root=study
    result=run_cell(p,c,SEED.to_dict(),rows,val,b,persist)
    return result


def test_three_frozen_slots_share_root_and_only_one_validation_winner(study):
    p,c,b,t,rows,val,_,root=study
    requests=[];original=b.evaluate
    def evaluate(prompt,rows,split,seed,ordinal):
        requests.append((prompt.prompt_hash,split,ordinal));return original(prompt,rows,split,seed,ordinal)
    b.evaluate=evaluate;result=run(study)
    assert t.draws==6 and len(result['metrics'])==2
    assert sum(split=='validation' for _,split,_ in requests)==3  # final reuses a recorded winner/root.
    assert all(len(m['selection']['selected_prompt_sha256'])==3 for m in result['metrics'])
    assert result['metrics'][0]['selection']['selected_prompt_sha256']==[SEED.prompt_hash]*3
    first_new={s['candidate']['strategy'] for s in result['metrics'][0]['slots']}
    assert len(first_new)==3
    assert all(not m['accepted'] for m in result['metrics'])  # All wrong; pool still retains children.
    assert len(read_json(root/c['cell_id']/'checkpoint_2.json')['pool'])==3
    packets=[json.loads(q['messages'][1]['content']) for role,q in t.calls if role=='reflection']
    assert all(x['parent_context']['diagnosis_source']=='running_root_not_selected_parent' for x in packets)
    assert all('validation' not in x and 'pool' not in x for x in packets)


def test_checkpoint_audit_and_tampering(study):
    p,c,b,t,rows,val,_,root=study;result=run(study)
    from multi_dataset_diverse_rl.research.ablations.generation import audit_optimizer_request
    audited=0
    for path in (root/'receipts').glob('*.json'):
        record=read_json(path)['record']
        if record['logical_identity']['role']=='reflection':
            audit_optimizer_request(root,c['cell_id'],record);audited+=1
    assert audited==6
    # Limit the audit to the actual subset of cells executed in this fixture.
    scope=dict(plan=deepcopy(p),initial_prompt_sha256=SEED.prompt_hash,
        optimize_sample_ids=[r['stable_example_id'] for r in rows],
        gate_membership=dict(sample_ids=[r['stable_example_id'] for r in val]))
    scope['plan']['cells']=[c]
    assert audit_cells(root,scope,[result])==[read_json(root/c['cell_id']/'result.json')]
    path=root/c['cell_id']/'checkpoint_1.json';value=read_json(path)
    value['metric']['selection']['selected_prompt_sha256'][0]='0'*64
    value['integrity_sha256']=digest({k:v for k,v in value.items() if k!='integrity_sha256'})
    atomic_write_json(path,value)
    with pytest.raises(SearchContractError):audit_cells(root,scope,[result])


def test_final_rejection_never_silently_substitutes_running_prompt(study):
    p,c,b,t,rows,val,_,root=study;original=b.evaluate
    def evaluate(prompt,rows,split,seed,ordinal):
        values=original(prompt,rows,split,seed,ordinal)
        if split=='validation' and ordinal==0:
            for v in values.values():v['correct']=True
        if split=='optimize' and prompt!=SEED:
            for v in values.values():v['correct']=True
        return values
    b.evaluate=evaluate;result=run(study)
    # Initial is tied for Optimize with children; make final selection deterministic via the pool's real hash order.
    final=result['final_admission']
    assert final['admitted'] is False and final['status']=='FINAL_PROPOSAL_REJECTED'
    assert result['final_prompt']==final['proposal_prompt']
    if not final['admitted']:
        from multi_dataset_diverse_rl.research.ablations import governance as gov
        with pytest.raises(SearchContractError,match='REJECTED_TEST_FORBIDDEN'):
            gov._execute_phase(root,root,root/'test',dict(phase='test'),train_scope=dict(plan=p),parent_results=[result])


def test_only_strict_winner_gain_updates_running_root(study):
    p,c,b,t,rows,val,_,root=study;original=b.evaluate
    def evaluate(prompt,rows,split,seed,ordinal):
        values=original(prompt,rows,split,seed,ordinal)
        if prompt!=SEED:
            number=int(prompt.strategy.split()[-1])
            correct=min(number,3) if split=='optimize' else 2
            for index,v in enumerate(values.values()):v['correct']=index<correct
        return values
    b.evaluate=evaluate;result=run(study)
    assert [m['accepted'] for m in result['metrics']]==[True,False]
    assert result['metrics'][0]['winner_slot']==2
    assert result['metrics'][0]['validation_candidate']==2
    assert result['metrics'][1]['validation_before']==result['metrics'][1]['validation_candidate']==2
    assert result['prompt_history'][1]==result['prompt_history'][2]


def test_validation42_membership_is_explicit_and_old_membership_unmodified():
    from multi_dataset_diverse_rl.research.ablations.governance import gate_membership
    reader=type('Reader',(),dict(members=[dict(project_split='validation',stable_example_id=str(i)) for i in range(300)]))()
    fresh=gate_membership(reader,42);old=gate_membership(reader)
    assert fresh!=old and fresh['count']==100 and 'SEED42' in fresh['identity']
    assert gate_membership(reader,42)==fresh and 'SEED88' in old['identity']


def test_C_diagnoses_root_once_and_never_replays_teacher(study):
    p,c,b,t,rows,val,persist,root=study
    c=p['cells'][2];folder=root/c['cell_id'];folder.mkdir()
    result=run_cell(p,c,SEED.to_dict(),rows,val,b,lambda name,v:atomic_write_json(folder/(name+'.json'),v))
    assert b.usage_view()['pattern_gradient']==6 and b.usage_view()['pattern_cluster']==2
    assert b.usage_view()['reflection']==6 and result['rounds_completed']==2
    packets=[json.loads(q['messages'][-1]['content']) for role,q in t.calls if role=='reflection']
    assert all(x['structured_feedback']['patterns'] for x in packets)
    scope=dict(plan=dict(p,cells=[c]),initial_prompt_sha256=SEED.prompt_hash,
        optimize_sample_ids=[r['stable_example_id'] for r in rows],
        gate_membership=dict(sample_ids=[r['stable_example_id'] for r in val]))
    assert audit_cells(root,scope,[result])==[read_json(folder/'result.json')]


@pytest.mark.parametrize('mode,draws,decision',[('no_op',6,'SKIPPED_NO_OP'),('malformed',24,'ABANDONED_GENERATION_RETRIES_EXHAUSTED')])
def test_slot_skip_and_exhaustion_leave_root_profiles_untouched(study,mode,draws,decision):
    p,c,b,t,rows,val,persist,root=study;original=t.for_role;calls=[];evaluate=b.evaluate
    def respond(role,request):
        v=original(role,request)
        if role=='reflection':
            parent=json.loads(request['messages'][1]['content'])['current_system_prompt']
            v['text']=json.dumps(dict(block='strategy',content=parent['strategy'])) if mode=='no_op' else '{bad'
        return v
    def measure(*args):calls.append(args[2]);return evaluate(*args)
    t.for_role=respond;b.evaluate=measure;result=run(study)
    assert t.draws==draws and all(m['decision']==decision for m in result['metrics'])
    assert calls==['optimize','validation']  # Neither slots nor closure reroll the incumbent.
    assert result['final_admission']['admitted'] and result['final_prompt']==SEED.to_dict()
