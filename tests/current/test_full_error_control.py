"""All-error control uses native editing/gating and never calls diagnosis roles."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from multi_dataset_diverse_rl.research.ablations import full_error_control as c, governance as g
from multi_dataset_diverse_rl.research.ablations.config import build_plan, validate_config, validate_plan
from multi_dataset_diverse_rl.research.ablations.engine import run_cell
from multi_dataset_diverse_rl.research.ablations.runtime import AblationBroker, execution_concurrency
from multi_dataset_diverse_rl.research.ablations.proposal_continuation import broker_type, SchedulingResumeBroker
from multi_dataset_diverse_rl.research.ablations.modules import NoModule
from multi_dataset_diverse_rl.research.ablations.interfaces import OptimizeContext
from multi_dataset_diverse_rl.research.ablations.synthetic import synthetic_rows, SyntheticTransport
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.system_prompt import SEED, SystemPrompt
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest, file_hash
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_PATTERN_96_UNCAPPED, audit_journal


def test_exact_control_and_old_defaults():
    from multi_dataset_diverse_rl.research.ablations.config import current_config
    from multi_dataset_diverse_rl.research.ablations.proposal_config import default_config
    plan = build_plan(c.default_config())
    validate_plan(plan)
    assert len(plan['cells']) == 3 and all(v['seed'] == 42 for v in plan['cells'])
    assert [v['repeat_id'] for v in plan['cells']] == [1, 2, 3]
    assert all(v['module'] == 'none' and v['error_source'] == 'all_wrong' for v in plan['cells'])
    assert plan['training_semantic_call_ceiling'] == 123480
    assert plan['final_test_semantic_call_ceiling'] == 3600
    assert plan['module_bindings']['A2']['semantic_calls_per_opportunity'] == 0
    assert current_config()['runtime']['solver_concurrency'] == 32
    assert len(default_config()['arms']) == 3


@pytest.mark.parametrize('change', ['rounds', 'seed', 'module', 'gate_seed', 'concurrency'])
def test_unfrozen_policy_change_rejected(change):
    value = c.default_config()
    if change == 'rounds': value['rounds'] = value['runtime']['rounds'] = 1
    elif change == 'seed': value['seed'] = 43
    elif change == 'module': value['arms'][0]['module'] = 'full_gradient_pattern'
    elif change == 'gate_seed': value['runtime']['validation_membership_seed'] = 88
    else: value['runtime']['solver_concurrency'] = 32
    with pytest.raises(SearchContractError): validate_config(value)


def observation(correct=False, answer='0'):
    return dict(correct=correct, valid=True, answer=answer,
        prediction=dict(text='Synthetic response', invalid_reason=None, semantic_attempt_count=1))


def test_more_than_three_errors_reach_native_packet_and_audit():
    from multi_dataset_diverse_rl.research.ablations.generation import optimizer_packet
    rows = synthetic_rows('optimize', 8)
    values = {r['stable_example_id']: observation(i == 3) for i,r in enumerate(rows)}
    wrong = tuple(r for r in rows if not values[r['stable_example_id']]['correct'])
    context = OptimizeContext('A2_seed42_repeat01',42,1,1,SEED,wrong,values)
    feedback = NoModule().prepare(context,None,None)
    packet = optimizer_packet(SEED,feedback.errors,feedback.structured_feedback,feedback.module_context)
    opportunity = dict(errors=feedback.errors,structured_feedback=None,module_context=None)
    c.audit_feedback(opportunity,dict(optimize=values),list(values))
    assert len(packet['errors']) == 7 and packet['structured_feedback'] is None
    assert 'module_context' not in packet
    bad = deepcopy(opportunity); bad['errors'] = bad['errors'][:3]
    with pytest.raises(SearchContractError,match='ALL_ERRORS'): c.audit_feedback(bad,dict(optimize=values),list(values))
    bad = deepcopy(opportunity); bad['errors'][0]['solver_response'] = 'Other Parent'
    with pytest.raises(SearchContractError,match='PARENT_RESPONSE'): c.audit_feedback(bad,dict(optimize=values),list(values))
    bad = deepcopy(opportunity); bad['structured_feedback'] = {'patterns': []}
    with pytest.raises(SearchContractError): c.audit_feedback(bad,dict(optimize=values),list(values))


def test_train_test_execution_and_no_resume_broker():
    from multi_dataset_diverse_rl.research.ablations.proposal_continuation import CONCURRENCY, PREP_SCHEMA
    plan = build_plan(c.default_config())
    scope = dict(source_sha='a'*40,scope_identity='b'*64,phase='train',plan=plan,
        test_plan=dict(successful_semantic_call_ceiling=3600),network_recovery={},execution_concurrency=c.CONCURRENCY)
    test = g.final_test_scope(scope,'c'*64)
    assert execution_concurrency(scope) == execution_concurrency(test) == c.CONCURRENCY
    assert 'continuation' not in test and 'source_hashes' not in test
    assert broker_type(scope) is AblationBroker and broker_type(test) is AblationBroker
    assert g.accounting_policy(scope,'train') == g.accounting_policy(test,'test') == POLICY_PATTERN_96_UNCAPPED
    old = dict(phase='train',execution_concurrency=CONCURRENCY,continuation=dict(identity=PREP_SCHEMA))
    assert broker_type(old) is SchedulingResumeBroker
    broker = AblationBroker.__new__(AblationBroker); broker.plan = plan; broker.phase = 'train'
    assert broker.legal_role_split('solver','validation') and broker.legal_role_split('reflection','optimize')
    assert not broker.legal_role_split('pattern_gradient','optimize')
    assert not broker.legal_role_split('pattern_cluster','optimize')
    broker.phase = 'test'
    assert broker.legal_role_split('solver','test') and not broker.legal_role_split('reflection','optimize')
    for field in c.CONCURRENCY:
        bad = deepcopy(scope); bad['execution_concurrency'][field] += 1
        with pytest.raises(SearchContractError): execution_concurrency(bad)


class ScriptedBroker:
    phase = 'train'
    scope = {}
    execution_policy = {}
    def __init__(self, exhaust_first=False, tie=False):
        self.calls=[]; self.evaluations=[]; self.events=[]; self.exhaust_first=exhaust_first; self.tie=tie
    def bind_cell(self,cell,plan): pass
    def usage_view(self): return dict(provider_calls=len(self.calls),input_tokens=0,output_tokens=0)
    def evaluate(self,prompt,rows,split,seed,ordinal):
        self.evaluations.append((prompt,split,ordinal));self.events.append(('evaluate',ordinal,split))
        count=(1 if self.tie or prompt.strategy=='Repair one' else 2) if prompt.strategy in ('Repair one','Repair two') else 0
        return {r['stable_example_id']: observation(split=='validation' and i<count,'0') for i,r in enumerate(rows)}
    def complete(self,**kwargs):
        self.calls.append(kwargs)
        packet=json.loads(kwargs['messages'][1]['content'])
        slot=int(kwargs['stage'].split('_proposal_')[1][0])
        self.events.append(('generate',slot,kwargs['stage']))
        if slot==0 and (self.exhaust_first or 'output_contract_feedback' not in packet):
            text='invalid synthetic format'
        else:
            text=json.dumps(dict(block='strategy',content='Repair one' if slot==0 else 'Repair two'))
        return dict(text=text,finish_reason='stop',request_sha256=digest(kwargs))


def run_scripted(broker):
    plan=build_plan(c.default_config());cell=plan['cells'][0];artifacts={}
    result=run_cell(plan,cell,SEED.to_dict(),synthetic_rows('optimize',7),synthetic_rows('validation',2),
        broker,lambda k,v:artifacts.setdefault(k,deepcopy(v)))
    return plan,cell,artifacts,result


def test_actual_c2_kernel_same_parent_all_errors_two_proposals_and_strict_gate():
    broker=ScriptedBroker();plan,cell,a,result=run_scripted(broker)
    from multi_dataset_diverse_rl.research.ablations.proposal_audit import audit_transition
    assert result['rounds_completed']==20 and result['final_prompt']['strategy']=='Repair two'
    m=result['metrics'][0]
    assert len(m['proposals'])==2 and m['proposals'][0]['generation_draws']==2
    assert [p['accepted'] for p in m['proposals']]==[False,True]
    assert all(p['parent_prompt']==SEED.to_dict() for p in m['proposals'])
    assert result['metrics'][1]['decision']=='REJECTED_NO_STRICT_VALIDATION_GAIN'
    assert all(v['role']=='reflection' for v in broker.calls)
    # Both full-error proposals are generated before the first candidate evaluation.
    first_eval=next(i for i,e in enumerate(broker.events) if e[0]=='evaluate' and e[1]==1)
    assert [e[1] for e in broker.events[2:first_eval] if e[0]=='generate']==[0,0,1]
    read=lambda name:deepcopy(a[name])
    for n in range(1,21):
        audit_transition(a[f'checkpoint_{n-1}'],a[f'checkpoint_{n}'],plan,cell,n,read)
        assert len(a[f'checkpoint_{n}']['metric']['proposals'])==2
        for slot in (0,1):
            op=a[f'round_{n}_proposal_{slot}_opportunity']
            c.audit_feedback(op,a[f'checkpoint_{n-1}'],list(a['checkpoint_0']['optimize']))
            assert len(op['errors'])==7 and op['structured_feedback'] is None
        assert all(v==0 for v in a[f'checkpoint_{n}']['metric']['shared_diagnostic_cost'].values())
    # Removing raw evidence or relabeling the paid Parent response fails the terminal audit.
    bad=deepcopy(a);bad['round_1_module']['audit']['raw_feedback']['errors'].pop()
    with pytest.raises(SearchContractError):
        audit_transition(bad['checkpoint_0'],bad['checkpoint_1'],plan,cell,1,lambda k:bad[k])
    bad=deepcopy(a);bad['round_1_proposal_1_opportunity']['errors'][0]['solver_response']='Another Parent'
    with pytest.raises(SearchContractError):
        audit_transition(bad['checkpoint_0'],bad['checkpoint_1'],plan,cell,1,lambda k:bad[k])


def test_slot_exhaustion_does_not_block_second_and_ties_use_slot_order():
    _,_,_,result=run_scripted(ScriptedBroker(exhaust_first=True))
    first=result['metrics'][0]
    assert first['proposals'][0]['generation_status']=='EXHAUSTED'
    assert first['proposals'][1]['accepted'] and first['accepted_update_count']==1
    _,_,_,result=run_scripted(ScriptedBroker(tie=True))
    assert [p['accepted'] for p in result['metrics'][0]['proposals']]==[True,False]
    assert result['metrics'][1]['accepted'] is False


def test_native_guarded_broker_cache_and_real_role_routing(tmp_path):
    plan=build_plan(c.default_config()); scope_id='d'*64
    scope=dict(scope_identity=scope_id,phase='train',call_ceiling=123480,token_ceiling=None,
        plan=plan,execution_concurrency=c.CONCURRENCY,generation_recovery=plan['config']['runtime']['diagnosis_recovery'])
    ledger=TokenLedger(tmp_path/'ledger',task_sha256=scope_id,policy=POLICY_PATTERN_96_UNCAPPED)
    transport=SyntheticTransport()
    broker=AblationBroker(plan=plan,phase='train',scope=scope,transport=transport,ledger=ledger,
        receipts=ProviderResponseReceipts(tmp_path/'receipts',attempt_id='fake',startup_identity_sha256=scope_id),
        dispatch_guard=lambda _:None)
    try:
        cell=plan['cells'][0];broker.bind_cell(cell,plan)
        rows=synthetic_rows('optimize',2)
        first=broker.evaluate(SEED,rows,'optimize',42,0)
        calls=broker.usage_view()['provider_calls']
        assert broker.evaluate(SEED,rows,'optimize',42,1)==first
        assert broker.usage_view()['provider_calls']==calls and broker.usage_view()['cache_hits']==2
        broker.bind_cell(plan['cells'][1],plan)
        broker.evaluate(SEED,rows,'optimize',42,0)
        assert broker.usage_view()['provider_calls']==calls+2
        assert all(role=='solver' for role,_ in transport.calls)
        assert broker.study_policy()['solver_concurrency']==96 and broker.grading_slots is not None
    finally: ledger.close()
    assert audit_journal(tmp_path/'ledger',task_sha256=scope_id,policy=POLICY_PATTERN_96_UNCAPPED)[0]['reserved_inflight']==0


def test_exact_auth_gate_and_entrypoint_before_transport(tmp_path,monkeypatch):
    from scripts.run_experiment import preflight
    assert preflight(c.default_config())['train_call_ceiling']==123480
    plan=build_plan(c.default_config());scope=dict(scope_identity='e'*64,phase='train',plan=plan,execution_concurrency=c.CONCURRENCY)
    folder=tmp_path/'runs/control/prep';handoff=dict(READY_TO_RUN=False,status='WAIT')
    monkeypatch.setattr(g,'require_selected_runtime',lambda root:None)
    monkeypatch.setattr(g,'_git',lambda *a:'')
    monkeypatch.setattr(c,'scope_template',lambda *a:deepcopy(scope))
    monkeypatch.setattr(c,'handoff',lambda *a:deepcopy(handoff))
    monkeypatch.setattr(g,'execution_plan_template',lambda *a:dict(command='pinned'))
    auth=g.authorization_template(scope)
    for name,value in [('prep',dict(schema_version=c.PREP_SCHEMA,scope=scope)),('authorization',auth),
                       ('handoff',handoff),('execution_plan',dict(command='pinned'))]:
        atomic_write_json(folder/(name+'.json'),value)
    assert g.validate_prep(tmp_path,folder)==scope
    def forbidden(): raise AssertionError('No provider before exact authorization')
    monkeypatch.setattr(g,'_transport',forbidden)
    with pytest.raises(SearchContractError): g._execute_phase(tmp_path,folder,folder.parent/'train_run',scope,train_scope=scope)
    proof=folder/'user_authorization_evidence.json';atomic_write_json(proof,dict(synthetic_grant=True))
    auth.update(approved=True,adaptive_validation_access=True,test_access=True,user_authorization_evidence_sha256=file_hash(proof))
    atomic_write_json(folder/'authorization.json',auth)
    atomic_write_json(folder/'handoff.json',dict(handoff,READY_TO_RUN=True))
    assert g.validate_prep(tmp_path,folder,authorized=True)==scope
    atomic_write_json(proof,dict(synthetic_grant='changed'))
    with pytest.raises(SearchContractError): g.validate_prep(tmp_path,folder,authorized=True)


def test_matched_data_metadata_only():
    root=Path(__file__).resolve().parents[2]
    parent,reader=c.inputs(root)
    # Public manifest comparison only: no private run or held-out content is needed.
    from multi_dataset_diverse_rl.research.ablations.proposal_governance import inputs
    parent2,reader2=inputs(root)
    assert parent['initial_prompt_sha256']==parent2['initial_prompt_sha256']
    assert g.gate_membership(reader,42)==g.gate_membership(reader2,42)
    assert g.test_plan(root,reader,build_plan(c.default_config()))['count']==300


@pytest.mark.parametrize('change',[None,'one_slot','wrong_gate','diagnosis'])
def test_repository_registry_explicitly_validates_new_control(tmp_path,change):
    from multi_dataset_diverse_rl.governance.repository import validate_registry_v2
    from multi_dataset_diverse_rl.governance.registries import load_yaml
    root=Path(__file__).resolve().parents[2]
    row=deepcopy(load_yaml(root/'experiments/registry.yaml')['experiments'][0])
    assert row['experiment_id']=='single_prompt_full_error_control_v2'
    row.update(parent_ids=[],manifest='experiments/manifests/control.json',report='report')
    (tmp_path/'report').mkdir();value=c.default_config()
    if change=='one_slot':value['runtime']['proposals_per_round']=1
    elif change=='wrong_gate':value['runtime']['validation_membership_seed']=88
    elif change=='diagnosis':value['arms'][0]['module']='full_gradient_pattern'
    atomic_write_json(tmp_path/row['manifest'],value)
    errors,_=validate_registry_v2(tmp_path,dict(experiments=[row]))
    assert not errors if change is None else any('FULL_ERROR_CONTROL_CONFIG' in e for e in errors)
