"""Fresh benchmark authority, conformance, and held-out denial with fake ports."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import os
import pytest
from multi_dataset_diverse_rl.benchmarks.math_v21_binding import MATHV21Binding
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.benchmarks.access import DataPurpose
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker, BenchmarkSolver, ReflectionProvider, PatternProvider
from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import import_frozen_gepa

ROOT=Path(__file__).resolve().parents[1]

def contract(phase='pilot'):
    return json.loads((ROOT/f'experiments/execution_bindings/math_v2_1_{phase}_v4.json').read_bytes())


def test_output_repair_is_immutable_and_request_accounting_matches_solver():
    from multi_dataset_diverse_rl.benchmarks.math_v21_interface import interface_for_contract, v5_interface_contract, MATH_SOLVER_INTERFACE_V4_USER_SUFFIX
    from multi_dataset_diverse_rl.benchmarks.math_accounting_prep import solver_request
    from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import final_payload
    c=contract();adapter=MATHV21Binding(ROOT,c).benchmark()
    assert adapter.solver_interface_contract()==v5_interface_contract()
    assert solver_request(c,'Synthetic decision procedure.','Synthetic arithmetic.')['messages'][0]['content']==adapter.output_contract
    request=solver_request(c,'Synthetic decision procedure.','Synthetic arithmetic.')
    assert request['messages'][1]['content'].endswith(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX)
    assert request['messages'][1]['content'].startswith('Synthetic decision procedure.\n\nSynthetic arithmetic.')
    captured=[]
    broker=RequestBroker(contract=c,transport=lambda r:captured.append(r) or dict(text='FINAL_ANSWER: 1',input_tokens=1,output_tokens=1),arm='A1',seed=81,ledger_writer=lambda _:None)
    item=protocol_input('math','synthetic',{'problem':'Synthetic arithmetic.'},adapter.output_contract,protocol=adapter.protocol)
    BenchmarkSolver(adapter,broker).solve('Synthetic decision procedure.',item,stage='initial',split='optimize')
    assert captured==[request]
    assert captured[0]['max_tokens']==3600
    broker.complete(role='reflection',split='optimize',stage='synthetic',messages=[{'role':'user','content':'Synthetic optimizer context.'}])
    assert captured[-1]['max_tokens']==1800
    from multi_dataset_diverse_rl.benchmarks.math_accounting_prep import ValidationReserve
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request,reservation
    problems=['Synthetic arithmetic.','Synthetic relation.'];prompt='Synthetic procedure.'
    metadata=dict(decoding=c['decoding'],examples=[dict(blank_prompt_serialized_request_bytes=len(serialized_request(solver_request(c,'',p)))) for p in problems])
    reserve=ValidationReserve(metadata,(prompt,)*5)
    assert reserve.remaining()==2*5*sum(reservation(solver_request(c,prompt,p))['amount'] for p in problems)
    bad=deepcopy(c);bad['decoding']['solver_max_output_tokens']=1800
    assert MATHV21Binding(ROOT,bad).blockers()
    assert final_payload('FINAL_ANSWER: 1')=='1'
    for bad in ('1','**FINAL_ANSWER: 1**','```FINAL_ANSWER: 1```','FINAL_ANSWER: 1\ncommentary','FINAL_ANSWER: 1\nFINAL_ANSWER: 2','FINAL_ANSWER:'):
        assert final_payload(bad) is None
    c['solver_output_interface']['sha256']='0'*64
    assert MATHV21Binding(ROOT,c).blockers()
    with pytest.raises(Exception,match='MATH_SOLVER_INTERFACE_BINDING_MISMATCH'):
        interface_for_contract(c)


def test_prior_v21_interface_contract_stays_reproducible():
    from multi_dataset_diverse_rl.benchmarks.math_interface import MATH_SOLVER_INTERFACE_V2,solver_interface_contract
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_canary_v1.json').read_bytes())
    binding=MATHV21Binding(ROOT,c)
    assert not binding.blockers()
    assert binding.benchmark().output_contract==MATH_SOLVER_INTERFACE_V2
    assert binding.benchmark().solver_interface_contract()==solver_interface_contract()
    from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATH_SOLVER_INTERFACE_V3, v3_interface_contract
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_canary_v2.json').read_bytes());binding=MATHV21Binding(ROOT,c)
    assert not binding.blockers()
    assert binding.benchmark().solver_interface_contract()==v3_interface_contract()
    item=protocol_input('math','synthetic',{'problem':'Synthetic arithmetic.'},MATH_SOLVER_INTERFACE_V3,protocol=binding.benchmark().protocol)
    assert binding.benchmark().solver_user_content('Synthetic procedure.',item)=='Synthetic procedure.\n\nSynthetic arithmetic.'

@pytest.mark.parametrize('phase',['canary','pilot'])
def test_fresh_binding_has_no_historical_execution_authority(phase):
    c=contract(phase);b=execution_binding(ROOT,c)
    assert isinstance(b,MATHV21Binding) and not b.blockers()
    assert b.method('A1').method=='unified_team_prompt_search_v2_1'
    assert b.method('A1').transition_policy=='initial_competence_team_gain_v2'
    assert b.method('A1').evidence_policy=='single_mechanism_evidence_v2'
    assert c['initial_competence_binding']['support_identity']==c['membership_hashes']['optimize']
    c['parent_binding_path']='experiments/execution_bindings/math_v2_execution_v1_2.json'
    assert b.blockers()==('HISTORICAL_EXECUTION_AUTHORITY_FORBIDDEN',)

@pytest.mark.parametrize('field',['method_identity','method_implementation_sha','models','budget','initial_competence_binding',
    'evaluator','verify_settings_sha256','transition_policy','allocation_policy','stop_policy','cache_namespace',
    'split_manifest_sha256','initial_team_sha256','membership_hashes','accounting_policy_sha256','provider_sdk'])
def test_semantic_and_operational_poison_denied_before_provider(field):
    c=contract();c[field]='poison'
    assert MATHV21Binding(ROOT,c).blockers()

@pytest.mark.parametrize('role',['validation','test'])
def test_heldout_raw_access_denied_before_file_open(role,monkeypatch):
    reader=MATHV21Binding(ROOT,contract()).reader()
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('heldout opened'))
    with pytest.raises(Exception,match='HELDOUT_SEARCH_ACCESS_FORBIDDEN'):
        reader.rows(role,DataPurpose.EVIDENCE)

def test_bound_manifest_requires_same_initial_support_and_fresh_components():
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,bound_preflight
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    m=preexecution_manifest(ROOT,source_sha='a'*40,binding_path='experiments/execution_bindings/math_v2_1_canary_v4.json',frozen=False)
    assert not validate_manifest_v2(ROOT,m)
    del m['initial_competence_binding']
    assert validate_manifest_v2(ROOT,m)
    m=preexecution_manifest(ROOT,source_sha='a'*40,binding_path='experiments/execution_bindings/math_v2_1_canary_v4.json',frozen=False)
    m['transition_identity']='common_safe_v1'
    assert 'MANIFEST_EXECUTION_BINDING_MISMATCH' in bound_preflight(ROOT,m)['blockers']

@pytest.mark.parametrize('arm',['A1','A2','A3','A4'])
def test_fresh_v21_binding_actual_gepa_four_arm_e2e(tmp_path,arm,monkeypatch):
    c=contract('canary');binding=MATHV21Binding(ROOT,c);adapter=binding.benchmark()
    prompts=tuple(m['prompt'] for m in json.loads((ROOT/c['initial_team_path']).read_bytes())['members'])
    def examples(role):
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',
            {'problem':f'Synthetic {role} arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'1')
            for i in range(6 if role=='optimize' else 300))
    monkeypatch.setattr(binding,'examples',examples)
    good='Check explicit constraints and verify relational consistency before deriving the mathematical result.'
    other='Enumerate cases carefully before deriving the mathematical result.'
    calls=[];contexts=[]
    def transport(req):
        calls.append(req)
        if req['model']=='qwen3-8b':
            assert req['messages'][0]['content']==adapter.output_contract
            prompt,problem=req['messages'][1]['content'].split('\n\n',1)
            from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATH_SOLVER_INTERFACE_V4_USER_SUFFIX
            assert problem.endswith(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX)
            problem=problem.removesuffix(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX)
            assert '"memory"' not in problem and '"pattern"' not in problem
            i=int(problem.rsplit(' ',1)[1].rstrip('.'))
            if problem.startswith('Synthetic shadow'):correct=prompt in {prompts[1],prompts[2],good,other}
            elif prompt in prompts:
                m=prompts.index(prompt);correct=m in {1,2} or m==0 and i>=3
            else:correct=i!=2 if prompt==good else i>=2
            return dict(text='FINAL_ANSWER: '+('1' if correct else '2'),input_tokens=2,output_tokens=2)
        if len(req['messages'])==2:
            data=json.loads(req['messages'][1]['content'])
            return dict(text=json.dumps({'patterns':[dict(pattern_id='synthetic',failure_mechanism='Missed constraints',
                corrective_principle='Check explicit constraints',support_ids=data['residual_ids'],counterexample_ids=[],risk_ids=[],confidence=.9)]}),input_tokens=2,output_tokens=2)
        return dict(text='```'+(good if len(contexts)==1 else other)+'```',input_tokens=2,output_tokens=2)
    broker=RequestBroker(contract=c,transport=transport,arm=arm,seed=81,ledger_writer=lambda _:None)
    solver=BenchmarkSolver(adapter,broker);reflection=ReflectionProvider(broker)
    pattern=PatternProvider(broker,json.loads((ROOT/c['pattern_prompt_path']).read_bytes())['prompt']) if c['arms'][arm][0] else None
    def official(**kwargs):
        contexts.append(kwargs['adapter'].optimization_context)
        return import_frozen_gepa().optimize(**kwargs)
    run=binding.compose(arm=arm,seed=81,solver=solver,reflection=reflection,pattern_provider=pattern,run_root=tmp_path,optimize_fn=official)
    run.state.initialize();initial=run.state.snapshot();result=asyncio.run(run.run(max_opportunities=3))
    assert result.transitions and run.state.initial_member_scores==initial.member_scores
    assert all(t.allocation_audit['realized_team_gain']>0 for t in result.trace if t.committed_candidate_id)
    assert all(t.allocation_audit['evaluator_identity']=='MATH_EQUIVALENCE_V2' for t in result.trace)
    assert all(t.allocation_audit['prior_exposure_counts']==dict.fromkeys(range(5),i) for i,t in enumerate(result.trace))
    assert broker.usage['validation']==broker.usage['test']==0
    assert bool(broker.usage['pattern'])==c['arms'][arm][0]
    if c['arms'][arm][1]:
        assert run.memory.private and all(e.action and e.lesson['successful_strategy'] for e in run.memory.private)
    output=dict(arm=arm,method=run.method.method,transitions=len(result.transitions),trace=[t.allocation_audit for t in result.trace],
        real_calls=0,ledger=broker.usage,initial_floor_immutable=True)
    dest=os.environ.get('FORMAL_V3_EVIDENCE_CAPTURE_DIR')
    if dest:
        Path(dest).mkdir(parents=True,exist_ok=True)
        (Path(dest)/f'math_v21_{arm}.json').write_text(json.dumps(output,sort_keys=True,indent=2)+'\n',encoding='utf-8')

@pytest.mark.parametrize('phase',['canary','pilot'])
def test_fresh_frozen_phase_fake_accounting_and_same_support_floor(tmp_path,monkeypatch,phase):
    from multi_dataset_diverse_rl.governance import unified_execution as gov,autonomous_math as execution
    c=contract(phase)
    c['token_ledger_directory']=(tmp_path/'budget').relative_to(ROOT).as_posix()
    c['binding_path']=(tmp_path/'binding.json').relative_to(ROOT).as_posix()
    (ROOT/c['binding_path']).write_text(json.dumps(c,sort_keys=True)+'\n',encoding='utf-8')
    monkeypatch.setattr(gov,'verify_source_commit',lambda *_:None)
    monkeypatch.setattr(execution,'consumption_path',lambda *_:tmp_path/'consumed.json')
    m=gov.preexecution_manifest(ROOT,source_sha='a'*40,binding_path=c['binding_path'],experiment_id='synthetic_'+phase)
    prep=tmp_path/'prep';gov.prepare_canary(ROOT,m,destination=prep)
    auth=json.loads((prep/'authorization.json').read_bytes());auth['explicit_user_authorized']=True
    (prep/'authorization.json').write_text(json.dumps(auth),encoding='utf-8')
    original=MATHV21Binding(ROOT,c);adapter=original.benchmark();metadata=original.reader().members
    def examples(self,role):
        return tuple(CorrectnessExample(protocol_input('math',r['stable_example_id'],
            {'problem':'Synthetic isolated arithmetic.'},adapter.output_contract,protocol=adapter.protocol),'1')
            for r in metadata if r['project_split']==role)
    monkeypatch.setattr(MATHV21Binding,'examples',examples)
    prompts=tuple(m['prompt'] for m in json.loads((ROOT/c['initial_team_path']).read_bytes())['members'])
    good='Check explicit constraints and verify relational consistency before deriving the mathematical result.'
    calls=[]
    def transport(request):
        calls.append(request)
        if request['model']=='qwen3-8b':
            prompt=request['messages'][1]['content'].split('\n\n')[0]
            text='FINAL_ANSWER: '+('1' if prompt in {prompts[1],prompts[2],good} else '2')
        else:text='```'+good+'```'
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop')
    class Client:
        def close(self):pass
    monkeypatch.setattr(execution,'create_transport',lambda _:(transport,Client()))
    run=tmp_path/'run';summary=asyncio.run(gov.execute_canary(ROOT,prep,run))
    floor=json.loads((run/'initial_competence_floor.json').read_bytes())
    assert floor['member_scores']==[0,150,150,0,0]
    assert floor['binding']['support_identity']==c['membership_hashes']['optimize']
    assert summary['result']['transitions'] and summary['accounting']['charged_total']==4*len(calls)
    assert summary['accounting']['reserved_inflight']==summary['validation_calls']==summary['test_calls']==0
    assert (run/'SEARCH_COMPLETE_RECEIPT.json').exists()==(phase=='pilot')
    assert summary['result']['trace'][0]['allocation_audit']['realized_team_gain']==150
    if phase=='pilot':
        receipt=json.loads((run/'SEARCH_COMPLETE_RECEIPT.json').read_bytes())
        assert receipt['search_closed_forever'] and receipt['validation_model_calls']==0
    with pytest.raises(Exception,match='EXACT_SINGLE_USE_AUTHORIZATION_REQUIRED'):
        gov.validate_prep(ROOT,prep,require_authorized=True)
