"""Low-cost membership, protocol poison, wire parity, and actual pinned GEPA."""
import asyncio,json,os
from pathlib import Path
from copy import deepcopy
import pytest
from multi_dataset_diverse_rl.benchmarks.math_low_cost_binding import MATHLowCostBinding
from multi_dataset_diverse_rl.benchmarks.math_low_cost import read_subsets,build_subsets,choose
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver,ReflectionProvider,PatternProvider
from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import import_frozen_gepa
ROOT=Path(__file__).resolve().parents[1]
def contract(phase='pilot'):
    return json.loads((ROOT/f'experiments/execution_bindings/math_v2_1_low_cost_{phase}_v1.json').read_bytes())

def test_subsets_exact_deterministic_nested_and_disjoint():
    c=contract();d=read_subsets(ROOT,c)
    assert d==build_subsets(d['metadata_universe'],c['split_manifest_sha256'])
    sets={k:{r['stable_example_id'] for r in v} for k,v in d['memberships'].items()}
    assert {k:len(v) for k,v in sets.items()}==d['counts']
    assert sets['canary_optimize']<=sets['pilot_optimize']
    assert not sets['pilot_optimize']&sets['pilot_shadow']
    assert not sets['pilot_optimize']&sets['pilot_validation']
    assert not sets['pilot_shadow']&sets['pilot_validation']
    # Small strata receive no artificial minimum allocation.
    rows=[dict(stable_example_id=str(i),subject='large' if i<99 else 'tiny',level='1') for i in range(100)]
    assert all(r['subject']=='large' for r in choose(rows,12,'synthetic'))

@pytest.mark.parametrize('phase',['canary','pilot'])
def test_binding_manifest_authorization_and_budget(phase):
    c=contract(phase);assert not MATHLowCostBinding(ROOT,c).blockers()
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,execution_scope
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    m=preexecution_manifest(ROOT,source_sha='a'*40,binding_path=c['binding_path'],frozen=False)
    assert not validate_manifest_v2(ROOT,m)
    scope=execution_scope(m,c)
    assert scope['accounting']['total_authorization']==40000000
    assert scope['invalid_recovery_policy']==c['invalid_recovery_policy']
    assert scope['low_cost_protocol']==c['low_cost_protocol']
    assert c['post_search_validation_policy']['logical_evaluations']==1000
    assert c['post_search_validation_policy']['successful_provider_call_ceiling']==4000


def test_all_solver_roles_exact_sdk_wire(monkeypatch):
    from tests import test_math_solver_decoding as wire
    monkeypatch.setattr(wire,'contract',lambda *_:contract('canary'))
    wire.test_all_solver_roles_and_evolved_prompts_reach_exact_sdk_wire(monkeypatch)

@pytest.mark.parametrize('field',['models','budget','solver_decoding_policy','decoding','invalid_recovery_policy',
    'initial_competence_binding','low_cost_protocol','low_cost_subsets_sha256','cache_namespace','shadow_count',
    'accounting_policy_sha256','transition_policy','allocation_policy','stop_policy','evaluator','initial_team_sha256'])
def test_scientific_poison_denied(field):
    c=contract();c[field]='poison'
    assert MATHLowCostBinding(ROOT,c).blockers()


@pytest.mark.parametrize('phase',['canary','pilot'])
def test_frozen_entrypoint_recovery_durable_floor_and_receipt(tmp_path,monkeypatch,phase):
    from multi_dataset_diverse_rl.governance import unified_execution as gov,autonomous_math as execution
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger
    from multi_dataset_diverse_rl.benchmarks.math_low_cost import read_subsets
    c=contract(phase);b=MATHLowCostBinding(ROOT,c);adapter=b.benchmark()
    source_members=read_subsets(ROOT,c)['memberships']
    good='Check explicit constraints and verify relational consistency before deriving the mathematical result.'
    prompts=tuple(m['prompt'] for m in json.loads((ROOT/c['initial_team_path']).read_bytes())['members'])
    monkeypatch.setattr(gov,'verify_source_commit',lambda *_:None)
    monkeypatch.setattr(execution,'consumption_path',lambda *_:tmp_path/'consumed.json')
    isolated=tmp_path/'budget'
    with TokenLedger(isolated,task_sha256=c['task_authorization_sha256']) as ledger:
        ledger.amend_authorization_40m(authorization_sha256=c['continuation_authorization_sha256'],expected_charged_total=0)
    monkeypatch.setattr(execution,'TokenLedger',lambda directory,**kwargs:TokenLedger(isolated,**kwargs))
    def examples(self,role):
        name='pilot_shadow' if role=='shadow' else 'canary_optimize' if phase=='canary' else 'pilot_optimize'
        return tuple(CorrectnessExample(protocol_input('math',r['stable_example_id'],
            {'problem':f'Synthetic {role} arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'1')
            for i,r in enumerate(source_members[name]))
    monkeypatch.setattr(MATHLowCostBinding,'examples',examples)
    calls=[];seen={}
    def transport(req):
        calls.append(req)
        if req['model']!='qwen3-8b':return dict(text='```'+good+'```',input_tokens=2,output_tokens=3,finish_reason='stop')
        prompt=req['messages'][1]['content'].split('\n\n',1)[0]
        key=json.dumps(req,sort_keys=True);seen[key]=seen.get(key,0)+1
        # The same successful invalid request receives a fresh semantic attempt.
        if prompt==prompts[4]:text='missing'
        elif prompt==prompts[3] and seen[key]==1:text='missing'
        else:text='FINAL_ANSWER: '+('1' if prompt in {prompts[1],prompts[2],good} else '2')
        return dict(text=text,input_tokens=2,output_tokens=3,finish_reason='stop')
    class Client:
        def close(self):pass
    monkeypatch.setattr(execution,'create_transport',lambda _: (transport,Client()))
    manifest=gov.preexecution_manifest(ROOT,source_sha='a'*40,binding_path=c['binding_path'],experiment_id='synthetic_low_'+phase)
    prep=tmp_path/'prep';gov.prepare_canary(ROOT,manifest,destination=prep)
    authorization=json.loads((prep/'authorization.json').read_bytes());authorization['explicit_user_authorized']=True
    (prep/'authorization.json').write_text(json.dumps(authorization))
    run_root=tmp_path/'run';summary=asyncio.run(gov.execute_canary(ROOT,prep,run_root))
    assert summary['accounting']['charged_total']==5*len(calls)
    assert summary['accounting']['authorized_total']==40000000 and summary['accounting']['reserved_inflight']==0
    floor=json.loads((run_root/'initial_competence_floor.json').read_bytes())
    assert floor['binding']['count']==(12 if phase=='canary' else 60)
    assert floor['binding']['support_identity']==c['initial_competence_binding']['support_identity']
    assert (run_root/'resolved_output_cache/scope.json').exists()
    initial=json.loads((run_root/'initial_state_private.json').read_bytes())
    assert initial['diagnostics']['raw_profiles'][4][0]['terminal_invalid']
    assert initial['diagnostics']['raw_profiles'][4][0]['semantic_attempt_count']==4
    assert initial['diagnostics']['raw_profiles'][3][0]['recovered_invalid']
    assert summary['pattern_calls']==summary['validation_calls']==summary['test_calls']==0
    if phase=='canary':
        assert len(summary['result']['trace'])==1
        assert summary['result']['stop_reason']=='CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'
        assert not (run_root/'SEARCH_COMPLETE_RECEIPT.json').exists()
    else:
        assert summary['result']['stop_reason'] in {'SATURATION_REACHED','NO_FEASIBLE_OPPORTUNITY'}
        assert json.loads((run_root/'SEARCH_COMPLETE_RECEIPT.json').read_bytes())['search_closed_forever']
        from multi_dataset_diverse_rl.governance import math_paired_validation as paired
        validation_prep=tmp_path/'validation_prep'
        paired.prepare_validation(ROOT,prep,run_root,validation_prep)
        auth=json.loads((validation_prep/'authorization.json').read_bytes());auth['explicit_user_authorized']=True
        (validation_prep/'authorization.json').write_text(json.dumps(auth))
        monkeypatch.setattr(execution,'consumption_path',lambda *_:tmp_path/'validation_consumed.json')
        monkeypatch.setattr(paired,'TokenLedger',lambda directory,**kwargs:TokenLedger(isolated,**kwargs))
        monkeypatch.setattr(paired,'create_transport',lambda _: (transport,Client()))
        rows=source_members['pilot_validation']
        monkeypatch.setattr(paired,'validation_rows',lambda *_args,**_kwargs:tuple(dict(
            stable_example_id=r['stable_example_id'],input_sha256=r['input_sha256'],
            problem=f'Synthetic validation arithmetic {i}.',reference='1') for i,r in enumerate(rows)))
        validation_root=tmp_path/'validation_run';before=len(calls)
        evaluated=asyncio.run(paired.execute_validation(ROOT,validation_prep,validation_root))
        assert evaluated['policy']['count']==100 and evaluated['search_closed_forever']
        assert evaluated['accounting']['charged_total']==5*len(calls)
        evidence=[json.loads(s) for s in (validation_root/'paired_evidence.jsonl').read_text().splitlines()]
        assert len(evidence)==200
        changed=sum(a!=b for a,b in zip(prompts,json.loads((run_root/'final_team_private.json').read_bytes())['prompts']))
        events=[json.loads(s) for s in (validation_root/'ledger.jsonl').read_text().splitlines()]
        physical=sum(r['kind']=='ATTEMPT' for r in events)
        semantic=sum(r['kind']=='SEMANTIC_ATTEMPT' and r['semantic_attempt_no']>1 for r in events)
        assert physical==len(calls)-before==500+100*changed+semantic
        assert sum(r['kind']=='CACHE_HIT' for r in events)==100*(5-changed)
        assert evaluated['prediction_invalidity']['initial']['denominator']==500
        assert evaluated['prediction_invalidity']['final']['denominator']==500
        assert evaluated['invalid_recovery']['initial']['terminal_invalid']==100

@pytest.mark.parametrize('arm',['A1','A2','A3','A4'])
def test_low_cost_actual_gepa_four_arm_e2e(tmp_path,arm,monkeypatch):
    c=contract('canary');binding=MATHLowCostBinding(ROOT,c);adapter=binding.benchmark()
    prompts=tuple(m['prompt'] for m in json.loads((ROOT/c['initial_team_path']).read_bytes())['members'])
    def examples(role):
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',
            {'problem':f'Synthetic {role} arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'1')
            for i in range(12 if role=='optimize' else 40))
    monkeypatch.setattr(binding,'examples',examples)
    good='Check explicit constraints and verify relational consistency before deriving the mathematical result.'
    other='Enumerate cases carefully before deriving the mathematical result.'
    calls=[];contexts=[]
    def transport(req):
        calls.append(req)
        if req['model']=='qwen3-8b':
            from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
            body=json.loads(serialized_request(req))
            assert {k:body[k] for k in ('temperature','top_p','top_k','min_p','presence_penalty','frequency_penalty','enable_thinking','max_tokens')} == dict(temperature=.7,top_p=.8,top_k=20,min_p=0,presence_penalty=0,frequency_penalty=0,enable_thinking=False,max_tokens=3600)
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
            if prompt == prompts[4] and i == 0:
                return dict(text='FINAL_ANSWER: 1',input_tokens=2,output_tokens=2,finish_reason='length')
            return dict(text='FINAL_ANSWER: '+('1' if correct else '2'),input_tokens=2,output_tokens=2,finish_reason="stop")
        if len(req['messages'])==2:
            data=json.loads(req['messages'][1]['content'])
            return dict(text=json.dumps({'patterns':[dict(pattern_id='synthetic',failure_mechanism='Missed constraints',
                corrective_principle='Check explicit constraints',support_ids=data['residual_ids'],counterexample_ids=[],risk_ids=[],confidence=.9)]}),input_tokens=2,output_tokens=2,finish_reason="stop")
        return dict(text='```'+(good if len(contexts)==1 else other)+'```',input_tokens=2,output_tokens=2,finish_reason="stop")
    broker=RequestBroker(contract=c,transport=transport,arm=arm,seed=81,ledger_writer=lambda _:None)
    solver=BenchmarkSolver(adapter,broker);reflection=ReflectionProvider(broker)
    pattern=PatternProvider(broker,json.loads((ROOT/c['pattern_prompt_path']).read_bytes())['prompt']) if c['arms'][arm][0] else None
    def official(**kwargs):
        contexts.append(kwargs['adapter'].optimization_context)
        return import_frozen_gepa().optimize(**kwargs)
    run=binding.compose(arm=arm,seed=81,solver=solver,reflection=reflection,pattern_provider=pattern,run_root=tmp_path,optimize_fn=official)
    run.state.initialize();initial=run.state.snapshot();result=asyncio.run(run.run(max_opportunities=3))
    assert len(result.trace)==1 and result.stop_reason=='CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'
    assert result.transitions and run.state.initial_member_scores==initial.member_scores
    assert not initial.member_outputs[4][0].valid
    assert initial.diagnostics['raw_profiles'][4][0].semantic_attempt_count==4
    assert initial.diagnostics['raw_profiles'][4][0].terminal_invalid
    assert adapter.invalid_predictions_are_incorrect and not run.gate.operational_failure
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
