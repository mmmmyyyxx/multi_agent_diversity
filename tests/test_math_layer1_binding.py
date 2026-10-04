"""Production graph smokes for the new backend with isolated synthetic data."""
import asyncio
import json
from pathlib import Path
import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.benchmarks.math_layer1_binding import MATHLayer1Binding
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver,ReflectionProvider,PatternProvider
from multi_dataset_diverse_rl.governance.token_accounting import serialized_request

ROOT=Path(__file__).resolve().parents[1]


def test_new_binding_manifest_enters_current_governed_preflight():
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,bound_preflight
    m=preexecution_manifest(ROOT,source_sha='0'*40,binding_path='experiments/execution_bindings/math_v2_1_layer1_canary_v1.json',experiment_id='synthetic_layer1_preflight',frozen=False)
    result=bound_preflight(ROOT,m)
    assert result['blockers']==['PREEXECUTION_NOT_FROZEN'],result
    assert result['provider_attempts']==0


@pytest.mark.parametrize('arm',['A1','A2','A3','A4'])
def test_frozen_layer1_four_arm_production_graph(tmp_path,arm,monkeypatch):
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_layer1_canary_v1.json').read_bytes())
    binding=MATHLayer1Binding(ROOT,c);assert not binding.blockers()
    adapter=binding.benchmark()
    prompts=tuple(m['prompt'] for m in json.loads((ROOT/c['initial_team_path']).read_bytes())['members'])
    def examples(role):
        assert role in ('optimize','shadow')
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',{'problem':f'Synthetic {role} arithmetic {i}.'},
            adapter.output_contract,protocol=adapter.protocol),'1') for i in range(12 if role=='optimize' else 40))
    monkeypatch.setattr(binding,'examples',examples)
    good='Check explicit constraints and verify relational consistency before deriving the mathematical result.'
    calls=[];generations=[];events=[]
    def transport(req):
        body=json.loads(serialized_request(req));calls.append(body)
        assert body['enable_thinking'] is False and body['top_k']==20 and body['top_p']==0.8
        if body['model']=='qwen3-8b':
            assert body['temperature']==0.2 and body['presence_penalty']==0 and body['max_tokens']==3600
            assert req['messages'][0]['content']==adapter.output_contract
            prompt,problem=req['messages'][1]['content'].split('\n\n',1)
            from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATH_SOLVER_INTERFACE_V4_USER_SUFFIX
            problem=problem.removesuffix(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX)
            assert '"memory"' not in problem and '"pattern"' not in problem
            i=int(problem.rsplit(' ',1)[1].rstrip('.'))
            if problem.startswith('Synthetic shadow'):correct=prompt in {prompts[1],prompts[2],good}
            elif prompt in prompts:
                m=prompts.index(prompt);correct=m in {1,2} or m==0 and i>=3
            else:correct=i!=2 if prompt==good else i>=2
            text='FINAL_ANSWER: '+('1' if correct else '2')
        else:
            assert body['temperature']==0.7 and body['presence_penalty']==1.5
            assert body['max_completion_tokens']==1800 and 'max_tokens' not in body
            if len(req['messages'])==2:
                data=json.loads(req['messages'][1]['content'])
                text=json.dumps({'patterns':[dict(pattern_id='synthetic',failure_mechanism='Missed constraints',corrective_principle='Check explicit constraints',support_ids=data['residual_ids'],counterexample_ids=[],risk_ids=[],confidence=.9)]})
            else:
                context=json.loads(req['messages'][0]['content'].rsplit('\n',1)[1]);generations.append(context)
                text=json.dumps({'decision_procedure':good if len(generations)==1 else
                    'Enumerate cases carefully and verify signs with '+str(len(generations))+' consistency checks.'})
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',provider_response_accepted=True,
            provider_metadata_loss_audited=True,provider_reasoning_content_present=False,
            provider_reasoning_character_count=None,provider_usage_details={},provider_thinking_indicators=[])
    broker=RequestBroker(contract=c,transport=transport,arm=arm,seed=81,ledger_writer=events.append)
    solver=BenchmarkSolver(adapter,broker);reflection=ReflectionProvider(broker)
    pattern=PatternProvider(broker,json.loads((ROOT/c['pattern_prompt_path']).read_bytes())['prompt']) if c['arms'][arm][0] else None
    run=binding.compose(arm=arm,seed=81,solver=solver,reflection=reflection,pattern_provider=pattern,run_root=tmp_path)
    run.state.initialize();initial=run.state.snapshot();result=asyncio.run(run.run(max_opportunities=3))
    assert len(result.trace)==1 and result.stop_reason=='CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'
    assert result.trace[0].candidate_ids and run.evaluation.provider.probed
    assert run.state.initial_member_scores==initial.member_scores
    assert generations and len(generations)<=6
    assert json.loads(generations[0]['search_context'].split('\n',1)[0])['responsibility']
    assert all(r['visible_reasoning_required'] is False for r in generations[0]['root_observations'])
    assert broker.usage['validation']==broker.usage['test']==0
    assert not any(e['kind']=='CACHE_HIT' and e['role']!='solver' for e in events)
    assert bool(broker.usage['pattern'])==c['arms'][arm][0]
    assert run.method.search_engine==versions.LAYER1_RESPONSIBILITY_SEARCH_VERSION
    assert all(t.allocation_audit['realized_team_gain']>0 for t in result.trace if t.committed_candidate_id)
