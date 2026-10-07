"""Phase wiring and observational equivalence under deterministic fake providers."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.benchmarks.gradient_pilot_contract import pilot_provider_bounds
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.governance.pilot_observation import attach_pilot_observer,memory_snapshot
from multi_dataset_diverse_rl.search.policies import GlobalStopPolicy,TargetPolicyV1,ResponsibilitySignal
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.search.schemas import Diagnosis,SearchContractError

ROOT=Path(__file__).resolve().parents[2]
BP='experiments/execution_bindings/math_v2_1_gradient_pattern_pilot_offline_profile_v4.json'
def contract():return json.loads((ROOT/BP).read_bytes())

def test_current_pilot_scope_and_scientific_identity_are_closed():
    c=contract();p=json.loads((ROOT/c['numeric_calibration_parent_binding_path']).read_bytes())
    assert not execution_binding(ROOT,c).blockers()
    changed={k for k in c.keys()|p.keys() if c.get(k)!=p.get(k)}
    assert changed=={'binding_path','execution_attempt_id','cache_namespace',
        'numeric_calibration_parent_binding_path','numeric_calibration_parent_binding_sha256',
        'numeric_calibration_amendment_path','numeric_calibration_amendment_sha256',
        'continuation_authorization_sha256',
        'pattern_abstraction_guard','pattern_policy'}
    assert c['provider_bounds']==p['provider_bounds']
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        execution_binding(ROOT,p)
    assert 'pilot_execution_authorization_path' not in c
    b=execution_binding(ROOT,c)
    assert len(b.examples('optimize'))==60 and len(b.examples('shadow'))==40
    with pytest.raises(SearchContractError,match='HELDOUT'):b.examples('validation')
    with pytest.raises(SearchContractError,match='HELDOUT'):b.examples('test')

@pytest.mark.parametrize('path,value',[
    ('execution_phase','canary'),('pilot_observation_policy','unknown'),
    ('cache_namespace','canary_reused'),('provider_bounds',{}),('execution_arm','A1')])
def test_pilot_amendment_tampering_fails_before_provider(path,value):
    c=contract();c[path]=value
    assert execution_binding(ROOT,c).blockers()

def test_carried_failure_bound_does_not_assume_round_robin_or_epoch_reset():
    from types import SimpleNamespace as NS
    bounds=pilot_provider_bounds(contract())
    assert bounds['max_opportunities']>100000
    assert bounds['bound_proof']['max_epoch_segments']==122
    history=HistoryState();history.failure_counts={0:100,1:0}
    diagnosis=Diagnosis(responsibility={0:ResponsibilitySignal(0,1,0,0,'direct_flip'),
        1:ResponsibilitySignal(1,60,0,0,'direct_flip')})
    state=NS();seen=set();count=0
    while len(seen)<2:
        m=TargetPolicyV1().select(state,diagnosis,(0,1),history).selected_member
        seen.add(m);history.observe_opportunity(m,committed=False);count+=1
    assert count>241
    threshold=241*(101)
    assert count<=5*(threshold+1)
    assert max(history.failure_counts.values())<=threshold+1
    assert bounds['successful_provider_calls']==sum(bounds[k] for k in (
        'solver_calls','reflection_calls','pattern_gradient_calls','pattern_cluster_calls'))
    assert bounds['reservation_peak_upper_bound']==40_000_000

def fake_run(root,monkeypatch,observe):
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver,ReflectionProvider
    from multi_dataset_diverse_rl.search.textual_gradients import PerExampleGradientProvider,GradientClusterProvider
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    c=contract();binding=execution_binding(ROOT,c);adapter=binding.benchmark()
    prompts=tuple(x['prompt'] for x in json.loads((ROOT/c['initial_team_path']).read_bytes())['members'])
    def examples(role):
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',
            {'problem':f'Synthetic {role} arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'1')
            for i in range(60 if role=='optimize' else 40))
    monkeypatch.setattr(binding,'examples',examples)
    requests=[];ledger=[];generations=[]
    def transport(req):
        requests.append(deepcopy(req));body=json.loads(serialized_request(req))
        assert body['enable_thinking'] is False
        if body['model']=='qwen3-8b':
            prompt,problem=req['messages'][1]['content'].split('\n\n',1)
            i=int(problem.split('arithmetic ')[1].split('.')[0])
            # All generic parents have failures, and proposals lose competence.
            correct=prompt in prompts and i>=7
            text='FINAL_ANSWER: '+('1' if correct else '2')
        elif len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            if 'example' in p:text=json.dumps({'gradient':'Check constraints before transforming intermediate expressions.'})
            else:text=json.dumps({'patterns':[{'generalized_gradient':'Check constraints before transforming intermediate expressions.',
                'support_ids':[g['example_id'] for g in p['gradients']]}],'unassigned_ids':[]})
        else:
            generations.append(1)
            text=json.dumps(dict(decision_procedure=f'Inspect constraints and check signs with {len(generations)} independent verifications.',change_summary='Add sign checks.'))
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',provider_response_accepted=True,
            provider_metadata_loss_audited=True,provider_reasoning_content_present=False,
            provider_reasoning_character_count=None,provider_usage_details={},provider_thinking_indicators=[])
    broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81,ledger_writer=ledger.append)
    provider=GradientClusterProvider(broker,gradient_provider=PerExampleGradientProvider(broker))
    run=binding.compose(arm='A4',seed=81,solver=BenchmarkSolver(adapter,broker),reflection=ReflectionProvider(broker),
        pattern_provider=provider,run_root=root)
    assert type(run.stop) is GlobalStopPolicy and run.stop.no_commit_patience==2
    if observe:attach_pilot_observer(run,root)
    run.state.initialize()
    result=asyncio.run(run.run(max_opportunities=50))
    assert result.stop_reason=='SATURATION_REACHED'
    assert len(result.trace)==10
    assert len(provider.input_audit)==10 and provider.gradient_provider.calls==70
    assert broker.usage['validation']==broker.usage['test']==0
    return result,requests,ledger,memory_snapshot(run.memory),run.state.snapshot()

def test_pilot_multiple_opportunities_and_observation_do_not_change_execution(tmp_path,monkeypatch):
    ordinary=tmp_path/'ordinary';ordinary.mkdir()
    observed=tmp_path/'observed';observed.mkdir()
    a=fake_run(ordinary,monkeypatch,False);b=fake_run(observed,monkeypatch,True)
    assert a==b
    rows=[json.loads(x) for x in (observed/'pilot_observation_private.jsonl').read_text().splitlines()]
    assert sum(r['stage']=='MEMORY_AFTER_OPPORTUNITY' for r in rows)==10
    assert sum(r['stage']=='GENERATION_COMPLETE' for r in rows)==60
    assert sum(r['stage']=='MEMORY_READ' for r in rows)==70
    assert rows[0]['memory_state']['audit']['stateful_write_count']==0
    assert all(r['data']['context_chars']<=1200 for r in rows if r['stage']=='MEMORY_READ')
    assert rows[-1]['memory_state']['audit']['failure_writes']>0
