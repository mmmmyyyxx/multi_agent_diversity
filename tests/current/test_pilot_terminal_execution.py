"""Real execution lifecycle with synthetic data and a fake transport."""
import asyncio,hashlib,json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
PROFILE='experiments/execution_bindings/math_v2_1_gradient_pattern_pilot_offline_profile_v5.json'

@pytest.mark.parametrize('changed',[False,True])
def test_current_pilot_runner_seals_terminal_receipts_and_validation_disposition(tmp_path,monkeypatch,changed):
    from multi_dataset_diverse_rl.governance import autonomous_math as execution
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    original=json.loads((ROOT/PROFILE).read_bytes());binding=execution_binding(ROOT,original)
    assert not binding.blockers()
    adapter=binding.benchmark()
    def examples(role):
        assert role in ('optimize','shadow')
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',
            {'problem':f'Synthetic {role} arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'1')
            for i in range(60 if role=='optimize' else 40))
    monkeypatch.setattr(binding,'examples',examples)
    c=deepcopy(original);c['token_ledger_directory']='runs/synthetic_ledger'
    c['initial_competence_binding']['support_identity']=hashlib.sha256(json.dumps(
        [e.item.input_id for e in examples('optimize')],separators=(',',':')).encode()).hexdigest()
    binding.contract=c
    # Freeze admission is covered separately; this fixture supplies synthetic data and storage.
    monkeypatch.setattr(binding,'blockers',lambda:())
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY
    with TokenLedger(tmp_path/c['token_ledger_directory'],task_sha256=c['task_authorization_sha256'],policy=POLICY) as ledger:
        ledger.amend_authorization_40m(authorization_sha256='0'*64,expected_charged_total=0)
    config=tmp_path/'binding.json';config.write_text(json.dumps(c),encoding='utf8')
    prep=tmp_path/'prep';prep.mkdir();run=tmp_path/'runs'/'synthetic_pilot'
    scope={'attempt_id':c['execution_attempt_id'],'source_sha':'0'*40}
    (prep/'authorization.json').write_text(json.dumps({'scope':scope,'explicit_user_authorized':True,'consumed':False}),encoding='utf8')
    payload={'manifest':{'execution_binding':{'path':'binding.json'},'source_sha':'0'*40},
        'scope':scope,'startup_identity_sha256':'0'*64}
    read_json=execution.read_json
    def read(path):
        return read_json(path if path.exists() else ROOT/path.relative_to(tmp_path))
    monkeypatch.setattr(execution,'read_json',read)
    monkeypatch.setattr(execution,'execution_binding',lambda root,contract:binding)
    requests=[];generic_calls={'optimize':0,'shadow':0};generations=[];closed=[]
    def transport(req):
        requests.append(deepcopy(req))
        if req['model']=='qwen3-8b':
            prompt,problem=req['messages'][1]['content'].split('\n\n',1)
            role='shadow' if 'Synthetic shadow ' in problem else 'optimize'
            i=int(problem.split('arithmetic ')[1].split('.')[0])
            if prompt=='Solve the problem.':
                m=generic_calls[role]//(40 if role=='shadow' else 60);generic_calls[role]+=1
                answer='1' if i>=7 or (changed and m==1) else str(m+2) if changed else '2'
            else:answer='1' if changed else '2'
            text='FINAL_ANSWER: '+answer
        elif len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            if 'example' in p:text=json.dumps({'gradient':'Check constraints before transforming intermediate expressions.'})
            else:text=json.dumps({'patterns':[{'generalized_gradient':'Check constraints before transforming intermediate expressions.',
                'support_ids':[g['example_id'] for g in p['gradients']]}],'unassigned_ids':[]})
        else:
            generations.append(1)
            text=json.dumps({'decision_procedure':f'Inspect constraints and check signs with {len(generations)} independent verifications.',
                'change_summary':'Add sign checks.'})
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',provider_response_accepted=True,
            provider_metadata_loss_audited=True,provider_reasoning_content_present=False,
            provider_reasoning_character_count=None,provider_usage_details={},provider_thinking_indicators=[])
    monkeypatch.setattr(execution,'create_transport',lambda contract:(transport,SimpleNamespace(close=lambda:closed.append(True))))
    summary=asyncio.run(execution.execute_search(tmp_path,prep,run,payload))
    assert summary['result']['stop_reason'] in ('SATURATION_REACHED','NO_FEASIBLE_OPPORTUNITY')
    assert summary['deployed_team_change']['team_changed'] is changed
    assert closed==[True]
    assert read_json(run/'lifecycle.json')['status']=='EXECUTION_COMPLETE'
    disposition=read_json(run/'VALIDATION_DISPOSITION.json')
    assert disposition['validation_status']==('DEFERRED_BY_USER_SCOPE' if changed else 'SKIPPED_NO_TEAM_CHANGE')
    assert disposition['validation_model_calls']==disposition['validation_provider_calls']==0
    assert summary['validation_calls']==summary['test_calls']==0
    assert read_json(run/'execution_summary.json')==summary
    receipt=read_json(run/'SEARCH_COMPLETE_RECEIPT.json')
    assert receipt['search_closed_forever'] and receipt['stop_reason']==summary['result']['stop_reason']
    assert receipt['execution_summary_sha256']==hashlib.sha256((run/'execution_summary.json').read_bytes()).hexdigest()
    assert receipt['raw_inventory_sha256']==hashlib.sha256((run/'raw_evidence_inventory.json').read_bytes()).hexdigest()
    assert read_json(run/'accounting_end.json')['reserved_inflight']==0
    assert summary['accounting']['charged_total']==4*len(requests)
