"""Synthetic largest-focus proofs, default compatibility and cache boundaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
from multi_dataset_diverse_rl.research import gated_prompt_largest as f
from multi_dataset_diverse_rl.research.gated_prompt import run_cell,generate_changed_candidate
from multi_dataset_diverse_rl.search.schemas import SearchContractError
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('largest_fixture',ROOT/'tests/current/test_gated_prompt_study.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def feedback():
    return dict(errors=[dict(example_id=str(i),problem=f'Synthetic fixture value: {i}',reference=str(i)) for i in range(3)],structured_feedback=dict(
        gradients=[dict(example_id=str(i),gradient=dict(reusable_correction=f'Synthetic correction {i}')) for i in range(3)],
        diagnostics=[dict(example_id=str(i),diagnosis=f'Synthetic diagnosis {i}') for i in range(3)],
        patterns=[dict(pattern_id='large',support_ids=['0','1'],generalized_gradient='Synthetic largest correction'),
            dict(pattern_id='small',support_ids=['2'],generalized_gradient='Synthetic excluded correction')],
        unassigned_ids=[],partition_audit=[dict(discarded_patterns=['Synthetic excluded draft'])],cluster_structural_errors=[]))


def test_unique_support_maximum_filters_every_optimizer_evidence_channel():
    old=feedback();new,a=f.project_feedback(old,88,1)
    assert old==feedback() and a['selected_pattern_id']=='large' and a['selected_support_ids']==['0','1']
    assert [e['example_id'] for e in new['errors']]==['0','1']
    sf=new['structured_feedback'];assert [p['pattern_id'] for p in sf['patterns']]==['large']
    assert [g['example_id'] for g in sf['gradients']]==[d['example_id'] for d in sf['diagnostics']]==['0','1']
    assert sf['partition_audit']==sf['unassigned_ids']==sf['cluster_structural_errors']==[]
    assert 'Synthetic excluded' not in json.dumps(new)


def test_tie_selection_is_seed_round_replayable_and_independent_of_pattern_order():
    old=feedback();old['structured_feedback']['patterns']=[dict(pattern_id=str(i),support_ids=[str(i)],generalized_gradient=f'Synthetic {i}') for i in range(3)]
    choices=set()
    for n in range(1,21):
        one,a=f.project_feedback(old,88,n);reverse=deepcopy(old);reverse['structured_feedback']['patterns'].reverse()
        two,b=f.project_feedback(reverse,88,n)
        assert one==two and a==b and len(one['errors'])==1 and len(a['tie_ids'])==3;choices.add(a['selected_pattern_id'])
    assert len(choices)>1


def test_duplicate_support_ids_do_not_inflate_largest_size():
    old=feedback();old['structured_feedback']['patterns'][1]['support_ids']=['2']*10
    _,a=f.project_feedback(old,88,1);assert a['selected_pattern_id']=='large' and a['pattern_sizes']['small']==1


@pytest.mark.parametrize('mutation',['unsupported_id','missing_gradient','duplicate_pattern','unknown_seed','unknown_channel'])
def test_illegal_selection_evidence_fails_closed(mutation):
    old=feedback();seed=88
    if mutation=='unsupported_id':old['structured_feedback']['patterns'][0]['support_ids'].append('foreign')
    elif mutation=='missing_gradient':old['structured_feedback']['gradients'].pop(0)
    elif mutation=='duplicate_pattern':old['structured_feedback']['patterns'].append(deepcopy(old['structured_feedback']['patterns'][0]))
    elif mutation=='unknown_channel':old['structured_feedback']['whole_batch_notes']='Synthetic excluded evidence'
    else:seed=90
    with pytest.raises(SearchContractError):f.project_feedback(old,seed,1)


def test_no_pattern_is_exact_B_input_and_never_carries_other_gradients():
    old=feedback();old['structured_feedback']['patterns']=[]
    new,a=f.project_feedback(old,89,4)
    assert new==dict(errors=old['errors'],structured_feedback=None) and a['fallback_to_B'] and a['selected_pattern_id'] is None


def test_full_support_single_pattern_preserves_original_wire_packet():
    old=feedback();sf=old['structured_feedback'];sf['patterns']=[dict(pattern_id='all',support_ids=['0','1','2'],generalized_gradient='Synthetic complete correction')];sf['partition_audit']=[dict(discarded_patterns=[],status='synthetic')]
    new,a=f.project_feedback(old,89,1)
    assert json.dumps(new,sort_keys=True,ensure_ascii=True)==json.dumps(old,sort_keys=True,ensure_ascii=True)


def test_exact_library_miss_never_coerces_an_old_response(tmp_path):
    entries=[dict(key='same',response=dict(text='synthetic B'),arm='batch_errors',source_scope_identity='b',reservation_id='2',source_receipt_sha256='2'),dict(key='same',response=dict(text='synthetic C'),arm='gradient_pattern',source_scope_identity='c',reservation_id='3',source_receipt_sha256='3')]
    cache=f.ExactResponseLibrary(entries,tmp_path/'reuse.jsonl')
    assert cache.take('different','same_guard') is None and not (tmp_path/'reuse.jsonl').exists()
    assert cache.take('same','guard')['text']=='synthetic C' and cache.hits==1
    cache.require_consumed()


def test_projection_is_the_actual_optimizer_packet_and_generation_retries_keep_focus(tmp_path):
    broker=fixture.broker_for(tmp_path);old=feedback();new,a=f.project_feedback(old,88,1)
    # Supply complete synthetic raw-error fields only needed by the generator.
    for e in new['errors']:e['reference']='0'
    try:
        candidate,draws=generate_changed_candidate(fixture.SEED,new,broker,1,lambda *_:None)
        request=next(req for role,req in broker.transport.calls if role=='reflection');packet=json.loads(request['messages'][-1]['content'])
        assert candidate is not None and draws==1 and packet['errors']==new['errors'] and packet['structured_feedback']==new['structured_feedback']
    finally:broker.ledger.close()


def test_method_requires_non_null_closed_contract():
    assert not f.enabled({}) and f.enabled(dict(method_amendment=f.contract()))
    for value in [None,dict(f.contract(),no_pattern='skip')]:
        with pytest.raises(SearchContractError):f.enabled(dict(method_amendment=value))


def test_run_cell_uses_selected_support_and_no_pattern_fallback_without_losing_opportunities(tmp_path,monkeypatch):
    import multi_dataset_diverse_rl.research.gated_prompt as loop
    broker=fixture.broker_for(tmp_path);broker.scope['method_amendment']=f.contract();rows=fixture.fixture.toy_rows(3);artifacts={}
    # Use the real loop with synthetic scoring and real generator/provider ports.
    broker.evaluate=lambda *args:{r['stable_example_id']:fixture.fixture.observation() for r in rows}
    def structure(*args):
        sf=feedback()['structured_feedback']
        for g in sf['gradients']:g['example_id']='toy_'+g['example_id']
        for d in sf['diagnostics']:d['example_id']='toy_'+d['example_id']
        for p in sf['patterns']:p['support_ids']=['toy_'+i for i in p['support_ids']]
        if broker.round_index%2==0:sf['patterns']=[]
        return sf
    monkeypatch.setattr(loop,'structured_feedback',structure)
    try:
        result=run_cell(fixture.fixture.cell('gradient_pattern',88),rows,rows,broker,artifacts.__setitem__)
        assert result['rounds_completed']==20 and len(result['metrics'])==20
        from multi_dataset_diverse_rl.research.gated_prompt_largest_governance import verify_trajectory
        from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
        for name,data in artifacts.items():atomic_write_json(tmp_path/'trajectory'/(name+'.json'),data)
        verify_trajectory(tmp_path/'trajectory',result,[r['stable_example_id'] for r in rows])
        altered=deepcopy(result);altered['metrics'][0]['accepted']=True
        with pytest.raises(SearchContractError):verify_trajectory(tmp_path/'trajectory',altered,[r['stable_example_id'] for r in rows])
        for n in range(1,21):
            focus=artifacts[f'round_{n}_focus'];packet=artifacts[f'round_{n}_opportunity']['feedback']
            assert focus['fallback_to_B']==(n%2==0) and len(packet['errors'])==(3 if n%2==0 else 2)
            assert result['metrics'][n-1]['optimizer_example_count']==len(packet['errors'])
    finally:broker.ledger.close()


@pytest.mark.parametrize('blocked',['authorization','train_path'])
def test_final_test_refuses_access_before_authorization_and_exact_training_path(tmp_path,monkeypatch,blocked):
    from multi_dataset_diverse_rl.research import gated_prompt_largest_governance as lg
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    prep=tmp_path/'prep';train=tmp_path/'train_run'
    atomic_write_json(prep/'authorization.json',dict(test_access=blocked!='authorization'))
    monkeypatch.setattr(lg,'validate_prep',lambda *a,**k:dict(execution=dict(run_root='expected_train')))
    monkeypatch.setattr(lg,'verify_training',lambda *a:pytest.fail('Blocked attempt must not read training/Test evidence'))
    with pytest.raises(SearchContractError,match='AUTHORIZATION_REQUIRED' if blocked=='authorization' else 'TRAIN_PATH_MISMATCH'):
        lg.execute_final_test(tmp_path,prep,train)
    assert not (tmp_path/'test_run').exists()


def test_final_phase_references_controls_and_reuses_only_exact_seed_solver_identity(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.research import gated_prompt_largest_governance as lg,gated_prompt_execution as ex
    from multi_dataset_diverse_rl.research.gated_prompt_test import FinalTestBroker
    from multi_dataset_diverse_rl.research.gated_prompt_network import operation_contract
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_V27_FORMAL_UNCAPPED
    from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
    from multi_dataset_diverse_rl.benchmarks.data_freeze import digest,file_hash
    import multi_dataset_diverse_rl.governance.autonomous_math as provider
    import multi_dataset_diverse_rl.research.single_prompt_runtime as runtime
    monkeypatch.setattr(runtime.MATHBenchmarkAdapter,'score_member_output',lambda *_:0.0)
    monkeypatch.setattr(runtime.MATHBenchmarkAdapter,'equivalent',lambda self,a,b:a==b)
    toy=fixture.fixture;rows=toy.toy_rows(300)
    oldcells=[dict(toy.cell(a,s),final_prompt=fixture.SEED.to_dict()) for a in ex.policy()['arms'] for s in [88,89]]
    parent_train=tmp_path/'runs/parent/train_run';parent_test=parent_train.parent/'test_run'
    atomic_write_json(parent_train/'results.json',oldcells)
    namespace='c'*64;transport=toy.ToyTransport();ledger=TokenLedger(tmp_path/'old_ledger',task_sha256=namespace,policy=POLICY_V27_FORMAL_UNCAPPED)
    oldbroker=FinalTestBroker(phase='test',scope=dict(scope_identity=namespace,call_ceiling=2400,token_ceiling=None,execution_amendment=ex.contract(),network_recovery=operation_contract()),transport=transport,ledger=ledger,receipts=ProviderResponseReceipts(tmp_path/'old_receipts',attempt_id='synthetic',startup_identity_sha256=namespace),dispatch_guard=lambda _:None)
    try:
        for cell in oldcells:
            oldbroker.cell_id=oldbroker.attempt_id=cell['cell_id'];oldbroker.seed=cell['seed']
            pred=oldbroker.evaluate(fixture.SEED,rows,'test',cell['seed'],20)
            atomic_write_json(parent_test/(cell['cell_id']+'.json'),dict(predictions=pred,count=300))
        oldusage=oldbroker.usage_view()
    finally:ledger.close()
    assert oldusage['provider_calls']==600
    prep=tmp_path/'runs/new/prep';train=prep.parent/'train_run'
    atomic_write_json(prep/'prep.json',dict(synthetic=True));atomic_write_json(prep/'authorization.json',dict(test_access=True))
    atomic_write_json(train/'terminal.json',dict(synthetic_seal=True));(tmp_path/'runs/gated_prompt_consumed').mkdir()
    corrected=[c for c in oldcells if c['arm']=='gradient_pattern'];newcells=[dict(c,attempt_id=c['cell_id']+'_train_new') for c in corrected]
    scope=dict(scope_identity='d'*64,source_sha='e'*40,source_hashes={},execution=dict(run_root=train.relative_to(tmp_path).as_posix()),execution_amendment=ex.contract(),cells=newcells,
        test_plan=dict(physical_call_ceiling=3000,sample_ids=[r['stable_example_id'] for r in rows]),continuation=dict(parent_run_root=parent_train.relative_to(tmp_path).as_posix(),parent_test_scope_identity=namespace,inherited_test_usage=oldusage,operation_policy=operation_contract()))
    monkeypatch.setattr(lg,'validate_prep',lambda *a,**k:scope)
    monkeypatch.setattr(lg,'verify_training',lambda *a:(corrected,file_hash(train/'terminal.json')))
    class Reader:
        manifest_sha256='f'*64
        def heldout_rows(self,capability):
            assert (train/'terminal.json').exists() and read_json(prep/'authorization.json')['test_access']
            return rows
    monkeypatch.setattr(lg.gov,'validate_manifest',lambda *a:Reader());monkeypatch.setattr(lg,'read_json',lambda path: {} if Path(path).as_posix().endswith(lg.gov.MANIFEST) else read_json(path))
    class Forbidden:
        def for_role(self,*args):pytest.fail('Exact final Solver identities must reuse cached observations')
    class Clients:
        def close(self):pass
    monkeypatch.setattr(provider,'create_transport',lambda *a:(Forbidden(),Clients()))
    result=lg.execute_final_test(tmp_path,prep,train);test=train.parent/'test_run'
    assert result['new_training_cells']==2 and result['referenced_controls']==4
    summary=read_json(test/'execution_summary.json')
    assert summary['usage']['provider_calls']==600 and summary['token_ledger']['charged_total']==0
    assert summary['usage']['cache_hits']==oldusage['cache_hits']+600
    assert len(read_json(test/'analysis.json')['all_15_pairs'])==15
    assert read_json(test/'lifecycle.json')['status']=='EXECUTION_COMPLETE'
    assert read_json(test/'scope.json')['policy']['physical_call_ceiling']==3000
    assert not any('seed90' in p.name for p in test.iterdir())
