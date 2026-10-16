"""Test-only authorization and phase closure, with no private files or network."""
from copy import deepcopy
import pytest

from multi_dataset_diverse_rl.research.ablations import completed_ab_test as ab
from multi_dataset_diverse_rl.research.ablations import population_governance as pg
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json


def test_test_only_identity_and_semantic_bounds():
    plan=ab.test_only_plan();policy=ab.operation_policy()
    assert len(plan['cells'])==6 and {c['arm_id'] for c in plan['cells']}=={'A','B'}
    assert {c['seed'] for c in plan['cells']}=={42} and {c['repeat_id'] for c in plan['cells']}=={1,2,3}
    assert policy['new_call_ceiling']==6*300*4 and policy['local_retry_new_model_calls']==0
    assert policy['train_access'] is policy['validation_access'] is False
    assert policy['scoring_worker']=='math_certified_worker' and policy['worker_deadline_seconds']==30


def test_V14_preflight_and_immutable_historical_V13():
    from pathlib import Path
    from scripts.run_experiment import preflight
    from multi_dataset_diverse_rl.governance.frozen_documents import validate_published_single_prompt
    root=Path(__file__).resolve().parents[2]
    assert preflight(ab.manifest_template())['real_api_calls']==0
    old_path='experiments/manifests/single_prompt_completed_ab_test_v13.json';old=read_json(root/old_path)
    assert validate_published_single_prompt(root,old_path,old)
    old['operation_policy']['new_call_ceiling']+=1
    with pytest.raises(ValueError,match='HISTORICAL_MANIFEST_CHANGED'):
        validate_published_single_prompt(root,old_path,old)


def test_fresh_population_prepare_rejects_dirty_source_before_writes(tmp_path,monkeypatch):
    monkeypatch.setattr(pg.gov,'require_selected_runtime',lambda *_:None)
    monkeypatch.setattr(pg.gov,'_git',lambda *_:'M tracked_source')
    with pytest.raises(SearchContractError,match='CLEAN_COMMITTED'):
        pg.prepare(tmp_path,tmp_path/'runs/new/prep')
    assert not (tmp_path/'runs').exists()


def test_unapproved_AB_cannot_create_provider_or_run(tmp_path,monkeypatch):
    called=[]
    atomic_write_json(tmp_path/'runs/a/prep/authorization.json',dict(approved=False,test_access=False))
    monkeypatch.setattr(ab,'validate_prep',lambda *a,**k:(_ for _ in ()).throw(SearchContractError('AB_TEST_UNAUTHORIZED')))
    monkeypatch.setattr(ab.gov,'_transport',lambda:called.append(True))
    with pytest.raises(SearchContractError,match='UNAUTHORIZED'):
        ab.execute(tmp_path,tmp_path/'runs/a/prep',tmp_path/'runs/a/test_run')
    assert not called and not (tmp_path/'runs/a/test_run').exists()


def test_AB_test_phase_has_no_training_or_teacher_or_C(tmp_path,monkeypatch):
    plan=ab.test_only_plan();prep=tmp_path/'runs/a/prep';run=prep.parent/'test_run'
    scope=dict(scope_identity=digest('fake AB'),source_sha='0'*40,source_hashes={},phase='test',plan=plan,
        parent_evidence_sha256=digest('completed cells'),test_plan=dict(sample_ids=['test_0','test_1']),call_ceiling=7200,
        execution=dict(run_root='runs/a/test_run'))
    parents=[dict(cell=deepcopy(c),rounds_completed=20,metrics=[dict(decision='REJECTED_NO_STRICT_VALIDATION_GAIN')]*20,
        final_prompt=SEED.to_dict(),final_validation_correct=0,validation_count=100,final_optimize_correct=0,optimize_count=150,
        usage=dict(input_tokens=0,output_tokens=0,provider_calls=0)) for c in plan['cells']]
    for n in ('prep','authorization','handoff','execution_plan'):atomic_write_json(prep/(n+'.json'),{})
    atomic_write_json(prep/'authorization.json',dict(approved=True,test_access=True))
    audit_calls=[];boundary_calls=[];snapshot=(None,parents,{})
    def full_audit(*_):
        audit_calls.append(True);return snapshot
    def boundary(*a,**k):
        assert k['authorized'] is True and k['parent_snapshot'] is snapshot
        boundary_calls.append(True);return scope
    monkeypatch.setattr(ab,'validate_prep',boundary)
    monkeypatch.setattr(ab,'parent_evidence',full_audit)
    class Clients:
        def close(self):pass
    monkeypatch.setattr(ab.gov,'_transport',lambda:(object(),Clients()))
    class Reader:
        manifest_sha256=digest('fixed test')
        def heldout_rows(self,access):
            access.require_authorization();return [dict(stable_example_id=i) for i in scope['test_plan']['sample_ids']]
    monkeypatch.setattr(pg,'inputs',lambda *_:(None,Reader()))
    calls=[]
    class Broker:
        def __init__(self,**kwargs):
            assert kwargs['phase']=='test' and kwargs['paid_prefix']==kwargs['terminal_observations']=={}
            self.local_retry_count=0
        def retry_local(self,fn,context):return fn()
        def bind_cell(self,cell,plan):self.cell=cell;assert cell['arm_id'] in ('A','B')
        def usage_view(self):
            roles=('solver','reflection','pattern_gradient','pattern_cluster')
            return dict(provider_calls=0,input_tokens=0,output_tokens=0,network_failed_calls=0,
                **{r:0 for r in roles},**{r+'_'+k:0 for r in roles for k in ('input_tokens','output_tokens')})
        def evaluate(self,prompt,rows,split,seed,ordinal):
            calls.append((self.cell['cell_id'],split,seed,ordinal))
            return {r['stable_example_id']:dict(correct=False,valid=True) for r in rows}
    monkeypatch.setattr(ab,'TestResumeBroker',Broker)
    result=ab.execute(tmp_path,prep,run)
    assert result==dict(status='EXECUTION_COMPLETE',cells_completed=6)
    assert len(calls)==6 and all(v[1:]==('test',42,0) for v in calls)
    assert read_json(run/'terminal.json')['phase']=='test'
    assert read_json(run/'access.json')['validation_access'] is False
    assert read_json(run/'summary.json')['matched_causal_ablation'] is False
    assert len(audit_calls)==1 and len(boundary_calls)==9


def checkpoint_fixture(folder):
    cell=dict(cell_id='A_seed42_repeat01',arm_id='A',seed=42,repeat_id=1)
    plan_identity=digest('test plan');checkpoint_scope=dict(cell=cell,plan_identity=plan_identity)
    prompt=SEED.to_dict();opt={'o':dict(valid=True,correct=False)};val={'v':dict(valid=True,correct=False)}
    scope=dict(plan=dict(plan_identity=plan_identity),initial_prompt_sha256=SEED.prompt_hash,
        optimize_sample_ids=['o'],gate_membership=dict(sample_ids=['v']))
    history=[];metrics=[]
    for n in range(21):
        cp=dict(prompt=deepcopy(prompt),optimize=deepcopy(opt),validation=deepcopy(val),
            scope_identity=digest(checkpoint_scope))
        if n==0:cp['scope']=checkpoint_scope
        else:
            cp['parent_prompt']=deepcopy(prompt)
            m=dict(round=n,validation_before=sum(v['correct'] for v in val.values()),accepted=False,
                generation_draws=4,decision='ABANDONED_GENERATION_RETRIES_EXHAUSTED')
            cp['candidate_prompt']=None
            if n in (1,2):
                candidate=deepcopy(prompt);candidate['strategy']+=' Generic strategy '+str(n)
                cp.update(candidate_prompt=candidate,candidate_optimize=deepcopy(opt),
                    candidate_validation={'v':dict(valid=True,correct=True)})
                m.update(accepted=n==1,validation_candidate=1,
                    decision='ACCEPTED_STRICT_VALIDATION_GAIN' if n==1 else 'REJECTED_NO_STRICT_VALIDATION_GAIN')
                if n==1:
                    cp['prompt']=candidate;cp['validation']=deepcopy(cp['candidate_validation'])
            cp['metric']=m;metrics.append(m)
        cp['integrity_sha256']=digest(cp)
        atomic_write_json(folder/f'checkpoint_{n}.json',cp)
        prompt=deepcopy(cp['prompt']);opt=deepcopy(cp['optimize']);val=deepcopy(cp['validation']);history.append(prompt)
    result=dict(scope_identity=digest(checkpoint_scope),prompt_history=history,metrics=metrics,final_prompt=prompt,
        final_optimize_correct=0,final_validation_correct=1)
    return cell,result,scope


def test_streaming_checkpoints_keep_at_most_two_full_objects(tmp_path,monkeypatch):
    import weakref
    cell,result,scope=checkpoint_fixture(tmp_path)
    refs=[];real_read=ab.read_json;max_live=0
    class Tracked(dict):pass
    def read(path):
        nonlocal max_live
        obj=Tracked(real_read(path));refs.append(weakref.ref(obj))
        max_live=max(max_live,sum(r() is not None for r in refs))
        return obj
    monkeypatch.setattr(ab,'read_json',read)
    ab.audit_cell_checkpoints(tmp_path,cell,result,scope)
    assert len(refs)==21 and max_live==2 and all(r() is None for r in refs)


@pytest.mark.parametrize('mutation',['hash','two_blocks','gate','retention','membership','boolean','history','final_count'])
def test_streaming_audit_preserves_historical_rejections(tmp_path,mutation):
    cell,result,scope=checkpoint_fixture(tmp_path);path=tmp_path/'checkpoint_2.json';cp=read_json(path)
    if mutation=='hash':cp['integrity_sha256']='0'*64
    elif mutation=='two_blocks':cp['candidate_prompt']['role']+=' Other role'
    elif mutation=='gate':cp['metric']['accepted']=True
    elif mutation=='retention':cp['prompt']=cp['candidate_prompt']
    elif mutation=='membership':cp['validation']['other']=cp['validation'].pop('v')
    elif mutation=='boolean':cp['candidate_validation']['v']['correct']=1
    elif mutation=='history':result['prompt_history'][2]=SEED.to_dict()
    elif mutation=='final_count':result['final_validation_correct']=0
    if mutation!='hash':cp['integrity_sha256']=digest({k:v for k,v in cp.items() if k!='integrity_sha256'})
    atomic_write_json(path,cp)
    with pytest.raises(SearchContractError):ab.audit_cell_checkpoints(tmp_path,cell,result,scope)


def test_parent_boundary_rehash_catches_same_size_and_mtime_mutation(tmp_path):
    import os
    from multi_dataset_diverse_rl.benchmarks.data_freeze import file_hash
    path=tmp_path/'parent.json';path.write_bytes(b'original')
    inventory={'parent.json':file_hash(path)};ab.verify_parent_inventory(tmp_path,inventory)
    stat=path.stat();path.write_bytes(b'mutation');os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(SearchContractError,match='PARENT_FILES_CHANGED'):ab.verify_parent_inventory(tmp_path,inventory)
    path.unlink()
    with pytest.raises(SearchContractError,match='PARENT_FILES_CHANGED'):ab.verify_parent_inventory(tmp_path,inventory)
