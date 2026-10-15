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


def test_fresh_population_prepare_rejects_dirty_source_before_writes(tmp_path,monkeypatch):
    monkeypatch.setattr(pg.gov,'require_selected_runtime',lambda *_:None)
    monkeypatch.setattr(pg.gov,'_git',lambda *_:'M tracked_source')
    with pytest.raises(SearchContractError,match='CLEAN_COMMITTED'):
        pg.prepare(tmp_path,tmp_path/'runs/new/prep')
    assert not (tmp_path/'runs').exists()


def test_unapproved_AB_cannot_create_provider_or_run(tmp_path,monkeypatch):
    called=[]
    monkeypatch.setattr(ab,'validate_prep',lambda *a,**k:(_ for _ in ()).throw(SearchContractError('AB_TEST_UNAUTHORIZED')))
    monkeypatch.setattr(ab.gov,'_transport',lambda:called.append(True))
    with pytest.raises(SearchContractError,match='UNAUTHORIZED'):
        ab.execute(tmp_path,tmp_path/'runs/a/prep',tmp_path/'runs/a/test_run')
    assert not called and not (tmp_path/'runs').exists()


def test_AB_test_phase_has_no_training_or_teacher_or_C(tmp_path,monkeypatch):
    plan=ab.test_only_plan();prep=tmp_path/'runs/a/prep';run=prep.parent/'test_run'
    scope=dict(scope_identity=digest('fake AB'),source_sha='0'*40,source_hashes={},phase='test',plan=plan,
        parent_evidence_sha256=digest('completed cells'),test_plan=dict(sample_ids=['test_0','test_1']),call_ceiling=7200,
        execution=dict(run_root='runs/a/test_run'))
    parents=[dict(cell=deepcopy(c),rounds_completed=20,metrics=[dict(decision='REJECTED_NO_STRICT_VALIDATION_GAIN')]*20,
        final_prompt=SEED.to_dict(),final_validation_correct=0,validation_count=100,final_optimize_correct=0,optimize_count=150,
        usage=dict(input_tokens=0,output_tokens=0,provider_calls=0)) for c in plan['cells']]
    for n in ('prep','authorization','handoff','execution_plan'):atomic_write_json(prep/(n+'.json'),{})
    monkeypatch.setattr(ab,'validate_prep',lambda *a,**k:scope)
    monkeypatch.setattr(ab,'parent_evidence',lambda *_:(None,parents,{}))
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
