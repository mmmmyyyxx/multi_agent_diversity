"""Fresh C-only scope cannot inherit controls, cache, paid state or authorization."""
from copy import deepcopy
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.research.ablations import aligned,governance as gov
from multi_dataset_diverse_rl.research.ablations.config import build_plan
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest,file_hash
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from test_ablation_execution import phase_smoke

ROOT=Path(__file__).resolve().parents[2]


def test_exact_reviewed_c_subset_preserves_runtime_and_fresh_budgets():
    c=read_json(ROOT/aligned.C_MANIFEST)
    base=read_json(ROOT/aligned.MANIFEST)
    aligned.validate_manifest(ROOT,c,c_only=True)
    assert c['config']['runtime']==base['config']['runtime']
    assert c['plan']==build_plan(c['config'])
    assert [(v['arm_id'],v['seed'],v['repeat_id']) for v in c['plan']['cells']]==[
        ('C',42,1),('C',42,2),('C',42,3)]
    assert c['plan']['training_semantic_call_ceiling']==90420
    assert c['plan']['final_test_semantic_call_ceiling']==3600
    assert c['config']['rounds']==20
    assert c['isolation']['import_old_caches'] is False
    assert c['isolation']['import_old_teacher_responses'] is False
    from scripts.run_experiment import preflight
    assert preflight(c)['status']=='ABLATION_C_ONLY_PREFLIGHT_PASS_NOT_AUTHORIZED'
    with pytest.raises(SearchContractError):aligned.validate_manifest(ROOT,c)


@pytest.mark.parametrize('alter',['arm','rounds','repeat','budget','isolation'])
def test_c_only_frozen_manifest_cannot_be_widened(alter):
    value=deepcopy(read_json(ROOT/aligned.C_MANIFEST))
    if alter=='arm':value['config']['arms'][0]['arm_id']='A'
    elif alter=='rounds':value['config']['rounds']=21
    elif alter=='repeat':value['config']['repeats_per_arm']=5
    elif alter=='budget':value['plan']['training_semantic_call_ceiling']+=1
    else:value['isolation']['import_old_caches']=True
    with pytest.raises(SearchContractError,match='MANIFEST_MISMATCH'):
        aligned.validate_manifest(ROOT,value,c_only=True)


def test_c_only_prep_freeze_scope_phase_grant_and_handoff(tmp_path,monkeypatch):
    prep=ROOT/'runs/ablations/verification_c_only_v7'/('fake_'+tmp_path.name)/'prep'
    original=gov._git
    def fake_git(root,*args):
        if args==('status','--porcelain','--untracked-files=no'):return ''
        if args==('rev-parse','HEAD'):return '0'*40
        return original(root,*args)
    monkeypatch.setattr(gov,'_git',fake_git)
    scope=aligned._scope(ROOT,prep,c_only=True)
    assert scope['call_ceiling']==90420 and 'continuation' not in scope
    assert scope['test_plan']['successful_semantic_call_ceiling']==3600
    assert all(c['arm_id']=='C' for c in scope['plan']['cells'])
    atomic_write_json(prep/'prep.json',dict(schema_version=aligned.C_PREP_SCHEMA,scope=scope))
    atomic_write_json(prep/'authorization.json',gov.authorization_template(scope))
    atomic_write_json(prep/'execution_plan.json',gov.execution_plan_template(scope))
    atomic_write_json(prep/'handoff.json',aligned.handoff_template(ROOT,prep,scope,c_only=True))
    assert gov.validate_prep(ROOT,prep)==scope
    with pytest.raises(SearchContractError,match='USER_PHASE_AUTHORIZATION'):
        gov.validate_prep(ROOT,prep,authorized=True)
    old=read_json(ROOT/aligned.MANIFEST)['plan']
    old_auth=gov.authorization_template(dict(scope_identity=digest('old'),plan=old))
    with pytest.raises(SearchContractError,match='AUTHORIZATION_SCOPE'):
        gov.validate_authorization(scope,old_auth,required=True)
    atomic_write_json(prep/'user_authorization_evidence.json',dict(synthetic_fixture_not_human_grant=True))
    auth=gov.authorization_template(scope)
    auth.update(approved=True,adaptive_validation_access=True,test_access=True,
        user_authorization_evidence_sha256=file_hash(prep/'user_authorization_evidence.json'))
    atomic_write_json(prep/'authorization.json',auth)
    handoff=aligned.handoff_template(ROOT,prep,scope,c_only=True);handoff['READY_TO_RUN']=True
    atomic_write_json(prep/'handoff.json',handoff)
    assert gov.validate_prep(ROOT,prep,authorized=True)==scope
    handoff['protocol']['manifest_path']=aligned.MANIFEST
    atomic_write_json(prep/'handoff.json',handoff)
    with pytest.raises(SearchContractError,match='HANDOFF_CHANGED'):
        gov.validate_prep(ROOT,prep,authorized=True)


@pytest.mark.parametrize('phase_smoke',['c_only'],indirect=True)
def test_c_only_actual_fake_phase_closure_and_test_exclude_AB(phase_smoke):
    root,prep,train,scope,transport=phase_smoke
    results,seal=gov.verify_training(train,scope)
    assert len(results)==1 and results[0]['cell']['arm_id']=='C'
    assert results[0]['rounds_completed']==1
    assert read_json(train/'access.json')['test_access'] is False
    assert not any(p.name.startswith(('A_','B_')) for p in train.iterdir())
    test=train.parent/'test_run'
    gov._execute_phase(root,prep,test,gov.final_test_scope(scope,seal),train_scope=scope,
        parent_results=results,terminal_sha=seal)
    values=read_json(test/'results.json')
    assert len(values)==1 and values[0]['cell']['arm_id']=='C'
    assert len(values[0]['predictions'])==300
    assert gov.verify_training(train,scope)[1]==seal
    assert read_json(test/'lifecycle.json')['status']=='EXECUTION_COMPLETE'
