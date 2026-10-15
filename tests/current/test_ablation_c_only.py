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
    from multi_dataset_diverse_rl.governance.frozen_documents import validate_published_single_prompt
    assert validate_published_single_prompt(ROOT,aligned.C_MANIFEST,c)
    with pytest.raises(SearchContractError):aligned.validate_manifest(ROOT,c,c_only=True)
    assert c['config']['runtime']==base['config']['runtime']
    with pytest.raises(SearchContractError):build_plan(c['config'])
    assert [(v['arm_id'],v['seed'],v['repeat_id']) for v in c['plan']['cells']]==[
        ('C',42,1),('C',42,2),('C',42,3)]
    assert c['plan']['training_semantic_call_ceiling']==90420
    assert c['plan']['final_test_semantic_call_ceiling']==3600
    assert c['config']['rounds']==20
    assert c['isolation']['import_old_caches'] is False
    assert c['isolation']['import_old_teacher_responses'] is False
    from scripts.run_experiment import preflight
    with pytest.raises(SearchContractError):preflight(c)
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


def test_historical_c_prep_cannot_freeze_revised_candidate_policy(tmp_path):
    prep=ROOT/'runs/structural_guard_v2_20261015'/tmp_path.name/'prep'
    with pytest.raises(SearchContractError,match='MANIFEST_MISMATCH'):
        aligned._scope(ROOT,prep,c_only=True)
    assert not prep.exists()


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
