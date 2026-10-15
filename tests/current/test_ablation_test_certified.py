from pathlib import Path
import json
import pytest
from tests.current import test_ablation_test_resume as old_tests
from tests.current.test_ablation_test_resume import factory
from multi_dataset_diverse_rl.research.ablations import test_certified as run,test_deadline as previous
from multi_dataset_diverse_rl.research.ablations.synthetic import synthetic_rows
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import SETTINGS


def test_certificate_phase_scores_and_records_without_extra_model_calls(factory,monkeypatch):
    monkeypatch.setattr(old_tests,'ResumeBroker',run.TestResumeBroker)
    monkeypatch.setattr(run,'require_scorable',lambda _:None)
    monkeypatch.setattr(run,'certified_matrix',lambda _:dict(valid=[True,True],equivalence=[[True,False],[False,True]],certified_pair_rejections=[[0,1],[1,0]]))
    b,t,p=factory();b.scope['local_evaluator_recovery']=run.operation_policy()
    result=b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)
    assert not result['correct'] and len(t.calls)==1 and b.local_retry_count==0
    entries=[json.loads(line) for line in (p/'receipts/local_scoring_certificates.jsonl').read_bytes().splitlines()]
    assert len(entries)==1 and entries[0]['certified_pair_rejections']==[[0,1],[1,0]]
    assert 'answer' not in entries[0]


def test_timeout_remains_unresolved_until_retry_succeeds_never_false_by_fallback(factory,monkeypatch):
    monkeypatch.setattr(old_tests,'ResumeBroker',run.TestResumeBroker)
    monkeypatch.setattr(run,'require_scorable',lambda _:None)
    count=[]
    def score(_):
        count.append(1)
        if len(count)<3:raise SearchContractError('MATH_EVALUATOR_TIMEOUT')
        return dict(valid=[True,True],equivalence=[[True,True],[True,True]],certified_pair_rejections=[])
    monkeypatch.setattr(run,'certified_matrix',score)
    monkeypatch.setattr('multi_dataset_diverse_rl.research.ablations.test_resume_runtime.time.sleep',lambda _:None)
    b,t,_=factory();b.scope['local_evaluator_recovery']=run.operation_policy()
    assert b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)['correct']
    assert len(t.calls)==1 and b.local_retry_count==2


def test_new_identity_and_exact_policy_keep_v9_manifest_unchanged():
    from scripts.run_experiment import preflight
    root=Path(__file__).resolve().parents[2]
    assert previous.manifest_template()==json.loads((root/previous.MANIFEST).read_text(encoding='utf-8'))
    value=json.loads((root/run.MANIFEST).read_text(encoding='utf-8'));run.validate_manifest(root,value)
    assert preflight(value)['status']=='TEST_CERTIFIED_PREFLIGHT_PASS_NOT_AUTHORIZED'
    assert run.operation_policy()['mathematical_settings']==SETTINGS
    assert 737+run.operation_policy()['new_call_ceiling']==3600
    value['operation_policy']['certified_rejection_policy']['precision_bits']+=1
    with pytest.raises(SearchContractError,match='MANIFEST_CHANGED'):run.validate_manifest(root,value)


def test_complete_fake_certified_phase_uses_actual_shared_runner(tmp_path,monkeypatch):
    monkeypatch.setattr(old_tests,'run',run)
    monkeypatch.setattr(run,'require_scorable',lambda _:None)
    monkeypatch.setattr(run,'certified_matrix',lambda _:dict(valid=[True,True],equivalence=[[True,False],[False,True]],certified_pair_rejections=[]))
    old_tests.test_governed_fake_test_phase_inherits_one_cell_and_finishes_two(tmp_path,monkeypatch)
