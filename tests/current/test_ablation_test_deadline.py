"""Scoped grading resources and preservation of paid Test evidence, under network guard."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from multi_dataset_diverse_rl.benchmarks import math_domain_v2 as domain
from multi_dataset_diverse_rl.research.ablations import test_deadline as run,test_resume as prior
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.research.ablations.synthetic import synthetic_rows
from tests.current import test_ablation_test_resume as old_tests
from tests.current.test_ablation_test_resume import factory


def test_worker_deadline_default_scope_and_thread_isolation(monkeypatch):
    seen=[]
    def fake(*args,**kwargs):
        seen.append(kwargs['timeout'])
        return SimpleNamespace(returncode=0,stdout=json.dumps(dict(valid=[True],equivalence=[[True]])))
    monkeypatch.setattr(domain.subprocess,'run',fake)
    domain.domain_matrix.cache_clear()
    try:
        domain.domain_matrix(('default',))
        with domain.worker_deadline(30):
            domain.domain_matrix(('extended',))
            with ThreadPoolExecutor(1) as pool:
                pool.submit(domain.domain_matrix,('independent-thread',)).result()
        domain.domain_matrix(('restored',))
        assert seen==[8,30,8,8] and domain.SETTINGS['process_deadline']==8
    finally:domain.domain_matrix.cache_clear()


def test_deadline_restores_after_exception():
    assert domain._WORKER_DEADLINE.get() is None
    with pytest.raises(RuntimeError):
        with domain.worker_deadline(30):raise RuntimeError('fixture')
    assert domain._WORKER_DEADLINE.get() is None
    with pytest.raises(SearchContractError,match='DEADLINE_INVALID'):
        with domain.worker_deadline(4):pass


@pytest.mark.parametrize('wrapped',[False,True])
def test_same_paid_answer_retries_at_thirty_seconds(factory,monkeypatch,wrapped):
    monkeypatch.setattr(old_tests,'ResumeBroker',run.TestResumeBroker)
    b,t,p=factory();b.scope['local_evaluator_recovery']=run.operation_policy()
    seen=[]
    def score(self,parsed,gold):
        seen.append((parsed.answer,gold,domain._WORKER_DEADLINE.get()))
        if len(seen)<4:
            try:raise SearchContractError('MATH_EVALUATOR_TIMEOUT')
            except SearchContractError as exc:
                if wrapped:raise SearchContractError('REFERENCE_UNSCORABLE') from exc
                raise
        return 0.
    monkeypatch.setattr(MATHBenchmarkAdapter,'score_member_output',score)
    monkeypatch.setattr('multi_dataset_diverse_rl.research.ablations.test_resume_runtime.time.sleep',lambda _:None)
    value=b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)
    assert not value['correct'] and len(t.calls)==1 and b.local_retry_count==3
    assert len(set(seen))==1 and seen[0][2]==30
    assert domain._WORKER_DEADLINE.get() is None


def test_paid_response_and_terminal_score_still_exact(factory,monkeypatch):
    prefix,value,row,cell=old_tests.paid(factory)
    monkeypatch.setattr(old_tests,'ResumeBroker',run.TestResumeBroker)
    b,t,_=factory(prefix,{(cell,value['request_sha256']):'f'*64})
    b.scope['local_evaluator_recovery']=run.operation_policy()
    with pytest.raises(SearchContractError,match='TERMINAL_SCORE_CHANGED'):
        b.prediction(SEED,row,'test',42,0)
    assert not t.calls and b.usage_view()['replayed_calls']==1


def test_reference_check_receives_deadline_and_restores(factory,monkeypatch):
    monkeypatch.setattr(old_tests,'ResumeBroker',run.TestResumeBroker)
    b,t,_=factory();b.scope['local_evaluator_recovery']=run.operation_policy()
    assert b.retry_local(lambda:domain._WORKER_DEADLINE.get(),dict(operation='test_membership_reference_check'))==30
    assert domain._WORKER_DEADLINE.get() is None and not t.calls


def test_v9_manifest_preserves_math_and_remaining_budget():
    from scripts.run_experiment import preflight
    root=Path(__file__).resolve().parents[2]
    value=json.loads((root/run.MANIFEST).read_text(encoding='utf-8'))
    run.validate_manifest(root,value)
    assert preflight(value)['status']=='TEST_DEADLINE_PREFLIGHT_PASS_NOT_AUTHORIZED'
    policy=run.operation_policy()
    assert policy['mathematical_settings']==domain.SETTINGS
    assert policy['worker_deadline_seconds']==30 and policy['new_call_ceiling']+737==3600
    assert prior.operation_policy()['mathematical_settings']['process_deadline']==8
    value['operation_policy']['worker_deadline_seconds']=60
    with pytest.raises(SearchContractError,match='MANIFEST_CHANGED'):run.validate_manifest(root,value)


def test_shared_runner_dispatches_explicit_contract_without_mutating_v8(monkeypatch):
    captured={}
    def execute(root,prep,target,*,contract=None):captured['contract']=contract;return dict(fake=True)
    monkeypatch.setattr(prior,'execute',execute)
    assert run.execute(Path('.'),Path('p'),Path('r'))==dict(fake=True)
    assert captured['contract'] is run and prior.TestResumeBroker is not run.TestResumeBroker


def test_fake_v9_phase_uses_shared_runner_and_finishes_remaining_cells(tmp_path,monkeypatch):
    monkeypatch.setattr(old_tests,'run',run)
    old_tests.test_governed_fake_test_phase_inherits_one_cell_and_finishes_two(tmp_path,monkeypatch)
