"""Fresh concurrency scope, immutable paid stages and bounded restoration."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.research.ablations import proposal_continuation as c
from multi_dataset_diverse_rl.research.ablations import governance as gov
from multi_dataset_diverse_rl.research.ablations.proposal_config import build_plan,default_config
from multi_dataset_diverse_rl.research.ablations.pattern_proposals import run_cell
from multi_dataset_diverse_rl.research.ablations.proposal_audit import audit_transition
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport,synthetic_rows
from multi_dataset_diverse_rl.research.ablations.runtime import AblationBroker
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_GATED_32_UNCAPPED,POLICY_PATTERN_96_UNCAPPED,audit_journal
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError


class OnePattern(SyntheticTransport):
    def for_role(self,role,request):
        value=super().for_role(role,request)
        if role=='pattern_cluster':
            ids=[g['example_id'] for g in json.loads(request['messages'][1]['content'])['gradients']]
            value['text']=json.dumps(dict(patterns=[dict(generalized_gradient='Check the algebraic transformation before computing.',support_ids=ids)],unassigned_ids=[]))
        return value


def test_96_ledger_is_separate_and_bounded(tmp_path):
    ledger=TokenLedger(tmp_path/'ledger',task_sha256='a'*64,policy=POLICY_PATTERN_96_UNCAPPED)
    request=dict(model='gpt-4o-mini',messages=[dict(role='user',content='synthetic')],max_tokens=1)
    try:
        ids=[ledger.reserve(request,attempt_id='fake',stage='synthetic',role='solver',model='gpt-4o-mini') for _ in range(96)]
        from multi_dataset_diverse_rl.governance.token_accounting import OperationalAbort
        with pytest.raises(OperationalAbort):ledger.reserve(request,attempt_id='fake',stage='synthetic',role='solver',model='gpt-4o-mini')
        for rid in ids:ledger.reconcile(rid,{},outcome='SYNTHETIC')
    finally:ledger.close()
    assert audit_journal(tmp_path/'ledger',task_sha256='a'*64,policy=POLICY_PATTERN_96_UNCAPPED)[0]['reserved_inflight']==0
    assert POLICY_GATED_32_UNCAPPED['concurrency']==32


@pytest.mark.parametrize('field',['solver_concurrency','gradient_concurrency','test_solver_concurrency','local_grading_concurrency'])
def test_override_is_exact_not_arbitrary(field):
    scope=dict(phase='train',execution_concurrency=deepcopy(c.CONCURRENCY),continuation=dict(identity=c.PREP_SCHEMA))
    assert c.execution_concurrency(scope)==c.CONCURRENCY
    scope['execution_concurrency'][field]+=1
    with pytest.raises(SearchContractError):c.execution_concurrency(scope)


def test_test_scope_has_96_but_no_training_cache():
    scope=dict(source_sha='b'*40,scope_identity='c'*64,plan={},test_plan=dict(successful_semantic_call_ceiling=10800),
        network_recovery={},execution_concurrency=c.CONCURRENCY,continuation=dict(identity=c.PREP_SCHEMA))
    test=gov.final_test_scope(scope,'d'*64)
    assert c.execution_concurrency(test)['test_solver_concurrency']==96
    assert 'continuation' not in test and gov.accounting_policy(test,'test')==POLICY_PATTERN_96_UNCAPPED


def test_resume_saved_diagnosis_and_candidate_without_teacher_replay(tmp_path):
    config=default_config();config['rounds']=config['runtime']['rounds']=1
    config['repeats_per_arm']=config['runtime']['repeats_per_arm']=1
    plan=build_plan(config);cell=next(v for v in plan['cells'] if v['arm_id']=='C2')
    rows=synthetic_rows('optimize',4);validation=synthetic_rows('validation',2);arts={};old_scope='e'*64
    def persist(name,value):
        assert name not in arts;arts[name]=deepcopy(value)
    ledger=TokenLedger(tmp_path/'parent/ledger',task_sha256=old_scope,policy=POLICY_GATED_32_UNCAPPED)
    scope=dict(scope_identity=old_scope,phase='train',call_ceiling=10000,token_ceiling=None,generation_recovery=config['runtime']['diagnosis_recovery'])
    broker=AblationBroker(plan=plan,phase='train',scope=scope,transport=OnePattern(),ledger=ledger,
        receipts=ProviderResponseReceipts(tmp_path/'parent/receipts',attempt_id='fake',startup_identity_sha256=old_scope),dispatch_guard=lambda _:None)
    broker.usage["network_failed_calls"]=0
    original=broker.evaluate
    def interrupt(prompt,rows,split,seed,ordinal):
        if ordinal:raise RuntimeError('synthetic interruption after paid generation')
        return original(prompt,rows,split,seed,ordinal)
    broker.evaluate=interrupt
    try:
        with pytest.raises(RuntimeError):run_cell(plan,cell,SEED.to_dict(),rows,validation,broker,persist)
        old=broker.usage_view();cache=deepcopy(broker.cache)
    finally:ledger.close()
    prefix=arts['checkpoint_0']['bootstrap_usage'];partial=c._difference(old,prefix)
    diagnostic=c._zero();generation=c._zero()
    for role,target in [('pattern_gradient',diagnostic),('pattern_cluster',diagnostic),('reflection',generation)]:
        target[role]=partial[role]
        for k in ('input_tokens','output_tokens'):
            target[role+'_'+k]=partial[role+'_'+k];target[k]+=partial[role+'_'+k]
        target['provider_calls']+=partial[role]
    diagnostic['supplier_reported_tokens']=diagnostic['input_tokens']+diagnostic['output_tokens']
    generation['supplier_reported_tokens']=generation['input_tokens']+generation['output_tokens']
    assert c._add(diagnostic,generation)==partial
    resume=dict(checkpoints=[arts['checkpoint_0']],historical_usage=prefix,artifacts=deepcopy(arts),
        partial=dict(ordinal=1,usage=partial,diagnostic_usage=diagnostic,generation_usage={'0':generation},evaluation_usage={}),
        fresh_continuation_identity=c.PREP_SCHEMA)
    child_scope=dict(scope,scope_identity='f'*64,execution_concurrency=c.CONCURRENCY,
        continuation=dict(identity=c.PREP_SCHEMA,inherited_usage=old,request_identity_scope=old_scope))
    transport=OnePattern();child_ledger=TokenLedger(tmp_path/'child/ledger',task_sha256='f'*64,policy=POLICY_PATTERN_96_UNCAPPED)
    child=c.SchedulingResumeBroker(plan=plan,phase='train',scope=child_scope,transport=transport,ledger=child_ledger,
        receipts=ProviderResponseReceipts(tmp_path/'child/receipts',attempt_id='fake',startup_identity_sha256='f'*64),dispatch_guard=lambda _:None)
    child.cache=cache
    try:
        result=run_cell(plan,cell,SEED.to_dict(),rows,validation,child,persist,resume=resume)
        assert all(role=='solver' for role,_ in transport.calls)
        assert result['usage']['pattern_gradient']==old['pattern_gradient']
        assert result['usage']['reflection']==old['reflection'] and child.usage_view()['replayed_calls']==0
        assert child.study_policy()['solver_concurrency']==96 and child.execution_policy['gradient_max_concurrency']==16
        audit_transition(arts['checkpoint_0'],arts['checkpoint_1'],plan,cell,1,lambda name:arts[name])
    finally:child_ledger.close()


def test_unsanctioned_partial_identity_fails():
    with pytest.raises(SearchContractError):
        c.validate_partial_resume(dict(partial=dict(ordinal=1),checkpoints=[{}]),dict(proposal_mode='parallel'))


def test_paid_draw_is_solver_only_exact_and_single_use(tmp_path,monkeypatch):
    import hashlib,threading
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    broker=c.SchedulingResumeBroker.__new__(c.SchedulingResumeBroker)
    request=dict(model='gpt-4o-mini',messages=[dict(role='user',content='synthetic')],max_tokens=1)
    broker._request_identity=lambda **kw:(request,'a'*64)
    broker.lock=threading.RLock();broker.cell_id='C2_seed42_repeat02';broker.used_solver=set()
    broker.dispatch_guard=lambda _:None;broker.receipts=SimpleNamespace(directory=tmp_path)
    broker._validated_response=lambda response,*args:response
    slot=(broker.cell_id,'optimize_checkpoint_13_draw_1',hashlib.sha256(serialized_request(request)).hexdigest())
    broker.paid_solver={slot:dict(record=dict(request=request,response=dict(text='paid synthetic')),
        reservation_id='1',receipt_sha256='b'*64)}
    monkeypatch.setattr(AblationBroker,'complete',lambda self,**kw:dict(text='fresh synthetic'))
    kwargs=dict(role='solver',split='optimize',stage=slot[1],messages=request['messages'])
    assert broker.complete(**kwargs)['text']=='paid synthetic'
    with pytest.raises(SearchContractError):broker.complete(**kwargs)
    assert broker.complete(**dict(kwargs,stage='optimize_checkpoint_13_draw_2'))['text']=='fresh synthetic'
    assert broker.complete(**dict(kwargs,role='reflection'))['text']=='fresh synthetic'


def test_restored_terminal_score_must_match(monkeypatch):
    value=dict(request_sha256='a'*64,correct=True)
    monkeypatch.setattr(AblationBroker,'prediction',lambda *args,**kw:deepcopy(value))
    broker=c.SchedulingResumeBroker.__new__(c.SchedulingResumeBroker)
    broker.expected_terminal={'a'*64:digest(value)}
    assert broker.prediction()==value
    broker.expected_terminal['a'*64]=digest(dict(value,correct=False))
    with pytest.raises(SearchContractError):broker.prediction()


def test_manifest_refuses_changed_science_or_budgets():
    assert c.validate_manifest(c.manifest_template())['training_semantic_call_ceiling']==391740
    for field in ('training_semantic_call_ceiling','test_semantic_call_ceiling'):
        value=c.manifest_template();value[field]+=1
        with pytest.raises(SearchContractError):c.validate_manifest(value)


def test_parser_and_scorer_share_local_admission_only(monkeypatch):
    import threading,time
    from concurrent.futures import ThreadPoolExecutor
    from multi_dataset_diverse_rl.research.single_prompt_runtime import StudyBroker
    lock=threading.Lock();active=0;peak=0
    def work(*args):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        try:time.sleep(0.005);return 'unchanged-native-result'
        finally:
            with lock:active-=1
    monkeypatch.setattr(StudyBroker,'classify_response',work)
    monkeypatch.setattr(StudyBroker,'score_resolved',work)
    broker=AblationBroker.__new__(AblationBroker);broker.grading_slots=threading.BoundedSemaphore(2)
    jobs=[lambda:broker.classify_response({}),lambda:broker.score_resolved(None,{})]*6
    with ThreadPoolExecutor(max_workers=12) as pool:results=list(pool.map(lambda f:f(),jobs))
    assert results==['unchanged-native-result']*12 and peak==2
    broker.grading_slots=None
    assert broker.classify_response({})==broker.score_resolved(None,{})=='unchanged-native-result'
