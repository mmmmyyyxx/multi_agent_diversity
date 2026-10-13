"""Timeout recovery and immutable paid-prefix continuation; no real APIs."""
from copy import deepcopy
from pathlib import Path

import httpx
from openai import APITimeoutError
import pytest

from test_single_prompt_study import broker_for, ToyTransport, toy_rows
from multi_dataset_diverse_rl.research.single_prompt import generate_candidate, run_cell, raw_error
from multi_dataset_diverse_rl.research.single_prompt_policy import policy, call_ceiling
from multi_dataset_diverse_rl.research.single_prompt_governance import phase_cells, inventory, PREP_SCHEMA
from multi_dataset_diverse_rl.research.single_prompt_continuation import (
    operation_contract, replay_key, replay_guard, ResponseReplay, audit_parent, frozen_parent,
)
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.governance.token_accounting import OperationalAbort
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json, read_json
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest

ROOT = Path(__file__).resolve().parents[2]


def timeout():
    return APITimeoutError(request=httpx.Request('POST','https://synthetic.invalid'))


def enable_recovery(broker, inherited=0):
    broker.scope['continuation'] = dict(operation_policy=operation_contract(),
        request_identity_scope=broker.scope['scope_identity'], inherited_usage={})
    broker.timeout_recovery = operation_contract()
    broker.usage.update(replayed_calls=0, timeout_retries=0)
    broker.usage['provider_calls'] = inherited


def test_timeout_retries_preserve_request_and_semantic_draw(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr('multi_dataset_diverse_rl.research.single_prompt_runtime.time.sleep', sleeps.append)
    class Transient(ToyTransport):
        def for_role(self, role, request):
            if len(self.calls)<2:
                self.calls.append((role,deepcopy(request)))
                raise timeout()
            return super().for_role(role,request)
    b = broker_for(tmp_path,transport=Transient(),cap=None)
    enable_recovery(b)
    try:
        obs = b.evaluate(SEED,toy_rows(1),'optimize',88,0)['toy_0']
        assert obs['prediction']['semantic_attempt_count'] == 1
        assert obs['prediction']['transport_retry_attempts'] == 2
        assert b.usage['provider_calls'] == 3 and b.usage['timeout_retries'] == 2
        assert sleeps == [2,4]
        assert all(req == b.transport.calls[0][1] for _,req in b.transport.calls)
        assert b.ledger.view()['fallback_charged']>0 and not b.ledger.inflight
        assert len(list((tmp_path/'receipts').glob('*.json'))) == 3
    finally:
        b.ledger.close()


def test_retries_obey_combined_call_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr('multi_dataset_diverse_rl.research.single_prompt_runtime.time.sleep', lambda _:None)
    class Timeout(ToyTransport):
        def for_role(self,role,request):
            self.calls.append((role,request))
            raise timeout()
    b=broker_for(tmp_path,transport=Timeout(),cap=None)
    enable_recovery(b,inherited=8)
    b.scope['call_ceiling']=10
    try:
        with pytest.raises(OperationalAbort,match='CALL_CEILING'):
            b.evaluate(SEED,toy_rows(1),'optimize',88,0)
        assert len(b.transport.calls)==2 and b.usage['provider_calls']==10
        assert not b.ledger.inflight
    finally:
        b.ledger.close()


def test_timeout_and_length_recovery_keep_distinct_draws(tmp_path, monkeypatch):
    monkeypatch.setattr('multi_dataset_diverse_rl.research.single_prompt_runtime.time.sleep', lambda _:None)
    class LengthThenTimeout(ToyTransport):
        def for_role(self,role,request):
            self.calls.append((role,deepcopy(request)))
            if len(self.calls)==2:
                raise timeout()
            text,finish=('incomplete','length') if len(self.calls)==1 else ('Final answer: 1','stop')
            return dict(text=text,finish_reason=finish,input_tokens=10,output_tokens=10)
    b=broker_for(tmp_path,transport=LengthThenTimeout(),cap=None)
    enable_recovery(b)
    try:
        obs=b.evaluate(SEED,toy_rows(1),'optimize',88,0)['toy_0']
        assert obs['prediction']['semantic_attempt_count']==2
        assert [r['max_tokens'] for _,r in b.transport.calls]==[3600,6144,6144]
        assert b.transport.calls[1][1]==b.transport.calls[2][1]
        assert b.usage['timeout_retries']==1 and b.usage['provider_calls']==3
    finally:
        b.ledger.close()


def test_replay_fifo_keeps_invalid_generated_outputs_and_gradient_subjects(tmp_path):
    rows=[dict(key='key',guard_key='guard',response=dict(text=t),source_scope_identity='p',
        reservation_id=str(n),source_receipt_sha256='h') for n,t in enumerate(('bad json','legal output'))]
    replay=ResponseReplay(rows,tmp_path/'log.jsonl')
    assert replay.take('key','guard')['text']=='bad json'
    assert replay.take('key','guard')['text']=='legal output'
    replay.require_consumed()
    requests=[dict(messages=[dict(content=__import__('json').dumps(dict(example=dict(example_id=i))))]) for i in ('a','b')]
    assert replay_guard(requests[0],88,1,'pattern_gradient','gradient') != replay_guard(requests[1],88,1,'pattern_gradient','gradient')


def test_other_provider_failure_and_old_timeout_scope_still_abort(tmp_path):
    for recovery, error in ((True,RuntimeError('synthetic')), (False,timeout())):
        class Failed(ToyTransport):
            def for_role(self,role,request):
                self.calls.append((role,request))
                raise error
        b=broker_for(tmp_path/str(recovery),transport=Failed(),cap=None)
        if recovery:
            enable_recovery(b)
        try:
            with pytest.raises(OperationalAbort,match='PROVIDER_FAILURE'):
                b.evaluate(SEED,toy_rows(1),'optimize',88,0)
            assert len(b.transport.calls)==1 and not b.ledger.inflight
        finally:
            b.ledger.close()


def synthetic_closed_parent(tmp_path, failure=None):
    root=tmp_path/'root'
    run=root/'runs/prefix/train_run'
    prep=root/'runs/prefix/train_prep'
    run.mkdir(parents=True)
    manifest=read_json(ROOT/'experiments/manifests/single_prompt_gradient_pattern_v1.json')
    scope=dict(study_id=manifest['experiment_id'],phase='train',source_sha='93c7f7f6804dfda30c0ab2452f78c806d3c3814c',
        policy=policy(),cells=phase_cells(manifest,'train'),call_ceiling=call_ceiling('train'),
        runtime_identity=__import__('multi_dataset_diverse_rl.research.single_prompt_governance',fromlist=['runtime_identity']).runtime_identity(),
        manifest_sha256=__import__('multi_dataset_diverse_rl.benchmarks.data_freeze',fromlist=['file_hash']).file_hash(ROOT/'experiments/manifests/single_prompt_gradient_pattern_v1.json'),
        token_ceiling=None,search_predecessor=None,
        execution=dict(prep_path='runs/prefix/train_prep',run_root='runs/prefix/train_run'))
    scope['scope_identity']=digest(scope)
    class Partial(ToyTransport):
        failed=False
        def for_role(self,role,request):
            if role=='solver' and self.reflections and not self.failed:
                self.failed=True
                self.calls.append((role,deepcopy(request)))
                raise timeout() if failure is None else failure
            return super().for_role(role,request)
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_V27_FORMAL_UNCAPPED
    from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
    from multi_dataset_diverse_rl.research.single_prompt_runtime import StudyBroker
    b=StudyBroker(phase='train',scope=scope,transport=Partial(),
        ledger=TokenLedger(run/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED),
        receipts=ProviderResponseReceipts(run/'provider_receipts',attempt_id=scope['scope_identity'],startup_identity_sha256=scope['scope_identity']),
        dispatch_guard=lambda _:None)
    c=scope['cells'][0]
    b.cell_id,b.attempt_id,b.seed=c['cell_id'],c['attempt_id'],88
    rows=toy_rows(3)
    initial=b.evaluate(SEED,rows,'optimize',88,0)
    b.round_index=1
    feedback=dict(errors=[raw_error(r,initial[r['stable_example_id']]) for r in rows
        if not initial[r['stable_example_id']]['correct']],structured_feedback=None)
    child,_=generate_candidate(SEED,feedback,b,1)
    with pytest.raises(OperationalAbort):
        b.evaluate(child,rows,'optimize',88,1)
    view=b.ledger.view()
    b.ledger.close()
    auth=dict(scope_identity=scope['scope_identity'],approved=True,
        user_authorization_evidence_sha256='e'*64,authorized_attempt_ids=[c['attempt_id'] for c in scope['cells']])
    atomic_write_json(prep/'prep.json',dict(schema_version=PREP_SCHEMA,scope=scope))
    atomic_write_json(prep/'authorization.json',auth)
    atomic_write_json(run/'scope.json',scope)
    atomic_write_json(run/'authorization.json',auth)
    atomic_write_json(run/'lifecycle.json',dict(status='EXECUTION_ABORTED',error_category='OperationalAbort'))
    atomic_write_json(run/'abort.json',dict(error_category='OperationalAbort',token_ledger=view))
    atomic_write_json(root/'runs/single_prompt_consumed'/(scope['scope_identity']+'.json'),
        dict(scope_identity=scope['scope_identity'],run_root='runs/prefix/train_run'))
    return root,run,scope,initial,child,feedback


def test_partial_checkpoint_resume_keeps_optimizer_and_successful_solver_draws(tmp_path):
    root,run,scope,initial,child,feedback=synthetic_closed_parent(tmp_path)
    before=inventory(run)
    parent=audit_parent(root,run)
    assert parent['inherited_usage']['provider_calls']==7 and len(parent['replay'])==6
    old_cost=parent['historical_cost_by_cell'][scope['cells'][0]['cell_id']]
    assert old_cost['0']['provider_calls']==3 and old_cost['1']['provider_calls']==4
    assert old_cost['1']['fallback_usage_calls']==1
    b=broker_for(tmp_path/'new',cap=None)
    enable_recovery(b,inherited=7)
    b.scope['continuation']['request_identity_scope']=scope['scope_identity']
    b.replay=ResponseReplay(parent['replay'],tmp_path/'new/replay.jsonl')
    try:
        rebuilt=b.evaluate(SEED,toy_rows(3),'optimize',88,0)
        assert rebuilt==initial
        b.round_index=1
        rebuilt_child,_=generate_candidate(SEED,feedback,b,1)
        assert rebuilt_child==child and b.transport.reflections==0
        b.evaluate(rebuilt_child,toy_rows(3),'optimize',88,1)
        b.replay.require_consumed()
        assert len(b.transport.calls)==1 and b.usage['replayed_calls']==6
        assert b.usage['provider_calls']==8
        assert inventory(run)==before
    finally:
        b.ledger.close()


def test_resumed_cell_cost_includes_all_historical_sends(tmp_path):
    root,run,scope,_,_,_=synthetic_closed_parent(tmp_path)
    parent=audit_parent(root,run)
    b=broker_for(tmp_path/'new',cap=None)
    enable_recovery(b,inherited=7)
    b.scope['continuation'].update(parent)
    b.replay=ResponseReplay(parent['replay'],tmp_path/'new/replay.jsonl')
    c=dict(scope['cells'][0],initial_prompt=SEED.to_dict())
    try:
        result=run_cell(c,toy_rows(3),b.evaluate,b,lambda *_:None)
        b.replay.require_consumed()
        assert result['initial_cost']['provider_calls']==3
        assert result['metrics'][0]['cost']['provider_calls']==5
        assert result['total_cost']['provider_calls']==7+len(b.transport.calls)
        assert result['total_cost']['fallback_usage_calls']==1
        assert sum(m['cost']['provider_calls'] for m in result['metrics'])+result['initial_cost']['provider_calls']==result['total_cost']['provider_calls']
    finally:
        b.ledger.close()


def test_parent_corruption_non_timeout_and_replay_mismatch_fail_closed(tmp_path):
    root,run,scope,_,_,_=synthetic_closed_parent(tmp_path)
    parent=audit_parent(root,run)
    b=broker_for(tmp_path/'new',cap=None)
    enable_recovery(b)
    b.replay=ResponseReplay(parent['replay'],tmp_path/'new/replay.jsonl')
    try:
        changed=SEED.edit('strategy','Use a different strategy.')
        with pytest.raises(SearchContractError,match='REPLAY_REQUEST_MISMATCH'):
            b.evaluate(changed,toy_rows(1),'optimize',88,0)
        assert not b.transport.calls
        with pytest.raises(SearchContractError,match='UNCONSUMED_PARENT'):
            b.replay.require_consumed()
    finally:
        b.ledger.close()
    frozen=frozen_parent(root,run)
    atomic_write_json(run/'unexpected.json',dict(changed=True))
    with pytest.raises(SearchContractError,match='PARENT_MUTATED'):
        audit_parent(root,run,expected_inventory=frozen['parent_inventory'])
    other_root,other_run,*_=synthetic_closed_parent(tmp_path/'other',failure=RuntimeError('synthetic non-timeout'))
    with pytest.raises(SearchContractError,match='NON_TIMEOUT_PARENT_FAILURE'):
        audit_parent(other_root,other_run)


def test_continuation_freeze_auth_and_parent_mutation_guards(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.research import single_prompt_governance as gov
    from multi_dataset_diverse_rl.research.single_prompt_continuation import MANIFEST as OPERATION_MANIFEST
    root,run,_,_,_,_=synthetic_closed_parent(tmp_path)
    manifest=read_json(ROOT/gov.MANIFEST)
    atomic_write_json(root/gov.MANIFEST,manifest)
    (root/gov.MANIFEST).write_bytes((ROOT/gov.MANIFEST).read_bytes())
    atomic_write_json(root/OPERATION_MANIFEST,read_json(ROOT/OPERATION_MANIFEST))
    monkeypatch.setattr(gov,'validate_manifest',lambda *_:None)
    monkeypatch.setattr(gov,'require_selected_runtime',lambda _:None)
    monkeypatch.setattr(gov,'source_snapshot',lambda _: {})
    monkeypatch.setattr(gov.subprocess,'check_output',lambda args,**_: 'a'*40 if args[1]=='rev-parse' else '')
    prep=root/'runs/resumed/train_prep'
    gov.prepare(root,prep,phase='train',continuation_root=run)
    scope=gov.validate_prep(root,prep)
    assert set(c['attempt_id'] for c in scope['cells']).isdisjoint(c['attempt_id'] for c in read_json(run/'scope.json')['cells'])
    with pytest.raises(SearchContractError,match='EXACT_USER_AUTHORIZATION'):
        gov.execute(root,prep,root/'runs/resumed/train_run')
    assert not (root/'runs/resumed/train_run').exists()
    atomic_write_json(run/'modified.json',dict(changed=True))
    with pytest.raises(SearchContractError,match='PARENT_CHANGED'):
        gov.validate_prep(root,prep)
