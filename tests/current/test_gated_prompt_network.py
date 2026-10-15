"""Network continuation fixtures; all model responses synthetic."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import httpx
import pytest
from openai import APITimeoutError,APIConnectionError
from multi_dataset_diverse_rl.research.gated_prompt_network import NetworkBroker,operation_contract,validate_operation_manifest
from multi_dataset_diverse_rl.research.gated_prompt import generate_changed_candidate,run_cell
from multi_dataset_diverse_rl.research.gated_prompt_test import FinalTestBroker
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.governance.token_accounting import OperationalAbort
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('network_fixture',ROOT/'tests/current/test_gated_prompt_study.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def broker_for(path,transport,cap=100000):
    old=fixture.broker_for(path,transport)
    scope=dict(old.scope,call_ceiling=cap,continuation=dict(operation_policy=operation_contract(),inherited_usage={},request_identity_scope='a'*64))
    broker=NetworkBroker(phase='train',scope=scope,transport=old.transport,ledger=old.ledger,receipts=old.receipts,dispatch_guard=lambda _:None)
    broker.seed=88;broker.attempt_id=broker.cell_id='network_fixture';return broker


@pytest.mark.parametrize('failure_type',(APITimeoutError,APIConnectionError))
def test_retries_beyond_physical_limit_keep_one_semantic_draw_and_account_all(tmp_path,monkeypatch,failure_type):
    import multi_dataset_diverse_rl.research.single_prompt_runtime as runtime
    class Many(fixture.fixture.ToyTransport):
        attempts=0
        def for_role(self,role,request):
            self.attempts+=1
            if self.attempts<=8:raise failure_type(request=httpx.Request('POST','https://synthetic.invalid'))
            return super().for_role(role,request)
    delays=[];monkeypatch.setattr(runtime.time,'sleep',delays.append)
    broker=broker_for(tmp_path,Many(),1)
    try:
        candidate,draws=generate_changed_candidate(SEED,dict(errors=[],structured_feedback=None),broker,1,lambda *_:None)
        assert candidate!=SEED and draws==1 and broker.usage['provider_calls']==9
        assert broker.usage['network_failed_calls']==8 and broker.budgeted_provider_calls()==1
        assert delays==[2,4,8,16,30,60,60,60] and broker.usage['fallback_usage_calls']==8
        assert not broker.ledger.inflight and len(list(broker.receipts.directory.glob('[0-9]*.json')))==9
    finally:broker.ledger.close()


def test_network_failure_does_not_consume_four_format_draws(tmp_path,monkeypatch):
    import multi_dataset_diverse_rl.research.single_prompt_runtime as runtime
    class Alternating(fixture.fixture.ToyTransport):
        attempts=0
        def for_role(self,role,request):
            self.attempts+=1
            if self.attempts%2:raise APIConnectionError(request=httpx.Request('POST','https://synthetic.invalid'))
            result=super().for_role(role,request);result['text']='{malformed';return result
    monkeypatch.setattr(runtime.time,'sleep',lambda _:None);broker=broker_for(tmp_path,Alternating())
    artifacts={}
    try:
        candidate,draws=generate_changed_candidate(SEED,dict(errors=[],structured_feedback=None),broker,1,artifacts.__setitem__)
        assert candidate is None and draws==4 and len(artifacts)==4
        assert broker.usage['provider_calls']==8 and broker.usage['network_failed_calls']==4
    finally:broker.ledger.close()


def test_non_network_failure_aborts_and_cancels_further_dispatch(tmp_path):
    class Fatal(fixture.fixture.ToyTransport):
        def for_role(self,role,request):raise RuntimeError('synthetic credential/protocol fault')
    broker=broker_for(tmp_path,Fatal())
    try:
        with pytest.raises(OperationalAbort,match='PROVIDER_FAILURE'):broker.complete(role='reflection',split='optimize',stage='fatal',messages=[])
        with pytest.raises(OperationalAbort,match='FATAL_DRAIN'):broker.complete(role='reflection',split='optimize',stage='later',messages=[])
        assert broker.usage['provider_calls']==1 and broker.usage['network_failed_calls']==0 and not broker.ledger.inflight
    finally:broker.ledger.close()


def test_restore_checkpoint_skips_completed_evaluation_and_preserves_metrics(tmp_path):
    old=fixture.broker_for(tmp_path/'old');rows=fixture.fixture.toy_rows(2);artifacts={}
    old.evaluate=lambda *args:{r['stable_example_id']:fixture.fixture.observation(False) for r in rows}
    try:result=run_cell(fixture.fixture.cell(),rows,rows,old,artifacts.__setitem__)
    finally:old.ledger.close()
    broker=broker_for(tmp_path/'new',fixture.fixture.ToyTransport());calls=[];saved={}
    def evaluate(*args):calls.append((args[2],args[4]));return {r['stable_example_id']:fixture.fixture.observation(False) for r in rows}
    broker.evaluate=evaluate
    resume=dict(checkpoints=[artifacts[f'checkpoint_{i}'] for i in range(4)],round_costs={'4':{'provider_calls':7}},historical_usage={'provider_calls':10})
    try:
        resumed=run_cell(fixture.fixture.cell(),rows,rows,broker,saved.__setitem__,resume=resume)
        assert resumed['rounds_completed']==20 and all(n>=4 for _,n in calls)
        assert resumed['metrics'][:3]==result['metrics'][:3] and resumed['prompt_history'][:4]==result['prompt_history'][:4]
        assert resumed['metrics'][3]['cost']['provider_calls']==8 and resumed['usage']['provider_calls']==27
    finally:broker.ledger.close()


def test_final_test_has_the_same_unbounded_network_recovery(tmp_path,monkeypatch):
    import multi_dataset_diverse_rl.research.single_prompt_runtime as runtime
    class Once(fixture.fixture.ToyTransport):
        failed=False
        def for_role(self,role,request):
            if not self.failed:self.failed=True;raise APIConnectionError(request=httpx.Request('POST','https://synthetic.invalid'))
            return super().for_role(role,request)
    old=fixture.broker_for(tmp_path,Once());monkeypatch.setattr(runtime.time,'sleep',lambda _:None)
    broker=FinalTestBroker(phase='test',scope=dict(old.scope,call_ceiling=1,network_recovery=operation_contract()),
        transport=old.transport,ledger=old.ledger,receipts=old.receipts,dispatch_guard=lambda _:None)
    broker.seed=88;broker.attempt_id=broker.cell_id='final_network'
    try:
        broker.evaluate(SEED,fixture.fixture.toy_rows(1),'test',88,20)
        assert broker.usage['provider_calls']==2 and broker.usage['network_failed_calls']==1 and broker.budgeted_provider_calls()==1
    finally:broker.ledger.close()


def test_closed_operation_contract_rejects_changed_budget():
    from multi_dataset_diverse_rl.persistence.durable_io import read_json
    from multi_dataset_diverse_rl.research.gated_prompt_network import MANIFEST
    value=read_json(ROOT/MANIFEST);validate_operation_manifest(value)
    value['operation_policy']['combined_physical_call_ceiling']=190440
    with pytest.raises(SearchContractError,match='MANIFEST_MISMATCH'):validate_operation_manifest(value)


def test_replay_wrong_request_cannot_dispatch_as_new_model_draw(tmp_path):
    from multi_dataset_diverse_rl.research.single_prompt_continuation import ResponseReplay,replay_key,replay_guard
    messages=[dict(role='system',content=SEED.render()),dict(role='user',content='Synthetic fixture value: 0')]
    broker=broker_for(tmp_path,fixture.fixture.ToyTransport())
    try:
        request,key=broker._request_identity(role='solver',split='optimize',messages=messages)
        entry=dict(key=replay_key(request,88,0,'solver','optimize_checkpoint_0_draw_1'),
            guard_key=replay_guard(request,88,0,'solver','optimize_checkpoint_0_draw_1'),response={},source_scope_identity='a'*64)
        broker.replay=ResponseReplay([entry],tmp_path/'replay.jsonl')
        with pytest.raises(SearchContractError,match='REQUEST_MISMATCH'):
            broker.complete(role='solver',split='optimize',stage='optimize_checkpoint_0_draw_1',messages=messages,capacity=6144)
        assert broker.usage['provider_calls']==0 and not broker.transport.calls
    finally:broker.ledger.close()


def test_changed_parent_inventory_rejected_before_reading_paid_evidence(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.research import gated_prompt_network as net,gated_prompt_governance as gov
    monkeypatch.setattr(gov.shared,'inventory',lambda _:dict(artifact='new_hash'))
    with pytest.raises(SearchContractError,match='PARENT_CHANGED'):
        net.audit_parent(tmp_path,tmp_path/'runs/parent',expected_inventory=dict(artifact='old_hash'))


def test_restore_initial_prompt_mismatch_does_not_dispatch(tmp_path):
    broker=broker_for(tmp_path,fixture.fixture.ToyTransport());rows=fixture.fixture.toy_rows(1)
    try:
        resume=dict(checkpoints=[dict(prompt=SEED.edit('strategy','Wrong parent.').to_dict())],round_costs={},historical_usage={})
        with pytest.raises(SearchContractError,match='INITIAL_PROMPT_MISMATCH'):
            run_cell(fixture.fixture.cell(),rows,rows,broker,lambda *_:None,resume=resume)
        assert broker.usage['provider_calls']==0
    finally:broker.ledger.close()
