"""Synthetic checks of meaningful opportunities, gate observations and boundaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

from multi_dataset_diverse_rl.research.gated_prompt import (
    GatedPromptBroker,generate_changed_candidate,run_cell,
)
from multi_dataset_diverse_rl.research.gated_prompt_policy import policy,call_ceiling,gate_membership
from multi_dataset_diverse_rl.research import gated_prompt_governance as gov
from multi_dataset_diverse_rl.research.single_prompt_runtime import StudyBroker
from multi_dataset_diverse_rl.governance.token_accounting import OperationalAbort
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.system_prompt import SEED,SystemPrompt
from multi_dataset_diverse_rl.benchmarks.experiment_splits import ExperimentSplitReader
from multi_dataset_diverse_rl.benchmarks.data_freeze import file_hash

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('gated_toy_fixture',ROOT/'tests/current/test_single_prompt_study.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def broker_for(path,transport=None):
    original=fixture.broker_for(path,transport=transport)
    broker=GatedPromptBroker(phase='train',scope=original.scope,transport=original.transport,
        ledger=original.ledger,receipts=original.receipts,dispatch_guard=lambda _:None)
    broker.attempt_id=broker.cell_id='synthetic_gated';broker.seed=88
    return broker


def test_frozen_settings_and_metadata_only_membership():
    manifest=gov.manifest_template(ROOT)
    # Historical cell hashes stay frozen; only metadata membership is read here.
    reader=fixture.metadata_reader()
    with pytest.raises(SearchContractError,match='STUDY_MANIFEST_POLICY_MISMATCH'):
        gov.validate_manifest(ROOT,manifest)
    subset=gate_membership(reader)
    assert subset==gate_membership(reader) and subset['count']==100 and subset['source_count']==300
    original=[r['stable_example_id'] for r in reader.members if r['project_split']=='validation']
    assert subset['sample_ids']==[i for i in original if i in set(subset['sample_ids'])]
    assert call_ceiling()==190440 and policy()['rounds']==20
    assert policy()['optimizer_structural_draws']==4 and policy()['optimizer_structural_retries']==3
    assert not policy()['optimize_gate']
    assert manifest['authorization']==dict(real_api_authorized=False,adaptive_validation_access=False,test_access=False)


def test_format_retry_then_long_candidate_without_character_budget(tmp_path):
    class Sequence(fixture.ToyTransport):
        def for_role(self,role,request):
            result=super().for_role(role,request)
            n=self.reflections
            result['text']=['bad json',json.dumps(dict(block='strategy',content='x'*3001)),
                json.dumps(dict(block='strategy',content=SEED.strategy)),
                json.dumps(dict(block='strategy',content='Check arithmetic.'))][n-1]
            return result
    broker=broker_for(tmp_path,Sequence());artifacts={}
    try:
        candidate,draws=generate_changed_candidate(SEED,dict(errors=[],structured_feedback=None),broker,1,artifacts.__setitem__)
        assert draws==2 and len(candidate.strategy)==3001 and len(artifacts)==2
        assert artifacts['round_1_generation_1']['candidate'] is None
        packets=[json.loads(r['messages'][1]['content']) for _,r in broker.transport.calls]
        assert 'output_contract_feedback' not in packets[0]
        assert all('x'*50 not in json.dumps(p) for p in packets)
        assert packets[-1]['output_contract_feedback']['errors']==['REFLECTION_OUTPUT_CONTRACT_INVALID']
        assert all('block_character_limits' not in p for p in packets)
    finally:broker.ledger.close()


@pytest.mark.parametrize('kind',('malformed',))
def test_four_failed_generations_abandon_each_of_twenty_opportunities_without_rescoring(tmp_path,kind):
    class Invalid(fixture.ToyTransport):
        def for_role(self,role,request):
            result=super().for_role(role,request)
            result['text']={'malformed':'invalid json',
                'overlong':json.dumps(dict(block='strategy',content='x'*3001)),
                'unchanged':json.dumps(dict(block='strategy',content=SEED.strategy))}[kind]
            return result
    broker=broker_for(tmp_path,Invalid());artifacts={};rows=fixture.toy_rows(2);evaluations=[]
    original={r['stable_example_id']:fixture.observation() for r in rows}
    def evaluate(*args):evaluations.append(args[2]);return deepcopy(original)
    broker.evaluate=evaluate
    try:
        result=run_cell(fixture.cell(),rows,rows,broker,artifacts.__setitem__)
        assert result['rounds_completed']==20 and result['opportunities_abandoned']==20
        assert result['candidates_evaluated']==0 and broker.usage['reflection']==80
        assert evaluations==['optimize','validation']
        assert len(result['prompt_history'])==21 and all(p==SEED.to_dict() for p in result['prompt_history'])
        assert all(m['decision']=='ABANDONED_GENERATION_RETRIES_EXHAUSTED' and m['generation_draws']==4 for m in result['metrics'])
        for ordinal in range(1,21):
            checkpoint=artifacts[f'checkpoint_{ordinal}']
            assert checkpoint['candidate_prompt'] is None and checkpoint['candidate_optimize'] is None
            assert checkpoint['candidate_validation'] is None and checkpoint['metric']['validation_candidate'] is None
            assert checkpoint['optimize']==checkpoint['validation']==original
    finally:broker.ledger.close()


def test_named_block_replacement_has_no_character_budget():
    parent=SystemPrompt(SEED.role,'x'*2650,SEED.answer)
    for block in ('role','strategy','answer'):
        child=parent.edit(block,'z'*20000)
        assert len(child.render())>20000
    assert len(parent.edit('strategy','Check calculations.').render())<len(parent.render())


def test_gate_strict_gain_and_rejection_preserves_both_observation_profiles(tmp_path):
    broker=broker_for(tmp_path);artifacts={};rows=fixture.toy_rows(5)
    retained_profiles={}
    def evaluate(prompt,rows,split,seed,ordinal):
        # Initial validation1; first candidate improves to2 while Optimize declines.
        # Second equals2; third worse1; remaining candidates equal2.
        correct=(4 if ordinal==0 else 0) if split=='optimize' else (1 if ordinal in (0,3) else 2)
        values={r['stable_example_id']:fixture.observation(i<correct) for i,r in enumerate(rows)}
        if ordinal==1:retained_profiles[split]=deepcopy(values)
        return values
    broker.evaluate=evaluate
    try:
        result=run_cell(fixture.cell(),rows,rows,broker,artifacts.__setitem__)
        assert result['rounds_completed']==20 and len(result['prompt_history'])==21
        first=result['metrics'][0]
        assert first['accepted'] and first['optimize_candidate']<first['optimize_before']
        assert not any(m['accepted'] for m in result['metrics'][1:])
        assert all(m['validation_before']==2 for m in result['metrics'][1:])
        for n in range(1,21):
            assert artifacts[f'checkpoint_{n}']['optimize']==retained_profiles['optimize']
            assert artifacts[f'checkpoint_{n}']['validation']==retained_profiles['validation']
        assert result['final_prompt']==result['prompt_history'][1]
    finally:broker.ledger.close()


@pytest.mark.parametrize('arm',policy()['arms'])
def test_no_errors_still_generates_twenty_legal_changed_candidates(tmp_path,arm):
    broker=broker_for(tmp_path);rows=fixture.toy_rows(2)
    broker.evaluate=lambda *args:{r['stable_example_id']:fixture.observation(True) for r in rows}
    try:
        result=run_cell(fixture.cell(arm),rows,rows,broker,lambda *_:None)
        assert result['rounds_completed']==20 and broker.usage['reflection']==20
        assert not any(m['accepted'] for m in result['metrics'])
        assert broker.usage['pattern_gradient']==broker.usage['pattern_cluster']==0
        for role,request in broker.transport.calls:
            packet=json.loads(request['messages'][1]['content'])
            assert role=='reflection' and packet['errors']==[]
            assert not any(k in packet for k in ('validation','validation_before','validation_candidate'))
    finally:broker.ledger.close()


def test_call_ceiling_preserves_unfinished_opportunity_never_checkpoint_fallback(tmp_path):
    class Invalid(fixture.ToyTransport):
        def for_role(self,role,request):
            result=super().for_role(role,request);result['text']='bad json';return result
    broker=broker_for(tmp_path,Invalid());broker.scope['call_ceiling']=3
    rows=fixture.toy_rows(1);broker.evaluate=lambda *args:{r['stable_example_id']:fixture.observation() for r in rows}
    artifacts={}
    try:
        with pytest.raises(OperationalAbort,match='CALL_CEILING'):
            run_cell(fixture.cell(),rows,rows,broker,artifacts.__setitem__)
        assert 'checkpoint_0' in artifacts and 'checkpoint_1' not in artifacts and 'result' not in artifacts
        assert artifacts['round_1_opportunity']['round_complete'] is False
        assert len([k for k in artifacts if '_generation_' in k])==3
        assert broker.usage['provider_calls']==3 and not broker.ledger.inflight
    finally:broker.ledger.close()


def test_validation_cache_ignores_round_arm_but_separates_seed_and_prompt(tmp_path):
    broker=broker_for(tmp_path);row=fixture.toy_rows(1)
    try:
        first=broker.evaluate(SEED,row,'validation',88,0)
        broker.cell_id='another_arm'
        assert broker.evaluate(SEED,row,'validation',88,19)==first and broker.usage['solver']==1
        broker.seed=89;broker.evaluate(SEED,row,'validation',89,19)
        assert broker.usage['solver']==2
        broker.evaluate(SEED.edit('strategy','Check arithmetic.'),row,'validation',89,19)
        assert broker.usage['solver']==3
        with pytest.raises(SearchContractError,match='ROLE_SPLIT'):
            broker.complete(role='reflection',split='validation',stage='forbidden',messages=[])
        with pytest.raises(SearchContractError,match='ROLE_SPLIT'):
            broker.evaluate(SEED,row,'test',89,0)
    finally:broker.ledger.close()


def test_gradient_once_per_opportunity_and_no_gate_feedback_leak(tmp_path):
    broker=broker_for(tmp_path);artifacts={}
    try:
        result=run_cell(fixture.cell('gradient_pattern'),fixture.toy_rows(5),fixture.toy_rows(2),broker,artifacts.__setitem__)
        assert result['rounds_completed']==20 and broker.usage['pattern_gradient']==60
        assert broker.usage['pattern_cluster']==20
        for role,request in broker.transport.calls:
            if role=='reflection':
                packet=json.loads(request['messages'][1]['content'])
                assert len(packet['errors'])==3 and len(packet['structured_feedback']['gradients'])==3
                assert not any(k in packet for k in ('validation','validation_before','validation_candidate'))
            if role=='solver':assert len(request['messages'])==2
    finally:broker.ledger.close()


def test_subset_reader_parses_only_selected_rows_and_rejects_foreign_members(tmp_path):
    rows=[dict(stable_example_id=f'v{i}',content_sha256=f'h{i}',input_sha256=f'i{i}') for i in range(3)]
    path=tmp_path/'raw/math/train.jsonl';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows[0])+'\nINVALID UNSELECTED JSON\n'+json.dumps(rows[2])+'\n',encoding='utf-8')
    reader=ExperimentSplitReader.__new__(ExperimentSplitReader);reader.root=tmp_path;reader.benchmark='math'
    reader.members=[dict(r,project_split='validation',source_split='train',source_index=i) for i,r in enumerate(rows)]
    reader.manifest=dict(counts=dict(validation=3),canonical_sources=[dict(name='train',canonical_sha256=file_hash(path))],protocol='synthetic')
    assert reader._read_rows('validation',frozen_subset_ids=['v2','v0'])==(rows[0],rows[2])
    for bad in ([],['v0','v0'],['test_member']):
        with pytest.raises(SearchContractError,match='SUBSET_MEMBERSHIP'):reader._read_rows('validation',frozen_subset_ids=bad)
    path.write_text('changed',encoding='utf-8')
    with pytest.raises(SearchContractError,match='SOURCE_HASH'):reader._read_rows('validation',frozen_subset_ids=['v0'])


def test_gate_capability_checks_auth_before_any_content_read(monkeypatch):
    class Reader:
        members=[dict(stable_example_id=f'v{i}',project_split='validation') for i in range(300)]
        def _read_rows(self,*args,**kwargs):pytest.fail('Unauthorized content read')
    reader=Reader()
    def denied():raise SearchContractError('SYNTHETIC_AUTH_REQUIRED')
    with pytest.raises(SearchContractError,match='AUTH_REQUIRED'):
        gov.read_gate_rows(reader,gov.AdaptiveGateAccess('a'*64,gate_membership(reader),denied))
    with pytest.raises(SearchContractError,match='ACCESS_REQUIRED'):gov.read_gate_rows(reader,None)


def test_unapproved_execution_blocks_before_clients_or_run_creation(tmp_path,monkeypatch):
    def denied(*args,**kwargs):
        assert kwargs['authorized'] is True
        raise SearchContractError('GATED_EXACT_TRAIN_AND_VALIDATION_AUTHORIZATION_REQUIRED')
    monkeypatch.setattr(gov,'validate_prep',denied)
    with pytest.raises(SearchContractError,match='AUTHORIZATION_REQUIRED'):
        gov.execute(tmp_path,tmp_path/'runs/prep',tmp_path/'runs/run')
    assert not (tmp_path/'runs/run').exists() and not (tmp_path/'runs/gated_prompt_consumed').exists()


def test_gated_timeout_reuses_exact_request_and_accounts_failure(tmp_path,monkeypatch):
    import httpx
    from openai import APITimeoutError
    import multi_dataset_diverse_rl.research.single_prompt_runtime as runtime
    class Once(fixture.ToyTransport):
        def __init__(self):super().__init__();self.failed_request=None
        def for_role(self,role,request):
            if self.failed_request is None:
                self.failed_request=deepcopy(request)
                raise APITimeoutError(request=httpx.Request('POST','https://synthetic.invalid'))
            assert request==self.failed_request
            return super().for_role(role,request)
    monkeypatch.setattr(runtime.time,'sleep',lambda _:None)
    broker=broker_for(tmp_path,Once())
    try:
        candidate,draws=generate_changed_candidate(SEED,dict(errors=[],structured_feedback=None),broker,1,lambda *_:None)
        assert candidate!=SEED and draws==1 and broker.usage['provider_calls']==2
        assert broker.usage['timeout_retries']==1 and broker.usage['fallback_usage_calls']==1
        assert len(list(broker.receipts.directory.glob('[0-9]*.json')))==2 and not broker.ledger.inflight
    finally:broker.ledger.close()


@pytest.mark.parametrize('mutation',('policy','membership','command','authorization','test_access'))
def test_prep_freeze_and_authorization_tampering_fail_closed(tmp_path,monkeypatch,mutation):
    from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    manifest=gov.manifest_template(ROOT)
    target=tmp_path/gov.MANIFEST;atomic_write_json(target,manifest)
    class Metadata:
        members=[dict(stable_example_id=f'v{i}',project_split='validation') for i in range(300)]
    monkeypatch.setattr(gov,'validate_manifest',lambda *_:Metadata())
    monkeypatch.setattr(gov,'final_test_plan',lambda *_:dict(identity='synthetic'))
    monkeypatch.setattr(gov,'source_snapshot',lambda _: {})
    monkeypatch.setattr(gov.shared,'require_selected_runtime',lambda _:None)
    monkeypatch.setattr(gov.shared,'runtime_identity',lambda:dict(identity='synthetic'))
    monkeypatch.setattr(gov.subprocess,'check_output',lambda args,**kwargs:'a'*40 if args[1]=='rev-parse' else '')
    prep=tmp_path/'runs/prep';path='runs/prep';run='runs/train_run'
    scope=dict(study_id=gov.STUDY,phase='train',source_sha='a'*40,source_hashes={},runtime_identity=dict(identity='synthetic'),
        policy=policy(),manifest_sha256=file_hash(target),gate_membership=gate_membership(Metadata()),test_plan=dict(identity='synthetic'),
        cells=gov.phase_cells(manifest,path),call_ceiling=call_ceiling(),token_ceiling=None,
        execution=dict(prep_path=path,run_root=run,command=[policy()['runtime_interpreter'],'scripts/run_experiment.py','--prep',path,'--run-root',run,'--execute']))
    scope['scope_identity']=digest(scope)
    auth=dict(scope_identity=scope['scope_identity'],approved=True,adaptive_validation_access=True,test_access=False,
        user_authorization_evidence_sha256='e'*64,authorized_attempt_ids=[c['attempt_id'] for c in scope['cells']])
    atomic_write_json(prep/'prep.json',dict(schema_version=gov.PREP_SCHEMA,scope=scope));atomic_write_json(prep/'authorization.json',auth)
    assert gov.validate_prep(tmp_path,prep,authorized=True)==scope
    if mutation=='policy':scope['policy']['rounds']=19
    elif mutation=='membership':scope['gate_membership']['sample_ids'][0]='foreign'
    elif mutation=='command':scope['execution']['command'][-1]='--other'
    elif mutation=='authorization':auth['adaptive_validation_access']=False
    else:auth['test_access']='unfrozen'
    atomic_write_json(prep/'prep.json',dict(schema_version=gov.PREP_SCHEMA,scope=scope));atomic_write_json(prep/'authorization.json',auth)
    with pytest.raises(SearchContractError):gov.validate_prep(tmp_path,prep,authorized=True)


def test_persistence_failure_does_not_complete_an_opportunity(tmp_path):
    broker=broker_for(tmp_path);rows=fixture.toy_rows(1);artifacts={}
    broker.evaluate=lambda *args:{r['stable_example_id']:fixture.observation() for r in rows}
    def persist(name,value):
        if '_generation_' in name:raise OSError('Synthetic persistence failure')
        artifacts[name]=value
    try:
        with pytest.raises(OSError,match='persistence failure'):run_cell(fixture.cell(),rows,rows,broker,persist)
        assert set(artifacts)=={'checkpoint_0','round_1_opportunity'}
        assert broker.usage['reflection']==1 and not broker.ledger.inflight
    finally:broker.ledger.close()


def test_formal_test_alignment_uses_metadata_and_same_inference_contract():
    from multi_dataset_diverse_rl.research.gated_prompt_test import formal_test_plan
    reader=fixture.metadata_reader()
    plan=formal_test_plan(ROOT,reader)
    assert plan['count']==300 and plan['physical_call_ceiling']==10800
    assert plan['solver_decoding']==policy()['solver_decoding_policy']
    assert plan['first_capacity']==3600 and plan['length_recovery_capacity']==6144
    assert plan['solver_concurrency']==16 and plan['roles']==['solver']
    assert set(plan['sample_ids']).isdisjoint(gate_membership(reader)['sample_ids'])


def test_final_test_broker_forbids_training_roles_and_shares_identical_seed_requests(tmp_path):
    from multi_dataset_diverse_rl.research.gated_prompt_test import FinalTestBroker
    old=broker_for(tmp_path)
    broker=FinalTestBroker(phase='test',scope=dict(old.scope,call_ceiling=10800),transport=old.transport,
        ledger=old.ledger,receipts=old.receipts,dispatch_guard=lambda _:None)
    broker.seed=88;broker.cell_id=broker.attempt_id='synthetic_test'
    try:
        first=broker.evaluate(SEED,fixture.toy_rows(1),'test',88,20)
        broker.cell_id='another_arm'
        assert broker.evaluate(SEED,fixture.toy_rows(1),'test',88,20)==first and broker.usage['solver']==1
        for role,split in [('reflection','test'),('solver','optimize'),('solver','validation')]:
            with pytest.raises(SearchContractError,match='ROLE_SPLIT'):
                broker.complete(role=role,split=split,stage='forbidden',messages=[])
        assert broker.timeout_recovery['combined_physical_call_ceiling']==10800
    finally:broker.ledger.close()


def test_final_test_denied_before_reading_parent_or_opening_test(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.research.gated_prompt_test import execute_final_test
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    prep=tmp_path/'runs/prep';atomic_write_json(prep/'authorization.json',dict(test_access=False))
    monkeypatch.setattr(gov,'validate_prep',lambda *args,**kwargs:dict(scope_identity='a'*64))
    with pytest.raises(SearchContractError,match='TEST_AUTHORIZATION_REQUIRED'):
        execute_final_test(tmp_path,prep,tmp_path/'runs/train_run')
    assert not (tmp_path/'runs/test_run').exists()


def test_incomplete_training_never_allows_test_access(tmp_path):
    from multi_dataset_diverse_rl.research.gated_prompt_test import verify_training
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    atomic_write_json(tmp_path/'terminal.json',dict(status='EXECUTION_ABORTED',phase='train'))
    with pytest.raises(SearchContractError,match='COMPLETE_EXACT_TRAIN'):
        verify_training(tmp_path,dict(scope_identity='a'*64,source_sha='a'*40))
