"""Canary uses native sealed training and refuses Test and unapproved dispatch."""
from copy import deepcopy
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.research.ablations import proposal_canary as canary
from multi_dataset_diverse_rl.research.ablations import governance as gov,proposal_governance as pg
from multi_dataset_diverse_rl.research.ablations.proposal_config import default_config,build_plan
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport,synthetic_rows
from multi_dataset_diverse_rl.persistence.durable_io import read_json,atomic_write_json
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest,file_hash
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.system_prompt import SEED

ROOT=Path(__file__).resolve().parents[2]


def test_canary_exact_budget_native_data_and_formal_defaults_unchanged():
    manifest=read_json(ROOT/canary.MANIFEST);plan=canary.validate_manifest(manifest)
    assert len(plan['cells'])==3 and all(c['seed']==42 and c['repeat_id']==1 for c in plan['cells'])
    assert plan['config']['rounds']==1 and plan['config']['runtime']['validation_membership_seed']==42
    assert plan['training_semantic_calls_by_arm']=={'C0':2457,'C1':3461,'C2':3461}
    assert manifest['successful_semantic_call_ceiling']==9379
    assert manifest['cumulative_accounted_token_ceiling']==3_000_000
    assert manifest['test_call_ceiling']==0 and manifest['test_access'] is False
    assert manifest['optimize_count']==150 and manifest['adaptive_validation_count']==100
    formal=build_plan(default_config())
    assert len(formal['cells'])==9 and formal['config']['rounds']==20
    assert formal['training_semantic_call_ceiling']==391740


@pytest.mark.parametrize('field,value',[
    ('test_access',True),('test_call_ceiling',3600),('successful_semantic_call_ceiling',9380),
    ('cumulative_accounted_token_ceiling',None),('real_api_authorized',True),('seed',88)])
def test_manifest_tamper_rejected(field,value):
    bad=canary.manifest_template();bad[field]=value
    with pytest.raises(SearchContractError,match='CANARY_MANIFEST_CHANGED'):canary.validate_manifest(bad)


def test_scope_handoff_and_main_entry_bind_actual_canary_limits(monkeypatch):
    destination=ROOT/'runs/ablations/fake_canary_scope/prep'
    formal=dict(scope_identity=digest('old'),study_id='formal',source_hashes={},
        plan=build_plan(default_config()),token_ceiling=None,call_ceiling=391740,
        execution=dict(run_root='runs/ablations/fake_canary_scope/train_run'),
        test_plan=dict(sample_ids=['metadata_only_test_id'],successful_semantic_call_ceiling=10800))
    monkeypatch.setattr(pg,'scope_template',lambda *a:deepcopy(formal))
    scope=canary.scope_template(ROOT,destination)
    assert scope['study_id']==canary.STUDY and len(scope['plan']['cells'])==3
    assert scope['scope_identity']==digest({k:v for k,v in scope.items() if k!='scope_identity'})
    assert scope['call_ceiling']==9379 and scope['token_ceiling']==3_000_000
    assert scope['test_plan']['sample_ids']==[] and scope['test_plan']['successful_semantic_call_ceiling']==0
    assert scope['source_hashes'][canary.MANIFEST]==file_hash(ROOT/canary.MANIFEST)
    monkeypatch.setattr(pg,'handoff',lambda *a:dict(protocol={},execution={},authorization={}))
    handoff=canary.handoff(ROOT,destination,scope)
    assert handoff['execution']['cumulative_token_ceiling']==3_000_000
    assert len(handoff['execution']['expected_checkpoint_paths'])==6
    assert handoff['authorization']['test_access_policy']=='prohibited_for_entire_canary'
    from scripts.run_experiment import preflight
    status=preflight(canary.manifest_template())
    assert status['cells']==3 and status['real_api_calls']==0 and status['test_access'] is False
    assert status['semantic_call_ceiling']==9379 and status['token_ceiling']==3_000_000


@pytest.mark.parametrize('authorized,test_grant',[(False,False),(False,True),(True,False),(True,True)])
def test_exact_authorization_and_test_deny_before_transport(tmp_path,monkeypatch,authorized,test_grant):
    scope=dict(scope_identity=digest('canary'),phase='train',plan=dict(cells=[dict(cell_id='C0_seed42_repeat01')]))
    folder=tmp_path/'runs/canary/prep';handoff=dict(READY_TO_RUN=False,status='WAIT',protocol=dict(identity='pinned'))
    plan=dict(identity='exact command')
    monkeypatch.setattr(gov,'require_selected_runtime',lambda root:None)
    monkeypatch.setattr(gov,'_git',lambda *a:'')
    monkeypatch.setattr(canary,'scope_template',lambda *a:deepcopy(scope))
    monkeypatch.setattr(canary,'handoff',lambda *a:deepcopy(handoff))
    monkeypatch.setattr(gov,'execution_plan_template',lambda *a:deepcopy(plan))
    def forbidden():raise AssertionError('No provider construction before grant')
    monkeypatch.setattr(gov,'_transport',forbidden)
    auth=gov.authorization_template(scope);auth['test_access']=test_grant
    for name,value in [('prep',dict(schema_version=canary.PREP_SCHEMA,scope=scope)),
        ('authorization',auth),('handoff',handoff),('execution_plan',plan)]:
        atomic_write_json(folder/(name+'.json'),value)
    if not authorized and not test_grant:assert gov.validate_prep(tmp_path,folder)==scope
    else:
        with pytest.raises(SearchContractError):gov.validate_prep(tmp_path,folder,authorized=authorized)


def test_ready_requires_exact_human_evidence_and_no_test(tmp_path,monkeypatch):
    scope=dict(scope_identity=digest('exact'),phase='train',plan=dict(cells=[dict(cell_id='C0_seed42_repeat01')]))
    folder=tmp_path/'runs/canary/prep';handoff=dict(READY_TO_RUN=False,status='WAIT')
    monkeypatch.setattr(gov,'require_selected_runtime',lambda root:None)
    monkeypatch.setattr(gov,'_git',lambda *a:'')
    monkeypatch.setattr(canary,'scope_template',lambda *a:deepcopy(scope))
    monkeypatch.setattr(canary,'handoff',lambda *a:deepcopy(handoff))
    monkeypatch.setattr(gov,'execution_plan_template',lambda *a:dict(command='pinned'))
    proof=folder/'user_authorization_evidence.json';atomic_write_json(proof,dict(fake_human_grant=True))
    auth=gov.authorization_template(scope)
    auth.update(approved=True,adaptive_validation_access=True,user_authorization_evidence_sha256=file_hash(proof))
    for name,value in [('prep',dict(schema_version=canary.PREP_SCHEMA,scope=scope)),
        ('authorization',auth),('handoff',handoff),('execution_plan',dict(command='pinned'))]:
        atomic_write_json(folder/(name+'.json'),value)
    with pytest.raises(SearchContractError):canary.validate_prep(tmp_path,folder,authorized=True)
    atomic_write_json(folder/'handoff.json',dict(handoff,READY_TO_RUN=True))
    assert canary.validate_prep(tmp_path,folder,authorized=True)==scope
    atomic_write_json(proof,dict(fake_human_grant='changed'))
    with pytest.raises(SearchContractError):canary.validate_prep(tmp_path,folder,authorized=True)


def test_prepare_persists_before_handoff_and_preserves_token_ceiling(tmp_path,monkeypatch):
    scope=dict(source_sha='0'*40,source_hashes={},scope_identity=digest('freeze'),call_ceiling=9379,
        token_ceiling=canary.TOKEN_CEILING,test_plan=dict(successful_semantic_call_ceiling=0),
        execution=dict(command=['frozen']),plan=dict(cells=[dict(cell_id='C0_seed42_repeat01')]))
    monkeypatch.setattr(gov,'require_selected_runtime',lambda root:None)
    monkeypatch.setattr(gov,'_git',lambda *a:'')
    monkeypatch.setattr(canary,'scope_template',lambda *a:deepcopy(scope))
    def handoff(root,folder,actual):
        assert actual==scope
        assert all((folder/(name+'.json')).exists() for name in ('prep','authorization','execution_plan'))
        return dict(prep_sha256=file_hash(folder/'prep.json'),READY_TO_RUN=False)
    monkeypatch.setattr(canary,'handoff',handoff)
    folder=tmp_path/'runs/fresh/prep';status=canary.prepare(tmp_path,folder)
    assert status['semantic_call_ceiling']==9379 and status['token_ceiling']==3_000_000
    assert not status['ready_to_run'] and status['real_api_calls']==0
    assert read_json(folder/'execution_plan.json')['token_ceiling']==3_000_000
    assert gov.execution_plan_template(dict(scope,token_ceiling=None))['token_ceiling'] is None
    with pytest.raises(SearchContractError):canary.prepare(tmp_path,folder)


def test_governed_fake_canary_complete_seals_without_test(tmp_path,monkeypatch):
    config=canary.manifest_template()['config'];plan=build_plan(config)
    optimize=synthetic_rows('optimize',150);pool=synthetic_rows('validation',300)
    prep=tmp_path/'runs/canary/prep';train=prep.parent/'train_run'
    class Reader:
        manifest_sha256=digest('fake split')
        members=[dict(stable_example_id=r['stable_example_id'],project_split='validation') for r in pool]
        def rows(self,split,purpose):assert split=='optimize';return optimize
        def _read_rows(self,split,frozen_subset_ids):assert split=='validation';return tuple(r for r in pool if r['stable_example_id'] in frozen_subset_ids)
        def heldout_rows(self,*a,**k):raise AssertionError('Canary must never open Test')
    reader=Reader();manifest=dict(initial_prompt_path='initial.json')
    atomic_write_json(tmp_path/'initial.json',dict(members=[dict(prompt=SEED.to_dict())]))
    for name in ('prep','handoff','execution_plan'):atomic_write_json(prep/(name+'.json'),dict(fake=True))
    scope=dict(phase='train',source_sha='0'*40,source_hashes={},plan=plan,gate_membership=gov.gate_membership(reader,42),
        initial_prompt_sha256=SEED.prompt_hash,optimize_sample_ids=[r['stable_example_id'] for r in optimize],
        test_plan=dict(identity='CANARY_TEST_PROHIBITED_V1',sample_ids=[],successful_semantic_call_ceiling=0),
        execution=dict(run_root='runs/canary/train_run'),call_ceiling=9379,token_ceiling=canary.TOKEN_CEILING,
        network_recovery=config['runtime']['timeout_recovery'],generation_recovery=config['runtime']['diagnosis_recovery'])
    scope['scope_identity']=digest(scope);auth=gov.authorization_template(scope)
    auth.update(approved=True,adaptive_validation_access=True,test_access=False,user_authorization_evidence_sha256=digest('fake grant'))
    atomic_write_json(prep/'authorization.json',auth)
    monkeypatch.setattr(canary,'validate_prep',lambda *a,**k:scope)
    monkeypatch.setattr(gov,'validate_prep',lambda *a,**k:scope)
    monkeypatch.setattr(pg,'inputs',lambda root:(manifest,reader))
    transport=SyntheticTransport()
    class Clients:
        def close(self):pass
    monkeypatch.setattr(gov,'_transport',lambda:(transport,Clients()))
    outcome=canary.execute(tmp_path,prep,train)
    summary=read_json(prep.parent/'canary_summary.json')
    assert outcome['canary_status']=='CANARY_EXECUTION_COMPLETE'
    assert not (prep.parent/'test_run').exists()
    assert summary['test_access'] is False and summary['formal_data_import_allowed'] is False
    assert summary['coverage']['two_direction_generation_exercised'] is False
    assert [row['attempted_proposals'] for row in summary['coverage']['rows']]==[1,1,1]
    assert read_json(train/'execution_summary.json')['usage']['reflection']==3
    assert all(role in {'solver','reflection','pattern_gradient','pattern_cluster'} for role,_ in transport.calls)
    assert gov.verify_training(train,scope)[1]==summary['terminal_sha256']


def test_token_reservation_cap_refuses_dispatch_and_leaves_no_inflight(tmp_path):
    from multi_dataset_diverse_rl.research.ablations.offline_runtime import OfflineAblationBroker
    from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_GATED_32_UNCAPPED
    from multi_dataset_diverse_rl.research.single_prompt_runtime import OperationalAbort
    plan=canary.validate_manifest(canary.manifest_template());sid=digest('bounded token test')
    ledger=TokenLedger(tmp_path/'ledger',task_sha256=sid,policy=POLICY_GATED_32_UNCAPPED)
    transport=SyntheticTransport()
    try:
        broker=OfflineAblationBroker(plan=plan,phase='train',scope=dict(scope_identity=sid,call_ceiling=9379,
            token_ceiling=1,network_recovery=True),transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(tmp_path/'receipts',attempt_id='fake',startup_identity_sha256=sid),
            dispatch_guard=lambda _:None)
        broker.bind_cell(plan['cells'][0],plan)
        broker.example_context.identity='offline_fixture_0'
        with pytest.raises(OperationalAbort,match='STUDY_TOKEN_CEILING'):
            broker.complete(role='solver',split='optimize',stage='canary_cap',messages=[
                dict(role='system',content=SEED.render()),dict(role='user',content='offline_fixture:0')])
        assert transport.calls==[] and not ledger.inflight and broker.usage_view()['provider_calls']==0
    finally:ledger.close()
