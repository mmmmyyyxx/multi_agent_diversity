"""Role-isolated output repair; no scientific selection, retry or schema change."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json
import pytest
from multi_dataset_diverse_rl.benchmarks.math_optimizer_generation import pattern_cluster_generation_contract,frozen_cluster_policy
from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import generation_request_fields
from multi_dataset_diverse_rl.governance.token_accounting import reservation,OperationalAbort
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.schemas import SearchContractError
ROOT=Path(__file__).resolve().parents[2]
PROFILE='experiments/execution_bindings/math_v2_2_gradient_pattern_seed81_pilot_v2.json'
OLD='experiments/execution_bindings/math_v2_2_gradient_pattern_seed81_pilot_v1.json'
def contract():return json.loads((ROOT/PROFILE).read_bytes())

@pytest.mark.parametrize('role,cap',[('reflection',1800),('pattern_gradient',1800),('pattern_cluster',8192)])
def test_only_cluster_wire_changes(role,cap):
    c=contract();old=json.loads((ROOT/OLD).read_bytes())
    after=generation_request_fields(c,role);before=generation_request_fields(old,role)
    assert after['max_completion_tokens']==cap and 'max_tokens' not in after
    before['max_completion_tokens']=cap
    assert after==before
    if role=='pattern_cluster':
        req=dict(model='qwen3.7-flash',messages=[dict(role='user',content='Synthetic partition.')],**after)
        r=reservation(req)
        assert r['output_hard_cap']==8202 and r['requested_output_cap']==8192
        assert r['amount']==r['serialized_request_bytes']+4096+8202
        assert r['generation_policy_identity']==pattern_cluster_generation_contract()['identity']

@pytest.mark.parametrize('field,value',[('max_completion_tokens',8191),('max_completion_tokens',True),
    ('accounting_output_ceiling',8192),('roles',['reflection']),('enable_thinking',True)])
def test_role_policy_tampering_fails_closed(field,value):
    c=contract();c['pattern_cluster_generation_policy'][field]=value
    with pytest.raises(SearchContractError):frozen_cluster_policy(c)

def test_historical_binding_cannot_inherit_new_output_policy():
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v6.json').read_bytes())
    c['pattern_cluster_generation_policy']=pattern_cluster_generation_contract()
    with pytest.raises(SearchContractError):frozen_cluster_policy(c)

@pytest.mark.parametrize('cap',[8191,16384,True])
def test_unfrozen_output_capacity_cannot_enter_reservation(cap):
    req=dict(model='qwen3.7-flash',messages=[],**generation_request_fields(contract(),'pattern_cluster'))
    req['max_completion_tokens']=cap
    with pytest.raises(OperationalAbort):reservation(req)

@pytest.mark.parametrize('tokens,finish,accepted',[(2200,'stop',True),(8202,'stop',True),
    (8203,'stop',False),(100,'length',False),(100,'max_tokens',False),(100,'max_output_tokens',False)])
def test_response_measurement_and_truncation_use_actual_role(tokens,finish,accepted):
    calls=[]
    def transport(req):
        calls.append(req)
        return dict(text='{}',input_tokens=2,output_tokens=tokens,finish_reason=finish,
            provider_response_accepted=True,provider_metadata_loss_audited=True,
            provider_reasoning_content_present=False,provider_reasoning_character_count=None,
            provider_usage_details={},provider_thinking_indicators=[])
    broker=RequestBroker(contract=contract(),transport=transport,arm='A4',seed=81)
    kwargs=dict(role='pattern_cluster',split='optimize',stage='synthetic',messages=[dict(role='user',content='Synthetic partition.')])
    if accepted:
        result=broker.complete(**kwargs)
        assert result['optimizer_generation_diagnostics']['max_completion_tokens']==8192
        assert result['optimizer_generation_diagnostics']['generation_policy_identity']==pattern_cluster_generation_contract()['identity']
    else:
        with pytest.raises(SearchContractError,match='OPERATIONAL_OUTPUT'):broker.complete(**kwargs)
    assert len(calls)==1 and broker.usage['pattern_cluster']==1

def test_operational_admission_reserves_entire_new_role_before_transport():
    c=contract();request=dict(model='qwen3.7-flash',messages=[],**generation_request_fields(c,'pattern_cluster'))
    c['operational_pilot']['token_ceiling']=reservation(request)['amount']-1
    calls=[];b=RequestBroker(contract=c,transport=lambda req:calls.append(req),arm='A4',seed=81)
    with pytest.raises(OperationalAbort,match='TOKEN_CEILING'):
        b.complete(role='pattern_cluster',split='optimize',stage='synthetic',messages=[])
    assert not calls and b.usage['attempts']==0

def test_fresh_binding_and_method_identity_record_cluster_policy():
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,execution_scope
    b=execution_binding(ROOT,contract());assert not b.blockers()
    m=b.method('A4');assert m.mechanism_config['pattern_cluster_generation_policy']==pattern_cluster_generation_contract()
    manifest=preexecution_manifest(ROOT,source_sha='0'*40,frozen=False,binding_path=PROFILE,experiment_id='synthetic')
    assert not validate_manifest_v2(ROOT,manifest)
    missing=deepcopy(manifest);missing.pop('pattern_cluster_generation_policy')
    assert validate_manifest_v2(ROOT,missing)
    assert execution_scope(manifest,contract())['pattern_cluster_generation_policy']==pattern_cluster_generation_contract()
    bad=deepcopy(contract());bad['pattern_cluster_generation_policy']['max_completion_tokens']=1800
    assert execution_binding(ROOT,bad).blockers()

@pytest.mark.parametrize('k,changed,completed',[(1,True,False),(64,False,True)])
def test_real_current_graph_with_new_role_synthetic_commit_and_scientific_stop(tmp_path,monkeypatch,k,changed,completed):
    spec=importlib.util.spec_from_file_location('synthetic_v22_regression',ROOT/'tests/current/test_v22_operational_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'contract',contract)
    module.test_current_runner_scientific_and_operational_receipts(tmp_path,monkeypatch,k,changed,completed)
