"""Exact optimizer body, total-usage tolerance and independent truncation gates."""
from copy import deepcopy
import json
from pathlib import Path
import httpx
from openai import OpenAI
import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.benchmarks.math_optimizer_generation import optimizer_generation_contract
from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import generation_request_fields
from multi_dataset_diverse_rl.governance.autonomous_math import create_transport
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,OperationalAbort,reservation,serialized_request
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT=Path(__file__).resolve().parents[1]


def binding(new=True,policy_id=versions.MATH_OPTIMIZER_GENERATION_POLICY_V1_VERSION):
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_low_cost_canary_v1.json').read_bytes())
    if new:
        c.update(identity=versions.MATH_LOW_COST_OPTIMIZER_EXECUTION_BINDING_VERSION,
            optimizer_generation_policy=optimizer_generation_contract(policy_id),
            optimizer_amendment_authorization_sha256='a'*64)
    return c


def broker(transport,policy_id=versions.MATH_OPTIMIZER_GENERATION_POLICY_V1_VERSION,**kwargs):
    return RequestBroker(contract=binding(policy_id=policy_id),transport=transport,arm='A2',seed=81,**kwargs)


@pytest.mark.parametrize('policy_id,thinking',[
    (versions.MATH_OPTIMIZER_GENERATION_POLICY_V1_VERSION,True),
    (versions.MATH_OPTIMIZER_GENERATION_POLICY_VERSION,False),
])
@pytest.mark.parametrize('role',['reflection','pattern'])
def test_provider_visible_total_cap_and_explicit_thinking(role,policy_id,thinking):
    captured=[]
    b=broker(lambda r:(captured.append(r) or dict(text='Synthetic proposal',input_tokens=10,output_tokens=20,finish_reason='stop')),policy_id=policy_id)
    b.complete(role=role,split='optimize',stage='synthetic',messages=[{'role':'user','content':'Fixture'}])
    body=json.loads(serialized_request(captured[0]))
    assert body['enable_thinking'] is thinking and body['max_completion_tokens']==1800
    assert body['temperature']==0.0 and 'max_tokens' not in body
    assert 'top_k' not in body and 'min_p' not in body
    bound=reservation(captured[0])
    assert bound['output_hard_cap']==1810 and bound['requested_output_cap']==1800
    assert bound['generation_policy_identity']==policy_id
    assert bound['amount']==len(serialized_request(captured[0]))+4096+1810


def test_solver_bytes_and_legacy_optimizer_wire_unchanged():
    assert generation_request_fields(binding(),'solver')==generation_request_fields(binding(False),'solver')
    legacy=generation_request_fields(binding(False),'reflection')
    assert legacy==dict(temperature=0.0,max_tokens=1800,extra_body={})
    request=dict(model='qwen3.7-flash',messages=[],**legacy)
    assert reservation(request)['output_hard_cap']==1800


@pytest.mark.parametrize('output,finish,expected',[
    (1807,'stop',None),(1810,'stop',None),
    (1811,'stop','OPERATIONAL_OUTPUT_CAP_NOT_ENFORCED'),
    (1807,'length','OPERATIONAL_OUTPUT_TRUNCATION'),
])
@pytest.mark.parametrize('policy_id',[versions.MATH_OPTIMIZER_GENERATION_POLICY_V1_VERSION,versions.MATH_OPTIMIZER_GENERATION_POLICY_VERSION])
def test_actual_usage_tolerance_does_not_relax_truncation(tmp_path,output,finish,expected,policy_id):
    ledger=TokenLedger(tmp_path,task_sha256='a'*64)
    captured=[]
    def transport(request):
        captured.append(request)
        return dict(text='Synthetic proposal',input_tokens=100,output_tokens=output,finish_reason=finish)
    b=broker(transport,policy_id=policy_id,token_ledger=ledger)
    try:
        args=dict(role='reflection',split='optimize',stage='reflection',messages=[{'role':'user','content':'Fixture'}])
        if expected:
            with pytest.raises(OperationalAbort,match=expected):b.complete(**args)
        else:
            assert b.complete(**args)['output_tokens']==output
        assert ledger.reserved==0 and captured[0]['max_completion_tokens']==1800
        if output<=1810:
            assert ledger.view()['provider_reported_actual']==100+output
            assert ledger.view()['fallback_charged']==0
        else:
            assert ledger.view()['fallback_charged']==reservation(captured[0])['amount']
    finally:ledger.close()


def test_ambiguous_caps_and_wrong_total_contract_fail_before_reservation():
    r=dict(model='qwen3.7-flash',max_completion_tokens=1800,extra_body={'enable_thinking':True},messages=[])
    with pytest.raises(OperationalAbort,match='AMBIGUOUS'):reservation(dict(r,max_tokens=1800))
    with pytest.raises(OperationalAbort,match='FROZEN_OPTIMIZER'):reservation(dict(r,max_completion_tokens=1810))
    with pytest.raises(OperationalAbort,match='FROZEN_OPTIMIZER'):reservation(dict(r,extra_body={}))


@pytest.mark.parametrize('field,value',[('enable_thinking',False),('max_completion_tokens',1810),('measurement_tolerance_tokens',11)])
def test_policy_drift_fails_before_transport(field,value):
    c=binding();c['optimizer_generation_policy'][field]=value
    with pytest.raises(SearchContractError,match='OPTIMIZER_GENERATION_POLICY'):
        RequestBroker(contract=c,transport=lambda r:pytest.fail('No dispatch on drift'),arm='A1',seed=81)


def test_optimizer_policy_is_explicit_cache_identity():
    b=broker(lambda r:None);old=RequestBroker(contract=binding(False),transport=lambda r:None,arm='A1',seed=81)
    args=dict(role='reflection',split='optimize',messages=[])
    assert b._request_identity(**args)[1]!=old._request_identity(**args)[1]
    forbidden=binding(False);forbidden['optimizer_generation_policy']=optimizer_generation_contract()
    with pytest.raises(SearchContractError,match='REQUIRES_FRESH_BINDING'):
        RequestBroker(contract=forbidden,transport=lambda r:None,arm='A1',seed=81)


@pytest.mark.parametrize('policy_id,thinking',[
    (versions.MATH_OPTIMIZER_GENERATION_POLICY_V1_VERSION,True),
    (versions.MATH_OPTIMIZER_GENERATION_POLICY_VERSION,False),
])
def test_exact_sdk_transport_retains_usage_categories(monkeypatch,policy_id,thinking):
    captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'synthetic','choices':[{'finish_reason':'stop',
            'message':{'content':'Synthetic proposal','reasoning_content':'Synthetic thought'}}],
            'usage':{'prompt_tokens':100,'completion_tokens':1807,
                'completion_tokens_details':{'reasoning_tokens':1600}}})
    client=OpenAI(api_key='synthetic-offline',base_url='https://example.invalid/v1',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),max_retries=0)
    monkeypatch.setattr('multi_dataset_diverse_rl.provider_factory.ProviderClientFactory.from_environment',lambda **kwargs:client)
    c=binding(policy_id=policy_id);transport,_=create_transport(c)
    try:
        result=transport(dict(model='qwen3.7-flash',messages=[],**generation_request_fields(c,'reflection')))
        assert captured[0]['max_completion_tokens']==1800 and captured[0]['enable_thinking'] is thinking
        assert 'max_tokens' not in captured[0]
        assert result['output_tokens']==1807
        assert result['provider_usage_details']['completion_tokens_details']['reasoning_tokens']==1600
        assert result['provider_reasoning_character_count']==len('Synthetic thought')
    finally:client.close()


def fresh_binding(phase='canary'):
    return json.loads((ROOT/f'experiments/execution_bindings/math_v2_1_low_cost_{phase}_v2.json').read_bytes())


@pytest.mark.parametrize('phase',['canary','pilot'])
def test_complete_policy_enters_manifest_scope_and_binding(phase):
    from multi_dataset_diverse_rl.benchmarks.math_low_cost_binding import MATHLowCostBinding
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,execution_scope
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    c=fresh_binding(phase);assert not MATHLowCostBinding(ROOT,c).blockers()
    m=preexecution_manifest(ROOT,source_sha='a'*40,binding_path=c['binding_path'],frozen=False)
    assert not validate_manifest_v2(ROOT,m)
    scope=execution_scope(m,c)
    assert m['optimizer_generation_policy']==scope['optimizer_generation_policy']==optimizer_generation_contract(versions.MATH_OPTIMIZER_GENERATION_POLICY_V1_VERSION)
    assert scope['optimizer_amendment_authorization_sha256']==c['optimizer_amendment_authorization_sha256']
    assert scope['accounting']['total_authorization']==40000000
    assert c['post_search_validation_policy']['count']==100
    drift=deepcopy(c);drift['optimizer_generation_policy']['enable_thinking']=False
    assert MATHLowCostBinding(ROOT,drift).blockers()


@pytest.mark.parametrize('arm',['A1','A2','A3','A4'])
def test_fresh_policy_actual_pinned_gepa_four_arm(tmp_path,monkeypatch,arm):
    from tests import test_math_low_cost as original
    monkeypatch.setattr(original,'contract',fresh_binding)
    original.test_low_cost_actual_gepa_four_arm_e2e(tmp_path,arm,monkeypatch)


@pytest.mark.parametrize('phase',['canary','pilot'])
def test_fresh_frozen_entrypoint_and_paired_validation(tmp_path,monkeypatch,phase):
    from tests import test_math_low_cost as original
    monkeypatch.setattr(original,'contract',fresh_binding)
    original.test_frozen_entrypoint_recovery_durable_floor_and_receipt(tmp_path,monkeypatch,phase)


def test_default_is_nonthinking_and_immutable_v1_still_dispatches():
    default=optimizer_generation_contract()
    assert default['identity']==versions.MATH_OPTIMIZER_GENERATION_POLICY_VERSION
    assert default['enable_thinking'] is False
    old=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_low_cost_canary_v2.json').read_bytes())
    assert generation_request_fields(old,'reflection')['extra_body']=={'enable_thinking':True}
    current=binding(policy_id=default['identity'])
    assert generation_request_fields(current,'reflection')['extra_body']=={'enable_thinking':False}
    assert generation_request_fields(current,'solver')==generation_request_fields(old,'solver')
    args=dict(role='reflection',split='optimize',messages=[])
    before=RequestBroker(contract=old,transport=lambda _:None,arm='A1',seed=81)
    after=RequestBroker(contract={**old,'optimizer_generation_policy':default},transport=lambda _:None,arm='A1',seed=81)
    assert before._request_identity(**args)[1]!=after._request_identity(**args)[1]
    drift={**current,'optimizer_generation_policy':{**default,'enable_thinking':True}}
    with pytest.raises(SearchContractError,match='OPTIMIZER_GENERATION_POLICY'):
        RequestBroker(contract=drift,transport=lambda _:pytest.fail('Drift must not dispatch'),arm='A1',seed=81)


def default_binding_fixture(tmp_path,phase):
    from multi_dataset_diverse_rl.benchmarks.math_low_cost import read_subsets
    from multi_dataset_diverse_rl.benchmarks.math_low_cost_binding import derive_contract
    from multi_dataset_diverse_rl.benchmarks.data_freeze import file_hash
    c=fresh_binding(phase);parent=json.loads((ROOT/c['amendment_parent_binding_path']).read_bytes())
    tmp_path.mkdir(parents=True,exist_ok=True)
    approval=tmp_path/'synthetic_approval.json'
    approval.write_text(json.dumps(dict(explicit_human_approval=True,policy=optimizer_generation_contract(),
        total_accounting_authorization=40000000,pattern_real_calls_A1=0,test_model_calls=0)),encoding='utf-8')
    destination=tmp_path/'synthetic_binding.json'
    kwargs=dict(phase=phase,attempt=f'math_v2_1_low_cost_A1_seed81_{phase}_attempt_synthetic_default',
        binding_path=destination.relative_to(ROOT).as_posix(),subsets_path=c['low_cost_subsets_path'],subsets_sha256=c['low_cost_subsets_sha256'],
        subsets=read_subsets(ROOT,c),parent_path=c['amendment_parent_binding_path'],parent_sha256=c['amendment_parent_binding_sha256'],
        accounting_path=c['accounting_policy_path'],accounting_sha256=c['accounting_policy_sha256'],metadata_path=c['validation_accounting_metadata_path'],
        metadata_sha256=c['validation_accounting_metadata_sha256'],authorization_sha256=c['continuation_authorization_sha256'])
    with pytest.raises(SearchContractError,match='OPTIMIZER_AMENDMENT_AUTHORITY_REQUIRED'):
        derive_contract(parent,**kwargs)
    current=derive_contract(parent,**kwargs,optimizer_authorization_path=approval.relative_to(ROOT).as_posix(),
        optimizer_authorization_sha256=file_hash(approval))
    destination.write_text(json.dumps(current),encoding='utf-8')
    return current


@pytest.mark.parametrize('phase',['canary','pilot'])
def test_default_off_binding_has_authority_and_full_identity(tmp_path,phase):
    from multi_dataset_diverse_rl.benchmarks.math_low_cost_binding import MATHLowCostBinding
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,execution_scope
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    c=default_binding_fixture(tmp_path,phase)
    assert c['optimizer_generation_policy']==optimizer_generation_contract()
    assert not MATHLowCostBinding(ROOT,c).blockers()
    manifest=preexecution_manifest(ROOT,source_sha='a'*40,binding_path=c['binding_path'],frozen=False)
    assert not validate_manifest_v2(ROOT,manifest)
    scope=execution_scope(manifest,c)
    assert manifest['optimizer_generation_policy']==scope['optimizer_generation_policy']==optimizer_generation_contract()


def test_default_off_frozen_entrypoint_keeps_solver_and_heldout_rules(tmp_path,monkeypatch):
    from tests import test_math_low_cost as original
    c=default_binding_fixture(tmp_path/'default_contract','canary')
    monkeypatch.setattr(original,'contract',lambda phase='canary':c)
    original.test_frozen_entrypoint_recovery_durable_floor_and_receipt(tmp_path/'execution',monkeypatch,'canary')
