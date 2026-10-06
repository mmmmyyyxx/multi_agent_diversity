"""Zero-API checks for minimal symmetric initialization and independent lanes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.benchmarks.current_math_dependencies import validate_current_initial_team
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.governance.provenance_receipts import verify_receipt_dependencies
from multi_dataset_diverse_rl.governance.unified_execution import (
    preexecution_manifest, bound_preflight, execution_scope, execution_identity)
from multi_dataset_diverse_rl.persistence.exact_output_cache import DurableExactOutputCache
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT=Path(__file__).resolve().parents[2]
PROFILE='experiments/execution_bindings/math_v2_1_gradient_pattern_offline_profile_v4.json'
PILOT_PROFILE='experiments/execution_bindings/math_v2_1_gradient_pattern_pilot_offline_profile_v3.json'


def contract(path=PROFILE):return json.loads((ROOT/path).read_bytes())
def team():return json.loads((ROOT/contract()['initial_team_path']).read_bytes())


def test_exact_minimal_team_identity():
    t=team();c=contract();validate_current_initial_team(t,c)
    assert len(t['members'])==5
    assert t['ordered_member_ids']==[m['member_id'] for m in t['members']]==[0,1,2,3,4]
    assert {m['prompt'].encode('utf-8') for m in t['members']}=={b'Solve the problem.'}
    hashes=[hashlib.sha256(m['prompt'].encode('utf-8')).hexdigest() for m in t['members']]
    assert set(hashes)=={'4e6dfb1595690c9487e7dea0deaafa72363cde09f5ce81186f1e4fbdc4205f80'}
    assert hashes==[m['prompt_sha256'] for m in t['members']]
    assert digest(hashes)==t['ordered_team_sha256']==c['initial_team_sha256']=='d1a04dbdd540371638e8271cca90469f0a8cf4d42a5f944eeee8c6dfcfca9110'
    assert t['team_version']==c['initial_team_version']=='MATH_GENERIC_TEAM_SEED_V1_2'
    assert t['initial_team_data_dependency']=='NONE' and t['semantic_contract']==['solve the problem']


@pytest.mark.parametrize('fault',[
    'asymmetry','all_nonminimal','duplicate_id','wrong_order','short_team','old_version',
    'data_hint','semantic_hint','prompt_hash','team_hash','binding_hash'])
def test_initial_contract_rejects_invalid_condition(fault):
    t=team();c=contract()
    if fault=='asymmetry':t['members'][4]['prompt']+=' Think carefully.'
    elif fault=='all_nonminimal':
        for m in t['members']:m['prompt']='Solve independently.'
    elif fault=='duplicate_id':t['members'][4]['member_id']=0
    elif fault=='wrong_order':t['ordered_member_ids'].reverse()
    elif fault=='short_team':t['members'].pop()
    elif fault=='old_version':t['team_version']=c['initial_team_version']='MATH_GENERIC_TEAM_SEED_V1_1'
    elif fault=='data_hint':t['initial_team_data_dependency']='OPTIMIZE'
    elif fault=='semantic_hint':t['semantic_contract'].append('careful reasoning')
    elif fault=='prompt_hash':t['members'][2]['prompt_sha256']='0'*64
    elif fault=='team_hash':t['ordered_team_sha256']='0'*64
    elif fault=='binding_hash':c['initial_team_sha256']='0'*64
    with pytest.raises(SearchContractError,match='CURRENT_DATA_INITIAL_'):validate_current_initial_team(t,c)


@pytest.mark.parametrize('path',[PROFILE,PILOT_PROFILE])
def test_amendment_changes_only_initial_condition_and_run_identity(path):
    c=contract(path);initial=json.loads((ROOT/c['numeric_parent_binding_path']).read_bytes())
    p=json.loads((ROOT/initial['initial_condition_parent_binding_path']).read_bytes())
    from multi_dataset_diverse_rl.benchmarks.initial_condition_contract import initial_condition_provenance
    from multi_dataset_diverse_rl.benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
    assert initial_condition_provenance(MATHGradientPatternBinding(ROOT,initial))[0]==p
    assert not execution_binding(ROOT,c).blockers()
    for field in ('models','decoding','provider_bounds','pattern_policy','memory_limits',
            'layer1_search_policy','solver_decoding_policy','membership_hashes','low_cost_protocol',
            'gradient_prompt_sha256','pattern_abstraction_guard','shared_risk_policy'):
        assert initial[field]==p[field]
    assert initial['execution_attempt_id']!=p['execution_attempt_id']
    assert initial['cache_namespace']!=p['cache_namespace']
    assert initial['initial_team_artifact_sha256']!=p['initial_team_artifact_sha256']
    m=preexecution_manifest(ROOT,source_sha=None,frozen=False,binding_path=path,
        experiment_id='math_identical_initial_condition_v1')
    status=bound_preflight(ROOT,m)
    assert status['gate']=='HOLD' and status['provider_attempts']==0
    assert status['blockers']==['PREEXECUTION_NOT_FROZEN']
    assert m['authorization']['real_api_authorized'] is False
    scope=execution_scope(m,c)
    assert scope['initial_memory_entries']==0
    assert scope['initial_condition']['initial_team_version']=='MATH_GENERIC_TEAM_SEED_V1_2'


@pytest.mark.parametrize('field,value',[
    ('initial_team_version','MATH_GENERIC_TEAM_SEED_V1_1'),
    ('initial_team_sha256','0'*64),('provider_bounds',{}),('cache_namespace','old'),
    ('gradient_prompt_sha256','0'*64),('initial_condition_amendment_sha256','0'*64)])
def test_amendment_rejects_identity_or_method_tampering(field,value):
    c=contract();c[field]=value
    assert execution_binding(ROOT,c).blockers()[0].startswith('CURRENT_NUMERIC_ADMISSIBILITY_')


def test_archive_preserves_old_bytes_without_authorizing_old_binding():
    c=contract();a=json.loads((ROOT/c['initial_condition_amendment_path']).read_bytes())
    archived=ROOT/a['archived_team_path']
    assert hashlib.sha256(archived.read_bytes()).hexdigest()==a['historical_team_artifact_sha256']
    t=json.loads(archived.read_bytes())
    assert t['team_version']=='MATH_GENERIC_TEAM_SEED_V1_1'
    assert len({m['prompt'] for m in t['members']})==5
    p=json.loads((ROOT/c['initial_condition_parent_binding_path']).read_bytes())
    with pytest.raises(SearchContractError,match='PROVENANCE_RECEIPT_HASH_MISMATCH'):
        verify_receipt_dependencies(ROOT,p)
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        execution_binding(ROOT,p)
    with pytest.raises(SearchContractError,match='PROVENANCE_RECEIPT_HASH_MISMATCH'):
        verify_receipt_dependencies(ROOT,p,historical_initial_team=(
            a['initial_team_path'],'0'*64,a['archived_team_path']))


def test_initial_amendment_and_archive_enter_execution_closure():
    c=contract();a=json.loads((ROOT/c['initial_condition_amendment_path']).read_bytes())
    paths={r['path'] for r in execution_identity(ROOT,c)['execution_closure']['files']}
    assert {c['initial_condition_amendment_path'],c['initial_condition_parent_binding_path'],
        a['archived_team_path'],c['initial_team_path']}<=paths


def messages():
    return [dict(role='system',content='Synthetic immutable interface.'),
        dict(role='user',content='Solve the problem.\n\nSynthetic arithmetic task.')]


def cache(tmp_path,c):
    from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import frozen_solver_policy
    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import frozen_recovery_policy
    context=dict(execution_attempt_id=c['execution_attempt_id'],cache_namespace=c['cache_namespace'],
        startup_identity_sha256='a'*64,source_sha='b'*40,authorization_sha256='c'*64,
        binding_sha256=digest(c),generation_policy_sha256=digest(frozen_solver_policy(c)),
        recovery_policy_sha256=digest(frozen_recovery_policy(c)))
    return DurableExactOutputCache(tmp_path,context)


def test_identical_wire_requests_have_five_independent_lanes_and_exact_reuse(tmp_path):
    c=contract();calls=[];ledger=[]
    def transport(request):
        calls.append(deepcopy(request))
        return dict(text='FINAL_ANSWER: '+str(len(calls)),input_tokens=2,output_tokens=2,finish_reason='stop')
    durable=cache(tmp_path,c)
    broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81,
        ledger_writer=ledger.append,durable_cache=durable)
    keys=[broker._request_identity(role='solver',split='optimize',messages=messages(),member_slot=i)[1]
        for i in range(5)]
    assert len(set(keys))==5
    def solve(b,i):return b.complete(role='solver',split='optimize',stage='initial',messages=messages(),member_slot=i)
    results=[solve(broker,i) for i in range(5)]
    assert [r['text'] for r in results]==['FINAL_ANSWER: '+str(i+1) for i in range(5)]
    assert len(calls)==5 and all(r==calls[0] for r in calls)
    assert [r['member_realization_lane'] for r in ledger if r['kind']=='ATTEMPT']==list(range(5))
    for i in range(5):
        repeated=solve(broker,i)
        assert repeated['text']==results[i]['text'] and repeated['provider_called'] is False
    def forbidden_transport(_):raise AssertionError('Exact durable hit must not dispatch.')
    resumed=RequestBroker(contract=c,transport=forbidden_transport,arm='A4',seed=81,durable_cache=cache(tmp_path,c))
    assert [solve(resumed,i)['text'] for i in range(5)]==[r['text'] for r in results]
    assert resumed.usage['attempts']==0 and len(calls)==5
    fresh=deepcopy(c);fresh['execution_attempt_id']+='_fresh';fresh['cache_namespace']+='_fresh'
    fresh_broker=RequestBroker(contract=fresh,transport=forbidden_transport,arm='A4',seed=81)
    fresh_keys=[fresh_broker._request_identity(role='solver',split='optimize',messages=messages(),member_slot=i)[1]
        for i in range(5)]
    assert not set(keys)&set(fresh_keys)
    with pytest.raises(SearchContractError,match='SCIENTIFIC_ATTEMPT_MISMATCH'):cache(tmp_path,fresh)
