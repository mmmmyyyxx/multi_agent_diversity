"""Closed execution identity, fresh budget and durability negative controls."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from tests.current.test_optimization_evidence_v23 import contract,response
from multi_dataset_diverse_rl.search.schemas import SearchContractError,SearchMethodConfig
from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.optimization_evidence import frozen_policy
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.benchmarks.math_v21_interface import interface_for_contract,MATH_SOLVER_INTERFACE_V6,solver_user_content
from multi_dataset_diverse_rl.governance.current_dependencies import current_dependency_graph
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_V23_2M,POLICY_40M,OperationalAbort,reservation
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts

ROOT=Path(__file__).resolve().parents[2]
FRESH='experiments/execution_bindings/a4_v23_only_seed81_20261008_attempt1.json'

@pytest.mark.parametrize('identity',['unified_team_prompt_search_v2_1','unified_team_prompt_search_v2_2','unknown',None])
def test_historical_or_missing_method_rejected_before_transport(identity):
    c=contract();c['method_identity']=identity;calls=[]
    with pytest.raises(SearchContractError):RequestBroker(contract=c,transport=lambda r:calls.append(r),arm='A4',seed=81)
    assert not calls

@pytest.mark.parametrize('change',['policy','trajectory','paired','arm','seed','heldout'])
def test_missing_or_legacy_current_inputs_fail_closed(change):
    c=contract();args=dict(arm='A4',seed=81)
    if change=='policy':c.pop('optimization_evidence_policy')
    if change=='trajectory':c.pop('solver_trajectory_policy')
    if change=='paired':c['paired_realization_policy']={'identity':'historical'}
    if change=='arm':args['arm']='A1'
    if change=='seed':args['seed']=82
    if change=='heldout':args['validation_only']=True
    with pytest.raises(SearchContractError):RequestBroker(contract=c,transport=lambda _:None,**args)

def test_null_constructors_cannot_choose_a_scientific_method():
    with pytest.raises(SearchContractError):SearchMethodConfig()
    with pytest.raises(SearchContractError):frozen_policy(None)
    with pytest.raises(SearchContractError):CURRENT_POLICY_BUNDLE.method(aggregation='x',provider_binding='0'*64,successful_provider_calls=1)
    for path in ('math_v2_2_gradient_pattern_seed81_pilot_v2.json','a4_v23_matched_seed81_20261008_attempt1_a.json'):
        with pytest.raises(SearchContractError):execution_binding(ROOT,json.loads((ROOT/'experiments/execution_bindings'/path).read_bytes()))
    b=execution_binding(ROOT,json.loads((ROOT/'experiments/execution_bindings/a4_v23_matched_seed81_20261008_attempt1_b.json').read_bytes()))
    assert b.blockers()  # A historical V23 paired receipt cannot authorize the new path.

def test_current_cli_historical_manifest_rejects_before_binding_or_credentials(monkeypatch):
    import yaml
    from multi_dataset_diverse_rl.governance import unified_execution as gov
    from scripts.run_experiment import preflight
    manifest=yaml.safe_load((ROOT/'experiments/manifests/a4_v23_matched_comparison_v1_a.yaml').read_bytes())
    monkeypatch.setattr(gov,'read_json',lambda _:(_ for _ in ()).throw(AssertionError('historical binding read')))
    result=preflight(manifest)
    assert result['gate']=='HOLD' and result['provider_attempts']==0
    assert 'CURRENT_V23_EXECUTION_BINDING_NOT_FROZEN' in result['blockers']

def test_invalid_prep_cannot_reach_transport(tmp_path,monkeypatch):
    import asyncio
    from scripts.run_experiment import execute_frozen
    from multi_dataset_diverse_rl.governance import autonomous_math
    prep=tmp_path/'prep';prep.mkdir()
    (prep/'prep.json').write_text(json.dumps(dict(schema_version='unified_canary_prep_v1',startup_identity_sha256='a'*64)))
    calls=[]
    monkeypatch.setattr(autonomous_math,'create_transport',lambda _:calls.append('transport'))
    with pytest.raises(SearchContractError,match='STARTUP_IDENTITY_MISMATCH'):
        asyncio.run(execute_frozen(prep,tmp_path/'execution'))
    assert not calls and not (tmp_path/'execution').exists()

def test_fresh_binding_scope_and_actual_closed_dependencies():
    c=json.loads((ROOT/FRESH).read_bytes());b=execution_binding(ROOT,c)
    assert b.blockers()==() and b.method('A4').method=='unified_team_prompt_search_v2_3'
    assert c['token_ledger_directory'].endswith('/accounting') and c['accounting_scope_policy']=='FRESH_V23_SINGLE_ARM_2M_V1'
    assert c['budget']['metric_calls']==42 and c['arms']=={'A4':[True,True]}
    assert c['canary_review_policy']['extra_provider_calls']==0
    assert c['heldout_accounting_reserve']==0 and c['operational_pilot']['token_ceiling']==2_000_000
    assert 'paired_realization_policy' not in c and 'gradient_recovery_policy' not in c
    assert interface_for_contract(c)[0]==MATH_SOLVER_INTERFACE_V6
    assert solver_user_content(c,'Synthetic procedure.','Synthetic public problem.')=='Synthetic procedure.\n\nSynthetic public problem.'
    graph=current_dependency_graph(ROOT)
    assert not graph['legacy_namespace_dependencies'] and not graph['legacy_class_definitions']
    for name in ('math_visible_binding.py','math_gradient_pattern_binding.py','gradient_pilot_contract.py','matched_realization.py','native_feed.py','gradient_recovery.py'):
        assert not any(path.endswith('/'+name) for path in graph['modules'])

@pytest.mark.parametrize('field',['method_identity','optimization_evidence_policy','accounting_policy_sha256','current_user_scope_sha256','initial_team_artifact_sha256'])
def test_fresh_readpoint_tampering_cannot_compose(field):
    c=json.loads((ROOT/FRESH).read_bytes());c[field]='tampered'
    assert execution_binding(ROOT,c).blockers()

def test_fresh_2m_journal_recovery_and_no_old_authorization_reuse(tmp_path):
    req=dict(model='synthetic',max_tokens=100,messages=[]);bound=reservation(req)
    with TokenLedger(tmp_path,task_sha256='a'*64,policy=POLICY_V23_2M) as ledger:
        key=ledger.reserve(req,attempt_id='fresh',stage='initial',role='solver',model='synthetic')
        assert ledger.reserved==bound['amount']
    with TokenLedger(tmp_path,task_sha256='a'*64,policy=POLICY_V23_2M) as ledger:
        assert ledger.view()['fallback_charged']==bound['amount'] and ledger.reserved==0
        assert ledger.remaining==2_000_000-bound['amount']
        with pytest.raises(OperationalAbort):ledger.amend_authorization_40m(authorization_sha256='b'*64,expected_charged_total=bound['amount'])
    for policy in (POLICY_40M,None):
        with pytest.raises(OperationalAbort):TokenLedger(tmp_path,task_sha256='a'*64,**({'policy':policy} if policy else {}))

def test_missing_usage_is_charged_and_next_request_stops_before_transport(tmp_path):
    c=contract();c['operational_pilot']['token_ceiling']=100000
    calls=[]
    with TokenLedger(tmp_path,task_sha256='a'*64,policy=POLICY_V23_2M) as ledger:
        def transport(request):
            calls.append(request);return dict(text='Steps.\nFINAL_ANSWER: 2',finish_reason='stop')
        broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81,token_ledger=ledger)
        broker.complete(role='solver',split='optimize',stage='initial',member_slot=0,messages=[])
        assert ledger.view()['fallback_charged']==reservation(calls[0])['amount']
        c['operational_pilot']['token_ceiling']=ledger.view()['charged_total']+reservation(calls[0])['amount']-1
        with pytest.raises(OperationalAbort):broker.complete(role='solver',split='optimize',stage='initial',member_slot=1,messages=[])
        assert len(calls)==1 and ledger.reserved==0

def test_response_receipt_precedes_charge_and_locked_snapshot_detaches(tmp_path):
    c=contract()
    with TokenLedger(tmp_path/'ledger',task_sha256='a'*64,policy=POLICY_V23_2M,best_effort_snapshots=True) as ledger:
        receipts=ProviderResponseReceipts(tmp_path/'raw',attempt_id=c['execution_attempt_id'],startup_identity_sha256='b'*64)
        original=ledger.reconcile
        def reconcile(key,result,**kw):
            assert receipts.read(key)['response']==result
            return original(key,result,**kw)
        ledger.reconcile=reconcile
        stream=ledger.snapshot.open('rb')
        try:
            broker=RequestBroker(contract=c,transport=lambda _:response('Steps.\nFINAL_ANSWER: 2'),arm='A4',seed=81,
                token_ledger=ledger,response_receipts=receipts)
            broker.complete(role='solver',split='optimize',stage='initial',member_slot=0,messages=[])
            assert ledger.view()['charged_total']==4 and ledger.reserved==0
            assert ledger.events[-1]['response_receipt']['integrity_sha256']
        finally:stream.close()
