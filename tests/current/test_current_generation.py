"""Current V2.4 checks ported from preserved replay tests; zero real APIs."""
import asyncio,json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from tests.current.test_structured_optimization_evidence import contract

from multi_dataset_diverse_rl.benchmarks.math_optimizer_generation import pattern_cluster_generation_contract,frozen_cluster_policy
from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import generation_request_fields
from multi_dataset_diverse_rl.governance.token_accounting import reservation,OperationalAbort
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
@pytest.mark.parametrize('field,value',[('max_completion_tokens',8191),('max_completion_tokens',True),
    ('accounting_output_ceiling',8192),('roles',['reflection']),('enable_thinking',True)])
def test_role_policy_tampering_fails_closed(field,value):
    c=contract();c['pattern_cluster_generation_policy'][field]=value
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
