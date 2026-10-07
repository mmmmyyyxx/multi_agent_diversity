"""First-valid recovery has an output-only boundary and identical requests."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from multi_dataset_diverse_rl.current_contract import TARGET_CORRECTNESS_SIGNAL_VERSION
from multi_dataset_diverse_rl.search.gradient_recovery import POLICY,statistics
from multi_dataset_diverse_rl.search.textual_gradients import GradientExtractor,PerExampleGradientProvider
from multi_dataset_diverse_rl.search.schemas import EvidenceItem,SearchContractError
from multi_dataset_diverse_rl.persistence.durable_io import append_jsonl
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding

ROOT=Path(__file__).resolve().parents[2]
PROFILE='experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v6.json'
VALID='Check intermediate transformations against the original constraints.'
LEAK='Expand 3x+5=14 before transforming the equation.'
def row():
    return EvidenceItem('synthetic','optimize',frozenset({'REPAIR'}),dict(input_payload='Solve 3x+5=14.',
        gold='31',target_output='23',target_member_valid=True,target_member_correct=False,
        correctness_signal_identity=TARGET_CORRECTNESS_SIGNAL_VERSION,responsibility_labels=('coverage',),
        team_margin=0,team_disagreement=0))

class Broker:
    def __init__(self,outputs):self.outputs=iter(outputs);self.requests=[]
    def _request_identity(self,**kw):return kw,'f'*64
    def complete(self,**kw):
        self.requests.append(deepcopy(kw));v=next(self.outputs)
        if isinstance(v,Exception):raise v
        return dict(text=v if isinstance(v,str) else json.dumps(v),input_tokens=10,output_tokens=7,request_sha256='f'*64)

def extractor(outputs,writer=None):
    broker=Broker(outputs)
    p=PerExampleGradientProvider(broker,recovery_policy=POLICY,recovery_writer=writer)
    return GradientExtractor(p),broker

@pytest.mark.parametrize('outputs,accepted',[
    ([{'gradient':VALID}],1),
    ([{'gradient':LEAK},{'gradient':VALID}],2),
    ([{'other':VALID},{'gradient':'x'*401},{'gradient':VALID}],3),
    (['{malformed',{'gradient':VALID}],2),
    ([{'gradient':'Check the reasoning.'}],1),
    ([{'gradient':''},{'gradient':VALID}],2),
    ([{'gradient':None},{'gradient':VALID}],2),
    ([{'gradient':VALID,'extra':1},{'gradient':VALID}],2),
    ([{'gradient':'You are a mathematical solver.'},{'gradient':VALID}],2),
    ([{'gradient':'Output FINAL_ANSWER: on the final line.'},{'gradient':VALID}],2),
])
def test_first_contract_valid_only_and_exact_request_equivalence(outputs,accepted):
    e,b=extractor(outputs);records=e.extract('Solve the problem.',(row(),))
    assert records==({'example_id':'synthetic','gradient':outputs[-1]['gradient']},)
    assert len(b.requests)==accepted
    assert all(r==b.requests[0] for r in b.requests)
    assert set(records[0])=={'example_id','gradient'}
    stats=statistics(e.recovery_audit)
    assert stats['logical_gradient_count']==stats['accepted_gradient_count']==1
    assert stats['physical_gradient_calls']==accepted
    assert e.recovery_audit[-1]['accepted_attempt_no']==accepted
    assert e.recovery_audit[-1]['prior_invalid_attempt_count']==accepted-1
    assert len({a['logical_gradient_id'] for a in e.recovery_audit})==1
    if accepted>1:
        scientific=b.requests[-1]['messages']
        assert not any('failure_category' in m['content'] or LEAK in m['content'] for m in scientific)

def test_three_failures_stop_and_no_example_dropped_or_cluster_called():
    e,b=extractor([{'gradient':LEAK}]*3)
    with pytest.raises(SearchContractError,match='PATTERN_GRADIENT_EXTRACTION_INVALID') as error:
        e.extract('Solve the problem.',(row(),))
    assert error.value.gradient_recovery_exhausted
    stats=statistics(e.recovery_audit)
    assert stats['physical_gradient_calls']==3 and stats['gradient_three_fail']==1
    assert stats['accepted_gradient_count']==0 and stats['strong_leakage_rejections']==3

@pytest.mark.parametrize('failure',[
    SearchContractError('PATTERN_GRADIENT_INPUT_INVALID'),
    SearchContractError('STOP_PATTERN_CONTEXT_LIMIT_POLICY_REQUIRED'),
    SearchContractError('PROVIDER_ROLE_SPLIT_FORBIDDEN'),
    SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID'),
    OSError('synthetic persistence failure'),RuntimeError('synthetic accounting failure'),
    ConnectionError('synthetic exhausted transport failure'),
])
def test_broker_or_input_failures_never_trigger_contract_retry(failure):
    e,b=extractor([failure,{'gradient':VALID}])
    with pytest.raises(type(failure)) as error:e.extract('Solve the problem.',(row(),))
    assert error.value is failure and len(b.requests)==1 and not e.recovery_audit

@pytest.mark.parametrize('bad_input',['empty_procedure','duplicate','heldout','correct'])
def test_corrupted_universe_is_checked_before_any_call(bad_input):
    from dataclasses import replace
    r=row();rows=(r,);procedure='Solve the problem.'
    if bad_input=='empty_procedure':procedure=''
    elif bad_input=='duplicate':rows=(r,r)
    elif bad_input=='heldout':rows=(replace(r,source_split='validation'),)
    else:rows=(replace(r,signals={**r.signals,'target_member_correct':True}),)
    e,b=extractor([])
    with pytest.raises(SearchContractError):e.extract(procedure,rows)
    assert not b.requests

def test_durable_audit_failure_propagates_before_retry():
    def broken(_):raise PermissionError('synthetic writer failure')
    e,b=extractor([{'gradient':LEAK},{'gradient':VALID}],broken)
    with pytest.raises(PermissionError):e.extract('Solve the problem.',(row(),))
    assert len(b.requests)==1 and not e.recovery_audit

def test_1005_journal_cycles_and_fresh_logical_ids(tmp_path):
    path=tmp_path/'attempts.jsonl'
    e,b=extractor([{'gradient':VALID}]*1005,lambda event:append_jsonl(path,event))
    for _ in range(1005):e.extract('Solve the problem.',(row(),))
    events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert events==e.recovery_audit and len({a['logical_gradient_id'] for a in events})==1005
    assert statistics(events)['gradient_first_pass_valid']==1005

def test_absent_policy_preserves_single_draw_historical_behavior():
    b=Broker([{'gradient':LEAK},{'gradient':VALID}])
    e=GradientExtractor(PerExampleGradientProvider(b))
    with pytest.raises(SearchContractError):e.extract('Solve the problem.',(row(),))
    assert len(b.requests)==1 and not e.recovery_audit

@pytest.mark.parametrize('field,value',[('max_physical_attempts_per_wrong',3.0),('output_cache',0),
    ('semantic_quality_selection',0),('selection','best_of_three')])
def test_policy_identity_requires_exact_typed_values(field,value):
    policy=deepcopy(POLICY);policy[field]=value
    with pytest.raises(SearchContractError,match='RECOVERY_POLICY_MISMATCH'):
        PerExampleGradientProvider(Broker([]),recovery_policy=policy)

def test_recovery_binding_changes_only_fresh_identity_gradient_ceiling_and_policy():
    c=json.loads((ROOT/PROFILE).read_bytes());p=json.loads((ROOT/c['gradient_recovery_parent_binding_path']).read_bytes())
    assert not execution_binding(ROOT,c).blockers()
    for key in ('pattern_policy','gradient_prompt_sha256','pattern_prompt_sha256','pattern_abstraction_guard',
        'models','decoding','memory_limits','shared_risk_policy','layer1_search_policy','low_cost_protocol',
        'stop_policy','runtime_persistence_policy','partition_completion_policy'):
        assert c[key]==p[key]
    assert c['provider_bounds']['pattern_gradient_calls']==3*p['provider_bounds']['pattern_gradient_calls']
    for key in ('solver_calls','reflection_calls','pattern_cluster_calls','max_opportunities'):
        assert c['provider_bounds'][key]==p['provider_bounds'][key]
    assert c['provider_bounds']['successful_provider_calls']-p['provider_bounds']['successful_provider_calls']==2*p['provider_bounds']['pattern_gradient_calls']

@pytest.mark.parametrize('field,value',[('gradient_recovery_policy',{}),('gradient_prompt_sha256','0'*64),
    ('provider_bounds',{}),('gradient_recovery_user_scope_sha256','0'*64),('pattern_abstraction_guard','changed'),
    ('stop_policy','changed'),('shadow_count',41)])
def test_recovery_binding_rejects_drift(field,value):
    c=json.loads((ROOT/PROFILE).read_bytes());c[field]=value
    try:binding=execution_binding(ROOT,c)
    except SearchContractError:return
    assert binding.blockers()
