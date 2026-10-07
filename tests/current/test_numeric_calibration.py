# V2.1 frozen replay assertions; V2.2 current conformance is tested separately.
"""Strong/weak numeric evidence, private telemetry and exact fresh identities."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from multi_dataset_diverse_rl.search.numeric_provenance import numeric_guard_result,numeric_content_leaked_v5
from multi_dataset_diverse_rl.search.textual_gradients import validate_gradient,GradientExtractor
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.legacy.current_math_domain_binding_v21 import execution_binding
from multi_dataset_diverse_rl.current_contract import TARGET_CORRECTNESS_SIGNAL_VERSION
ROOT=Path(__file__).resolve().parents[2]
PROFILE='experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v5.json'
def row(problem,gold='x',prediction='z'):
    return NS(example_id='synthetic',source_split='optimize',signals=dict(input_payload=problem,gold=gold,
        target_output=prediction,target_member_correct=False,target_member_valid=True,
        correctness_signal_identity=TARGET_CORRECTNESS_SIGNAL_VERSION,responsibility_labels=['coverage'],
        team_margin=0,team_disagreement=0))

@pytest.mark.parametrize('value',['100','180','360','0.5','0.125','1/2','3','10','123','1.25e-1','17/43'])
def test_bare_overlap_warns_without_changing_admissibility(value):
    # Literal instantiated expressions retain their branch. An equivalent
    # slash/LaTeX ratio has real value overlap without copying a literal span.
    source='The source quantity is '+('\\frac{17}{43}' if value=='17/43' else value)+'.'
    r=row(source);gradient='Check a reusable threshold of '+value+'.'
    result=numeric_guard_result(gradient,(r,))
    assert result['hard_reject'] is False and not result['strong_reasons']
    assert result['warning_reasons']
    if value=='17/43':assert '17/43' in result['matched_values']
    assert validate_gradient(gradient,(r,))==gradient

@pytest.mark.parametrize('problem,gold,prediction,gradient,reason',[
    ('A symbolic example.','42','x','42','ANSWER_LITERAL_MATCH'),
    ('A symbolic example.','42','x','The answer is forty-two.','ANSWER_LITERAL_MATCH'),
    ('A symbolic example.','x','17','Return 17.','ANSWER_LITERAL_MATCH'),
    ('Solve 3x+5=14.','x','z','Transform 3x + 5 = 14.','COPIED_NUMERIC_EXPRESSION'),
    ('Transform $\\frac{3}{7}$.','x','z','Expand $\\frac{3}{7}$.','COPIED_NUMERIC_EXPRESSION'),
    ('Factor $(Cx-19)(Dx-7),$ over integers.','31','23',
        'Verify by expanding $(Cx-19)(Dx-7)$ and equating coefficients.','COPIED_NUMERIC_EXPRESSION'),
    ('A length is 137 meters.','x','z','Check 137 meters.','VALUE_AND_UNIT_MATCH'),
    ('A ratio is 17/43.','x','z','Check the specified ratio 17/43.','INSTANCE_CUED_VALUE_MATCH'),
    ('A length is 256.','x','z','Use the provided value 256.','INSTANCE_CUED_VALUE_MATCH'),
    ('Alice buys 137 items.','x','z','Check Alice against a threshold of 137.','VALUE_WITH_ENTITY_CONTEXT'),
    ('A sequence with 137 items requires arranging all the selected values in ascending order before completing the task.',
        'x','z','Check 137 items by arranging all the selected values in ascending order before completing the task.',
        'VALUE_WITH_LONG_SOURCE_FRAGMENT'),
])
def test_strong_provenance_remains_terminal(problem,gold,prediction,gradient,reason):
    r=row(problem,gold,prediction);result=numeric_guard_result(gradient,(r,))
    assert result['hard_reject'] and reason in result['strong_reasons']
    assert result['matches'] and all(len(m['source_span_sha256'])==64 for m in result['matches'])
    for generalized in (False,True):
        with pytest.raises(SearchContractError,match='GRADIENT_.*INVALID'):
            validate_gradient(gradient,(r,),generalized=generalized)

def test_generic_rule_unit_does_not_match_an_unrelated_source_unit():
    r=row('A source quantity is 360 meters.')
    g='Angles in a full turn sum to 360 degrees.'
    result=numeric_guard_result(g,(r,))
    assert not result['hard_reject'] and result['warning_reasons']==['BARE_DISTINCTIVE_VALUE_MATCH']
    assert validate_gradient(g,(r,))==g
    assert not numeric_guard_result('Check that probabilities sum to 1.',(row('A quantity is 1.'),))['hard_reject']

def test_v5_detector_is_preserved_as_an_explicit_replay_function():
    r=row('A quantity is 123.')
    assert numeric_content_leaked_v5('Check a threshold of 123.',(r,))
    assert not numeric_guard_result('Check a threshold of 123.',(r,))['hard_reject']

@pytest.mark.parametrize('field,value',[('shadow_count',41),('stop_policy','changed'),('models',{}),
    ('gradient_prompt_sha256','0'*64),('provider_bounds',{}),('partition_completion_policy',None),
    ('numeric_calibration_parent_binding_sha256','0'*64),('numeric_calibration_amendment_sha256','0'*64)])
def test_calibration_binding_rejects_scientific_or_receipt_drift(field,value):
    c=json.loads((ROOT/PROFILE).read_bytes());c[field]=value
    try:blockers=execution_binding(ROOT,c).blockers()
    except SearchContractError:return
    assert blockers

def test_new_guard_changes_only_admissibility_and_fresh_identity():
    c=json.loads((ROOT/PROFILE).read_bytes());p=json.loads((ROOT/c['numeric_calibration_parent_binding_path']).read_bytes())
    assert not execution_binding(ROOT,c).blockers()
    assert c['partition_completion_policy']==p['partition_completion_policy']=='GRADIENT_PATTERN_PARTITION_COMPLETION_V1'
    for key in ('models','decoding','provider_bounds','memory_limits','shared_risk_policy','layer1_search_policy',
        'low_cost_protocol','stop_policy','gradient_prompt_sha256','pattern_prompt_sha256','runtime_persistence_policy'):
        assert c[key]==p[key]
    policy=deepcopy(c['pattern_policy']);policy['gradient_policy']['abstraction_guard']=p['pattern_abstraction_guard']
    assert policy==p['pattern_policy']
    with pytest.raises(SearchContractError,match='LEGACY_POLICY_FORBIDDEN'):execution_binding(ROOT,p)

def test_warning_journal_never_enters_provider_input_or_gradient_records():
    calls=[];events=[]
    class Provider:
        numeric_guard_writer=staticmethod(events.append)
        def extract(self,payload):calls.append(deepcopy(payload));return {'gradient':'Check a threshold of 123.'}
    extractor=GradientExtractor(Provider());r=row('A source quantity is 123.')
    values=extractor.extract('Solve the problem.',(r,))
    assert len(calls)==len(events)==1 and values==({'example_id':'synthetic','gradient':'Check a threshold of 123.'},)
    assert set(calls[0])=={'schema','current_member_procedure','example'}
    assert events[0]['numeric_guard_result']['warning_reasons']==['BARE_DISTINCTIVE_VALUE_MATCH']

def test_critical_warning_journal_failure_does_not_regenerate():
    calls=[]
    class Provider:
        def numeric_guard_writer(self,event):raise OSError('synthetic durable write failure')
        def extract(self,payload):calls.append(payload);return {'gradient':'Check a threshold of 123.'}
    with pytest.raises(OSError):GradientExtractor(Provider()).extract('Solve the problem.',(row('A quantity is 123.'),))
    assert len(calls)==1

def test_scientific_boundary_blocks_pilot_preparation_without_any_provider(tmp_path):
    from multi_dataset_diverse_rl.governance.legacy.current_unified_execution_v21 import preexecution_manifest,bound_preflight,prepare_canary
    manifest=preexecution_manifest(ROOT,source_sha=None,frozen=False,binding_path=PROFILE,
        experiment_id='math_v2_1_gradient_pattern_seed81_pilot_v5')
    result=bound_preflight(ROOT,manifest)
    assert result['gate']=='HOLD' and result['provider_attempts']==0
    assert 'ATTEMPT4_CONFIRMED_STRONG_NUMERIC_LEAKAGE' in result['blockers']
    with pytest.raises(SearchContractError,match='CONFIRMED_STRONG_NUMERIC_LEAKAGE'):
        prepare_canary(ROOT,manifest,destination=tmp_path/'blocked_prep')
    assert not (tmp_path/'blocked_prep').exists()
