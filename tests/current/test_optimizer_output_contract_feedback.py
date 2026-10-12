"""Output rejection and recovery use fake providers, never efficacy assertions."""
import asyncio
from copy import deepcopy
import json

import pytest

from tests.current.population_support import graph, BASE, GOOD, HYPOTHESIS, response
from tests.current.test_gradient_diagnostic_limits_400 import output
from multi_dataset_diverse_rl.search.textual_gradients import (
    GradientExtractor, PerExampleGradientProvider, GRADIENT_OUTPUT_LIMITS, REFERENCE_GRADIENT_PROMPT)
from multi_dataset_diverse_rl.search.generation_failures import GeneratedOutputFailure, POLICY
from multi_dataset_diverse_rl.search.optimizer_output_contract import (
    validate_gradient_output, validate_reflection_output, parse_output, feedback_for_error, validate_feedback)
from multi_dataset_diverse_rl.search.schemas import SearchContractError


@pytest.mark.parametrize('value,field,code',[
    (output(suggested_block={'role':'private rejected text','strategy':None}),'suggested_block','TYPE'),
    (output(suggested_block=['role','strategy']),'suggested_block','TYPE'),
    (output(diagnosis='数'*402),'diagnosis','OVER_LIMIT'),
    (output(diagnosis='数'*411),'diagnosis','OVER_LIMIT'),
    (output(disposition={'ACTIONABLE':True}),'disposition','TYPE'),
    (output(suggested_block='role, strategy, or answer'),'suggested_block','ENUM'),
    (output(expected_effect='e'*241),'expected_effect','OVER_LIMIT'),
    (output(disposition='UNCERTAIN'),'reusable_correction','MUST_BE_NULL'),
])
def test_rejects_existing_failure_shapes_without_repair_or_text_feedback(value,field,code):
    original=deepcopy(value)
    with pytest.raises(GeneratedOutputFailure) as caught:validate_gradient_output(value,GRADIENT_OUTPUT_LIMITS)
    feedback=feedback_for_error(caught.value,stage='gradient')
    assert any(i['field']==field and i['code']==code for i in feedback['issues'])
    assert value==original and 'private rejected text' not in json.dumps(feedback)
    if field=='diagnosis':
        issue=next(i for i in feedback['issues'] if i['field']==field)
        assert issue['actual_characters']==len(value[field]) and issue['limit']==400


def test_json_escaping_is_not_repaired_and_instructions_show_single_value_shape():
    with pytest.raises(GeneratedOutputFailure) as caught:parse_output(r'{"diagnosis":"\implies"}',stage='gradient')
    assert feedback_for_error(caught.value,stage='gradient')['issues'][0]['code']=='JSON_INVALID_ESCAPE'
    assert 'must be ONE string' in REFERENCE_GRADIENT_PROMPT
    assert '"suggested_block":"strategy"' in REFERENCE_GRADIENT_PROMPT
    assert r'\\alpha' in REFERENCE_GRADIENT_PROMPT


@pytest.mark.parametrize('mutation',[{'issues':[]},{'question':'private'}, {'stage':'reflection'},
    {'category':{'raw':'private'}},{'issues':[{'field':'raw_question','code':'TYPE','actual_type':'string','actual_characters':2,'limit':400}]}])
def test_feedback_integrity_is_fatal(mutation):
    with pytest.raises(GeneratedOutputFailure) as caught:validate_gradient_output(output(diagnosis='x'*401),GRADIENT_OUTPUT_LIMITS)
    feedback=feedback_for_error(caught.value,stage='gradient');feedback.update(mutation)
    with pytest.raises(SearchContractError,match='OPTIMIZER_OUTPUT_FEEDBACK_INTEGRITY_REQUIRED'):
        validate_feedback(feedback,stage='gradient')


def test_gradient_retry_feedback_keeps_full_task_evidence_and_stops_at_valid_draw(tmp_path):
    run,broker,*_=graph(tmp_path,long_response=True)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    row=run.opportunities.source.for_member(state,diagnosis,0)[0]
    draws=[json.dumps(output(suggested_block={'strategy':'secret rejected text'})),
        json.dumps(output(diagnosis='数'*402)),json.dumps(output())]
    requests=[]
    def transport(req):
        requests.append(deepcopy(req));return response(draws[len(requests)-1])
    broker.transport=transport
    extractor=GradientExtractor(PerExampleGradientProvider(broker))
    assert extractor.extract(BASE,(row,))==({'example_id':row.example_id,'gradient':HYPOTHESIS},)
    packets=[json.loads(req['messages'][1]['content']) for req in requests]
    assert len(packets)==3 and 'output_contract_feedback' not in packets[0]
    for packet in packets[1:]:
        feedback=packet.pop('output_contract_feedback')
        assert len(feedback['issues'])==1 and 'secret rejected text' not in json.dumps(packet)+json.dumps(feedback)
        assert packet==packets[0]
    diagnostic=extractor.evidence_diagnostics[0]
    assert [f['issues'][0]['field'] for f in diagnostic['output_contract_errors']]==['suggested_block','diagnosis']
    assert diagnostic['recovery_succeeded'] and not diagnostic['exhausted']
    assert packets[0]['example']['solver_trajectory']['feedback_truncated']
    assert len(packets[0]['example']['solver_trajectory']['observed_response'])>4096


def test_gradient_feedback_is_not_carried_to_another_example_and_uncertain_stops(tmp_path):
    run,broker,*_=graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    rows=run.opportunities.source.for_member(state,diagnosis,0)[:2];requests=[]
    def transport(req):
        requests.append(deepcopy(req))
        if len(requests)==1:return response(json.dumps(output(suggested_block={'strategy':'private'})))
        return response(json.dumps(output(disposition='UNCERTAIN',reusable_correction=None,expected_effect=None)))
    broker.transport=transport;extractor=GradientExtractor(PerExampleGradientProvider(broker))
    assert extractor.extract(BASE,rows)==()
    packets=[json.loads(r['messages'][1]['content']) for r in requests]
    assert len(packets)==3 and 'output_contract_feedback' in packets[1]
    assert 'output_contract_feedback' not in packets[2]
    assert [d['generation_draws'] for d in extractor.evidence_diagnostics]==[2,1]




def test_reflection_valid_schema_does_not_override_content_guards():
    value=dict(target_block='strategy',new_content='Copy the gold answer.',change_summary='Change.')
    # Schema acceptance is deliberately separate from the existing content guard.
    assert validate_reflection_output(value)==value
    from multi_dataset_diverse_rl.search.optimization_evidence import executability_checks
    for content in [value['new_content'],'Read output_contract_feedback before solving.']:
        assert 'optimizer_only_dependency' in executability_checks(BASE.edit('strategy',content))
