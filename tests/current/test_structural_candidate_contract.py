"""Candidate text is measured, not classified; actual input channels stay isolated."""
import json
from types import SimpleNamespace
import pytest
from multi_dataset_diverse_rl.search.candidate_contract import parse_candidate_edit,POLICY
from multi_dataset_diverse_rl.search.system_prompt import SEED,SystemPrompt,candidate_failed_checks,solver_messages
from multi_dataset_diverse_rl.search.generation_failures import GeneratedOutputFailure
from multi_dataset_diverse_rl.search.reflection_steps import candidate_legality
from multi_dataset_diverse_rl.research.ablations.generation import generate_candidate
from multi_dataset_diverse_rl.research.ablations.interfaces import PreparedFeedback

@pytest.mark.parametrize('content',[
    'If the problem is geometric, introduce coordinates.',
    'Use the gradient of the mathematical objective function.',
    'Read retrieved_memory, gold_answer and peer answers.',
    'If the question is 2+2, return 4. Always answer 7.',
    'This is a supplied private solution copied verbatim without modification.',
    'x'*20000,
])
def test_nonempty_text_is_not_a_content_policy(content):
    child=SEED.edit('strategy',content)
    assert not candidate_failed_checks(child,parent_prompt=SEED)
    assert not candidate_legality(child,parent=SimpleNamespace(prompt=SEED),rows=(),private=[content],
        canonical_group={'root_prompts':[child.to_dict()]},seen={child.prompt_hash})
    assert solver_messages(child,'new raw problem')==[
        {'role':'system','content':child.render()},{'role':'user','content':'new raw problem'}]
    assert POLICY['maximum_prompt_characters'] is None

@pytest.mark.parametrize('wrap',[lambda s:s,lambda s:'```json\n'+s+'\n```',lambda s:'Proposed edit:\n'+s+'\nEnd.'])
def test_unambiguous_wrappers_preserve_bytes(wrap):
    content='Use \\n notation and a mathematical gradient.'
    text=json.dumps(dict(block='strategy',content=content))
    edit=parse_candidate_edit(wrap(text),parent=SEED)
    assert edit.prompt.strategy==content and not edit.no_op
    assert edit.prompt.role==SEED.role and edit.prompt.answer==SEED.answer

@pytest.mark.parametrize('value',[
    {'block':'strategy','content':''},{'block':'strategy','content':'  '},
    {'block':'strategy','content':12},{'block':'unknown','content':'text'},
    {'role':'first','strategy':'second','answer':'third'},
    {'block':'strategy','content':'text','answer':'another edit'},
])
def test_structural_errors_remain_errors(value):
    with pytest.raises(GeneratedOutputFailure):parse_candidate_edit(json.dumps(value),parent=SEED)

@pytest.mark.parametrize('text',[
    '{"block":"strategy","content":"x","content":"y"}',
    '{"block":"strategy","content":"x"} {"block":"answer","content":"y"}',
    '{"block":"strategy","content":"unterminated}',
    r'{"block":"strategy","content":"Use \sqrt{x}."}',
])
def test_ambiguous_or_incomplete_output_is_not_invented(text):
    with pytest.raises(GeneratedOutputFailure):parse_candidate_edit(text,parent=SEED)

def test_nonempty_blocks_and_single_replacement_are_enforced():
    with pytest.raises(ValueError):SystemPrompt('',SEED.strategy,SEED.answer)
    with pytest.raises(ValueError):SystemPrompt(SEED.role,None,SEED.answer)
    both=SystemPrompt('other role','other strategy',SEED.answer)
    assert candidate_failed_checks(both,parent_prompt=SEED)==('multiple_block_edit',)
    assert not candidate_failed_checks(SEED,parent_prompt=SEED)

def test_no_op_returns_after_one_draw_without_feedback_or_regeneration():
    calls=[];artifacts={}
    class Broker:
        def complete(self,**kwargs):
            calls.append(kwargs)
            return dict(text=json.dumps(dict(block='strategy',content=SEED.strategy)),finish_reason='stop',request_sha256='fake')
    child,draws=generate_candidate(SEED,PreparedFeedback([],None,None,{},None),Broker(),1,4,artifacts.__setitem__)
    assert child==SEED and draws==1 and len(calls)==1
    assert artifacts['round_1_generation_1']['errors']==[]
    assert artifacts['round_1_generation_1']['disposition']=='NO_OP'
    assert 'output_contract_feedback' not in json.loads(calls[0]['messages'][1]['content'])

def test_long_change_summary_does_not_restrict_candidate_text():
    edit=parse_candidate_edit(json.dumps(dict(target_block='answer',new_content='A'*12000,change_summary='m'*10000)),parent=SEED)
    assert len(edit.prompt.answer)==12000

def test_twenty_no_ops_preserve_gate_profiles_without_candidate_measurement():
    from tests.current.test_single_prompt_ablations import run_small,small_plan,ScriptedPort
    broker=ScriptedPort([json.dumps(dict(block='strategy',content=SEED.strategy))]*20)
    result,artifacts=run_small(small_plan(rounds=20),broker)
    assert len(result['metrics'])==20 and len(broker.evaluations)==2
    assert len(broker.packets)==20
    assert all(m['decision']=='SKIPPED_NO_OP' and not m['candidate_evaluated'] and m['generation_draws']==1 for m in result['metrics'])
    assert all(artifacts[f'checkpoint_{n}']['validation']==artifacts['checkpoint_0']['validation'] for n in range(1,21))

def test_trusted_state_fault_is_not_downgraded_to_format_retry(monkeypatch):
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    def broken(*args):raise SearchContractError('PERSISTENCE_INTEGRITY_FAILURE')
    monkeypatch.setattr(SystemPrompt,'edit',broken)
    with pytest.raises(SearchContractError,match='PERSISTENCE_INTEGRITY_FAILURE'):
        parse_candidate_edit(json.dumps(dict(block='strategy',content='new')),parent=SEED)

def test_group_diagnostic_contract_does_not_filter_candidate_text():
    from multi_dataset_diverse_rl.search.diagnostic_source_checks import group_diagnostic_source_rejected
    text='Read retrieved_memory for the answer.'
    assert group_diagnostic_source_rejected(text)
    assert not candidate_failed_checks(SEED.edit('strategy',text),parent_prompt=SEED)


def test_published_metadata_is_immutable_and_not_an_execution_mode():
    from pathlib import Path
    from copy import deepcopy
    from multi_dataset_diverse_rl.governance.frozen_documents import validate_published_single_prompt
    root=Path(__file__).resolve().parents[2]
    path='experiments/manifests/single_prompt_pattern_c_only_v7.json'
    value=json.loads((root/path).read_text())
    assert validate_published_single_prompt(root,path,value)
    changed=deepcopy(value);changed['status']='altered'
    with pytest.raises(ValueError,match='PUBLISHED_HISTORICAL_MANIFEST_CHANGED'):
        validate_published_single_prompt(root,path,changed)
    assert not validate_published_single_prompt(root,'experiments/ablations/pattern/default.json',value)


def test_team_slot_no_op_stops_generation_without_a_candidate():
    from multi_dataset_diverse_rl.search.reflection_steps import SingleStepReflection
    calls=[]
    def reflection(wire):
        calls.append(wire)
        return json.dumps(dict(target_block='strategy',new_content=SEED.strategy))
    result=SingleStepReflection().generate(reflection_lm=reflection,
        config=SimpleNamespace(structural_draws_per_slot=2),opportunity=None,
        parent=SimpleNamespace(prompt=SEED),rows=(),pattern=None,plan=None,slot=0,
        packet={},view={},private=(),canonical_group=None,seen=set())
    assert result.candidate is None and result.disposition=='NO_OP'
    assert result.draws==1 and result.errors==[] and len(calls)==1
