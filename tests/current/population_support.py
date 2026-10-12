"""Current synthetic provider fixtures. Never contacts a model provider."""
import json,re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from multi_dataset_diverse_rl.benchmarks.math_structured_binding import derive_structured_contract
from multi_dataset_diverse_rl.benchmarks.math_structured_interface import MATHStructuredSystemBenchmark
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
from multi_dataset_diverse_rl.search.current_composition import build_current_team_prompt_search
from multi_dataset_diverse_rl.search.current_layer1 import CurrentOptimizer
from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
from multi_dataset_diverse_rl.search.optimization_evidence import POLICY
from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver, ReflectionProvider, RequestBroker
from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
from multi_dataset_diverse_rl.search.textual_gradients import GradientClusterProvider, PerExampleGradientProvider
from multi_dataset_diverse_rl.search.system_prompt import SEED
ROOT=Path(__file__).resolve().parents[2]
HYPOTHESIS='Check constraints before transforming intermediate expressions.'
BASE=SEED
GOOD=BASE.edit('strategy','Check signs and substitute the result to verify constraints.')
def contract():
    from multi_dataset_diverse_rl.search.solver_execution import POLICY as serial_reference
    parent='experiments/execution_bindings/math_v2_2_gradient_pattern_seed81_pilot_v2.json'
    return derive_structured_contract(json.loads((ROOT/parent).read_bytes()),attempt='population_offline',
        binding_path='runs/population_binding.json',parent_path=parent,parent_sha256=sha256((ROOT/parent).read_bytes()).hexdigest(),
        authorization_path='runs/population_scope.json',authorization_sha256='a'*64,
        gradient_prompt_path='runs/population_gradient.json',gradient_prompt_sha256='b'*64,
        validation_metadata_path='runs/population_metadata.json',validation_metadata_sha256='c'*64,
        initial_team_path='runs/population_team.json',initial_team_artifact_sha256='d'*64,
        pattern_prompt_path='runs/population_pattern.json',pattern_prompt_sha256='e'*64,
        accounting_policy_path='runs/population_accounting.json',accounting_policy_sha256='f'*64,
        execution_policy=serial_reference)
def response(text):
    return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',
        provider_response_accepted=True,provider_metadata_loss_audited=True,
        provider_reasoning_content_present=False,provider_reasoning_character_count=None,
        provider_usage_details={},provider_thinking_indicators=[])
def graph(tmp_path,*,reflection=None,solver_behavior=None,uncertain=False,long_response=False,
        answer_only=False,answer_only_form='markdown',block='strategy',format_repair=False,
        prompts=None,solver_answer=None):
    c=contract();adapter=MATHStructuredSystemBenchmark(c);requests=[];packets=[]
    def examples(split):
        return tuple(CorrectnessExample(protocol_input('math',f'{split}{i}',{'problem':f'Synthetic case {i}: find the result.'},
            adapter.output_contract,protocol=adapter.protocol),'2',f'REFERENCE_{i}: the operation gives the reference result.' if split=='optimize' else None)
            for i in range(18 if split=='optimize' else 3))
    def transport(req):
        requests.append(deepcopy(req))
        if req['model']=='gpt-4o-mini':
            index=int(re.search(r'case (\d+):',req['messages'][1]['content'])[1]);prompt=req['messages'][0]['content']
            correct=solver_behavior(prompt,index,req) if solver_behavior else index>=6 or 'conditional repair' in prompt and index==0
            written='' if answer_only else 'Actual substitution operation is recorded.\n'
            if long_response:written+='A synthetic ordinary operation is written here. Unicode数学核验 π。\n'*120
            answer=solver_answer(prompt,index,req) if solver_answer else '2' if correct else '3'
            if answer_only_form=='natural_confirmation':return response('Final answer: '+answer+'\nTherefore, x = '+answer+'.')
            return response(written+'Final answer: '+answer)
        if len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            if 'example' in p:
                return response(json.dumps(dict(disposition='UNCERTAIN' if uncertain else 'ACTIONABLE',
                    observed_failure='The observed operation omits a constraint.',diagnosis='A domain check may be needed.',
                    suggested_block='strategy',reusable_correction=None if uncertain else HYPOTHESIS,
                    expected_effect=None if uncertain else 'Apply constraints before transformation.')))
            return response(json.dumps(dict(patterns=[dict(generalized_gradient=HYPOTHESIS,
                support_ids=[g['example_id'] for g in p['gradients']])],unassigned_ids=[])))
        packet=json.loads(req['messages'][0]['content'].splitlines()[-1]);packets.append(packet)
        if reflection:return response(reflection(packet,len(packets)))
        return response(json.dumps(dict(target_block=block,new_content=f'Apply a conditional repair rule {len(packets)}: check domains before transformations.',
            change_summary='Add one conditional domain repair.')))
    broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81)
    solver=BenchmarkSolver(adapter,broker);reflection_provider=ReflectionProvider(broker)
    gradient=PerExampleGradientProvider(broker);pattern=GradientClusterProvider(broker,gradient_provider=gradient)
    method=CURRENT_POLICY_BUNDLE.method(aggregation=EquivalencePluralityAggregation.identity,
        provider_binding='0'*64,successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
        solver_trajectory_policy=c['solver_trajectory_policy'],optimization_evidence_policy=POLICY)
    optimizer=CurrentOptimizer(evaluator=solver,reflection_lm=reflection_provider,
        accounting_reader=reflection_provider.accounting,run_root=tmp_path)
    run=build_current_team_prompt_search(benchmark=adapter,aggregation=EquivalencePluralityAggregation(),
        examples=examples('optimize'),prompts=prompts or (BASE,)*5,solver=solver,optimizer=optimizer,method=method,
        seed=81,shadow_loader=lambda:examples('shadow'),shadow_count=3,runtime_readiness=lambda:(),
        pattern_provider=pattern,provider_call_reader=lambda:broker.successes)
    run.state.initialize()
    return run,broker,requests,packets
