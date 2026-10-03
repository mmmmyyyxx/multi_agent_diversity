"""Count physical realizations across production reuse boundaries, offline."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from multi_dataset_diverse_rl.benchmarks.math_v21_binding import MATHV21Binding
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker, BenchmarkSolver
from multi_dataset_diverse_rl.search.binary_runtime import (
    BinaryTeamStateStore, CorrectnessExample, FixedPeerTeamEvaluationProvider,
    BinaryEvidenceSource,
)
from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
from multi_dataset_diverse_rl.search.binary_responsibility import BinaryPluralityResponsibilityAnalyzer
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.governance.math_paired_validation import evaluate_team

ROOT = Path(__file__).resolve().parents[1]


def ports(*, validation=False):
    c = json.loads((ROOT/'experiments/execution_bindings/math_v2_1_canary_v6.json').read_bytes())
    b = MATHV21Binding(ROOT, c).benchmark()
    calls=[]; events=[]
    broker = RequestBroker(contract=c, transport=lambda r:calls.append(r) or dict(
        text='FINAL_ANSWER: 1', finish_reason='stop', input_tokens=2, output_tokens=3),
        arm='A1', seed=81, ledger_writer=events.append, validation_only=validation)
    return b, broker, BenchmarkSolver(b, broker), calls, events


def test_initial_responsibility_evidence_fixed_peers_overlap_commit():
    b, broker, solver, calls, events = ports()
    examples = tuple(CorrectnessExample(protocol_input('math',str(i),
        {'problem':f'Synthetic arithmetic {i}.'},b.output_contract,protocol=b.protocol),'1') for i in range(3))
    prompts = tuple(f'Synthetic procedure {i}.' for i in range(5))
    store = BinaryTeamStateStore(benchmark=b,examples=examples,prompts=prompts,
        solver=solver,aggregation=EquivalencePluralityAggregation(),freeze_initial_competence=True)
    store.initialize(); assert len(calls)==15
    parent=store.snapshot(); history=HistoryState()
    diagnosis=BinaryPluralityResponsibilityAnalyzer(b.capabilities).analyze(parent, history)
    BinaryEvidenceSource(store,history).for_member(parent,diagnosis,0)
    assert len(calls)==15
    candidate=SimpleNamespace(prompt='Synthetic candidate procedure.')
    # GEPA-local evaluation and outer TeamProbe overlap share one realization.
    solver.solve(candidate.prompt,examples[0].item,stage='gepa_local',split='optimize')
    opportunity=SimpleNamespace(parent_state_id=parent.team_state_id,target_member=0)
    provider=FixedPeerTeamEvaluationProvider(store)
    provider._evaluate(opportunity,candidate,{'0','1'},'team_probe')
    assert len(calls)==17
    provider._evaluate(opportunity,candidate,{'0','1','2'},'full')
    assert len(calls)==18
    old_peers=tuple(store.profiles[i] for i in range(1,5))
    store.replace_member(0,candidate.prompt)
    assert len(calls)==18 and tuple(store.profiles[i] for i in range(1,5))==old_peers
    assert sum(e['kind']=='CACHE_HIT' for e in events)==3


def test_stage_member_reuse_and_shadow_capability():
    b, broker, solver, calls, events=ports()
    item=protocol_input('math','synthetic',{'problem':'Synthetic arithmetic.'},b.output_contract,protocol=b.protocol)
    solver.observe_member(0);solver.solve('Synthetic procedure.',item,stage='initial',split='optimize')
    solver.observe_member(4);solver.solve('Synthetic procedure.',item,stage='full',split='optimize')
    assert len(calls)==1
    gate=solver.for_gate()
    gate.solve('Synthetic procedure.',item,stage='adaptive_gate',split='shadow')
    gate.solve('Synthetic procedure.',item,stage='adaptive_gate',split='shadow')
    assert len(calls)==2 and gate.broker.cache is not broker.cache
    # A fresh process broker loses in-memory outputs, even for the same attempt.
    restarted=RequestBroker(contract=broker.contract,transport=broker.transport,arm='A1',seed=81)
    BenchmarkSolver(b,restarted).solve('Synthetic procedure.',item,stage='full',split='optimize')
    assert len(calls)==3


def test_paired_validation_unchanged_members_have_zero_physical_calls():
    b,broker,solver,calls,events=ports(validation=True)
    rows=[dict(stable_example_id=str(i),problem=f'Synthetic arithmetic {i}.',reference='1') for i in range(3)]
    prompts=tuple(f'Synthetic procedure {i}.' for i in range(5))
    a=evaluate_team(rows,prompts,solver,b,'initial',lambda _:None)
    assert len(calls)==15
    final=(prompts[0],'Synthetic changed procedure.',*prompts[2:])
    z=evaluate_team(rows,final,solver,b,'final',lambda _:None)
    assert len(calls)==18
    assert len(a)==len(z)==3 and broker.usage['validation']==18
    assert sum(e['kind']=='CACHE_HIT' for e in events)==12
