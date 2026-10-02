import asyncio
from copy import deepcopy
import json
from pathlib import Path
import pytest
from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import MATHBenchmarkAdapterV2,domain_matrix,require_scorable,final_payload
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import MATHDomainBinding
from multi_dataset_diverse_rl.search.benchmark import BenchmarkInput
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT=Path(__file__).resolve().parents[1]
CASES=[('(1,2,3)','(3,2,1)'),(r'\{1,2\}',r'\{1,3\}'),('[1,2]','[1,2)'),
       ('1,2,3','1,2,4'),('1','2'),('x+1','x+2')]


def test_v2_protocol_is_carried_by_inference_input():
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input,MATH_PROTOCOL_V2,PROTOCOLS
    b=MATHBenchmarkAdapterV2()
    item=protocol_input('math','synthetic',{'problem':'Synthetic one'},b.output_contract,protocol=b.protocol)
    assert item.parser_contract=='MATH_PAYLOAD_PARSER_V2'
    assert item.benchmark_version==MATH_PROTOCOL_V2.identity()
    historical=protocol_input('math','synthetic',{'problem':'Synthetic one'},b.output_contract)
    assert historical.benchmark_version==PROTOCOLS['math'].identity()


@pytest.mark.parametrize('good,bad',CASES)
def test_representation_positive_and_negative(good,bad):
    b=MATHBenchmarkAdapterV2();require_scorable(good)
    assert b.equivalent(good,good)
    assert not b.equivalent(good,bad)


def test_order_set_and_interval_boundaries():
    b=MATHBenchmarkAdapterV2()
    assert b.equivalent(r'\{1,2\}',r'\{2,1\}')
    assert not b.equivalent('(a,b,c)','(b,a,c)')
    assert not b.equivalent('(1,2,3)',r'\{1,2,3\}')
    assert not b.equivalent('[1,2]',r'\{1,2\}')
    for expression in ('(1,2)','(a,b)',r'\left(1,2\right)',r'\boxed{(1,2)}'):
        # Native parse attempted, representation remains genuinely ambiguous.
        result=domain_matrix((expression,))
        assert result['parsed_types'][0]
        with pytest.raises(SearchContractError,match='REFERENCE_UNSCORABLE'):
            require_scorable(expression)
    for a,c in [('[1,2]','[1,2)'),('[1,2)','(1,2]'),('(1,2]','[1,2]')]:
        assert not b.equivalent(a,c)
    assert b.equivalent('x=2','x=2')
    # Pinned strict verify does not normalize reversed equation sides.
    assert not b.equivalent('x=2','2=x')
    assert not b.equivalent('x=2','y=2')
    assert not b.equivalent('x=2','2')


@pytest.mark.parametrize('raw',['1','FINAL_ANSWER:','FINAL_ANSWER: 1\nmore','FINAL_ANSWER: 1\nFINAL_ANSWER: 1'])
def test_strict_marker_unchanged(raw):
    assert final_payload(raw) is None


def test_reference_failure_before_any_transport():
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_answer_domain_canary_v1_4_final.json').read_bytes())
    solver=BenchmarkSolver(MATHBenchmarkAdapterV2(),RequestBroker(contract=c,transport=lambda r:pytest.fail('unscorable reference dispatched'),arm='A1',seed=81))
    from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample
    example=LocalEvidenceExample(example_id='synthetic',input_payload='synthetic',gold='(1,2)')
    with pytest.raises(SearchContractError,match='REFERENCE_UNSCORABLE'):
        solver.evaluate('reason',example)


@pytest.mark.parametrize('field',['models','budget','aggregation','responsibility','initial_team_sha256','evaluator','verify_settings_sha256','membership_hashes'])
def test_unamended_science_and_domain_poison_rejected(field):
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_answer_domain_canary_v1_4_final.json').read_bytes())
    assert not MATHDomainBinding(ROOT,c).blockers()
    c[field]='poison'
    assert MATHDomainBinding(ROOT,c).blockers()


@pytest.mark.parametrize('arm',['A1','A2','A3','A4'])
@pytest.mark.parametrize('good,bad',CASES)
def test_actual_gepa_structured_four_arm_e2e(tmp_path,arm,good,bad):
    from test_math_preexecution import run_fake_arm
    run_fake_arm(tmp_path,arm,adapter=MATHBenchmarkAdapterV2(),gold=good,wrong=bad)


def test_reference_projection_cannot_be_search_context():
    from multi_dataset_diverse_rl.benchmarks.math_domain_prep import source_references
    with pytest.raises(ValueError,match='PREPARATION_CONTEXT_REQUIRED'):
        next(source_references(ROOT,expected_manifest_sha256='0'*64,context='search'))


def test_paired_validation_identical_requests_share_realization():
    from multi_dataset_diverse_rl.governance.math_paired_validation import evaluate_team,comparison
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_answer_domain_pilot_v1_4_final.json').read_bytes())
    seen=[]
    def fake(request):
        seen.append(request)
        return dict(text='FINAL_ANSWER: 1',input_tokens=3,output_tokens=2)
    broker=RequestBroker(contract=c,transport=fake,arm='A1',seed=81,validation_only=True)
    benchmark=MATHBenchmarkAdapterV2();solver=BenchmarkSolver(benchmark,broker)
    prompts=tuple('Synthetic reasoning '+str(i) for i in range(5))
    rows=[dict(stable_example_id='synthetic_'+str(i),problem='Synthetic one '+str(i),reference='1') for i in range(2)]
    initial=evaluate_team(rows,prompts,solver,benchmark,'validation_initial',lambda r:None)
    final=evaluate_team(rows,prompts,solver,benchmark,'validation_final',lambda r:None)
    assert initial==final and len(seen)==10
    result=comparison(initial,final)
    assert result['signal']=='NEUTRAL' and all(v['delta']==0 for v in result['metrics'].values())
    assert all(v==0 for v in result['paired'].values())
    with pytest.raises(SearchContractError,match='ROLE_SPLIT_FORBIDDEN'):
        broker.complete(role='solver',split='optimize',stage='search',messages=[])


def test_unscorable_reference_not_silently_wrong():
    from multi_dataset_diverse_rl.search.schemas import ParsedOutput
    with pytest.raises(SearchContractError,match='REFERENCE_UNSCORABLE'):
        MATHBenchmarkAdapterV2().score_member_output(ParsedOutput('',False),'(1,2)')


@pytest.mark.parametrize('failure',['MATH_EVALUATOR_FAILURE','MATH_EVALUATOR_TIMEOUT'])
def test_evaluator_operational_failure_not_prediction_wrong(monkeypatch,failure):
    import multi_dataset_diverse_rl.benchmarks.math_domain_v2 as domain
    def broken(expressions):
        raise SearchContractError(failure)
    monkeypatch.setattr(domain,'domain_matrix',broken)
    with pytest.raises(SearchContractError,match=failure):
        domain.MATHBenchmarkAdapterV2().parse_member_output('FINAL_ANSWER: 1',BenchmarkInput('synthetic','Synthetic one','FINAL_ANSWER',benchmark_id='math'))


def test_real_accounting_local_fault_bypasses_gepa_exception_handler(monkeypatch):
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.governance.token_accounting import OperationalAbort
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_answer_domain_canary_v1_4_final.json').read_bytes())
    broker=RequestBroker(contract=c,transport=lambda _:pytest.fail('transport must remain untouched'),arm='A1',seed=81,token_ledger=object())
    solver=BenchmarkSolver(MATHBenchmarkAdapterV2(),broker)
    def broken(*args):raise SearchContractError('MATH_EVALUATOR_TIMEOUT')
    monkeypatch.setattr(solver,'_evaluate',broken)
    caught=False
    with pytest.raises(OperationalAbort,match='MATH_EVALUATOR_TIMEOUT'):
        try:
            solver.evaluate('synthetic',SimpleNamespace(gold='1'))
        except Exception:
            caught=True
    assert not caught and broker.usage['attempts']==0


@pytest.mark.parametrize('phase',['canary','pilot'])
def test_v2_binding_actual_frozen_fake_execution(tmp_path,monkeypatch,phase):
    from test_math_token_accounting import test_actual_public_gepa_with_accounting_and_phase_persistence
    test_actual_public_gepa_with_accounting_and_phase_persistence(tmp_path,monkeypatch,phase,
        binding_path=f'experiments/execution_bindings/math_v2_answer_domain_{phase}_v1_4_final.json',adapter=MATHBenchmarkAdapterV2())
