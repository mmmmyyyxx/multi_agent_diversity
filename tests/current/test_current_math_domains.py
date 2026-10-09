"""Preserved mathematical scoring and reference-boundary physics on V2.4."""
import pytest
from pathlib import Path
from tests.current.test_structured_optimization_evidence import contract
from multi_dataset_diverse_rl.benchmarks.math_structured_interface import MATHStructuredSystemBenchmark
from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import domain_matrix,require_scorable,final_payload
from multi_dataset_diverse_rl.search.benchmark import BenchmarkInput
from multi_dataset_diverse_rl.search.schemas import SearchContractError
ROOT=Path(__file__).resolve().parents[2]
CASES=[('(1,2,3)','(3,2,1)'),(r'\{1,2\}',r'\{1,3\}'),('[1,2]','[1,2)'),
       ('1,2,3','1,2,4'),('1','2'),('x+1','x+2')]
def MATHBenchmarkAdapterV2():return MATHStructuredSystemBenchmark(contract())
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


@pytest.mark.parametrize('raw',['unidentified prose','###','### 1\nmore','### 1\n### 2'])
def test_ambiguous_final_boundary_is_invalid(raw):
    assert final_payload(raw) is None


def test_reference_projection_cannot_be_search_context():
    from multi_dataset_diverse_rl.benchmarks.math_domain_prep import source_references
    with pytest.raises(ValueError,match='PREPARATION_CONTEXT_REQUIRED'):
        next(source_references(ROOT,expected_manifest_sha256='0'*64,context='search'))


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
        domain.MATHBenchmarkAdapterV2().parse_member_output('### 1',BenchmarkInput('synthetic','Synthetic one','FINAL_ANSWER',benchmark_id='math'))



def test_unscorable_reference_rejected_before_actual_solver_port():
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver
    from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample
    calls=[];c=contract()
    solver=BenchmarkSolver(MATHStructuredSystemBenchmark(c),RequestBroker(contract=c,arm='A4',seed=81,transport=lambda r:calls.append(r)))
    with pytest.raises(SearchContractError,match='REFERENCE_UNSCORABLE'):
        solver.evaluate(__import__('multi_dataset_diverse_rl.search.system_prompt',fromlist=['SEED']).SEED,LocalEvidenceExample(example_id='synthetic',input_payload='Synthetic problem.',gold='(1,2)'))
    assert not calls
