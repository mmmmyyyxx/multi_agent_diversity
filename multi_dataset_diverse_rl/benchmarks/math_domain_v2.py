"""Legacy path: current correctness plus separate unchanged vote equivalence."""
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
import json
import subprocess
import sys
from ..search.schemas import SearchContractError
from ..search.schemas import ParsedOutput, BenchmarkCapabilities
from .protocols import MATH_PROTOCOL_HENDRYCKS
from ..math_scoring import CONTRACT_ID

SETTINGS = dict(identity='MATH_VERIFY_SETTINGS_V2',strict=True,float_rounding=6,
    numeric_precision=15,parsing_timeout=3,verification_timeout=3,process_deadline=8,
    fallback_mode='no_fallback',extraction_mode='first_match',
    latex_extraction=dict(try_extract_without_anchor=True,boxed_match_priority=50,
        normalization_config=dict(basic_latex=True,units=True,malformed_operators=True,nits=True,boxed='all',equations=False)),
    conversion_config=dict(interpret_as_mixed_fractions=True,interpret_simple_eq_as_assignment=False,interpret_contains_as_eq=True),
    allow_set_relation_comp='NOT_EXPOSED_BY_PINNED_API; DISJOINT_CONTAINER_TYPES_REQUIRED',
    two_component_parentheses='AMBIGUOUS_REFERENCE_UNSCORABLE_AFTER_NATIVE_PARSE',
    correctness_direction='gold_first_prediction_second')
SETTINGS['object_families']=['expression','relation','tuple','matrix','finite_set','infinite_set']
SETTINGS['maximum_expression_characters']=2048

_WORKER_DEADLINE=ContextVar('math_worker_deadline',default=None)


@contextmanager
def worker_deadline(seconds):
    """Explicit caller-scoped operational deadline; default mathematical policy is unchanged."""
    if type(seconds) is not int or seconds<SETTINGS['process_deadline']:
        raise SearchContractError('MATH_WORKER_DEADLINE_INVALID')
    token=_WORKER_DEADLINE.set(seconds)
    try:yield
    finally:_WORKER_DEADLINE.reset(token)


def final_payload(raw):
    from .math_structured_answer import extract_answer
    answer,reason=extract_answer(raw)
    return answer if reason is None else None


@lru_cache(maxsize=2048)
def domain_matrix(expressions):
    if not expressions or len(expressions)>6 or any(not isinstance(x,str) or len(x)>SETTINGS['maximum_expression_characters'] for x in expressions):
        raise SearchContractError('MATH_EXPRESSION_SHAPE_INVALID')
    try:
        result=subprocess.run([sys.executable,'-m','multi_dataset_diverse_rl.benchmarks.math_domain_worker'],
            input=json.dumps({'expressions':expressions}),capture_output=True,text=True,timeout=_WORKER_DEADLINE.get() or SETTINGS['process_deadline'],check=False)
        if result.returncode:
            raise SearchContractError('MATH_EVALUATOR_FAILURE')
        data=json.loads(result.stdout)
        if (len(data['valid'])!=len(expressions) or len(data['equivalence'])!=len(expressions)
                or any(len(row)!=len(expressions) for row in data['equivalence'])
                or any(type(v)!=bool for v in data['valid'])
                or any(type(v)!=bool for row in data['equivalence'] for v in row)):
            raise SearchContractError('MATH_EVALUATOR_RESPONSE_INVALID')
        return data
    except subprocess.TimeoutExpired as exc:
        raise SearchContractError('MATH_EVALUATOR_TIMEOUT') from exc
    except (ValueError,KeyError,OSError) as exc:
        raise SearchContractError('MATH_EVALUATOR_FAILURE') from exc


def require_scorable(reference):
    from ..math_scoring import prepare_ground_truth
    prepare_ground_truth(reference)


class MATHBenchmarkAdapterV2:
    benchmark_id='math'
    preferred_aggregation='equivalence_plurality'
    capabilities=BenchmarkCapabilities(True,True,True,True,True)
    output_contract='hendrycks_last_boxed_answer_v1'
    protocol=MATH_PROTOCOL_HENDRYCKS
    parser_identity=CONTRACT_ID+'/last_boxed'
    evaluator_identity=CONTRACT_ID
    final_payload=staticmethod(final_payload)
    require_scorable=staticmethod(require_scorable)

    def format_input(self,item):return item.problem

    def build_task_feedback(self,parsed,gold):
        return 'correct' if self.score_member_output(parsed,gold) else 'incorrect'

    def parse_member_output(self,raw,item):
        from .math_flexible_answer import classify_prediction
        if item.benchmark_id!=self.benchmark_id:
            return ParsedOutput('',False)
        return classify_prediction(raw).parsed()

    parse_output=parse_member_output

    def equivalent(self,left,right):
        result=domain_matrix((left,right))
        if result['equivalence'][0][1]!=result['equivalence'][1][0]:
            raise SearchContractError('EQUIVALENCE_RELATION_INCONSISTENT')
        return all(result['valid']) and result['equivalence'][0][1]

    def score_member_output(self,parsed,gold):
        from ..math_scoring import score_parsed_answer
        return float(score_parsed_answer(parsed.answer, parsed.valid, gold).score)
