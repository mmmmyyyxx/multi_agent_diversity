"""Versioned mathematical payload support; FINAL_ANSWER framing stays separate."""
from functools import lru_cache
import json
import subprocess
import sys
from ..search.schemas import SearchContractError
from .math import MATHBenchmarkAdapter
from ..search.schemas import ParsedOutput
from .protocols import MATH_PROTOCOL_V2

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


def final_payload(raw):
    lines=[line.strip() for line in raw.splitlines() if line.strip()]
    marked=[line for line in lines if line.startswith('FINAL_ANSWER:')]
    if len(marked)!=1 or not lines or lines[-1]!=marked[0]:
        return None
    return marked[0].removeprefix('FINAL_ANSWER:').strip() or None


@lru_cache(maxsize=2048)
def domain_matrix(expressions):
    if not expressions or len(expressions)>6 or any(not isinstance(x,str) or len(x)>SETTINGS['maximum_expression_characters'] for x in expressions):
        raise SearchContractError('MATH_EXPRESSION_SHAPE_INVALID')
    try:
        result=subprocess.run([sys.executable,'-m','multi_dataset_diverse_rl.benchmarks.math_domain_worker'],
            input=json.dumps({'expressions':expressions}),capture_output=True,text=True,timeout=SETTINGS['process_deadline'],check=False)
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
    if not isinstance(reference,str) or not reference.strip():
        raise SearchContractError('REFERENCE_UNSCORABLE')
    try:
        result=domain_matrix((reference,))
        if not result['valid'][0] or not result['equivalence'][0][0]:
            raise SearchContractError('REFERENCE_UNSCORABLE')
    except SearchContractError as exc:
        raise SearchContractError('REFERENCE_UNSCORABLE') from exc


class MATHBenchmarkAdapterV2(MATHBenchmarkAdapter):
    protocol=MATH_PROTOCOL_V2
    parser_identity='MATH_PAYLOAD_PARSER_V2'
    evaluator_identity='MATH_EQUIVALENCE_V2'
    final_payload=staticmethod(final_payload)
    require_scorable=staticmethod(require_scorable)

    def parse_member_output(self,raw,item):
        expression=final_payload(raw)
        if item.benchmark_id!=self.benchmark_id or expression is None:
            return ParsedOutput('',False)
        valid=domain_matrix((expression,))['valid'][0]
        return ParsedOutput(expression,valid)

    parse_output=parse_member_output

    def equivalent(self,left,right):
        result=domain_matrix((left,right))
        if result['equivalence'][0][1]!=result['equivalence'][1][0]:
            raise SearchContractError('EQUIVALENCE_RELATION_INCONSISTENT')
        return all(result['valid']) and result['equivalence'][0][1]

    def score_member_output(self,parsed,gold):
        require_scorable(gold)
        if not parsed.valid:
            return 0.0
        # Correctness is directed. Aggregation uses the separate symmetric seam.
        result=domain_matrix((str(gold),parsed.answer))
        if not result['valid'][0]:
            raise SearchContractError('REFERENCE_UNSCORABLE')
        return float(result['valid'][1] and result['equivalence'][0][1])
