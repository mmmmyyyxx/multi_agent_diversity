"""Fresh scoped client; timeout retains a private stack instead of a wrong score."""
from functools import lru_cache
import json,subprocess,sys
from .math_domain_v2 import SETTINGS
from .math_power_certificate import POLICY
from ..search.schemas import SearchContractError


@lru_cache(maxsize=2048)
def power_matrix(expressions):
    if not expressions or len(expressions)>6 or any(not isinstance(x,str) or
            len(x)>SETTINGS['maximum_expression_characters'] for x in expressions):
        raise SearchContractError('MATH_EXPRESSION_SHAPE_INVALID')
    try:
        result=subprocess.run([sys.executable,'-m','multi_dataset_diverse_rl.benchmarks.math_power_worker'],
            input=json.dumps(dict(expressions=expressions,policy=POLICY)),capture_output=True,text=True,timeout=30,check=False)
    except subprocess.TimeoutExpired as exc:
        error=SearchContractError('MATH_EVALUATOR_TIMEOUT')
        trace=exc.stderr or ''
        error.private_worker_trace=trace.decode('utf-8',errors='replace') if isinstance(trace,bytes) else trace
        raise error from exc
    except OSError as exc:raise SearchContractError('MATH_EVALUATOR_FAILURE') from exc
    if result.returncode:raise SearchContractError('MATH_EVALUATOR_FAILURE')
    try:
        value=json.loads(result.stdout);n=len(expressions)
        if (value['certificate_policy']!=POLICY['identity'] or len(value['valid'])!=n
            or any(type(v) is not bool for v in value['valid']) or len(value['equivalence'])!=n
            or any(len(row)!=n or any(type(v) is not bool for v in row) for row in value['equivalence'])
            or any(type(pair) is not list or len(pair)!=2 or any(type(v) is not int or not 0<=v<n for v in pair)
                for pair in value['certified_pair_rejections'])):
            raise SearchContractError('MATH_EVALUATOR_RESPONSE_INVALID')
        return value
    except (ValueError,KeyError,TypeError) as exc:raise SearchContractError('MATH_EVALUATOR_RESPONSE_INVALID') from exc
