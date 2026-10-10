"""Private, provenance-bound ordinary response text; never latent reasoning.

The prediction retains every complete original response. Complete ordinary content
crosses the Optimize feedback boundary with an optional bounded solution view. No answer is guessed
from a malformed response, and none of these fields affects mathematical scoring.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import re

from .. import versions
from ..search.schemas import SearchContractError

from .math_flexible_answer import prediction_from_persisted, extract_answer, boundaries


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode()).hexdigest()


def text_hash(text):
    from ..search.system_prompt import SystemPrompt
    if isinstance(text, SystemPrompt):
        return text.prompt_hash
    return hashlib.sha256(text.encode()).hexdigest()


def trajectory_policy():
    return dict(identity='MATH_FLEXIBLE_OBSERVED_RESPONSE_POLICY_V4',
        profile_schema=versions.MATH_FLEXIBLE_PROFILE_VERSION,
        trajectory_schema=versions.MATH_FLEXIBLE_RESPONSE_VERSION,
        gradient_input_schema='STRUCTURED_SYSTEM_GRADIENT_INPUT_V5',
        gradient_prompt_identity='STRUCTURED_SYSTEM_GRADIENT_PROMPT_V7',
        optimizer_input_schema='STRUCTURED_SYSTEM_PATTERN_EDIT_INPUT_V8',
        max_visible_solution_characters=4096, observed_response='complete_ordinary_content',
        truncation='optional_solution_unicode_prefix_explicit_metadata',
        original_response_retention='complete_private_resolved_prediction_and_attempts',
        source='ordinary_assistant_content', adaptive_split='optimize',
        visible_trajectory_required=False, final_only='TRAJECTORY_UNAVAILABLE',
        hidden_reasoning_claim=False, memory_trajectory_writes=False)


def frozen_trajectory_policy(contract):
    policy = contract.get('solver_trajectory_policy')
    if contract.get('identity')==versions.MATH_FLEXIBLE_ANSWER_BINDING_VERSION:
        from ..search.optimization_evidence import frozen_policy
        frozen_policy(contract.get('optimization_evidence_policy'))
        from .math_structured_interface import system_interface_contract
        expected = trajectory_policy()
        if (policy != expected or any(type(policy.get(k)) is not type(v) for k, v in expected.items())
                or contract.get('solver_output_interface') != system_interface_contract(contract.get('initial_team_version'))
                or contract.get('models', {}).get('solver') != 'qwen3-8b'
                or contract.get('models', {}).get('solver_thinking') is not False):
            raise SearchContractError('MATH_VISIBLE_TRAJECTORY_BINDING_MISMATCH')
        return deepcopy(policy)
    if policy is not None or contract.get('solver_output_interface', {}).get('identity') == versions.MATH_SOLVER_INTERFACE_V9_VERSION:
        raise SearchContractError('MATH_VISIBLE_TRAJECTORY_REQUIRES_FRESH_BINDING')
    raise SearchContractError('CURRENT_VISIBLE_TRAJECTORY_REQUIRED')


def _projection(prediction, source):
    raw = prediction.text or ''
    # This is the existing boundary parser, not a new answer fallback. With an
    # invalid boundary expose the response as unsegmented, explicitly incomplete
    # written evidence; its marker text is never scored as an answer.
    boundary_valid = extract_answer(raw)[1] is None
    records = boundaries(raw) if boundary_valid else None
    terminal = [r for r in (records or ()) if r[3]]
    solution = raw[:min(r[1] for r in terminal)].rstrip() if terminal else '' if boundary_valid else raw
    solution = re.sub(r'(?im)(?:^|\n)\s*(?:###|Final answer:)\s*[$]*$', '', solution).rstrip()
    solution = re.sub(r'(?im)(?:^|\n)\s*(?:#{1,6}\s+)?(?:\*\*|__)?(?:Final[ _]+answer|Answer)\s*[:=]\s*(?:\*\*|__)?\s*$', '', solution).rstrip()
    if re.fullmatch(r'\s*(?:\$+|\\[([]|(?:Final answer:|Thus[:,]?|Therefore[:,]?|The answer is)\s*)*', solution, re.I):
        solution = ''
    truncated = prediction.finish_reason in {'length', 'max_tokens', 'max_output_tokens'}
    status = ('RESPONSE_TRUNCATED' if truncated else 'FINAL_BOUNDARY_INVALID'
        if not boundary_valid else 'WRITTEN_SOLUTION_PRESENT' if solution.strip()
        else 'TRAJECTORY_UNAVAILABLE')
    limit = trajectory_policy()['max_visible_solution_characters']
    attempts=getattr(prediction,'original_predictions',(prediction,))
    reasons=[p.invalid_reason for p in attempts]
    retry_summary=dict(semantic_attempt_count=len(attempts),invalid_reasons=reasons,
        repeated_format_failure=len(attempts)==4 and all(not p.prediction_valid for p in attempts) and len(set(reasons))==1,
        distinct_response_count=len({p.text for p in attempts}))
    return dict(schema=versions.MATH_FLEXIBLE_RESPONSE_VERSION,retry_summary=retry_summary,
        observed_response=raw,observed_response_truncated=False,
        original_response_characters=len(raw),
        source=deepcopy(source), visible_solution=solution[:limit], solution_status=status,
        response_truncated=truncated, feedback_truncated=len(solution) > limit,
        original_solution_characters=len(solution), exposed_solution_characters=min(len(solution), limit),
        raw_response_sha256=text_hash(raw), prediction_identity_sha256=digest(asdict(prediction)),
        finish_reason=prediction.finish_reason, prediction_valid=prediction.prediction_valid,
        invalid_reason=prediction.invalid_reason, written_text_is_hidden_reasoning=False)


def solver_profile(result, prediction, *, member_id, prompt, example_id, split, policy, problem):
    if policy != trajectory_policy() or type(member_id) is not int or member_id not in range(5):
        raise SearchContractError('MATH_TRAJECTORY_PROVENANCE_REQUIRED')
    request_hash = result.get('request_sha256')
    if (not isinstance(request_hash, str) or len(request_hash) != 64
            or split not in {'optimize', 'shadow', 'validation'}):
        raise SearchContractError('MATH_TRAJECTORY_PROVENANCE_REQUIRED')
    source = dict(member_id=member_id, mutable_prompt_sha256=text_hash(prompt),
        benchmark_input_sha256=text_hash(problem),
        example_id=example_id, split=split, request_sha256=request_hash,
        solver_interface_identity=versions.MATH_SOLVER_INTERFACE_V9_VERSION)
    return dict(schema=versions.MATH_FLEXIBLE_PROFILE_VERSION,
        prediction=asdict(prediction), solver_trajectory=_projection(prediction, source))


def profile_prediction(profile, *, example_id=None):
    if (not isinstance(profile, dict) or set(profile) != {'schema', 'prediction', 'solver_trajectory'}
            or profile['schema'] != versions.MATH_FLEXIBLE_PROFILE_VERSION):
        raise SearchContractError('MATH_VISIBLE_PROFILE_SCHEMA_INVALID')
    prediction = prediction_from_persisted(profile['prediction'])
    trajectory = profile['solver_trajectory']
    source = trajectory.get('source') if isinstance(trajectory, dict) else None
    expected_keys = {'member_id', 'mutable_prompt_sha256', 'benchmark_input_sha256', 'example_id', 'split',
        'request_sha256', 'solver_interface_identity'}
    if (not isinstance(source, dict) or set(source) != expected_keys
            or type(source['member_id']) is not int or source['member_id'] not in range(5)
            or not isinstance(source['example_id'], str) or not source['example_id']
            or source['split'] not in {'optimize', 'shadow', 'validation'}
            or source['solver_interface_identity'] != versions.MATH_SOLVER_INTERFACE_V9_VERSION
            or any(not isinstance(source[k], str) or re.fullmatch('[0-9a-f]{64}', source[k]) is None
                for k in ('mutable_prompt_sha256', 'benchmark_input_sha256', 'request_sha256'))
            or example_id is not None and source['example_id'] != example_id
            or digest(trajectory) != digest(_projection(prediction, source))):
        raise SearchContractError('MATH_VISIBLE_PROFILE_PROVENANCE_MISMATCH')
    return prediction


def adaptive_trajectory(profile, *, member_id, prompt, example_id, problem=None):
    profile_prediction(profile, example_id=example_id)
    trajectory = profile['solver_trajectory']
    source = trajectory['source']
    if (source['split'] != 'optimize' or source['member_id'] != member_id
            or source['mutable_prompt_sha256'] != text_hash(prompt)
            or problem is not None and source['benchmark_input_sha256'] != text_hash(problem)):
        raise SearchContractError('MATH_TRAJECTORY_ADAPTIVE_PROVENANCE_MISMATCH')
    return deepcopy(trajectory)


def validate_adaptive_trajectory(trajectory, *, example_id, member_id=None, prompt=None, problem=None):
    # Full profile validation occurs at the Solver/state boundary. This guards
    # the lossless response and bounded optional view at the feedback provider.
    source = trajectory.get('source') if isinstance(trajectory, dict) else None
    retry=trajectory.get('retry_summary',{}) if isinstance(trajectory,dict) else {}
    if (not isinstance(retry,dict) or set(retry)!={'semantic_attempt_count','invalid_reasons','repeated_format_failure','distinct_response_count'}
            or type(retry['semantic_attempt_count']) is not int or not 1<=retry['semantic_attempt_count']<=4
            or not isinstance(retry['invalid_reasons'],list) or len(retry['invalid_reasons'])!=retry['semantic_attempt_count']
            or any(r is not None and not isinstance(r,str) for r in retry['invalid_reasons'])
            or type(retry['distinct_response_count']) is not int or not 1<=retry['distinct_response_count']<=retry['semantic_attempt_count']
            or type(retry['repeated_format_failure']) is not bool
            or retry['repeated_format_failure']!=(retry['semantic_attempt_count']==4 and
                all(r is not None for r in retry['invalid_reasons']) and len(set(retry['invalid_reasons']))==1)):
        raise SearchContractError('MATH_TRAJECTORY_ADAPTIVE_PROVENANCE_MISMATCH')
    expected_keys = {'schema', 'source', 'visible_solution', 'solution_status', 'response_truncated','retry_summary',
        'observed_response','observed_response_truncated','original_response_characters',
        'feedback_truncated', 'original_solution_characters', 'exposed_solution_characters',
        'raw_response_sha256', 'prediction_identity_sha256', 'finish_reason', 'prediction_valid',
        'invalid_reason', 'written_text_is_hidden_reasoning'}
    source_keys = {'member_id', 'mutable_prompt_sha256', 'benchmark_input_sha256', 'example_id',
        'split', 'request_sha256', 'solver_interface_identity'}
    if (not isinstance(trajectory, dict) or set(trajectory) != expected_keys
            or not isinstance(source, dict) or set(source) != source_keys
            or type(source['member_id']) is not int or source['member_id'] not in range(5)
            or source['split'] != 'optimize' or source['example_id'] != example_id
            or source['solver_interface_identity'] != versions.MATH_SOLVER_INTERFACE_V9_VERSION
            or any(not isinstance(source[k], str) or re.fullmatch('[0-9a-f]{64}', source[k]) is None
                for k in ('mutable_prompt_sha256', 'benchmark_input_sha256', 'request_sha256'))
            or trajectory.get('schema') != versions.MATH_FLEXIBLE_RESPONSE_VERSION
            or not isinstance(trajectory.get('visible_solution'), str)
            or not isinstance(trajectory.get('observed_response'),str)
            or type(trajectory.get('observed_response_truncated')) is not bool
            or type(trajectory.get('original_response_characters')) is not int
            or trajectory['original_response_characters']<0
            or len(trajectory['observed_response'])!=trajectory['original_response_characters']
            or trajectory['observed_response_truncated'] is not False
            or text_hash(trajectory['observed_response'])!=trajectory['raw_response_sha256']
            or len(trajectory['visible_solution']) > trajectory_policy()['max_visible_solution_characters']
            or type(trajectory.get('feedback_truncated')) is not bool
            or type(trajectory.get('response_truncated')) is not bool
            or type(trajectory.get('prediction_valid')) is not bool
            or trajectory.get('written_text_is_hidden_reasoning') is not False
            or not isinstance(trajectory.get('retry_summary'),dict)
            or set(trajectory['retry_summary'])!={'semantic_attempt_count','invalid_reasons','repeated_format_failure','distinct_response_count'}
            or type(trajectory['retry_summary']['repeated_format_failure']) is not bool
            or type(trajectory.get('original_solution_characters')) is not int
            or type(trajectory.get('exposed_solution_characters')) is not int
            or trajectory['exposed_solution_characters'] != len(trajectory['visible_solution'])
            or trajectory['exposed_solution_characters'] != min(trajectory['original_solution_characters'],
                trajectory_policy()['max_visible_solution_characters'])
            or trajectory['original_solution_characters'] < trajectory['exposed_solution_characters']
            or trajectory['feedback_truncated'] != (trajectory['original_solution_characters'] > trajectory_policy()['max_visible_solution_characters'])
            or any(not isinstance(trajectory[k], str) or re.fullmatch('[0-9a-f]{64}', trajectory[k]) is None
                for k in ('raw_response_sha256', 'prediction_identity_sha256'))
            or trajectory.get('solution_status') not in {'RESPONSE_TRUNCATED', 'FINAL_BOUNDARY_INVALID',
                'WRITTEN_SOLUTION_PRESENT', 'TRAJECTORY_UNAVAILABLE'}
            or member_id is not None and trajectory['source'].get('member_id') != member_id
            or prompt is not None and source['mutable_prompt_sha256'] != text_hash(prompt)
            or problem is not None and source['benchmark_input_sha256'] != text_hash(
                problem['problem'] if isinstance(problem, dict) else problem)):
        raise SearchContractError('MATH_TRAJECTORY_ADAPTIVE_PROVENANCE_MISMATCH')
    return deepcopy(trajectory)
