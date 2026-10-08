"""Private, provenance-bound ordinary response text; never latent reasoning.

The prediction retains every complete original response. Only this deterministic
bounded projection crosses the Optimize feedback boundary. No answer is guessed
from a malformed response, and none of these fields affects mathematical scoring.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import re

from .. import versions
from ..search.schemas import SearchContractError
from .math_domain_v2 import final_payload
from .math_prediction_validity import prediction_from_persisted


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def trajectory_policy():
    return dict(identity=versions.MATH_VISIBLE_TRAJECTORY_POLICY_VERSION,
        profile_schema=versions.MATH_SOLVER_PROFILE_VERSION,
        trajectory_schema=versions.MATH_SOLVER_TRAJECTORY_VERSION,
        gradient_input_schema=versions.GRADIENT_VISIBLE_INPUT_VERSION,
        gradient_prompt_identity=versions.GRADIENT_VISIBLE_PROMPT_VERSION,
        optimizer_input_schema=versions.GRADIENT_VISIBLE_OPTIMIZER_INPUT_VERSION,
        max_visible_solution_characters=4096, truncation='unicode_prefix_explicit_metadata',
        original_response_retention='complete_private_resolved_prediction_and_attempts',
        source='ordinary_assistant_content', adaptive_split='optimize',
        hidden_reasoning_claim=False, memory_trajectory_writes=False)


def frozen_trajectory_policy(contract):
    policy = contract.get('solver_trajectory_policy')
    if contract.get('identity') in {versions.MATH_VISIBLE_TRAJECTORY_BINDING_VERSION, versions.MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION}:
        if contract.get('identity')==versions.MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION:
            from ..search.optimization_evidence import frozen_policy
            if frozen_policy(contract.get('optimization_evidence_policy')) is None:
                raise SearchContractError('OPTIMIZATION_EVIDENCE_POLICY_REQUIRED')
        from .math_v21_interface import v6_interface_contract
        expected = trajectory_policy()
        if (policy != expected or any(type(policy.get(k)) is not type(v) for k, v in expected.items())
                or contract.get('solver_output_interface') != v6_interface_contract()
                or contract.get('models', {}).get('solver') != 'qwen3-8b'
                or contract.get('models', {}).get('solver_thinking') is not False):
            raise SearchContractError('MATH_VISIBLE_TRAJECTORY_BINDING_MISMATCH')
        return deepcopy(policy)
    if policy is not None or contract.get('solver_output_interface', {}).get('identity') == versions.MATH_SOLVER_INTERFACE_V6_VERSION:
        raise SearchContractError('MATH_VISIBLE_TRAJECTORY_REQUIRES_FRESH_BINDING')
    return None


def _projection(prediction, source):
    raw = prediction.text or ''
    # This is the existing boundary parser, not a new answer fallback. With an
    # invalid boundary expose the response as unsegmented, explicitly incomplete
    # written evidence; its marker text is never scored as an answer.
    boundary_valid = final_payload(raw) is not None
    lines = raw.splitlines(keepends=True)
    if boundary_valid:
        marker_index = max(i for i, line in enumerate(lines) if line.strip())
        solution = ''.join(lines[:marker_index]).rstrip()
    else:
        solution = raw
    truncated = prediction.finish_reason in {'length', 'max_tokens', 'max_output_tokens'}
    status = ('RESPONSE_TRUNCATED' if truncated else 'FINAL_BOUNDARY_INVALID'
        if not boundary_valid else 'WRITTEN_SOLUTION_PRESENT' if solution.strip()
        else 'WRITTEN_SOLUTION_MISSING')
    limit = trajectory_policy()['max_visible_solution_characters']
    return dict(schema=versions.MATH_SOLVER_TRAJECTORY_VERSION,
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
        solver_interface_identity=versions.MATH_SOLVER_INTERFACE_V6_VERSION)
    return dict(schema=versions.MATH_SOLVER_PROFILE_VERSION,
        prediction=asdict(prediction), solver_trajectory=_projection(prediction, source))


def profile_prediction(profile, *, example_id=None):
    if (not isinstance(profile, dict) or set(profile) != {'schema', 'prediction', 'solver_trajectory'}
            or profile['schema'] != versions.MATH_SOLVER_PROFILE_VERSION):
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
            or source['solver_interface_identity'] != versions.MATH_SOLVER_INTERFACE_V6_VERSION
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
    # the bounded projection when consumed independently by a feedback provider.
    source = trajectory.get('source') if isinstance(trajectory, dict) else None
    expected_keys = {'schema', 'source', 'visible_solution', 'solution_status', 'response_truncated',
        'feedback_truncated', 'original_solution_characters', 'exposed_solution_characters',
        'raw_response_sha256', 'prediction_identity_sha256', 'finish_reason', 'prediction_valid',
        'invalid_reason', 'written_text_is_hidden_reasoning'}
    source_keys = {'member_id', 'mutable_prompt_sha256', 'benchmark_input_sha256', 'example_id',
        'split', 'request_sha256', 'solver_interface_identity'}
    if (not isinstance(trajectory, dict) or set(trajectory) != expected_keys
            or not isinstance(source, dict) or set(source) != source_keys
            or type(source['member_id']) is not int or source['member_id'] not in range(5)
            or source['split'] != 'optimize' or source['example_id'] != example_id
            or source['solver_interface_identity'] != versions.MATH_SOLVER_INTERFACE_V6_VERSION
            or any(not isinstance(source[k], str) or re.fullmatch('[0-9a-f]{64}', source[k]) is None
                for k in ('mutable_prompt_sha256', 'benchmark_input_sha256', 'request_sha256'))
            or trajectory.get('schema') != versions.MATH_SOLVER_TRAJECTORY_VERSION
            or not isinstance(trajectory.get('visible_solution'), str)
            or len(trajectory['visible_solution']) > trajectory_policy()['max_visible_solution_characters']
            or type(trajectory.get('feedback_truncated')) is not bool
            or type(trajectory.get('response_truncated')) is not bool
            or type(trajectory.get('prediction_valid')) is not bool
            or trajectory.get('written_text_is_hidden_reasoning') is not False
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
                'WRITTEN_SOLUTION_PRESENT', 'WRITTEN_SOLUTION_MISSING'}
            or member_id is not None and trajectory['source'].get('member_id') != member_id
            or prompt is not None and source['mutable_prompt_sha256'] != text_hash(prompt)
            or problem is not None and source['benchmark_input_sha256'] != text_hash(
                problem['problem'] if isinstance(problem, dict) else problem)):
        raise SearchContractError('MATH_TRAJECTORY_ADAPTIVE_PROVENANCE_MISMATCH')
    return deepcopy(trajectory)
