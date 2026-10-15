"""Study inference ports reuse mathematical and durable accounting primitives.

This is a fresh study binding, not a forged current team policy bundle.
Provider construction is owned by governed execution after authorization.
"""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import threading
import time

from ..benchmarks.math import MATHBenchmarkAdapter
from ..benchmarks.math_flexible_answer import classify_prediction
from ..benchmarks.math_prediction_validity import resolve_predictions
from ..benchmarks.math_response_evidence import solver_profile, trajectory_policy
from ..governance.token_accounting import serialized_request, reservation, OperationalAbort
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..persistence.durable_io import append_jsonl
from ..search.schemas import SearchContractError
from ..search.system_prompt import solver_messages
from ..search.solver_execution import bounded_ordered_map
from ..search.textual_gradients import (
    PerExampleGradientProvider, GradientClusterProvider, REFERENCE_GRADIENT_PROMPT,
)
from ..search.optimizer_output_contract import parse_output
from ..search.generation_failures import generation_failure
from ..search.optimization_evidence import POLICY as EVIDENCE_POLICY
from .single_prompt_policy import policy


class StudyGradientProvider(PerExampleGradientProvider):
    """Reuse the exact Gradient input validator/extractor with a study broker."""
    def __init__(self, broker):
        self.broker = broker
        self.prompt = REFERENCE_GRADIENT_PROMPT
        self.calls = 0
        self.input_audit = []
        self.solver_trajectory_policy = trajectory_policy()
        # Extractor's guard contract is retained. Its team selection fields have
        # no read point in this provider or in the independent study loop.
        self.optimization_evidence_policy = deepcopy(EVIDENCE_POLICY)
        self.input_schema = 'STRUCTURED_SYSTEM_GRADIENT_INPUT_V8'
        self.numeric_guard_writer = None

    def independent_provider(self):
        return StudyGradientProvider(self.broker)

    def _call(self, payload):
        self.calls += 1
        result = self.broker.complete(role=self.role, split='optimize', stage=self.stage,
            messages=[dict(role='system', content=self.prompt),
                      dict(role='user', content=json.dumps(payload, sort_keys=True, ensure_ascii=True))])
        self.last_response = deepcopy(result)
        return parse_output(result['text'], stage='gradient')


class StudyClusterProvider(GradientClusterProvider):
    """Retain exact alias partition/abstraction/recovery; omit team ranking."""
    def _call(self, payload):
        self.calls += 1
        result = self.broker.complete(role=self.role, split='optimize', stage=self.stage,
            messages=[dict(role='system', content=self.prompt),
                      dict(role='user', content=json.dumps(payload, sort_keys=True, ensure_ascii=True))])
        self.last_response = deepcopy(result)
        try:
            return json.loads(result['text'])
        except (TypeError, ValueError) as exc:
            raise generation_failure('PATTERN_GRADIENT_CLUSTER_INVALID') from exc


class StudyBroker:
    def __init__(self, *, phase, scope, transport, ledger, receipts, dispatch_guard):
        self.phase = phase
        self.scope = scope
        self.transport = transport
        self.ledger = ledger
        self.receipts = receipts
        self.dispatch_guard = dispatch_guard
        self.execution_policy = dict(gradient_max_concurrency=self.study_policy()['gradient_concurrency'])
        self.lock = threading.RLock()
        self.cache = {}
        self.inflight = {}
        self.cell_id = None
        self.seed = None
        self.round_index = 0
        self.attempt_id = None
        self.usage = dict(provider_calls=0, solver=0, reflection=0, pattern_gradient=0,
                          pattern_cluster=0, input_tokens=0, output_tokens=0, cache_hits=0,
                          supplier_reported_tokens=0, fallback_usage_calls=0)
        for role in ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'):
            self.usage.update({role + '_input_tokens': 0, role + '_output_tokens': 0})
        self.replay = None
        self.timeout_recovery = self.timeout_policy()
        if self.timeout_recovery is not None:
            self.usage.update(scope.get('continuation', {}).get('inherited_usage', {}))
            self.usage.update(replayed_calls=0, timeout_retries=0)

    def usage_view(self):
        with self.lock:
            return dict(self.usage)

    def study_policy(self):
        return policy()

    def timeout_policy(self):
        return self.scope.get('continuation', {}).get('operation_policy')

    def legal_role_split(self, role, split):
        return (split == 'optimize' and role in ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster')
                if self.phase in ('train', 'canary') else split == self.phase and role == 'solver')

    def cache_ordinal(self, split):
        return self.round_index

    def recoverable_transport_error(self, exc):
        from openai import APITimeoutError
        return isinstance(exc, APITimeoutError)

    def budgeted_provider_calls(self):
        return self.usage['provider_calls']

    def record_recovery_failure(self):
        pass

    def _request_identity(self, *, role, split, messages, capacity=None):
        p = self.study_policy()
        legal = self.legal_role_split(role, split)
        if not legal or not self.attempt_id:
            raise SearchContractError('STUDY_PROVIDER_ROLE_SPLIT_FORBIDDEN')
        route = p['provider_routing_policy']['routes'][role]
        request = dict(model=route['model'], messages=messages)
        if role == 'solver':
            q = p['solver_decoding_policy']
            request.update(temperature=q['temperature'], top_p=q['top_p'],
                frequency_penalty=q['frequency_penalty'], presence_penalty=q['presence_penalty'],
                max_tokens=capacity or p['first_solver_capacity'])
        else:
            q = p['cluster_generation_policy'] if role == 'pattern_cluster' else p['optimizer_generation_policy']
            request.update(temperature=q['temperature'], top_p=q['top_p'],
                presence_penalty=q['presence_penalty'], frequency_penalty=q['frequency_penalty'],
                max_completion_tokens=q['max_completion_tokens'],
                extra_body=dict(enable_thinking=False, top_k=q['top_k']))
            estimate = len(serialized_request(request)) + 1024
            if estimate > p['optimizer_input_limit'] or estimate + q['accounting_output_ceiling'] > p['optimizer_context_limit']:
                raise SearchContractError('STUDY_FULL_EVIDENCE_CONTEXT_LIMIT')
        key = hashlib.sha256(serialized_request(request) + json.dumps([
            self.scope.get('continuation', {}).get('request_identity_scope', self.scope['scope_identity']),
            split, self.seed, self.cache_ordinal(split), role]).encode()).hexdigest()
        return request, key

    def complete(self, *, role, split, stage, messages, capacity=None):
        request, key = self._request_identity(role=role, split=split, messages=messages, capacity=capacity)
        bound = reservation(request)
        if self.replay is not None:
            from .single_prompt_continuation import replay_key, replay_guard
            with self.lock:
                self.dispatch_guard(self.cell_id)
                response = self.replay.take(replay_key(request,self.seed,self.round_index,role,stage),
                    replay_guard(request,self.seed,self.round_index,role,stage))
                if response is not None:
                    self.usage['replayed_calls'] += 1
            if response is not None:
                return self._validated_response(response, request, role, key, bound)
        timeout_count = 0
        while True:
            with self.lock:
                self.dispatch_guard(self.cell_id)
                if self.budgeted_provider_calls() >= self.scope['call_ceiling']:
                    raise OperationalAbort('STUDY_CALL_CEILING')
                if self.scope['token_ceiling'] is not None and self.ledger.totals['charged_total'] + self.ledger.reserved + bound['amount'] > self.scope['token_ceiling']:
                    raise OperationalAbort('STUDY_TOKEN_CEILING')
                reservation_id = self.ledger.reserve(request, attempt_id=self.attempt_id,
                    stage=stage, role=role, model=request['model'])
                receipt_store = ProviderResponseReceipts(self.receipts.directory,
                    attempt_id=self.attempt_id, startup_identity_sha256=self.receipts.startup)
                self.usage['provider_calls'] += 1
                self.usage[role] += 1
            logical = dict(seed=self.seed, ordinal=self.round_index, role=role, stage=stage)
            try:
                response = self.transport.for_role(role, request) if hasattr(self.transport, 'for_role') else self.transport(request)
            except BaseException as exc:
                from openai import APITimeoutError
                retry = self.timeout_recovery is not None and self.recoverable_transport_error(exc)
                evidence = dict(error_category=type(exc).__name__, request=request,
                                provider_evidence=getattr(exc, 'provider_evidence', None))
                if self.timeout_recovery is not None:
                    evidence['logical_identity'] = logical
                receipt = receipt_store.persist(reservation_id, evidence)
                charged = self.ledger.reconcile(reservation_id, getattr(exc, 'token_usage', {}),
                    outcome='PROVIDER_TIMEOUT' if retry else 'PROVIDER_FAILURE', response_receipt=receipt)
                self._usage(charged, role)
                if not retry:
                    raise OperationalAbort('STUDY_PROVIDER_FAILURE') from exc
                with self.lock:
                    self.record_recovery_failure()
                    self.usage['timeout_retries'] += 1
                    schedule = self.timeout_recovery['backoff_seconds']
                    delay = schedule[min(timeout_count, len(schedule)-1)]
                    append_jsonl(self.receipts.directory/'timeout_recovery.jsonl',
                        dict(request_sha256=key, reservation_id=reservation_id, logical_identity=logical,
                            backoff_seconds=delay, retry_index=timeout_count+1))
                timeout_count += 1
                # The durable failure receipt/charge precede the retry delay.
                # The finite cumulative call ceiling is checked before every send.
                time.sleep(delay)
                continue
            record = dict(request=request, response=response)
            if self.timeout_recovery is not None:
                response = dict(response, study_transport_timeout_attempts=timeout_count)
                record['response'] = response
                record['logical_identity'] = logical
            receipt = receipt_store.persist(reservation_id, record)
            charged = self.ledger.reconcile(reservation_id, response, outcome='PROVIDER_RESPONSE', response_receipt=receipt)
            self._usage(charged, role)
            return self._validated_response(response, request, role, key, bound)

    def _validated_response(self, response, request, role, key, bound):
        if not isinstance(response, dict) or not isinstance(response.get('text'), str):
            raise OperationalAbort('STUDY_PROVIDER_RESPONSE_INVALID')
        if type(response.get('output_tokens')) is int and response['output_tokens'] > bound['output_hard_cap']:
            raise OperationalAbort('STUDY_PROVIDER_OUTPUT_BOUND_EXCEEDED')
        if role != 'solver':
            self._check_nonthinking(request, response, role, key)
            if response.get('finish_reason') != 'stop' and role in ('pattern_gradient', 'pattern_cluster'):
                raise generation_failure('GENERATED_OUTPUT_TRUNCATED')
        return dict(response, request_sha256=key)

    def _check_nonthinking(self, request, response, role, key):
        from ..benchmarks.math_optimizer_diagnostics import optimizer_response_telemetry, nonthinking_evidence_contract
        p = self.study_policy()
        generation = p['cluster_generation_policy'] if role == 'pattern_cluster' else p['optimizer_generation_policy']
        audit = optimizer_response_telemetry(request, response, generation, nonthinking_evidence_contract())
        with self.lock:
            append_jsonl(self.receipts.directory / 'optimizer_telemetry.jsonl',
                         dict(request_sha256=key, role=role, cell_id=self.cell_id, telemetry=audit))
        if not audit['nonthinking_wire_confirmed']:
            raise OperationalAbort('STUDY_OPTIMIZER_NONTHINKING_UNCONFIRMED_OR_CONTRADICTORY')

    def _usage(self, charged, role):
        with self.lock:
            for k in ('input_tokens', 'output_tokens'):
                self.usage[k] += charged[k]
                self.usage[role + '_' + k] += charged[k]
            if charged['usage_reliable']:
                self.usage['supplier_reported_tokens'] += charged['input_tokens'] + charged['output_tokens']
            else:
                self.usage['fallback_usage_calls'] += 1

    def prediction(self, prompt, row, split, seed, ordinal):
        messages = solver_messages(prompt, row['content']['problem'])
        _, key = self._request_identity(role='solver', split=split, messages=messages)
        with self.lock:
            if key in self.cache:
                self.usage['cache_hits'] += 1
                return deepcopy(self.cache[key])
            future = self.inflight.get(key)
            owner = future is None
            if owner:
                future = self.inflight[key] = Future()
        if not owner:
            result = future.result()
            with self.lock:
                self.usage['cache_hits'] += 1
            return deepcopy(result)
        try:
            attempts = []
            transport_timeout_attempts = 0
            for draw in range(self.study_policy()['solver_semantic_draws']):
                capacity = (self.study_policy()['length_recovery_capacity'] if attempts and
                    attempts[-1].finish_reason in ('length', 'max_tokens', 'max_output_tokens') else self.study_policy()['first_solver_capacity'])
                response = self.complete(role='solver', split=split, stage=f'{split}_checkpoint_{ordinal}_draw_{draw + 1}',
                                         messages=messages, capacity=capacity)
                prediction = classify_prediction(response['text'], response.get('finish_reason'))
                transport_timeout_attempts += response.get('study_transport_timeout_attempts', 0)
                attempts.append(prediction)
                if prediction.prediction_valid:
                    break
            resolved = resolve_predictions(attempts, transport_retry_attempts=transport_timeout_attempts)
            correct = self.score_resolved(resolved, row)
            value = dict(prediction=asdict(resolved), valid=resolved.prediction_valid,
                         answer=resolved.answer, correct=correct, request_sha256=key,
                         task_type=row['content'].get('type', 'SYNTHETIC'),
                         outcome_category='CORRECT' if correct else 'VALID_WRONG' if resolved.prediction_valid
                            else 'INVALID_' + str(resolved.invalid_reason))
            if split == 'optimize':
                value['profile'] = solver_profile(dict(request_sha256=key), resolved, member_id=0,
                    prompt=prompt, example_id=row['stable_example_id'], split=split,
                    policy=trajectory_policy(), problem=row['content']['problem'])
            with self.lock:
                self.cache[key] = deepcopy(value)
                future.set_result(value)
            return value
        except BaseException as exc:
            with self.lock:
                future.set_exception(exc)
            raise
        finally:
            with self.lock:
                self.inflight.pop(key, None)

    def score_resolved(self, resolved, row):
        """Default mathematical scoring; scoped Test recovery may retry this boundary."""
        return bool(MATHBenchmarkAdapter().score_member_output(resolved.parsed(), row['reference_final_answer']))

    def evaluate(self, prompt, rows, split, seed, ordinal):
        if not rows or seed != self.seed:
            raise SearchContractError('STUDY_EVALUATION_MEMBERSHIP_INVALID')
        self.round_index = ordinal
        values = bounded_ordered_map(lambda row: self.prediction(prompt, row, split, seed, ordinal),
                                    rows, self.study_policy()['solver_concurrency'])
        return {r['stable_example_id']: value for r, value in zip(rows, values)}
