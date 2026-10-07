"""Versioned role accounting and inference-only requests, with isolated caches."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import threading
import time

from ..local_optimizers.base import LocalSolverObservation
from ..benchmarks.protocols import PROTOCOLS
from .benchmark import BenchmarkInput
from .schemas import SearchContractError
from ..benchmarks.math_solver_decoding import generation_request_fields, frozen_solver_policy
from ..benchmarks.math_prediction_validity import frozen_prediction_policy, frozen_recovery_policy
from ..benchmarks.math_optimizer_generation import frozen_optimizer_policy
from .. import versions
from ..current_contract import PATTERN_SPECIFIC_CONTENT_GUARD_VERSION as CURRENT_CONTENT_GUARD


class RequestBroker:
    def __init__(self, *, contract, transport, arm, seed, ledger_writer=None, raw_writer=None, usage=None, lock=None,
                 token_ledger=None, reserve_reader=None, validation_only=False, durable_cache=None,
                 response_receipts=None):
        self.contract = contract
        self.transport = transport
        self.arm = arm
        self.seed = seed
        self.ledger_writer = ledger_writer
        self.raw_writer = raw_writer
        self.response_receipts = response_receipts
        if response_receipts is not None and token_ledger is None:
            raise SearchContractError("PROVIDER_RECEIPTS_REQUIRE_DURABLE_ACCOUNTING")
        self.cache = {}
        self.cache_seals = {}
        self.usage = usage if usage is not None else dict(attempts=0, successes=0, failures=0, input_tokens=0, output_tokens=0,
            solver=0, reflection=0, pattern=0, validation=0, test=0)
        self.lock = lock if lock is not None else threading.RLock()
        self.token_ledger = token_ledger
        self.reserve_reader = reserve_reader or (lambda: 0)
        self.validation_only = validation_only
        self.prediction_policy = frozen_prediction_policy(contract)
        self.recovery_policy = frozen_recovery_policy(contract)
        self.optimizer_policy = frozen_optimizer_policy(contract)
        self.gradient_pattern = contract.get('pattern_policy',{}).get('discovery') == versions.GRADIENT_PATTERN_DISCOVERY_VERSION
        from .gradient_recovery import validate_policy
        validate_policy(contract.get('gradient_recovery_policy'))
        if self.gradient_pattern:
            from .textual_gradients import POLICY
            if (contract.get('identity') not in {versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION, versions.MATH_V2_2_EXECUTION_BINDING_VERSION}
                    or contract['pattern_policy']!=POLICY
                    or contract.get('pattern_abstraction_guard')!=CURRENT_CONTENT_GUARD
                    or not contract.get('gradient_prompt_sha256')):
                raise SearchContractError('GRADIENT_PATTERN_PROVIDER_BINDING_MISMATCH')
            for role in ('pattern_gradient','pattern_cluster'):self.usage.setdefault(role,0)
        self.member_lane_policy = contract.get('cache_policy') == versions.SOLVER_MEMBER_LANE_CACHE_VERSION
        if self.member_lane_policy != (contract.get('identity') in {versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION, versions.MATH_V2_2_EXECUTION_BINDING_VERSION}):
            raise SearchContractError('SOLVER_MEMBER_LANE_POLICY_BINDING_MISMATCH')
        self.durable_cache = durable_cache
        if durable_cache is not None and self.recovery_policy is None:
            raise SearchContractError('DURABLE_CACHE_REQUIRES_RECOVERY_BINDING')
        if durable_cache is not None:
            from ..persistence.exact_output_cache import digest
            expected=dict(execution_attempt_id=contract['execution_attempt_id'],
                cache_namespace=contract['cache_namespace'],binding_sha256=digest(contract),
                generation_policy_sha256=digest(frozen_solver_policy(contract)),
                recovery_policy_sha256=digest(self.recovery_policy))
            if any(durable_cache.context[k]!=v for k,v in expected.items()):
                raise SearchContractError('DURABLE_CACHE_PROVIDER_BINDING_MISMATCH')
        if contract.get('identity') == versions.MATH_V2_2_EXECUTION_BINDING_VERSION:
            if (contract.get('method_identity') != versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_2_VERSION
                    or contract.get('transition_policy') != versions.UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION):
                raise SearchContractError('CURRENT_V2_2_METHOD_BINDING_MISMATCH')
        self.prompt_observer = None

    @property
    def successes(self):
        return self.usage["successes"]

    def private_capability(self):
        return RequestBroker(contract=self.contract, transport=self.transport, arm=self.arm, seed=self.seed,
            ledger_writer=self.ledger_writer, raw_writer=self.raw_writer, usage=self.usage, lock=self.lock,
            token_ledger=self.token_ledger, reserve_reader=self.reserve_reader, validation_only=self.validation_only,
            durable_cache=self.durable_cache, response_receipts=self.response_receipts)

    def _request_identity(self, *, role, split, messages, member_slot=None):
        pattern_roles={'pattern_gradient','pattern_cluster'} if self.gradient_pattern else {'pattern'}
        legal = (role == "solver" and split == "validation") if self.validation_only else (
            role in {"solver", "reflection", *pattern_roles} and split in {"optimize", "shadow"}
            and (role == "solver" or split == "optimize"))
        if not legal:
            raise SearchContractError("PROVIDER_ROLE_SPLIT_FORBIDDEN")
        if role in pattern_roles and not self.contract["arms"][self.arm][0]:
            raise SearchContractError("PATTERN_CALL_FORBIDDEN_IN_NULL_ARM")
        c = self.contract
        model = c["models"]["solver" if role == "solver" else "optimizer_reflection" if role == "reflection" else "pattern"]
        request = dict(model=model, messages=messages, **generation_request_fields(c, role))
        identity = {"provider": c["provider"], "role": role, "split": split, "request": request,
                    "cache_namespace": c["cache_namespace"]}
        if c.get('identity') == versions.MATH_V2_2_EXECUTION_BINDING_VERSION:
            identity['method_treatment'] = {k:c[k] for k in ('method_identity', 'transition_policy')}
        if c.get('identity') in {versions.MATH_LAYER1_MEMORY_EXECUTION_BINDING_VERSION,versions.MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION, versions.MATH_V2_2_EXECUTION_BINDING_VERSION}:
            identity['memory_treatment']={k:c[k] for k in ('memory_policy_identity','memory_limits','layer1_search_policy','optimizer_input_schema','panel_evidence_policy')}
        if c.get('pattern_policy'):
            identity['pattern_treatment']={k:c[k] for k in ('pattern_policy','shared_risk_policy','pattern_amendment_authorization_sha256')}
            if 'pattern_support_id_transport' in c:
                identity['pattern_treatment']['pattern_support_id_transport']=c['pattern_support_id_transport']
            if 'pattern_abstraction_guard' in c:
                identity['pattern_treatment']['pattern_abstraction_guard']=c['pattern_abstraction_guard']
            if self.gradient_pattern:
                identity['pattern_treatment'].update(gradient_prompt_sha256=c['gradient_prompt_sha256'],
                    cluster_prompt_sha256=c['pattern_prompt_sha256'])
                if 'gradient_recovery_policy' in c:
                    identity['pattern_treatment']['gradient_recovery_policy']=c['gradient_recovery_policy']
        if role in {'reflection','pattern','pattern_gradient','pattern_cluster'} and self.optimizer_policy:
            identity['optimizer_generation_policy'] = self.optimizer_policy
            if c.get('optimizer_nonthinking_evidence_policy'):
                identity['optimizer_nonthinking_evidence_policy']=c['optimizer_nonthinking_evidence_policy']
        if role == "solver":
            if self.member_lane_policy:
                if type(member_slot) is not int or member_slot not in range(5):
                    raise SearchContractError('SOLVER_MEMBER_REALIZATION_LANE_REQUIRED')
                identity['member_realization_lane']=member_slot
                identity['scientific_attempt']=c['execution_attempt_id']
            identity["solver_output_interface"] = c["solver_output_interface"]
            if frozen_solver_policy(c) is not None:
                identity["solver_decoding_policy"] = frozen_solver_policy(c)
            if self.prediction_policy:
                identity["prediction_validity_policy"] = self.prediction_policy
            if self.recovery_policy:
                identity.update(invalid_recovery_policy=self.recovery_policy,
                    benchmark_protocol=c['benchmark_protocol_sha256'],
                    development_protocol=c['low_cost_protocol'],cache_policy=c['cache_policy'])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return request,key

    def _cache_hit(self, key, role, split, stage):
        if self.prediction_policy and self.cache_seals.get(key) != hashlib.sha256(
                json.dumps(self.cache[key], sort_keys=True, separators=(",", ":")).encode()).hexdigest():
            self.abort("PROVIDER_CACHE_CORRUPTION")
        self._write(dict(kind="CACHE_HIT", role=role, split=split, stage=stage, request_sha256=key))
        return {**self.cache[key], "provider_called": False, "input_tokens": 0, "output_tokens": 0}

    def _store_cache(self, key, result):
        self.cache[key]=result
        if self.prediction_policy:
            self.cache_seals[key]=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def complete(self, *, role, split, stage, messages, member_slot=None):
        if role != 'solver' or not self.recovery_policy:
            return self._complete_one(role=role,split=split,stage=stage,messages=messages,member_slot=member_slot)
        _,key=self._request_identity(role=role,split=split,messages=messages,member_slot=member_slot)
        with self.lock:
            if key not in self.cache and self.durable_cache:
                previous=self.durable_cache.get(key)
                if previous is not None:self._store_cache(key,previous)
            if key in self.cache:return self._cache_hit(key,role,split,stage)
            from ..benchmarks.math_prediction_validity import classify_prediction,resolve_predictions
            predictions=[];realizations=[];transport_retries=0
            for semantic_attempt_no in range(1,5):
                try:
                    result=self._complete_one(role=role,split=split,stage=stage,messages=messages,
                        semantic_attempt_no=semantic_attempt_no,member_slot=member_slot)
                except BaseException as exc:
                    if semantic_attempt_no>1 and 'TOKEN_BUDGET' in str(exc):
                        self.abort('STOP_TOKEN_BUDGET_INSUFFICIENT_FOR_INVALID_RECOVERY')
                    raise
                transport_retries+=result['transport_retry_attempts']
                prediction=classify_prediction(result['text'],result.get('finish_reason'))
                predictions.append(prediction);realizations.append(result)
                self._write(dict(kind='SEMANTIC_ATTEMPT',role=role,split=split,stage=stage,
                    request_sha256=key,semantic_attempt_no=semantic_attempt_no,
                    prediction_valid=prediction.prediction_valid,invalid_reason=prediction.invalid_reason,
                    finish_reason=prediction.finish_reason,transport_retry_attempts=result['transport_retry_attempts']))
                if prediction.prediction_valid:break
            resolved=resolve_predictions(predictions,transport_retries)
            output={**result,'resolved_prediction':asdict(resolved),'original_realizations':realizations,
                'input_tokens':sum(r['input_tokens'] for r in realizations),
                'output_tokens':sum(r['output_tokens'] for r in realizations)}
            if self.member_lane_policy:
                hashes=[hashlib.sha256((r['text'] or '').encode()).hexdigest() for r in realizations]
                retries=hashes[1:]
                diversity=dict(raw_invalid_attempts=sum(not p.prediction_valid for p in predictions),
                    recovered_logical_requests=int(resolved.recovered_invalid),terminal_invalids=int(resolved.terminal_invalid),
                    identical_retry_response_count=sum(h in hashes[:i] for i,h in enumerate(hashes) if i>0),
                    unique_retry_response_rate=len(set(retries))/len(retries) if retries else None,
                    same_invalid_repeated_all_retries=resolved.terminal_invalid and len(set(hashes))==1)
                output['retry_diversity']=diversity
                self._write(dict(kind='SEMANTIC_RETRY_DIVERSITY',role=role,split=split,stage=stage,
                    request_sha256=key,member_slot=member_slot,**diversity))
            if self.durable_cache:self.durable_cache.put(key,output)
            self._store_cache(key,output)
            return output

    def _complete_one(self, *, role, split, stage, messages, semantic_attempt_no=None, member_slot=None):
        c=self.contract
        request,key=self._request_identity(role=role,split=split,messages=messages,member_slot=member_slot)
        model=request['model']
        cacheable=(not self.member_lane_policy or role=='solver') and role not in {'pattern_gradient','pattern_cluster'}
        with self.lock:
            if cacheable and semantic_attempt_no is None and key in self.cache:
                return self._cache_hit(key,role,split,stage)
            for retry in range(c["decoding"]["transport_retries"] + 1):
                bounds = c["provider_bounds"]
                role_bound = bounds[role+'_calls']
                if (self.usage["attempts"] >= bounds["transport_attempts"] or self.successes >= bounds["successful_provider_calls"]
                        or self.usage[role] >= role_bound or role in {'pattern_gradient','pattern_cluster'} and self.usage['pattern']>=bounds['pattern_calls']):
                    self.abort("PROVIDER_CEILING_PRE_TRANSPORT")
                token_reservation = None
                if self.token_ledger:
                    token_reservation = self.token_ledger.reserve(request,
                        attempt_id=c["execution_attempt_id"],
                        stage=stage if self.validation_only else c["execution_phase"], role=role, model=model,
                        protected_validation=self.reserve_reader())
                self.usage["attempts"] += 1
                self._write(dict(kind="ATTEMPT", role=role, split=split, stage=stage, request_sha256=key, attempt=self.usage["attempts"],
                    semantic_attempt_no=semantic_attempt_no,transport_retry_no=retry,
                    **({'member_realization_lane':member_slot} if self.member_lane_policy and role=='solver' else {})))
                try:
                    result = self.transport(request)
                except Exception as exc:
                    receipt = None
                    if self.response_receipts is not None:
                        receipt = self.response_receipts.persist(token_reservation, dict(
                            role=role, split=split, stage=stage, request_sha256=key,
                            request=request, error_category=type(exc).__name__,
                            token_usage=getattr(exc, "token_usage", None),
                            provider_evidence=getattr(exc, "provider_evidence", None),
                            semantic_attempt_no=semantic_attempt_no,
                            physical_attempt_no=self.usage["attempts"]))
                    if token_reservation is not None:
                        charge = self.token_ledger.reconcile(token_reservation, getattr(exc, "token_usage", None),
                            outcome=type(exc).__name__, **({"response_receipt": receipt} if receipt is not None else {}))
                        for k in ("input_tokens", "output_tokens"):
                            self.usage[k] += charge[k]
                    self.usage["failures"] += 1
                    self._write(dict(kind="FAILURE", role=role, split=split, stage=stage, request_sha256=key, error_category=type(exc).__name__))
                    if self.raw_writer:
                        self.raw_writer(dict(role=role, split=split, stage=stage, request_sha256=key,
                            semantic_attempt_no=semantic_attempt_no,physical_attempt_no=self.usage['attempts'],
                            request=request, error_category=type(exc).__name__,
                            provider_evidence=getattr(exc, "provider_evidence", None)))
                    # Retry transport failures only. SDK retries are disabled.
                    from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
                    if not isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)) or retry == c["decoding"]["transport_retries"]:
                        if self.token_ledger:
                            self.abort("PROVIDER_TERMINAL_" + type(exc).__name__)
                        raise
                    time.sleep(min(c["decoding"]["retry_sleep_seconds"] * 2**retry, c["decoding"]["retry_backoff_ceiling_seconds"]))
                    continue
                receipt = None
                if self.response_receipts is not None:
                    receipt = self.response_receipts.persist(token_reservation, dict(
                        role=role, split=split, stage=stage, request_sha256=key,
                        request=request, response=result, semantic_attempt_no=semantic_attempt_no,
                        physical_attempt_no=self.usage["attempts"]))
                if token_reservation is not None:
                    charge = self.token_ledger.reconcile(token_reservation, result, outcome="RESPONSE",
                        **({"response_receipt": receipt} if receipt is not None else {}))
                    if isinstance(result, dict):
                        result = {**result, "provider_reported_input_tokens":result.get("input_tokens"),
                                  "provider_reported_output_tokens":result.get("output_tokens"), **charge}
                if not isinstance(result, dict) or (not isinstance(result.get("text"), str)
                        and not (role == "solver" and self.prediction_policy and "text" in result and result["text"] is None)) or any(type(result.get(k)) is not int or result[k] < 0 for k in ("input_tokens", "output_tokens")):
                    self.usage["failures"] += 1
                    self._write(dict(kind="FAILURE", role=role, split=split, stage=stage, request_sha256=key, error_category="RESPONSE_ACCOUNTING_INVALID"))
                    self.abort("PROVIDER_RESPONSE_ACCOUNTING_INVALID")
                self.usage["successes"] += 1
                self.usage[role] += 1
                if role in {'pattern_gradient','pattern_cluster'}:self.usage['pattern']+=1
                if split == "validation":
                    self.usage["validation"] += 1
                for k in ("input_tokens", "output_tokens"):
                    self.usage[k] += result[k]
                self._write(dict(kind="SUCCESS", role=role, split=split, stage=stage, request_sha256=key,
                    response_sha256=hashlib.sha256(result["text"].encode() if result["text"] is not None else b"null").hexdigest(), input_tokens=result["input_tokens"], output_tokens=result["output_tokens"]))
                if role in {'reflection','pattern','pattern_gradient','pattern_cluster'} and self.optimizer_policy:
                    from ..benchmarks.math_optimizer_diagnostics import optimizer_response_telemetry
                    telemetry=optimizer_response_telemetry(request,result,self.optimizer_policy,c.get('optimizer_nonthinking_evidence_policy'))
                    result={**result,'optimizer_generation_diagnostics':telemetry}
                    self._write(dict(kind='OPTIMIZER_GENERATION_DIAGNOSTICS',role=role,split=split,stage=stage,
                        request_sha256=key,**telemetry))
                if self.raw_writer:
                    self.raw_writer(dict(role=role, split=split, stage=stage, request_sha256=key, request=request, response=result,
                        semantic_attempt_no=semantic_attempt_no,physical_attempt_no=self.usage['attempts']))
                if role in {'reflection','pattern','pattern_gradient','pattern_cluster'} and self.optimizer_policy and self.optimizer_policy['enable_thinking'] is False:
                    reasoning=telemetry['reasoning_tokens']
                    if telemetry['nonthinking_evidence_level']=='CONTRADICTORY':
                        self.abort('PROVIDER_NONTHINKING_CONTROL_NOT_HONORED')
                if (self.token_ledger or role == "solver" and frozen_solver_policy(c) or role != 'solver' and self.optimizer_policy) and not (role == "solver" and self.prediction_policy) and result.get("finish_reason") in {"length", "max_tokens", "max_output_tokens"}:
                    self.abort("STOP_SOLVER_DECODING_POLICY_INSUFFICIENT" if role == "solver" and frozen_solver_policy(c) else "OPERATIONAL_OUTPUT_TRUNCATION")
                if 'max_completion_tokens' in request:
                    reported = result.get('provider_reported_output_tokens', result.get('output_tokens'))
                    ceiling = self.optimizer_policy['accounting_output_ceiling']
                else:
                    reported = result.get('provider_reported_output_tokens')
                    ceiling = request['max_tokens']
                if (self.token_ledger or role != 'solver' and self.optimizer_policy) and type(reported) is int and reported > ceiling:
                    self.abort("OPERATIONAL_OUTPUT_CAP_NOT_ENFORCED")
                result = {**result, "provider_called": True}
                if self.prediction_policy:
                    result["request_sha256"] = key
                if semantic_attempt_no is None and cacheable:
                    self._store_cache(key,result)
                if semantic_attempt_no is not None:
                    result['transport_retry_attempts']=retry
                return result

    def abort(self, reason):
        if self.token_ledger:
            from ..governance.token_accounting import OperationalAbort
            raise OperationalAbort(reason)
        raise SearchContractError(reason)

    def _write(self, row):
        if self.ledger_writer:
            self.ledger_writer({**row, "arm": self.arm, "seed": self.seed})


class BenchmarkSolver:
    solver_contract_id = "COMMON_SOLVER_CONTRACT_V1"
    def __init__(self, benchmark, broker):
        self.benchmark = benchmark
        self.broker = broker
        self.output_contract_id = PROTOCOLS[benchmark.benchmark_id].output_contract_id
        self.invalid_predictions_are_incorrect = bool(getattr(benchmark, "invalid_predictions_are_incorrect", False))
        self.member_id = None  # New lane policy binds this slot to identity, never model messages.
        if self.invalid_predictions_are_incorrect != bool(broker.prediction_policy):
            raise SearchContractError("SOLVER_PREDICTION_POLICY_PORT_MISMATCH")

    def for_gate(self):
        return BenchmarkSolver(self.benchmark, self.broker.private_capability())

    def observe_member(self, member_id):
        if type(member_id) is not int or member_id not in range(5):
            raise SearchContractError("SOLVER_TELEMETRY_MEMBER_INVALID")
        self.member_id = member_id

    def _request(self, prompt, item, *, stage, split):
        if item.benchmark_id != self.benchmark.benchmark_id:
            raise SearchContractError("SOLVER_BENCHMARK_INTERFACE_MISMATCH")
        # Benchmark-owned immutable formatting is authoritative even if an
        # evolved example or caller supplies a missing/stale per-item contract.
        interface = self.benchmark.output_contract
        contract = self.benchmark.solver_interface_contract()
        if self.broker.contract.get("solver_output_interface") != contract:
            raise SearchContractError("SOLVER_INTERFACE_BINDING_MISMATCH")
        if self.broker.prompt_observer:
            self.broker.prompt_observer(prompt)
        user_content = (self.benchmark.solver_user_content(prompt, item)
            if hasattr(self.benchmark, "solver_user_content")
            else prompt + "\n\n" + self.benchmark.format_input(item))
        effective_contract = dict(solver_interface=contract,
            mutable_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            benchmark_input_sha256=hashlib.sha256(self.benchmark.format_input(item).encode()).hexdigest(),
            model=self.broker.contract["models"]["solver"], decoding=self.broker.contract["decoding"],
            solver_thinking=self.broker.contract["models"]["solver_thinking"])
        if frozen_solver_policy(self.broker.contract) is not None:
            effective_contract["solver_decoding_policy"] = frozen_solver_policy(self.broker.contract)
        if self.broker.prediction_policy:
            effective_contract["prediction_validity_policy"] = self.broker.prediction_policy
        if self.broker.member_lane_policy:
            effective_contract['member_realization_lane']=self.member_id
            effective_contract['cache_policy']=self.broker.contract['cache_policy']
        effective = hashlib.sha256(json.dumps(effective_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.broker._write(dict(kind="SOLVER_REQUEST_CONTRACT", role="solver", split=split, stage=stage,
            solver_interface_identity=contract["identity"], mutable_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            effective_solver_request_contract_hash=effective))
        return self.broker.complete(role="solver", split=split, stage=stage,
            messages=[{"role": "system", "content": interface},
                      {"role": "user", "content": user_content}],member_slot=self.member_id)

    def solve(self, prompt, item, *, stage, split):
        result = self._request(prompt, item, stage=stage, split=split)
        if self.invalid_predictions_are_incorrect:
            return self._prediction(result, prompt, item, stage=stage, split=split)
        if hasattr(self.benchmark,"final_payload"):
            if self.benchmark.final_payload(result["text"]) is None:
                self.broker.abort("STOP_SOLVER_DECODING_POLICY_INSUFFICIENT" if frozen_solver_policy(self.broker.contract) else "SOLVER_INVALID_RESPONSE_NO_REGENERATION")
        elif not self.benchmark.parse_member_output(result["text"], item).valid:
            self.broker.abort("STOP_SOLVER_DECODING_POLICY_INSUFFICIENT" if frozen_solver_policy(self.broker.contract) else "SOLVER_INVALID_RESPONSE_NO_REGENERATION")
        return result["text"]

    def _prediction(self, result, prompt, item, *, stage, split):
        prediction = self.benchmark.prediction_result(result)
        telemetry=({k:getattr(prediction,k) for k in ('semantic_attempt_count','raw_invalid_count','recovered_invalid',
            'terminal_invalid','transport_retry_attempts')} if self.broker.recovery_policy else {})
        self.broker._write(dict(kind="PREDICTION_VALIDITY", role="solver", stage=stage, split=split,**telemetry,
            policy_identity=self.broker.prediction_policy["identity"],
            mutable_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), input_id=item.input_id,
            member_id=self.member_id,
            request_sha256=result["request_sha256"],
            prediction_sha256=hashlib.sha256(prediction.identity_bytes()).hexdigest(),
            prediction_valid=prediction.prediction_valid, invalid_reason=prediction.invalid_reason,
            finish_reason=prediction.finish_reason, provider_called=result["provider_called"]))
        return prediction

    def evaluate(self, prompt, example):
        if hasattr(self.benchmark,"require_scorable"):
            try:
                return self._evaluate(prompt,example)
            except Exception as exc:
                # Pinned GEPA catches ordinary exceptions in its loop. Real
                # accounting scopes must terminate on evaluator/runtime faults.
                reason=str(exc) if isinstance(exc,SearchContractError) else 'MATH_LOCAL_EVALUATION_FAILURE_'+type(exc).__name__
                self.broker.abort(reason)
        return self._evaluate(prompt,example)

    def _evaluate(self, prompt, example):
        if hasattr(self.benchmark,"require_scorable"):
            self.benchmark.require_scorable(example.gold)
        item = BenchmarkInput(example.example_id, example.input_payload, self.benchmark.output_contract,
            benchmark_id=self.benchmark.benchmark_id)
        if hasattr(self.benchmark,'protocol'):
            from ..benchmarks.protocols import protocol_input
            item=protocol_input(self.benchmark.benchmark_id,example.example_id,{'problem':example.input_payload},
                self.benchmark.output_contract,protocol=self.benchmark.protocol)
        result = self._request(prompt, item, stage="gepa_local", split="optimize")
        if self.invalid_predictions_are_incorrect:
            prediction = self._prediction(result, prompt, item, stage="gepa_local", split="optimize")
            parsed = prediction.parsed()
            return LocalSolverObservation(parsed.answer, result["text"],
                self.benchmark.score_member_output(parsed, example.gold) == 1, parsed.valid,
                failure_reason=prediction.invalid_reason, input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"], provider_called=result["provider_called"])
        parsed = self.benchmark.parse_member_output(result["text"], item)
        if hasattr(self.benchmark,"final_payload"):
            if self.benchmark.final_payload(result["text"]) is None:
                self.broker.abort("STOP_SOLVER_DECODING_POLICY_INSUFFICIENT" if frozen_solver_policy(self.broker.contract) else "SOLVER_INVALID_RESPONSE_NO_REGENERATION")
        elif not parsed.valid:
            self.broker.abort("STOP_SOLVER_DECODING_POLICY_INSUFFICIENT" if frozen_solver_policy(self.broker.contract) else "SOLVER_INVALID_RESPONSE_NO_REGENERATION")
        return LocalSolverObservation(parsed.answer, result["text"], self.benchmark.score_member_output(parsed, example.gold) == 1,
            parsed.valid, input_tokens=result["input_tokens"], output_tokens=result["output_tokens"], provider_called=result["provider_called"])


class ReflectionProvider:
    def __init__(self, broker):
        self.broker = broker
        self.calls = self.input_tokens = self.output_tokens = 0

    def __call__(self, prompt):
        result = self.broker.complete(role="reflection", split="optimize", stage="gepa_reflection", messages=[{"role": "user", "content": prompt}])
        self.calls += int(result["provider_called"])
        self.input_tokens += result["input_tokens"]
        self.output_tokens += result["output_tokens"]
        return result["text"]

    def accounting(self):
        return dict(successful_calls=self.calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens)
