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
from .patterns import PatternHypothesis
from .schemas import SearchContractError


class RequestBroker:
    def __init__(self, *, contract, transport, arm, seed, ledger_writer=None, raw_writer=None, usage=None, lock=None,
                 token_ledger=None, reserve_reader=None, validation_only=False):
        self.contract = contract
        self.transport = transport
        self.arm = arm
        self.seed = seed
        self.ledger_writer = ledger_writer
        self.raw_writer = raw_writer
        self.cache = {}
        self.usage = usage if usage is not None else dict(attempts=0, successes=0, failures=0, input_tokens=0, output_tokens=0,
            solver=0, reflection=0, pattern=0, validation=0, test=0)
        self.lock = lock if lock is not None else threading.RLock()
        self.token_ledger = token_ledger
        self.reserve_reader = reserve_reader or (lambda: 0)
        self.validation_only = validation_only
        self.prompt_observer = None

    @property
    def successes(self):
        return self.usage["successes"]

    def private_capability(self):
        return RequestBroker(contract=self.contract, transport=self.transport, arm=self.arm, seed=self.seed,
            ledger_writer=self.ledger_writer, raw_writer=self.raw_writer, usage=self.usage, lock=self.lock,
            token_ledger=self.token_ledger, reserve_reader=self.reserve_reader, validation_only=self.validation_only)

    def complete(self, *, role, split, stage, messages):
        legal = (role == "solver" and split == "validation") if self.validation_only else (
            role in {"solver", "reflection", "pattern"} and split in {"optimize", "shadow"}
            and (role == "solver" or split == "optimize"))
        if not legal:
            raise SearchContractError("PROVIDER_ROLE_SPLIT_FORBIDDEN")
        if role == "pattern" and not self.contract["arms"][self.arm][0]:
            raise SearchContractError("PATTERN_CALL_FORBIDDEN_IN_NULL_ARM")
        c = self.contract
        model = c["models"]["solver" if role == "solver" else "optimizer_reflection" if role == "reflection" else "pattern"]
        request = dict(model=model, messages=messages, temperature=c["decoding"]["temperature"],
            max_tokens=c["decoding"]["max_output_tokens"], extra_body={"enable_thinking": False} if role == "solver" else {})
        identity = {"provider": c["provider"], "role": role, "split": split, "request": request,
                    "cache_namespace": c["cache_namespace"]}
        if role == "solver":
            identity["solver_output_interface"] = c["solver_output_interface"]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self._write(dict(kind="CACHE_HIT", role=role, split=split, stage=stage, request_sha256=key))
                return {**self.cache[key], "provider_called": False, "input_tokens": 0, "output_tokens": 0}
            for retry in range(c["decoding"]["transport_retries"] + 1):
                bounds = c["provider_bounds"]
                role_bound = bounds["solver_calls" if role == "solver" else "reflection_calls" if role == "reflection" else "pattern_calls"]
                if self.usage["attempts"] >= bounds["transport_attempts"] or self.successes >= bounds["successful_provider_calls"] or self.usage[role] >= role_bound:
                    self.abort("PROVIDER_CEILING_PRE_TRANSPORT")
                token_reservation = None
                if self.token_ledger:
                    token_reservation = self.token_ledger.reserve(request,
                        attempt_id=c["execution_attempt_id"],
                        stage=stage if self.validation_only else c["execution_phase"], role=role, model=model,
                        protected_validation=self.reserve_reader())
                self.usage["attempts"] += 1
                self._write(dict(kind="ATTEMPT", role=role, split=split, stage=stage, request_sha256=key, attempt=self.usage["attempts"]))
                try:
                    result = self.transport(request)
                except Exception as exc:
                    if token_reservation is not None:
                        charge = self.token_ledger.reconcile(token_reservation, getattr(exc, "token_usage", None), outcome=type(exc).__name__)
                        for k in ("input_tokens", "output_tokens"):
                            self.usage[k] += charge[k]
                    self.usage["failures"] += 1
                    self._write(dict(kind="FAILURE", role=role, split=split, stage=stage, request_sha256=key, error_category=type(exc).__name__))
                    # Retry transport failures only. SDK retries are disabled.
                    from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
                    if not isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)) or retry == c["decoding"]["transport_retries"]:
                        if self.token_ledger:
                            self.abort("PROVIDER_TERMINAL_" + type(exc).__name__)
                        raise
                    time.sleep(min(c["decoding"]["retry_sleep_seconds"] * 2**retry, c["decoding"]["retry_backoff_ceiling_seconds"]))
                    continue
                if token_reservation is not None:
                    charge = self.token_ledger.reconcile(token_reservation, result, outcome="RESPONSE")
                    if isinstance(result, dict):
                        result = {**result, "provider_reported_input_tokens":result.get("input_tokens"),
                                  "provider_reported_output_tokens":result.get("output_tokens"), **charge}
                if not isinstance(result, dict) or not isinstance(result.get("text"), str) or any(type(result.get(k)) is not int or result[k] < 0 for k in ("input_tokens", "output_tokens")):
                    self.usage["failures"] += 1
                    self._write(dict(kind="FAILURE", role=role, split=split, stage=stage, request_sha256=key, error_category="RESPONSE_ACCOUNTING_INVALID"))
                    self.abort("PROVIDER_RESPONSE_ACCOUNTING_INVALID")
                self.usage["successes"] += 1
                self.usage[role] += 1
                if split == "validation":
                    self.usage["validation"] += 1
                for k in ("input_tokens", "output_tokens"):
                    self.usage[k] += result[k]
                self._write(dict(kind="SUCCESS", role=role, split=split, stage=stage, request_sha256=key,
                    response_sha256=hashlib.sha256(result["text"].encode()).hexdigest(), input_tokens=result["input_tokens"], output_tokens=result["output_tokens"]))
                if self.raw_writer:
                    self.raw_writer(dict(role=role, split=split, stage=stage, request_sha256=key, request=request, response=result))
                if self.token_ledger and result.get("finish_reason") in {"length", "max_tokens", "max_output_tokens"}:
                    self.abort("OPERATIONAL_OUTPUT_TRUNCATION")
                if self.token_ledger and type(result.get("provider_reported_output_tokens")) is int and result["provider_reported_output_tokens"] > request["max_tokens"]:
                    self.abort("OPERATIONAL_OUTPUT_CAP_NOT_ENFORCED")
                result = {**result, "provider_called": True}
                self.cache[key] = result
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

    def for_gate(self):
        return BenchmarkSolver(self.benchmark, self.broker.private_capability())

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
        effective = hashlib.sha256(json.dumps(dict(solver_interface=contract,
            mutable_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            benchmark_input_sha256=hashlib.sha256(self.benchmark.format_input(item).encode()).hexdigest(),
            model=self.broker.contract["models"]["solver"], decoding=self.broker.contract["decoding"],
            solver_thinking=self.broker.contract["models"]["solver_thinking"]), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.broker._write(dict(kind="SOLVER_REQUEST_CONTRACT", role="solver", split=split, stage=stage,
            solver_interface_identity=contract["identity"], mutable_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            effective_solver_request_contract_hash=effective))
        return self.broker.complete(role="solver", split=split, stage=stage,
            messages=[{"role": "system", "content": interface},
                      {"role": "user", "content": prompt + "\n\n" + self.benchmark.format_input(item)}])

    def solve(self, prompt, item, *, stage, split):
        result = self._request(prompt, item, stage=stage, split=split)
        if not self.benchmark.parse_member_output(result["text"], item).valid:
            self.broker.abort("SOLVER_INVALID_RESPONSE_NO_REGENERATION")
        return result["text"]

    def evaluate(self, prompt, example):
        item = BenchmarkInput(example.example_id, example.input_payload, self.benchmark.output_contract,
            benchmark_id=self.benchmark.benchmark_id)
        result = self._request(prompt, item, stage="gepa_local", split="optimize")
        parsed = self.benchmark.parse_member_output(result["text"], item)
        if not parsed.valid:
            self.broker.abort("SOLVER_INVALID_RESPONSE_NO_REGENERATION")
        return LocalSolverObservation(parsed.answer, result["text"], self.benchmark.score_member_output(parsed, example.gold) == 1,
            True, input_tokens=result["input_tokens"], output_tokens=result["output_tokens"], provider_called=result["provider_called"])


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


class PatternProvider:
    def __init__(self, broker, prompt):
        self.broker = broker
        self.prompt = prompt
        self.seen = set()

    def diagnose(self, request):
        key = (request.parent_state_id, request.target_member, request.structured_history)
        if key in self.seen:
            raise SearchContractError("PATTERN_ONE_SUCCESS_PER_OPPORTUNITY")
        if any(r.source_split != "optimize" for r in request.evidence_rows):
            raise SearchContractError("PATTERN_HELDOUT_ACCESS")
        payload = json.dumps(asdict(request), sort_keys=True, default=lambda x: sorted(x) if isinstance(x, frozenset) else None)
        result = self.broker.complete(role="pattern", split="optimize", stage="pattern_diagnosis",
            messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": payload}])
        self.seen.add(key)
        value = json.loads(result["text"])
        if not isinstance(value, dict) or set(value) != {"patterns"} or not isinstance(value["patterns"], list):
            raise SearchContractError("PATTERN_RESPONSE_SCHEMA_INVALID")
        allowed = set(PatternHypothesis.__dataclass_fields__)
        rows = []
        for row in value["patterns"]:
            if not isinstance(row, dict) or set(row) != allowed:
                raise SearchContractError("PATTERN_RESPONSE_SCHEMA_INVALID")
            rows.append(PatternHypothesis(**{k: tuple(v) if k.endswith("_ids") else v for k, v in row.items()}))
        return tuple(rows)
