"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


def is_transient_llm_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "timeout", "timed out", "deadline", "rate limit", "too many requests",
            "temporarily", "temporary", "connection", "server", "overloaded",
            "try again", "429", "503", "502", "504", "负载", "饱和", "稍后再试",
        )
    )


class SolverServiceMixin:
    async def _chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        client_role: str = "evaluator",
        audit_context: Optional[Mapping[str, Any]] = None,
    ) -> str:
        client = self.solver_client if client_role == "solver" else self.evaluator_client
        last_err: Optional[Exception] = None
        attempt = 0
        transient_failures = 0
        timeout_sec = float(self.cfg.llm_call_timeout or 0.0)
        prompt_estimate = self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt)
        while True:
            start_time = time.time()
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if timeout_sec > 0:
                    kwargs["timeout"] = timeout_sec
                resp = await client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                self._record_llm_call(
                    stage=stage,
                    client_role=client_role,
                    model=model,
                    temperature=temperature,
                    prompt_tokens=self._usage_value(usage, "prompt_tokens", prompt_estimate),
                    completion_tokens=self._usage_value(usage, "completion_tokens", self._estimate_tokens(text)),
                    latency_seconds=time.time() - start_time,
                    success=True,
                    audit_context={**dict(audit_context or {}), "response_empty": not bool(str(text or "").strip())},
                )
                return text
            except Exception as e:
                last_err = e
                transient = is_transient_llm_error(e)
                if not transient:
                    if attempt >= max(1, int(self.cfg.max_retries)):
                        self._record_llm_call(
                            stage=stage,
                            client_role=client_role,
                            model=model,
                            temperature=temperature,
                            prompt_tokens=prompt_estimate,
                            completion_tokens=0,
                            latency_seconds=time.time() - start_time,
                            success=False,
                            error_type=type(e).__name__,
                            audit_context=audit_context,
                        )
                        break
                else:
                    transient_failures += 1
                    if not self.cfg.transient_retry_forever and transient_failures > int(self.cfg.max_transient_retries or self.cfg.max_retries):
                        self._record_llm_call(
                            stage=stage,
                            client_role=client_role,
                            model=model,
                            temperature=temperature,
                            prompt_tokens=prompt_estimate,
                            completion_tokens=0,
                            latency_seconds=time.time() - start_time,
                            success=False,
                            error_type=type(e).__name__,
                            audit_context=audit_context,
                        )
                        break
                sleep_sec = min(float(self.cfg.max_retry_backoff), float(self.cfg.retry_sleep) * (2 ** attempt))
                if self.cfg.llm_call_logging:
                    print(f"[LLM][retry] stage={stage} model={model} attempt={attempt + 1} sleep={sleep_sec:.2f} error={normalize_spaces(str(e))[:240]}", flush=True)
                await asyncio.sleep(sleep_sec)
                attempt += 1
        raise RuntimeError(f"LLM call failed at {stage}: {last_err}")

    async def solve_once(self, question: str, agent_id: int, prompt_text: str) -> Tuple[str, str]:
        effective_task = infer_task_type(task_type=self.cfg.task_type, question=question, answer=None)
        answer_format = str(getattr(self.cfg, "answer_format", "") or "").strip().lower()
        if answer_format == "option_letter":
            answer_hint = "<A/B/C/D>"
        elif answer_format == "boolean":
            answer_hint = "<true/false>"
        elif answer_format == "yes_no":
            answer_hint = "<yes/no>"
        elif answer_format == "valid_invalid":
            answer_hint = "<valid/invalid>"
        elif answer_format == "numeric":
            answer_hint = "<number>"
        else:
            answer_hint = "<answer>"
        if effective_task == "mmlu" or answer_format == "option_letter":
            system_prompt = (
                "You are solving a multiple-choice question. Follow the agent role faithfully and make the role's "
                "decision procedure visible in a compact trace. Do not merely name the role; execute it. "
                f"Avoid filler, avoid copying the question, and end with exactly one line: FINAL_ANSWER: {answer_hint}.\n\n"
                f"Agent role:\n{prompt_text}"
            )
        else:
            system_prompt = (
                "You are solving a reasoning problem. Follow the agent role faithfully and make the role's "
                "decision procedure visible in a compact trace. Do not merely name the role; execute it. "
                f"Avoid filler, avoid copying the question, and end with exactly one line: FINAL_ANSWER: {answer_hint}.\n\n"
                f"Agent role:\n{prompt_text}"
            )
        text = await self._chat(
            model=self.cfg.agent_model,
            system_prompt=system_prompt,
            user_prompt=f"Question:\n{question}\n\nSolve with the assigned role and keep the trace concise.",
            temperature=float(self.cfg.temperature),
            max_tokens=int(self.cfg.max_tokens),
            stage=f"solver_agent_{agent_id}",
            client_role="solver",
        )
        return text, self.task_spec.extract_pred(text, question)

    async def solve_with_prompts(self, question: str, prompts: List[str]) -> Tuple[List[str], List[str]]:
        return await self.solve_with_prompts_limited(question, prompts, self.solver_call_semaphore)

    async def solve_with_prompts_limited(
        self,
        question: str,
        prompts: List[str],
        solver_call_semaphore: asyncio.Semaphore,
    ) -> Tuple[List[str], List[str]]:
        async def solve_agent(agent_id: int):
            async with solver_call_semaphore:
                return await self.solve_once(question, agent_id, prompts[agent_id])

        outs = await asyncio.gather(*[solve_agent(i) for i in range(len(self.agents))])
        return [x[0] for x in outs], [x[1] for x in outs]

    async def get_or_create_solver_rollout(
        self,
        *,
        cache_key: str,
        lookup: Callable[[], Optional[Dict[str, Any]]],
        call_factory: Callable[[], Awaitable[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], str]:
        """Read a rollout cache or coalesce an identical in-flight API request."""
        cached = lookup()
        if isinstance(cached, dict) and "trace" in cached and "answer" in cached:
            return cached, "persisted_cache" if cached.get("cache_origin") == "persisted" else "memory_cache"

        if not bool(getattr(self.cfg, "solver_rollout_singleflight", True)):
            return await call_factory(), "api_call"
        if not hasattr(self, "solver_rollout_inflight"):
            self.solver_rollout_inflight = {}
        if not hasattr(self, "solver_rollout_inflight_lock"):
            self.solver_rollout_inflight_lock = asyncio.Lock()

        owner = False
        async with self.solver_rollout_inflight_lock:
            future = self.solver_rollout_inflight.get(cache_key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                # Consume exceptions if an owner fails before another waiter attaches.
                future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                self.solver_rollout_inflight[cache_key] = future
                owner = True
        if not owner:
            return await future, "inflight_reuse"
        try:
            row = await call_factory()
            future.set_result(row)
            return row, "api_call"
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self.solver_rollout_inflight_lock:
                self.solver_rollout_inflight.pop(cache_key, None)

    async def _solve_agent_rollout(
        self,
        *,
        question: str,
        question_hash: str,
        prompt: str,
        agent_id: int,
        source: str,
    ) -> Tuple[str, str, str]:
        cache_key = self._solver_rollout_cache_key(question_hash, prompt, agent_id)

        async def call_factory() -> Dict[str, Any]:
            async with self.solver_call_semaphore:
                trace, answer = await self.solve_once(question, agent_id, prompt)
            self._record_solver_rollout(
                question_hash=question_hash,
                prompt=prompt,
                trace=trace,
                answer=answer,
                agent_id=agent_id,
                source=source,
            )
            return {"trace": trace, "answer": answer, "cache_origin": "current_run"}

        if not self.cfg.candidate_reuse_recorded_rollouts:
            trace, answer = await self.solve_once(question, agent_id, prompt)
            return trace, answer, "api_call"
        row, origin = await self.get_or_create_solver_rollout(
            cache_key=cache_key,
            lookup=lambda: self._lookup_solver_rollout(question_hash, prompt, agent_id),
            call_factory=call_factory,
        )
        return str(row.get("trace", "")), str(row.get("answer", "")), origin

    async def solve_with_prompts_reusing_records(
        self,
        question: str,
        prompts: List[str],
        source: str = "candidate_eval",
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        prompts = list(prompts)
        while len(prompts) < len(self.agents):
            prompts.append(self.agents[len(prompts)].current_prompt)
        qh = self._hash(question)
        n = len(self.agents)
        outs = await asyncio.gather(
            *[
                self._solve_agent_rollout(
                    question=question, question_hash=qh, prompt=prompts[agent_id], agent_id=agent_id, source=source
                )
                for agent_id in range(n)
            ]
        )
        final_traces = [str(row[0] or "") for row in outs]
        final_answers = [str(row[1] or "") for row in outs]
        origins = [str(row[2]) for row in outs]
        reuse_hits = sum(origin in {"memory_cache", "persisted_cache", "inflight_reuse"} for origin in origins)
        api_calls = sum(origin == "api_call" for origin in origins)
        stats = {
            "solver_reuse_enabled": bool(self.cfg.candidate_reuse_recorded_rollouts),
            "solver_reuse_hits": int(reuse_hits),
            "solver_reuse_misses": int(api_calls),
            "solver_calls": int(api_calls),
            "solver_reuse_total": int(n),
            "solver_memory_cache_hits": int(sum(origin == "memory_cache" for origin in origins)),
            "solver_persisted_cache_hits": int(sum(origin == "persisted_cache" for origin in origins)),
            "solver_inflight_reuses": int(sum(origin == "inflight_reuse" for origin in origins)),
        }
        return final_traces, final_answers, stats

    async def ensure_recorded_rollouts_for_prompts(
        self,
        eval_batch: List[Dict[str, str]],
        prompts: List[str],
        source: str,
    ) -> Dict[str, Any]:
        if not self.cfg.candidate_reuse_recorded_rollouts or not eval_batch:
            return {"enabled": bool(self.cfg.candidate_reuse_recorded_rollouts), "solver_calls": 0, "solver_reuse_hits": 0, "solver_reuse_total": 0}
        totals = {"solver_calls": 0, "solver_reuse_hits": 0, "solver_reuse_total": 0}
        for ex in eval_batch:
            q = str(ex.get("question", ""))
            if not q:
                continue
            _, _, stats = await self.solve_with_prompts_reusing_records(q, prompts, source=source)
            totals["solver_calls"] += int(stats.get("solver_calls", 0) or 0)
            totals["solver_reuse_hits"] += int(stats.get("solver_reuse_hits", 0) or 0)
            totals["solver_reuse_total"] += int(stats.get("solver_reuse_total", 0) or 0)
        totals["enabled"] = True
        totals["solver_reuse_hit_rate"] = float(totals["solver_reuse_hits"] / totals["solver_reuse_total"]) if totals["solver_reuse_total"] else 0.0
        return totals

    async def _prewarm_factorized_candidate_rollouts(
        self,
        *,
        agent_id: int,
        eval_batch: List[Dict[str, str]],
        peer_prompts: List[str],
        candidate_pool: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Populate per-question rollouts for fixed peers and unique target prompts.

        Team metrics are deliberately *not* cached here. Every candidate later calls
        the normal evaluator, which recomposes a team on the current batch.
        """
        unique_prompts: Dict[str, str] = {}
        for candidate in candidate_pool:
            prompt = str(candidate.get("prompt", ""))
            if prompt:
                unique_prompts.setdefault(self._hash(normalize_spaces(prompt)), prompt)
        active_prompt = str(peer_prompts[agent_id]) if agent_id < len(peer_prompts) else str(self.agents[agent_id].current_prompt)
        unique_prompts.setdefault(self._hash(normalize_spaces(active_prompt)), active_prompt)
        requests: List[Tuple[str, int, str, str]] = []
        for ex in eval_batch:
            question = str(ex.get("question", ""))
            if not question:
                continue
            question_hash = self._hash(question)
            for peer_id, prompt in enumerate(peer_prompts):
                if peer_id != agent_id:
                    requests.append((question, peer_id, str(prompt), question_hash))
            for prompt in unique_prompts.values():
                requests.append((question, agent_id, prompt, question_hash))

        async def prewarm(row: Tuple[str, int, str, str]):
            question, row_agent_id, prompt, question_hash = row
            trace, answer, origin = await self._solve_agent_rollout(
                question=question,
                question_hash=question_hash,
                prompt=prompt,
                agent_id=row_agent_id,
                source=f"candidate_factorized_{'peer' if row_agent_id != agent_id else 'target'}_agent_{agent_id}",
            )
            return trace, answer, origin

        results = await asyncio.gather(*[prewarm(request) for request in requests])
        origins = [origin for _, _, origin in results]
        candidate_count = len(candidate_pool)
        example_count = len(eval_batch)
        naive = candidate_count * len(self.agents) * example_count
        factorized = (max(0, len(self.agents) - 1) + len(unique_prompts)) * example_count
        api_calls = sum(origin == "api_call" for origin in origins)
        memory_hits = sum(origin == "memory_cache" for origin in origins)
        persisted_hits = sum(origin == "persisted_cache" for origin in origins)
        inflight = sum(origin == "inflight_reuse" for origin in origins)
        return {
            "candidate_eval_execution_mode": "factorized_cached",
            "candidate_eval_candidate_object_count": candidate_count,
            "candidate_eval_unique_target_prompt_count": len(unique_prompts),
            "candidate_eval_duplicate_target_prompt_count": max(0, candidate_count - len(unique_prompts)),
            "candidate_eval_example_count": example_count,
            "candidate_eval_repeat_count": 1,
            "candidate_eval_naive_rollout_request_count": naive,
            "candidate_eval_factorized_rollout_request_count": factorized,
            "candidate_eval_unique_rollout_key_count": len(requests),
            "candidate_eval_memory_cache_hit_count": memory_hits,
            "candidate_eval_persisted_cache_hit_count": persisted_hits,
            "candidate_eval_inflight_reuse_count": inflight,
            "candidate_eval_solver_api_call_count": api_calls,
            "candidate_eval_rollout_failure_count": 0,
            "candidate_eval_calls_saved_vs_naive": naive - api_calls,
            "candidate_eval_cache_hit_rate": float((memory_hits + persisted_hits + inflight) / len(requests)) if requests else 0.0,
            "candidate_eval_peer_rollout_key_count": max(0, len(self.agents) - 1) * example_count,
            "candidate_eval_target_rollout_key_count": len(unique_prompts) * example_count,
            "candidate_eval_prompt_dedup_savings": max(0, candidate_count - len(unique_prompts)) * example_count,
        }
