import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from .config import Config
from .policy import AgentState
from .tasks import get_task_spec
from .utils import (
    ensure_dir,
    extract_json_obj,
    infer_task_type,
    majority_vote_with_diagnostics,
    normalize_spaces,
    set_seed,
)


class TraceBeamSearchSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.beam_refresh_each_epoch = bool(int(self.cfg.beam_refresh_each_epoch))
        self.cfg.transient_retry_forever = bool(int(self.cfg.transient_retry_forever))
        self.cfg.llm_call_logging = bool(int(self.cfg.llm_call_logging))
        self.cfg.local_validity_binary = bool(int(self.cfg.local_validity_binary))
        self.cfg.invalid_binary = bool(int(self.cfg.invalid_binary))
        self.cfg.use_joint_trace_diversity_evaluator = bool(int(self.cfg.use_joint_trace_diversity_evaluator))
        self.cfg.candidate_reuse_recorded_rollouts = bool(int(getattr(self.cfg, "candidate_reuse_recorded_rollouts", 1)))
        self.cfg.use_baseline_relative_reward = bool(int(getattr(self.cfg, "use_baseline_relative_reward", 1)))
        self.task_spec = get_task_spec(self.cfg.task_type)

        self.homogeneity_window = max(1, int(self.cfg.update_every))
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        solver_key_env = str(self.cfg.solver_api_key_env or "").strip()
        solver_base_env = str(self.cfg.solver_base_url_env or "").strip()
        evaluator_key_env = str(self.cfg.evaluator_api_key_env or "").strip()
        evaluator_base_env = str(self.cfg.evaluator_base_url_env or "").strip()
        solver_key = os.getenv(solver_key_env) if solver_key_env else api_key
        solver_base = os.getenv(solver_base_env) if solver_base_env else base_url
        evaluator_key = os.getenv(evaluator_key_env) if evaluator_key_env else api_key
        evaluator_base = os.getenv(evaluator_base_env) if evaluator_base_env else base_url
        self.solver_client = AsyncOpenAI(api_key=solver_key or api_key, base_url=solver_base)
        self.evaluator_client = AsyncOpenAI(api_key=evaluator_key or api_key, base_url=evaluator_base)

        ensure_dir(self.cfg.out_dir)
        set_seed(int(self.cfg.seed))

        self.initial_prompt_bank = self._default_prompt_bank()
        self.initial_agent_prompts = self._build_initial_prompts()
        self.initial_agent_prompt_hashes = [self._hash(p) for p in self.initial_agent_prompts]
        self.agents = [AgentState(p, homogeneity_window=self.homogeneity_window) for p in self.initial_agent_prompts]
        self._initialize_prompt_beams()

        self.history: List[Dict[str, Any]] = []
        self.update_logs: List[Dict[str, Any]] = []
        self.train_step_logs: List[Dict[str, Any]] = []
        self.train_trace_history_logs: List[Dict[str, Any]] = []
        self.test_trace_history_logs: List[Dict[str, Any]] = []
        self.recent_window_records: List[Dict[str, Any]] = []
        self.prompt_history = self._init_prompt_history()
        self.local_validity_cache: Dict[str, Dict[str, Any]] = {}
        self.joint_diversity_cache: Dict[str, Dict[str, Any]] = {}
        self.solver_rollout_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.embedding_model = None
        self.embedding_cache: Dict[str, List[float]] = {}
        self.solver_call_limit = max(1, int(getattr(self.cfg, "eval_solver_call_concurrency", 225) or 225))
        self.solver_call_semaphore = asyncio.Semaphore(self.solver_call_limit)

        if not self._is_accuracy_only_mode():
            self._load_embedding_model()
        self._load_recorded_solver_rollouts()
        self.write_run_meta()
        self.flush_prompt_history()

    def _is_accuracy_only_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "accuracy_only"

    def _is_guarded_reward_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "guarded_diversity"

    def _default_prompt_bank(self) -> List[str]:
        if str(self.cfg.task_type).lower() == "mmlu":
            return [
                "Use a concept-first procedure: name the tested concept, map it to the options, then choose one final answer.",
                "Use a contradiction-checking procedure: state a quick inconsistency test, apply it to plausible options, then choose one final answer.",
                "Use a boundary-and-scope procedure: inspect qualifiers, exceptions, and scope before comparing options and choosing.",
                "Use a backward-validation procedure: test what must be true if each plausible option were correct, then choose.",
                "Use an evidence-alignment procedure: tie the decision to specific clues in the stem before choosing.",
                "Use a mechanism-first procedure: explain the underlying rule or mechanism before selecting an option.",
            ]
        return [
            "Use an equation-first procedure: define variables, derive equations, solve, then check units or constraints.",
            "Use a backward-checking procedure: solve, then verify by substitution or reverse reasoning.",
            "Use a decomposition procedure: split the problem into sub-results, solve each, then combine carefully.",
            "Use a boundary-case procedure: check hidden assumptions, off-by-one cases, and impossible values before finalizing.",
            "Use a representation procedure: create a compact table, relation, or diagram in words before computing.",
            "Use an invariant procedure: track totals, conserved quantities, or repeated structure before calculating.",
        ]

    def _build_initial_prompts(self) -> List[str]:
        if self.cfg.agents <= 0:
            return []
        if str(self.cfg.init_mode).lower() == "bank":
            return [self.initial_prompt_bank[i % len(self.initial_prompt_bank)] for i in range(self.cfg.agents)]
        return [self.cfg.shared_prompt for _ in range(self.cfg.agents)]

    def _hash(self, value: str) -> str:
        return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]

    def _solver_cache_settings(self) -> Dict[str, Any]:
        return {
            "task_type": str(self.cfg.task_type),
            "agent_model": str(self.cfg.agent_model),
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens),
        }

    def _solver_rollout_cache_key_from_hashes(self, question_hash: str, prompt_hash: str, agent_id: int) -> str:
        settings = self._solver_cache_settings()
        return self._hash(
            "|".join(
                [
                    str(int(agent_id)),
                    str(settings["task_type"]),
                    str(settings["agent_model"]),
                    f"{float(settings['temperature']):.8g}",
                    str(settings["max_tokens"]),
                    str(question_hash),
                    str(prompt_hash),
                ]
            )
        )

    def _solver_rollout_cache_key(self, question_hash: str, prompt: str, agent_id: int) -> str:
        return self._solver_rollout_cache_key_from_hashes(question_hash, self._hash(prompt), agent_id)

    def _record_solver_rollout(
        self,
        question_hash: str,
        prompt: str,
        trace: str,
        answer: str,
        agent_id: Optional[int] = None,
        source: str = "",
        prompt_hash: Optional[str] = None,
    ):
        qh = str(question_hash or "").strip()
        ph = str(prompt_hash or self._hash(prompt)).strip()
        if agent_id is None:
            return
        try:
            aid = int(agent_id)
        except Exception:
            return
        if aid < 0 or not qh or not ph:
            return
        key = self._solver_rollout_cache_key_from_hashes(qh, ph, aid)
        row = {
            **self._solver_cache_settings(),
            "question_hash": qh,
            "prompt_hash": ph,
            "agent_id": aid,
            "trace": str(trace or ""),
            "answer": str(answer or ""),
            "source": str(source or ""),
        }
        self._add_solver_rollout_cache_row(row)
        self._append_solver_rollout_record(row)

    def _add_solver_rollout_cache_row(self, row: Dict[str, Any]):
        try:
            qh = str(row.get("question_hash", "")).strip()
            ph = str(row.get("prompt_hash", "")).strip()
            aid = int(row.get("agent_id", -1))
        except Exception:
            return
        if aid < 0 or not qh or not ph:
            return
        key = self._solver_rollout_cache_key_from_hashes(qh, ph, aid)
        self.solver_rollout_cache.setdefault(key, []).append(dict(row))

    def _record_solver_rollouts(
        self,
        question_hash: str,
        prompts: List[str],
        traces: List[str],
        answers: List[str],
        source: str,
    ):
        for i, prompt in enumerate(prompts):
            if i >= len(traces) or i >= len(answers):
                continue
            self._record_solver_rollout(
                question_hash=question_hash,
                prompt=str(prompt),
                trace=str(traces[i]),
                answer=str(answers[i]),
                agent_id=i,
                source=source,
            )

    def _lookup_solver_rollout(self, question_hash: str, prompt: str, agent_id: int) -> Optional[Dict[str, Any]]:
        try:
            aid = int(agent_id)
        except Exception:
            return None
        key = self._solver_rollout_cache_key(question_hash, prompt, aid)
        cached = self.solver_rollout_cache.get(key)
        if not isinstance(cached, list) or not cached:
            return None
        return dict(cached[-1])

    def _append_solver_rollout_record(self, row: Dict[str, Any]):
        if not self.cfg.candidate_reuse_recorded_rollouts:
            return
        path = os.path.join(self.cfg.out_dir, "solver_rollout_records.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _existing_run_meta_matches_solver_cache(self) -> bool:
        meta_path = os.path.join(self.cfg.out_dir, "run_meta.json")
        if not os.path.exists(meta_path):
            return True
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False
        cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
        if not cfg:
            return False
        checks = {
            "task_type": str(self.cfg.task_type),
            "agent_model": str(self.cfg.agent_model),
            "max_tokens": int(self.cfg.max_tokens),
        }
        for key, expected in checks.items():
            if str(cfg.get(key, "")) != str(expected):
                return False
        try:
            return abs(float(cfg.get("temperature", 0.0)) - float(self.cfg.temperature)) < 1e-12
        except Exception:
            return False

    def _iter_recorded_rollout_files(self) -> List[str]:
        out_dir = str(self.cfg.out_dir)
        names = ["solver_rollout_records.jsonl", "train_trace_history.jsonl", "test_trace_history.jsonl"]
        paths = [os.path.join(out_dir, name) for name in names]
        if os.path.isdir(out_dir):
            for name in sorted(os.listdir(out_dir)):
                if name.endswith("_predictions.jsonl") or (name.startswith("val_epoch") and name.endswith("_predictions.jsonl")):
                    paths.append(os.path.join(out_dir, name))
        seen = set()
        deduped = []
        for path in paths:
            if path not in seen and os.path.exists(path):
                seen.add(path)
                deduped.append(path)
        return deduped

    def _load_recorded_solver_rollouts(self):
        if not self.cfg.candidate_reuse_recorded_rollouts:
            return
        if not self._existing_run_meta_matches_solver_cache():
            return
        loaded = 0
        for path in self._iter_recorded_rollout_files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        if "prompt_hash" in row and "trace" in row and "answer" in row and "agent_id" in row:
                            self._add_solver_rollout_cache_row(row)
                            loaded += 1
                            continue
                        qh = str(row.get("question_hash", "")).strip()
                        agents = row.get("agents", [])
                        if not qh or not isinstance(agents, list):
                            continue
                        for agent in agents:
                            if not isinstance(agent, dict):
                                continue
                            prompt_hash = str(agent.get("prompt_hash", "")).strip()
                            if not prompt_hash:
                                continue
                            try:
                                agent_id = int(agent.get("agent_id", -1))
                            except Exception:
                                continue
                            self._add_solver_rollout_cache_row(
                                {
                                    **self._solver_cache_settings(),
                                    "question_hash": qh,
                                    "prompt_hash": prompt_hash,
                                    "agent_id": agent_id,
                                    "trace": str(agent.get("trace", "")),
                                    "answer": str(agent.get("answer", "")),
                                    "source": os.path.basename(path),
                                }
                            )
                            loaded += 1
            except Exception:
                continue
        if loaded and self.cfg.llm_call_logging:
            print(f"[solver-reuse] loaded recorded rollouts={loaded} unique_keys={len(self.solver_rollout_cache)}", flush=True)

    def _initialize_prompt_beams(self):
        for agent in self.agents:
            agent.prompt_beam = [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]

    def _make_beam_item(
        self,
        prompt: str,
        score: Optional[float],
        metrics: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        generation: int = 0,
        candidate_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        gen = int(generation)
        prompt_text = str(prompt)
        return {
            "id": candidate_id or f"g{gen}_{self._hash(prompt_text)}",
            "prompt": prompt_text,
            "score": None if score is None else float(score),
            "metrics": dict(metrics or {}),
            "parent_id": parent_id,
            "generation": gen,
        }

    def _active_prompt_list(self) -> List[str]:
        prompts = []
        for agent in self.agents:
            beam = getattr(agent, "prompt_beam", [])
            if beam and isinstance(beam[0], dict):
                prompts.append(str(beam[0].get("prompt", agent.current_prompt)))
            else:
                prompts.append(str(agent.current_prompt))
        return prompts

    def _base_log_fields(self) -> Dict[str, Any]:
        return {
            "agent_model": self.cfg.agent_model,
            "optimizer_model": self.cfg.optimizer_model,
            "evaluator_model": self.cfg.evaluator_model,
            "search_mode": self.cfg.search_mode,
            "reward_mode": self.cfg.reward_mode,
            "diversity_metric": self.cfg.diversity_metric,
            "embedding_model": self.cfg.embedding_model,
        }

    def write_run_meta(self):
        meta = {
            **self._base_log_fields(),
            "init_mode": self.cfg.init_mode,
            "agents": self.cfg.agents,
            "update_every": self.cfg.update_every,
            "beam_size": self.cfg.beam_size,
            "initial_agent_prompts": self.initial_agent_prompts,
            "initial_agent_prompt_hashes": self.initial_agent_prompt_hashes,
            "config": asdict(self.cfg),
            "framework": "accuracy_only_evolutionary_beam" if self._is_accuracy_only_mode() else "trace_embedding_evolutionary_beam",
        }
        with open(os.path.join(self.cfg.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _init_prompt_history(self) -> Dict[str, Any]:
        return {
            str(i): {
                "initial_prompt": agent.initial_prompt,
                "initial_prompt_hash": self._hash(agent.initial_prompt),
                "current_prompt": agent.current_prompt,
                "current_prompt_hash": self._hash(agent.current_prompt),
                "prompt_beam": agent.prompt_beam,
                "events": [],
            }
            for i, agent in enumerate(self.agents)
        }

    def _append_prompt_history_event(self, agent_id: int, epoch: int, step: int, decision: str, changed: bool):
        key = str(agent_id)
        agent = self.agents[agent_id]
        self.prompt_history.setdefault(key, {"events": []})
        self.prompt_history[key]["current_prompt"] = agent.current_prompt
        self.prompt_history[key]["current_prompt_hash"] = self._hash(agent.current_prompt)
        self.prompt_history[key]["prompt_beam"] = agent.prompt_beam
        self.prompt_history[key].setdefault("events", []).append(
            {
                "epoch": epoch,
                "step": step,
                "decision": decision,
                "changed": bool(changed),
                "current_prompt_hash": self._hash(agent.current_prompt),
            }
        )

    def _contains_task_specific_content(self, prompt: str, question: Optional[str] = None) -> bool:
        text = normalize_spaces(str(prompt)).lower()
        if "final_answer:" in text:
            return True
        if question:
            q = normalize_spaces(question).lower()
            words = [w for w in re.findall(r"[a-zA-Z0-9]{4,}", q) if len(w) >= 6]
            if len(words) >= 4:
                hits = sum(1 for w in set(words) if w in text)
                if hits >= min(4, max(2, len(set(words)) // 3)):
                    return True
        return False

    def _sanitize_prompt(self, prompt: str, agent_id: int, question: Optional[str] = None) -> Tuple[str, bool]:
        if self._contains_task_specific_content(prompt, question):
            return self.agents[agent_id].initial_prompt, True
        return str(prompt).strip(), False

    async def _chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        client_role: str = "evaluator",
    ) -> str:
        client = self.solver_client if client_role == "solver" else self.evaluator_client
        last_err: Optional[Exception] = None
        attempt = 0
        transient_failures = 0
        timeout_sec = float(self.cfg.llm_call_timeout or 0.0)
        while True:
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
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                transient = any(
                    x in msg
                    for x in [
                        "timeout",
                        "timed out",
                        "deadline",
                        "rate limit",
                        "temporarily",
                        "temporary",
                        "connection",
                        "server",
                        "overloaded",
                        "try again",
                        "503",
                        "502",
                        "504",
                    ]
                )
                if not transient:
                    if attempt >= max(1, int(self.cfg.max_retries)):
                        break
                else:
                    transient_failures += 1
                    if not self.cfg.transient_retry_forever and transient_failures > int(self.cfg.max_transient_retries or self.cfg.max_retries):
                        break
                sleep_sec = min(float(self.cfg.max_retry_backoff), float(self.cfg.retry_sleep) * (2 ** attempt))
                if self.cfg.llm_call_logging:
                    print(f"[LLM][retry] stage={stage} model={model} attempt={attempt + 1} sleep={sleep_sec:.2f} error={normalize_spaces(str(e))[:240]}", flush=True)
                await asyncio.sleep(sleep_sec)
                attempt += 1
        raise RuntimeError(f"LLM call failed at {stage}: {last_err}")

    async def solve_once(self, question: str, agent_id: int, prompt_text: str) -> Tuple[str, str]:
        effective_task = infer_task_type(task_type=self.cfg.task_type, question=question, answer=None)
        if effective_task == "mmlu":
            system_prompt = (
                "You are solving a multiple-choice question. Follow the agent role faithfully and make the role's "
                "decision procedure visible in a compact trace. Do not merely name the role; execute it. "
                "Avoid filler, avoid copying the question, and end with exactly one line: FINAL_ANSWER: <A/B/C/D>.\n\n"
                f"Agent role:\n{prompt_text}"
            )
        else:
            system_prompt = (
                "You are solving a reasoning problem. Follow the agent role faithfully and make the role's "
                "decision procedure visible in a compact trace. Do not merely name the role; execute it. "
                "Avoid filler, avoid copying the question, and end with exactly one line: FINAL_ANSWER: <answer>.\n\n"
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
        traces: List[Optional[str]] = [None for _ in range(n)]
        answers: List[Optional[str]] = [None for _ in range(n)]
        missing: List[int] = []
        reuse_hits = 0

        if self.cfg.candidate_reuse_recorded_rollouts:
            for agent_id in range(n):
                cached = self._lookup_solver_rollout(qh, prompts[agent_id], agent_id=agent_id)
                if isinstance(cached, dict) and "trace" in cached and "answer" in cached:
                    traces[agent_id] = str(cached.get("trace", ""))
                    answers[agent_id] = str(cached.get("answer", ""))
                    reuse_hits += 1
                else:
                    missing.append(agent_id)
        else:
            missing = list(range(n))

        async def solve_agent(agent_id: int):
            async with self.solver_call_semaphore:
                trace, answer = await self.solve_once(question, agent_id, prompts[agent_id])
                return agent_id, trace, answer

        if missing:
            outs = await asyncio.gather(*[solve_agent(i) for i in missing])
            for agent_id, trace, answer in outs:
                traces[agent_id] = trace
                answers[agent_id] = answer
                self._record_solver_rollout(
                    question_hash=qh,
                    prompt=prompts[agent_id],
                    trace=trace,
                    answer=answer,
                    agent_id=agent_id,
                    source=source,
                )

        final_traces = [str(t or "") for t in traces]
        final_answers = [str(a or "") for a in answers]
        stats = {
            "solver_reuse_enabled": bool(self.cfg.candidate_reuse_recorded_rollouts),
            "solver_reuse_hits": int(reuse_hits),
            "solver_reuse_misses": int(len(missing)),
            "solver_calls": int(len(missing)),
            "solver_reuse_total": int(n),
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

    async def solve_with_current_prompts(self, question: str) -> Tuple[List[str], List[str]]:
        return await self.solve_with_prompts(question, self._active_prompt_list())

    async def solve_with_agent_prompt_override(
        self,
        question: str,
        agent_id: int,
        prompt: str,
        peer_prompts: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        prompts = list(peer_prompts or self._active_prompt_list())
        while len(prompts) < len(self.agents):
            prompts.append(self.agents[len(prompts)].current_prompt)
        prompts[agent_id] = prompt
        return await self.solve_with_prompts(question, prompts)

    def rule_invalid_check(self, trace: str, answer: str = "") -> Dict[str, Any]:
        text = str(trace or "")
        reasons = []
        tokens = re.findall(r"\w+", text)
        if len(normalize_spaces(text)) < 40:
            reasons.append("trace_too_short")
        if "FINAL_ANSWER:" not in text:
            reasons.append("missing_final_answer")
        if len(tokens) < 12:
            reasons.append("too_few_tokens")
        if not str(answer or "").strip():
            reasons.append("empty_extracted_answer")
        if len(tokens) >= 12:
            bigrams = list(zip(tokens, tokens[1:]))
            repeated = len(bigrams) - len(set(bigrams))
            ratio = repeated / max(1, len(bigrams))
            if ratio > 0.35:
                reasons.append("bigram_repetition")
        return {"invalid": int(bool(reasons)), "reasons": reasons}

    def _load_embedding_model(self):
        if self.embedding_model is not None:
            return self.embedding_model
        from sentence_transformers import SentenceTransformer

        model_name = str(self.cfg.embedding_model)
        try:
            self.embedding_model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.embedding_model = SentenceTransformer(model_name)
        return self.embedding_model

    def _split_trace_for_embedding(self, text: str) -> List[str]:
        words = normalize_spaces(text).split()
        if not words:
            return []
        chunk_words = max(1, int(self.cfg.trace_embedding_chunk_words or 320))
        overlap = max(0, min(int(self.cfg.trace_embedding_chunk_overlap or 0), chunk_words - 1))
        if len(words) <= chunk_words:
            return [" ".join(words)]
        chunks = []
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            chunk = words[start : start + chunk_words]
            if chunk:
                chunks.append(" ".join(chunk))
            if start + chunk_words >= len(words):
                break
        return chunks

    def _normalize_vector(self, vector: Any) -> List[float]:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= 0.0 or np.isnan(norm):
            return []
        return (arr / norm).astype(float).tolist()

    def _encode_trace_document(self, trace: str) -> List[float]:
        cleaned = normalize_spaces(trace)
        if not cleaned:
            return []
        cache_key = self._hash(f"{self.cfg.embedding_model}|{self.cfg.trace_embedding_chunk_words}|{self.cfg.trace_embedding_chunk_overlap}|{cleaned}")
        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        chunks = self._split_trace_for_embedding(cleaned)
        if not chunks:
            return []
        model = self._load_embedding_model()
        embeddings = model.encode(chunks, normalize_embeddings=True)
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            pooled = arr
        else:
            pooled = np.mean(arr, axis=0)
        vector = self._normalize_vector(pooled)
        self.embedding_cache[cache_key] = vector
        return vector

    def _vector_cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 1.0
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom <= 0.0 or np.isnan(denom):
            return 1.0
        sim = float(np.dot(va, vb) / denom)
        return float(max(-1.0, min(1.0, sim)))

    def embedding_overlap_diagnostics(
        self,
        traces: List[str],
        prompts: Optional[List[str]] = None,
        invalids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        n = len(traces)
        invalids = list(invalids or [0 for _ in traces])
        embeddings = [self._encode_trace_document(trace) for trace in traces]
        pair_rows = []
        per_agent_scores = [0.0 for _ in range(n)]
        per_agent_counts = [0 for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (i < len(invalids) and int(invalids[i]) > 0) or (j < len(invalids) and int(invalids[j]) > 0):
                    sim = 1.0
                else:
                    sim = self._vector_cosine_similarity(embeddings[i], embeddings[j])
                    sim = max(0.0, sim)
                pair_rows.append({"pair": [i, j], "overlap": sim})
                per_agent_scores[i] += sim
                per_agent_scores[j] += sim
                per_agent_counts[i] += 1
                per_agent_counts[j] += 1
        per_agent_overlap = [
            float(per_agent_scores[i] / per_agent_counts[i]) if per_agent_counts[i] else 0.0
            for i in range(n)
        ]
        mean_overlap = float(np.mean([p["overlap"] for p in pair_rows])) if pair_rows else 0.0
        threshold = float(self.cfg.homogeneity_overlap_threshold)
        for row in pair_rows:
            i, j = row["pair"]
            row["invalid_pair"] = bool(
                (i < len(invalids) and int(invalids[i]) > 0)
                or (j < len(invalids) and int(invalids[j]) > 0)
            )
        high_pairs = [p for p in pair_rows if float(p["overlap"]) >= threshold]
        roles = [
            {
                "agent_id": i,
                "prompt_preview": normalize_spaces((prompts or self._active_prompt_list())[i])[:220] if i < len(prompts or []) else "",
                "trace_hash": self._hash(traces[i]),
                "trace_preview": normalize_spaces(traces[i])[:360],
                "overlap_pressure": per_agent_overlap[i],
            }
            for i in range(n)
        ]
        embedding_diversity = max(0.0, min(1.0, 1.0 - mean_overlap))
        return {
            "mean_embedding_overlap": mean_overlap,
            "embedding_diversity": embedding_diversity,
            "trace_embedding_model": str(self.cfg.embedding_model),
            "trace_embedding_chunk_words": int(self.cfg.trace_embedding_chunk_words),
            "trace_embedding_chunk_overlap": int(self.cfg.trace_embedding_chunk_overlap),
            "per_agent_overlap": per_agent_overlap,
            "pair_overlaps": pair_rows,
            "high_overlap_pairs": high_pairs,
            "homogeneity_overlap_threshold": threshold,
            "roles": roles,
        }

    def _vote_with_diagnostics(self, answers: List[str], question_hash: str = "") -> Dict[str, Any]:
        return majority_vote_with_diagnostics(
            answers,
            tie_break_method=str(getattr(self.cfg, "vote_tie_break", "random")),
            seed=int(getattr(self.cfg, "seed", 0) or 0),
            question_hash=question_hash,
        )

    def compute_rollout_metrics(
        self,
        traces: List[str],
        answers: List[str],
        gold: str,
        prompts: Optional[List[str]] = None,
        question_hash: str = "",
    ) -> Dict[str, Any]:
        vote = self._vote_with_diagnostics(answers, question_hash=question_hash)
        vote_answer = str(vote.get("vote_answer", ""))
        individual_correct = [int(self.task_spec.match_answer(a, gold)) for a in answers]
        vote_correct = int(self.task_spec.match_answer(vote_answer, gold))
        vote_fields = {
            "vote_answer": vote_answer,
            "vote_correct": vote_correct,
            "individual_correct": individual_correct,
            "vote_tie": bool(vote.get("vote_tie", False)),
            "tie_candidates": list(vote.get("tie_candidates", [])),
            "vote_counts": dict(vote.get("vote_counts", {})),
            "tie_break_method": str(vote.get("tie_break_method", "")),
        }
        if self._is_accuracy_only_mode():
            n = len(traces)
            active_prompts = prompts or self._active_prompt_list()
            roles = [
                {
                    "agent_id": i,
                    "prompt_preview": normalize_spaces(active_prompts[i])[:220] if i < len(active_prompts) else "",
                    "trace_hash": self._hash(traces[i]) if i < len(traces) else "",
                    "trace_preview": self._trace_method_preview(traces[i]) if i < len(traces) else "",
                    "overlap_pressure": 0.0,
                }
                for i in range(n)
            ]
            return {
                **vote_fields,
                "invalid_rate": 0.0,
                "invalid_score": 1.0,
                "invalid_flags": [0 for _ in traces],
                "mean_embedding_overlap": 0.0,
                "embedding_diversity": 0.0,
                "trace_embedding_model": "",
                "trace_embedding_chunk_words": int(self.cfg.trace_embedding_chunk_words),
                "trace_embedding_chunk_overlap": int(self.cfg.trace_embedding_chunk_overlap),
                "per_agent_overlap": [0.0 for _ in traces],
                "pair_overlaps": [],
                "high_overlap_pairs": [],
                "homogeneity_overlap_threshold": float(self.cfg.homogeneity_overlap_threshold),
                "roles": roles,
            }
        invalids = [self.rule_invalid_check(traces[i], answers[i] if i < len(answers) else "").get("invalid", 1) for i in range(len(traces))]
        overlap = self.embedding_overlap_diagnostics(traces, prompts, invalids=invalids)
        return {
            **vote_fields,
            "invalid_rate": float(np.mean(invalids)) if invalids else 1.0,
            "invalid_score": 1.0 - (float(np.mean(invalids)) if invalids else 1.0),
            "invalid_flags": [int(x) for x in invalids],
            **overlap,
        }

    def _trace_method_preview(self, trace: str, max_chars: int = 420) -> str:
        text = re.sub(r"FINAL_ANSWER\s*:\s*.*", "", str(trace or ""), flags=re.IGNORECASE)
        text = re.sub(r"\b(answer|final answer)\b\s*[:=-]?\s*[A-Da-d0-9.+/, \\-]+", "[answer redacted]", text, flags=re.IGNORECASE)
        return normalize_spaces(text)[:max_chars]

    def _case_key(self, sample_hash: str, a: int, b: int) -> str:
        left, right = sorted([int(a), int(b)])
        return f"{sample_hash}:{left}-{right}"

    def _build_homogeneous_cases(
        self,
        sample_hash: str,
        traces: List[str],
        answers: List[str],
        prompts: List[str],
        metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        invalids = [int(x) for x in metrics.get("invalid_flags", [])]
        cases: List[Dict[str, Any]] = []
        for pair in metrics.get("high_overlap_pairs", []):
            if not isinstance(pair, dict):
                continue
            ids = pair.get("pair", [])
            if not isinstance(ids, list) or len(ids) != 2:
                continue
            a, b = int(ids[0]), int(ids[1])
            if a >= len(traces) or b >= len(traces):
                continue
            if (a < len(invalids) and invalids[a]) or (b < len(invalids) and invalids[b]):
                continue
            overlap = float(pair.get("overlap", 0.0))
            for target, peer in [(a, b), (b, a)]:
                cases.append(
                    {
                        "case_id": self._hash(f"{sample_hash}|{target}|{peer}|{overlap:.6f}"),
                        "case_key": self._case_key(sample_hash, target, peer),
                        "sample_hash": sample_hash,
                        "target_agent_id": target,
                        "peer_agent_id": peer,
                        "pair_overlap": overlap,
                        "target_trace_preview": self._trace_method_preview(traces[target]),
                        "peer_trace_preview": self._trace_method_preview(traces[peer]),
                        "target_answer": str(answers[target]) if target < len(answers) else "",
                        "peer_answer": str(answers[peer]) if peer < len(answers) else "",
                        "target_prompt_preview": normalize_spaces(prompts[target])[:260] if target < len(prompts) else "",
                        "peer_prompt_preview": normalize_spaces(prompts[peer])[:260] if peer < len(prompts) else "",
                        "target_valid": True,
                        "peer_valid": True,
                        "team_correct": bool(metrics.get("vote_correct", 0)),
                        "case_type": "homogeneous_valid_pair",
                    }
                )
        return cases

    def _build_validity_cases(
        self,
        sample_hash: str,
        traces: List[str],
        answers: List[str],
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        rows = []
        for agent_id, trace in enumerate(traces):
            answer = answers[agent_id] if agent_id < len(answers) else ""
            invalid = self.rule_invalid_check(trace, answer)
            if not int(invalid.get("invalid", 0)):
                continue
            rows.append(
                {
                    "case_id": self._hash(f"{sample_hash}|invalid|{agent_id}|{self._hash(trace)}"),
                    "sample_hash": sample_hash,
                    "target_agent_id": agent_id,
                    "trace_preview": self._trace_method_preview(trace),
                    "answer_present": bool(str(answer).strip()),
                    "invalid_reasons": list(invalid.get("reasons", [])),
                    "target_prompt_preview": normalize_spaces(prompts[agent_id])[:260] if agent_id < len(prompts) else "",
                    "case_type": "hard_validity_case",
                }
            )
        return rows

    def is_homogeneity_window_warmup_done(self) -> bool:
        return all(len(a.recent_homogeneity_flags) >= self.homogeneity_window for a in self.agents)

    def is_update_window_ready(self) -> bool:
        if self._is_accuracy_only_mode():
            return len(self.recent_window_records) >= self.homogeneity_window
        return self.is_homogeneity_window_warmup_done()

    def clear_homogeneity_windows(self):
        for agent in self.agents:
            agent.recent_homogeneity_flags.clear()
            agent.homogeneity_count = 0
        self.recent_window_records = []

    def select_agents_for_update(self, metrics: Dict[str, Any]) -> List[int]:
        if not self.is_homogeneity_window_warmup_done():
            return []
        diagnosis = self._window_overlap_diagnosis(self.recent_window_records)
        pressures = list(diagnosis.get("per_agent_overlap_pressure", metrics.get("per_agent_overlap", [])))
        if not pressures or all(float(x) <= 0 for x in pressures):
            return []
        case_counts = diagnosis.get("homogeneous_case_counts", [])
        invalid_rates = diagnosis.get("per_agent_invalid_rate", [])
        tie_eps = float(self.cfg.homogeneity_pressure_tie_eps)
        ids = list(range(len(self.agents)))
        random.shuffle(ids)
        max_pressure = max(float(x) for x in pressures) if pressures else 0.0
        ids.sort(
            key=lambda i: (
                1 if (i < len(invalid_rates) and float(invalid_rates[i]) >= float(self.cfg.invalid_repair_rate_threshold)) else 0,
                round(float(pressures[i]) / max(tie_eps, 1e-6)) if i < len(pressures) else 0,
                int(case_counts[i]) if i < len(case_counts) else 0,
                int(self.agents[i].homogeneity_count),
                float(pressures[i]) if i < len(pressures) else 0.0,
            ),
            reverse=True,
        )
        ids = [
            i for i in ids
            if (
                (i < len(invalid_rates) and float(invalid_rates[i]) >= float(self.cfg.invalid_repair_rate_threshold))
                or (float(pressures[i]) if i < len(pressures) else 0.0) >= max(0.0, max_pressure - tie_eps)
            )
        ]
        if not ids:
            ids = list(range(len(self.agents)))
        active = sum(1 for a in self.agents if a.homogeneity_count > 0)
        return ids[: (2 if active >= 2 else 1)]

    def select_error_agents_for_update(self) -> List[int]:
        if not self.is_update_window_ready():
            return []
        wrong_counts = [0 for _ in range(len(self.agents))]
        team_wrong_counts = [0 for _ in range(len(self.agents))]
        seen_counts = [0 for _ in range(len(self.agents))]
        for rec in self.recent_window_records:
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            individual = list(metrics.get("individual_correct", []))
            team_correct = int(metrics.get("vote_correct", 0) or 0)
            for agent_id in range(len(self.agents)):
                if agent_id >= len(individual):
                    continue
                seen_counts[agent_id] += 1
                if not int(individual[agent_id]):
                    wrong_counts[agent_id] += 1
                    if not team_correct:
                        team_wrong_counts[agent_id] += 1
        ids = list(range(len(self.agents)))
        random.shuffle(ids)
        ids.sort(
            key=lambda i: (
                int(wrong_counts[i]),
                int(team_wrong_counts[i]),
                seen_counts[i] - wrong_counts[i],
            ),
            reverse=True,
        )
        ids = [i for i in ids if wrong_counts[i] > 0]
        return ids[: (2 if len(ids) >= 2 else 1)]

    def _window_accuracy_diagnosis(self, window_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        per_agent_seen = [0 for _ in range(len(self.agents))]
        per_agent_correct = [0 for _ in range(len(self.agents))]
        per_agent_team_wrong = [0 for _ in range(len(self.agents))]
        all_error_cases: List[Dict[str, Any]] = []
        focus_cases: List[Dict[str, Any]] = []
        for idx, rec in enumerate(window_records):
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            traces = list(rec.get("traces", []))
            answers = list(rec.get("answers", []))
            prompts = list(rec.get("prompts", []))
            individual = list(metrics.get("individual_correct", []))
            team_correct = int(metrics.get("vote_correct", 0) or 0)
            vote_answer = str(metrics.get("vote_answer", ""))
            row_cases = []
            for agent_id in range(len(self.agents)):
                if agent_id >= len(individual):
                    continue
                per_agent_seen[agent_id] += 1
                per_agent_correct[agent_id] += int(individual[agent_id])
                if int(individual[agent_id]):
                    continue
                if not team_correct:
                    per_agent_team_wrong[agent_id] += 1
                case = {
                    "case_id": self._hash(f"{rec.get('question_hash', '')}|accuracy_error|{agent_id}|{idx}"),
                    "case_type": "target_agent_answer_error",
                    "window_index": idx,
                    "sample_hash": rec.get("question_hash", ""),
                    "target_agent_id": agent_id,
                    "target_trace_preview": self._trace_method_preview(traces[agent_id]) if agent_id < len(traces) else "",
                    "target_answer": str(answers[agent_id]) if agent_id < len(answers) else "",
                    "team_vote_answer": vote_answer,
                    "team_correct": bool(team_correct),
                    "target_prompt_preview": normalize_spaces(prompts[agent_id])[:260] if agent_id < len(prompts) else "",
                }
                row_cases.append(case)
                all_error_cases.append(case)
            if row_cases:
                focus_cases.append(
                    {
                        "window_index": idx,
                        "team_correct": bool(team_correct),
                        "wrong_agent_ids": [int(c.get("target_agent_id", -1)) for c in row_cases],
                        "vote_answer": vote_answer,
                    }
                )
        per_agent_accuracy = [
            float(per_agent_correct[i] / per_agent_seen[i]) if per_agent_seen[i] else 0.0
            for i in range(len(self.agents))
        ]
        current_prompts = [
            {"agent_id": i, "prompt_preview": normalize_spaces(p)[:260], "prompt_hash": self._hash(p)}
            for i, p in enumerate(self._active_prompt_list())
        ]
        return {
            "window_size": len(window_records),
            "prompt_roles": current_prompts,
            "focus_cases": focus_cases[:5],
            "error_cases": all_error_cases,
            "per_agent_accuracy": per_agent_accuracy,
            "per_agent_error_count": [int(per_agent_seen[i] - per_agent_correct[i]) for i in range(len(self.agents))],
            "per_agent_team_wrong_error_count": per_agent_team_wrong,
            "team_accuracy": float(np.mean([int((rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}).get("vote_correct", 0) or 0) for rec in window_records])) if window_records else 0.0,
        }

    def _window_overlap_diagnosis(self, window_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = []
        for idx, rec in enumerate(window_records):
            metrics = rec.get("metrics", {})
            scored.append((float(metrics.get("mean_embedding_overlap", 0.0)), idx, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        focus_items = scored[: min(3, max(1, len(scored)))]
        all_homogeneous_cases: List[Dict[str, Any]] = []
        all_validity_cases: List[Dict[str, Any]] = []
        per_agent_invalid = [0 for _ in range(len(self.agents))]
        per_agent_seen = [0 for _ in range(len(self.agents))]
        per_agent_pressure_rows = [[] for _ in range(len(self.agents))]
        focus_cases = []
        for score, idx, rec in focus_items:
            metrics = rec.get("metrics", {})
            focus_cases.append(
                {
                    "window_index": idx,
                    "overlap_score": round(score, 4),
                    "high_overlap_pairs": metrics.get("high_overlap_pairs", []),
                    "roles": metrics.get("roles", []),
                }
            )
        for idx, rec in enumerate(window_records):
            metrics = rec.get("metrics", {})
            all_homogeneous_cases.extend(list(rec.get("homogeneous_cases", [])))
            all_validity_cases.extend(list(rec.get("validity_cases", [])))
            invalids = list(metrics.get("invalid_flags", []))
            pressures = list(metrics.get("per_agent_overlap", []))
            for agent_id in range(len(self.agents)):
                if agent_id < len(invalids):
                    per_agent_seen[agent_id] += 1
                    per_agent_invalid[agent_id] += int(invalids[agent_id])
                if agent_id < len(pressures):
                    per_agent_pressure_rows[agent_id].append(float(pressures[agent_id]))
        homogeneous_case_counts = [0 for _ in range(len(self.agents))]
        for case in all_homogeneous_cases:
            agent_id = int(case.get("target_agent_id", -1))
            if 0 <= agent_id < len(homogeneous_case_counts):
                homogeneous_case_counts[agent_id] += 1
        per_agent_invalid_rate = [
            float(per_agent_invalid[i] / per_agent_seen[i]) if per_agent_seen[i] else 0.0
            for i in range(len(self.agents))
        ]
        per_agent_pressure = [
            float(np.mean(rows)) if rows else 0.0
            for rows in per_agent_pressure_rows
        ]
        current_prompts = [
            {"agent_id": i, "prompt_preview": normalize_spaces(p)[:260], "prompt_hash": self._hash(p)}
            for i, p in enumerate(self._active_prompt_list())
        ]
        return {
            "window_size": len(window_records),
            "focus_cases": focus_cases,
            "prompt_roles": current_prompts,
            "mean_window_overlap": float(np.mean([x[0] for x in scored])) if scored else 0.0,
            "homogeneous_cases": sorted(all_homogeneous_cases, key=lambda c: float(c.get("pair_overlap", 0.0)), reverse=True),
            "validity_cases": all_validity_cases,
            "homogeneous_case_counts": homogeneous_case_counts,
            "per_agent_invalid_rate": per_agent_invalid_rate,
            "per_agent_overlap_pressure": per_agent_pressure,
            "homogeneity_overlap_threshold": float(self.cfg.homogeneity_overlap_threshold),
        }

    def _cases_for_agent(self, diagnosis: Dict[str, Any], agent_id: int) -> List[Dict[str, Any]]:
        return [
            c for c in diagnosis.get("homogeneous_cases", [])
            if isinstance(c, dict) and int(c.get("target_agent_id", -1)) == int(agent_id)
        ]

    def _validity_cases_for_agent(self, diagnosis: Dict[str, Any], agent_id: int) -> List[Dict[str, Any]]:
        return [
            c for c in diagnosis.get("validity_cases", [])
            if isinstance(c, dict) and int(c.get("target_agent_id", -1)) == int(agent_id)
        ]

    def _accuracy_cases_for_agent(self, diagnosis: Dict[str, Any], agent_id: int) -> List[Dict[str, Any]]:
        return [
            c for c in diagnosis.get("error_cases", [])
            if isinstance(c, dict) and int(c.get("target_agent_id", -1)) == int(agent_id)
        ]

    def _window_random_case_summaries(self, agent_id: int, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0 or not self.recent_window_records:
            return []
        records = list(self.recent_window_records)
        random.shuffle(records)
        rows = []
        for rec in records[:limit]:
            traces = list(rec.get("traces", []))
            answers = list(rec.get("answers", []))
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            if agent_id >= len(traces):
                continue
            rows.append(
                {
                    "case_type": "random_window_case",
                    "sample_hash": rec.get("question_hash", ""),
                    "target_agent_id": agent_id,
                    "target_trace_preview": self._trace_method_preview(traces[agent_id]),
                    "target_answer": str(answers[agent_id]) if agent_id < len(answers) else "",
                    "target_overlap_pressure": float(metrics.get("per_agent_overlap", [0.0] * len(self.agents))[agent_id]) if agent_id < len(metrics.get("per_agent_overlap", [])) else 0.0,
                    "team_correct": bool(metrics.get("vote_correct", 0)),
                }
            )
        return rows

    def _build_case_generation_batches(self, agent_id: int, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._is_accuracy_only_mode():
            error_cases = self._accuracy_cases_for_agent(diagnosis, agent_id)
            random_cases = [
                c for c in self._window_random_case_summaries(agent_id, max(0, int(self.cfg.random_window_cases_per_agent)))
                if isinstance(c, dict)
            ]
            batches = [
                {
                    "batch_type": "accuracy_error_cases",
                    "priority": 0,
                    "cases": error_cases[: max(1, int(self.cfg.max_homogeneous_cases_per_agent))],
                    "purpose": "repair target-agent answer mistakes observed in the current update window",
                },
                {
                    "batch_type": "mixed_window_accuracy_cases",
                    "priority": 1,
                    "cases": random_cases,
                    "purpose": "keep the revised prompt robust on nearby window examples while improving accuracy",
                },
            ]
            return [b for b in batches if b.get("cases") or str(b.get("batch_type")) == "accuracy_error_cases"]
        top_cases = self._cases_for_agent(diagnosis, agent_id)[: max(0, int(self.cfg.max_homogeneous_cases_per_agent))]
        random_cases = self._window_random_case_summaries(agent_id, max(0, int(self.cfg.random_window_cases_per_agent)))
        validity_cases = self._validity_cases_for_agent(diagnosis, agent_id)[: max(0, int(self.cfg.hard_validity_cases_per_agent))]
        invalid_rate = 0.0
        rates = diagnosis.get("per_agent_invalid_rate", [])
        if agent_id < len(rates):
            invalid_rate = float(rates[agent_id])
        batches = [
            {
                "batch_type": "high_overlap_cases",
                "priority": 0,
                "cases": top_cases,
                "purpose": "repair valid high-overlap reasoning pairs involving the target agent",
            },
            {
                "batch_type": "mixed_window_cases",
                "priority": 1,
                "cases": random_cases,
                "purpose": "avoid overfitting to only the highest-overlap cases",
            },
        ]
        if validity_cases or invalid_rate >= float(self.cfg.invalid_repair_rate_threshold):
            batches.append(
                {
                    "batch_type": "validity_focused_cases",
                    "priority": -1 if invalid_rate >= float(self.cfg.invalid_repair_rate_threshold) else 2,
                    "cases": validity_cases,
                    "purpose": "repair invalid or fragile target-agent outputs before pushing diversity",
                }
            )
        batches.sort(key=lambda x: int(x.get("priority", 0)))
        return batches

    def _optimizer_case_payload(self, case: Dict[str, Any]) -> Dict[str, Any]:
        allowed = [
            "target_agent_id",
            "peer_agent_id",
            "pair_overlap",
            "target_trace_preview",
            "peer_trace_preview",
            "target_prompt_preview",
            "peer_prompt_preview",
            "target_valid",
            "peer_valid",
            "case_type",
            "target_overlap_pressure",
            "invalid_reasons",
            "answer_present",
            "purpose",
            "target_answer",
            "team_vote_answer",
            "team_correct",
            "window_index",
        ]
        return {k: case.get(k) for k in allowed if k in case}

    def _target_case_keys(self, cases: List[Dict[str, Any]]) -> set:
        return {str(c.get("case_key", "")) for c in cases if str(c.get("case_key", ""))}

    def _homogeneity_impact_metrics(
        self,
        agent_id: int,
        rollout: Dict[str, Any],
        baseline_case_keys: set,
        sample_hash: str,
    ) -> Dict[str, Any]:
        high_pairs = [
            p for p in rollout.get("high_overlap_pairs", [])
            if isinstance(p, dict) and not bool(p.get("invalid_pair", False))
        ]
        target_pairs = []
        current_keys = set()
        target_pressure = 0.0
        pressures = list(rollout.get("per_agent_overlap", []))
        if agent_id < len(pressures):
            target_pressure = float(pressures[agent_id])
        for pair in high_pairs:
            ids = pair.get("pair", [])
            if not isinstance(ids, list) or len(ids) != 2:
                continue
            a, b = int(ids[0]), int(ids[1])
            if agent_id not in (a, b):
                continue
            target_pairs.append(pair)
            suffix = f"{a}-{b}" if a < b else f"{b}-{a}"
            current_keys.add(f"{sample_hash}:{suffix}")
        relevant_baselines = {x for x in baseline_case_keys if str(x).startswith(f"{sample_hash}:")}
        resolved = relevant_baselines - current_keys
        new_cases = current_keys - relevant_baselines
        return {
            "target_overlap_pressure": target_pressure,
            "homogeneous_case_count": int(len(target_pairs)),
            "resolved_case_count": int(len(resolved)),
            "new_homogeneous_case_count": int(len(new_cases)),
        }

    async def _propose_accuracy_candidates(
        self,
        agent_id: int,
        parent_prompt: str,
        accuracy_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt_roles = [r for r in accuracy_diagnosis.get("prompt_roles", []) if isinstance(r, dict)]
        target_role_spec = next((r for r in prompt_roles if int(r.get("agent_id", -1)) == int(agent_id)), {})
        peer_role_specs = [r for r in prompt_roles if int(r.get("agent_id", -1)) != int(agent_id)]
        error_counts = accuracy_diagnosis.get("per_agent_error_count", [])
        agent_acc = accuracy_diagnosis.get("per_agent_accuracy", [])
        window_stats = {
            "window_size": accuracy_diagnosis.get("window_size", 0),
            "team_accuracy": accuracy_diagnosis.get("team_accuracy", 0.0),
            "target_error_count": error_counts[agent_id] if agent_id < len(error_counts) else 0,
            "target_accuracy": agent_acc[agent_id] if agent_id < len(agent_acc) else 0.0,
        }
        safe_generation_batches = []
        for batch in generation_batches:
            if not isinstance(batch, dict):
                continue
            safe_generation_batches.append(
                {
                    **batch,
                    "cases": [
                        self._optimizer_case_payload(c)
                        for c in batch.get("cases", [])
                        if isinstance(c, dict)
                    ],
                }
            )
        if not safe_generation_batches:
            safe_generation_batches = [{"batch_type": "accuracy_error_cases", "cases": [], "purpose": "general accuracy repair"}]
        system_prompt = (
            "You are a prompt optimizer for a multi-agent reasoning team.\n"
            "Your only objective in this experiment is to improve final team answer accuracy.\n"
            "Use the parent prompt, prompt-role previews, window accuracy statistics, and target-agent error cases.\n"
            "Do not optimize for diversity, semantic overlap, local validity scores, invalid-rate metrics, or stylistic novelty.\n"
            "Do not use gold answers, concrete task text, options, labels, or answer-specific content.\n"
            "Treat trace previews as behavioral evidence of mistakes; do not copy their wording into the new prompt.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            "Revise the target agent prompt to reduce the observed answer mistakes.\n"
            "Each candidate must describe an executable reasoning procedure that can improve correctness on similar examples. "
            "Prefer concrete checks such as concept disambiguation, option comparison, contradiction testing, qualifier inspection, "
            "or final verification when they fit the observed mistake pattern.\n"
            "The prompt should remain short and usable by a solver agent. It must still end with exactly one final answer in normal solving, "
            "but do not include concrete answer labels or sample content inside candidate_prompt.\n"
            "Do not mention reward, beam search, candidates, evaluation metrics, or this optimizer instruction inside candidate_prompt.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"candidate_prompt": str, "role_name": str, "decision_procedure": [str, ...], "when_to_use": str, "fallback_strategy": str, "accuracy_checks": [str, ...], "rationale": str, "source_batch_type": str},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Return exactly requested_candidates distinct candidates. "
            "If multiple candidates use the same source_batch_type, they must repair the mistakes with meaningfully different executable procedures.\n\n"
            f"target_agent_id: {agent_id}\n"
            f"requested_candidates: {num_candidates}\n\n"
            f"current_parent_prompt:\n{parent_prompt}\n\n"
            f"target_role_spec:\n{json.dumps(target_role_spec, ensure_ascii=False, indent=2)}\n\n"
            f"peer_role_specs:\n{json.dumps(peer_role_specs, ensure_ascii=False, indent=2)}\n\n"
            f"window_accuracy_statistics:\n{json.dumps(window_stats, ensure_ascii=False, indent=2)}\n\n"
            f"generation_batches:\n{json.dumps(safe_generation_batches, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.cfg.optimizer_temperature),
            max_tokens=int(self.cfg.optimizer_max_tokens),
            stage=f"accuracy_optimizer_agent_{agent_id}",
        )
        obj = extract_json_obj(text) or {}
        candidates = obj.get("candidates", []) if isinstance(obj, dict) else []
        parsed: List[Dict[str, Any]] = []
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                prompt = str(item.get("candidate_prompt", "")).strip()
                prompt, _ = self._sanitize_prompt(prompt, agent_id)
                if not prompt:
                    continue
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(item.get("role_name", "")),
                        "decision_procedure": item.get("decision_procedure", []),
                        "when_to_use": str(item.get("when_to_use", "")),
                        "fallback_strategy": str(item.get("fallback_strategy", "")),
                        "accuracy_checks": item.get("accuracy_checks", []),
                        "rationale": str(item.get("rationale", "")),
                        "generation_batch_type": str(item.get("source_batch_type", "")) or str(generation_batches[batch_idx].get("batch_type", "")),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[batch_idx].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
                if len(parsed) >= num_candidates:
                    break
        while len(parsed) < num_candidates:
            batch_idx = min(len(parsed), len(generation_batches) - 1)
            parsed.append(
                {
                    "candidate_prompt": (
                        parent_prompt
                        + " Before finalizing, identify the most plausible trap in the question, compare the two strongest answer candidates against that trap, and perform one concise consistency check."
                    ),
                    "role_name": "fallback_accuracy_repair",
                    "decision_procedure": ["identify likely trap", "compare strongest candidates", "run one consistency check"],
                    "when_to_use": "Use when recent traces show answer-selection mistakes.",
                    "fallback_strategy": "If no trap is visible, use direct concept matching with one verification step.",
                    "accuracy_checks": ["compare plausible alternatives", "verify the final choice against the stem"],
                    "rationale": "Fallback candidate when optimizer returns too few usable prompts.",
                    "generation_batch_type": str(generation_batches[batch_idx].get("batch_type", "")),
                    "generation_case_ids": [
                        str(c.get("case_id", ""))
                        for c in generation_batches[batch_idx].get("cases", [])
                        if isinstance(c, dict)
                    ],
                }
            )
        return parsed[:num_candidates]

    async def propose_candidates(
        self,
        agent_id: int,
        parent_prompt: str,
        overlap_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batch: Optional[Dict[str, Any]] = None,
        generation_batches: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        prompt_roles = [
            r for r in overlap_diagnosis.get("prompt_roles", [])
            if isinstance(r, dict)
        ]
        target_role_spec = next((r for r in prompt_roles if int(r.get("agent_id", -1)) == int(agent_id)), {})
        peer_role_specs = [r for r in prompt_roles if int(r.get("agent_id", -1)) != int(agent_id)]
        if generation_batches is None:
            generation_batches = [dict(generation_batch or {"batch_type": "window_overlap_diagnosis", "cases": [], "purpose": "general window repair"})]
        generation_batches = [dict(x) for x in generation_batches if isinstance(x, dict)]
        if not generation_batches:
            generation_batches = [{"batch_type": "window_overlap_diagnosis", "cases": [], "purpose": "general window repair"}]
        if self._is_accuracy_only_mode():
            return await self._propose_accuracy_candidates(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                accuracy_diagnosis=overlap_diagnosis,
                num_candidates=num_candidates,
                generation_batches=generation_batches,
            )
        agent_pressures = overlap_diagnosis.get("per_agent_overlap_pressure", [])
        agent_invalid_rates = overlap_diagnosis.get("per_agent_invalid_rate", [])
        window_stats = {
            "mean_window_overlap": overlap_diagnosis.get("mean_window_overlap", 0.0),
            "homogeneity_overlap_threshold": overlap_diagnosis.get("homogeneity_overlap_threshold", self.cfg.homogeneity_overlap_threshold),
            "target_overlap_pressure": agent_pressures[agent_id] if agent_id < len(agent_pressures) else 0.0,
            "target_homogeneous_case_count": (overlap_diagnosis.get("homogeneous_case_counts", [0] * len(self.agents))[agent_id] if agent_id < len(overlap_diagnosis.get("homogeneous_case_counts", [])) else 0),
            "target_invalid_rate": agent_invalid_rates[agent_id] if agent_id < len(agent_invalid_rates) else 0.0,
        }
        validity_constraints = {
            "invalid_repair_priority": bool(window_stats["target_invalid_rate"] >= float(self.cfg.invalid_repair_rate_threshold)),
            "required_final_answer_line": True,
            "avoid_empty_or_repetitive_trace": True,
            "do_not_copy_case_content": True,
        }
        safe_generation_batches = []
        for batch in generation_batches:
            safe_generation_batches.append(
                {
                    **batch,
                    "cases": [
                        self._optimizer_case_payload(c)
                        for c in batch.get("cases", [])
                        if isinstance(c, dict)
                    ],
                }
            )
        system_prompt = (
            "You are a prompt optimizer for a multi-agent reasoning team.\n"
            "Generate executable role prompts that reduce full-trace embedding overlap while preserving answer reliability.\n"
            "Use only the supplied parent prompt, prompt-role previews, window statistics, and generation-batch diagnoses.\n"
            "The homogeneous cases were selected by the system, not by you. You are only a candidate prompt proposer.\n"
            "Do not use gold answers, concrete task text, options, labels, or answer-specific content.\n"
            "Treat trace previews as abstract behavioral evidence; do not copy their wording into the new prompt.\n"
            "Optimize for behavior that will be visible in the solver trace and easy to evaluate for role execution.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            "Revise the target agent prompt using the case-aware generation batches below.\n"
            "Each candidate must primarily address one supplied generation batch; do not merge all batches into one generic prompt.\n"
            "The new prompt must address the provided cases as reasoning-pattern evidence, not as sample content to memorize.\n"
            "Write concrete reasoning behavior, not slogans such as 'be diverse' or 'avoid redundancy'.\n"
            "Prefer a short role prompt with 2-4 explicit procedure steps, a fallback strategy, and validity checks.\n"
            "The prompt should create a different reasoning route from peer roles without making the answer less reliable.\n"
            "Never include concrete question text, answer text, options, labels, sample hashes, or FINAL_ANSWER templates.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"candidate_prompt": str, "role_name": str, "decision_procedure": [str, ...], "when_to_use": str, "fallback_strategy": str, "anti_overlap_rule": str, "validity_checks": [str, ...], "rationale": str, "source_batch_type": str},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Return exactly requested_candidates distinct candidates. "
            "Set source_batch_type to the exact batch_type that the candidate primarily addresses. "
            "If source_batch_type repeats, the repeated candidates must use meaningfully different executable procedures. "
            "Do not include batch names, sample identifiers, or meta-evaluation language inside candidate_prompt.\n\n"
            f"target_agent_id: {agent_id}\n"
            f"requested_candidates: {num_candidates}\n\n"
            f"current_parent_prompt:\n{parent_prompt}\n\n"
            f"target_role_spec:\n{json.dumps(target_role_spec, ensure_ascii=False, indent=2)}\n\n"
            f"peer_role_specs:\n{json.dumps(peer_role_specs, ensure_ascii=False, indent=2)}\n\n"
            f"window_overlap_statistics:\n{json.dumps(window_stats, ensure_ascii=False, indent=2)}\n\n"
            f"validity_constraints:\n{json.dumps(validity_constraints, ensure_ascii=False, indent=2)}\n\n"
            f"generation_batches:\n{json.dumps(safe_generation_batches, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.cfg.optimizer_temperature),
            max_tokens=int(self.cfg.optimizer_max_tokens),
            stage=f"optimizer_agent_{agent_id}",
        )
        obj = extract_json_obj(text) or {}
        candidates = obj.get("candidates", []) if isinstance(obj, dict) else []
        parsed: List[Dict[str, Any]] = []
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                prompt = str(item.get("candidate_prompt", "")).strip()
                prompt, _ = self._sanitize_prompt(prompt, agent_id)
                if not prompt:
                    continue
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(item.get("role_name", "")),
                        "decision_procedure": item.get("decision_procedure", []),
                        "when_to_use": str(item.get("when_to_use", "")),
                        "fallback_strategy": str(item.get("fallback_strategy", "")),
                        "anti_overlap_rule": str(item.get("anti_overlap_rule", "")),
                        "validity_checks": item.get("validity_checks", []),
                        "rationale": str(item.get("rationale", "")),
                        "generation_batch_type": str(item.get("source_batch_type", "")) or str(safe_generation_batches[min(len(parsed), len(safe_generation_batches) - 1)].get("batch_type", "")),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[min(len(parsed), len(generation_batches) - 1)].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
                if len(parsed) >= num_candidates:
                    break
        while len(parsed) < num_candidates:
            parsed.append(
                {
            "candidate_prompt": (
                parent_prompt
                        + " Use a distinct decision procedure: first state which reasoning route you will use, then approach the problem through boundary checks, reverse validation, or an alternative representation. If that procedure is not useful, fall back to direct reasoning with one explicit verification step."
                    ),
                    "role_name": "fallback_overlap_reducer",
                    "decision_procedure": ["detect likely overlap", "choose boundary/reverse/representation route", "verify"],
                    "when_to_use": "Use when the team's traces are semantically similar.",
                    "fallback_strategy": "Return to direct reasoning with one explicit check.",
                    "anti_overlap_rule": "Do not repeat the same decomposition order as the default solver.",
                    "validity_checks": ["trace shows the selected route", "answer line is present"],
                    "rationale": "Fallback candidate when optimizer returns too few usable prompts.",
                    "generation_batch_type": str(generation_batches[min(len(parsed), len(generation_batches) - 1)].get("batch_type", "")),
                    "generation_case_ids": [
                        str(c.get("case_id", ""))
                        for c in generation_batches[min(len(parsed), len(generation_batches) - 1)].get("cases", [])
                        if isinstance(c, dict)
                    ],
                }
            )
        return parsed[:num_candidates]

    async def evaluate_local_role_execution(
        self,
        agent_id: int,
        candidate_prompt: str,
        role_spec: Dict[str, Any],
        trace: str,
        answer: str,
    ) -> Dict[str, Any]:
        invalid = self.rule_invalid_check(trace, answer)
        cache_key = self._hash("|".join([str(agent_id), candidate_prompt, json.dumps(role_spec, sort_keys=True), trace]))
        if cache_key in self.local_validity_cache:
            return dict(self.local_validity_cache[cache_key])
        system_prompt = (
            "You evaluate whether one solver trace executed the candidate prompt's role.\n"
            "Judge role execution only. Do not judge answer correctness. Do not compare to other agents. "
            "Be strict: a trace is locally valid only when it actually follows the specified procedure, not when it merely mentions it. "
            "Return strict JSON only."
        )
        user_prompt = (
            "Return JSON with keys:\n"
            "{\n"
            '  "local_validity": 0,\n'
            '  "role_alignment": "aligned / partially_aligned / not_aligned",\n'
            '  "evidence": ["..."],\n'
            '  "reason": "..."\n'
            "}\n\n"
            "local_validity is 1 only if the trace clearly performs the executable role behavior with observable steps. "
            "It is 0 if the trace is invalid, only mentions the role superficially, follows a generic path, or skips the role's distinctive procedure. "
            "Use short evidence strings copied or paraphrased from the trace to justify the role-execution judgment. "
            "Do not require the answer to be correct; judge only role execution and output validity.\n\n"
            f"candidate_prompt:\n{candidate_prompt}\n\n"
            f"role_spec:\n{json.dumps(role_spec, ensure_ascii=False, indent=2)}\n\n"
            f"trace:\n{normalize_spaces(trace)}"
        )
        try:
            text = await self._chat(
                model=self.cfg.evaluator_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(self.cfg.evaluator_temperature),
                max_tokens=int(self.cfg.evaluator_max_tokens),
                stage=f"local_role_eval_agent_{agent_id}",
            )
            obj = extract_json_obj(text) or {}
            if not isinstance(obj, dict):
                obj = {}
        except Exception as e:
            obj = {"local_validity": 0, "role_alignment": "not_aligned", "evidence": [], "reason": normalize_spaces(str(e))[:240]}
        local_validity = 1 if int(obj.get("local_validity", 0) or 0) > 0 else 0
        if invalid["invalid"]:
            local_validity = 0
        evidence = obj.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        result = {
            "local_validity": int(local_validity),
            "role_alignment": str(obj.get("role_alignment", "aligned" if local_validity else "not_aligned")),
            "evidence": evidence,
            "reason": str(obj.get("reason", "")),
        }
        self.local_validity_cache[cache_key] = dict(result)
        return result

    async def evaluate_local_role_execution_batch(
        self,
        agent_id: int,
        candidate_prompt: str,
        role_spec: Dict[str, Any],
        trace_answer_rows: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        results: List[Optional[Dict[str, Any]]] = [None for _ in trace_answer_rows]
        pending: List[Dict[str, Any]] = []
        role_spec = dict(role_spec or {})
        for idx, row in enumerate(trace_answer_rows):
            trace = str(row.get("trace", ""))
            answer = str(row.get("answer", ""))
            invalid = self.rule_invalid_check(trace, answer)
            cache_key = self._hash("|".join([str(agent_id), candidate_prompt, json.dumps(role_spec, sort_keys=True), trace]))
            if cache_key in self.local_validity_cache:
                results[idx] = dict(self.local_validity_cache[cache_key])
                continue
            if int(invalid.get("invalid", 0)):
                result = {
                    "local_validity": 0,
                    "role_alignment": "not_aligned",
                    "evidence": [],
                    "reason": "Invalid trace: " + ", ".join(str(x) for x in invalid.get("reasons", [])),
                }
                self.local_validity_cache[cache_key] = dict(result)
                results[idx] = result
                continue
            pending.append({"idx": idx, "trace": trace, "answer": answer, "cache_key": cache_key})

        batch_size = max(1, int(getattr(self.cfg, "local_evaluator_batch_size", 5) or 5))
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            try:
                chunk_results = await self._evaluate_local_role_execution_chunk(
                    agent_id=agent_id,
                    candidate_prompt=candidate_prompt,
                    role_spec=role_spec,
                    rows=chunk,
                )
            except Exception:
                chunk_results = []
                for item in chunk:
                    chunk_results.append(
                        await self.evaluate_local_role_execution(
                            agent_id=agent_id,
                            candidate_prompt=candidate_prompt,
                            role_spec=role_spec,
                            trace=str(item.get("trace", "")),
                            answer=str(item.get("answer", "")),
                        )
                    )
            for item, result in zip(chunk, chunk_results):
                idx = int(item["idx"])
                result = dict(result or {})
                result["local_validity"] = 1 if int(result.get("local_validity", 0) or 0) > 0 else 0
                result.setdefault("role_alignment", "aligned" if result["local_validity"] else "not_aligned")
                evidence = result.get("evidence", [])
                result["evidence"] = evidence if isinstance(evidence, list) else []
                result.setdefault("reason", "")
                self.local_validity_cache[str(item["cache_key"])] = dict(result)
                results[idx] = result

        return [
            r
            if isinstance(r, dict)
            else {"local_validity": 0, "role_alignment": "not_aligned", "evidence": [], "reason": "missing batch evaluation"}
            for r in results
        ]

    async def _evaluate_local_role_execution_chunk(
        self,
        agent_id: int,
        candidate_prompt: str,
        role_spec: Dict[str, Any],
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []
        system_prompt = (
            "You evaluate whether each solver trace executed the candidate prompt's role.\n"
            "Judge each item independently for role execution only. Do not judge answer correctness. "
            "Do not compare items in this batch and do not compare to other agents. "
            "Be strict: a trace is locally valid only when it actually follows the specified procedure, not when it merely mentions it. "
            "Return strict JSON only."
        )
        payload = [
            {
                "item_id": int(row["idx"]),
                "trace": normalize_spaces(str(row.get("trace", ""))),
            }
            for row in rows
        ]
        user_prompt = (
            "Return JSON with keys:\n"
            "{\n"
            '  "evaluations": [\n'
            '    {"item_id": 0, "local_validity": 0, "role_alignment": "aligned / partially_aligned / not_aligned", "evidence": ["..."], "reason": "..."},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Return exactly one evaluation for every item_id, preserving each item_id. "
            "local_validity is 1 only if the trace clearly performs the executable role behavior with observable steps. "
            "It is 0 if the trace only mentions the role superficially, follows a generic path, or skips the role's distinctive procedure. "
            "Use at most two short evidence strings copied or paraphrased from that same trace only. "
            "If evidence is absent or purely generic, set local_validity to 0. "
            "Keep reason concise. Do not require the answer to be correct.\n\n"
            f"candidate_prompt:\n{candidate_prompt}\n\n"
            f"role_spec:\n{json.dumps(role_spec, ensure_ascii=False, indent=2)}\n\n"
            f"items:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.evaluator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.cfg.evaluator_temperature),
            max_tokens=int(self.cfg.evaluator_max_tokens),
            stage=f"local_role_eval_batch_agent_{agent_id}",
        )
        obj = extract_json_obj(text) or {}
        evaluations = obj.get("evaluations", []) if isinstance(obj, dict) else []
        by_id = {int(x.get("item_id", -1)): x for x in evaluations if isinstance(x, dict)}
        normalized = []
        for row in rows:
            item_id = int(row["idx"])
            obj_row = dict(by_id.get(item_id, {}))
            local_validity = 1 if int(obj_row.get("local_validity", 0) or 0) > 0 else 0
            evidence = obj_row.get("evidence", [])
            normalized.append(
                {
                    "local_validity": local_validity,
                    "role_alignment": str(obj_row.get("role_alignment", "aligned" if local_validity else "not_aligned")),
                    "evidence": evidence if isinstance(evidence, list) else [],
                    "reason": str(obj_row.get("reason", "")),
                }
            )
        return normalized

    async def evaluate_joint_trace_diversity(self, traces: List[str], candidate_agent_id: int) -> Dict[str, Any]:
        cache_key = self._hash("|".join([str(candidate_agent_id), *[self._hash(t) for t in traces]]))
        if cache_key in self.joint_diversity_cache:
            return dict(self.joint_diversity_cache[cache_key])
        system_prompt = (
            "You evaluate semantic diversity among a team's solver traces for diagnosis only. Return strict JSON only."
        )
        trace_payload = [
            {"agent_id": i, "trace": normalize_spaces(t)[:1800]}
            for i, t in enumerate(traces)
        ]
        user_prompt = (
            "Judge whether the candidate agent contributes a distinct reasoning behavior relative to the team.\n"
            "Do not use gold answers. Do not reward nonsense; invalid, vacuous, or copied traces are not diverse. "
            "This judgment is diagnostic and must not assume it controls prompt adoption.\n"
            "Return JSON:\n"
            "{\n"
            '  "joint_trace_diversity": 0.0,\n'
            '  "semantic_overlap_score": 0.0,\n'
            '  "candidate_agent_contribution": "distinct / redundant / harmful",\n'
            '  "redundant_agent_pairs": [[0, 1]],\n'
            '  "reason": "..."\n'
            "}\n\n"
            f"candidate_agent_id: {candidate_agent_id}\n"
            f"traces:\n{json.dumps(trace_payload, ensure_ascii=False, indent=2)}"
        )
        try:
            text = await self._chat(
                model=self.cfg.evaluator_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(self.cfg.evaluator_temperature),
                max_tokens=int(self.cfg.evaluator_max_tokens),
                stage=f"joint_trace_diversity_agent_{candidate_agent_id}",
            )
            obj = extract_json_obj(text) or {}
            if not isinstance(obj, dict):
                obj = {}
        except Exception as e:
            obj = {"joint_trace_diversity": 0.0, "semantic_overlap_score": 1.0, "candidate_agent_contribution": "harmful", "redundant_agent_pairs": [], "reason": normalize_spaces(str(e))[:240]}
        result = {
            "joint_trace_diversity": self._clip01(obj.get("joint_trace_diversity", 0.0)),
            "semantic_overlap_score": self._clip01(obj.get("semantic_overlap_score", 1.0)),
            "candidate_agent_contribution": str(obj.get("candidate_agent_contribution", "")),
            "redundant_agent_pairs": obj.get("redundant_agent_pairs", []) if isinstance(obj.get("redundant_agent_pairs", []), list) else [],
            "reason": str(obj.get("reason", "")),
        }
        self.joint_diversity_cache[cache_key] = dict(result)
        return result

    def _clip01(self, value: Any) -> float:
        try:
            x = float(value)
        except Exception:
            x = 0.0
        if np.isnan(x):
            return 0.0
        return float(max(0.0, min(1.0, x)))

    def _candidate_reward(self, diversity: float, local_validity: float, team_accuracy: float, invalid_score: float) -> float:
        return float(
            float(self.cfg.reward_weight_diversity) * self._clip01(diversity)
            + float(self.cfg.reward_weight_local_validity) * self._clip01(local_validity)
            + float(self.cfg.reward_weight_team_accuracy) * self._clip01(team_accuracy)
            + float(self.cfg.reward_weight_invalid_score) * self._clip01(invalid_score)
        )

    def _candidate_reward_guarded(
        self,
        baseline_team_accuracy: float,
        candidate_team_accuracy: float,
        baseline_embedding_diversity: float,
        candidate_embedding_diversity: float,
        baseline_invalid_rate: float,
        candidate_invalid_rate: float,
        local_validity: float,
    ) -> Dict[str, Any]:
        baseline_team_accuracy = self._clip01(baseline_team_accuracy)
        candidate_team_accuracy = self._clip01(candidate_team_accuracy)
        baseline_embedding_diversity = self._clip01(baseline_embedding_diversity)
        candidate_embedding_diversity = self._clip01(candidate_embedding_diversity)
        baseline_invalid_rate = self._clip01(baseline_invalid_rate)
        candidate_invalid_rate = self._clip01(candidate_invalid_rate)
        local_validity = self._clip01(local_validity)

        acc_delta = candidate_team_accuracy - baseline_team_accuracy
        div_delta = candidate_embedding_diversity - baseline_embedding_diversity
        invalid_delta = candidate_invalid_rate - baseline_invalid_rate
        guard_passed = candidate_team_accuracy >= baseline_team_accuracy - float(self.cfg.accuracy_guard_epsilon)
        if not guard_passed:
            reward = -1.0 + acc_delta - float(self.cfg.reward_weight_invalid_delta) * max(0.0, invalid_delta)
        else:
            reward = (
                candidate_team_accuracy
                + float(self.cfg.reward_weight_div_delta) * div_delta
                + float(self.cfg.reward_weight_local_validity) * local_validity
                - float(self.cfg.reward_weight_invalid_delta) * max(0.0, invalid_delta)
            )
        return {
            "reward": float(reward),
            "accuracy_delta": float(acc_delta),
            "diversity_delta": float(div_delta),
            "invalid_delta": float(invalid_delta),
            "accuracy_guard_passed": bool(guard_passed),
        }

    async def _evaluate_candidate_prompt_accuracy_only(
        self,
        agent_id: int,
        candidate_prompt: str,
        peer_prompts: Optional[List[str]],
        eval_batch: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        peer_prompts = list(peer_prompts or self._active_prompt_list())

        async def run_one(ex: Dict[str, str]) -> Dict[str, Any]:
            q = ex["question"]
            gold = self.task_spec.parse_gold(ex["answer"], q)
            eval_prompts = list(peer_prompts)
            while len(eval_prompts) < len(self.agents):
                eval_prompts.append(self.agents[len(eval_prompts)].current_prompt)
            eval_prompts[agent_id] = candidate_prompt
            traces, answers, reuse_stats = await self.solve_with_prompts_reusing_records(
                q,
                eval_prompts,
                source=f"candidate_accuracy_agent_{agent_id}",
            )
            vote = self._vote_with_diagnostics(answers, question_hash=self._hash(q))
            vote_answer = str(vote.get("vote_answer", ""))
            target_answer = answers[agent_id] if agent_id < len(answers) else ""
            return {
                "team_accuracy": int(self.task_spec.match_answer(vote_answer, gold)),
                "target_agent_accuracy": int(self.task_spec.match_answer(target_answer, gold)),
                "vote_answer": vote_answer,
                "vote_tie": bool(vote.get("vote_tie", False)),
                "tie_candidates": list(vote.get("tie_candidates", [])),
                "vote_counts": dict(vote.get("vote_counts", {})),
                "tie_break_method": str(vote.get("tie_break_method", "")),
                "target_answer": target_answer,
                "target_trace_hash": self._hash(traces[agent_id]) if agent_id < len(traces) else "",
                **reuse_stats,
            }

        raw = await asyncio.gather(*[run_one(ex) for ex in eval_batch], return_exceptions=True)
        rows = [r for r in raw if isinstance(r, dict)]
        errors = [normalize_spaces(str(r))[:240] for r in raw if isinstance(r, Exception)]
        team_accuracy = self._clip01(float(np.mean([float(r.get("team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        target_agent_accuracy = self._clip01(float(np.mean([float(r.get("target_agent_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        solver_reuse_hits = int(sum(int(r.get("solver_reuse_hits", 0) or 0) for r in rows))
        solver_reuse_misses = int(sum(int(r.get("solver_reuse_misses", 0) or 0) for r in rows))
        solver_calls = int(sum(int(r.get("solver_calls", 0) or 0) for r in rows))
        solver_reuse_total = int(sum(int(r.get("solver_reuse_total", 0) or 0) for r in rows))
        return {
            "reward": team_accuracy,
            "embedding_diversity": 0.0,
            "mean_embedding_overlap": 0.0,
            "target_overlap_pressure": 0.0,
            "homogeneous_case_count": 0.0,
            "resolved_case_count": 0.0,
            "new_homogeneous_case_count": 0.0,
            "local_validity_mean": 0.0,
            "team_accuracy": team_accuracy,
            "target_agent_accuracy": target_agent_accuracy,
            "invalid_rate": 0.0,
            "invalid_score": 1.0,
            "num_eval_samples": len(rows),
            "candidate_prompt": candidate_prompt,
            "errors": errors,
            "accuracy_only": True,
            "solver_reuse_enabled": bool(self.cfg.candidate_reuse_recorded_rollouts),
            "solver_reuse_hits": solver_reuse_hits,
            "solver_reuse_misses": solver_reuse_misses,
            "solver_calls": solver_calls,
            "solver_reuse_total": solver_reuse_total,
            "solver_reuse_hit_rate": float(solver_reuse_hits / solver_reuse_total) if solver_reuse_total else 0.0,
            "candidate_eval_strategy": str(getattr(self.cfg, "candidate_eval_strategy", "random")),
            "candidate_eval_pool_size": int(getattr(self.cfg, "candidate_eval_pool_size", 0) or 0),
            "candidate_eval_pool_actual_size": int(getattr(self.cfg, "candidate_eval_pool_actual_size", 0) or 0),
            "candidate_eval_batch_size": int(getattr(self.cfg, "candidate_eval_batch_size", 0) or 0),
            "actual_eval_batch_size": len(eval_batch),
            "num_eval_repeats": int(getattr(self.cfg, "candidate_eval_repeats", 1) or 1),
        }

    async def evaluate_candidate_prompt(
        self,
        agent_id: int,
        candidate_prompt: str,
        peer_prompts: Optional[List[str]],
        eval_batch: List[Dict[str, str]],
        role_spec: Optional[Dict[str, Any]] = None,
        baseline_homogeneous_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if self._is_accuracy_only_mode():
            return await self._evaluate_candidate_prompt_accuracy_only(
                agent_id=agent_id,
                candidate_prompt=candidate_prompt,
                peer_prompts=peer_prompts,
                eval_batch=eval_batch,
            )
        role_spec = dict(role_spec or {})
        peer_prompts = list(peer_prompts or self._active_prompt_list())
        baseline_case_keys = self._target_case_keys(list(baseline_homogeneous_cases or []))

        async def run_one(ex: Dict[str, str]) -> Dict[str, Any]:
            q = ex["question"]
            sample_hash = self._hash(q)
            gold = self.task_spec.parse_gold(ex["answer"], q)
            baseline_prompts = list(peer_prompts)
            while len(baseline_prompts) < len(self.agents):
                baseline_prompts.append(self.agents[len(baseline_prompts)].current_prompt)
            eval_prompts = list(baseline_prompts)
            eval_prompts[agent_id] = candidate_prompt
            baseline_rollout = {}
            baseline_reuse_stats: Dict[str, Any] = {}
            if self._is_guarded_reward_mode():
                baseline_traces, baseline_answers, baseline_reuse_stats = await self.solve_with_prompts_reusing_records(
                    q,
                    baseline_prompts,
                    source=f"candidate_baseline_agent_{agent_id}",
                )
                baseline_rollout = self.compute_rollout_metrics(
                    baseline_traces,
                    baseline_answers,
                    gold,
                    prompts=baseline_prompts,
                    question_hash=sample_hash,
                )
            traces, answers, reuse_stats = await self.solve_with_prompts_reusing_records(
                q,
                eval_prompts,
                source=f"candidate_eval_agent_{agent_id}",
            )
            rollout = self.compute_rollout_metrics(traces, answers, gold, prompts=eval_prompts, question_hash=sample_hash)
            agent_invalid = self.rule_invalid_check(traces[agent_id], answers[agent_id] if agent_id < len(answers) else "")
            diversity = float(rollout.get("embedding_diversity", 0.0))
            joint = {}
            if self.cfg.use_joint_trace_diversity_evaluator:
                joint = await self.evaluate_joint_trace_diversity(traces, agent_id)
            impact = self._homogeneity_impact_metrics(agent_id, rollout, baseline_case_keys, sample_hash)
            row = {
                "trace": traces[agent_id] if agent_id < len(traces) else "",
                "answer": answers[agent_id] if agent_id < len(answers) else "",
                "embedding_diversity": self._clip01(diversity),
                "team_accuracy": int(rollout.get("vote_correct", 0)),
                "invalid": float(agent_invalid.get("invalid", 1)),
                "invalid_reasons": agent_invalid.get("reasons", []),
                "mean_embedding_overlap": float(rollout.get("mean_embedding_overlap", 0.0)),
                **impact,
                "joint_trace_evaluation": joint,
                "trace_hash": self._hash(traces[agent_id]),
                **reuse_stats,
            }
            if self._is_guarded_reward_mode():
                row.update(
                    {
                        "baseline_team_accuracy": float(baseline_rollout.get("vote_correct", 0.0)),
                        "baseline_embedding_diversity": float(baseline_rollout.get("embedding_diversity", 0.0)),
                        "baseline_invalid_rate": float(baseline_rollout.get("invalid_rate", 1.0)),
                        "baseline_mean_embedding_overlap": float(baseline_rollout.get("mean_embedding_overlap", 0.0)),
                        "candidate_team_accuracy": float(rollout.get("vote_correct", 0.0)),
                        "candidate_embedding_diversity": float(rollout.get("embedding_diversity", 0.0)),
                        "candidate_invalid_rate": float(rollout.get("invalid_rate", 1.0)),
                        "candidate_mean_embedding_overlap": float(rollout.get("mean_embedding_overlap", 0.0)),
                        "baseline_solver_reuse_hits": int(baseline_reuse_stats.get("solver_reuse_hits", 0) or 0),
                        "baseline_solver_reuse_misses": int(baseline_reuse_stats.get("solver_reuse_misses", 0) or 0),
                        "baseline_solver_calls": int(baseline_reuse_stats.get("solver_calls", 0) or 0),
                        "baseline_solver_reuse_total": int(baseline_reuse_stats.get("solver_reuse_total", 0) or 0),
                    }
                )
            return row

        raw = await asyncio.gather(*[run_one(ex) for ex in eval_batch], return_exceptions=True)
        rows = [r for r in raw if isinstance(r, dict)]
        errors = [normalize_spaces(str(r))[:240] for r in raw if isinstance(r, Exception)]
        local_results = await self.evaluate_local_role_execution_batch(
            agent_id=agent_id,
            candidate_prompt=candidate_prompt,
            role_spec=role_spec,
            trace_answer_rows=[{"trace": str(r.get("trace", "")), "answer": str(r.get("answer", ""))} for r in rows],
        )
        for row, local in zip(rows, local_results):
            row["local_validity"] = int(local.get("local_validity", 0))
        diversity = self._clip01(float(np.mean([float(r.get("embedding_diversity", 0.0)) for r in rows])) if rows else 0.0)
        local_validity = self._clip01(float(np.mean([float(r.get("local_validity", 0.0)) for r in rows])) if rows else 0.0)
        team_accuracy = self._clip01(float(np.mean([float(r.get("team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        invalid_rate = self._clip01(float(np.mean([float(r.get("invalid", 1.0)) for r in rows])) if rows else 1.0)
        invalid_score = self._clip01(1.0 - invalid_rate)
        guarded_metrics: Dict[str, Any] = {}
        if self._is_guarded_reward_mode():
            baseline_team_accuracy = self._clip01(float(np.mean([float(r.get("baseline_team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
            candidate_team_accuracy = self._clip01(float(np.mean([float(r.get("candidate_team_accuracy", 0.0)) for r in rows])) if rows else team_accuracy)
            baseline_embedding_diversity = self._clip01(float(np.mean([float(r.get("baseline_embedding_diversity", 0.0)) for r in rows])) if rows else 0.0)
            candidate_embedding_diversity = self._clip01(float(np.mean([float(r.get("candidate_embedding_diversity", 0.0)) for r in rows])) if rows else diversity)
            baseline_invalid_rate = self._clip01(float(np.mean([float(r.get("baseline_invalid_rate", 1.0)) for r in rows])) if rows else 1.0)
            candidate_invalid_rate = self._clip01(float(np.mean([float(r.get("candidate_invalid_rate", 1.0)) for r in rows])) if rows else invalid_rate)
            guarded_metrics = self._candidate_reward_guarded(
                baseline_team_accuracy=baseline_team_accuracy,
                candidate_team_accuracy=candidate_team_accuracy,
                baseline_embedding_diversity=baseline_embedding_diversity,
                candidate_embedding_diversity=candidate_embedding_diversity,
                baseline_invalid_rate=baseline_invalid_rate,
                candidate_invalid_rate=candidate_invalid_rate,
                local_validity=local_validity,
            )
            guarded_metrics.update(
                {
                    "baseline_team_accuracy": baseline_team_accuracy,
                    "candidate_team_accuracy": candidate_team_accuracy,
                    "baseline_embedding_diversity": baseline_embedding_diversity,
                    "candidate_embedding_diversity": candidate_embedding_diversity,
                    "baseline_invalid_rate": baseline_invalid_rate,
                    "candidate_invalid_rate": candidate_invalid_rate,
                    "baseline_mean_embedding_overlap": self._clip01(float(np.mean([float(r.get("baseline_mean_embedding_overlap", 0.0)) for r in rows])) if rows else 0.0),
                    "candidate_mean_embedding_overlap": self._clip01(float(np.mean([float(r.get("candidate_mean_embedding_overlap", 0.0)) for r in rows])) if rows else 0.0),
                }
            )
            reward = float(guarded_metrics.get("reward", 0.0))
        else:
            reward = self._candidate_reward(diversity, local_validity, team_accuracy, invalid_score)
        solver_reuse_hits = int(sum(int(r.get("solver_reuse_hits", 0) or 0) for r in rows))
        solver_reuse_misses = int(sum(int(r.get("solver_reuse_misses", 0) or 0) for r in rows))
        solver_calls = int(sum(int(r.get("solver_calls", 0) or 0) for r in rows))
        solver_reuse_total = int(sum(int(r.get("solver_reuse_total", 0) or 0) for r in rows))
        result = {
            "reward": reward,
            "embedding_diversity": diversity,
            "mean_embedding_overlap": self._clip01(float(np.mean([float(r.get("mean_embedding_overlap", 0.0)) for r in rows])) if rows else 0.0),
            "target_overlap_pressure": self._clip01(float(np.mean([float(r.get("target_overlap_pressure", 0.0)) for r in rows])) if rows else 0.0),
            "homogeneous_case_count": float(np.mean([float(r.get("homogeneous_case_count", 0.0)) for r in rows])) if rows else 0.0,
            "resolved_case_count": float(np.mean([float(r.get("resolved_case_count", 0.0)) for r in rows])) if rows else 0.0,
            "new_homogeneous_case_count": float(np.mean([float(r.get("new_homogeneous_case_count", 0.0)) for r in rows])) if rows else 0.0,
            "local_validity_mean": local_validity,
            "team_accuracy": team_accuracy,
            "invalid_rate": invalid_rate,
            "invalid_score": invalid_score,
            "num_eval_samples": len(rows),
            "candidate_prompt": candidate_prompt,
            "errors": errors,
            "solver_reuse_enabled": bool(self.cfg.candidate_reuse_recorded_rollouts),
            "solver_reuse_hits": solver_reuse_hits,
            "solver_reuse_misses": solver_reuse_misses,
            "solver_calls": solver_calls,
            "solver_reuse_total": solver_reuse_total,
            "solver_reuse_hit_rate": float(solver_reuse_hits / solver_reuse_total) if solver_reuse_total else 0.0,
            "baseline_solver_calls": int(sum(int(r.get("baseline_solver_calls", 0) or 0) for r in rows)),
            "baseline_solver_reuse_hits": int(sum(int(r.get("baseline_solver_reuse_hits", 0) or 0) for r in rows)),
            "baseline_solver_reuse_misses": int(sum(int(r.get("baseline_solver_reuse_misses", 0) or 0) for r in rows)),
            "baseline_solver_reuse_total": int(sum(int(r.get("baseline_solver_reuse_total", 0) or 0) for r in rows)),
            "candidate_eval_strategy": str(getattr(self.cfg, "candidate_eval_strategy", "random")),
            "candidate_eval_pool_size": int(getattr(self.cfg, "candidate_eval_pool_size", 0) or 0),
            "candidate_eval_pool_actual_size": int(getattr(self.cfg, "candidate_eval_pool_actual_size", 0) or 0),
            "candidate_eval_batch_size": int(getattr(self.cfg, "candidate_eval_batch_size", 0) or 0),
            "actual_eval_batch_size": len(eval_batch),
            "num_eval_repeats": int(getattr(self.cfg, "candidate_eval_repeats", 1) or 1),
        }
        result.update(guarded_metrics)
        return result

    async def update_prompt_with_beam(
        self,
        agent_id: int,
        overlap_diagnosis: Dict[str, Any],
        eval_batch: List[Dict[str, str]],
        step_id: int,
        epoch_id: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        agent = self.agents[agent_id]
        beam = getattr(agent, "prompt_beam", []) or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]
        generation = max([int(x.get("generation", 0) or 0) for x in beam] + [0]) + 1
        candidate_pool: List[Dict[str, Any]] = []
        seen = set()
        generation_batches = self._build_case_generation_batches(agent_id, overlap_diagnosis)
        if not generation_batches:
            generation_batches = [{"batch_type": "window_overlap_diagnosis", "cases": [], "purpose": "general window repair"}]
        requested = max(1, int(self.cfg.num_candidates_per_parent))
        for parent in beam:
            parent_prompt = str(parent.get("prompt", agent.current_prompt))
            parent_id = str(parent.get("id", self._hash(parent_prompt)))
            parent_batches = [generation_batches[i % len(generation_batches)] for i in range(requested)]
            proposals = await self.propose_candidates(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                overlap_diagnosis=overlap_diagnosis,
                num_candidates=requested,
                generation_batches=parent_batches,
            )
            for idx, proposal in enumerate(proposals):
                prompt = str(proposal.get("candidate_prompt", "")).strip()
                prompt, _ = self._sanitize_prompt(prompt, agent_id)
                key = normalize_spaces(prompt).lower()
                if not prompt or key in seen:
                    continue
                seen.add(key)
                batch = parent_batches[idx % len(parent_batches)]
                candidate_pool.append(
                    {
                        "candidate_id": f"g{generation}_a{agent_id}_p{self._hash(parent_id)}_{idx}_{self._hash(prompt)}",
                        "prompt": prompt,
                        "parent_id": parent_id,
                        "generation": generation,
                        "source": "optimizer",
                        "generation_batch_type": str(proposal.get("generation_batch_type", "")) or str(batch.get("batch_type", "")),
                        "generation_case_ids": proposal.get("generation_case_ids", []),
                        "proposal": proposal,
                    }
                )
        for parent in beam:
            prompt = str(parent.get("prompt", agent.current_prompt))
            key = normalize_spaces(prompt).lower()
            if key in seen:
                continue
            seen.add(key)
            candidate_pool.append(
                {
                    "candidate_id": str(parent.get("id", "")) or f"beam_{self._hash(prompt)}",
                    "prompt": prompt,
                    "parent_id": parent.get("parent_id"),
                    "generation": int(parent.get("generation", 0) or 0),
                    "source": "existing_beam",
                    "proposal": {},
                }
            )

        evaluated = []
        peer_prompts = self._active_prompt_list()
        await self.ensure_recorded_rollouts_for_prompts(
            eval_batch=eval_batch,
            prompts=peer_prompts,
            source=f"candidate_peer_prewarm_agent_{agent_id}",
        )
        baseline_cases = self._cases_for_agent(overlap_diagnosis, agent_id)
        configured_concurrency = int(getattr(self.cfg, "candidate_eval_concurrency", 0) or 0)
        eval_concurrency = len(candidate_pool) if configured_concurrency <= 0 else min(configured_concurrency, len(candidate_pool))
        sem = asyncio.Semaphore(max(1, eval_concurrency))

        async def evaluate_one_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                metrics = await self.evaluate_candidate_prompt(
                    agent_id=agent_id,
                    candidate_prompt=str(candidate["prompt"]),
                    peer_prompts=peer_prompts,
                    eval_batch=eval_batch,
                    role_spec=candidate.get("proposal", {}),
                    baseline_homogeneous_cases=baseline_cases,
                )
                return {**candidate, "metrics": metrics, "reward": float(metrics.get("reward", 0.0))}

        raw_evaluated = await asyncio.gather(*[evaluate_one_candidate(c) for c in candidate_pool], return_exceptions=True)
        for idx, item in enumerate(raw_evaluated):
            if isinstance(item, dict):
                evaluated.append(item)
                continue
            candidate = candidate_pool[idx]
            metrics = await self.evaluate_candidate_prompt(
                agent_id=agent_id,
                candidate_prompt=str(candidate["prompt"]),
                peer_prompts=peer_prompts,
                eval_batch=eval_batch,
                role_spec=candidate.get("proposal", {}),
                baseline_homogeneous_cases=baseline_cases,
            )
            evaluated.append({**candidate, "metrics": metrics, "reward": float(metrics.get("reward", 0.0))})
        evaluated.sort(key=lambda x: float(x.get("reward", 0.0)), reverse=True)

        old_hash = self._hash(agent.current_prompt)
        beam_size = max(1, int(self.cfg.beam_size))
        selected = evaluated[:beam_size]
        agent.prompt_beam = [
            self._make_beam_item(
                prompt=str(x["prompt"]),
                score=float(x.get("reward", 0.0)),
                metrics=x.get("metrics", {}),
                parent_id=x.get("parent_id"),
                generation=int(x.get("generation", generation) or generation),
                candidate_id=str(x.get("candidate_id", "")) or None,
            )
            for x in selected
        ] or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]
        agent.current_prompt = str(agent.prompt_beam[0]["prompt"])
        changed = old_hash != self._hash(agent.current_prompt)
        if changed:
            agent.history.append(agent.current_prompt)
            agent.accept_count += 1
        else:
            agent.reject_count += 1

        for rank, item in enumerate(evaluated, start=1):
            metrics = item.get("metrics", {})
            accepted = rank <= len(agent.prompt_beam)
            self.update_logs.append(
                {
                    **self._base_log_fields(),
                    "epoch": epoch_id,
                    "step": step_id,
                    "agent_id": agent_id,
                    "search_mode": "evolutionary_beam",
                    "beam_size": beam_size,
                    "candidate_id": item.get("candidate_id", ""),
                    "parent_id": item.get("parent_id"),
                    "reward": float(metrics.get("reward", 0.0)),
                    "embedding_diversity": float(metrics.get("embedding_diversity", 0.0)),
                    "mean_embedding_overlap": float(metrics.get("mean_embedding_overlap", 0.0)),
                    "target_overlap_pressure": float(metrics.get("target_overlap_pressure", 0.0)),
                    "homogeneous_case_count": float(metrics.get("homogeneous_case_count", 0.0)),
                    "resolved_case_count": float(metrics.get("resolved_case_count", 0.0)),
                    "new_homogeneous_case_count": float(metrics.get("new_homogeneous_case_count", 0.0)),
                    "local_validity_mean": float(metrics.get("local_validity_mean", 0.0)),
                    "team_accuracy": float(metrics.get("team_accuracy", 0.0)),
                    "target_agent_accuracy": float(metrics.get("target_agent_accuracy", 0.0)),
                    "invalid_rate": float(metrics.get("invalid_rate", 0.0)),
                    "invalid_score": float(metrics.get("invalid_score", 0.0)),
                    "baseline_team_accuracy": float(metrics.get("baseline_team_accuracy", 0.0)),
                    "candidate_team_accuracy": float(metrics.get("candidate_team_accuracy", metrics.get("team_accuracy", 0.0))),
                    "accuracy_delta": float(metrics.get("accuracy_delta", 0.0)),
                    "baseline_embedding_diversity": float(metrics.get("baseline_embedding_diversity", 0.0)),
                    "candidate_embedding_diversity": float(metrics.get("candidate_embedding_diversity", metrics.get("embedding_diversity", 0.0))),
                    "diversity_delta": float(metrics.get("diversity_delta", 0.0)),
                    "baseline_invalid_rate": float(metrics.get("baseline_invalid_rate", 0.0)),
                    "candidate_invalid_rate": float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0))),
                    "invalid_delta": float(metrics.get("invalid_delta", 0.0)),
                    "accuracy_guard_passed": bool(metrics.get("accuracy_guard_passed", True)),
                    "solver_reuse_enabled": bool(metrics.get("solver_reuse_enabled", False)),
                    "solver_reuse_hits": int(metrics.get("solver_reuse_hits", 0)),
                    "solver_reuse_misses": int(metrics.get("solver_reuse_misses", 0)),
                    "solver_calls": int(metrics.get("solver_calls", 0)),
                    "solver_reuse_total": int(metrics.get("solver_reuse_total", 0)),
                    "solver_reuse_hit_rate": float(metrics.get("solver_reuse_hit_rate", 0.0)),
                    "accepted": bool(accepted),
                    "rank_in_beam": rank if accepted else None,
                    "beam_rank": rank if accepted else None,
                    "prompt_preview": normalize_spaces(str(item.get("prompt", "")))[:220],
                    "optimizer_model": self.cfg.optimizer_model,
                    "evaluator_model": self.cfg.evaluator_model,
                    "candidate_source": item.get("source", ""),
                    "generation_batch_type": item.get("generation_batch_type", ""),
                    "generation_case_ids": item.get("generation_case_ids", []),
                    "num_eval_samples": int(metrics.get("num_eval_samples", 0)),
                    "candidate_eval_strategy": str(metrics.get("candidate_eval_strategy", getattr(self.cfg, "candidate_eval_strategy", "random"))),
                    "candidate_eval_pool_size": int(metrics.get("candidate_eval_pool_size", getattr(self.cfg, "candidate_eval_pool_size", 0))),
                    "candidate_eval_pool_actual_size": int(metrics.get("candidate_eval_pool_actual_size", getattr(self.cfg, "candidate_eval_pool_actual_size", 0))),
                    "candidate_eval_batch_size": int(metrics.get("candidate_eval_batch_size", getattr(self.cfg, "candidate_eval_batch_size", 0))),
                    "actual_eval_batch_size": int(metrics.get("actual_eval_batch_size", metrics.get("num_eval_samples", 0))),
                    "num_eval_repeats": int(metrics.get("num_eval_repeats", getattr(self.cfg, "candidate_eval_repeats", 1))),
                }
            )
        self._append_prompt_history_event(agent_id, epoch_id, step_id, "beam_accept" if changed else "beam_keep", changed)
        summary = {
            "agent_id": agent_id,
            "updated": bool(changed),
            "candidate_count": len(candidate_pool),
            "generation_batches": generation_batches,
            "baseline_homogeneous_case_count": len(baseline_cases),
            "top_reward": float(agent.prompt_beam[0].get("score", 0.0) or 0.0),
            "top_metrics": agent.prompt_beam[0].get("metrics", {}),
        }
        agent.last_update_record = summary
        return bool(changed), summary

    async def refresh_all_prompt_beams(self, eval_batch: List[Dict[str, str]], epoch_id: int) -> Dict[str, Any]:
        if not self.cfg.beam_refresh_each_epoch or not eval_batch:
            return {"event": "beam_refresh", "enabled": False, "agent_count": 0}
        records = []
        for agent_id, agent in enumerate(self.agents):
            old_scores = [x.get("score") for x in getattr(agent, "prompt_beam", []) if isinstance(x, dict)]
            old_hash = self._hash(agent.current_prompt)
            refreshed = []
            peer_prompts = self._active_prompt_list()
            for item in getattr(agent, "prompt_beam", []) or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]:
                prompt = str(item.get("prompt", agent.current_prompt))
                metrics = await self.evaluate_candidate_prompt(agent_id, prompt, peer_prompts, eval_batch, role_spec=item.get("metrics", {}))
                refreshed.append(
                    self._make_beam_item(
                        prompt=prompt,
                        score=float(metrics.get("reward", 0.0)),
                        metrics=metrics,
                        parent_id=item.get("parent_id"),
                        generation=int(item.get("generation", 0) or 0),
                        candidate_id=str(item.get("id", "")) or None,
                    )
                )
            refreshed.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            agent.prompt_beam = refreshed[: max(1, int(self.cfg.beam_size))]
            agent.current_prompt = str(agent.prompt_beam[0]["prompt"])
            changed = old_hash != self._hash(agent.current_prompt)
            if changed:
                agent.history.append(agent.current_prompt)
            record = {
                **self._base_log_fields(),
                "event": "beam_refresh",
                "epoch": epoch_id,
                "step": 0,
                "agent_id": agent_id,
                "old_beam_scores": old_scores,
                "new_beam_scores": [x.get("score") for x in agent.prompt_beam],
                "active_prompt_changed": bool(changed),
                "beam_size": int(self.cfg.beam_size),
            }
            self.update_logs.append(record)
            self._append_prompt_history_event(agent_id, epoch_id, 0, "beam_refresh_changed" if changed else "beam_refresh_keep", changed)
            records.append(record)
        self.flush_update_logs()
        self.flush_prompt_history()
        return {
            "event": "beam_refresh",
            "enabled": True,
            "agent_count": len(records),
            "active_prompt_changed_count": int(sum(1 for r in records if r.get("active_prompt_changed"))),
        }

    async def maybe_update_prompts(self, metrics: Dict[str, Any], eval_batch: List[Dict[str, str]], step_id: int, epoch_id: int) -> Dict[str, Any]:
        if not self.is_update_window_ready():
            return {"update_requested": True, "update_ready": False, "selected_agent_ids": [], "updated_agent_ids": [], "skipped_reason": "window_not_ready"}
        if self._is_accuracy_only_mode():
            selected = self.select_error_agents_for_update()
            diagnosis = self._window_accuracy_diagnosis(self.recent_window_records)
            no_selection_reason = "no_agent_errors"
        else:
            selected = self.select_agents_for_update(metrics)
            diagnosis = self._window_overlap_diagnosis(self.recent_window_records)
            no_selection_reason = "no_overlap_pressure"
        if not selected:
            self.clear_homogeneity_windows()
            return {"update_requested": True, "update_ready": True, "selected_agent_ids": [], "updated_agent_ids": [], "skipped_reason": no_selection_reason}
        updated = []
        top_metrics = []
        for agent_id in selected:
            changed, summary = await self.update_prompt_with_beam(agent_id, diagnosis, eval_batch, step_id, epoch_id)
            if changed:
                updated.append(agent_id)
            if isinstance(summary.get("top_metrics", {}), dict):
                top_metrics.append(summary["top_metrics"])
        self.clear_homogeneity_windows()
        self.flush_update_logs()
        self.flush_prompt_history()
        return {
            "update_requested": True,
            "update_ready": True,
            "selected_agent_ids": selected,
            "updated_agent_ids": updated,
            "skipped_reason": "none",
            "candidate_behavior_diagnostics": self._mean_metric_dict(top_metrics),
        }

    def _mean_metric_dict(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        keys = [
            "reward",
            "embedding_diversity",
            "mean_embedding_overlap",
            "target_overlap_pressure",
            "homogeneous_case_count",
            "resolved_case_count",
            "new_homogeneous_case_count",
            "local_validity_mean",
            "team_accuracy",
            "target_agent_accuracy",
            "invalid_rate",
            "invalid_score",
            "solver_reuse_hits",
            "solver_reuse_misses",
            "solver_calls",
            "solver_reuse_total",
            "solver_reuse_hit_rate",
        ]
        return {k: float(np.mean([float(r.get(k, 0.0)) for r in rows])) if rows else 0.0 for k in keys}

    async def solve_train_example_without_update(
        self,
        question: str,
        gold: str,
    ) -> Dict[str, Any]:
        for i, agent in enumerate(self.agents):
            sanitized, changed = self._sanitize_prompt(agent.current_prompt, i, question)
            if changed:
                agent.current_prompt = sanitized
                if agent.prompt_beam:
                    agent.prompt_beam[0]["prompt"] = sanitized
        prompts = self._active_prompt_list()
        traces, answers = await self.solve_with_prompts(question, prompts)
        question_hash = self._hash(question)
        self._record_solver_rollouts(question_hash, prompts, traces, answers, source="train_rollout")
        metrics = self.compute_rollout_metrics(traces, answers, gold, prompts, question_hash=question_hash)
        if self._is_accuracy_only_mode():
            homogeneous_cases = []
            validity_cases = []
        else:
            homogeneous_cases = self._build_homogeneous_cases(question_hash, traces, answers, prompts, metrics)
            validity_cases = self._build_validity_cases(question_hash, traces, answers, prompts)
        return {
            "question_hash": question_hash,
            "gold": gold,
            "traces": traces,
            "answers": answers,
            "prompts": prompts,
            "metrics": metrics,
            "homogeneous_cases": homogeneous_cases,
            "validity_cases": validity_cases,
        }

    async def record_train_rollout(
        self,
        solved: Dict[str, Any],
        do_update: bool = True,
        eval_batch: Optional[List[Dict[str, str]]] = None,
        step_id: int = 0,
        epoch_id: int = 0,
    ) -> Dict[str, Any]:
        question_hash = str(solved.get("question_hash", ""))
        traces = list(solved.get("traces", []))
        answers = list(solved.get("answers", []))
        prompts = list(solved.get("prompts", []))
        metrics = dict(solved.get("metrics", {}))
        homogeneous_cases = list(solved.get("homogeneous_cases", []))
        validity_cases = list(solved.get("validity_cases", []))
        self.recent_window_records.append(
            {
                "question_hash": question_hash,
                "traces": traces,
                "answers": answers,
                "prompts": prompts,
                "metrics": metrics,
                "homogeneous_cases": homogeneous_cases,
                "validity_cases": validity_cases,
            }
        )
        self.recent_window_records = self.recent_window_records[-self.homogeneity_window :]
        if not self._is_accuracy_only_mode():
            for i, pressure in enumerate(metrics.get("per_agent_overlap", [])):
                self.agents[i].observe_homogeneity_result(1 if float(pressure) >= float(self.cfg.homogeneity_overlap_threshold) else 0)

        update_summary = {"update_requested": bool(do_update), "update_ready": self.is_update_window_ready(), "selected_agent_ids": [], "updated_agent_ids": []}
        if do_update and eval_batch is not None:
            update_summary = await self.maybe_update_prompts(metrics, eval_batch, step_id, epoch_id)
        record = {
            **self._base_log_fields(),
            "epoch": epoch_id,
            "step": step_id,
            "vote_correct": int(metrics.get("vote_correct", 0)),
            "vote_answer": metrics.get("vote_answer", ""),
            "vote_tie": bool(metrics.get("vote_tie", False)),
            "tie_candidates": metrics.get("tie_candidates", []),
            "vote_counts": metrics.get("vote_counts", {}),
            "tie_break_method": metrics.get("tie_break_method", ""),
            "embedding_diversity": float(metrics.get("embedding_diversity", 0.0)),
            "mean_embedding_overlap": float(metrics.get("mean_embedding_overlap", 0.0)),
            "homogeneous_case_count": len(homogeneous_cases),
            "validity_case_count": len(validity_cases),
            "invalid_rate": float(metrics.get("invalid_rate", 0.0)),
            "update_summary": update_summary,
        }
        self.train_step_logs.append(record)
        self.train_trace_history_logs.append(
            {
                **record,
                "question_hash": question_hash,
                "homogeneous_cases": homogeneous_cases,
                "validity_cases": validity_cases,
                "agents": [
                    {
                        "agent_id": i,
                        "prompt_hash": self._hash(prompts[i]),
                        "trace": traces[i],
                        "answer": answers[i],
                        "invalid": {"invalid": 0, "reasons": ["skipped_accuracy_only"]} if self._is_accuracy_only_mode() else self.rule_invalid_check(traces[i], answers[i]),
                    }
                    for i in range(len(self.agents))
                ],
            }
        )
        if len(self.train_step_logs) >= 20:
            self.flush_train_step_logs()
        if len(self.train_trace_history_logs) >= 20:
            self.flush_train_trace_history_logs()
        return {
            **metrics,
            "homogeneous_case_count": len(homogeneous_cases),
            "validity_case_count": len(validity_cases),
            "update_summary": update_summary,
        }

    async def rollout_train_example(
        self,
        question: str,
        gold: str,
        do_update: bool = True,
        eval_batch: Optional[List[Dict[str, str]]] = None,
        step_id: int = 0,
        epoch_id: int = 0,
    ) -> Dict[str, Any]:
        solved = await self.solve_train_example_without_update(question, gold)
        return await self.record_train_rollout(
            solved,
            do_update=do_update,
            eval_batch=eval_batch,
            step_id=step_id,
            epoch_id=epoch_id,
        )

    async def evaluate_dataset(self, data: List[Dict[str, str]], split_name: str = "test") -> Dict[str, Any]:
        prompts = self._active_prompt_list()

        async def evaluate_one(idx: int, ex: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            q = ex["question"]
            gold = self.task_spec.parse_gold(ex["answer"], q)
            traces, answers = await self.solve_with_prompts(q, prompts)
            question_hash = self._hash(q)
            self._record_solver_rollouts(question_hash, prompts, traces, answers, source=f"{split_name}_rollout")
            metrics = self.compute_rollout_metrics(traces, answers, gold, prompts, question_hash=question_hash)
            row = {"index": idx, "question_hash": question_hash, **metrics}
            prediction = {
                "index": idx,
                "question_hash": question_hash,
                "vote_answer": metrics.get("vote_answer", ""),
                "gold": gold,
                "vote_correct": int(metrics.get("vote_correct", 0)),
                "vote_tie": bool(metrics.get("vote_tie", False)),
                "tie_candidates": metrics.get("tie_candidates", []),
                "vote_counts": metrics.get("vote_counts", {}),
                "tie_break_method": metrics.get("tie_break_method", ""),
                "embedding_diversity": float(metrics.get("embedding_diversity", 0.0)),
                "mean_embedding_overlap": float(metrics.get("mean_embedding_overlap", 0.0)),
                "invalid_rate": float(metrics.get("invalid_rate", 0.0)),
                "agents": [
                    {
                        "agent_id": i,
                        "prompt_hash": self._hash(prompts[i]),
                        "trace": traces[i],
                        "answer": answers[i],
                        "invalid": {"invalid": 0, "reasons": ["skipped_accuracy_only"]} if self._is_accuracy_only_mode() else self.rule_invalid_check(traces[i], answers[i]),
                    }
                    for i in range(len(self.agents))
                ],
            }
            return row, prediction

        evaluated = await asyncio.gather(*[evaluate_one(idx, ex) for idx, ex in enumerate(data)])
        evaluated.sort(key=lambda x: int(x[0].get("index", 0)))
        rows = [row for row, _ in evaluated]
        predictions = [prediction for _, prediction in evaluated]
        pred_path = os.path.join(self.cfg.out_dir, f"{split_name}_predictions.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if split_name.startswith("test") or split_name.startswith("val"):
            self.test_trace_history_logs.extend(predictions)
            self.flush_test_trace_history_logs()
        return {
            "size": len(rows),
            "vote_acc": float(np.mean([r.get("vote_correct", 0) for r in rows])) if rows else 0.0,
            "vote_tie_rate": float(np.mean([1 if r.get("vote_tie", False) else 0 for r in rows])) if rows else 0.0,
            "mean_embedding_diversity": float(np.mean([r.get("embedding_diversity", 0.0) for r in rows])) if rows else 0.0,
            "mean_embedding_overlap": float(np.mean([r.get("mean_embedding_overlap", 0.0) for r in rows])) if rows else 0.0,
            "mean_invalid_rate": float(np.mean([r.get("invalid_rate", 0.0) for r in rows])) if rows else 0.0,
        }

    def save_state(self, name: str, extra: Optional[Dict[str, Any]] = None):
        payload = {
            **self._base_log_fields(),
            "agents": [
                {
                    "agent_id": i,
                    "initial_prompt": a.initial_prompt,
                    "current_prompt": a.current_prompt,
                    "prompt_beam": a.prompt_beam,
                    "history": a.history,
                    "accept_count": a.accept_count,
                    "reject_count": a.reject_count,
                }
                for i, a in enumerate(self.agents)
            ],
            "extra": extra or {},
        }
        with open(os.path.join(self.cfg.out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _flush_jsonl(self, filename: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        path = os.path.join(self.cfg.out_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def flush_update_logs(self):
        self._flush_jsonl("update_logs.jsonl", self.update_logs)
        self.update_logs = []

    def flush_train_step_logs(self):
        self._flush_jsonl("train_step_logs.jsonl", self.train_step_logs)
        self.train_step_logs = []

    def flush_train_trace_history_logs(self):
        self._flush_jsonl("train_trace_history.jsonl", self.train_trace_history_logs)
        self.train_trace_history_logs = []

    def flush_test_trace_history_logs(self):
        self._flush_jsonl("test_trace_history.jsonl", self.test_trace_history_logs)
        self.test_trace_history_logs = []

    def flush_prompt_history(self):
        with open(os.path.join(self.cfg.out_dir, "prompt_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.prompt_history, f, ensure_ascii=False, indent=2)


TextualGradientRLSystem = TraceBeamSearchSystem
