"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *
from ..mechanisms import cosine_similarity


class RuntimeStateMixin:
    def _attach_stable_mechanism_representation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metrics = item.setdefault("metrics", {})
        steps = metrics.get("mechanism_steps", metrics.get("mechanism_signature", []))
        representation = normalize_mechanism_representation(str(item.get("prompt", "")), steps)
        cache_key = representation["mechanism_hash"]
        vector = self.mechanism_embedding_cache.get(cache_key)
        if vector is not None:
            self.mechanism_embedding_cache_hit_count = int(getattr(self, "mechanism_embedding_cache_hit_count", 0)) + 1
        if vector is None and representation["mechanism_embedding_text"]:
            self.mechanism_embedding_cache_miss_count = int(getattr(self, "mechanism_embedding_cache_miss_count", 0)) + 1
            model = self._load_embedding_model()
            encoded = model.encode([representation["mechanism_embedding_text"]], normalize_embeddings=True)
            vector = self._normalize_vector(np.asarray(encoded)[0])
            self.mechanism_embedding_cache[cache_key] = list(vector)
        representation["mechanism_embedding"] = list(vector or [])
        if representation.get("family_kind") == "semantic" and vector:
            threshold = float(getattr(self.cfg, "semantic_niche_merge_threshold", 0.88))
            families = getattr(self, "semantic_mechanism_families", {})
            matching = [
                (family_id, cosine_similarity(vector, row.get("embedding", [])))
                for family_id, row in families.items()
                if str(family_id).startswith("semantic:")
            ]
            best_id, best_similarity = max(matching, key=lambda item: (item[1], item[0]), default=("", -1.0))
            if best_similarity >= threshold:
                representation["family_id"] = best_id
            else:
                family_id = str(representation["family_id"])
                families[family_id] = {
                    "representative_text": representation.get("semantic_residual_text", ""),
                    "embedding": list(vector),
                }
                self.semantic_mechanism_families = families
        metrics["mechanism_representation"] = representation
        metrics["normalized_operation_sequence"] = list(representation["normalized_operation_sequence"])
        return representation

    def _empty_cost_summary(self) -> Dict[str, Any]:
        return {
            "solver_calls": 0,
            "optimizer_calls": 0,
            "evaluator_calls": 0,
            "total_llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "tcs_teacher_calls": 0,
            "tcs_critic_calls": 0,
            "tcs_rewrite_calls": 0,
            "tcs_student_calls": 0,
            "open_exploration_calls": 0,
            "legacy_beam_refresh_calls": 0,
            "joint_refresh_count": 0,
            "joint_refresh_skipped_count": 0,
            "new_full_probe_prompt_count": 0,
            "new_full_probe_pair_count": 0,
            "full_probe_cache_hits": 0,
            "offline_team_combination_count": 0,
            "team_level_solver_calls": 0,
            "calls_saved_by_skipped_joint_refresh": 0,
            "calls_saved_by_dirty_prompt_cache": 0,
            "calls_saved_by_tcs_round_reduction": 0,
            "latency_seconds": 0.0,
            "candidate_eval_solver_api_calls": 0,
            "candidate_eval_cache_hits": 0,
            "candidate_eval_inflight_reuses": 0,
            "candidate_eval_calls_saved_vs_naive": 0,
            "candidate_eval_prompt_dedup_savings": 0,
            "full_probe_cache_hits": 0,
            "full_probe_missing_pair_evaluations": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
        }

    def _client_role_from_stage(self, stage: str, client_role: str) -> str:
        role = str(client_role or "").strip().lower()
        if role in {"solver", "optimizer"}:
            return role
        if "optimizer" in str(stage or "").lower():
            return "optimizer"
        return "evaluator"

    def _estimate_tokens(self, text: str) -> int:
        text = str(text or "")
        if not text:
            return 0
        words = len(re.findall(r"\S+", text))
        chars = len(text)
        return max(1, max(words, int(chars / 4)))

    def _usage_value(self, usage: Any, key: str, default: int = 0) -> int:
        if usage is None:
            return int(default)
        value = usage.get(key, default) if isinstance(usage, dict) else getattr(usage, key, default)
        try:
            return int(value or default)
        except Exception:
            return int(default)

    def _record_llm_call(
        self,
        *,
        stage: str,
        client_role: str,
        model: str,
        temperature: float,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        success: bool,
        error_type: str = "",
        audit_context: Optional[Mapping[str, Any]] = None,
    ):
        role = self._client_role_from_stage(stage, client_role)
        prompt_tokens = int(max(0, prompt_tokens))
        completion_tokens = int(max(0, completion_tokens))
        total_tokens = prompt_tokens + completion_tokens
        latency_seconds = float(max(0.0, latency_seconds))
        context = dict(TCS_AUDIT_CONTEXT.get() or {})
        context.update(dict(audit_context or {}))
        stage_name = str(context.get("llm_call_stage", "") or self._normalize_llm_call_stage(stage))
        model_role = str(context.get("model_role", "") or self._model_role_for_client_role(role))
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": str(stage or ""),
            "llm_call_stage": stage_name,
            "optimizer_architecture": str(context.get("optimizer_architecture", getattr(self.cfg, "optimizer_architecture", "")) or ""),
            "epoch": context.get("epoch"),
            "step": context.get("step"),
            "agent_id": context.get("agent_id"),
            "parent_id": context.get("parent_id"),
            "teacher_critic_round": context.get("teacher_critic_round"),
            "tcs_call_group_id": str(context.get("tcs_call_group_id", "") or ""),
            "execution_session_id": str(context.get("execution_session_id", getattr(self, "execution_session_id", "")) or getattr(self, "execution_session_id", "")),
            "update_attempt_id": str(context.get("update_attempt_id", "") or ""),
            "model_role": model_role,
            "model_name": str(model or ""),
            "client_role": role,
            "model": str(model or ""),
            "temperature": float(temperature),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": latency_seconds,
            "success": bool(success),
            "call_succeeded": bool(success),
            "response_empty": bool(context.get("response_empty", False)),
            "error_type": str(error_type or ""),
        }
        self.llm_call_logs.append(row)

        summary = self.cost_summary
        summary[f"{role}_calls"] = int(summary.get(f"{role}_calls", 0) or 0) + 1
        summary["total_llm_calls"] = int(summary.get("total_llm_calls", 0) or 0) + 1
        summary["prompt_tokens"] = int(summary.get("prompt_tokens", 0) or 0) + prompt_tokens
        summary["completion_tokens"] = int(summary.get("completion_tokens", 0) or 0) + completion_tokens
        summary["total_tokens"] = int(summary.get("total_tokens", 0) or 0) + total_tokens
        summary["estimated_cost"] = float(summary.get("estimated_cost", 0.0) or 0.0)
        summary["latency_seconds"] = float(summary.get("latency_seconds", 0.0) or 0.0) + latency_seconds

        if len(self.llm_call_logs) >= 20:
            self.flush_llm_call_logs()
            self.write_cost_summary()

    @staticmethod
    def _model_role_for_client_role(client_role: str) -> str:
        return {"optimizer": "optimizer", "evaluator": "evaluator", "solver": "agent"}.get(str(client_role), "evaluator")

    @staticmethod
    def _normalize_llm_call_stage(stage: str) -> str:
        lowered = str(stage or "").lower()
        if "teacher_rewrite" in lowered:
            return "teacher_rewrite"
        if "teacher_critic" in lowered:
            return "critic"
        if lowered.startswith("teacher_"):
            return "teacher"
        if "student_json_retry" in lowered:
            return "student_json_retry"
        if "student_json_repair" in lowered:
            return "student_json_repair"
        if "student_" in lowered:
            return "student"
        if "solver" in lowered:
            return "solver"
        return "one_shot_optimizer" if "optimizer" in lowered else lowered

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
            "cache_origin": "current_run",
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
        normalized = dict(row)
        normalized.setdefault("cache_origin", "current_run")
        self.solver_rollout_cache.setdefault(key, []).append(normalized)

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
                            persisted = dict(row)
                            persisted["cache_origin"] = "persisted"
                            self._add_solver_rollout_cache_row(persisted)
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
                                    "cache_origin": "persisted",
                                }
                            )
                            loaded += 1
            except Exception:
                continue
        if loaded and self.cfg.llm_call_logging:
            print(f"[solver-reuse] loaded recorded rollouts={loaded} unique_keys={len(self.solver_rollout_cache)}", flush=True)

    def _initialize_prompt_beams(self):
        for agent in self.agents:
            incumbent = self._make_beam_item(agent.current_prompt, None, {}, None, 0)
            incumbent.update({"is_incumbent": True, "archive_bucket": "safe"})
            agent.safe_qd_archive = [dict(incumbent)]
            agent.prompt_beam = [incumbent]

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
            "prompt_hash": self._normalized_prompt_hash(prompt_text),
        }

    def _refresh_joint_representatives(self, agent: AgentState) -> None:
        archive = list(getattr(agent, "safe_qd_archive", []) or [])
        if not archive:
            archive = [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]
        representatives = [dict(item) for item in select_joint_representatives(
            archive, self._normalized_prompt_hash(agent.current_prompt),
            int(self.cfg.joint_representative_beam_size), self.cfg,
        )] or [dict(archive[0])]
        active_hash = self._normalized_prompt_hash(agent.current_prompt)
        for item in representatives:
            metrics = item.setdefault("metrics", {})
            default_source = "incumbent" if str(item.get("prompt_hash", "")) == active_hash else "safe_archive_niche"
            item["beam_slot"] = str(item.get("beam_slot") or metrics.get("beam_slot") or default_source)
            metrics["beam_slot"] = item["beam_slot"]
        agent.prompt_beam = representatives

    def _record_stable_qd_archive_snapshot(
        self,
        *,
        agent_id: int,
        epoch: int,
        step: int,
        evaluated: Sequence[Dict[str, Any]],
        parent_sources: Sequence[str],
    ) -> None:
        agent = self.agents[agent_id]
        bucket_counts = {
            bucket: sum(str(item.get("archive_bucket", "")) == bucket for item in evaluated)
            for bucket in ("safe", "probation", "catastrophic")
        }
        self.quality_diversity_archive_history.append({
            "event": "quality_diversity_archive",
            "epoch": int(epoch),
            "step": int(step),
            "agent_id": int(agent_id),
            "safe_archive_size": len(getattr(agent, "safe_qd_archive", [])),
            "probation_archive_size": len(getattr(agent, "probation_archive", [])),
            "representative_count": len(getattr(agent, "prompt_beam", [])),
            "safe_candidate_count": int(bucket_counts["safe"]),
            "probation_candidate_count": int(bucket_counts["probation"]),
            "catastrophic_candidate_count": int(bucket_counts["catastrophic"]),
            "parent_sources": list(parent_sources),
            "safe_prompt_hashes": [str(item.get("prompt_hash", "")) for item in getattr(agent, "safe_qd_archive", [])],
            "probation_prompt_hashes": [str(item.get("prompt_hash", "")) for item in getattr(agent, "probation_archive", [])],
            "representative_prompt_hashes": [str(item.get("prompt_hash", "")) for item in getattr(agent, "prompt_beam", [])],
            "safe_niches": [
                [str(niche[0]), list(niche[1])]
                for item in getattr(agent, "safe_qd_archive", [])
                for niche in [mechanism_niche_key(item.get("metrics", {}).get("mechanism_representation", {}))]
            ],
        })

    def _make_refill_candidate(
        self,
        *,
        proposal: Dict[str, Any],
        prompt: str,
        parent_id: str,
        parent_prompt: str,
        agent_id: int,
        candidate_index: int,
        refill_round: int,
        generation: int,
    ) -> Dict[str, Any]:
        diagnostics = dict(proposal.get("optimizer_generation_diagnostics", {}) or {})
        return {
            "candidate_id": f"refill{refill_round}_a{agent_id}_{candidate_index}_{self._hash(prompt)}",
            "prompt": prompt,
            "prompt_hash": self._normalized_prompt_hash(prompt),
            "parent_id": parent_id,
            "parent_prompt": parent_prompt,
            "generation": generation + refill_round,
            "source": "optimizer",
            "candidate_pool_source": "optimizer",
            "candidate_source": str(
                proposal.get("candidate_source", "teacher_critic_student") or "teacher_critic_student"
            ),
            "optimizer_architecture": str(
                proposal.get("optimizer_architecture", diagnostics.get("optimizer_architecture", "")) or ""
            ),
            "optimizer_generation_diagnostics": diagnostics,
            "tcs_call_group_id": str(proposal.get("tcs_call_group_id", diagnostics.get("tcs_call_group_id", "")) or ""),
            "execution_session_id": str(
                proposal.get("execution_session_id", diagnostics.get("execution_session_id", self._current_execution_session_id()))
                or self._current_execution_session_id()
            ),
            "update_attempt_id": str(
                proposal.get("update_attempt_id", diagnostics.get("update_attempt_id", "")) or ""
            ),
            "refill_candidate": True,
            "proposal": proposal,
        }

    def _expire_probation_branches(self, epoch_id: int) -> int:
        expired = 0
        for agent in self.agents:
            expired += self._expire_agent_probation_branches(agent)
        self.probation_expired_count += expired
        return expired

    def _expire_agent_probation_branches(self, agent: AgentState) -> int:
        retained = []
        update_count = sum(int(value or 0) for value in agent.optimizer_update_count_by_epoch.values())
        expired = 0
        for item in getattr(agent, "probation_archive", []):
            born = int(item.get("probation_created_update", update_count) or update_count)
            if update_count - born >= int(self.cfg.probation_archive_ttl_updates):
                expired += 1
            else:
                retained.append(item)
        agent.probation_archive = retained
        return expired

    def expire_probation_branches(self, epoch_id: int) -> int:
        """Public epoch-end hook; TTL is measured in each agent's update turns."""
        return self._expire_probation_branches(epoch_id)

    def _current_joint_change_limit(self, epoch: int) -> int:
        early = float(self.specialization_strength) < float(self.cfg.joint_team_change_limit_switch_strength)
        base = int(self.cfg.joint_team_max_active_changes_early if early else self.cfg.joint_team_max_active_changes_late)
        return base + int(self.cfg.joint_team_change_limit_relaxation) if int(getattr(self, "qd_change_limit_relaxed_epoch", -1)) == int(epoch) else base

    def _select_stable_qd_parents(self, agent: AgentState, epoch_id: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Keep active exploitation while guaranteeing archived niches reproduction turns."""
        expired = self._expire_agent_probation_branches(agent)
        self.probation_expired_count += expired
        active_hash = self._normalized_prompt_hash(agent.current_prompt)
        active = next(
            (item for item in getattr(agent, "safe_qd_archive", []) if str(item.get("prompt_hash", "")) == active_hash),
            self._make_beam_item(agent.current_prompt, None, {}, None, 0),
        )
        alternate, source, niche = select_reproduction_parent(
            active,
            getattr(agent, "safe_qd_archive", []),
            getattr(agent, "probation_archive", []),
            agent.per_niche_parent_count,
            epoch=int(epoch_id),
            min_opportunities=int(self.cfg.qd_niche_min_parent_opportunities_per_epoch),
            allow_probation=bool(self.cfg.probation_parent_enabled),
        )
        parents, sources = [active], ["active"]
        if alternate is not None and str(alternate.get("prompt_hash", "")) != str(active.get("prompt_hash", "")):
            parents.append(alternate)
            sources.append(source)
            key = f"{int(epoch_id)}:{niche}"
            agent.per_niche_parent_count[key] = int(agent.per_niche_parent_count.get(key, 0) or 0) + 1
            if source == "probation_niche":
                agent.probation_parent_count += 1
        return parents, sources

    def _mark_mechanism_novelty(
        self,
        item: Dict[str, Any],
        *,
        parent: Optional[Dict[str, Any]],
        existing: Sequence[Dict[str, Any]],
    ) -> bool:
        novel = mechanism_is_novel(
            item,
            parent,
            existing,
            near_duplicate_threshold=float(self.cfg.mechanism_near_duplicate_similarity_threshold),
        )
        item.setdefault("metrics", {})["mechanism_novel"] = bool(novel)
        return bool(novel)

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
        requested_aggregation_mode = str(getattr(self.cfg, "aggregation_mode", "majority") or "majority")
        fields = {
            "execution_session_id": self._current_execution_session_id(),
            "comparison_task_id": getattr(self.cfg, "comparison_task_id", ""),
            "setting": getattr(self.cfg, "experiment_setting", ""),
            "benchmark": getattr(self.cfg, "benchmark", ""),
            "answer_format": getattr(self.cfg, "answer_format", ""),
            "task_type": self.cfg.task_type,
            "dataset_format": getattr(self.cfg, "dataset_format", ""),
            "agent_model": self.cfg.agent_model,
            "optimizer_model": self.cfg.optimizer_model,
            "evaluator_model": self.cfg.evaluator_model,
            "search_mode": self.cfg.search_mode,
            "reward_mode": self.cfg.reward_mode,
            "diversity_metric": self.cfg.diversity_metric,
            "embedding_model": self.cfg.embedding_model,
            "aggregation_mode": requested_aggregation_mode,
            "requested_aggregation_mode": requested_aggregation_mode,
            "effective_aggregation_mode": canonical_aggregation_mode(requested_aggregation_mode),
        }
        if bool(getattr(self.cfg, "competence_depth_enabled", False)):
            fields["plurality_boundary_version"] = PLURALITY_BOUNDARY_VERSION
        if self._is_rollout_qd_method():
            fields.update({
                "method_version": str(self.cfg.method_version),
                "mechanism_diversity_enabled": False,
                "mechanism_metadata_required": False,
                "mechanism_distance_used_for_selection": False,
                "mechanism_based_decision_count": int(getattr(self, "mechanism_based_decision_count", 0)),
                "capability_labeling_enabled": False,
                "prompt_text_diversity_used": False,
                "joint_active_team_selection_enabled": True,
                "quality_diversity_archive_enabled": True,
                    "rollout_diversity_enabled": True,
            })
        if self._is_state_conditioned_method():
            fields.update({
                "method_version": str(self.cfg.method_version),
                "state_conditioned_enabled": True,
                "state_conditioned_checkpoint_version": int(STATE_CONDITIONED_CHECKPOINT_VERSION),
                "fixed_acceptance_probe_enabled": True,
                "rollout_qd_method": False,
                "rollout_archive_enabled": False,
                "accuracy_is_primary_objective": True,
                "true_plurality_vote_delta_used": True,
                "wrong_answer_dispersion_used_for_generation": False,
                "wrong_answer_dispersion_used_for_reward": False,
                "wrong_answer_dispersion_used_for_selection": False,
                "diversity_is_noncollapse_constraint": True,
                "joint_team_enumeration_enabled": False,
                "joint_team_combination_count": 0,
                "per_agent_prompt_memory_capacity": int(self.cfg.state_prompt_memory_capacity),
                "update_mode": "sequential_single_agent",
            })
        if self._is_v82_hybrid():
            fields.update({
                "method_version": str(getattr(self.cfg, "method_version", "legacy")),
                "competence_schedule_version": str(getattr(self.cfg, "competence_schedule_version", "legacy")),
                "target_selector_version": str(getattr(self.cfg, "target_selector_version", "legacy")),
                "beam_policy_version": str(getattr(self.cfg, "beam_policy_version", "legacy")),
                "tcs_candidate_policy_version": str(getattr(self.cfg, "tcs_candidate_policy_version", "legacy")),
                "mechanism_signature_version": str(getattr(self.cfg, "mechanism_signature_version", "legacy")),
                "candidate_generation_policy_version": str(getattr(self.cfg, "candidate_generation_policy_version", "legacy")),
                "joint_refresh_policy_version": str(getattr(self.cfg, "joint_refresh_policy_version", "legacy")),
                "representative_probe_policy_version": str(getattr(self.cfg, "representative_probe_policy_version", "legacy")),
                "requested_beam_refresh_each_epoch": bool(getattr(self.cfg, "beam_refresh_each_epoch", False)),
                "effective_legacy_beam_refresh_each_epoch": bool(
                    getattr(self.cfg, "legacy_beam_rescore_each_epoch", False)
                    and not self._is_stable_qd_lineage()
                ),
            })
            if self._is_stable_qd_lineage():
                fields.update({
                    "active_team_selector_version": str(self.cfg.active_team_selector_version),
                    "lineage_policy_version": str(self.cfg.lineage_policy_version),
                    "mechanism_distance_version": str(self.cfg.mechanism_distance_version),
                    "joint_active_team_selection_enabled": True,
                    "quality_diversity_archive_enabled": True,
                    "probation_archive_enabled": bool(self.cfg.probation_archive_enabled),
                    "candidate_refill_enabled": bool(self.cfg.candidate_refill_enabled),
                    "early_self_drift_disabled": True,
                    "behavior_diversity_primary": True,
                    "mechanism_embedding_secondary": True,
                    "candidate_refill_version": str(self.cfg.candidate_refill_version),
                    "archive_policy_version": str(self.cfg.archive_policy_version),
                    "joint_quality_filter_version": str(self.cfg.joint_quality_filter_version),
                    "probe_stability_version": str(self.cfg.probe_stability_version),
                    "parent_selection_version": str(self.cfg.parent_selection_version),
                    "joint_refresh_mode": str(getattr(self.cfg, "joint_refresh_mode", "legacy")),
                    "joint_refresh_policy_version": str(getattr(self.cfg, "joint_refresh_policy_version", "legacy")),
                    "representative_probe_policy_version": str(getattr(self.cfg, "representative_probe_policy_version", "legacy")),
                    "joint_refresh_interval_epochs": int(getattr(self.cfg, "joint_refresh_interval_epochs", 0)),
                    "joint_refresh_max_dirty_candidates_per_agent": int(getattr(self.cfg, "joint_refresh_max_dirty_candidates_per_agent", 0)),
                })
        return fields

    def _read_previous_execution_session_id(self) -> str:
        path = os.path.join(self.cfg.out_dir, "run_meta.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return ""
        return str(payload.get("execution_session_id", "") or "") if isinstance(payload, dict) else ""

    def _current_execution_session_id(self) -> str:
        return str(getattr(self, "execution_session_id", "") or "")

    def _update_attempt_id(self, epoch_id: int, step_id: int, agent_id: int) -> str:
        return f"{self._current_execution_session_id()}_e{int(epoch_id)}_s{int(step_id)}_a{int(agent_id)}"

    def _tcs_call_group_id(
        self,
        update_attempt_id: str,
        parent_id: str,
        parent_prompt: str,
        generation_round: int = 0,
    ) -> str:
        return (
            f"{update_attempt_id}_p{self._hash(str(parent_id))}_"
            f"{self._hash(str(parent_prompt))}_r{int(generation_round)}"
        )

    @staticmethod
    def _git_provenance() -> Dict[str, Any]:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                # Generated run directories are often untracked. Dirty here
                # means tracked source/configuration changes, not fresh output.
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            return {"git_commit": commit, "git_dirty": bool(status.strip())}
        except (OSError, subprocess.SubprocessError):
            return {"git_commit": "", "git_dirty": None}

    def _split_integrity_metadata(self) -> Dict[str, Any]:
        raw = str(getattr(self.cfg, "split_integrity_json", "") or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"parse_error": "invalid_split_integrity_json"}
        return payload if isinstance(payload, dict) else {"parse_error": "split_integrity_json_is_not_an_object"}

    def write_run_meta(self):
        provenance = self._git_provenance()
        meta = {
            **self._base_log_fields(),
            "comparison_task_id": getattr(self.cfg, "comparison_task_id", ""),
            "benchmark": getattr(self.cfg, "benchmark", ""),
            "answer_format": getattr(self.cfg, "answer_format", ""),
            "dataset_format": getattr(self.cfg, "dataset_format", ""),
            "init_mode": self.cfg.init_mode,
            "agents": self.cfg.agents,
            "epochs": self.cfg.epochs,
            "train_size": self.cfg.train_size,
            "val_size": self.cfg.val_size,
            "test_size": self.cfg.test_size,
            "update_every": self.cfg.update_every,
            "beam_size": self.cfg.beam_size,
            "candidate_eval_batch_size": self.cfg.candidate_eval_batch_size,
            "candidate_eval_strategy": self.cfg.candidate_eval_strategy,
            "candidate_eval_data_source": str(getattr(self.cfg, "candidate_eval_data_source", "optimization_train")),
            "candidate_eval_repeats": self.cfg.candidate_eval_repeats,
            "candidate_eval_execution_mode": getattr(self.cfg, "candidate_eval_execution_mode", "legacy"),
            "solver_rollout_singleflight": bool(getattr(self.cfg, "solver_rollout_singleflight", True)),
            "candidate_eval_prompt_dedup": bool(getattr(self.cfg, "candidate_eval_prompt_dedup", True)),
            "candidate_eval_cache_logging": bool(getattr(self.cfg, "candidate_eval_cache_logging", True)),
            "candidate_selection_mode": getattr(self.cfg, "candidate_selection_mode", "scalar_reward"),
            "best_state_selection_mode": getattr(self.cfg, "best_state_selection_mode", "vote_first"),
            "method_version": str(getattr(self.cfg, "method_version", "legacy")),
            "state_conditioned_enabled": bool(self._is_state_conditioned_method()),
            "state_vote_objective_enabled": bool(getattr(self.cfg, "state_vote_objective_enabled", True)),
            "state_coverage_enabled": bool(getattr(self.cfg, "state_coverage_enabled", True)),
            "state_c2_correct_conversion_enabled": bool(getattr(self.cfg, "state_c2_correct_conversion_enabled", True)),
            "state_c2_wrong_split_enabled": bool(getattr(self.cfg, "state_c2_wrong_split_enabled", True)),
            "state_trace_tiebreak_enabled": bool(getattr(self.cfg, "state_trace_tiebreak_enabled", True)),
            "state_rollout_exploration_enabled": bool(getattr(self.cfg, "state_rollout_exploration_enabled", False)),
            "v9_update_mode": "sequential_single_agent" if self._is_state_conditioned_method() else None,
            "joint_team_enumeration_enabled": False if self._is_state_conditioned_method() else None,
            "joint_team_combination_count": 0 if self._is_state_conditioned_method() else None,
            "equal_vote_weighting": True if self._is_state_conditioned_method() else None,
            "wrong_answer_dispersion_used_for_reward": False if self._is_state_conditioned_method() else None,
            "wrong_answer_dispersion_used_for_selection": False if self._is_state_conditioned_method() else None,
            "accuracy_is_primary_objective": True if self._is_state_conditioned_method() else None,
            "diversity_is_constraint": True if self._is_state_conditioned_method() else None,
            "fixed_probe_state_snapshot_version": str(
                getattr(self, "fixed_probe_state_snapshot", {}).get("snapshot_version", "")
            ),
            "fixed_probe_state_snapshot_epoch": int(
                getattr(self, "fixed_probe_state_snapshot", {}).get("snapshot_epoch", 0) or 0
            ),
            "exploration_parent_use_count": int(getattr(self, "exploration_parent_use_count", 0)),
            "exploration_descendant_count": int(getattr(self, "exploration_descendant_count", 0)),
            "exploration_descendant_safe_count": int(getattr(self, "exploration_descendant_safe_count", 0)),
            "exploration_descendant_state_gain_count": int(getattr(self, "exploration_descendant_state_gain_count", 0)),
            "state_archive_slot_fill_counts": dict(getattr(self, "state_archive_slot_fill_counts", {})),
            "state_active_selection_source_counts": dict(getattr(self, "state_active_selection_source_counts", {})),
            "state_parent_selection_source_counts": dict(getattr(self, "state_parent_selection_source_counts", {})),
            "target_selector_mode": str(getattr(self.cfg, "target_selector_mode", "legacy")),
            "target_selector_version": str(getattr(self.cfg, "target_selector_version", "legacy")),
            "beam_policy_version": str(getattr(self.cfg, "beam_policy_version", "legacy")),
            "tcs_candidate_policy_version": str(getattr(self.cfg, "tcs_candidate_policy_version", "legacy")),
            "mechanism_signature_version": str(getattr(self.cfg, "mechanism_signature_version", "legacy")),
            "optimizer_architecture": getattr(self.cfg, "optimizer_architecture", ""),
            "optimizer_fallback_mode": getattr(self.cfg, "optimizer_fallback_mode", ""),
            "teacher_critic_max_rounds": getattr(self.cfg, "teacher_critic_max_rounds", 0),
            "teacher_question_pass_threshold": getattr(self.cfg, "teacher_question_pass_threshold", 0.0),
            "teacher_critic_use_voting_failure": bool(getattr(self.cfg, "teacher_critic_use_voting_failure", False)),
            "competence_schedule_mode": str(getattr(self.cfg, "competence_schedule_mode", "absolute_legacy")),
            "competence_schedule_version": str(getattr(self.cfg, "competence_schedule_version", "competence_depth_v1")),
            "competence_probe_size": int(getattr(self.cfg, "competence_probe_size", 0) or 0),
            "competence_probe_seed_offset": int(getattr(self.cfg, "competence_probe_seed_offset", 7000)),
            "competence_probe_question_hashes": list(getattr(self, "competence_probe_question_hashes", [])),
            "initial_competence_probe_metrics": dict(getattr(self, "initial_competence_probe_metrics", {})),
            "competence_relative_low_delta": float(getattr(self.cfg, "competence_relative_low_delta", 0.01)),
            "competence_relative_high_delta": float(getattr(self.cfg, "competence_relative_high_delta", 0.06)),
            "competence_schedule_ema": float(getattr(self.cfg, "competence_schedule_ema", 0.50)),
            "competence_schedule_max_step": float(getattr(self.cfg, "competence_schedule_max_step", 0.35)),
            "competence_schedule_monotonic": bool(getattr(self.cfg, "competence_schedule_monotonic", True)),
            "competence_mean_guard_epsilon": float(getattr(self.cfg, "competence_mean_guard_epsilon", 0.01)),
            "competence_c1_guard_epsilon": float(getattr(self.cfg, "competence_c1_guard_epsilon", 0.01)),
            "competence_c2_guard_epsilon": float(getattr(self.cfg, "competence_c2_guard_epsilon", 0.01)),
            "competence_depth1_candidate_guard_enabled": bool(getattr(self.cfg, "competence_depth1_candidate_guard_enabled", False)),
            "competence_depth1_candidate_guard_epsilon": float(getattr(self.cfg, "competence_depth1_candidate_guard_epsilon", 0.0)),
            "execution_session_id": self.execution_session_id,
            "previous_execution_session_id": self.previous_execution_session_id,
            "experiment_protocol_version": self._experiment_protocol_version(),
            "checkpoint_version": CHECKPOINT_VERSION,
            "state_conditioned_checkpoint_version": (
                int(STATE_CONDITIONED_CHECKPOINT_VERSION)
                if self._is_state_conditioned_method() else None
            ),
            **provenance,
            "split_integrity": self._split_integrity_metadata(),
            "model_role_map": {
                "agent_model": "solver rollouts for train/validation/test answering",
                "optimizer_model": (
                    "prompt-evolution generator calls: one_shot optimizer, TCS Teacher, "
                    "Teacher rewrite, Student, Student JSON retry, and Student JSON repair"
                ),
                "evaluator_model": (
                    "TCS Critic and optional joint trace diversity evaluator"
                ),
                "embedding_model": "local trace-embedding encoder for diversity diagnostics",
            },
            "initial_agent_prompts": self.initial_agent_prompts,
            "initial_agent_prompt_hashes": self.initial_agent_prompt_hashes,
            "config": self.cfg.to_flat_dict(),
            "framework": "accuracy_only_evolutionary_beam" if self._is_accuracy_only_mode() else "vote_oriented_evolutionary_beam",
        }
        if self._is_rollout_qd_method():
            meta.update({
                "rollout_distance_weights": {
                    "correctness_set": float(self.cfg.rollout_correct_distance_weight),
                    "useful_wrong_answer": float(self.cfg.rollout_wrong_distance_weight),
                    "trace_embedding": float(self.cfg.rollout_trace_distance_weight),
                },
                "rollout_quality_guards": {
                    "accuracy_guard_epsilon": float(self.cfg.accuracy_guard_epsilon),
                    "invalid_guard_epsilon": float(self.cfg.invalid_guard_epsilon),
                    "c3_loss_epsilon": int(self.cfg.rollout_c3_loss_epsilon),
                    "vote_loss_epsilon": int(self.cfg.rollout_vote_loss_epsilon),
                },
                "capability_profile_per_agent": None,
                "top_capability_family_per_agent": None,
            })
        if self._is_state_conditioned_method():
            diagnostics = dict(getattr(self, "state_search_diagnostics", {}) or {})
            informative = any(int(diagnostics.get(key, 0) or 0) > 0 for key in (
                "candidate_diversity_constraint_rejection_count",
                "correct_set_constraint_binding_count",
                "safe_trace_constraint_binding_count",
                "paired_safe_trace_available_count",
                "safe_diversity_parent_use_count",
            ))
            if not bool(self.cfg.state_diversity_constraints_enabled):
                uninformative_reason = "diversity_constraints_disabled"
            elif informative:
                uninformative_reason = ""
            elif int(diagnostics.get("candidate_diversity_constraint_evaluated_count", 0) or 0) == 0:
                uninformative_reason = "no_accuracy_valid_candidate_reached_diversity_constraints"
            else:
                uninformative_reason = "no_binding_rejection_paired_support_or_diversity_parent_use"
            initial_metrics = list(getattr(self, "initial_sequential_team_metrics", []) or [])
            initial_anchor = initial_metrics[0] if initial_metrics else {}
            meta.update({
                "state_conditioned_checkpoint_version": int(STATE_CONDITIONED_CHECKPOINT_VERSION),
                "accuracy_is_primary_objective": True,
                "true_plurality_vote_delta_used": True,
                "wrong_answer_dispersion_used_for_generation": False,
                "wrong_answer_dispersion_used_for_reward": False,
                "wrong_answer_dispersion_used_for_selection": False,
                "diversity_is_noncollapse_constraint": True,
                "joint_team_enumeration_enabled": False,
                "joint_team_combination_count": 0,
                "per_agent_prompt_memory_capacity": int(self.cfg.state_prompt_memory_capacity),
                "update_mode": "sequential_single_agent",
                "equal_vote_weighting": True,
                "rollout_qd_method": False,
                "rollout_archive_enabled": False,
                "deprecated_v9_fields_present": True,
                "deprecated_v9_fields_ignored": [
                    "state_c2_wrong_split_enabled", "state_trace_tiebreak_enabled",
                    "state_rollout_exploration_enabled", "state_exploration_parent_enabled",
                    "state_vote_objective_enabled", "state_joint_total_correct_slack_rate",
                    "state_representative_capacity",
                ],
                "initial_correct_set_diversity_mean": float(initial_anchor.get("correct_set_diversity_mean", 0.0)),
                "initial_correct_set_diversity_min": float(initial_anchor.get("correct_set_diversity_min", 0.0)),
                "initial_safe_trace_pair_count": int(sum(
                    int(row.get("safe_trace_pair_count", 0) or 0) for row in initial_metrics
                )),
                "state_search_diagnostics": diagnostics,
                "prompt_memory_occupancy_per_agent": [
                    len(getattr(agent, "prompt_memory", []) or []) for agent in self.agents
                ],
                "accepted_update_count": int(sum(
                    int(getattr(agent, "accept_count", 0) or 0) for agent in self.agents
                )),
                "accepted_accuracy_regression_count": int(
                    diagnostics.get("accepted_accuracy_regression_count", 0) or 0
                ),
                "diversity_ablation_informative": bool(informative),
                "diversity_ablation_uninformative_reason": uninformative_reason,
                "fixed_probe_state_snapshot": dict(getattr(self, "fixed_probe_state_snapshot", {})),
                "parent_selection_source_counts": dict(getattr(self, "state_parent_selection_source_counts", {})),
            })
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
