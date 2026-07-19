"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class TrainingControllerMixin:
    async def refresh_all_prompt_beams(self, eval_batch: List[Dict[str, str]], epoch_id: int) -> Dict[str, Any]:
        if self._is_stable_qd_lineage():
            for agent in self.agents:
                self._refresh_joint_representatives(agent)
            return {
                "event": "beam_refresh", "enabled": True, "mode": "safe_archive_representatives",
                "agent_count": len(self.agents), "active_prompt_changed_count": 0,
            }
        if not self.cfg.beam_refresh_each_epoch or not eval_batch:
            if self._residual_specialization_enabled():
                for agent in self.agents:
                    self._flush_capability_profile(agent, epoch_id, force=True)
            return {"event": "beam_refresh", "enabled": False, "agent_count": 0}
        records = []
        for agent_id, agent in enumerate(self.agents):
            old_scores = [x.get("score") for x in getattr(agent, "prompt_beam", []) if isinstance(x, dict)]
            old_hash = self._hash(agent.current_prompt)
            refreshed = []
            peer_prompts = self._active_prompt_list()
            for item in getattr(agent, "prompt_beam", []) or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]:
                prompt = str(item.get("prompt", agent.current_prompt))
                prior_metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
                metrics = await self.evaluate_candidate_prompt(agent_id, prompt, peer_prompts, eval_batch, role_spec=prior_metrics)
                if self._is_v82_hybrid():
                    for key in (
                        "candidate_type", "mechanism_signature", "parent_mechanism_signature",
                        "peer_dominant_mechanism_signature", "mechanism_signature_distance", "beam_slot",
                    ):
                        if key in prior_metrics:
                            metrics[key] = prior_metrics[key]
                refreshed_item = {
                        "candidate_id": str(item.get("id", "")) or self._hash(prompt),
                        "prompt": prompt,
                        "parent_id": item.get("parent_id"),
                        "parent_prompt": agent.current_prompt,
                        "generation": int(item.get("generation", 0) or 0),
                        "source": "existing_beam",
                        "candidate_pool_source": "existing_beam",
                        "candidate_source": "existing_beam",
                        "metrics": metrics,
                        "reward": float(metrics.get("reward", 0.0)),
                    }
                if self._v7_residual_protocol_enabled():
                    metrics.update(self._candidate_trajectory_feasibility(agent, refreshed_item))
                    refreshed_item["metrics"] = metrics
                if self._is_v82_hybrid():
                    metrics = self._apply_hybrid_soft_guards(metrics)
                    refreshed_item["metrics"] = metrics
                    refreshed_item["reward"] = float(metrics.get("penalized_reward", metrics.get("reward", 0.0)) or 0.0)
                self._apply_competence_depth1_candidate_guard(metrics)
                refreshed.append(refreshed_item)
            if self._v7_residual_protocol_enabled() or bool(getattr(self.cfg, "competence_depth1_candidate_guard_enabled", False)):
                refreshed = [item for item in refreshed if not str(item.get("metrics", {}).get("rejection_reason", ""))]
                if not refreshed:
                    raise RuntimeError("Beam refresh trajectory guard removed the current active prompt")
            if self._is_v82_hybrid():
                retained, _ = self._select_hybrid_beam(
                    refreshed, max(1, int(self.cfg.beam_size)), agent.current_prompt,
                    agent_id=agent_id, epoch_id=epoch_id, step_id=0,
                )
            elif self._uses_vote_pareto_selection():
                retained, _ = self._select_vote_pareto_beam(refreshed, max(1, int(self.cfg.beam_size)), agent.current_prompt)
            else:
                retained = sorted(refreshed, key=lambda item: float(item.get("reward", 0.0)), reverse=True)[: max(1, int(self.cfg.beam_size))]
            agent.prompt_beam = [
                self._make_beam_item(
                    prompt=str(item["prompt"]),
                    score=float(item.get("reward", 0.0)),
                    metrics=item.get("metrics", {}),
                    parent_id=item.get("parent_id"),
                    generation=int(item.get("generation", 0) or 0),
                    candidate_id=str(item.get("candidate_id", "")) or None,
                )
                for item in retained
            ]
            agent.current_prompt = str(agent.prompt_beam[0]["prompt"])
            changed = old_hash != self._hash(agent.current_prompt)
            if changed:
                agent.history.append(agent.current_prompt)
                agent.accept_count += 1
                if self._v7_residual_protocol_enabled():
                    profile_before = dict(agent.capability_profile)
                    active_metrics = retained[0].get("metrics", {})
                    if self._residual_specialization_enabled():
                        self._update_vote_context_profile(agent, active_metrics)
                        self._accumulate_capability_evidence(agent, active_metrics, epoch_id)
                        self._flush_capability_profile(agent, epoch_id, force=False)
                    agent.last_accepted_prompt_hash = self._normalized_prompt_hash(agent.current_prompt)
                    fingerprint = {
                        str(key): BehaviorFingerprintEntry.from_dict(value)
                        for key, value in dict(active_metrics.get("behavior_fingerprint", {})).items()
                        if isinstance(value, dict)
                    }
                    self._append_bounded_archive(
                        agent.accepted_behavior_archive,
                        BehaviorStateSummary(
                            state_id=f"e{int(epoch_id)}_refresh_a{int(agent_id)}_{str(retained[0].get('candidate_id', ''))}",
                            epoch=int(epoch_id),
                            prompt_hash=agent.last_accepted_prompt_hash,
                            behavior_fingerprint=fingerprint,
                            transition_vector={str(key): float(value) for key, value in dict(active_metrics.get("candidate_transition_vector", {})).items()},
                            target_accuracy=float(active_metrics.get("candidate_target_accuracy", 0.0) or 0.0),
                            team_vote_accuracy=float(active_metrics.get("candidate_team_accuracy", 0.0) or 0.0),
                            mean_vote_margin=float(active_metrics.get("candidate_mean_vote_margin", 0.0) or 0.0),
                            preserved_mechanisms=[],
                            capability_profile=dict(agent.capability_profile),
                            paired_behavior_utility=self.behavior_fingerprint_utility(fingerprint),
                        ),
                    )
                    self.trajectory_events.append(self._trajectory_event(
                        agent_id=agent_id,
                        epoch_id=epoch_id,
                        step_id=0,
                        item=retained[0],
                        accepted=True,
                        profile_before=profile_before,
                        profile_after=dict(agent.capability_profile),
                    ))
                    self.trajectory_events[-1]["decision"] = "beam_refresh_activated"
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
        if self._residual_specialization_enabled():
            for agent in self.agents:
                self._flush_capability_profile(agent, epoch_id, force=True)
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
            diagnosis = self._window_accuracy_diagnosis(self.recent_window_records)
            selected = self.select_reward_agents_for_update(diagnosis, metrics)
            no_selection_reason = "no_reward_relevant_agent"
        else:
            diagnosis = self._window_update_diagnosis(self.recent_window_records)
            selected = self.select_reward_agents_for_update(diagnosis, metrics)
            no_selection_reason = "no_reward_relevant_agent"
        if self._is_v82_hybrid():
            for row in diagnosis.get("hybrid_selector_diagnostics", []):
                row["selected"] = int(row.get("agent_id", -1)) in selected
            missed_niche_agents = []
            if self._is_stable_qd_lineage():
                for agent_id, agent in enumerate(self.agents):
                    if agent_id in selected:
                        continue
                    active_hash = self._normalized_prompt_hash(agent.current_prompt)
                    has_branch = any(
                        str(item.get("prompt_hash", "")) != active_hash
                        for item in getattr(agent, "safe_qd_archive", [])
                    ) or bool(getattr(agent, "probation_archive", []))
                    if has_branch:
                        missed_niche_agents.append(agent_id)
                diagnosis["niche_parent_opportunity_missed_due_to_agent_not_selected"] = missed_niche_agents
            selector_event = {
                "epoch": int(epoch_id),
                "step": int(step_id),
                "applied_specialization_strength": float(getattr(self, "specialization_strength", 0.0)),
                "weights": dict(diagnosis.get("hybrid_selector_weights", {})),
                "agents": list(diagnosis.get("hybrid_selector_diagnostics", [])),
                "fairness_slot_selected": diagnosis.get("fairness_slot_selected"),
                "fairness_slot_skipped_no_evidence": bool(diagnosis.get("fairness_slot_skipped_no_evidence", False)),
                "per_agent_optimizer_update_count": {
                    str(agent_id): int(self.per_agent_optimizer_update_count.get(f"{epoch_id}:{agent_id}", 0))
                    for agent_id in range(len(self.agents))
                },
                "niche_parent_opportunity_missed_due_to_agent_not_selected": list(
                    diagnosis.get("niche_parent_opportunity_missed_due_to_agent_not_selected", [])
                ),
            }
            self.hybrid_selector_history.append(selector_event)
            self.update_logs.append({**self._base_log_fields(), "event": "hybrid_target_selection", **selector_event})
        if not selected:
            self.clear_homogeneity_windows()
            return {
                "update_requested": True,
                "update_ready": True,
                "selected_agent_ids": [],
                "updated_agent_ids": [],
                "skipped_reason": no_selection_reason,
                "requested_optimizer_candidates": 0,
                "num_optimizer_candidates": 0,
                "num_fallback_candidates": 0,
                "num_existing_beam_candidates": 0,
                "active_prompt_changed_count": 0,
                "optimizer_underfilled": False,
            }
        updated = []
        top_metrics = []
        update_summaries = []
        for agent_id in selected:
            changed, summary = await self.update_prompt_with_beam(agent_id, diagnosis, eval_batch, step_id, epoch_id)
            update_summaries.append(summary)
            if changed:
                updated.append(agent_id)
            if isinstance(summary.get("top_metrics", {}), dict):
                top_metrics.append(summary["top_metrics"])
        self.clear_homogeneity_windows()
        self.flush_update_logs()
        self.flush_prompt_history()
        requested_optimizer_candidates = int(sum(int(s.get("requested_optimizer_candidates", 0) or 0) for s in update_summaries))
        num_optimizer_candidates = int(sum(int(s.get("num_optimizer_candidates", 0) or 0) for s in update_summaries))
        num_fallback_candidates = int(sum(int(s.get("num_fallback_candidates", 0) or 0) for s in update_summaries))
        num_existing_beam_candidates = int(sum(int(s.get("num_existing_beam_candidates", 0) or 0) for s in update_summaries))
        diagnostic_keys = [
            "optimizer_raw_response_empty",
            "optimizer_json_parse_failed",
            "optimizer_raw_candidate_count",
            "optimizer_empty_prompt_count",
            "optimizer_sanitized_count",
            "optimizer_redundant_filtered_count",
            "optimizer_schema_filtered_count",
            "optimizer_final_candidate_count",
            "student_missing_required_field_count",
        ]
        optimizer_generation_diagnostics = {
            key: int(sum(int(s.get(key, 0) or 0) for s in update_summaries))
            for key in diagnostic_keys
        }
        metadata_keys = [
            "optimizer_architecture",
            "teacher_question",
            "teacher_question_approved",
            "teacher_question_rejected",
            "teacher_question_rejection_reason",
            "teacher_question_forced_best_score",
            "teacher_question_forced_best_round",
            "teacher_question_forced_below_threshold",
            "teacher_question_score",
            "teacher_critic_rounds",
            "teacher_quality_critique",
            "teacher_specificity_critique",
            "teacher_task_alignment_critique",
            "teacher_error_alignment_critique",
            "teacher_diversity_critique",
            "teacher_rewrite_count",
            "student_candidate_count_raw",
            "student_candidate_count_final",
            "student_candidate_filtered_count",
            "student_candidate_filter_reasons",
            "student_all_candidates_filtered",
            "student_missing_required_fields",
            "student_raw_response_empty",
            "student_raw_response_preview",
            "student_json_parse_failed",
            "student_json_parse_error",
            "student_json_has_candidates_key",
            "student_candidates_is_list",
            "student_candidates_empty_list",
            "student_refusal_or_explanation",
            "student_failure_stage",
        ]
        optimizer_generation_metadata = {}
        for key in metadata_keys:
            values = [s.get(key) for s in update_summaries if isinstance(s, dict) and s.get(key) not in (None, "", [])]
            if values:
                optimizer_generation_metadata[key] = values[-1]
        return {
            "update_requested": True,
            "update_ready": True,
            "selected_agent_ids": selected,
            "updated_agent_ids": updated,
            "skipped_reason": "none",
            "requested_optimizer_candidates": requested_optimizer_candidates,
            "num_optimizer_candidates": num_optimizer_candidates,
            "num_fallback_candidates": num_fallback_candidates,
            "num_existing_beam_candidates": num_existing_beam_candidates,
            "active_prompt_changed_count": int(len(updated)),
            "optimizer_underfilled": bool(num_optimizer_candidates < requested_optimizer_candidates),
            **optimizer_generation_diagnostics,
            **optimizer_generation_metadata,
            "candidate_behavior_diagnostics": self._mean_metric_dict(top_metrics),
            "hybrid_selector_diagnostics": list(diagnosis.get("hybrid_selector_diagnostics", [])),
            "hybrid_selector_weights": dict(diagnosis.get("hybrid_selector_weights", {})),
        }

    def _apply_no_effective_evolution_tracking(
        self,
        update_summary: Dict[str, Any],
        epoch_id: int = 0,
        step_id: int = 0,
    ) -> Dict[str, Any]:
        if not isinstance(update_summary, dict):
            update_summary = {}
        if not bool(update_summary.get("update_requested", False)) or not bool(update_summary.get("update_ready", False)):
            update_summary["no_effective_evolution_counter"] = int(getattr(self, "no_effective_evolution_counter", 0) or 0)
            update_summary["no_effective_evolution_stopped"] = bool(getattr(self, "no_effective_evolution_stopped", False))
            update_summary["no_effective_evolution_reason"] = str(getattr(self, "no_effective_evolution_reason", ""))
            return update_summary

        min_optimizer_candidates = max(
            0,
            int(getattr(self.cfg, "no_effective_evolution_min_optimizer_candidates", 1) or 0),
        )
        num_optimizer_candidates = int(update_summary.get("num_optimizer_candidates", 0) or 0)
        active_prompt_changed_count = int(
            update_summary.get(
                "active_prompt_changed_count",
                len(update_summary.get("updated_agent_ids", []) or []),
            )
            or 0
        )
        ineffective = num_optimizer_candidates < min_optimizer_candidates and active_prompt_changed_count <= 0
        if ineffective:
            self.no_effective_evolution_counter = int(getattr(self, "no_effective_evolution_counter", 0) or 0) + 1
        else:
            self.no_effective_evolution_counter = 0
            self.no_effective_evolution_stopped = False
            self.no_effective_evolution_reason = ""

        enabled = bool(int(getattr(self.cfg, "no_effective_evolution_stop_enabled", True)))
        patience = max(1, int(getattr(self.cfg, "no_effective_evolution_patience", 10) or 10))
        if enabled and self.no_effective_evolution_counter >= patience:
            self.no_effective_evolution_stopped = True
            self.no_effective_evolution_reason = (
                f"num_optimizer_candidates<{min_optimizer_candidates} and no active prompt changed"
            )

        update_summary["no_effective_evolution_counter"] = int(self.no_effective_evolution_counter)
        update_summary["no_effective_evolution_stopped"] = bool(self.no_effective_evolution_stopped)
        update_summary["no_effective_evolution_reason"] = str(self.no_effective_evolution_reason)
        self.update_logs.append(
            {
                **self._base_log_fields(),
                "event": "no_effective_evolution_check",
                "epoch": epoch_id,
                "step": step_id,
                "no_effective_evolution_counter": int(self.no_effective_evolution_counter),
                "no_effective_evolution_stopped": bool(self.no_effective_evolution_stopped),
                "no_effective_evolution_reason": str(self.no_effective_evolution_reason),
                "requested_optimizer_candidates": int(update_summary.get("requested_optimizer_candidates", 0) or 0),
                "num_optimizer_candidates": int(update_summary.get("num_optimizer_candidates", 0) or 0),
                "num_fallback_candidates": int(update_summary.get("num_fallback_candidates", 0) or 0),
                "num_existing_beam_candidates": int(update_summary.get("num_existing_beam_candidates", 0) or 0),
                "active_prompt_changed_count": int(update_summary.get("active_prompt_changed_count", 0) or 0),
                "optimizer_underfilled": bool(update_summary.get("optimizer_underfilled", False)),
            }
        )
        return update_summary

    def _mean_metric_dict(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        keys = [
            "reward",
            "embedding_diversity",
            "mean_embedding_overlap",
            "target_overlap_pressure",
            "homogeneous_case_count",
            "resolved_case_count",
            "new_homogeneous_case_count",
            "team_accuracy",
            "target_agent_accuracy",
            "baseline_team_accuracy",
            "candidate_team_accuracy",
            "baseline_oracle_acc",
            "candidate_oracle_acc",
            "coverage_delta",
            "rescue_rate",
            "useful_diversity",
            "rescue_useful_diversity",
            "vote_delta",
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
                "gold": str(solved.get("gold", "")),
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
        update_summary = self._apply_no_effective_evolution_tracking(update_summary, epoch_id=epoch_id, step_id=step_id)
        record = {
            **self._base_log_fields(),
            "epoch": epoch_id,
            "step": step_id,
            "vote_correct": int(metrics.get("vote_correct", 0)),
            "vote_answer": metrics.get("vote_answer", ""),
            "plurality_vote_correct": int(metrics.get("plurality_vote_correct", metrics.get("vote_correct", 0))),
            "plurality_vote_answer": metrics.get("plurality_vote_answer", metrics.get("vote_answer", "")),
            "majority_vote_correct": int(metrics.get("majority_vote_correct", metrics.get("vote_correct", 0))),
            "majority_vote_answer": metrics.get("majority_vote_answer", metrics.get("vote_answer", "")),
            "weighted_vote_correct": int(metrics.get("weighted_vote_correct", 0)),
            "weighted_vote_answer": metrics.get("weighted_vote_answer", ""),
            "aggregation_mode": metrics.get("aggregation_mode", "majority"),
            "requested_aggregation_mode": metrics.get("requested_aggregation_mode", metrics.get("aggregation_mode", "majority")),
            "effective_aggregation_mode": metrics.get("effective_aggregation_mode", "plurality"),
            "any_correct": int(metrics.get("any_correct", 0)),
            "aggregation_gap_available": int(metrics.get("any_correct", 0)) - int(metrics.get("vote_correct", 0)),
            "useful_diversity": float(metrics.get("useful_diversity", 0.0)),
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
            "no_effective_evolution_counter": int(update_summary.get("no_effective_evolution_counter", 0) or 0),
            "no_effective_evolution_stopped": bool(update_summary.get("no_effective_evolution_stopped", False)),
            "no_effective_evolution_reason": str(update_summary.get("no_effective_evolution_reason", "")),
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

    def _stable_probe_cache_key(self, agent_id: int, prompt: str, question_hash: str) -> str:
        payload = {
            "agent_id": int(agent_id), "prompt_hash": self._normalized_prompt_hash(prompt),
            "agent_model": self.cfg.agent_model, "question_hash": question_hash,
            "task_type": self.cfg.task_type, "answer_format": getattr(self.cfg, "answer_format", ""),
            "aggregation_mode": self.cfg.aggregation_mode, "vote_tie_break": self.cfg.vote_tie_break,
            "seed": int(self.cfg.seed),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    async def _evaluate_prompt_on_stable_probe(
        self, agent_id: int, prompt: str, probe_data: List[Dict[str, str]], mechanism_steps: Sequence[Any] = (),
    ) -> Dict[str, Any]:
        async def evaluate_example(example: Dict[str, str]) -> Dict[str, Any]:
            question = example["question"]
            question_hash = self._hash(question)
            gold = self.task_spec.parse_gold(example["answer"], question)
            cache_key = self._stable_probe_cache_key(agent_id, prompt, question_hash)
            cached = self.prompt_probe_cache.get(cache_key)
            if cached is None:
                self.full_probe_missing_pair_evaluation_count = int(getattr(self, "full_probe_missing_pair_evaluation_count", 0)) + 1
                async with self.solver_call_semaphore:
                    trace, answer = await self.solve_once(question, agent_id, prompt)
                cached = {"trace": trace, "answer": answer, "gold": gold, "question_hash": question_hash}
                self.prompt_probe_cache[cache_key] = dict(cached)
                self._record_solver_rollout(
                    question_hash=question_hash,
                    prompt=prompt,
                    trace=trace,
                    answer=answer,
                    agent_id=agent_id,
                    source="stable_qd_probe",
                )
            else:
                self.full_probe_cache_hit_count = int(getattr(self, "full_probe_cache_hit_count", 0)) + 1
            answer = str(cached.get("answer", ""))
            return {
                "answer": answer,
                "correct": int(self.task_spec.match_answer(answer, gold)),
                "question_hash": question_hash,
                "gold": gold,
            }

        rows = await asyncio.gather(*[evaluate_example(example) for example in probe_data])
        answers = [row["answer"] for row in rows]
        correctness = [row["correct"] for row in rows]
        question_hashes = [row["question_hash"] for row in rows]
        gold_answers = [row["gold"] for row in rows]
        item = {"prompt": prompt, "metrics": {"mechanism_steps": list(mechanism_steps)}}
        representation = self._attach_stable_mechanism_representation(item)
        return {
            "prompt": prompt,
            "prompt_hash": self._normalized_prompt_hash(prompt),
            "answer_vector": answers,
            "correctness_vector": correctness,
            "accuracy": float(np.mean(correctness)) if correctness else 0.0,
            "question_hashes": question_hashes,
            "gold_answers": gold_answers,
            "mechanism_representation": representation,
            "prompt_static_profile": build_prompt_static_profile(answers, correctness),
        }
