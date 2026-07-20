"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class CandidateEvaluatorMixin:
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

    def _nonnegative(self, value: Any) -> float:
        try:
            x = float(value)
        except Exception:
            x = 0.0
        if np.isnan(x):
            return 0.0
        return float(max(0.0, x))

    def _reward_phase_state(self) -> Dict[str, float]:
        agents = list(getattr(self, "agents", []) or [])
        accepted_updates = sum(int(getattr(agent, "accept_count", 0) or 0) for agent in agents)
        attempted_updates = sum(
            int(getattr(agent, "accept_count", 0) or 0) + int(getattr(agent, "reject_count", 0) or 0)
            for agent in agents
        )
        prompt_hashes = [self._hash(getattr(agent, "current_prompt", "")) for agent in agents]
        unique_prompt_ratio = float(len(set(prompt_hashes)) / max(1, len(prompt_hashes)))
        update_progress = min(
            1.0,
            float(accepted_updates) / max(1, int(getattr(self.cfg, "reward_diversity_warmup_updates", 10) or 10)),
        )
        phase_progress = update_progress if self._v7_residual_protocol_enabled() else max(unique_prompt_ratio, update_progress)
        diversity_need = 1.0 - phase_progress
        return {
            "accepted_updates": float(accepted_updates),
            "attempted_updates": float(attempted_updates),
            "unique_prompt_ratio": float(unique_prompt_ratio),
            "update_progress": float(update_progress),
            "phase_progress": float(phase_progress),
            "diversity_need": float(diversity_need),
        }

    def _effective_reward_weights(self) -> Dict[str, float]:
        if str(getattr(self.cfg, "reward_schedule_mode", "static") or "static").lower() == "static":
            state = self._reward_phase_state()
            return {
                "target_accuracy": 1.0,
                "div_delta": self._nonnegative(getattr(self.cfg, "reward_weight_div_delta", 0.0)),
                "vote_delta": self._nonnegative(getattr(self.cfg, "reward_weight_vote_delta", 0.0)),
                "vote_margin": self._nonnegative(getattr(self.cfg, "reward_weight_vote_margin", 0.0)),
                "boundary_diversity": self._nonnegative(getattr(self.cfg, "reward_weight_boundary_diversity", 0.0)),
                "coverage": self._nonnegative(getattr(self.cfg, "reward_weight_coverage", 0.3)),
                "useful_diversity": self._nonnegative(getattr(self.cfg, "reward_weight_useful_diversity", 0.2)),
                "invalid_delta": self._nonnegative(getattr(self.cfg, "reward_weight_invalid_delta", 0.0)),
                "accuracy_guard_epsilon": self._nonnegative(getattr(self.cfg, "accuracy_guard_epsilon", 0.0)),
                **state,
            }

        state = self._reward_phase_state()
        need = float(state["diversity_need"])
        progress = float(state["phase_progress"])
        target_weight = (
            float(getattr(self.cfg, "reward_weight_target_accuracy_late", 1.0)) * progress
            + float(getattr(self.cfg, "reward_weight_target_accuracy_early", 0.9)) * need
        )
        div_weight = (
            float(getattr(self.cfg, "reward_weight_div_delta_late", 0.2)) * progress
            + float(getattr(self.cfg, "reward_weight_div_delta_early", 0.8)) * need
        )
        vote_delta_weight = (
            float(getattr(self.cfg, "reward_weight_vote_delta_late", 0.3)) * progress
            + float(getattr(self.cfg, "reward_weight_vote_delta_early", 0.4)) * need
        )
        vote_margin_weight = (
            float(getattr(self.cfg, "reward_weight_vote_margin_late", 0.25)) * progress
            + float(getattr(self.cfg, "reward_weight_vote_margin_early", 0.5)) * need
        )
        boundary_diversity_weight = (
            float(getattr(self.cfg, "reward_weight_boundary_diversity_late", 0.2)) * progress
            + float(getattr(self.cfg, "reward_weight_boundary_diversity_early", 0.3)) * need
        )
        guard_epsilon = (
            float(getattr(self.cfg, "accuracy_guard_epsilon_late", 0.01)) * progress
            + float(getattr(self.cfg, "accuracy_guard_epsilon_early", 0.03)) * need
        )
        coverage_weight = float(getattr(self.cfg, "reward_weight_coverage_late", 0.3)) * progress + float(getattr(self.cfg, "reward_weight_coverage_early", 0.4)) * need
        useful_weight = float(getattr(self.cfg, "reward_weight_useful_diversity_late", 0.25)) * progress + float(getattr(self.cfg, "reward_weight_useful_diversity_early", 0.5)) * need
        return {
            "target_accuracy": self._nonnegative(target_weight),
            "div_delta": self._nonnegative(div_weight),
            "vote_delta": self._nonnegative(vote_delta_weight),
            "vote_margin": self._nonnegative(vote_margin_weight),
            "boundary_diversity": self._nonnegative(boundary_diversity_weight),
            "coverage": self._nonnegative(coverage_weight),
            "useful_diversity": self._nonnegative(useful_weight),
            "invalid_delta": self._nonnegative(getattr(self.cfg, "reward_weight_invalid_delta", 0.0)),
            "accuracy_guard_epsilon": self._nonnegative(guard_epsilon),
            **state,
        }

    def _effective_reward_log_fields(self, weights: Dict[str, float]) -> Dict[str, float]:
        return {
            "effective_weight_target_accuracy": float(weights.get("target_accuracy", 0.0)),
            "effective_weight_div_delta": float(weights.get("div_delta", 0.0)),
            "effective_weight_vote_delta": float(weights.get("vote_delta", 0.0)),
            "effective_weight_vote_margin": float(weights.get("vote_margin", 0.0)),
            "effective_weight_boundary_diversity": float(weights.get("boundary_diversity", 0.0)),
            "effective_weight_coverage": float(weights.get("coverage", 0.0)),
            "effective_weight_useful_diversity": float(weights.get("useful_diversity", 0.0)),
            "effective_accuracy_guard_epsilon": float(weights.get("accuracy_guard_epsilon", 0.0)),
            "reward_phase_progress": float(weights.get("phase_progress", 0.0)),
            "reward_diversity_need": float(weights.get("diversity_need", 0.0)),
            "reward_unique_prompt_ratio": float(weights.get("unique_prompt_ratio", 0.0)),
            "reward_accepted_updates": float(weights.get("accepted_updates", 0.0)),
        }

    def _candidate_eval_audit_fields(self, eval_batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        question_hashes = {
            hashlib.sha256(normalize_spaces(str(example.get("question", ""))).lower().encode("utf-8")).hexdigest()
            for example in eval_batch
            if isinstance(example, Mapping)
        }
        return {
            "candidate_eval_data_source": str(getattr(self.cfg, "candidate_eval_data_source", "optimization_train")),
            "candidate_eval_total_count": len(eval_batch),
            "candidate_eval_unique_question_count": len(question_hashes),
            "candidate_eval_repeat_count": int(getattr(self.cfg, "candidate_eval_repeats", 1) or 1),
        }

    def _candidate_reward_guarded(
        self,
        baseline_team_accuracy: float,
        candidate_team_accuracy: float,
        baseline_target_accuracy: float,
        candidate_target_accuracy: float,
        baseline_embedding_diversity: float,
        candidate_embedding_diversity: float,
        baseline_invalid_rate: float,
        candidate_invalid_rate: float,
    ) -> Dict[str, Any]:
        baseline_team_accuracy = self._clip01(baseline_team_accuracy)
        candidate_team_accuracy = self._clip01(candidate_team_accuracy)
        baseline_target_accuracy = self._clip01(baseline_target_accuracy)
        candidate_target_accuracy = self._clip01(candidate_target_accuracy)
        baseline_embedding_diversity = self._clip01(baseline_embedding_diversity)
        candidate_embedding_diversity = self._clip01(candidate_embedding_diversity)
        baseline_invalid_rate = self._clip01(baseline_invalid_rate)
        candidate_invalid_rate = self._clip01(candidate_invalid_rate)

        deltas = compute_candidate_metric_deltas(
            baseline_target_accuracy=baseline_target_accuracy,
            candidate_target_accuracy=candidate_target_accuracy,
            baseline_team_accuracy=baseline_team_accuracy,
            candidate_team_accuracy=candidate_team_accuracy,
            baseline_oracle_accuracy=0.0,
            candidate_oracle_accuracy=0.0,
            baseline_embedding_diversity=baseline_embedding_diversity,
            candidate_embedding_diversity=candidate_embedding_diversity,
            baseline_invalid_rate=baseline_invalid_rate,
            candidate_invalid_rate=candidate_invalid_rate,
        )
        acc_delta = deltas["accuracy_delta"]
        vote_delta = deltas["vote_delta"]
        div_delta = deltas["diversity_delta"]
        invalid_delta = deltas["invalid_delta"]
        weights = self._effective_reward_weights()
        guard_passed = candidate_target_accuracy >= baseline_target_accuracy - float(weights["accuracy_guard_epsilon"])
        if not guard_passed:
            reward = -1.0 + acc_delta - float(weights["invalid_delta"]) * max(0.0, invalid_delta)
        else:
            reward = (
                float(weights["target_accuracy"]) * candidate_target_accuracy
                + float(weights["div_delta"]) * div_delta
                - float(weights["invalid_delta"]) * max(0.0, invalid_delta)
            )
        result = {
            "reward": float(reward),
            "reward_total": float(reward),
            **deltas,
            "accuracy_guard_passed": bool(guard_passed),
            "baseline_target_accuracy": float(baseline_target_accuracy),
            "candidate_target_accuracy": float(candidate_target_accuracy),
            "target_agent_accuracy": float(candidate_target_accuracy),
        }
        result.update(self._effective_reward_log_fields(weights))
        result.update(
            {
                "baseline_team_accuracy": float(baseline_team_accuracy),
                "candidate_team_accuracy": float(candidate_team_accuracy),
                "baseline_invalid_rate": float(baseline_invalid_rate),
                "candidate_invalid_rate": float(candidate_invalid_rate),
                "baseline_embedding_diversity": float(baseline_embedding_diversity),
                "candidate_embedding_diversity": float(candidate_embedding_diversity),
            }
        )
        return result

    def _candidate_reward_vote_useful_diversity(
        self,
        *,
        baseline_team_accuracy: float,
        candidate_team_accuracy: float,
        baseline_target_accuracy: float,
        candidate_target_accuracy: float,
        baseline_invalid_rate: float,
        candidate_invalid_rate: float,
        baseline_mean_vote_margin: float,
        candidate_mean_vote_margin: float,
        baseline_boundary_useful_diversity: float,
        candidate_boundary_useful_diversity: float,
        baseline_oracle_accuracy: Optional[float] = None,
        candidate_oracle_accuracy: Optional[float] = None,
        baseline_embedding_diversity: float = 0.0,
        candidate_embedding_diversity: float = 0.0,
    ) -> Dict[str, Any]:
        baseline_team_accuracy = self._clip01(baseline_team_accuracy)
        candidate_team_accuracy = self._clip01(candidate_team_accuracy)
        baseline_target_accuracy = self._clip01(baseline_target_accuracy)
        candidate_target_accuracy = self._clip01(candidate_target_accuracy)
        baseline_invalid_rate = self._clip01(baseline_invalid_rate)
        candidate_invalid_rate = self._clip01(candidate_invalid_rate)
        baseline_mean_vote_margin = float(np.clip(baseline_mean_vote_margin, -1.0, 1.0))
        candidate_mean_vote_margin = float(np.clip(candidate_mean_vote_margin, -1.0, 1.0))
        baseline_boundary_useful_diversity = self._clip01(baseline_boundary_useful_diversity)
        candidate_boundary_useful_diversity = self._clip01(candidate_boundary_useful_diversity)

        deltas = compute_candidate_metric_deltas(
            baseline_target_accuracy=baseline_target_accuracy,
            candidate_target_accuracy=candidate_target_accuracy,
            baseline_team_accuracy=baseline_team_accuracy,
            candidate_team_accuracy=candidate_team_accuracy,
            baseline_oracle_accuracy=float(baseline_oracle_accuracy or 0.0),
            candidate_oracle_accuracy=float(candidate_oracle_accuracy or 0.0),
            baseline_embedding_diversity=baseline_embedding_diversity,
            candidate_embedding_diversity=candidate_embedding_diversity,
            baseline_invalid_rate=baseline_invalid_rate,
            candidate_invalid_rate=candidate_invalid_rate,
        )
        vote_delta = deltas["vote_delta"]
        invalid_delta = deltas["invalid_delta"]
        vote_margin_delta = candidate_mean_vote_margin - baseline_mean_vote_margin
        boundary_diversity_delta = candidate_boundary_useful_diversity - baseline_boundary_useful_diversity
        # Boundary diversity is an auxiliary signal only while the team remains
        # near a gold-vs-wrong vote boundary. Leaving that boundary through a
        # stronger correct vote must not turn into a diversity penalty.
        boundary_diversity_gain = max(0.0, boundary_diversity_delta)
        weights = self._effective_reward_weights()
        target_guard_passed = candidate_target_accuracy >= baseline_target_accuracy - float(weights["accuracy_guard_epsilon"])
        invalid_guard_passed = candidate_invalid_rate <= baseline_invalid_rate + float(self.cfg.invalid_guard_epsilon)
        reward_components = {
            "reward_component_target_accuracy": 0.0,
            "reward_component_vote_delta": 0.0,
            "reward_component_vote_margin": 0.0,
            "reward_component_boundary_diversity": 0.0,
            "reward_component_invalid_penalty": 0.0,
            "reward_component_guard_penalty": 0.0,
        }
        if not target_guard_passed or not invalid_guard_passed:
            reward = -1.0
            reward_components["reward_component_guard_penalty"] = -1.0
        else:
            reward_components.update(
                {
                    "reward_component_target_accuracy": float(weights["target_accuracy"]) * candidate_target_accuracy,
                    "reward_component_vote_delta": float(weights["vote_delta"]) * vote_delta,
                    "reward_component_vote_margin": float(weights["vote_margin"]) * vote_margin_delta,
                    "reward_component_boundary_diversity": float(weights["boundary_diversity"]) * boundary_diversity_gain,
                    "reward_component_invalid_penalty": -float(weights["invalid_delta"]) * max(0.0, invalid_delta),
                }
            )
            reward = sum(reward_components.values())
        result = {
            "reward": float(reward),
            "reward_total": float(reward),
            "coverage_delta": float(deltas["coverage_delta"]),
            **deltas,
            "baseline_mean_vote_margin": baseline_mean_vote_margin,
            "candidate_mean_vote_margin": candidate_mean_vote_margin,
            "vote_margin_delta": vote_margin_delta,
            "baseline_boundary_useful_diversity": baseline_boundary_useful_diversity,
            "candidate_boundary_useful_diversity": candidate_boundary_useful_diversity,
            "boundary_useful_diversity_delta": boundary_diversity_delta,
            "boundary_diversity_gain": boundary_diversity_gain,
            "baseline_team_accuracy": float(baseline_team_accuracy),
            "candidate_team_accuracy": float(candidate_team_accuracy),
            "baseline_target_accuracy": float(baseline_target_accuracy),
            "candidate_target_accuracy": float(candidate_target_accuracy),
            "target_agent_accuracy": float(candidate_target_accuracy),
            "baseline_invalid_rate": float(baseline_invalid_rate),
            "candidate_invalid_rate": float(candidate_invalid_rate),
            "accuracy_guard_passed": bool(target_guard_passed),
            "invalid_guard_passed": bool(invalid_guard_passed),
            **reward_components,
        }
        result.update(self._effective_reward_log_fields(weights))
        return result

    def _candidate_reward_competence_depth(self, metrics: Dict[str, Any], v7_reward: float) -> Dict[str, Any]:
        accuracy_delta = float(metrics.get("accuracy_delta", 0.0) or 0.0)
        competence_component = (
            float(getattr(self.cfg, "competence_weight_accuracy_gain", 1.0)) * max(0.0, accuracy_delta)
            - float(getattr(self.cfg, "competence_weight_accuracy_loss", 1.5)) * max(0.0, -accuracy_delta)
            + float(getattr(self.cfg, "competence_weight_depth2_gain", 0.8)) * float(metrics.get("depth2_gain_rate", 0.0) or 0.0)
            - float(getattr(self.cfg, "competence_weight_depth2_loss", 1.0)) * float(metrics.get("depth2_loss_rate", 0.0) or 0.0)
            + float(getattr(self.cfg, "competence_weight_vote_gain_early", 0.4)) * float(metrics.get("vote_gain_rate", 0.0) or 0.0)
            - float(getattr(self.cfg, "competence_weight_vote_loss_early", 1.0)) * float(metrics.get("vote_loss_rate", 0.0) or 0.0)
        )
        if self._is_v82_hybrid():
            competence_component += (
                float(getattr(self.cfg, "competence_weight_depth1_gain", 0.8)) * float(metrics.get("depth1_gain_rate", 0.0) or 0.0)
                - float(getattr(self.cfg, "competence_weight_depth1_loss", 1.2)) * float(metrics.get("depth1_loss_rate", 0.0) or 0.0)
            )
        strength = float(self.specialization_strength)
        competence_mix = max(float(getattr(self.cfg, "competence_residual_floor", 0.30)), 1.0 - strength) if self._is_v82_hybrid() else 1.0 - strength
        specialization_mix = 1.0 - competence_mix if self._is_v82_hybrid() else strength
        reward = competence_mix * competence_component + specialization_mix * float(v7_reward)
        depth2_component = float(metrics.get("depth2_net_delta", 0.0) or 0.0) if bool(
            getattr(self.cfg, "competence_depth2_aux_enabled", False)
        ) else 0.0
        boundary_component = float(metrics.get(
            "plurality_boundary_shared_error_net_gain",
            metrics.get("boundary_shared_error_net_gain", 0.0),
        ) or 0.0)
        return {
            "reward": float(reward),
            "reward_total": float(reward),
            "final_reward": float(reward),
            "competence_reward_component": float(competence_component),
            "v7_reward_component": float(v7_reward),
            "effective_reward_specialization_strength": strength,
            "competence_mix": competence_mix,
            "specialization_mix": specialization_mix,
            "stage_aux_depth2_component": competence_mix * depth2_component,
            "stage_aux_boundary_component": specialization_mix * boundary_component,
            "stage_aux_objective": competence_mix * (
                0.5 * float(metrics.get("depth1_net_delta", 0.0) or 0.0) + 0.5 * depth2_component
            ) + specialization_mix * boundary_component if self._is_v82_hybrid() else (1.0 - strength) * depth2_component + strength * boundary_component,
        }

    def _candidate_reward_coverage_useful_diversity(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        weights = self._effective_reward_weights()
        invalid_passed = float(metrics.get("candidate_invalid_rate", 1.0)) <= float(metrics.get("baseline_invalid_rate", 1.0)) + float(self.cfg.invalid_guard_epsilon)
        reward = -1.0 if not invalid_passed else (
            float(weights["target_accuracy"]) * float(metrics.get("candidate_target_accuracy", 0.0))
            + float(weights["coverage"]) * float(metrics.get("coverage_delta", 0.0))
            + float(weights["useful_diversity"]) * float(metrics.get("useful_diversity", 0.0))
        )
        return {"reward": reward, "reward_total": reward, "invalid_guard_passed": invalid_passed, **self._effective_reward_log_fields(weights)}

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
            baseline_prompts = list(peer_prompts)
            while len(baseline_prompts) < len(self.agents):
                baseline_prompts.append(self.agents[len(baseline_prompts)].current_prompt)
            eval_prompts = list(baseline_prompts)
            eval_prompts[agent_id] = candidate_prompt
            question_hash = self._hash(q)
            baseline_traces, baseline_answers, baseline_reuse_stats = await self.solve_with_prompts_reusing_records(
                q,
                baseline_prompts,
                source=f"candidate_accuracy_baseline_agent_{agent_id}",
            )
            baseline_rollout = self.compute_rollout_metrics(
                baseline_traces,
                baseline_answers,
                gold,
                prompts=baseline_prompts,
                question_hash=question_hash,
            )
            traces, answers, reuse_stats = await self.solve_with_prompts_reusing_records(
                q,
                eval_prompts,
                source=f"candidate_accuracy_agent_{agent_id}",
            )
            rollout = self.compute_rollout_metrics(
                traces,
                answers,
                gold,
                prompts=eval_prompts,
                question_hash=question_hash,
            )
            baseline_target_answer = baseline_answers[agent_id] if agent_id < len(baseline_answers) else ""
            target_answer = answers[agent_id] if agent_id < len(answers) else ""
            return {
                "baseline_team_accuracy": int(baseline_rollout.get("vote_correct", 0)),
                "team_accuracy": int(rollout.get("vote_correct", 0)),
                "baseline_target_accuracy": int(self.task_spec.match_answer(baseline_target_answer, gold)),
                "target_agent_accuracy": int(self.task_spec.match_answer(target_answer, gold)),
                "baseline_any_correct": int(baseline_rollout.get("any_correct", 0)),
                "candidate_any_correct": int(rollout.get("any_correct", 0)),
                "baseline_individual_correct": list(baseline_rollout.get("individual_correct", [])),
                "candidate_individual_correct": list(rollout.get("individual_correct", [])),
                "baseline_mean_vote_margin": float(baseline_rollout.get("normalized_vote_margin", -1.0)),
                "candidate_mean_vote_margin": float(rollout.get("normalized_vote_margin", -1.0)),
                "baseline_boundary_useful_diversity": float(baseline_rollout.get("boundary_useful_diversity", 0.0)),
                "candidate_boundary_useful_diversity": float(rollout.get("boundary_useful_diversity", 0.0)),
                "vote_answer": str(rollout.get("vote_answer", "")),
                "vote_tie": bool(rollout.get("vote_tie", False)),
                "tie_candidates": list(rollout.get("tie_candidates", [])),
                "vote_counts": dict(rollout.get("vote_counts", {})),
                "tie_break_method": str(rollout.get("tie_break_method", "")),
                "majority_vote_answer": str(rollout.get("majority_vote_answer", "")),
                "weighted_vote_answer": str(rollout.get("weighted_vote_answer", "")),
                "majority_vote_correct": int(rollout.get("majority_vote_correct", 0)),
                "weighted_vote_correct": int(rollout.get("weighted_vote_correct", 0)),
                "aggregation_mode": str(rollout.get("aggregation_mode", "majority")),
                "target_answer": target_answer,
                "target_trace_hash": self._hash(traces[agent_id]) if agent_id < len(traces) else "",
                "baseline_solver_reuse_hits": int(baseline_reuse_stats.get("solver_reuse_hits", 0) or 0),
                "baseline_solver_reuse_misses": int(baseline_reuse_stats.get("solver_reuse_misses", 0) or 0),
                "baseline_solver_calls": int(baseline_reuse_stats.get("solver_calls", 0) or 0),
                "baseline_solver_reuse_total": int(baseline_reuse_stats.get("solver_reuse_total", 0) or 0),
                **reuse_stats,
            }

        raw = await asyncio.gather(*[run_one(ex) for ex in eval_batch], return_exceptions=True)
        rows = [r for r in raw if isinstance(r, dict)]
        errors = [normalize_spaces(str(r))[:240] for r in raw if isinstance(r, Exception)]
        baseline_team_accuracy = self._clip01(float(np.mean([float(r.get("baseline_team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        team_accuracy = self._clip01(float(np.mean([float(r.get("team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        baseline_target_accuracy = self._clip01(float(np.mean([float(r.get("baseline_target_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        target_agent_accuracy = self._clip01(float(np.mean([float(r.get("target_agent_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        baseline_oracle_acc = self._clip01(float(np.mean([float(r.get("baseline_any_correct", 0.0)) for r in rows])) if rows else 0.0)
        candidate_oracle_acc = self._clip01(float(np.mean([float(r.get("candidate_any_correct", 0.0)) for r in rows])) if rows else 0.0)
        baseline_mean_vote_margin = float(np.mean([float(r.get("baseline_mean_vote_margin", -1.0)) for r in rows])) if rows else -1.0
        candidate_mean_vote_margin = float(np.mean([float(r.get("candidate_mean_vote_margin", -1.0)) for r in rows])) if rows else -1.0
        baseline_boundary = self._clip01(float(np.mean([float(r.get("baseline_boundary_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
        candidate_boundary = self._clip01(float(np.mean([float(r.get("candidate_boundary_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
        vote_transitions = compute_vote_transitions(
            [bool(row.get("baseline_team_accuracy", 0)) for row in rows],
            [bool(row.get("team_accuracy", 0)) for row in rows],
        )
        coverage_transitions = compute_oracle_coverage_transitions(
            [list(row.get("baseline_individual_correct", [])) for row in rows],
            [list(row.get("candidate_individual_correct", [])) for row in rows],
        )
        deltas = compute_candidate_metric_deltas(
            baseline_target_accuracy=baseline_target_accuracy,
            candidate_target_accuracy=target_agent_accuracy,
            baseline_team_accuracy=baseline_team_accuracy,
            candidate_team_accuracy=team_accuracy,
            baseline_oracle_accuracy=baseline_oracle_acc,
            candidate_oracle_accuracy=candidate_oracle_acc,
            baseline_embedding_diversity=0.0,
            candidate_embedding_diversity=0.0,
            baseline_invalid_rate=0.0,
            candidate_invalid_rate=0.0,
        )
        if abs(float(vote_transitions["net_vote_delta"]) - float(deltas["vote_delta"])) > PARETO_EPSILON:
            raise RuntimeError("Accuracy-only vote transition delta is inconsistent")
        if abs(float(coverage_transitions["net_coverage_delta"]) - float(deltas["coverage_delta"])) > PARETO_EPSILON:
            raise RuntimeError("Accuracy-only coverage transition delta is inconsistent")
        boundary_delta = candidate_boundary - baseline_boundary
        solver_reuse_hits = int(sum(int(r.get("solver_reuse_hits", 0) or 0) for r in rows))
        solver_reuse_misses = int(sum(int(r.get("solver_reuse_misses", 0) or 0) for r in rows))
        solver_calls = int(sum(int(r.get("solver_calls", 0) or 0) for r in rows))
        solver_reuse_total = int(sum(int(r.get("solver_reuse_total", 0) or 0) for r in rows))
        majority_team_accuracy = self._clip01(float(np.mean([float(r.get("majority_vote_correct", 0.0)) for r in rows])) if rows else 0.0)
        weighted_team_accuracy = self._clip01(float(np.mean([float(r.get("weighted_vote_correct", 0.0)) for r in rows])) if rows else 0.0)
        return {
            "reward": target_agent_accuracy,
            "reward_total": target_agent_accuracy,
            "reward_component_target_accuracy": target_agent_accuracy,
            "reward_component_vote_delta": 0.0,
            "reward_component_vote_margin": 0.0,
            "reward_component_boundary_diversity": 0.0,
            "reward_component_invalid_penalty": 0.0,
            "reward_component_guard_penalty": 0.0,
            "embedding_diversity": 0.0,
            "mean_embedding_overlap": 0.0,
            "target_overlap_pressure": 0.0,
            "homogeneous_case_count": 0.0,
            "resolved_case_count": 0.0,
            "new_homogeneous_case_count": 0.0,
            "team_accuracy": team_accuracy,
            "baseline_team_accuracy": baseline_team_accuracy,
            "candidate_team_accuracy": team_accuracy,
            "majority_team_accuracy": majority_team_accuracy,
            "weighted_team_accuracy": weighted_team_accuracy,
            "aggregation_mode": str(getattr(self.cfg, "aggregation_mode", "majority") or "majority"),
            "target_agent_accuracy": target_agent_accuracy,
            "baseline_target_accuracy": baseline_target_accuracy,
            "candidate_target_accuracy": target_agent_accuracy,
            "baseline_oracle_acc": baseline_oracle_acc,
            "candidate_oracle_acc": candidate_oracle_acc,
            "baseline_mean_vote_margin": baseline_mean_vote_margin,
            "candidate_mean_vote_margin": candidate_mean_vote_margin,
            "vote_margin_delta": candidate_mean_vote_margin - baseline_mean_vote_margin,
            "baseline_boundary_useful_diversity": baseline_boundary,
            "candidate_boundary_useful_diversity": candidate_boundary,
            "boundary_useful_diversity_delta": boundary_delta,
            "boundary_diversity_gain": max(0.0, boundary_delta),
            "baseline_embedding_diversity": 0.0,
            "candidate_embedding_diversity": 0.0,
            "baseline_invalid_rate": 0.0,
            "candidate_invalid_rate": 0.0,
            "accuracy_guard_passed": True,
            "invalid_guard_passed": True,
            **deltas,
            **vote_transitions,
            **coverage_transitions,
            "invalid_rate": 0.0,
            "invalid_score": 1.0,
            "num_eval_samples": len(rows),
            "candidate_prompt": candidate_prompt,
            "errors": errors,
            "accuracy_only": True,
            "accuracy_only_reward_basis": "target_agent_accuracy",
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
            **self._candidate_eval_audit_fields(eval_batch),
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
            if self._uses_baseline_candidate_metrics():
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
            if self._uses_baseline_candidate_metrics():
                baseline_vote_correct = int(baseline_rollout.get("vote_correct", 0))
                candidate_vote_correct = int(rollout.get("vote_correct", 0))
                baseline_any_correct = self._rollout_any_correct(baseline_rollout)
                candidate_any_correct = self._rollout_any_correct(rollout)
                baseline_target_correct = self._safe_agent_correct(baseline_rollout, agent_id)
                target_agent_correct = self._safe_agent_correct(rollout, agent_id)
                target_trace_novelty = self._target_trace_novelty(traces, agent_id)
                target_useful_diversity = (
                    target_trace_novelty
                    * float(target_agent_correct)
                    * (1.0 - float(agent_invalid.get("invalid", 1)))
                )
                rescue = int((baseline_vote_correct == 0) and (target_agent_correct == 1))
                peer_wrong_count = sum(
                    int(not self.task_spec.match_answer(answer, gold))
                    for idx, answer in enumerate(baseline_answers)
                    if idx != agent_id
                )
                counterfactual_answers = list(baseline_answers)
                if agent_id < len(counterfactual_answers):
                    counterfactual_answers[agent_id] = gold
                counterfactual_vote = self._vote_with_diagnostics(
                    counterfactual_answers, question_hash=sample_hash
                )
                counterfactual_gold_vote_correct = bool(
                    self.task_spec.match_answer(str(counterfactual_vote.get("vote_answer", "")), gold)
                )
                counterfactual_gold_diagnostics = compute_gold_vote_diagnostics(
                    counterfactual_answers,
                    gold,
                    self.task_spec.match_answer,
                    len(self.agents),
                )

                def in_dominant_wrong_cluster(candidate_answers: Sequence[str], target_id: int) -> bool:
                    target_value = str(candidate_answers[target_id] if target_id < len(candidate_answers) else "").strip()
                    wrong_counts = Counter(
                        str(answer or "").strip()
                        for answer in candidate_answers
                        if str(answer or "").strip() and not self.task_spec.match_answer(str(answer), gold)
                    )
                    return bool(
                        target_value
                        and not self.task_spec.match_answer(target_value, gold)
                        and wrong_counts.get(target_value, 0) == max(wrong_counts.values(), default=0)
                        and wrong_counts.get(target_value, 0) > 1
                    )

                residual_info = (
                    {}
                    if self._is_rollout_qd_method()
                    else self._infer_target_error_pattern(
                        target_trace=baseline_traces[agent_id] if agent_id < len(baseline_traces) else "",
                        target_answer=baseline_answers[agent_id] if agent_id < len(baseline_answers) else "",
                        peer_traces=[trace for idx, trace in enumerate(baseline_traces) if idx != agent_id],
                        rollout=baseline_rollout,
                        agent_id=agent_id,
                    )
                )
                row.update(
                    {
                        "question_hash": sample_hash,
                        "baseline_vote_correct": baseline_vote_correct,
                        "candidate_vote_correct": candidate_vote_correct,
                        "baseline_plurality_vote_correct": baseline_vote_correct,
                        "candidate_plurality_vote_correct": candidate_vote_correct,
                        "baseline_vote_tie": bool(baseline_rollout.get("vote_tie", False)),
                        "candidate_vote_tie": bool(rollout.get("vote_tie", False)),
                        "baseline_plurality_vote_tie": bool(baseline_rollout.get("plurality_vote_tie", False)),
                        "candidate_plurality_vote_tie": bool(rollout.get("plurality_vote_tie", False)),
                        "baseline_plurality_tie_candidates": list(baseline_rollout.get("plurality_tie_candidates", [])),
                        "candidate_plurality_tie_candidates": list(rollout.get("plurality_tie_candidates", [])),
                        "plurality_tie_break_method": str(baseline_rollout.get("plurality_tie_break_method", "")),
                        "plurality_tie_break_question_hash": sample_hash,
                        "baseline_any_correct": baseline_any_correct,
                        "candidate_any_correct": candidate_any_correct,
                        "baseline_individual_correct": [bool(value) for value in baseline_rollout.get("individual_correct", [])],
                        "candidate_individual_correct": [bool(value) for value in rollout.get("individual_correct", [])],
                        "baseline_answers": list(baseline_answers),
                        "candidate_answers": list(answers),
                        "baseline_traces": list(baseline_traces),
                        "candidate_traces": list(traces),
                        "baseline_invalid_flags": [int(value) for value in baseline_rollout.get("invalid_flags", [])],
                        "candidate_invalid_flags": [int(value) for value in rollout.get("invalid_flags", [])],
                        "baseline_target_correct": baseline_target_correct,
                        "candidate_target_correct": target_agent_correct,
                        "target_agent_correct": target_agent_correct,
                        "peer_wrong_count": int(peer_wrong_count),
                        "baseline_vote_margin": float(baseline_rollout.get("normalized_vote_margin", -1.0)),
                        "candidate_vote_margin": float(rollout.get("normalized_vote_margin", -1.0)),
                        "baseline_gold_vote_count": int(baseline_rollout.get("gold_vote_count", 0) or 0),
                        "candidate_gold_vote_count": int(rollout.get("gold_vote_count", 0) or 0),
                        "baseline_largest_wrong_vote_count": int(baseline_rollout.get("largest_wrong_vote_count", 0) or 0),
                        "candidate_largest_wrong_vote_count": int(rollout.get("largest_wrong_vote_count", 0) or 0),
                        "baseline_plurality_margin_votes": int(baseline_rollout.get("plurality_margin_votes", 0) or 0),
                        "candidate_plurality_margin_votes": int(rollout.get("plurality_margin_votes", 0) or 0),
                        "baseline_normalized_plurality_margin": float(baseline_rollout.get("normalized_plurality_margin", -1.0)),
                        "candidate_normalized_plurality_margin": float(rollout.get("normalized_plurality_margin", -1.0)),
                        "counterfactual_gold_vote_correct": counterfactual_gold_vote_correct,
                        "plurality_pivotal_fix_opportunity": bool(
                            not baseline_vote_correct and counterfactual_gold_vote_correct
                        ),
                        "plurality_pivotal_fix": bool(
                            not baseline_vote_correct and counterfactual_gold_vote_correct and candidate_vote_correct
                        ),
                        "plurality_pivotal_loss": bool(baseline_vote_correct and not candidate_vote_correct),
                        "counterfactual_gold_margin": float(counterfactual_gold_diagnostics.get("normalized_vote_margin", -1.0)),
                        "baseline_target_in_dominant_wrong_cluster": in_dominant_wrong_cluster(baseline_answers, agent_id),
                        "candidate_target_in_dominant_wrong_cluster": in_dominant_wrong_cluster(answers, agent_id),
                        **({} if self._is_rollout_qd_method() else {
                            "capability_residual_family": str(residual_info.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                            "capability_residual_confidence": float(residual_info.get("confidence", 0.0) or 0.0),
                        }),
                        "target_answer": answers[agent_id] if agent_id < len(answers) else "",
                        "target_trace_novelty": target_trace_novelty,
                        "target_useful_diversity": target_useful_diversity,
                        "rescue": rescue,
                        "rescue_useful_diversity": target_useful_diversity * float(rescue),
                        "baseline_team_accuracy": float(baseline_vote_correct),
                        "baseline_mean_vote_margin": float(baseline_rollout.get("normalized_vote_margin", -1.0)),
                        "baseline_boundary_useful_diversity": float(baseline_rollout.get("boundary_useful_diversity", 0.0)),
                        "baseline_embedding_diversity": float(baseline_rollout.get("embedding_diversity", 0.0)),
                        "baseline_invalid_rate": float(baseline_rollout.get("invalid_rate", 1.0)),
                        "baseline_mean_embedding_overlap": float(baseline_rollout.get("mean_embedding_overlap", 0.0)),
                        "candidate_team_accuracy": float(candidate_vote_correct),
                        "candidate_mean_vote_margin": float(rollout.get("normalized_vote_margin", -1.0)),
                        "candidate_boundary_useful_diversity": float(rollout.get("boundary_useful_diversity", 0.0)),
                        "candidate_embedding_diversity": float(rollout.get("embedding_diversity", 0.0)),
                        "candidate_invalid_rate": float(rollout.get("invalid_rate", 1.0)),
                        "candidate_mean_embedding_overlap": float(rollout.get("mean_embedding_overlap", 0.0)),
                        "baseline_solver_reuse_hits": int(baseline_reuse_stats.get("solver_reuse_hits", 0) or 0),
                        "baseline_solver_reuse_misses": int(baseline_reuse_stats.get("solver_reuse_misses", 0) or 0),
                        "baseline_solver_calls": int(baseline_reuse_stats.get("solver_calls", 0) or 0),
                        "baseline_solver_reuse_total": int(baseline_reuse_stats.get("solver_reuse_total", 0) or 0),
                    }
                )
                if self._v7_residual_protocol_enabled():
                    row["behavior_context"] = self._behavior_context_for_baseline(
                        agent_id=agent_id,
                        answers=baseline_answers,
                        gold=gold,
                        rollout=baseline_rollout,
                        question_hash=sample_hash,
                    )
            return row

        raw = await asyncio.gather(*[run_one(ex) for ex in eval_batch], return_exceptions=True)
        rows = [r for r in raw if isinstance(r, dict)]
        errors = [normalize_spaces(str(r))[:240] for r in raw if isinstance(r, Exception)]
        diversity = self._clip01(float(np.mean([float(r.get("embedding_diversity", 0.0)) for r in rows])) if rows else 0.0)
        team_accuracy = self._clip01(float(np.mean([float(r.get("team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        invalid_rate = self._clip01(float(np.mean([float(r.get("invalid", 1.0)) for r in rows])) if rows else 1.0)
        invalid_score = self._clip01(1.0 - invalid_rate)
        baseline_candidate_metrics: Dict[str, Any] = {}
        if self._uses_baseline_candidate_metrics():
            baseline_team_accuracy = self._clip01(float(np.mean([float(r.get("baseline_team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
            candidate_team_accuracy = self._clip01(float(np.mean([float(r.get("candidate_team_accuracy", 0.0)) for r in rows])) if rows else team_accuracy)
            baseline_embedding_diversity = self._clip01(float(np.mean([float(r.get("baseline_embedding_diversity", 0.0)) for r in rows])) if rows else 0.0)
            candidate_embedding_diversity = self._clip01(float(np.mean([float(r.get("candidate_embedding_diversity", 0.0)) for r in rows])) if rows else diversity)
            baseline_invalid_rate = self._clip01(float(np.mean([float(r.get("baseline_invalid_rate", 1.0)) for r in rows])) if rows else 1.0)
            candidate_invalid_rate = self._clip01(float(np.mean([float(r.get("candidate_invalid_rate", 1.0)) for r in rows])) if rows else invalid_rate)
            baseline_target_accuracy = self._clip01(float(np.mean([float(r.get("baseline_target_correct", 0.0)) for r in rows])) if rows else 0.0)
            candidate_target_accuracy = self._clip01(float(np.mean([float(r.get("target_agent_correct", 0.0)) for r in rows])) if rows else 0.0)
            baseline_oracle_acc = self._clip01(float(np.mean([float(r.get("baseline_any_correct", 0.0)) for r in rows])) if rows else 0.0)
            candidate_oracle_acc = self._clip01(float(np.mean([float(r.get("candidate_any_correct", 0.0)) for r in rows])) if rows else 0.0)
            baseline_mean_vote_margin = float(np.mean([float(r.get("baseline_mean_vote_margin", -1.0)) for r in rows])) if rows else -1.0
            candidate_mean_vote_margin = float(np.mean([float(r.get("candidate_mean_vote_margin", -1.0)) for r in rows])) if rows else -1.0
            baseline_boundary_useful_diversity = self._clip01(float(np.mean([float(r.get("baseline_boundary_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            candidate_boundary_useful_diversity = self._clip01(float(np.mean([float(r.get("candidate_boundary_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            vote_transitions = compute_vote_transitions(
                [bool(row.get("baseline_vote_correct", 0)) for row in rows],
                [bool(row.get("candidate_vote_correct", 0)) for row in rows],
            )
            vote_delta = candidate_team_accuracy - baseline_team_accuracy
            if abs(float(vote_transitions["net_vote_delta"]) - float(vote_delta)) > PARETO_EPSILON:
                raise RuntimeError("Vote transition delta does not match candidate evaluation vote delta")
            coverage_delta = candidate_oracle_acc - baseline_oracle_acc
            coverage_transitions = compute_oracle_coverage_transitions(
                [list(row.get("baseline_individual_correct", [bool(row.get("baseline_any_correct", 0))])) for row in rows],
                [list(row.get("candidate_individual_correct", [bool(row.get("candidate_any_correct", 0))])) for row in rows],
            )
            if abs(float(coverage_transitions["net_coverage_delta"]) - float(coverage_delta)) > PARETO_EPSILON:
                raise RuntimeError("Oracle coverage transition delta does not match candidate evaluation coverage delta")
            coverage_depth_transitions = compute_coverage_depth_transitions(
                [list(row.get("baseline_individual_correct", [])) for row in rows],
                [list(row.get("candidate_individual_correct", [])) for row in rows],
                max_depth=len(self.agents),
            )
            if abs(float(coverage_depth_transitions.get("depth1_net_delta", 0.0)) - float(coverage_delta)) > PARETO_EPSILON:
                raise RuntimeError("Coverage depth-1 delta does not match oracle delta")
            rescue_rate = self._clip01(float(np.mean([float(r.get("rescue", 0.0)) for r in rows])) if rows else 0.0)
            useful_diversity = self._clip01(float(np.mean([float(r.get("target_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            rescue_useful_diversity = self._clip01(float(np.mean([float(r.get("rescue_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            if self._is_vote_useful_diversity_mode() or self._is_competence_depth_reward_mode():
                baseline_candidate_metrics = self._candidate_reward_vote_useful_diversity(
                    baseline_team_accuracy=baseline_team_accuracy,
                    candidate_team_accuracy=candidate_team_accuracy,
                    baseline_target_accuracy=baseline_target_accuracy,
                    candidate_target_accuracy=candidate_target_accuracy,
                    baseline_invalid_rate=baseline_invalid_rate,
                    candidate_invalid_rate=candidate_invalid_rate,
                    baseline_mean_vote_margin=baseline_mean_vote_margin,
                    candidate_mean_vote_margin=candidate_mean_vote_margin,
                    baseline_boundary_useful_diversity=baseline_boundary_useful_diversity,
                    candidate_boundary_useful_diversity=candidate_boundary_useful_diversity,
                    baseline_oracle_accuracy=baseline_oracle_acc,
                    candidate_oracle_accuracy=candidate_oracle_acc,
                    baseline_embedding_diversity=baseline_embedding_diversity,
                    candidate_embedding_diversity=candidate_embedding_diversity,
                )
            else:
                baseline_candidate_metrics = self._candidate_reward_guarded(
                    baseline_team_accuracy=baseline_team_accuracy,
                    candidate_team_accuracy=candidate_team_accuracy,
                    baseline_target_accuracy=baseline_target_accuracy,
                    candidate_target_accuracy=candidate_target_accuracy,
                    baseline_embedding_diversity=baseline_embedding_diversity,
                    candidate_embedding_diversity=candidate_embedding_diversity,
                    baseline_invalid_rate=baseline_invalid_rate,
                    candidate_invalid_rate=candidate_invalid_rate,
                )
            baseline_candidate_metrics.update(
                {
                    "baseline_team_accuracy": baseline_team_accuracy,
                    "candidate_team_accuracy": candidate_team_accuracy,
                    "baseline_oracle_acc": baseline_oracle_acc,
                    "candidate_oracle_acc": candidate_oracle_acc,
                    "coverage_delta": float(coverage_delta),
                    **vote_transitions,
                    "plurality_vote_gain_count": int(vote_transitions["vote_gain_count"]),
                    "plurality_vote_gain_rate": float(vote_transitions["vote_gain_rate"]),
                    "plurality_vote_loss_count": int(vote_transitions["vote_loss_count"]),
                    "plurality_vote_loss_rate": float(vote_transitions["vote_loss_rate"]),
                    "plurality_vote_net_count": int(vote_transitions["net_vote_count"]),
                    "plurality_vote_net_delta": float(vote_transitions["net_vote_delta"]),
                    **coverage_transitions,
                    **coverage_depth_transitions,
                    "baseline_gold_vote_count": float(np.mean([float(row.get("baseline_gold_vote_count", 0.0)) for row in rows])) if rows else 0.0,
                    "candidate_gold_vote_count": float(np.mean([float(row.get("candidate_gold_vote_count", 0.0)) for row in rows])) if rows else 0.0,
                    "baseline_largest_wrong_vote_count": float(np.mean([float(row.get("baseline_largest_wrong_vote_count", 0.0)) for row in rows])) if rows else 0.0,
                    "candidate_largest_wrong_vote_count": float(np.mean([float(row.get("candidate_largest_wrong_vote_count", 0.0)) for row in rows])) if rows else 0.0,
                    "baseline_plurality_margin_votes": float(np.mean([float(row.get("baseline_plurality_margin_votes", 0.0)) for row in rows])) if rows else 0.0,
                    "candidate_plurality_margin_votes": float(np.mean([float(row.get("candidate_plurality_margin_votes", 0.0)) for row in rows])) if rows else 0.0,
                    "plurality_margin_vote_delta": float(np.mean([
                        float(row.get("candidate_plurality_margin_votes", 0.0))
                        - float(row.get("baseline_plurality_margin_votes", 0.0)) for row in rows
                    ])) if rows else 0.0,
                    "baseline_normalized_plurality_margin": float(np.mean([float(row.get("baseline_normalized_plurality_margin", -1.0)) for row in rows])) if rows else -1.0,
                    "candidate_normalized_plurality_margin": float(np.mean([float(row.get("candidate_normalized_plurality_margin", -1.0)) for row in rows])) if rows else -1.0,
                    "normalized_plurality_margin_delta": float(np.mean([
                        float(row.get("candidate_normalized_plurality_margin", -1.0))
                        - float(row.get("baseline_normalized_plurality_margin", -1.0)) for row in rows
                    ])) if rows else 0.0,
                    "baseline_plurality_vote_tie": float(np.mean([int(bool(row.get("baseline_plurality_vote_tie", False))) for row in rows])) if rows else 0.0,
                    "candidate_plurality_vote_tie": float(np.mean([int(bool(row.get("candidate_plurality_vote_tie", False))) for row in rows])) if rows else 0.0,
                    "baseline_mean_vote_margin": baseline_mean_vote_margin,
                    "candidate_mean_vote_margin": candidate_mean_vote_margin,
                    "vote_margin_delta": candidate_mean_vote_margin - baseline_mean_vote_margin,
                    "baseline_boundary_useful_diversity": baseline_boundary_useful_diversity,
                    "candidate_boundary_useful_diversity": candidate_boundary_useful_diversity,
                    "boundary_useful_diversity_delta": candidate_boundary_useful_diversity - baseline_boundary_useful_diversity,
                    "baseline_target_accuracy": baseline_target_accuracy,
                    "candidate_target_accuracy": candidate_target_accuracy,
                    "target_agent_accuracy": candidate_target_accuracy,
                    "rescue_rate": rescue_rate,
                    "useful_diversity": useful_diversity,
                    "rescue_useful_diversity": rescue_useful_diversity,
                    "baseline_embedding_diversity": baseline_embedding_diversity,
                    "candidate_embedding_diversity": candidate_embedding_diversity,
                    "baseline_invalid_rate": baseline_invalid_rate,
                    "candidate_invalid_rate": candidate_invalid_rate,
                    "baseline_mean_embedding_overlap": self._clip01(float(np.mean([float(r.get("baseline_mean_embedding_overlap", 0.0)) for r in rows])) if rows else 0.0),
                    "candidate_mean_embedding_overlap": self._clip01(float(np.mean([float(r.get("candidate_mean_embedding_overlap", 0.0)) for r in rows])) if rows else 0.0),
                }
            )
            if self._is_rollout_qd_method():
                def profile_for(agent_index: int, candidate: bool) -> Dict[str, Any]:
                    prefix = "candidate" if candidate else "baseline"
                    answers = [
                        list(row.get(f"{prefix}_answers", []))[agent_index]
                        if agent_index < len(row.get(f"{prefix}_answers", [])) else ""
                        for row in rows
                    ]
                    correctness = [
                        int(list(row.get(f"{prefix}_individual_correct", []))[agent_index])
                        if agent_index < len(row.get(f"{prefix}_individual_correct", [])) else 0
                        for row in rows
                    ]
                    invalid = [
                        int(list(row.get(f"{prefix}_invalid_flags", []))[agent_index])
                        if agent_index < len(row.get(f"{prefix}_invalid_flags", [])) else 1
                        for row in rows
                    ]
                    traces = [
                        list(row.get(f"{prefix}_traces", []))[agent_index]
                        if agent_index < len(row.get(f"{prefix}_traces", [])) else ""
                        for row in rows
                    ]
                    useful_wrong = [
                        int(wrong_diversity_is_useful(row, candidate=candidate))
                        for row in rows
                    ]
                    profile = {
                        "answer_vector": answers,
                        "correctness_vector": correctness,
                        "invalid_vector": invalid,
                        "trace_embedding_vector_per_question": [
                            self._encode_trace_document(trace) if not invalid[index] else []
                            for index, trace in enumerate(traces)
                        ],
                        "wrong_diversity_useful_vector": useful_wrong,
                        "question_hashes": [str(row.get("question_hash", "")) for row in rows],
                    }
                    profile["rollout_signature_hash"] = rollout_signature(profile)
                    return profile

                baseline_profiles = [profile_for(index, False) for index in range(len(self.agents))]
                candidate_profiles = [profile_for(index, True) for index in range(len(self.agents))]
                def target_diversity(profiles: Sequence[Mapping[str, Any]]) -> tuple[float, Dict[str, float]]:
                    distances = [
                        rollout_distance(
                            profiles[agent_id], profiles[peer_id],
                            correctness_weight=self.cfg.rollout_correct_distance_weight,
                            wrong_weight=self.cfg.rollout_wrong_distance_weight,
                            trace_weight=self.cfg.rollout_trace_distance_weight,
                        )
                        for peer_id in range(len(profiles)) if peer_id != agent_id
                    ]
                    keys = (
                        "correct_set_rollout_distance", "useful_wrong_answer_dispersion",
                        "rollout_trace_embedding_distance", "rollout_distance",
                    )
                    mean = {key: float(np.mean([row[key] for row in distances])) if distances else 0.0 for key in keys}
                    return mean["rollout_distance"], mean

                baseline_rollout_diversity, baseline_components = target_diversity(baseline_profiles)
                candidate_rollout_diversity, candidate_components = target_diversity(candidate_profiles)
                transitions = candidate_transition_metrics(rows)
                baseline_candidate_metrics.update({
                    **transitions,
                    "rollout_profile": candidate_profiles[agent_id],
                    "baseline_rollout_profile": baseline_profiles[agent_id],
                    "baseline_rollout_diversity": baseline_rollout_diversity,
                    "candidate_rollout_diversity": candidate_rollout_diversity,
                    "rollout_diversity_delta": candidate_rollout_diversity - baseline_rollout_diversity,
                    "baseline_rollout_distance_components": baseline_components,
                    "candidate_rollout_distance_components": candidate_components,
                    "mechanism_based_decision_count": 0,
                })
                guard = rollout_quality_guard(baseline_candidate_metrics, self.cfg)
                baseline_candidate_metrics.update(guard)
                reward = rollout_candidate_reward(
                    baseline_candidate_metrics,
                    self.cfg,
                    vote_ready=self._is_vote_ready_rollout_method(),
                )
                if not guard["rollout_quality_guard_passed"]:
                    reward = -1.0
                baseline_candidate_metrics.update({
                    "reward": float(reward),
                    "reward_total": float(reward),
                    "rollout_reward_mode": (
                        "vote_ready" if self._is_vote_ready_rollout_method() else "accuracy_rollout_embedding"
                    ),
                })
            if bool(getattr(self.cfg, "shared_error_metrics_enabled", False)) or self._uses_vote_error_pareto_selection() or self._uses_competence_depth_pareto_selection() or self._residual_specialization_enabled():
                baseline_candidate_metrics.update(self._candidate_boundary_error_metrics(rows))
                paired_keys = (
                    "question_hash", "baseline_target_correct", "candidate_target_correct",
                    "peer_wrong_count", "baseline_vote_correct", "candidate_vote_correct",
                    "baseline_vote_margin", "candidate_vote_margin", "counterfactual_gold_vote_correct",
                    "counterfactual_gold_margin", "baseline_target_in_dominant_wrong_cluster",
                    "candidate_target_in_dominant_wrong_cluster", "capability_residual_family",
                )
                baseline_candidate_metrics["paired_boundary_transition_rows"] = [
                    {key: row.get(key) for key in paired_keys} for row in rows
                ]
            if self._residual_specialization_enabled():
                baseline_candidate_metrics.update(self._candidate_residual_metrics(rows))
                baseline_candidate_metrics["capability_alignment"] = self.capability_alignment(
                    self.agents[agent_id], baseline_candidate_metrics
                )
            if self._v7_residual_protocol_enabled():
                baseline_candidate_metrics.update(self._candidate_behavior_metrics(rows))
            baseline_candidate_metrics.update(
                compute_candidate_metric_deltas(
                    baseline_target_accuracy=baseline_target_accuracy,
                    candidate_target_accuracy=candidate_target_accuracy,
                    baseline_team_accuracy=baseline_team_accuracy,
                    candidate_team_accuracy=candidate_team_accuracy,
                    baseline_oracle_accuracy=baseline_oracle_acc,
                    candidate_oracle_accuracy=candidate_oracle_acc,
                    baseline_embedding_diversity=baseline_embedding_diversity,
                    candidate_embedding_diversity=candidate_embedding_diversity,
                    baseline_invalid_rate=baseline_invalid_rate,
                    candidate_invalid_rate=candidate_invalid_rate,
                )
            )
            if self._is_coverage_useful_diversity_mode():
                baseline_candidate_metrics.update(self._candidate_reward_coverage_useful_diversity(baseline_candidate_metrics))
            if self._is_competence_depth_reward_mode():
                v7_reward = float(baseline_candidate_metrics.get("reward", 0.0) or 0.0)
                baseline_candidate_metrics.update(
                    self._candidate_reward_competence_depth(baseline_candidate_metrics, v7_reward)
                )
            reward = float(baseline_candidate_metrics.get("reward", 0.0))
        else:
            reward = team_accuracy
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
            **self._candidate_eval_audit_fields(eval_batch),
        }
        result.update(baseline_candidate_metrics)
        return result
