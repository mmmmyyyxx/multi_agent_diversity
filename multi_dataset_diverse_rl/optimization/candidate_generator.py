"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class CandidateGeneratorMixin:
    _V9_GENERATION_FORBIDDEN_TOKENS = (
        "dominant_wrong", "wrong_cluster", "wrong_split", "dispersion",
        "boundary_useful_diversity", "dominant wrong", "wrong cluster",
        "wrong-answer split", "wrong-answer dispersion", "boundary useful diversity",
        "diversity",
    )

    @classmethod
    def _sanitize_v9_generation_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._sanitize_v9_generation_value(item)
                for key, item in value.items()
                if not any(token in str(key).lower() for token in cls._V9_GENERATION_FORBIDDEN_TOKENS)
                and not (
                    isinstance(item, str)
                    and any(token in item.lower() for token in cls._V9_GENERATION_FORBIDDEN_TOKENS)
                )
            }
        if isinstance(value, list):
            return [
                cls._sanitize_v9_generation_value(item)
                for item in value
                if not (
                    isinstance(item, str)
                    and any(token in item.lower() for token in cls._V9_GENERATION_FORBIDDEN_TOKENS)
                )
            ]
        return value

    def _build_v9_sequential_teacher_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        focus = dict(context.get("diagnostic_focus", {}) or {})
        target_error_summary = str(focus.get("target_error_summary", ""))
        target_error_summary = target_error_summary.split(
            "; target_dominant_wrong_redundancy_count=", 1
        )[0]
        v9_context = {
            "target_agent_id": int(context.get("target_agent_id", 0) or 0),
            "parent_prompt_preview": str(context.get("parent_prompt_preview", "")),
            "peer_prompt_summaries": list(context.get("peer_role_specs", []) or []),
            "validity_constraints": dict(context.get("validity_constraints", {}) or {}),
            "optimization_routes": list(context.get("optimization_routes", []) or []),
            "generation_batches": list(context.get("generation_batches", []) or []),
            "diagnostic_focus": {
                "problem_type": str(focus.get("problem_type", "")),
                "answer_format": str(focus.get("answer_format", "")),
                "target_error_patterns": list(focus.get("target_error_patterns", []) or []),
                "invalid_output_patterns": list(focus.get("invalid_output_patterns", []) or []),
                "prompt_redundancy_summary": str(focus.get("prompt_redundancy_summary", "")),
                "peer_behavior_summary": list(focus.get("peer_behavior_summary", []) or []),
                "target_error_summary": target_error_summary,
                "invalid_output_summary": str(focus.get("invalid_output_summary", "")),
                "optimization_goal": (
                    "Follow the requested route. Repair target errors and invalid output while preserving correct "
                    "cases. Coverage repair makes the target correct on C0/C1 cases; vote conversion makes the "
                    "target correct on C2/C3 cases. Generic procedural redundancy may be reduced, but "
                    "disagreement and trace difference are not objectives."
                ),
            },
        }
        sanitized = self._sanitize_v9_generation_value(v9_context)
        serialized = json.dumps(sanitized, ensure_ascii=False, sort_keys=True).lower()
        forbidden_count = sum(serialized.count(token) for token in self._V9_GENERATION_FORBIDDEN_TOKENS)
        if not isinstance(getattr(self, "state_search_diagnostics", None), dict):
            self.state_search_diagnostics = {}
        self.state_search_diagnostics["optimizer_context_audit_count"] = int(
            self.state_search_diagnostics.get("optimizer_context_audit_count", 0) or 0
        ) + 1
        self.state_search_diagnostics["optimizer_context_wrong_cluster_field_count"] = int(
            self.state_search_diagnostics.get("optimizer_context_wrong_cluster_field_count", 0) or 0
        ) + forbidden_count
        self.state_search_diagnostics["optimizer_context_forbidden_signal_count"] = int(
            self.state_search_diagnostics.get("optimizer_context_forbidden_signal_count", 0) or 0
        ) + forbidden_count
        if forbidden_count:
            raise RuntimeError("V9 optimizer context contains forbidden wrong-cluster diagnostics")
        return sanitized

    def _ensure_candidate_channel_funnel(self) -> None:
        if not isinstance(getattr(self, "candidate_channel_funnel", None), dict):
            self.candidate_channel_funnel = empty_candidate_channel_funnel()
        if not isinstance(getattr(self, "candidate_channel_funnel_seen", None), dict):
            self.candidate_channel_funnel_seen = {}

    def _record_candidate_funnel_item(self, item: Mapping[str, Any], agent_id: int, stage: str) -> bool:
        self._ensure_candidate_channel_funnel()
        return record_candidate_stage(
            self.candidate_channel_funnel,
            self.candidate_channel_funnel_seen,
            item,
            agent_id=agent_id,
            stage=stage,
        )

    def _record_candidate_funnel_classification(
        self, item: Mapping[str, Any], agent_id: int, stage: str
    ) -> bool:
        self._ensure_candidate_channel_funnel()
        return record_candidate_classification(
            self.candidate_channel_funnel,
            self.candidate_channel_funnel_seen,
            item,
            agent_id=agent_id,
            stage=stage,
        )

    def _record_candidate_funnel_outcomes(
        self,
        *,
        agent_id: int,
        evaluated: Sequence[Dict[str, Any]],
        safe_archive: Sequence[Mapping[str, Any]],
        epoch: int,
    ) -> None:
        bucket_stages = {
            "safe": "safe_count",
            "probation": "probation_count",
            "catastrophic": "catastrophic_count",
        }
        for item in evaluated:
            self._record_candidate_funnel_item(item, agent_id, "evaluated_candidate_count")
            bucket = str(item.get("archive_bucket", "catastrophic"))
            if bucket == "safe":
                item.setdefault("safe_created_epoch", int(epoch))
            self._record_candidate_funnel_classification(
                item, agent_id, bucket_stages.get(bucket, "catastrophic_count")
            )
        for item in safe_archive:
            self._record_candidate_funnel_item(item, agent_id, "archive_retained_count")

    def _record_generation_channel_funnel(
        self,
        *,
        agent_id: int,
        parent_id: str,
        channel: str,
        refill_round: int,
        raw_candidate_count: int,
    ) -> None:
        self._ensure_candidate_channel_funnel()
        audit = dict(TCS_AUDIT_CONTEXT.get() or {})
        identity = (
            f"e{int(audit.get('epoch', 0) or 0)}:s{int(audit.get('step', 0) or 0)}:"
            f"a{int(agent_id)}:p{parent_id}:r{int(refill_round)}:{channel}"
        )
        record_funnel_event(
            self.candidate_channel_funnel,
            self.candidate_channel_funnel_seen,
            channel=channel,
            stage="generation_call_count",
            identity=identity,
        )
        record_funnel_event(
            self.candidate_channel_funnel,
            self.candidate_channel_funnel_seen,
            channel=channel,
            stage="raw_candidate_count",
            identity=identity,
            amount=raw_candidate_count,
        )

    @staticmethod
    def _parse_optional_strict_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return False

    def _critic_review_passed(self, review: Mapping[str, Any], threshold: float) -> bool:
        score = self._safe_float(review.get("score", 0.0), 0.0)
        declared_pass = self._parse_optional_strict_bool(review.get("passed"))
        if declared_pass is None:
            return score >= float(threshold)
        return declared_pass and score >= float(threshold)

    def _build_teacher_context(
        self,
        agent_id: int,
        parent_prompt: str,
        target_role_spec: Dict[str, Any],
        peer_role_specs: List[Dict[str, Any]],
        window_stats: Dict[str, Any],
        validity_constraints: Dict[str, Any],
        generation_batches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        safe_generation_batches: List[Dict[str, Any]] = []
        target_error_patterns: List[str] = []
        invalid_output_patterns: List[str] = []
        peer_behavior_summary: List[str] = []
        batch_types: List[str] = []
        for batch in generation_batches:
            if not isinstance(batch, dict):
                continue
            safe_cases = []
            batch_type = str(batch.get("batch_type", ""))
            optimization_route = str(batch.get("optimization_route", "general_accuracy") or "general_accuracy")
            if batch_type:
                batch_types.append(batch_type)
            for case in batch.get("cases", []):
                if not isinstance(case, dict):
                    continue
                case_type = str(case.get("case_type", "") or case.get("purpose", "") or batch_type)
                if case_type:
                    target_error_patterns.append(case_type)
                invalids = case.get("invalid_reasons", [])
                if isinstance(invalids, list):
                    invalid_output_patterns.extend(str(x) for x in invalids if str(x))
                elif invalids:
                    invalid_output_patterns.append(str(invalids))
                peer_summary = str(case.get("peer_behavior_summary", "") or case.get("purpose", "") or "").strip()
                if peer_summary:
                    peer_behavior_summary.append(normalize_spaces(peer_summary)[:180])
                safe_case = {
                        "case_type": case_type,
                        "target_agent_id": int(case.get("target_agent_id", agent_id) or agent_id),
                        "target_correct": case.get("target_correct", ""),
                        "target_invalid": case.get("target_invalid", ""),
                        "peer_correct_available": case.get("peer_correct_available", ""),
                        "purpose": normalize_spaces(str(case.get("purpose", "")))[:160],
                        "repair_hint": normalize_spaces(str(case.get("repair_hint", "")))[:180],
                        "target_overlap_pressure": case.get("target_overlap_pressure", ""),
                }
                if self._is_state_conditioned_method():
                    safe_case.update({
                        "state": str(case.get("state", "")),
                        "baseline_correct_count": int(case.get("baseline_correct_count", -1) or 0),
                        "option_count": int(case.get("option_count", 0) or 0),
                        "coverage_assigned_agents": list(case.get("coverage_assigned_agents", [])),
                    })
                    for field in (
                        "shared_failure_category", "generalized_failure_mechanism",
                        "missing_reasoning_step", "misleading_reasoning_step",
                        "generalizable_repair_rule", "ambiguity_handling_rule", "memorization_risk",
                    ):
                        if field in case:
                            safe_case[field] = normalize_spaces(str(case.get(field, "")))[:220]
                if self._v7_residual_protocol_enabled():
                    safe_case.update({
                        "capability_residual_family": str(case.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                        "vote_context": str(case.get("vote_context", "")),
                    })
                safe_cases.append(safe_case)
            safe_generation_batches.append(
                {
                    "batch_type": batch_type,
                    "optimization_route": optimization_route,
                    "purpose": normalize_spaces(str(batch.get("purpose", "")))[:200],
                    "case_count": len(safe_cases),
                    "cases": safe_cases,
                }
            )
        answer_format = str(getattr(self.cfg, "answer_format", "") or "").strip() or str(getattr(self.cfg, "task_type", "auto"))
        problem_type = str(getattr(self.cfg, "comparison_task_id", "") or getattr(self.cfg, "benchmark", "") or getattr(self.cfg, "task_type", "auto"))
        target_pressure = float(window_stats.get("target_overlap_pressure", 0.0) or 0.0)
        mean_overlap = float(window_stats.get("mean_window_overlap", 0.0) or 0.0)
        target_invalid_rate = float(window_stats.get("target_invalid_rate", 0.0) or 0.0)
        target_error_count = int(window_stats.get("target_error_count", 0) or 0)
        target_team_wrong_error_count = int(window_stats.get("target_team_wrong_error_count", 0) or 0)
        target_pivotal_fix_count = int(window_stats.get("target_pivotal_fix_count", 0) or 0)
        target_dominant_wrong_count = int(window_stats.get("target_dominant_wrong_redundancy_count", 0) or 0)
        window_vote_acc = float(window_stats.get("window_vote_acc", window_stats.get("team_accuracy", 0.0)) or 0.0)
        window_vote_margin = float(window_stats.get("window_mean_vote_margin", -1.0) if window_stats.get("window_mean_vote_margin") is not None else -1.0)
        window_boundary_diversity = float(window_stats.get("window_mean_boundary_useful_diversity", 0.0) or 0.0)
        residual_families = sorted({
            str(case.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value))
            for batch in generation_batches if isinstance(batch, dict)
            for case in batch.get("cases", []) if isinstance(case, dict)
        })
        context = {
            "target_agent_id": agent_id,
            "parent_prompt_preview": normalize_spaces(parent_prompt)[:600],
            "target_role_spec": target_role_spec,
            "peer_role_specs": peer_role_specs,
            "window_stats": window_stats,
            "validity_constraints": validity_constraints,
            "generation_batches": safe_generation_batches,
            "optimization_routes": sorted({
                str(batch.get("optimization_route", "general_accuracy") or "general_accuracy")
                for batch in safe_generation_batches
            }),
            "diagnostic_focus": {
                "problem_type": problem_type,
                "answer_format": answer_format,
                "target_error_patterns": sorted(set(target_error_patterns))[:12],
                "invalid_output_patterns": sorted(set(invalid_output_patterns))[:12],
                "diversity_gap_summary": (
                    f"window_vote_acc={window_vote_acc:.3f}; window_mean_vote_margin={window_vote_margin:.3f}; "
                    f"window_boundary_useful_diversity={window_boundary_diversity:.3f}; "
                    f"target_pivotal_fix_count={target_pivotal_fix_count}; target_dominant_wrong_redundancy_count={target_dominant_wrong_count}; "
                    f"diagnostic_embedding_overlap={mean_overlap:.3f}; target_embedding_overlap_pressure={target_pressure:.3f}; "
                    f"batch_types={sorted(set(batch_types))[:8]}"
                ),
                "prompt_redundancy_summary": (
                    f"target prompt preview is compared against {len(peer_role_specs)} peer role previews; "
                    f"avoid duplicating peer procedures and parent wording."
                ),
                "error_correlation_summary": (
                    "Use target error cases and vote-boundary diagnostics as abstract repair signals. "
                    "Voting failures are included by default."
                    if not bool(getattr(self.cfg, "teacher_critic_use_voting_failure", False))
                    else "Voting failures, pivotal fixes, and dominant wrong-answer redundancy are primary diagnostics."
                ),
                "peer_behavior_summary": peer_behavior_summary[:8],
                "target_error_summary": (
                    f"target_error_count={target_error_count}; "
                    f"target_team_wrong_error_count={target_team_wrong_error_count}; "
                    f"target_pivotal_fix_count={target_pivotal_fix_count}; "
                    f"target_dominant_wrong_redundancy_count={target_dominant_wrong_count}"
                ),
                "invalid_output_summary": f"target_invalid_rate={target_invalid_rate:.3f}; invalid patterns are abstracted above.",
            },
        }
        if self._v7_residual_protocol_enabled():
            context["diagnostic_focus"]["capability_residual_families"] = residual_families[:12]
            context["target_prompt_state"] = context.pop("target_role_spec")
            context["peer_prompt_summaries"] = context.pop("peer_role_specs")
        if self._residual_specialization_enabled():
            agent = self.agents[agent_id]
            ordered_profile = sorted(agent.capability_profile.items(), key=lambda item: (-item[1], item[0]))
            context["observed_long_term_capability_profile"] = {
                "strongest_supported_residual_families": [key for key, value in ordered_profile[:3] if value > 0.0],
                "capability_coverage_gap": dict(window_stats.get("capability_coverage_gap", {})),
                "residual_guidance_strength": float(self.specialization_strength) if bool(getattr(self.cfg, "competence_progressive_residual_enabled", False)) else 1.0,
                "guidance": (
                    "Treat residual-family evidence as observation only; do not steer the prompt from it yet."
                    if bool(getattr(self.cfg, "competence_progressive_residual_enabled", False)) and float(self.specialization_strength) <= 0.0
                    else "Use residual-family evidence as a strength-scaled historical affinity, never as an assigned role."
                ),
            }
        if self._is_stable_qd_lineage():
            target_state = self.agents[agent_id].lineage_state
            target_profile = getattr(self, "behavior_profile_by_prompt_hash", {}).get(
                self._normalized_prompt_hash(parent_prompt), {}
            )
            context["stable_lineage_context"] = {
                "lineage_status": str(target_state.get("lineage_status", "uncommitted")),
                "anchor_mechanism": list(target_state.get("lineage_anchor_mechanism_signature", [])),
                "stay_near_anchor_required": bool(target_state.get("lineage_status") == "committed"),
                "committed_peer_mechanisms": [
                    {
                        "agent_id": peer_id,
                        "mechanism": list(peer.lineage_state.get("lineage_anchor_mechanism_signature", [])),
                    }
                    for peer_id, peer in enumerate(self.agents)
                    if peer_id != agent_id and peer.lineage_state.get("lineage_status") == "committed"
                ],
                "target_behavior_residual": {
                    "rescue_support": int(sum(target_profile.get("rescue_vector", []))),
                    "unique_correct_support": int(sum(target_profile.get("unique_correct_vector", []))),
                    "shared_error_support": int(sum(target_profile.get("shared_error_vector", []))),
                    "window_target_error_count": target_error_count,
                    "window_target_team_wrong_error_count": target_team_wrong_error_count,
                },
                "guidance": (
                    "The target has no committed lineage: permit substantial mechanism changes while preserving competence."
                    if target_state.get("lineage_status") != "committed"
                    else "Prefer structural variants near the committed anchor, but allow a justified alternative for joint selection."
                ),
            }
        if isinstance(window_stats.get("refill_feedback"), dict):
            context["candidate_refill_feedback"] = dict(window_stats["refill_feedback"])
        if self._is_rollout_qd_method():
            context.pop("parent_prompt_preview", None)
            context.pop("target_role_spec", None)
            context.pop("peer_role_specs", None)
            context["diagnostic_focus"].pop("prompt_redundancy_summary", None)
            context["diagnostic_focus"].pop("capability_residual_families", None)
            context["diagnostic_focus"]["rollout_optimization_goal"] = (
                "Improve target correctness first; prioritize C2-to-C3, C3-to-C4, and vote recovery; "
                "use valid solver-trace diversity only after quality guards."
            )
        if self._is_state_conditioned_method():
            return self._build_v9_sequential_teacher_context(context)
        return context

    async def propose_teacher_question(
        self,
        agent_id: int,
        parent_prompt: str,
        teacher_context: Dict[str, Any],
        requested_candidates: int,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the Teacher in a Teacher-Critic-Student prompt optimization system.\n\n"
            "Your job is not to write a prompt.\n"
            "Your job is to formulate a high-quality Socratic guiding question that will help the Student rewrite the target agent prompt.\n\n"
            "The guiding question must be grounded in:\n"
            "- problem type\n- answer format\n- target-agent error patterns\n- diversity gap\n"
            "- prompt redundancy\n- error correlation with peer agents\n- peer behavior summaries\n"
            "- invalid-output patterns if present\n\n"
            "Do not use gold answers.\nDo not use concrete question text.\nDo not use concrete answer labels.\n"
            "Do not create task-specific hard-coded roles.\nDo not optimize for voting failure in this step.\n"
            "Do not ask a generic question such as 'How can the prompt be improved?'\n\n"
            "A good guiding question should force the Student to create a candidate prompt that:\n"
            "- aligns with the task/problem type\n- repairs a specific observed error pattern\n"
            "- improves target-agent accuracy\n- contributes useful reasoning diversity\n"
            "- avoids duplicating peer prompts\n- avoids invalid or overlong outputs\n\n"
            "Return strict JSON only."
        )
        if self._is_rollout_qd_method():
            system_prompt = (
                "You are the Teacher in a Teacher-Critic-Student prompt optimization system. Formulate one Socratic "
                "question from observed solver rollout errors. Seek an executable accuracy repair, especially C2-to-C3 "
                "conversion, vote recovery, margin improvement, or dominant wrong-cluster reduction. Rollout diversity is "
                "secondary and must never reward random errors or invalid traces. Do not compare prompt wording, propose "
                "named mechanisms, assign capability labels or roles, use gold answers, or quote samples. Return strict JSON only."
            )
        if self._is_state_conditioned_method():
            system_prompt = (
                "You are the Teacher in a state-conditioned prompt optimization system. Formulate one Socratic "
                "question for the requested route: general accuracy, C0/C1 correct coverage, or C2/C3 vote conversion. "
                "Correctness comes first. Vote conversion must make the target correct; label-only changes have no "
                "value. Never praise random disagreement, prompt wording "
                "difference, personas, assigned roles, or trace difference. Do not quote cases or reveal answers. "
                "Return strict JSON only."
            )
        if self._v7_residual_protocol_enabled():
            system_prompt = system_prompt.replace(
                "Do not optimize for voting failure in this step.\n",
                "Use voting failures only as abstract evidence of harmful shared-error mechanisms. "
                "Do not game the vote directly, memorize sample answers, or optimize for disagreement by itself.\n",
            )
            system_prompt += (
                "\nFocus the guiding question on one residual error mechanism, a possible pivotal correction, "
                "and how to preserve pivotal-correct behavior. Ask which local executable reasoning mechanism "
                "could make the target correct when several peers also fail."
            )
        user_prompt = (
            "Create one Socratic guiding question for the Student.\n"
            "Return JSON with keys: problem_type_analysis, answer_format_analysis, target_error_analysis, "
            "diversity_gap_analysis, error_correlation_analysis, peer_difference_analysis, socratic_guiding_question, "
            "question_objective, expected_prompt_change, expected_accuracy_effect, expected_diversity_effect, risk_to_avoid.\n\n"
            f"target_agent_id: {agent_id}\nrequested_candidates: {requested_candidates}\n"
            f"parent_prompt_preview:\n{normalize_spaces(parent_prompt)[:600]}\n\n"
            f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
        )
        if self._is_state_conditioned_method():
            user_prompt = (
                "Create one Socratic guiding question for an accuracy-first prompt repair. Return JSON with keys: "
                "problem_type_analysis, answer_format_analysis, target_error_analysis, invalid_output_analysis, "
                "correct_case_preservation, socratic_guiding_question, question_objective, expected_prompt_change, "
                "expected_accuracy_effect, risk_to_avoid.\n\n"
                f"target_agent_id: {agent_id}\nrequested_candidates: {requested_candidates}\n"
                f"parent_prompt_preview:\n{normalize_spaces(parent_prompt)[:600]}\n\n"
                f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
            )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "teacher_temperature", self.cfg.optimizer_temperature)),
            max_tokens=int(getattr(self.cfg, "teacher_max_tokens", self.cfg.optimizer_max_tokens)),
            stage=f"teacher_agent_{agent_id}",
            client_role="optimizer",
        )
        obj = extract_json_obj(text) or {}
        return obj if isinstance(obj, dict) else {}

    async def critique_teacher_question(
        self,
        agent_id: int,
        teacher_question: Dict[str, Any],
        teacher_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the Critic in a Teacher-Critic-Student prompt optimization system.\n\n"
            "Your job is to audit the Teacher's Socratic guiding question before the Student sees it.\n\n"
            "Reject the question if it is:\n"
            "- generic\n- not grounded in observed diagnostics\n- not aligned with the problem type\n"
            "- not aligned with the target error pattern\n- only about surface-level diversity\n"
            "- likely to duplicate peer prompts\n- likely to reduce answer accuracy\n"
            "- using gold answers, concrete sample text, or answer labels\n"
            "- using hard-coded task-specific roles\n"
            "- focused on voting failure rather than prompt quality/diversity/accuracy\n\n"
            "Also reject persona-only changes, wholesale rewrites for a narrow error, deletion of repeatedly effective mechanisms, "
            "repetition of recent failed edits, or vague non-executable reasoning steps. Prefer a concrete local mechanism edit.\n\n"
            "Return strict JSON only."
        )
        if self._is_rollout_qd_method():
            system_prompt = (
                "Audit whether the Teacher question is specific to observed solver rollout failures and proposes an executable "
                "accuracy-first repair. Reject prompt-wording novelty, named mechanisms, capability labels, personas, random "
                "disagreement, invalid behavior, leakage, or any diversity goal that can reduce Vote, C3, or target accuracy. "
                "Prefer C2-to-C3 or C3-to-C4 conversion, target correctness, and vote recovery. Return strict JSON only."
            )
        if self._is_state_conditioned_method():
            system_prompt = (
                "Audit whether the Teacher question proposes one executable accuracy-first repair grounded in target "
                "errors or invalid output. Vote failures may identify cases only when the repair makes the target correct. "
                "Reject disagreement goals, trace-difference goals, personas, assigned roles, leakage, sample quoting, "
                "generic advice, or changes that risk previously correct cases. Return strict JSON only."
            )
        if self._v7_residual_protocol_enabled():
            system_prompt = system_prompt.replace(
                "- focused on voting failure rather than prompt quality/diversity/accuracy\n\n",
                "- gaming vote outcomes, memorizing answers, or pursuing disagreement by itself\n\n",
            )
            system_prompt += (
                "\nAudit whether the question targets a concrete residual error mechanism, can reduce a shared-error "
                "mechanism through a pivotal correction, preserves pivotal-correct behavior, avoids persona-only or "
                "wholesale rewrites, avoids repeated failed mechanisms, and specifies one executable local decision step."
            )
        user_prompt = (
            "Audit the Teacher question. Pass only if score >= threshold, the question is specific, grounded in diagnostics, "
            "contains no leakage or hard-coded task role, and is useful for both accuracy and diversity.\n"
            "Return JSON with keys: passed, score, quality_critique, specificity_critique, task_alignment_critique, "
            "error_alignment_critique, diversity_critique, redundancy_critique, safety_critique, rewrite_instruction.\n\n"
            f"target_agent_id: {agent_id}\n"
            f"pass_threshold: {float(getattr(self.cfg, 'teacher_question_pass_threshold', 0.75))}\n"
            f"teacher_question:\n{json.dumps(teacher_question, ensure_ascii=False, indent=2)}\n\n"
            f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
        )
        if self._is_state_conditioned_method():
            user_prompt = (
                "Audit the Teacher question. Pass only if it is a specific executable accuracy repair with explicit "
                "correct-case preservation and no leakage. Return JSON with keys: passed, score, quality_critique, "
                "specificity_critique, task_alignment_critique, error_alignment_critique, safety_critique, "
                "rewrite_instruction.\n\n"
                f"target_agent_id: {agent_id}\n"
                f"pass_threshold: {float(getattr(self.cfg, 'teacher_question_pass_threshold', 0.75))}\n"
                f"teacher_question:\n{json.dumps(teacher_question, ensure_ascii=False, indent=2)}\n\n"
                f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
            )
        text = await self._chat(
            model=self.cfg.evaluator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "critic_temperature", self.cfg.evaluator_temperature)),
            max_tokens=int(getattr(self.cfg, "critic_max_tokens", self.cfg.evaluator_max_tokens)),
            stage=f"teacher_critic_agent_{agent_id}",
            client_role="evaluator",
        )
        obj = extract_json_obj(text) or {}
        return obj if isinstance(obj, dict) else {}

    async def rewrite_teacher_question(
        self,
        agent_id: int,
        previous_question: Dict[str, Any],
        critic_review: Dict[str, Any],
        teacher_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the Teacher revising a Socratic guiding question after Critic feedback.\n"
            "Revise only the guiding question and its rationale. Do not write candidate prompts.\n"
            "Do not use gold answers, concrete sample text, answer labels, or hard-coded task-specific roles.\n"
            "Return strict JSON only."
        )
        if self._is_rollout_qd_method():
            system_prompt = (
                "Revise the Teacher question using Critic feedback. Keep it grounded in observed rollout failures and "
                "accuracy-first vote readiness. Do not add mechanism names, capability labels, prompt-text novelty, roles, "
                "gold answers, or sample text. Return strict JSON only."
            )
        if self._is_state_conditioned_method():
            system_prompt = (
                "Revise the Teacher question using Critic feedback. Keep one concrete accuracy repair, preserve correct "
                "cases, and address invalid output when present. Do not pursue disagreement, trace difference, personas, "
                "roles, leakage, or sample-specific rules. Return strict JSON only."
            )
        user_prompt = (
            "Rewrite the Teacher JSON so it can pass Critic review while staying grounded in the abstract diagnostics.\n"
            "Keep the same JSON schema as the Teacher output.\n\n"
            f"target_agent_id: {agent_id}\n"
            f"previous_question:\n{json.dumps(previous_question, ensure_ascii=False, indent=2)}\n\n"
            f"critic_review:\n{json.dumps(critic_review, ensure_ascii=False, indent=2)}\n\n"
            f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "teacher_temperature", self.cfg.optimizer_temperature)),
            max_tokens=int(getattr(self.cfg, "teacher_max_tokens", self.cfg.optimizer_max_tokens)),
            stage=f"teacher_rewrite_agent_{agent_id}",
            client_role="optimizer",
        )
        obj = extract_json_obj(text) or {}
        return obj if isinstance(obj, dict) else {}

    async def generate_approved_teacher_question(
        self,
        agent_id: int,
        parent_prompt: str,
        teacher_context: Dict[str, Any],
        requested_candidates: int,
    ) -> Dict[str, Any]:
        stable_qd = self._is_stable_qd_lineage()
        direct_threshold = float(getattr(self.cfg, "teacher_critic_direct_pass_threshold", 0.75) or 0.75) if stable_qd else float(getattr(self.cfg, "teacher_question_pass_threshold", 0.75) or 0.75)
        rewrite_threshold = float(getattr(self.cfg, "teacher_critic_rewrite_threshold", 0.50) or 0.50)
        forced_threshold = float(getattr(self.cfg, "teacher_critic_forced_best_threshold", 0.60) or 0.60)
        max_rounds = min(2, max(1, int(getattr(self.cfg, "teacher_critic_max_rounds", 2) or 2))) if stable_qd else max(1, int(getattr(self.cfg, "teacher_critic_max_rounds", 3) or 3))
        max_rewrites = min(1, max(0, int(getattr(self.cfg, "teacher_rewrite_max_count", 1) or 0))) if stable_qd else max(0, max_rounds - 1)
        teacher_question = await self.propose_teacher_question(agent_id, parent_prompt, teacher_context, requested_candidates)
        if self._is_state_conditioned_method():
            teacher_question = self._sanitize_v9_generation_value(teacher_question)
        reviews: List[Dict[str, Any]] = []
        question_versions: List[Dict[str, Any]] = []
        rewrite_count = 0

        def has_guiding_question(question: Any) -> bool:
            return bool(
                isinstance(question, dict)
                and str(question.get("socratic_guiding_question", "")).strip()
            )

        for round_id in range(max_rounds):
            round_context = dict(TCS_AUDIT_CONTEXT.get() or {})
            round_context["teacher_critic_round"] = round_id + 1
            round_token = TCS_AUDIT_CONTEXT.set(round_context)
            try:
                review = await self.critique_teacher_question(agent_id, teacher_question, teacher_context)
            finally:
                TCS_AUDIT_CONTEXT.reset(round_token)
            if self._is_state_conditioned_method():
                review = self._sanitize_v9_generation_value(review)
            reviews.append(review)
            question_versions.append(teacher_question)
            score = self._safe_float(review.get("score", 0.0), 0.0)
            if has_guiding_question(teacher_question) and self._critic_review_passed(
                review, direct_threshold
            ):
                return {
                    "approved": True,
                    "teacher_question": teacher_question,
                    "critic_reviews": reviews,
                    "teacher_critic_rounds": round_id + 1,
                    "teacher_rewrite_count": rewrite_count,
                    "teacher_question_forced_best_score": False,
                    "teacher_question_forced_best_round": 0,
                    "teacher_question_forced_below_threshold": False,
                }
            if stable_qd and round_id == 0 and score < rewrite_threshold:
                return {
                    "approved": False,
                    "teacher_question": teacher_question if has_guiding_question(teacher_question) else {},
                    "critic_reviews": reviews,
                    "teacher_critic_rounds": 1,
                    "teacher_rewrite_count": 0,
                    "teacher_question_forced_best_score": False,
                    "teacher_question_forced_best_round": 0,
                    "teacher_question_forced_below_threshold": True,
                    "teacher_question_forced_best_review": review,
                    "teacher_question_rejection_reason": "critic_score_below_rewrite_threshold",
                }
            if round_id < max_rounds - 1 and rewrite_count < max_rewrites:
                rewrite_context = dict(TCS_AUDIT_CONTEXT.get() or {})
                rewrite_context["teacher_critic_round"] = round_id + 1
                rewrite_token = TCS_AUDIT_CONTEXT.set(rewrite_context)
                try:
                    teacher_question = await self.rewrite_teacher_question(agent_id, teacher_question, review, teacher_context)
                finally:
                    TCS_AUDIT_CONTEXT.reset(rewrite_token)
                rewrite_count += 1
                if self._is_state_conditioned_method():
                    teacher_question = self._sanitize_v9_generation_value(teacher_question)
        usable_indices = [
            idx for idx, question in enumerate(question_versions)
            if has_guiding_question(question)
        ]
        if not usable_indices:
            return {
                "approved": False,
                "teacher_question": {},
                "critic_reviews": reviews,
                "teacher_critic_rounds": len(reviews),
                "teacher_rewrite_count": rewrite_count,
                "teacher_question_forced_best_score": False,
                "teacher_question_forced_best_round": 0,
                "teacher_question_forced_below_threshold": True,
                "teacher_question_forced_best_review": reviews[-1] if reviews else {},
                "teacher_question_rejection_reason": "empty_teacher_question",
            }

        best_idx = usable_indices[0]
        best_score = -1.0
        for idx in usable_indices:
            review = reviews[idx]
            score = self._safe_float(review.get("score", 0.0), 0.0)
            if score > best_score:
                best_idx = idx
                best_score = score
        best_review = reviews[best_idx] if reviews else {}
        best_question = question_versions[best_idx] if question_versions else teacher_question
        forced = (best_score >= forced_threshold) if stable_qd else True
        return {
            "approved": False,
            "teacher_question": best_question if forced else {},
            "critic_reviews": reviews,
            "teacher_critic_rounds": len(reviews),
            "teacher_rewrite_count": rewrite_count,
            "teacher_question_forced_best_score": forced,
            "teacher_question_forced_best_round": best_idx + 1,
            "teacher_question_forced_below_threshold": True,
            "teacher_question_forced_best_review": best_review,
            "teacher_question_rejection_reason": "" if forced else "critic_score_below_forced_best_threshold",
        }

    async def retry_student_candidates_json_only(
        self,
        previous_raw_text: str,
        approved_teacher_question: Dict[str, Any],
        num_candidates: int,
        agent_id: int = 0,
    ) -> str:
        schema = self._student_candidate_schema_json()
        prompt_max = int(
            getattr(self.cfg, "student_candidate_prompt_hard_max_chars", 1400)
            if bool(getattr(self.cfg, "competence_depth_enabled", False))
            else getattr(self.cfg, "student_candidate_prompt_max_chars", 900)
        )
        field_max = int(getattr(self.cfg, "student_candidate_max_chars_per_field", 320) or 320)
        system_prompt = (
            "Your previous response was not valid JSON.\n\n"
            "Return only valid minified JSON matching this exact schema:\n"
            f"{schema}\n\n"
            "Do not add markdown.\n"
            "Do not add explanations.\n"
            "Do not use multiline strings.\n"
            "Use double quotes for every key and string value.\n"
            "Do not include trailing commas or comments.\n"
            f"Use at most {int(num_candidates)} candidates.\n"
            f"candidate_prompt must be <= {prompt_max} characters.\n"
            f"Every other field must be <= {field_max} characters.\n"
            "Each candidate_prompt must be concise.\n"
            'If you cannot comply, return {"candidates":[]}.'
        )
        user_prompt = (
            "Approved Teacher question and context for the retry:\n"
            f"{json.dumps(approved_teacher_question, ensure_ascii=False, indent=2)}\n\n"
            "Previous invalid JSON-like response, for reference only:\n"
            f"{str(previous_raw_text or '')[:4000]}"
        )
        return await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "student_temperature", self.cfg.optimizer_temperature)),
            max_tokens=int(getattr(self.cfg, "student_max_tokens", self.cfg.optimizer_max_tokens)),
            stage=f"student_json_retry_agent_{agent_id}",
            client_role="optimizer",
        )

    async def repair_student_json_response(
        self,
        raw_text: str,
        expected_num_candidates: int,
    ) -> Dict[str, Any]:
        schema = self._student_candidate_schema_json()
        if not str(raw_text or "").strip():
            return {
                "repaired": False,
                "repair_raw_response_preview": "",
                "repair_json_parse_failed": True,
                "repair_failure_reason": "empty_raw_text",
                "obj": None,
            }
        system_prompt = (
            "You are a JSON repair utility.\n\n"
            "You will receive malformed JSON-like text that was intended to match this schema:\n"
            f"{schema}\n\n"
            "Your job is only to repair JSON syntax:\n"
            "- close braces and brackets if needed\n"
            "- escape unescaped quotes inside strings\n"
            "- remove trailing commas\n"
            "- keep only the candidates that are already present in the input\n"
            "- do not invent new candidates\n"
            "- do not change semantic content\n"
            "- do not add explanations\n"
            "- return minified JSON only\n\n"
            'If the input cannot be repaired without inventing content, return {"candidates":[]}.'
        )
        user_prompt = (
            f"expected_num_candidates: {int(expected_num_candidates)}\n\n"
            "Malformed JSON-like input:\n"
            f"{str(raw_text or '')[:6000]}"
        )
        try:
            repair_text = await self._chat(
                model=self.cfg.optimizer_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(getattr(self.cfg, "student_json_repair_temperature", 0.0) or 0.0),
                max_tokens=int(getattr(self.cfg, "student_json_repair_max_tokens", 1200) or 1200),
                stage="student_json_repair",
                client_role="optimizer",
            )
        except Exception as exc:
            return {
                "repaired": False,
                "repair_raw_response_preview": "",
                "repair_json_parse_failed": True,
                "repair_failure_reason": type(exc).__name__,
                "obj": None,
            }
        obj = extract_json_obj(repair_text or "")
        parse_failed = obj is None or not isinstance(obj, dict)
        return {
            "repaired": not parse_failed,
            "repair_raw_response_preview": normalize_spaces(repair_text or "")[:1000],
            "repair_json_parse_failed": bool(parse_failed),
            "repair_failure_reason": "" if not parse_failed else "repair_json_parse_failed",
            "obj": obj if isinstance(obj, dict) else None,
        }

    async def generate_student_candidates(
        self,
        agent_id: int,
        parent_prompt: str,
        approved_teacher_question: Dict[str, Any],
        teacher_context: Dict[str, Any],
        num_candidates: int,
        generation_channel: str = "tcs_repair",
    ) -> Dict[str, Any]:
        open_exploration = generation_channel in {"open_mechanism_exploration", "open_rollout_exploration"}
        rollout_exploration = generation_channel == "open_rollout_exploration"
        schema = self._student_candidate_schema_json()
        schema_mode = str(getattr(self.cfg, "student_candidate_schema_mode", "compact") or "compact").lower()
        prompt_max = int(
            getattr(self.cfg, "student_candidate_prompt_hard_max_chars", 1400)
            if bool(getattr(self.cfg, "competence_depth_enabled", False))
            else getattr(self.cfg, "student_candidate_prompt_max_chars", 900)
        )
        field_max = int(getattr(self.cfg, "student_candidate_max_chars_per_field", 320) or 320)
        compact_output_rules = (
            "Output format requirements:\n"
            "- Return exactly one JSON object.\n"
            "- The first character must be `{`.\n"
            "- The last character must be `}`.\n"
            "- Return minified JSON only.\n"
            "- Use double quotes for all JSON keys and string values.\n"
            "- Do not use Markdown.\n"
            "- Do not wrap the JSON in code fences.\n"
            "- Do not add explanations before or after the JSON.\n"
            "- Do not use multiline strings.\n"
            "- Do not include newline characters inside string values.\n"
            "- Do not include bullet lists inside string values.\n"
            "- Escape all quotes inside strings.\n"
            "- Do not include trailing commas.\n"
            "- Do not include comments.\n"
            f"- candidate_prompt must be <= {prompt_max} characters.\n"
            f"- Every other field must be <= {field_max} characters.\n"
            "- Each non-prompt field must be one short sentence.\n"
            "- Each candidate_prompt should be a concise solver instruction, not a long essay.\n"
            "- Return a complete standalone prompt. Do not end mid-sentence.\n"
            + (
                "- Keep only executable accuracy, correct-case preservation, and validity instructions.\n"
                if self._is_state_conditioned_method()
                else "- Keep only executable accuracy and validity instructions; do not describe named mechanisms or capabilities.\n"
                if self._is_rollout_qd_method()
                else "- Preserve useful mechanisms but merge repeated instructions; avoid repeatedly saying explicitly, systematically, before final selection, or check every constraint.\n"
            )
            +
            "- Prefer semicolon-separated steps over numbered multiline lists.\n"
            '- If you cannot safely generate a candidate, return {"candidates":[]}.\n'
            f"Exact schema:\n{schema}"
        )
        if schema_mode == "compact":
            return_mode = (
                "Return minified JSON only. Do not use Markdown, code fences, explanations, or multiline strings. "
                "The JSON must match the exact compact schema."
            )
            item_instruction = (
                "Each item must match the compact schema. Keep candidate_prompt concise and standalone; "
                "all other fields must be one short sentence."
            )
            format_rules = compact_output_rules
        else:
            return_mode = "Return strict JSON only."
            item_instruction = (
                "Each item must include candidate_prompt, student_interpretation_of_question, target_error_pattern, "
                "accuracy_repair_rule, diversity_contribution, error_correlation_reduction, task_alignment_rule, "
                "peer_redundancy_avoidance, expected_accuracy_effect, expected_diversity_effect, risk_control, rationale."
            )
            format_rules = (
                "Return JSON with a candidates list. Do not use Markdown or code fences. "
                f"Exact schema:\n{schema}"
            )
        if self._is_state_conditioned_method():
            item_instruction = (
                "Each item must match the schema and include one executable accuracy repair, one correct-case "
                "preservation rule, and one validity risk control."
            )
        system_prompt = (
            "You are the Student in a Teacher-Critic-Student prompt optimization system.\n\n"
            "You will receive:\n- the current parent prompt\n- an approved Socratic guiding question from the Teacher\n"
            "- Critic reviews of that question\n- abstract diagnostics about problem type, error type, diversity gap, "
            "error correlation, and peer behavior\n\n"
            "Your job is to generate candidate prompts for the target agent.\n\n"
            "Each candidate prompt must:\n- directly answer the approved guiding question\n- be a complete standalone prompt\n"
            "- align with the problem type and answer format\n- repair the target error pattern\n"
            "- improve target-agent accuracy\n- contribute useful reasoning diversity\n"
            "- reduce redundant behavior with peer prompts\n- avoid invalid, overlong, or generic outputs\n\n"
            + ("Make one executable accuracy-oriented change grounded in observed solver rollouts. " if self._is_rollout_qd_method() else "Preserve effective mechanisms from the parent and make one local, executable reasoning change. ")
            +
            "Do not create superficial diversity by changing persona or expert-role labels.\n"
            "Do not use gold answers.\nDo not include concrete sample text.\nDo not include answer labels from examples.\n"
            "Do not write hard-coded task-specific roles.\nDo not simply ask the solver to 'think more carefully'.\n"
            f"Do not only paraphrase the parent prompt.\n\n{return_mode}"
        )
        if rollout_exploration:
            system_prompt = (
                "You directly explore solver prompts from observed rollout failures. Do not infer behavior from prompt wording. "
                "Generate a complete standalone solver prompt that may improve target accuracy, convert C2 to C3, "
                "or convert C3 to C4 by making the target correct. Do not propose named mechanisms, capability labels, personas, "
                "random disagreement, or invalid output. Do not use gold answers or concrete sample text.\n\n"
                f"{return_mode}"
            )
        if self._is_state_conditioned_method():
            system_prompt = (
                "You are the Student in an accuracy-first sequential prompt optimization system. Generate complete "
                "standalone solver prompts that implement the approved repair. Obey optimization_route: repair target "
                "errors for general_accuracy, make the target correct on assigned C0/C1 cases for coverage_repair, and "
                "make the target correct on C2/C3 cases for vote_conversion. Preserve behavior on already-correct cases "
                "and keep output valid. Do not optimize disagreement, answer-label changes, trace difference, personas, "
                "roles, prompt wording novelty, or sample-specific rules. Do not use gold answers or quote cases.\n\n"
                f"{return_mode}"
            )
        elif open_exploration:
            system_prompt = (
                "You are a direct prompt mutation generator exploring a new reasoning mechanism.\n"
                "Do not rely on a Teacher or Critic question. Produce a complete standalone solver prompt that uses "
                "a structurally different, executable decision procedure from the parent. Do not merely paraphrase, "
                "add a persona, add generic care, or copy the parent's operation order. You may reorganize the reasoning "
                "sequence, but preserve answer-format compliance and solver competence. Do not use gold answers, concrete "
                "question text, answer labels, or peer prompts. Every candidate must set candidate_type to "
                "mechanism_alternative.\n\n"
                f"{return_mode}"
            )
        if self._v7_residual_protocol_enabled():
            item_instruction += (
                " Include non-empty preserved_mechanisms, exactly one modified_mechanism, change_summary, "
                "target_residual_family, expected_shared_error_effect, and risk_control."
            )
            system_prompt += (
                "\nMake exactly one local mechanism change. Preserve the listed effective and pivotal-correct mechanisms. "
                "State how the change should reduce shared error without manufacturing disagreement."
            )
        if self._is_stable_qd_lineage() and not open_exploration:
            item_instruction += (
                " Return only task_specific_repair candidates. Each must include candidate_type, ordered "
                "mechanism_steps, target_failure_buckets, and expected_effect."
            )
            system_prompt += (
                "\nEach candidate must be a task-specific repair grounded in the supplied failure buckets."
            )
        elif self._is_stable_qd_lineage():
            item_instruction += (
                " Return only mechanism_alternative candidates. Each must include candidate_type, ordered "
                "mechanism_steps, target_failure_buckets, and expected_effect."
            )
        if self._is_stable_qd_lineage():
            lineage_context = teacher_context.get("stable_lineage_context", {})
            committed = str(lineage_context.get("lineage_status", "uncommitted")) == "committed"
            system_prompt += (
                "\nThe mechanism_alternative must differ from the parent or committed peer mechanisms in at least one core operation. "
                "It must target the same observed task errors; do not create novelty unrelated to competence. "
                + (
                    "For this committed agent, prefer a structural variant near its anchor, but still emit a genuine alternative when justified."
                    if committed
                    else "This agent is not committed: a substantial mechanism departure is allowed and must not be reduced to a local paraphrase."
                )
            )
        visible_context = teacher_context
        if rollout_exploration:
            visible_context = {
                key: teacher_context[key]
                for key in ("diagnostic_focus", "candidate_refill_feedback")
                if key in teacher_context
            }
        elif open_exploration:
            visible_context = {
                key: teacher_context[key]
                for key in (
                    "diagnostic_focus", "observed_long_term_capability_profile",
                    "stable_lineage_context", "candidate_refill_feedback",
                )
                if key in teacher_context
            }
        user_prompt = (
            "Generate up to requested_candidates candidate prompts. Return JSON with a candidates list. "
            f"{item_instruction}\n\n"
            f"{format_rules}\n\n"
            f"target_agent_id: {agent_id}\nrequested_candidates: {num_candidates}\n\n"
            f"parent_prompt:\n{parent_prompt}\n\n"
            + (
                f"approved_teacher_question:\n{json.dumps(approved_teacher_question, ensure_ascii=False, indent=2)}\n\n"
                if not open_exploration else ""
            )
            + f"abstract_generation_context:\n{json.dumps(visible_context, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "student_temperature", self.cfg.optimizer_temperature)),
            max_tokens=int(getattr(self.cfg, "student_max_tokens", self.cfg.optimizer_max_tokens)),
            stage=(f"open_rollout_exploration_agent_{agent_id}" if rollout_exploration else f"open_mechanism_exploration_agent_{agent_id}" if open_exploration else f"student_optimizer_agent_{agent_id}"),
            client_role="optimizer",
        )
        raw_text = text or ""
        raw_preview = normalize_spaces(raw_text)[:1000]
        diagnostics = self._empty_optimizer_generation_diagnostics()
        diagnostics.update(
            {
                "student_raw_response_empty": not bool(raw_text.strip()),
                "student_raw_response_preview": raw_preview,
                "student_json_parse_failed": False,
                "student_json_parse_error": "",
                "student_json_has_candidates_key": False,
                "student_candidates_is_list": False,
                "student_candidates_empty_list": False,
                "student_refusal_or_explanation": False,
                "student_failure_stage": "none",
            }
        )
        obj = extract_json_obj(raw_text)
        if diagnostics["student_raw_response_empty"] or obj is None or not isinstance(obj, dict):
            diagnostics["student_failure_stage"] = "raw_empty" if diagnostics["student_raw_response_empty"] else "json_parse_failed"
            diagnostics["student_json_parse_failed"] = not diagnostics["student_raw_response_empty"]
            diagnostics["student_refusal_or_explanation"] = self._student_refusal_or_explanation(raw_preview)
            if diagnostics["student_refusal_or_explanation"]:
                diagnostics["student_failure_stage"] = "refusal_or_explanation"

            max_retries = max(0, int(getattr(self.cfg, "student_json_max_retries", 1) or 0))
            retry_enabled = bool(int(getattr(self.cfg, "student_json_retry_on_parse_fail", True)))
            retry_text = ""
            if retry_enabled and max_retries > 0:
                diagnostics["student_json_retry_attempted"] = True
                for _ in range(max_retries):
                    retry_text = await self.retry_student_candidates_json_only(
                        previous_raw_text=raw_text,
                        approved_teacher_question=approved_teacher_question,
                        num_candidates=num_candidates,
                        agent_id=agent_id,
                    )
                    diagnostics["student_json_retry_raw_response_preview"] = normalize_spaces(retry_text or "")[:1000]
                    if retry_text and retry_text.strip():
                        diagnostics["student_raw_response_empty"] = False
                    retry_obj = extract_json_obj(retry_text or "")
                    if isinstance(retry_obj, dict):
                        obj = retry_obj
                        diagnostics["student_json_retry_succeeded"] = True
                        diagnostics["student_json_parse_failed"] = False
                        diagnostics["student_json_parse_error"] = ""
                        diagnostics["student_failure_stage"] = "none"
                        break
                if not diagnostics["student_json_retry_succeeded"]:
                    diagnostics["student_json_parse_error"] = (
                        "retry_raw_empty"
                        if diagnostics["student_failure_stage"] == "raw_empty"
                        else "retry_json_parse_failed"
                    )

            if obj is None or not isinstance(obj, dict):
                repair_enabled = bool(int(getattr(self.cfg, "student_json_repair_enabled", True)))
                repair_source = retry_text or raw_text
                if repair_enabled and bool(str(repair_source or "").strip()):
                    repair_source = retry_text or raw_text
                    repair = await self.repair_student_json_response(
                        raw_text=repair_source,
                        expected_num_candidates=num_candidates,
                    )
                    diagnostics["student_json_repair_attempted"] = True
                    diagnostics["student_json_repair_succeeded"] = bool(repair.get("repaired", False))
                    diagnostics["student_json_repair_raw_response_preview"] = str(repair.get("repair_raw_response_preview", ""))[:1000]
                    diagnostics["student_json_repair_failure_reason"] = str(repair.get("repair_failure_reason", ""))[:500]
                    repair_obj = repair.get("obj")
                    if isinstance(repair_obj, dict):
                        obj = repair_obj
                        diagnostics["student_json_parse_failed"] = False
                        diagnostics["student_json_parse_error"] = ""
                        diagnostics["student_failure_stage"] = "none"

            if obj is None or not isinstance(obj, dict):
                if diagnostics.get("student_failure_stage") != "raw_empty":
                    diagnostics["student_failure_stage"] = "json_parse_failed"
                    diagnostics["student_json_parse_failed"] = True
                if not diagnostics.get("student_json_parse_error"):
                    diagnostics["student_json_parse_error"] = (
                        str(diagnostics.get("student_json_repair_failure_reason", ""))
                        or ("raw_empty" if diagnostics["student_failure_stage"] == "raw_empty" else "json_parse_failed")
                    )
                return {"candidates": [], "diagnostics": diagnostics}

        diagnostics["student_json_has_candidates_key"] = "candidates" in obj
        if "candidates" not in obj:
            diagnostics["student_failure_stage"] = "missing_candidates_key"
            return {"candidates": [], "diagnostics": diagnostics}

        candidates = obj.get("candidates", None)
        if not isinstance(candidates, list):
            diagnostics["student_candidates_is_list"] = False
            diagnostics["student_failure_stage"] = "candidates_not_list"
            return {"candidates": [], "diagnostics": diagnostics}

        diagnostics["student_candidates_is_list"] = True
        diagnostics["student_candidates_empty_list"] = len(candidates) == 0
        if len(candidates) == 0:
            diagnostics["student_failure_stage"] = "empty_candidates_list"
        return {"candidates": candidates, "diagnostics": diagnostics}

    async def propose_candidates_teacher_critic_student(
        self,
        agent_id: int,
        parent_prompt: str,
        overlap_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batch: Optional[Dict[str, Any]] = None,
        generation_batches: Optional[List[Dict[str, Any]]] = None,
        refill_feedback: Optional[Dict[str, Any]] = None,
        generation_channel: str = "tcs_repair",
    ) -> List[Dict[str, Any]]:
        open_exploration = generation_channel in {"open_mechanism_exploration", "open_rollout_exploration"}
        rollout_exploration = generation_channel == "open_rollout_exploration"
        update_diagnosis = overlap_diagnosis
        prompt_roles = [
            r for r in update_diagnosis.get("prompt_roles", [])
            if isinstance(r, dict)
        ]
        target_role_spec = next((r for r in prompt_roles if int(r.get("agent_id", -1)) == int(agent_id)), {})
        peer_role_specs = [r for r in prompt_roles if int(r.get("agent_id", -1)) != int(agent_id)]
        if generation_batches is None:
            generation_batches = [dict(generation_batch or {"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"})]
        generation_batches = [dict(x) for x in generation_batches if isinstance(x, dict)]
        if not generation_batches:
            generation_batches = [{"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"}]

        agent_pressures = update_diagnosis.get("per_agent_overlap_pressure", [])
        agent_invalid_rates = update_diagnosis.get("per_agent_invalid_rate", [])
        agent_error_counts = update_diagnosis.get("per_agent_error_count", [])
        agent_team_wrong_counts = update_diagnosis.get("per_agent_team_wrong_error_count", [])
        agent_pivotal_fix_counts = update_diagnosis.get("per_agent_pivotal_fix_count", [])
        agent_dominant_wrong_counts = update_diagnosis.get("per_agent_dominant_wrong_redundancy_count", [])
        window_stats = {
            "diagnosis_type": update_diagnosis.get("diagnosis_type", "vote_update"),
            "window_vote_acc": update_diagnosis.get("window_vote_acc", update_diagnosis.get("team_accuracy", 0.0)),
            "window_mean_vote_margin": update_diagnosis.get("window_mean_vote_margin", -1.0),
            "window_mean_boundary_useful_diversity": update_diagnosis.get("window_mean_boundary_useful_diversity", 0.0),
            "mean_reward_pressure": update_diagnosis.get("mean_reward_pressure", 0.0),
            "mean_window_overlap": update_diagnosis.get("mean_window_overlap", 0.0),
            "homogeneity_overlap_threshold": update_diagnosis.get("homogeneity_overlap_threshold", self.cfg.homogeneity_overlap_threshold),
            "target_overlap_pressure": agent_pressures[agent_id] if agent_id < len(agent_pressures) else 0.0,
            "target_homogeneous_case_count": (update_diagnosis.get("homogeneous_case_counts", [0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("homogeneous_case_counts", [])) else 0),
            "target_invalid_rate": agent_invalid_rates[agent_id] if agent_id < len(agent_invalid_rates) else 0.0,
            "target_error_count": agent_error_counts[agent_id] if agent_id < len(agent_error_counts) else 0,
            "target_team_wrong_error_count": agent_team_wrong_counts[agent_id] if agent_id < len(agent_team_wrong_counts) else 0,
            "target_pivotal_fix_count": agent_pivotal_fix_counts[agent_id] if agent_id < len(agent_pivotal_fix_counts) else 0,
            "target_dominant_wrong_redundancy_count": agent_dominant_wrong_counts[agent_id] if agent_id < len(agent_dominant_wrong_counts) else 0,
            "target_pivotal_fix_rate": (update_diagnosis.get("per_agent_pivotal_fix_rate", [0.0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("per_agent_pivotal_fix_rate", [])) else 0.0),
            "target_near_boundary_error_rate": (update_diagnosis.get("per_agent_near_boundary_error_rate", [0.0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("per_agent_near_boundary_error_rate", [])) else 0.0),
            "target_shared_error_rate": (update_diagnosis.get("per_agent_shared_error_rate", [0.0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("per_agent_shared_error_rate", [])) else 0.0),
            "target_pivotal_hold_rate": (update_diagnosis.get("per_agent_pivotal_hold_rate", [0.0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("per_agent_pivotal_hold_rate", [])) else 0.0),
            "capability_coverage_gap": dict(update_diagnosis.get("capability_coverage_gap", {})),
            "refill_feedback": dict(refill_feedback or {}),
        }
        validity_constraints = {
            "invalid_repair_priority": bool(window_stats["target_invalid_rate"] >= float(self.cfg.invalid_repair_rate_threshold)),
            "required_final_answer_line": True,
            "avoid_empty_or_repetitive_trace": True,
            "do_not_copy_case_content": True,
        }
        teacher_context = self._build_teacher_context(
            agent_id=agent_id,
            parent_prompt=parent_prompt,
            target_role_spec=target_role_spec,
            peer_role_specs=peer_role_specs,
            window_stats=window_stats,
            validity_constraints=validity_constraints,
            generation_batches=generation_batches,
        )
        # Preserve the beam parent's stable ID for call-level provenance.
        parent_id = str((TCS_AUDIT_CONTEXT.get() or {}).get("parent_id") or self._hash(parent_prompt))
        tcs_call_group_id = (
            ""
            if open_exploration
            else str((TCS_AUDIT_CONTEXT.get() or {}).get("tcs_call_group_id") or "")
        )
        execution_session_id = str((TCS_AUDIT_CONTEXT.get() or {}).get("execution_session_id") or getattr(self, "execution_session_id", ""))
        update_attempt_id = str((TCS_AUDIT_CONTEXT.get() or {}).get("update_attempt_id") or "")
        call_context = dict(TCS_AUDIT_CONTEXT.get() or {})
        call_context.update(
            {
                "optimizer_architecture": (
                    "open_rollout_exploration" if rollout_exploration else "open_mechanism_exploration" if open_exploration else "teacher_critic_student"
                ),
                "agent_id": int(agent_id),
                "parent_id": parent_id,
                "teacher_critic_round": 1,
                "tcs_call_group_id": tcs_call_group_id,
            }
        )
        if open_exploration:
            approved = {
                "approved": True,
                "teacher_question": {},
                "critic_reviews": [],
                "teacher_critic_rounds": 0,
                "teacher_rewrite_count": 0,
                "teacher_question_forced_best_score": False,
            }
        else:
            context_token = TCS_AUDIT_CONTEXT.set(call_context)
            try:
                approved = await self.generate_approved_teacher_question(
                    agent_id=agent_id,
                    parent_prompt=parent_prompt,
                    teacher_context=teacher_context,
                    requested_candidates=num_candidates,
                )
            finally:
                TCS_AUDIT_CONTEXT.reset(context_token)
        diagnostics = self._empty_optimizer_generation_diagnostics()
        diagnostics["optimizer_architecture"] = (
            "open_rollout_exploration" if rollout_exploration else "open_mechanism_exploration" if open_exploration else "teacher_critic_student"
        )
        diagnostics["tcs_call_group_id"] = tcs_call_group_id
        diagnostics["execution_session_id"] = execution_session_id
        diagnostics["update_attempt_id"] = update_attempt_id
        teacher_question = approved.get("teacher_question", {}) if isinstance(approved, dict) else {}
        critic_reviews = approved.get("critic_reviews", []) if isinstance(approved, dict) else []
        forced_best = bool(approved.get("teacher_question_forced_best_score", False)) if isinstance(approved, dict) else False
        approved_for_student = open_exploration or bool(approved.get("approved", False)) or forced_best
        forced_best_review = approved.get("teacher_question_forced_best_review", {}) if isinstance(approved, dict) else {}
        last_review = (
            forced_best_review
            if forced_best and isinstance(forced_best_review, dict)
            else (critic_reviews[-1] if critic_reviews and isinstance(critic_reviews[-1], dict) else {})
        )
        guiding_question = (
            str(teacher_question.get("socratic_guiding_question", "")).strip()
            if isinstance(teacher_question, dict)
            else ""
        )
        teacher_question_usable = open_exploration or bool(guiding_question)
        approved_for_student = approved_for_student and teacher_question_usable
        diagnostics.update(
            {
                "teacher_question": guiding_question,
                "teacher_question_approved": bool(approved.get("approved", False)) and teacher_question_usable,
                "teacher_question_rejected": not approved_for_student,
                "teacher_question_forced_best_score": forced_best,
                "teacher_question_forced_best_round": int(approved.get("teacher_question_forced_best_round", 0) or 0),
                "teacher_question_forced_below_threshold": bool(approved.get("teacher_question_forced_below_threshold", False)),
                "teacher_question_score": self._safe_float(last_review.get("score", 0.0), 0.0),
                "teacher_critic_rounds": int(approved.get("teacher_critic_rounds", len(critic_reviews)) or 0),
                "teacher_quality_critique": str(last_review.get("quality_critique", "")),
                "teacher_specificity_critique": str(last_review.get("specificity_critique", "")),
                "teacher_task_alignment_critique": str(last_review.get("task_alignment_critique", "")),
                "teacher_error_alignment_critique": str(last_review.get("error_alignment_critique", "")),
                "teacher_diversity_critique": str(last_review.get("diversity_critique", "")),
                "teacher_rewrite_count": int(approved.get("teacher_rewrite_count", 0) or 0),
                "num_teacher_calls": 0 if open_exploration else 1,
                "num_critic_calls": int(approved.get("teacher_critic_rounds", len(critic_reviews)) or 0),
                "num_teacher_rewrite_calls": int(approved.get("teacher_rewrite_count", 0) or 0),
            }
        )
        if hasattr(self, "cost_summary"):
            self.cost_summary["tcs_teacher_calls"] = int(self.cost_summary.get("tcs_teacher_calls", 0)) + int(diagnostics["num_teacher_calls"])
            self.cost_summary["tcs_critic_calls"] = int(self.cost_summary.get("tcs_critic_calls", 0)) + int(diagnostics["num_critic_calls"])
            self.cost_summary["tcs_rewrite_calls"] = int(self.cost_summary.get("tcs_rewrite_calls", 0)) + int(diagnostics["num_teacher_rewrite_calls"])
            if self._is_stable_qd_lineage() and not open_exploration:
                saved_calls = max(0, 3 - int(diagnostics["num_critic_calls"]))
                saved_calls += max(0, 2 - int(diagnostics["num_teacher_rewrite_calls"]))
                self.cost_summary["calls_saved_by_tcs_round_reduction"] = int(
                    self.cost_summary.get("calls_saved_by_tcs_round_reduction", 0)
                ) + saved_calls
        if not approved_for_student:
            diagnostics["teacher_question_rejection_reason"] = (
                "empty_teacher_question"
                if not teacher_question_usable
                else str(
                    approved.get("teacher_question_rejection_reason", "")
                    or last_review.get("rewrite_instruction", "")
                    or last_review.get("quality_critique", "")
                    or "teacher question failed critic review"
                )
            )
            diagnostics["optimizer_underfilled"] = True
            self._record_optimizer_generation_diagnostics(agent_id, parent_prompt, diagnostics)
            self._record_generation_channel_funnel(
                agent_id=agent_id,
                parent_id=parent_id,
                channel="open_rollout_exploration" if rollout_exploration else "open_mechanism_exploration" if open_exploration else "teacher_critic_student",
                refill_round=int((refill_feedback or {}).get("refill_round", 0) or 0),
                raw_candidate_count=0,
            )
            return []

        student_context = dict(TCS_AUDIT_CONTEXT.get() or {})
        student_context.update(
            {
                "optimizer_architecture": (
                    "open_rollout_exploration" if rollout_exploration else "open_mechanism_exploration" if open_exploration else "teacher_critic_student"
                ),
                "agent_id": int(agent_id),
                "parent_id": parent_id,
                "teacher_critic_round": int(diagnostics["teacher_critic_rounds"]),
                "tcs_call_group_id": tcs_call_group_id,
            }
        )
        student_context_token = TCS_AUDIT_CONTEXT.set(student_context)
        try:
            approved_for_generation = (
                self._sanitize_v9_generation_value(approved)
                if self._is_state_conditioned_method()
                else approved
            )
            student_result = await self.generate_student_candidates(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                approved_teacher_question=approved_for_generation,
                teacher_context=teacher_context,
                num_candidates=num_candidates,
                generation_channel=generation_channel,
            )
        finally:
            TCS_AUDIT_CONTEXT.reset(student_context_token)
        if isinstance(student_result, dict):
            student_candidates = student_result.get("candidates", [])
            student_diag = student_result.get("diagnostics", {})
            if isinstance(student_diag, dict):
                self._merge_student_diagnostics(diagnostics, student_diag)
        else:
            student_candidates = student_result
        diagnostics["student_candidate_count_raw"] = len(student_candidates) if isinstance(student_candidates, list) else 0
        diagnostics["num_student_calls"] = 1
        if open_exploration:
            if hasattr(self, "cost_summary"):
                self.cost_summary["open_exploration_calls"] = int(self.cost_summary.get("open_exploration_calls", 0)) + 1
            self.open_exploration_generation_count = int(getattr(self, "open_exploration_generation_count", 0)) + 1
        else:
            if hasattr(self, "cost_summary"):
                self.cost_summary["tcs_student_calls"] = int(self.cost_summary.get("tcs_student_calls", 0)) + 1
            self.tcs_repair_generation_count = int(getattr(self, "tcs_repair_generation_count", 0)) + 1
        diagnostics["num_student_retry_calls"] = int(bool(diagnostics.get("student_json_retry_attempted", False)))
        diagnostics["num_student_repair_calls"] = int(bool(diagnostics.get("student_json_repair_attempted", False)))
        diagnostics["optimizer_raw_candidate_count"] = int(diagnostics["student_candidate_count_raw"])
        parsed: List[Dict[str, Any]] = []
        seen_signatures: set = set()
        seen_candidate_types: set = set()
        filter_reasons: List[str] = []
        if isinstance(student_candidates, list):
            for item in student_candidates:
                if not isinstance(item, dict):
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    filter_reasons.append("schema")
                    continue
                length_audit: Dict[str, Any] = {}
                if bool(getattr(self.cfg, "competence_depth_enabled", False)):
                    prepared, length_audit = self._prepare_v8_candidate_text_fields(item)
                    if prepared is None:
                        diagnostics["optimizer_schema_filtered_count"] += 1
                        filter_reasons.append("candidate_prompt_overlength")
                        continue
                    item = prepared
                else:
                    original_prompt = str(item.get("candidate_prompt", ""))
                    item = self._truncate_candidate_text_fields(item)
                    self.truncated_prompt_count = int(getattr(self, "truncated_prompt_count", 0)) + int(
                        str(item.get("candidate_prompt", "")) != normalize_spaces(original_prompt)
                    )
                if self._v7_residual_protocol_enabled():
                    if not str(item.get("modified_mechanism", "")).strip():
                        item["modified_mechanism"] = str(item.get("new_or_modified_mechanism", "")).strip()
                    if not str(item.get("target_residual_family", "")).strip():
                        item["target_residual_family"] = CapabilityResidualFamily.UNKNOWN.value
                    if not str(item.get("expected_shared_error_effect", "")).strip():
                        item["expected_shared_error_effect"] = str(item.get("error_correlation_reduction", "")).strip()
                    if not str(item.get("change_summary", "")).strip():
                        item["change_summary"] = str(item.get("modified_mechanism", "")).strip()
                if self._is_stable_qd_lineage():
                    candidate_type = str(item.get("candidate_type", "")).strip().lower()
                    expected_type = "mechanism_alternative" if open_exploration else "task_specific_repair"
                    type_rejection = (
                        "unexpected_candidate_type"
                        if candidate_type != expected_type
                        else ""
                    )
                    if type_rejection:
                        diagnostics["optimizer_schema_filtered_count"] += 1
                        filter_reasons.append(type_rejection)
                        continue
                missing_fields = self._missing_optimizer_fields(item, architecture="teacher_critic_student")
                if missing_fields:
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    if "candidate_prompt" in missing_fields:
                        diagnostics["optimizer_empty_prompt_count"] += 1
                    diagnostics["student_missing_required_field_count"] += len(missing_fields)
                    existing = list(diagnostics.get("student_missing_required_fields", []))
                    existing.extend(missing_fields)
                    diagnostics["student_missing_required_fields"] = sorted(set(str(x) for x in existing))
                    reason = "missing_required_student_fields:" + ",".join(missing_fields)
                    filter_reasons.append(reason)
                    continue
                prompt = str(item.get("candidate_prompt", "")).strip()
                prompt, sanitized = self._sanitize_prompt(prompt, agent_id)
                diagnostics["optimizer_sanitized_count"] += int(bool(sanitized))
                mechanism_signature = normalize_mechanism_signature(item.get("mechanism_steps", []))
                allow_substantive_parent_extension = bool(
                    self._is_v82_hybrid()
                    and mechanism_signature
                    and self._prompt_signature(prompt) != self._prompt_signature(parent_prompt)
                )
                if self._is_redundant_candidate_prompt(
                    parent_prompt,
                    prompt,
                    seen_signatures,
                    allow_substantive_parent_extension=allow_substantive_parent_extension,
                ):
                    diagnostics["optimizer_redundant_filtered_count"] += 1
                    filter_reasons.append("redundant")
                    continue
                if not prompt:
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    filter_reasons.append("empty_prompt")
                    continue
                mechanism_alternative_invalid = bool(
                    self._is_v82_hybrid()
                    and str(item.get("candidate_type", "")).strip().lower() == "mechanism_alternative"
                    and parsed
                    and mechanism_signature == list(parsed[0].get("mechanism_signature", []))
                )
                if mechanism_alternative_invalid:
                    diagnostics["mechanism_alternative_invalid"] = True
                    diagnostics["mechanism_alternative_invalid_count"] = int(
                        diagnostics.get("mechanism_alternative_invalid_count", 0) or 0
                    ) + 1
                    filter_reasons.append("mechanism_alternative_same_signature")
                    continue
                seen_signatures.add(self._prompt_signature(prompt))
                if self._is_v82_hybrid():
                    seen_candidate_types.add(str(item.get("candidate_type", "")).strip().lower())
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "student_interpretation_of_question": str(item.get("student_interpretation_of_question", "")),
                        "target_error_pattern": str(item.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(item.get("accuracy_repair_rule", "")),
                        "diversity_contribution": str(item.get("diversity_contribution", "")),
                        "error_correlation_reduction": str(item.get("error_correlation_reduction", "")),
                        "task_alignment_rule": str(item.get("task_alignment_rule", "")),
                        "peer_redundancy_avoidance": str(item.get("peer_redundancy_avoidance", "")),
                        "expected_accuracy_effect": str(item.get("expected_accuracy_effect", "")),
                        "rollout_diversity_intent": str(item.get("rollout_diversity_intent", "")),
                        "expected_diversity_effect": str(item.get("expected_diversity_effect", "")),
                        "risk_control": str(item.get("risk_control", "")),
                        "rationale": str(item.get("rationale", "")),
                        "change_summary": str(item.get("change_summary", "")),
                        "preserved_mechanisms": list(item.get("preserved_mechanisms", [])) if isinstance(item.get("preserved_mechanisms", []), list) else [],
                        "new_or_modified_mechanism": str(item.get("new_or_modified_mechanism", "")),
                        "modified_mechanism": str(item.get("modified_mechanism", item.get("new_or_modified_mechanism", ""))),
                        "target_residual_family": str(item.get("target_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                        "expected_shared_error_effect": str(item.get("expected_shared_error_effect", "")),
                        "candidate_type": str(item.get("candidate_type", "")),
                        "mechanism_steps": [str(value) for value in item.get("mechanism_steps", [])] if isinstance(item.get("mechanism_steps", []), list) else [],
                        "target_failure_buckets": [str(value) for value in item.get("target_failure_buckets", [])] if isinstance(item.get("target_failure_buckets", []), list) else [],
                        "expected_effect": str(item.get("expected_effect", "")),
                        "mechanism_signature": mechanism_signature,
                        "mechanism_alternative_invalid": mechanism_alternative_invalid,
                        **length_audit,
                        "candidate_source": (
                            "open_rollout_exploration" if rollout_exploration else "open_mechanism_exploration" if open_exploration else "teacher_critic_student"
                        ),
                        "tcs_call_group_id": tcs_call_group_id,
                        "execution_session_id": execution_session_id,
                        "update_attempt_id": update_attempt_id,
                        "teacher_question": teacher_question,
                        "teacher_question_score": self._safe_float(diagnostics.get("teacher_question_score", 0.0), 0.0),
                        "teacher_question_approved": bool(diagnostics.get("teacher_question_approved", False)),
                        "teacher_critic_rounds": int(diagnostics["teacher_critic_rounds"]),
                        "critic_reviews": critic_reviews,
                        "generation_batch_type": str(generation_batches[batch_idx].get("batch_type", "")),
                        "optimization_route": str(
                            generation_batches[batch_idx].get("optimization_route", "general_accuracy")
                            or "general_accuracy"
                        ),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[batch_idx].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
                if len(parsed) >= num_candidates:
                    break
        if self._is_v82_hybrid():
            type_order = {"task_specific_repair": 0, "mechanism_alternative": 1}
            parsed.sort(key=lambda item: type_order.get(str(item.get("candidate_type", "")), 99))
        diagnostics["student_candidate_count_final"] = len(parsed)
        diagnostics["student_candidate_filtered_count"] = max(0, int(diagnostics["student_candidate_count_raw"]) - len(parsed))
        diagnostics["student_candidate_filter_reasons"] = filter_reasons
        diagnostics["student_all_candidates_filtered"] = bool(int(diagnostics["student_candidate_count_raw"]) > 0 and not parsed)
        if diagnostics["student_all_candidates_filtered"]:
            has_schema = any(
                "missing_required" in str(reason)
                or "schema" in str(reason)
                or "empty_prompt" in str(reason)
                for reason in filter_reasons
            )
            has_redundant = any("redundant" in str(reason) for reason in filter_reasons)
            if has_schema and has_redundant:
                diagnostics["student_failure_stage"] = "all_candidates_filtered_mixed"
            elif has_schema:
                diagnostics["student_failure_stage"] = "all_candidates_filtered_schema"
            elif has_redundant:
                diagnostics["student_failure_stage"] = "all_candidates_filtered_redundant"
            else:
                diagnostics["student_failure_stage"] = "unknown"
        diagnostics["optimizer_final_candidate_count"] = len(parsed)
        diagnostics["optimizer_underfilled"] = bool(len(parsed) < int(num_candidates))
        if diagnostics.get("teacher_question_approved"):
            diagnostics["teacher_question_forced_best_score"] = False
            diagnostics["teacher_question_forced_best_round"] = 0
        if diagnostics.get("teacher_question_forced_best_score"):
            diagnostics["teacher_question_approved"] = False
        diagnostics = self._record_optimizer_generation_diagnostics(agent_id, parent_prompt, diagnostics)
        metadata = self._teacher_metadata_from_diagnostics(diagnostics)
        for item in parsed:
            item["optimizer_generation_diagnostics"] = dict(diagnostics)
            item.update(metadata)
            item["optimizer_architecture"] = diagnostics["optimizer_architecture"]
        if open_exploration:
            self.open_exploration_candidate_count = int(getattr(self, "open_exploration_candidate_count", 0)) + len(parsed)
        else:
            self.tcs_repair_candidate_count = int(getattr(self, "tcs_repair_candidate_count", 0)) + len(parsed)
        self._record_generation_channel_funnel(
            agent_id=agent_id,
            parent_id=parent_id,
            channel="open_rollout_exploration" if rollout_exploration else "open_mechanism_exploration" if open_exploration else "teacher_critic_student",
            refill_round=int((refill_feedback or {}).get("refill_round", 0) or 0),
            raw_candidate_count=int(diagnostics.get("optimizer_raw_candidate_count", 0) or 0),
        )
        return parsed[:num_candidates]

    def _tcs_diagnostic_support_count(
        self,
        agent_id: int,
        diagnosis: Dict[str, Any],
        generation_batches: Optional[List[Dict[str, Any]]],
    ) -> int:
        support = 0
        for field in (
            "per_agent_error_count", "per_agent_team_wrong_error_count",
            "per_agent_pivotal_fix_count", "per_agent_dominant_wrong_redundancy_count",
        ):
            values = diagnosis.get(field, [])
            if isinstance(values, list) and agent_id < len(values) and float(values[agent_id] or 0) > 0:
                support += 1
        invalid_rates = diagnosis.get("per_agent_invalid_rate", [])
        if isinstance(invalid_rates, list) and agent_id < len(invalid_rates) and float(invalid_rates[agent_id] or 0) > 0:
            support += 1
        supported_batch_types = {
            "target_error_repair", "c1_c2_creation", "actual_plurality_boundary",
            "residual_shared_error", "invalid_output_repair",
        }
        support += sum(
            1 for batch in (generation_batches or [])
            if isinstance(batch, dict)
            and str(batch.get("batch_type", "")) in supported_batch_types
            and bool(batch.get("cases", []))
        )
        return support

    @staticmethod
    def _merge_generation_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        additive = {
            "num_teacher_calls", "num_critic_calls", "num_teacher_rewrite_calls",
            "num_student_calls", "num_student_retry_calls", "num_student_repair_calls",
            "optimizer_raw_candidate_count", "optimizer_final_candidate_count",
            "student_candidate_count_raw", "student_candidate_count_final",
        }
        for record in records:
            for key, value in record.items():
                if key in additive:
                    merged[key] = int(merged.get(key, 0) or 0) + int(value or 0)
                elif key not in merged or (not merged[key] and value):
                    merged[key] = value
        return merged

    async def propose_candidates(
        self,
        agent_id: int,
        parent_prompt: str,
        overlap_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batch: Optional[Dict[str, Any]] = None,
        generation_batches: Optional[List[Dict[str, Any]]] = None,
        refill_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        architecture = str(getattr(self.cfg, "optimizer_architecture", "teacher_critic_student") or "teacher_critic_student").lower()
        if architecture == "teacher_critic_student":
            if self._is_stable_qd_lineage() or self._is_rollout_qd_method():
                forced_generator = str((refill_feedback or {}).get("refill_generator_type", ""))
                support_count = self._tcs_diagnostic_support_count(
                    agent_id, overlap_diagnosis, generation_batches,
                )
                if forced_generator == "tcs_repair":
                    repair_count, open_count = num_candidates, 0
                elif forced_generator in {"open_mechanism_exploration", "open_rollout_exploration"}:
                    repair_count, open_count = 0, num_candidates
                elif support_count > 0:
                    repair_count = min(num_candidates, int(self.cfg.tcs_repair_candidates_per_parent))
                    open_count = max(0, num_candidates - repair_count)
                else:
                    repair_count, open_count = 0, num_candidates
                outputs: List[Dict[str, Any]] = []
                diagnostics: List[Dict[str, Any]] = []
                if repair_count:
                    repair = await self.propose_candidates_teacher_critic_student(
                        agent_id=agent_id, parent_prompt=parent_prompt,
                        overlap_diagnosis=overlap_diagnosis, num_candidates=repair_count,
                        generation_batch=generation_batch, generation_batches=generation_batches,
                        refill_feedback=refill_feedback, generation_channel="tcs_repair",
                    )
                    outputs.extend(repair)
                    diagnostics.append(self._optimizer_generation_diagnostics_for_parent(agent_id, parent_prompt))
                if open_count:
                    opened = await self.propose_candidates_teacher_critic_student(
                        agent_id=agent_id, parent_prompt=parent_prompt,
                        overlap_diagnosis=overlap_diagnosis, num_candidates=open_count,
                        generation_batch=generation_batch, generation_batches=generation_batches,
                        refill_feedback=refill_feedback,
                        generation_channel="open_rollout_exploration" if self._is_rollout_qd_method() else "open_mechanism_exploration",
                    )
                    outputs.extend(opened)
                    diagnostics.append(self._optimizer_generation_diagnostics_for_parent(agent_id, parent_prompt))
                merged = self._merge_generation_diagnostics(diagnostics)
                merged.update({
                    "tcs_repair_triggered": bool(repair_count),
                    "tcs_repair_skip_reason": "" if repair_count else "no_diagnostic_support",
                    "diagnostic_support_count": int(support_count),
                })
                self._record_optimizer_generation_diagnostics(agent_id, parent_prompt, merged)
                # Keep channel-specific provenance on each candidate. The merged
                # record is only an update-level summary; copying it back would
                # mix TCS and open-branch flags for the same parent.
                return outputs[:num_candidates]
            return await self.propose_candidates_teacher_critic_student(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                overlap_diagnosis=overlap_diagnosis,
                num_candidates=num_candidates,
                generation_batch=generation_batch,
                generation_batches=generation_batches,
                refill_feedback=refill_feedback,
            )
        return await self.propose_candidates_one_shot(
            agent_id=agent_id,
            parent_prompt=parent_prompt,
            overlap_diagnosis=overlap_diagnosis,
            num_candidates=num_candidates,
            generation_batch=generation_batch,
            generation_batches=generation_batches,
        )

    async def propose_candidates_one_shot(
        self,
        agent_id: int,
        parent_prompt: str,
        overlap_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batch: Optional[Dict[str, Any]] = None,
        generation_batches: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        update_diagnosis = overlap_diagnosis
        prompt_roles = [
            r for r in update_diagnosis.get("prompt_roles", [])
            if isinstance(r, dict)
        ]
        target_role_spec = next((r for r in prompt_roles if int(r.get("agent_id", -1)) == int(agent_id)), {})
        peer_role_specs = [r for r in prompt_roles if int(r.get("agent_id", -1)) != int(agent_id)]
        if generation_batches is None:
            generation_batches = [dict(generation_batch or {"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"})]
        generation_batches = [dict(x) for x in generation_batches if isinstance(x, dict)]
        if not generation_batches:
            generation_batches = [{"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"}]
        if self._is_accuracy_only_mode():
            return await self._propose_accuracy_candidates(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                accuracy_diagnosis=update_diagnosis,
                num_candidates=num_candidates,
                generation_batches=generation_batches,
            )
        agent_pressures = update_diagnosis.get("per_agent_overlap_pressure", [])
        agent_invalid_rates = update_diagnosis.get("per_agent_invalid_rate", [])
        agent_error_counts = update_diagnosis.get("per_agent_error_count", [])
        agent_team_wrong_counts = update_diagnosis.get("per_agent_team_wrong_error_count", [])
        agent_pivotal_fix_counts = update_diagnosis.get("per_agent_pivotal_fix_count", [])
        agent_dominant_wrong_counts = update_diagnosis.get("per_agent_dominant_wrong_redundancy_count", [])
        window_stats = {
            "diagnosis_type": update_diagnosis.get("diagnosis_type", "vote_update"),
            "window_vote_acc": update_diagnosis.get("window_vote_acc", update_diagnosis.get("team_accuracy", 0.0)),
            "window_mean_vote_margin": update_diagnosis.get("window_mean_vote_margin", -1.0),
            "window_mean_boundary_useful_diversity": update_diagnosis.get("window_mean_boundary_useful_diversity", 0.0),
            "mean_reward_pressure": update_diagnosis.get("mean_reward_pressure", 0.0),
            "mean_window_overlap": update_diagnosis.get("mean_window_overlap", 0.0),
            "homogeneity_overlap_threshold": update_diagnosis.get("homogeneity_overlap_threshold", self.cfg.homogeneity_overlap_threshold),
            "target_overlap_pressure": agent_pressures[agent_id] if agent_id < len(agent_pressures) else 0.0,
            "target_homogeneous_case_count": (update_diagnosis.get("homogeneous_case_counts", [0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("homogeneous_case_counts", [])) else 0),
            "target_invalid_rate": agent_invalid_rates[agent_id] if agent_id < len(agent_invalid_rates) else 0.0,
            "target_error_count": agent_error_counts[agent_id] if agent_id < len(agent_error_counts) else 0,
            "target_team_wrong_error_count": agent_team_wrong_counts[agent_id] if agent_id < len(agent_team_wrong_counts) else 0,
            "target_pivotal_fix_count": agent_pivotal_fix_counts[agent_id] if agent_id < len(agent_pivotal_fix_counts) else 0,
            "target_dominant_wrong_redundancy_count": agent_dominant_wrong_counts[agent_id] if agent_id < len(agent_dominant_wrong_counts) else 0,
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
        has_target_error_batches = any(str(b.get("batch_type", "")) == "target_error_repair" and b.get("cases") for b in generation_batches)
        if has_target_error_batches:
            system_prompt = (
                "You are a prompt optimizer for a multi-agent reasoning team.\n"
                "Generate executable role prompts that improve the target agent's answer accuracy on its observed error patterns while preserving useful reasoning diversity.\n"
                "Diversity is valuable only when it creates valid, answer-improving reasoning behavior, not superficial wording differences.\n"
                "Use the supplied target-error cases, peer behavior summaries, parent prompt, role previews, and batch diagnoses.\n"
                "Do not use gold answers, concrete task text, answer labels, or sample-specific content.\n"
                "Treat trace previews as abstract behavioral evidence; do not copy their wording into the new prompt.\n"
                "Optimize for behavior that will be visible in the solver trace and easy to evaluate for role execution.\n"
                "Return strict JSON only."
            )
        else:
            system_prompt = (
                "You are a prompt optimizer for a multi-agent reasoning team.\n"
                "Generate executable role prompts that preserve answer reliability while adding useful reasoning diversity.\n"
                "Use only the supplied parent prompt, prompt-role previews, window statistics, and generation-batch diagnoses.\n"
                "The homogeneous cases were selected by the system, not by you. You are only a candidate prompt proposer.\n"
                "Do not use gold answers, concrete task text, options, labels, or answer-specific content.\n"
                "Treat trace previews as abstract behavioral evidence; do not copy their wording into the new prompt.\n"
                "Optimize for behavior that will be visible in the solver trace and easy to evaluate for role execution.\n"
                "Return strict JSON only."
            )
        user_prompt = (
            "Revise the target agent prompt using the case-aware generation batches below.\n"
            "Priority order:\n"
            "1. Repair the target agent's observed error patterns.\n"
            "2. Preserve or improve target-agent answer accuracy.\n"
            "3. Add useful reasoning diversity only when it helps correctness or error rescue.\n"
            "4. Avoid invalid, verbose, generic, or merely paraphrased prompts.\n"
            "5. Do not optimize for trace difference alone.\n"
            "Each candidate must primarily address one supplied generation batch; do not merge all batches into one generic prompt.\n"
            "The new prompt must address the provided cases as reasoning-pattern evidence, not as sample content to memorize.\n"
            "Write concrete reasoning behavior, not slogans such as 'be diverse' or 'avoid redundancy'.\n"
            "A candidate is invalid if it only paraphrases the parent prompt, appends generic caution, asks the solver to be more accurate, "
            "or changes style without adding a concrete error-repair procedure.\n"
            "Each candidate_prompt must contain a concrete reasoning procedure, a specific error-repair behavior, final answer discipline, "
            "and a short verification step.\n"
            "Write a complete short role prompt, not a suffix to append to the parent prompt. "
            "Do not repeat generic instructions already present in the parent prompt. "
            "Do not use the phrase 'Use a distinct decision procedure'. "
            "Prefer a short role prompt with 2-4 explicit procedure steps, a fallback strategy, and validity checks.\n"
            "The prompt should create a different reasoning route from peer roles only when that helps correctness or error rescue.\n"
            "Never include concrete question text, answer text, options, labels, sample hashes, or FINAL_ANSWER templates.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"candidate_prompt": str, "role_name": str, "decision_procedure": [str, ...], "when_to_use": str, "fallback_strategy": str, "anti_overlap_rule": str, "validity_checks": [str, ...], "target_error_pattern": str, "accuracy_repair_rule": str, "expected_accuracy_effect": str, "rationale": str, "source_batch_type": str},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Return exactly requested_candidates distinct candidates. "
            "Set source_batch_type to the exact batch_type that the candidate primarily addresses. "
            "target_error_pattern names the main observed pattern repaired by the candidate. "
            "accuracy_repair_rule is the concrete behavior enforced to improve target-agent correctness. "
            "expected_accuracy_effect explains why this improves the target agent rather than merely changing wording. "
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
        if self._v7_residual_protocol_enabled():
            system_prompt = (
                system_prompt.replace("role prompts", "solver instructions")
                .replace("prompt-role previews", "prompt summaries")
                .replace("role previews", "prompt summaries")
                .replace("role execution", "mechanism execution")
            )
            user_prompt = (
                user_prompt.replace("role_name", "mechanism_name")
                .replace("target_role_spec", "target_prompt_state")
                .replace("peer_role_specs", "peer_prompt_summaries")
                .replace("complete short role prompt", "complete short solver instruction")
                .replace("short role prompt", "short solver instruction")
                .replace("peer roles", "peer prompts")
                .replace("role prompts", "solver instructions")
                .replace("prompt-role previews", "prompt summaries")
            )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.cfg.optimizer_temperature),
            max_tokens=int(self.cfg.optimizer_max_tokens),
            stage=f"optimizer_agent_{agent_id}",
        )
        diagnostics = self._empty_optimizer_generation_diagnostics()
        diagnostics["optimizer_raw_response_empty"] = int(not str(text or "").strip())
        obj = extract_json_obj(text)
        diagnostics["optimizer_json_parse_failed"] = int(bool(str(text or "").strip()) and obj is None)
        if obj is None:
            obj = {}
        candidates = obj.get("candidates", []) if isinstance(obj, dict) else []
        if isinstance(candidates, list):
            diagnostics["optimizer_raw_candidate_count"] = len(candidates)
        else:
            diagnostics["optimizer_schema_filtered_count"] += 1
            candidates = []
        parsed: List[Dict[str, Any]] = []
        seen_signatures: set = set()
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    continue
                if not self._candidate_has_required_optimizer_fields(item, architecture="one_shot"):
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    continue
                prompt = str(item.get("candidate_prompt", "")).strip()
                prompt, sanitized = self._sanitize_prompt(prompt, agent_id)
                diagnostics["optimizer_sanitized_count"] += int(bool(sanitized))
                if self._is_redundant_candidate_prompt(parent_prompt, prompt, seen_signatures):
                    diagnostics["optimizer_redundant_filtered_count"] += 1
                    continue
                if not prompt:
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    continue
                seen_signatures.add(self._prompt_signature(prompt))
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(item.get("role_name", "")),
                        "mechanism_name": str(item.get("mechanism_name", item.get("role_name", ""))),
                        "decision_procedure": item.get("decision_procedure", []),
                        "when_to_use": str(item.get("when_to_use", "")),
                        "fallback_strategy": str(item.get("fallback_strategy", "")),
                        "anti_overlap_rule": str(item.get("anti_overlap_rule", "")),
                        "validity_checks": item.get("validity_checks", []),
                        "target_error_pattern": str(item.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(item.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(item.get("expected_accuracy_effect", "")),
                        "rationale": str(item.get("rationale", "")),
                        "candidate_source": "optimizer",
                        "optimizer_generation_diagnostics": dict(diagnostics),
                        "generation_batch_type": str(item.get("source_batch_type", "")) or str(safe_generation_batches[min(len(parsed), len(safe_generation_batches) - 1)].get("batch_type", "")),
                        "optimization_route": str(
                            safe_generation_batches[min(len(parsed), len(safe_generation_batches) - 1)].get(
                                "optimization_route", "general_accuracy"
                            ) or "general_accuracy"
                        ),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[min(len(parsed), len(generation_batches) - 1)].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
                if len(parsed) >= num_candidates:
                    break
        fallback_mode_cfg = str(getattr(self.cfg, "optimizer_fallback_mode", "none") or "none").lower()
        if fallback_mode_cfg == "template":
            while len(parsed) < num_candidates:
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                fallback_mode = "accuracy_repair" if has_target_error_batches else "diversity"
                fallback = self._structured_fallback_role(agent_id, len(parsed), mode=fallback_mode)
                prompt = str(fallback["candidate_prompt"])
                seen_signatures.add(self._prompt_signature(prompt))
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(fallback.get("role_name", fallback.get("mechanism_name", ""))),
                        "mechanism_name": str(fallback.get("mechanism_name", fallback.get("role_name", ""))),
                        "decision_procedure": list(fallback["decision_procedure"]),
                        "when_to_use": str(fallback["when_to_use"]),
                        "fallback_strategy": str(fallback["fallback_strategy"]),
                        "anti_overlap_rule": str(fallback["anti_overlap_rule"]),
                        "validity_checks": list(fallback["validity_checks"]),
                        "target_error_pattern": str(fallback.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(fallback.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(fallback.get("expected_accuracy_effect", "")),
                        "rationale": "Fallback candidate when optimizer returns too few usable prompts.",
                        "candidate_source": f"{fallback_mode}_fallback",
                        "optimizer_generation_diagnostics": dict(diagnostics),
                        "generation_batch_type": str(generation_batches[batch_idx].get("batch_type", "")),
                        "optimization_route": str(
                            generation_batches[batch_idx].get("optimization_route", "general_accuracy")
                            or "general_accuracy"
                        ),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[batch_idx].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
        diagnostics["optimizer_final_candidate_count"] = sum(1 for item in parsed[:num_candidates] if str(item.get("candidate_source", "")) == "optimizer")
        diagnostics["optimizer_underfilled"] = bool(diagnostics["optimizer_final_candidate_count"] < int(num_candidates))
        diagnostics = self._record_optimizer_generation_diagnostics(agent_id, parent_prompt, diagnostics)
        for item in parsed:
            item["optimizer_generation_diagnostics"] = dict(diagnostics)
        return parsed[:num_candidates]
