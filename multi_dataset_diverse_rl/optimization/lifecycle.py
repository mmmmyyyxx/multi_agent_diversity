"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class LifecycleMixin:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.beam_refresh_each_epoch = bool(int(self.cfg.beam_refresh_each_epoch))
        self.cfg.transient_retry_forever = bool(int(self.cfg.transient_retry_forever))
        self.cfg.llm_call_logging = bool(int(self.cfg.llm_call_logging))
        self.cfg.invalid_binary = bool(int(self.cfg.invalid_binary))
        self.cfg.use_joint_trace_diversity_evaluator = bool(int(self.cfg.use_joint_trace_diversity_evaluator))
        self.cfg.candidate_reuse_recorded_rollouts = bool(int(getattr(self.cfg, "candidate_reuse_recorded_rollouts", 1)))
        self.cfg.solver_rollout_singleflight = bool(int(getattr(self.cfg, "solver_rollout_singleflight", 1)))
        self.cfg.candidate_eval_prompt_dedup = bool(int(getattr(self.cfg, "candidate_eval_prompt_dedup", 1)))
        self.cfg.candidate_eval_cache_logging = bool(int(getattr(self.cfg, "candidate_eval_cache_logging", 1)))
        self.cfg.use_baseline_relative_reward = bool(int(getattr(self.cfg, "use_baseline_relative_reward", 1)))
        for name in (
            "boundary_selector_enabled",
            "shared_error_metrics_enabled",
            "residual_specialization_enabled",
            "error_dependence_guard_enabled",
            "residual_cycle_guard_enabled",
            "mechanism_trust_region_enabled",
            "competence_depth_enabled",
            "competence_depth2_aux_enabled",
            "competence_progressive_residual_enabled",
            "competence_schedule_monotonic",
            "competence_depth1_candidate_guard_enabled",
        ):
            setattr(self.cfg, name, bool(int(getattr(self.cfg, name, 0))))
        self.cfg.behavior_cycle_guard_enabled = bool(int(getattr(self.cfg, "behavior_cycle_guard_enabled", 1)))
        self.cfg.prompt_trust_region_enabled = bool(int(getattr(self.cfg, "prompt_trust_region_enabled", 1)))
        self.task_spec = self._build_task_spec()

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
        from ..persistence.artifacts import ArtifactWriter
        self.artifact_writer = ArtifactWriter(self.cfg.out_dir)
        self.previous_execution_session_id = self._read_previous_execution_session_id()
        # A resumed process is a distinct provenance session, even for a repeated step.
        self.execution_session_id = uuid.uuid4().hex[:12]
        open(os.path.join(self.cfg.out_dir, "llm_calls.jsonl"), "a", encoding="utf-8").close()
        set_seed(int(self.cfg.seed))

        self.initial_prompt_bank = self._default_prompt_bank()
        self.initial_agent_prompts = self._build_initial_prompts()
        self.initial_agent_prompt_hashes = [self._hash(p) for p in self.initial_agent_prompts]
        self.agents = [AgentState(p, homogeneity_window=self.homogeneity_window) for p in self.initial_agent_prompts]
        self._initialize_prompt_beams()

        self.history: List[Dict[str, Any]] = []
        self.update_logs: List[Dict[str, Any]] = []
        self.trajectory_events: List[Dict[str, Any]] = []
        self.train_step_logs: List[Dict[str, Any]] = []
        self.train_trace_history_logs: List[Dict[str, Any]] = []
        self.test_trace_history_logs: List[Dict[str, Any]] = []
        self.recent_window_records: List[Dict[str, Any]] = []
        self.prompt_history = self._init_prompt_history()
        self.joint_diversity_cache: Dict[str, Dict[str, Any]] = {}
        self.solver_rollout_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.solver_rollout_inflight: Dict[str, asyncio.Future] = {}
        self.solver_rollout_inflight_lock = asyncio.Lock()
        self.optimizer_generation_diagnostics: Dict[str, Dict[str, Any]] = {}
        self.no_effective_evolution_counter = 0
        self.no_effective_evolution_stopped = False
        self.no_effective_evolution_reason = ""
        self.specialization_strength = 0.0
        self.effective_residual_strength = 0.0
        self.previous_epoch_per_agent_acc: List[float] = []
        self.previous_epoch_bottom2_mean_acc = 0.0
        self.competence_phase_epoch = 1
        self.competence_schedule_version = str(getattr(self.cfg, "competence_schedule_version", "competence_depth_v1"))
        self.specialization_strength_history: List[float] = []
        self.competence_probe_indices: List[int] = []
        self.competence_probe_question_hashes: List[str] = []
        self.initial_competence_probe_metrics: Dict[str, Any] = {}
        self.latest_competence_probe_metrics: Dict[str, Any] = {}
        self.competence_probe_history: List[Dict[str, Any]] = []
        self.initial_active_prompt_hashes: List[str] = list(self.initial_agent_prompt_hashes)
        self.first_nonzero_specialization_epoch: Optional[int] = None
        self.effective_specialization_epoch_count = 0
        self.depth1_guard_rejection_count = 0
        self.catastrophic_accuracy_guard_rejection_count = 0
        self.soft_error_dependence_penalty_count = 0
        self.soft_cycle_penalty_count = 0
        self.soft_mechanism_shift_penalty_count = 0
        self.exploration_candidate_count = 0
        self.exploration_slot_occupancy_count = 0
        self.exploration_to_active_conversion_count = 0
        self.hybrid_selector_history: List[Dict[str, Any]] = []
        self.mechanism_signature_history: List[Dict[str, Any]] = []
        self.mechanism_signature_by_prompt_hash: Dict[str, List[str]] = {}
        self.beam_slot_state: Dict[str, Any] = {}
        self.exploration_slot_candidates: List[Dict[str, Any]] = []
        self.mechanism_embedding_cache: Dict[str, List[float]] = {}
        self.semantic_mechanism_families: Dict[str, Dict[str, Any]] = {}
        self.prompt_probe_cache: Dict[str, Dict[str, Any]] = {}
        self.mechanism_embedding_cache_hit_count = 0
        self.mechanism_embedding_cache_miss_count = 0
        self.full_probe_cache_hit_count = 0
        self.full_probe_missing_pair_evaluation_count = 0
        self.behavior_profile_by_prompt_hash: Dict[str, Dict[str, Any]] = {}
        self.joint_team_selection_history: List[Dict[str, Any]] = []
        self.lineage_history: List[Dict[str, Any]] = []
        self.quality_diversity_archive_history: List[Dict[str, Any]] = []
        self.behavior_profile_history: List[Dict[str, Any]] = []
        self.total_agent_update_count = 0
        self.task_repair_niche_occupancy_count = 0
        self.mechanism_niche_occupancy_count = 0
        self.peer_collapse_soft_count = 0
        self.peer_collapse_hard_rejection_count = 0
        self.latest_joint_team_metrics: Dict[str, Any] = {}
        self.joint_quality_anchor_metrics: Dict[str, Any] = {}
        self.quality_anchor_archive: List[Dict[str, Any]] = []
        self.quality_anchor_created_count = 0
        self.qd_no_diversification_epochs = 0
        self.qd_change_limit_relaxed_epoch = -1
        self.qd_previous_active_niche_count = 0
        self.probation_to_safe_conversion_count = 0
        self.probation_expired_count = 0
        self.candidate_starvation_count = 0
        self.mechanism_starvation_count = 0
        self.search_branch_starvation_count = 0
        self.refill_requirements_unmet_count = 0
        self.per_agent_optimizer_update_count: Dict[str, int] = {}
        self.prompt_overlength_rejection_count = 0
        self.truncated_prompt_count = 0
        self.llm_call_logs: List[Dict[str, Any]] = []
        self.cost_summary: Dict[str, Any] = self._empty_cost_summary()
        self.embedding_model = None
        self.embedding_cache: Dict[str, List[float]] = {}
        self.solver_call_limit = max(1, int(getattr(self.cfg, "eval_solver_call_concurrency", 225) or 225))
        self.solver_call_semaphore = asyncio.Semaphore(self.solver_call_limit)

        self._load_recorded_solver_rollouts()
        self.write_run_meta()
        resume_existing = bool(int(getattr(self.cfg, "resume_from_checkpoint", False) or False))
        if not (resume_existing and os.path.exists(os.path.join(self.cfg.out_dir, "prompt_history.json"))):
            self.flush_prompt_history()
        if not (resume_existing and os.path.exists(os.path.join(self.cfg.out_dir, "cost_summary.json"))):
            self.write_cost_summary()

    def _build_task_spec(self) -> TaskSpec:
        answer_format = str(getattr(self.cfg, "answer_format", "") or "").strip()
        if not answer_format:
            return get_task_spec(self.cfg.task_type)
        return TaskSpec(
            name=f"{self.cfg.task_type}:{answer_format}",
            parse_gold=lambda answer, question=None: canonical_answer_format(answer, answer_format),
            extract_pred=lambda text, question=None: extract_prediction_format(text, answer_format),
            match_answer=lambda pred, gold: match_answer_format(pred, gold, answer_format),
        )

    def _is_accuracy_only_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "accuracy_only"

    def _is_guarded_reward_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "guarded_diversity"

    def _is_vote_useful_diversity_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "vote_useful_diversity"

    def _is_coverage_useful_diversity_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "coverage_useful_diversity"

    def _is_competence_depth_reward_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "competence_depth_schedule"

    def _uses_competence_depth_pareto_selection(self) -> bool:
        return str(getattr(self.cfg, "candidate_selection_mode", "")).lower() == "competence_depth_pareto"

    def _is_v82_hybrid(self) -> bool:
        return str(getattr(self.cfg, "method_version", "legacy")) in {"v8_2_hybrid_progressive", "v8_stable_qd_lineage"}

    def _is_stable_qd_lineage(self) -> bool:
        return str(getattr(self.cfg, "method_version", "legacy")) == "v8_stable_qd_lineage"

    def _apply_competence_depth1_candidate_guard(self, metrics: Dict[str, Any]) -> bool:
        enabled = bool(getattr(self.cfg, "competence_depth1_candidate_guard_enabled", False))
        epsilon = float(getattr(self.cfg, "competence_depth1_candidate_guard_epsilon", 0.0) or 0.0)
        passed = (not enabled) or float(metrics.get("depth1_net_delta", 0.0) or 0.0) >= -epsilon
        metrics.update({
            "competence_depth1_guard_enabled": enabled,
            "competence_depth1_guard_epsilon": epsilon,
            "competence_depth1_guard_passed": bool(passed),
        })
        if not passed and not str(metrics.get("rejection_reason", "")):
            metrics["rejection_reason"] = "competence_depth1_guard"
        return bool(passed)

    def complete_competence_epoch(
        self,
        per_agent_acc: Optional[Sequence[float]] = None,
        epoch: int = 0,
        *,
        snapshot_metrics: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Advance competence scheduling; v2 accepts only a static probe snapshot."""
        if not bool(getattr(self.cfg, "competence_depth_enabled", False)):
            return float(self.specialization_strength)
        mode = str(getattr(self.cfg, "competence_schedule_mode", "absolute_legacy") or "absolute_legacy")
        if mode == "baseline_relative_opt_snapshot":
            if not isinstance(snapshot_metrics, dict):
                raise ValueError("baseline_relative_opt_snapshot requires snapshot_metrics from the optimization probe")
            if not self.initial_competence_probe_metrics:
                raise ValueError("initial_competence_probe_metrics is required before advancing the v2 schedule")
            record = competence_relative_specialization_strength(
                initial_metrics=self.initial_competence_probe_metrics,
                snapshot_metrics=snapshot_metrics,
                probe_size=int(snapshot_metrics.get("probe_size", len(self.competence_probe_indices)) or len(self.competence_probe_indices)),
                current_strength=float(self.specialization_strength),
                low_delta=float(getattr(self.cfg, "competence_relative_low_delta", 0.01)),
                high_delta=float(getattr(self.cfg, "competence_relative_high_delta", 0.06)),
                ema=float(getattr(self.cfg, "competence_schedule_ema", 0.50)),
                max_step=float(getattr(self.cfg, "competence_schedule_max_step", 0.35)),
                monotonic=bool(getattr(self.cfg, "competence_schedule_monotonic", True)),
                mean_guard_epsilon=float(getattr(self.cfg, "competence_mean_guard_epsilon", 0.01)),
                c1_guard_epsilon=float(getattr(self.cfg, "competence_c1_guard_epsilon", 0.01)),
                c2_guard_epsilon=float(getattr(self.cfg, "competence_c2_guard_epsilon", 0.01)),
            )
            record.update({"epoch": int(epoch), "version": str(self.competence_schedule_version)})
            self.previous_epoch_per_agent_acc = [float(value) for value in snapshot_metrics.get("per_agent_acc", [])]
            self.previous_epoch_bottom2_mean_acc = float(snapshot_metrics.get("bottom2_mean_acc", 0.0) or 0.0)
            self.latest_competence_probe_metrics = dict(snapshot_metrics)
            self.specialization_strength = float(record["next_specialization_strength"])
            self.competence_phase_epoch = int(epoch) + 1
            self._recompute_effective_residual_strength()
            return record
        values = [float(value) for value in (per_agent_acc or [])]
        ordered = sorted(values)
        bottom2 = float(np.mean(ordered[: min(2, len(ordered))])) if ordered else 0.0
        self.previous_epoch_per_agent_acc = values
        self.previous_epoch_bottom2_mean_acc = bottom2
        self.specialization_strength = competence_specialization_strength(
            bottom2,
            float(getattr(self.cfg, "competence_floor_low", 0.55)),
            float(getattr(self.cfg, "competence_floor_high", 0.65)),
        )
        self.competence_phase_epoch = int(epoch) + 1
        self._recompute_effective_residual_strength()
        return float(self.specialization_strength)

    def _recompute_effective_residual_strength(self, qd_ready: Optional[bool] = None) -> float:
        if not self._is_stable_qd_lineage():
            self.effective_residual_strength = float(self.specialization_strength)
            return self.effective_residual_strength
        latest = dict(getattr(self, "latest_joint_team_metrics", {}) or {})
        ready = bool(latest.get("qd_readiness_passed", False)) if qd_ready is None else bool(qd_ready)
        self.effective_residual_strength = max(
            float(self.specialization_strength),
            float(self.cfg.residual_specialization_qd_floor) if ready else 0.0,
        )
        latest.update({
            "competence_schedule_strength": float(self.specialization_strength),
            "qd_residual_floor_applied": bool(ready and self.effective_residual_strength > self.specialization_strength),
            "effective_residual_strength": float(self.effective_residual_strength),
        })
        if latest:
            self.latest_joint_team_metrics = latest
        return self.effective_residual_strength

    def _effective_progressive_weight(self, configured: float) -> float:
        if bool(getattr(self.cfg, "competence_progressive_residual_enabled", False)):
            return float(configured) * float(getattr(self, "effective_residual_strength", self.specialization_strength))
        return float(configured)

    def _effective_support_shrinkage(self) -> float:
        base = float(getattr(self.cfg, "specialization_support_shrinkage", 3.0) or 3.0)
        if not bool(getattr(self.cfg, "competence_progressive_residual_enabled", False)):
            return base
        extra = float(getattr(self.cfg, "competence_extra_support_shrinkage", 3.0) or 0.0)
        return base + (1.0 - float(self.specialization_strength)) * extra

    def _uses_baseline_candidate_metrics(self) -> bool:
        return self._is_guarded_reward_mode() or self._is_vote_useful_diversity_mode() or self._is_coverage_useful_diversity_mode() or self._is_competence_depth_reward_mode()

    def _uses_vote_pareto_selection(self) -> bool:
        return str(getattr(self.cfg, "candidate_selection_mode", "scalar_reward") or "scalar_reward").lower() in {
            "vote_pareto", "vote_error_pareto", "competence_depth_pareto"
        }

    def _uses_vote_error_pareto_selection(self) -> bool:
        cfg = getattr(self, "cfg", None)
        return str(getattr(cfg, "candidate_selection_mode", "scalar_reward") or "scalar_reward").lower() == "vote_error_pareto"

    def _residual_specialization_enabled(self) -> bool:
        return bool(getattr(self.cfg, "residual_specialization_enabled", False))

    def _v7_residual_protocol_enabled(self) -> bool:
        cfg = getattr(self, "cfg", None)
        return bool(
            self._uses_vote_error_pareto_selection()
            or any(bool(getattr(cfg, name, False)) for name in (
                "boundary_selector_enabled",
                "shared_error_metrics_enabled",
                "residual_specialization_enabled",
                "error_dependence_guard_enabled",
                "residual_cycle_guard_enabled",
                "mechanism_trust_region_enabled",
            ))
        )

    def _experiment_protocol_version(self) -> str:
        if self._is_stable_qd_lineage():
            return "vote_oriented_v8_stable_qd_lineage"
        if bool(getattr(self.cfg, "competence_depth_enabled", False)):
            return "vote_oriented_v8_competence_depth"
        return EXPERIMENT_PROTOCOL_VERSION

    def _normalized_prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(self._prompt_signature(prompt).encode("utf-8")).hexdigest()

    def prompt_change_ratio(self, parent_prompt: str, candidate_prompt: str) -> float:
        parent = self._prompt_signature(parent_prompt)
        candidate = self._prompt_signature(candidate_prompt)
        return self._clip01(1.0 - SequenceMatcher(None, parent, candidate).ratio())

    def _behavior_context_for_baseline(
        self,
        *,
        agent_id: int,
        answers: Sequence[str],
        gold: str,
        rollout: Dict[str, Any],
        question_hash: str = "",
    ) -> str:
        invalids = list(rollout.get("invalid_flags", []))
        if agent_id >= len(answers) or (agent_id < len(invalids) and int(invalids[agent_id]) > 0):
            return BehaviorContext.INVALID.value
        target_correct = bool(self._safe_agent_correct(rollout, agent_id))
        team_correct = bool(rollout.get("vote_correct", 0))
        gold_count = int(rollout.get("gold_vote_count", 0) or 0)
        largest_wrong = int(rollout.get("largest_wrong_vote_count", 0) or 0)
        if target_correct:
            if bool(getattr(self.cfg, "competence_depth_enabled", False)) and team_correct:
                without_target = list(answers)
                without_target[agent_id] = ""
                counterfactual = self._vote_with_diagnostics(without_target, question_hash=question_hash)
                if not self.task_spec.match_answer(str(counterfactual.get("vote_answer", "")), gold):
                    return BehaviorContext.TARGET_CORRECT_PIVOTAL_HOLD.value
                return BehaviorContext.TARGET_CORRECT_ROBUST.value
            if gold_count - largest_wrong <= 1:
                return BehaviorContext.TARGET_CORRECT_PIVOTAL_HOLD.value
            return BehaviorContext.TARGET_CORRECT_ROBUST.value

        if not team_correct:
            counterfactual_answers = list(answers)
            counterfactual_answers[agent_id] = str(gold)
            counterfactual = self._vote_with_diagnostics(counterfactual_answers, question_hash=question_hash)
            pivotal = self.task_spec.match_answer(str(counterfactual.get("vote_answer", "")), gold)
            return (
                BehaviorContext.TEAM_WRONG_PIVOTAL_FIX.value
                if pivotal else BehaviorContext.TEAM_WRONG_NONPIVOTAL.value
            )

        target_answer = str(answers[agent_id] or "").strip()
        wrong_counts = Counter(
            str(answer or "").strip()
            for answer in answers
            if str(answer or "").strip() and not self.task_spec.match_answer(str(answer), gold)
        )
        is_dominant_wrong = bool(
            target_answer
            and wrong_counts.get(target_answer, 0) == max(wrong_counts.values(), default=0)
            and (largest_wrong > 0 or gold_count - largest_wrong <= 1)
        )
        return (
            BehaviorContext.TEAM_CORRECT_DOMINANT_WRONG_REDUNDANCY.value
            if is_dominant_wrong or gold_count - largest_wrong <= 1
            else BehaviorContext.TEAM_CORRECT_TARGET_WRONG_OTHER.value
        )

    def _candidate_behavior_metrics(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        values = {context: [] for context in BEHAVIOR_CONTEXT_NAMES}
        context_counts = {context: 0 for context in BEHAVIOR_CONTEXT_NAMES}
        fingerprint: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            context = str(row.get("behavior_context", BehaviorContext.INVALID.value))
            if context not in values:
                context = BehaviorContext.INVALID.value
            vote_gain = int(not bool(row.get("baseline_vote_correct", 0)) and bool(row.get("candidate_vote_correct", 0)))
            vote_loss = int(bool(row.get("baseline_vote_correct", 0)) and not bool(row.get("candidate_vote_correct", 0)))
            margin_delta = float(row.get("candidate_mean_vote_margin", -1.0)) - float(row.get("baseline_mean_vote_margin", -1.0))
            wrong_to_correct = int(not bool(row.get("baseline_target_correct", 0)) and bool(row.get("target_agent_correct", 0)))
            correct_to_wrong = int(bool(row.get("baseline_target_correct", 0)) and not bool(row.get("target_agent_correct", 0)))
            transition = 2.0 * vote_gain - 2.0 * vote_loss + max(0.0, margin_delta) - max(0.0, -margin_delta) + 0.5 * wrong_to_correct - 0.5 * correct_to_wrong
            values[context].append(float(transition))
            context_counts[context] += 1
            question_hash = str(row.get("question_hash", ""))
            if question_hash:
                answer_signature = self._prompt_signature(str(row.get("target_answer", "")))
                fingerprint[question_hash] = {
                    "target_correct": bool(row.get("target_agent_correct", 0)),
                    "target_answer_hash": hashlib.sha256(answer_signature.encode("utf-8")).hexdigest(),
                    "team_vote_correct": bool(row.get("candidate_vote_correct", 0)),
                    "vote_margin_bucket": int(round(10.0 * float(row.get("candidate_mean_vote_margin", -1.0)))),
                    "behavior_context": context,
                }
        return {
            "behavior_context_counts": context_counts,
            "candidate_transition_vector": {
                context: float(np.mean(context_values)) if context_values else 0.0
                for context, context_values in values.items()
            },
            "candidate_transition_support": context_counts,
            "behavior_fingerprint": fingerprint,
        }

    def _candidate_boundary_error_metrics(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(rows)
        denominator = float(count) if count else 1.0
        individual_fix = individual_regression = 0
        pivotal_rescue = pivotal_loss = 0
        plurality_opportunity = plurality_fix = plurality_loss = 0
        same_wrong_break = same_wrong_create = 0
        shared_rescue = shared_creation = 0.0
        for row in rows:
            baseline_correct = bool(row.get("baseline_target_correct", False))
            candidate_correct = bool(row.get("candidate_target_correct", row.get("target_agent_correct", False)))
            baseline_vote = bool(row.get("baseline_vote_correct", False))
            candidate_vote = bool(row.get("candidate_vote_correct", False))
            fixed = not baseline_correct and candidate_correct
            regressed = baseline_correct and not candidate_correct
            individual_fix += int(fixed)
            individual_regression += int(regressed)
            pivotal_rescue += int(fixed and not baseline_vote and candidate_vote)
            pivotal_loss += int(regressed and baseline_vote and not candidate_vote)
            plurality_opportunity += int(bool(row.get("plurality_pivotal_fix_opportunity", False)))
            plurality_fix += int(bool(row.get("plurality_pivotal_fix", False)))
            plurality_loss += int(bool(row.get("plurality_pivotal_loss", False)))
            peer_wrong_count = int(row.get("peer_wrong_count", 0) or 0)
            shared_weight = float(peer_wrong_count) / max(1, len(self.agents) - 1)
            shared_rescue += shared_weight * int(fixed)
            shared_creation += shared_weight * int(regressed)
            baseline_cluster = bool(row.get("baseline_target_in_dominant_wrong_cluster", False))
            candidate_cluster = bool(row.get("candidate_target_in_dominant_wrong_cluster", False))
            same_wrong_break += int(baseline_cluster and not candidate_cluster)
            same_wrong_create += int(not baseline_cluster and candidate_cluster)
        pivotal_rescue_rate = float(pivotal_rescue) / denominator if count else 0.0
        pivotal_loss_rate = float(pivotal_loss) / denominator if count else 0.0
        shared_rescue_score = float(shared_rescue) / denominator if count else 0.0
        shared_creation_score = float(shared_creation) / denominator if count else 0.0
        same_wrong_break_rate = float(same_wrong_break) / denominator if count else 0.0
        same_wrong_create_rate = float(same_wrong_create) / denominator if count else 0.0
        legacy_net_gain = (
            4.0 * pivotal_rescue_rate
            - 4.0 * pivotal_loss_rate
            + shared_rescue_score
            - 1.5 * shared_creation_score
            + 0.5 * same_wrong_break_rate
            - 0.5 * same_wrong_create_rate
        )
        plurality_opportunity_rate = float(plurality_opportunity) / denominator if count else 0.0
        plurality_fix_rate = float(plurality_fix) / denominator if count else 0.0
        plurality_loss_rate = float(plurality_loss) / denominator if count else 0.0
        plurality_net_gain = (
            4.0 * plurality_fix_rate
            - 4.0 * plurality_loss_rate
            + shared_rescue_score
            - 1.5 * shared_creation_score
            + 0.5 * same_wrong_break_rate
            - 0.5 * same_wrong_create_rate
        )
        active_net_gain = plurality_net_gain if bool(getattr(self.cfg, "competence_depth_enabled", False)) else legacy_net_gain
        return {
            "individual_fix_count": int(individual_fix),
            "individual_regression_count": int(individual_regression),
            "pivotal_rescue_count": int(pivotal_rescue),
            "pivotal_rescue_rate": pivotal_rescue_rate,
            "pivotal_loss_count": int(pivotal_loss),
            "pivotal_loss_rate": pivotal_loss_rate,
            "shared_error_rescue_score": shared_rescue_score,
            "shared_error_creation_score": shared_creation_score,
            "same_wrong_cluster_break_count": int(same_wrong_break),
            "same_wrong_cluster_create_count": int(same_wrong_create),
            "same_wrong_cluster_break_rate": same_wrong_break_rate,
            "same_wrong_cluster_create_rate": same_wrong_create_rate,
            "plurality_pivotal_fix_opportunity_count": int(plurality_opportunity),
            "plurality_pivotal_fix_opportunity_rate": plurality_opportunity_rate,
            "plurality_pivotal_fix_count": int(plurality_fix),
            "plurality_pivotal_fix_rate": plurality_fix_rate,
            "plurality_pivotal_loss_count": int(plurality_loss),
            "plurality_pivotal_loss_rate": plurality_loss_rate,
            "plurality_boundary_shared_error_net_gain": float(plurality_net_gain),
            "boundary_shared_error_net_gain": float(active_net_gain),
            "pivotal_definition": (
                "actual_plurality_counterfactual"
                if bool(getattr(self.cfg, "competence_depth_enabled", False))
                else "legacy_vote_boundary"
            ),
        }

    def _candidate_residual_metrics(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tau = self._effective_support_shrinkage()
        support = {family: 0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
        weighted_gain = {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
        weighted_loss = {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
        weighted_sum = {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
        for row in rows:
            family = str(row.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value))
            if family not in support:
                family = CapabilityResidualFamily.UNKNOWN.value
            vote_gain = int(not bool(row.get("baseline_vote_correct", 0)) and bool(row.get("candidate_vote_correct", 0)))
            vote_loss = int(bool(row.get("baseline_vote_correct", 0)) and not bool(row.get("candidate_vote_correct", 0)))
            margin_delta = float(row.get("candidate_mean_vote_margin", -1.0)) - float(row.get("baseline_mean_vote_margin", -1.0))
            fixed = int(not bool(row.get("baseline_target_correct", 0)) and bool(row.get("candidate_target_correct", row.get("target_agent_correct", 0))))
            regressed = int(bool(row.get("baseline_target_correct", 0)) and not bool(row.get("candidate_target_correct", row.get("target_agent_correct", 0))))
            raw = 2.0 * vote_gain - 2.0 * vote_loss + margin_delta + 0.5 * fixed - 0.5 * regressed
            context = str(row.get("behavior_context", BehaviorContext.INVALID.value))
            context_weight = float(VOTE_CONTEXT_WEIGHTS.get(context, 1.0))
            weighted = context_weight * raw
            support[family] += 1
            weighted_sum[family] += weighted
            weighted_gain[family] += max(0.0, weighted)
            weighted_loss[family] += max(0.0, -weighted)
            row["raw_transition_value"] = float(raw)
            row["vote_context_weight"] = context_weight
        shrunk = {}
        reliability = {}
        for family in CAPABILITY_RESIDUAL_FAMILY_NAMES:
            reliability[family] = float(support[family] / (support[family] + tau)) if support[family] else 0.0
            shrunk[family] = float(weighted_sum[family] / (support[family] + tau)) if support[family] else 0.0
        for row in rows:
            family = str(row.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value))
            if family not in support:
                family = CapabilityResidualFamily.UNKNOWN.value
            row["support_reliability"] = reliability[family]
            row["shrunk_transition_value"] = shrunk[family]
        return {
            "capability_transition_support": support,
            "capability_weighted_gain": weighted_gain,
            "capability_weighted_loss": weighted_loss,
            "capability_support_reliability": reliability,
            "capability_shrunk_transition": shrunk,
            "capability_evidence_rows": [
                {
                    "question_hash": str(row.get("question_hash", "")),
                    "capability_residual_family": str(row.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                    "raw_transition_value": float(row.get("raw_transition_value", 0.0) or 0.0),
                    "vote_context_weight": float(row.get("vote_context_weight", 0.0) or 0.0),
                    "support_reliability": float(row.get("support_reliability", 0.0) or 0.0),
                    "shrunk_transition_value": float(row.get("shrunk_transition_value", 0.0) or 0.0),
                }
                for row in rows
            ],
        }

    @staticmethod
    def _candidate_v7_log_fields(metrics: Mapping[str, Any]) -> Dict[str, Any]:
        count_fields = (
            "individual_fix_count",
            "individual_regression_count",
            "pivotal_rescue_count",
            "pivotal_loss_count",
            "same_wrong_cluster_break_count",
            "same_wrong_cluster_create_count",
        )
        rate_fields = (
            "pivotal_rescue_rate",
            "pivotal_loss_rate",
            "shared_error_rescue_score",
            "shared_error_creation_score",
            "same_wrong_cluster_break_rate",
            "same_wrong_cluster_create_rate",
            "boundary_shared_error_net_gain",
            "capability_alignment",
        )
        fields: Dict[str, Any] = {
            key: int(metrics.get(key, 0) or 0) for key in count_fields
        }
        fields.update({
            key: float(metrics.get(key, 0.0) or 0.0) for key in rate_fields
        })
        fields.update({
            "error_dependence_guard_passed": bool(metrics.get("error_dependence_guard_passed", True)),
            "paired_boundary_transition_rows": metrics.get("paired_boundary_transition_rows", []),
            "capability_transition_support": metrics.get("capability_transition_support", {}),
            "capability_weighted_gain": metrics.get("capability_weighted_gain", {}),
            "capability_weighted_loss": metrics.get("capability_weighted_loss", {}),
            "capability_support_reliability": metrics.get("capability_support_reliability", {}),
            "capability_shrunk_transition": metrics.get("capability_shrunk_transition", {}),
            "capability_evidence_rows": metrics.get("capability_evidence_rows", []),
        })
        return fields

    def capability_alignment(self, agent: AgentState, metrics: Dict[str, Any]) -> float:
        positive = np.array([
            max(0.0, float(metrics.get("capability_shrunk_transition", {}).get(family, 0.0)))
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        ], dtype=float)
        profile = np.array([
            max(0.0, float(agent.capability_profile.get(family, 0.0)))
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        ], dtype=float)
        denominator = float(np.linalg.norm(positive) * np.linalg.norm(profile))
        return self._clip01(float(np.dot(positive, profile) / denominator)) if denominator > 0.0 else 0.0

    def _accumulate_capability_evidence(self, agent: AgentState, metrics: Dict[str, Any], epoch_id: int) -> None:
        agent.pending_capability_evidence.append({
            "epoch": int(epoch_id),
            "support": dict(metrics.get("capability_transition_support", {})),
            "weighted_gain": dict(metrics.get("capability_weighted_gain", {})),
            "weighted_loss": dict(metrics.get("capability_weighted_loss", {})),
        })
        agent.pending_capability_update_count += 1

    def _flush_capability_profile(self, agent: AgentState, epoch_id: int, force: bool = False) -> bool:
        period = max(1, int(getattr(self.cfg, "specialization_update_period", 2) or 2))
        if not agent.pending_capability_evidence or (not force and agent.pending_capability_update_count < period):
            return False
        tau = self._effective_support_shrinkage()
        loss_weight = float(getattr(self.cfg, "capability_loss_weight", 1.5) or 1.5)
        for pending in agent.pending_capability_evidence:
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES:
                evidence = agent.capability_evidence[family]
                evidence.support += int(pending.get("support", {}).get(family, 0) or 0)
                evidence.weighted_gain += float(pending.get("weighted_gain", {}).get(family, 0.0) or 0.0)
                evidence.weighted_loss += float(pending.get("weighted_loss", {}).get(family, 0.0) or 0.0)
                reliability = float(evidence.support / (evidence.support + tau)) if evidence.support else 0.0
                evidence.posterior_value = reliability * (
                    evidence.weighted_gain - loss_weight * evidence.weighted_loss
                )
                evidence.last_updated_epoch = int(epoch_id)
        positive = {
            family: max(0.0, agent.capability_evidence[family].posterior_value)
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        }
        total_positive = sum(positive.values())
        changed = False
        if total_positive > 0.0:
            target = {family: value / total_positive for family, value in positive.items()}
            old = dict(agent.capability_profile or empty_capability_profile())
            mu = float(getattr(self.cfg, "specialization_ema", 0.20))
            updated = {
                family: (1.0 - mu) * float(old.get(family, 0.0)) + mu * target[family]
                for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
            }
            total = sum(updated.values())
            agent.capability_profile = {family: value / total for family, value in updated.items()}
            changed = True
        agent.pending_capability_evidence.clear()
        agent.pending_capability_update_count = 0
        if changed:
            agent.capability_profile_update_count += 1
        return changed

    def _update_vote_context_profile(self, agent: AgentState, metrics: Dict[str, Any]) -> bool:
        transition = metrics.get("candidate_transition_vector", {})
        positive = {context: max(0.0, float(transition.get(context, 0.0))) for context in BEHAVIOR_CONTEXT_NAMES}
        total = sum(positive.values())
        if total <= 0.0:
            return False
        target = {context: value / total for context, value in positive.items()}
        mu = float(getattr(self.cfg, "specialization_ema", 0.20))
        old = dict(agent.vote_context_profile or uniform_vote_context_profile())
        updated = {context: (1.0 - mu) * old.get(context, 0.0) + mu * target[context] for context in BEHAVIOR_CONTEXT_NAMES}
        norm = sum(updated.values())
        agent.vote_context_profile = {context: value / norm for context, value in updated.items()}
        return True

    @staticmethod
    def behavior_fingerprint_similarity(
        candidate: Mapping[str, Any],
        history: Mapping[str, Any],
    ) -> Tuple[float, int]:
        overlap = sorted(set(candidate).intersection(history))
        if not overlap:
            return 0.0, 0
        correctness_matches = 0
        answer_matches = 0
        for key in overlap:
            current = candidate[key]
            previous = history[key]
            current_correct = current.target_correct if isinstance(current, BehaviorFingerprintEntry) else bool(current.get("target_correct", False))
            previous_correct = previous.target_correct if isinstance(previous, BehaviorFingerprintEntry) else bool(previous.get("target_correct", False))
            current_answer = current.target_answer_hash if isinstance(current, BehaviorFingerprintEntry) else str(current.get("target_answer_hash", ""))
            previous_answer = previous.target_answer_hash if isinstance(previous, BehaviorFingerprintEntry) else str(previous.get("target_answer_hash", ""))
            correctness_matches += int(current_correct == previous_correct)
            answer_matches += int(current_answer == previous_answer)
        count = len(overlap)
        return 0.7 * correctness_matches / count + 0.3 * answer_matches / count, count

    @staticmethod
    def behavior_fingerprint_utility(fingerprint: Mapping[str, Any]) -> Dict[str, float]:
        utility: Dict[str, float] = {}
        for key, entry in fingerprint.items():
            target_correct = entry.target_correct if isinstance(entry, BehaviorFingerprintEntry) else bool(entry.get("target_correct", False))
            team_correct = entry.team_vote_correct if isinstance(entry, BehaviorFingerprintEntry) else bool(entry.get("team_vote_correct", False))
            margin_bucket = entry.vote_margin_bucket if isinstance(entry, BehaviorFingerprintEntry) else int(entry.get("vote_margin_bucket", 0) or 0)
            utility[str(key)] = 2.0 * float(team_correct) + float(target_correct) + 0.5 * float(margin_bucket) / 10.0
        return utility

    @staticmethod
    def paired_utility_improvement(
        candidate: Mapping[str, float], history: Mapping[str, float]
    ) -> Tuple[float, int]:
        overlap = sorted(set(candidate).intersection(history))
        if not overlap:
            return 0.0, 0
        return float(np.mean([float(candidate[key]) - float(history[key]) for key in overlap])), len(overlap)

    def _append_bounded_archive(self, archive: List[Any], value: Any) -> None:
        archive.append(value)
        limit = max(0, int(getattr(self.cfg, "behavior_archive_size", 16)))
        if limit == 0:
            archive.clear()
        elif len(archive) > limit:
            del archive[:-limit]

    def _candidate_trajectory_feasibility(
        self,
        agent: AgentState,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        prompt = str(item.get("prompt", ""))
        parent_prompt = str(item.get("parent_prompt", agent.current_prompt))
        prompt_hash = self._normalized_prompt_hash(prompt)
        parent_hash = self._normalized_prompt_hash(parent_prompt)
        change_ratio = self.prompt_change_ratio(parent_prompt, prompt)
        diagnostics: Dict[str, Any] = {
            "prompt_hash": prompt_hash,
            "parent_prompt_hash": parent_hash,
            "prompt_change_ratio": change_ratio,
            "max_behavior_cycle_similarity": 0.0,
            "behavior_cycle_overlap": 0,
            "matched_behavior_state_id": "",
            "exact_prompt_cycle": False,
            "behavior_cycle_guard_passed": True,
            "prompt_trust_region_passed": True,
            "rejection_reason": "",
        }
        if not (
            bool(getattr(self.cfg, "residual_cycle_guard_enabled", False))
            or bool(getattr(self.cfg, "mechanism_trust_region_enabled", False))
        ):
            return diagnostics

        source = self._candidate_pool_source(item)
        current_hash = self._normalized_prompt_hash(agent.current_prompt)
        if prompt_hash == current_hash and source in {"existing_beam", "current_active_fallback"}:
            return diagnostics

        _, _, original_guards_passed = self._vote_pareto_feasibility(metrics)
        if not original_guards_passed:
            return diagnostics

        proposal = item.get("proposal", {}) if isinstance(item.get("proposal", {}), dict) else {}
        if source == "optimizer" and bool(getattr(self.cfg, "mechanism_trust_region_enabled", False)):
            preserved = proposal.get("preserved_mechanisms", [])
            modified = proposal.get("modified_mechanism", proposal.get("new_or_modified_mechanism", ""))
            change_summary = str(proposal.get("change_summary", "")).strip()
            mechanism_contract_passed = bool(
                isinstance(preserved, list)
                and any(str(value).strip() for value in preserved)
                and isinstance(modified, str)
                and bool(modified.strip())
                and bool(change_summary)
            )
            diagnostics["mechanism_contract_passed"] = mechanism_contract_passed
            if not mechanism_contract_passed:
                diagnostics["prompt_trust_region_passed"] = False
                diagnostics["rejection_reason"] = "mechanism_contract_missing"
                return diagnostics

        historic_hashes = {self._normalized_prompt_hash(value) for value in agent.history}
        historic_hashes.update(state.prompt_hash for state in agent.accepted_behavior_archive)
        historic_hashes.update(state.prompt_hash for state in agent.rejected_behavior_archive)
        exact_cycle = bool(prompt_hash in historic_hashes)
        diagnostics["exact_prompt_cycle"] = exact_cycle
        if exact_cycle:
            diagnostics["rejection_reason"] = "exact_prompt_cycle"
            return diagnostics

        candidate_fingerprint = metrics.get("behavior_fingerprint", {})
        best_similarity = 0.0
        best_overlap = 0
        best_state_id = ""
        residual_cycle = bool(getattr(self.cfg, "residual_cycle_guard_enabled", False))
        candidate_utility = self.behavior_fingerprint_utility(candidate_fingerprint) if isinstance(candidate_fingerprint, dict) else {}
        matched_kind = ""
        utility_improvement = 0.0
        if bool(getattr(self.cfg, "behavior_cycle_guard_enabled", True)) and isinstance(candidate_fingerprint, dict):
            for state in agent.accepted_behavior_archive:
                similarity, overlap = self.behavior_fingerprint_similarity(candidate_fingerprint, state.behavior_fingerprint)
                if (similarity, overlap, state.state_id) > (best_similarity, best_overlap, best_state_id):
                    best_similarity, best_overlap, best_state_id = similarity, overlap, state.state_id
                    matched_kind = "accepted"
                    historical_utility = state.paired_behavior_utility or self.behavior_fingerprint_utility(state.behavior_fingerprint)
                    utility_improvement, _ = self.paired_utility_improvement(candidate_utility, historical_utility)
            if residual_cycle:
                for state in agent.rejected_behavior_archive:
                    similarity, overlap = self.behavior_fingerprint_similarity(candidate_fingerprint, state.behavior_fingerprint)
                    if (similarity, overlap, state.state_id) > (best_similarity, best_overlap, best_state_id):
                        best_similarity, best_overlap, best_state_id = similarity, overlap, state.state_id
                        matched_kind = "rejected"
                        utility_improvement, _ = self.paired_utility_improvement(
                            candidate_utility, state.paired_behavior_utility
                        )
        diagnostics.update(
            {
                "max_behavior_cycle_similarity": float(best_similarity),
                "behavior_cycle_overlap": int(best_overlap),
                "matched_behavior_state_id": best_state_id,
                "matched_behavior_archive": matched_kind,
                "paired_behavior_utility_improvement": float(utility_improvement),
            }
        )
        meaningful_improvement = bool(
            float(metrics.get("vote_delta", 0.0) or 0.0) > float(getattr(self.cfg, "behavior_cycle_improvement_epsilon", 0.01))
            or float(metrics.get("accuracy_delta", 0.0) or 0.0) > float(getattr(self.cfg, "behavior_cycle_improvement_epsilon", 0.01))
            or float(metrics.get("vote_margin_delta", 0.0) or 0.0) > float(getattr(self.cfg, "behavior_cycle_margin_epsilon", 0.05))
        )
        if residual_cycle:
            meaningful_improvement = bool(
                utility_improvement > float(getattr(self.cfg, "behavior_cycle_improvement_epsilon", 0.01))
            )
        behavior_cycle = bool(
            bool(getattr(self.cfg, "behavior_cycle_guard_enabled", True))
            and best_overlap >= int(getattr(self.cfg, "behavior_cycle_min_overlap", 16))
            and best_similarity >= float(getattr(self.cfg, "behavior_cycle_similarity_threshold", 0.95))
            and not meaningful_improvement
        )
        diagnostics["behavior_cycle_guard_passed"] = not behavior_cycle
        if behavior_cycle:
            diagnostics["rejection_reason"] = (
                "rejected_failure_cycle" if residual_cycle and matched_kind == "rejected"
                else "accepted_state_cycle" if residual_cycle
                else "behavior_cycle"
            )
            return diagnostics

        large_shift = bool(
            source == "optimizer"
            and
            bool(getattr(self.cfg, "prompt_trust_region_enabled", True))
            and agent.accept_count >= int(getattr(self.cfg, "prompt_large_shift_warmup_accepts", 2))
            and change_ratio > float(getattr(self.cfg, "prompt_max_change_ratio", 0.45))
        )
        large_shift_supported = bool(
            float(metrics.get("vote_delta", 0.0) or 0.0) >= float(getattr(self.cfg, "prompt_large_shift_min_vote_delta", 0.02))
            and float(metrics.get("accuracy_delta", 0.0) or 0.0) >= 0.0
            and float(metrics.get("vote_loss_rate", 0.0) or 0.0) <= float(getattr(self.cfg, "baseline_allowed_vote_loss", 0.0))
        )
        if bool(getattr(self.cfg, "mechanism_trust_region_enabled", False)):
            large_shift_supported = bool(
                large_shift_supported
                and float(metrics.get("pivotal_loss_rate", 0.0) or 0.0) <= 0.0
                and float(metrics.get("shared_error_creation_score", 0.0) or 0.0)
                <= float(metrics.get("shared_error_rescue_score", 0.0) or 0.0)
            )
        diagnostics["prompt_trust_region_passed"] = bool(not large_shift or large_shift_supported)
        if large_shift and not large_shift_supported:
            diagnostics["rejection_reason"] = "unsupported_large_prompt_shift"
        return diagnostics

    def _trajectory_event(
        self,
        *,
        agent_id: int,
        epoch_id: int,
        step_id: int,
        item: Dict[str, Any],
        accepted: bool,
        profile_before: Dict[str, float],
        profile_after: Dict[str, float],
    ) -> Dict[str, Any]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        return {
            **self._base_log_fields(),
            "event": "trajectory_evolution",
            "epoch": int(epoch_id),
            "step": int(step_id),
            "agent_id": int(agent_id),
            "candidate_id": str(item.get("candidate_id", "")),
            "candidate_pool_source": self._candidate_pool_source(item),
            "candidate_source": self._candidate_generation_source(item),
            "accepted": bool(accepted),
            "behavior_context_counts": metrics.get("behavior_context_counts", {}),
            "candidate_transition_vector": metrics.get("candidate_transition_vector", {}),
            "candidate_transition_support": metrics.get("candidate_transition_support", {}),
            "capability_profile_before": profile_before,
            "capability_profile_after": profile_after,
            "vote_context_profile": dict(self.agents[agent_id].vote_context_profile),
            "capability_profile": dict(self.agents[agent_id].capability_profile),
            "capability_transition_support": metrics.get("capability_transition_support", {}),
            "capability_support_reliability": metrics.get("capability_support_reliability", {}),
            "capability_shrunk_transition": metrics.get("capability_shrunk_transition", {}),
            "capability_alignment": float(metrics.get("capability_alignment", 0.0) or 0.0),
            **{key: metrics.get(key) for key in (
                "prompt_hash", "parent_prompt_hash", "prompt_change_ratio",
                "max_behavior_cycle_similarity", "behavior_cycle_overlap", "matched_behavior_state_id",
                "exact_prompt_cycle", "behavior_cycle_guard_passed", "prompt_trust_region_passed", "rejection_reason",
                "matched_behavior_archive", "paired_behavior_utility_improvement", "mechanism_contract_passed",
            )},
        }

    def _vote_pareto_feasibility(self, metrics: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        weights = self._effective_reward_weights()
        baseline_target = float(metrics.get("baseline_target_accuracy", 0.0) or 0.0)
        candidate_target = float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0)
        baseline_invalid = float(metrics.get("baseline_invalid_rate", 0.0) or 0.0)
        candidate_invalid = float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0)
        guard_epsilon = float(weights.get("accuracy_guard_epsilon", 0.0))
        if self._is_v82_hybrid():
            guard_epsilon = float(getattr(self.cfg, "catastrophic_target_accuracy_loss_epsilon", 0.05))
        elif self._uses_competence_depth_pareto_selection():
            guard_epsilon = float(self.specialization_strength) * float(getattr(self.cfg, "accuracy_guard_epsilon", guard_epsilon))
        accuracy_guard_passed = candidate_target >= baseline_target - guard_epsilon
        metrics["effective_accuracy_guard_epsilon"] = guard_epsilon
        invalid_guard_passed = candidate_invalid <= baseline_invalid + float(getattr(self.cfg, "invalid_guard_epsilon", 0.0) or 0.0)
        dependence_guard_passed = bool(self._is_v82_hybrid() or (
            not bool(getattr(self.cfg, "error_dependence_guard_enabled", False))
            or (
                float(metrics.get("pivotal_loss_rate", 0.0) or 0.0)
                <= float(getattr(self.cfg, "pivotal_loss_guard_epsilon", 0.0) or 0.0)
                and float(metrics.get("shared_error_creation_score", 0.0) or 0.0)
                <= float(metrics.get("shared_error_rescue_score", 0.0) or 0.0)
                + float(getattr(self.cfg, "shared_error_creation_epsilon", 0.02) or 0.0)
            )
        ))
        metrics["error_dependence_guard_passed"] = dependence_guard_passed
        return bool(accuracy_guard_passed), bool(invalid_guard_passed), bool(
            accuracy_guard_passed and invalid_guard_passed and dependence_guard_passed
        )

    def _vote_pareto_active_sort_key(self, item: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float, int, str]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        rank = item.get("pareto_rank")
        normalized_rank = int(rank) if isinstance(rank, int) and rank >= 0 else 10**9
        return (
            -float(metrics.get("vote_delta", 0.0) or 0.0),
            float(metrics.get("vote_loss_rate", 0.0) or 0.0),
            -float(metrics.get("vote_gain_rate", 0.0) or 0.0),
            -float(metrics.get("vote_margin_delta", 0.0) or 0.0),
            -float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0),
            -float(
                metrics.get("boundary_shared_error_net_gain", 0.0)
                if self._uses_vote_error_pareto_selection()
                else metrics.get("boundary_useful_diversity_delta", 0.0)
                or 0.0
            ),
            float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0),
            normalized_rank,
            str(item.get("candidate_id", "")),
        )

    def _vote_pareto_crowding_sort_key(self, item: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float, float, str]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        distance = float(item.get("pareto_crowding_distance", 0.0) or 0.0)
        return (
            -distance,
            -float(metrics.get("vote_delta", 0.0) or 0.0),
            float(metrics.get("vote_loss_rate", 0.0) or 0.0),
            -float(metrics.get("vote_gain_rate", 0.0) or 0.0),
            -float(metrics.get("vote_margin_delta", 0.0) or 0.0),
            -float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0),
            -float(
                metrics.get("boundary_shared_error_net_gain", 0.0)
                if self._uses_vote_error_pareto_selection()
                else metrics.get("boundary_useful_diversity_delta", 0.0)
                or 0.0
            ),
            float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0),
            str(item.get("candidate_id", "")),
        )

    def _competence_depth_sort_key(self, item: Dict[str, Any]) -> Tuple[Any, ...]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        strength = float(self.specialization_strength)
        early = (
            -float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
            -float(metrics.get("depth2_gain_rate", 0.0) or 0.0),
            -float(metrics.get("depth2_net_delta", 0.0) or 0.0),
        )
        late = (
            -float(metrics.get("vote_gain_rate", 0.0) or 0.0),
            float(metrics.get("vote_loss_rate", 0.0) or 0.0),
            -float(metrics.get("boundary_shared_error_net_gain", 0.0) or 0.0),
        )
        order = late + early if strength >= 0.5 else early + late
        return order + (-float(metrics.get("reward", item.get("reward", 0.0)) or 0.0), str(item.get("candidate_id", "")))

    def _select_vote_pareto_beam(
        self,
        evaluated: List[Dict[str, Any]],
        beam_size: int,
        current_prompt: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select a retained beam using feasibility, Pareto fronts, then deterministic crowding."""
        feasible: List[Dict[str, Any]] = []
        for item in evaluated:
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
            accuracy_passed, invalid_passed, is_feasible = self._vote_pareto_feasibility(metrics)
            metrics.update(
                {
                    "accuracy_guard_passed": accuracy_passed,
                    "invalid_guard_passed": invalid_passed,
                    "error_dependence_guard_passed": bool(metrics.get("error_dependence_guard_passed", True)),
                }
            )
            item["metrics"] = metrics
            item["pareto_feasible"] = is_feasible
            item["pareto_rank"] = None
            item["pareto_crowding_distance"] = None
            item["pareto_selected"] = False
            item["pareto_forced_fallback"] = False
            if is_feasible:
                feasible.append(item)

        original_feasible_count = len(feasible)
        forced_fallback = False
        if not feasible:
            current_hash = self._hash(current_prompt)
            fallback = next((item for item in evaluated if self._hash(str(item.get("prompt", ""))) == current_hash), None)
            if fallback is None:
                raise RuntimeError("Vote Pareto selection requires the current active prompt in the candidate pool")
            fallback["pareto_feasible"] = True
            fallback["pareto_forced_fallback"] = True
            feasible = [fallback]
            forced_fallback = True

        include_competence = self._uses_competence_depth_pareto_selection()
        include_boundary_error = self._uses_vote_error_pareto_selection()
        fronts_by_item = competence_non_dominated_sort(feasible) if include_competence else non_dominated_sort(
            feasible, include_boundary_error=include_boundary_error
        )
        retained: List[Dict[str, Any]] = []
        for rank, front_indices in enumerate(fronts_by_item):
            distances = compute_crowding_distances(
                feasible, front_indices, include_boundary_error=include_boundary_error,
                include_competence_depth=include_competence,
            )
            front = []
            for index in front_indices:
                item = feasible[index]
                item["pareto_rank"] = rank
                item["pareto_crowding_distance"] = distances.get(index, 0.0)
                front.append(item)
            slots = beam_size - len(retained)
            if slots <= 0:
                continue
            if len(front) <= slots:
                retained.extend(sorted(front, key=lambda item: str(item.get("candidate_id", ""))))
            else:
                if include_competence:
                    retained.extend(sorted(front, key=lambda item: (
                        -float(item.get("pareto_crowding_distance", 0.0) or 0.0),
                        *self._competence_depth_sort_key(item),
                    ))[:slots])
                else:
                    retained.extend(sorted(front, key=self._vote_pareto_crowding_sort_key)[:slots])
                break

        if not retained:
            raise RuntimeError("Vote Pareto selection produced an empty beam")
        retained.sort(key=self._competence_depth_sort_key if include_competence else self._vote_pareto_active_sort_key)
        for item in retained:
            item["pareto_selected"] = True
        for item in evaluated:
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
            metrics["pareto_rank"] = item.get("pareto_rank")
            metrics["pareto_crowding_distance"] = item.get("pareto_crowding_distance")
            metrics["pareto_feasible"] = bool(item.get("pareto_feasible", False))
            metrics["pareto_selected"] = bool(item.get("pareto_selected", False))
            item["metrics"] = metrics
        return retained, {
            "num_pareto_feasible": int(original_feasible_count),
            "num_pareto_infeasible": int(len(evaluated) - original_feasible_count),
            "num_pareto_fronts": int(len(fronts_by_item)),
            "pareto_front0_size": int(len(fronts_by_item[0])) if fronts_by_item else 0,
            "pareto_forced_current_fallback": bool(forced_fallback),
        }

    def _apply_hybrid_soft_guards(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        raw_reward = float(metrics.get("reward", 0.0) or 0.0)
        pivotal_excess = max(
            0.0,
            float(metrics.get("pivotal_loss_rate", 0.0) or 0.0)
            - float(getattr(self.cfg, "pivotal_loss_guard_epsilon", 0.0) or 0.0),
        )
        shared_excess = max(
            0.0,
            float(metrics.get("shared_error_creation_score", 0.0) or 0.0)
            - float(metrics.get("shared_error_rescue_score", 0.0) or 0.0)
            - float(getattr(self.cfg, "shared_error_creation_epsilon", 0.02) or 0.0),
        )
        error_dependence_excess = pivotal_excess + shared_excess
        cycle_excess = 0.0
        if int(metrics.get("behavior_cycle_overlap", 0) or 0) >= int(getattr(self.cfg, "behavior_cycle_min_overlap", 16)):
            cycle_excess = max(
                0.0,
                float(metrics.get("max_behavior_cycle_similarity", 0.0) or 0.0)
                - float(getattr(self.cfg, "behavior_cycle_similarity_threshold", 0.95) or 0.95),
            )
        mechanism_shift_excess = max(
            0.0,
            float(metrics.get("prompt_change_ratio", 0.0) or 0.0)
            - float(getattr(self.cfg, "prompt_max_change_ratio", 0.45) or 0.45),
        )
        if metrics.get("mechanism_contract_passed") is False:
            mechanism_shift_excess = max(mechanism_shift_excess, 1.0)
        if self._is_stable_qd_lineage():
            cycle_excess = 0.0
            mechanism_shift_excess = 0.0
        mild_accuracy_regression = min(
            float(getattr(self.cfg, "catastrophic_target_accuracy_loss_epsilon", 0.05) or 0.05),
            max(0.0, -float(metrics.get("accuracy_delta", 0.0) or 0.0)),
        )
        components = {
            "soft_error_dependence_penalty": float(getattr(self.cfg, "soft_guard_error_dependence_weight", 0.5)) * error_dependence_excess,
            "soft_cycle_penalty": float(getattr(self.cfg, "soft_guard_cycle_weight", 0.2)) * cycle_excess,
            "soft_mechanism_shift_penalty": float(getattr(self.cfg, "soft_guard_mechanism_shift_weight", 0.2)) * mechanism_shift_excess,
            "soft_accuracy_regression_penalty": float(getattr(self.cfg, "soft_guard_accuracy_regression_weight", 0.5)) * mild_accuracy_regression,
        }
        penalty = sum(components.values())
        soft_reasons = []
        if error_dependence_excess > 0.0:
            soft_reasons.append("error_dependence")
        if cycle_excess > 0.0:
            soft_reasons.append("residual_cycle")
        if mechanism_shift_excess > 0.0:
            soft_reasons.append("mechanism_shift")
        if mild_accuracy_regression > 0.0:
            soft_reasons.append("mild_accuracy_regression")
        trajectory_reason = str(metrics.get("rejection_reason", ""))
        trajectory_soft_reasons = {
            "behavior_cycle", "accepted_state_cycle", "rejected_failure_cycle",
            "unsupported_large_prompt_shift", "mechanism_contract_missing",
        }
        if self._is_stable_qd_lineage():
            trajectory_soft_reasons.remove("mechanism_contract_missing")
        if trajectory_reason in trajectory_soft_reasons:
            if trajectory_reason not in soft_reasons:
                soft_reasons.append(trajectory_reason)
            metrics["rejection_reason"] = ""
        metrics.update({
            "raw_reward": raw_reward,
            "soft_guard_penalty": float(penalty),
            "penalized_reward": raw_reward - penalty,
            "soft_guard_reasons": soft_reasons,
            "hard_guard_passed": not bool(metrics.get("rejection_reason", "")),
            **components,
        })
        return metrics

    def _select_hybrid_beam(
        self,
        evaluated: List[Dict[str, Any]],
        beam_size: int,
        current_prompt: str,
        *,
        agent_id: Optional[int] = None,
        epoch_id: Optional[int] = None,
        step_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self._is_stable_qd_lineage():
            for item in evaluated:
                self._attach_stable_mechanism_representation(item)
                item["prompt_hash"] = self._normalized_prompt_hash(str(item.get("prompt", "")))
            retained, summary = select_quality_diversity_archive(
                evaluated, beam_size, self._normalized_prompt_hash(current_prompt), self.cfg
            )
            for item in evaluated:
                item["pareto_selected"] = item in retained
            self.quality_diversity_archive_history.append({
                "event": "quality_diversity_archive", "agent_id": agent_id,
                "epoch": epoch_id, "step": step_id, "niche_count": summary["niche_count"],
                "retained_prompt_hashes": [item.get("prompt_hash", "") for item in retained],
                "retained_sources": [item.get("beam_slot", "") for item in retained],
                "retained_niches": [item.get("qd_niche_key", item.get("metrics", {}).get("qd_niche_key", [])) for item in retained],
            })
            return retained, summary
        _, summary = self._select_vote_pareto_beam(evaluated, len(evaluated), current_prompt)
        current_hash = self._normalized_prompt_hash(current_prompt)
        safe = next(
            (item for item in evaluated if self._normalized_prompt_hash(str(item.get("prompt", ""))) == current_hash),
            None,
        )
        hard_pass = [item for item in evaluated if bool(item.get("pareto_feasible", False))]
        exploit_pool = [item for item in hard_pass if int(item.get("pareto_rank", 999) or 0) == 0 and item is not safe]
        exploit = max(
            exploit_pool,
            key=lambda item: (float(item.get("metrics", {}).get("penalized_reward", item.get("reward", 0.0)) or 0.0), str(item.get("candidate_id", ""))),
            default=None,
        )
        explore_pool = []
        for item in hard_pass:
            if item is safe or item is exploit:
                continue
            if self._candidate_pool_source(item) != "optimizer" and str(item.get("metrics", {}).get("beam_slot", "")) != "explore":
                continue
            metrics = item.get("metrics", {})
            evidence = float(metrics.get("accuracy_delta", 0.0) or 0.0) >= 0.0 or (
                float(metrics.get("depth1_net_delta", 0.0) or 0.0)
                + float(metrics.get("depth2_net_delta", 0.0) or 0.0)
            ) >= 0.0
            if evidence and float(metrics.get("mechanism_signature_distance", 0.0) or 0.0) > 0.0:
                explore_pool.append(item)
        explore = max(
            explore_pool,
            key=lambda item: (
                float(item.get("metrics", {}).get("penalized_reward", item.get("reward", 0.0)) or 0.0)
                + float(item.get("metrics", {}).get("mechanism_novelty_bonus", 0.0) or 0.0),
                str(item.get("candidate_id", "")),
            ),
            default=None,
        )
        retained: List[Dict[str, Any]] = []
        for item, slot in ((exploit, "exploit"), (safe, "safe"), (explore, "explore")):
            if item is None or item in retained:
                continue
            item["beam_slot"] = slot
            retained.append(item)
        for item in sorted(
            hard_pass,
            key=lambda row: -float(row.get("metrics", {}).get("penalized_reward", row.get("reward", 0.0)) or 0.0),
        ):
            if len(retained) >= beam_size:
                break
            if item not in retained:
                item["beam_slot"] = "exploit" if not retained else "safe_fill"
                retained.append(item)
        retained = retained[:beam_size]
        for item in evaluated:
            if item not in retained:
                item["beam_slot"] = "not_retained"
                item["pareto_selected"] = False
            else:
                item["pareto_selected"] = True
        summary.update({
            "safe_slot_occupancy": int(any(item.get("beam_slot") == "safe" for item in retained)),
            "exploit_slot_occupancy": int(any(item.get("beam_slot") == "exploit" for item in retained)),
            "explore_slot_occupancy": int(any(item.get("beam_slot") == "explore" for item in retained)),
        })
        return retained, summary
