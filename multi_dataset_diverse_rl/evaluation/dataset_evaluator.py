"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class DatasetEvaluatorMixin:
    def _summarize_rollout_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        individual_matrix = [list(r.get("individual_correct", [])) for r in rows]
        flat_individual = [int(x) for row in individual_matrix for x in row]
        per_agent_acc = []
        agent_count = max((len(row) for row in individual_matrix), default=0)
        for agent_id in range(agent_count):
            vals = [int(row[agent_id]) for row in individual_matrix if agent_id < len(row)]
            per_agent_acc.append(float(np.mean(vals)) if vals else 0.0)
        vote_acc = float(np.mean([r.get("vote_correct", 0) for r in rows])) if rows else 0.0
        plurality_vote_acc = float(np.mean([
            r.get("plurality_vote_correct", r.get("majority_vote_correct", r.get("vote_correct", 0)))
            for r in rows
        ])) if rows else 0.0
        oracle_acc = float(np.mean([1 if any(int(x) for x in r.get("individual_correct", [])) else 0 for r in rows])) if rows else 0.0
        rescue_available_rate = float(
            np.mean([
                1 if int(r.get("vote_correct", 0)) == 0 and any(int(x) for x in r.get("individual_correct", [])) else 0
                for r in rows
            ])
        ) if rows else 0.0
        correct_disagreement_rate = float(
            np.mean([
                1
                if len({str(a).strip() for a in r.get("vote_counts", {}).keys() if str(a).strip()}) > 1
                and any(int(x) for x in r.get("individual_correct", []))
                else 0
                for r in rows
            ])
        ) if rows else 0.0
        pair_double_fault: List[float] = []
        pair_covariance: List[float] = []
        if agent_count >= 2 and rows:
            for left in range(agent_count):
                for right in range(left + 1, agent_count):
                    pairs = [row for row in individual_matrix if left < len(row) and right < len(row)]
                    if not pairs:
                        continue
                    left_errors = np.array([1.0 - float(row[left]) for row in pairs], dtype=float)
                    right_errors = np.array([1.0 - float(row[right]) for row in pairs], dtype=float)
                    pair_double_fault.append(float(np.mean(left_errors * right_errors)))
                    pair_covariance.append(float(np.mean(left_errors * right_errors) - np.mean(left_errors) * np.mean(right_errors)))
        same_wrong_pair_values = []
        dominant_wrong_sizes = []
        boundary_conditional_errors = []
        pivotal_fix_values = []
        pivotal_hold_values = []
        shared_rescue_values = []
        shared_creation_values = []
        correct_depths = []
        plurality_opportunity_values: List[int] = []
        plurality_hold_values: List[int] = []
        for row in rows:
            flags = [int(value) for value in row.get("individual_correct", [])]
            n = len(flags)
            correct_count = sum(flags)
            wrong_count = n - correct_count
            correct_depths.append(correct_count)
            plurality_opportunity_values.extend(
                int(value) for value in row.get("plurality_pivotal_fix_opportunity_per_agent", [])
            )
            plurality_hold_values.extend(
                int(value) for value in row.get("plurality_pivotal_hold_per_agent", [])
            )
            largest_wrong = int(row.get("largest_wrong_vote_count", 0) or 0)
            dominant_wrong_sizes.append(largest_wrong)
            vote_counts = row.get("vote_counts", {}) if isinstance(row.get("vote_counts", {}), dict) else {}
            all_same_pairs = sum(int(count) * (int(count) - 1) / 2 for count in vote_counts.values())
            gold_same_pairs = correct_count * (correct_count - 1) / 2
            same_wrong_pair_values.append(
                max(0.0, all_same_pairs - gold_same_pairs) / max(1.0, n * (n - 1) / 2)
            )
            per_row_pivotal_fix = []
            per_row_pivotal_hold = []
            per_row_boundary_error = []
            per_row_shared_rescue = []
            per_row_shared_creation = []
            for agent_id, correct in enumerate(flags):
                peer_wrong = wrong_count - int(not correct)
                near_shared_boundary = peer_wrong >= max(1, (n - 1) // 2)
                if near_shared_boundary:
                    per_row_boundary_error.append(float(not correct))
                    per_row_shared_rescue.append(float(correct))
                    per_row_shared_creation.append(float(not correct))
                if not correct:
                    per_row_pivotal_fix.append(float(
                        (not bool(row.get("vote_correct", 0)))
                        and correct_count + 1 > largest_wrong
                    ))
                else:
                    per_row_pivotal_hold.append(float(correct_count - 1 <= largest_wrong))
            if per_row_pivotal_fix:
                pivotal_fix_values.append(float(np.mean(per_row_pivotal_fix)))
            if per_row_pivotal_hold:
                pivotal_hold_values.append(float(np.mean(per_row_pivotal_hold)))
            if per_row_boundary_error:
                boundary_conditional_errors.append(float(np.mean(per_row_boundary_error)))
                shared_rescue_values.append(float(np.mean(per_row_shared_rescue)))
                shared_creation_values.append(float(np.mean(per_row_shared_creation)))
        triple_joint_error_rate = float(np.mean([int((agent_count - depth) >= 3) for depth in correct_depths])) if correct_depths else 0.0
        shared_rescue_rate = float(np.mean(shared_rescue_values)) if shared_rescue_values else 0.0
        shared_creation_rate = float(np.mean(shared_creation_values)) if shared_creation_values else 0.0
        ordered_acc = sorted(per_agent_acc)
        min_acc = ordered_acc[0] if ordered_acc else 0.0
        bottom2 = float(np.mean(ordered_acc[: min(2, len(ordered_acc))])) if ordered_acc else 0.0
        bottom3 = float(np.mean(ordered_acc[: min(3, len(ordered_acc))])) if ordered_acc else 0.0
        max_acc = ordered_acc[-1] if ordered_acc else 0.0
        minority_rescue_counts = [0 for _ in range(agent_count)]
        unique_correct_counts = [0 for _ in range(agent_count)]
        for row in rows:
            flags = [int(value) for value in row.get("individual_correct", [])]
            for agent_id, correct in enumerate(flags):
                if correct and not int(row.get("vote_correct", 0)):
                    minority_rescue_counts[agent_id] += 1
                if correct and sum(flags) == 1:
                    unique_correct_counts[agent_id] += 1
        rescue_total = sum(minority_rescue_counts)
        rescue_shares = [count / rescue_total if rescue_total else 0.0 for count in minority_rescue_counts]
        result = {
            "size": len(rows),
            "num_test_samples": len(rows),
            "vote_acc": vote_acc,
            "plurality_vote_acc": plurality_vote_acc,
            "majority_vote_acc": float(np.mean([r.get("majority_vote_correct", r.get("vote_correct", 0)) for r in rows])) if rows else 0.0,
            "weighted_vote_acc": float(np.mean([r.get("weighted_vote_correct", 0) for r in rows])) if rows else 0.0,
            "mean_individual_acc": float(np.mean(flat_individual)) if flat_individual else 0.0,
            "best_individual_acc": float(max(per_agent_acc)) if per_agent_acc else 0.0,
            "per_agent_acc": per_agent_acc,
            "min_individual_acc": min_acc,
            "bottom2_mean_acc": bottom2,
            "bottom3_mean_acc": bottom3,
            "max_individual_acc": max_acc,
            "individual_acc_std": float(np.std(per_agent_acc)) if per_agent_acc else 0.0,
            "best_minus_worst_gap": max_acc - min_acc,
            "best_minus_bottom2_gap": max_acc - bottom2,
            "minority_rescue_count_per_agent": minority_rescue_counts,
            "unique_correct_count_per_agent": unique_correct_counts,
            "minority_rescue_share_per_agent": rescue_shares,
            "max_minority_rescue_share": max(rescue_shares, default=0.0),
            "minority_rescue_hhi": sum(value * value for value in rescue_shares),
            "oracle_acc": oracle_acc,
            "all_wrong_rate": 1.0 - oracle_acc,
            "aggregation_gap": float(oracle_acc - plurality_vote_acc),
            "oracle_minus_plurality_vote": float(oracle_acc - plurality_vote_acc),
            "rescue_available_rate": rescue_available_rate,
            "correct_disagreement_rate": correct_disagreement_rate,
            "mean_useful_diversity": float(np.mean([r.get("useful_diversity", 0.0) for r in rows])) if rows else 0.0,
            "mean_vote_margin": float(np.mean([r.get("normalized_vote_margin", -1.0) for r in rows])) if rows else -1.0,
            "mean_plurality_margin_votes": float(np.mean([r.get("plurality_margin_votes", 0.0) for r in rows])) if rows else 0.0,
            "mean_normalized_plurality_margin": float(np.mean([r.get("normalized_plurality_margin", -1.0) for r in rows])) if rows else -1.0,
            "strict_plurality_win_rate": float(np.mean([int(bool(r.get("strict_plurality_win", False))) for r in rows])) if rows else 0.0,
            "plurality_top_tie_rate": float(np.mean([int(bool(r.get("plurality_gold_top_tied", False))) for r in rows])) if rows else 0.0,
            "plurality_pivotal_fix_opportunity_rate": float(np.mean(plurality_opportunity_values)) if plurality_opportunity_values else 0.0,
            "plurality_pivotal_fix_rate": float(np.mean(plurality_hold_values)) if plurality_hold_values else 0.0,
            "plurality_pivotal_hold_rate": float(np.mean(plurality_hold_values)) if plurality_hold_values else 0.0,
            "mean_boundary_useful_diversity": float(np.mean([r.get("boundary_useful_diversity", 0.0) for r in rows])) if rows else 0.0,
            "aggregation_mode": str(getattr(self.cfg, "aggregation_mode", "majority") or "majority"),
            "requested_aggregation_mode": str(getattr(self.cfg, "aggregation_mode", "majority") or "majority"),
            "effective_aggregation_mode": canonical_aggregation_mode(str(getattr(self.cfg, "aggregation_mode", "majority") or "majority")),
            "plurality_boundary_version": PLURALITY_BOUNDARY_VERSION,
            "vote_tie_rate": float(np.mean([1 if r.get("vote_tie", False) else 0 for r in rows])) if rows else 0.0,
            "mean_embedding_diversity": float(np.mean([r.get("embedding_diversity", 0.0) for r in rows])) if rows else 0.0,
            "mean_embedding_overlap": float(np.mean([r.get("mean_embedding_overlap", 0.0) for r in rows])) if rows else 0.0,
            "mean_invalid_rate": float(np.mean([r.get("invalid_rate", 0.0) for r in rows])) if rows else 0.0,
            "mean_pairwise_double_fault": float(np.mean(pair_double_fault)) if pair_double_fault else 0.0,
            "mean_pairwise_error_covariance": float(np.mean(pair_covariance)) if pair_covariance else 0.0,
            "same_wrong_pair_rate": float(np.mean(same_wrong_pair_values)) if same_wrong_pair_values else 0.0,
            "triple_joint_error_rate": triple_joint_error_rate,
            "majority_failure_tail_rate": float(np.mean([int((agent_count - depth) >= ((agent_count // 2) + 1)) for depth in correct_depths])) if correct_depths else 0.0,
            **{
                f"coverage_depth_c{depth}": float(np.mean([int(value >= depth) for value in correct_depths])) if correct_depths else 0.0
                for depth in range(1, 6)
            },
            **{f"correct_agent_count_{depth}": int(sum(value == depth for value in correct_depths)) for depth in range(6)},
            "c1_minus_c2": float(np.mean([int(value >= 1) - int(value >= 2) for value in correct_depths])) if correct_depths else 0.0,
            "c2_minus_c3": float(np.mean([int(value >= 2) - int(value >= 3) for value in correct_depths])) if correct_depths else 0.0,
            "c2_minus_plurality_vote": float(np.mean([int(value >= 2) for value in correct_depths])) - plurality_vote_acc if correct_depths else 0.0,
            "c3_minus_plurality_vote": float(np.mean([int(value >= 3) for value in correct_depths])) - plurality_vote_acc if correct_depths else 0.0,
            "specialization_strength_final": float(getattr(self, "specialization_strength", 0.0)),
            "mean_specialization_strength": float(np.mean(getattr(self, "specialization_strength_history", [0.0]))) if getattr(self, "specialization_strength_history", None) else 0.0,
            "first_nonzero_specialization_epoch": getattr(self, "first_nonzero_specialization_epoch", None),
            "effective_specialization_epoch_count": int(getattr(self, "effective_specialization_epoch_count", 0)),
            "max_specialization_strength": max(getattr(self, "specialization_strength_history", [0.0]) or [0.0]),
            "progressive_stage_exercised": int(getattr(self, "effective_specialization_epoch_count", 0)) >= int(getattr(self.cfg, "competence_min_effective_specialization_epochs", 1)),
            "progressive_stage_not_exercised_reason": (
                "" if int(getattr(self, "effective_specialization_epoch_count", 0)) >= int(getattr(self.cfg, "competence_min_effective_specialization_epochs", 1))
                else "activation_after_final_epoch" if float(getattr(self, "specialization_strength", 0.0)) > 0.0
                else "never_activated"
            ),
            "depth1_guard_rejection_count": int(getattr(self, "depth1_guard_rejection_count", 0)),
            "catastrophic_accuracy_guard_rejection_count": int(getattr(self, "catastrophic_accuracy_guard_rejection_count", 0)),
            "soft_error_dependence_penalty_count": int(getattr(self, "soft_error_dependence_penalty_count", 0)),
            "soft_cycle_penalty_count": int(getattr(self, "soft_cycle_penalty_count", 0)),
            "soft_mechanism_shift_penalty_count": int(getattr(self, "soft_mechanism_shift_penalty_count", 0)),
            "exploration_candidate_count": int(getattr(self, "exploration_candidate_count", 0)),
            "exploration_slot_occupancy_rate": float(np.clip(
                float(getattr(self, "exploration_slot_occupancy_count", 0))
                / max(1, int(getattr(self, "total_agent_update_count", 0) or len(getattr(self, "mechanism_signature_history", [])))),
                0.0, 1.0,
            )),
            "exploration_to_active_conversion_count": int(getattr(self, "exploration_to_active_conversion_count", 0)),
            "prompt_overlength_rejection_count": int(getattr(self, "prompt_overlength_rejection_count", 0)),
            "truncated_prompt_count": int(getattr(self, "truncated_prompt_count", 0)),
            "mean_boundary_conditional_error": float(np.mean(boundary_conditional_errors)) if boundary_conditional_errors else 0.0,
            "mean_pivotal_fix_rate": float(np.mean(pivotal_fix_values)) if pivotal_fix_values else 0.0,
            "mean_pivotal_hold_rate": float(np.mean(pivotal_hold_values)) if pivotal_hold_values else 0.0,
            "shared_error_rescue_rate": shared_rescue_rate,
            "shared_error_creation_rate": shared_creation_rate,
            "boundary_shared_error_net_gain": shared_rescue_rate - 1.5 * shared_creation_rate,
            "dominant_wrong_cluster_size": float(np.mean(dominant_wrong_sizes)) if dominant_wrong_sizes else 0.0,
            "gold_vs_largest_wrong_margin": float(np.mean([r.get("normalized_vote_margin", -1.0) for r in rows])) if rows else -1.0,
            **summarize_vote_conversion(rows),
        }
        if self._residual_specialization_enabled():
            result.update({
                "capability_profile_per_agent": [dict(agent.capability_profile) for agent in self.agents],
                "vote_context_profile_per_agent": [dict(agent.vote_context_profile) for agent in self.agents],
                "capability_profile_update_count_per_agent": [int(agent.capability_profile_update_count) for agent in self.agents],
                **self._capability_specialization_diagnostics(),
            })
        if self._is_rollout_qd_method():
            signatures = [
                str(row.get("rollout_signature_hash", ""))
                for row in getattr(self, "accepted_rollout_archive", [])
                if str(row.get("rollout_signature_hash", ""))
            ]
            counts = Counter(signatures)
            latest = dict(getattr(self, "latest_joint_team_metrics", {}) or {})
            selected = dict(latest.get("selected_metrics", {}) or {})
            result.update({
                "method_version": str(self.cfg.method_version),
                "mechanism_diversity_enabled": False,
                "mechanism_metadata_required": False,
                "mechanism_distance_used_for_selection": False,
                "mechanism_based_decision_count": int(getattr(self, "mechanism_based_decision_count", 0)),
                "capability_labeling_enabled": False,
                "capability_profile_per_agent": None,
                "top_capability_family_per_agent": None,
                "prompt_text_diversity_used": False,
                "rollout_embedding_diversity": float(selected.get("rollout_diversity_score", result.get("mean_embedding_diversity", 0.0)) or 0.0),
                "correct_set_rollout_distance": float(selected.get("correct_set_rollout_distance", 0.0) or 0.0),
                "useful_wrong_answer_dispersion": float(selected.get("useful_wrong_answer_dispersion", result.get("same_wrong_pair_rate", 0.0)) or 0.0),
                "rollout_signature_count": len(counts),
                "duplicate_rollout_signature_count": int(sum(max(0, count - 1) for count in counts.values())),
                "joint_team_solver_call_count": int(latest.get("joint_team_solver_call_count", 0) or 0),
                "legacy_beam_refresh_call_count": int(getattr(self, "legacy_beam_refresh_call_count", 0)),
                "candidate_channel_funnel": json.loads(json.dumps(getattr(self, "candidate_channel_funnel", {}))),
                "active_candidate_source_by_agent": dict(getattr(self, "active_candidate_source_by_agent", {})),
            })
            if self._is_state_conditioned_method():
                result.update(state_dataset_metrics(rows))
                result.update({
                    "state_conditioned_enabled": True,
                    "state_coverage_enabled": bool(getattr(self.cfg, "state_coverage_enabled", True)),
                    "state_c2_wrong_split_enabled": bool(getattr(self.cfg, "state_c2_wrong_split_enabled", True)),
                    "state_trace_tiebreak_enabled": bool(getattr(self.cfg, "state_trace_tiebreak_enabled", True)),
                    "composite_rollout_distance_used_for_selection": False,
                    "trace_diversity_role": "diagnostic_or_last_tiebreak_only",
                    "coverage_case_assignment_per_agent": dict(
                        getattr(self, "coverage_case_assignment_per_agent", {})
                    ),
                    "c0_rescue_count_per_agent": dict(
                        getattr(self, "c0_rescue_count_per_agent", {})
                    ),
                    "c1_deepening_count_per_agent": dict(
                        getattr(self, "c1_deepening_count_per_agent", {})
                    ),
                    "state_search_diagnostics": dict(
                        getattr(self, "state_search_diagnostics", {})
                    ),
                })
        if self._is_v82_hybrid():
            final_signatures = []
            for agent in self.agents:
                metrics = agent.prompt_beam[0].get("metrics", {}) if agent.prompt_beam else {}
                signature = list(metrics.get("mechanism_signature", []))
                if not signature:
                    signature = list(self.mechanism_signature_by_prompt_hash.get(
                        self._normalized_prompt_hash(agent.current_prompt), []
                    ))
                final_signatures.append(signature)
            encoded = [json.dumps(value, ensure_ascii=True, separators=(",", ":")) for value in final_signatures]
            counts = Counter(encoded)
            pair_distances = [
                mechanism_signature_distance(final_signatures[left], final_signatures[right])
                for left in range(len(final_signatures))
                for right in range(left + 1, len(final_signatures))
            ]
            result.update({
                "distinct_final_mechanism_signature_count": len(counts),
                "dominant_final_mechanism_signature_share": max(counts.values(), default=0) / max(1, len(final_signatures)),
                "mean_pairwise_mechanism_signature_distance": float(np.mean(pair_distances)) if pair_distances else 0.0,
                "final_mechanism_signatures": final_signatures,
            })
        if self._is_stable_qd_lineage():
            latest = dict(getattr(self, "latest_joint_team_metrics", {}) or {})
            statuses = [str(agent.lineage_state.get("lineage_status", "uncommitted")) for agent in self.agents]
            lineage_drifts = []
            for agent in self.agents:
                state = agent.lineage_state
                lineage_drifts.append(0.0 if state.get("lineage_status") != "committed" else float(
                    state.get("last_lineage_drift", 0.0) or 0.0
                ))
            mean_behavior = float(latest.get("mean_behavior_distance", 0.0) or 0.0)
            min_behavior = float(latest.get("min_behavior_distance", 0.0) or 0.0)
            mean_mechanism = float(latest.get("mean_mechanism_distance", 0.0) or 0.0)
            mean_drift = float(np.mean(lineage_drifts)) if lineage_drifts else 0.0
            task_rate = float(np.clip(self.task_repair_niche_occupancy_count / max(1, self.total_agent_update_count), 0.0, 1.0))
            mechanism_rate = float(np.clip(self.mechanism_niche_occupancy_count / max(1, self.total_agent_update_count), 0.0, 1.0))
            exploration_rate = float(result["exploration_slot_occupancy_rate"])
            assert 0.0 <= exploration_rate <= 1.0
            result.update({
                "mean_inter_agent_behavior_distance": mean_behavior,
                "min_inter_agent_behavior_distance": min_behavior,
                "mean_inter_agent_mechanism_distance": mean_mechanism,
                "mean_intra_agent_lineage_drift": mean_drift,
                "max_intra_agent_lineage_drift": max(lineage_drifts, default=0.0),
                "stable_specialization_score": mean_behavior + 0.5 * min_behavior + 0.25 * mean_mechanism - 0.5 * mean_drift,
                "uncommitted_agent_count": statuses.count("uncommitted"),
                "provisional_agent_count": statuses.count("provisional"),
                "committed_agent_count": statuses.count("committed"),
                "lineage_commit_count": sum(int(agent.lineage_state.get("lineage_commit_count", 0)) for agent in self.agents),
                "lineage_switch_attempt_count": sum(int(agent.lineage_state.get("lineage_switch_attempt_count", 0)) for agent in self.agents),
                "lineage_switch_commit_count": sum(int(agent.lineage_state.get("lineage_switch_commit_count", 0)) for agent in self.agents),
                "lineage_switch_cancel_count": sum(int(agent.lineage_state.get("lineage_switch_cancel_count", 0)) for agent in self.agents),
                "lineage_committed_but_not_exercised": sum(
                    int(
                        agent.lineage_state.get("lineage_status") == "committed"
                        and int(agent.lineage_state.get("lineage_anchor_epoch", -1)) >= int(self.cfg.epochs)
                    )
                    for agent in self.agents
                ),
                "peer_collapse_soft_count": int(self.peer_collapse_soft_count),
                "peer_collapse_hard_rejection_count": int(self.peer_collapse_hard_rejection_count),
                "joint_team_combination_count": int(latest.get("combination_count", 0) or 0),
                "joint_team_feasible_count": int(latest.get("feasible_count", 0) or 0),
                "joint_team_quality_frontier_count": int(latest.get("quality_frontier_count", 0) or 0),
                "joint_team_quality_floor_feasible_count": int(latest.get("quality_floor_feasible_count", latest.get("feasible_count", 0)) or 0),
                "joint_team_final_candidate_count": int(latest.get("final_candidate_team_count", latest.get("quality_frontier_count", 0)) or 0),
                "joint_team_change_limit_rejection_count": int(latest.get("combination_rejected_by_change_limit_count", 0) or 0),
                "joint_team_fold_quality_rejection_count": int(latest.get("fold_quality_rejection_count", 0) or 0),
                "joint_team_selected_diversity_score": float(latest.get("team_diversity_score", 0.0) or 0.0),
                "joint_team_selected_stability_score": float(latest.get("stable_team_score", 0.0) or 0.0),
                "active_from_incumbent_count": list(latest.get("selected_beam_sources", [])).count("incumbent"),
                "active_from_task_repair_niche_count": list(latest.get("selected_beam_sources", [])).count("task_repair_niche"),
                "active_from_mechanism_niche_count": list(latest.get("selected_beam_sources", [])).count("mechanism_niche"),
                "mechanism_niche_occupancy_rate": mechanism_rate,
                "task_repair_niche_occupancy_rate": task_rate,
                "candidate_starvation_count": int(self.candidate_starvation_count),
                "mechanism_starvation_count": int(self.mechanism_starvation_count),
                "search_branch_starvation_count": int(self.search_branch_starvation_count),
                "probation_to_safe_conversion_count": int(self.probation_to_safe_conversion_count),
                "probation_expired_count": int(self.probation_expired_count),
                "refill_requirements_unmet_count": int(self.refill_requirements_unmet_count),
                "method_version": self.cfg.method_version,
                "active_team_selector_version": self.cfg.active_team_selector_version,
                "lineage_policy_version": self.cfg.lineage_policy_version,
                "mechanism_distance_version": self.cfg.mechanism_distance_version,
                "candidate_refill_version": self.cfg.candidate_refill_version,
                "archive_policy_version": self.cfg.archive_policy_version,
                "joint_quality_filter_version": self.cfg.joint_quality_filter_version,
                "probe_stability_version": self.cfg.probe_stability_version,
                "parent_selection_version": self.cfg.parent_selection_version,
                "candidate_channel_funnel": json.loads(json.dumps(getattr(self, "candidate_channel_funnel", {}))),
                **{
                    key: latest.get(key)
                    for key in (
                        "safe_archive_profile_current_count_per_agent",
                        "safe_archive_unprofiled_count_per_agent",
                        "safe_archive_profile_fraction_per_agent",
                        "dirty_shortlist_excluded_count_per_agent",
                        "oldest_unprofiled_safe_age_epochs_per_agent",
                        "representative_profile_current_count_per_agent",
                        "representative_mean_behavior_distance",
                        "representative_min_behavior_distance",
                        "representative_behavior_span",
                    )
                },
            })
        initial_probe = dict(getattr(self, "initial_competence_probe_metrics", {}) or {})
        final_probe = dict(getattr(self, "latest_competence_probe_metrics", {}) or initial_probe)
        if initial_probe:
            for label, key in (
                ("bottom2", "bottom2_mean_acc"), ("mean_acc", "mean_individual_acc"),
                ("c1", "coverage_depth_c1"), ("c2", "coverage_depth_c2"),
            ):
                initial_value = float(initial_probe.get(key, 0.0) or 0.0)
                final_value = float(final_probe.get(key, initial_value) or 0.0)
                result[f"initial_competence_probe_{label}"] = initial_value
                result[f"final_competence_probe_{label}"] = final_value
                result[f"competence_probe_{label}_gain"] = final_value - initial_value
            baseline_gap = float(initial_probe.get("oracle_acc", 0.0) or 0.0) - float(
                initial_probe.get("plurality_vote_acc", initial_probe.get("vote_acc", 0.0)) or 0.0
            )
            initial_c1 = float(initial_probe.get("coverage_depth_c1", 0.0) or 0.0)
            final_c1 = float(final_probe.get("coverage_depth_c1", initial_c1) or 0.0)
            result.update({
                "baseline_aggregation_gap": baseline_gap,
                "oracle_preserving_gap_reduction": bool(
                    float(result.get("aggregation_gap", 0.0)) < baseline_gap
                    and final_c1 >= initial_c1 - float(getattr(self.cfg, "competence_c1_guard_epsilon", 0.01))
                ),
            })
        return result

    async def evaluate_dataset(self, data: List[Dict[str, str]], split_name: str = "test") -> Dict[str, Any]:
        prompts = self._active_prompt_list()

        async def evaluate_one(idx: int, ex: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            q = ex["question"]
            gold = self.task_spec.parse_gold(ex["answer"], q)
            traces, answers = await self.solve_with_prompts(q, prompts)
            question_hash = self._hash(q)
            self._record_solver_rollouts(question_hash, prompts, traces, answers, source=f"{split_name}_rollout")
            metrics = self.compute_rollout_metrics(traces, answers, gold, prompts, question_hash=question_hash)
            option_counter = getattr(self.task_spec, "option_count", None)
            option_count = int(option_counter(q) or 0) if callable(option_counter) else 0
            row = {
                "index": idx,
                "question_hash": question_hash,
                "option_count": option_count,
                **metrics,
            }
            agent_correct = [int(x) for x in metrics.get("individual_correct", [])]
            prediction = {
                "index": idx,
                "sample_id": idx,
                "question_hash": question_hash,
                "question": q,
                "option_count": option_count,
                "vote_answer": metrics.get("vote_answer", ""),
                "plurality_vote_answer": metrics.get("plurality_vote_answer", metrics.get("vote_answer", "")),
                "majority_vote_answer": metrics.get("majority_vote_answer", metrics.get("vote_answer", "")),
                "weighted_vote_answer": metrics.get("weighted_vote_answer", ""),
                "gold": gold,
                "agent_answers": list(answers),
                "agent_correct": agent_correct,
                "vote_correct": int(metrics.get("vote_correct", 0)),
                "plurality_vote_correct": int(metrics.get("plurality_vote_correct", metrics.get("vote_correct", 0))),
                "majority_vote_correct": int(metrics.get("majority_vote_correct", metrics.get("vote_correct", 0))),
                "weighted_vote_correct": int(metrics.get("weighted_vote_correct", 0)),
                "aggregation_mode": metrics.get("aggregation_mode", "majority"),
                "requested_aggregation_mode": metrics.get("requested_aggregation_mode", metrics.get("aggregation_mode", "majority")),
                "effective_aggregation_mode": metrics.get("effective_aggregation_mode", "plurality"),
                "aggregation_fallback": metrics.get("aggregation_fallback", ""),
                "vote_tie": bool(metrics.get("vote_tie", False)),
                "tie_candidates": metrics.get("tie_candidates", []),
                "vote_counts": metrics.get("vote_counts", {}),
                "gold_vote_count": int(metrics.get("gold_vote_count", 0)),
                "largest_wrong_vote_count": int(metrics.get("largest_wrong_vote_count", 0)),
                "correct_agent_count": int(metrics.get("correct_agent_count", 0)),
                "max_wrong_vote_count": int(metrics.get("max_wrong_vote_count", 0)),
                "gold_plurality_margin": int(metrics.get("gold_plurality_margin", 0)),
                "oracle_correct": int(metrics.get("oracle_correct", 0)),
                "gold_in_top_tie": bool(metrics.get("gold_in_top_tie", False)),
                "top_tie_size": int(metrics.get("top_tie_size", 0)),
                "invalid_agent_count": int(metrics.get("invalid_agent_count", 0)),
                "vote_normalization_anomaly": bool(metrics.get("vote_normalization_anomaly", False)),
                "plurality_margin_votes": int(metrics.get("plurality_margin_votes", 0)),
                "normalized_plurality_margin": float(metrics.get("normalized_plurality_margin", -1.0)),
                "normalized_vote_margin": float(metrics.get("normalized_vote_margin", -1.0)),
                "boundary_useful_diversity": float(metrics.get("boundary_useful_diversity", 0.0)),
                "tie_break_method": metrics.get("tie_break_method", ""),
                "weighted_vote_scores": metrics.get("weighted_vote_scores", {}),
                "weighted_vote_agent_weights": metrics.get("weighted_vote_agent_weights", []),
                "any_correct": int(metrics.get("any_correct", 0)),
                "useful_diversity": float(metrics.get("useful_diversity", 0.0)),
                "embedding_diversity": float(metrics.get("embedding_diversity", 0.0)),
                "mean_embedding_overlap": float(metrics.get("mean_embedding_overlap", 0.0)),
                "invalid_rate": float(metrics.get("invalid_rate", 0.0)),
                "agents": [
                    {
                        "agent_id": i,
                        "prompt_hash": self._hash(prompts[i]),
                        "trace": traces[i],
                        "answer": answers[i],
                        "correct": agent_correct[i] if i < len(agent_correct) else 0,
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
        return self._summarize_rollout_rows(rows)
