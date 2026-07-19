"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class JointControllerMixin:
    async def select_joint_active_team(self, probe_data: List[Dict[str, str]], *, epoch: int) -> Dict[str, Any]:
        if not self._is_stable_qd_lineage():
            return {"enabled": False, "combination_count": 0}
        beams: List[List[Dict[str, Any]]] = []
        for agent_id, agent in enumerate(self.agents):
            archive = list(getattr(agent, "safe_qd_archive", []) or agent.prompt_beam)
            profiled_archive = []
            for item in archive:
                metrics = item.get("metrics", {})
                profile = await self._evaluate_prompt_on_stable_probe(
                    agent_id, str(item.get("prompt", agent.current_prompt)), probe_data,
                    metrics.get("mechanism_steps", metrics.get("mechanism_signature", [])),
                )
                candidate = dict(item)
                candidate_metrics = dict(metrics)
                candidate_metrics["behavior_profile"] = build_prompt_static_profile(
                    profile.get("answer_vector", []), profile.get("correctness_vector", [])
                )
                candidate["metrics"] = candidate_metrics
                profiled_archive.append(candidate)
            agent.safe_qd_archive = profiled_archive
            self._refresh_joint_representatives(agent)
            agent_profiles = []
            for beam_index, item in enumerate(agent.prompt_beam):
                metrics = item.get("metrics", {})
                profile = await self._evaluate_prompt_on_stable_probe(
                    agent_id, str(item.get("prompt", agent.current_prompt)), probe_data,
                    metrics.get("mechanism_steps", metrics.get("mechanism_signature", [])),
                )
                profile.update({
                    "beam_index": beam_index,
                    "beam_source": str(item.get("beam_slot", metrics.get("beam_slot", "incumbent"))),
                })
                self.behavior_profile_by_prompt_hash[profile["prompt_hash"]] = dict(profile)
                agent_profiles.append(profile)
            beams.append(agent_profiles)
        question_hashes = beams[0][0]["question_hashes"]
        gold_answers = beams[0][0]["gold_answers"]
        teams = enumerate_joint_teams(
            beams, gold_answers, question_hashes,
            vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
            tie_break_method=self.cfg.vote_tie_break, seed=self.cfg.seed,
        )
        incumbent_indices = []
        for agent_id, agent in enumerate(self.agents):
            current_hash = self._normalized_prompt_hash(agent.current_prompt)
            incumbent_indices.append(next((index for index, profile in enumerate(beams[agent_id]) if profile["prompt_hash"] == current_hash), 0))
        incumbent = next(team for team in teams if team["beam_indices"] == incumbent_indices)
        initial_per_agent = list(self.initial_competence_probe_metrics.get("per_agent_acc", incumbent["per_agent_acc"]))
        initial_profiles = list(self.initial_competence_probe_metrics.get("behavior_profiles", []))
        initial_anchor = (
            team_quality_metrics(
                initial_profiles, gold_answers, question_hashes,
                vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
                tie_break_method=self.cfg.vote_tie_break, seed=self.cfg.seed,
            )
            if len(initial_profiles) == len(self.agents)
            else incumbent
        )
        initial_anchor["prompt_hashes"] = list(
            getattr(self, "initial_active_prompt_hashes", [])
            or [str(profile.get("prompt_hash", "")) for profile in initial_profiles]
        )
        incumbent["prompt_hashes"] = team_prompt_hashes(incumbent)
        new_anchors = []
        if not getattr(self, "quality_anchor_archive", []):
            new_anchors.append(build_quality_anchor(
                initial_anchor, epoch=0, created_order=int(getattr(self, "quality_anchor_created_count", 0)),
            ))
            self.quality_anchor_created_count = int(getattr(self, "quality_anchor_created_count", 0)) + 1
        new_anchors.append(build_quality_anchor(
            incumbent, epoch=epoch, created_order=int(getattr(self, "quality_anchor_created_count", 0)),
        ))
        self.quality_anchor_created_count = int(getattr(self, "quality_anchor_created_count", 0)) + 1
        anchor_objects = update_quality_anchor_archive(
            getattr(self, "quality_anchor_archive", []), new_anchors,
            capacity=int(self.cfg.quality_anchor_archive_size),
        )
        self.quality_anchor_archive = [anchor.to_dict() for anchor in anchor_objects]
        joint_selection = select_stable_joint_team(
            teams, incumbent, initial_per_agent,
            [agent.lineage_state for agent in self.agents], len(probe_data), self.cfg,
            gold_answers=gold_answers, question_hashes=question_hashes,
            vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
            tie_break_method=self.cfg.vote_tie_break, seed=self.cfg.seed,
            change_limit=self._current_joint_change_limit(epoch),
            quality_anchors=anchor_objects,
        )
        selected = joint_selection["selected"]
        self.peer_collapse_soft_count += int(joint_selection["selected_has_soft_peer_collapse"])
        self.peer_collapse_hard_rejection_count += int(joint_selection["hard_rejection_count"])
        selected_sources, changed_count = [], 0
        for agent_id, beam_index in enumerate(selected["beam_indices"]):
            agent = self.agents[agent_id]
            chosen = agent.prompt_beam[beam_index]
            old_hash = self._normalized_prompt_hash(agent.current_prompt)
            agent.prompt_beam = [chosen] + [item for index, item in enumerate(agent.prompt_beam) if index != beam_index]
            agent.current_prompt = str(chosen["prompt"])
            changed_count += int(old_hash != self._normalized_prompt_hash(agent.current_prompt))
            selected_sources.append(str(chosen.get("beam_slot", chosen.get("metrics", {}).get("beam_slot", "incumbent"))))
            selected_profile = selected["prompt_profiles"][agent_id]
            selected_profile["behavior_profile"] = selected["behavior_profiles"][agent_id]
            selected_profile["cross_fold_diversity_gap"] = float(selected.get("cross_fold_diversity_gap", 0.0))
            selected_profile["fold_quality_gate_passed"] = bool(selected.get("fold_quality_gate_passed", True))
            per_agent_fold_gaps = list(selected.get("per_agent_cross_fold_behavior_gap", []))
            selected_profile["cross_fold_behavior_gap"] = (
                float(per_agent_fold_gaps[agent_id]) if agent_id < len(per_agent_fold_gaps) else 0.0
            )
            fold_profiles = list(selected.get("per_agent_fold_specialization_profiles", []))
            selected_profile["fold_specialization_profiles"] = [
                list(fold[agent_id]) for fold in fold_profiles if agent_id < len(fold)
            ]
            selected_profile["fold_behavior_stable"] = bool(
                selected_profile["cross_fold_behavior_gap"] <= float(self.cfg.qd_readiness_max_fold_gap)
            )
            selected_drift = selected_profile.get("lineage_drift", {})
            agent.lineage_state["last_lineage_drift"] = float(selected_drift.get("lineage_drift", 0.0) or 0.0)
            lineage_record = update_lineage_state(
                agent.lineage_state,
                selected_profile,
                epoch=epoch,
                quality_gate_passed=bool(selected.get("fold_quality_gate_passed", True)),
                config=self.cfg,
            )
            agent.lineage_state = {key: value for key, value in lineage_record.items() if key not in {"old_status", "new_status", "reason"}}
            self.lineage_history.append({"epoch": epoch, "agent_id": agent_id, **lineage_record})
        record = {
            "epoch": epoch, "combination_count": len(teams),
            "representative_count_per_agent": [len(beam) for beam in beams],
            "theoretical_combination_count": int(np.prod([len(beam) for beam in beams])) if beams else 0,
            "post_change_limit_combination_count": int(
                len(teams) - int(joint_selection.get("combination_rejected_by_change_limit_count", 0))
            ),
            "feasible_count": int(joint_selection["feasible_count"]),
            "quality_floor_feasible_count": int(joint_selection.get("quality_floor_feasible_count", joint_selection["feasible_count"])),
            "quality_anchor_feasible_team_count": int(joint_selection.get("quality_anchor_feasible_team_count", 0)),
            "quality_anchor_fallback_reason": str(joint_selection.get("quality_anchor_fallback_reason", "")),
            "quality_frontier_count": int(joint_selection["quality_frontier_count"]),
            "final_candidate_team_count": int(joint_selection.get("final_candidate_team_count", joint_selection["quality_frontier_count"])),
            "hierarchical_band_counts": list(joint_selection.get("hierarchical_band_counts", [])),
            "hierarchical_band_count_by_name": dict(joint_selection.get("hierarchical_band_count_by_name", {})),
            "combination_rejected_by_change_limit_count": int(joint_selection.get("combination_rejected_by_change_limit_count", 0)),
            "fold_quality_rejection_count": int(joint_selection.get("fold_quality_rejection_count", 0)),
            "incumbent_metrics": {
                key: incumbent[key] for key in (*QUALITY_KEYS, "vote_correct_count", "total_agent_correct_count", "bottom2_correct_count", "per_agent_correct_count", "coverage_depth_c1_correct_count", "coverage_depth_c2_correct_count")
            },
            "selected_metrics": {
                key: selected[key] for key in (*QUALITY_KEYS, "vote_correct_count", "total_agent_correct_count", "bottom2_correct_count", "per_agent_correct_count", "coverage_depth_c1_correct_count", "coverage_depth_c2_correct_count")
            },
            "allowed_quality_losses": {
                "vote_correct_count": int(self.cfg.joint_allowed_vote_loss_questions),
                "total_agent_correct_count": int(self.cfg.joint_allowed_total_agent_correct_loss),
                "bottom2_correct_count": int(self.cfg.joint_allowed_bottom2_correct_loss),
                "c1_correct_count": int(self.cfg.joint_allowed_c1_loss_questions),
                "c2_correct_count": int(self.cfg.joint_allowed_c2_loss_questions),
                "per_agent_correct_count": int(self.cfg.joint_allowed_per_agent_correct_loss),
            },
            "quality_anchor_archive": [anchor.to_dict() for anchor in anchor_objects],
            "quality_anchor_count": len(anchor_objects),
            "safe_archive_size_per_agent": [len(getattr(agent, "safe_qd_archive", [])) for agent in self.agents],
            "probation_archive_size_per_agent": [len(getattr(agent, "probation_archive", [])) for agent in self.agents],
            "selected_prompt_hashes": selected["prompt_hashes"], "selected_beam_sources": selected_sources,
            "team_diversity_score": selected["team_diversity_score"],
            "stable_team_score": selected["stable_team_score"],
            "mean_behavior_distance": selected["mean_behavior_distance"],
            "min_behavior_distance": selected["min_behavior_distance"],
            "mean_mechanism_distance": selected["mean_mechanism_distance"],
            "fold_diversities": list(selected.get("fold_diversities", [])),
            "fold_quality_gate_passed": bool(selected.get("fold_quality_gate_passed", True)),
            "per_agent_cross_fold_behavior_gap": list(selected.get("per_agent_cross_fold_behavior_gap", [])),
            "cross_fold_diversity_mean": float(selected.get("cross_fold_diversity_mean", 0.0)),
            "cross_fold_diversity_gap": float(selected.get("cross_fold_diversity_gap", 0.0)),
            "stable_diversity_score": float(selected.get("stable_diversity_score", 0.0)),
            "lineage_drift_penalty_mean": float(selected.get("lineage_drift_penalty_mean", 0.0) or 0.0),
            "peer_collapse_penalty_mean": float(selected.get("peer_collapse_penalty_mean", 0.0) or 0.0),
            "active_prompt_changed_count": changed_count,
            "specialization_strength": float(self.specialization_strength),
            "full_probe_cache_hits": int(getattr(self, "full_probe_cache_hit_count", 0)),
            "full_probe_missing_pair_evaluations": int(getattr(self, "full_probe_missing_pair_evaluation_count", 0)),
            "embedding_cache_hits": int(getattr(self, "mechanism_embedding_cache_hit_count", 0)),
            "embedding_cache_misses": int(getattr(self, "mechanism_embedding_cache_miss_count", 0)),
        }
        band_counts = record["hierarchical_band_count_by_name"]
        fold_diversities = record["fold_diversities"]
        record.update({
            "actual_combination_count": record["post_change_limit_combination_count"],
            "vote_band_remaining_count": int(band_counts.get("vote", 0)),
            "mean_band_remaining_count": int(band_counts.get("mean", 0)),
            "bottom2_band_remaining_count": int(band_counts.get("bottom2", 0)),
            "c1_band_remaining_count": int(band_counts.get("c1", 0)),
            "c2_band_remaining_count": int(band_counts.get("c2", 0)),
            "fold_a_diversity": float(fold_diversities[0]) if fold_diversities else 0.0,
            "fold_b_diversity": float(fold_diversities[1]) if len(fold_diversities) > 1 else 0.0,
        })
        selected_anchor = build_quality_anchor(
            selected, epoch=epoch, created_order=int(getattr(self, "quality_anchor_created_count", 0)),
        )
        self.quality_anchor_created_count = int(getattr(self, "quality_anchor_created_count", 0)) + 1
        anchor_objects = update_quality_anchor_archive(
            self.quality_anchor_archive, [selected_anchor], capacity=int(self.cfg.quality_anchor_archive_size),
        )
        self.quality_anchor_archive = [anchor.to_dict() for anchor in anchor_objects]
        self.joint_quality_anchor_metrics = {}
        actual_losses = {
            key: max(0, int(record["incumbent_metrics"][key]) - int(record["selected_metrics"][key]))
            for key in ("vote_correct_count", "total_agent_correct_count", "bottom2_correct_count", "coverage_depth_c1_correct_count", "coverage_depth_c2_correct_count")
        }
        actual_losses["per_agent_correct_count"] = [
            max(0, int(before) - int(after))
            for before, after in zip(
                record["incumbent_metrics"]["per_agent_correct_count"],
                record["selected_metrics"]["per_agent_correct_count"],
            )
        ]
        quality_passed = (
            actual_losses["vote_correct_count"] <= record["allowed_quality_losses"]["vote_correct_count"]
            and actual_losses["total_agent_correct_count"] <= record["allowed_quality_losses"]["total_agent_correct_count"]
            and actual_losses["bottom2_correct_count"] <= record["allowed_quality_losses"]["bottom2_correct_count"]
            and actual_losses["coverage_depth_c1_correct_count"] <= record["allowed_quality_losses"]["c1_correct_count"]
            and actual_losses["coverage_depth_c2_correct_count"] <= record["allowed_quality_losses"]["c2_correct_count"]
            and all(
                loss <= record["allowed_quality_losses"]["per_agent_correct_count"]
                for loss in actual_losses["per_agent_correct_count"]
            )
        )
        record.update({
            "actual_quality_losses": actual_losses,
            "quality_constraints_passed": bool(quality_passed),
            "quality_constraint_violation": not bool(quality_passed),
        })
        self.joint_team_selection_history.append(record)
        self.latest_joint_team_metrics = dict(record)
        safe_niches = {
            mechanism_niche_key(item.get("metrics", {}).get("mechanism_representation", {}))
            for agent in self.agents
            for item in getattr(agent, "safe_qd_archive", [])
        }
        initial = dict(self.initial_competence_probe_metrics or {})
        initial_mean = float(initial.get("mean_individual_acc", 0.0) or 0.0)
        initial_c1 = float(initial.get("coverage_depth_c1", 0.0) or 0.0)
        initial_c2 = float(initial.get("coverage_depth_c2", 0.0) or 0.0)
        selected_mean = float(selected.get("mean_individual_acc", 0.0) or 0.0)
        selected_c1 = float(selected.get("coverage_depth_c1", 0.0) or 0.0)
        selected_c2 = float(selected.get("coverage_depth_c2", 0.0) or 0.0)
        competence_mean_gate_passed = selected_mean >= initial_mean - float(self.cfg.competence_mean_guard_epsilon)
        competence_c1_gate_passed = selected_c1 >= initial_c1 - float(self.cfg.competence_c1_guard_epsilon)
        competence_c2_gate_passed = selected_c2 >= initial_c2 - float(self.cfg.competence_c2_guard_epsilon)
        qd_ready = (
            competence_mean_gate_passed
            and competence_c1_gate_passed
            and competence_c2_gate_passed
            and len(safe_niches) >= int(self.cfg.qd_readiness_min_distinct_niches)
            and float(record.get("stable_diversity_score", 0.0)) >= float(self.cfg.qd_readiness_min_diversity)
            and float(record.get("cross_fold_diversity_gap", 0.0)) <= float(self.cfg.qd_readiness_max_fold_gap)
        )
        self._recompute_effective_residual_strength(qd_ready)
        record.update({
            "qd_readiness_passed": bool(qd_ready),
            "safe_distinct_mechanism_niche_count": len(safe_niches),
            "competence_mean_gate_passed": bool(competence_mean_gate_passed),
            "competence_c1_gate_passed": bool(competence_c1_gate_passed),
            "competence_c2_gate_passed": bool(competence_c2_gate_passed),
            "competence_schedule_strength": float(self.specialization_strength),
            "qd_residual_floor_applied": bool(qd_ready and self.effective_residual_strength > self.specialization_strength),
            "effective_residual_strength": float(self.effective_residual_strength),
        })
        self.latest_joint_team_metrics = dict(record)
        active_niches = len({
            tuple(profile["mechanism_representation"].get("normalized_operation_sequence", [])[:4])
            for profile in selected["prompt_profiles"]
        })
        no_new_niche = active_niches <= int(getattr(self, "qd_previous_active_niche_count", 0) or 0)
        incumbent_retained = changed_count == 0
        self.qd_no_diversification_epochs = int(self.qd_no_diversification_epochs) + 1 if (incumbent_retained and no_new_niche) else 0
        self.qd_previous_active_niche_count = int(active_niches)
        if self.qd_no_diversification_epochs >= int(self.cfg.joint_team_no_diversification_patience):
            self.qd_change_limit_relaxed_epoch = int(epoch) + 1
        self._flush_jsonl("joint_team_selection_history.jsonl", [record])
        self._flush_jsonl("lineage_history.jsonl", self.lineage_history[-len(self.agents):])
        self._flush_jsonl("quality_diversity_archive.jsonl", self.quality_diversity_archive_history)
        self.quality_diversity_archive_history = []
        return record

    async def evaluate_competence_probe(
        self,
        probe_data: List[Dict[str, str]],
        *,
        probe_name: str,
        epoch: int,
    ) -> Dict[str, Any]:
        """Evaluate current prompts on a fixed optimization-only probe without training side effects."""
        prompts = list(self._active_prompt_list())
        prompt_hashes = [self._hash(prompt) for prompt in prompts]

        if self._is_stable_qd_lineage():
            profiles = await asyncio.gather(*[
                self._evaluate_prompt_on_stable_probe(
                    agent_id, prompt, probe_data,
                    self.agents[agent_id].prompt_beam[0].get("metrics", {}).get("mechanism_signature", []),
                )
                for agent_id, prompt in enumerate(prompts)
            ])
            question_hashes = profiles[0]["question_hashes"] if profiles else []
            gold_answers = profiles[0]["gold_answers"] if profiles else []
            summary = team_quality_metrics(
                profiles, gold_answers, question_hashes,
                vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
                tie_break_method=self.cfg.vote_tie_break, seed=self.cfg.seed,
            )
            behavior_profiles = build_team_behavior_profiles(summary["answer_vectors"], summary["correctness_vectors"])
            for profile, behavior in zip(profiles, behavior_profiles):
                self.behavior_profile_by_prompt_hash[profile["prompt_hash"]] = dict(behavior)
            record = {
                "probe_name": str(probe_name), "probe_source": "optimization_train", "epoch": int(epoch),
                "probe_size": len(probe_data), "question_hashes": question_hashes,
                "active_prompt_hashes": prompt_hashes, "per_agent_acc": summary["per_agent_acc"],
                "mean_individual_acc": summary["mean_individual_acc"],
                "min_individual_acc": min(summary["per_agent_acc"], default=0.0),
                "bottom2_mean_acc": summary["bottom2_mean_acc"],
                "bottom3_mean_acc": float(np.mean(sorted(summary["per_agent_acc"])[:3])) if summary["per_agent_acc"] else 0.0,
                "max_individual_acc": max(summary["per_agent_acc"], default=0.0),
                "individual_acc_std": float(np.std(summary["per_agent_acc"])) if summary["per_agent_acc"] else 0.0,
                "best_minus_worst_gap": max(summary["per_agent_acc"], default=0.0) - min(summary["per_agent_acc"], default=0.0),
                "best_minus_bottom2_gap": max(summary["per_agent_acc"], default=0.0) - summary["bottom2_mean_acc"],
                "coverage_depth_c1": summary["coverage_depth_c1"], "coverage_depth_c2": summary["coverage_depth_c2"],
                **{f"coverage_depth_c{depth}": float(np.mean([sum(row) >= depth for row in zip(*summary["correctness_vectors"])])) if summary["correctness_vectors"] else 0.0 for depth in range(3, 6)},
                "behavior_profiles": behavior_profiles,
            }
            self.behavior_profile_history.append({"epoch": epoch, "probe_name": probe_name, "profiles": behavior_profiles})
            self._flush_jsonl("behavior_profile_history.jsonl", [self.behavior_profile_history[-1]])
            self.competence_probe_history.append(dict(record))
            self._flush_jsonl("competence_probe_history.jsonl", [record])
            return record

        async def evaluate_one(index: int, example: Dict[str, str]) -> Dict[str, Any]:
            question = example["question"]
            gold = self.task_spec.parse_gold(example["answer"], question)
            traces, answers = await self.solve_with_prompts(question, prompts)
            question_hash = self._hash(question)
            self._record_solver_rollouts(
                question_hash, prompts, traces, answers, source=f"competence_probe_{probe_name}"
            )
            return {
                "index": index,
                "question_hash": question_hash,
                **self.compute_rollout_metrics(traces, answers, gold, prompts, question_hash=question_hash),
            }

        rows = await asyncio.gather(*[
            evaluate_one(index, example) for index, example in enumerate(probe_data)
        ])
        summary = self._summarize_rollout_rows(rows)
        record = {
            "probe_name": str(probe_name),
            "probe_source": "optimization_train",
            "epoch": int(epoch),
            "probe_size": len(probe_data),
            "question_hashes": [str(row["question_hash"]) for row in rows],
            "active_prompt_hashes": prompt_hashes,
            **{key: summary.get(key) for key in (
                "per_agent_acc", "mean_individual_acc", "min_individual_acc",
                "bottom2_mean_acc", "bottom3_mean_acc", "max_individual_acc",
                "individual_acc_std", "best_minus_worst_gap", "best_minus_bottom2_gap",
                "coverage_depth_c1", "coverage_depth_c2", "coverage_depth_c3",
                "coverage_depth_c4", "coverage_depth_c5",
            )},
        }
        self.competence_probe_history.append(dict(record))
        with open(os.path.join(self.cfg.out_dir, "competence_probe_history.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def _capability_specialization_diagnostics(self) -> Dict[str, Any]:
        profiles = [dict(getattr(agent, "capability_profile", {}) or {}) for agent in self.agents]
        families = sorted({str(key) for profile in profiles for key in profile})
        top = []
        for profile in profiles:
            top.append(max(profile, key=lambda key: float(profile[key])) if profile else "")
        nonempty_top = [value for value in top if value]
        counts = {value: nonempty_top.count(value) for value in sorted(set(nonempty_top))}
        total = len(nonempty_top)
        shares = [count / total for count in counts.values()] if total else []
        cosines = []
        for left in range(len(profiles)):
            for right in range(left + 1, len(profiles)):
                a = np.array([float(profiles[left].get(key, 0.0) or 0.0) for key in families])
                b = np.array([float(profiles[right].get(key, 0.0) or 0.0) for key in families])
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denom > 0.0:
                    cosines.append(float(np.dot(a, b) / denom))
        return {
            "top_capability_family_per_agent": top,
            "distinct_top_capability_family_count": len(set(nonempty_top)),
            "dominant_capability_family_share": max(shares, default=0.0),
            "capability_family_hhi": sum(value * value for value in shares),
            "mean_pairwise_capability_profile_cosine": float(np.mean(cosines)) if cosines else 0.0,
        }
