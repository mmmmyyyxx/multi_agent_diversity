"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class JointControllerMixin:
    @staticmethod
    def _active_mechanism_niche_count(prompt_profiles: Sequence[Mapping[str, Any]]) -> int:
        return len({
            mechanism_niche_key(profile.get("mechanism_representation", {}))
            for profile in prompt_profiles
        })

    def _joint_material_snapshot(self) -> Dict[str, Any]:
        return {
            "safe": {
                str(agent_id): sorted(
                    self._normalized_prompt_hash(str(item.get("prompt", "")))
                    for item in getattr(agent, "safe_qd_archive", [])
                )
                for agent_id, agent in enumerate(self.agents)
            },
            "representatives": {
                str(agent_id): [
                    self._normalized_prompt_hash(str(item.get("prompt", "")))
                    for item in getattr(agent, "prompt_beam", [])
                ]
                for agent_id, agent in enumerate(self.agents)
            },
            "active": [self._normalized_prompt_hash(agent.current_prompt) for agent in self.agents],
            "probation_promotions": int(getattr(self, "probation_to_safe_conversion_count", 0)),
        }

    def _fixed_probe_hash(self, probe_data: List[Dict[str, str]]) -> str:
        payload = [
            (self._hash(str(item.get("question", ""))), self._hash(str(item.get("answer", ""))))
            for item in probe_data
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _profile_is_current(self, agent_id: int, prompt_hash: str) -> bool:
        key = f"{int(agent_id)}:{prompt_hash}"
        profile = dict(getattr(self, "behavior_profile_by_prompt_hash", {}).get(key, {}) or {})
        return bool(
            profile
            and str(profile.get("fixed_probe_hash", "")) == str(getattr(self, "current_fixed_probe_hash", ""))
            and str(profile.get("fixed_probe_version", "")) == str(getattr(self, "prompt_probe_version", "legacy"))
        )

    @staticmethod
    def _dirty_shortlist_score(item: Dict[str, Any]) -> tuple:
        metrics = dict(item.get("metrics", {}) or {})
        candidate_type = str(metrics.get("candidate_type", item.get("proposal", {}).get("candidate_type", "")))
        return (
            float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0),
            float(metrics.get("depth1_net_count", 0) or 0) + float(metrics.get("depth2_net_count", 0) or 0),
            float(metrics.get("mechanism_signature_distance", 0.0) or 0.0),
            int(candidate_type == "mechanism_alternative"),
            str(item.get("prompt_hash", "")),
        )

    async def refresh_joint_active_team_if_needed(
        self,
        probe_data: List[Dict[str, str]],
        *,
        epoch: int,
        final_epoch: bool = False,
    ) -> Dict[str, Any]:
        if self._is_rollout_qd_method():
            if not probe_data:
                return {"enabled": False, "joint_refresh_triggered": False, "joint_refresh_skip_reason": "empty_probe"}
            self.current_fixed_probe_hash = self._fixed_probe_hash(probe_data)
            record = await self._select_rollout_joint_active_team(probe_data, epoch=epoch)
            record.update({
                "joint_refresh_triggered": True,
                "joint_refresh_trigger_reasons": ["epoch_end" if not final_epoch else "final_epoch"],
                "joint_team_solver_call_count": 0,
            })
            self.joint_team_selection_history.append(dict(record))
            self._flush_jsonl("joint_team_selection_history.jsonl", [record])
            return record
        if not self._is_stable_qd_lineage():
            return {"enabled": False, "joint_refresh_triggered": False}
        snapshot = self._joint_material_snapshot()
        previous = dict(getattr(self, "last_archive_material_snapshot", {}) or {})
        previous_safe = dict(previous.get("safe", {}) or {})
        new_safe_count = sum(
            len(set(values) - set(previous_safe.get(agent_id, [])))
            for agent_id, values in snapshot["safe"].items()
        ) if previous else sum(len(values) for values in snapshot["safe"].values())
        safe_changed = bool(previous and snapshot["safe"] != previous.get("safe"))
        representative_changed = bool(previous and snapshot["representatives"] != previous.get("representatives"))
        active_changed = bool(previous and snapshot["active"] != previous.get("active"))
        probation_promoted = snapshot["probation_promotions"] > int(previous.get("probation_promotions", 0) or 0)
        epochs_since = int(epoch) - int(getattr(self, "last_joint_refresh_epoch", 0) or 0)
        interval_due = epochs_since >= max(1, int(self.cfg.joint_refresh_interval_epochs))
        reasons: List[str] = []
        if bool(self.cfg.joint_refresh_on_safe_archive_change) and (
            safe_changed or new_safe_count >= int(self.cfg.joint_refresh_min_new_safe_candidates)
        ):
            reasons.append("safe_archive_change")
        if bool(self.cfg.joint_refresh_on_probation_promotion) and probation_promoted:
            reasons.append("probation_promotion")
        if bool(self.cfg.joint_refresh_on_representative_change) and representative_changed:
            reasons.append("representative_change")
        if active_changed:
            reasons.append("active_prompt_change")
        if interval_due:
            reasons.append("interval")
        if bool(self.cfg.joint_refresh_force_final_epoch) and final_epoch:
            reasons.append("final_epoch")
        triggered = str(self.cfg.joint_refresh_mode) != "event_driven" or bool(reasons)
        self.epochs_since_last_joint_refresh = epochs_since
        if not triggered:
            self.joint_refresh_skipped_count = int(getattr(self, "joint_refresh_skipped_count", 0)) + 1
            record = {
                "epoch": int(epoch),
                "joint_refresh_triggered": False,
                "joint_refresh_skip_reason": "no_material_archive_change",
                "joint_refresh_trigger_reasons": [],
                "epochs_since_last_joint_refresh": epochs_since,
                "new_full_probe_prompt_count": 0,
                "new_full_probe_pair_count": 0,
                "joint_team_combination_count": 0,
                "joint_team_solver_call_count": 0,
            }
            self.joint_team_selection_history.append(record)
            self._flush_jsonl("joint_team_selection_history.jsonl", [record])
            return record
        self.current_fixed_probe_hash = self._fixed_probe_hash(probe_data)
        self.last_archive_material_snapshot = snapshot
        self.archive_material_change_version = int(getattr(self, "archive_material_change_version", 0)) + int(
            safe_changed or probation_promoted
        )
        before_pairs = int(getattr(self, "full_probe_missing_pair_evaluation_count", 0))
        before_prompts = int(getattr(self, "new_full_probe_prompt_count", 0))
        record = await self.select_joint_active_team(probe_data, epoch=epoch)
        self.last_joint_refresh_epoch = int(epoch)
        self.epochs_since_last_joint_refresh = 0
        self.joint_refresh_count = int(getattr(self, "joint_refresh_count", 0)) + 1
        record.update({
            "joint_refresh_triggered": True,
            "joint_refresh_skip_reason": "",
            "joint_refresh_trigger_reasons": reasons,
            "epochs_since_last_joint_refresh": epochs_since,
            "new_full_probe_prompt_count": int(getattr(self, "new_full_probe_prompt_count", 0)) - before_prompts,
            "new_full_probe_pair_count": int(getattr(self, "full_probe_missing_pair_evaluation_count", 0)) - before_pairs,
            "joint_team_combination_count": int(record.get("combination_count", 0)),
            "joint_team_solver_call_count": 0,
        })
        self.last_archive_material_snapshot = self._joint_material_snapshot()
        self.last_representative_snapshot = dict(self.last_archive_material_snapshot.get("representatives", {}))
        self.last_active_prompt_hashes = list(self.last_archive_material_snapshot.get("active", []))
        self._flush_jsonl("joint_team_selection_history.jsonl", [record])
        return record

    async def _select_rollout_joint_active_team(
        self, probe_data: List[Dict[str, str]], *, epoch: int,
    ) -> Dict[str, Any]:
        beams: List[List[Dict[str, Any]]] = []
        beam_items: List[List[Dict[str, Any]]] = []
        profile_cache_hits = 0
        for agent_id, agent in enumerate(self.agents):
            active_hash = self._normalized_prompt_hash(agent.current_prompt)
            archive = list(getattr(agent, "safe_qd_archive", []) or agent.prompt_beam)
            by_hash: Dict[str, Dict[str, Any]] = {}
            for item in archive:
                by_hash.setdefault(
                    self._normalized_prompt_hash(str(item.get("prompt", agent.current_prompt))), dict(item)
                )
            by_hash.setdefault(active_hash, self._make_beam_item(agent.current_prompt, None, {}, None, 0))
            profiled = []
            for item in by_hash.values():
                prompt_hash = self._normalized_prompt_hash(str(item.get("prompt", agent.current_prompt)))
                cache_key = f"{agent_id}:{prompt_hash}:{self.current_fixed_probe_hash}"
                profile = dict(self.rollout_profile_by_prompt_hash.get(cache_key, {}) or {})
                if profile:
                    profile_cache_hits += 1
                else:
                    profile = await self._evaluate_prompt_on_stable_probe(
                        agent_id, str(item.get("prompt", agent.current_prompt)), probe_data
                    )
                    self.rollout_profile_by_prompt_hash[cache_key] = dict(profile)
                candidate = dict(item)
                metrics = dict(candidate.get("metrics", {}) or {})
                metrics["rollout_profile"] = dict(profile)
                candidate["metrics"] = metrics
                candidate["prompt_hash"] = prompt_hash
                profiled.append(candidate)
            agent.safe_qd_archive = select_rollout_archive(
                profiled, active_hash, int(self.cfg.qd_archive_size_per_agent), self.cfg,
                vote_ready=self._is_vote_ready_rollout_method(),
            )
            representatives = select_rollout_representatives(
                agent.safe_qd_archive, active_hash, int(self.cfg.joint_representative_beam_size), self.cfg,
                vote_ready=self._is_vote_ready_rollout_method(),
            )
            agent.prompt_beam = representatives or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]
            for item in agent.prompt_beam:
                self._record_candidate_funnel_item(item, agent_id, "representative_selected_count")
            beam_items.append(list(agent.prompt_beam))
            beams.append([dict(item.get("metrics", {}).get("rollout_profile", {})) for item in agent.prompt_beam])
            for item, profile in zip(agent.prompt_beam, beams[-1]):
                profile["prompt_hash"] = str(item.get("prompt_hash", ""))
                profile["candidate_source"] = self._candidate_generation_source(item)
        question_hashes = [self._hash(str(example.get("question", ""))) for example in probe_data]
        gold_answers = [self.task_spec.parse_gold(example.get("answer"), str(example.get("question", ""))) for example in probe_data]
        teams = enumerate_rollout_teams(
            beams,
            gold_answers,
            question_hashes,
            vote_fn=plurality_vote_with_diagnostics,
            match_fn=self.task_spec.match_answer,
            tie_break_method=self.cfg.vote_tie_break,
            seed=self.cfg.seed,
            config=self.cfg,
        )
        incumbent_indices = []
        for agent_id, agent in enumerate(self.agents):
            active_hash = self._normalized_prompt_hash(agent.current_prompt)
            incumbent_indices.append(next((
                index for index, item in enumerate(beam_items[agent_id])
                if str(item.get("prompt_hash", "")) == active_hash
            ), 0))
        incumbent = next(team for team in teams if team["beam_indices"] == incumbent_indices)
        key_fn = rollout_team_key if self._is_vote_ready_rollout_method() else accuracy_rollout_team_key
        selected = max(teams, key=key_fn)
        changed_count = 0
        selected_sources = []
        for agent_id, beam_index in enumerate(selected["beam_indices"]):
            agent = self.agents[agent_id]
            chosen = beam_items[agent_id][beam_index]
            old_hash = self._normalized_prompt_hash(agent.current_prompt)
            new_hash = str(chosen.get("prompt_hash", ""))
            agent.prompt_beam = [chosen] + [item for index, item in enumerate(beam_items[agent_id]) if index != beam_index]
            agent.current_prompt = str(chosen.get("prompt", agent.current_prompt))
            changed = old_hash != new_hash
            changed_count += int(changed)
            source = self._candidate_generation_source(chosen) or "incumbent"
            selected_sources.append(source)
            self.active_candidate_source_by_agent[str(agent_id)] = source
            profile = dict(chosen.get("metrics", {}).get("rollout_profile", {}) or {})
            signature = str(profile.get("rollout_signature_hash", "") or rollout_signature(profile))
            self.rollout_signature_history.append({
                "epoch": int(epoch), "agent_id": int(agent_id), "prompt_hash": new_hash,
                "rollout_signature_hash": signature, "candidate_source": source,
            })
            self.accepted_rollout_archive.append({
                "epoch": int(epoch), "agent_id": int(agent_id), "prompt_hash": new_hash,
                "rollout_signature_hash": signature, "candidate_source": source,
            })
            if changed:
                agent.history.append(agent.current_prompt)
                agent.accept_count += 1
            else:
                agent.reject_count += 1
            self._record_candidate_funnel_item(chosen, agent_id, "active_selected_count")
            self._append_prompt_history_event(
                agent_id, epoch, 0, "rollout_joint_selected" if changed else "rollout_joint_keep", changed
            )
        validate_candidate_channel_funnel(self.candidate_channel_funnel)
        record = {
            "enabled": True,
            "epoch": int(epoch),
            "combination_count": len(teams),
            "joint_team_solver_call_count": 0,
            "cached_prompt_profile_count": int(profile_cache_hits),
            "representative_count_per_agent": [len(beam) for beam in beams],
            "incumbent_metrics": {key: value for key, value in incumbent.items() if key not in {"prompt_profiles", "answer_vectors", "correctness_vectors", "invalid_vectors"}},
            "selected_metrics": {key: value for key, value in selected.items() if key not in {"prompt_profiles", "answer_vectors", "correctness_vectors", "invalid_vectors"}},
            "selected_prompt_hashes": list(selected.get("prompt_hashes", [])),
            "selected_beam_sources": selected_sources,
            "active_prompt_changed_count": int(changed_count),
            "rollout_diversity_score": float(selected.get("rollout_diversity_score", 0.0)),
            "mechanism_based_decision_count": 0,
            "candidate_channel_funnel": json.loads(json.dumps(self.candidate_channel_funnel)),
        }
        self.latest_joint_team_metrics = dict(record)
        self.flush_prompt_history()
        return record

    async def select_joint_active_team(self, probe_data: List[Dict[str, str]], *, epoch: int) -> Dict[str, Any]:
        if self._is_rollout_qd_method():
            return await self._select_rollout_joint_active_team(probe_data, epoch=epoch)
        if not self._is_stable_qd_lineage():
            return {"enabled": False, "combination_count": 0}
        if not str(getattr(self, "current_fixed_probe_hash", "")):
            self.current_fixed_probe_hash = self._fixed_probe_hash(probe_data)
        beams: List[List[Dict[str, Any]]] = []
        cached_profile_count = 0
        dirty_prompt_count = 0
        dirty_shortlist_count = 0
        safe_profile_current_counts: List[int] = []
        safe_unprofiled_counts: List[int] = []
        safe_profile_fractions: List[float] = []
        dirty_shortlist_excluded_counts: List[int] = []
        oldest_unprofiled_safe_ages: List[int] = []
        representative_profile_current_counts: List[int] = []
        for agent_id, agent in enumerate(self.agents):
            archive = list(getattr(agent, "safe_qd_archive", []) or agent.prompt_beam)
            for archive_item in archive:
                self._record_candidate_funnel_item(
                    archive_item, agent_id, "evaluated_candidate_count"
                )
                self._record_candidate_funnel_classification(
                    archive_item, agent_id, "safe_count"
                )
                self._record_candidate_funnel_item(
                    archive_item, agent_id, "archive_retained_count"
                )
            active_hash = self._normalized_prompt_hash(agent.current_prompt)
            existing_representatives = list(getattr(agent, "prompt_beam", []) or [])
            dirty = [
                item for item in archive
                if not self._profile_is_current(agent_id, self._normalized_prompt_hash(str(item.get("prompt", ""))))
            ]
            dirty_prompt_count += len(dirty)
            dirty.sort(key=self._dirty_shortlist_score, reverse=True)
            shortlist = dirty[: max(0, int(self.cfg.joint_refresh_max_dirty_candidates_per_agent))]
            dirty_shortlist_count += len(shortlist)
            dirty_shortlist_excluded_counts.append(max(0, len(dirty) - len(shortlist)))
            pool_by_hash: Dict[str, Dict[str, Any]] = {}
            for item in [
                next((row for row in archive if self._normalized_prompt_hash(str(row.get("prompt", ""))) == active_hash),
                     self._make_beam_item(agent.current_prompt, None, {}, None, 0)),
                *existing_representatives,
                *shortlist,
            ]:
                pool_by_hash.setdefault(self._normalized_prompt_hash(str(item.get("prompt", ""))), dict(item))
            profiled_pool = []
            for item in pool_by_hash.values():
                metrics = item.get("metrics", {})
                prompt_hash = self._normalized_prompt_hash(str(item.get("prompt", agent.current_prompt)))
                was_current = self._profile_is_current(agent_id, prompt_hash)
                cached_profile = None
                if was_current:
                    cached_profile = dict(
                        self.behavior_profile_by_prompt_hash.get(f"{agent_id}:{prompt_hash}")
                        or self.behavior_profile_by_prompt_hash.get(prompt_hash)
                        or {}
                    )
                profile = cached_profile or await self._evaluate_prompt_on_stable_probe(
                    agent_id, str(item.get("prompt", agent.current_prompt)), probe_data,
                    metrics.get("mechanism_steps", metrics.get("mechanism_signature", [])),
                )
                profile["prompt_hash"] = prompt_hash
                profile["fixed_probe_hash"] = str(self.current_fixed_probe_hash)
                profile["fixed_probe_version"] = str(getattr(self, "prompt_probe_version", "legacy"))
                candidate = dict(item)
                candidate_metrics = dict(metrics)
                candidate_metrics["behavior_profile"] = build_prompt_static_profile(
                    profile.get("answer_vector", []), profile.get("correctness_vector", [])
                )
                candidate["metrics"] = candidate_metrics
                profiled_pool.append(candidate)
                key = f"{agent_id}:{profile['prompt_hash']}"
                self.behavior_profile_by_prompt_hash[key] = dict(profile)
                self.behavior_profile_by_prompt_hash[profile["prompt_hash"]] = dict(profile)
                if was_current:
                    cached_profile_count += 1
                else:
                    self.new_full_probe_prompt_count = int(getattr(self, "new_full_probe_prompt_count", 0)) + 1
                dirty_hashes = getattr(self, "dirty_prompt_hashes", {})
                dirty_hashes.setdefault(str(agent_id), [])
                if profile["prompt_hash"] in dirty_hashes[str(agent_id)]:
                    dirty_hashes[str(agent_id)].remove(profile["prompt_hash"])
                self.dirty_prompt_hashes = dirty_hashes
            previous_hashes = [self._normalized_prompt_hash(str(item.get("prompt", ""))) for item in existing_representatives]
            representatives = [dict(item) for item in select_joint_representatives(
                profiled_pool, active_hash, int(self.cfg.joint_representative_beam_size), self.cfg,
            )]
            agent.prompt_beam = representatives or [dict(profiled_pool[0])]
            current_hashes = [self._normalized_prompt_hash(str(item.get("prompt", ""))) for item in agent.prompt_beam]
            if current_hashes != previous_hashes:
                key = str(agent_id)
                versions = getattr(self, "representative_version_per_agent", {})
                versions[key] = int(versions.get(key, 0)) + 1
                self.representative_version_per_agent = versions
            agent_profiles = []
            for beam_index, item in enumerate(agent.prompt_beam):
                metrics = item.get("metrics", {})
                prompt_hash = self._normalized_prompt_hash(str(item.get("prompt", "")))
                profile = dict(self.behavior_profile_by_prompt_hash.get(f"{agent_id}:{prompt_hash}") or self.behavior_profile_by_prompt_hash.get(prompt_hash) or {})
                if not profile:
                    profile = await self._evaluate_prompt_on_stable_probe(
                        agent_id, str(item.get("prompt", agent.current_prompt)), probe_data,
                        metrics.get("mechanism_steps", metrics.get("mechanism_signature", [])),
                    )
                    profile["prompt_hash"] = prompt_hash
                    profile["fixed_probe_hash"] = str(self.current_fixed_probe_hash)
                    profile["fixed_probe_version"] = str(getattr(self, "prompt_probe_version", "legacy"))
                    self.behavior_profile_by_prompt_hash[f"{agent_id}:{profile['prompt_hash']}"] = dict(profile)
                profile.update({
                    "beam_index": beam_index,
                    "beam_source": str(item.get("beam_slot", metrics.get("beam_slot", "incumbent"))),
                })
                self.behavior_profile_by_prompt_hash[profile["prompt_hash"]] = dict(profile)
                agent_profiles.append(profile)
            beams.append(agent_profiles)
            current_safe = [
                item for item in archive
                if self._profile_is_current(
                    agent_id, self._normalized_prompt_hash(str(item.get("prompt", "")))
                )
            ]
            unprofiled_safe = [item for item in archive if item not in current_safe]
            safe_profile_current_counts.append(len(current_safe))
            safe_unprofiled_counts.append(len(unprofiled_safe))
            safe_profile_fractions.append(float(len(current_safe) / len(archive)) if archive else 0.0)
            oldest_unprofiled_safe_ages.append(max([
                max(0, int(epoch) - int(item.get("safe_created_epoch", epoch) or epoch))
                for item in unprofiled_safe
            ], default=0))
            representative_profile_current_counts.append(sum(
                self._profile_is_current(
                    agent_id, self._normalized_prompt_hash(str(item.get("prompt", "")))
                )
                for item in agent.prompt_beam
            ))
            assert representative_profile_current_counts[-1] == len(agent.prompt_beam)
            assert self._profile_is_current(agent_id, active_hash)
        representative_distances: List[float] = []
        for beam in beams:
            static_profiles = [
                build_prompt_static_profile(
                    profile.get("answer_vector", []),
                    profile.get("correctness_vector", []),
                    profile.get("invalid_vector", []),
                )
                for profile in beam
            ]
            for left in range(len(static_profiles)):
                for right in range(left + 1, len(static_profiles)):
                    representative_distances.append(behavior_distance(
                        static_profiles[left], static_profiles[right],
                        correct_set_weight=self.cfg.behavior_correct_set_weight,
                        rescue_weight=self.cfg.behavior_rescue_weight,
                        shared_wrong_weight=self.cfg.behavior_error_overlap_weight,
                        wrong_answer_dispersion_weight=self.cfg.behavior_wrong_answer_dispersion_weight,
                        support_shrinkage=self.cfg.behavior_support_shrinkage,
                        wrong_support_shrinkage=self.cfg.behavior_wrong_support_shrinkage,
                    )["behavior_distance"])
        representative_mean_behavior_distance = (
            float(np.mean(representative_distances)) if representative_distances else 0.0
        )
        representative_min_behavior_distance = min(representative_distances, default=0.0)
        representative_behavior_span = max(representative_distances, default=0.0)
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
        self.offline_team_combination_count = int(getattr(self, "offline_team_combination_count", 0)) + len(teams)
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
        for agent_id, agent in enumerate(self.agents):
            for item in agent.prompt_beam:
                self._record_candidate_funnel_item(
                    item, agent_id, "representative_selected_count"
                )
        selected_sources, changed_count = [], 0
        for agent_id, beam_index in enumerate(selected["beam_indices"]):
            agent = self.agents[agent_id]
            chosen = agent.prompt_beam[beam_index]
            old_hash = self._normalized_prompt_hash(agent.current_prompt)
            agent.prompt_beam = [chosen] + [item for index, item in enumerate(agent.prompt_beam) if index != beam_index]
            agent.current_prompt = str(chosen["prompt"])
            self._record_candidate_funnel_item(chosen, agent_id, "active_selected_count")
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
        validate_candidate_channel_funnel(self.candidate_channel_funnel)
        record = {
            "epoch": epoch, "combination_count": len(teams),
            "cached_prompt_profile_count": int(cached_profile_count),
            "dirty_prompt_count": int(dirty_prompt_count),
            "dirty_prompt_shortlist_count": int(dirty_shortlist_count),
            "safe_archive_profile_current_count_per_agent": safe_profile_current_counts,
            "safe_archive_unprofiled_count_per_agent": safe_unprofiled_counts,
            "safe_archive_profile_fraction_per_agent": safe_profile_fractions,
            "dirty_shortlist_excluded_count_per_agent": dirty_shortlist_excluded_counts,
            "oldest_unprofiled_safe_age_epochs_per_agent": oldest_unprofiled_safe_ages,
            "representative_profile_current_count_per_agent": representative_profile_current_counts,
            "representative_mean_behavior_distance": representative_mean_behavior_distance,
            "representative_min_behavior_distance": representative_min_behavior_distance,
            "representative_behavior_span": representative_behavior_span,
            "candidate_channel_funnel": json.loads(json.dumps(self.candidate_channel_funnel)),
            "joint_team_solver_call_count": 0,
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
            **{
                key: selected[key]
                for key in (
                    "oracle_correct_count", "vote_correct_count", "oracle_vote_gap_count",
                    "oracle_to_vote_conversion_rate", "c0_count", "c1_count", "c2_count",
                    "c3plus_count", "c1_vote_correct_count", "c1_vote_fail_count",
                    "c2_vote_correct_count", "c2_vote_fail_count", "c3plus_vote_fail_count",
                    "gold_top_tie_count", "gold_top_tie_win_count", "gold_top_tie_loss_count",
                    "mean_gold_plurality_margin", "mean_gold_margin_on_oracle_vote_gap",
                    "mean_max_wrong_vote_on_oracle_vote_gap", "dominant_wrong_concentration",
                    "vote_normalization_anomaly_count",
                    "vote_normalization_anomaly_question_hashes",
                )
            },
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
        active_niches = self._active_mechanism_niche_count(selected["prompt_profiles"])
        no_new_niche = active_niches <= int(getattr(self, "qd_previous_active_niche_count", 0) or 0)
        incumbent_retained = changed_count == 0
        self.qd_no_diversification_epochs = int(self.qd_no_diversification_epochs) + 1 if (incumbent_retained and no_new_niche) else 0
        self.qd_previous_active_niche_count = int(active_niches)
        if self.qd_no_diversification_epochs >= int(self.cfg.joint_team_no_diversification_patience):
            self.qd_change_limit_relaxed_epoch = int(epoch) + 1
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
