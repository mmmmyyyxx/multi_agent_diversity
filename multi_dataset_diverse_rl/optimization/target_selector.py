"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class TargetSelectorMixin:
    def _window_update_diagnosis(self, window_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = []
        all_homogeneous_cases: List[Dict[str, Any]] = []
        all_validity_cases: List[Dict[str, Any]] = []
        num_agents = len(self.agents)
        per_agent_invalid = [0 for _ in range(num_agents)]
        per_agent_seen = [0 for _ in range(num_agents)]
        per_agent_pressure_rows = [[] for _ in range(num_agents)]
        pivotal_fix_counts = [0 for _ in range(num_agents)]
        dominant_wrong_counts = [0 for _ in range(num_agents)]
        near_boundary_error_counts = [0 for _ in range(num_agents)]
        shared_error_counts = [0 for _ in range(num_agents)]
        c1_creation_counts = [0 for _ in range(num_agents)]
        c2_creation_counts = [0 for _ in range(num_agents)]
        pivotal_hold_counts = [0 for _ in range(num_agents)]
        vote_values: List[int] = []
        vote_margin_values: List[float] = []
        boundary_diversity_values: List[float] = []
        embedding_overlap_values: List[float] = []

        for idx, rec in enumerate(window_records):
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            individual = [int(value) for value in metrics.get("individual_correct", [])]
            invalids = [int(value) for value in metrics.get("invalid_flags", [])]
            pressures = list(metrics.get("per_agent_overlap", []))
            answers = [str(answer or "").strip() for answer in rec.get("answers", [])]
            gold = str(rec.get("gold", ""))
            question_hash = str(rec.get("question_hash", ""))
            vote_correct = int(metrics.get("vote_correct", 0) or 0)
            vote_tie = bool(metrics.get("vote_tie", False))
            gold_count = int(metrics.get("gold_vote_count", sum(individual)) or 0)
            largest_wrong = int(metrics.get("largest_wrong_vote_count", 0) or 0)
            margin = float(metrics.get("normalized_vote_margin", -1.0) if metrics.get("normalized_vote_margin") is not None else -1.0)
            boundary_diversity = float(metrics.get("boundary_useful_diversity", 0.0) or 0.0)
            invalid_rate = float(metrics.get("invalid_rate", 0.0) or 0.0)
            reward_pressure = float(1 - vote_correct) + max(0.0, -margin) + invalid_rate
            scored.append((reward_pressure, idx, rec))
            vote_values.append(vote_correct)
            vote_margin_values.append(margin)
            boundary_diversity_values.append(boundary_diversity)
            embedding_overlap_values.append(float(metrics.get("mean_embedding_overlap", 0.0) or 0.0))
            all_homogeneous_cases.extend(list(rec.get("homogeneous_cases", [])))
            all_validity_cases.extend(list(rec.get("validity_cases", [])))

            wrong_counts = Counter(
                answers[agent_id]
                for agent_id in range(min(len(answers), len(individual)))
                if answers[agent_id] and not individual[agent_id]
            )
            for agent_id in range(num_agents):
                if agent_id < len(invalids):
                    per_agent_seen[agent_id] += 1
                    per_agent_invalid[agent_id] += invalids[agent_id]
                if agent_id < len(pressures):
                    per_agent_pressure_rows[agent_id].append(float(pressures[agent_id]))
                if agent_id >= len(individual):
                    continue
                if individual[agent_id]:
                    if bool(getattr(self.cfg, "competence_depth_enabled", False)):
                        without_target = list(answers)
                        if agent_id < len(without_target):
                            without_target[agent_id] = ""
                        without_vote = self._vote_with_diagnostics(without_target, question_hash=question_hash)
                        if vote_correct and not self.task_spec.match_answer(
                            str(without_vote.get("vote_answer", "")), gold
                        ):
                            pivotal_hold_counts[agent_id] += 1
                    elif gold_count - 1 <= largest_wrong:
                        pivotal_hold_counts[agent_id] += 1
                    continue
                if gold_count == 0:
                    c1_creation_counts[agent_id] += 1
                elif gold_count == 1:
                    c2_creation_counts[agent_id] += 1
                peer_wrong_count = sum(
                    int(not individual[peer_id])
                    for peer_id in range(len(individual))
                    if peer_id != agent_id
                )
                plurality_pivotal = False
                if bool(getattr(self.cfg, "competence_depth_enabled", False)):
                    counterfactual_answers = list(answers)
                    if agent_id < len(counterfactual_answers):
                        counterfactual_answers[agent_id] = gold
                    counterfactual_vote = self._vote_with_diagnostics(
                        counterfactual_answers, question_hash=question_hash
                    )
                    plurality_pivotal = bool(
                        not vote_correct
                        and self.task_spec.match_answer(str(counterfactual_vote.get("vote_answer", "")), gold)
                    )
                    if plurality_pivotal:
                        near_boundary_error_counts[agent_id] += 1
                elif abs(gold_count - largest_wrong) <= 1:
                    near_boundary_error_counts[agent_id] += 1
                if peer_wrong_count > 0:
                    shared_error_counts[agent_id] += 1
                answer = answers[agent_id] if agent_id < len(answers) else ""
                remaining_wrong = dict(wrong_counts)
                if answer and answer in remaining_wrong:
                    remaining_wrong[answer] -= 1
                    if remaining_wrong[answer] <= 0:
                        remaining_wrong.pop(answer, None)
                counterfactual_largest_wrong = max(remaining_wrong.values(), default=0)
                if (
                    plurality_pivotal
                    if bool(getattr(self.cfg, "competence_depth_enabled", False))
                    else ((not vote_correct or vote_tie) and gold_count + 1 > counterfactual_largest_wrong)
                ):
                    pivotal_fix_counts[agent_id] += 1
                if answer and gold_count > 0 and wrong_counts.get(answer, 0) == largest_wrong and abs(gold_count - largest_wrong) <= 1:
                    dominant_wrong_counts[agent_id] += 1

        scored.sort(key=lambda item: item[0], reverse=True)
        focus_cases = []
        for score, idx, rec in scored[: min(3, max(1, len(scored)))]:
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            individual = list(metrics.get("individual_correct", []))
            focus_cases.append(
                {
                    "window_index": idx,
                    "reward_pressure": round(score, 4),
                    "vote_correct": bool(metrics.get("vote_correct", 0)),
                    "vote_tie": bool(metrics.get("vote_tie", False)),
                    "normalized_vote_margin": float(metrics.get("normalized_vote_margin", -1.0) if metrics.get("normalized_vote_margin") is not None else -1.0),
                    "boundary_useful_diversity": float(metrics.get("boundary_useful_diversity", 0.0) or 0.0),
                    "wrong_agent_ids": [agent_id for agent_id, correct in enumerate(individual) if not int(correct)],
                    "invalid_rate": float(metrics.get("invalid_rate", 0.0) or 0.0),
                }
            )

        homogeneous_case_counts = [0 for _ in range(num_agents)]
        for case in all_homogeneous_cases:
            agent_id = int(case.get("target_agent_id", -1))
            if 0 <= agent_id < num_agents:
                homogeneous_case_counts[agent_id] += 1
        accuracy_diagnosis = self._window_accuracy_diagnosis(window_records)
        boundary_case_types = {
            "pivotal": "target_wrong_pivotal_vote_fix",
            "near": "target_wrong_near_vote_boundary",
            "shared": "target_wrong_shared_error",
            "dominant": "target_wrong_dominant_wrong_cluster",
            "hold": "pivotal_correct_protection",
        }
        boundary_counts = {
            key: [0 for _ in range(num_agents)] for key in boundary_case_types
        }
        capability_pressure = [
            {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
            for _ in range(num_agents)
        ]
        for case in accuracy_diagnosis.get("target_error_cases", []):
            if not isinstance(case, dict):
                continue
            agent_id = int(case.get("target_agent_id", -1))
            if not 0 <= agent_id < num_agents:
                continue
            case_type = str(case.get("case_type", ""))
            for key, expected in boundary_case_types.items():
                boundary_counts[key][agent_id] += int(case_type == expected)
            family = str(case.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value))
            if family not in capability_pressure[agent_id]:
                family = CapabilityResidualFamily.UNKNOWN.value
            capability_pressure[agent_id][family] += 1.0
        seen = [max(1, int(value)) for value in per_agent_seen]
        coverage_depth = {
            family: sum(
                int(
                    agent.capability_evidence[family].support > 0
                    and agent.capability_evidence[family].posterior_value > 0.0
                )
                for agent in self.agents
            )
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        }
        majority_threshold = 2 if bool(getattr(self.cfg, "competence_depth_enabled", False)) else (num_agents // 2) + 1
        coverage_gap = {
            family: max(0, majority_threshold - depth)
            for family, depth in coverage_depth.items()
        }
        current_prompts = [
            {"agent_id": i, "prompt_preview": normalize_spaces(prompt)[:260], "prompt_hash": self._hash(prompt)}
            for i, prompt in enumerate(self._active_prompt_list())
        ]
        return {
            "diagnosis_type": "vote_update",
            "window_size": len(window_records),
            "focus_cases": focus_cases,
            "prompt_roles": current_prompts,
            "mean_window_overlap": float(np.mean(embedding_overlap_values)) if embedding_overlap_values else 0.0,
            "mean_embedding_overlap": float(np.mean(embedding_overlap_values)) if embedding_overlap_values else 0.0,
            "mean_reward_pressure": float(np.mean([item[0] for item in scored])) if scored else 0.0,
            "window_vote_acc": float(np.mean(vote_values)) if vote_values else 0.0,
            "window_mean_vote_margin": float(np.mean(vote_margin_values)) if vote_margin_values else -1.0,
            "window_mean_boundary_useful_diversity": self._clip01(float(np.mean(boundary_diversity_values))) if boundary_diversity_values else 0.0,
            "homogeneous_cases": sorted(all_homogeneous_cases, key=lambda case: float(case.get("pair_overlap", 0.0)), reverse=True),
            "validity_cases": all_validity_cases,
            "error_cases": list(accuracy_diagnosis.get("error_cases", [])),
            "target_error_cases": list(accuracy_diagnosis.get("target_error_cases", [])),
            "per_agent_accuracy": list(accuracy_diagnosis.get("per_agent_accuracy", [])),
            "per_agent_error_count": list(accuracy_diagnosis.get("per_agent_error_count", [])),
            "per_agent_team_wrong_error_count": list(accuracy_diagnosis.get("per_agent_team_wrong_error_count", [])),
            "team_accuracy": float(accuracy_diagnosis.get("team_accuracy", 0.0)),
            "homogeneous_case_counts": homogeneous_case_counts,
            "per_agent_invalid_rate": [float(per_agent_invalid[i] / per_agent_seen[i]) if per_agent_seen[i] else 0.0 for i in range(num_agents)],
            "per_agent_overlap_pressure": [float(np.mean(values)) if values else 0.0 for values in per_agent_pressure_rows],
            "per_agent_pivotal_fix_count": pivotal_fix_counts,
            "per_agent_dominant_wrong_redundancy_count": dominant_wrong_counts,
            "per_agent_pivotal_fix_rate": [pivotal_fix_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_plurality_pivotal_fix_rate": [pivotal_fix_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_plurality_boundary_error_rate": [near_boundary_error_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_near_boundary_error_rate": [near_boundary_error_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_shared_error_rate": [shared_error_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_c1_creation_opportunity": [c1_creation_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_c2_creation_opportunity": [c2_creation_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_dominant_wrong_rate": [dominant_wrong_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_general_error_rate": [
                float(accuracy_diagnosis.get("per_agent_error_count", [0] * num_agents)[i]) / seen[i]
                for i in range(num_agents)
            ],
            "per_agent_pivotal_hold_rate": [pivotal_hold_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_plurality_pivotal_hold_rate": [pivotal_hold_counts[i] / seen[i] for i in range(num_agents)],
            "pivotal_definition": "actual_plurality_counterfactual" if bool(getattr(self.cfg, "competence_depth_enabled", False)) else "legacy_vote_boundary",
            "per_agent_capability_pressure": capability_pressure,
            "capability_coverage_depth": coverage_depth,
            "capability_coverage_gap": coverage_gap,
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

    def _target_error_cases_for_agent(self, diagnosis: Dict[str, Any], agent_id: int) -> List[Dict[str, Any]]:
        if bool(getattr(self.cfg, "boundary_selector_enabled", False)):
            priority = {
                "target_wrong_pivotal_vote_fix": 0,
                "target_wrong_near_vote_boundary": 1,
                "target_wrong_shared_error": 2,
                "target_wrong_dominant_wrong_cluster": 3,
                "target_wrong_peer_correct_nonboundary": 4,
                "target_wrong_vote_already_correct": 5,
                "pivotal_correct_protection": 6,
                "target_invalid": 7,
            }
        else:
            priority = {
                "target_agent_wrong_and_peer_correct": 0,
                "target_agent_wrong_and_vote_correct": 1,
                "target_agent_wrong_and_vote_wrong": 2,
                "target_agent_invalid": 3,
            }
        cases = [
            c for c in diagnosis.get("target_error_cases", [])
            if isinstance(c, dict) and int(c.get("target_agent_id", -1)) == int(agent_id)
        ]
        cases.sort(key=lambda c: (priority.get(str(c.get("case_type", "")), 99), int(c.get("window_index", 0) or 0)))
        return cases

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
        target_error_cases = self._target_error_cases_for_agent(diagnosis, agent_id)
        target_error_limit = max(1, int(self.cfg.max_homogeneous_cases_per_agent))
        if self._is_state_conditioned_method():
            general = [case for case in target_error_cases if not bool(case.get("target_correct", False))]
            coverage = []
            conversion = []
            assignments = dict(getattr(self, "coverage_case_assignment_per_agent", {}) or {})
            assigned_to_target = set(str(value) for value in assignments.get(str(agent_id), []))
            for case in general:
                correct_count = int(case.get("baseline_correct_count", -1) or 0)
                question_hash = str(case.get("question_hash", case.get("sample_hash", case.get("case_id", ""))))
                if correct_count in {0, 1}:
                    assignees = coverage_case_assignees(
                        question_hash,
                        len(self.agents),
                        seed=int(self.cfg.seed),
                        assignment_count=min(2, len(self.agents)),
                    )
                    if agent_id in assignees:
                        enriched = dict(case)
                        enriched["coverage_assigned_agents"] = assignees
                        enriched["state"] = f"C{correct_count}"
                        coverage.append(enriched)
                        assigned_to_target.add(question_hash)
                elif correct_count == 2:
                    enriched = dict(case)
                    enriched["state"] = "C2"
                    conversion.append(enriched)
            assignments[str(agent_id)] = sorted(assigned_to_target)
            self.coverage_case_assignment_per_agent = assignments

            batches = [{
                "batch_type": "state_general_accuracy",
                "optimization_route": "general_accuracy",
                "priority": 0,
                "cases": general[:target_error_limit],
                "purpose": (
                    "repair target-agent errors while preserving overall accuracy; "
                    "do not treat changed wrong answers as progress"
                ),
            }]
            if bool(getattr(self.cfg, "state_coverage_enabled", True)):
                batches.append({
                    "batch_type": "state_coverage_repair",
                    "optimization_route": "coverage_repair",
                    "priority": 1,
                    "cases": coverage[:target_error_limit],
                    "purpose": (
                        "repair deterministically assigned C0/C1 residual cases to create C0-to-C1 "
                        "or C1-to-C2 correct coverage without sacrificing existing correct cases"
                    ),
                })
            if bool(getattr(self.cfg, "state_c2_wrong_split_enabled", True)):
                batches.append({
                    "batch_type": "state_vote_conversion",
                    "optimization_route": "vote_conversion",
                    "priority": 2,
                    "cases": conversion[:target_error_limit],
                    "purpose": (
                        "first make the target correct on C2 cases to create C2-to-C3; only if it remains "
                        "wrong, reduce a dominant wrong cluster on theoretically rescuable C2 cases"
                    ),
                })
            return [batch for batch in batches if batch["optimization_route"] == "general_accuracy" or batch.get("cases")]
        if self._is_accuracy_only_mode():
            error_cases = self._accuracy_cases_for_agent(diagnosis, agent_id)
            random_cases = [
                c for c in self._window_random_case_summaries(agent_id, max(0, int(self.cfg.random_window_cases_per_agent)))
                if isinstance(c, dict)
            ]
            batches = [
                {
                    "batch_type": "target_error_repair",
                    "priority": -2,
                    "cases": target_error_cases[:target_error_limit],
                    "purpose": "repair target-agent observed error patterns before changing diversity",
                },
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
            return [b for b in batches if b.get("cases") or str(b.get("batch_type")) in {"target_error_repair", "accuracy_error_cases"}]
        if self._is_v82_hybrid():
            def take(predicate: Any, count: int, used: set) -> List[Dict[str, Any]]:
                if count <= 0:
                    return []
                rows = []
                for case in target_error_cases:
                    case_id = str(case.get("case_id", ""))
                    if case_id in used or not predicate(case):
                        continue
                    rows.append(case)
                    used.add(case_id)
                    if len(rows) >= count:
                        break
                return rows

            used: set = set()
            c1 = take(lambda case: int(case.get("baseline_correct_count", -1)) == 0, 1, used)
            c2 = take(lambda case: int(case.get("baseline_correct_count", -1)) == 1, 1, used)
            boundary = take(
                lambda case: str(case.get("case_type", "")) in {
                    "target_wrong_pivotal_vote_fix", "target_wrong_near_vote_boundary"
                },
                1,
                used,
            )
            residual = take(
                lambda case: str(case.get("case_type", "")) in {
                    "target_wrong_shared_error", "target_wrong_dominant_wrong_cluster"
                },
                1 if float(getattr(self, "specialization_strength", 0.0)) > 0.0 else 0,
                used,
            )
            general = take(lambda case: not bool(case.get("target_correct", False)), 2, used)
            budget = max(1, int(self.cfg.max_homogeneous_cases_per_agent) + int(self.cfg.random_window_cases_per_agent))
            chosen = general + c1 + c2 + boundary + residual
            for case in target_error_cases:
                if len(chosen) >= budget:
                    break
                case_id = str(case.get("case_id", ""))
                if case_id not in used:
                    chosen.append(case)
                    used.add(case_id)
            chosen_ids = {str(case.get("case_id", "")) for case in chosen}
            buckets = [
                ("general_error", general, "repair general target-agent errors and preserve competence"),
                ("c1_c2_creation", c1 + c2, "create C1/C2 coverage even when one repair cannot yet flip plurality"),
                ("actual_plurality_boundary", boundary, "repair cases verified by the actual plurality counterfactual"),
                ("residual_shared_error", residual, "reduce shared residual errors after specialization activates"),
            ]
            batches = []
            for priority, (batch_type, rows, purpose) in enumerate(buckets):
                retained = [case for case in rows if str(case.get("case_id", "")) in chosen_ids]
                if retained:
                    batches.append({"batch_type": batch_type, "priority": priority, "cases": retained, "purpose": purpose})
            if not batches and chosen:
                batches.append({"batch_type": "general_error", "priority": 0, "cases": chosen, "purpose": "repair general target-agent errors"})
            return batches
        if bool(getattr(self.cfg, "boundary_selector_enabled", False)):
            limit = max(1, int(self.cfg.max_homogeneous_cases_per_agent))
            by_type = lambda *names: [
                case for case in target_error_cases if str(case.get("case_type", "")) in set(names)
            ]
            pivotal = by_type("target_wrong_pivotal_vote_fix")
            near_shared = by_type(
                "target_wrong_near_vote_boundary",
                "target_wrong_shared_error",
                "target_wrong_dominant_wrong_cluster",
            )
            protection = by_type("pivotal_correct_protection")
            for state in self.agents[agent_id].accepted_behavior_archive[-5:]:
                for question_hash, entry in state.behavior_fingerprint.items():
                    target_correct = entry.target_correct if isinstance(entry, BehaviorFingerprintEntry) else bool(entry.get("target_correct", False))
                    team_correct = entry.team_vote_correct if isinstance(entry, BehaviorFingerprintEntry) else bool(entry.get("team_vote_correct", False))
                    if target_correct and not team_correct:
                        protection.append({
                            "case_id": self._hash(f"{question_hash}|historical_unique_correct|{agent_id}"),
                            "case_type": "pivotal_correct_protection",
                            "sample_hash": str(question_hash),
                            "target_agent_id": agent_id,
                            "target_correct": True,
                            "team_correct": False,
                            "historical_unique_correct": True,
                            "repair_hint": "preserve the mechanism that supplied a historically unique correct path",
                            "capability_residual_family": CapabilityResidualFamily.UNKNOWN.value,
                        })
            general = by_type(
                "target_wrong_peer_correct_nonboundary",
                "target_wrong_vote_already_correct",
                "target_invalid",
            )
            gaps = diagnosis.get("capability_coverage_gap", {})
            residual = [
                case for case in target_error_cases
                if str(case.get("case_type", "")) != "pivotal_correct_protection"
            ]
            residual.sort(
                key=lambda case: (
                    -float(gaps.get(str(case.get("capability_residual_family", "unknown")), 0.0)),
                    -float(self.agents[agent_id].capability_profile.get(str(case.get("capability_residual_family", "unknown")), 0.0)),
                    int(case.get("window_index", 0) or 0),
                )
            )
            random_cases = self._window_random_case_summaries(
                agent_id, max(0, int(self.cfg.random_window_cases_per_agent))
            )
            invalid_rate = float(diagnosis.get("per_agent_invalid_rate", [0.0] * len(self.agents))[agent_id])
            batches = [
                {
                    "batch_type": "pivotal_error_repair",
                    "priority": 0,
                    "cases": pivotal[:limit],
                    "purpose": "repair errors whose correction can directly recover a team vote",
                },
                {
                    "batch_type": "near_boundary_shared_error_repair",
                    "priority": 1,
                    "cases": near_shared[:limit],
                    "purpose": "reduce harmful shared-error mechanisms near the vote boundary",
                },
                {
                    "batch_type": "residual_capability_repair",
                    "priority": 2,
                    "cases": residual[:limit],
                    "purpose": "make one local executable repair for the highest-pressure residual capability family",
                },
                {
                    "batch_type": "pivotal_correct_protection",
                    "priority": 3,
                    "cases": protection[:limit],
                    "purpose": "preserve mechanisms that already supply pivotal correct votes",
                },
                {
                    "batch_type": "general_accuracy_repair",
                    "priority": 4,
                    "cases": general[:limit],
                    "purpose": "repair remaining target-agent errors without broad prompt replacement",
                },
                {
                    "batch_type": "random_robustness",
                    "priority": 5,
                    "cases": random_cases,
                    "purpose": "check that the local repair remains robust on nearby cases",
                },
            ]
            if invalid_rate >= float(self.cfg.invalid_repair_rate_threshold):
                invalid_cases = by_type("target_invalid") or self._validity_cases_for_agent(diagnosis, agent_id)
                batches.insert(0, {
                    "batch_type": "hard_validity_repair",
                    "priority": -1,
                    "cases": invalid_cases[:limit],
                    "purpose": "repair elevated invalid output failures before other mechanisms",
                })
            return [batch for batch in batches if batch.get("cases")]
        top_cases = self._cases_for_agent(diagnosis, agent_id)[: max(0, int(self.cfg.max_homogeneous_cases_per_agent))]
        random_cases = self._window_random_case_summaries(agent_id, max(0, int(self.cfg.random_window_cases_per_agent)))
        validity_cases = self._validity_cases_for_agent(diagnosis, agent_id)[: max(0, int(self.cfg.hard_validity_cases_per_agent))]
        invalid_rate = 0.0
        rates = diagnosis.get("per_agent_invalid_rate", [])
        if agent_id < len(rates):
            invalid_rate = float(rates[agent_id])
        batches = [
            {
                "batch_type": "target_error_repair",
                "priority": -3,
                "cases": target_error_cases[:target_error_limit],
                "purpose": "repair target-agent wrong, invalid, or unhelpful behavior using abstract error-pattern evidence",
            },
            {
                "batch_type": "target_error_repair",
                "priority": -2,
                "cases": target_error_cases[target_error_limit : target_error_limit * 2] or target_error_cases[:target_error_limit],
                "purpose": "produce an alternative accuracy-repair procedure for the same target-agent blind spots",
            },
            {
                "batch_type": "useful_diversity_repair",
                "priority": 1,
                "cases": top_cases,
                "purpose": "turn redundant or correlated target-agent behavior into useful complementary reasoning",
            },
            {
                "batch_type": "random_window",
                "priority": 2,
                "cases": random_cases,
                "purpose": "avoid overfitting to only the highest-overlap cases",
            },
        ]
        if validity_cases or invalid_rate >= float(self.cfg.invalid_repair_rate_threshold):
            batches.append(
                {
                    "batch_type": "hard_validity_repair",
                    "priority": 0 if invalid_rate >= float(self.cfg.invalid_repair_rate_threshold) else 3,
                    "cases": validity_cases,
                    "purpose": "repair invalid or fragile target-agent outputs before pushing diversity",
                }
            )
        batches.sort(key=lambda x: int(x.get("priority", 0)))
        return [b for b in batches if b.get("cases") or str(b.get("batch_type")) != "target_error_repair"]

    def _optimizer_case_payload(self, case: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        allowed = [
            "case_id",
            "target_agent_id",
            "peer_agent_id",
            "pair_overlap",
            "target_prompt_preview",
            "peer_prompt_preview",
            "target_valid",
            "peer_valid",
            "case_type",
            "target_overlap_pressure",
            "invalid_reasons",
            "answer_present",
            "purpose",
            "team_correct",
            "window_index",
            "target_answer_preview",
            "peer_behavior_summary",
            "target_invalid",
            "target_correct",
            "peer_correct_available",
            "error_pattern",
            "repair_hint",
        ]
        if self._is_state_conditioned_method():
            allowed.extend([
                "baseline_correct_count", "state", "coverage_assigned_agents",
                "option_count", "question_hash", "sample_hash",
            ])
        if self._v7_residual_protocol_enabled():
            allowed.extend(["capability_residual_family", "confidence", "peer_wrong_count", "vote_context"])
        for key in allowed:
            if key in case:
                payload[key] = case.get(key)
        if "target_trace_preview" in case:
            payload["target_trace_preview"] = self._redact_optimizer_text(str(case.get("target_trace_preview", "")))
        if "trace_preview" in case and "target_trace_preview" not in payload:
            payload["target_trace_preview"] = self._redact_optimizer_text(str(case.get("trace_preview", "")))
        if "peer_trace_preview" in case:
            value = case.get("peer_trace_preview")
            if isinstance(value, list):
                payload["peer_trace_preview"] = [self._redact_optimizer_text(str(x), max_chars=240) for x in value[:2]]
            else:
                payload["peer_trace_preview"] = self._redact_optimizer_text(str(value), max_chars=240)
        if "target_answer_preview" not in payload and "target_answer" in case:
            payload["target_answer_preview"] = self._answer_behavior_preview(str(case.get("target_answer", "")))
        return payload

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
            "Your objective is to improve the target agent's answer accuracy on observed error patterns.\n"
            "Use the parent prompt, prompt-role previews, window accuracy statistics, and target-agent error cases.\n"
            "Useful reasoning diversity is allowed only when it helps the target agent repair mistakes.\n"
            "Do not optimize for semantic overlap, invalid-rate metrics, trace difference alone, or stylistic novelty.\n"
            "Do not use gold answers, concrete task text, options, labels, or answer-specific content.\n"
            "Treat trace previews as behavioral evidence of mistakes; do not copy their wording into the new prompt.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            "Revise the target agent prompt to reduce the observed answer mistakes.\n"
            "Priority order:\n"
            "1. Repair the target agent's observed error patterns.\n"
            "2. Preserve or improve target-agent answer accuracy.\n"
            "3. Add useful reasoning diversity only when it helps correctness or error rescue.\n"
            "4. Avoid invalid, verbose, generic, or merely paraphrased prompts.\n"
            "5. Do not optimize for trace difference alone.\n"
            "Each candidate must describe an executable reasoning procedure that can improve correctness on similar examples. "
            "Prefer concrete checks such as concept disambiguation, option comparison, contradiction testing, qualifier inspection, "
            "or final verification when they fit the observed mistake pattern.\n"
            "A candidate is invalid if it only paraphrases the parent prompt, appends generic caution, asks the solver to be more accurate, "
            "or changes style without adding a concrete error-repair procedure.\n"
            "Each candidate_prompt must contain a concrete reasoning procedure, a specific error-repair behavior, final answer discipline, "
            "and a short verification step.\n"
            "Write a complete short role prompt, not a suffix to append to the parent prompt. "
            "Do not repeat generic instructions already present in the parent prompt. "
            "Do not use the phrase 'Use a distinct decision procedure'. "
            "The prompt should remain short and usable by a solver agent. It must still end with exactly one final answer in normal solving, "
            "but do not include concrete answer labels or sample content inside candidate_prompt.\n"
            "Do not mention reward, beam search, candidates, evaluation metrics, or this optimizer instruction inside candidate_prompt.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"candidate_prompt": str, "role_name": str, "decision_procedure": [str, ...], "when_to_use": str, "fallback_strategy": str, "accuracy_checks": [str, ...], "target_error_pattern": str, "accuracy_repair_rule": str, "expected_accuracy_effect": str, "rationale": str, "source_batch_type": str},\n'
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
        if self._v7_residual_protocol_enabled():
            system_prompt = (
                system_prompt.replace("role prompts", "solver instructions")
                .replace("prompt-role previews", "prompt summaries")
                .replace("role previews", "prompt summaries")
            )
            user_prompt = (
                user_prompt.replace("role_name", "mechanism_name")
                .replace("target_role_spec", "target_prompt_state")
                .replace("peer_role_specs", "peer_prompt_summaries")
                .replace("complete short role prompt", "complete short solver instruction")
                .replace("role prompt", "solver instruction")
                .replace("peer roles", "peer prompts")
                .replace("prompt-role previews", "prompt summaries")
            )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.cfg.optimizer_temperature),
            max_tokens=int(self.cfg.optimizer_max_tokens),
            stage=f"accuracy_optimizer_agent_{agent_id}",
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
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(item.get("role_name", "")),
                        "mechanism_name": str(item.get("mechanism_name", item.get("role_name", ""))),
                        "decision_procedure": item.get("decision_procedure", []),
                        "when_to_use": str(item.get("when_to_use", "")),
                        "fallback_strategy": str(item.get("fallback_strategy", "")),
                        "accuracy_checks": item.get("accuracy_checks", []),
                        "target_error_pattern": str(item.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(item.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(item.get("expected_accuracy_effect", "")),
                        "rationale": str(item.get("rationale", "")),
                        "candidate_source": "optimizer",
                        "optimizer_generation_diagnostics": dict(diagnostics),
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
        fallback_mode_cfg = str(getattr(self.cfg, "optimizer_fallback_mode", "none") or "none").lower()
        if fallback_mode_cfg == "template":
            while len(parsed) < num_candidates:
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                fallback = self._structured_fallback_role(agent_id, len(parsed), mode="accuracy")
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
                        "accuracy_checks": list(fallback["accuracy_checks"]),
                        "target_error_pattern": str(fallback.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(fallback.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(fallback.get("expected_accuracy_effect", "")),
                        "rationale": "Fallback candidate when optimizer returns too few usable prompts.",
                        "candidate_source": "accuracy_repair_fallback",
                        "optimizer_generation_diagnostics": dict(diagnostics),
                        "generation_batch_type": str(generation_batches[batch_idx].get("batch_type", "")),
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
