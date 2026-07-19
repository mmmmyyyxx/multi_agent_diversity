"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class CandidateSchemaMixin:
    def sync_prompt_history_current_state(
        self,
        event: str = "sync_current_state",
        epoch: Any = "final",
        step: int = 0,
        selected_epoch: Optional[int] = None,
    ):
        for agent_id, agent in enumerate(self.agents):
            key = str(agent_id)
            row = self.prompt_history.setdefault(
                key,
                {
                    "initial_prompt": getattr(agent, "initial_prompt", ""),
                    "initial_prompt_hash": self._hash(getattr(agent, "initial_prompt", "")),
                    "events": [],
                },
            )
            row["current_prompt"] = agent.current_prompt
            row["current_prompt_hash"] = self._hash(agent.current_prompt)
            row["prompt_beam"] = agent.prompt_beam
            event_row = {
                "epoch": epoch,
                "step": step,
                "decision": event,
                "changed": self._hash(agent.current_prompt) != row.get("initial_prompt_hash", ""),
                "current_prompt_hash": self._hash(agent.current_prompt),
            }
            if selected_epoch is not None:
                event_row["selected_epoch"] = int(selected_epoch)
            row.setdefault("events", []).append(event_row)

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

    def _prompt_signature(self, prompt: str) -> str:
        return normalize_spaces(str(prompt or "")).lower()

    def _is_redundant_candidate_prompt(
        self,
        parent_prompt: str,
        candidate_prompt: str,
        seen_signatures: Optional[set] = None,
        *,
        allow_substantive_parent_extension: bool = False,
    ) -> bool:
        candidate_sig = self._prompt_signature(candidate_prompt)
        if not candidate_sig:
            return True
        if seen_signatures and candidate_sig in seen_signatures:
            return True

        parent_sig = self._prompt_signature(parent_prompt)
        stock_sig = self._prompt_signature(self.GENERIC_DISTINCT_PROCEDURE)
        stock_count = candidate_sig.count(stock_sig)
        parent_stock_count = parent_sig.count(stock_sig)

        if candidate_sig == parent_sig:
            return True
        if stock_count > 1:
            return True
        if (
            parent_sig
            and candidate_sig.startswith(parent_sig)
            and len(candidate_sig) > len(parent_sig) + 40
            and not allow_substantive_parent_extension
        ):
            return True
        if stock_count > parent_stock_count and parent_stock_count > 0:
            return True
        return False

    def _empty_optimizer_generation_diagnostics(self) -> Dict[str, Any]:
        return {
            "optimizer_architecture": str(getattr(self.cfg, "optimizer_architecture", "one_shot") or "one_shot"),
            "optimizer_raw_response_empty": 0,
            "optimizer_json_parse_failed": 0,
            "optimizer_raw_candidate_count": 0,
            "optimizer_empty_prompt_count": 0,
            "optimizer_sanitized_count": 0,
            "optimizer_redundant_filtered_count": 0,
            "optimizer_schema_filtered_count": 0,
            "optimizer_final_candidate_count": 0,
            "optimizer_underfilled": False,
            "teacher_question": "",
            "tcs_call_group_id": "",
            "execution_session_id": "",
            "update_attempt_id": "",
            "teacher_question_approved": False,
            "teacher_question_rejected": False,
            "teacher_question_rejection_reason": "",
            "teacher_question_forced_best_score": False,
            "teacher_question_forced_best_round": 0,
            "teacher_question_forced_below_threshold": False,
            "teacher_question_score": 0.0,
            "teacher_critic_rounds": 0,
            "teacher_quality_critique": "",
            "teacher_specificity_critique": "",
            "teacher_task_alignment_critique": "",
            "teacher_error_alignment_critique": "",
            "teacher_diversity_critique": "",
            "teacher_rewrite_count": 0,
            "student_candidate_count_raw": 0,
            "student_candidate_count_final": 0,
            "student_candidate_filtered_count": 0,
            "student_candidate_filter_reasons": [],
            "student_all_candidates_filtered": False,
            "student_missing_required_field_count": 0,
            "student_missing_required_fields": [],
            "student_raw_response_empty": False,
            "student_raw_response_preview": "",
            "student_json_parse_failed": False,
            "student_json_parse_error": "",
            "student_json_retry_attempted": False,
            "student_json_retry_succeeded": False,
            "student_json_retry_raw_response_preview": "",
            "student_json_repair_attempted": False,
            "student_json_repair_succeeded": False,
            "student_json_repair_raw_response_preview": "",
            "student_json_repair_failure_reason": "",
            "student_json_has_candidates_key": False,
            "student_candidates_is_list": False,
            "student_candidates_empty_list": False,
            "student_refusal_or_explanation": False,
            "student_failure_stage": "",
            "num_teacher_calls": 0,
            "num_critic_calls": 0,
            "num_teacher_rewrite_calls": 0,
            "num_student_calls": 0,
            "num_student_retry_calls": 0,
            "num_student_repair_calls": 0,
        }

    def _record_optimizer_generation_diagnostics(
        self,
        agent_id: int,
        parent_id: str,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = self._empty_optimizer_generation_diagnostics()
        normalized.update(diagnostics or {})
        normalized["optimizer_final_candidate_count"] = int(normalized.get("optimizer_final_candidate_count", 0) or 0)
        normalized["optimizer_underfilled"] = bool(normalized.get("optimizer_underfilled", False))
        if not hasattr(self, "optimizer_generation_diagnostics"):
            self.optimizer_generation_diagnostics = {}
        key = f"{int(agent_id)}:{str(parent_id)}"
        self.optimizer_generation_diagnostics[key] = dict(normalized)
        return normalized

    def _optimizer_generation_diagnostics_for_parent(self, agent_id: int, parent_id: str) -> Dict[str, Any]:
        if not hasattr(self, "optimizer_generation_diagnostics"):
            self.optimizer_generation_diagnostics = {}
        key = f"{int(agent_id)}:{str(parent_id)}"
        return dict(self.optimizer_generation_diagnostics.get(key, self._empty_optimizer_generation_diagnostics()))

    def _required_optimizer_fields(self, architecture: Optional[str] = None) -> List[str]:
        arch = str(architecture or getattr(self.cfg, "optimizer_architecture", "one_shot") or "one_shot").lower()
        if arch == "teacher_critic_student":
            required = [
                "candidate_prompt",
                "student_interpretation_of_question",
                "target_error_pattern",
                "accuracy_repair_rule",
                "diversity_contribution",
                "error_correlation_reduction",
                "task_alignment_rule",
                "peer_redundancy_avoidance",
                "expected_accuracy_effect",
                "expected_diversity_effect",
                "risk_control",
                "rationale",
            ]
            if self._v7_residual_protocol_enabled():
                required.extend([
                    "preserved_mechanisms",
                    "modified_mechanism",
                    "change_summary",
                    "target_residual_family",
                    "expected_shared_error_effect",
                ])
            if self._is_v82_hybrid():
                required.extend(["candidate_type", "mechanism_steps", "target_failure_buckets", "expected_effect"])
            return required
        return ["candidate_prompt"]

    def _missing_optimizer_fields(
        self,
        item: Dict[str, Any],
        architecture: Optional[str] = None,
    ) -> List[str]:
        missing = []
        for field in self._required_optimizer_fields(architecture):
            value = item.get(field, None)
            if value is None:
                missing.append(field)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(field)
                continue
            if isinstance(value, list) and len(value) == 0:
                missing.append(field)
                continue
        return missing

    def _candidate_has_required_optimizer_fields(
        self,
        item: Dict[str, Any],
        architecture: Optional[str] = None,
    ) -> bool:
        return not self._missing_optimizer_fields(item, architecture)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
            text = str(value).strip()
            if not text:
                return float(default)
            try:
                return float(text)
            except Exception:
                pass
            match = re.search(r"[-+]?\d*\.?\d+", text)
            if match:
                return float(match.group(0))
            return float(default)
        except Exception:
            return float(default)

    def _is_optimizer_generated_candidate_source(self, source: Any) -> bool:
        text = str(source or "").strip().lower()
        return text in {"optimizer", "teacher_critic_student", "open_mechanism_exploration"}

    @staticmethod
    def _candidate_pool_source(item: Mapping[str, Any]) -> str:
        """Return where a candidate entered the pool, with legacy checkpoint support."""
        return str(item.get("candidate_pool_source") or item.get("source") or "").strip()

    @staticmethod
    def _candidate_generation_source(item: Mapping[str, Any]) -> str:
        """Return the mechanism that generated the candidate prompt."""
        return str(item.get("candidate_source") or "").strip()

    def _teacher_metadata_from_diagnostics(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
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
            "student_missing_required_field_count",
            "student_missing_required_fields",
            "student_raw_response_empty",
            "student_raw_response_preview",
            "student_json_parse_failed",
            "student_json_parse_error",
            "student_json_retry_attempted",
            "student_json_retry_succeeded",
            "student_json_retry_raw_response_preview",
            "student_json_repair_attempted",
            "student_json_repair_succeeded",
            "student_json_repair_raw_response_preview",
            "student_json_repair_failure_reason",
            "student_json_has_candidates_key",
            "student_candidates_is_list",
            "student_candidates_empty_list",
            "student_refusal_or_explanation",
            "student_failure_stage",
        ]
        return {key: diagnostics.get(key, self._empty_optimizer_generation_diagnostics().get(key)) for key in keys}

    @staticmethod
    def _merge_student_diagnostics(diagnostics: Dict[str, Any], student_diagnostics: Mapping[str, Any]) -> None:
        """Student defaults must not erase Teacher/Critic provenance from the same parent."""
        for key, value in student_diagnostics.items():
            if str(key).startswith("student_"):
                diagnostics[key] = value

    def _student_failure_log_fields(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics = diagnostics or {}
        return {
            "student_raw_response_empty": bool(diagnostics.get("student_raw_response_empty", False)),
            "student_raw_response_preview": str(diagnostics.get("student_raw_response_preview", ""))[:1000],
            "student_json_parse_failed": bool(diagnostics.get("student_json_parse_failed", False)),
            "student_json_parse_error": str(diagnostics.get("student_json_parse_error", ""))[:500],
            "student_json_retry_attempted": bool(diagnostics.get("student_json_retry_attempted", False)),
            "student_json_retry_succeeded": bool(diagnostics.get("student_json_retry_succeeded", False)),
            "student_json_retry_raw_response_preview": str(diagnostics.get("student_json_retry_raw_response_preview", ""))[:1000],
            "student_json_repair_attempted": bool(diagnostics.get("student_json_repair_attempted", False)),
            "student_json_repair_succeeded": bool(diagnostics.get("student_json_repair_succeeded", False)),
            "student_json_repair_raw_response_preview": str(diagnostics.get("student_json_repair_raw_response_preview", ""))[:1000],
            "student_json_repair_failure_reason": str(diagnostics.get("student_json_repair_failure_reason", ""))[:500],
            "student_json_has_candidates_key": bool(diagnostics.get("student_json_has_candidates_key", False)),
            "student_candidates_is_list": bool(diagnostics.get("student_candidates_is_list", False)),
            "student_candidates_empty_list": bool(diagnostics.get("student_candidates_empty_list", False)),
            "student_refusal_or_explanation": bool(diagnostics.get("student_refusal_or_explanation", False)),
            "student_failure_stage": str(diagnostics.get("student_failure_stage", "")),
        }

    def _student_candidate_schema_json(self) -> str:
        prompt_limit = int(
            getattr(self.cfg, "student_candidate_prompt_hard_max_chars", 1400)
            if bool(getattr(self.cfg, "competence_depth_enabled", False))
            else getattr(self.cfg, "student_candidate_prompt_max_chars", 900)
        )
        candidate_schema = {
                    "candidate_prompt": f"standalone complete prompt, <= {prompt_limit} chars",
                    "student_interpretation_of_question": "one short sentence",
                    "target_error_pattern": "short phrase",
                    "accuracy_repair_rule": "one short sentence",
                    "diversity_contribution": "one short sentence",
                    "error_correlation_reduction": "one short sentence",
                    "task_alignment_rule": "one short sentence",
                    "peer_redundancy_avoidance": "one short sentence",
                    "expected_accuracy_effect": "one short sentence",
                    "expected_diversity_effect": "one short sentence",
                    "risk_control": "one short sentence",
                    "rationale": "one short sentence",
                    "change_summary": "optional short local-edit summary",
                    "preserved_mechanisms": ["optional preserved mechanism"],
                    "new_or_modified_mechanism": "optional changed mechanism",
        }
        if self._v7_residual_protocol_enabled():
            candidate_schema.update({
                "modified_mechanism": "one local mechanism changed in v7",
                "target_residual_family": "task-independent residual family",
                "expected_shared_error_effect": "one short sentence",
            })
        if self._is_v82_hybrid():
            candidate_schema.update({
                "candidate_type": "task_specific_repair or mechanism_alternative",
                "mechanism_steps": ["ordered executable decision operation"],
                "target_failure_buckets": ["general_error, c1_creation, c2_creation, boundary, or residual"],
                "expected_effect": "one short sentence",
            })
        schema = {"candidates": [candidate_schema]}
        return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _hybrid_candidate_type_rejection_reason(candidate_type: str, seen_types: set) -> str:
        normalized = str(candidate_type or "").strip().lower()
        if normalized not in {"task_specific_repair", "mechanism_alternative"}:
            return f"invalid_candidate_type:{normalized or 'missing'}"
        if normalized in seen_types:
            return f"duplicate_candidate_type:{normalized}"
        return ""

    def _student_refusal_or_explanation(self, text: str) -> bool:
        lowered = normalize_spaces(text).lower()
        refusal_markers = [
            "i cannot",
            "i can't",
            "unable to",
            "cannot comply",
            "sorry",
            "as an ai",
            "instead",
            "here is",
            "i will",
        ]
        return any(marker in lowered for marker in refusal_markers)

    def _truncate_candidate_text_fields(
        self,
        item: Dict[str, Any],
        prompt_max_chars: Optional[int] = None,
        field_max_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt_max = int(prompt_max_chars or getattr(self.cfg, "student_candidate_prompt_max_chars", 900) or 900)
        field_max = int(field_max_chars or getattr(self.cfg, "student_candidate_max_chars_per_field", 320) or 320)
        out = dict(item or {})
        for key, value in list(out.items()):
            if isinstance(value, str):
                value = normalize_spaces(value)
                if key == "candidate_prompt":
                    out[key] = value[:prompt_max]
                else:
                    out[key] = value[:field_max]
        return out

    @staticmethod
    def _prompt_ends_with_sentence_boundary(prompt: str) -> bool:
        return bool(re.search(r"[.!?;:)\]}'\"]\s*$", str(prompt or "").strip()))

    def _prepare_v8_candidate_text_fields(self, item: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        out = dict(item or {})
        field_max = int(getattr(self.cfg, "student_candidate_max_chars_per_field", 320) or 320)
        for key, value in list(out.items()):
            if isinstance(value, str):
                out[key] = normalize_spaces(value) if key == "candidate_prompt" else normalize_spaces(value)[:field_max]
        prompt = str(out.get("candidate_prompt", "")).strip()
        soft = int(getattr(self.cfg, "student_candidate_prompt_soft_max_chars", 1100) or 1100)
        hard = int(getattr(self.cfg, "student_candidate_prompt_hard_max_chars", 1400) or 1400)
        audit = {
            "candidate_prompt_char_count": len(prompt),
            "candidate_prompt_over_soft_limit": len(prompt) > soft,
            "candidate_prompt_over_hard_limit": len(prompt) > hard,
            "candidate_prompt_overlength_rejected": len(prompt) > hard,
            "candidate_prompt_ends_with_sentence_boundary": self._prompt_ends_with_sentence_boundary(prompt),
        }
        if len(prompt) > hard:
            self.prompt_overlength_rejection_count += 1
            return None, audit
        if prompt and not audit["candidate_prompt_ends_with_sentence_boundary"]:
            audit["candidate_prompt_incomplete_rejected"] = True
            return None, audit
        return out, audit

    def _structured_fallback_role(self, agent_id: int, index: int, mode: str = "diversity") -> Dict[str, Any]:
        def finalize(row: Dict[str, Any]) -> Dict[str, Any]:
            if not self._v7_residual_protocol_enabled():
                return row
            result = dict(row)
            result["mechanism_name"] = str(result.pop("role_name", "local_reasoning_mechanism"))
            prompt = str(result.get("candidate_prompt", ""))
            prompt = re.sub(r"^You are (?:an?|the) [^.]+\.\s*", "", prompt, flags=re.IGNORECASE)
            result["candidate_prompt"] = "Use this solver instruction: " + prompt
            return result

        accuracy_repair_roles = [
            {
                "role_name": "constraint_verifier",
                "candidate_prompt": (
                    "You are a constraint verifier. Before answering, list the explicit constraints and qualifiers in the question. "
                    "Reject any answer that satisfies the general pattern but violates a stated constraint. "
                    "Then give exactly one final answer after a brief consistency check."
                ),
                "decision_procedure": [
                    "list explicit constraints",
                    "compare plausible answers against constraints",
                    "reject constraint violations",
                    "final consistency check",
                ],
                "when_to_use": "Use when the target agent misses qualifiers, exceptions, or stated constraints.",
                "fallback_strategy": "If no explicit constraints are visible, compare the two most plausible answers and verify the final choice.",
                "target_error_pattern": "missed_constraint",
                "accuracy_repair_rule": "force explicit constraint listing before selecting the final answer",
                "expected_accuracy_effect": "reduces premature selections that violate stated conditions",
            },
            {
                "role_name": "option_elimination_specialist",
                "candidate_prompt": (
                    "You are an option-elimination specialist. Compare the plausible answer choices one by one. "
                    "For each choice, state the strongest reason it could be correct and the strongest reason it could fail. "
                    "Choose the answer with the fewest unresolved failures, then output exactly one final answer."
                ),
                "decision_procedure": [
                    "identify plausible choices",
                    "test each choice",
                    "eliminate unsupported choices",
                    "select final answer",
                ],
                "when_to_use": "Use when the target agent jumps to a plausible answer without eliminating alternatives.",
                "fallback_strategy": "If choices are implicit, name the plausible interpretations and eliminate them as alternatives.",
                "target_error_pattern": "insufficient_option_elimination",
                "accuracy_repair_rule": "require option-by-option or interpretation-by-interpretation elimination",
                "expected_accuracy_effect": "makes the target agent compare alternatives instead of following the first plausible route",
            },
            {
                "role_name": "reverse_answer_validator",
                "candidate_prompt": (
                    "You are a reverse-answer validator. Start from the most plausible candidate answers and ask what must be true for each to be correct. "
                    "Reject candidates whose required assumptions conflict with the question. "
                    "Select the answer with the strongest support and provide exactly one final answer."
                ),
                "decision_procedure": [
                    "name plausible candidates",
                    "derive required assumptions",
                    "reject conflicting assumptions",
                    "final answer",
                ],
                "when_to_use": "Use when the target agent gives weakly supported answers or fails to verify assumptions.",
                "fallback_strategy": "If assumptions are hard to name, run a contradiction check on the selected answer before finalizing.",
                "target_error_pattern": "weak_verification",
                "accuracy_repair_rule": "validate the selected answer by checking the assumptions it requires",
                "expected_accuracy_effect": "catches unsupported selections before the final answer is emitted",
            },
            {
                "role_name": "format_and_answer_auditor",
                "candidate_prompt": (
                    "You are a format-and-answer auditor. Solve the problem normally, then audit the final answer format before responding. "
                    "Ensure the final response contains exactly one answer in the required format and no extra alternatives."
                ),
                "decision_procedure": [
                    "solve",
                    "check answer format",
                    "remove extra alternatives",
                    "emit exactly one final answer",
                ],
                "when_to_use": "Use when the target agent omits, duplicates, or malforms the final answer.",
                "fallback_strategy": "If the reasoning is uncertain, still emit one best-supported final answer in the required format.",
                "target_error_pattern": "invalid_or_missing_final_answer",
                "accuracy_repair_rule": "add a final answer audit that enforces exactly one valid answer",
                "expected_accuracy_effect": "reduces invalid outputs and missing-answer failures",
            },
        ]
        if str(mode).lower() in {"accuracy_repair", "accuracy"}:
            role = accuracy_repair_roles[(int(agent_id) + int(index)) % len(accuracy_repair_roles)]
            return finalize({
                **role,
                "anti_overlap_rule": "Use the named repair procedure because it fixes a target-agent error pattern, not because it sounds different.",
                "validity_checks": ["trace shows the repair procedure", "final answer is explicit", "no sample text is copied"],
                "accuracy_checks": ["repair rule is executed", "final answer is verified before output"],
            })

        roles = [
            {
                "role_name": "boundary_condition_checker",
                "candidate_prompt": (
                    "You are a boundary-condition checker. Before answering, list the explicit constraints, "
                    "edge cases, and qualifiers in the question. Eliminate choices or interpretations that violate "
                    "any constraint, then verify the final answer against each constraint."
                ),
                "decision_procedure": ["list constraints", "check edge cases", "eliminate violations", "verify final answer"],
                "when_to_use": "Use when errors come from missing qualifiers, edge cases, or hidden constraints.",
                "fallback_strategy": "If there are no clear constraints, switch to direct reasoning with one contradiction check.",
            },
            {
                "role_name": "reverse_validator",
                "candidate_prompt": (
                    "You are a reverse validator. Start from the strongest candidate answers and ask what would need "
                    "to be true for each one. Reject candidates whose required assumptions conflict with the question, "
                    "then choose the answer with the fewest unsupported assumptions."
                ),
                "decision_procedure": ["name strongest candidates", "derive required assumptions", "reject conflicts", "choose supported answer"],
                "when_to_use": "Use when several answers look plausible but one fails under reverse checking.",
                "fallback_strategy": "If no candidate can be reverse-checked, compare the two most plausible answers directly.",
            },
            {
                "role_name": "counterexample_tester",
                "candidate_prompt": (
                    "You are a counterexample tester. For each plausible answer, try to construct a minimal counterexample "
                    "or contradiction. Prefer the answer that survives counterexample search, and perform one final consistency check."
                ),
                "decision_procedure": ["identify plausible answers", "search counterexamples", "compare survivors", "run consistency check"],
                "when_to_use": "Use when the team is overconfident in a tempting but brittle answer.",
                "fallback_strategy": "If counterexamples are not meaningful, use explicit option elimination.",
            },
            {
                "role_name": "representation_converter",
                "candidate_prompt": (
                    "You are a representation converter. Rewrite the problem into a compact alternative form such as "
                    "a table, symbolic relation, coordinate list, or cause-effect chain. Solve from that representation "
                    "and verify that it preserves the original question."
                ),
                "decision_procedure": ["convert representation", "solve converted form", "map back to original", "verify preservation"],
                "when_to_use": "Use when direct wording is confusing or spatial, logical, or relational structure matters.",
                "fallback_strategy": "If conversion adds no clarity, return to concise direct reasoning.",
            },
            {
                "role_name": "ambiguity_resolver",
                "candidate_prompt": (
                    "You are an ambiguity resolver. Identify the key ambiguous phrase, pronoun, rule, or label. "
                    "Test each interpretation against the full context, discard interpretations that require unstated facts, "
                    "and answer using the interpretation best supported by the text."
                ),
                "decision_procedure": ["find ambiguity", "test interpretations", "discard unstated assumptions", "answer supported reading"],
                "when_to_use": "Use when mistakes come from pronoun, label, or wording ambiguity.",
                "fallback_strategy": "If no ambiguity exists, use boundary-condition checking.",
            },
        ]
        order = [0, 3, 1, 2, 4]
        role = roles[order[(int(agent_id) + int(index)) % len(order)]]
        return finalize({
            **role,
            "anti_overlap_rule": "Use the named procedure explicitly instead of repeating the default decomposition order.",
            "validity_checks": ["trace shows the named procedure", "final answer is explicit", "no sample text is copied"],
            "accuracy_checks": ["compare plausible alternatives", "verify the final choice against the question"],
        })
