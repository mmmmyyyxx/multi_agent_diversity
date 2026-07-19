"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class RolloutMetricsMixin:
    async def solve_with_current_prompts(self, question: str) -> Tuple[List[str], List[str]]:
        return await self.solve_with_prompts(question, self._active_prompt_list())

    async def solve_with_agent_prompt_override(
        self,
        question: str,
        agent_id: int,
        prompt: str,
        peer_prompts: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        prompts = list(peer_prompts or self._active_prompt_list())
        while len(prompts) < len(self.agents):
            prompts.append(self.agents[len(prompts)].current_prompt)
        prompts[agent_id] = prompt
        return await self.solve_with_prompts(question, prompts)

    def rule_invalid_check(self, trace: str, answer: str = "") -> Dict[str, Any]:
        text = str(trace or "")
        reasons = []
        tokens = re.findall(r"\w+", text)
        if len(normalize_spaces(text)) < 40:
            reasons.append("trace_too_short")
        if "FINAL_ANSWER:" not in text:
            reasons.append("missing_final_answer")
        if len(tokens) < 12:
            reasons.append("too_few_tokens")
        if not str(answer or "").strip():
            reasons.append("empty_extracted_answer")
        if len(tokens) >= 12:
            bigrams = list(zip(tokens, tokens[1:]))
            repeated = len(bigrams) - len(set(bigrams))
            ratio = repeated / max(1, len(bigrams))
            if ratio > 0.35:
                reasons.append("bigram_repetition")
        return {"invalid": int(bool(reasons)), "reasons": reasons}

    def _load_embedding_model(self):
        if self.embedding_model is not None:
            return self.embedding_model
        from sentence_transformers import SentenceTransformer

        model_name = str(self.cfg.embedding_model)
        try:
            self.embedding_model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.embedding_model = SentenceTransformer(model_name)
        return self.embedding_model

    def _split_trace_for_embedding(self, text: str) -> List[str]:
        words = normalize_spaces(text).split()
        if not words:
            return []
        chunk_words = max(1, int(self.cfg.trace_embedding_chunk_words or 320))
        overlap = max(0, min(int(self.cfg.trace_embedding_chunk_overlap or 0), chunk_words - 1))
        if len(words) <= chunk_words:
            return [" ".join(words)]
        chunks = []
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            chunk = words[start : start + chunk_words]
            if chunk:
                chunks.append(" ".join(chunk))
            if start + chunk_words >= len(words):
                break
        return chunks

    def _normalize_vector(self, vector: Any) -> List[float]:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= 0.0 or np.isnan(norm):
            return []
        return (arr / norm).astype(float).tolist()

    def _encode_trace_document(self, trace: str) -> List[float]:
        cleaned = normalize_spaces(trace)
        if not cleaned:
            return []
        cache_key = self._hash(f"{self.cfg.embedding_model}|{self.cfg.trace_embedding_chunk_words}|{self.cfg.trace_embedding_chunk_overlap}|{cleaned}")
        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        chunks = self._split_trace_for_embedding(cleaned)
        if not chunks:
            return []
        model = self._load_embedding_model()
        embeddings = model.encode(chunks, normalize_embeddings=True)
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            pooled = arr
        else:
            pooled = np.mean(arr, axis=0)
        vector = self._normalize_vector(pooled)
        self.embedding_cache[cache_key] = vector
        return vector

    def _vector_cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 1.0
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom <= 0.0 or np.isnan(denom):
            return 1.0
        sim = float(np.dot(va, vb) / denom)
        return float(max(-1.0, min(1.0, sim)))

    def embedding_overlap_diagnostics(
        self,
        traces: List[str],
        prompts: Optional[List[str]] = None,
        invalids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        n = len(traces)
        invalids = list(invalids or [0 for _ in traces])
        embeddings = [self._encode_trace_document(trace) for trace in traces]
        pair_rows = []
        per_agent_scores = [0.0 for _ in range(n)]
        per_agent_counts = [0 for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (i < len(invalids) and int(invalids[i]) > 0) or (j < len(invalids) and int(invalids[j]) > 0):
                    sim = 1.0
                else:
                    sim = self._vector_cosine_similarity(embeddings[i], embeddings[j])
                    sim = max(0.0, sim)
                pair_rows.append({"pair": [i, j], "overlap": sim})
                per_agent_scores[i] += sim
                per_agent_scores[j] += sim
                per_agent_counts[i] += 1
                per_agent_counts[j] += 1
        per_agent_overlap = [
            float(per_agent_scores[i] / per_agent_counts[i]) if per_agent_counts[i] else 0.0
            for i in range(n)
        ]
        mean_overlap = float(np.mean([p["overlap"] for p in pair_rows])) if pair_rows else 0.0
        threshold = float(self.cfg.homogeneity_overlap_threshold)
        for row in pair_rows:
            i, j = row["pair"]
            row["invalid_pair"] = bool(
                (i < len(invalids) and int(invalids[i]) > 0)
                or (j < len(invalids) and int(invalids[j]) > 0)
            )
        high_pairs = [p for p in pair_rows if float(p["overlap"]) >= threshold]
        roles = [
            {
                "agent_id": i,
                "prompt_preview": normalize_spaces((prompts or self._active_prompt_list())[i])[:220] if i < len(prompts or []) else "",
                "trace_hash": self._hash(traces[i]),
                "trace_preview": normalize_spaces(traces[i])[:360],
                "overlap_pressure": per_agent_overlap[i],
            }
            for i in range(n)
        ]
        embedding_diversity = max(0.0, min(1.0, 1.0 - mean_overlap))
        return {
            "mean_embedding_overlap": mean_overlap,
            "embedding_diversity": embedding_diversity,
            "trace_embedding_model": str(self.cfg.embedding_model),
            "trace_embedding_chunk_words": int(self.cfg.trace_embedding_chunk_words),
            "trace_embedding_chunk_overlap": int(self.cfg.trace_embedding_chunk_overlap),
            "per_agent_overlap": per_agent_overlap,
            "pair_overlaps": pair_rows,
            "high_overlap_pairs": high_pairs,
            "homogeneity_overlap_threshold": threshold,
            "roles": roles,
        }

    def _vote_with_diagnostics(self, answers: List[str], question_hash: str = "") -> Dict[str, Any]:
        return plurality_vote_with_diagnostics(
            answers,
            tie_break_method=str(getattr(self.cfg, "vote_tie_break", "random")),
            seed=int(getattr(self.cfg, "seed", 0) or 0),
            question_hash=question_hash,
        )

    def _rollout_any_correct(self, rollout: Dict[str, Any]) -> int:
        return int(any(int(x) for x in rollout.get("individual_correct", [])))

    def _safe_agent_correct(self, rollout: Dict[str, Any], agent_id: int) -> int:
        individual = list(rollout.get("individual_correct", []))
        return int(individual[agent_id]) if 0 <= int(agent_id) < len(individual) else 0

    def _target_trace_novelty(self, traces: List[str], target_agent_id: int) -> float:
        try:
            target_agent_id = int(target_agent_id)
            if target_agent_id < 0 or target_agent_id >= len(traces):
                return 0.0
            target = self._encode_trace_document(str(traces[target_agent_id]))
            peers = [
                self._encode_trace_document(str(trace))
                for i, trace in enumerate(traces)
                if i != target_agent_id and str(trace or "").strip()
            ]
            sims = [self._vector_cosine_similarity(target, peer) for peer in peers if peer]
            if not target or not sims:
                return 0.0
            return self._clip01(1.0 - float(np.mean([max(0.0, sim) for sim in sims])))
        except Exception:
            return 0.0

    def _trace_diversity_for_indices(self, traces: List[str], indices: List[int]) -> float:
        try:
            embeddings = [self._encode_trace_document(str(traces[i])) for i in indices if 0 <= i < len(traces)]
            embeddings = [vec for vec in embeddings if vec]
            if len(embeddings) < 2:
                return 0.0
            diversities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    diversities.append(1.0 - max(0.0, self._vector_cosine_similarity(embeddings[i], embeddings[j])))
            return self._clip01(float(np.mean(diversities)) if diversities else 0.0)
        except Exception:
            return 0.0

    def _useful_trace_diversity(
        self,
        traces: List[str],
        individual_correct: List[int],
        invalid_flags: List[int],
    ) -> float:
        indices = [
            i
            for i, correct in enumerate(individual_correct)
            if int(correct) > 0 and i < len(invalid_flags) and int(invalid_flags[i]) <= 0
        ]
        return self._trace_diversity_for_indices(traces, indices)

    def _weighted_vote_with_diagnostics(
        self,
        answers: List[str],
        invalid_flags: Optional[List[int]] = None,
        per_agent_overlap: Optional[List[float]] = None,
        question_hash: str = "",
    ) -> Dict[str, Any]:
        invalid_flags = list(invalid_flags or [0 for _ in answers])
        per_agent_overlap = list(per_agent_overlap or [0.0 for _ in answers])
        scores: Dict[str, float] = {}
        agent_weights = []
        for i, raw_answer in enumerate(answers):
            answer = str(raw_answer or "").strip()
            invalid = int(invalid_flags[i]) if i < len(invalid_flags) else 0
            overlap = self._clip01(per_agent_overlap[i]) if i < len(per_agent_overlap) else 0.0
            reliability = 1.0
            validity = 0.0 if invalid else 1.0
            independence = min(max(0.0, 1.0 - overlap), 0.5)
            weight = float(reliability * validity * independence)
            agent_weights.append(
                {
                    "agent_id": i,
                    "answer": answer,
                    "reliability": reliability,
                    "validity": validity,
                    "independence": independence,
                    "weight": weight,
                }
            )
            if answer and weight > 0.0:
                scores[answer] = scores.get(answer, 0.0) + weight

        fallback = False
        if not scores:
            fallback = True
            majority = self._vote_with_diagnostics(answers, question_hash=question_hash)
            return {
                "weighted_vote_answer": str(majority.get("vote_answer", "")),
                "weighted_vote_scores": {},
                "weighted_vote_tie": bool(majority.get("vote_tie", False)),
                "weighted_tie_candidates": list(majority.get("tie_candidates", [])),
                "weighted_tie_break_method": str(majority.get("tie_break_method", "")),
                "weighted_vote_agent_weights": agent_weights,
                "weighted_vote_fallback": fallback,
            }

        max_score = max(scores.values())
        tied = [answer for answer, score in scores.items() if abs(float(score) - float(max_score)) <= 1e-12]
        tied_set = set(tied)
        method = str(getattr(self.cfg, "vote_tie_break", "random") or "random").lower()
        if len(tied) <= 1:
            selected = tied[0]
        elif method == "abstain":
            selected = ""
        elif method == "random":
            seed_material = f"{int(getattr(self.cfg, 'seed', 0) or 0)}|{question_hash}|weighted_vote"
            rng = random.Random(int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:12], 16))
            selected = rng.choice(sorted(tied))
        else:
            selected = next((str(answer or "").strip() for answer in answers if str(answer or "").strip() in tied_set), sorted(tied)[0])
        return {
            "weighted_vote_answer": selected,
            "weighted_vote_scores": {key: float(value) for key, value in scores.items()},
            "weighted_vote_tie": len(tied) > 1,
            "weighted_tie_candidates": sorted(tied),
            "weighted_tie_break_method": method,
            "weighted_vote_agent_weights": agent_weights,
            "weighted_vote_fallback": fallback,
        }

    def compute_rollout_metrics(
        self,
        traces: List[str],
        answers: List[str],
        gold: str,
        prompts: Optional[List[str]] = None,
        question_hash: str = "",
    ) -> Dict[str, Any]:
        plurality_vote = self._vote_with_diagnostics(answers, question_hash=question_hash)
        plurality_vote_answer = str(plurality_vote.get("vote_answer", ""))
        individual_correct = [int(self.task_spec.match_answer(a, gold)) for a in answers]
        plurality_vote_correct = int(self.task_spec.match_answer(plurality_vote_answer, gold))
        gold_vote_diagnostics = compute_gold_vote_diagnostics(
            answers,
            gold,
            self.task_spec.match_answer,
            len(self.agents),
        )
        plurality_margin_votes = int(
            gold_vote_diagnostics.get("gold_vote_count", 0)
            - gold_vote_diagnostics.get("largest_wrong_vote_count", 0)
        )
        gold_vote_diagnostics.update({
            "plurality_margin_votes": plurality_margin_votes,
            "normalized_plurality_margin": float(
                gold_vote_diagnostics.get("normalized_vote_margin", -1.0)
            ),
            "strict_plurality_win": bool(plurality_margin_votes > 0),
            "plurality_gold_leading": bool(plurality_margin_votes > 0),
            "plurality_gold_top_tied": bool(plurality_margin_votes == 0 and bool(answers)),
            "plurality_gold_one_vote_behind": bool(plurality_margin_votes == -1),
            "plurality_gold_far_behind": bool(plurality_margin_votes <= -2),
        })
        if self._is_accuracy_only_mode():
            n = len(traces)
            active_prompts = prompts or self._active_prompt_list()
            roles = [
                {
                    "agent_id": i,
                    "prompt_preview": normalize_spaces(active_prompts[i])[:220] if i < len(active_prompts) else "",
                    "trace_hash": self._hash(traces[i]) if i < len(traces) else "",
                    "trace_preview": self._trace_method_preview(traces[i]) if i < len(traces) else "",
                    "overlap_pressure": 0.0,
                }
                for i in range(n)
            ]
            invalids = [0 for _ in traces]
            overlap = {
                "mean_embedding_overlap": 0.0,
                "embedding_diversity": 0.0,
                "trace_embedding_model": "",
                "trace_embedding_chunk_words": int(self.cfg.trace_embedding_chunk_words),
                "trace_embedding_chunk_overlap": int(self.cfg.trace_embedding_chunk_overlap),
                "per_agent_overlap": [0.0 for _ in traces],
                "pair_overlaps": [],
                "high_overlap_pairs": [],
                "homogeneity_overlap_threshold": float(self.cfg.homogeneity_overlap_threshold),
                "roles": roles,
            }
        else:
            invalids = [self.rule_invalid_check(traces[i], answers[i] if i < len(answers) else "").get("invalid", 1) for i in range(len(traces))]
            overlap = self.embedding_overlap_diagnostics(traces, prompts, invalids=invalids)
        weighted_vote = self._weighted_vote_with_diagnostics(
            answers,
            invalid_flags=[int(x) for x in invalids],
            per_agent_overlap=list(overlap.get("per_agent_overlap", [])),
            question_hash=question_hash,
        )
        weighted_vote_answer = str(weighted_vote.get("weighted_vote_answer", ""))
        weighted_vote_correct = int(self.task_spec.match_answer(weighted_vote_answer, gold))
        requested_aggregation_mode = str(getattr(self.cfg, "aggregation_mode", "majority") or "majority").lower()
        effective_aggregation_mode = canonical_aggregation_mode(requested_aggregation_mode)
        aggregation_fallback = ""
        if effective_aggregation_mode == "weighted_vote":
            vote_answer = weighted_vote_answer
            vote_correct = weighted_vote_correct
            vote_tie = bool(weighted_vote.get("weighted_vote_tie", False))
            tie_candidates = list(weighted_vote.get("weighted_tie_candidates", []))
            tie_break_method = str(weighted_vote.get("weighted_tie_break_method", ""))
        else:
            if effective_aggregation_mode == "verifier_select":
                aggregation_fallback = "verifier_select_not_implemented_fallback_majority"
                effective_aggregation_mode = "plurality"
            elif effective_aggregation_mode != "plurality":
                effective_aggregation_mode = "plurality"
            vote_answer = plurality_vote_answer
            vote_correct = plurality_vote_correct
            vote_tie = bool(plurality_vote.get("vote_tie", False))
            tie_candidates = list(plurality_vote.get("tie_candidates", []))
            tie_break_method = str(plurality_vote.get("tie_break_method", ""))
        any_correct = int(any(individual_correct))
        pivotal_fix_opportunities = []
        pivotal_holds = []
        for agent_id, correct in enumerate(individual_correct):
            opportunity = False
            hold = False
            if not correct and not plurality_vote_correct:
                counterfactual_answers = list(answers)
                counterfactual_answers[agent_id] = gold
                counterfactual = self._vote_with_diagnostics(counterfactual_answers, question_hash=question_hash)
                opportunity = bool(self.task_spec.match_answer(str(counterfactual.get("vote_answer", "")), gold))
            if correct and plurality_vote_correct:
                without_target = list(answers)
                without_target[agent_id] = ""
                counterfactual = self._vote_with_diagnostics(without_target, question_hash=question_hash)
                hold = not bool(self.task_spec.match_answer(str(counterfactual.get("vote_answer", "")), gold))
            pivotal_fix_opportunities.append(int(opportunity))
            pivotal_holds.append(int(hold))
        useful_diversity = 0.0 if self._is_accuracy_only_mode() else self._useful_trace_diversity(traces, individual_correct, [int(x) for x in invalids])
        return {
            "vote_answer": vote_answer,
            "vote_correct": vote_correct,
            "individual_correct": individual_correct,
            "vote_tie": vote_tie,
            "tie_candidates": tie_candidates,
            "vote_counts": dict(plurality_vote.get("vote_counts", {})),
            "tie_break_method": tie_break_method,
            "aggregation_mode": requested_aggregation_mode,
            "requested_aggregation_mode": requested_aggregation_mode,
            "effective_aggregation_mode": effective_aggregation_mode,
            "aggregation_fallback": aggregation_fallback,
            "plurality_boundary_version": PLURALITY_BOUNDARY_VERSION,
            "plurality_vote_answer": plurality_vote_answer,
            "plurality_vote_correct": plurality_vote_correct,
            "plurality_vote_tie": bool(plurality_vote.get("vote_tie", False)),
            "plurality_tie_candidates": list(plurality_vote.get("tie_candidates", [])),
            "plurality_vote_counts": dict(plurality_vote.get("vote_counts", {})),
            "plurality_tie_break_method": str(plurality_vote.get("tie_break_method", "")),
            "plurality_tie_break_question_hash": str(question_hash),
            "plurality_pivotal_fix_opportunity_per_agent": pivotal_fix_opportunities,
            "plurality_pivotal_hold_per_agent": pivotal_holds,
            "plurality_pivotal_fix_opportunity_rate": float(np.mean(pivotal_fix_opportunities)) if pivotal_fix_opportunities else 0.0,
            "plurality_pivotal_hold_rate": float(np.mean(pivotal_holds)) if pivotal_holds else 0.0,
            # Historical names remain diagnostic aliases for old readers.
            "majority_vote_answer": plurality_vote_answer,
            "majority_vote_correct": plurality_vote_correct,
            "majority_vote_tie": bool(plurality_vote.get("vote_tie", False)),
            "majority_tie_candidates": list(plurality_vote.get("tie_candidates", [])),
            "majority_vote_counts": dict(plurality_vote.get("vote_counts", {})),
            "majority_tie_break_method": str(plurality_vote.get("tie_break_method", "")),
            "weighted_vote_answer": weighted_vote_answer,
            "weighted_vote_correct": weighted_vote_correct,
            "weighted_vote_tie": bool(weighted_vote.get("weighted_vote_tie", False)),
            "weighted_tie_candidates": list(weighted_vote.get("weighted_tie_candidates", [])),
            "weighted_vote_scores": dict(weighted_vote.get("weighted_vote_scores", {})),
            "weighted_vote_agent_weights": list(weighted_vote.get("weighted_vote_agent_weights", [])),
            "weighted_vote_fallback": bool(weighted_vote.get("weighted_vote_fallback", False)),
            "any_correct": any_correct,
            **gold_vote_diagnostics,
            "useful_diversity": useful_diversity,
            "invalid_rate": float(np.mean(invalids)) if invalids else 1.0,
            "invalid_score": 1.0 - (float(np.mean(invalids)) if invalids else 1.0),
            "invalid_flags": [int(x) for x in invalids],
            **overlap,
        }

    def _trace_method_preview(self, trace: str, max_chars: int = 420) -> str:
        text = re.sub(r"FINAL_ANSWER\s*:\s*.*", "", str(trace or ""), flags=re.IGNORECASE)
        text = re.sub(r"\b(answer|final answer)\b\s*[:=-]?\s*[A-Da-d0-9.+/, \\-]+", "[answer redacted]", text, flags=re.IGNORECASE)
        return normalize_spaces(text)[:max_chars]

    def _redact_optimizer_text(self, text: str, max_chars: int = 420) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"FINAL_ANSWER\s*:\s*.*", "FINAL_ANSWER: [redacted]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(final answer|answer)\b\s*[:=-]\s*[^\n.;,]+", r"\1: [redacted]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(option|choice)\s+[A-Z]\b", "option [label]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\([A-Z]\)", "([label])", cleaned)
        cleaned = re.sub(r"\b[A-D]\s*[\).]\s*", "[label]. ", cleaned)
        cleaned = re.sub(r"\b(true|false|yes|no|valid|invalid)\b", "[boolean-like]", cleaned, flags=re.IGNORECASE)
        return normalize_spaces(cleaned)[:max_chars]

    def _answer_behavior_preview(self, answer: str) -> Dict[str, Any]:
        raw = str(answer or "").strip()
        lowered = raw.lower()
        if not raw:
            kind = "missing"
        elif re.fullmatch(r"\(?[A-Za-z]\)?\.?", raw):
            kind = "option_like"
        elif lowered in {"true", "false", "yes", "no", "valid", "invalid"}:
            kind = "boolean_like"
        elif re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", raw):
            kind = "numeric_like"
        else:
            kind = "text_like"
        return {
            "present": bool(raw),
            "length": len(raw),
            "kind": kind,
            "has_multiple_tokens": len(re.findall(r"\S+", raw)) > 1,
        }

    def _peer_behavior_summary(
        self,
        peer_traces: List[str],
        peer_correct_flags: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        flags = [int(x) for x in list(peer_correct_flags or [])]
        previews = [self._redact_optimizer_text(t, max_chars=240) for t in peer_traces[:2]]
        word_counts = [len(re.findall(r"\w+", str(t or ""))) for t in peer_traces]
        verification_terms = re.compile(r"\b(check|verify|therefore|because|constraint|eliminate|assumption|contradiction)\b", re.IGNORECASE)
        return {
            "num_peer_traces": len(peer_traces),
            "num_peer_correct": int(sum(flags)) if flags else 0,
            "peer_trace_previews": previews,
            "peer_longer_than_target": False,
            "mean_peer_trace_words": float(np.mean(word_counts)) if word_counts else 0.0,
            "peer_uses_verification_terms": bool(any(verification_terms.search(str(t or "")) for t in peer_traces)),
        }

    def _infer_target_error_pattern(
        self,
        target_trace: str,
        target_answer: str,
        peer_traces: List[str],
        rollout: Dict[str, Any],
        agent_id: int,
    ) -> Dict[str, Any]:
        invalid_flags = list(rollout.get("invalid_flags", [])) if isinstance(rollout, dict) else []
        invalid = int(invalid_flags[agent_id]) if agent_id < len(invalid_flags) else int(self.rule_invalid_check(target_trace, target_answer).get("invalid", 0))
        text = normalize_spaces(str(target_trace or ""))
        lower = text.lower()
        answer_preview = self._answer_behavior_preview(target_answer)
        target_words = len(re.findall(r"\w+", text))
        peer_words = [len(re.findall(r"\w+", str(t or ""))) for t in peer_traces]
        peer_mean_words = float(np.mean(peer_words)) if peer_words else 0.0

        def result(pattern: str, hint: str, family: CapabilityResidualFamily, confidence: float) -> Dict[str, Any]:
            return {
                "error_pattern": pattern,
                "repair_hint": hint,
                "capability_residual_family": family.value,
                "confidence": self._clip01(confidence),
            }

        if invalid or not answer_preview["present"]:
            if not answer_preview["present"] or "final_answer:" not in str(target_trace or ""):
                return result(
                    "invalid_or_missing_final_answer",
                    "add a final answer audit that emits exactly one answer in the required format",
                    CapabilityResidualFamily.OUTPUT_VALIDITY,
                    1.0,
                )
            return result(
                "format_violation",
                "check answer format and remove extra alternatives before finalizing",
                CapabilityResidualFamily.OUTPUT_VALIDITY,
                1.0,
            )
        if target_words < 35:
            return result(
                "premature_answer",
                "delay the final answer until after evidence comparison and a short verification step",
                CapabilityResidualFamily.FINAL_VERIFICATION,
                0.8,
            )
        if re.search(r"\b(calculate|compute|equation|number|sum|difference|multiply|divide|symbol|formula)\b", lower):
            if not re.search(r"\b(check|verify|substitut|unit|sanity)\b", lower):
                return result(
                    "calculation_or_symbolic_slip",
                    "add a numeric or symbolic sanity check before the final answer",
                    CapabilityResidualFamily.NUMERIC_SYMBOLIC,
                    0.9,
                )
        if re.search(r"\b(option|choice|alternative|candidate)\b", lower) and not re.search(r"\b(eliminate|reject|compare|fail|against)\b", lower):
            return result(
                "insufficient_option_elimination",
                "force option-by-option elimination before selecting the final answer",
                CapabilityResidualFamily.OPTION_COMPARISON,
                0.9,
            )
        if re.search(r"\b(constraint|except|unless|only|must|not|qualifier|condition)\b", lower) and not re.search(r"\b(list|check|satisfy|violate)\b", lower):
            return result(
                "missed_constraint",
                "force the agent to list explicit constraints before selecting an answer",
                CapabilityResidualFamily.QUALIFIER_NEGATION,
                0.85,
            )
        if re.search(r"\b(before|after|earlier|later|first|last|sequence|order|timeline|simultaneous)\b", lower):
            return result(
                "temporal_order_confusion",
                "construct and verify an explicit temporal ordering before selecting the answer",
                CapabilityResidualFamily.TEMPORAL_ORDER,
                0.75,
            )
        if re.search(r"\b(entity|person|object|name|pronoun|refer|correspond|bind)\b", lower):
            return result(
                "entity_binding_confusion",
                "bind each entity and reference explicitly before propagating constraints",
                CapabilityResidualFamily.ENTITY_BINDING,
                0.7,
            )
        if re.search(r"\b(relation|left|right|above|below|inside|between|adjacent|relative)\b", lower):
            return result(
                "relation_tracking_slip",
                "track each relation in a compact normalized representation and verify composition",
                CapabilityResidualFamily.RELATION_TRACKING,
                0.75,
            )
        if not re.search(r"\b(check|verify|therefore|because|contradiction|assumption|consistent)\b", lower):
            return result(
                "weak_verification",
                "add a final consistency check against the question before output",
                CapabilityResidualFamily.FINAL_VERIFICATION,
                0.75,
            )
        if re.search(r"\b(contradiction|inconsistent|assumption|counterexample|impossible)\b", lower):
            return result(
                "contradiction_check_failure",
                "test the provisional conclusion for contradiction or a concrete counterexample",
                CapabilityResidualFamily.CONTRADICTION_CHECK,
                0.7,
            )
        if peer_mean_words >= max(45.0, float(target_words) * 1.35):
            return result(
                "peer_has_more_specific_reasoning",
                "require grounding the answer in specific clues rather than generic reasoning",
                CapabilityResidualFamily.COMMONSENSE_CONSISTENCY,
                0.55,
            )
        generic_terms = len(re.findall(r"\b(careful|think|analyze|reason|solve|answer)\b", lower))
        evidence_terms = len(re.findall(r"\b(because|therefore|constraint|eliminate|verify|assumption|example|case)\b", lower))
        if generic_terms >= 4 and evidence_terms <= 1:
            return result(
                "overly_generic_reasoning",
                "replace generic reasoning with a concrete evidence-comparison procedure",
                CapabilityResidualFamily.COMMONSENSE_CONSISTENCY,
                0.5,
            )
        return result(
            "unknown_error_pattern",
            "use a concrete compare-then-verify procedure before the final answer",
            CapabilityResidualFamily.UNKNOWN,
            0.0,
        )

    def _case_key(self, sample_hash: str, a: int, b: int) -> str:
        left, right = sorted([int(a), int(b)])
        return f"{sample_hash}:{left}-{right}"

    def _build_homogeneous_cases(
        self,
        sample_hash: str,
        traces: List[str],
        answers: List[str],
        prompts: List[str],
        metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        invalids = [int(x) for x in metrics.get("invalid_flags", [])]
        cases: List[Dict[str, Any]] = []
        for pair in metrics.get("high_overlap_pairs", []):
            if not isinstance(pair, dict):
                continue
            ids = pair.get("pair", [])
            if not isinstance(ids, list) or len(ids) != 2:
                continue
            a, b = int(ids[0]), int(ids[1])
            if a >= len(traces) or b >= len(traces):
                continue
            if (a < len(invalids) and invalids[a]) or (b < len(invalids) and invalids[b]):
                continue
            overlap = float(pair.get("overlap", 0.0))
            for target, peer in [(a, b), (b, a)]:
                cases.append(
                    {
                        "case_id": self._hash(f"{sample_hash}|{target}|{peer}|{overlap:.6f}"),
                        "case_key": self._case_key(sample_hash, target, peer),
                        "sample_hash": sample_hash,
                        "target_agent_id": target,
                        "peer_agent_id": peer,
                        "pair_overlap": overlap,
                        "target_trace_preview": self._trace_method_preview(traces[target]),
                        "peer_trace_preview": self._trace_method_preview(traces[peer]),
                        "target_answer": str(answers[target]) if target < len(answers) else "",
                        "peer_answer": str(answers[peer]) if peer < len(answers) else "",
                        "target_prompt_preview": normalize_spaces(prompts[target])[:260] if target < len(prompts) else "",
                        "peer_prompt_preview": normalize_spaces(prompts[peer])[:260] if peer < len(prompts) else "",
                        "target_valid": True,
                        "peer_valid": True,
                        "team_correct": bool(metrics.get("vote_correct", 0)),
                        "case_type": "homogeneous_valid_pair",
                    }
                )
        return cases

    def _build_validity_cases(
        self,
        sample_hash: str,
        traces: List[str],
        answers: List[str],
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        rows = []
        for agent_id, trace in enumerate(traces):
            answer = answers[agent_id] if agent_id < len(answers) else ""
            invalid = self.rule_invalid_check(trace, answer)
            if not int(invalid.get("invalid", 0)):
                continue
            rows.append(
                {
                    "case_id": self._hash(f"{sample_hash}|invalid|{agent_id}|{self._hash(trace)}"),
                    "sample_hash": sample_hash,
                    "target_agent_id": agent_id,
                    "trace_preview": self._trace_method_preview(trace),
                    "answer_present": bool(str(answer).strip()),
                    "invalid_reasons": list(invalid.get("reasons", [])),
                    "target_prompt_preview": normalize_spaces(prompts[agent_id])[:260] if agent_id < len(prompts) else "",
                    "case_type": "hard_validity_case",
                }
            )
        return rows

    def is_homogeneity_window_warmup_done(self) -> bool:
        return all(len(a.recent_homogeneity_flags) >= self.homogeneity_window for a in self.agents)

    def is_update_window_ready(self) -> bool:
        return len(self.recent_window_records) >= self.homogeneity_window

    def clear_homogeneity_windows(self):
        for agent in self.agents:
            agent.recent_homogeneity_flags.clear()
            agent.homogeneity_count = 0
        self.recent_window_records = []

    def select_agents_for_update(self, metrics: Dict[str, Any]) -> List[int]:
        if not self.is_homogeneity_window_warmup_done():
            return []
        diagnosis = self._window_update_diagnosis(self.recent_window_records)
        pressures = list(diagnosis.get("per_agent_overlap_pressure", metrics.get("per_agent_overlap", [])))
        if not pressures or all(float(x) <= 0 for x in pressures):
            return []
        case_counts = diagnosis.get("homogeneous_case_counts", [])
        invalid_rates = diagnosis.get("per_agent_invalid_rate", [])
        tie_eps = float(self.cfg.homogeneity_pressure_tie_eps)
        ids = list(range(len(self.agents)))
        random.shuffle(ids)
        max_pressure = max(float(x) for x in pressures) if pressures else 0.0
        ids.sort(
            key=lambda i: (
                1 if (i < len(invalid_rates) and float(invalid_rates[i]) >= float(self.cfg.invalid_repair_rate_threshold)) else 0,
                round(float(pressures[i]) / max(tie_eps, 1e-6)) if i < len(pressures) else 0,
                int(case_counts[i]) if i < len(case_counts) else 0,
                int(self.agents[i].homogeneity_count),
                float(pressures[i]) if i < len(pressures) else 0.0,
            ),
            reverse=True,
        )
        ids = [
            i for i in ids
            if (
                (i < len(invalid_rates) and float(invalid_rates[i]) >= float(self.cfg.invalid_repair_rate_threshold))
                or (float(pressures[i]) if i < len(pressures) else 0.0) >= max(0.0, max_pressure - tie_eps)
            )
        ]
        if not ids:
            ids = list(range(len(self.agents)))
        active = sum(1 for a in self.agents if a.homogeneity_count > 0)
        return ids[: (2 if active >= 2 else 1)]

    def select_error_agents_for_update(self) -> List[int]:
        if not self.is_update_window_ready():
            return []
        wrong_counts = [0 for _ in range(len(self.agents))]
        team_wrong_counts = [0 for _ in range(len(self.agents))]
        seen_counts = [0 for _ in range(len(self.agents))]
        for rec in self.recent_window_records:
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            individual = list(metrics.get("individual_correct", []))
            team_correct = int(metrics.get("vote_correct", 0) or 0)
            for agent_id in range(len(self.agents)):
                if agent_id >= len(individual):
                    continue
                seen_counts[agent_id] += 1
                if not int(individual[agent_id]):
                    wrong_counts[agent_id] += 1
                    if not team_correct:
                        team_wrong_counts[agent_id] += 1
        ids = list(range(len(self.agents)))
        random.shuffle(ids)
        ids.sort(
            key=lambda i: (
                int(wrong_counts[i]),
                int(team_wrong_counts[i]),
                seen_counts[i] - wrong_counts[i],
            ),
            reverse=True,
        )
        ids = [i for i in ids if wrong_counts[i] > 0]
        return ids[: (2 if len(ids) >= 2 else 1)]

    def select_reward_agents_for_update(self, diagnosis: Dict[str, Any], metrics: Dict[str, Any]) -> List[int]:
        if str(getattr(self.cfg, "target_selector_mode", "legacy")) == "hybrid_competence_boundary":
            selected = self._select_hybrid_reward_agents(diagnosis)
            if self._is_stable_qd_lineage() and bool(self.cfg.target_selector_fairness_enabled):
                rows = diagnosis.get("hybrid_selector_diagnostics", [])
                positive = [row for row in rows if float(row.get("hybrid_target_score", 0.0) or 0.0) > 0.0]
                epoch = int(getattr(self, "competence_phase_epoch", 1))
                minimum = int(self.cfg.min_optimizer_updates_per_agent_per_epoch)
                under_minimum = [
                    row for row in positive
                    if int(self.per_agent_optimizer_update_count.get(f"{epoch}:{int(row['agent_id'])}", 0)) < minimum
                ]
                if not positive:
                    diagnosis["fairness_slot_selected"] = None
                    diagnosis["fairness_slot_skipped_no_evidence"] = True
                elif under_minimum:
                    fairness = min(
                        under_minimum,
                        key=lambda row: (
                            int(self.per_agent_optimizer_update_count.get(f"{epoch}:{int(row['agent_id'])}", 0)),
                            -float(row.get("hybrid_target_score", 0.0) or 0.0), int(row["agent_id"]),
                        ),
                    )
                    fairness_id = int(fairness["agent_id"])
                    selected = (selected[:1] + ([fairness_id] if fairness_id not in selected[:1] else selected[1:2]))[:2]
                    diagnosis["fairness_slot_selected"] = fairness_id
                    diagnosis["fairness_slot_skipped_no_evidence"] = False
                else:
                    selected = [int(row["agent_id"]) for row in sorted(
                        positive, key=lambda row: (-float(row.get("hybrid_target_score", 0.0) or 0.0), int(row["agent_id"])),
                    )[:2]]
                    diagnosis["fairness_slot_selected"] = None
                    diagnosis["fairness_slot_skipped_no_evidence"] = False
            return selected
        if bool(getattr(self.cfg, "boundary_selector_enabled", False)):
            return self._select_boundary_reward_agents(diagnosis)
        error_counts = list(diagnosis.get("per_agent_error_count", []))
        team_wrong_counts = list(diagnosis.get("per_agent_team_wrong_error_count", []))
        invalid_rates = list(diagnosis.get("per_agent_invalid_rate", []))
        pivotal_fix_counts = list(diagnosis.get("per_agent_pivotal_fix_count", []))
        dominant_wrong_counts = list(diagnosis.get("per_agent_dominant_wrong_redundancy_count", []))

        ids = list(range(len(self.agents)))
        random.shuffle(ids)

        def value(rows: List[Any], idx: int, default: float = 0.0) -> float:
            if idx >= len(rows):
                return float(default)
            try:
                return float(rows[idx])
            except Exception:
                return float(default)

        scored = []
        for agent_id in ids:
            base_score = (
                3.0 * value(error_counts, agent_id)
                + 2.0 * value(team_wrong_counts, agent_id)
                + 2.0 * value(invalid_rates, agent_id)
                + 2.0 * value(pivotal_fix_counts, agent_id)
                + 1.0 * value(dominant_wrong_counts, agent_id)
            )
            if base_score > 0.0:
                scored.append((float(base_score), agent_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [agent_id for _, agent_id in scored]
        return selected[: (2 if len(selected) >= 2 else 1)]

    def _select_hybrid_reward_agents(self, diagnosis: Dict[str, Any]) -> List[int]:
        strength = self._clip01(float(getattr(self, "specialization_strength", 0.0)))
        weights = {
            "individual_error_rate": 1.0 - 0.4 * strength,
            "weakness_score": 0.5 - 0.2 * strength,
            "c1_creation_opportunity": 1.2 - 0.4 * strength,
            "c2_creation_opportunity": 1.0,
            "plurality_pivotal_fix_opportunity": 0.25 + strength,
            "dominant_wrong_redundancy": 0.2 + 0.8 * strength,
            "shared_error_residual": 0.8 * strength,
            "capability_gap_affinity": 0.5 * strength,
        }

        def value(name: str, agent_id: int) -> float:
            values = diagnosis.get(name, [])
            return self._clip01(float(values[agent_id])) if agent_id < len(values) else 0.0

        latest = dict(getattr(self, "latest_competence_probe_metrics", {}) or {})
        probe_acc = [float(v) for v in latest.get("per_agent_acc", [])]
        probe_mean = float(latest.get("mean_individual_acc", np.mean(probe_acc) if probe_acc else 0.0) or 0.0)
        pressure = diagnosis.get("per_agent_capability_pressure", [])
        gaps = diagnosis.get("capability_coverage_gap", {})
        diagnostics: List[Dict[str, Any]] = []
        ids = list(range(len(self.agents)))
        scored: List[Tuple[float, int]] = []
        for agent_id in ids:
            family_pressure = pressure[agent_id] if agent_id < len(pressure) and isinstance(pressure[agent_id], dict) else {}
            pressure_total = sum(max(0.0, float(v)) for v in family_pressure.values())
            capability_gap = 0.0
            if pressure_total > 0.0:
                capability_gap = sum(
                    max(0.0, float(family_pressure.get(family, 0.0)))
                    * max(0.0, float(gaps.get(family, 0.0)))
                    for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
                ) / pressure_total
                capability_gap /= max(1.0, float(len(self.agents)))
            components = {
                "individual_error_rate": value("per_agent_general_error_rate", agent_id),
                "weakness_score": self._clip01(max(0.0, probe_mean - probe_acc[agent_id])) if agent_id < len(probe_acc) else 0.0,
                "c1_creation_opportunity": value("per_agent_c1_creation_opportunity", agent_id),
                "c2_creation_opportunity": value("per_agent_c2_creation_opportunity", agent_id),
                "plurality_pivotal_fix_opportunity": value("per_agent_plurality_pivotal_fix_rate", agent_id),
                "dominant_wrong_redundancy": value("per_agent_dominant_wrong_rate", agent_id),
                "shared_error_residual": value("per_agent_shared_error_rate", agent_id),
                "capability_gap_affinity": self._clip01(capability_gap),
            }
            score = sum(weights[name] * components[name] for name in weights)
            diagnostics.append({
                "agent_id": agent_id,
                "applied_specialization_strength": strength,
                **components,
                "hybrid_target_score": float(score),
                "selected": False,
            })
            if score > 0.0:
                scored.append((float(score), agent_id))
        scored.sort(key=lambda row: (-row[0], row[1]))
        selected = [agent_id for _, agent_id in scored[: (2 if len(scored) >= 2 else 1)]]
        for row in diagnostics:
            row["selected"] = int(row["agent_id"]) in selected
        diagnosis["hybrid_selector_weights"] = weights
        diagnosis["hybrid_selector_diagnostics"] = sorted(diagnostics, key=lambda row: int(row["agent_id"]))
        return selected

    def _select_boundary_reward_agents(self, diagnosis: Dict[str, Any]) -> List[int]:
        ids = list(range(len(self.agents)))
        random.shuffle(ids)

        def rate(name: str, agent_id: int) -> float:
            values = diagnosis.get(name, [])
            return float(values[agent_id]) if agent_id < len(values) else 0.0

        pressures = diagnosis.get("per_agent_capability_pressure", [])
        gaps = diagnosis.get("capability_coverage_gap", {})
        scored: List[Tuple[float, int]] = []
        for agent_id in ids:
            plurality_boundary = bool(getattr(self.cfg, "competence_depth_enabled", False))
            pivotal_fix_rate = rate(
                "per_agent_plurality_pivotal_fix_rate"
                if plurality_boundary and diagnosis.get("per_agent_plurality_pivotal_fix_rate")
                else "per_agent_pivotal_fix_rate",
                agent_id,
            )
            pivotal_hold_rate = rate(
                "per_agent_plurality_pivotal_hold_rate"
                if plurality_boundary and diagnosis.get("per_agent_plurality_pivotal_hold_rate")
                else "per_agent_pivotal_hold_rate",
                agent_id,
            )
            boundary_error_rate = rate(
                "per_agent_plurality_boundary_error_rate"
                if plurality_boundary and diagnosis.get("per_agent_plurality_boundary_error_rate")
                else "per_agent_near_boundary_error_rate",
                agent_id,
            )
            base_score = (
                4.0 * pivotal_fix_rate
                + 2.0 * boundary_error_rate
                + 0.5 * pivotal_hold_rate
                + 1.5 * rate("per_agent_dominant_wrong_rate", agent_id)
                + 1.0 * rate("per_agent_shared_error_rate", agent_id)
                + 0.5 * rate("per_agent_general_error_rate", agent_id)
                + 1.0 * rate("per_agent_invalid_rate", agent_id)
            )
            affinity = coverage_bonus = 0.0
            if agent_id < len(pressures) and isinstance(pressures[agent_id], dict):
                family_pressure = {
                    family: max(0.0, float(pressures[agent_id].get(family, 0.0)))
                    for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
                }
                total_pressure = sum(family_pressure.values())
                if total_pressure > 0.0:
                    profile = self.agents[agent_id].capability_profile
                    affinity = sum(
                        float(profile.get(family, 0.0)) * family_pressure[family] / total_pressure
                        for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
                    )
                    coverage_bonus = sum(
                        float(gaps.get(family, 0.0)) * family_pressure[family] / total_pressure
                        for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
                    ) / max(1, (len(self.agents) // 2) + 1)
            score = (
                base_score
                + self._effective_progressive_weight(float(getattr(self.cfg, "capability_affinity_weight", 0.25))) * affinity
                + self._effective_progressive_weight(float(getattr(self.cfg, "capability_coverage_gap_weight", 0.25))) * coverage_bonus
            )
            if bool(getattr(self.cfg, "competence_depth_enabled", False)):
                references = self.previous_epoch_per_agent_acc or list(diagnosis.get("per_agent_accuracy", []))
                reference = float(references[agent_id]) if agent_id < len(references) else 0.0
                deficit = max(0.0, float(getattr(self.cfg, "competence_floor_high", 0.65)) - reference)
                score += (
                    (1.0 - float(self.specialization_strength))
                    * float(getattr(self.cfg, "competence_selector_weight", 1.0))
                    * deficit
                )
            if base_score > 0.0 and score > 0.0:
                scored.append((score, agent_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [agent_id for _, agent_id in scored]
        return selected[: (2 if len(selected) >= 2 else 1)]

    def _window_accuracy_diagnosis(self, window_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        per_agent_seen = [0 for _ in range(len(self.agents))]
        per_agent_correct = [0 for _ in range(len(self.agents))]
        per_agent_team_wrong = [0 for _ in range(len(self.agents))]
        all_error_cases: List[Dict[str, Any]] = []
        target_error_cases: List[Dict[str, Any]] = []
        focus_cases: List[Dict[str, Any]] = []
        for idx, rec in enumerate(window_records):
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            traces = list(rec.get("traces", []))
            answers = list(rec.get("answers", []))
            prompts = list(rec.get("prompts", []))
            individual = list(metrics.get("individual_correct", []))
            invalids = list(metrics.get("invalid_flags", []))
            team_correct = int(metrics.get("vote_correct", 0) or 0)
            vote_answer = str(metrics.get("vote_answer", ""))
            gold = str(rec.get("gold", ""))
            question_hash = str(rec.get("question_hash", ""))
            row_cases = []
            for agent_id in range(len(self.agents)):
                if agent_id >= len(individual):
                    continue
                per_agent_seen[agent_id] += 1
                per_agent_correct[agent_id] += int(individual[agent_id])
                target_invalid = int(invalids[agent_id]) if agent_id < len(invalids) else 0
                if int(individual[agent_id]) and not target_invalid:
                    if bool(getattr(self.cfg, "boundary_selector_enabled", False)) and gold:
                        context = self._behavior_context_for_baseline(
                            agent_id=agent_id,
                            answers=answers,
                            gold=gold,
                            rollout=metrics,
                            question_hash=question_hash,
                        )
                        if context == BehaviorContext.TARGET_CORRECT_PIVOTAL_HOLD.value:
                            target_error_cases.append({
                                "case_id": self._hash(f"{question_hash}|pivotal_correct_protection|{agent_id}|{idx}"),
                                "case_type": "pivotal_correct_protection",
                                "window_index": idx,
                                "question_hash": question_hash,
                                "sample_hash": question_hash,
                                "target_agent_id": agent_id,
                                "target_trace_preview": self._redact_optimizer_text(traces[agent_id] if agent_id < len(traces) else ""),
                                "target_answer_preview": self._answer_behavior_preview(answers[agent_id] if agent_id < len(answers) else ""),
                                "target_invalid": False,
                                "target_correct": True,
                                "team_correct": bool(team_correct),
                                "peer_correct_available": True,
                                "error_pattern": "pivotal_correct_behavior",
                                "repair_hint": "preserve the local mechanism that keeps this answer correct near the vote boundary",
                                "capability_residual_family": CapabilityResidualFamily.UNKNOWN.value,
                                "confidence": 1.0,
                                "vote_context": context,
                            })
                    continue
                if not team_correct:
                    per_agent_team_wrong[agent_id] += 1
                peer_correct_flags = [
                    int(individual[i])
                    for i in range(len(individual))
                    if i != agent_id
                ]
                peer_correct_ids = [
                    int(i)
                    for i in range(len(individual))
                    if i != agent_id and int(individual[i])
                ]
                peer_trace_candidates = [
                    str(traces[i])
                    for i in peer_correct_ids
                    if i < len(traces)
                ]
                if not peer_trace_candidates:
                    peer_trace_indices = [
                        i for i in range(len(traces))
                        if i != agent_id
                    ][:2]
                    peer_trace_candidates = [
                        str(traces[i]) for i in peer_trace_indices
                    ]
                    selected_peer_flags = [
                        int(individual[i]) for i in peer_trace_indices
                        if i < len(individual)
                    ]
                else:
                    selected_peer_flags = [1 for _ in peer_trace_candidates]
                error_info = self._infer_target_error_pattern(
                    target_trace=str(traces[agent_id]) if agent_id < len(traces) else "",
                    target_answer=str(answers[agent_id]) if agent_id < len(answers) else "",
                    peer_traces=peer_trace_candidates,
                    rollout=metrics,
                    agent_id=agent_id,
                )
                context = self._behavior_context_for_baseline(
                    agent_id=agent_id,
                    answers=answers,
                    gold=gold,
                    rollout=metrics,
                    question_hash=question_hash,
                ) if gold and (
                    bool(getattr(self.cfg, "boundary_selector_enabled", False))
                    or self._v7_residual_protocol_enabled()
                ) else BehaviorContext.INVALID.value
                peer_wrong_count = sum(1 for flag in peer_correct_flags if not flag)
                gold_count = int(metrics.get("gold_vote_count", sum(individual)) or 0)
                largest_wrong = int(metrics.get("largest_wrong_vote_count", 0) or 0)
                if bool(getattr(self.cfg, "boundary_selector_enabled", False)):
                    target_answer = str(answers[agent_id] if agent_id < len(answers) else "")
                    wrong_counts = Counter(
                        str(answer or "").strip()
                        for i, answer in enumerate(answers)
                        if i < len(individual) and not int(individual[i]) and str(answer or "").strip()
                    )
                    in_dominant_wrong = bool(
                        target_answer.strip()
                        and wrong_counts.get(target_answer.strip(), 0) == max(wrong_counts.values(), default=0)
                        and max(wrong_counts.values(), default=0) > 1
                    )
                    if target_invalid:
                        target_case_type = "target_invalid"
                    elif context == BehaviorContext.TEAM_WRONG_PIVOTAL_FIX.value:
                        target_case_type = "target_wrong_pivotal_vote_fix"
                    elif abs(gold_count - largest_wrong) <= 1:
                        target_case_type = "target_wrong_near_vote_boundary"
                    elif in_dominant_wrong:
                        target_case_type = "target_wrong_dominant_wrong_cluster"
                    elif peer_wrong_count > 0:
                        target_case_type = "target_wrong_shared_error"
                    elif any(peer_correct_flags) and not team_correct:
                        target_case_type = "target_wrong_peer_correct_nonboundary"
                    else:
                        target_case_type = "target_wrong_vote_already_correct"
                elif not int(individual[agent_id]) and any(peer_correct_flags):
                    target_case_type = "target_agent_wrong_and_peer_correct"
                elif not int(individual[agent_id]) and team_correct:
                    target_case_type = "target_agent_wrong_and_vote_correct"
                elif not int(individual[agent_id]):
                    target_case_type = "target_agent_wrong_and_vote_wrong"
                else:
                    target_case_type = "target_agent_invalid"
                peer_summary = self._peer_behavior_summary(peer_trace_candidates, peer_correct_flags=selected_peer_flags)
                target_words = len(re.findall(r"\w+", str(traces[agent_id]) if agent_id < len(traces) else ""))
                peer_summary["peer_longer_than_target"] = bool(peer_summary.get("mean_peer_trace_words", 0.0) > max(0, target_words))
                case = {
                    "case_id": self._hash(f"{rec.get('question_hash', '')}|accuracy_error|{agent_id}|{idx}"),
                    "case_type": "target_agent_answer_error",
                    "window_index": idx,
                    "sample_hash": rec.get("question_hash", ""),
                    "target_agent_id": agent_id,
                    "target_trace_preview": self._trace_method_preview(traces[agent_id]) if agent_id < len(traces) else "",
                    "target_answer": str(answers[agent_id]) if agent_id < len(answers) else "",
                    "team_vote_answer": vote_answer,
                    "team_correct": bool(team_correct),
                    "target_prompt_preview": normalize_spaces(prompts[agent_id])[:260] if agent_id < len(prompts) else "",
                }
                target_error_cases.append(
                    {
                        "case_id": self._hash(f"{rec.get('question_hash', '')}|target_error_repair|{agent_id}|{idx}"),
                        "case_type": target_case_type,
                        "window_index": idx,
                        "question_hash": rec.get("question_hash", ""),
                        "sample_hash": rec.get("question_hash", ""),
                        "target_agent_id": agent_id,
                        "target_trace_preview": self._redact_optimizer_text(traces[agent_id] if agent_id < len(traces) else ""),
                        "target_answer_preview": self._answer_behavior_preview(answers[agent_id] if agent_id < len(answers) else ""),
                        "peer_trace_preview": peer_summary.get("peer_trace_previews", []),
                        "peer_behavior_summary": peer_summary,
                        "target_invalid": bool(target_invalid),
                        "target_correct": bool(int(individual[agent_id])),
                        "team_correct": bool(team_correct),
                        "peer_correct_available": bool(any(peer_correct_flags)),
                        "error_pattern": str(error_info.get("error_pattern", "unknown_error_pattern")),
                        "repair_hint": str(error_info.get("repair_hint", "")),
                        "capability_residual_family": str(error_info.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                        "confidence": float(error_info.get("confidence", 0.0) or 0.0),
                        "peer_wrong_count": int(peer_wrong_count),
                        "baseline_correct_count": int(sum(int(value) for value in individual)),
                        "vote_context": context,
                        "target_prompt_preview": normalize_spaces(prompts[agent_id])[:260] if agent_id < len(prompts) else "",
                    }
                )
                row_cases.append(case)
                all_error_cases.append(case)
            if row_cases:
                focus_cases.append(
                    {
                        "window_index": idx,
                        "team_correct": bool(team_correct),
                        "wrong_agent_ids": [int(c.get("target_agent_id", -1)) for c in row_cases],
                        "vote_answer": vote_answer,
                    }
                )
        per_agent_accuracy = [
            float(per_agent_correct[i] / per_agent_seen[i]) if per_agent_seen[i] else 0.0
            for i in range(len(self.agents))
        ]
        current_prompts = [
            {"agent_id": i, "prompt_preview": normalize_spaces(p)[:260], "prompt_hash": self._hash(p)}
            for i, p in enumerate(self._active_prompt_list())
        ]
        return {
            "window_size": len(window_records),
            "prompt_roles": current_prompts,
            "focus_cases": focus_cases[:5],
            "error_cases": all_error_cases,
            "target_error_cases": target_error_cases,
            "per_agent_accuracy": per_agent_accuracy,
            "per_agent_error_count": [int(per_agent_seen[i] - per_agent_correct[i]) for i in range(len(self.agents))],
            "per_agent_team_wrong_error_count": per_agent_team_wrong,
            "team_accuracy": float(np.mean([int((rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}).get("vote_correct", 0) or 0) for rec in window_records])) if window_records else 0.0,
        }
