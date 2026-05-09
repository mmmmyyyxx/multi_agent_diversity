import asyncio
import hashlib
import json
import os
import random
import re
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from .config import Config
from .policy import AgentState
from .utils import (
    compute_strategy_family_profile_metrics,
    ensure_dir,
    extract_pred_answer_by_task,
    extract_json_obj,
    infer_task_type,
    majority_vote,
    infer_strategy_family_major,
    normalize_strategy_family_label,
    strategy_family_major_categories,
    strategy_family_to_major_map,
    normalize_spaces,
    parse_gold,
    set_seed,
)


@dataclass
class ReasoningProfile:
    agent_id: int
    trace_hash: str
    primary_family: str
    secondary_family: str
    reasoning_summary: str
    strategy_steps: List[str]
    distinctive_features: List[str]
    evidence_spans: List[str]
    confidence: float
    source: str


def _empty_strategy_family_metrics() -> Dict[str, Any]:
    return {
        "primary_families": [],
        "secondary_families": [],
        "agent_family_distributions": [],
        "primary_family_counts": {},
        "weighted_family_distribution": {},
        "major_family_distribution": {},
        "per_agent_same_family_count": [],
        "per_agent_same_family_ratio": [],
        "per_agent_family_diversity": [],
        "team_family_homogeneity_rate": 0.0,
        "team_family_diversity": 0.0,
        "team_family_entropy": 0.0,
        "team_major_family_diversity": 0.0,
        "team_intra_family_diversity": 0.0,
        "dominant_family_share": 0.0,
        "dominant_major_family_share": 0.0,
        "reasoning_summaries": [],
        "family_judge_metric": "unknown",
    }


class TextualGradientRLSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Unify homogeneity-stat window with update cadence so each update decision
        # uses exactly one update batch worth of recent examples.
        unified_window = max(1, int(self.cfg.update_every))
        self.cfg.homogeneity_window = unified_window
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        ensure_dir(cfg.out_dir)
        set_seed(cfg.seed)

        if str(self.cfg.task_type).lower() == "mmlu":
            self.initial_prompt_bank = [
                "You are a concept-first multiple-choice reasoner. Identify the tested concept, eliminate distractors, then decide.",
                "You are a contradiction-checking verifier. Test each option against the question and remove inconsistent choices.",
                "You are a definition-and-principle solver. Ground reasoning in core definitions before choosing an option.",
                "You are a skeptical critic-solver. Look for subtle wording traps, scope mismatches, and overgeneralization.",
                "You are a decomposition-oriented planner. Break the question into facts, constraints, and implications before selecting.",
                "You are a comparative evaluator. Score options one-by-one and choose the highest-consistency option.",
            ]
        else:
            self.initial_prompt_bank = [
                "You are an equation-first math solver. Define variables explicitly, derive equations, and solve step by step.",
                "You are a backward-checking verifier. Solve the problem, then verify the answer by substitution or reverse reasoning.",
                "You are a commonsense-first quantitative reasoner. Translate the story into quantities carefully before computing.",
                "You are a skeptical critic-solver. Look for hidden assumptions, unit mistakes, and off-by-one errors.",
                "You are a decomposition-oriented planner. Break the problem into sub-steps and compute each sub-result carefully.",
                "You are a pattern-and-invariant solver. Identify repeated structures, totals, and conserved quantities before calculating.",
            ]
        self.initial_agent_prompts = self._build_initial_prompts()
        self.initial_agent_prompt_hashes = [self._prompt_hash(p) for p in self.initial_agent_prompts]
        self.agents = [
            AgentState(
                initial_prompt=self.initial_agent_prompts[i],
                bandit_lr=self.cfg.bandit_lr,
                baseline_momentum=self.cfg.baseline_momentum,
                homogeneity_window=unified_window,
            )
            for i in range(self.cfg.agents)
        ]

        self.strategy_family_labels_path = self._resolve_strategy_family_taxonomy_path()
        self.strategy_family_labels = self._load_strategy_family_labels()
        self.strategy_family_label_set = set(self.strategy_family_labels)
        self.strategy_family_cache: Dict[str, Any] = {}
        self.strategy_family_label_resolution_cache: Dict[str, Dict[str, Any]] = {}

        self.history: List[Dict[str, Any]] = []
        self.update_logs: List[Dict[str, Any]] = []
        self.train_step_logs: List[Dict[str, Any]] = []
        self.train_trace_history_logs: List[Dict[str, Any]] = []
        self.test_trace_history_logs: List[Dict[str, Any]] = []
        self.reasoning_summary_history_logs: List[Dict[str, Any]] = []
        self.recent_window_records: List[Dict[str, Any]] = []
        self.prompt_history = self._init_prompt_history()
        self.write_run_meta()
        self.flush_prompt_history()

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

    def _default_strategy_family_labels(self) -> List[str]:
        return [
            "decomposition",
            "symbolic_formulation",
            "algebraic_derivation",
            "equation_solving",
            "direct_computation",
            "case_analysis",
            "exhaustive_enumeration",
            "constraint_propagation",
            "option_elimination",
            "comparative_reasoning",
            "backward_reasoning",
            "consistency_verification",
            "counterexample_search",
            "proof_by_contradiction",
            "invariant_reasoning",
            "symmetry_reasoning",
            "probabilistic_reasoning",
            "expected_value_reasoning",
            "combinatorial_counting",
            "pattern_generalization",
            "inductive_reasoning",
            "analogy_mapping",
            "causal_reasoning",
            "temporal_sequential_reasoning",
            "spatial_visualization",
            "definition_application",
            "rule_based_classification",
            "theorem_property_application",
            "edge_case_analysis",
            "dimensional_unit_analysis",
            "optimization_extremal_reasoning",
            "approximation_bounding",
            "simulation_tracing",
            "recursive_reasoning",
            "abductive_inference",
            "counterfactual_reasoning",
        ]

    def _resolve_strategy_family_taxonomy_path(self) -> str:
        path = str(getattr(self.cfg, "family_taxonomy_path", "") or "").strip()
        if not path:
            return os.path.join(self.cfg.out_dir, "family_taxonomy.json")
        return path

    def _canonicalize_family_label(self, label: str) -> str:
        raw = normalize_spaces(str(label or "")).lower()
        raw = raw.replace("-", "_").replace(" ", "_")
        raw = re.sub(r"[^a-z0-9_]+", "", raw)
        raw = re.sub(r"_+", "_", raw).strip("_")
        if raw == "other":
            return ""
        aliases = {
            "algebra": "algebraic_derivation",
            "algebraic_reasoning": "algebraic_derivation",
            "symbolic_reasoning": "symbolic_formulation",
            "equation_reasoning": "equation_solving",
            "contradiction": "proof_by_contradiction",
            "contradiction_proof": "proof_by_contradiction",
            "proof_by_contradiction_proof": "proof_by_contradiction",
            "verification": "consistency_verification",
            "verify": "consistency_verification",
            "check": "consistency_verification",
            "backward_verification": "backward_reasoning",
            "backward_checking": "backward_reasoning",
            "computation": "direct_computation",
            "arithmetic": "direct_computation",
            "elimination": "option_elimination",
            "elimination_comparison": "option_elimination",
            "option_comparison": "comparative_reasoning",
            "comparison": "comparative_reasoning",
            "case_split": "case_analysis",
            "enumeration": "exhaustive_enumeration",
            "constraint_reasoning": "constraint_propagation",
            "invariant_symmetry": "invariant_reasoning",
            "invariant": "invariant_reasoning",
            "symmetry": "symmetry_reasoning",
            "probabilistic_estimation": "probabilistic_reasoning",
            "probability_reasoning": "probabilistic_reasoning",
            "estimation": "approximation_bounding",
            "bounding": "approximation_bounding",
            "counting": "combinatorial_counting",
            "combinatorics": "combinatorial_counting",
            "pattern_recognition": "pattern_generalization",
            "analogy": "analogy_mapping",
            "causal": "causal_reasoning",
            "temporal_reasoning": "temporal_sequential_reasoning",
            "spatial_reasoning": "spatial_visualization",
            "definition_lookup": "definition_application",
            "rule_matching": "rule_based_classification",
            "property_application": "theorem_property_application",
            "edge_cases": "edge_case_analysis",
            "unit_analysis": "dimensional_unit_analysis",
            "extremal_reasoning": "optimization_extremal_reasoning",
            "simulation": "simulation_tracing",
            "recursion": "recursive_reasoning",
            "abduction": "abductive_inference",
            "counterfactual": "counterfactual_reasoning",
        }
        raw = aliases.get(raw, raw)
        return raw

    def _load_strategy_family_labels(self) -> List[str]:
        base = self._default_strategy_family_labels()
        path = self.strategy_family_labels_path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return base

            if isinstance(data, dict):
                labels = data.get("labels") or []
            elif isinstance(data, list):
                labels = data
            else:
                labels = []

            merged: List[str] = []
            for label in list(base) + list(labels):
                cleaned = self._canonicalize_family_label(label)
                if not cleaned:
                    continue
                if cleaned not in merged:
                    merged.append(cleaned)
            return merged or base
        return base

    def _persist_strategy_family_labels(self):
        path = self.strategy_family_labels_path
        dir_name = os.path.dirname(path)
        if dir_name:
            ensure_dir(dir_name)
        payload = {
            "labels": list(self.strategy_family_labels),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _strategy_family_definitions(self) -> Dict[str, str]:
        return {
            "decomposition": "breaks a task into sub-goals, components, assumptions, or intermediate claims",
            "symbolic_formulation": "translates the problem into variables, symbols, equations, logical forms, or structured representations",
            "algebraic_derivation": "manipulates symbolic relations to derive a needed expression or conclusion",
            "equation_solving": "sets up and solves equations, systems, or inequalities for unknown quantities",
            "direct_computation": "computes, recalls, or applies a short formula directly with minimal intermediate structure",
            "case_analysis": "splits the solution into mutually relevant cases, branches, or conditions",
            "exhaustive_enumeration": "lists possibilities systematically and evaluates them to find the valid answer",
            "constraint_propagation": "uses constraints to narrow possible states, values, or relationships step by step",
            "option_elimination": "removes impossible or less plausible answer options until a remaining option is selected",
            "comparative_reasoning": "compares alternatives, quantities, hypotheses, or criteria to decide among them",
            "backward_reasoning": "starts from the desired result, answer form, or goal condition and reasons backward",
            "consistency_verification": "checks whether an intermediate or final conclusion satisfies the original conditions",
            "counterexample_search": "searches for disconfirming examples or edge cases",
            "proof_by_contradiction": "assumes the negation or an incompatible claim and derives a contradiction",
            "invariant_reasoning": "uses conserved quantities, stable relationships, monotonicity, or unchanged properties",
            "symmetry_reasoning": "uses interchangeable roles, mirrored structures, or symmetry to simplify the task",
            "probabilistic_reasoning": "reasons with likelihoods, conditional probabilities, uncertainty, or stochastic structure",
            "expected_value_reasoning": "uses averages, expectation, weighted outcomes, or long-run value calculations",
            "combinatorial_counting": "counts arrangements, selections, paths, or possibilities using combinatorial structure",
            "pattern_generalization": "detects a recurring pattern and extends or generalizes it",
            "inductive_reasoning": "infers a general rule from examples, smaller cases, or observed regularities",
            "analogy_mapping": "maps the problem to a similar known structure or transfers a parallel solution pattern",
            "causal_reasoning": "tracks cause-effect relationships, mechanisms, interventions, or explanatory chains",
            "temporal_sequential_reasoning": "reasons over order, time, process steps, or before-after relationships",
            "spatial_visualization": "uses mental diagrams, geometry, layout, orientation, or spatial transformations",
            "definition_application": "applies definitions, terminology, or conceptual criteria directly",
            "rule_based_classification": "matches facts to a rule, category, diagnostic criterion, or decision procedure",
            "theorem_property_application": "uses a known theorem, law, identity, property, or domain principle",
            "edge_case_analysis": "tests boundary conditions, special cases, degeneracies, or limiting scenarios",
            "dimensional_unit_analysis": "uses units, dimensions, scales, or quantity types to constrain the answer",
            "optimization_extremal_reasoning": "seeks maxima, minima, worst cases, best cases, or extremal configurations",
            "approximation_bounding": "uses estimates, bounds, orders of magnitude, or inequalities to locate the answer",
            "simulation_tracing": "steps through a process, algorithm, scenario, or state transition explicitly",
            "recursive_reasoning": "uses recurrence, self-similar structure, dynamic programming, or reduction to smaller instances",
            "abductive_inference": "selects the best explanation for observed facts among plausible hypotheses",
            "counterfactual_reasoning": "changes an assumption or condition to test what would follow",
        }

    def _strategy_family_major_categories(self) -> Dict[str, List[str]]:
        return strategy_family_major_categories()

    def _strategy_family_major_label(self, family_label: str) -> str:
        mapping = strategy_family_to_major_map()
        family = normalize_strategy_family_label(
            family_label,
            allowed_labels=self.strategy_family_labels,
            allow_fallback=True,
        )
        return mapping.get(family, infer_strategy_family_major(family))

    async def _review_new_family_label(
        self,
        raw_label: str,
        agent_trace: str,
        agent_id: Optional[int],
        trace_hash: str,
        reasoning_summary: str,
        judge_reason: str,
        judge_confidence: Optional[float],
        allow_expand: bool,
    ) -> Dict[str, Any]:
        existing_labels = list(self.strategy_family_labels)
        definitions = self._strategy_family_definitions()
        definition_lines = "\n".join([f"- {k}: {v}" for k, v in definitions.items()])
        agent_trace = normalize_spaces(agent_trace)
        reasoning_summary = normalize_spaces(reasoning_summary)
        expand_note = "Expansion is ENABLED." if allow_expand else "Expansion is DISABLED; you MUST map to an existing family."

        system_prompt = (
            "You are a taxonomy gatekeeper for reasoning strategy families.\n"
            "Review exactly one agent trace and decide whether to accept a proposed new family label or map it to an existing family.\n"
            "Use the single agent trace as the authoritative evidence. Do not use other agents or group-level behavior.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            f"Proposed label: {raw_label}\n"
            f"Existing families: {', '.join(existing_labels)}\n"
            f"{expand_note}\n\n"
            "Family definitions (base set):\n"
            f"{definition_lines}\n\n"
            "Context:\n"
            f"- agent_id: {agent_id}\n"
            f"- trace_hash: {trace_hash}\n"
            f"- trace_length: {len(agent_trace)}\n"
            f"- reasoning_summary_from_judge: {reasoning_summary}\n"
            f"- judge_reason: {judge_reason}\n"
            f"- judge_confidence: {judge_confidence}\n\n"
            "Single agent trace:\n"
            f"{agent_trace}\n\n"
            "Rules:\n"
            "- Judge only whether the proposed label describes a reusable reasoning strategy shown in this single trace.\n"
            "- Do not infer from other agents, family distribution, group diagnosis, or answer correctness.\n"
            "- Accept a new family only if it is clearly distinct and not a synonym of existing families.\n"
            "- If accepted, use a concise snake_case label (2-4 words).\n"
            "- If rejected, map to exactly one existing family.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "action": "accept_new" | "map_existing",\n'
            '  "new_family": "...",\n'
            '  "map_to": "...",\n'
            '  "reason": "..."\n'
            "}"
        )
        text = await self._chat(
            model=self.cfg.family_expansion_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=700,
            stage="family_expansion_review",
        )
        obj = extract_json_obj(text) or {}
        obj["_raw_response"] = text
        return obj

    async def _resolve_strategy_family_label(
        self,
        raw_label: str,
        agent_trace: str,
        agent_id: Optional[int],
        trace_hash: str,
        reasoning_summary: str,
        judge_reason: str,
        judge_confidence: Optional[float],
    ) -> Tuple[str, Dict[str, Any]]:
        candidate = self._canonicalize_family_label(raw_label)
        if not candidate:
            fallback = self.strategy_family_labels[0] if self.strategy_family_labels else "decomposition"
            return fallback, {"action": "invalid", "resolved": fallback}

        if candidate in self.strategy_family_label_set:
            return candidate, {"action": "known", "resolved": candidate}

        cached = self.strategy_family_label_resolution_cache.get(candidate)
        if isinstance(cached, dict) and cached.get("resolved"):
            return str(cached["resolved"]), dict(cached)

        allow_expand = bool(getattr(self.cfg, "family_expansion_enabled", True))
        if not allow_expand:
            resolved = normalize_strategy_family_label(
                candidate,
                allowed_labels=list(self.strategy_family_labels),
                allow_fallback=True,
            )
            resolution_info = {
                "action": "local_map_existing",
                "raw_label": raw_label,
                "candidate": candidate,
                "accepted_new": False,
                "resolved": resolved,
                "review_context": "local_taxonomy_fallback",
                "agent_id": agent_id,
                "trace_hash": trace_hash,
                "trace_length": len(normalize_spaces(agent_trace)),
                "used_reasoning_summary": bool(reasoning_summary),
            }
            self.strategy_family_label_resolution_cache[candidate] = dict(resolution_info)
            return resolved, resolution_info
        decision = await self._review_new_family_label(
            raw_label=raw_label,
            agent_trace=agent_trace,
            agent_id=agent_id,
            trace_hash=trace_hash,
            reasoning_summary=reasoning_summary,
            judge_reason=judge_reason,
            judge_confidence=judge_confidence,
            allow_expand=allow_expand,
        )

        action = str(decision.get("action", "")).strip().lower()
        resolved = ""
        resolution_info = {
            "action": action,
            "raw_label": raw_label,
            "candidate": candidate,
            "model": self.cfg.family_expansion_model,
            "reason": str(decision.get("reason", "")),
            "review_context": "single_agent_trace",
            "agent_id": agent_id,
            "trace_hash": trace_hash,
            "trace_length": len(normalize_spaces(agent_trace)),
            "used_reasoning_summary": bool(reasoning_summary),
        }

        existing_labels = list(self.strategy_family_labels)
        if action == "accept_new" and allow_expand:
            new_family = self._canonicalize_family_label(decision.get("new_family") or candidate)
            if new_family and new_family not in self.strategy_family_label_set:
                self.strategy_family_labels.append(new_family)
                self.strategy_family_label_set.add(new_family)
                self._persist_strategy_family_labels()
                resolved = new_family
                resolution_info["accepted_new"] = True
                resolution_info["resolved"] = resolved
            else:
                action = "map_existing"

        if action != "accept_new" or not resolved:
            target = self._canonicalize_family_label(decision.get("map_to") or "")
            resolved = normalize_strategy_family_label(
                target or candidate,
                allowed_labels=existing_labels,
                allow_fallback=True,
            )
            resolution_info["accepted_new"] = False
            resolution_info["resolved"] = resolved

        self.strategy_family_label_resolution_cache[candidate] = dict(resolution_info)
        return resolved, resolution_info

    def _build_initial_prompts(self) -> List[str]:
        mode = str(self.cfg.init_mode).strip().lower()
        if mode not in {"shared", "bank"}:
            mode = "shared"
            self.cfg.init_mode = mode
        if self.cfg.agents <= 0:
            return []
        if mode == "shared":
            return [self.cfg.shared_prompt for _ in range(self.cfg.agents)]
        return [self.initial_prompt_bank[i % len(self.initial_prompt_bank)] for i in range(self.cfg.agents)]

    def write_run_meta(self):
        meta = {
            "init_mode": self.cfg.init_mode,
            "agents": self.cfg.agents,
            "update_every": self.cfg.update_every,
            "homogeneity_window": self.cfg.homogeneity_window,
            "family_judge_metric": "single_trace_judge",
            "trace_homogeneity_metric": "disabled",
            "initial_agent_prompts": self.initial_agent_prompts,
            "initial_agent_prompt_hashes": self.initial_agent_prompt_hashes,
            "all_agents_shared_origin": len(set(self.initial_agent_prompt_hashes)) <= 1,
            "config": asdict(self.cfg),
        }
        path = os.path.join(self.cfg.out_dir, "run_meta.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def is_homogeneity_window_warmup_done(self) -> bool:
        if not self.agents:
            return False
        return all(len(a.recent_homogeneity_flags) >= a.homogeneity_window for a in self.agents)

    # Backward-compat shim for old call sites.
    def is_wrong_window_warmup_done(self) -> bool:
        return self.is_homogeneity_window_warmup_done()

    def clear_homogeneity_windows(self):
        for a in self.agents:
            a.recent_homogeneity_flags.clear()
            a.homogeneity_count = 0
        self.recent_window_records = []

    # Backward-compat shim for old call sites.
    def clear_wrong_windows(self):
        self.clear_homogeneity_windows()

    def _base_log_fields(self) -> Dict[str, Any]:
        return {
            "init_mode": self.cfg.init_mode,
            "all_agents_shared_origin": len(set(self.initial_agent_prompt_hashes)) <= 1,
            "initial_agent_prompt_hashes": self.initial_agent_prompt_hashes,
        }

    def _build_strategy_family_profile(self, agent_id: int, trace: str, family_label: str = "") -> Dict[str, Any]:
        cleaned = normalize_spaces(trace)
        family = normalize_strategy_family_label(
            family_label,
            allowed_labels=self.strategy_family_labels,
            allow_fallback=True,
        )
        return {
            "agent_id": agent_id,
            "trace_hash": self._prompt_hash(cleaned) if cleaned else "",
            "trace_length": len(cleaned),
            "primary_family": family,
        }

    def _heuristic_strategy_family(self, trace: str) -> str:
        cleaned = normalize_spaces(trace)
        head = cleaned[:240].lower()
        if not head:
            return "decomposition"
        if any(token in head for token in ["counterexample", "edge case", "boundary case", "limit case"]):
            return "counterexample_search"
        if any(token in head for token in ["contradiction", "contradict", "assume the opposite", "suppose not"]):
            return "proof_by_contradiction"
        if any(token in head for token in ["work backward", "backward", "from the answer", "goal condition"]):
            return "backward_reasoning"
        if any(token in head for token in ["verify", "check", "validation", "substitute back", "consistent", "consistency"]):
            return "consistency_verification"
        if any(token in head for token in ["case", "split", "branch"]):
            return "case_analysis"
        if any(token in head for token in ["enumerate", "list all", "try all", "exhaustive"]):
            return "exhaustive_enumeration"
        if any(token in head for token in ["constraint", "must be", "cannot be", "narrow down"]):
            return "constraint_propagation"
        if any(token in head for token in ["eliminate", "rule out", "option a", "option b", "option c", "option d"]):
            return "option_elimination"
        if any(token in head for token in ["compare", "larger than", "smaller than", "better than", "contrast"]):
            return "comparative_reasoning"
        if any(token in head for token in ["equation", "solve for", "system of equations", "inequality"]):
            return "equation_solving"
        if any(token in head for token in ["derive", "algebra", "simplify", "expand", "factor"]):
            return "algebraic_derivation"
        if any(token in head for token in ["variable", "symbol", "represent", "model the problem"]):
            return "symbolic_formulation"
        if any(token in head for token in ["expected value", "expectation", "average outcome", "weighted average"]):
            return "expected_value_reasoning"
        if any(token in head for token in ["probability", "likely", "chance", "conditional"]):
            return "probabilistic_reasoning"
        if any(token in head for token in ["count", "arrangement", "combination", "permutation", "ways"]):
            return "combinatorial_counting"
        if any(token in head for token in ["invariant", "conserved", "unchanged", "monotonic"]):
            return "invariant_reasoning"
        if any(token in head for token in ["symmetry", "symmetric", "mirror", "interchangeable"]):
            return "symmetry_reasoning"
        if any(token in head for token in ["pattern", "generalize", "sequence", "regularity"]):
            return "pattern_generalization"
        if any(token in head for token in ["induction", "base case", "inductive"]):
            return "inductive_reasoning"
        if any(token in head for token in ["analogous", "analogy", "similar to"]):
            return "analogy_mapping"
        if any(token in head for token in ["cause", "effect", "because", "mechanism"]):
            return "causal_reasoning"
        if any(token in head for token in ["before", "after", "sequence of steps", "timeline", "then"]):
            return "temporal_sequential_reasoning"
        if any(token in head for token in ["diagram", "visualize", "spatial", "geometry", "rotate"]):
            return "spatial_visualization"
        if any(token in head for token in ["definition", "means that", "by definition", "term"]):
            return "definition_application"
        if any(token in head for token in ["classify", "category", "criterion", "rule says"]):
            return "rule_based_classification"
        if any(token in head for token in ["theorem", "property", "law", "identity"]):
            return "theorem_property_application"
        if any(token in head for token in ["unit", "dimension", "scale"]):
            return "dimensional_unit_analysis"
        if any(token in head for token in ["maximum", "minimum", "best case", "worst case", "extreme"]):
            return "optimization_extremal_reasoning"
        if any(token in head for token in ["approximate", "estimate", "bound", "at least", "at most"]):
            return "approximation_bounding"
        if any(token in head for token in ["simulate", "trace the process", "step through", "state transition"]):
            return "simulation_tracing"
        if any(token in head for token in ["recursive", "recurrence", "smaller instance", "dynamic programming"]):
            return "recursive_reasoning"
        if any(token in head for token in ["best explanation", "hypothesis", "explain the observation"]):
            return "abductive_inference"
        if any(token in head for token in ["what if", "counterfactual", "if instead"]):
            return "counterfactual_reasoning"
        if len(cleaned) < 160:
            return "direct_computation"
        return "decomposition"

    def _trace_invalid_penalty(self, trace: str, answer: str) -> float:
        cleaned = normalize_spaces(trace)
        if not cleaned:
            return 1.0
        penalty = 0.0
        if len(cleaned) < 80:
            penalty += 0.35
        if not re.search(r"FINAL_ANSWER\s*:", cleaned, flags=re.IGNORECASE):
            penalty += 0.25

        tokens = re.findall(r"\w+", cleaned.lower())
        if len(tokens) >= 12:
            bigrams = list(zip(tokens, tokens[1:]))
            if bigrams:
                repeat_ratio = 1.0 - (len(set(bigrams)) / len(bigrams))
                if repeat_ratio > 0.35:
                    penalty += min(0.25, repeat_ratio)
        elif len(tokens) < 6:
            penalty += 0.25

        if str(answer or "").strip() == "":
            penalty += 0.25
        return float(min(1.0, penalty))

    def _strategy_family_cache_key(self, trace: str) -> str:
        normalized_trace = normalize_spaces(trace)
        return self._prompt_hash(normalized_trace)

    def _fallback_reasoning_summary(self, trace: str) -> str:
        cleaned = normalize_spaces(trace)
        if not cleaned:
            return "The trace is empty, so only a heuristic strategy label could be assigned."
        head = self._truncate_words(cleaned, max(80, int(getattr(self.cfg, "min_summary_words", 60))))
        summary = (
            "The trace is summarized heuristically because the judge output was unavailable or incomplete. "
            "It appears to begin by organizing the available information, then follows the visible reasoning "
            f"steps in the trace before converging to a final answer. Trace excerpt: {head}"
        )
        return self._truncate_profile_text(summary)

    def _truncate_words(self, text: Any, max_words: int) -> str:
        words = re.findall(r"\S+", normalize_spaces(str(text or "")))
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words])

    def _truncate_profile_text(self, text: Any, max_tokens: Optional[int] = None) -> str:
        max_tokens = int(max_tokens or getattr(self.cfg, "max_summary_tokens", 512))
        cleaned = normalize_spaces(str(text or ""))
        if not cleaned:
            return ""
        limit = max(1, max_tokens)
        encoder = self._summary_token_encoder()
        if encoder is None:
            return self._truncate_words(cleaned, limit)
        try:
            token_ids = encoder.encode(cleaned)
            if len(token_ids) <= limit:
                return cleaned
            return normalize_spaces(encoder.decode(token_ids[:limit]))
        except Exception:
            return self._truncate_words(cleaned, limit)

    def _word_count(self, text: Any) -> int:
        return len(re.findall(r"\S+", normalize_spaces(str(text or ""))))

    def _summary_token_encoder(self):
        if hasattr(self, "_cached_summary_token_encoder"):
            return getattr(self, "_cached_summary_token_encoder")
        encoder = None
        try:
            import tiktoken

            model_name = str(getattr(self.cfg, "critic_model", "") or getattr(self.cfg, "model", ""))
            try:
                encoder = tiktoken.encoding_for_model(model_name)
            except Exception:
                encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoder = None
        setattr(self, "_cached_summary_token_encoder", encoder)
        return encoder

    def _summary_token_count(self, text: Any) -> int:
        cleaned = normalize_spaces(str(text or ""))
        if not cleaned:
            return 0
        encoder = self._summary_token_encoder()
        if encoder is None:
            return self._word_count(cleaned)
        try:
            return int(len(encoder.encode(cleaned)))
        except Exception:
            return self._word_count(cleaned)

    def _compact_reasoning_summary(self, summary: Any, trace: str) -> str:
        cleaned = normalize_spaces(str(summary or ""))
        if not cleaned:
            return self._fallback_reasoning_summary(trace)
        compact = self._truncate_profile_text(cleaned)
        return compact if compact else self._fallback_reasoning_summary(trace)

    def _normalize_text_list(self, values: Any, max_items: int = 6, max_words: int = 28) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        seen = set()
        for value in values:
            text = self._truncate_words(value, max_words)
            key = text.lower()
            if text and key not in seen:
                out.append(text)
                seen.add(key)
            if len(out) >= max_items:
                break
        return out

    def check_summary_support(
        self,
        reasoning_summary: str,
        evidence_spans: List[str],
        trace: str,
    ) -> Dict[str, Any]:
        summary = normalize_spaces(reasoning_summary)
        cleaned_trace = normalize_spaces(trace)
        min_words = int(getattr(self.cfg, "min_summary_words", 60))
        min_spans = int(getattr(self.cfg, "min_evidence_spans", 1))
        valid_spans = []
        for span in evidence_spans:
            cleaned_span = normalize_spaces(span)
            if cleaned_span and cleaned_span.lower() in cleaned_trace.lower():
                valid_spans.append(cleaned_span)
        process_markers = [
            "first",
            "then",
            "next",
            "finally",
            "compare",
            "eliminate",
            "verify",
            "derive",
            "check",
            "reason",
            "infer",
            "identify",
            "construct",
            "test",
        ]
        word_count = self._word_count(summary)
        issues: List[str] = []
        if word_count < min_words:
            issues.append("summary_too_short")
        if len(evidence_spans) < min_spans:
            issues.append("missing_evidence_spans")
        if evidence_spans and len(valid_spans) < min_spans:
            issues.append("evidence_not_found_in_trace")
        if not any(marker in summary.lower() for marker in process_markers):
            issues.append("weak_reasoning_process_description")
        return {
            "ok": not issues,
            "issues": issues,
            "summary_word_count": word_count,
            "summary_token_count": self._summary_token_count(summary),
            "evidence_span_count": len(evidence_spans),
            "valid_evidence_span_count": len(valid_spans),
        }

    def build_summary_embedding_text(self, profile: Dict[str, Any]) -> str:
        return self._truncate_profile_text(profile.get("reasoning_summary", ""))

    def _reasoning_summary_prompt_requirements(self) -> str:
        max_tokens = int(getattr(self.cfg, "max_summary_tokens", 512))
        min_words = int(getattr(self.cfg, "min_summary_words", 60))
        return (
            "Reasoning summary requirements:\n"
            f"- Write a detailed natural-language paragraph with at least about {min_words} words when the trace contains enough information.\n"
            f"- Maximum length: {max_tokens} tokens.\n"
            "- Preserve the semantic structure of the trace as much as possible.\n"
            "- Describe how the agent understands the problem, what information it prioritizes, how it organizes intermediate reasoning, "
            "whether it compares options, reasons backward, constructs constraints, derives equations, estimates, verifies, or handles uncertainty, "
            "and how it converges to a final answer.\n"
            "- Focus on reasoning trajectory and method, not answer correctness.\n"
            "- Do not include evaluative judgments about quality, such as saying the reasoning is thorough, careful, robust, weak, effective, or well-structured.\n"
            "- Do not mention gold answers, vote results, other agents, or group behavior.\n"
            "- Use continuous natural language suitable for embedding-based comparison; do not make this field a bullet list.\n"
        )

    def _generic_candidate_reason(self, text: str) -> str:
        cleaned = normalize_spaces(text).lower()
        if not cleaned:
            return "empty_prompt"
        generic_patterns = [
            "be diverse",
            "use diverse strategies",
            "use a variety of reasoning",
            "adopt diverse reasoning",
            "think differently",
            "avoid redundancy",
            "be more complementary",
        ]
        has_generic = any(pattern in cleaned for pattern in generic_patterns)
        behavior_markers = [
            "if ",
            "when ",
            "first ",
            "before ",
            "check",
            "test",
            "compare",
            "derive",
            "verify",
            "fallback",
            "fall back",
            "return to",
        ]
        has_behavior = any(marker in cleaned for marker in behavior_markers)
        if has_generic and not has_behavior:
            return "generic_diversity_slogan"
        if not has_behavior:
            return "missing_executable_reasoning_behavior"
        if "fallback" not in cleaned and "fall back" not in cleaned and "return to" not in cleaned:
            return "missing_fallback_strategy"
        return ""

    def validate_candidate_prompt(self, candidate: Dict[str, Any], question: str = "") -> Tuple[bool, str]:
        prompt = normalize_spaces(str(candidate.get("task_agnostic_prompt") or candidate.get("prompt") or ""))
        if not prompt:
            return False, "missing_task_agnostic_prompt"
        question_text = normalize_spaces(question)
        if question_text and len(question_text) > 20 and question_text[:80].lower() in prompt.lower():
            return False, "contains_question_text"
        reason = self._generic_candidate_reason(prompt)
        if reason:
            return False, reason
        if not normalize_spaces(str(candidate.get("fallback_strategy", ""))):
            return False, "missing_fallback_strategy"
        return True, ""

    def _token_cosine_distance(self, a: Any, b: Any) -> float:
        ta = re.findall(r"\w+", normalize_spaces(str(a or "")).lower())
        tb = re.findall(r"\w+", normalize_spaces(str(b or "")).lower())
        if not ta and not tb:
            return 0.0
        ca: Dict[str, int] = {}
        cb: Dict[str, int] = {}
        for t in ta:
            ca[t] = ca.get(t, 0) + 1
        for t in tb:
            cb[t] = cb.get(t, 0) + 1
        keys = set(ca) | set(cb)
        dot = sum(ca.get(k, 0) * cb.get(k, 0) for k in keys)
        na = sum(v * v for v in ca.values()) ** 0.5
        nb = sum(v * v for v in cb.values()) ** 0.5
        if na <= 0.0 or nb <= 0.0:
            return 1.0
        sim = max(0.0, min(1.0, dot / (na * nb)))
        return float(1.0 - sim)

    def _build_single_trace_profile_from_obj(
        self,
        obj: Dict[str, Any],
        agent_id: int,
        trace: str,
        source: str,
    ) -> Dict[str, Any]:
        cleaned_trace = normalize_spaces(trace)
        trace_hash = self._strategy_family_cache_key(trace)
        raw_primary = str(obj.get("primary_family", ""))
        raw_secondary = str(obj.get("secondary_family", raw_primary))
        primary = self._canonicalize_family_label(raw_primary) or self._heuristic_strategy_family(trace)
        secondary = self._canonicalize_family_label(raw_secondary) or primary
        try:
            confidence = float(obj.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = float(max(0.0, min(1.0, confidence)))
        summary = self._compact_reasoning_summary(obj.get("reasoning_summary", ""), trace)
        strategy_steps = self._normalize_text_list(obj.get("strategy_steps", []), max_items=8, max_words=24)
        distinctive_features = self._normalize_text_list(obj.get("distinctive_features", []), max_items=8, max_words=20)
        evidence_spans = self._normalize_text_list(obj.get("evidence_spans", []), max_items=4, max_words=36)
        support = self.check_summary_support(summary, evidence_spans, trace)
        if not evidence_spans and cleaned_trace:
            evidence_spans = [self._truncate_words(cleaned_trace, 36)]
            support = self.check_summary_support(summary, evidence_spans, trace)
        profile = ReasoningProfile(
            agent_id=agent_id,
            trace_hash=trace_hash,
            primary_family=primary,
            secondary_family=secondary,
            reasoning_summary=summary,
            strategy_steps=strategy_steps,
            distinctive_features=distinctive_features,
            evidence_spans=evidence_spans,
            confidence=confidence,
            source=source,
        )
        data = asdict(profile)
        data.update(
            {
                "raw_primary_family": raw_primary,
                "raw_secondary_family": raw_secondary,
                "reason": str(obj.get("reason", "")),
                "summary_support": support,
                "summary_token_count": self._summary_token_count(summary),
                "summary_embedding_text": self.build_summary_embedding_text(data),
            }
        )
        return data

    async def _judge_strategy_family_single(
        self,
        agent_id: int,
        trace: str,
        answer: str = "",
        question: str = "",
    ) -> Dict[str, Any]:
        cleaned_trace = normalize_spaces(trace)
        trace_hash = self._strategy_family_cache_key(trace)
        cached = self.strategy_family_cache.get(trace_hash)
        if isinstance(cached, dict) and cached.get("primary_family") and cached.get("reasoning_summary"):
            return dict(cached)

        labels_for_prompt = list(self.strategy_family_labels)
        definitions = self._strategy_family_definitions()
        definition_lines = "\n".join([f"- {k}: {v}." for k, v in definitions.items()])
        major_lines = "\n".join(
            [f"- {major}: {', '.join(families)}" for major, families in self._strategy_family_major_categories().items()]
        )
        use_dual = bool(getattr(self.cfg, "use_dual_family_labels", True))
        system_prompt = (
            "You judge the reasoning strategy family for exactly one agent trace.\n"
            "Ignore answer correctness. Judge only the reasoning trajectory.\n"
            "Do not use other agents, group behavior, vote results, or gold answers.\n"
            "Choose the most specific existing leaf family that captures the trace.\n"
            "Do not output major/coarse category labels and never output an 'other' family.\n"
            "Return strict JSON only."
        )
        secondary_line = (
            "Return primary_family and secondary_family. If only one clear strategy is present, set secondary_family equal to primary_family.\n"
            if use_dual
            else "Return primary_family only; secondary_family may equal primary_family.\n"
        )
        user_prompt = (
            "Assign reasoning family labels for this single trace only.\n"
            f"{secondary_line}"
            f"{self._reasoning_summary_prompt_requirements()}\n"
            "Also provide strategy_steps, distinctive_features, and evidence_spans copied as short spans from the trace.\n\n"
            f"Existing leaf families: {', '.join(labels_for_prompt)}.\n"
            "You must output only labels from Existing leaf families, or a genuinely new reusable leaf family label.\n"
            "Do NOT output major/coarse category labels such as representation_formalization, algebra_computation, logical_proof, probability_statistics, induction_pattern, process_structure_simulation, or optimization_boundary_meta.\n"
            "Major-family tree:\n"
            f"{major_lines}\n\n"
            "Family definitions (base set):\n"
            f"{definition_lines}\n\n"
            "Allowed input context:\n"
            f"- agent_id: {agent_id}\n"
            f"- task_type: {self.cfg.task_type}\n"
            f"- question_hash: {self._prompt_hash(normalize_spaces(question)) if question else ''}\n"
            f"- trace_hash: {trace_hash}\n"
            f"- trace_length: {len(cleaned_trace)}\n"
            f"- extracted_answer: {normalize_spaces(answer)[:80]}\n\n"
            "Single agent trace:\n"
            f"{cleaned_trace}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "agent_id": 0,\n'
            '  "primary_family": "...",\n'
            '  "secondary_family": "...",\n'
            '  "reasoning_summary": "...",\n'
            '  "strategy_steps": ["...", "..."],\n'
            '  "distinctive_features": ["...", "..."],\n'
            '  "evidence_spans": ["short exact span from trace", "..."],\n'
            '  "confidence": 0.0,\n'
            '  "reason": "..."\n'
            "}"
        )
        text = await self._chat(
            model=self.cfg.critic_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=max(1600, int(self.cfg.critic_max_tokens)),
            stage=f"family_judge_agent_{agent_id}",
        )
        obj = extract_json_obj(text) or {}
        if not isinstance(obj, dict):
            obj = {}
        obj["agent_id"] = agent_id
        profile = self._build_single_trace_profile_from_obj(obj, agent_id, trace, source="single_trace_judge")
        profile["_raw_family_judge_response"] = text
        self.strategy_family_cache[trace_hash] = dict(profile)
        return profile

    async def _rejudge_low_confidence_family(
        self,
        trace: str,
        answer: str,
        question: str,
        original_judgment: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_id = int(original_judgment.get("agent_id", -1))
        cleaned_trace = normalize_spaces(trace)
        definitions = self._strategy_family_definitions()
        definition_lines = "\n".join([f"- {k}: {v}." for k, v in definitions.items()])
        system_prompt = (
            "You are an audit model rejudging one low-confidence reasoning-family label.\n"
            "Use only the single trace and the taxonomy. Ignore answer correctness and do not use group context.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            "The previous family judgment had low confidence. Rejudge the same single trace and return legal leaf family labels.\n"
            f"Existing leaf families: {', '.join(self.strategy_family_labels)}.\n"
            "Prefer existing labels. If none fits and expansion is enabled, you may propose a concise reusable snake_case leaf label.\n\n"
            f"{self._reasoning_summary_prompt_requirements()}\n"
            "Family definitions:\n"
            f"{definition_lines}\n\n"
            "Original judgment:\n"
            f"{json.dumps(original_judgment, ensure_ascii=False, indent=2)}\n\n"
            f"Question hash: {self._prompt_hash(normalize_spaces(question)) if question else ''}\n"
            f"Extracted answer: {normalize_spaces(answer)[:80]}\n"
            "Single agent trace:\n"
            f"{cleaned_trace}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "primary_family": "...",\n'
            '  "secondary_family": "...",\n'
            '  "reasoning_summary": "...",\n'
            '  "strategy_steps": ["...", "..."],\n'
            '  "distinctive_features": ["...", "..."],\n'
            '  "evidence_spans": ["short exact span from trace", "..."],\n'
            '  "confidence": 0.0,\n'
            '  "reason": "..."\n'
            "}"
        )
        text = await self._chat(
            model=self.cfg.family_expansion_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=max(1600, int(self.cfg.critic_max_tokens)),
            stage=f"family_low_confidence_rejudge_agent_{agent_id}",
        )
        obj = extract_json_obj(text) or {}
        if not isinstance(obj, dict):
            obj = {}
        obj["agent_id"] = agent_id
        profile = self._build_single_trace_profile_from_obj(obj, agent_id, trace, source="review_model_rejudge")
        profile["low_confidence_before_rejudge"] = True
        profile["original_confidence"] = float(original_judgment.get("confidence", 0.0) or 0.0)
        profile["rejudged_confidence"] = float(profile.get("confidence", 0.0))
        profile["original_judgment"] = {
            "primary_family": original_judgment.get("primary_family", ""),
            "secondary_family": original_judgment.get("secondary_family", ""),
            "confidence": original_judgment.get("confidence", 0.0),
            "reason": original_judgment.get("reason", ""),
        }
        profile["_raw_rejudge_response"] = text
        self.strategy_family_cache[self._strategy_family_cache_key(trace)] = dict(profile)
        return profile

    async def _finalize_strategy_family_judgment(
        self,
        judgment: Dict[str, Any],
        trace: str,
        answer: str = "",
        question: str = "",
    ) -> Dict[str, Any]:
        agent_id = int(judgment.get("agent_id", -1))
        threshold = float(getattr(self.cfg, "family_confidence_threshold", 0.4))
        support = judgment.get("summary_support", {}) if isinstance(judgment.get("summary_support", {}), dict) else {}
        low_conf = float(judgment.get("confidence", 0.0) or 0.0) < threshold
        weak_summary = not bool(support.get("ok", True))
        if bool(getattr(self.cfg, "family_rejudge_on_low_confidence", True)) and (low_conf or weak_summary):
            try:
                judgment = await self._rejudge_low_confidence_family(trace, answer, question, judgment)
            except Exception as e:
                judgment["rejudge_error"] = normalize_spaces(str(e))[:300]

        raw_family = str(judgment.get("primary_family", ""))
        resolved_family, resolution_info = await self._resolve_strategy_family_label(
            raw_family,
            agent_trace=trace,
            agent_id=agent_id,
            trace_hash=self._strategy_family_cache_key(trace),
            reasoning_summary=str(judgment.get("reasoning_summary", "")),
            judge_reason=str(judgment.get("reason", "")),
            judge_confidence=judgment.get("confidence", None),
        )
        raw_secondary = str(judgment.get("secondary_family", raw_family))
        resolved_secondary = resolved_family
        secondary_resolution_info: Dict[str, Any] = {"action": "same_as_primary", "resolved": resolved_family}
        if bool(getattr(self.cfg, "use_dual_family_labels", True)):
            resolved_secondary, secondary_resolution_info = await self._resolve_strategy_family_label(
                raw_secondary,
                agent_trace=trace,
                agent_id=agent_id,
                trace_hash=self._strategy_family_cache_key(trace),
                reasoning_summary=str(judgment.get("reasoning_summary", "")),
                judge_reason=str(judgment.get("reason", "")),
                judge_confidence=judgment.get("confidence", None),
            )

        judgment["raw_primary_family"] = raw_family
        judgment["raw_secondary_family"] = raw_secondary
        judgment["primary_family"] = resolved_family
        judgment["secondary_family"] = resolved_secondary
        judgment["family_resolution"] = resolution_info
        judgment["secondary_family_resolution"] = secondary_resolution_info
        judgment["family_source"] = judgment.get("source", "single_trace_judge")
        self.strategy_family_cache[self._strategy_family_cache_key(trace)] = dict(judgment)
        return judgment

    async def _judge_strategy_families(
        self,
        traces: List[str],
        answers: Optional[List[str]] = None,
        question: str = "",
    ) -> Tuple[List[str], List[Dict[str, Any]], Optional[float], str]:
        if not traces:
            return [], [], None, ""

        labels: List[str] = [""] * len(traces)
        secondary_labels: List[str] = [""] * len(traces)
        judgments: List[Dict[str, Any]] = [{} for _ in traces]
        answers = answers or ["" for _ in traces]
        tasks = [
            self._judge_strategy_family_single(
                i,
                traces[i],
                answers[i] if i < len(answers) else "",
                question,
            )
            for i in range(len(traces))
        ]
        initial = await asyncio.gather(*tasks, return_exceptions=True)

        finalize_tasks = []
        finalize_indices = []
        for i, item in enumerate(initial):
            if isinstance(item, Exception):
                continue
            if isinstance(item, dict):
                finalize_indices.append(i)
                finalize_tasks.append(
                    self._finalize_strategy_family_judgment(
                        item,
                        traces[i],
                        answers[i] if i < len(answers) else "",
                        question,
                    )
                )
        finalized = await asyncio.gather(*finalize_tasks, return_exceptions=True) if finalize_tasks else []
        for idx, item in zip(finalize_indices, finalized):
            if isinstance(item, Exception) or not isinstance(item, dict):
                continue
            labels[idx] = str(item.get("primary_family", ""))
            secondary_labels[idx] = str(item.get("secondary_family", labels[idx]))
            judgments[idx] = item

        for i in range(len(traces)):
            if not labels[i]:
                fallback_family = self._heuristic_strategy_family(traces[i])
                fallback_secondary_family = fallback_family
                reasoning_summary = self._fallback_reasoning_summary(traces[i])
                cache_value = {
                    "primary_family": fallback_family,
                    "secondary_family": fallback_secondary_family,
                    "reasoning_summary": reasoning_summary,
                    "strategy_steps": [],
                    "distinctive_features": [],
                    "evidence_spans": [self._truncate_words(traces[i], 36)] if traces[i] else [],
                    "summary_support": self.check_summary_support(reasoning_summary, [self._truncate_words(traces[i], 36)] if traces[i] else [], traces[i]),
                    "summary_token_count": self._summary_token_count(reasoning_summary),
                    "confidence": 0.2,
                    "reason": "heuristic fallback",
                    "source": "heuristic",
                    "family_source": "heuristic",
                }
                self.strategy_family_cache[self._strategy_family_cache_key(traces[i])] = cache_value
                labels[i] = fallback_family
                secondary_labels[i] = fallback_secondary_family
                judgments[i] = cache_value

        return labels, judgments, None, ""

    async def compute_rewards_async(self, traces: List[str], answers: List[str], gold: str) -> Dict[str, Any]:
        family_labels, family_judgments, direct_score, direct_reason = await self._judge_strategy_families(traces, answers)
        return self.compute_rewards(
            traces,
            answers,
            gold,
            primary_family_labels=family_labels,
            family_judgments=family_judgments,
            family_group_judgment={
                "llm_direct_diversity_score": direct_score,
                "llm_direct_diversity_reason": direct_reason,
            },
        )

    def _init_prompt_history(self) -> Dict[str, Any]:
        history: Dict[str, Any] = {}
        for i, agent in enumerate(self.agents):
            agent_key = f"agent{i}"
            init_hash = self._prompt_hash(agent.initial_prompt)
            history[agent_key] = {
                "agent_id": i,
                "initial_prompt": agent.initial_prompt,
                "initial_prompt_hash": init_hash,
                "current_prompt": agent.current_prompt,
                "current_prompt_hash": init_hash,
                "events": [
                    {
                        "event": "init",
                        "epoch": 0,
                        "step": 0,
                        "decision": "init",
                        "selected_action_id": None,
                        "selected_action_name": "init",
                        "changed": 0,
                        "current_prompt": agent.current_prompt,
                        "current_prompt_hash": init_hash,
                    }
                ],
            }
        return history

    def _append_prompt_history_event(
        self,
        agent_id: int,
        epoch_id: int,
        step_id: int,
        decision: str,
        selected_action_id: Optional[int],
        selected_action_name: str,
        current_prompt: str,
        current_prompt_hash: str,
        changed: int,
    ):
        agent_key = f"agent{agent_id}"
        if agent_key not in self.prompt_history:
            self.prompt_history[agent_key] = {
                "agent_id": agent_id,
                "initial_prompt": current_prompt,
                "initial_prompt_hash": current_prompt_hash,
                "current_prompt": current_prompt,
                "current_prompt_hash": current_prompt_hash,
                "events": [],
            }

        event = {
            "event": "update",
            "epoch": int(epoch_id),
            "step": int(step_id),
            "decision": str(decision),
            "selected_action_id": selected_action_id,
            "selected_action_name": str(selected_action_name),
            "changed": int(changed),
            "current_prompt": str(current_prompt),
            "current_prompt_hash": str(current_prompt_hash),
        }
        self.prompt_history[agent_key]["events"].append(event)
        self.prompt_history[agent_key]["current_prompt"] = str(current_prompt)
        self.prompt_history[agent_key]["current_prompt_hash"] = str(current_prompt_hash)

    def flush_prompt_history(self):
        path = os.path.join(self.cfg.out_dir, "prompt_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.prompt_history, f, ensure_ascii=False, indent=2)

    def _compact_update_log_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        compact = dict(record)

        question = str(compact.get("question", ""))
        if question:
            compact["question_hash"] = self._prompt_hash(question)
        compact.pop("question", None)

        gold = str(compact.get("gold", ""))
        if gold:
            compact["gold_hash"] = self._prompt_hash(gold)
        compact.pop("gold", None)

        tg = str(compact.get("textual_gradient", ""))
        if tg:
            compact["textual_gradient_hash"] = self._prompt_hash(tg)
            compact["textual_gradient_excerpt"] = normalize_spaces(tg)[:220]
        compact.pop("textual_gradient", None)

        group_diagnosis = compact.get("group_diagnosis")
        if isinstance(group_diagnosis, dict):
            aid = str(compact.get("agent_id", ""))
            hints = group_diagnosis.get("target_role_hints", {})
            target_hint = ""
            if isinstance(hints, dict):
                target_hint = str(hints.get(aid, ""))
            compact["group_diagnosis"] = {
                "group_summary": str(group_diagnosis.get("group_summary", "")),
                "missing_modes": list(group_diagnosis.get("missing_modes", [])),
                "redundant_agents": list(group_diagnosis.get("redundant_agents", [])),
                "critical_agents": list(group_diagnosis.get("critical_agents", [])),
                "target_role_hint": target_hint,
            }

        candidates = compact.get("candidates")
        if isinstance(candidates, list):
            compact["candidates"] = [
                {
                    "name": str(c.get("name", "candidate")),
                    "prompt_hash": self._prompt_hash(str(c.get("prompt", ""))),
                }
                for c in candidates
                if isinstance(c, dict)
            ]

        current_prompt = compact.get("current_prompt")
        if current_prompt is not None:
            compact["current_prompt_hash"] = self._prompt_hash(str(current_prompt))
        compact.pop("current_prompt", None)

        current_prompt_excerpt = compact.get("current_prompt_excerpt")
        if current_prompt_excerpt is not None:
            compact["current_prompt_excerpt"] = normalize_spaces(str(current_prompt_excerpt))[:180]

        selected_prompt = compact.get("selected_prompt")
        if selected_prompt is not None:
            compact["selected_prompt_hash"] = self._prompt_hash(str(selected_prompt))
        compact.pop("selected_prompt", None)

        compact.pop("vote_answer", None)

        return compact

    def _build_train_step_log(self, epoch_id: int, step_id: int, reward_pack: Dict[str, Any], update_summary: Dict[str, Any]) -> Dict[str, Any]:
        family_metrics = reward_pack.get("family_metrics", {}) if isinstance(reward_pack.get("family_metrics", {}), dict) else {}
        invalid_penalties = reward_pack.get("per_agent_invalid_trace_penalty", [])
        return {
            **self._base_log_fields(),
            "epoch": epoch_id,
            "step": step_id,
            "vote_answer": reward_pack["vote_answer"],
            "vote_correct": reward_pack["vote_correct"],
            "llm_direct_diversity_score": reward_pack.get("llm_direct_diversity_score"),
            "primary_family_labels": family_metrics.get("primary_families", []),
            "secondary_family_labels": family_metrics.get("secondary_families", []),
            "reasoning_summaries": family_metrics.get("reasoning_summaries", []),
            "strategy_steps": family_metrics.get("strategy_steps", []),
            "distinctive_features": family_metrics.get("distinctive_features", []),
            "evidence_spans": family_metrics.get("evidence_spans", []),
            "family_confidences": family_metrics.get("family_confidences", []),
            "family_sources": family_metrics.get("family_sources", []),
            "agent_family_distributions": family_metrics.get("agent_family_distributions", []),
            "primary_family_counts": family_metrics.get("primary_family_counts", {}),
            "weighted_family_distribution": family_metrics.get("weighted_family_distribution", {}),
            "major_family_distribution": family_metrics.get("major_family_distribution", {}),
            "team_family_homogeneity_rate": family_metrics.get("team_family_homogeneity_rate", 0.0),
            "team_family_diversity": family_metrics.get("team_family_diversity", 0.0),
            "team_major_family_diversity": family_metrics.get("team_major_family_diversity", 0.0),
            "team_intra_family_diversity": family_metrics.get("team_intra_family_diversity", 0.0),
            "dominant_family_share": family_metrics.get("dominant_family_share", 0.0),
            "dominant_major_family_share": family_metrics.get("dominant_major_family_share", 0.0),
            "family_judge_metric": family_metrics.get("family_judge_metric", "unknown"),
            "all_same_primary": bool(family_metrics.get("all_same_primary", False)),
            "all_same_pair": bool(family_metrics.get("all_same_pair", False)),
            "primary_dominant_share": family_metrics.get("primary_dominant_share", 0.0),
            "pair_dominant_share": family_metrics.get("pair_dominant_share", 0.0),
            "mean_family_confidence": family_metrics.get("mean_family_confidence", 0.0),
            "low_confidence_share": family_metrics.get("low_confidence_share", 0.0),
            "rejudge_count": family_metrics.get("rejudge_count", 0),
            "mean_summary_words": family_metrics.get("mean_summary_words", 0.0),
            "mean_summary_tokens": family_metrics.get("mean_summary_tokens", 0.0),
            "mean_evidence_spans": family_metrics.get("mean_evidence_spans", 0.0),
            "generic_prompt_candidate_rate": update_summary.get("generic_prompt_candidate_rate", 0.0),
            "mean_reward": float(np.mean(reward_pack.get("rewards", []))) if reward_pack.get("rewards") else 0.0,
            "mean_invalid_trace_penalty": float(np.mean(invalid_penalties)) if invalid_penalties else 0.0,
            "invalid_trace_penalty": invalid_penalties,
            "update": {
                "update_requested": bool(update_summary.get("update_requested", False)),
                "update_ready": bool(update_summary.get("update_ready", False)),
                "group_diagnosis_ok": update_summary.get("group_diagnosis_ok"),
                "selected_agent_ids": list(update_summary.get("selected_agent_ids", [])),
                "updated_agent_ids": list(update_summary.get("updated_agent_ids", [])),
                "skipped_reason": update_summary.get("skipped_reason", ""),
            },
            "candidate_behavior_diagnostics": update_summary.get("candidate_behavior_diagnostics", {}),
        }

    def _build_reasoning_summary_history_record(
        self,
        split: str,
        epoch_id: int,
        step_id: int,
        question: str,
        family_metrics: Dict[str, Any],
        traces: List[str],
    ) -> Dict[str, Any]:
        primary_families = list(family_metrics.get("primary_families", []))
        secondary_families = list(family_metrics.get("secondary_families", primary_families))
        family_distributions = list(family_metrics.get("agent_family_distributions", []))
        reasoning_summaries = list(family_metrics.get("reasoning_summaries", []))
        strategy_steps_list = list(family_metrics.get("strategy_steps", []))
        distinctive_features_list = list(family_metrics.get("distinctive_features", []))
        evidence_spans_list = list(family_metrics.get("evidence_spans", []))
        confidence_list = list(family_metrics.get("family_confidences", []))
        source_list = list(family_metrics.get("family_sources", []))
        family_judgments = family_metrics.get("family_judgments", [])
        if not isinstance(family_judgments, list):
            family_judgments = []

        agents: List[Dict[str, Any]] = []
        for i, trace in enumerate(traces):
            cleaned_trace = normalize_spaces(trace)
            judgment = family_judgments[i] if i < len(family_judgments) and isinstance(family_judgments[i], dict) else {}
            primary_family = primary_families[i] if i < len(primary_families) else str(judgment.get("primary_family", ""))
            secondary_family = secondary_families[i] if i < len(secondary_families) else str(judgment.get("secondary_family", primary_family))
            reasoning_summary = reasoning_summaries[i] if i < len(reasoning_summaries) else str(judgment.get("reasoning_summary", ""))
            if not reasoning_summary:
                reasoning_summary = self._fallback_reasoning_summary(trace)
            strategy_steps = strategy_steps_list[i] if i < len(strategy_steps_list) and isinstance(strategy_steps_list[i], list) else self._normalize_text_list(judgment.get("strategy_steps", []), max_items=8, max_words=24)
            distinctive_features = distinctive_features_list[i] if i < len(distinctive_features_list) and isinstance(distinctive_features_list[i], list) else self._normalize_text_list(judgment.get("distinctive_features", []), max_items=8, max_words=20)
            evidence_spans = evidence_spans_list[i] if i < len(evidence_spans_list) and isinstance(evidence_spans_list[i], list) else self._normalize_text_list(judgment.get("evidence_spans", []), max_items=4, max_words=36)
            try:
                confidence = float(confidence_list[i]) if i < len(confidence_list) else float(judgment.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            source = str(source_list[i]) if i < len(source_list) else str(judgment.get("family_source", judgment.get("source", "")))
            agents.append(
                {
                    "agent_id": i,
                    "primary_family": primary_family,
                    "secondary_family": secondary_family or primary_family,
                    "family_resolution": judgment.get("family_resolution", {}) if isinstance(judgment, dict) else {},
                    "secondary_family_resolution": judgment.get("secondary_family_resolution", {}) if isinstance(judgment, dict) else {},
                    "family_distribution": family_distributions[i] if i < len(family_distributions) else {},
                    "reasoning_summary": reasoning_summary,
                    "summary_token_count": self._summary_token_count(reasoning_summary),
                    "strategy_steps": strategy_steps,
                    "distinctive_features": distinctive_features,
                    "evidence_spans": evidence_spans,
                    "confidence": confidence,
                    "source": source,
                    "low_confidence_before_rejudge": bool(judgment.get("low_confidence_before_rejudge", False)) if isinstance(judgment, dict) else False,
                    "original_confidence": judgment.get("original_confidence", None) if isinstance(judgment, dict) else None,
                    "rejudged_confidence": judgment.get("rejudged_confidence", None) if isinstance(judgment, dict) else None,
                    "family_source": judgment.get("family_source", source) if isinstance(judgment, dict) else source,
                    "summary_embedding_text": self.build_summary_embedding_text(
                        {
                            "reasoning_summary": reasoning_summary,
                            "strategy_steps": strategy_steps,
                            "distinctive_features": distinctive_features,
                        }
                    ),
                    "summary_support": judgment.get("summary_support", {}) if isinstance(judgment, dict) else self.check_summary_support(reasoning_summary, evidence_spans, trace),
                    "trace_hash": self._prompt_hash(cleaned_trace) if cleaned_trace else "",
                    "trace_length": len(cleaned_trace),
                }
            )

        compact_question = normalize_spaces(question)
        return {
            "split": str(split),
            "epoch": int(epoch_id),
            "step": int(step_id),
            "question_hash": self._prompt_hash(compact_question) if compact_question else "",
            "question_excerpt": compact_question[:300],
            "family_judge_metric": family_metrics.get("family_judge_metric", "unknown"),
            "primary_family_counts": family_metrics.get("primary_family_counts", {}),
            "weighted_family_distribution": family_metrics.get("weighted_family_distribution", {}),
            "major_family_distribution": family_metrics.get("major_family_distribution", {}),
            "team_family_homogeneity_rate": family_metrics.get("team_family_homogeneity_rate", 0.0),
            "team_family_diversity": family_metrics.get("team_family_diversity", 0.0),
            "team_major_family_diversity": family_metrics.get("team_major_family_diversity", 0.0),
            "team_intra_family_diversity": family_metrics.get("team_intra_family_diversity", 0.0),
            "agents": agents,
        }

    def _is_transient_llm_error(self, err: Exception) -> bool:
        if isinstance(err, TimeoutError):
            return True
        msg = str(err).lower()
        transient_markers = [
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "rate limit",
            "too many requests",
            "connection reset",
            "connection aborted",
            "connection error",
            "connect error",
            "network error",
            "remoteprotocolerror",
            "read timeout",
            "write timeout",
            "service unavailable",
            "apiconnectionerror",
        ]
        return any(m in msg for m in transient_markers)

    def _contains_task_specific_content(self, prompt: str, question: Optional[str] = None) -> bool:
        p = normalize_spaces(str(prompt)).lower()
        if not p:
            return False

        hard_markers = [
            "question:",
            "select the best option from the following",
            "output final_answer",
            "final_answer:",
            "options:",
        ]
        if any(m in p for m in hard_markers):
            return True

        # Detect copied multiple-choice option blocks.
        if re.search(r"\ba\.\s+.*\bb\.\s+.*\bc\.\s+", p, flags=re.DOTALL):
            return True

        if question:
            q = normalize_spaces(str(question)).lower()
            if len(q) >= 40 and q[:40] in p:
                return True
        return False

    def _fallback_general_prompt(self, agent_id: int) -> str:
        base = self.agents[agent_id].initial_prompt
        return normalize_spaces(
            base
            + " Keep the prompt task-agnostic: never include concrete question text, options, or answer-format templates."
        )

    def _sanitize_prompt(self, prompt: str, agent_id: int, question: Optional[str] = None) -> Tuple[str, bool]:
        if self._contains_task_specific_content(prompt, question=question):
            return self._fallback_general_prompt(agent_id), True
        return prompt, False

    async def _chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str = "unknown",
    ) -> str:
        last_err = None
        attempt = 0
        transient_failures = 0
        call_id = self._prompt_hash(f"{stage}|{model}|{time.time()}|{random.random()}")
        prompt_chars = len(str(system_prompt)) + len(str(user_prompt))
        timeout_sec = float(getattr(self.cfg, "llm_call_timeout", 120.0) or 0.0)
        log_enabled = bool(getattr(self.cfg, "llm_call_logging", True))
        while True:
            started = time.time()
            try:
                if log_enabled:
                    print(
                        f"[LLM][start] id={call_id} stage={stage} model={model} "
                        f"attempt={attempt + 1} prompt_chars={prompt_chars} max_tokens={max_tokens} timeout={timeout_sec}",
                        flush=True,
                    )
                request = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if timeout_sec > 0:
                    resp = await asyncio.wait_for(request, timeout=timeout_sec)
                else:
                    resp = await request
                content = resp.choices[0].message.content or ""
                if log_enabled:
                    elapsed = time.time() - started
                    print(
                        f"[LLM][ok] id={call_id} stage={stage} model={model} "
                        f"attempt={attempt + 1} elapsed={elapsed:.2f}s response_chars={len(content)}",
                        flush=True,
                    )
                return content
            except Exception as e:
                last_err = e
                is_transient = self._is_transient_llm_error(e)
                elapsed = time.time() - started
                if is_transient:
                    transient_failures += 1
                    if (not self.cfg.transient_retry_forever) and self.cfg.max_transient_retries > 0:
                        if transient_failures >= self.cfg.max_transient_retries:
                            break
                else:
                    if attempt >= max(1, int(self.cfg.max_retries)):
                        break

                backoff = self.cfg.retry_sleep * (2 ** attempt)
                backoff = min(backoff, float(self.cfg.max_retry_backoff))
                jitter = 1.0 + random.uniform(0.0, 0.3)
                sleep_sec = backoff * jitter
                if log_enabled:
                    print(
                        f"[LLM][retry] id={call_id} stage={stage} model={model} "
                        f"attempt={attempt + 1} elapsed={elapsed:.2f}s transient={int(is_transient)} "
                        f"sleep={sleep_sec:.2f}s error={normalize_spaces(str(e))[:300]}",
                        flush=True,
                    )
                await asyncio.sleep(sleep_sec)
                attempt += 1
        if log_enabled:
            print(
                f"[LLM][failed] id={call_id} stage={stage} model={model} "
                f"attempts={attempt + 1} error={normalize_spaces(str(last_err))[:500]}",
                flush=True,
            )
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    async def solve_once(self, question: str, agent_id: int, prompt_text: str) -> Tuple[str, str]:
        effective_task = infer_task_type(task_type=self.cfg.task_type, question=question, answer=None)
        if effective_task == "mmlu":
            system_prompt = (
                "You are solving an MMLU-style multiple-choice question.\n"
                "Reason briefly and carefully using the options provided.\n"
                "At the end, output exactly one line in the format: FINAL_ANSWER: <A/B/C/D>\n"
                "Do not output multiple final answers.\n\n"
                f"Agent specialization:\n{prompt_text}"
            )
        else:
            system_prompt = (
                "You are solving a GSM8K-style arithmetic word problem.\n"
                "Always reason step by step.\n"
                "At the end, output exactly one line in the format: FINAL_ANSWER: <number>\n"
                "Do not output multiple final answers.\n\n"
                f"Agent specialization:\n{prompt_text}"
            )
        user_prompt = f"Question:\n{question}\n\nSolve carefully."
        text = await self._chat(
            model=self.cfg.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            stage=f"solver_agent_{agent_id}",
        )
        answer = extract_pred_answer_by_task(text, task_type=self.cfg.task_type, question=question)
        return text, answer

    async def solve_with_current_prompts(self, question: str) -> Tuple[List[str], List[str]]:
        tasks = []
        for i, agent in enumerate(self.agents):
            tasks.append(self.solve_once(question, i, agent.current_prompt))
        outs = await asyncio.gather(*tasks)
        traces = [x[0] for x in outs]
        answers = [x[1] for x in outs]
        return traces, answers

    async def solve_with_current_prompts_with_family(
        self,
        question: str,
    ) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
        traces, answers = await self.solve_with_current_prompts(question)
        family_labels, family_judgments, _, _ = await self._judge_strategy_families(traces, answers, question)
        return traces, answers, family_labels, family_judgments

    async def solve_with_agent_prompt_override(
        self,
        question: str,
        agent_id: int,
        prompt: str,
    ) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
        tasks = []
        for i, agent in enumerate(self.agents):
            agent_prompt = prompt if i == agent_id else agent.current_prompt
            tasks.append(self.solve_once(question, i, agent_prompt))
        outs = await asyncio.gather(*tasks)
        traces = [x[0] for x in outs]
        answers = [x[1] for x in outs]
        family_labels, family_judgments, _, _ = await self._judge_strategy_families(traces, answers, question)
        return traces, answers, family_labels, family_judgments

    def _family_diversity_reward_from_metrics(self, family_metrics: Dict[str, Any]) -> float:
        diversity = float(family_metrics.get("team_family_diversity", 0.0))
        entropy = float(family_metrics.get("team_family_entropy", 0.0))
        dominant = float(family_metrics.get("dominant_family_share", 0.0))
        return float(0.55 * diversity + 0.25 * (1.0 - dominant) + 0.20 * min(1.0, entropy))

    def _dominant_label(self, counts: Dict[str, Any]) -> str:
        if not isinstance(counts, dict) or not counts:
            return self.strategy_family_labels[0] if self.strategy_family_labels else "decomposition"
        cleaned = []
        for label, count in counts.items():
            try:
                cleaned.append((str(label), int(count)))
            except Exception:
                continue
        if not cleaned:
            return self.strategy_family_labels[0] if self.strategy_family_labels else "decomposition"
        cleaned.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return cleaned[0][0]

    def compute_rewards(
        self,
        traces: List[str],
        answers: List[str],
        gold: str,
        primary_family_labels: Optional[List[str]] = None,
        family_judgments: Optional[List[Dict[str, Any]]] = None,
        family_group_judgment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        vote_answer = majority_vote(answers)
        vote_correct = int(vote_answer == gold)
        individual_correct = [int(a == gold) for a in answers]

        if primary_family_labels is not None:
            family_labels = list(primary_family_labels)
        else:
            family_labels = [self._heuristic_strategy_family(trace) for trace in traces]

        secondary_family_labels = list(family_labels)
        if bool(getattr(self.cfg, "use_dual_family_labels", True)) and isinstance(family_judgments, list):
            for i, judgment in enumerate(family_judgments):
                if i >= len(secondary_family_labels):
                    break
                if isinstance(judgment, dict):
                    secondary_family_labels[i] = str(judgment.get("secondary_family", secondary_family_labels[i]))

        family_profiles = [
            self._build_strategy_family_profile(i, trace, family_labels[i] if i < len(family_labels) else self.strategy_family_labels[0])
            for i, trace in enumerate(traces)
        ]
        family_metrics = compute_strategy_family_profile_metrics(
            family_labels,
            secondary_family_labels,
            allowed_labels=self.strategy_family_labels,
            use_dual_family=bool(getattr(self.cfg, "use_dual_family_labels", True)),
            primary_weight=float(getattr(self.cfg, "primary_family_weight", 0.7)),
            secondary_weight=float(getattr(self.cfg, "secondary_family_weight", 0.3)),
            same_major_weight=float(getattr(self.cfg, "same_major_family_weight", 0.5)),
            macro_diversity_weight=float(getattr(self.cfg, "macro_diversity_weight", 0.5)),
            allow_fallback=True,
        )
        reasoning_summaries: List[str] = []
        strategy_steps_list: List[List[str]] = []
        distinctive_features_list: List[List[str]] = []
        evidence_spans_list: List[List[str]] = []
        confidence_list: List[float] = []
        source_list: List[str] = []
        rejudge_count = 0
        if isinstance(family_judgments, list):
            for i, judgment in enumerate(family_judgments):
                if i >= len(traces):
                    break
                if isinstance(judgment, dict):
                    reasoning_summaries.append(self._compact_reasoning_summary(judgment.get("reasoning_summary", ""), traces[i]))
                    strategy_steps_list.append(self._normalize_text_list(judgment.get("strategy_steps", []), max_items=8, max_words=24))
                    distinctive_features_list.append(self._normalize_text_list(judgment.get("distinctive_features", []), max_items=8, max_words=20))
                    evidence_spans_list.append(self._normalize_text_list(judgment.get("evidence_spans", []), max_items=4, max_words=36))
                    try:
                        confidence_list.append(float(judgment.get("confidence", 0.0) or 0.0))
                    except Exception:
                        confidence_list.append(0.0)
                    source = str(judgment.get("family_source", judgment.get("source", "")))
                    source_list.append(source)
                    if source == "review_model_rejudge" or judgment.get("low_confidence_before_rejudge"):
                        rejudge_count += 1
                else:
                    reasoning_summaries.append(self._fallback_reasoning_summary(traces[i]))
                    strategy_steps_list.append([])
                    distinctive_features_list.append([])
                    evidence_spans_list.append([])
                    confidence_list.append(0.0)
                    source_list.append("heuristic")
        while len(reasoning_summaries) < len(traces):
            reasoning_summaries.append(self._fallback_reasoning_summary(traces[len(reasoning_summaries)]))
        while len(strategy_steps_list) < len(traces):
            strategy_steps_list.append([])
        while len(distinctive_features_list) < len(traces):
            distinctive_features_list.append([])
        while len(evidence_spans_list) < len(traces):
            evidence_spans_list.append([])
        while len(confidence_list) < len(traces):
            confidence_list.append(0.0)
        while len(source_list) < len(traces):
            source_list.append("unknown")
        family_metrics["reasoning_summaries"] = reasoning_summaries
        family_metrics["strategy_steps"] = strategy_steps_list
        family_metrics["distinctive_features"] = distinctive_features_list
        family_metrics["evidence_spans"] = evidence_spans_list
        family_metrics["family_confidences"] = confidence_list
        family_metrics["family_sources"] = source_list
        family_metrics["mean_family_confidence"] = float(np.mean(confidence_list)) if confidence_list else 0.0
        threshold = float(getattr(self.cfg, "family_confidence_threshold", 0.4))
        family_metrics["low_confidence_share"] = float(np.mean([1.0 if c < threshold else 0.0 for c in confidence_list])) if confidence_list else 0.0
        family_metrics["rejudge_count"] = int(rejudge_count)
        family_metrics["mean_summary_words"] = float(np.mean([self._word_count(s) for s in reasoning_summaries])) if reasoning_summaries else 0.0
        family_metrics["mean_summary_tokens"] = float(np.mean([self._summary_token_count(s) for s in reasoning_summaries])) if reasoning_summaries else 0.0
        family_metrics["mean_evidence_spans"] = float(np.mean([len(x) for x in evidence_spans_list])) if evidence_spans_list else 0.0
        normalized_pairs = list(zip(family_metrics.get("primary_families", []), family_metrics.get("secondary_families", [])))
        primary_counts = family_metrics.get("primary_family_counts", {})
        pair_counts: Dict[str, int] = {}
        for primary, secondary in normalized_pairs:
            key = f"{primary}|{secondary}"
            pair_counts[key] = pair_counts.get(key, 0) + 1
        family_metrics["all_same_primary"] = bool(len(set(family_metrics.get("primary_families", []))) == 1 and len(traces) > 0)
        family_metrics["all_same_pair"] = bool(len(set(normalized_pairs)) == 1 and len(traces) > 0)
        family_metrics["primary_dominant_share"] = float(max(primary_counts.values()) / len(traces)) if isinstance(primary_counts, dict) and primary_counts and traces else 0.0
        family_metrics["pair_dominant_share"] = float(max(pair_counts.values()) / len(traces)) if pair_counts and traces else 0.0
        family_metrics["family_profile_summaries"] = [
            {
                "agent_id": p["agent_id"],
                "primary_family": family_metrics.get("primary_families", [])[p["agent_id"]] if p["agent_id"] < len(family_metrics.get("primary_families", [])) else p["primary_family"],
                "secondary_family": family_metrics.get("secondary_families", [])[p["agent_id"]] if p["agent_id"] < len(family_metrics.get("secondary_families", [])) else p["primary_family"],
                "family_distribution": family_metrics.get("agent_family_distributions", [])[p["agent_id"]] if p["agent_id"] < len(family_metrics.get("agent_family_distributions", [])) else {p["primary_family"]: 1.0},
                "reasoning_summary": reasoning_summaries[p["agent_id"]] if p["agent_id"] < len(reasoning_summaries) else "",
                "strategy_steps": strategy_steps_list[p["agent_id"]] if p["agent_id"] < len(strategy_steps_list) else [],
                "distinctive_features": distinctive_features_list[p["agent_id"]] if p["agent_id"] < len(distinctive_features_list) else [],
                "evidence_spans": evidence_spans_list[p["agent_id"]] if p["agent_id"] < len(evidence_spans_list) else [],
                "confidence": confidence_list[p["agent_id"]] if p["agent_id"] < len(confidence_list) else 0.0,
                "source": source_list[p["agent_id"]] if p["agent_id"] < len(source_list) else "unknown",
                "trace_hash": p["trace_hash"],
                "trace_length": p["trace_length"],
            }
            for p in family_profiles
        ]
        if family_judgments is None:
            family_metrics["family_judge_metric"] = "heuristic_fallback"
        elif any(isinstance(j, dict) and j.get("source") == "heuristic" for j in family_judgments):
            family_metrics["family_judge_metric"] = "single_trace_judge_with_fallback"
        else:
            family_metrics["family_judge_metric"] = "single_trace_judge"
        if family_judgments is not None:
            family_metrics["family_judgments"] = family_judgments
        if isinstance(family_group_judgment, dict):
            family_metrics["llm_direct_diversity_score"] = family_group_judgment.get("llm_direct_diversity_score")
            family_metrics["llm_direct_diversity_reason"] = family_group_judgment.get("llm_direct_diversity_reason", "")
        family_diversity = float(family_metrics.get("team_family_diversity", 0.0))
        per_agent_family_diversity = list(
            family_metrics.get("per_agent_family_diversity", [family_diversity for _ in traces])
        )
        per_agent_same_family_ratio = list(
            family_metrics.get("per_agent_same_family_ratio", [0.0 for _ in traces])
        )
        per_agent_invalid_trace_penalty = [
            self._trace_invalid_penalty(trace, answers[i] if i < len(answers) else "")
            for i, trace in enumerate(traces)
        ]

        rewards = []
        for i in range(len(answers)):
            same_ratio = per_agent_same_family_ratio[i] if i < len(per_agent_same_family_ratio) else 0.0
            invalid_penalty = per_agent_invalid_trace_penalty[i] if i < len(per_agent_invalid_trace_penalty) else 0.0
            family_div_term = per_agent_family_diversity[i] if i < len(per_agent_family_diversity) else family_diversity
            effective_div = float(0.75 * family_diversity + 0.25 * family_div_term)
            r = (
                self.cfg.lambda_diversity * effective_div
                - self.cfg.lambda_homogeneity * same_ratio
                - self.cfg.lambda_invalid_trace * invalid_penalty
            )
            rewards.append(float(r))

        return {
            "vote_answer": vote_answer,
            "vote_correct": vote_correct,
            "individual_correct": individual_correct,
            "llm_direct_diversity_score": float(family_group_judgment.get("llm_direct_diversity_score")) if isinstance(family_group_judgment, dict) and family_group_judgment.get("llm_direct_diversity_score") is not None else None,
            "llm_direct_diversity_reason": str(family_group_judgment.get("llm_direct_diversity_reason", "")) if isinstance(family_group_judgment, dict) else "",
            "team_family_homogeneity_rate": float(family_metrics.get("team_family_homogeneity_rate", 0.0)),
            "team_family_diversity": float(family_metrics.get("team_family_diversity", 0.0)),
            "per_agent_family_diversity": per_agent_family_diversity,
            "per_agent_invalid_trace_penalty": per_agent_invalid_trace_penalty,
            "per_agent_same_family_count": family_metrics["per_agent_same_family_count"],
            "per_agent_same_family_ratio": family_metrics["per_agent_same_family_ratio"],
            "family_metrics": family_metrics,
            "rewards": rewards,
        }

    def build_group_context(
        self,
        question: str,
        traces: List[str],
        reward_pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        role_positions = []
        family_metrics = reward_pack.get("family_metrics", {}) if isinstance(reward_pack.get("family_metrics", {}), dict) else {}
        families = list(family_metrics.get("primary_families", []))
        secondary_families = list(family_metrics.get("secondary_families", families))
        agent_family_distributions = list(family_metrics.get("agent_family_distributions", []))
        reasoning_summaries = list(family_metrics.get("reasoning_summaries", []))
        fallback_family = self.strategy_family_labels[0] if self.strategy_family_labels else "decomposition"
        num_agents = max(
            len(self.agents),
            len(traces),
            len(families),
            len(secondary_families),
            len(agent_family_distributions),
            len(reasoning_summaries),
        )
        for i in range(num_agents):
            pos = {
                "agent_id": i,
                "primary_family": families[i] if i < len(families) else fallback_family,
                "secondary_family": secondary_families[i] if i < len(secondary_families) else (families[i] if i < len(families) else fallback_family),
                "family_distribution": agent_family_distributions[i] if i < len(agent_family_distributions) else {},
                "reasoning_summary": reasoning_summaries[i] if i < len(reasoning_summaries) else self._fallback_reasoning_summary(traces[i] if i < len(traces) else ""),
                "same_family_ratio": float(reward_pack["per_agent_same_family_ratio"][i]) if i < len(reward_pack.get("per_agent_same_family_ratio", [])) else 0.0,
                "same_family_count": int(reward_pack["per_agent_same_family_count"][i]) if i < len(reward_pack.get("per_agent_same_family_count", [])) else 0,
            }
            role_positions.append(pos)

        critical_agents = [
            i
            for i, c in enumerate(reward_pack.get("per_agent_same_family_count", []))
            if int(c) == 0
        ]
        high_homogeneity_agents = [
            i
            for i, c in enumerate(reward_pack.get("per_agent_same_family_count", []))
            if int(c) > 0
        ]
        return {
            "question_hash": self._prompt_hash(normalize_spaces(question)) if question else "",
            "llm_direct_diversity_score": reward_pack.get("llm_direct_diversity_score"),
            "llm_direct_diversity_reason": reward_pack.get("llm_direct_diversity_reason", ""),
            "primary_family_counts": family_metrics.get("primary_family_counts", {}),
            "weighted_family_distribution": family_metrics.get("weighted_family_distribution", {}),
            "major_family_distribution": family_metrics.get("major_family_distribution", {}),
            "team_family_homogeneity_rate": family_metrics.get("team_family_homogeneity_rate", 0.0),
            "team_family_diversity": family_metrics.get("team_family_diversity", 0.0),
            "team_major_family_diversity": family_metrics.get("team_major_family_diversity", 0.0),
            "team_intra_family_diversity": family_metrics.get("team_intra_family_diversity", 0.0),
            "dominant_family_share": family_metrics.get("dominant_family_share", 0.0),
            "dominant_major_family_share": family_metrics.get("dominant_major_family_share", 0.0),
            "family_judge_metric": family_metrics.get("family_judge_metric", "heuristic"),
            "per_agent_same_family_count": family_metrics.get("per_agent_same_family_count", []),
            "per_agent_same_family_ratio": family_metrics.get("per_agent_same_family_ratio", []),
            "role_positions": role_positions,
            "critical_agents": critical_agents,
            "high_homogeneity_agents": high_homogeneity_agents,
        }

    def build_group_context_summary(self, group_context: Dict[str, Any]) -> Dict[str, Any]:
        question_hash = str(group_context.get("question_hash", ""))
        role_positions = []
        for pos in group_context.get("role_positions", []):
            if not isinstance(pos, dict):
                continue
            role_positions.append(
                {
                    "agent_id": int(pos.get("agent_id", -1)),
                    "primary_family": str(pos.get("primary_family", "decomposition")),
                    "secondary_family": str(pos.get("secondary_family", pos.get("primary_family", "decomposition"))),
                    "family_distribution": pos.get("family_distribution", {}),
                    "reasoning_summary": self._clean_template_field(str(pos.get("reasoning_summary", "")), 360),
                    "same_family_ratio": round(float(pos.get("same_family_ratio", 0.0)), 4),
                    "same_family_count": int(pos.get("same_family_count", 0)),
                }
            )
        return {
            "question_hash": question_hash,
            "primary_family_counts": group_context.get("primary_family_counts", {}),
            "weighted_family_distribution": group_context.get("weighted_family_distribution", {}),
            "major_family_distribution": group_context.get("major_family_distribution", {}),
            "team_family_homogeneity_rate": round(float(group_context.get("team_family_homogeneity_rate", 0.0)), 4),
            "team_family_diversity": round(float(group_context.get("team_family_diversity", 0.0)), 4),
            "team_major_family_diversity": round(float(group_context.get("team_major_family_diversity", 0.0)), 4),
            "team_intra_family_diversity": round(float(group_context.get("team_intra_family_diversity", 0.0)), 4),
            "llm_direct_diversity_score": group_context.get("llm_direct_diversity_score"),
            "llm_direct_diversity_reason": str(group_context.get("llm_direct_diversity_reason", "")),
            "critical_agents": list(group_context.get("critical_agents", [])),
            "high_homogeneity_agents": list(group_context.get("high_homogeneity_agents", [])),
            "role_positions": role_positions,
        }

    def build_peer_trace_summary(self, agent_id: int, trace: str, reward_pack: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_trace = normalize_spaces(trace)
        family_metrics = reward_pack.get("family_metrics", {}) if isinstance(reward_pack.get("family_metrics", {}), dict) else {}
        families = family_metrics.get("primary_families", [])
        secondary_families = family_metrics.get("secondary_families", families)
        distributions = family_metrics.get("agent_family_distributions", [])
        reasoning_summaries = family_metrics.get("reasoning_summaries", [])
        distinctive_features = family_metrics.get("distinctive_features", [])
        confidences = family_metrics.get("family_confidences", [])
        fallback_family = self.strategy_family_labels[0] if self.strategy_family_labels else "decomposition"
        return {
            "agent_id": agent_id,
            "trace_hash": self._prompt_hash(cleaned_trace) if cleaned_trace else "",
            "primary_family": families[agent_id] if agent_id < len(families) else fallback_family,
            "secondary_family": secondary_families[agent_id] if agent_id < len(secondary_families) else (families[agent_id] if agent_id < len(families) else fallback_family),
            "family_distribution": distributions[agent_id] if agent_id < len(distributions) else {},
            "reasoning_summary": reasoning_summaries[agent_id] if agent_id < len(reasoning_summaries) else self._fallback_reasoning_summary(trace),
            "distinctive_features": distinctive_features[agent_id] if agent_id < len(distinctive_features) else [],
            "confidence": confidences[agent_id] if agent_id < len(confidences) else 0.0,
            "trace_diversity": round(float(reward_pack["per_agent_family_diversity"][agent_id]), 4),
            "same_family_ratio": round(float(reward_pack["per_agent_same_family_ratio"][agent_id]), 4),
            "same_family_count": int(reward_pack["per_agent_same_family_count"][agent_id]),
        }

    def build_agent_trace_profile(self, agent_id: int, trace: str, reward_pack: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_trace = normalize_spaces(trace)
        family_metrics = reward_pack.get("family_metrics", {}) if isinstance(reward_pack.get("family_metrics", {}), dict) else {}
        families = family_metrics.get("primary_families", [])
        secondary_families = family_metrics.get("secondary_families", families)
        distributions = family_metrics.get("agent_family_distributions", [])
        reasoning_summaries = family_metrics.get("reasoning_summaries", [])
        strategy_steps = family_metrics.get("strategy_steps", [])
        distinctive_features = family_metrics.get("distinctive_features", [])
        evidence_spans = family_metrics.get("evidence_spans", [])
        confidences = family_metrics.get("family_confidences", [])
        fallback_family = self.strategy_family_labels[0] if self.strategy_family_labels else "decomposition"
        return {
            "agent_id": agent_id,
            "trace_hash": self._prompt_hash(cleaned_trace) if cleaned_trace else "",
            "trace_length": len(cleaned_trace),
            "primary_family": families[agent_id] if agent_id < len(families) else fallback_family,
            "secondary_family": secondary_families[agent_id] if agent_id < len(secondary_families) else (families[agent_id] if agent_id < len(families) else fallback_family),
            "family_distribution": distributions[agent_id] if agent_id < len(distributions) else {},
            "reasoning_summary": reasoning_summaries[agent_id] if agent_id < len(reasoning_summaries) else self._fallback_reasoning_summary(trace),
            "strategy_steps": strategy_steps[agent_id] if agent_id < len(strategy_steps) else [],
            "distinctive_features": distinctive_features[agent_id] if agent_id < len(distinctive_features) else [],
            "evidence_spans": evidence_spans[agent_id] if agent_id < len(evidence_spans) else [],
            "confidence": confidences[agent_id] if agent_id < len(confidences) else 0.0,
            "trace_diversity": round(float(reward_pack["per_agent_family_diversity"][agent_id]), 4),
            "same_family_ratio": round(float(reward_pack["per_agent_same_family_ratio"][agent_id]), 4),
            "same_family_count": int(reward_pack["per_agent_same_family_count"][agent_id]),
            "rho_i": round(float(reward_pack["per_agent_same_family_ratio"][agent_id]), 4),
            "invalid_trace_penalty": round(float(reward_pack.get("per_agent_invalid_trace_penalty", [0.0])[agent_id]), 4) if agent_id < len(reward_pack.get("per_agent_invalid_trace_penalty", [])) else 0.0,
        }

    def _clean_template_field(self, value: str, max_len: int) -> str:
        text = normalize_spaces(str(value)).replace(";", ",").replace("|", "/")
        return text[:max_len]

    def _normalize_agent_id_list(self, values: Any) -> List[int]:
        if not isinstance(values, list):
            return []
        valid: List[int] = []
        seen = set()
        upper = len(self.agents)
        for v in values:
            try:
                idx = int(v)
            except Exception:
                continue
            if 0 <= idx < upper and idx not in seen:
                valid.append(idx)
                seen.add(idx)
        return valid

    def _parse_hint_fields(self, hint: Any) -> Dict[str, str]:
        cleaned = normalize_spaces(str(hint))
        role_match = re.search(r"ROLE\s*[:=]\s*([^;]+)", cleaned, flags=re.IGNORECASE)
        focus_match = re.search(r"FOCUS\s*[:=]\s*([^;]+)", cleaned, flags=re.IGNORECASE)
        avoid_match = re.search(r"AVOID\s*[:=]\s*([^;]+)", cleaned, flags=re.IGNORECASE)
        role = self._clean_template_field(role_match.group(1), 72) if role_match else ""
        focus = self._clean_template_field(focus_match.group(1), 180) if focus_match else ""
        avoid = self._clean_template_field(avoid_match.group(1), 180) if avoid_match else ""
        return {
            "role": role,
            "focus": focus,
            "avoid": avoid,
        }

    def _stable_group_summary(
        self,
        raw_summary: Any,
        missing_modes: List[str],
        redundant_agents: List[int],
        critical_agents: List[int],
    ) -> str:
        cleaned = normalize_spaces(str(raw_summary))
        pattern_match = re.search(r"PATTERN\s*[:=]\s*([^;]+)", cleaned, flags=re.IGNORECASE)
        gap_match = re.search(r"GAP\s*[:=]\s*([^;]+)", cleaned, flags=re.IGNORECASE)
        action_match = re.search(r"ACTION\s*[:=]\s*([^;]+)", cleaned, flags=re.IGNORECASE)

        pattern = self._clean_template_field(pattern_match.group(1), 180) if pattern_match else ""
        gap = self._clean_template_field(gap_match.group(1), 180) if gap_match else ""
        action = self._clean_template_field(action_match.group(1), 180) if action_match else ""

        if not pattern:
            pattern = self._clean_template_field(
                cleaned or "Window shows recurring overlap in decomposition style and verification behavior.",
                180,
            )
        if not gap:
            gap = self._clean_template_field(
                ", ".join(missing_modes[:3]) if missing_modes else "underused reasoning modes",
                180,
            )
        if not action:
            red_text = ",".join(str(x) for x in redundant_agents) if redundant_agents else "none"
            crit_text = ",".join(str(x) for x in critical_agents) if critical_agents else "none"
            action = self._clean_template_field(
                f"Reduce overlap of agents {red_text} while preserving distinct roles of agents {crit_text}.",
                180,
            )

        return f"PATTERN={pattern};GAP={gap};ACTION={action}"

    def _stable_target_role_hints(
        self,
        raw_hints: Any,
        missing_modes: List[str],
        redundant_agents: List[int],
        critical_agents: List[int],
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
        hints_obj = raw_hints if isinstance(raw_hints, dict) else {}
        hints: Dict[str, str] = {}
        structured: Dict[str, Dict[str, str]] = {}

        for i in range(len(self.agents)):
            key = str(i)
            raw_hint = hints_obj.get(key, hints_obj.get(i, ""))
            parsed = self._parse_hint_fields(raw_hint)
            fallback_mode = missing_modes[i % len(missing_modes)] if missing_modes else "underused reasoning mode"

            if not parsed["role"]:
                if i in critical_agents:
                    parsed["role"] = "critical_diversifier"
                elif i in redundant_agents:
                    parsed["role"] = "redundancy_breaker"
                else:
                    parsed["role"] = "complementary_specialist"

            if not parsed["focus"]:
                if i in critical_agents:
                    parsed["focus"] = f"Preserve your distinct mode and deepen verification around {fallback_mode}."
                elif i in redundant_agents:
                    parsed["focus"] = f"Shift to {fallback_mode} with a distinctly different decomposition order."
                else:
                    parsed["focus"] = f"Adopt a complementary reasoning mode centered on {fallback_mode}."

            if not parsed["avoid"]:
                parsed["avoid"] = "Copying peer decomposition order, wording, or intermediate representation."

            role = self._clean_template_field(parsed["role"], 72)
            focus = self._clean_template_field(parsed["focus"], 180)
            avoid = self._clean_template_field(parsed["avoid"], 180)

            hints[key] = f"ROLE={role};FOCUS={focus};AVOID={avoid}"
            structured[key] = {
                "role": role,
                "focus": focus,
                "avoid": avoid,
            }

        return hints, structured

    def _build_rewriter_diagnosis_context(self, group_diagnosis: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
        missing_modes = group_diagnosis.get("missing_modes", [])
        if not isinstance(missing_modes, list):
            missing_modes = []

        redundant_agents = self._normalize_agent_id_list(group_diagnosis.get("redundant_agents", []))
        critical_agents = self._normalize_agent_id_list(group_diagnosis.get("critical_agents", []))
        target_hint = group_diagnosis.get("target_role_hints", {}).get(str(agent_id), "")
        target_hint_structured = group_diagnosis.get("target_role_hints_structured", {}).get(str(agent_id), {})

        return {
            "version": "v1",
            "window_size": int(group_diagnosis.get("window_size", 1)),
            "focus_k": int(group_diagnosis.get("focus_k", 1)),
            "group_summary": str(group_diagnosis.get("group_summary", "")),
            "missing_modes": [self._clean_template_field(x, 120) for x in missing_modes[:5]],
            "redundant_agents": redundant_agents,
            "critical_agents": critical_agents,
            "target_role_hint": str(target_hint),
            "target_role_hint_structured": target_hint_structured if isinstance(target_hint_structured, dict) else {},
            "agent_id": agent_id,
        }

    async def generate_group_diagnosis(
        self,
        question: str,
        traces: List[str],
        reward_pack: Dict[str, Any],
        window_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        group_context = self.build_group_context(question, traces, reward_pack)

        window_payload: Dict[str, Any]
        if window_records:
            scored = []
            for idx, rec in enumerate(window_records):
                rp = rec.get("reward_pack", {}) if isinstance(rec, dict) else {}
                score = float(rp.get("team_family_homogeneity_rate", 0.0))
                scored.append((score, idx, rec))
            scored.sort(key=lambda x: x[0], reverse=True)

            focus_k = min(3, max(1, len(scored) // 2))
            focus_items = scored[:focus_k]
            rest_items = scored[focus_k:]

            focus_cases = []
            for score, idx, rec in focus_items:
                q = str(rec.get("question", ""))
                ts = list(rec.get("traces", []))
                rp = rec.get("reward_pack", {}) if isinstance(rec.get("reward_pack", {}), dict) else {}
                ctx = self.build_group_context(q, ts, rp)
                focus_cases.append(
                    {
                        "window_index": idx,
                        "homogeneity_score": round(float(score), 4),
                        "summary": self.build_group_context_summary(ctx),
                    }
                )

            brief_cases = []
            for score, idx, rec in rest_items:
                q = normalize_spaces(str(rec.get("question", "")))
                rp = rec.get("reward_pack", {}) if isinstance(rec.get("reward_pack", {}), dict) else {}
                brief_cases.append(
                    {
                        "window_index": idx,
                        "question_hash": self._prompt_hash(q) if q else "",
                        "homogeneity_score": round(float(score), 4),
                        "team_family_diversity": round(float(rp.get("team_family_diversity", 0.0)), 4),
                    }
                )

            window_payload = {
                "window_size": len(window_records),
                "focus_k": focus_k,
                "focus_cases": focus_cases,
                "brief_cases": brief_cases,
                "current_case": self.build_group_context_summary(group_context),
            }
        else:
            window_payload = {
                "window_size": 1,
                "focus_k": 1,
                "focus_cases": [{"window_index": 0, "homogeneity_score": round(float(reward_pack.get("team_family_homogeneity_rate", 0.0)), 4), "summary": self.build_group_context_summary(group_context)}],
                "brief_cases": [],
                "current_case": self.build_group_context_summary(group_context),
            }

        system_prompt = (
            "You are a group-aware critic for homogeneous multi-agent reasoning.\n"
            "Your job is to summarize the window-level collaboration pattern, not to solve the task.\n"
            "Prioritize diversity, complementarity, anti-redundancy, and missing reasoning modes over answer correctness.\n"
            "Use the structured window evidence to infer repeated failure modes and specialization gaps.\n"
            "Return strict JSON only.\n"
        )
        user_prompt = (
            "Diagnose diversity only. Focus on overused reasoning families, missing reasoning modes, "
            "which agents should move away from crowded families, and which general strategy each agent should adopt next.\n"
            "Do not discuss answer correctness, do not mention gold answer, and do not optimize accuracy.\n\n"
            "Important: focus on focus_cases, but summarize them at the level of trace patterns and statistics rather than copying trace content.\n"
            "For each focus case, use its structured summary to infer the agent's role, not the raw question wording.\n"
            "Use brief_cases only for calibration of window-wide trends.\n\n"
            "Formatting constraints (strict):\n"
            "- group_summary must be exactly one line in this format:\n"
            "  PATTERN=<summary>;GAP=<missing_modes>;ACTION=<team-level adjustment>.\n"
            "- target_role_hints must contain all agent IDs as string keys from 0 to "
            f"{len(self.agents) - 1}.\n"
            "- Each target_role_hints value must use exactly:\n"
            "  ROLE=<role_label>;FOCUS=<agent-specific shift>;AVOID=<anti-overlap rule>.\n"
            "- Keep each field concise and strategy-level only.\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "group_summary": str,\n'
            '  "missing_modes": [str, ...],\n'
            '  "redundant_agents": [int, ...],\n'
            '  "critical_agents": [int, ...],\n'
            '  "target_role_hints": {"0": str, "1": str, ...}\n'
            "}\n\n"
            f"Window context:\n{json.dumps(window_payload, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.critic_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.cfg.critic_temperature,
            max_tokens=self.cfg.critic_max_tokens,
            stage="group_diagnosis",
        )
        obj = extract_json_obj(text) or {}
        if not isinstance(obj, dict):
            obj = {}

        raw_missing_modes = obj.get("missing_modes", ["equation-first verification", "backward checking"])
        missing_modes: List[str] = []
        seen_modes = set()
        if isinstance(raw_missing_modes, list):
            for m in raw_missing_modes:
                mode = self._clean_template_field(m, 120)
                key = mode.lower()
                if mode and key not in seen_modes:
                    missing_modes.append(mode)
                    seen_modes.add(key)
        if not missing_modes:
            missing_modes = ["equation-first verification", "backward checking"]

        default_critical = self._normalize_agent_id_list(group_context.get("critical_agents", []))
        redundant_agents = self._normalize_agent_id_list(obj.get("redundant_agents", []))
        critical_agents = self._normalize_agent_id_list(obj.get("critical_agents", default_critical))
        stable_group_summary = self._stable_group_summary(
            obj.get("group_summary", ""),
            missing_modes,
            redundant_agents,
            critical_agents,
        )
        stable_hints, stable_hints_structured = self._stable_target_role_hints(
            obj.get("target_role_hints", {}),
            missing_modes,
            redundant_agents,
            critical_agents,
        )
        return {
            "group_summary": stable_group_summary,
            "missing_modes": missing_modes,
            "redundant_agents": redundant_agents,
            "critical_agents": critical_agents,
            "target_role_hints": stable_hints,
            "target_role_hints_structured": stable_hints_structured,
            "window_size": int(window_payload.get("window_size", 1)),
            "focus_k": int(window_payload.get("focus_k", 1)),
        }

    async def generate_textual_gradient(
        self,
        question: str,
        traces: List[str],
        reward_pack: Dict[str, Any],
        agent_id: int,
        group_diagnosis: Dict[str, Any],
    ) -> str:
        agent_trace = traces[agent_id] if agent_id < len(traces) else ""
        agent_info = {
            "agent_id": agent_id,
            "trace_profile": self.build_agent_trace_profile(agent_id, agent_trace, reward_pack),
            "target_role_hint_structured": group_diagnosis.get("target_role_hints_structured", {}).get(str(agent_id), {}),
        }
        compact_group_diagnosis = self._build_rewriter_diagnosis_context(group_diagnosis, agent_id)
        peer_summary = [
            self.build_peer_trace_summary(i, traces[i], reward_pack)
            for i in range(len(traces))
            if i != agent_id
        ]
        target_profile = agent_info["trace_profile"]
        peer_profiles = [
            {
                "agent_id": p.get("agent_id"),
                "primary_family": p.get("primary_family"),
                "secondary_family": p.get("secondary_family"),
                "reasoning_summary": p.get("reasoning_summary"),
                "distinctive_features": p.get("distinctive_features", []),
                "confidence": p.get("confidence", 0.0),
            }
            for p in peer_summary
        ]

        system_prompt = (
            "You are a prompt optimizer that writes group-aware textual gradients.\n"
            "A textual gradient is a compact optimization signal that should explain what to change in the prompt and why it helps the team.\n"
            "Use only the agent's own trace profile and the peers' compressed summaries, not raw peer traces.\n"
            "Return strict JSON only."
        )
        task_name = "MMLU" if str(self.cfg.task_type).lower() == "mmlu" else "GSM8K"
        user_prompt = (
            f"Write a group-aware textual gradient for one agent in a homogeneous multi-agent {task_name} system.\n"
            "The advice must be relative to the rest of the team, but the final instruction must stay reusable across future questions.\n\n"
            "Important constraints:\n"
            "- Do NOT include or paraphrase the current question text.\n"
            "- Do NOT include options, entity names, numbers, or answer templates from any single example.\n"
            "- Do NOT discuss answer correctness or the gold answer.\n"
            "- Use the agent's own trace profile to identify its current failure mode, then prescribe a prompt-level shift.\n"
            "- Use peer summaries only to explain the team gap, not to mirror their wording or trajectories.\n"
            "- Output only general strategy-level guidance reusable across future questions.\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "diagnosis": str,\n'
            '  "redundant_pattern": str,\n'
            '  "desired_shift": str,\n'
            '  "prompt_edit_instruction": str\n'
            "}\n\n"
            f"Group diagnosis:\n{json.dumps(compact_group_diagnosis, ensure_ascii=False, indent=2)}\n\n"
            f"Target structured profile:\n{json.dumps(target_profile, ensure_ascii=False, indent=2)}\n\n"
            f"Target role hint fields:\n{json.dumps(agent_info['target_role_hint_structured'], ensure_ascii=False, indent=2)}\n\n"
            f"Peer structured profiles:\n{json.dumps(peer_profiles, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.critic_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.cfg.critic_temperature,
            max_tokens=self.cfg.critic_max_tokens,
            stage=f"textual_gradient_agent_{agent_id}",
        )
        obj = extract_json_obj(text) or {}
        diagnosis = obj.get("diagnosis", "Your current reasoning style is insufficiently differentiated from peers.")
        redundant_pattern = obj.get("redundant_pattern", "The target profile overlaps with peer reasoning profiles.")
        desired_shift = obj.get("desired_shift", obj.get("behavior_shift", "Shift toward a more complementary reasoning trajectory relative to peers."))
        prompt_edit_instruction = obj.get(
            "prompt_edit_instruction",
            "Edit the prompt to emphasize a distinct, non-overlapping problem-solving strategy.",
        )
        return (
            f"Diagnosis: {diagnosis}\n"
            f"Redundant Pattern: {redundant_pattern}\n"
            f"Desired Shift: {desired_shift}\n"
            f"Prompt Edit Instruction: {prompt_edit_instruction}"
        )

    async def propose_candidates(
        self,
        question: str,
        agent_id: int,
        textual_gradient: str,
        group_diagnosis: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        target_role_hint_structured = group_diagnosis.get("target_role_hints_structured", {}).get(str(agent_id), {})
        compact_group_diagnosis = self._build_rewriter_diagnosis_context(group_diagnosis, agent_id)
        system_prompt = (
            "You are a group-aware prompt rewriter for homogeneous multi-agent reasoning.\n"
            "You will propose candidate prompt edits for one agent.\n"
            "The edits must improve coverage, reduce redundancy, and make the agent more complementary to the team.\n"
            "Use the agent's own trace profile as the main evidence source; use peer summaries only to avoid overlap.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            "Given the current prompt, the group diagnosis, and the agent-specific textual gradient,\n"
            "propose exactly 3 candidate prompt rewrites with distinct styles:\n"
            "1) conservative_specialization\n"
            "2) coverage_gap_shift\n"
            "3) anti_redundancy_shift\n\n"
            "Important constraints:\n"
            "- The rewritten prompt must be general and reusable across future tasks.\n"
            "- Never include any concrete question text, options, named entities, or FINAL_ANSWER templates.\n"
            "- Do not include task-specific entities, numbers, options, or answers.\n"
            "- Do not force the agent to use one fixed family on every task.\n"
            "- Each candidate must express a transferable reasoning bias, an applicability condition, and a fallback strategy.\n"
            "- Make the prompt changes operational: tell the agent how to think, when to prioritize that behavior, and how to fall back when it is not suitable.\n"
            "- Focus only on reasoning style, decomposition strategy, verification behavior, and role separation.\n\n"
            "Stable input contract:\n"
            "- Use the compact group diagnosis object as the only team-level input.\n"
            "- Treat target_role_hint_structured as the canonical per-agent role specification.\n"
            "- Do not depend on any other free-form fields in the full diagnosis object.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"name": str, "reasoning_bias": str, "trajectory_shift": str, "applicability_condition": str, "fallback_strategy": str, "task_agnostic_prompt": str, "rationale": str},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            f"Current prompt:\n{self.agents[agent_id].current_prompt}\n\n"
            f"Textual gradient:\n{textual_gradient}\n\n"
            f"Group diagnosis:\n{json.dumps(compact_group_diagnosis, ensure_ascii=False, indent=2)}\n\n"
            f"Target role hint fields:\n{json.dumps(target_role_hint_structured, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.rewriter_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.cfg.rewriter_temperature,
            max_tokens=self.cfg.rewriter_max_tokens,
            stage=f"rewriter_agent_{agent_id}",
        )
        obj = extract_json_obj(text) or {}
        candidates = obj.get("candidates", [])

        parsed = []
        if isinstance(candidates, list):
            for item in candidates[:3]:
                if isinstance(item, dict) and (item.get("task_agnostic_prompt") or item.get("prompt")):
                    candidate_prompt = str(item.get("task_agnostic_prompt") or item.get("prompt")).strip()
                    candidate_prompt, _ = self._sanitize_prompt(candidate_prompt, agent_id, question=question)
                    item = dict(item)
                    item["task_agnostic_prompt"] = candidate_prompt
                    valid, invalid_reason = self.validate_candidate_prompt(item, question)
                    parsed.append(
                        {
                            "name": str(item.get("name", "candidate")),
                            "prompt": candidate_prompt,
                            "reasoning_bias": str(item.get("reasoning_bias", "")),
                            "trajectory_shift": str(item.get("trajectory_shift", "")),
                            "applicability_condition": str(item.get("applicability_condition", "")),
                            "fallback_strategy": str(item.get("fallback_strategy", "")),
                            "task_agnostic_prompt": candidate_prompt,
                            "rationale": str(item.get("rationale", "")),
                            "valid_candidate_prompt": bool(valid),
                            "candidate_validation_error": invalid_reason,
                        }
                    )

        if len(parsed) < 3:
            base = self.agents[agent_id].current_prompt
            fallback = [
                {
                    "name": "conservative_specialization",
                    "reasoning_bias": "explicit intermediate representation",
                    "trajectory_shift": "Keep the current role but make the intermediate representation more explicit before selecting an answer.",
                    "applicability_condition": "Use when the problem has relationships, categories, or constraints that can be written down compactly.",
                    "fallback_strategy": "If no useful representation emerges, fall back to concise concept matching and verification.",
                    "task_agnostic_prompt": base + " Prefer to make one compact intermediate representation of the key relationships before selecting an answer. If that representation is not useful, fall back to concise concept matching and a final consistency check.",
                    "prompt": base + " Prefer to make one compact intermediate representation of the key relationships before selecting an answer. If that representation is not useful, fall back to concise concept matching and a final consistency check.",
                    "rationale": "Strength-preserving edit.",
                    "valid_candidate_prompt": True,
                    "candidate_validation_error": "",
                },
                {
                    "name": "coverage_gap_shift",
                    "reasoning_bias": "boundary and condition checking",
                    "trajectory_shift": "Before direct selection, identify boundaries, exceptions, or conditions under which an answer would fail.",
                    "applicability_condition": "Use when options or claims differ by definitions, rules, assumptions, or absolute wording.",
                    "fallback_strategy": "If boundary testing is not informative, fall back to direct concept matching and option contrast.",
                    "task_agnostic_prompt": base + " When a problem contains rules, definitions, or subtly different options, first test boundaries, exceptions, or failure conditions. If no meaningful boundary exists, fall back to direct concept matching and option comparison.",
                    "prompt": base + " When a problem contains rules, definitions, or subtly different options, first test boundaries, exceptions, or failure conditions. If no meaningful boundary exists, fall back to direct concept matching and option comparison.",
                    "rationale": "Shift toward missing equation-centric coverage.",
                    "valid_candidate_prompt": True,
                    "candidate_validation_error": "",
                },
                {
                    "name": "anti_redundancy_shift",
                    "reasoning_bias": "answer-to-question backward validation",
                    "trajectory_shift": "Start from plausible answers and check what would need to be true for each to fit.",
                    "applicability_condition": "Use when multiple options look plausible or differ by subtle conceptual conditions.",
                    "fallback_strategy": "If backward validation is not useful, return to direct reasoning with an explicit uncertainty check.",
                    "task_agnostic_prompt": base + " For each plausible answer, briefly reason backward: if it were correct, what conditions would need to hold? Compare those conditions with the problem statement. If this backward check is not informative, return to direct reasoning and state the key uncertainty.",
                    "prompt": base + " For each plausible answer, briefly reason backward: if it were correct, what conditions would need to hold? Compare those conditions with the problem statement. If this backward check is not informative, return to direct reasoning and state the key uncertainty.",
                    "rationale": "Reduce redundancy with peers.",
                    "valid_candidate_prompt": True,
                    "candidate_validation_error": "",
                },
            ]
            need = 3 - len(parsed)
            parsed.extend(fallback[:need])

        return parsed[:3]

    async def evaluate_candidate_minibatch(
        self,
        eval_batch: List[Dict[str, str]],
        agent_id: int,
        candidate_prompt: str,
    ) -> Dict[str, Any]:
        original_prompt = self.agents[agent_id].current_prompt

        async def eval_one(ex: Dict[str, str], prompt: str) -> Dict[str, Any]:
            q = ex["question"]
            gold = parse_gold(ex["answer"], self.cfg.task_type)
            traces, answers, family_labels, family_judgments = await self.solve_with_agent_prompt_override(q, agent_id, prompt)
            reward_pack = self.compute_rewards(
                traces,
                answers,
                gold,
                primary_family_labels=family_labels,
                family_judgments=family_judgments,
            )
            family_metrics = reward_pack.get("family_metrics", {}) if isinstance(reward_pack.get("family_metrics", {}), dict) else {}
            summaries = family_metrics.get("reasoning_summaries", [])
            primary = family_metrics.get("primary_families", [])
            secondary = family_metrics.get("secondary_families", primary)
            rhos = reward_pack.get("per_agent_same_family_ratio", [])
            invalids = reward_pack.get("per_agent_invalid_trace_penalty", [])
            return {
                "reward": float(reward_pack["rewards"][agent_id]),
                "primary_family": primary[agent_id] if agent_id < len(primary) else "",
                "secondary_family": secondary[agent_id] if agent_id < len(secondary) else "",
                "rho_i": float(rhos[agent_id]) if agent_id < len(rhos) else 0.0,
                "invalid_trace_penalty": float(invalids[agent_id]) if agent_id < len(invalids) else 0.0,
                "reasoning_summary": summaries[agent_id] if agent_id < len(summaries) else "",
            }

        tasks = [eval_one(ex, candidate_prompt) for ex in eval_batch]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        self.agents[agent_id].current_prompt = original_prompt

        results = [r for r in results_raw if isinstance(r, dict)]
        rewards = [float(r.get("reward", 0.0)) for r in results]

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "batch_size": len(eval_batch),
            "evaluated_size": len(results),
            "family_agent_i": [r.get("primary_family", "") for r in results],
            "secondary_family_agent_i": [r.get("secondary_family", "") for r in results],
            "rho_i": [float(r.get("rho_i", 0.0)) for r in results],
            "invalid_trace_penalty_i": [float(r.get("invalid_trace_penalty", 0.0)) for r in results],
            "reasoning_summary_i": [str(r.get("reasoning_summary", "")) for r in results],
            "errors": [normalize_spaces(str(r))[:300] for r in results_raw if isinstance(r, Exception)],
        }

    def _candidate_eval_behavior_diagnostics(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        before_family = list(before.get("family_agent_i", []))
        after_family = list(after.get("family_agent_i", []))
        n = min(len(before_family), len(after_family))
        family_shift_rate = float(np.mean([1.0 if before_family[i] != after_family[i] else 0.0 for i in range(n)])) if n else 0.0
        before_rho = [float(x) for x in before.get("rho_i", [])]
        after_rho = [float(x) for x in after.get("rho_i", [])]
        rn = min(len(before_rho), len(after_rho))
        rho_reduction = float(np.mean([before_rho[i] - after_rho[i] for i in range(rn)])) if rn else 0.0
        before_invalid = [float(x) for x in before.get("invalid_trace_penalty_i", [])]
        after_invalid = [float(x) for x in after.get("invalid_trace_penalty_i", [])]
        vn = min(len(before_invalid), len(after_invalid))
        invalid_delta = float(np.mean([after_invalid[i] - before_invalid[i] for i in range(vn)])) if vn else 0.0
        before_summaries = list(before.get("reasoning_summary_i", []))
        after_summaries = list(after.get("reasoning_summary_i", []))
        sn = min(len(before_summaries), len(after_summaries))
        summary_shift = float(np.mean([self._token_cosine_distance(before_summaries[i], after_summaries[i]) for i in range(sn)])) if sn else 0.0
        return {
            "family_before_agent_i": before_family,
            "family_after_agent_i": after_family,
            "rho_before_i": before_rho,
            "rho_after_i": after_rho,
            "reward_before_i": before.get("mean_reward", 0.0),
            "reward_after_i": after.get("mean_reward", 0.0),
            "invalid_before_i": before_invalid,
            "invalid_after_i": after_invalid,
            "summary_before_i": before_summaries,
            "summary_after_i": after_summaries,
            "family_shift_rate": family_shift_rate,
            "rho_reduction": rho_reduction,
            "invalid_delta": invalid_delta,
            "summary_embedding_shift": summary_shift,
        }

    async def maybe_update_prompts(
        self,
        question: str,
        traces: List[str],
        reward_pack: Dict[str, Any],
        eval_batch: List[Dict[str, str]],
        step_id: int,
        epoch_id: int,
    ) -> Dict[str, Any]:
        try:
            window_records = list(self.recent_window_records)
            group_diagnosis = await self.generate_group_diagnosis(
                question,
                traces,
                reward_pack,
                window_records=window_records,
            )
        except Exception as e:
            err_msg = normalize_spaces(str(e))[:500]
            print(f"[WARN] Step {step_id} update skipped: group diagnosis failed: {err_msg}")
            self.update_logs.append(
                {
                    **self._base_log_fields(),
                    "epoch": epoch_id,
                    "step": step_id,
                    "agent_id": None,
                    "selected_agent_ids": [],
                    "question_hash": self._prompt_hash(question) if question else "",
                    "decision": "skip_error",
                    "error_stage": "group_diagnosis",
                    "error": err_msg,
                }
            )
            self.flush_update_logs()
            # Window-based stats are batch-like: once window is full and reaches an update
            # checkpoint, clear it for the next batch even if update generation failed.
            self.clear_homogeneity_windows()
            return {
                "update_requested": True,
                "update_ready": True,
                "group_diagnosis_ok": False,
                "selected_agent_ids": [],
                "updated_agent_ids": [],
                "skipped_reason": "group_diagnosis_error",
            }

        selected_agent_ids = self.select_agents_for_update(reward_pack)
        updated_agent_ids: List[int] = []
        skip_reason = "none" if selected_agent_ids else "no_agents_selected"
        generic_prompt_candidates = 0
        total_prompt_candidates = 0
        candidate_behavior_records: List[Dict[str, Any]] = []

        for i in selected_agent_ids:
            # For update-stage failures, retry transient/network errors indefinitely,
            # but skip on persistent/non-transient errors while logging them.
            while True:
                try:
                    # Hard guard: sanitize contaminated prompts before any further optimization.
                    sanitized_current, current_was_sanitized = self._sanitize_prompt(
                        self.agents[i].current_prompt,
                        i,
                        question=question,
                    )
                    if current_was_sanitized:
                        self.agents[i].current_prompt = sanitized_current
                        sanitized_hash = self._prompt_hash(sanitized_current)
                        self._append_prompt_history_event(
                            agent_id=i,
                            epoch_id=epoch_id,
                            step_id=step_id,
                            decision="sanitize_leak",
                            selected_action_id=None,
                            selected_action_name="sanitize_leak",
                            current_prompt=sanitized_current,
                            current_prompt_hash=sanitized_hash,
                            changed=1,
                        )

                    tg = await self.generate_textual_gradient(question, traces, reward_pack, i, group_diagnosis)
                    self.agents[i].gradient_history.append(tg)

                    candidates = await self.propose_candidates(question, i, tg, group_diagnosis)
                    for cand in candidates:
                        total_prompt_candidates += 1
                        if isinstance(cand, dict) and not bool(cand.get("valid_candidate_prompt", True)):
                            generic_prompt_candidates += 1
                    actions = [{"name": "keep_current", "prompt": self.agents[i].current_prompt, "rationale": "Keep current prompt."}] + candidates
                    while len(actions) < 5:
                        actions.append({"name": "keep_current", "prompt": self.agents[i].current_prompt, "rationale": "Pad action."})

                    action_id, probs = self.agents[i].bandit.sample()
                    selected = actions[action_id]
                    chosen_prompt = selected["prompt"]
                    chosen_prompt, chosen_was_sanitized = self._sanitize_prompt(chosen_prompt, i, question=question)
                    before_prompt_hash = self._prompt_hash(self.agents[i].current_prompt)
                    selected_is_keep = str(selected.get("name", "")).strip().lower() == "keep_current"

                    current_batch_metrics = await self.evaluate_candidate_minibatch(eval_batch, i, self.agents[i].current_prompt)
                    current_batch_reward = current_batch_metrics["mean_reward"]

                    log_record = {
                        **self._base_log_fields(),
                        "epoch": epoch_id,
                        "step": step_id,
                        "agent_id": i,
                        "selected_agent_ids": selected_agent_ids,
                        "question_hash": self._prompt_hash(question) if question else "",
                        "group_diagnosis": group_diagnosis,
                        "textual_gradient": tg,
                        "agent_initial_prompt_hash": self._prompt_hash(self.agents[i].initial_prompt),
                        "current_prompt_hash": self._prompt_hash(self.agents[i].current_prompt),
                        "candidates": candidates,
                        "bandit_probs": probs.tolist(),
                        "selected_action_id": action_id,
                        "selected_action_name": selected["name"],
                        "selected_prompt_hash": self._prompt_hash(chosen_prompt),
                        "current_batch_metrics": current_batch_metrics,
                    }

                    if selected_is_keep:
                        self.agents[i].reject_count += 1
                        self.agents[i].bandit.update(action_id, current_batch_reward)
                        log_record["decision"] = "keep"
                        log_record["selected_batch_metrics"] = current_batch_metrics
                        self._append_prompt_history_event(
                            agent_id=i,
                            epoch_id=epoch_id,
                            step_id=step_id,
                            decision="keep",
                            selected_action_id=action_id,
                            selected_action_name=selected["name"],
                            current_prompt=self.agents[i].current_prompt,
                            current_prompt_hash=before_prompt_hash,
                            changed=0,
                        )
                        compact_record = self._compact_update_log_record(log_record)
                        self.agents[i].last_update_record = compact_record
                        self.update_logs.append(compact_record)
                        break

                    candidate_batch_metrics = await self.evaluate_candidate_minibatch(eval_batch, i, chosen_prompt)
                    candidate_reward = candidate_batch_metrics["mean_reward"]
                    behavior_diagnostics = self._candidate_eval_behavior_diagnostics(current_batch_metrics, candidate_batch_metrics)
                    candidate_behavior_records.append(behavior_diagnostics)
                    log_record["candidate_behavior_diagnostics"] = behavior_diagnostics
                    self.agents[i].bandit.update(action_id, candidate_reward)
                    log_record["selected_batch_metrics"] = candidate_batch_metrics
                    if chosen_was_sanitized:
                        log_record["selected_prompt_sanitized"] = True

                    invalid_ok = float(behavior_diagnostics.get("invalid_delta", 0.0)) <= float(getattr(self.cfg, "invalid_tolerance", 0.1))
                    reward_delta = float(candidate_reward - current_batch_reward)
                    if abs(reward_delta) < float(getattr(self.cfg, "reward_tie_eps", 0.03)):
                        tie_score = (
                            float(behavior_diagnostics.get("summary_embedding_shift", 0.0))
                            + float(behavior_diagnostics.get("rho_reduction", 0.0))
                            - max(0.0, float(behavior_diagnostics.get("invalid_delta", 0.0)))
                        )
                        accept_candidate = invalid_ok and (reward_delta >= 0.0 or tie_score > 0.0)
                        log_record["tie_break_score"] = tie_score
                    else:
                        accept_candidate = reward_delta >= 0.0 and invalid_ok
                    log_record["invalid_tolerance_ok"] = bool(invalid_ok)

                    if accept_candidate:
                        self.agents[i].current_prompt = chosen_prompt
                        self.agents[i].history.append(chosen_prompt)
                        self.agents[i].accept_count += 1
                        log_record["decision"] = "accept"
                        updated_agent_ids.append(i)
                    else:
                        self.agents[i].reject_count += 1
                        log_record["decision"] = "reject"

                    after_prompt_hash = self._prompt_hash(self.agents[i].current_prompt)
                    after_prompt = self.agents[i].current_prompt
                    if before_prompt_hash != after_prompt_hash:
                        self._append_prompt_history_event(
                            agent_id=i,
                            epoch_id=epoch_id,
                            step_id=step_id,
                            decision=log_record["decision"],
                            selected_action_id=action_id,
                            selected_action_name=selected["name"],
                            current_prompt=after_prompt,
                            current_prompt_hash=after_prompt_hash,
                            changed=1,
                        )

                    compact_record = self._compact_update_log_record(log_record)
                    self.agents[i].last_update_record = compact_record
                    self.update_logs.append(compact_record)
                    break
                except Exception as e:
                    err_msg = normalize_spaces(str(e))[:500]
                    tb = traceback.format_exc()
                    is_transient = self._is_transient_llm_error(e)
                    report = {
                        **self._base_log_fields(),
                        "epoch": epoch_id,
                        "step": step_id,
                        "agent_id": i,
                        "selected_agent_ids": selected_agent_ids,
                        "question_hash": self._prompt_hash(question) if question else "",
                        "decision": "retrying" if is_transient else "skip_error",
                        "error_stage": "agent_update",
                        "error": err_msg,
                        "traceback": tb,
                        "transient": bool(is_transient),
                        "time": time.time(),
                    }
                    self.update_logs.append(self._compact_update_log_record(report))
                    self.flush_update_logs()
                    if is_transient:
                        print(f"[WARN] Step {step_id} agent {i} update transient error, retrying: {err_msg}")
                        await asyncio.sleep(self.cfg.retry_sleep)
                        continue
                    else:
                        print(f"[WARN] Step {step_id} agent {i} update skipped (non-transient): {err_msg}")
                        log_record = {
                            **self._base_log_fields(),
                            "epoch": epoch_id,
                            "step": step_id,
                            "agent_id": i,
                            "selected_agent_ids": selected_agent_ids,
                            "question_hash": self._prompt_hash(question) if question else "",
                            "decision": "skip_error",
                            "error_stage": "agent_update",
                            "error": err_msg,
                        }
                        compact_record = self._compact_update_log_record(log_record)
                        self.agents[i].last_update_record = compact_record
                        self.update_logs.append(compact_record)
                        break

        # Window-based stats are consumed in batches.
        # Once a full-window checkpoint is reached, clear windows regardless of whether
        # selected_agent_ids is empty (e.g., all-zero window) to avoid stale carryover.
        self.clear_homogeneity_windows()

        self.flush_update_logs()
        self.flush_prompt_history()
        mean_behavior = {
            "family_shift_rate": float(np.mean([float(x.get("family_shift_rate", 0.0)) for x in candidate_behavior_records])) if candidate_behavior_records else 0.0,
            "rho_reduction": float(np.mean([float(x.get("rho_reduction", 0.0)) for x in candidate_behavior_records])) if candidate_behavior_records else 0.0,
            "invalid_delta": float(np.mean([float(x.get("invalid_delta", 0.0)) for x in candidate_behavior_records])) if candidate_behavior_records else 0.0,
            "summary_embedding_shift": float(np.mean([float(x.get("summary_embedding_shift", 0.0)) for x in candidate_behavior_records])) if candidate_behavior_records else 0.0,
        }
        return {
            "update_requested": True,
            "update_ready": True,
            "group_diagnosis_ok": True,
            "selected_agent_ids": selected_agent_ids,
            "updated_agent_ids": updated_agent_ids,
            "skipped_reason": skip_reason,
            "generic_prompt_candidate_rate": float(generic_prompt_candidates / total_prompt_candidates) if total_prompt_candidates else 0.0,
            "candidate_behavior_diagnostics": mean_behavior,
        }

    def select_agents_for_update(self, reward_pack: Dict[str, Any]) -> List[int]:
        num_agents = len(self.agents)
        if num_agents == 0:
            return []
        # Use sliding-window homogeneity count to decide updates.
        # If windows are not yet warm (not full), do not select any agents here.
        if not self.is_homogeneity_window_warmup_done():
            return []

        family_counts = reward_pack.get("per_agent_same_family_count", [0 for _ in self.agents])
        if not isinstance(family_counts, list):
            family_counts = [0 for _ in self.agents]
        same_ratios = reward_pack.get("per_agent_same_family_ratio", [0.0 for _ in self.agents])
        invalid_penalties = reward_pack.get("per_agent_invalid_trace_penalty", [0.0 for _ in self.agents])

        pressures = []
        for i in range(num_agents):
            family_pressure = float(same_ratios[i]) if i < len(same_ratios) else 0.0
            invalid_pressure = float(invalid_penalties[i]) if i < len(invalid_penalties) else 0.0
            pressure = 0.85 * family_pressure + 0.15 * invalid_pressure
            pressures.append(float(pressure))

        # If within the window all agents have zero mixed homogeneity pressure, skip updating.
        if all(p <= 0.0 for p in pressures):
            return []

        # Otherwise, select top agents with highest recent family-homogeneity count.
        indices = list(range(num_agents))
        random.shuffle(indices)
        indices.sort(
            key=lambda i: (
                int(self.agents[i].homogeneity_count),
                pressures[i],
                int(family_counts[i]) if i < len(family_counts) else 0,
            ),
            reverse=True,
        )

        num_with_family_history = sum(1 for a in self.agents if a.homogeneity_count > 0)
        select_k = 2 if num_with_family_history >= 2 else 1
        return indices[:select_k]

    async def rollout_train_example(
        self,
        question: str,
        gold: str,
        do_update: bool = True,
        eval_batch: Optional[List[Dict[str, str]]] = None,
        step_id: int = 0,
        epoch_id: int = 0,
    ) -> Dict[str, Any]:
        # Pre-step hard guard: keep all agent prompts task-agnostic.
        for i, agent in enumerate(self.agents):
            sanitized_prompt, was_sanitized = self._sanitize_prompt(agent.current_prompt, i, question=question)
            if was_sanitized:
                agent.current_prompt = sanitized_prompt
                prompt_hash = self._prompt_hash(sanitized_prompt)
                self._append_prompt_history_event(
                    agent_id=i,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    decision="sanitize_leak",
                    selected_action_id=None,
                    selected_action_name="sanitize_leak",
                    current_prompt=sanitized_prompt,
                    current_prompt_hash=prompt_hash,
                    changed=1,
                )

        traces, answers, family_labels, family_judgments = await self.solve_with_current_prompts_with_family(question)
        reward_pack = self.compute_rewards(
            traces,
            answers,
            gold,
            primary_family_labels=family_labels,
            family_judgments=family_judgments,
        )

        self.recent_window_records.append(
            {
                "question": question,
                "traces": traces,
                "reward_pack": reward_pack,
            }
        )
        max_window = max(1, int(self.cfg.homogeneity_window))
        if len(self.recent_window_records) > max_window:
            self.recent_window_records = self.recent_window_records[-max_window:]

        same_ratios = reward_pack.get("per_agent_same_family_ratio", [])
        invalid_penalties = reward_pack.get("per_agent_invalid_trace_penalty", [])
        for i in range(len(self.agents)):
            family_pressure = float(same_ratios[i]) if i < len(same_ratios) else 0.0
            invalid_pressure = float(invalid_penalties[i]) if i < len(invalid_penalties) else 0.0
            pressure = 0.85 * family_pressure + 0.15 * invalid_pressure
            self.agents[i].observe_homogeneity_result(1 if pressure > 0.0 else 0)

        homogeneity_counts_before_update = [int(a.homogeneity_count) for a in self.agents]
        ready_for_update = self.is_homogeneity_window_warmup_done()
        update_summary = {
            "update_requested": bool(do_update),
            "update_ready": bool(ready_for_update),
            "group_diagnosis_ok": None,
            "selected_agent_ids": [],
            "updated_agent_ids": [],
            "skipped_reason": "not_requested" if not do_update else "not_ready",
        }
        if do_update and eval_batch is not None and ready_for_update:
            update_summary = await self.maybe_update_prompts(question, traces, reward_pack, eval_batch, step_id, epoch_id)

        step_log = self._build_train_step_log(epoch_id, step_id, reward_pack, update_summary)
        self.train_step_logs.append(step_log)
        family_metrics = reward_pack.get("family_metrics", {}) if isinstance(reward_pack.get("family_metrics", {}), dict) else {}
        primary_families = list(family_metrics.get("primary_families", []))
        secondary_families = list(family_metrics.get("secondary_families", primary_families))
        family_distributions = list(family_metrics.get("agent_family_distributions", []))
        reasoning_summaries = list(family_metrics.get("reasoning_summaries", []))
        strategy_steps_list = list(family_metrics.get("strategy_steps", []))
        distinctive_features_list = list(family_metrics.get("distinctive_features", []))
        evidence_spans_list = list(family_metrics.get("evidence_spans", []))
        confidence_list = list(family_metrics.get("family_confidences", []))
        trace_log = {
            "epoch": epoch_id,
            "step": step_id,
            "question_hash": self._prompt_hash(normalize_spaces(question)) if question else "",
            "agents": [
                {
                    "agent_id": i,
                    "primary_family": primary_families[i] if i < len(primary_families) else "",
                    "secondary_family": secondary_families[i] if i < len(secondary_families) else "",
                    "family_distribution": family_distributions[i] if i < len(family_distributions) else {},
                    "reasoning_summary": reasoning_summaries[i] if i < len(reasoning_summaries) else "",
                    "strategy_steps": strategy_steps_list[i] if i < len(strategy_steps_list) else [],
                    "distinctive_features": distinctive_features_list[i] if i < len(distinctive_features_list) else [],
                    "evidence_spans": evidence_spans_list[i] if i < len(evidence_spans_list) else [],
                    "confidence": confidence_list[i] if i < len(confidence_list) else 0.0,
                    "trace": traces[i],
                }
                for i in range(len(answers))
            ],
        }
        trace_log["split"] = "train"
        self.train_trace_history_logs.append(trace_log)
        self.reasoning_summary_history_logs.append(
            self._build_reasoning_summary_history_record(
                split="train",
                epoch_id=epoch_id,
                step_id=step_id,
                question=question,
                family_metrics=family_metrics,
                traces=traces,
            )
        )
        if len(self.train_step_logs) >= 100:
            self.flush_train_step_logs()
        if len(self.train_trace_history_logs) >= 20:
            self.flush_train_trace_history_logs()
        if len(self.reasoning_summary_history_logs) >= 20:
            self.flush_reasoning_summary_history_logs()
        return {
            "traces": traces,
            "answers": answers,
            "update_ready": ready_for_update,
            "update_summary": update_summary,
            **reward_pack,
        }

    def flush_update_logs(self):
        if not self.update_logs:
            return
        path = os.path.join(self.cfg.out_dir, "update_logs.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for record in self.update_logs:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.update_logs = []

    def flush_train_step_logs(self):
        if not self.train_step_logs:
            return
        path = os.path.join(self.cfg.out_dir, "train_step_logs.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for record in self.train_step_logs:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.train_step_logs = []

    def _flush_trace_history_records(self, file_name: str, records: List[Dict[str, Any]]):
        if not records:
            return
        path = os.path.join(self.cfg.out_dir, file_name)
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush_train_trace_history_logs(self):
        self._flush_trace_history_records("train_trace_history.jsonl", self.train_trace_history_logs)
        self.train_trace_history_logs = []

    def flush_test_trace_history_logs(self):
        self._flush_trace_history_records("test_trace_history.jsonl", self.test_trace_history_logs)
        self.test_trace_history_logs = []

    def flush_reasoning_summary_history_logs(self):
        self._flush_trace_history_records("reasoning_summary_history.jsonl", self.reasoning_summary_history_logs)
        self.reasoning_summary_history_logs = []

    def flush_trace_history_logs(self):
        # Backward-compatibility shim for old call sites.
        self.flush_train_trace_history_logs()
        self.flush_test_trace_history_logs()
        self.flush_reasoning_summary_history_logs()

    def agent_to_dict(self, a: AgentState) -> Dict[str, Any]:
        return {
            "initial_prompt": a.initial_prompt,
            "initial_prompt_hash": self._prompt_hash(a.initial_prompt),
            "current_prompt": a.current_prompt,
            "current_prompt_hash": self._prompt_hash(a.current_prompt),
            "history": a.history,
            "gradient_history": a.gradient_history[-50:],
            "bandit": a.bandit.to_dict(),
            "homogeneity_window": a.homogeneity_window,
            "recent_homogeneity_flags": list(a.recent_homogeneity_flags),
            "homogeneity_count": a.homogeneity_count,
            "accept_count": a.accept_count,
            "reject_count": a.reject_count,
            "last_update_record": a.last_update_record,
        }

    def save_state(self, name: str, extra: Optional[Dict[str, Any]] = None):
        payload = {
            "config": asdict(self.cfg),
            "agents": [self.agent_to_dict(a) for a in self.agents],
            "history": self.history,
            "prompt_history": self.prompt_history,
        }
        if extra:
            payload["extra"] = extra
        path = os.path.join(self.cfg.out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    async def evaluate_dataset(self, data: List[Dict[str, str]], split_name: str = "test") -> Dict[str, Any]:
        family_homogeneity_rate_list = []
        family_diversity_list = []
        direct_diversity_list = []
        vote_correct_list = []
        details_path = os.path.join(self.cfg.out_dir, f"{split_name}_predictions.jsonl")

        async def eval_one(step_id: int, ex: Dict[str, str]) -> Dict[str, Any]:
            q = ex["question"]
            gold = parse_gold(ex["answer"], self.cfg.task_type, question=q)
            traces, answers, family_labels, family_judgments = await self.solve_with_current_prompts_with_family(q)
            reward_pack = self.compute_rewards(
                traces,
                answers,
                gold,
                primary_family_labels=family_labels,
                family_judgments=family_judgments,
            )
            family_metrics = reward_pack.get("family_metrics", {}) if isinstance(reward_pack.get("family_metrics", {}), dict) else {}
            strategy_steps = family_metrics.get("strategy_steps", [])
            distinctive_features = family_metrics.get("distinctive_features", [])
            evidence_spans = family_metrics.get("evidence_spans", [])
            confidences = family_metrics.get("family_confidences", [])
            record = {
                **self._base_log_fields(),
                "question_hash": self._prompt_hash(q) if q else "",
                "answers": answers,
                "vote_answer": reward_pack["vote_answer"],
                "vote_correct": reward_pack["vote_correct"],
                "llm_direct_diversity_score": reward_pack.get("llm_direct_diversity_score"),
                "llm_direct_diversity_reason": reward_pack.get("llm_direct_diversity_reason", ""),
                "primary_family_labels": family_metrics.get("primary_families", family_labels),
                "secondary_family_labels": family_metrics.get("secondary_families", family_labels),
                "reasoning_summaries": family_metrics.get("reasoning_summaries", []),
                "strategy_steps": strategy_steps,
                "distinctive_features": distinctive_features,
                "evidence_spans": evidence_spans,
                "family_confidences": confidences,
                "agent_family_distributions": family_metrics.get("agent_family_distributions", []),
                "family_judgments": family_judgments,
                "primary_family_counts": family_metrics.get("primary_family_counts", {}),
                "weighted_family_distribution": family_metrics.get("weighted_family_distribution", {}),
                "major_family_distribution": family_metrics.get("major_family_distribution", {}),
                "team_family_homogeneity_rate": family_metrics.get("team_family_homogeneity_rate", 0.0),
                "team_family_diversity": family_metrics.get("team_family_diversity", 0.0),
                "team_major_family_diversity": family_metrics.get("team_major_family_diversity", 0.0),
                "team_intra_family_diversity": family_metrics.get("team_intra_family_diversity", 0.0),
                "family_judge_metric": family_metrics.get("family_judge_metric", "unknown"),
                "all_same_primary": bool(family_metrics.get("all_same_primary", False)),
                "all_same_pair": bool(family_metrics.get("all_same_pair", False)),
                "primary_dominant_share": family_metrics.get("primary_dominant_share", 0.0),
                "pair_dominant_share": family_metrics.get("pair_dominant_share", 0.0),
                "mean_family_confidence": family_metrics.get("mean_family_confidence", 0.0),
                "low_confidence_share": family_metrics.get("low_confidence_share", 0.0),
                "rejudge_count": family_metrics.get("rejudge_count", 0),
                "mean_summary_words": family_metrics.get("mean_summary_words", 0.0),
                "mean_summary_tokens": family_metrics.get("mean_summary_tokens", 0.0),
                "mean_evidence_spans": family_metrics.get("mean_evidence_spans", 0.0),
                "gold": gold,
            }
            trace_record = {
                "epoch": 0,
                "step": step_id,
                "split": "test",
                "question_hash": self._prompt_hash(normalize_spaces(q)) if q else "",
                "agents": [
                    {
                        "agent_id": i,
                        "primary_family": family_metrics.get("primary_families", [])[i] if i < len(family_metrics.get("primary_families", [])) else "",
                        "secondary_family": family_metrics.get("secondary_families", [])[i] if i < len(family_metrics.get("secondary_families", [])) else "",
                        "family_distribution": family_metrics.get("agent_family_distributions", [])[i] if i < len(family_metrics.get("agent_family_distributions", [])) else {},
                        "reasoning_summary": family_metrics.get("reasoning_summaries", [])[i] if i < len(family_metrics.get("reasoning_summaries", [])) else "",
                        "strategy_steps": strategy_steps[i] if i < len(strategy_steps) else [],
                        "distinctive_features": distinctive_features[i] if i < len(distinctive_features) else [],
                        "evidence_spans": evidence_spans[i] if i < len(evidence_spans) else [],
                        "confidence": confidences[i] if i < len(confidences) else 0.0,
                        "trace": traces[i],
                    }
                    for i in range(len(answers))
                ],
            }
            summary_record = self._build_reasoning_summary_history_record(
                split=split_name,
                epoch_id=0,
                step_id=step_id,
                question=q,
                family_metrics=family_metrics,
                traces=traces,
            )
            return {
                "record": record,
                "trace_record": trace_record,
                "summary_record": summary_record,
                "family_homogeneity": float(family_metrics.get("team_family_homogeneity_rate", 0.0)),
                "family_diversity": float(family_metrics.get("team_family_diversity", 0.0)),
                "direct_diversity": reward_pack.get("llm_direct_diversity_score"),
                "vote_correct": reward_pack["vote_correct"],
            }

        results_raw = await asyncio.gather(
            *[eval_one(step_id, ex) for step_id, ex in enumerate(data, start=1)],
            return_exceptions=True,
        )
        results = [r for r in results_raw if isinstance(r, dict)]

        with open(details_path, "w", encoding="utf-8") as fw:
            for result in results:
                family_homogeneity_rate_list.append(float(result.get("family_homogeneity", 0.0)))
                family_diversity_list.append(float(result.get("family_diversity", 0.0)))
                if result.get("direct_diversity") is not None:
                    direct_diversity_list.append(float(result.get("direct_diversity", 0.0)))
                vote_correct_list.append(int(result.get("vote_correct", 0)))
                fw.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                self.test_trace_history_logs.append(result["trace_record"])
                self.reasoning_summary_history_logs.append(result["summary_record"])

                if len(self.test_trace_history_logs) >= 20:
                    self.flush_test_trace_history_logs()
                if len(self.reasoning_summary_history_logs) >= 20:
                    self.flush_reasoning_summary_history_logs()

        vote_acc = float(np.mean(vote_correct_list)) if vote_correct_list else 0.0
        metrics = {
            "mean_family_homogeneity_rate": float(np.mean(family_homogeneity_rate_list)) if family_homogeneity_rate_list else 0.0,
            "mean_family_diversity": float(np.mean(family_diversity_list)) if family_diversity_list else 0.0,
            "mean_llm_direct_diversity_score": float(np.mean(direct_diversity_list)) if direct_diversity_list else 0.0,
            "vote_acc": vote_acc,
            "size": len(data),
        }
        return metrics
