import json
import math
import os
import csv
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .tasks import (
    canonical_number_str as task_canonical_number_str,
    extract_pred_answer_bbh,
    extract_pred_answer_gsm8k,
    extract_pred_answer_mmlu as task_extract_pred_answer_mmlu,
    get_task_spec,
    infer_task_type as task_infer_task_type,
    match_bbh_answer,
    match_gsm8k_answer,
    match_mmlu_answer,
    normalize_bbh_answer,
    parse_gold_bbh,
    parse_gsm8k_gold as task_parse_gsm8k_gold,
    parse_mmlu_gold as task_parse_mmlu_gold,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl(path: str, limit: int = -1) -> List[Dict[str, Any]]:
    if os.path.splitext(str(path))[1].lower() == ".csv":
        data = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
                if limit > 0 and len(data) >= limit:
                    break
        return data

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if limit > 0 and len(data) >= limit:
                break
    return data


def extract_question_answer(ex: Dict[str, Any]) -> Tuple[str, str]:
    q_keys = ["question", "input", "problem", "query"]
    a_keys = ["answer", "output", "target", "label", "response"]

    q = None
    a = None
    for k in q_keys:
        if k in ex and ex[k] is not None:
            q = str(ex[k])
            break
    for k in a_keys:
        if k in ex and ex[k] is not None:
            a = str(ex[k])
            break
    if q is None or a is None:
        raise ValueError(f"Cannot find question/answer fields in record: {ex}")
    return q, a


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _string_similarity(s1: str, s2: str) -> float:
    """Compute simple character-level similarity between two strings (0.0 to 1.0)."""
    s1 = s1.lower().replace("-", "_").replace(" ", "_")
    s2 = s2.lower().replace("-", "_").replace(" ", "_")
    
    # Exact match
    if s1 == s2:
        return 1.0
    
    # Substring match
    if s1 in s2 or s2 in s1:
        return 0.9
    
    # Compute Jaccard similarity on character trigrams
    def get_trigrams(s):
        return set([s[i:i+3] for i in range(len(s)-2)]) if len(s) >= 3 else set()
    
    t1, t2 = get_trigrams(s1), get_trigrams(s2)
    if not t1 or not t2:
        return 0.0
    
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    return float(intersection / union) if union > 0 else 0.0


def strategy_family_major_categories() -> Dict[str, List[str]]:
    return {
        "mmlu_option_semantics": [
            "concept_definition_match",
            "option_contrast",
            "distractor_elimination",
            "option_contradiction_check",
            "answer_to_stem_backward_check",
            "stem_evidence_alignment",
            "scope_qualifier_analysis",
            "negation_exception_handling",
        ],
        "mmlu_domain_reasoning": [
            "rule_or_principle_application",
            "causal_mechanism_reasoning",
            "historical_context_reasoning",
            "scientific_model_reasoning",
            "quantitative_formula_application",
            "classification_taxonomy_reasoning",
            "example_counterexample_reasoning",
            "statistical_method_reasoning",
            "psychodynamic_analysis",
            "convergent_evolution_analysis",
            "cultural_context_reasoning",
            "buffer_capacity_analysis",
            "ethical_reasoning",
            "countertransference",
        ],
        "representation_formalization": [
            "decomposition",
            "symbolic_formulation",
            "spatial_visualization",
            "dimensional_unit_analysis",
            "assumption_analysis",
        ],
        "algebra_computation": [
            "algebraic_derivation",
            "equation_solving",
            "direct_computation",
            "combinatorial_counting",
        ],
        "logical_proof": [
            "case_analysis",
            "exhaustive_enumeration",
            "constraint_propagation",
            "consistency_verification",
            "counterexample_search",
            "proof_by_contradiction",
            "invariant_reasoning",
            "symmetry_reasoning",
        ],
        "probability_statistics": [
            "probabilistic_reasoning",
            "expected_value_reasoning",
        ],
        "induction_pattern": [
            "pattern_generalization",
            "inductive_reasoning",
            "analogy_mapping",
        ],
        "process_structure_simulation": [
            "simulation_tracing",
            "recursive_reasoning",
            "temporal_sequential_reasoning",
            "causal_reasoning",
        ],
        "optimization_boundary_meta": [
            "optimization_extremal_reasoning",
            "approximation_bounding",
            "edge_case_analysis",
            "abductive_inference",
            "counterfactual_reasoning",
        ],
    }


def strategy_family_to_major_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for major, families in strategy_family_major_categories().items():
        for family in families:
            mapping[family] = major
    return mapping


def _normalize_strategy_family_alias(label: Optional[str], *, mmlu: bool = True) -> str:
    raw = normalize_spaces(str(label or "")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    raw = re.sub(r"[^a-z0-9_]+", "", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        return ""

    aliases = {
        "other": "decomposition",
        "algebra": "algebraic_derivation",
        "algebraic_reasoning": "algebraic_derivation",
        "symbolic_reasoning": "symbolic_formulation",
        "equation_reasoning": "equation_solving",
        "contradiction": "proof_by_contradiction",
        "contradiction_proof": "proof_by_contradiction",
        "verification": "consistency_verification",
        "verify": "consistency_verification",
        "check": "consistency_verification",
        "computation": "direct_computation",
        "arithmetic": "direct_computation",
        "elimination_comparison": "distractor_elimination",
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
        "edge_cases": "edge_case_analysis",
        "unit_analysis": "dimensional_unit_analysis",
        "extremal_reasoning": "optimization_extremal_reasoning",
        "simulation": "simulation_tracing",
        "recursion": "recursive_reasoning",
        "abduction": "abductive_inference",
        "counterfactual": "counterfactual_reasoning",
    }
    if mmlu:
        aliases.update(
            {
                "concept_recall": "concept_definition_match",
                "concept_match": "concept_definition_match",
                "definition_lookup": "concept_definition_match",
                "definition_application": "concept_definition_match",
                "terminology_match": "concept_definition_match",
                "option_comparison": "option_contrast",
                "comparison": "option_contrast",
                "comparative_reasoning": "option_contrast",
                "elimination": "distractor_elimination",
                "option_elimination": "distractor_elimination",
                "contradiction_check": "option_contradiction_check",
                "contradiction": "option_contradiction_check",
                "backward_verification": "answer_to_stem_backward_check",
                "backward_checking": "answer_to_stem_backward_check",
                "backward_reasoning": "answer_to_stem_backward_check",
                "property_application": "rule_or_principle_application",
                "theorem_property_application": "rule_or_principle_application",
                "rule_matching": "rule_or_principle_application",
                "rule_based_classification": "rule_or_principle_application",
            }
        )
    return aliases.get(raw, raw)


def infer_strategy_family_major(family_label: str) -> str:
    major_map = strategy_family_to_major_map()
    label = _normalize_strategy_family_alias(family_label, mmlu=True)
    if label in major_map:
        return major_map[label]
    if not label:
        return "representation_formalization"
    best_family = max(major_map, key=lambda cand: _string_similarity(label, cand))
    return major_map[best_family]


def normalize_strategy_family_label(
    label: Optional[str],
    allowed_labels: Optional[List[str]] = None,
    allow_fallback: bool = True,
) -> str:
    """
    Normalize a strategy family label.

    - If allowed_labels is None, returns the normalized label (aliases applied).
    - If allowed_labels is provided, normalizes to that set and optionally falls back
      to the most similar allowed label when unknown.
    """
    raw = _normalize_strategy_family_alias(label, mmlu=False)
    if not raw:
        if allowed_labels:
            return normalize_spaces(str(allowed_labels[0])).lower().replace("-", "_").replace(" ", "_")
        return "decomposition"

    if allowed_labels is not None:
        allowed_probe = {
            normalize_spaces(str(x)).lower().replace("-", "_").replace(" ", "_")
            for x in allowed_labels
        }
        if allowed_probe.intersection(
            {
                "concept_definition_match",
                "option_contrast",
                "distractor_elimination",
                "option_contradiction_check",
                "answer_to_stem_backward_check",
                "rule_or_principle_application",
            }
        ):
            raw = _normalize_strategy_family_alias(label, mmlu=True)

    if allowed_labels is None:
        return raw

    allowed = {normalize_spaces(str(x)).lower().replace("-", "_").replace(" ", "_") for x in allowed_labels}
    if raw in allowed:
        return raw

    if allow_fallback and allowed:
        candidates = list(allowed)
        best_match = max(candidates, key=lambda cand: _string_similarity(raw, cand))
        return best_match

    if allowed:
        return sorted(allowed)[0]
    return raw or "decomposition"


def normalize_strategy_family_pair(
    primary_family: Optional[str],
    secondary_family: Optional[str],
    allowed_labels: Optional[List[str]] = None,
    allow_fallback: bool = True,
) -> Tuple[str, str]:
    primary = normalize_strategy_family_label(
        primary_family,
        allowed_labels=allowed_labels,
        allow_fallback=allow_fallback,
    )
    secondary_raw = secondary_family if secondary_family not in (None, "") else primary
    secondary = normalize_strategy_family_label(
        secondary_raw,
        allowed_labels=allowed_labels,
        allow_fallback=allow_fallback,
    )
    return primary, secondary


def build_strategy_family_distributions(
    primary_families: List[str],
    secondary_families: Optional[List[str]] = None,
    *,
    allowed_labels: Optional[List[str]] = None,
    use_dual_family: bool = False,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
    allow_fallback: bool = True,
) -> Tuple[List[str], List[str], List[Dict[str, float]]]:
    if not use_dual_family:
        normalized = [
            normalize_strategy_family_label(f, allowed_labels=allowed_labels, allow_fallback=allow_fallback)
            for f in primary_families
        ]
        return normalized, list(normalized), [{fam: 1.0} for fam in normalized]

    primary_weight = float(max(0.0, min(1.0, primary_weight)))
    secondary_weight = float(max(0.0, secondary_weight))
    total = primary_weight + secondary_weight
    if total <= 0.0:
        primary_weight, secondary_weight = 0.7, 0.3
        total = 1.0
    primary_weight = primary_weight / total
    secondary_weight = secondary_weight / total

    secondary_families = secondary_families or []
    primary_out: List[str] = []
    secondary_out: List[str] = []
    distributions: List[Dict[str, float]] = []

    for i, primary_raw in enumerate(primary_families):
        secondary_raw = secondary_families[i] if i < len(secondary_families) else primary_raw
        primary, secondary = normalize_strategy_family_pair(
            primary_raw,
            secondary_raw,
            allowed_labels=allowed_labels,
            allow_fallback=allow_fallback,
        )
        if primary == secondary:
            dist = {primary: 1.0}
        else:
            dist = {primary: primary_weight, secondary: secondary_weight}
        primary_out.append(primary)
        secondary_out.append(secondary)
        distributions.append(dist)

    return primary_out, secondary_out, distributions


def _family_pair_similarity(
    family_a: str,
    family_b: str,
    major_map: Dict[str, str],
    same_major_weight: float,
) -> float:
    if family_a == family_b:
        return 1.0
    major_a = major_map.get(family_a, "")
    major_b = major_map.get(family_b, "")
    if major_a and major_a == major_b:
        return float(max(0.0, min(1.0, same_major_weight)))
    return 0.0


def _distribution_similarity(
    dist_a: Dict[str, float],
    dist_b: Dict[str, float],
    major_map: Dict[str, str],
    same_major_weight: float,
) -> float:
    def kernel_score(left: Dict[str, float], right: Dict[str, float]) -> float:
        score = 0.0
        for family_a, weight_a in left.items():
            for family_b, weight_b in right.items():
                score += float(weight_a) * float(weight_b) * _family_pair_similarity(
                    family_a,
                    family_b,
                    major_map,
                    same_major_weight,
                )
        return float(max(0.0, score))

    cross_score = kernel_score(dist_a, dist_b)
    self_a = kernel_score(dist_a, dist_a)
    self_b = kernel_score(dist_b, dist_b)
    denom = math.sqrt(self_a * self_b) if self_a > 0.0 and self_b > 0.0 else 0.0
    if denom <= 0.0:
        return 0.0
    score = cross_score / denom
    return float(max(0.0, min(1.0, score)))


def compute_strategy_family_profile_metrics(
    primary_families: List[str],
    secondary_families: Optional[List[str]] = None,
    *,
    allowed_labels: Optional[List[str]] = None,
    use_dual_family: bool = False,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
    same_major_weight: float = 0.5,
    macro_diversity_weight: float = 0.5,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    normalized_primary, normalized_secondary, agent_distributions = build_strategy_family_distributions(
        primary_families,
        secondary_families,
        allowed_labels=allowed_labels,
        use_dual_family=use_dual_family,
        primary_weight=primary_weight,
        secondary_weight=secondary_weight,
        allow_fallback=allow_fallback,
    )
    n = len(agent_distributions)
    if n == 0:
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
            "use_dual_family": bool(use_dual_family),
        }

    major_map = strategy_family_to_major_map()
    same_major_weight = float(max(0.0, min(1.0, same_major_weight)))
    effective_same_major_weight = same_major_weight if use_dual_family else 0.0
    macro_diversity_weight = float(max(0.0, min(1.0, macro_diversity_weight)))

    weighted_counts: Counter = Counter()
    for dist in agent_distributions:
        for family, weight in dist.items():
            weighted_counts[family] += float(weight)
    weighted_distribution = {k: float(v / n) for k, v in weighted_counts.items()}

    major_counts: Counter = Counter()
    for family, weight in weighted_counts.items():
        major = major_map.get(family, infer_strategy_family_major(family))
        major_counts[major] += float(weight)
    major_distribution = {k: float(v / n) for k, v in major_counts.items()}

    def normalized_entropy(distribution: Dict[str, float]) -> float:
        probs = [float(v) for v in distribution.values() if float(v) > 0.0]
        if not probs:
            return 0.0
        entropy = -sum(p * math.log(p) for p in probs)
        if len(probs) > 1:
            entropy = entropy / math.log(len(probs))
        return float(max(0.0, min(1.0, entropy)))

    family_entropy = normalized_entropy(weighted_distribution)
    major_diversity = normalized_entropy(major_distribution)

    intra_terms = []
    for major, major_mass in major_counts.items():
        if major_mass <= 0.0:
            continue
        sub_distribution = {
            family: float(weight / major_mass)
            for family, weight in weighted_counts.items()
            if major_map.get(family, infer_strategy_family_major(family)) == major and weight > 0.0
        }
        intra_terms.append(float(major_mass / n) * normalized_entropy(sub_distribution))
    intra_diversity = float(sum(intra_terms))
    hierarchical_diversity = float(
        max(0.0, min(1.0, macro_diversity_weight * major_diversity + (1.0 - macro_diversity_weight) * intra_diversity))
    )

    pairwise_sims: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    total_pair_similarity = 0.0
    total_pairs = n * (n - 1) / 2
    for i in range(n):
        for j in range(i + 1, n):
            sim = _distribution_similarity(agent_distributions[i], agent_distributions[j], major_map, effective_same_major_weight)
            pairwise_sims[i][j] = sim
            pairwise_sims[j][i] = sim
            total_pair_similarity += sim

    team_family_homogeneity_rate = float(total_pair_similarity / total_pairs) if total_pairs > 0 else 0.0
    per_agent_same_family_ratio = [
        float(sum(pairwise_sims[i]) / (n - 1)) if n > 1 else 0.0
        for i in range(n)
    ]
    per_agent_same_family_count = [
        int(round(ratio * max(0, n - 1)))
        for ratio in per_agent_same_family_ratio
    ]
    per_agent_family_diversity = [float(1.0 - x) for x in per_agent_same_family_ratio]

    primary_counts = Counter(normalized_primary)
    dominant_family_share = float(max(weighted_distribution.values())) if weighted_distribution else 0.0
    dominant_major_family_share = float(max(major_distribution.values())) if major_distribution else 0.0

    return {
        "primary_families": normalized_primary,
        "secondary_families": normalized_secondary,
        "agent_family_distributions": agent_distributions,
        "family_major_categories": {family: major_map.get(family, infer_strategy_family_major(family)) for family in weighted_distribution},
        "primary_family_counts": dict(primary_counts),
        "weighted_family_distribution": weighted_distribution,
        "major_family_distribution": major_distribution,
        "per_agent_same_family_count": per_agent_same_family_count,
        "per_agent_same_family_ratio": per_agent_same_family_ratio,
        "per_agent_family_diversity": per_agent_family_diversity,
        "team_family_homogeneity_rate": team_family_homogeneity_rate,
        "team_family_diversity": hierarchical_diversity if use_dual_family else family_entropy,
        "team_family_entropy": family_entropy,
        "team_major_family_diversity": major_diversity,
        "team_intra_family_diversity": intra_diversity,
        "dominant_family_share": dominant_family_share,
        "dominant_major_family_share": dominant_major_family_share,
        "use_dual_family": bool(use_dual_family),
        "primary_family_weight": float(primary_weight),
        "secondary_family_weight": float(secondary_weight),
        "same_major_family_weight": effective_same_major_weight,
        "macro_diversity_weight": macro_diversity_weight,
    }


def compute_strategy_family_metrics(
    families: List[str],
    allowed_labels: Optional[List[str]] = None,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    return compute_strategy_family_profile_metrics(
        families,
        allowed_labels=allowed_labels,
        use_dual_family=False,
        allow_fallback=allow_fallback,
    )


def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    candidates = []

    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)

    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1))

    def _escape_invalid_json_backslashes(value: str) -> str:
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", value)

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            try:
                return json.loads(_escape_invalid_json_backslashes(cand))
            except Exception:
                continue
    return None


def extract_all_numbers(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums


def canonical_number_str(x: str) -> str:
    return task_canonical_number_str(x)


def parse_gsm8k_gold(answer: str) -> str:
    return task_parse_gsm8k_gold(answer)


def parse_mmlu_gold(answer: str) -> str:
    return task_parse_mmlu_gold(answer)


def infer_task_type(task_type: str = "auto", question: Optional[str] = None, answer: Optional[str] = None) -> str:
    return task_infer_task_type(task_type=task_type, question=question, answer=answer)


def parse_gold(answer: str, task_type: str = "auto", question: Optional[str] = None) -> str:
    return get_task_spec(task_type).parse_gold(answer, question)


def extract_pred_answer(text: Optional[str]) -> str:
    return extract_pred_answer_gsm8k(text)


def extract_pred_answer_mmlu(text: Optional[str]) -> str:
    return task_extract_pred_answer_mmlu(text)


def extract_pred_answer_by_task(
    text: Optional[str],
    task_type: str = "auto",
    question: Optional[str] = None,
) -> str:
    return get_task_spec(task_type).extract_pred(text, question)


def match_answer_by_task(pred: str, gold: str, task_type: str = "auto") -> bool:
    return get_task_spec(task_type).match_answer(pred, gold)


def majority_vote_with_diagnostics(
    answers: List[str],
    tie_break_method: str = "first",
    seed: int = 0,
    question_hash: str = "",
) -> Dict[str, Any]:
    cleaned = [str(a) for a in answers if str(a).strip() != ""]
    if not cleaned:
        return {
            "vote_answer": "",
            "vote_tie": False,
            "tie_candidates": [],
            "vote_counts": {},
            "tie_break_method": str(tie_break_method or "first"),
        }
    cnt = Counter(cleaned)
    best_count = max(cnt.values())
    cands = [k for k, v in cnt.items() if v == best_count]
    vote_tie = len(cands) > 1
    method = str(tie_break_method or "first").lower()
    if method not in {"first", "random", "abstain"}:
        method = "first"
    if not vote_tie:
        vote_answer = cands[0]
    elif method == "abstain":
        vote_answer = ""
    elif method == "random":
        rng = random.Random(f"{int(seed)}:{question_hash}:{'|'.join(sorted(cands))}")
        vote_answer = rng.choice(sorted(cands))
    else:
        vote_answer = next((a for a in cleaned if a in cands), cands[0])
    return {
        "vote_answer": vote_answer,
        "vote_tie": bool(vote_tie),
        "tie_candidates": list(cands),
        "vote_counts": dict(cnt),
        "tie_break_method": method,
    }


def majority_vote(answers: List[str]) -> str:
    return str(majority_vote_with_diagnostics(answers, tie_break_method="first").get("vote_answer", ""))




