import json
import math
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl(path: str, limit: int = -1) -> List[Dict[str, Any]]:
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
        "representation_formalization": [
            "decomposition",
            "symbolic_formulation",
            "spatial_visualization",
            "dimensional_unit_analysis",
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
            "option_elimination",
            "backward_reasoning",
            "consistency_verification",
            "counterexample_search",
            "proof_by_contradiction",
            "invariant_reasoning",
            "symmetry_reasoning",
            "definition_application",
            "rule_based_classification",
            "theorem_property_application",
        ],
        "probability_statistics": [
            "probabilistic_reasoning",
            "expected_value_reasoning",
        ],
        "induction_pattern": [
            "pattern_generalization",
            "inductive_reasoning",
            "analogy_mapping",
            "comparative_reasoning",
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


def infer_strategy_family_major(family_label: str) -> str:
    major_map = strategy_family_to_major_map()
    label = normalize_spaces(str(family_label or "")).lower().replace("-", "_").replace(" ", "_")
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
    raw = normalize_spaces(str(label or "")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if not raw:
        if allowed_labels:
            return normalize_spaces(str(allowed_labels[0])).lower().replace("-", "_").replace(" ", "_")
        return "decomposition"

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
    score = 0.0
    for family_a, weight_a in dist_a.items():
        for family_b, weight_b in dist_b.items():
            score += float(weight_a) * float(weight_b) * _family_pair_similarity(
                family_a,
                family_b,
                major_map,
                same_major_weight,
            )
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

    for cand in candidates:
        try:
            return json.loads(cand)
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
    try:
        v = float(x.replace(",", "").strip())
        if abs(v - int(v)) < 1e-9:
            return str(int(v))
        return ("%f" % v).rstrip("0").rstrip(".")
    except Exception:
        return x.strip()


def parse_gsm8k_gold(answer: str) -> str:
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer.replace(",", ""))
    if m:
        return canonical_number_str(m.group(1))
    nums = extract_all_numbers(answer)
    if nums:
        return canonical_number_str(nums[-1])
    return normalize_spaces(answer)


def parse_mmlu_gold(answer: str) -> str:
    s = normalize_spaces(str(answer)).upper()
    if s in {"A", "B", "C", "D"}:
        return s
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([0-3])\b", s)
    if m:
        return ["A", "B", "C", "D"][int(m.group(1))]
    return s


def infer_task_type(task_type: str = "auto", question: Optional[str] = None, answer: Optional[str] = None) -> str:
    declared = str(task_type).strip().lower()
    if declared in {"gsm8k", "mmlu"}:
        return declared

    q = normalize_spaces(str(question or "")).upper()
    a = normalize_spaces(str(answer or "")).upper()

    if a in {"A", "B", "C", "D"}:
        return "mmlu"
    if re.search(r"\bOPTIONS\b", q) and re.search(r"\bA\.|\bB\.|\bC\.|\bD\.", q):
        return "mmlu"
    return "gsm8k"


def parse_gold(answer: str, task_type: str = "auto", question: Optional[str] = None) -> str:
    task = infer_task_type(task_type=task_type, question=question, answer=answer)
    if task == "mmlu":
        return parse_mmlu_gold(answer)
    return parse_gsm8k_gold(answer)


def extract_pred_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    patterns = [
        r"FINAL_ANSWER\s*:\s*([-+]?\d+(?:\.\d+)?)",
        r"Answer\s*:\s*([-+]?\d+(?:\.\d+)?)",
        r"The answer is\s*([-+]?\d+(?:\.\d+)?)",
    ]
    raw = text.replace(",", "")
    for p in patterns:
        m = re.search(p, raw, flags=re.IGNORECASE)
        if m:
            return canonical_number_str(m.group(1))
    nums = extract_all_numbers(raw)
    if nums:
        return canonical_number_str(nums[-1])
    return normalize_spaces(text)


def extract_pred_answer_mmlu(text: Optional[str]) -> str:
    if text is None:
        return ""
    raw = str(text)
    patterns = [
        r"FINAL_ANSWER\s*:\s*([ABCD])\b",
        r"Answer\s*:\s*([ABCD])\b",
        r"The answer is\s*([ABCD])\b",
        r"\boption\s*([ABCD])\b",
    ]
    for p in patterns:
        m = re.search(p, raw, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # fallback: prefer the last standalone option token
    toks = re.findall(r"\b([ABCD])\b", raw.upper())
    if toks:
        return toks[-1]
    return normalize_spaces(raw).upper()


def extract_pred_answer_by_task(
    text: Optional[str],
    task_type: str = "auto",
    question: Optional[str] = None,
) -> str:
    task = infer_task_type(task_type=task_type, question=question, answer=None)
    if task == "mmlu":
        return extract_pred_answer_mmlu(text)
    return extract_pred_answer(text)


def majority_vote(answers: List[str]) -> str:
    cleaned = [a for a in answers if str(a).strip() != ""]
    if not cleaned:
        return ""
    cnt = Counter(cleaned)
    best_count = max(cnt.values())
    cands = [k for k, v in cnt.items() if v == best_count]
    if len(cands) == 1:
        return cands[0]
    for a in cleaned:
        if a in cands:
            return a
    return cands[0]




