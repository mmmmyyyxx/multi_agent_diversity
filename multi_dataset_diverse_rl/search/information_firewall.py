"""Recursive evaluation-context firewall shared by public inference boundaries."""
from collections.abc import Mapping

from .schemas import SearchContractError


EVALUATOR_ONLY_KEYS = frozenset({
    "gold", "gold_answer", "gold_titles", "gold_documents", "reward", "correctness",
    "correct_vector", "evaluation_result", "evaluation_feedback", "evaluator_feedback",
    "label", "reference", "reference_answer", "reference_response", "score", "scores",
    "member_scores", "team_scores", "instruction_id_list", "instruction_ids", "kwargs",
    "success_vector", "coverage_success", "supporting_facts", "supporting_documents",
    "pii", "pii_str", "judge_outputs", "num_satisfied", "num_constraints",
})


def require_public_context(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key.casefold() in EVALUATOR_ONLY_KEYS:
                raise SearchContractError("INFERENCE_EVALUATOR_CONTEXT_FORBIDDEN: inference input contains evaluation-only context")
            require_public_context(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            require_public_context(child)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise SearchContractError("INFERENCE_CONTEXT_TYPE_FORBIDDEN")
