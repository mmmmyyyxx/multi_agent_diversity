"""Closed deterministic edit concepts shared by current and replay memory."""
import re

# Procedures, not benchmark topics, entities, questions or final answers.
STRATEGIES = {
    "explicit_constraint_check": r"\b(?:check|verify|compare|enforce|track|list|respect)\b[^.!?\n]{0,80}\bconstraints?\b",
    "case_analysis": r"\b(?:enumerate|consider|split|analy[sz]e|check)\b[^.!?\n]{0,80}\bcases?\b",
    "boundary_check": r"\b(?:check|test|verify|consider)\b[^.!?\n]{0,80}\b(?:boundar(?:y|ies)|edge cases?|endpoints?)\b",
    "independent_verification": r"\b(?:verify|check)\b[^.!?\n]{0,80}\b(?:independent(?:ly)?|substitut(?:e|ion)|original equation|consistency)\b",
    "relational_consistency": r"\b(?:check|verify|compare|track|resolve)\b[^.!?\n]{0,80}\b(?:relational|relations?|referents?|semantic roles?)\b",
    "subproblem_decomposition": r"\b(?:decompose|break down|separate)\b[^.!?\n]{0,80}\b(?:steps?|subproblems?|goals?)\b",
    "assumption_check": r"\b(?:check|verify|identify|state|track)\b[^.!?\n]{0,80}\bassumptions?\b",
}


def abstract_actions(text):
    # Do not interpret "never check constraints" as an added positive check.
    # Ambiguous/negated clauses are omitted rather than guessed from keywords.
    clauses = re.split(r"[.!?\n]", text)
    text = ". ".join(c for c in clauses if not re.search(
        r"\b(?:not|never|avoid|skip|omit|without|don't|doesn't|no)\b", c, re.I))
    return frozenset(k for k, expression in STRATEGIES.items()
                     if re.search(expression, text, re.I))
