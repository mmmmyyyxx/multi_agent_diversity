from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample, PromptAnswer
from multi_dataset_diverse_rl.tasks import get_task_spec, normalize_bbh_answer
from multi_dataset_diverse_rl.team_differentiation import team_behavior_metrics, vote_transition_decomposition


def profiles(answers):
    return [(PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True),) for answer in answers]


if __name__ == "__main__":
    spec = get_task_spec("bbh")
    examples = (ProbeExample("q", "q", "A"),)
    old = profiles(["A", "A", "B", "B", "B"])
    new = profiles(["A", "A", "B", "C", "D"])
    metrics = team_behavior_metrics(
        examples=examples, profiles=new, normalize_answer=normalize_bbh_answer,
        match_answer=spec.match_answer, tie_break="abstain", seed=42,
    )
    transition = vote_transition_decomposition(
        examples=examples, incumbent_profiles=old, candidate_profiles=new,
        normalize_answer=normalize_bbh_answer, match_answer=spec.match_answer,
        tie_break="abstain", seed=42,
    )
    assert metrics["mean_H"] == 1
    assert transition["vote_gain_source_counts"] == {"concentration_driven": 1}
    print("deterministic team differentiation metrics smoke: PASS")
