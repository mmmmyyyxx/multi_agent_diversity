from multi_dataset_diverse_rl.evaluation.fixed_probe import (
    ProbeExample,
    PromptAnswer,
    evaluate_candidate_profile,
)


def answer(value):
    return PromptAnswer(value, f"FINAL_ANSWER: {value}", True)


def test_unique_and_pivotal_gains_and_losses_are_audited_symmetrically():
    examples = tuple(
        ProbeExample(f"q{index}", f"h{index}", "A")
        for index in range(4)
    )
    active_answers = (
        ("B", "B", "A", "A"),
        ("B", "A", "B", "A"),
        ("B", "A", "B", "A"),
        ("B", "B", "B", "B"),
        ("B", "B", "B", "B"),
    )
    active_profiles = [
        tuple(answer(value) for value in values)
        for values in active_answers
    ]
    candidate_profile = tuple(
        answer(value) for value in ("A", "A", "B", "B")
    )
    evaluation = evaluate_candidate_profile(
        prompt="candidate",
        prompt_hash="candidate-hash",
        examples=examples,
        active_profiles=active_profiles,
        initial_profiles=active_profiles,
        candidate_profile=candidate_profile,
        target_agent_id=0,
        assigned_question_hashes=set(),
        normalize_answer=lambda value: value.strip().lower(),
        match_answer=lambda prediction, gold: (
            prediction.lower() == gold.lower()
        ),
        tie_break="abstain",
        seed=42,
        tau=1.0,
    )
    assert evaluation.protection.unique_correct_gain_count == 1
    assert evaluation.protection.unique_correct_loss_count == 1
    assert evaluation.protection.pivotal_correct_gain_count == 1
    assert evaluation.protection.pivotal_correct_loss_count == 1
