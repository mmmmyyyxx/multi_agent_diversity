from multi_dataset_diverse_rl.mechanisms import (
    mechanism_distance,
    mechanisms_are_near_duplicate,
    normalize_mechanism_representation,
)


def test_full_prompt_template_is_not_used_as_mechanism_embedding_input():
    prompt = "You are a careful reasoning solver. Verify your answer. Give exactly one final answer."
    rep = normalize_mechanism_representation(prompt, ["Extract constraints", "Hard elimination"])
    assert rep["mechanism_embedding_text"] == "extract_constraints ; hard_elimination"
    assert "careful" not in rep["mechanism_embedding_text"]
    assert "final answer" not in rep["mechanism_embedding_text"]


def test_same_operation_sequence_is_same_niche_and_near_duplicate():
    left = normalize_mechanism_representation("persona A", ["List constraints", "Rule out impossible options"])
    right = normalize_mechanism_representation("persona B", ["Identify constraints", "Discard impossible choices"])
    left["mechanism_embedding"] = [1.0, 0.0]
    right["mechanism_embedding"] = [0.999, 0.001]
    assert mechanism_distance(left, right)["sequence_distance"] == 0.0
    assert mechanisms_are_near_duplicate(left, right)


def test_substantive_mechanism_change_has_larger_distance():
    elimination = normalize_mechanism_representation("", ["Hard elimination"])
    paraphrase = normalize_mechanism_representation("", ["Rule out impossible options"])
    alternative = normalize_mechanism_representation("", ["Weighted scoring", "Counterfactual check"])
    elimination["mechanism_embedding"] = paraphrase["mechanism_embedding"] = [1.0, 0.0]
    alternative["mechanism_embedding"] = [0.0, 1.0]
    assert mechanism_distance(elimination, alternative)["mechanism_distance"] > mechanism_distance(elimination, paraphrase)["mechanism_distance"]
