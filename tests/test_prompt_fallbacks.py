from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def test_redundant_distinct_procedure_append_is_rejected():
    system = object.__new__(TraceBeamSearchSystem)
    parent = "You are a careful reasoning solver."
    stock = TraceBeamSearchSystem.GENERIC_DISTINCT_PROCEDURE

    assert system._is_redundant_candidate_prompt(parent, parent + " " + stock)
    assert system._is_redundant_candidate_prompt(parent, stock + " " + stock)


def test_structured_fallback_role_is_not_parent_append():
    system = object.__new__(TraceBeamSearchSystem)
    parent = "You are a careful reasoning solver."

    fallback = system._structured_fallback_role(agent_id=0, index=0)
    prompt = fallback["candidate_prompt"]

    assert not prompt.startswith(parent)
    assert "Use a distinct decision procedure" not in prompt
    assert not system._is_redundant_candidate_prompt(parent, prompt)
    assert fallback["role_name"]
    assert fallback["decision_procedure"]
