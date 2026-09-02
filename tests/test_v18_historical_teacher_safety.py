from scripts.analyze_v18_historical_teacher_safety import classify_field


def test_field_classifier_separates_output_and_peer_copying():
    output = classify_field("Keep strict adherence to the output contract and final answer format.")
    assert output["output_contract"] is True
    assert output["anti_cheating"] is False
    assert "preservation_rule" not in output

    copying = classify_field("Copy the peer prompt procedure exactly.")
    assert copying["anti_cheating"] is True
    assert "peer_procedure_copying" in copying["anti_subtypes"]


def test_field_classifier_leaves_reasoning_only_rule_safe():
    result = classify_field("Compare candidate antecedents using syntax and semantics.")
    assert result["unsafe"] is False
    assert result["output_subtypes"] == []
    assert result["anti_subtypes"] == []
