from scripts.audit_v18_critic_safety_only_counterfactual import (
    AMBIGUOUS,
    ANTI_CHEATING,
    OUTPUT_CONTRACT,
    SCHEMA_OR_FORMAT,
    SEMANTIC_QUALITY_ONLY,
    classify_rejection,
)


def row(checks, **overrides):
    value = {
        "schema_valid": True,
        "response_truncated": False,
        "parse_error": "",
        "failed_checks": checks,
    }
    value.update(overrides)
    return value


def test_structural_categories_precede_semantic_mapping():
    assert classify_rejection(row(["actionable_specificity"], schema_valid=False), None)[0] == SCHEMA_OR_FORMAT
    assert classify_rejection(row(["shortcut_or_copying"]), None)[0] == ANTI_CHEATING
    assert classify_rejection(row(["evidence_mismatch"]), None)[0] == SEMANTIC_QUALITY_ONLY


def test_preservation_or_output_is_split_only_by_direct_plan_evidence():
    check = row(["preservation_or_output_risk"])
    output = {"repair_rule": "Return only the selected answer label."}
    preservation = {"preservation_rule": "Preserve previously correct answers."}
    mixed = {**output, **preservation}
    neither = {"repair_rule": "Use a clearer decision rule."}

    assert classify_rejection(check, output)[0] == OUTPUT_CONTRACT
    assert classify_rejection(check, preservation)[0] == SEMANTIC_QUALITY_ONLY
    assert classify_rejection(check, mixed)[0] == AMBIGUOUS
    assert classify_rejection(check, neither)[0] == AMBIGUOUS
