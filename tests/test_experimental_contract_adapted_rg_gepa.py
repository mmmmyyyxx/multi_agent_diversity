from __future__ import annotations

import json

import pytest

from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    mutable_prompt_violation_reasons,
)
from multi_dataset_diverse_rl.experimental_contract_adapted_rg_gepa import (
    AVOIDANCE_PRIORITIES,
    AVOID_IDS,
    BEHAVIORAL_CHANGES,
    ContractAdaptedProtocol,
    ContractAdaptedProtocolV2,
    EDIT_IDS,
    EditHypothesis,
    FAILURE_PATTERNS,
    FAILURE_IDS,
    HypothesisSelectionV2,
    PRESERVE_IDS,
    PRESERVATION_PRIORITIES,
    SchemaQualificationV2Protocol,
    build_hypothesis_request_v2,
    hypothesis_interface_v2_identity,
    parse_hypothesis_selection_v2,
    parse_edit_hypothesis,
    render_contract_adapted_prompt,
    render_contract_adapted_prompt_v2,
    renderer_vocabulary_identity,
)


def _hypothesis() -> EditHypothesis:
    return EditHypothesis(
        failure_pattern="ambiguous_referent",
        behavioral_change="trace_entities",
        preserve="global_consistency",
        avoid="unsupported_assumptions",
    )


def test_protocol_is_frozen_and_identified() -> None:
    protocol = ContractAdaptedProtocol()
    assert len(protocol.identity()) == 64
    assert len(renderer_vocabulary_identity()) == 64
    assert protocol.commit_enabled is False
    assert protocol.validation_enabled is False
    assert protocol.test_enabled is False
    assert protocol.memory_enabled is False


def test_parser_accepts_only_exact_closed_vocabulary_json() -> None:
    raw = json.dumps({
        "failure_pattern": "ambiguous_referent",
        "behavioral_change": "trace_entities",
        "preserve": "global_consistency",
        "avoid": "unsupported_assumptions",
    })
    assert parse_edit_hypothesis(raw) == _hypothesis()
    with pytest.raises(ValueError, match="exact JSON"):
        parse_edit_hypothesis(f"```json\n{raw}\n```")
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_edit_hypothesis(raw[:-1] + ', "comment": "free text"}')
    with pytest.raises(ValueError, match="unknown"):
        parse_edit_hypothesis(raw.replace("trace_entities", "rewrite_the_interface"))


def test_renderer_never_copies_reflection_text() -> None:
    parent = "Reason carefully and compare the choices using the stated evidence."
    raw = json.dumps({
        "failure_pattern": "ambiguous_referent",
        "behavioral_change": "trace_entities",
        "preserve": "global_consistency",
        "avoid": "unsupported_assumptions",
    })
    rendered = render_contract_adapted_prompt(parent, parse_edit_hypothesis(raw))
    assert rendered.startswith(parent)
    assert raw not in rendered
    assert not mutable_prompt_violation_reasons(rendered)


def test_every_representable_hypothesis_renders_contract_valid() -> None:
    parent = "Use a careful, evidence-grounded decision procedure."
    count = 0
    for failure in FAILURE_PATTERNS:
        for behavior in BEHAVIORAL_CHANGES:
            for preserve in PRESERVATION_PRIORITIES:
                for avoid in AVOIDANCE_PRIORITIES:
                    prompt = render_contract_adapted_prompt(
                        parent,
                        EditHypothesis(failure, behavior, preserve, avoid),
                    )
                    assert not mutable_prompt_violation_reasons(prompt)
                    count += 1
    assert count == 6 * 6 * 4 * 4


def test_parent_contract_violation_fails_closed() -> None:
    with pytest.raises(ValueError, match="mutable_prompt_contract_violation"):
        render_contract_adapted_prompt("Include a FINAL_ANSWER marker.", _hypothesis())


def test_runner_protocol_comparison_survives_json_round_trip() -> None:
    from scripts import run_contract_adapted_rg_gepa_fixed_parent_pilot as runner

    serialized = json.loads(json.dumps(runner._serialized_protocol()))
    assert serialized == runner._serialized_protocol()
    assert serialized["selection_arms"] == [
        "B0_PRIME_CURRENT",
        "B1_PRIME_TEAM_PARETO",
    ]


def test_v2_parser_has_no_model_authored_keys_or_aliases() -> None:
    selected = parse_hypothesis_selection_v2('["F1", "E3", "P4", "A1"]')
    assert selected == HypothesisSelectionV2("F1", "E3", "P4", "A1")
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_hypothesis_selection_v2(
            '{"failure_id":"F1","edit_id":"E3","preserve_id":"P4","avoid_id":"A1"}'
        )
    with pytest.raises(ValueError, match="exact JSON"):
        parse_hypothesis_selection_v2('```json\n["F1","E3","P4","A1"]\n```')
    with pytest.raises(ValueError, match="unknown"):
        parse_hypothesis_selection_v2('["failure", "E3", "P4", "A1"]')


def test_v2_every_selection_materializes_to_contract_valid_prompt() -> None:
    parent = "Use a careful, evidence-grounded decision procedure."
    count = 0
    for failure in FAILURE_IDS:
        for edit in EDIT_IDS:
            for preserve in PRESERVE_IDS:
                for avoid in AVOID_IDS:
                    prompt = render_contract_adapted_prompt_v2(
                        parent,
                        HypothesisSelectionV2(failure, edit, preserve, avoid),
                    )
                    assert not mutable_prompt_violation_reasons(prompt)
                    count += 1
    assert count == 6 * 6 * 4 * 4
    assert len(hypothesis_interface_v2_identity()) == 64


def test_v2_schema_qualification_is_isolated_and_fail_closed() -> None:
    protocol = SchemaQualificationV2Protocol()
    assert protocol.request_count == 12
    assert protocol.scientific_evidence_eligible is False
    assert protocol.schema_retry_enabled is False
    assert protocol.solver_enabled is False
    assert protocol.validation_enabled is False
    assert protocol.test_enabled is False
    assert len(protocol.identity()) == 64


def test_contract_adapted_scientific_protocol_v2_is_separate_and_frozen() -> None:
    v1 = ContractAdaptedProtocol()
    v2 = ContractAdaptedProtocolV2()
    assert v2.identity() != v1.identity()
    assert v2.hypothesis_interface_version == "rg_gepa_hypothesis_interface_v2"
    assert v2.selection_arms == v1.selection_arms
    assert v2.commit_enabled is False
    assert v2.validation_enabled is False
    assert v2.test_enabled is False
    assert v2.memory_enabled is False
    assert v2.optimizer_model == "qwen3.7-flash"
    assert v2.reflection_temperature == 0.0
    assert v2.reflection_max_tokens == 64
    assert len(v2.hypothesis_interface_hash) == 64
    assert len(v2.renderer_vocabulary_hash) == 64


def test_v2_request_builder_is_shared_and_fail_closed() -> None:
    system, user = build_hypothesis_request_v2(
        responsibility_lane="coverage",
        context_lines=("Symbolic evidence: one uncovered responsibility",),
        mutation_index=0,
    )
    assert "exact JSON array" in system
    assert '["F1","E1","P1","A1"]' in user
    with pytest.raises(ValueError, match="responsibility lane"):
        build_hypothesis_request_v2(
            responsibility_lane="unknown",
            context_lines=("evidence",),
            mutation_index=0,
        )
