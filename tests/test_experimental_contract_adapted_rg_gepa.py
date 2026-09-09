from __future__ import annotations

import json

import pytest

from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    mutable_prompt_violation_reasons,
)
from multi_dataset_diverse_rl.experimental_contract_adapted_rg_gepa import (
    AVOIDANCE_PRIORITIES,
    BEHAVIORAL_CHANGES,
    ContractAdaptedProtocol,
    EditHypothesis,
    FAILURE_PATTERNS,
    PRESERVATION_PRIORITIES,
    parse_edit_hypothesis,
    render_contract_adapted_prompt,
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
