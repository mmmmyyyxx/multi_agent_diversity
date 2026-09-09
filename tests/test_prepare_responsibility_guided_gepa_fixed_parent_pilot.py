from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path=Path(__file__).parents[1]/"scripts"/"prepare_responsibility_guided_gepa_fixed_parent_pilot.py"
    spec=importlib.util.spec_from_file_location("rg_gepa_prepare",path)
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parent_reconstruction_matches_historical_hashes() -> None:
    module=load()
    source=module.source_rows(76)
    cases=module.select_cases(76,source)
    assert [row["responsibility_type"] for row in cases]==["coverage","margin_support","direct_flip"]
    for case in cases:
        assert module.team_hash(case["parent_prompts"])==case["parent_team_hash"]


def test_minibatches_are_deterministic_and_optimize_only() -> None:
    module=load(); source=module.source_rows(77); case=module.select_cases(77,source)[0]
    first=module.evidence_for_case(case); second=module.evidence_for_case(case)
    assert first==second and len(first)==12
    assert {row.example_id for row in first}.issubset({row["example_id"] for row in case["questions"]})
