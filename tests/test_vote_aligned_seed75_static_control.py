from __future__ import annotations

import inspect

from scripts import run_vote_aligned_seed75_static_control as static


def test_member_classifier_uses_frozen_tolerance() -> None:
    assert static.classify_member_effect(0.01001) == "P1_ABSOLUTE_MEMBER_GAIN"
    assert static.classify_member_effect(0.01) == "P1_MEMBER_ROUGHLY_PRESERVED"
    assert static.classify_member_effect(-0.01) == "P1_MEMBER_ROUGHLY_PRESERVED"
    assert static.classify_member_effect(-0.01001) == "P1_ABSOLUTE_MEMBER_DEGRADATION"


def test_ensemble_classifier_uses_sign_only() -> None:
    assert static.classify_ensemble_effect(0.1) == "POSITIVE"
    assert static.classify_ensemble_effect(0.0) == "NEUTRAL"
    assert static.classify_ensemble_effect(-0.1) == "NEGATIVE"


def test_versioned_test_call_field_remains_fail_closed() -> None:
    assert static._recorded_test_calls({"new_test_calls": 0}) == 0
    assert static._recorded_test_calls({"test_calls": 0}) == 0
    try:
        static._recorded_test_calls({})
    except KeyError:
        pass
    else:
        raise AssertionError("missing test-call evidence must fail closed")


def test_execution_path_contains_no_training_or_test_call() -> None:
    source = inspect.getsource(static.execute)
    assert "update_once" not in source
    assert "evaluate_final_test" not in source
    assert "Teacher" not in source
    assert "Student" not in source
    assert '"epochs": 0' in source


def test_static_control_paths_are_project_local() -> None:
    for path in (static.DEFAULT_PREP_ROOT, static.DEFAULT_RUN_ROOT, static.DEFAULT_REPORT_ROOT):
        path.resolve().relative_to(static.ROOT.resolve())
