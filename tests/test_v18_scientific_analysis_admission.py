from __future__ import annotations

from scripts.admit_v18_hybrid_online_scientific_analysis import (
    EXPECTED_ORIGINAL_BLOCKERS,
    build_admission,
)


def inputs(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "execution_summary.json").write_text(
        '{"trajectory_count":6,"new_test_calls":0,'
        '"infrastructure_failure_count":0}',
        encoding="utf-8",
    )
    # Synthetic tests do not exercise source-freeze byte identity; an empty
    # manifest makes that check fail safely unless the three required files are
    # explicitly supplied. Patch in the current frozen files below.
    from scripts.admit_v18_hybrid_online_scientific_analysis import ROOT, sha256_file

    required = (
        "scripts/analyze_v18_hybrid_online_accumulation.py",
        "scripts/v18_hybrid_online_accumulation_support.py",
        "experiments/v18_hybrid_online_accumulation_pilot_20260822/classifier_spec.json",
    )
    freeze = {"files": [
        {"path": rel, "sha256": sha256_file(ROOT / rel)} for rel in required
    ]}
    from scripts.admit_v18_hybrid_online_scientific_analysis import artifact_tree_identity

    semantics = {
        "semantics_audit_gate": "PASS",
        "semantics_audit_blockers": [],
        "scientific_interpretation_status": "not_performed",
        "raw_artifact_identity": artifact_tree_identity(root),
    }
    corrected = {
        "version": "post_hoc_corrected_gate_v1",
        "gate": "PASS",
        "blockers": [],
        "replaces_original_gate": False,
        "original_gate_remains": "FAIL/HOLD",
        "correction_scope": "revision_parity_representation_only",
    }
    original = {
        "audit_version": "v18_hybrid_online_accumulation_audit_v1",
        "gate": "FAIL",
        "blockers": sorted(EXPECTED_ORIGINAL_BLOCKERS),
        "new_test_calls": 0,
        "infrastructure_failure_count": 0,
    }
    return root, original, semantics, corrected, freeze


def admit(tmp_path, **changes):
    root, original, semantics, corrected, freeze = inputs(tmp_path)
    values = {
        "run_root": root,
        "original_audit": original,
        "semantics_audit": semantics,
        "corrected_gate": corrected,
        "source_freeze": freeze,
        "canonical_report_dir": tmp_path / "report",
    }
    values.update(changes)
    return build_admission(**values)


def test_corrected_gate_admits_without_rewriting_original_hold(tmp_path):
    result = admit(tmp_path)
    assert result["scientific_analysis_admitted"] is True
    assert result["gate"] == "PASS"
    assert result["original_frozen_audit_status"] == "FAIL/HOLD"
    assert result["original_frozen_blockers"] == sorted(EXPECTED_ORIGINAL_BLOCKERS)
    assert result["new_api_calls"] == result["new_test_calls"] == 0


def test_semantics_failure_blocks_scientific_analyzer(tmp_path):
    root, original, semantics, corrected, freeze = inputs(tmp_path)
    semantics["semantics_audit_gate"] = "FAIL"
    result = build_admission(
        run_root=root,
        original_audit=original,
        semantics_audit=semantics,
        corrected_gate=corrected,
        source_freeze=freeze,
        canonical_report_dir=tmp_path / "report",
    )
    assert result["scientific_analysis_admitted"] is False
    assert "semantics_audit" in result["admission_blockers"]


def test_missing_corrected_gate_blocks_scientific_analyzer(tmp_path):
    root, original, semantics, corrected, freeze = inputs(tmp_path)
    corrected["gate"] = "FAIL"
    result = build_admission(
        run_root=root,
        original_audit=original,
        semantics_audit=semantics,
        corrected_gate=corrected,
        source_freeze=freeze,
        canonical_report_dir=tmp_path / "report",
    )
    assert result["scientific_analysis_admitted"] is False
    assert "corrected_gate" in result["admission_blockers"]


def test_raw_hash_drift_blocks_scientific_analyzer(tmp_path):
    root, original, semantics, corrected, freeze = inputs(tmp_path)
    (root / "drift.txt").write_text("drift", encoding="utf-8")
    result = build_admission(
        run_root=root,
        original_audit=original,
        semantics_audit=semantics,
        corrected_gate=corrected,
        source_freeze=freeze,
        canonical_report_dir=tmp_path / "report",
    )
    assert result["scientific_analysis_admitted"] is False
    assert "raw_hash_match" in result["admission_blockers"]


def test_existing_scientific_report_blocks_admission(tmp_path):
    root, original, semantics, corrected, freeze = inputs(tmp_path)
    report = tmp_path / "report"
    report.mkdir()
    result = build_admission(
        run_root=root,
        original_audit=original,
        semantics_audit=semantics,
        corrected_gate=corrected,
        source_freeze=freeze,
        canonical_report_dir=report,
    )
    assert result["scientific_analysis_admitted"] is False
    assert "scientific_analyzer_previously_run" in result["admission_blockers"]


def test_frozen_analyzer_source_has_no_test_artifact_access():
    from pathlib import Path

    source = Path("scripts/analyze_v18_hybrid_online_accumulation.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "test_states.jsonl", "final_test", "test_split", "test_examples",
        "evaluate_final_test",
    )
    assert not any(token in source for token in forbidden)
