import ast
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.persistence.artifacts import ArtifactWriter


ROOT = Path(__file__).resolve().parents[1]


def test_system_facade_contains_no_search_or_persistence_formulas():
    source = (ROOT / "multi_dataset_diverse_rl" / "system.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert functions == []
    assert len(source.splitlines()) <= 1800


def test_prompt_update_is_explicit_short_stage_pipeline():
    source = (ROOT / "multi_dataset_diverse_rl" / "optimization" / "prompt_update_controller.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    expected = {
        "CandidateGenerationStage", "CheapPrescreenStage", "CandidateEvaluationStage",
        "CandidateClassificationAndRefillStage", "ArchiveSelectionStage",
        "CandidateEventStage", "UpdateSummaryStage",
    }
    assert expected <= set(classes)
    wrapper = next(
        node for node in classes["PromptUpdateMixin"].body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_prompt_with_beam"
    )
    assert wrapper.end_lineno - wrapper.lineno + 1 < 30
    assert max(node.end_lineno - node.lineno + 1 for name, node in classes.items() if name in expected) < 200


def test_artifact_writer_rejects_nonfinite_json(tmp_path):
    writer = ArtifactWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.append_jsonl("bad.jsonl", [{"value": float("nan")}])
    assert not (tmp_path / "bad.jsonl").exists() or not (tmp_path / "bad.jsonl").read_text(encoding="utf-8")
