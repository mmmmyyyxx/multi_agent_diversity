from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_v16_responsibility_reports.py"
SPEC = importlib.util.spec_from_file_location("package_v16_reports", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def _write_sources(root: Path, study: str) -> None:
    root.mkdir()
    for name in PACKAGE.SOURCE_ALLOWLIST[study]:
        path = root / name
        if path.suffix == ".json":
            payload = {"status": "PASS", "value": 1}
            if name == "protocol_gate.json":
                payload["gate"] = "PASS"
            if name == "source_freeze_sanitized.json":
                payload["source_freeze_status"] = "PASS"
            path.write_text(
                json.dumps(payload) + "\n",
                encoding="utf-8",
            )
        else:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
                writer.writeheader()
                writer.writerow({"metric": "safe", "value": 1})


@pytest.mark.parametrize("study", ["study_a", "study_b"])
def test_packager_copies_only_allowlist_and_generates_manifest(
    tmp_path, monkeypatch, study
):
    source = tmp_path / "source"
    destination = tmp_path / "report"
    _write_sources(source, study)
    (source / "unrelated.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(PACKAGE, "ROOT", tmp_path)
    monkeypatch.setitem(PACKAGE.REPORT_ROOTS, study, destination)
    manifest = PACKAGE.package_study(
        study=study, source=source, destination=destination
    )
    assert {path.name for path in destination.iterdir()} == PACKAGE.EXPECTED_OUTPUTS[
        study
    ]
    assert "unrelated.json" not in {path.name for path in destination.iterdir()}
    assert manifest["sanitization_status"] == "PASS"
    written = json.loads(
        (destination / "sanitized_manifest.json").read_text(encoding="utf-8")
    )
    for row in written["files"]:
        path = destination / row["name"]
        assert row["sha256"] == PACKAGE.sha256(path)
        assert row["size"] == path.stat().st_size


@pytest.mark.parametrize(
    "name,payload",
    [
        ("bad.json", {"gold_answer": "a"}),
        ("bad.json", {"safe": r"D:\\private\\run"}),
        ("bad.json", {"safe": "https://provider.invalid/v1"}),
        ("bad.json", {"api_key": "sk-secret-value"}),
    ],
)
def test_structured_scanner_rejects_sensitive_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        PACKAGE.scan_sanitized_file(path)


def test_csv_scanner_rejects_sensitive_header_and_absolute_value(tmp_path):
    header = tmp_path / "header.csv"
    header.write_text("question_text,value\nsecret,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sensitive structured field"):
        PACKAGE.scan_sanitized_file(header)
    hashes = tmp_path / "hashes.csv"
    hashes.write_text(
        "responsibility_question_hash_count,value\n2,1\n", encoding="utf-8"
    )
    PACKAGE.scan_sanitized_file(hashes)
    value = tmp_path / "value.csv"
    value.write_text("metric,value\nsafe,C:\\private\\run\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absolute-path"):
        PACKAGE.scan_sanitized_file(value)
    extra = tmp_path / "extra.csv"
    extra.write_text("metric,value\nsafe,1,hidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra unnamed columns"):
        PACKAGE.scan_sanitized_file(extra)


def test_scanner_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"PASS","status":"FAIL"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        PACKAGE.scan_sanitized_file(path)


@pytest.mark.parametrize("study", ["study_a", "study_b"])
def test_packager_refuses_failed_scientific_gate(tmp_path, monkeypatch, study):
    source = tmp_path / "source"
    destination = tmp_path / "report"
    _write_sources(source, study)
    if study == "study_a":
        (source / "audit_summary.json").write_text(
            json.dumps({"status": "FAIL"}), encoding="utf-8"
        )
    else:
        (source / "protocol_gate.json").write_text(
            json.dumps({"gate": "FAIL"}), encoding="utf-8"
        )
    monkeypatch.setattr(PACKAGE, "ROOT", tmp_path)
    monkeypatch.setitem(PACKAGE.REPORT_ROOTS, study, destination)
    with pytest.raises(ValueError, match="must PASS"):
        PACKAGE.package_study(
            study=study, source=source, destination=destination
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "name", ["solver_cache.sqlite", "training_checkpoint.json", "raw_response.json"]
)
def test_scanner_rejects_forbidden_runtime_filenames(tmp_path, name):
    path = tmp_path / name
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden report artifact filename"):
        PACKAGE.scan_sanitized_file(path)


def test_packager_rejects_source_or_destination_outside_repo(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(PACKAGE, "ROOT", repository)
    with pytest.raises(PermissionError, match="repository root"):
        PACKAGE.package_study(
            study="study_a", source=outside, destination=repository / "report"
        )


def test_packager_requires_fresh_frozen_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "report"
    _write_sources(source, "study_a")
    destination.mkdir()
    monkeypatch.setattr(PACKAGE, "ROOT", tmp_path)
    monkeypatch.setitem(PACKAGE.REPORT_ROOTS, "study_a", destination)
    with pytest.raises(FileExistsError, match="fresh"):
        PACKAGE.package_study(
            study="study_a", source=source, destination=destination
        )
