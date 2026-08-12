from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOTS = {
    "study_a": (
        ROOT
        / "experiments"
        / "v16_responsibility_coherence_audit_20260812"
    ),
    "study_b": (
        ROOT
        / "experiments"
        / "v16_generic_vs_m20_fixed_parent_probe_20260812"
    ),
}
SOURCE_ALLOWLIST = {
    "study_a": {
        "audit_summary.json",
        "branch_coherence_metrics.csv",
        "coherence_outcome_strata.csv",
        "candidate_repair_focus.csv",
        "seed_summary.csv",
        "audit_assertions.json",
    },
    "study_b": {
        "protocol_gate.json",
        "variant_metrics.csv",
        "case_metrics.csv",
        "paired_comparisons.csv",
        "responsibility_repair_metrics.csv",
        "pipeline_cost_metrics.csv",
        "seed_metrics.csv",
        "analysis_summary.json",
        "source_freeze_sanitized.json",
    },
}
EXPECTED_OUTPUTS = {
    study: allowlist | {"README.md", "sanitized_manifest.json"}
    for study, allowlist in SOURCE_ALLOWLIST.items()
}
FORBIDDEN_FILE_SUFFIXES = {
    ".sqlite",
    ".db",
    ".ckpt",
    ".checkpoint",
    ".bin",
    ".pkl",
    ".pickle",
}
FORBIDDEN_FILE_PARTS = (
    "cache",
    "checkpoint",
    "raw_response",
    "llm_calls",
    "best_prompts",
    "history",
)
FORBIDDEN_KEY_PARTS = (
    "question_text",
    "question_body",
    "question_content",
    "question_answer",
    "question_output",
    "gold_answer",
    "gold_label",
    "model_answer",
    "raw_answer",
    "raw_prompt",
    "prompt_text",
    "raw_response",
    "response_text",
    "credential",
    "api_key",
    "secret",
    "endpoint",
    "cache_path",
    "checkpoint_path",
)
FORBIDDEN_VALUE_PATTERNS = (
    re.compile(
        r"(?i)(?:[a-z]:[\\/]|file://|\\\\[^\\/\s]+[\\/][^\\/\s]+|"
        r"(?:^|[\s\"'=])/(?:[^\s\"']*))"
    ),
    re.compile(r"(?i)\b(?:https?|file|sqlite)://"),
    re.compile(r"(?i)final_answer:"),
    re.compile(r"(?i)openai_api_key"),
    re.compile(r"(?i)\b(?:sk-[a-z0-9_-]{8,}|bearer\s+[a-z0-9._-]{8,})\b"),
    re.compile(r"(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
        raise PermissionError("report path must remain under repository root")
    return resolved


def _validate_filename(name: str) -> None:
    lowered = name.lower()
    suffix = Path(name).suffix.lower()
    if suffix in FORBIDDEN_FILE_SUFFIXES or any(
        token in lowered for token in FORBIDDEN_FILE_PARTS
    ):
        raise ValueError(f"forbidden report artifact filename: {name}")


def _validate_key(key: str, location: str) -> None:
    lowered = str(key).lower()
    if any(token in lowered for token in FORBIDDEN_KEY_PARTS):
        raise ValueError(f"sensitive structured field at {location}/{key}")


def _validate_scalar(value: Any, location: str) -> None:
    if not isinstance(value, str):
        return
    if any(pattern.search(value) for pattern in FORBIDDEN_VALUE_PATTERNS):
        raise ValueError(f"sensitive or absolute-path content at {location}")


def _scan_structured(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_key(str(key), location)
            _scan_structured(child, f"{location}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_structured(child, f"{location}/{index}")
    else:
        _validate_scalar(value, location)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def scan_sanitized_file(path: Path) -> None:
    _validate_filename(path.name)
    suffix = path.suffix.lower()
    if suffix == ".json":
        _scan_structured(json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        ))
        return
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV lacks a header: {path.name}")
            for field in reader.fieldnames:
                _validate_key(field, f"{path.name}/header")
            for row_index, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"CSV row has extra unnamed columns: {path.name}/{row_index}"
                    )
                for field, value in row.items():
                    _validate_scalar(value, f"{path.name}/{row_index}/{field}")
        return
    raise ValueError(f"only JSON and CSV source artifacts may be packaged: {path.name}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readme(study: str, files: list[str]) -> str:
    if study == "study_a":
        title = "Responsibility Coherence Audit"
        scope = (
            "Offline train-trajectory mechanism audit over preregistered Seeds "
            "48-51. No API, validation, or test evaluation was used."
        )
    else:
        title = "Generic vs M20 Fixed-Parent Probe"
        scope = (
            "Preregistered fixed-parent, fixed-target diagnostic comparison. "
            "No team update, validation evaluation, or test evaluation was used."
        )
    listing = "\n".join(f"- `{name}`" for name in files)
    return (
        f"# {title}\n\n{scope}\n\n"
        "This directory contains sanitized, analysis-ready evidence only. It "
        "excludes prompts, question text, gold/model answers, raw provider "
        "responses, caches, checkpoints, credentials, endpoints, and absolute "
        "local paths.\n\n"
        f"## Files\n\n{listing}\n"
    )


def package_study(*, study: str, source: Path, destination: Path) -> dict[str, Any]:
    if study not in SOURCE_ALLOWLIST:
        raise ValueError(f"unknown report study: {study}")
    source = _inside_repo(source)
    destination = _inside_repo(destination)
    if source == destination:
        raise ValueError("source and destination must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination != REPORT_ROOTS[study].resolve():
        raise PermissionError("destination is not the frozen publish directory")
    if destination.exists():
        raise FileExistsError("publish directory must be fresh")
    actual = {path.name for path in source.iterdir() if path.is_file()}
    missing = SOURCE_ALLOWLIST[study] - actual
    if missing:
        raise FileNotFoundError(
            "missing required sanitized inputs: " + ", ".join(sorted(missing))
        )
    if study == "study_a":
        summary = json.loads(
            (source / "audit_summary.json").read_text(encoding="utf-8")
        )
        assertions = json.loads(
            (source / "audit_assertions.json").read_text(encoding="utf-8")
        )
        if summary.get("status") != "PASS" or assertions.get(
            "status"
        ) != "PASS":
            raise ValueError("Study-A audit status must PASS before packaging")
    else:
        protocol = json.loads(
            (source / "protocol_gate.json").read_text(encoding="utf-8")
        )
        freeze = json.loads(
            (source / "source_freeze_sanitized.json").read_text(
                encoding="utf-8"
            )
        )
        if protocol.get("gate") != "PASS":
            raise ValueError("Study-B protocol gate must PASS before packaging")
        if freeze.get("source_freeze_status") != "PASS":
            raise ValueError("Study-B source freeze must PASS before packaging")
    for name in sorted(SOURCE_ALLOWLIST[study]):
        scan_sanitized_file(source / name)

    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in sorted(SOURCE_ALLOWLIST[study]):
            shutil.copyfile(source / name, destination / name)
        readme_files = sorted(SOURCE_ALLOWLIST[study]) + ["sanitized_manifest.json"]
        (destination / "README.md").write_text(
            _readme(study, readme_files), encoding="utf-8"
        )
        manifest_files = []
        for name in sorted(SOURCE_ALLOWLIST[study] | {"README.md"}):
            path = destination / name
            manifest_files.append(
                {"name": name, "sha256": sha256(path), "size": path.stat().st_size}
            )
        manifest = {
            "manifest_version": "v16_sanitized_report_manifest_v1",
            "study": study,
            "sanitization_status": "PASS",
            "files": manifest_files,
        }
        (destination / "sanitized_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in destination.iterdir():
            if path.name != "README.md":
                scan_sanitized_file(path)
        if {path.name for path in destination.iterdir()} != EXPECTED_OUTPUTS[study]:
            raise AssertionError("packaged report file scope differs from allowlist")
    except Exception:
        # The directory is fresh and created by this function, so rollback is exact.
        shutil.rmtree(destination)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_a_source", type=Path, required=True)
    parser.add_argument("--study_b_source", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "study_a": package_study(
            study="study_a",
            source=args.study_a_source,
            destination=REPORT_ROOTS["study_a"],
        ),
        "study_b": package_study(
            study="study_b",
            source=args.study_b_source,
            destination=REPORT_ROOTS["study_b"],
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
