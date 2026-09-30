"""Rebuild deterministic, relative-path evidence hashes; never reads run artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
BASE_SHA = "b8555b9005fb117a086c639e51e0a5e6f9ba1443"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from multi_dataset_diverse_rl.benchmarks.registry import BENCHMARKS  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (REPORT / name).write_text(json.dumps(value, indent=2, sort_keys=True,
                                          ensure_ascii=False) + "\n", encoding="utf-8")


source_paths = sorted((ROOT / "multi_dataset_diverse_rl").rglob("*.py"))
source_paths += [ROOT / "scripts" / "run_experiment.py",
                 ROOT / "docs" / "design" / "CURRENT_SPEC.md"]
source_paths += sorted((ROOT / "tests").glob("test_*.py"))
source_paths = sorted(path for path in source_paths if path.is_file())
write("source_closure.json", {
    "base_sha": BASE_SHA,
    "scope": "Conservative superset: complete package, production CLI, current spec, all test modules",
    "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
              for path in source_paths],
})
write("benchmark_capabilities.json", {
    key: {**asdict(value), "blockers": list(value.blockers()),
          "aggregation_ready": value.aggregation_ready,
          "responsibility_ready": value.responsibility_ready,
          "unified_search_ready": value.unified_search_ready}
    for key, value in sorted(BENCHMARKS.items())
})
write("freeze_identity.json", {
    "base_sha": BASE_SHA,
    "task": "unified_benchmark_migration_v1_20260930",
    "execution_status": "ZERO_API_SOFTWARE_FREEZE",
    "formal_execution_ready": False,
    "source_closure_sha256": sha(REPORT / "source_closure.json"),
    "real_api_calls": 0,
    "bbh_validation50_calls": 0,
    "bbh_test50_calls": 0,
    "new_commit_sha": "RECORDED_BY_GIT_COMMIT_NOT_SELF_REFERENTIAL",
})
report_files = sorted(path for path in REPORT.iterdir()
                      if path.is_file() and path.name != "sha256_manifest.json")
write("sha256_manifest.json", {
    "hash_algorithm": "SHA-256",
    "scope": "All report files except this manifest, which cannot self-hash",
    "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
              for path in report_files],
})
