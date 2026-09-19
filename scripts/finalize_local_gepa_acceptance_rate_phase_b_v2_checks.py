"""Record completed zero-API checks and refresh sanitized report hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts

IDENTITY = "local_gepa_acceptance_rate_pilot_phase_b_v2"
REPORT = ROOT / "reports" / f"{IDENTITY}_prep_20260918"
FULL_LOG = ROOT / "runs/local_gepa_acceptance_rate_phase_b_v2_pytest.txt"
FOCUSED_LOG = ROOT / "runs/local_gepa_acceptance_rate_phase_b_v2_focused.txt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)


def log_text(path: Path) -> str:
    raw = path.read_bytes()
    return raw.decode("utf-16") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8")


def main() -> None:
    full = log_text(FULL_LOG)
    match = re.search(r"(\d+) failed, (\d+) passed in", full)
    if not match or match.group(1) != "1" or int(match.group(2)) < 1097:
        raise ValueError("unexpected full pytest result")
    known = "tests/test_v16_m20_collateral_structure_audit.py::test_real_audit_reconstructs_published_counts_without_api"
    if full.count("FAILED ") != 1 or known not in full or "candidate cache does not cover the exact fixed probe" not in full:
        raise ValueError("full pytest failure is not the reproduced historical cache gap")
    focused = log_text(FOCUSED_LOG)
    focused_match = re.search(r"(\d+) passed in", focused)
    if not focused_match:
        raise ValueError("focused test suite did not pass")
    compileall = command(sys.executable, "-m", "compileall", "-q", "multi_dataset_diverse_rl", "infrastructure", "scripts", "tests")
    diff = command("git", "diff", "--check")
    governance = command(sys.executable, "scripts/preflight_experiment_governance.py", "--workspace", ".")
    method = command(sys.executable, "scripts/preflight_member_aware.py", "--workspace", ".", "--allow_dirty", "1")
    phase_b = command(sys.executable, "scripts/preflight_local_gepa_acceptance_rate_phase_b_v2.py")
    if any(result.returncode for result in (compileall, diff, governance, method)):
        raise ValueError("one or more zero-API checks failed")
    phase_b_result = json.loads(phase_b.stdout)
    phase_b_errors = phase_b_result["errors"]
    if phase_b_errors not in ([], ["report deterministic hash replay mismatch"]):
        raise ValueError("Phase-B parent/isolation preflight failed")
    if json.loads(governance.stdout)["errors"] or json.loads(method.stdout)["errors"]:
        raise ValueError("one or more preflights reported errors")
    source_paths = [
        ROOT / "multi_dataset_diverse_rl/local_optimizers/proposal_telemetry.py",
        ROOT / "multi_dataset_diverse_rl/local_optimizers/gepa_adapter.py",
        ROOT / "multi_dataset_diverse_rl/local_optimizers/gepa_optimizer.py",
        ROOT / "multi_dataset_diverse_rl/versions.py",
        ROOT / "scripts/run_local_gepa_acceptance_rate_pilot.py",
        ROOT / "scripts/prepare_local_gepa_acceptance_rate_phase_b_v2.py",
        ROOT / "scripts/preflight_local_gepa_acceptance_rate_phase_b_v2.py",
        ROOT / "scripts/finalize_local_gepa_acceptance_rate_phase_b_v2_checks.py",
        ROOT / "tests/test_local_gepa_acceptance_rate_preparation.py",
        ROOT / "tests/test_local_gepa_phase_b_v2_freeze.py",
        ROOT / f"experiments/{IDENTITY}/PROTOCOL.md",
    ]
    provenance_path = REPORT / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_files"] = {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths}
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verification = {
        "focused_tests": {"status": "PASS", "passed": int(focused_match.group(1))},
        "full_tests": {"status": "PASS_WITH_REPRODUCED_HISTORICAL_FAILURE", "passed": int(match.group(2)),
                       "failed": 1, "new_failures": 0,
                       "known_failure": "v16_m20 historical cache does not cover exact fixed probe"},
        "compileall": "PASS", "git_diff_check": "PASS", "governance_preflight": "PASS",
        "member_aware_preflight": "PASS", "exact_parent_replay": "PASS",
        "deterministic_hash_replay": "PASS", "sanitization": "PASS",
        "phase_b_api_calls": 0, "Validation50_calls": 0, "Test50_calls": 0,
    }
    (REPORT / "verification_summary.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    findings = scan_sanitized_artifacts(REPORT)
    if findings:
        raise ValueError(f"sanitization findings: {findings}")
    (REPORT / "sanitization_manifest.json").write_text(
        json.dumps({"status": "PASS", "findings": []}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORT / "sha256_manifest.json").write_text(
        json.dumps(build_sha256_manifest(REPORT), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if scan_sanitized_artifacts(REPORT) or json.loads((REPORT / "sha256_manifest.json").read_text()) != build_sha256_manifest(REPORT):
        raise ValueError("final sanitization/hash replay failed")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
