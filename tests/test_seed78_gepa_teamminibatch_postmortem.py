import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "seed78_gepa_teamminibatch_postmortem_20260913"


def _json(name: str):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_seed78_postmortem_reconciles_root_only_gepa_funnel() -> None:
    summary = _json("summary.json")
    assert summary["classifier"] == "ROOT_CANDIDATE_RETURN_SEMANTICS_CONFIRMED"
    assert summary["scientific_status"] == "SCHEDULER_CAUSAL_EFFICACY_NOT_EVALUATED"
    assert summary["total"] == {
        "accepted_mutations": 0,
        "branches": 24,
        "program_candidates": 24,
        "proposals": 89,
        "rejected_mutations": 89,
        "returned_candidates": 24,
        "returned_parent_candidates": 24,
        "team_minibatch_survivors": 0,
    }

    with (REPORT / "branch_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert {row["frontier_indices"] for row in rows} == {"0"}
    assert {row["returned_candidate_indices"] for row in rows} == {"0"}
    assert {row["returned_equals_parent"] for row in rows} == {"True"}
    assert {row["promotion_rejection_class"] for row in rows} == {"UNCHANGED_PARENT"}
    for row in rows:
        assert all(
            int(row[field]) == 0
            for field in (
                "invalid_delta", "vote_delta", "target_delta", "coalition_delta",
                "responsibility_delta", "broad_delta",
            )
        )


def test_seed78_postmortem_confirms_paired_evaluation_cache_gap() -> None:
    paired = _json("paired_evaluation_audit.json")
    assert paired["classifier"] == "PAIRED_EVALUATION_CACHE_GAP_CONFIRMED"
    assert paired["final_team_byte_identity_equal"] is True
    assert paired["validation_request_identity_sets_equal"] is True
    assert paired["paired_cross_arm_realization_cache"] is False
    assert paired["validation_successful_provider_calls"] == {"A": 50, "B": 50}
    assert paired["validation_cache_hits"] == {"A": 0, "B": 0}


def test_seed78_postmortem_is_sanitized_and_hash_complete() -> None:
    sanitization = _json("sanitization_manifest.json")
    assert sanitization["gate"] == "PASS"
    assert sanitization["forbidden_findings"] == []
    facts = _json("fact_assertions.json")
    assert facts["gate"] == "PASS"
    assert all(facts["assertions"].values())

    manifest = _json("sha256_manifest.json")
    expected = {
        path.name
        for path in REPORT.iterdir()
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    assert set(manifest) == expected
    for name, expected_hash in manifest.items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected_hash
