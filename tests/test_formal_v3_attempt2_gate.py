"""Formal attempt2 evidence gate and Native representation, with no provider."""

import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.formal_final_team import persist_final_team, team_hash
from multi_dataset_diverse_rl.formal_validation_policy import formal_validation50_policy
from multi_dataset_diverse_rl.governance.production_execution import (
    _expected_bundle, validate_execution,
)
from multi_dataset_diverse_rl.governance.startup_identity import (
    StartupIdentityError, write_bundle,
)
from scripts.prepare_gepa_saturation_comparison_v3 import frozen_payload
from scripts.prepare_post_refactor_gepa_canary import _private_splits

ROOT = Path(__file__).resolve().parents[1]


def _prep(tmp_path, monkeypatch, *, attempt_number):
    monkeypatch.setattr(
        "scripts.prepare_gepa_saturation_comparison_v3.endpoint_fingerprint_from_environment",
        lambda: "f" * 64,
    )
    monkeypatch.setattr(
        "multi_dataset_diverse_rl.governance.production_execution.endpoint_fingerprint_from_environment",
        lambda: "f" * 64,
    )
    source = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    manifest, protocol = frozen_payload(
        execution_source_sha=source, seed=80, scope="native",
        attempt_number=attempt_number,
    )
    _private_splits(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    write_bundle(tmp_path / "startup_identity", _expected_bundle(
        root=ROOT, manifest=manifest, protocol=protocol,
        execution_source_sha=source, prep=tmp_path,
    ))
    return manifest, protocol


def test_formal_attempt1_remains_superseded(tmp_path, monkeypatch):
    manifest, _ = _prep(tmp_path, monkeypatch, attempt_number=1)
    assert manifest["execution_gate"] == {"real_v4_diagnostic": "PENDING_SCIENTIFIC_VALIDITY"}
    with pytest.raises(StartupIdentityError, match="superseded before execution"):
        validate_execution(root=ROOT, prep=tmp_path, require_authorized=True)
    manifest["execution_gate"] = {"real_v4_diagnostic": "SCIENTIFICALLY_VALID"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StartupIdentityError, match="historical formal attempt1 identity mismatch"):
        validate_execution(root=ROOT, prep=tmp_path, require_authorized=False)


def test_formal_attempt2_binds_validity_not_efficacy(tmp_path, monkeypatch):
    manifest, protocol = _prep(tmp_path, monkeypatch, attempt_number=2)
    policy = formal_validation50_policy(ROOT)
    assert manifest["post_freeze_validation50"] == protocol["post_freeze_validation50"] == policy
    assert policy["split_identity"] == "anti_overfitting_split_v1/validation"
    assert policy["count"] == 50
    assert policy["question_hash_set_sha256"] == "d6514ca588f124460502afbc44a5314e9ef7f141ee826411cb6f02abad736df5"
    assert policy["split_manifest_sha256"] == "9639db6f1a83bbe54704d012edd7b843e31cbb1141f9957437daa3803a954886"
    assert policy["question_hash_set_sha256"] != policy["shadow50_question_hash_set_sha256"]
    historical, historical_protocol = frozen_payload(
        execution_source_sha=manifest["execution"]["execution_source_sha"],
        seed=80, scope="native", attempt_number=1,
    )
    assert manifest["scientific"] == historical["scientific"]
    assert manifest["spec_identity"] == historical["spec_identity"]
    assert manifest["formal_v3_contract"] == historical["formal_v3_contract"]
    assert protocol["formal_v3_contract"] == historical_protocol["formal_v3_contract"]
    assert manifest["diagnostic_prerequisite"]["efficacy"] == "NOT_EVALUABLE"
    assert validate_execution(root=ROOT, prep=tmp_path, require_authorized=False).attempt_id.endswith("attempt2")
    for field, bad in (("execution_source_sha", "0" * 40),
                       ("audit_commit", "1" * 40), ("scientific_validity", "INVALID")):
        altered = json.loads(json.dumps(manifest))
        altered["diagnostic_prerequisite"][field] = bad
        (tmp_path / "manifest.json").write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(StartupIdentityError, match="diagnostic prerequisite mismatch"):
            validate_execution(root=ROOT, prep=tmp_path, require_authorized=False)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    protocol["diagnostic_prerequisite"]["scientific_validity"] = "INVALID"
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(StartupIdentityError, match="diagnostic prerequisite mismatch"):
        validate_execution(root=ROOT, prep=tmp_path, require_authorized=False)
    protocol["diagnostic_prerequisite"]["scientific_validity"] = "VALID"
    wrong = json.loads(json.dumps(policy))
    wrong["split_identity"] = "anti_overfitting_split_v1/fold_c"
    wrong["question_hash_set_sha256"] = policy["shadow50_question_hash_set_sha256"]
    manifest["post_freeze_validation50"] = wrong
    protocol["post_freeze_validation50"] = wrong
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(StartupIdentityError, match="diagnostic prerequisite mismatch"):
        validate_execution(root=ROOT, prep=tmp_path, require_authorized=False)


@pytest.mark.parametrize("other", ["optimize100", "shadow50", "test50"])
def test_validation50_question_hashes_are_disjoint(other):
    split = json.loads((ROOT / "experiments/anti_overfitting_split_v1/split_manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((ROOT / "experiments/anti_overfitting_split_v1/fold_assignment.json").read_text(encoding="utf-8"))["folds"]
    alternatives = {
        "optimize100": set(folds["fold_a"]) | set(folds["fold_b"]),
        "shadow50": set(folds["fold_c"]),
        "test50": set(split["question_hashes"]["test"]),
    }
    validation = set(split["question_hashes"]["validation"])
    assert len(validation) == 50
    assert not validation & alternatives[other]
    assert formal_validation50_policy(ROOT)["split_identity"] == "anti_overfitting_split_v1/validation"


def test_split_manifest_cannot_relabel_fold_c_as_validation50(tmp_path):
    source = ROOT / "experiments/anti_overfitting_split_v1"
    destination = tmp_path / "experiments/anti_overfitting_split_v1"
    destination.mkdir(parents=True)
    manifest = json.loads((source / "split_manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((source / "fold_assignment.json").read_text(encoding="utf-8"))
    manifest["question_hashes"]["validation"] = folds["folds"]["fold_c"]
    (destination / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (destination / "fold_assignment.json").write_text(json.dumps(folds), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        formal_validation50_policy(tmp_path)


def test_formal_policy_text_keeps_shadow_and_validation_distinct():
    text = (ROOT / "experiments/gepa_saturation_comparison_v3/POST_FREEZE_VALIDATION50_EVALUATION.md").read_text(encoding="utf-8")
    assert "fold_c = Shadow50" in text
    assert "validation = Validation50" in text
    assert "Evaluate exactly Validation50 `fold_c`" not in text


def test_native_final_team_reconstructs_selected_candidate(tmp_path):
    initial = ["P0"] * 5
    final = ["P1"] * 5
    identity = persist_final_team(
        tmp_path / "final_team_materialization.json", mode="GEPA_NATIVE",
        initial_prompts=initial, final_prompts=final,
        candidate_id="candidate-1", initial_team_hash=team_hash(initial),
        search_final_identity="candidate-1",
    )
    frozen = json.loads((tmp_path / "final_team_materialization.json").read_text(encoding="utf-8"))
    assert frozen["prompts"] == final
    assert frozen["final_team_hash"] == team_hash(frozen["prompts"]) == identity["final_team_hash"]
    assert len(frozen["prompt_hashes"]) == 5
    with pytest.raises(ValueError, match="selected candidate/final team mismatch"):
        persist_final_team(
            tmp_path / "bad.json", mode="GEPA_NATIVE",
            initial_prompts=initial, final_prompts=["P1", "P0", "P0", "P0", "P0"],
            candidate_id="candidate-1", initial_team_hash=team_hash(initial),
            search_final_identity="candidate-1",
        )
