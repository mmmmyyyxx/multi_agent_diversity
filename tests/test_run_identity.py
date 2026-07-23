from dataclasses import replace

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.identity import RunIdentity, config_fingerprint, validate_run_identity


def identity(fingerprint):
    return RunIdentity(
        method_version="member_aware_peer_state_v1",
        experiment_setting="shared_member_aware_full",
        git_commit="commit",
        git_dirty=False,
        config_fingerprint=fingerprint,
        manifest_sha256="manifest",
        train_file_sha256="train",
        val_file_sha256="val",
        test_file_sha256="test",
        train_question_set_hash="train-q",
        val_question_set_hash="val-q",
        test_question_set_hash="test-q",
    )


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"agent_model": "different-model"},
        {"vote_tie_break": "first"},
        {"local_accuracy_loss_epsilon": 0.1},
        {"solver_output_contract_version": "different-contract"},
        {"shared_solver_cache_path": "different-cache.sqlite"},
    ],
)
def test_behavioral_config_changes_fingerprint_and_reject_resume(override):
    baseline = config_fingerprint(Config())
    changed = config_fingerprint(Config.from_flat(**override))
    assert baseline != changed
    with pytest.raises(ValueError, match="Run identity mismatch"):
        validate_run_identity(identity(baseline), identity(changed).to_dict())


def test_split_sha_mismatch_rejects_resume():
    expected = identity("same")
    actual = replace(expected, val_file_sha256="different")
    with pytest.raises(ValueError, match="val_file_sha256"):
        validate_run_identity(expected, actual.to_dict())
