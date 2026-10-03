"""Append-only authorization extension; reservation physics and charges stay V2."""
import json
import pytest
from multi_dataset_diverse_rl.governance.token_accounting import (
    TokenLedger, POLICY_40M, OperationalAbort,
)

TASK = "a" * 64
AMENDMENT = "b" * 64


def test_extension_preserves_journal_charges_and_replays(tmp_path):
    with TokenLedger(tmp_path, task_sha256=TASK) as ledger:
        request = dict(model="synthetic", max_tokens=3600, messages=[])
        key = ledger.reserve(request, attempt_id="old", stage="initial",
                             role="solver", model="synthetic")
        ledger.reconcile(key, dict(input_tokens=20, output_tokens=10), outcome="OK")
        prefix = ledger.journal.read_bytes()
        ledger.amend_authorization_40m(authorization_sha256=AMENDMENT,
                                      expected_charged_total=30)
        assert ledger.journal.read_bytes().startswith(prefix)
        assert ledger.view()["charged_total"] == 30
        assert ledger.remaining == 39_999_970
    with TokenLedger(tmp_path, task_sha256=TASK, policy=POLICY_40M) as ledger:
        assert ledger.view()["authorized_total"] == 40_000_000
        assert ledger.view()["provider_reported_actual"] == 30
        assert ledger.view()["authorization_amendments"][0]["authorization_sha256"] == AMENDMENT
    with pytest.raises(OperationalAbort, match="AMENDMENT_MISMATCH"):
        TokenLedger(tmp_path, task_sha256=TASK)


def test_extension_cannot_create_fresh_ledger_or_erase_charges(tmp_path):
    with pytest.raises(OperationalAbort, match="ORIGINAL_JOURNAL"):
        TokenLedger(tmp_path / "new", task_sha256=TASK, policy=POLICY_40M)
    with TokenLedger(tmp_path / "old", task_sha256=TASK) as ledger:
        before = ledger.journal.read_bytes()
        with pytest.raises(OperationalAbort, match="AMENDMENT_MISMATCH"):
            ledger.amend_authorization_40m(authorization_sha256=AMENDMENT,
                                          expected_charged_total=1)
        assert ledger.journal.read_bytes() == before
        ledger.amend_authorization_40m(authorization_sha256=AMENDMENT,
                                      expected_charged_total=0)
        with pytest.raises(OperationalAbort, match="AMENDMENT_MISMATCH"):
            ledger.amend_authorization_40m(authorization_sha256=AMENDMENT,
                                          expected_charged_total=0)


def test_extension_keeps_hard_ceiling(tmp_path):
    with TokenLedger(tmp_path, task_sha256=TASK) as ledger:
        ledger.amend_authorization_40m(authorization_sha256=AMENDMENT,
                                      expected_charged_total=0)
        request = dict(model="synthetic", max_tokens=39_990_000, messages=[])
        key = ledger.reserve(request, attempt_id="new", stage="initial",
                             role="solver", model="synthetic")
        ledger.reconcile(key, None, outcome="UNKNOWN_USAGE")
        with pytest.raises(OperationalAbort, match="BUDGET_EXHAUSTED"):
            ledger.reserve(request, attempt_id="new", stage="retry",
                           role="solver", model="synthetic")
        assert ledger.view()["charged_total"] <= 40_000_000
