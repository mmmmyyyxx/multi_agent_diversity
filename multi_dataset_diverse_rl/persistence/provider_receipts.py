"""Private immutable transport receipts, durable before accounting finalization.

These receipts are operational evidence, never a cache or an experiment resume
capability. A recovered incomplete attempt still requires a fresh scientific run.
"""
import hashlib
import json
from pathlib import Path

from .durable_io import atomic_write_json, read_json


IDENTITY = "JOURNAL_FIRST_PROVIDER_DURABILITY_V1"
POLICY = dict(identity=IDENTITY, response_before_charge=True,
    derived_token_snapshot="best_effort_detach_on_io_failure",
    unresolved_reservation="CHARGE_FULL_RESERVATION",
    scientific_resume=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class ProviderResponseReceipts:
    def __init__(self, directory, *, attempt_id, startup_identity_sha256):
        self.directory = Path(directory)
        self.attempt_id = attempt_id
        self.startup = startup_identity_sha256
        self.directory.mkdir(parents=True, exist_ok=True)

    def persist(self, reservation_id, record):
        if not isinstance(reservation_id, str) or not reservation_id.isdecimal():
            raise ValueError("PROVIDER_RECEIPT_RESERVATION_INVALID")
        payload = dict(identity=IDENTITY, attempt_id=self.attempt_id,
            startup_identity_sha256=self.startup, reservation_id=reservation_id,
            record=record)
        seal = digest(payload)
        target = self.directory / (reservation_id + ".json")
        row = dict(payload, integrity_sha256=seal)
        if target.exists():
            if read_json(target) != row:
                raise ValueError("PROVIDER_RECEIPT_IMMUTABLE_CONFLICT")
        else:
            atomic_write_json(target, row)
        return dict(identity=IDENTITY, reservation_id=reservation_id,
            attempt_id=self.attempt_id, integrity_sha256=seal)

    def read(self, reservation_id):
        row = read_json(self.directory / (reservation_id + ".json"))
        payload = {k: v for k, v in row.items() if k != "integrity_sha256"}
        if (digest(payload) != row.get("integrity_sha256")
                or payload.get("identity") != IDENTITY
                or payload.get("reservation_id") != reservation_id
                or payload.get("attempt_id") != self.attempt_id
                or payload.get("startup_identity_sha256") != self.startup):
            raise ValueError("PROVIDER_RECEIPT_CORRUPTION")
        return payload["record"]
