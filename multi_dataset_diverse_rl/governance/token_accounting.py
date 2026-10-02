"""User-authorized operational reservations; not a provider billing proof.

The hash-chained, fsynced journal is authoritative. The JSON snapshot is a
derived view. An OS lock enforces one live owner and survives process crashes
by releasing automatically; unresolved reservations are charged on recovery.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading

from ..persistence.durable_io import append_jsonl, atomic_write_json

POLICY = {
    "identity": "MATH_TOKEN_ACCOUNTING_V2",
    "ledger_identity": "AUTONOMOUS_REAL_TOKEN_LEDGER_V2",
    "authorized_total": 30_000_000,
    "input_tier": "UTF8_SERIALIZED_REQUEST_PLUS_4096",
    "input_margin": 4096,
    "bound_kind": "OPERATIONAL_ACCOUNTING_BOUND_NOT_PROVIDER_BILLING_PROOF",
    "missing_invalid_untrusted_usage": "CHARGE_FULL_RESERVATION",
    "failure_without_reliable_usage": "CHARGE_FULL_RESERVATION",
    "crash_unresolved_reservation": "CHARGE_FULL_RESERVATION",
    "concurrency": 1,
}


class OperationalAbort(BaseException):
    """Cannot be swallowed by optimizer code handling ordinary exceptions."""


def serialized_request(request):
    body = dict(request)
    extra = body.pop("extra_body", {})
    if not isinstance(extra, dict) or set(extra) & set(body):
        raise OperationalAbort("EXACT_REQUEST_SERIALIZATION_FAILED")
    body.update(extra)
    try:
        return json.dumps(body, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise OperationalAbort("EXACT_REQUEST_SERIALIZATION_FAILED") from exc


def reservation(request):
    cap = request.get("max_tokens")
    if type(cap) is not int or cap <= 0:
        raise OperationalAbort("OUTPUT_HARD_CAP_REQUIRED")
    count = len(serialized_request(request))
    return dict(serialized_request_bytes=count, input_upper_bound=count + 4096,
                output_hard_cap=cap, amount=count + 4096 + cap)


def reliable_usage(result, bound):
    return (isinstance(result, dict)
            and all(type(result.get(k)) is int and 0 <= result[k] <= bound[b]
                    for k, b in (("input_tokens", "input_upper_bound"),
                                 ("output_tokens", "output_hard_cap"))))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class TokenLedger:
    def __init__(self, directory: Path, *, task_sha256: str, policy=POLICY):
        if policy != POLICY or len(task_sha256) != 64:
            raise OperationalAbort("TOKEN_ACCOUNTING_POLICY_IDENTITY_MISMATCH")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = self.directory / "events.jsonl"
        self.snapshot = self.directory / "AUTONOMOUS_REAL_TOKEN_LEDGER_V2.json"
        self.mutex = threading.RLock()
        self.owner = (self.directory / "owner.lock").open("a+b")
        self.owner.seek(0, 2)
        if self.owner.tell() == 0:
            self.owner.write(b"1")
            self.owner.flush()
        self.owner.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.owner.close()
            raise OperationalAbort("TOKEN_LEDGER_ALREADY_OWNED") from exc
        self.task_sha256 = task_sha256
        self.events = []
        self.inflight = {}
        self.totals = dict(charged_total=0, provider_reported_actual=0,
                           fallback_charged=0, input_tokens=0, output_tokens=0)
        self.groups = {k: {} for k in ("by_stage", "by_attempt", "by_role", "by_model")}
        try:
            if self.journal.exists():
                data = self.journal.read_bytes()
                if not data.endswith(b"\n"):
                    raise OperationalAbort("TOKEN_LEDGER_CORRUPTION")
                try:
                    for line in data.splitlines():
                        self._apply(json.loads(line))
                except (ValueError, KeyError, TypeError) as exc:
                    raise OperationalAbort("TOKEN_LEDGER_CORRUPTION") from exc
                if not self.events or self.events[0]["task_sha256"] != task_sha256:
                    raise OperationalAbort("TOKEN_LEDGER_AUTHORIZATION_MISMATCH")
            else:
                self._append(dict(kind="AUTHORIZE", task_sha256=task_sha256, policy=policy))
            for key in tuple(self.inflight):
                self.reconcile(key, None, outcome="CRASH_RECOVERY_FULL_CHARGE")
            self._snapshot()
        except BaseException:
            self.close()
            raise

    def _apply(self, row):
        data = {k: v for k, v in row.items() if k != "event_sha256"}
        if (row.get("event_sha256") != _digest(data)
                or row.get("sequence") != len(self.events)
                or row.get("previous_sha256") != (self.events[-1]["event_sha256"] if self.events else None)):
            raise OperationalAbort("TOKEN_LEDGER_CORRUPTION")
        if row["kind"] == "AUTHORIZE":
            if self.events or row["policy"] != POLICY:
                raise OperationalAbort("TOKEN_LEDGER_CORRUPTION")
        elif row["kind"] == "RESERVE":
            if row["reservation_id"] in self.inflight or row["bound"]["amount"] <= 0:
                raise OperationalAbort("TOKEN_LEDGER_CORRUPTION")
            self.inflight[row["reservation_id"]] = row
        elif row["kind"] == "CHARGE":
            old = self.inflight.pop(row["reservation_id"], None)
            if old is None or row["charged"] < 0 or row["charged"] > old["bound"]["amount"]:
                raise OperationalAbort("TOKEN_LEDGER_CORRUPTION")
            if row["input_tokens"] + row["output_tokens"] != row["charged"]:
                raise OperationalAbort("TOKEN_LEDGER_CORRUPTION")
            charge = {"charged_total": row["charged"],
                      "provider_reported_actual": row["charged"] if row["reliable_usage"] else 0,
                      "fallback_charged": 0 if row["reliable_usage"] else row["charged"],
                      "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"]}
            for k, n in charge.items():
                self.totals[k] += n
            for group, field in (("by_stage", "stage"), ("by_attempt", "attempt_id"),
                                 ("by_role", "role"), ("by_model", "model")):
                values = self.groups[group].setdefault(old[field], {**dict.fromkeys(charge, 0), "physical_attempts": 0})
                values["physical_attempts"] += 1
                for k, n in charge.items():
                    values[k] += n
        else:
            raise OperationalAbort("TOKEN_LEDGER_CORRUPTION")
        self.events.append(row)
        if self.totals["charged_total"] + self.reserved > POLICY["authorized_total"]:
            raise OperationalAbort("TOKEN_LEDGER_CORRUPTION")

    def _append(self, payload):
        row = {**payload, "sequence": len(self.events),
               "previous_sha256": self.events[-1]["event_sha256"] if self.events else None}
        row["event_sha256"] = _digest(row)
        append_jsonl(self.journal, row)
        self._apply(row)
        self._snapshot()

    @property
    def reserved(self):
        return sum(r["bound"]["amount"] for r in self.inflight.values())

    @property
    def remaining(self):
        return POLICY["authorized_total"] - self.totals["charged_total"] - self.reserved

    def view(self):
        return dict(identity=POLICY["ledger_identity"], task_sha256=self.task_sha256,
                    authorized_total=POLICY["authorized_total"], **self.totals,
                    reserved_inflight=self.reserved, remaining=self.remaining,
                    **self.groups, last_event_sha256=self.events[-1]["event_sha256"])

    def _snapshot(self):
        atomic_write_json(self.snapshot, self.view())

    def reserve(self, request, *, attempt_id, stage, role, model, protected_validation=0):
        with self.mutex:
            bound = reservation(request)
            if self.inflight:
                raise OperationalAbort("TOKEN_LEDGER_CONCURRENCY_VIOLATION")
            if bound["amount"] > self.remaining:
                raise OperationalAbort("STOP_TOKEN_BUDGET_EXHAUSTED")
            if type(protected_validation) is not int or protected_validation < 0:
                raise OperationalAbort("VALIDATION_RESERVE_INVALID")
            if self.remaining - bound["amount"] < protected_validation:
                raise OperationalAbort("STOP_TOKEN_BUDGET_INSUFFICIENT_FOR_VALIDATION")
            key = str(len(self.events))
            self._append(dict(kind="RESERVE", reservation_id=key, bound=bound,
                request_sha256=hashlib.sha256(serialized_request(request)).hexdigest(),
                attempt_id=attempt_id, stage=stage, role=role, model=model,
                protected_validation=protected_validation))
            return key

    def reconcile(self, key, result, *, outcome):
        with self.mutex:
            bound = self.inflight[key]["bound"]
            reliable = reliable_usage(result, bound)
            inp = result["input_tokens"] if reliable else bound["input_upper_bound"]
            out = result["output_tokens"] if reliable else bound["output_hard_cap"]
            self._append(dict(kind="CHARGE", reservation_id=key, reliable_usage=reliable,
                input_tokens=inp, output_tokens=out, charged=inp + out, outcome=outcome))
            return dict(input_tokens=inp, output_tokens=out, usage_reliable=reliable)

    def close(self):
        self.owner.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
