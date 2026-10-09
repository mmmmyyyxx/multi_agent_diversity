"""Fresh Stage 1 accounting; no current-search policy or old ledger is changed."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .math_baseline_calibration import digest
from ..governance.token_accounting import reservation, reliable_usage, serialized_request
from ..persistence.durable_io import append_jsonl, atomic_write_json
import hashlib

IDENTITY = 'FRESH_MATH_BASELINE_STAGE1_ACCOUNTING_V1'
CAP = 1_450_000
MAX_TRANSPORT = 15_120


class CalibrationAbort(RuntimeError):
    """Closed category only; provider messages never reach public stdout."""


def replay_events(rows):
    charged = reported = fallback = transports = 0
    inflight = {}
    previous = None
    terminal = None
    scope = None
    for sequence, row in enumerate(rows):
        payload = {k: v for k, v in row.items() if k != 'event_sha256'}
        if (digest(payload) != row.get('event_sha256') or row.get('sequence') != sequence
                or row.get('previous_sha256') != previous):
            raise CalibrationAbort('CALIBRATION_LEDGER_CORRUPT')
        previous = row['event_sha256']
        kind = row['kind']
        if kind == 'AUTHORIZE':
            if sequence != 0 or row['policy_identity'] != IDENTITY or row['cap'] != CAP:
                raise CalibrationAbort('CALIBRATION_LEDGER_SCOPE_MISMATCH')
            scope = row['scope_sha256']
        elif kind == 'RESERVE':
            key = row['reservation_id']; bound = row['bound']
            if (scope is None or terminal or inflight or key != str(sequence)
                    or type(bound['amount']) is not int or bound['amount'] <= 0
                    or charged + bound['amount'] > CAP or transports >= MAX_TRANSPORT):
                raise CalibrationAbort('CALIBRATION_LEDGER_ADMISSION_MISMATCH')
            inflight[key] = row; transports += 1
        elif kind == 'CHARGE':
            key = row['reservation_id']
            if key not in inflight or terminal:
                raise CalibrationAbort('CALIBRATION_LEDGER_CHARGE_MISMATCH')
            bound = inflight.pop(key)['bound']
            value = row['charged']
            if (type(value) is not int or not 0 <= value <= bound['amount']
                    or type(row['reliable_usage']) is not bool
                    or not row['reliable_usage'] and value != bound['amount']):
                raise CalibrationAbort('CALIBRATION_LEDGER_CHARGE_MISMATCH')
            charged += value
            if row['reliable_usage']: reported += value
            else: fallback += value
        elif kind == 'TERMINAL':
            if terminal or inflight or row['status'] not in {'EXECUTION_COMPLETE', 'EXECUTION_ABORTED'}:
                raise CalibrationAbort('CALIBRATION_LEDGER_TERMINAL_MISMATCH')
            terminal = row['status']
        else:
            raise CalibrationAbort('CALIBRATION_LEDGER_EVENT_UNKNOWN')
        if charged + sum(r['bound']['amount'] for r in inflight.values()) > CAP:
            raise CalibrationAbort('CALIBRATION_LEDGER_BUDGET_OVERRUN')
    return dict(policy_identity=IDENTITY,authorized_total=CAP,charged_total=charged,
        provider_reported_reliable_tokens=reported,unknown_usage_charge=fallback,
        reserved_total=sum(r['bound']['amount'] for r in inflight.values()),
        physical_attempts=transports,inflight=inflight,terminal=terminal,
        scope_sha256=scope,last_event_sha256=previous,event_count=len(rows))


class CalibrationLedger:
    def __init__(self, directory, scope_sha256):
        self.directory = Path(directory)
        if self.directory.exists():
            raise CalibrationAbort('FRESH_CALIBRATION_ACCOUNTING_REQUIRED')
        self.directory.mkdir(parents=True)
        self.owner = (self.directory / 'owner.lock').open('a+b')
        self.owner.write(b'1'); self.owner.flush(); self.owner.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.events = []; self.snapshot_detached = False; self.journal_failed = False
            self._append(dict(kind='AUTHORIZE',policy_identity=IDENTITY,cap=CAP,
                scope_sha256=scope_sha256,concurrency=1,old_scope_reuse=False))
        except BaseException:
            self.owner.close(); raise

    def _append(self, payload):
        if self.journal_failed:raise CalibrationAbort('CALIBRATION_JOURNAL_PERSISTENCE_FAILED')
        row = dict(payload,sequence=len(self.events),
            previous_sha256=self.events[-1]['event_sha256'] if self.events else None)
        row['event_sha256'] = digest(row)
        state = replay_events([*self.events,row])
        try:append_jsonl(self.directory / 'events.jsonl',row)
        except BaseException:
            self.journal_failed=True;raise
        self.events.append(row); self.state = state
        if not self.snapshot_detached:
            try: atomic_write_json(self.directory / 'ledger_snapshot.json',state)
            except OSError: self.snapshot_detached = True
        return row

    def reserve(self, request, lane):
        bound = reservation(request)
        if (self.state['charged_total'] + self.state['reserved_total'] + bound['amount'] > CAP
                or self.state['physical_attempts'] >= MAX_TRANSPORT):
            raise CalibrationAbort('STOP_CALIBRATION_RESOURCE_CEILING')
        key = str(len(self.events))
        self._append(dict(kind='RESERVE',reservation_id=key,bound=bound,lane=lane,
            wire_sha256=hashlib.sha256(serialized_request(request)).hexdigest()))
        return key

    def charge(self, key, usage, outcome, receipt=None):
        bound = self.state['inflight'][key]['bound']
        reliable = reliable_usage(usage,bound)
        value = usage['input_tokens'] + usage['output_tokens'] if reliable else bound['amount']
        self._append(dict(kind='CHARGE',reservation_id=key,reliable_usage=reliable,
            charged=value,outcome=outcome,response_receipt=receipt))
        return value, reliable

    def terminal(self, status):
        # A persistence/process anomaly cannot create a free unresolved call.
        for key in tuple(self.state['inflight']):
            self.charge(key,None,'UNRESOLVED_FULL_RESERVATION')
        self._append(dict(kind='TERMINAL',status=status))

    def close(self):
        self.owner.close()


def read_ledger(directory):
    data = (Path(directory) / 'events.jsonl').read_bytes()
    if not data.endswith(b'\n'): raise CalibrationAbort('CALIBRATION_LEDGER_TORN_JOURNAL')
    return replay_events([json.loads(line) for line in data.splitlines()])
