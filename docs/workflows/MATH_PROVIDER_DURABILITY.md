# MATH provider durability

`JOURNAL_FIRST_PROVIDER_DURABILITY_V1` is an explicitly bound operational
policy. It changes no scientific request, generation policy, cache realization,
accounting reservation rule, search decision, or method identity.

The single owner reserves durably before transport. After receiving a result,
it writes a private immutable receipt containing the request, original response
and usage, reservation ID, attempt, and startup identity. The receipt is flushed
and fsynced before its seal enters the authoritative accounting CHARGE event.
Successful HTTP response bodies and malformed HTTP evidence are retained.
Scientific evaluation starts only after the receipt and charge are durable.

The hash-chained accounting journal remains authoritative. Token snapshots
are derived: an I/O failure detaches snapshot updates for that owner and never
regenerates a paid response. Monitoring reads complete journal lines, closes
handles promptly, and does not read live replacement targets. Other critical
state writes continue to fail closed. Unique receipt/checkpoint destinations
are immutable; readers cannot cause destination-replacement conflicts.

Recovery replays the existing journal and preserves its byte prefix. Already
charged reservations are not charged again. A response receipt whose reservation
was not finalized remains forensic evidence; the existing unresolved-reservation
full-charge rule still applies. No receipt, cache, checkpoint, or replay grants
scientific continuation of an invalid attempt. Every operational retry receives
a fresh attempt, namespace, empty Memory, initial team, and authorization.

Frozen historical bindings retain the strict snapshot behavior. A fresh
operational Pilot binding copies every scientific field from the checked parent
and admits only new execution identities, exact user scope, and this persistence
policy. Provider-visible bytes and member-lane identities have offline witnesses.
