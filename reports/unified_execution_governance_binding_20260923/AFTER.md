# After: governed unified entry and fresh GEPA canary freeze

| Question | Answer |
| --- | --- |
| Did unified `run_experiment` become governed? | YES |
| Can provider construction occur before authorization admission? | NO |
| Can the new runner fall back to `myx`? | NO |
| Can source, model, or provider mismatch reach the provider? | NO |
| Does the real production path depend on historical runners? | NO |
| Did the scientific method change? | NO |

The unified CLI now consumes a disk-serialized canonical startup bundle,
replays source and dependency identity, checks the `lwj` provider/model binding
and one-time authorization, then admits a fresh run before creating any
network-capable client. An unauthorized execution of the frozen attempt was
rejected before a formal run root, consumed marker, or ledger existed.

The canary is a technical local-empirical-path check through the current
Layer-2 controller and official GEPA backend. It stops before team evaluation,
write-back, and any Validation50/Test50 call. A fake-provider full-path test
reached reflection, a changed contract-valid proposal, and candidate Solver
evaluation. The eight scientific golden traces remained exact. No real API was
called. The existing 33 historical full-suite failures are unchanged.

The new attempt is **`PREREGISTERED_NOT_EXECUTED`** and
**`READY_FOR_AUTHORIZATION=true`**. It is **not authorized to run**; a separate
one-time API authorization is required. Previous freezes and aborted attempts
remain separate historical evidence and were not resumed or reused.
