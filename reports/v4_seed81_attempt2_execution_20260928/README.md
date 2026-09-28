# Seed81 V4 attempt2: aborted execution and validity audit

The user authorized this exact real API attempt. It ran once from frozen
execution source `85812a7d891e6a2c3bfdca00a1cb4d14074735a4` after
authorization, startup identity, GEPA, source closure, split, and fresh-root
checks passed. The attempt-local grant was consumed. No second run was started.

The process exited with `DiagnosticSamplingIntegrityError: diagnostic local
acceptance exceeds capacity` at update index 3. Official GEPA lineage records
local accepted-event counts of **0, 1, 1, 2** for updates 0–3. The frozen
diagnostic controller assumes at most one accepted event per 36-call local
opportunity. It stopped before TeamMiniBatch or Full for update 3. This is an
execution/conformance failure; it is not a negative scientific result.

The lifecycle is `ABORTED`, and no `execution_summary.json` or complete
trajectory/evidence trace was written. The raw lifecycle, ledger, GEPA lineage,
system artifacts, and private console log remain under the ignored attempt2
run directory. A sibling `raw_evidence_freeze.json` records SHA256 and size for
all **94** original files, including prep authorization and the private log;
its SHA256 is
`767edd031a57c5e6957b232a43cbcd229fa7c35bcbb6103a0099d7011394a319`.
A read-back check found zero hash mismatches. None of those raw artifacts are
copied into this report.

The durable ledger records **337 physical attempts, 337 successes, zero failed
attempts, 538 cache hits, and 247,685 total recorded tokens**. It has no
Validation50 or Test50 phase. The terminal summary is absent, so its split
counters cannot be independently reconciled. Attempts stayed below the frozen
1200-success / 4800-attempt emergency limits. The frozen read-only auditor
returned `HOLD` because the execution summary is absent.

Classification: **scientific validity `INVALID`; efficacy `NOT_EVALUABLE`**.
The V4 pilot remains open, and the prerequisite for Formal V3 real execution
is unmet. An engineering repair may be prepared without changing V4 scientific
semantics, but a new real attempt requires a new explicit user authorization.
