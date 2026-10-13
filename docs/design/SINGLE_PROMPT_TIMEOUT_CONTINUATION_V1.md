# Timeout continuation for the independent single-Prompt study

The user explicitly authorized continuation and future timeout recovery without
repeated authorization. This operational amendment applies only to the independent
three-arm study's training phase. It supersedes that study's original timeout-abort
and no-continuation prerequisite when a fresh continuation scope names this policy.
Original manifests, consumed scopes, journals, responses and reports stay immutable.
The current Unified Team Prompt Search transport and historical team protocols
retain their own rules. No Validation/Test authorization is added.

The scientific study is unchanged: seeds88/89/90, sequential cells,20 rounds,
Optimize150, initial Arm B Prompt, Batch3, Solver16 and independent Gradient8,
same models/decoding/context/output limits, structural recovery and unconditional
replacement. No accuracy gate, outcome-driven scheduling or prompt selection is added.

## Timeout recovery

Only typed OpenAI `APITimeoutError` is recoverable. Each physical send uses the
same120-second timeout, request bytes, role and semantic draw. Failed attempts
receive durable private receipts and conservative full-reservation charges before
retrying. Backoff is2/4/8/16/30/60 seconds, then60 seconds. Retry stops on a successful
response, non-timeout fault, integrity/access/persistence failure or the finite
combined114480-call ceiling. All old and new physical calls count toward that same
ceiling; there is no cumulative Token hard ceiling. SDK automatic retries remain
disabled. Timeout retry does not consume a successful semantic draw or generate
another optimizer candidate. Transport success still passes every output guard.

## Durable prefix continuation

Freeze a new source SHA, operational manifest, command, fresh destination and
unique cell attempt IDs. A closed timeout-aborted training parent must have its
entire immutable inventory frozen, audited journal, resolved reservations, sealed
response receipts, exact model/data/scientific policy and valid original authorization.
Only timeout failures are eligible. The original pre-amendment migration is pinned
to its known source SHA; other historical runs are forbidden.

Re-enter the deterministic study loop with the original initial Prompt. Replay
successful responses in their original draw order, keyed by exact request bytes,
Seed, round, role and stage. Never replay failed transports or choose responses by
correctness. Preserve the root parent's request namespace so optimizer evidence
bytes remain identical, while all new sends use the new authorization and receipt
attempt IDs. Recompute deterministic scoring and rebuild checkpoints from those
same responses; retain the exact already-generated candidate, including a malformed
draw if one occurred. Only missing successful draws reach the provider. Replay is
logged with source receipt provenance and is never charged twice. A mismatch or
unused parent response is fatal. Completed checkpoints are compared during the
owner's prefix audit. Ancestor calls and Token charges remain visible beside the
new journal. No previous artifact is modified and no consumed scope is reopened.

This is an explicit evidence-continuation lineage, not a claim that every response
was generated under the new operational source. Report original and resumed source
identities and costs separately. A future process restart after a closed timeout
failure may create a fresh derived scope under the same human timeout-continuation
grant; source/data/budgets and prefix audits remain mandatory. Other crashes or
provider errors are not covered. Hourly monitoring reads only complete journal
prefixes and recovery logs. It does not initiate an unrelated run or held-out phase.

## Verification

Fake-provider tests must cover successful prefix replay, preserved optimizer draws,
partial Solver checkpoint repair, typed timeout retries, successful semantic draw
counts, each physical charge, combined ceilings, unrelated failure rejection,
inventory corruption and request mismatch before new sends. Before execution run
production compileall, the classified current suite, governance/manifest checks,
deterministic study smoke, sanitization and diff checks. Freeze and authorize the
exact new handoff under the human continuation instruction.
