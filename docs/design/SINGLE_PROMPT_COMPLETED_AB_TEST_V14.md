# Historical A/B Test completion with bounded audit V14

This explicitly authorized execution repair derives from
[V13](SINGLE_PROMPT_COMPLETED_AB_TEST_V13.md), whose fresh attempt stopped with
`MemoryError` during parent checkpoint audit before any provider request.
Its failed artifacts and single-use authorization remain immutable. V14 has a
fresh source, manifest, scope, authorization evidence, command and run directory.
The direct human instruction is to repair and finish the remaining experiment.

All V13 scientific read points are unchanged: six historical A/B final Prompts,
original MATH Test300, Seed42, three repeat IDs, gpt-4o-mini, Solver16, SDK2.37.0,
identical decoding/capacity, exact certified-negative scorer and thirty-second
local worker. There is no training, Validation, C evaluation or new candidate.
The successful semantic/in-flight ceiling is7200, failed network calls and total
Tokens are uncapped, and existing accounted network and same-paid-answer local
timeout recovery remain unchanged. No old Test/cache is imported.

`BOUNDED_ADJACENT_CHECKPOINT_AUDIT_V1` replaces trajectory-wide JSON retention
with a sequential audit holding at most two complete checkpoints. It still
checks every original integrity hash, cell/scope identity, Boolean observation,
membership, single-block edit, strict Validation decision, retained state,
history, metrics, final counts and per-cell ledger totals. Prompt/metric summaries
are small retained values; complete historical answer graphs are released.
Verified journal events are released before reading checkpoint graphs.

Each executing process performs one full historical semantic audit before any
provider is constructed. This produces an in-process snapshot of validated
parent metadata, final results and the complete frozen file inventory. Every
subsequent authorization/reference/cell/closure boundary recomputes all parent
and source file hashes and the full frozen scope, instead of reparsing the
unchanged historical checkpoint graphs. A same-size, same-mtime modification
still fails. No persisted audit cache or earlier-process assertion bypasses the
full semantic audit. Any changed or missing input fails closed.

Offline verification must compare the original frozen V13 audit's returned
scope, results and inventory against V14 on unchanged historical artifacts;
measure process memory separately; test all preserved semantic checks and
boundary mutations; and run the six-cell fake execution without network.
The new private handoff pins the old zero-call failure evidence. Completing
Test still does not create a matched causal comparison to corrected C or erase
historical Test exposure. Report actual results and phase costs separately.
