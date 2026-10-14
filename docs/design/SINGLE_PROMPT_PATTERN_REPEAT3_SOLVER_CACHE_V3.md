# Single Prompt Gradient–Pattern: three fixed-seed repeats

This explicit user amendment changes the default from five to three repeats per
arm. It supersedes the unexecuted five-repeat preparation, which remains immutable
at its original manifest and source. No authorization, cache, checkpoint, paid
response or execution scope from that preparation is imported.

All scientific and operational rules of
[the Solver-only cache protocol V2](SINGLE_PROMPT_PATTERN_REPEAT_SOLVER_CACHE_V2.md)
are preserved except these frozen counts:

- A/B/C each use three repeats, all seed42, repeat IDs1–3.
- Order is A1–A3, B1–B3, C1–C3, sequential.
- Each of nine cells has twenty opportunities:180 total,189 checkpoints.
- Successful semantic responses plus in-flight attempts:training190440,
  final Test10800. Network-failed physical attempts and total Tokens remain uncapped.
- Final Test evaluates nine retained Prompts on the same formal-aligned Test300.
- Within-arm repeat comparisons have three pairs per arm,nine total.

Optimize150, the same Validation100 membership seed88, strict gain of one correct
answer, initial-plus-three generation retries, largest unique-support Pattern
with Seed+round ties and supported-only evidence, and raw-Batch B fallback remain.
Solver observations share only within the same repeat across arms/rounds, with
split/phase isolation. All other roles call providers fresh. No historical cache
or response replay is allowed. Training Solver32/Gradient8, final TestSolver16,
models, SDK2.37.0, limits, scoring and indefinite typed network recovery remain.

Registration and fresh source/data/phase freeze precede execution. The user's
explicit three-repeat amendment and execution authorization applies to the newly
derived nine-cell scope with the reduced ceilings. It is not a reusable grant
for the superseded five-repeat scope or another attempt. Conditional Test starts
only after all nine training cells, source/access/checkpoint/receipt/ledger checks
and final Prompt selection have passed. The requested monitor interval is two
hours; monitoring is read-only and cannot change the frozen runner's policy.
Historical Test exposure and first-dispatch cache cost allocation are disclosed
as in V2. Runtime evidence stays private; publication contains sanitized metrics.
