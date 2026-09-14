# Seed78 GEPA Differential Audit

This is a zero-API, read-only comparison of the completed Seed78 local GEPA
searches against the successful Independent-GEPA capacity probe.

## Result

`LOCAL_GEPA_PROPOSER_MUTABLE_CONTRACT_MISMATCH_CONFIRMED`

- The official GEPA commit, Solver model, reflection model, correctness metric,
  100-example search count, and reflection minibatch size match.
- Diversity generated **89/89 byte-changed proposals**, but **all 89 were
  rejected before a candidate Solver call**. Candidate Solver calls were `0`.
- Production-order rejection counts were: `3` compact-length failures, `82`
  output-contract contamination failures, and `4` append-only mutations.
- Independent-GEPA produced `66` proposals and retained `29` mutations. All
  `29/29` retained mutations would fail the current Diversity mutable-prompt
  boundary (`27` compact-length, `2` output-contract contamination).

The immediate divergence is therefore not evidence that qwen3-8b cannot be
optimized, nor evidence that TeamMiniBatch rejects useful local mutations.
The default GEPA proposer emits complete prompts, while the Diversity Layer-1
interface accepts only compact decision-procedure replacements and rejects
duplicated output-interface text. No Seed78 proposal reached empirical candidate
evaluation, so validation-size, merge, and skip-perfect differences remain
secondary, untested explanations.

## Frozen interpretation

- Scheduler implementation: verified.
- Scheduler causal efficacy: **NOT EVALUATED**.
- Local-to-TeamMiniBatch transfer: **NOT EVALUATED**.
- Primary blocker: **proposer / mutable-prompt contract mismatch**.

The hard boundary should not be loosened. A future change should constrain the
proposer to decision-procedure-only replacement text, prove nonzero Solver reach
offline, and then use a minimal fresh canary before any scheduler A/B rerun.

API calls: `0`; Validation calls: `0`; Test calls: `0`.
