# Solver stage-attribution closure

Status: **PASS — ZERO API**.

The deterministic `KeyError: phase` was an attribution-contract failure at the
TeamMiniBatch producer boundary. It was not a GEPA search failure: the preserved
aborted attempt already established real proposal generation, candidate Solver
reach, positive local delta, and one accepted local mutation. The run remains
`ABORTED`; no full-path scientific conclusion is reported.

The repair introduces one canonical, fail-closed Solver stage schema. `phase` is
authoritative. Every Solver-capable producer supplies it explicitly; when the
legacy diagnostic field `evaluation_stage` is present it must equal `phase`.
Missing, empty, unknown, or divergent attribution fails before provider or cache
access. No consumer fallback exists.

Initialization, Local GEPA Solver evaluation, TeamMiniBatch, Full, Shadow, and
final-validation producers were audited. Cache-hit and cache-miss tests pass,
the fake-provider positive path reaches TeamMiniBatch, Full, Shadow, and exactly
one commit, and phase/token ledger arithmetic reconciles exactly.

No scientific setting changed. The official GEPA checkout, Level-B ownership,
reflection representation, mutable component, local budget, scheduler,
TeamMiniBatch, Common-Safe, Shadow, models, and data identities remain frozen.

A fresh attempt identity is prepared but not authorized. Its formal run root has
not been created. Starting it requires a new explicit API authorization.

`API_CALLS=0`, `VALIDATION_CALLS=0`, `TEST_CALLS=0`.
