# Primary Responsibility Online Binding

Status: `IMPLEMENTED_NOT_RUN`.

The opt-in Layer-2 lifecycle is now wired as:

```text
scheduler.select
-> two frozen TeamSearchAssignments
-> TeamSearchController.run_frozen_opportunity
-> one global Common-Safe winner
-> winner-only Shadow
-> at most one commit
-> scheduler.record_outcome exactly once
```

Both target branches are evaluated before any Shadow call or prompt-team
mutation. The existing one-target controller entry point delegates to the same
implementation and retains its previous behavior. Local GEPA, TeamMiniBatch12,
full-team evaluation, Common-Safe constraints/ranking, Shadow policy, and commit
criteria are unchanged.

The frozen target-count contract is `always_two_targets`. If only one member
has a positive score, deterministic RR fills the second zero-score slot. This
preserves dual-target compute parity for the first prospective A/B.

Realizability is explicitly `eventual_write_back_realizability_v1`. A selected
branch that does not write back, including a feasible branch that loses global
competition, increments that member's failure count. Only that member's own
commit resets it. Operational aborts leave counters unchanged. Every update
index receives one durable outcome marker; duplicate recording fails closed,
and the marker survives checkpoint/resume.

Zero-API tests cover commit, valid no-commit, operational abort, duplicate
outcome rejection, checkpoint persistence, two-target compute parity, primary
lane propagation, single-target compatibility, and winner-only global Shadow.

No prospective execution is frozen: Solver/optimizer models, seeds, data
identity, update budget, and API roles remain unset. Consequently this report
makes no efficacy, Vote, commit-efficiency, or token-efficiency claim.

API calls: 0. Validation calls: 0. Test calls: 0. Historical artifacts
modified: 0.

Verification: 25 focused tests passed. The full suite reported 991 passes and
the one known historical-cache coverage failure. Compileall, governance
preflight, binding preflight, sanitization, SHA256 verification, and
`git diff --check` passed.
