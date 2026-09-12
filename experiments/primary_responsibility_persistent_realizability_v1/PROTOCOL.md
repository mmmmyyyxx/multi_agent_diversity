# Primary Responsibility + Persistent Realizability v1

Status: `IMPLEMENTED_NOT_RUN`.

This paired prospective experiment changes only Layer 2 target allocation.
Arm A retains the current GEPA scheduler: strict direct-flip, near-margin,
coverage hierarchy with lane-local round robin. Arm B selects the two highest
distinct member scores under:

```text
V_i = max(4 D_i, 2 N_i, C_i)
R_i = 1 / (1 + f_i)
TargetScore_i = V_i R_i
```

Within-member ties use direct-flip, near-margin, coverage order. Equal target
scores use deterministic stateful round robin. If every score is zero, the
existing deterministic fallback round robin is used.

The compute contract is `always_two_targets`: when exactly one member has a
positive score, the second slot is still filled by the deterministic RR
ordering over zero-score members. This preserves the frozen dual-target
compute structure rather than introducing a second simultaneous intervention.

The responsibility snapshot and assignments are frozen once at the beginning
of each opportunity. A selected member's valid completed no-commit outcome
increments `f_i`; its own successful commit resets `f_i` to zero. Unselected
members, teammate commits, team-state hash changes, and operational aborts do
not change `f_i`.

Here realizability means **eventual write-back realizability**. A feasible
branch that loses cross-branch competition is a no-write-back outcome for that
member and increments its failure count. It is not interpreted as an isolated
measure of prompt repairability.

The opt-in online binding freezes both assignments from one scheduler decision,
runs both branches from the same request/parent before mutation, applies the
unchanged Common-Safe cross-branch choice, shadows only the global winner, and
commits at most one prompt. It durably records one scheduler outcome for every
completed, no-commit, or operational-abort opportunity. Duplicate outcome
recording for an update index fails closed, including after checkpoint resume.

Both arms freeze identical initial teams, splits, seeds, official GEPA, local
optimizer budget, Solver contract, TeamMiniBatch12, full-team evaluation,
Common-Safe, Shadow, and commit policy. The primary hypotheses are reduced
target/commit mismatch, lower tokens per commit, and no systematic team Vote
degradation. Risk telemetry is observational and cannot modify the scheduler.

API, Validation, and Test50 are unauthorized by this preregistration until the
user separately authorizes a frozen execution.
