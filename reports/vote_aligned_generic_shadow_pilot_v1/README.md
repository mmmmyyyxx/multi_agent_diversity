# Vote-Aligned Generic Shadow Pilot v1

Phase A is complete and freezes a zero-API paired experiment. Both arms use the
existing Shadow-gated D2 Generic pipeline. P1 changes only target scheduling to
the deterministic hierarchy `direct_flip > near_margin > pure_coverage`, with
canonical actionable-member round robin as fallback.

All ten Phase A gates passed. The user then narrowed Phase B to Seed75 and
authorized its P0 then P1 pair. The execution was stopped by the user after P0
began but before the first complete trajectory checkpoint existed. P1 never
started. The run has therefore been classified as
`USER_STOPPED_BEFORE_FIRST_TRAJECTORY_CHECKPOINT` and must not be interpreted
as efficacy or mechanism evidence. API work occurred, but exact provider-call
accounting is unavailable because no trajectory checkpoint was written.

The existing D2 Generic Teacher–Critic–Student pipeline is shared unchanged by
both arms. The prohibition on a semantic Critic is interpreted as prohibiting
any new lane-specific or scheduler-dependent Critic behavior; removing the
existing shared Critic would violate the frozen one-factor comparison.

The stopped scope was exactly two trajectories for Seed75. Completed
trajectories: 0/2. Validation evaluations: 0. Test calls: 0. Resume authority
has been revoked; Seeds76/77 remain unexecuted.

## Phase B completion path audit

The stopped execution exposed stale completion constants inherited from an
earlier three-seed design. The runner summary expected six trajectories and
six evaluations per final dataset, while the auditor expected twelve final
evaluation artifacts. Those values did not match the frozen Seed75-only scope.

The engineering repair now derives one canonical completion scope from the
frozen seeds, arms, and final-evaluation dataset roles. It expects two
trajectories and exactly four identity-bearing evaluation artifacts:
Seed75/P0/shadow, Seed75/P0/validation, Seed75/P1/shadow, and
Seed75/P1/validation. Missing, extra, duplicate, test, wrong-seed, wrong-arm,
or dataset-role-inconsistent artifacts fail the audit.

The `pure_coverage` telemetry is now mutually exclusive with `direct_flip` and
`near_margin`. Deterministic replay confirms this reporting cleanup does not
change target scheduling or the frozen priority hierarchy.

- Scientific protocol changed: NO
- Repair-task API/model calls: 0
- Repair-task Validation evaluation calls: 0
- Repair-task Test calls: 0
- Seed75 execution resumed: NO
- Current scientific status: USER_ABORTED_INCOMPLETE; 0/2 trajectories
- Final scientific analysis: NOT RUN
