# Vote-Aligned Generic Shadow Pilot v1

Phase A is complete and freezes a zero-API paired experiment. Both arms use the
existing Shadow-gated D2 Generic pipeline. P1 changes only target scheduling to
the deterministic hierarchy `direct_flip > near_margin > pure_coverage`, with
canonical actionable-member round robin as fallback.

All ten Phase A gates passed on the clean implementation commit. Phase B has
not been authorized or run. Consequently this directory contains
no efficacy result, no validation result, and no test result. Test50 remains
unaccessed.

The existing D2 Generic Teacher–Critic–Student pipeline is shared unchanged by
both arms. The prohibition on a semantic Critic is interpreted as prohibiting
any new lane-specific or scheduler-dependent Critic behavior; removing the
existing shared Critic would violate the frozen one-factor comparison.

Planned Phase B is exactly six paired trajectories: Seeds 75, 76, and 77, each
in P0 then P1 order. Each trajectory has at most 32 update opportunities and
may stop only after six consecutive opportunities without a Shadow-approved
commit. Validation50 is evaluated once after each final state is frozen;
Test50 has zero permitted calls.
