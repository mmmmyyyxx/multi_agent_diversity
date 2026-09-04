# Vote-Aligned Generic Shadow Pilot v1

Phase A implements and freezes a zero-API paired experiment. Both arms use the
existing Shadow-gated D2 Generic pipeline. P1 changes only target scheduling to
the deterministic hierarchy `direct_flip > near_margin > pure_coverage`, with
canonical actionable-member round robin as fallback.

Phase B has not been authorized or run. Consequently this directory contains
no efficacy result, no validation result, and no test result. Test50 remains
unaccessed.

The existing D2 Generic Teacher–Critic–Student pipeline is shared unchanged by
both arms. The prohibition on a semantic Critic is interpreted as prohibiting
any new lane-specific or scheduler-dependent Critic behavior; removing the
existing shared Critic would violate the frozen one-factor comparison.
