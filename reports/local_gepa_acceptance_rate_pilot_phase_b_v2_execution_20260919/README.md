# Phase-B Local GEPA 4x8 result

Classifier: **VALID_COMPLETE_4X8**.

Pooled descriptive acceptance: 5/10 = 0.5. This is a dependent, adaptive single-state pilot; it is not an IID population estimate.

Accepted parents: 4/4. Primary diagnostic: `LOCAL_GEPA_STRICT_IMPROVEMENT_OBSERVED_ACROSS_MULTIPLE_MEMBER_TASKS`. Exact-equal behavioral no-ops: 2; repair/preservation cancellations: 1. Proposal validity was the main funnel loss: 21/32 proposals were contract-invalid, all with an over-length failure. Eligible proposals produced 6 fixes and 3 breaks.

Operational note: the start-time `run_lifecycle.json` marker remained `RUNNING`; `execution.json` and `completion.json` both record `EXECUTION_COMPLETE`, the process exited successfully, and post-run source hashes matched.

Interpretation is limited to member/task heterogeneity within one frozen baseline state. `TEAM_TRANSFER_NOT_EVALUATED` remains true. No further Layer-1 budget was run.
