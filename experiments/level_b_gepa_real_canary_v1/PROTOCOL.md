# Level-B GEPA real-provider canary v1

This is an engineering path canary, not an efficacy experiment. It executes
exactly one Seed78 parent, one target selected deterministically at update zero,
and one local GEPA opportunity. It preserves the frozen GEPA configuration,
36-call local metric budget, Common-Safe, TeamMiniBatch12, full evaluation, and
winner-only Shadow behavior.

The canary may initialize the shared P0 fixed Optimize100 probe. It performs no
Validation50 or Test50 evaluation and cannot be extended, resumed, or retried.

Frozen diagnostic classification:

- `BACKEND_EMPIRICAL_PATH_CONFIRMED`: proposal attempts > 0, candidate Solver
  calls beyond the 12-call root evaluation > 0, a changed valid candidate is
  returned, and a changed candidate enters TeamMiniBatch.
- `LOCAL_SEARCH_EMPIRICAL_BUT_NO_CHANGED_FRONTIER`: real candidate evaluation
  occurs but no changed valid frontier candidate is returned.
- `PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH`: proposals occur but no
  candidate Solver evaluation occurs beyond the root.
- `NO_REAL_PROPOSAL_ATTEMPT`: no proposal is attempted.
- `PROTOCOL_HOLD`: an integrity, infrastructure, persistence, access, source,
  or accounting invariant fails.

Accepted mutation, positive minibatch delta, TeamMiniBatch promotion, full
evaluation, Shadow, and commit are reported observations, not prerequisites for
declaring the local empirical path active.
