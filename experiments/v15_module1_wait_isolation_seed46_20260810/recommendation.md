# v15 Module 1 Wait-Isolation Recommendation

`RECOMMENDED_MODULE1_WAIT_DESIGN = W1`

## Two-stage conclusion

W1 is the only supported minimal correction. Moving wait inside `rho_state`
retains 27/27
known-feasible branch events and
22/22
actual commit targets. It demotes 12 known
failed branches without adding persistent state.

| Policy | Agent1+4 | Feasible retention | Commit retention | Failure demotion | Changes vs W1 |
|---|---:|---:|---:|---:|---:|
| W1 | 75 | 1.000 | 1.000 | 0.075 | 0 |
| W2 | 77 | 0.889 | 0.909 | 0.217 | 60 |
| W3 | 74 | 0.926 | 0.955 | 0.193 | 58 |

W2 loses 3 known-feasible events and 2 commit targets relative to W1 while
increasing aggregate Agent1+4 occupancy. W3 loses 2 known-feasible events and
1 commit target; its small aggregate Agent1+4 reduction is inconsistent across
settings and requires 58 Top-2 changes versus W1. Neither persistent extension
has sufficient incremental evidence.

## Direct low-rho witnesses

W1 reduces the S3 update-30 Agent1 score from
0.075926 to
0.031481, but its rank remains
1. It reduces the update-31 Agent4 score from
0.073333 to
0.028333, but its rank remains
2.

Thus W1 fixes the wait-placement semantics and is empirically safe, but it is
not evidence that long-run Agent1/4 budget occupation is solved. The supported
v15 design scope is W1 only; no persistent realizability state is recommended.

This is fixed-actual-trajectory replay. No alternative train/test outcome or
counterfactual acceptance rate is claimed.

`V15_IMPLEMENTATION_AUTHORIZED = false`
