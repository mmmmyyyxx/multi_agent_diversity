# V17 Module 1 fixed-parent 2x2 causal isolation

Status: Phase A source design. No API calls, validation rollouts, test
rollouts, commits, or trajectory mutations are permitted while preparing this
design.

## Frozen estimand

Six historical V17 S2 parents are reconstructed exactly and crossed with:

| Cell | Target allocation | Search/residual context |
|---|---|---|
| A / `RR_GENERIC` | dual round-robin | generic accuracy |
| B / `W1_GENERIC` | frozen W1 | generic accuracy |
| C / `RR_MEMBER_AWARE` | dual round-robin | member-aware residual |
| D / `W1_MEMBER_AWARE` | frozen W1 | member-aware residual |

Every parent-cell uses two distinct targets, two source candidates per target,
and one loss-blind generic revision for every valid source candidate. The
proposal mechanism remains generic evolution in all cells. M20, compatibility
repair, RCRU, proposal memory, validation selection, and test evaluation are
forbidden.

Each target branch is evaluated from the same immutable parent against the
same 75-row training pool and four fixed peers. Unchanged Common-Safe,
`common_monotone_safe` branch ranking, and the unchanged common cross-branch
key produce either one hypothetical transition (`WOULD_COMMIT`) or no update.
No prompt, profile, optimizer state, routing state, or historical artifact is
written back.

## Parent selection frozen before outcomes

The six distinct parent hashes are selected from the historical V17 S2 runs by
the explicit seed/update inventory in `parent_selection.json`. Categories are
diagnostic strata from the already-published zero-API V17 decomposition, not
new outcome filters:

- two concentration witnesses from the two seeds supporting H3;
- two throughput witnesses from rejected/low-throughput plateaus;
- two neutral controls from Seed56 states not used by those strata.

The registry builder must independently reconstruct every prompt-team hash,
the matching 75-row member profiles, service portfolios, active lanes, W1
total order, and dual round-robin targets. Any mismatch is a hard Phase-A
failure.

## Frozen endpoints

For each parent-cell, after Phase B candidate generation and train-only
WOULD_COMMIT simulation:

```text
realized_validation_vote_delta =
  V_val(hypothetical team) - V_val(parent), if WOULD_COMMIT
  0, otherwise

realized_validation_oracle_delta =
  O_val(hypothetical team) - O_val(parent), if WOULD_COMMIT
  0, otherwise
```

The vote delta is primary. Oracle coverage delta is the most important
secondary. Validation is evaluated in Phase B only; no test candidate call is
ever allowed.

The four paired contrasts are `B-A`, `D-C`, `C-A`, and `D-B`; the interaction
is `(D-C)-(B-A)`. Classification is performed only by the frozen rule in
`analysis_spec.json` and must return exactly one of its five labels.
