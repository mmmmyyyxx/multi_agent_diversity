# V17 Hybrid Target Allocation Prospective Pilot

This selector-local fixed-parent pilot compares three compute-matched arms:

```text
W1_TOP2                  = W1 rank 1 + W1 rank 2
RR_TOP2                  = first two responsibility-eligible members in the
                           frozen deterministic RR order
HYBRID_EXPLOIT_EXPLORE   = W1 rank 1 + first remaining responsibility-eligible
                           member in that same RR order
```

The responsibility-eligible set is frozen before proposal generation. No
validation, feasibility, candidate, or test outcome can alter either target.

All arms use two distinct targets, two source slots per target, one loss-blind
generic revision per valid source, member-aware residual/search context,
generic proposal evolution, unchanged Common-Safe, unchanged ranking,
unchanged max-one, fixed peers, and the same train and validation pools.

Six prospective parents are selected deterministically: two each from V17 S2
Seeds 56, 57, and 58, at the nearest eligible states to the one-third and
two-thirds positions of each seed's sorted unused eligible-state sequence. The
six prior diagnostic parents are excluded. Parent selection uses no validation,
test, branch-feasibility, selector-outcome, or WOULD_COMMIT evidence.

Equivalent branches across arms share a canonical branch artifact. Conceptual
compute remains two branches and four source slots per cell. The experiment
never commits a prompt, mutates a trajectory, or evaluates test data.

Primary endpoint: realized validation vote delta, with zero assigned to a cell
that has no hypothetical Common-Safe commit. The key secondary endpoint is the
identically defined realized validation oracle delta.

The frozen classifiers are defined in `classifier_spec.json`. They must not be
changed after Phase B begins.
