# V18 Hybrid Online Accumulation Pilot

This preregistered development pilot asks one question: does singleton
competence recovered by the frozen Hybrid selector naturally deepen across
later online responsibility updates?

The two compute-matched arms are:

```text
W1_TOP2   = W1 rank 1 + W1 rank 2
HYBRID_BASE = W1 rank 1 + the first distinct actionable member in the
              frozen responsibility-constrained round-robin order
```

Both arms use the existing V17 member-aware residual/search context, generic
proposal evolution, two source candidates per target, one loss-blind generic
revision per valid source, unchanged Common-Safe, unchanged ranking, and at
most one experimental prompt commit per update.

The online horizon is recovered from the canonical V17 training protocol:
one epoch over 75 optimization examples with `update_every=10`, yielding
exactly eight update opportunities. Seeds 59, 60, and 61 are frozen before any
API result is observed. Both arms start from the same initial prompt team and
initial fixed-probe observations within each seed.

Validation is evaluated only at trajectory start and after a changed team
state. It is analysis-only and cannot affect target selection, generation,
candidate evaluation, ranking, commit, or stopping. Test evaluation is
forbidden.

The primary endpoint is the count of validation examples following a temporal
path `G=0 -> G=1 -> G>=2`. The classifiers and five allowed final diagnoses
are frozen in `classifier_spec.json`.
