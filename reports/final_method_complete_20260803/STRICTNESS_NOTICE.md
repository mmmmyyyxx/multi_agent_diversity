# Strictness notice

All 36 planned runs completed operationally, and the original v1 artifact
gates found no missing runs, budget violations, validation use, test-lifecycle
violations, scheduler violations, or recorded request failures.

Post-run analysis found that the matched test comparison was not strict. Each
setting used a separate mutable cache cloned from the task-seed initialization
cache. The baseline test observations were therefore not present when the
optimized settings were cloned, so an unchanged prompt could be sent to the
provider again in another setting.

The decisive witness is `geometric_shapes`, seed 46: the full method accepted
zero updates, so its five prompts were unchanged, yet every member's test
correct count was 75 instead of the baseline's 73. The difference can only be
an observation mismatch across repeated identical requests. Consequently, the
test deltas and module comparisons in this directory are exploratory and must
not be treated as strict causal setting comparisons.

The runner has been repaired for future runs: every task-seed now has one
cumulative, content-addressed observation reference. Each setting starts from
an independent clone, then contributes only newly completed exact-request
entries back to the reference before the next setting starts. This preserves
run-local mutable caches while ensuring that any repeated exact request uses
the first recorded observation. The v2 gate requires a continuous cache chain
and rejects drift for unchanged prompts.

No experiment was rerun as part of this publication step.
