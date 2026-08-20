# V17 Module-1 2x2 fixed-parent causal isolation

Both phases are complete. Phase A froze six pairwise-distinct historical V17
S2 parents, the four-cell matrix, target selectors, generic-revision budget,
WOULD_COMMIT simulation, realized validation endpoints, and the five-label
classifier without an API call. Phase B executed only that frozen matrix.

## Protocol result

```text
gate = PASS
parents = 6
cells = 24
branches = 48
WOULD_COMMIT = 9
actual prompt commits = 0
trajectory mutations = 0
test calls = 0
```

Every branch used at most two source candidates. Every valid Stage-B source
received exactly one loss-blind generic revision. Parent validation results
were identical across the four cells of each parent.

## Realized validation result

| Cell | Target allocation | Context | Vote delta | Oracle delta |
|---|---|---|---:|---:|
| A | Round-robin | Generic | +1 | +2 |
| B | W1 | Generic | 0 | 0 |
| C | Round-robin | Member-aware | +2 | -3 |
| D | W1 | Member-aware | 0 | +2 |

Frozen contrasts:

| Contrast | Vote | Oracle |
|---|---:|---:|
| B-A | -1 | -2 |
| D-C | -2 | +5 |
| C-A | +1 | -5 |
| D-B | 0 | +2 |
| `(D-C)-(B-A)` | -1 | +7 |

The preregistered classifier returns:

```text
TARGET_ALLOCATION_DOMINANT
```

Locally, W1 under both contexts produced lower realized validation vote delta
than its matched round-robin arm. Member-aware context was not uniformly
harmful: it improved vote delta under round-robin and improved oracle delta
under W1, while reducing oracle coverage under round-robin. This is a local
fixed-parent causal diagnosis, not a new trajectory-level efficacy claim.

Four invalidated attempts are retained only as execution incidents. They ended
because of probe-runner or credential-inheritance defects and are excluded
from mechanism evidence. No method, target policy, context policy, acceptance
rule, candidate budget, metric, or classifier was changed while fixing them.

This published directory contains only hashes, counters, categorical labels,
and aggregated metrics. It excludes prompts, questions, answers, raw model
responses, credentials, endpoints, caches, checkpoints, and absolute paths.
