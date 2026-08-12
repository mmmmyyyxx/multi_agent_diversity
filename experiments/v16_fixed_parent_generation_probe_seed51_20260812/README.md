# v16 fixed-parent C0/C2/C3 generation probe

## Status

The authorized retry2 run passed the strict protocol gate:

```text
gate = PASS
cells = 9 / 9
commit_count = 0
validation_calls = 0
test_calls = 0
parent_state_mutations = 0
infrastructure_blockers = 0
```

All three variants were evaluated on the same frozen Seed51 final parent team,
the same three targets, the same assigned residuals, the same model and the
same two-candidate budget. C2 and C3 had identical Repair/Preservation set
membership in every case. The three case orders were rotated before execution.

## Fixed-parent results

| Variant | Critic semantic rejection | Valid / budget | Feasible / evaluated | Repair gain | P1 loss | Repair + preservation safe |
|---|---:|---:|---:|---:|---:|---:|
| C0 current v15 | 1/4 (25.0%) | 6/6 (100.0%) | 3/6 (50.0%) | 4/6 (66.7%) | 2/6 (33.3%) | 2/6 (33.3%) |
| C2 unlabeled preservation | 2/5 (40.0%) | 5/6 (83.3%) | 0/5 (0.0%) | 0/5 (0.0%) | 4/5 (80.0%) | 0/5 (0.0%) |
| C3 coalition-aware metadata | 4/6 (66.7%) | 4/6 (66.7%) | 0/4 (0.0%) | 1/4 (25.0%) | 2/4 (50.0%) | 1/4 (25.0%) |

C0 was the only variant to produce common-safe feasible candidates in this
probe. C2 produced no assigned-residual repair among five evaluated candidates
and had the highest observed P1-loss rate. C3 recovered one repair-safe
candidate relative to C2, but one of its three branches exhausted the Critic
semantic gate and none of its four evaluated candidates was feasible.

These fixed-parent observations reinforce the Seed51 trajectory warning rather
than supporting C2 or C3 promotion. Metadata did not rescue C3 to C0-level
candidate quality in this sample. In particular, the result does not support a
claim that C3 lowers collateral damage or improves feasible retention.

## Interpretation boundary

This is a small development mechanism probe with three parent-target cases and
six requested candidates per variant. It isolates context-generation behavior;
it is not a new seed, training trajectory, final-test evaluation, formal
efficacy estimate, or generalization result. No method threshold or acceptance
rule should be tuned from these nine cells alone.

The earlier transport-failure attempt and retry1 serialization failure are
excluded from all mechanism metrics and authoritative cost totals. They remain
local execution-incident evidence only.

## Authoritative cost

The valid retry2 used 1,164 provider attempts: 1,163 succeeded and one was
recovered by the existing retry policy. It consumed 347,749 prompt tokens and
19,414 completion tokens, for 367,163 total tokens. Role attempts were 15
Teacher, 15 Critic, 8 Student, and 1,126 Solver. There was no terminal
infrastructure failure.

Only sanitized counters, hashes, aggregate metrics, and protocol provenance
are tracked here. Prompts, questions, answers, raw responses, endpoints,
credentials, SQLite contents, checkpoints, and absolute paths are excluded.
