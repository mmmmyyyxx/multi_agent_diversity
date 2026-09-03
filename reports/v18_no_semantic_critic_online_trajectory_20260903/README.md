# Canonical vs No-Semantic-Critic Online Trajectory

Official audit: **PASS**. Frozen classifier: **ONLINE_THROUGHPUT_AND_VOTE_SUPPORTED**. This is one-seed prospective evidence, not a multi-seed efficacy claim.

| Metric | A Canonical | C No Semantic Critic |
|---|---:|---:|
| Student reaches / 16 branches | 2 | 14 |
| Feasible candidates | 2 | 12 |
| Accepted commits | 2 | 5 |
| Distinct members updated | 2 | 5 |
| Final train Vote | 51/75 | 64/75 |
| Final train Oracle | 64/75 | 69/75 |
| Final validation Vote | 33/50 | 34/50 |
| Final validation Oracle | 39/50 | 42/50 |
| Coverage to vote conversions | 0 | 5 |
| Persistent singleton coverage | 4 | 3 |

C-A final validation Vote = +1/50 and W/T/L = 1/0/0. C reached all five members and exceeded the historical diagnostic reference of about four commits per seed.

Validation was evaluated only after each online trajectory was frozen and never affected target selection, candidate acceptance, ranking, or commits. Intermediate frozen states were replayed post hoc only to attribute accepted-transition validation gains and losses. Test125 was not loaded for evaluation and received zero calls.
