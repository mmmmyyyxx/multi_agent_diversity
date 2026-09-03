# Canonical vs No-Semantic-Critic Online Trajectory

Official audit: **PASS**. Frozen classifier: **NO_CLEAR_ONLINE_ADVANTAGE**. Evidence scope is 2 frozen seed pair(s).

**Provenance:** Seeds69/70 are a post-Seed68-result extension, not an untouched three-seed preregistration. Any combined Seed68-70 result is descriptive only.

| Metric | A Canonical | C No Semantic Critic |
|---|---:|---:|
| Student reaches | 9 | 22 |
| Feasible candidates | 11 | 30 |
| Accepted commits | 5 | 10 |
| Distinct member-seed updates | 5 | 8 |
| Final train Vote total | 104/150 | 121/150 |
| Final train Oracle total | 130/150 | 135/150 |
| Final validation Vote total | 64/100 | 70/100 |
| Final validation Oracle total | 77/100 | 83/100 |
| Coverage to vote conversions | 0 | 8 |
| Persistent singleton coverage | 7 | 7 |

C-A final validation Vote total = +6/100 and W/T/L = 1/0/1. The historical four-commit reference remains diagnostic only.

Validation was evaluated only after each online trajectory was frozen and never affected target selection, candidate acceptance, ranking, or commits. Intermediate frozen states were replayed post hoc only to attribute accepted-transition validation gains and losses. Test125 was not loaded for evaluation and received zero calls.
